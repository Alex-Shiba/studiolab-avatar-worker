"""Regression coverage for resident engine recovery, boot deadlines and sampling."""

import asyncio
from contextlib import suppress
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import psutil
import pytest

import avatar_engine
import boot
from contract import validate_start
import engine_process
from engine_process import GenerationFailed, ProcessTree
import gpu_watch
import server
import worker
from conftest import payload
from test_engine import fake_backend, job_files


async def until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(.01)
    await asyncio.wait_for(wait(), 10)


@pytest.mark.parametrize("reload_failure", ["crash", "timeout"])
async def test_failed_reloads_fail_all_jobs_and_exit_server(tmp_path, monkeypatch, reload_failure):
    backend = fake_backend(env={**os.environ, "AVATAR_TEST_GATE": str(tmp_path / "gate")}, poll=.02)
    managers, launches = [], []
    original_manager, original_command = server.JobManager, backend.command

    def manager(*args, **kwargs):
        result = original_manager(*args, **kwargs)
        managers.append(result)
        return result

    def command(*args):
        launches.append(1)
        return original_command(*args)

    backend.command = command
    monkeypatch.setattr(server, "EngineBackend", lambda: backend)
    monkeypatch.setattr(server, "JobManager", manager)
    # Exercise the real app lifecycle without reserving the production port.
    monkeypatch.setattr(server.web, "TCPSite", lambda *a, **k: SimpleNamespace(start=AsyncMock()))
    monkeypatch.setattr(asyncio.get_running_loop(), "add_signal_handler", lambda *a: None)
    monkeypatch.setenv("AVATAR_JOB_ROOT", str(tmp_path / "jobs"))
    monkeypatch.setenv("AVATAR_LOG_FILE", str(tmp_path / "markers.log"))

    async def serve():
        try:
            await server.serve()
        except SystemExit as error:
            return error.code

    task = asyncio.create_task(serve())
    try:
        await until(lambda: backend.healthy)
        manager = managers[0]
        jobs = []
        for number in range(3):
            prepared = manager.root / f"prepared-{number}"
            prepared.mkdir()
            clip = validate_start(payload(), prepared)
            jobs.append(manager.submit(clip, prepared))
        await until(lambda: (jobs[0].directory / "entered").exists())
        assert [job.state for job in jobs] == ["running", "queued", "queued"]
        if reload_failure == "crash":
            backend.env["AVATAR_TEST_LOAD_CRASH"] = "1"
        else:
            backend.env["AVATAR_TEST_LOAD_DELAY"] = "120"
            backend.startup_timeout = .15
        backend.process.kill()
        assert await asyncio.wait_for(task, 15) == 1
        assert backend.dead and backend.reload_failures == 3 and len(launches) == 4
        assert all(job.state == "failed" for job in jobs)
        assert jobs[0].error.startswith("engine exited rc=")
        assert all(job.error == "generation failed" for job in jobs[1:])
        assert all(job.finished_at is not None for job in jobs)
        assert manager.active == manager.retirement.jobs == 0
        assert manager.queue.empty() and backend.process is None and backend.tree_safe
        assert "AVATAR_FATAL server GenerationFailed" in (tmp_path / "markers.log").read_text()
        with pytest.raises(GenerationFailed, match="^generation failed$"):
            await backend.render({}, tmp_path, lambda _: None)
        with pytest.raises(GenerationFailed, match="^generation failed$"):
            await backend.restart()
        assert len(launches) == 4
    finally:
        if not task.done():
            task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        await backend.close()


async def test_unconfirmed_stop_is_immediately_fatal_without_another_launch(tmp_path, monkeypatch):
    backend = fake_backend(poll=.02)
    await backend.start(tmp_path)
    process = backend.process
    with monkeypatch.context() as patch:
        stop = AsyncMock(return_value=False)
        patch.setattr(backend.tree, "stop", stop)
        with pytest.raises(GenerationFailed, match="engine did not stop"):
            await backend.restart()
        assert backend.dead and not backend.healthy and not backend.tree_safe
        assert backend.process is process and process.returncode is None
        assert backend.reload_failures == 1
        with pytest.raises(GenerationFailed, match="^generation failed$"):
            await asyncio.wait_for(backend.wait_ready(), 1)
        with pytest.raises(GenerationFailed, match="^generation failed$"):
            await backend.restart()
        stop.assert_awaited_once()
    await backend.close()
    assert backend.tree_safe and not psutil.pid_exists(process.pid)


async def test_waiting_render_fails_when_recovery_is_exhausted(tmp_path):
    backend = fake_backend(poll=.02)
    await backend.start(tmp_path)
    backend.env["AVATAR_TEST_LOAD_CRASH"] = "1"
    backend.process.kill()
    await backend.process.wait()
    directory = tmp_path / "job"
    with pytest.raises(GenerationFailed, match="^generation failed$"):
        await asyncio.wait_for(backend.render(job_files(directory), directory, lambda _: None), 10)
    assert backend.dead and backend.reload_failures == 3


async def test_successful_reload_resets_consecutive_failure_budget(tmp_path, monkeypatch):
    backend = fake_backend(poll=.02)
    await backend.start(tmp_path)
    original_launch = backend._launch
    attempts = []

    async def launch():
        attempts.append(1)
        if len(attempts) <= 2:
            raise GenerationFailed("engine exited before ready")
        await original_launch()

    monkeypatch.setattr(backend, "_launch", launch)
    backend.process.kill()
    await backend.process.wait()
    await asyncio.wait_for(backend.wait_ready(), 10)
    assert len(attempts) == 3 and backend.reload_failures == 0 and not backend.dead


@pytest.mark.parametrize("failure", ["timeout", "crash"])
async def test_inflight_failure_returns_before_blocked_reload(tmp_path, monkeypatch, failure):
    backend = fake_backend(env={**os.environ, "AVATAR_TEST_GATE": str(tmp_path / "gate")},
                           poll=.02, timeout=.5 if failure == "timeout" else 30)
    await backend.start(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    original_launch = backend._launch

    async def blocked_launch():
        entered.set()
        await release.wait()
        await original_launch()

    monkeypatch.setattr(backend, "_launch", blocked_launch)
    directory = tmp_path / "job"
    task = asyncio.create_task(backend.render(job_files(directory), directory, lambda _: None))
    try:
        await until(lambda: (directory / "entered").exists())
        if failure == "crash":
            backend.process.kill()
        with pytest.raises(GenerationFailed, match="timed out" if failure == "timeout" else "engine exited rc="):
            await asyncio.wait_for(asyncio.shield(task), 3)
        assert backend.tree_safe and not backend.busy and not backend.healthy
        await asyncio.wait_for(entered.wait(), 3)
        assert not release.is_set()
        release.set()
        await asyncio.wait_for(backend.wait_ready(), 5)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        with suppress(asyncio.CancelledError, GenerationFailed):
            await task
        await backend.close()


@pytest.mark.parametrize("error_type", [worker.BootTimeout, RuntimeError])
def test_fa3_build_does_not_swallow_boot_watchdog(tmp_path, monkeypatch, error_type):
    pin = boot.manifest()["flash_attention"]
    calls = []

    def command(args, **kwargs):
        calls.append(args)
        if "setup.py" in args:
            raise error_type("boot deadline exceeded")
        return pin["commit"] if "rev-parse" in args else ""

    monkeypatch.setattr(boot, "command", command)
    with pytest.raises(error_type, match="boot deadline exceeded"):
        boot.install_fa3(tmp_path, Path("/venv/bin/python"))
    assert "setup.py" in calls[-1]
    assert not (tmp_path / "fa3-status.json").exists()


def test_fa3_build_timeout_falls_back_with_shorter_budget(tmp_path, monkeypatch):
    calls = []
    pin = boot.manifest()["flash_attention"]

    def run(args, **kwargs):
        calls.append((args, kwargs))
        if "setup.py" in args:
            assert kwargs["timeout"] == 900
            raise subprocess.TimeoutExpired(args, kwargs["timeout"])
        return SimpleNamespace(returncode=0, stdout=pin["commit"].encode() if "rev-parse" in args else b"", stderr=b"")

    monkeypatch.setattr(boot.subprocess, "run", run)
    assert boot.install_fa3(tmp_path, Path("/venv/bin/python")) is False
    assert "setup.py" in calls[-1][0]
    assert json.loads((tmp_path / "fa3-status.json").read_text())["available"] is False
    assert calls[0][1]["timeout"] == 1800


@pytest.mark.parametrize("rank,fail_job", [(0, False), (0, True), (1, False), (1, True)])
def test_sampler_lives_only_during_rank_zero_jobs(tmp_path, monkeypatch, rank, fail_job):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(avatar_engine.tempfile, "tempdir", str(tmp_path))
    active, samplers, jobs = [], [], []
    runtime = SimpleNamespace(global_rank=rank, control=object(), state={"attention": "fa2", "compiled": False})

    class Sampler:
        def start(self):
            assert not active
            active.append(self)
            samplers.append(self)
            return self

        def stop(self):
            assert active == [self]
            active.clear()

    def next_job(inbox):
        assert not active, "sampling must stop before the idle wait"
        if len(jobs) == 2:
            raise StopIteration
        directory = tmp_path / str(len(jobs))
        directory.mkdir()
        return {"directory": str(directory), "job": {}, "token": "test"}

    def broadcast(message, **kwargs):
        assert not active
        if rank != 0:
            message[0] = next_job(tmp_path)

    def generate(*args):
        assert len(active) == (1 if rank == 0 else 0)
        jobs.append(1)
        if fail_job:
            raise ValueError("job failed")

    distributed = ModuleType("torch.distributed")
    distributed.broadcast_object_list = broadcast
    distributed.destroy_process_group = lambda: None
    torch = ModuleType("torch")
    torch.distributed = distributed
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", distributed)
    monkeypatch.setattr(avatar_engine, "load_runtime", lambda *a: runtime)
    monkeypatch.setattr(avatar_engine, "wait_job", next_job)
    monkeypatch.setattr(avatar_engine, "generate", generate)
    monkeypatch.setattr(gpu_watch, "Sampler", Sampler)
    with pytest.raises(ValueError if fail_job else StopIteration):
        avatar_engine.run(tmp_path, 2)
    assert not active
    assert len(samplers) == ((1 if fail_job else 2) if rank == 0 else 0)


def test_sampler_stop_waits_for_outstanding_query():
    queried, release = threading.Event(), threading.Event()

    def query(args):
        queried.set()
        assert release.wait(3)
        return ""

    sampler = gpu_watch.Sampler(query=query).start()
    stopper = threading.Thread(target=sampler.stop)
    try:
        assert queried.wait(2)
        stopper.start()
        assert sampler.stopped.wait(2)
        assert stopper.is_alive()
    finally:
        release.set()
        stopper.join(3)
        sampler.thread.join(3)
    assert not stopper.is_alive() and not sampler.thread.is_alive()


def test_process_tree_prunes_exited_children_but_keeps_living_orphans(monkeypatch):
    def process(pid, running=True):
        return SimpleNamespace(pid=pid, create_time=lambda: 1, is_running=lambda: running,
                               status=lambda: psutil.STATUS_RUNNING, children=lambda **kw: [])

    parent, orphan, exited = process(10), process(11), process(12, running=False)
    monkeypatch.setattr(engine_process.psutil, "Process", lambda pid: parent)
    monkeypatch.setattr(engine_process.psutil, "process_iter", lambda: [])
    tree = ProcessTree(parent.pid)
    tree.known.update({(orphan.pid, 1): orphan, (exited.pid, 1): exited})
    tree.remember()
    assert set(tree.known) == {(parent.pid, 1), (orphan.pid, 1)}
