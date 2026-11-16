"""Own provisioning, the model server and Vast proxy; start_server.sh owns public TLS."""

import asyncio
import logging
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

from aiohttp import web
from vastai import BenchmarkConfig, HandlerConfig, LogActionConfig, Worker, WorkerConfig

from boot import WORKSPACE, CommandFailed, prepare, wait_ready
from contract import MAX_BODY_BYTES
from download import manifest
import exam
from diagnostics import Diagnostics, model_environment
from self_destruct import retire
from server import log_marker


def build_config(benchmark_id=None):
    benchmark_id = benchmark_id or os.environ["AVATAR_BENCHMARK_ID"]
    return WorkerConfig(
        model_server_url="http://127.0.0.1", model_server_port=18001,
        model_log_file=os.environ.get("AVATAR_LOG_FILE", "/var/log/portal/avatar-worker.log"),
        model_healthcheck_url="/health",
        handlers=[
            HandlerConfig(route="/clip/start", allow_parallel_requests=True, max_queue_time=30,
                          workload_calculator=lambda payload: 100.0),
            HandlerConfig(route="/clip/state", allow_parallel_requests=True, max_queue_time=30,
                          workload_calculator=lambda payload: 1.0,
                          benchmark_config=BenchmarkConfig(generator=lambda: {"id": benchmark_id},
                                                           runs=1, concurrency=1, do_warmup=False)),
            HandlerConfig(route="/clip/file", allow_parallel_requests=True, max_queue_time=30,
                          workload_calculator=lambda payload: 1.0),
        ],
        log_action_config=LogActionConfig(on_load=["AVATAR_READY"], on_error=[
            "AVATAR_FATAL install", "AVATAR_FATAL download", "AVATAR_FATAL exam",
            "AVATAR_FATAL server", "AVATAR_FATAL proxy", "AVATAR_FATAL"]),
    )


def bounded_route(handler):
    async def limited(request):
        # SDK 1.8.3 otherwise constructs an unlimited public aiohttp application.
        bounded = request.clone(client_max_size=MAX_BODY_BYTES + 65536)
        try:
            return await handler(bounded)
        except web.HTTPRequestEntityTooLarge:
            return web.json_response({"error": "invalid", "detail": "request exceeds size limit"}, status=400)
    return limited


def run_proxy():
    proxy = Worker(build_config())
    # SDK sets DEBUG during construction and logs full payloads at that level.
    logging.disable(logging.DEBUG)
    close_sessions(proxy.backend)
    proxy.routes = [web.post(route.path, bounded_route(route.handler)) for route in proxy.routes]
    proxy.run()


async def unknown_session(request):
    # Do not read/decompress JSON or invoke any SDK session machinery.
    return web.json_response({"error": "unknown"}, status=404)


def close_sessions(backend):
    # SDK 1.8.3 reads these attributes for both its public and webhook apps.
    for name in ("create", "get", "end", "health"):
        setattr(backend, f"session_{name}_handler", unknown_session)


class BootTimeout(RuntimeError):
    # Not OSError/TimeoutError: readiness and download retries must not swallow it.
    pass


class ExamFailed(RuntimeError):
    """The machine cannot finish a boot: it is given back before anything is installed."""


def examine(log, *, measure=exam.measure, env=None):
    """The first minute's look at the machine; returns its numbers or raises ExamFailed. The line it logs carries only numbers."""
    facts = measure(WORKSPACE, manifest(), WORKSPACE / "studiolab-avatar" / "weights", env=env)
    reason = exam.verdict(facts, env)
    line = exam.marker(facts, reason)
    log(line)
    # The marker file is the SDK's; this copy goes to the container's own output, so the exam is read in Vast's instance log too.
    print(line, flush=True)
    if reason:
        raise ExamFailed(reason)
    return facts


class BootWatchdog:
    def __init__(self, seconds=30 * 60):
        self.seconds = seconds
        self.previous = None

    def expired(self, signum, frame):
        raise BootTimeout("boot deadline exceeded")

    def start(self):
        # Provisioning is Linux-only. SIGALRM interrupts even a blocked install/download.
        if hasattr(signal, "setitimer"):
            self.previous = signal.signal(signal.SIGALRM, self.expired)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)

    def cancel(self):
        if self.previous is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self.previous)
            self.previous = None


def stop_children(children):
    for child in reversed(children):
        if child.poll() is None:
            child.terminate()
    for child in reversed(children):
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def main():
    children, runtime, diagnostics = [], None, None
    watchdog = BootWatchdog()
    current_stage = "install"

    def stage(name):
        nonlocal current_stage
        current_stage = name
        log_marker("AVATAR_STAGE " + name)

    def launch(args, env, **kwargs):
        child = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding="utf-8", errors="replace", bufsize=1,
                                 env=env, **kwargs)
        children.append(child)
        diagnostics.attach(child)
        return child

    def stop(signum, frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        os.umask(0o077)
        log = Path(os.environ.get("AVATAR_LOG_FILE", "/var/log/portal/avatar-worker.log"))
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("", encoding="utf-8")
        diagnostics = Diagnostics()
        watchdog.start()
        benchmark = uuid.uuid4().hex
        proxy_env = {**model_environment(os.environ), **{k: os.environ[k] for k in ("MASTER_TOKEN", "VAST_MASTER_TOKEN") if k in os.environ}, "AVATAR_BENCHMARK_ID": benchmark, "PYTHONUNBUFFERED": "1"}
        proxy_env.pop("CONTAINER_API_KEY", None)
        stage("proxy")
        launch([sys.executable, str(Path(__file__)), "--proxy"], proxy_env)
        # Before the install and the tens of gigabytes of models: a machine that cannot finish the boot leaves now.
        stage("exam")
        booted = time.monotonic()
        facts = examine(log_marker)
        stage("install")
        source, python, runtime = prepare(stage=stage)
        retire_file = runtime / "retire"
        # What the studio keeps about this machine travels with every state answer: numbers and Vast's own ids only.
        facts["boot_s"] = round(time.monotonic() - booted)
        child_env = {**model_environment(os.environ), "AVATAR_JOB_ROOT": str(runtime / "jobs"),
                     "AVATAR_MODEL_ROOT": str(source.parent / "weights"),
                     "AVATAR_SOURCE": str(source), "AVATAR_PYTHON": str(python), "AVATAR_READY_FILE": str(runtime / "ready"),
                     "AVATAR_RETIRE_FILE": str(retire_file), "AVATAR_BENCHMARK_ID": benchmark,
                     "AVATAR_MACHINE": exam.public(facts), "PYTHONUNBUFFERED": "1"}
        stage("server")
        launch([sys.executable, str(Path(__file__).with_name("server.py"))], child_env)
        wait_ready("http://127.0.0.1:18001/health", children, timeout=60)
        log_marker("AVATAR_READY")
        (runtime / "ready").touch(mode=0o600)
        watchdog.cancel()
        while True:
            for child in children:
                if child.poll() is not None:
                    current_stage = "proxy" if child is children[0] else "server"
                    raise ChildProcessError("child exited")
            if retire_file.exists():
                asyncio.run(retire(log_marker, reason="idle"))
                return
            time.sleep(1)
    except Exception as error:
        watchdog.cancel()
        if children and children[0].poll() is not None:
            current_stage = "proxy"
        detail = f" {error.what} rc={error.code}" if isinstance(error, CommandFailed) else ""
        log_marker(f"AVATAR_FATAL {current_stage} {type(error).__name__}{detail}")
        # Stop generation, but leave the proxy alive through its 10 s metrics cadence.
        stop_children(children[1:])
        time.sleep(15)
        asyncio.run(retire(log_marker, reason="fatal"))
        raise SystemExit(1) from None
    finally:
        watchdog.cancel()
        stop_children(children)
        if diagnostics is not None:
            diagnostics.close()
        if runtime is not None:
            shutil.rmtree(runtime, ignore_errors=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--proxy"]:
        run_proxy()
    else:
        main()
