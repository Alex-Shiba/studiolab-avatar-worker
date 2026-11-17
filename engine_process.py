"""Resident torchrun supervisor with bounded progress, reload and verified teardown."""

import asyncio
from contextlib import suppress
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import tempfile
import uuid

from resident_protocol import atomic_json

import psutil

from diagnostics import model_environment
from media import video_info


class GenerationFailed(RuntimeError):
    pass


def gpu_group(count):
    if type(count) is not int or count < 1:
        raise RuntimeError("no visible GPU")
    return max(n for n in (1, 2, 4, 8) if n <= count)


def engine_environment(env, python, gpus):
    result = model_environment(env)
    site = Path(python).parent.parent / "lib/python3.10/site-packages"
    libraries = [site / "nvidia/cudnn/lib", site / "nvidia/cublas/lib", site / "nvidia/cuda_runtime/lib",
                 site / "torch/lib"]
    result["LD_LIBRARY_PATH"] = ":".join([*(str(p) for p in libraries),
                                          *([result["LD_LIBRARY_PATH"]] if result.get("LD_LIBRARY_PATH") else [])])
    result.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                  PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    result.pop("LONGCAT_A2A_WORKSPACE", None)
    if gpus > 2:
        result["LONGCAT_A2A_WORKSPACE"] = "1"
    if gpus > 1:
        # Which transport carries the all-to-all: a 2xH100 machine ran 2.4-3x slow (2026-11-17).
        result.update(NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT,P2P,SHM,NET")
    return result


def visible_gpus(python, env):
    result = subprocess.run([str(python), "-c", "import torch; print(torch.cuda.device_count())"],
                            capture_output=True, timeout=30, check=True, env=model_environment(env))
    return gpu_group(int(result.stdout.strip()))


def engine_command(python, directory, gpus):
    return [str(Path(python).with_name("torchrun")), "--standalone", f"--nproc_per_node={gpus}",
            str(Path(__file__).with_name("avatar_engine.py")), "--inbox", str(directory),
            f"--context_parallel_size={gpus}"]


def live(process):
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


class ProcessTree:
    def __init__(self, pid):
        self.pid = pid
        self.known = {}
        self.remember()

    def remember(self):
        try:
            parent = psutil.Process(self.pid)
            for process in [parent, *parent.children(recursive=True)]:
                self.known[(process.pid, process.create_time())] = process
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        if os.name != "nt":
            # Also find reparented children in our session/process group.
            for process in psutil.process_iter():
                try:
                    if os.getpgid(process.pid) == self.pid:
                        self.known[(process.pid, process.create_time())] = process
                except (ProcessLookupError, PermissionError, psutil.NoSuchProcess):
                    pass

    def alive(self):
        self.remember()
        return [p for p in self.known.values() if live(p)]

    async def stop(self, process):
        self.remember()
        if os.name != "nt":
            with suppress(ProcessLookupError, PermissionError):
                os.killpg(self.pid, signal.SIGKILL)
        # The test host has no POSIX groups. Remember and kill descendants too,
        # including ones whose parent has already exited.
        for child in reversed(list(self.known.values())):
            with suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                child.kill()
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(process.wait(), 10)
        deadline = time.monotonic() + 5
        while self.alive() and time.monotonic() < deadline:
            await asyncio.sleep(.05)
        return not self.alive()


class EngineBackend:
    def __init__(self, python=None, *, env=None, gpus=None, command=None, timeout=3600, poll=.1,
                 startup_timeout=1200):
        self.env = dict(os.environ if env is None else env)
        self.python = python or self.env.get("AVATAR_PYTHON", sys.executable)
        self.gpus = visible_gpus(self.python, self.env) if gpus is None else gpu_group(gpus)
        self.command = command or engine_command
        self.timeout, self.poll, self.startup_timeout = timeout, poll, startup_timeout
        self._healthy, self.busy, self.closing, self.tree_safe = False, False, False, True
        self.process = self.tree = self.output = self.monitor = self.inbox = None
        self.state = {}
        self.lock = asyncio.Lock()

    @property
    def healthy(self):
        return self._healthy and self.process is not None and self.process.returncode is None

    @healthy.setter
    def healthy(self, value):
        self._healthy = value

    async def start(self, root):
        async with self.lock:
            if self.inbox is None:
                root = Path(root).resolve()
                root.mkdir(parents=True, exist_ok=True, mode=0o700)
                self.inbox = Path(tempfile.mkdtemp(prefix="engine-", dir=root))
                self.inbox.chmod(0o700)
            self.closing = False
            if not self.healthy:
                await self._launch()
            if self.monitor is None:
                self.monitor = asyncio.create_task(self._monitor())

    async def _stop(self):
        self.healthy = False
        if self.process is not None:
            self.tree_safe = await self.tree.stop(self.process)
            if self.tree_safe:
                self.process = self.tree = None
        if self.output is not None:
            self.output.cancel()
            with suppress(asyncio.CancelledError):
                await self.output
            self.output = None
        return self.tree_safe

    async def _launch(self):
        if not await self._stop():
            raise GenerationFailed("engine did not stop")
        self.state = {}
        for name in ("ready.json", "job.json", "active.json"):
            (self.inbox / name).unlink(missing_ok=True)
        env = engine_environment(self.env, self.python, self.gpus)
        env.update(TMPDIR=str(self.inbox), TEMP=str(self.inbox), TMP=str(self.inbox),
                   TORCHINDUCTOR_CACHE_DIR=str(self.inbox / "inductor-cache"),
                   TRITON_CACHE_DIR=str(self.inbox / "triton-cache"))
        flags = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
        process = await asyncio.create_subprocess_exec(
            *self.command(self.python, self.inbox, self.gpus), cwd=self.inbox,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=env, **flags)
        self.process, self.tree = process, ProcessTree(process.pid)

        async def drain():
            while block := await process.stdout.read(8192):
                # The parent Diagnostics tee redacts before disk/stdout.
                sys.stdout.write(block.decode("utf-8", errors="replace"))
                sys.stdout.flush()

        self.output = asyncio.create_task(drain())
        deadline = time.monotonic() + self.startup_timeout
        try:
            while process.returncode is None:
                self.tree.remember()
                if time.monotonic() >= deadline:
                    raise GenerationFailed("engine load timed out")
                try:
                    state = json.loads((self.inbox / "ready.json").read_text(encoding="utf-8"))
                except FileNotFoundError:
                    await asyncio.sleep(self.poll)
                    continue
                except ValueError:
                    raise GenerationFailed("invalid engine readiness") from None
                if (not isinstance(state, dict) or state.get("attention") not in ("fa2", "fa3")
                        or type(state.get("compiled")) is not bool):
                    raise GenerationFailed("invalid engine readiness")
                self.state = {key: state[key] for key in ("attention", "compiled")}
                self.healthy = True
                return
            raise GenerationFailed("engine exited before ready")
        except BaseException:
            await self._stop()
            raise

    async def _monitor(self):
        while True:
            await asyncio.sleep(max(.5, self.poll))
            async with self.lock:
                if self.closing:
                    return
                if self.process is not None:
                    self.tree.remember()
                if not self.healthy:
                    try:
                        await self._launch()
                    except (GenerationFailed, OSError):
                        # Pending jobs remain in the server queue while the models reload.
                        pass

    async def restart(self):
        async with self.lock:
            await self._launch()

    async def close(self):
        self.closing = True
        if self.monitor is not None:
            self.monitor.cancel()
            with suppress(asyncio.CancelledError):
                await self.monitor
            self.monitor = None
        async with self.lock:
            await self._stop()

    def progress(self, directory, job, callback):
        try:
            path = directory / "progress.json"
            if path.stat().st_size > 4096:
                return
            value = json.loads(path.read_text(encoding="utf-8"))
            n = value.get("completed")
            if type(n) is not int or not 0 <= n <= job["segments"] or value.get("segments") != job["segments"]:
                return
            stage = value.get("stage")
            allowed = {"loading", "muxing", f"segment {n}/{job['segments']}"}
            if type(stage) is not str or stage not in allowed:
                return
            update = {"progress": n / job["segments"], "stage": stage, "gpus": self.gpus}
            for key in ("width", "height"):
                dimension = value.get(key)
                if type(dimension) is int and 0 < dimension <= 8192:
                    update[key] = dimension
            callback(update)
        except (OSError, ValueError, AttributeError):
            pass

    async def render(self, job, directory, callback):
        if self.busy or self.closing:
            raise GenerationFailed("backend unavailable")
        if self.inbox is None:
            await self.start(Path(directory).parent)
        async with self.lock:
            self.busy = True
            try:
                while not self.healthy:
                    try:
                        await self._launch()
                    except (GenerationFailed, OSError):
                        # No message was submitted: preserve this queued job through reload failures.
                        await asyncio.sleep(max(.5, self.poll))
                directory = Path(directory).resolve()
                token = uuid.uuid4().hex
                (directory / "completion.json").unlink(missing_ok=True)
                (directory / "progress.json").unlink(missing_ok=True)
                atomic_json(self.inbox / "job.json", {"directory": str(directory), "token": token, "job": job})
                deadline = time.monotonic() + self.timeout
                callback({"progress": 0.0, "stage": "loading", "gpus": self.gpus, **self.state})
                while True:
                    if self.process.returncode is not None:
                        raise GenerationFailed("generation failed")
                    if time.monotonic() >= deadline:
                        raise GenerationFailed("generation timed out")
                    self.tree.remember()
                    self.progress(directory, job, callback)
                    try:
                        completed = json.loads((directory / "completion.json").read_text(encoding="utf-8"))
                    except FileNotFoundError:
                        completed = None
                    if completed is not None:
                        if completed.get("token") != token:
                            raise GenerationFailed("invalid engine completion")
                        break
                    await asyncio.sleep(self.poll)
                self.progress(directory, job, callback)
                result = directory / "result.mp4"
                info = await asyncio.to_thread(video_info, result)
                if info["frames"] != job["frames"]:
                    raise GenerationFailed("invalid output frame count")
                callback({**info, **self.state})
                return result
            except asyncio.CancelledError:
                await self._stop()
                raise
            except Exception as error:
                # Reload before the server dequeues its next job. Only this job fails.
                try:
                    await self._launch()
                except (GenerationFailed, OSError):
                    pass
                if isinstance(error, GenerationFailed):
                    raise
                raise GenerationFailed("generation failed") from None
            finally:
                self.busy = False
