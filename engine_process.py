"""One torchrun process group per job, with bounded progress and verified teardown."""

import asyncio
from contextlib import suppress
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

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
        # Which transport carries the all-to-all (P2P/NVLink, SHM or sockets): a 2×H100 machine ran 2.4–3× slow (2026-11-17).
        result.update(NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT,P2P,SHM,NET")
    return result


def visible_gpus(python, env):
    result = subprocess.run([str(python), "-c", "import torch; print(torch.cuda.device_count())"],
                            capture_output=True, timeout=30, check=True, env=model_environment(env))
    return gpu_group(int(result.stdout.strip()))


def engine_command(python, directory, gpus):
    return [str(Path(python).with_name("torchrun")), "--standalone", f"--nproc_per_node={gpus}",
            str(Path(__file__).with_name("avatar_engine.py")), "--job", str(directory),
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
    def __init__(self, python=None, *, env=None, gpus=None, command=None, timeout=3600, poll=.1):
        self.env = dict(os.environ if env is None else env)
        self.python = python or self.env.get("AVATAR_PYTHON", sys.executable)
        self.gpus = visible_gpus(self.python, self.env) if gpus is None else gpu_group(gpus)
        self.command = command or engine_command
        self.timeout, self.poll, self.healthy = timeout, poll, True
        self.process = None

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
        if not self.healthy or self.process is not None:
            raise GenerationFailed("backend unavailable")
        env = engine_environment(self.env, self.python, self.gpus)
        # Prevent private data/cache from leaking outside this job.
        env.update(TMPDIR=str(directory), TEMP=str(directory), TMP=str(directory))
        flags = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
        process = await asyncio.create_subprocess_exec(
            *self.command(self.python, directory, self.gpus), cwd=directory,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=env, **flags)
        self.process = process
        tree = ProcessTree(process.pid)

        async def drain():
            while block := await process.stdout.read(8192):
                # The parent Diagnostics tee redacts before disk/stdout.
                sys.stdout.write(block.decode("utf-8", errors="replace"))
                sys.stdout.flush()

        output = asyncio.create_task(drain())
        deadline = time.monotonic() + self.timeout
        callback({"progress": 0.0, "stage": "loading", "gpus": self.gpus})
        try:
            while process.returncode is None:
                if time.monotonic() >= deadline:
                    raise GenerationFailed("generation timed out")
                tree.remember()
                self.progress(directory, job, callback)
                await asyncio.sleep(self.poll)
            self.progress(directory, job, callback)
            if process.returncode != 0:
                raise GenerationFailed("generation failed")
            # A successful launcher must also leave no children behind.
            if tree.alive():
                raise GenerationFailed("engine did not stop")
            result = directory / "result.mp4"
            info = await asyncio.to_thread(video_info, result)
            if info["frames"] != job["frames"]:
                raise GenerationFailed("invalid output frame count")
            callback(info)
            return result
        finally:
            self.healthy = False
            try:
                self.healthy = await tree.stop(process)
            finally:
                self.process = None
                output.cancel()
                with suppress(asyncio.CancelledError):
                    await output
