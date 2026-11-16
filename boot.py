"""Linux provisioning, invoked only on the rented worker after its proxy starts."""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from urllib.request import urlopen

from diagnostics import model_environment
from download import download_models, manifest

WORKSPACE = Path("/workspace")
FLASH_WHEEL = ("https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
               "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl")
EXCLUDED = {"torch", "torchvision", "torchaudio", "flash-attn", "libsndfile1",
            "tritonserverclient", "streamlit", "openai"}


class CommandFailed(RuntimeError):
    """An install command failed; `what` is its program and first argument (no paths, URLs or values)."""

    def __init__(self, what, code):
        super().__init__(what)
        self.what, self.code = what, code


def command_name(argv):
    words = [Path(str(a)).name for a in argv[:4] if not str(a).startswith("-")]
    return " ".join(words[:2])


def command(argv, **kwargs):
    result = subprocess.run([str(a) for a in argv], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=1800, env=model_environment(os.environ), **kwargs)
    if result.returncode != 0:
        # The children run without credentials in their environment; the tail goes to Vast's log so a failed boot is readable.
        tail = result.stderr.decode("utf-8", errors="replace").strip().splitlines()[-12:]
        for line in tail:
            print("AVATAR_INSTALL_STDERR " + line[:240], flush=True)
        raise CommandFailed(command_name(argv), result.returncode)
    return result.stdout.decode("utf-8").strip()


def system_tools_present():
    if any(shutil.which(binary) is None for binary in ("ffmpeg", "ffprobe", "git")):
        return False
    try:
        probe = subprocess.run(["ldconfig", "-p"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return b"libsndfile.so.1" in probe.stdout


def apt_install(packages, *, tries=10, pause=15, run=None, sleep=time.sleep):
    """apt-get with a lock wait and retries: the base image's own boot may hold the apt lock in a fresh container's first minutes."""
    run = run or command
    lock = ["-o", "DPkg::Lock::Timeout=120"]
    for attempt in range(tries):
        try:
            run(["apt-get", *lock, "update", "-qq"])
            run(["apt-get", *lock, "install", "-y", "-qq", *packages])
            return
        except CommandFailed:
            if attempt == tries - 1:
                raise
            print(f"AVATAR_APT_RETRY {attempt + 1}", flush=True)
            sleep(pause)


def filtered_requirements(text):
    kept = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name = re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0].lower().replace("_", "-")
        if name not in EXCLUDED:
            kept.append(line)
    return "\n".join(kept) + "\n"


def prepare(stage=lambda name: None):
    stage("install")
    import sys
    if sys.platform != "linux":
        raise RuntimeError("GPU provisioning requires Linux")
    os.umask(0o077)
    root = WORKSPACE / "studiolab-avatar"
    root.mkdir(parents=True, exist_ok=True)
    if not system_tools_present():
        apt_install(["ffmpeg", "git", "libsndfile1"])
    for binary in ("ffmpeg", "ffprobe"):
        if shutil.which(binary) is None:
            raise RuntimeError("media tools unavailable")
    pin = manifest()["code"]
    source = root / "LongCat-Video"
    if not source.exists():
        command(["git", "clone", "--filter=blob:none", "--no-checkout", pin["repository"], source])
        command(["git", "-C", source, "checkout", "--detach", pin["commit"]])
    if command(["git", "-C", source, "rev-parse", "HEAD"]) != pin["commit"]:
        raise RuntimeError("upstream commit mismatch")
    command(["git", "-C", source, "diff", "--exit-code", "HEAD", "--"])
    environment = root / "venv-py310"
    python = environment / "bin/python"
    if not python.exists():
        command(["python3.10", "-m", "venv", environment])
    command([python, "-c", "import sys; assert sys.version_info[:2] == (3, 10)"])
    pip = [python, "-m", "pip", "install", "--disable-pip-version-check"]
    command([*pip, "--index-url", "https://download.pytorch.org/whl/cu124",
             "torch==2.6.0+cu124", "torchvision==0.21.0+cu124", "torchaudio==2.6.0+cu124"])
    command([*pip, "--no-deps", "--only-binary=:all:", FLASH_WHEEL])
    requirements = root / "upstream-filtered.txt"
    requirements.write_text("".join(filtered_requirements((source / name).read_text())
                                    for name in ("requirements.txt", "requirements_avatar.txt")), encoding="utf-8")
    command([*pip, "-r", requirements, "-c", Path(__file__).with_name("runtime-constraints.txt"),
             "huggingface_hub==0.35.3", "accelerate==1.10.1", "Pillow==12.2.0"])
    command([python, "-c", "import torch, flash_attn, ctypes, pathlib; "
             "assert torch.__version__ == '2.6.0+cu124'; assert torch.version.cuda == '12.4'; "
             "assert not torch._C._GLIBCXX_USE_CXX11_ABI; assert torch.cuda.is_available(); "
             "assert flash_attn.__version__ == '2.7.4.post1'; "
             "p = pathlib.Path(torch.__file__).parent.parent / 'nvidia/cudnn/lib/libcudnn_adv.so.9'; "
             "assert p.is_file()"])
    stage("download")
    download_models(root / "weights")
    private = root / "private"
    private.mkdir(exist_ok=True, mode=0o700)
    for old in private.glob("run-*"):
        if old.is_dir() and not old.is_symlink() and old.resolve().parent == private.resolve():
            shutil.rmtree(old)
    runtime = Path(tempfile.mkdtemp(prefix="run-", dir=private))
    (runtime / "jobs").mkdir(mode=0o700)
    return source, python, runtime


def wait_ready(url, children, *, timeout=60, clock=time.monotonic, sleep=time.sleep, opener=urlopen):
    deadline = clock() + timeout
    while clock() < deadline:
        if any(child.poll() is not None for child in children):
            raise RuntimeError("child exited before ready")
        try:
            with opener(url, timeout=5) as response:
                if response.status == 200 and json.load(response).get("status") == "ready":
                    return
        except (OSError, ValueError):
            pass
        sleep(1)
    raise RuntimeError("readiness timeout")
