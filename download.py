"""Pinned downloads with resumable, validated parallel byte ranges."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

MANIFEST_PATH = Path(__file__).with_name("model-manifest.json")
# Anonymous Hugging Face downloads were throttled to ~14 MB/s on 2026-11-17 after a day of repeated 60 GB boots from one host
# (the exam's 8-second sample still saw 226 MB/s), and the boot ran out of its 30 minutes. With HF_TOKEN in the machine's
# environment (a template or account variable the owner sets; never printed: diagnostics redact *TOKEN*) requests are
# authenticated. The token goes only to Hugging Face hosts and is dropped when a redirect leaves them.
HF_HOSTS = ("huggingface.co", "hf.co")


def hf_host(url):
    host = (urlsplit(url).hostname or "").lower()
    return any(host == name or host.endswith("." + name) for name in HF_HOSTS)


def hf_headers(url, env=None):
    token = (os.environ if env is None else env).get("HF_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}"} if token and hf_host(url) else {}


class HFRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and not hf_host(newurl):
            new.remove_header("Authorization")
        return new


hf_open = build_opener(HFRedirect).open


def manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def valid_file(path, size, digest):
    if not path.is_file() or path.stat().st_size != size:
        return False
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest() == digest


def download_file(url, target, size, digest, *, connections=6, opener=hf_open, sleep=time.sleep,
                  progress=lambda value: None, max_attempts=None, cancelled=None):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if valid_file(target, size, digest):
        progress(1.0)
        return target
    target.unlink(missing_ok=True)
    # Ranges depend only on the manifest size, never the available prefix.
    # Their names remain stable across interruptions and reboots.
    count = min(max(1, connections), size)
    step = (size + count - 1) // count
    segments = [(start, min(size - 1, start + step - 1),
                 target.with_name(f"{target.name}.range-{start}-{min(size - 1, start + step - 1)}"))
                for start in range(0, size, step)]
    stopped = threading.Event()

    def is_stopped():
        return stopped.is_set() or (cancelled is not None and cancelled.is_set())

    progress_lock = threading.Lock()
    completed = {}

    def report(start, have):
        with progress_lock:
            completed[start] = have
            progress(min(.99, sum(completed.values()) / size))

    def segment(spec):
        start, end, path = spec
        length = end - start + 1
        attempt = 0
        while not is_stopped():
            previous_attempt = attempt
            try:
                have = path.stat().st_size if path.exists() else 0
                report(start, have)
                if have == length:
                    return
                if have > length:
                    path.unlink()
                    have = 0
                request = Request(url, headers={**hf_headers(url), "Range": f"bytes={start + have}-{end}",
                                                "Accept-Encoding": "identity"})
                with opener(request, timeout=120) as response:
                    expected = f"bytes {start + have}-{end}/{size}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError("invalid byte range response")
                    remaining = length - have
                    with path.open("ab") as output:
                        while remaining:
                            if is_stopped():
                                return
                            data = response.read(min(8 * 1024 * 1024, remaining))
                            if not data:
                                raise OSError("short range")
                            output.write(data)
                            remaining -= len(data)
                            report(start, length - remaining)
                            attempt = 0
                        if response.read(1):
                            raise ValueError("oversized range")
                return
            except ValueError:
                # A bad response must never become a trusted resume prefix.
                path.unlink(missing_ok=True)
                attempt = previous_attempt
            except OSError:
                # Includes URLError, HTTPError (429/5xx), timeouts and short reads.
                pass
            # Boot downloads rely on the parent's watchdog; optional files bound
            # consecutive failures so the next refs job can retry. Progress resets delay.
            if max_attempts is not None and attempt >= max_attempts - 1:
                raise RuntimeError("reference download retries exhausted")
            delay = min(30, 2 ** min(attempt, 5))
            attempt = min(attempt + 1, 5)
            if sleep is time.sleep:
                stopped.wait(delay)
            else:
                sleep(delay)

    pool = ThreadPoolExecutor(max_workers=count)
    try:
        list(pool.map(segment, segments))
    except BaseException:
        # In particular, propagate the parent's boot watchdog immediately.
        # Boot cannot wait for HTTP transfers. Optional transfers must finish
        # their threads before a retry can reuse the same range files.
        stopped.set()
        pool.shutdown(wait=cancelled is not None, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    if is_stopped():
        raise RuntimeError("download cancelled")
    assembled = target.with_name(target.name + ".assembling")
    try:
        with assembled.open("wb") as output:
            for _, _, path in segments:
                with path.open("rb") as source:
                    shutil.copyfileobj(source, output, 8 * 1024 * 1024)
        if not valid_file(assembled, size, digest):
            for _, _, path in segments:
                path.unlink(missing_ok=True)
            raise RuntimeError("download checksum mismatch")
        assembled.replace(target)
        for _, _, path in segments:
            path.unlink(missing_ok=True)
    finally:
        assembled.unlink(missing_ok=True)
    progress(1.0)
    return target


def download_models(model_root, *, emit=None, clock=time.monotonic, every=30.0, fetch=None):
    """All pinned files in order; one numbers-only progress line every `every` seconds (MB/s since the last line)."""
    emit = emit or (lambda line: print(line, flush=True))
    fetch = fetch or download_file
    files = manifest()["files"]
    total = sum(item["size"] for item in files)
    started = clock()
    last = {"at": started, "bytes": 0}
    done = 0
    for item in files:
        repo, revision = item["repository"], item["revision"]
        source = item.get("source_path", item["path"])
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{source}"

        def progress(fraction, base=done, size=item["size"]):
            now, have = clock(), base + int(fraction * size)
            if now - last["at"] >= every:
                rate = (have - last["bytes"]) / max(now - last["at"], 1e-9) / 1e6
                emit("AVATAR_DOWNLOAD %.1f/%.1f GB %.0f MB/s" % (have / 1e9, total / 1e9, rate))
                last["at"], last["bytes"] = now, have

        fetch(url, Path(model_root) / item["path"], item["size"], item["sha256"], progress=progress)
        done += item["size"]
    emit("AVATAR_DOWNLOAD done %.1f GB in %.0f s" % (total / 1e9, clock() - started))
