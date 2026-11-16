import hashlib
import io
import re
import threading

import pytest

from download import download_file, download_models, manifest, valid_file


class Response(io.BytesIO):
    def __init__(self, content, start, end, size, status=206):
        super().__init__(content)
        self.status = status
        self.headers = {"Content-Range": f"bytes {start}-{end}/{size}"}


def test_parallel_resume_integrity_and_second_boot(tmp_path):
    data = b"abcdefghijklmnopqrstuvwxyz012345"
    path = tmp_path / "model"
    path.with_name("model.range-0-15").write_bytes(data[:5])
    calls, threads = [], set()
    barrier = threading.Barrier(2)

    def opener(request, timeout):
        start, end = map(int, request.headers["Range"].split("=")[1].split("-"))
        calls.append((start, end))
        threads.add(threading.get_ident())
        barrier.wait(timeout=5)
        return Response(data[start:end + 1], start, end, len(data))

    digest = hashlib.sha256(data).hexdigest()
    download_file("https://example.invalid/pinned/model", path, len(data), digest, connections=2, opener=opener)
    assert path.read_bytes() == data and set(calls) == {(5, 15), (16, 31)} and len(threads) == 2
    assert list(tmp_path.iterdir()) == [path]
    download_file("unused", path, len(data), digest, opener=lambda *a, **k: pytest.fail("cached model fetched"))
    assert valid_file(path, len(data), digest)


@pytest.mark.parametrize("bad", ["range", "checksum", "oversized"])
def test_bad_download_is_never_installed(tmp_path, bad):
    data = b"good"

    def opener(request, timeout):
        body = b"evil" if bad == "checksum" else data + b"x" if bad == "oversized" else data
        return Response(body, 0, 3, 4, status=200 if bad == "range" else 206)

    path = tmp_path / "model"
    sleeps = []
    def deadline(delay):
        from worker import BootTimeout
        sleeps.append(delay)
        if len(sleeps) == 8:
            raise BootTimeout("test watchdog")
    with pytest.raises(RuntimeError):
        download_file("https://example.invalid/model", path, 4, hashlib.sha256(data).hexdigest(),
                      connections=1, opener=opener, sleep=deadline)
    assert not path.exists() and not list(tmp_path.glob("*.range-*"))
    assert len(sleeps) == (0 if bad == "checksum" else 8)


def test_partial_network_failure_resumes(tmp_path):
    data = b"abcdef"
    calls = []

    def opener(request, timeout):
        start, end = map(int, request.headers["Range"].split("=")[1].split("-"))
        calls.append(start)
        return Response(data[:2] if len(calls) == 1 else data[start:], start, end, len(data))

    path = tmp_path / "model"
    download_file("https://example.invalid/model", path, 6, hashlib.sha256(data).hexdigest(), connections=1,
                  opener=opener, sleep=lambda _: None)
    assert calls == [0, 2] and path.read_bytes() == data


def test_same_size_corrupt_cache_is_rechecked(tmp_path):
    path = tmp_path / "model"
    path.write_bytes(b"evil")
    download_file("https://example.invalid/model", path, 4, hashlib.sha256(b"good").hexdigest(), connections=1,
                  opener=lambda *a, **k: Response(b"good", 0, 3, 4))
    assert path.read_bytes() == b"good"
