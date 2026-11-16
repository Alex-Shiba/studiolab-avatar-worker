import base64
from functools import lru_cache
import io
from pathlib import Path
import socket
import subprocess
import uuid

import aiohttp
from PIL import Image
import pytest


def pytest_configure(config):
    (Path(__file__).resolve().parents[1] / ".tmp").mkdir(exist_ok=True)


@pytest.fixture(autouse=True)
def no_external_network(monkeypatch):
    connect, connect_ex = socket.socket.connect, socket.socket.connect_ex

    def guard(original):
        def checked(sock, address):
            if isinstance(address, tuple) and address[0] not in ("127.0.0.1", "::1"):
                raise AssertionError("external networking is forbidden in tests")
            return original(sock, address)
        return checked

    monkeypatch.setattr(socket.socket, "connect", guard(connect))
    monkeypatch.setattr(socket.socket, "connect_ex", guard(connect_ex))


def encoded_image(fmt="PNG", size=(32, 24)):
    stream = io.BytesIO()
    Image.new("RGB", size, (25, 60, 90)).save(stream, format=fmt)
    return base64.b64encode(stream.getvalue()).decode("ascii")


@lru_cache(maxsize=16)
def audio_bytes(duration=2.0, codec="pcm_s16le", fmt="wav", frequency=440):
    return subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
        f"sine=frequency={frequency}:sample_rate=16000:duration={duration}",
        "-c:a", codec, "-f", fmt, "pipe:1"], capture_output=True, check=True, timeout=20).stdout


def payload(**overrides):
    return {"request": uuid.uuid4().hex, "mode": "avatar", "prompt": "A person speaks",
            "size": "480p", "seed": 123, "image": encoded_image(),
            "audio": base64.b64encode(audio_bytes()).decode("ascii"), **overrides}


def connector():
    return aiohttp.TCPConnector(resolver=aiohttp.ThreadedResolver())
