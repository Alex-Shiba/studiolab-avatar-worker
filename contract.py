"""Strict wire validation. This module never imports a model or the Vast SDK."""

import base64
import binascii
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_UP
import io
import hashlib
import json
import re
import warnings

from PIL import Image, ImageOps
from pathlib import Path
import math
import subprocess

FPS = 25
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_IMAGE_BASE64 = 4 * ((MAX_IMAGE_BYTES + 2) // 3)
MAX_IMAGE_PIXELS = 16_777_216
MAX_IMAGE_EDGE = 8192
MAX_AUDIO_BYTES = 24 * 1024 * 1024
MAX_AUDIO_BASE64 = 4 * ((MAX_AUDIO_BYTES + 2) // 3)
MAX_BODY_BYTES = MAX_IMAGE_BASE64 + MAX_AUDIO_BASE64 + 65536
# Finished records and files share the video worker's one-hour retention.
TTL_S = 3600
ID_PATTERN = re.compile(r"[0-9a-f]{32}\Z")


class Invalid(ValueError):
    """Only fixed, non-user-controlled messages may cross the HTTP boundary."""


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise Invalid("duplicate key")
            result[key] = value
        return result

    def constant(_):
        raise Invalid("invalid JSON")

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, TypeError, RecursionError):
        raise Invalid("invalid JSON") from None


@dataclass(frozen=True)
class InputImage:
    data: bytes
    extension: str


def decode_image(encoded):
    if not isinstance(encoded, str) or not 1 <= len(encoded) <= MAX_IMAGE_BASE64:
        raise Invalid("invalid image encoding or size")
    try:
        raw = base64.b64decode(encoded, validate=True)
        if not raw or len(raw) > MAX_IMAGE_BYTES:
            raise Invalid("image exceeds 12 MiB")
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as image:
                fmt = image.format
                if fmt not in ("JPEG", "PNG") or getattr(image, "n_frames", 1) != 1:
                    raise Invalid("image must be a single JPEG or PNG")
                w, h = image.size
                if max(w, h) > MAX_IMAGE_EDGE or w * h > MAX_IMAGE_PIXELS:
                    raise Invalid("image dimensions exceed limit")
                image.verify()
            # verify() checks structure; load() checks compressed pixel contents.
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
        return InputImage(raw, ".jpg" if fmt == "JPEG" else ".png")
    except Invalid:
        raise
    except (ValueError, binascii.Error, OSError, SyntaxError, Image.DecompressionBombError,
            Image.DecompressionBombWarning):
        raise Invalid("invalid JPEG or PNG") from None


def validate_id(payload):
    if (not isinstance(payload, dict) or set(payload) != {"id"}
            or not isinstance(payload["id"], str) or not ID_PATTERN.fullmatch(payload["id"])):
        raise Invalid("id must be 32 lowercase hex characters")
    return payload["id"]


def segment_count(duration):
    # Decimal arithmetic avoids a spurious continuation at an exact boundary.
    extra = max(Decimal(0), Decimal(str(duration)) - Decimal("3.72"))
    return 1 + int((extra / Decimal("3.2")).to_integral_value(rounding=ROUND_CEILING))


def frame_count(duration):
    return int((Decimal(str(duration)) * FPS).to_integral_value(rounding=ROUND_HALF_UP))


@dataclass(frozen=True)
class InputAudio:
    data: bytes
    duration: float


@dataclass(frozen=True)
class ClipRequest:
    request: str
    mode: str
    prompt: str
    size: str
    seed: int
    image: InputImage
    audio: InputAudio

    @property
    def frames(self):
        return frame_count(self.audio.duration)

    @property
    def segments(self):
        return segment_count(self.audio.duration)

    def fingerprint(self):
        fields = {key: getattr(self, key) for key in ("mode", "prompt", "size", "seed")}
        for key in ("image", "audio"):
            fields[key] = hashlib.sha256(getattr(self, key).data).hexdigest()
        return hashlib.sha256(json.dumps(fields, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def save_image(image, path):
    with Image.open(io.BytesIO(image.data)) as source:
        oriented = ImageOps.exif_transpose(source)
        if oriented.mode.startswith("I"):
            oriented = oriented.convert("I").point(lambda value: value / 257).convert("L")
        oriented.convert("RGB").save(path, format="PNG")


def probe_audio(path):
    from diagnostics import model_environment
    import os
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe",
            "-format_whitelist", "wav,mp3,aac,mov,ogg,flac",
            "-show_streams", "-show_format", "-of", "json", str(path)],
            capture_output=True, timeout=20, check=True, env=model_environment(os.environ))
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        if len(streams) != 1 or streams[0].get("codec_type") != "audio":
            raise Invalid("audio must contain one audio stream")
        codec = streams[0].get("codec_name", "")
        containers = set(data.get("format", {}).get("format_name", "").split(","))
        if not (codec in {"mp3", "aac", "opus", "vorbis"} or codec.startswith("pcm_")):
            raise Invalid("unsupported audio codec")
        if not containers.intersection({"wav", "mp3", "aac", "mov", "mp4", "m4a", "ogg"}):
            raise Invalid("unsupported audio container")
        duration = float(streams[0].get("duration") or data["format"]["duration"])
        if not math.isfinite(duration) or not 2 <= duration <= 60:
            raise Invalid("audio must last 2 to 60 seconds")
        return duration
    except Invalid:
        raise
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        raise Invalid("invalid audio file") from None


def decode_audio(encoded, directory):
    if not isinstance(encoded, str) or not 1 <= len(encoded) <= MAX_AUDIO_BASE64:
        raise Invalid("invalid audio encoding or size")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise Invalid("invalid audio encoding or size") from None
    if not raw or len(raw) > MAX_AUDIO_BYTES:
        raise Invalid("audio exceeds 24 MiB")
    # A caller-owned private job folder, never a client path or system-wide temp.
    path = Path(directory) / "audio.input"
    path.write_bytes(raw)
    return InputAudio(raw, probe_audio(path))


def validate_start(payload, directory):
    required = {"request", "mode", "prompt", "size", "seed", "image", "audio"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise Invalid("missing or unknown keys")
    validate_id({"id": payload["request"]})
    if type(payload["mode"]) is not str or payload["mode"] != "avatar":
        raise Invalid("invalid mode")
    if type(payload["prompt"]) is not str or not 1 <= len(payload["prompt"]) <= 2000:
        raise Invalid("prompt must contain 1 to 2000 characters")
    if type(payload["size"]) is not str or payload["size"] not in ("480p", "720p"):
        raise Invalid("invalid size")
    if type(payload["seed"]) is not int or not 0 <= payload["seed"] <= 2147483647:
        raise Invalid("invalid seed")
    image = decode_image(payload["image"])
    audio = decode_audio(payload["audio"], directory)
    save_image(image, Path(directory) / "image.png")
    return ClipRequest(**{key: payload[key] for key in ("request", "mode", "prompt", "size", "seed")},
                       image=image, audio=audio)
