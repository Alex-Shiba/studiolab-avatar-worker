import base64
from dataclasses import replace
import io
import json
import subprocess

from PIL import Image
import pytest

import contract
from contract import Invalid, decode_image, frame_count, save_image, segment_count, strict_json, validate_start
from conftest import audio_bytes, encoded_image, payload


def test_valid_request_preserves_bytes_and_no_crop(tmp_path):
    body = payload(size="720p", seed=2147483647, prompt="Человек рассказывает историю.")
    clip = validate_start(body, tmp_path)
    assert clip.mode == "avatar" and clip.size == "720p" and clip.segments == 1 and clip.frames == 50
    assert (tmp_path / "audio.input").read_bytes() == base64.b64decode(body["audio"])
    with Image.open(tmp_path / "image.png") as image:
        assert image.size == (32, 24) and image.mode == "RGB"


@pytest.mark.parametrize("key", ["request", "mode", "prompt", "size", "seed", "image", "audio"])
def test_every_missing_key(tmp_path, key):
    body = payload()
    del body[key]
    with pytest.raises(Invalid, match="keys"):
        validate_start(body, tmp_path)


@pytest.mark.parametrize("key", ["seconds", "aspect", "first", "last", "refs", "lab", "url", "path", "graph", "extra"])
def test_every_foreign_key(tmp_path, key):
    with pytest.raises(Invalid, match="keys"):
        validate_start(payload(**{key: None}), tmp_path)


@pytest.mark.parametrize("key,value", [
    ("request", "A" * 32), ("request", "../file"), ("request", 12), ("request", None),
    ("mode", "text"), ("mode", []), ("mode", True), ("prompt", ""), ("prompt", "a" * 2001),
    ("prompt", 1), ("size", "1080p"), ("size", []), ("seed", True), ("seed", "1"),
    ("seed", -1), ("seed", 2147483648), ("seed", 1.5), ("seed", None),
    ("image", "data:image/png;base64,abc"), ("image", "%%%"), ("image", False),
    ("audio", "data:audio/wav;base64,abc"), ("audio", "%%%"), ("audio", []), ("audio", ""),
])
def test_bad_values(tmp_path, key, value):
    with pytest.raises(Invalid):
        validate_start(payload(**{key: value}), tmp_path)


@pytest.mark.parametrize("text", ['{"seed":1,"seed":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}', '{'])
def test_strict_json(text):
    with pytest.raises(Invalid):
        strict_json(text)


@pytest.mark.parametrize("duration,good", [(1.9, False), (2, True), (3.73, True), (60, True), (60.1, False)])
def test_audio_duration_from_real_ffprobe(tmp_path, duration, good):
    body = payload(audio=base64.b64encode(audio_bytes(duration)).decode())
    if good:
        assert validate_start(body, tmp_path).audio.duration == pytest.approx(duration)
    else:
        with pytest.raises(Invalid, match="audio must last 2 to 60 seconds"):
            validate_start(body, tmp_path)


@pytest.mark.parametrize("codec,fmt", [("pcm_s16le", "wav"), ("libmp3lame", "mp3"), ("aac", "adts"),
                                      ("libopus", "ogg"), ("libvorbis", "ogg")])
def test_accepted_codecs(tmp_path, codec, fmt):
    clip = validate_start(payload(audio=base64.b64encode(audio_bytes(2.1, codec, fmt)).decode()), tmp_path)
    assert 2 <= clip.audio.duration < 2.5


def test_m4a_and_disallowed_codec_and_multiple_streams(tmp_path):
    output = tmp_path / "source.m4a"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=duration=2.1",
                    "-c:a", "aac", str(output)], check=True, timeout=20)
    assert validate_start(payload(audio=base64.b64encode(output.read_bytes()).decode()), tmp_path)
    with pytest.raises(Invalid, match="codec"):
        validate_start(payload(audio=base64.b64encode(audio_bytes(2.1, "flac", "flac")).decode()), tmp_path)
    output = tmp_path / "multiple.m4a"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=duration=2.1",
                    "-map", "0:a", "-map", "0:a", "-c:a", "aac", str(output)], check=True, timeout=20)
    with pytest.raises(Invalid, match="one audio stream"):
        validate_start(payload(audio=base64.b64encode(output.read_bytes()).decode()), tmp_path)


def test_audio_and_image_size_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(contract, "MAX_AUDIO_BYTES", 10)
    with pytest.raises(Invalid, match="24 MiB"):
        validate_start(payload(), tmp_path)
    monkeypatch.setattr(contract, "MAX_AUDIO_BASE64", 10)
    with pytest.raises(Invalid, match="size"):
        validate_start(payload(), tmp_path)
    monkeypatch.setattr(contract, "MAX_IMAGE_BYTES", 10)
    with pytest.raises(Invalid, match="12 MiB"):
        decode_image(encoded_image())


@pytest.mark.parametrize("encoded", [encoded_image("GIF"), encoded_image(size=(8193, 1)),
                                     base64.b64encode(b"not image").decode()])
def test_bad_images(encoded):
    with pytest.raises(Invalid):
        decode_image(encoded)


def test_image_orientation_and_16bit_gray(tmp_path):
    stream = io.BytesIO()
    image = Image.new("RGB", (20, 30))
    exif = Image.Exif()
    exif[274] = 6
    image.save(stream, format="JPEG", exif=exif)
    save_image(decode_image(base64.b64encode(stream.getvalue()).decode()), tmp_path / "oriented.png")
    with Image.open(tmp_path / "oriented.png") as result:
        assert result.size == (30, 20)
    stream = io.BytesIO()
    Image.new("I;16", (4, 4), 32768).save(stream, format="PNG")
    save_image(decode_image(base64.b64encode(stream.getvalue()).decode()), tmp_path / "gray.png")
    with Image.open(tmp_path / "gray.png") as result:
        assert result.getpixel((0, 0))[0] in (127, 128)


@pytest.mark.parametrize("duration,segments,frames", [(2, 1, 50), (3.72, 1, 93), (3.73, 1, 93),
    (3.82, 1, 96), (3.83, 2, 96), (4.1, 2, 103), (6.92, 2, 173), (10.12, 3, 253), (10.14, 3, 254), (60, 19, 1500)])
def test_segment_arithmetic(duration, segments, frames):
    assert segment_count(duration) == segments and frame_count(duration) == frames
    assert 93 + (segments - 1) * 80 + 3 >= frames


def test_fingerprint_covers_all_content_but_not_request(tmp_path):
    clip = validate_start(payload(), tmp_path)
    assert replace(clip, request="a" * 32).fingerprint() == clip.fingerprint()
    for key, value in {"mode": "other", "prompt": "other", "size": "720p", "seed": 1,
                       "image": replace(clip.image, data=b"other"),
                       "audio": replace(clip.audio, data=b"other")}.items():
        assert replace(clip, **{key: value}).fingerprint() != clip.fingerprint()


def test_probe_refuses_network_protocols(tmp_path, monkeypatch):
    original = subprocess.run
    def run(args, **kwargs):
        if args[0] == "ffprobe":
            assert args[args.index("-protocol_whitelist") + 1] == "file,pipe"
        return original(args, **kwargs)
    monkeypatch.setattr(contract.subprocess, "run", run)
    assert validate_start(payload(), tmp_path).audio.duration == 2
