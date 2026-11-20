"""CPU-only final encoding: bounded video frames, then the original audio."""

import json
import os
from pathlib import Path
import subprocess

from contract import FPS
from diagnostics import model_environment
from resident_protocol import atomic_json


def bounded_frames(frames, count):
    """Trim excess frames or hold the last frame for at most three missing frames."""
    iterator = iter(frames)
    last, held = None, 0
    for _ in range(count):
        try:
            last = next(iterator)
        except StopIteration:
            held += 1
            if last is None or held > 3:
                raise ValueError("video tail exceeds three frames") from None
        yield last


MIX_RATE = 48000
MIX_PEAK = 0.95


def decode_track(path):
    """Plain decode to 48 kHz mono float samples from the track's first sample.

    Mono, as a single avatar keeps its mono voice: a stereo mix spread a mono voice over two channels 3 dB down, so a
    dialog sounded quieter than the same voice alone (live 2026-11-20)."""
    import numpy as np
    raw = subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-protocol_whitelist", "file,pipe", "-i", str(path), "-vn",
        "-ar", str(MIX_RATE), "-ac", "1", "-f", "f32le", "pipe:1"],
        stdin=subprocess.DEVNULL, capture_output=True, check=True, timeout=120, env=model_environment(os.environ)).stdout
    return np.frombuffer(raw, dtype="<f4")


def mix_audio(audio1, audio2, directory, duration):
    """Mix the original tracks from t=0, pad to the longer one, and keep the sum under MIX_PEAK.

    Summed in numpy: the image's ffmpeg is 4.4.2 (Ubuntu 22.04), whose alimiter has no `latency` option, and the
    filter graph that used it failed every dialog take (live 2026-11-20). ffmpeg here only decodes and writes.
    A sum louder than MIX_PEAK is turned down as a whole; a voice beside a silent track is left as it is.
    """
    import numpy as np
    output = Path(directory) / "mixed.wav"
    mix = np.zeros(round(duration * MIX_RATE), dtype=np.float32)
    for path in (audio1, audio2):
        track = decode_track(path)[:len(mix)]
        mix[:len(track)] += track
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > MIX_PEAK:
        mix *= MIX_PEAK / peak
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "f32le", "-ar", str(MIX_RATE), "-ac", "1", "-i", "pipe:0",
        "-c:a", "pcm_f32le", "-map_metadata", "-1", str(output)],
        input=mix.astype("<f4").tobytes(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=120,
        env=model_environment(os.environ))
    return output


def encode_frames(frames, original_audio, directory, count, width, height):
    directory = Path(directory)
    silent, output = directory / "silent.mp4", directory / "result.mp4"
    env = model_environment(os.environ)
    process = subprocess.Popen([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", f"{width}x{height}",
        "-r", str(FPS), "-i", "pipe:0", "-an", "-c:v", "libx264", "-crf", "16",
        "-preset", "fast", "-pix_fmt", "yuv420p", "-map_metadata", "-1", str(silent)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    try:
        for frame in bounded_frames(frames, count):
            if len(frame) != width * height * 3:
                raise ValueError("invalid RGB frame")
            process.stdin.write(frame)
        process.stdin.close()
        if process.wait(timeout=300) != 0:
            raise RuntimeError("video encoding failed")
        # No -shortest: the original voice must survive the rounding to video frames.
        # The separated stem is never passed to this function.
        subprocess.run([
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(silent), "-protocol_whitelist", "file,pipe", "-i", str(original_audio),
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-af", "asetpts=PTS-STARTPTS", "-map_metadata", "-1", "-movflags", "+faststart", str(output)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300, check=True, env=env)
        return output
    except BaseException:
        if process.poll() is None:
            process.kill()
        process.wait()
        output.unlink(missing_ok=True)
        raise
    finally:
        if not process.stdin.closed:
            process.stdin.close()
        silent.unlink(missing_ok=True)


def video_info(path):
    result = subprocess.run([
        "ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
        capture_output=True, timeout=20, check=True, env=model_environment(os.environ))
    streams = json.loads(result.stdout)["streams"]
    video = [s for s in streams if s["codec_type"] == "video"]
    audio = [s for s in streams if s["codec_type"] == "audio"]
    if (len(video) != 1 or len(audio) != 1 or video[0]["codec_name"] != "h264"
            or video[0]["pix_fmt"] != "yuv420p" or audio[0]["codec_name"] != "aac"
            or video[0]["r_frame_rate"] != "25/1"):
        raise RuntimeError("invalid result media")
    return {key: int(video[0][source]) for key, source in
            (("width", "width"), ("height", "height"), ("frames", "nb_frames"))}


def write_progress(directory, completed, segments, *, stage=None, width=None, height=None):
    value = {"completed": completed, "segments": segments,
             "stage": stage or f"segment {completed}/{segments}"}
    if width is not None and height is not None:
        value.update(width=width, height=height)
    atomic_json(Path(directory) / "progress.json", value)
