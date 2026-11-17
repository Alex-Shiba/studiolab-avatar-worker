import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys

import psutil
import pytest

from contract import frame_count
from engine_process import EngineBackend, GenerationFailed, engine_command, engine_environment, gpu_group, live
from media import encode_frames, video_info
from conftest import audio_bytes

FAKE = Path(__file__).with_name("fake_engine.py").resolve()
FAKE_BACKENDS = []


def fake_command(python, directory, gpus):
    return [sys.executable, str(FAKE), "--inbox", str(directory)]


def fake_backend(**kwargs):
    backend = EngineBackend(gpus=2, command=fake_command, **kwargs)
    FAKE_BACKENDS.append(backend)
    return backend


def job_files(directory, duration=2):
    directory.mkdir(exist_ok=True)
    spec = {"frames": frame_count(duration), "segments": 1, "duration": duration,
            "prompt": "Synthetic test", "size": "480p", "seed": 1}
    (directory / "audio.input").write_bytes(audio_bytes(duration))
    (directory / "job.json").write_text(json.dumps(spec))
    return spec


@pytest.mark.parametrize("count,gpus", [(1, 1), (2, 2), (3, 2), (4, 4), (5, 4), (7, 4), (8, 8), (12, 8)])
def test_gpu_group(count, gpus):
    assert gpu_group(count) == gpus


@pytest.mark.parametrize("count", [0, -1, True, "2"])
def test_no_gpu(count):
    with pytest.raises(RuntimeError):
        gpu_group(count)


def test_command_and_cudnn_environment():
    python = Path("/workspace/avatar/venv-py310/bin/python")
    env = {"LD_LIBRARY_PATH": "/system/cuda", "CONTAINER_API_KEY": "synthetic-secret",
           "HF_TOKEN": "synthetic-token", "CUDA_VISIBLE_DEVICES": "GPU-abc,GPU-def",
           "LONGCAT_A2A_WORKSPACE": "99"}
    for gpus in (1, 2, 4, 8):
        result = engine_environment(env, python, gpus)
        assert result["LD_LIBRARY_PATH"].split(":")[0].replace("\\", "/").endswith(
            "/venv-py310/lib/python3.10/site-packages/nvidia/cudnn/lib")
        assert result["LD_LIBRARY_PATH"].endswith(":/system/cuda")
        assert "HF_TOKEN" not in result and "CONTAINER_API_KEY" not in result
        assert result["CUDA_VISIBLE_DEVICES"] == "GPU-abc,GPU-def"
        assert result.get("LONGCAT_A2A_WORKSPACE") == ("1" if gpus > 2 else None)
        args = engine_command(python, Path("/job"), gpus)
        assert f"--nproc_per_node={gpus}" in args and f"--context_parallel_size={gpus}" in args
        assert Path(args[0]).name == "torchrun" and "avatar_engine.py" in args[3]


async def test_fake_process_progress_and_real_mp4(tmp_path):
    spec = job_files(tmp_path)
    backend = fake_backend()
    updates = []
    output = await backend.render(spec, tmp_path, updates.append)
    assert output == tmp_path / "result.mp4"
    assert video_info(output) == {"width": 32, "height": 24, "frames": 50}
    assert updates[0]["stage"] == "loading" and any(v.get("stage") == "segment 1/1" for v in updates)
    assert backend.healthy and backend.process is not None
    await backend.close()


async def test_timeout_kills_process_group_and_next_job_recovers(tmp_path):
    spec = job_files(tmp_path)
    backend = fake_backend(env={**os.environ, "AVATAR_TEST_MODE": "tree"}, timeout=1.5)
    with pytest.raises(GenerationFailed, match="timed out"):
        await backend.render(spec, tmp_path, lambda v: None)
    pids = json.loads((tmp_path / "pids.json").read_text())
    assert len(pids) == 2
    assert not any(psutil.pid_exists(pid) and live(psutil.Process(pid)) for pid in pids)
    assert not backend.healthy and backend.process is None
    backend.env["AVATAR_TEST_MODE"] = "normal"
    backend.timeout = 15
    await backend.restart()
    assert (await backend.render(spec, tmp_path, lambda v: None)).is_file()
    await backend.close()


async def test_cancel_kills_child_tree(tmp_path):
    spec = job_files(tmp_path)
    backend = fake_backend(env={**os.environ, "AVATAR_TEST_MODE": "tree"})
    task = asyncio.create_task(backend.render(spec, tmp_path, lambda v: None))
    for _ in range(200):
        if (tmp_path / "pids.json").exists():
            break
        await asyncio.sleep(.02)
    pids = json.loads((tmp_path / "pids.json").read_text())
    await asyncio.sleep(.15)  # Let the supervisor remember the descendant on Windows.
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not any(psutil.pid_exists(pid) and live(psutil.Process(pid)) for pid in pids)
    assert backend.tree_safe
    await backend.close()


def test_untrusted_progress_is_bounded(tmp_path):
    backend = fake_backend()
    spec = {"segments": 3}
    updates = []
    for value in [{"completed": 99, "segments": 3, "stage": "muxing"},
                  {"completed": 1, "segments": 3, "stage": "private prompt"},
                  {"completed": True, "segments": 3, "stage": "loading"},
                  {"completed": 1, "segments": 3, "stage": {}}, []]:
        (tmp_path / "progress.json").write_text(json.dumps(value))
        backend.progress(tmp_path, spec, updates.append)
    assert updates == []
    (tmp_path / "progress.json").write_text(json.dumps({"completed": 1, "segments": 3,
                                                       "stage": "segment 1/3", "width": "bad", "height": -1}))
    backend.progress(tmp_path, spec, updates.append)
    assert updates == [{"progress": 1 / 3, "stage": "segment 1/3", "gpus": 2}]


@pytest.mark.parametrize("duration", [2, 3.73, 4.137])
def test_mux_trims_video_preserves_original_voice(tmp_path, duration):
    original = tmp_path / "original.wav"
    original.write_bytes(audio_bytes(duration, frequency=440))
    # A deliberately different hypothetical separator output must never be used.
    (tmp_path / "separated.wav").write_bytes(audio_bytes(duration, frequency=900))
    count = frame_count(duration)
    consumed = []
    def frames():
        for i in range(count + 80):
            consumed.append(i)
            yield bytes((120, 60, 30)) * (32 * 24)
    output = encode_frames(frames(), original, tmp_path, count, 32, 24)
    assert len(consumed) == count and video_info(output)["frames"] == count
    # Compare AAC packets against independently transcoding the ORIGINAL audio.
    expected = tmp_path / "expected.m4a"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(original), "-c:a", "aac", "-b:a", "192k",
                    "-af", "asetpts=PTS-STARTPTS", str(expected)], check=True, timeout=20)
    def audio_packets(path):
        return subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0",
                               "-c:a", "copy", "-f", "adts", "pipe:1"],
                              capture_output=True, check=True, timeout=20).stdout
    assert audio_packets(output) == audio_packets(expected)
    data = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(output)],
                                     capture_output=True, check=True).stdout)
    audio = next(s for s in data["streams"] if s["codec_type"] == "audio")
    assert float(audio["duration"]) == pytest.approx(duration, abs=.005)
