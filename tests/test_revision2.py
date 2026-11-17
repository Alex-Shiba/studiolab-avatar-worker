import asyncio
import ast
import base64
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import psutil
import pytest

from acceleration import attention_overrides, compile_dit, select_attention, shim_fa3
import avatar_engine
import boot
from contract import Invalid, validate_start
from dialog import pad_speeches, pixel_boxes, target_masks
from download import manifest
from engine_process import GenerationFailed
from media import bounded_frames, encode_frames, mix_audio, video_info
from resident_protocol import atomic_json, replace_file, wait_job
from conftest import audio_bytes, payload
from test_engine import fake_backend, job_files

ROOT = Path(__file__).resolve().parents[1]


def dialog_payload(**changes):
    body = payload(mode="dialog")
    body["audio1"] = body.pop("audio")
    body["audio2"] = base64.b64encode(audio_bytes(3.82, frequency=700)).decode()
    return {**body, **changes}


def test_dialog_duration_files_and_fingerprint(tmp_path):
    body = dialog_payload(boxes={"person1": [0, 0, .5, 1], "person2": [.5, 0, 1, 1]})
    clip = validate_start(body, tmp_path)
    assert clip.mode == "dialog" and clip.audio is None
    assert clip.audio1.duration == 2 and clip.duration == 3.82
    assert (clip.frames, clip.segments) == (96, 1)
    for key in ("audio1", "audio2"):
        assert (tmp_path / (key + ".input")).read_bytes() == base64.b64decode(body[key])
    assert not (tmp_path / "audio.input").exists()
    assert replace(clip, request="a" * 32).fingerprint() == clip.fingerprint()
    for key, value in {"prompt": "changed", "size": "720p", "seed": 99,
                       "image": replace(clip.image, data=b"other"), "boxes": None,
                       "audio1": replace(clip.audio1, data=b"other"),
                       "audio2": replace(clip.audio2, data=b"other")}.items():
        assert replace(clip, **{key: value}).fingerprint() != clip.fingerprint()
    # Numeric spellings and object key order do not change box semantics.
    equivalent = {"person2": [.5, 0.0, 1.0, 1.0], "person1": [0.0, 0.0, .5, 1.0]}
    assert validate_start({**body, "boxes": equivalent}, tmp_path).fingerprint() == clip.fingerprint()


@pytest.mark.parametrize("key", ["request", "mode", "prompt", "size", "seed", "image", "audio1", "audio2"])
def test_dialog_missing_keys(tmp_path, key):
    body = dialog_payload()
    del body[key]
    with pytest.raises(Invalid):
        validate_start(body, tmp_path)


@pytest.mark.parametrize("changes", [{"audio": "unexpected"}, {"audio_type": "add"}, {"audio1": "%%%"},
                                     {"audio2": "%%%"}, {"boxes": None}, {"boxes": {}},
                                     {"boxes": {"person1": [0, 0, 1, 1]}}])
def test_dialog_invalid_keys_and_tracks(tmp_path, changes):
    with pytest.raises(Invalid):
        validate_start(dialog_payload(**changes), tmp_path)


@pytest.mark.parametrize("box", [[0, 0, 0, 1], [1, 0, 0, 1], [0, .5, 1, .5], [-.01, 0, 1, 1],
                                [0, 0, 1.01, 1], [0, 0, 1], [False, 0, 1, 1], [0, 0, float('nan'), 1],
                                [0, 0, float('inf'), 1], [0, 0, 10 ** 400, 1], [0, 0, "1", 1], "box"])
def test_dialog_box_bounds(tmp_path, box):
    with pytest.raises(Invalid, match="boxes"):
        validate_start(dialog_payload(boxes={"person1": box, "person2": [0, 0, 1, 1]}), tmp_path)


def test_avatar_rejects_dialog_fields(tmp_path):
    for key in ("boxes", "audio1", "audio2"):
        with pytest.raises(Invalid, match="keys"):
            validate_start(payload(**{key: None}), tmp_path)


def test_padding_keeps_longer_track_and_silence():
    left, right = pad_speeches([np.array([1., 2.]), np.array([3., 4., 5.])], 1, 3)
    np.testing.assert_array_equal(left, [1, 2, 0])
    np.testing.assert_array_equal(right, [3, 4, 5])
    left, right = pad_speeches([left, right], 1, 4)
    np.testing.assert_array_equal(left, [1, 2, 0, 0])
    np.testing.assert_array_equal(right, [3, 4, 5, 0])


def test_pixel_boxes_default_and_fractional_order():
    assert pixel_boxes(None, 100, 80) == [[8, 5, 72, 45], [8, 55, 72, 95]]
    assert pixel_boxes({"person1": [.1, .2, .6, .7], "person2": [.9, .9, 1, 1]}, 100, 80) == [
        [16, 10, 56, 60], [72, 90, 80, 100]]
    # An arbitrarily small valid fractional box still covers a pixel.
    assert pixel_boxes({"person1": [.001, .001, .002, .002], "person2": [0, 0, 1, 1]}, 10, 10)[0] == [0, 0, 1, 1]


def test_background_mask_matches_upstream_without_others():
    class Array(np.ndarray):
        def to(self, _):
            return self
    torch = SimpleNamespace(zeros=lambda shape: np.zeros(shape).view(Array),
                            stack=lambda arrays, dim: np.stack(arrays, axis=dim).view(Array))
    masks = target_masks(torch, None, 100, 80, "fake")
    assert masks.shape == (3, 80, 100)
    assert masks[0, 40, 20] == 1 and masks[1, 40, 70] == 1 and masks[2, 0, 0] == 1
    assert np.all(masks.sum(axis=0) == 1)


@pytest.mark.parametrize("missing", [0, 1, 2, 3])
def test_tail_holds_only_last_frame(missing):
    assert list(bounded_frames([b"a", b"b"], 2 + missing)) == [b"a", b"b"] + [b"b"] * missing
    assert list(bounded_frames([b"a", b"b", b"c"], 2)) == [b"a", b"b"]


@pytest.mark.parametrize("frames,count", [([], 1), ([b"a"], 5)])
def test_tail_refuses_large_gap(frames, count):
    with pytest.raises(ValueError, match="three frames"):
        list(bounded_frames(frames, count))


def test_tail_real_mux(tmp_path):
    original = tmp_path / "original.wav"
    original.write_bytes(audio_bytes(3.82))
    frame = bytes((40, 90, 110)) * (32 * 24)
    result = encode_frames([frame] * 93, original, tmp_path, 96, 32, 24)
    assert video_info(result)["frames"] == 96


def read_samples(path):
    return np.frombuffer(subprocess.run([
        "ffmpeg", "-v", "error", "-i", str(path), "-ar", "16000", "-ac", "1", "-f", "f32le", "pipe:1"],
        capture_output=True, check=True, timeout=20).stdout, dtype="<f4")


def test_mix_sums_originals_and_pads_shorter_without_normalizing(tmp_path):
    first, second = tmp_path / "a.wav", tmp_path / "b.wav"
    first.write_bytes(audio_bytes(2, frequency=440))
    second.write_bytes(audio_bytes(3, frequency=700))
    a, b = read_samples(first), read_samples(second)
    result = read_samples(mix_audio(first, second, tmp_path, 3))
    assert len(result) == len(b)
    expected = np.pad(a, (0, len(b) - len(a))) + b
    # Resampling filter boundary transients aside, preserve both tracks and the longer tail.
    np.testing.assert_allclose(result[100:31000], expected[100:31000], atol=.001)
    np.testing.assert_allclose(result[33000:-100], b[33000:-100], atol=.001)


def test_mix_limits_loud_sum(tmp_path):
    import wave
    path = tmp_path / "loud.wav"
    samples = (.9 * np.sin(2 * np.pi * 440 * np.arange(32000) / 16000) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        out.writeframes(samples.tobytes())
    mixed = read_samples(mix_audio(path, path, tmp_path, 2))
    assert len(mixed) == 32000 and np.max(np.abs(mixed)) < 1
    assert np.max(np.abs(mixed)) > .9


def test_fa3_shim_preserves_tuples_and_is_idempotent():
    calls = []
    tensor = object()
    interface = SimpleNamespace(flash_attn_func=lambda *a, **k: (calls.append((a, k)) or tensor),
                                flash_attn_varlen_func=lambda *a, **k: (tensor, "lse"))
    shim_fa3(interface)
    wrapper = interface.flash_attn_func
    shim_fa3(interface)
    assert interface.flash_attn_func is wrapper
    assert interface.flash_attn_func(1, scale=2) == (tensor, None)
    assert interface.flash_attn_varlen_func() == (tensor, "lse")
    assert calls == [((1,), {"scale": 2})]


@pytest.mark.parametrize("capability", [(8, 0), (8, 9), (10, 0)])
def test_fa3_only_hopper(capability, capsys):
    assert select_attention(capability, importer=lambda _: pytest.fail("must not import FA3")) == "fa2"
    assert "AVATAR_FA3 off compute_capability" in capsys.readouterr().out


def test_fa3_import_and_failed_build_fallback(capsys):
    assert select_attention((9, 0), installed=False, importer=lambda _: pytest.fail("stale package")) == "fa2"
    def fail(_):
        raise ImportError("private path")
    assert select_attention((9, 0), importer=fail) == "fa2"
    interface = SimpleNamespace(flash_attn_func=lambda: 1, flash_attn_varlen_func=lambda: 2)
    assert select_attention((9, 0), importer=lambda _: interface) == "fa3"
    assert interface.flash_attn_func() == (1, None)
    assert attention_overrides("fa3") == {"enable_flashattn3": True, "enable_flashattn2": False}
    assert attention_overrides("fa2") == {"enable_flashattn3": False, "enable_flashattn2": True}
    assert "private path" not in capsys.readouterr().out


def test_compile_uses_pinned_recipe_and_never_vae():
    calls = []
    torch = SimpleNamespace(_dynamo=SimpleNamespace(config=SimpleNamespace()),
                            compile=lambda block, **kw: (calls.append((block, kw)) or ("compiled", block)))
    dit = SimpleNamespace(blocks=["a", "b"])
    assert compile_dit(torch, dit, False) is False and not calls
    assert compile_dit(torch, dit) is True
    assert dit.blocks == [("compiled", "a"), ("compiled", "b")]
    assert torch._dynamo.config.cache_size_limit == 128
    assert torch._dynamo.config.accumulated_cache_size_limit == 1024
    assert calls[0][1] == {"fullgraph": False, "dynamic": False,
                           "options": {"triton.cudagraphs": False, "epilogue_fusion": False,
                                       "emulate_precision_casts": True}}


@pytest.mark.parametrize("cores,ram,expected", [(224, 512, 64), (32, 512, 32), (224, 128, 30), (8, 24, 4), (1, 4, 1)])
def test_fa3_parallelism(cores, ram, expected):
    assert boot.fa3_jobs(cores, ram * 1024 ** 3) == expected


@pytest.mark.parametrize("fail_build", [False, True])
def test_fa3_install_pin_flags_isolation_and_fallback(tmp_path, monkeypatch, capsys, fail_build):
    calls = []
    pin = manifest()["flash_attention"]
    assert pin["commit"] == "060c9188beec3a8b62b33a3bfa6d5d2d44975fab"
    monkeypatch.setenv("FLASH_ATTENTION_DISABLE_HDIM128", "TRUE")
    monkeypatch.setenv("VAST_API_KEY", "synthetic-key")
    def command(args, **kwargs):
        calls.append((args, kwargs))
        if "setup.py" in args and fail_build:
            raise boot.CommandFailed("python setup.py", 1)
        return pin["commit"] if "rev-parse" in args else ""
    monkeypatch.setattr(boot, "command", command)
    assert boot.install_fa3(tmp_path, Path("/venv/bin/python")) is not fail_build
    build = next(c for c in calls if "setup.py" in c[0])
    assert build[0] == [Path("/venv/bin/python"), "setup.py", "install"]
    assert build[1]["cwd"] == tmp_path / "flash-attention/hopper"
    assert build[1]["venv_bin"] == Path("/venv/bin")
    env = boot.install_environment(build[1]["env"])
    expected = {"BACKWARD", "SPLIT", "PAGEDKV", "APPENDKV", "LOCAL", "SOFTCAP", "FP16", "FP8",
                "HDIM64", "HDIM96", "HDIM192", "HDIM256"}
    assert {k.removeprefix("FLASH_ATTENTION_DISABLE_") for k in env if k.startswith("FLASH_ATTENTION_")} == expected
    assert all(env["FLASH_ATTENTION_DISABLE_" + name] == "TRUE" for name in expected)
    assert "VAST_API_KEY" not in env
    assert json.loads((tmp_path / "fa3-status.json").read_text())["available"] is not fail_build
    assert any(pin["commit"] in args for args, _ in calls)
    if fail_build:
        assert "AVATAR_FA3 off CommandFailed" in capsys.readouterr().out


def test_build_command_finds_worker_venv_ninja_after_sdk_isolation(monkeypatch):
    calls = []
    monkeypatch.setattr(boot.subprocess, "run", lambda argv, **kw: (
        calls.append(kw) or SimpleNamespace(returncode=0, stdout=b"", stderr=b"")))
    boot.command(["python", "setup.py", "install"], venv_bin="/worker/venv/bin",
                 env={"PATH": "/sdk/venv/bin", "PYTHONPATH": "/sdk/site", "VAST_API_KEY": "fake"})
    env = calls[0]["env"]
    assert env["PATH"].split(os.pathsep)[0] == "/worker/venv/bin"
    assert "/sdk/venv/bin" not in env["PATH"] and "PYTHONPATH" not in env and "VAST_API_KEY" not in env


def test_inbox_atomic_publish_and_idle_wait(tmp_path, monkeypatch):
    sleeps = []
    (tmp_path / "job.pending").write_text("partial")
    original = Path.replace
    def rename(path, target):
        if path.name == "job.pending":
            assert not Path(target).exists()
            assert json.loads(path.read_text()) == {"token": "test"}
        return original(path, target)
    monkeypatch.setattr(Path, "replace", rename)
    def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 3:
            atomic_json(tmp_path / "job.json", {"token": "test"})
    assert wait_job(tmp_path, sleep=sleep) == {"token": "test"}
    assert sleeps == [.25] * 3
    assert not (tmp_path / "job.json").exists() and (tmp_path / "active.json").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows file sharing semantics")
def test_atomic_replace_retries_windows_readers(tmp_path, monkeypatch):
    source, target = tmp_path / "pending", tmp_path / "ready"
    source.write_text("new")
    target.write_text("old")
    calls, sleeps = [], []
    original = Path.replace
    def replace(path, dest):
        calls.append(1)
        if len(calls) < 3:
            assert dest.read_text() == "old"
            raise PermissionError("reader")
        return original(path, dest)
    monkeypatch.setattr(Path, "replace", replace)
    replace_file(source, target, sleep=sleeps.append)
    assert target.read_text() == "new" and sleeps == [.01, .01]


async def test_ready_waits_for_models_and_resident_jobs_share_process(tmp_path):
    backend = fake_backend(env={**os.environ, "AVATAR_TEST_LOAD_DELAY": ".3"})
    task = asyncio.create_task(backend.start(tmp_path))
    try:
        await asyncio.sleep(.1)
        assert not backend.healthy and not task.done()
        await task
        pid = backend.process.pid
        for i in range(2):
            directory = tmp_path / str(i)
            spec = job_files(directory)
            await backend.render(spec, directory, lambda _: None)
            assert backend.process.pid == pid and backend.healthy
        process = psutil.Process(pid)
        before = process.cpu_times()
        await asyncio.sleep(.6)
        after = process.cpu_times()
        assert (after.user + after.system) - (before.user + before.system) < .15
    finally:
        await backend.close()


async def test_idle_crash_reloads_and_discards_stale_ready(tmp_path):
    backend = fake_backend(env={**os.environ, "AVATAR_TEST_LOAD_DELAY": ".2"})
    await backend.start(tmp_path)
    try:
        old = backend.process
        old.kill()
        await old.wait()
        assert not backend.healthy
        for _ in range(100):
            if backend.healthy and backend.process.pid != old.pid:
                break
            await asyncio.sleep(.05)
        assert backend.healthy and backend.process.pid != old.pid
        directory = tmp_path / "job"
        await backend.render(job_files(directory), directory, lambda _: None)
    finally:
        await backend.close()


async def test_one_job_at_a_time_and_crash_fails_only_inflight(tmp_path):
    gate = tmp_path / "gate"
    backend = fake_backend(env={**os.environ, "AVATAR_TEST_GATE": str(gate)})
    directory = tmp_path / "first"
    spec = job_files(directory)
    task = asyncio.create_task(backend.render(spec, directory, lambda _: None))
    try:
        for _ in range(100):
            if (directory / "entered").exists():
                break
            await asyncio.sleep(.05)
        assert (directory / "entered").exists()
        with pytest.raises(GenerationFailed, match="unavailable"):
            await backend.render(spec, directory, lambda _: None)
        old = backend.process.pid
        backend.process.kill()
        with pytest.raises(GenerationFailed, match="generation failed"):
            await task
        await asyncio.wait_for(backend.wait_ready(), 5)
        assert backend.healthy and backend.process.pid != old
        gate.touch()
        second = tmp_path / "second"
        await backend.render(job_files(second), second, lambda _: None)
    finally:
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        await backend.close()


async def test_load_timeout_kills_unready_engine(tmp_path):
    backend = fake_backend(env={**os.environ, "AVATAR_TEST_LOAD_DELAY": "120"}, startup_timeout=.2)
    with pytest.raises(GenerationFailed, match="load timed out"):
        await backend.start(tmp_path)
    assert not backend.healthy and backend.process is None and backend.tree_safe
    await backend.close()


async def test_server_shutdown_during_model_loading_kills_engine(tmp_path, monkeypatch):
    import server
    backend = fake_backend(env={**os.environ, "AVATAR_TEST_LOAD_DELAY": "120"})
    monkeypatch.setattr(server, "EngineBackend", lambda: backend)
    monkeypatch.setenv("AVATAR_JOB_ROOT", str(tmp_path / "jobs"))
    # Exercise cancellation on every OS without installing process-wide handlers in pytest.
    monkeypatch.setattr(asyncio.get_running_loop(), "add_signal_handler", lambda *args: None)
    task = asyncio.create_task(server.serve())
    for _ in range(100):
        if backend.process is not None:
            break
        await asyncio.sleep(.02)
    assert backend.process is not None and not backend.healthy
    pid = backend.process.pid
    task.cancel()
    await asyncio.wait_for(task, 5)
    assert backend.process is None and backend.tree_safe
    assert not psutil.pid_exists(pid)


def test_rank_zero_loads_once_and_broadcasts_each_job_on_gloo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    control = object()
    calls = []
    runtime = SimpleNamespace(global_rank=0, control=control, sampler=None,
                              state={"attention": "fa2", "compiled": False})
    def load(inbox, size):
        calls.append("loaded")
        return runtime
    def broadcast(message, src, group):
        assert src == 0 and group is control
        calls.append("broadcast")
    distributed = ModuleType("torch.distributed")
    distributed.broadcast_object_list = broadcast
    distributed.destroy_process_group = lambda: calls.append("destroyed")
    torch = ModuleType("torch")
    torch.distributed = distributed
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", distributed)
    monkeypatch.setattr(avatar_engine, "load_runtime", load)
    monkeypatch.setattr(avatar_engine.gpu_watch, "Sampler", lambda: SimpleNamespace(
        start=lambda: SimpleNamespace(stop=lambda: None)))
    monkeypatch.setattr(avatar_engine, "generate", lambda directory, job, runtime: calls.append(job["id"]))
    def next_job(inbox):
        assert (inbox / "ready.json").exists()
        number = calls.count("broadcast")
        if number == 2:
            raise StopIteration
        directory = tmp_path / str(number)
        directory.mkdir()
        return {"directory": str(directory), "job": {"id": number}, "token": str(number)}
    monkeypatch.setattr(avatar_engine, "wait_job", next_job)
    with pytest.raises(StopIteration):
        avatar_engine.run(tmp_path, 2)
    assert calls == ["loaded", "broadcast", 0, "broadcast", 1, "destroyed"]
    assert json.loads((tmp_path / "1/completion.json").read_text())["token"] == "1"


def test_pinned_model_accepts_attention_config_and_upstream_recipe():
    source = (ROOT / "evidence/upstream-dit.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LongCatVideoAvatarTransformer3DModel")
    assert [b.id for b in model.bases] == ["ModelMixin", "ConfigMixin"]
    assert not any(isinstance(n, ast.FunctionDef) and n.name == "from_pretrained" for n in model.body)
    init = next(n for n in model.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    assert "register_to_config" in [d.id for d in init.decorator_list]
    assert {"enable_flashattn3", "enable_flashattn2"} <= {a.arg for a in init.args.args}
    engine = (ROOT / "avatar_engine.py").read_text(encoding="utf-8")
    assert engine.index("select_attention(torch.cuda") < engine.index("from longcat_video")
    assert '**attention_overrides(attention)' in engine
    assert 'os.environ.get("AVATAR_COMPILE", "1") != "0"' in engine
    assert 'backend="gloo", timeout=datetime.timedelta(days=7)' in engine
    assert 'backend="nccl", timeout=datetime.timedelta(seconds=3600)' in engine
    upstream = (ROOT / "evidence/upstream-multi-demo.py").read_text(encoding="utf-8")
    assert "audio_type = 'para'" in upstream and "ref_target_masks=ref_target_masks" in upstream
