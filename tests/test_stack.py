import ast
import hashlib
import io
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

import boot
import download
import exam
from build_manifest import REPOSITORIES, selected
from diagnostics import Redactor

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_matches_every_selected_hub_entry():
    data = download.manifest()
    assert data["code"] == {"repository": "https://github.com/meituan-longcat/LongCat-Video",
                             "commit": "58eb336edc5c8c63389838727e40b8c18434e54e", "license": "MIT"}
    assert len(data["files"]) == 44
    assert data["total_bytes"] == sum(f["size"] for f in data["files"]) == 63163373768
    paths = set()
    for name, revision, evidence in REPOSITORIES:
        tree = json.loads((ROOT / "evidence" / evidence).read_text())
        selected_files = [item for item in tree if item["type"] == "file" and selected(name, item["path"])]
        actual = [item for item in data["files"] if item["repository"] == "meituan-longcat/" + name]
        assert {f["source_path"] for f in actual} == {f["path"] for f in selected_files}
        for item in actual:
            source = next(f for f in selected_files if f["path"] == item["source_path"])
            assert item["revision"] == revision and re.fullmatch(r"[0-9a-f]{40}", revision)
            assert item["path"] == name + "/" + item["source_path"]
            assert item["size"] == source["size"] > 0
            assert re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
            assert ".." not in Path(item["path"]).parts and not Path(item["path"]).is_absolute()
            if "lfs" in source:
                assert item["sha256"] == source["lfs"]["oid"]
            assert item["path"] not in paths
            paths.add(item["path"])
    assert not any("int8" in p or "fp32" in p or "pytorch_model.bin" in p or "flax" in p for p in paths)


def test_download_uses_pinned_sources_and_layout(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(download, "download_file", lambda *a, **k: calls.append(a))
    download.download_models(tmp_path, emit=lambda line: None)
    assert calls == [(f"https://huggingface.co/{f['repository']}/resolve/{f['revision']}/{f['source_path']}",
                      tmp_path / f["path"], f["size"], f["sha256"]) for f in download.manifest()["files"]]


def test_exam_hard_refusal_defaults_and_real_largest_weight(tmp_path):
    assert exam.NEED == {"ram_gb": 110.0, "disk_free_gb": 180.0, "boot_download_s": 1200.0}
    assert exam.verdict({"ram_gb": 109.9}, {}) == "memory"
    assert exam.verdict({"disk_free_gb": 179.9}, {}) == "disk"
    assert exam.verdict({"ram_gb": 110, "disk_free_gb": 180}, {}) is None
    seen = []
    def speed(url, size):
        seen.append((url, size))
        return 40
    facts = exam.measure(tmp_path, download.manifest(), tmp_path / "missing", env={}, probes={
        "identity": lambda env: {}, "memory": lambda: 128,
        "disk": lambda folder: 500, "download": speed, "upload": lambda: None})
    largest = max(download.manifest()["files"], key=lambda f: f["size"])
    assert seen == [(exam.source_url(download.manifest(), largest), largest["size"])]
    assert exam.verdict({**facts, "disk_free_gb": 200}, {}) == "download"


def test_filter_retains_pinned_upstream_requirements():
    result = "".join(boot.filtered_requirements((ROOT / "evidence" / name).read_text())
                     for name in ("requirements.txt", "requirements_avatar.txt"))
    assert "diffusers==0.35.1" in result and "audio-separator==0.30.2" in result
    assert "onnxruntime-gpu==1.20.2" in result
    assert not any(line.split("==")[0] in boot.EXCLUDED for line in result.splitlines())
    assert boot.filtered_requirements("torch>=2\nflash_attn==2\n# comment\nopenai\nfoo==1\n") == "foo==1\n"


def test_install_plan_verified_commit_and_binary_flash_only(tmp_path, monkeypatch):
    monkeypatch.setattr(boot, "WORKSPACE", tmp_path)
    monkeypatch.setattr(boot.shutil, "which", lambda name: "/usr/bin/" + name)
    import sys
    monkeypatch.setattr(sys, "platform", "linux")
    source = tmp_path / "studiolab-avatar/LongCat-Video"
    source.mkdir(parents=True)
    for name in ("requirements.txt", "requirements_avatar.txt"):
        (source / name).write_text((ROOT / "evidence" / name).read_text())
    commands, stages = [], []
    def command(args, **kwargs):
        commands.append([str(a) for a in args])
        return download.manifest()["code"]["commit"] if "rev-parse" in args else ""
    monkeypatch.setattr(boot, "command", command)
    monkeypatch.setattr(boot, "download_models", lambda path: stages.append(("models", path)))
    actual_source, python, runtime = boot.prepare(stages.append)
    assert actual_source == source and python.name == "python" and (runtime / "jobs").is_dir()
    assert stages == ["install", "download", ("models", source.parent / "weights")]
    torch = next(c for c in commands if "torch==2.6.0+cu124" in c)
    assert torch[torch.index("--index-url") + 1] == "https://download.pytorch.org/whl/cu124"
    assert "torchvision==0.21.0+cu124" in torch and "torchaudio==2.6.0+cu124" in torch
    flash = next(c for c in commands if boot.FLASH_WHEEL in c)
    assert "--only-binary=:all:" in flash and "--no-deps" in flash
    assert "cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" in flash[-1]
    assert any("huggingface_hub==0.35.3" in c and "accelerate==1.10.1" in c for c in commands)
    assert any(c[-1] == "import sys; assert sys.version_info[:2] == (3, 10)" for c in commands)


def test_install_commit_mismatch_refuses_download(tmp_path, monkeypatch):
    monkeypatch.setattr(boot, "WORKSPACE", tmp_path)
    import sys
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(boot.shutil, "which", lambda name: name)
    monkeypatch.setattr(boot, "command", lambda *a, **k: "wrong")
    monkeypatch.setattr(boot, "download_models", lambda *a: pytest.fail("download before verified commit"))
    with pytest.raises(RuntimeError, match="commit mismatch"):
        boot.prepare()


def test_readiness_checks_health_and_live_children():
    class Response(io.BytesIO):
        status = 200
    now, calls = [0], []
    def opener(url, timeout):
        calls.append(url)
        return Response(b'{"status":"ready"}' if len(calls) == 2 else b'{"status":"unavailable"}')
    boot.wait_ready("unused", [SimpleNamespace(poll=lambda: None)], opener=opener,
                    clock=lambda: now[0], sleep=lambda n: now.__setitem__(0, now[0] + n))
    assert len(calls) == 2
    with pytest.raises(RuntimeError, match="exited"):
        boot.wait_ready("unused", [SimpleNamespace(poll=lambda: 1)])


def test_engine_import_does_not_import_gpu_stack_and_matches_upstream_negative_prompt():
    # Parse source, never import or run torch. AST also checks Python 3.10 syntax.
    source = (ROOT / "avatar_engine.py").read_text()
    tree = ast.parse(source, feature_version=(3, 10))
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            assert "torch" not in ast.unparse(node) and "longcat_video" not in ast.unparse(node)
    assignments = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
                   if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    upstream = ast.parse((ROOT / "evidence/demo.py").read_text())
    negative = next(node.value.value for node in ast.walk(upstream) if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "negative_prompt" for t in node.targets))
    # Upstream text first, then only our text words (live fake-caption finding, 2026-11-17).
    assert assignments["NEGATIVE_PROMPT"] == negative + ", text, captions, letters, watermark, logo"
    assert assignments["MODEL_TYPE"] == "avatar-v1.6"
    assert assignments["MIX_STEPS"] == 4 and assignments["MIX_ALPHA"] == 1.0 and assignments["INFERENCE_STEPS"] == 8
    assert "save_video_ffmpeg" not in source and "job[\"seed\"]" in source


def test_short_credential_values_are_redacted_too():
    redact = Redactor(["abc", "private-long-key"])
    text = redact.feed("a") + redact.feed("bc private-long-") + redact.feed("key", final=True)
    assert text == "[REDACTED] [REDACTED]"
