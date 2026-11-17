"""gpu_watch: numbers-only lines, never fails a job, NCCL transport logging for multi-GPU engines."""

import time

import engine_process
import gpu_watch


def test_sampler_emits_one_line_per_gpu_and_stops():
    lines = []
    fake = lambda argv: "0, 100, 81, 690.1, 700.0, 1755, 1980, 70000, Not Active, Not Active, Not Active, Active\n1, 99, 80, 688.0, 700.0, 1755, 1980, 69000, Not Active, Not Active, Not Active, Not Active"
    sampler = gpu_watch.Sampler(emit=lines.append, every=0.05, query=fake).start()
    time.sleep(0.12)
    sampler.stop()
    sampler.thread.join(1)
    assert not sampler.thread.is_alive()
    assert lines and all(line.startswith("AVATAR_GPU ") for line in lines)
    assert any(line.startswith("AVATAR_GPU 0, 100, 81") for line in lines) and any(line.startswith("AVATAR_GPU 1,") for line in lines)


def test_topology_tolerates_missing_nvidia_smi(monkeypatch):
    lines = []
    monkeypatch.setattr(gpu_watch, "run", lambda argv, timeout=20: "")
    gpu_watch.topology(emit=lines.append)
    assert lines == ["AVATAR_GPU_NVLINK lines=0 active=0"]


def test_nccl_transport_logging_only_with_several_gpus():
    one = engine_process.engine_environment({"PATH": "/usr/bin"}, "/w/venv/bin/python", 1)
    two = engine_process.engine_environment({"PATH": "/usr/bin"}, "/w/venv/bin/python", 2)
    assert "NCCL_DEBUG" not in one
    assert two["NCCL_DEBUG"] == "WARN" and "NCCL_DEBUG_SUBSYS" not in two
    loud = engine_process.engine_environment({"PATH": "/usr/bin", "AVATAR_NCCL_DEBUG": "1"}, "/w/venv/bin/python", 2)
    assert loud["NCCL_DEBUG"] == "INFO" and loud["NCCL_DEBUG_SUBSYS"] == "INIT,P2P,SHM,NET"
