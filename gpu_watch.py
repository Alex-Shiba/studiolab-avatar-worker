"""GPU facts for Vast's log, rank 0 only (2026-11-17).

A live 2×H100 SXM machine ran 2.4–3× slower than the trial's 2×H100 SXM box; the step time rose for the first ~100 s.
Two suspects: heat or power throttling, and GPU-to-GPU traffic that goes through host memory instead of NVLink/P2P.
These lines tell them apart: the topology, peer access and copy speed once per job, then clocks, power, temperature
and throttle reasons every 10 s while the engine runs. Lines carry numbers only.
"""

import subprocess
import threading
import time

QUERY = ("index,utilization.gpu,temperature.gpu,power.draw,power.limit,clocks.current.graphics,clocks.max.graphics,"
         "memory.used,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.sw_thermal_slowdown,"
         "clocks_event_reasons.hw_power_brake_slowdown,clocks_event_reasons.sw_power_cap")


def run(argv, timeout=20):
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def topology(emit=print):
    for line in run(["nvidia-smi", "topo", "-m"]).splitlines()[:12]:
        emit("AVATAR_GPU_TOPO " + line.rstrip()[:200])
    nvlink = run(["nvidia-smi", "nvlink", "-s"]).splitlines()
    emit("AVATAR_GPU_NVLINK lines=%d active=%d" % (len(nvlink), sum("GB/s" in line for line in nvlink)))


def peer_copy(torch, emit=print, gib=1, repeats=10):
    """GPU 0 → GPU 1 copy speed in GB/s and whether CUDA reports peer access (no model involved)."""
    if torch.cuda.device_count() < 2:
        return None
    access = torch.cuda.can_device_access_peer(0, 1)
    source = torch.empty(gib * (1 << 30), dtype=torch.uint8, device="cuda:0")
    target = torch.empty_like(source, device="cuda:1")
    target.copy_(source)
    torch.cuda.synchronize(0); torch.cuda.synchronize(1)
    started = time.perf_counter()
    for _ in range(repeats):
        target.copy_(source, non_blocking=True)
    torch.cuda.synchronize(0); torch.cuda.synchronize(1)
    speed = gib * repeats * (1 << 30) / (time.perf_counter() - started) / 1e9
    del source, target
    torch.cuda.empty_cache()
    emit("AVATAR_GPU_PEER access=%s copy_gb_s=%.1f" % (access, speed))
    return speed


class Sampler:
    """nvidia-smi every `every` seconds in a daemon thread until stop()."""

    def __init__(self, emit=print, every=10, query=run):
        self.emit, self.every, self.query = emit, every, query
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def loop(self):
        while not self.stopped.is_set():
            out = self.query(["nvidia-smi", "--query-gpu=" + QUERY, "--format=csv,noheader,nounits"])
            for line in out.splitlines():
                self.emit("AVATAR_GPU " + " ".join(line.split())[:240])
            self.stopped.wait(self.every)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.stopped.set()
