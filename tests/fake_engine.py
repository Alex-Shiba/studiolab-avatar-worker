"""Subprocess stand-in: progress, real CPU MP4, or a hung process tree."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from media import encode_frames, mix_audio, write_progress
from resident_protocol import atomic_json, wait_job


def generate(directory, spec):
    mode = os.environ.get("AVATAR_TEST_MODE", "normal")
    (directory / "entered").touch()
    if mode == "tree":
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        (directory / "pids.json").write_text(json.dumps([os.getpid(), child.pid]))
        time.sleep(120)
    if mode == "failure":
        raise SystemExit(1)
    gate = os.environ.get("AVATAR_TEST_GATE")
    if gate:
        while not Path(gate).exists():
            time.sleep(.02)
    for completed in range(1, spec["segments"] + 1):
        write_progress(directory, completed, spec["segments"], width=32, height=24)
        time.sleep(.4)  # longer than a busy test host's poll gap: the segment stage must be observable
    write_progress(directory, spec["segments"], spec["segments"], stage="muxing", width=32, height=24)
    # More generated frames than the voice: the production writer must trim them.
    frame = bytes((25, 60, 90)) * (32 * 24)
    audio = (mix_audio(directory / "audio1.input", directory / "audio2.input", directory, spec["duration"])
             if spec.get("mode") == "dialog" else directory / "audio.input")
    encode_frames(iter([frame] * (spec["frames"] + 80)), audio,
                  directory, spec["frames"], 32, 24)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inbox", type=Path, required=True)
    args = parser.parse_args()
    inbox = args.inbox
    time.sleep(float(os.environ.get("AVATAR_TEST_LOAD_DELAY", "0")))
    if os.environ.get("AVATAR_TEST_LOAD_CRASH") == "1":
        raise SystemExit(1)
    atomic_json(inbox / "ready.json", {"attention": "fa3", "compiled": True})
    while True:
        request = wait_job(inbox, interval=.05)
        directory = Path(request["directory"])
        generate(directory, request["job"])
        atomic_json(directory / "completion.json", {"token": request["token"]})
        (inbox / "active.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
