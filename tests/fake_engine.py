"""Subprocess stand-in: progress, real CPU MP4, or a hung process tree."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from media import encode_frames, write_progress


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    directory = args.job
    spec = json.loads((directory / "job.json").read_text())
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
        time.sleep(.15)
    write_progress(directory, spec["segments"], spec["segments"], stage="muxing", width=32, height=24)
    # More generated frames than the voice: the production writer must trim them.
    frame = bytes((25, 60, 90)) * (32 * 24)
    encode_frames(iter([frame] * (spec["frames"] + 80)), directory / "audio.input",
                  directory, spec["frames"], 32, 24)


if __name__ == "__main__":
    main()
