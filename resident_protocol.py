"""Private, single-slot file protocol shared by the GPU engine and CPU fakes."""

import json
import os
from pathlib import Path
import time


def atomic_json(path, value):
    path = Path(path)
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(value, ensure_ascii=True), encoding="utf-8")
    pending.chmod(0o600)
    replace_file(pending, path)


def replace_file(source, target, *, sleep=time.sleep):
    # Windows readers briefly deny replacement of an open file. Retrying keeps
    # publication atomic; never expose partially written JSON as a workaround.
    for attempt in range(101):
        try:
            source.replace(target)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 100:
                raise
            sleep(.01)


def wait_job(inbox, *, sleep=time.sleep, interval=.25):
    """Rank zero sleeps between checks; other ranks block on the gloo broadcast."""
    inbox = Path(inbox)
    path = inbox / "job.json"
    while True:
        try:
            # Rename claims only a fully published message. The server permits one in flight.
            path.replace(inbox / "active.json")
        except FileNotFoundError:
            sleep(interval)
            continue
        return json.loads((inbox / "active.json").read_text(encoding="utf-8"))
