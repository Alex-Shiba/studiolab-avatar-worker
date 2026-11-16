"""Refresh pins from Hub metadata; never fetch LFS objects or model weights."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
CODE_COMMIT = "58eb336edc5c8c63389838727e40b8c18434e54e"
REPOSITORIES = (
    ("LongCat-Video", "03b55529b1d1d4045f5fbe14d65c8c6e8116b278", "base-tree.json"),
    ("LongCat-Video-Avatar-1.6", "acfc628244201fc53fb6fe8a8903c5bac9bdf68f", "avatar-tree.json"),
)


def selected(name, path):
    if name == "LongCat-Video":
        return (path.startswith(("tokenizer/", "scheduler/"))
                or path.startswith(("text_encoder/", "vae/")) and path.endswith((".safetensors", ".json"))
                or "/" not in path and path.endswith(".json"))
    return (path.startswith(("base_model/", "loraA/", "loraB/", "scheduler/", "vocal_separator/"))
            or path in ("config.json", "model_index.json", "whisper-large-v3/model.safetensors")
            or path.startswith("whisper-large-v3/") and path.endswith((".json", ".txt"))
            and "fp32" not in path and "pytorch_model" not in path)


def pin(entry):
    name, revision, item = entry
    path = item["path"]
    repo = "meituan-longcat/" + name
    if "lfs" in item:
        assert item["lfs"]["size"] == item["size"]
        digest = item["lfs"]["oid"]
        origin = "Hub LFS oid"
    else:
        # Git blobs have SHA-1, not SHA-256. Read only small textual metadata;
        # raw never dereferences an LFS pointer. Verify its Git blob id too.
        assert path.endswith((".json", ".txt")) and item["size"] < 3_000_000
        url = f"https://huggingface.co/{repo}/raw/{revision}/{path}"
        with urlopen(url, timeout=60) as response:
            content = response.read(3_000_001)
        assert len(content) == item["size"]
        blob = b"blob " + str(len(content)).encode() + b"\0" + content
        assert hashlib.sha1(blob).hexdigest() == item["oid"]
        digest = hashlib.sha256(content).hexdigest()
        origin = "SHA-256 of textual Git blob, verified against Hub oid"
    return {"repository": repo, "revision": revision, "source_path": path,
            "path": name + "/" + path, "size": item["size"], "sha256": digest,
            "hash_source": origin}


def main():
    entries = []
    for name, revision, evidence in REPOSITORIES:
        tree = json.loads((ROOT / "evidence" / evidence).read_text())
        entries.extend((name, revision, item) for item in tree
                       if item["type"] == "file" and selected(name, item["path"]))
    with ThreadPoolExecutor(max_workers=6) as pool:
        files = list(pool.map(pin, entries))
    result = {"schema_version": 1, "code": {
        "repository": "https://github.com/meituan-longcat/LongCat-Video",
        "commit": CODE_COMMIT, "license": "MIT"},
        "files": files, "total_bytes": sum(f["size"] for f in files)}
    (ROOT / "model-manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{len(files)} files; {result['total_bytes']} bytes; no LFS objects downloaded")


if __name__ == "__main__":
    main()
