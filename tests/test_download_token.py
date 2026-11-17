"""HF token on downloads (only to Hugging Face hosts, dropped on a foreign redirect) and the progress lines (2026-11-17)."""

from urllib.request import Request

import download


def test_token_only_for_hugging_face_hosts():
    env = {"HF_TOKEN": " hf_fake_token_value "}
    assert download.hf_headers("https://huggingface.co/x/resolve/abc/f", env) == {"Authorization": "Bearer hf_fake_token_value"}
    assert download.hf_headers("https://cas-bridge.xethub.hf.co/obj", env) == {"Authorization": "Bearer hf_fake_token_value"}
    assert download.hf_headers("https://example.com/huggingface.co/f", env) == {}
    assert download.hf_headers("https://evilhuggingface.co/f", env) == {}
    assert download.hf_headers("https://huggingface.co/f", {}) == {}


def test_redirect_away_from_hugging_face_drops_the_token():
    handler = download.HFRedirect()
    request = Request("https://huggingface.co/x/resolve/abc/f", headers={"Authorization": "Bearer hf_fake", "Range": "bytes=0-9"})
    away = handler.redirect_request(request, None, 302, "Found", {}, "https://cdn.example.net/blob?sig=1")
    assert "Authorization" not in away.headers and away.headers.get("Range") == "bytes=0-9"
    request = Request("https://huggingface.co/x/resolve/abc/f", headers={"Authorization": "Bearer hf_fake"})
    stay = handler.redirect_request(request, None, 302, "Found", {}, "https://cas-bridge.xethub.hf.co/blob")
    assert stay.headers.get("Authorization") == "Bearer hf_fake"


def test_progress_lines_numbers_only(monkeypatch):
    files = [{"repository": "a/b", "revision": "r", "path": "one.bin", "size": 6_000_000_000, "sha256": "0" * 64},
             {"repository": "a/b", "revision": "r", "path": "two.bin", "size": 4_000_000_000, "sha256": "1" * 64}]
    monkeypatch.setattr(download, "manifest", lambda: {"files": files})
    now = [0.0]
    lines = []

    def fetch(url, target, size, digest, *, progress):
        for fraction in (.5, 1.0):
            now[0] += 30
            progress(fraction)

    download.download_models("/tmp/w", emit=lines.append, clock=lambda: now[0], fetch=fetch)
    assert lines[0] == "AVATAR_DOWNLOAD 3.0/10.0 GB 100 MB/s"
    assert lines[-1] == "AVATAR_DOWNLOAD done 10.0 GB in 120 s"
    assert all("hf_" not in line and "huggingface" not in line for line in lines)
