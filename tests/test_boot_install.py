"""Install-stage robustness added after the first live boot (2026-11-17): readable failures, apt lock retries, no apt when tools exist."""

import sys

import pytest

import boot
from server import log_marker


def test_command_failure_names_the_program_and_prints_the_stderr_tail(capsys):
    script = "import sys; sys.stderr.write('line one\\nE: Could not get lock /var/lib/dpkg/lock-frontend\\n'); sys.exit(100)"
    with pytest.raises(boot.CommandFailed) as caught:
        boot.command([sys.executable, "-c", script])
    assert caught.value.code == 100
    assert caught.value.what == boot.command_name([sys.executable, "-c", script])
    out = capsys.readouterr().out
    assert "AVATAR_INSTALL_STDERR E: Could not get lock" in out


def test_command_name_keeps_program_and_first_word_only():
    assert boot.command_name(["/usr/bin/apt-get", "-o", "DPkg::Lock::Timeout=120", "update", "-qq"]) == "apt-get DPkg::Lock::Timeout=120"
    assert boot.command_name(["/w/venv/bin/python", "-m", "pip", "install"]) == "python pip"
    assert boot.command_name(["git", "-C", "/w/src", "checkout"]) == "git src"


def test_apt_retries_on_failure_then_succeeds(capsys):
    calls, slept = [], []

    def run(argv):
        calls.append(argv)
        if len(calls) <= 2:
            raise boot.CommandFailed("apt-get update", 100)

    boot.apt_install(["ffmpeg"], tries=5, pause=3, run=run, sleep=slept.append)
    assert slept == [3, 3]
    assert calls[-1][-1] == "ffmpeg" and "DPkg::Lock::Timeout=120" in calls[-1]
    assert "AVATAR_APT_RETRY 2" in capsys.readouterr().out


def test_apt_gives_up_after_the_last_try():
    def run(argv):
        raise boot.CommandFailed("apt-get update", 100)

    with pytest.raises(boot.CommandFailed):
        boot.apt_install(["ffmpeg"], tries=3, pause=0, run=run, sleep=lambda _: None)


def test_markers_also_reach_the_container_output(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AVATAR_LOG_FILE", str(tmp_path / "markers.log"))
    log_marker("AVATAR_STAGE install")
    assert (tmp_path / "markers.log").read_text(encoding="utf-8") == "AVATAR_STAGE install\n"
    assert "AVATAR_STAGE install" in capsys.readouterr().out
