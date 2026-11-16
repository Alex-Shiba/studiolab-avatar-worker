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


def test_install_environment_drops_image_pip_settings():
    env = {"PATH": "/usr/bin", "PIP_INDEX_URL": "https://mirror.invalid/simple", "pip_extra_index_url": "x", "UV_INDEX_URL": "y",
           "VAST_API_KEY": "fake-secret-value", "HOME": "/root"}
    result = boot.install_environment(env)
    assert "/usr/bin" in result["PATH"].split(boot.os.pathsep) and result["HOME"] == "/root"
    assert not any(name.upper().startswith(("PIP_INDEX", "PIP_EXTRA", "UV_")) for name in result)
    assert "VAST_API_KEY" not in result
    assert result["PIP_CONFIG_FILE"] == boot.os.devnull


def test_install_environment_drops_the_sdk_venv():
    sep = boot.os.pathsep
    env = {"PATH": sep.join(["/workspace/worker-env/bin", "/venv/main/bin", "/usr/bin", "/opt/x"]), "PYTHONPATH": "/workspace/worker-env/lib",
           "VIRTUAL_ENV": "/workspace/worker-env", "CONDA_PREFIX": "/venv", "HOME": "/root"}
    result = boot.install_environment(env)
    assert "PYTHONPATH" not in result and "VIRTUAL_ENV" not in result and "CONDA_PREFIX" not in result
    parts = result["PATH"].split(sep)
    assert parts[0] == "/usr/local/sbin" and "/opt/x" in parts and not any("venv" in p or "worker-env" in p for p in parts)
    assert result["PYTHONNOUSERSITE"] == "1"
    assert boot.leaky_names(env) == ["CONDA_PREFIX", "PYTHONPATH", "VIRTUAL_ENV"]
