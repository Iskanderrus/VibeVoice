"""Model-free integration tests of run_local.sh: CPU quota launch and fail-closed behavior."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

LOCAL = Path(__file__).resolve().parents[1] / "scripts" / "local-hybrid"
HAS_USER_SYSTEMD = shutil.which("systemd-run") is not None and subprocess.run(
    ["systemd-run", "--user", "--scope", "-q", "true"], capture_output=True, check=False
).returncode == 0


def write_tool(path: Path, body: str) -> None:
    path.write_text("#!/bin/bash\n" + body, encoding="utf-8")
    path.chmod(0o755)


def setup_run(tmp_path: Path, quota: str = "150%", extra_profile: str = "") -> tuple[dict, Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    seq = tmp_path / "cpu.txt"
    seq.write_text("50\n", encoding="utf-8")
    write_tool(bin_dir / "sensors", 'printf "Tctl:  +%s.0C\\n" 50\n')
    write_tool(bin_dir / "nvidia-smi", "echo 50\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"revision": "test-revision"}), encoding="utf-8")
    profile = tmp_path / "profile.env"
    profile.write_text(
        f"VIBEVOICE_MODEL_MANIFEST={manifest}\n"
        f"THERMAL_CPU_QUOTA={quota}\n"
        "THERMAL_PREFLIGHT_WAIT_SECONDS=5\nTHERMAL_PREFLIGHT_POLL_SECONDS=0.2\n"
        "THERMAL_PREFLIGHT_STABLE_SAMPLES=2\nTHERMAL_PREFLIGHT_MIN_IDLE_PERCENT=0\n"
        f"{extra_profile}",
        encoding="utf-8",
    )
    cgroup_out = tmp_path / "cgroup.txt"
    fake_python = tmp_path / "fake_python"
    write_tool(fake_python, f'cat /proc/self/cgroup > "{cgroup_out}"\nsleep 1\n')
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "VIBEVOICE_PYTHON": str(fake_python),
        "THERMAL_HWMON_ROOT": str(tmp_path / "no-hwmon"),
    }
    dialogue = tmp_path / "dialogue.json"
    dialogue.write_text("{}", encoding="utf-8")
    logdir = tmp_path / "run"
    cmd = ["bash", str(LOCAL / "run_local.sh"), str(dialogue), str(tmp_path / "out"),
           str(profile), str(logdir)]
    return {"env": env, "cmd": cmd}, logdir, cgroup_out


@pytest.mark.skipif(not HAS_USER_SYSTEMD, reason="systemd --user scopes unavailable")
def test_worker_runs_inside_cpu_quota_scope(tmp_path: Path):
    spec, logdir, cgroup_out = setup_run(tmp_path)
    result = subprocess.run(spec["cmd"], env=spec["env"], capture_output=True,
                            text=True, timeout=120, check=False)
    assert result.returncode == 0, result.stderr
    assert ".scope" in cgroup_out.read_text(encoding="utf-8")
    assert "exit=0" in (logdir / "synthesis.log").read_text(encoding="utf-8")


def test_quota_that_cannot_be_applied_blocks_run(tmp_path: Path):
    spec, logdir, cgroup_out = setup_run(tmp_path)
    # A systemd-run that always fails stands in for "user scopes unavailable".
    write_tool(tmp_path / "bin" / "systemd-run", "exit 1\n")
    spec["env"]["PATH"] = f"{tmp_path / 'bin'}{os.pathsep}{spec['env']['PATH']}"
    result = subprocess.run(spec["cmd"], env=spec["env"], capture_output=True,
                            text=True, timeout=120, check=False)
    assert result.returncode == 2
    assert "BLOCKED: CPU quota" in (logdir / "thermal.log").read_text(encoding="utf-8")
    assert not cgroup_out.exists(), "worker must not start uncapped"


@pytest.mark.skipif(not HAS_USER_SYSTEMD, reason="systemd --user scopes unavailable")
def test_worker_is_pinned_to_the_configured_cpus_and_thread_budget(tmp_path: Path):
    spec, logdir, _ = setup_run(tmp_path, extra_profile="THERMAL_CPU_AFFINITY=0,2\n")
    probe = tmp_path / "probe.txt"
    write_tool(tmp_path / "fake_python", f'grep Cpus_allowed_list /proc/self/status > "{probe}"\n'
               f'echo "OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS" >> "{probe}"\n')
    spec["env"]["OMP_NUM_THREADS"] = "9"  # the profile must win over an inherited value
    profile = Path(spec["cmd"][4])
    profile.write_text(profile.read_text() + "OMP_NUM_THREADS=2\nMKL_NUM_THREADS=2\n")
    result = subprocess.run(spec["cmd"], env=spec["env"], capture_output=True, text=True,
                            timeout=120, check=False)
    assert result.returncode == 0, result.stderr
    text = probe.read_text()
    assert "0,2" in text and "OMP=2 MKL=2" in text
    assert "RESIDUAL_WORKERS=0" in (logdir / "thermal.log").read_text()


def test_affinity_that_cannot_be_applied_blocks_run(tmp_path: Path):
    spec, logdir, cgroup_out = setup_run(tmp_path, quota="", extra_profile="THERMAL_CPU_AFFINITY=0,2\n")
    write_tool(tmp_path / "bin" / "taskset", "exit 1\n")
    result = subprocess.run(spec["cmd"], env=spec["env"], capture_output=True, text=True,
                            timeout=120, check=False)
    assert result.returncode == 2
    assert "BLOCKED: CPU affinity" in (logdir / "thermal.log").read_text()
    assert not cgroup_out.exists()


def test_worker_is_stopped_when_the_watchdog_dies(tmp_path: Path):
    spec, logdir, _ = setup_run(tmp_path, quota="")
    # A long-running worker plus a guard that dies immediately: the worker must not run unwatched.
    write_tool(tmp_path / "fake_python", "sleep 30\n")
    guard_stub = tmp_path / "guard_dies.sh"
    write_tool(guard_stub, "exit 0\n")
    run_copy = tmp_path / "local-hybrid"
    shutil.copytree(LOCAL, run_copy)
    shutil.copy(guard_stub, run_copy / "thermal_guard.sh")
    (run_copy / "thermal_guard.sh").chmod(0o755)
    cmd = ["bash", str(run_copy / "run_local.sh"), *spec["cmd"][2:]]
    # run_local.sh derives FORK from its own location; the copy keeps git metadata out of scope, so
    # provide the identity it would read from the real checkout.
    spec["env"]["GIT_DIR"] = str(LOCAL.parents[1] / ".git") if (LOCAL.parents[1] / ".git").exists() else ""
    started = time.monotonic()
    result = subprocess.run(cmd, env=spec["env"], capture_output=True, text=True,
                            timeout=60, check=False)
    assert time.monotonic() - started < 20, "worker kept running after the watchdog died"
    assert result.returncode != 0, result.stderr  # 143: the worker was terminated, never success
    assert "GUARD_LOST" in (logdir / "thermal.log").read_text()
