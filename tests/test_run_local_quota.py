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
    ["systemd-run", "--user", "--scope", "-q", "-p", "CPUQuota=150%", "bash", "-c",
     'cat /sys/fs/cgroup$(sed -n "s/^0:://p" /proc/self/cgroup)/cpu.max | grep -q "^150000 100000$"'],
    capture_output=True, check=False,
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
        "THERMAL_MIN_AVAILABLE_MEM_MIB=0\nOMP_NUM_THREADS=2\nMKL_NUM_THREADS=2\n"
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
def test_affinity_that_cannot_be_applied_blocks_run(tmp_path: Path):
    spec, logdir, cgroup_out = setup_run(tmp_path, extra_profile="THERMAL_CPU_AFFINITY=0,2\n")
    write_tool(tmp_path / "bin" / "taskset", "exit 1\n")
    result = subprocess.run(spec["cmd"], env=spec["env"], capture_output=True, text=True,
                            timeout=120, check=False)
    assert result.returncode == 2
    assert "BLOCKED: CPU affinity" in (logdir / "thermal.log").read_text()
    assert not cgroup_out.exists()


def test_run_without_an_owned_quota_scope_is_refused(tmp_path: Path):
    spec, logdir, cgroup_out = setup_run(tmp_path, quota="")
    result = subprocess.run(spec["cmd"], env=spec["env"], capture_output=True, text=True,
                            timeout=120, check=False)
    assert result.returncode == 2
    assert "owned CPU-quota scope is required" in (logdir / "thermal.log").read_text()
    assert not cgroup_out.exists()


@pytest.mark.skipif(not HAS_USER_SYSTEMD, reason="systemd --user scopes unavailable")
def test_limits_that_are_not_actually_in_force_block_before_the_worker_starts(tmp_path: Path):
    spec, logdir, cgroup_out = setup_run(tmp_path, extra_profile="THERMAL_CPU_AFFINITY=0,2\n")
    # A taskset that passes its probe but never pins: the in-scope check must catch it.
    write_tool(tmp_path / "bin" / "taskset", "shift 2\nexec \"$@\"\n")
    result = subprocess.run(spec["cmd"], env=spec["env"], capture_output=True, text=True,
                            timeout=120, check=False)
    assert result.returncode != 0
    assert "limits not applied: affinity" in (logdir / "thermal.log").read_text()
    assert not cgroup_out.exists(), "the worker must not start with unapplied limits"


@pytest.mark.skipif(not HAS_USER_SYSTEMD, reason="systemd --user scopes unavailable")
def test_worker_is_pinned_to_the_configured_cpus_and_thread_budget(tmp_path: Path):
    spec, logdir, _ = setup_run(tmp_path, extra_profile="THERMAL_CPU_AFFINITY=0,2\n")
    probe = tmp_path / "probe.txt"
    write_tool(tmp_path / "fake_python", f'grep Cpus_allowed_list /proc/self/status > "{probe}"\n'
               f'echo "OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS" >> "{probe}"\n')
    spec["env"]["OMP_NUM_THREADS"] = "9"  # the profile must win over an inherited value
    result = subprocess.run(spec["cmd"], env=spec["env"], capture_output=True, text=True,
                            timeout=120, check=False)
    assert result.returncode == 0, result.stderr
    text = probe.read_text()
    assert "0,2" in text and "OMP=2 MKL=2" in text
    log = (logdir / "thermal.log").read_text()
    assert "LIMITS_APPLIED" in log and "RESIDUAL_WORKERS=0" in log


def _fake_systemctl(tmp_path: Path, fail: str) -> None:
    """Delegate to the real systemctl except for one failing sub-command."""
    real = shutil.which("systemctl")
    write_tool(tmp_path / "bin" / "systemctl",
               f'for a in "$@"; do if [ "$a" = "{fail}" ]; then exit 1; fi; done\nexec {real} "$@"\n')


def _pid_alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError):
        return False
    return state != "Z"


@pytest.mark.skipif(not HAS_USER_SYSTEMD, reason="systemd --user scopes unavailable")
def test_a_late_orphan_child_cannot_escape_cleanup_and_fails_the_run(tmp_path: Path):
    spec, logdir, _ = setup_run(tmp_path)
    child_pid = tmp_path / "child.pid"
    write_tool(tmp_path / "fake_python", f'setsid sleep 120 &\necho $! > "{child_pid}"\nsleep 0.3\nexit 0\n')
    result = subprocess.run(spec["cmd"], env=spec["env"], capture_output=True, text=True,
                            timeout=120, check=False)
    pid = int(child_pid.read_text())
    try:
        assert result.returncode == 1, "a worker exit of 0 must not hide a surviving child"
        assert "RESIDUAL_WORKERS_KILLED" in (logdir / "thermal.log").read_text()
        deadline = time.time() + 5
        while _pid_alive(pid) and time.time() < deadline:
            time.sleep(0.1)
        assert not _pid_alive(pid)
    finally:
        if _pid_alive(pid):
            os.kill(pid, 9)


@pytest.mark.skipif(not HAS_USER_SYSTEMD, reason="systemd --user scopes unavailable")
def test_failed_scope_kill_is_reported_as_unverified_cleanup_not_success(tmp_path: Path):
    spec, logdir, _ = setup_run(tmp_path)
    child_pid = tmp_path / "child.pid"
    write_tool(tmp_path / "fake_python", f'setsid sleep 120 &\necho $! > "{child_pid}"\nsleep 0.3\nexit 0\n')
    _fake_systemctl(tmp_path, "kill")
    result = subprocess.run(spec["cmd"], env=spec["env"], capture_output=True, text=True,
                            timeout=120, check=False)
    pid = int(child_pid.read_text())
    try:
        assert result.returncode == 1
        assert "CLEANUP_UNVERIFIED" in (logdir / "thermal.log").read_text()
    finally:
        if _pid_alive(pid):
            os.kill(pid, 9)  # the test removes what the deliberately failing stub left behind


@pytest.mark.skipif(not HAS_USER_SYSTEMD, reason="systemd --user scopes unavailable")
def test_unknown_scope_state_is_never_treated_as_no_residual_workers(tmp_path: Path):
    spec, logdir, _ = setup_run(tmp_path)
    _fake_systemctl(tmp_path, "show")
    result = subprocess.run(spec["cmd"], env=spec["env"], capture_output=True, text=True,
                            timeout=120, check=False)
    assert result.returncode == 1
    log = (logdir / "thermal.log").read_text()
    assert "CLEANUP_UNVERIFIED" in log and "RESIDUAL_WORKERS=0" not in log


@pytest.mark.skipif(not HAS_USER_SYSTEMD, reason="systemd --user scopes unavailable")
def test_worker_is_stopped_when_the_watchdog_dies(tmp_path: Path):
    spec, logdir, _ = setup_run(tmp_path)
    write_tool(tmp_path / "fake_python", "sleep 30\n")
    run_copy = tmp_path / "local-hybrid"
    shutil.copytree(LOCAL, run_copy)
    guard = run_copy / "thermal_guard.sh"
    write_tool(guard, "exit 0\n")  # a watchdog that dies immediately
    cmd = ["bash", str(run_copy / "run_local.sh"), *spec["cmd"][2:]]
    started = time.monotonic()
    result = subprocess.run(cmd, env=spec["env"], capture_output=True, text=True,
                            timeout=60, check=False)
    assert time.monotonic() - started < 20, "worker kept running after the watchdog died"
    assert result.returncode != 0, result.stderr
    assert "GUARD_LOST" in (logdir / "thermal.log").read_text()
