"""Model-free integration tests of run_local.sh: CPU quota launch and fail-closed behavior."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

LOCAL = Path(__file__).resolve().parents[1] / "scripts" / "local-hybrid"
HAS_USER_SYSTEMD = shutil.which("systemd-run") is not None and subprocess.run(
    ["systemd-run", "--user", "--scope", "-q", "true"], capture_output=True, check=False
).returncode == 0


def write_tool(path: Path, body: str) -> None:
    path.write_text("#!/bin/bash\n" + body, encoding="utf-8")
    path.chmod(0o755)


def setup_run(tmp_path: Path, quota: str = "150%") -> tuple[dict, Path, Path]:
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
        "THERMAL_PREFLIGHT_WAIT_SECONDS=1\nTHERMAL_PREFLIGHT_POLL_SECONDS=1\n",
        encoding="utf-8",
    )
    cgroup_out = tmp_path / "cgroup.txt"
    fake_python = tmp_path / "fake_python"
    write_tool(fake_python, f'cat /proc/self/cgroup > "{cgroup_out}"\nsleep 1\n')
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "VIBEVOICE_PYTHON": str(fake_python),
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
