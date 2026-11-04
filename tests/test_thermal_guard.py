"""Model-free tests of the hybrid thermal preflight and watchdog, using stubbed sensors."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "local-hybrid"


def make_sensor_stubs(bin_dir: Path, cpu_seq: list[int], gpu: int) -> Path:
    """Fake `sensors` returns the CPU values in order (last one repeats); `nvidia-smi` is fixed."""
    bin_dir.mkdir()
    seq = bin_dir / "cpu_seq.txt"
    seq.write_text("\n".join(map(str, cpu_seq)) + "\n", encoding="utf-8")
    sensors = bin_dir / "sensors"
    sensors.write_text(
        "#!/bin/bash\n"
        f'f="{seq}"\n'
        'v=$(head -n1 "$f")\n'
        'if [ "$(wc -l < "$f")" -gt 1 ]; then sed -i 1d "$f"; fi\n'
        'printf "k10temp-pci-00c3\\nAdapter: PCI\\nTctl:         +%s.0\\302\\260C\\n" "$v"\n',
        encoding="utf-8",
    )
    nvidia = bin_dir / "nvidia-smi"
    nvidia.write_text(f"#!/bin/bash\necho {gpu}\n", encoding="utf-8")
    for tool in (sensors, nvidia):
        tool.chmod(0o755)
    return bin_dir


def stub_env(bin_dir: Path, **extra: str) -> dict:
    return {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", **extra}


def run_guard(bin_dir: Path, pid: int, log: Path, timeout: int = 60, **extra: str):
    return subprocess.run(
        ["bash", str(SCRIPTS / "thermal_guard.sh"), str(pid), str(log)],
        env=stub_env(bin_dir, **extra), capture_output=True, text=True,
        timeout=timeout, check=False,
    )


def process_alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError):
        return False
    return state != "Z"


def test_guard_pauses_hot_worker_then_resumes_after_cooling(tmp_path: Path):
    bin_dir = make_sensor_stubs(tmp_path / "bin", cpu_seq=[85, 85, 60], gpu=60)
    log = tmp_path / "thermal.log"
    worker = subprocess.Popen(["sleep", "3"])
    guard = subprocess.Popen(
        ["bash", str(SCRIPTS / "thermal_guard.sh"), str(worker.pid), str(log)],
        env=stub_env(bin_dir), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        # Reap the worker first: an unreaped zombie still answers `kill -0`, as in run_local.sh.
        worker.wait(timeout=30)
        _, stderr = guard.communicate(timeout=30)
    finally:
        for proc in (worker, guard):
            if proc.poll() is None:
                proc.kill()
    assert guard.returncode == 0, stderr
    text = log.read_text(encoding="utf-8")
    assert text.index("PAUSE") < text.index("RESUME") < text.index("process exited")


def test_single_cpu_burst_does_not_pause_worker(tmp_path: Path):
    bin_dir = make_sensor_stubs(tmp_path / "bin", cpu_seq=[60, 60, 85, 60, 60], gpu=60)
    log = tmp_path / "thermal.log"
    worker = subprocess.Popen(["sleep", "2"])
    guard = subprocess.Popen(
        ["bash", str(SCRIPTS / "thermal_guard.sh"), str(worker.pid), str(log)],
        env=stub_env(bin_dir), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        worker.wait(timeout=30)
        _, stderr = guard.communicate(timeout=30)
    finally:
        for proc in (worker, guard):
            if proc.poll() is None:
                proc.kill()
    assert guard.returncode == 0, stderr
    text = log.read_text(encoding="utf-8")
    assert "PAUSE" not in text and "cpu_med=" in text


def test_critical_temperature_stops_immediately_without_smoothing(tmp_path: Path):
    bin_dir = make_sensor_stubs(tmp_path / "bin", cpu_seq=[60, 60, 95], gpu=60)
    log = tmp_path / "thermal.log"
    worker = subprocess.Popen(["sleep", "60"])
    try:
        result = run_guard(bin_dir, worker.pid, log, timeout=30)
        worker.wait(timeout=10)
    finally:
        if worker.poll() is None:
            worker.kill()
    assert result.returncode == 3, result.stderr
    assert "CRITICAL_THERMAL_STOP cpu=95" in log.read_text(encoding="utf-8")


def test_guard_terminates_whole_tree_with_reason_when_cooling_never_happens(tmp_path: Path):
    bin_dir = make_sensor_stubs(tmp_path / "bin", cpu_seq=[85], gpu=60)
    log = tmp_path / "thermal.log"
    child_pid_file = tmp_path / "child.pid"
    worker = subprocess.Popen([
        "bash", "-c", f"sleep 120 & echo $! > {child_pid_file}; wait",
    ])
    try:
        deadline = time.time() + 10
        while not child_pid_file.exists() and time.time() < deadline:
            time.sleep(0.05)
        child_pid = int(child_pid_file.read_text())
        result = run_guard(bin_dir, worker.pid, log, timeout=60,
                           THERMAL_PAUSE_MAX_SECONDS="2")
        worker.wait(timeout=10)
    finally:
        if worker.poll() is None:
            worker.kill()
    assert result.returncode == 4, result.stderr
    assert "THERMAL_PAUSE_TIMEOUT" in log.read_text(encoding="utf-8")
    deadline = time.time() + 5
    while process_alive(child_pid) and time.time() < deadline:
        time.sleep(0.05)
    assert not process_alive(child_pid), "descendant worker survived the thermal stop"


def test_preflight_refuses_to_start_when_already_hot(tmp_path: Path):
    bin_dir = make_sensor_stubs(tmp_path / "bin", cpu_seq=[85], gpu=60)
    log = tmp_path / "thermal.log"
    result = subprocess.run(
        ["bash", str(SCRIPTS / "thermal_preflight.sh"), str(log)],
        env=stub_env(bin_dir, THERMAL_PREFLIGHT_WAIT_SECONDS="1",
                     THERMAL_PREFLIGHT_POLL_SECONDS="1"),
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert result.returncode == 2, result.stderr
    assert "BLOCKED: thermal preflight not met" in log.read_text(encoding="utf-8")


def test_preflight_allows_start_when_cool(tmp_path: Path):
    bin_dir = make_sensor_stubs(tmp_path / "bin", cpu_seq=[60], gpu=60)
    log = tmp_path / "thermal.log"
    result = subprocess.run(
        ["bash", str(SCRIPTS / "thermal_preflight.sh"), str(log)],
        env=stub_env(bin_dir), capture_output=True, text=True, timeout=60, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "PREFLIGHT_OK" in log.read_text(encoding="utf-8")
