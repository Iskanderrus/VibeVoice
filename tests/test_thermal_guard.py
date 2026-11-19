"""Model-free tests of the hybrid thermal preflight and watchdog, using stubbed sensors."""

from __future__ import annotations

import os
import subprocess
import threading
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


class StatWriter:
    """Writes a monotonic fake /proc/stat; each phase is (seconds, idle_fraction)."""

    def __init__(self, path: Path, phases: list[tuple[float, float]]):
        self.path, self.phases = path, phases
        self.stop = threading.Event()
        self.busy = self.idle = 0
        self._write()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _write(self) -> None:
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(f"cpu  {self.busy} 0 0 {self.idle} 0 0 0 0 0 0\n", encoding="utf-8")
        os.replace(tmp, self.path)

    def _run(self) -> None:
        started = time.monotonic()
        while not self.stop.is_set():
            elapsed, frac = time.monotonic() - started, self.phases[-1][1]
            acc = 0.0
            for seconds, phase_frac in self.phases:
                acc += seconds
                if elapsed < acc:
                    frac = phase_frac
                    break
            self.idle += int(100 * frac)
            self.busy += 100 - int(100 * frac)
            self._write()
            time.sleep(0.05)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=2)


def stub_env(bin_dir: Path, **extra: str) -> dict:
    # An empty hwmon root forces the `sensors` stub; the preflight gets a fixed idle /proc/stat so
    # the result never depends on whatever else this workstation is doing.
    stat = bin_dir / "proc_stat"
    if not stat.exists():  # constant counters: valid shape, but never advancing (stale)
        stat.write_text("cpu  100 0 100 100000 0 0 0 0 0 0\n", encoding="utf-8")
    meminfo = bin_dir / "meminfo"
    if not meminfo.exists():
        meminfo.write_text("MemTotal: 30000000 kB\nMemAvailable: 24000000 kB\n", encoding="utf-8")
    return {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "THERMAL_HWMON_ROOT": str(bin_dir / "no-hwmon"),
        "THERMAL_PROC_STAT": str(stat),
        "THERMAL_PROC_MEMINFO": str(meminfo),
        "THERMAL_SENSOR_TIMEOUT_SECONDS": "2",
        "THERMAL_RESUME_SAMPLES": "3",
        "THERMAL_PREFLIGHT_STABLE_SAMPLES": "2",
        "THERMAL_PREFLIGHT_POLL_SECONDS": "0.2",
        "THERMAL_POLL_SECONDS": "0.2",
        **extra,
    }


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
    worker = subprocess.Popen(["sleep", "4"])
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


def test_raw_hot_sample_pauses_even_when_the_median_is_cool(tmp_path: Path):
    # Median of the window stays 60 while one raw reading is 85: raw safety wins.
    bin_dir = make_sensor_stubs(tmp_path / "bin", cpu_seq=[60, 60, 60, 60, 85, 60, 60, 60, 60, 60], gpu=60)
    log = tmp_path / "thermal.log"
    worker = subprocess.Popen(["sleep", "6"])
    guard = subprocess.Popen(
        ["bash", str(SCRIPTS / "thermal_guard.sh"), str(worker.pid), str(log)],
        env=stub_env(bin_dir), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        worker.wait(timeout=30)
        guard.communicate(timeout=30)
    finally:
        for proc in (worker, guard):
            if proc.poll() is None:
                proc.kill()
    text = log.read_text(encoding="utf-8")
    assert "PAUSE raw_cpu=85" in text and "cpu=85 cpu_med=60" in text


def test_resume_needs_sustained_raw_cooling_and_a_hot_raw_reading_resets_it(tmp_path: Path):
    # After the pause, a raw 75 (median cool) must reset the streak; resume needs 3 clean samples.
    seq = [85, 60, 60, 75, 60, 60, 60, 60, 60, 60]
    bin_dir = make_sensor_stubs(tmp_path / "bin", cpu_seq=seq, gpu=60)
    log = tmp_path / "thermal.log"
    worker = subprocess.Popen(["sleep", "8"])
    guard = subprocess.Popen(
        ["bash", str(SCRIPTS / "thermal_guard.sh"), str(worker.pid), str(log)],
        env=stub_env(bin_dir), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        worker.wait(timeout=30)
        guard.communicate(timeout=30)
    finally:
        for proc in (worker, guard):
            if proc.poll() is None:
                proc.kill()
    lines = log.read_text(encoding="utf-8").splitlines()
    hot75 = next(i for i, line in enumerate(lines) if " cpu=75 " in line)
    resume = next(i for i, line in enumerate(lines) if line.startswith("RESUME"))
    clean_after_hot = [ln for ln in lines[hot75 + 1:resume] if " cpu=60 " in ln]
    assert resume > hot75 and len(clean_after_hot) >= 3, lines


def test_hung_sensor_command_is_bounded_and_stops_the_worker(tmp_path: Path):
    bin_dir = make_sensor_stubs(tmp_path / "bin", cpu_seq=[60], gpu=60)
    (bin_dir / "nvidia-smi").write_text("#!/bin/bash\nsleep 30\n", encoding="utf-8")
    log = tmp_path / "thermal.log"
    worker = subprocess.Popen(["sleep", "60"])
    started = time.monotonic()
    try:
        result = run_guard(bin_dir, worker.pid, log, timeout=30, THERMAL_SENSOR_TIMEOUT_SECONDS="1")
        worker.wait(timeout=10)
    finally:
        if worker.poll() is None:
            worker.kill()
    assert result.returncode == 2 and time.monotonic() - started < 10
    assert "sensor unavailable" in log.read_text(encoding="utf-8")


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


def run_preflight(bin_dir: Path, log: Path, **extra: str):
    return subprocess.run(
        ["bash", str(SCRIPTS / "thermal_preflight.sh"), str(log)],
        env=stub_env(bin_dir, **extra), capture_output=True, text=True, timeout=60, check=False,
    )


def test_preflight_requires_a_sustained_window_not_one_cool_reading(tmp_path: Path):
    # One cool sample between hot ones never satisfies a 3-sample stable window.
    bin_dir = make_sensor_stubs(tmp_path / "bin", cpu_seq=[60, 85, 60, 85, 60, 85], gpu=60)
    log = tmp_path / "thermal.log"
    result = run_preflight(bin_dir, log, THERMAL_PREFLIGHT_STABLE_SAMPLES="3",
                           THERMAL_PREFLIGHT_WAIT_SECONDS="2")
    assert result.returncode == 2, result.stderr
    assert "PREFLIGHT_OK" not in log.read_text(encoding="utf-8")


def test_preflight_allows_start_only_when_cool_idle_and_counters_advance(tmp_path: Path):
    bin_dir = make_sensor_stubs(tmp_path / "bin", cpu_seq=[60], gpu=60)
    log = tmp_path / "thermal.log"
    with StatWriter(bin_dir / "proc_stat", [(60, 0.95)]):
        result = run_preflight(bin_dir, log, THERMAL_PREFLIGHT_WAIT_SECONDS="10")
    assert result.returncode == 0, result.stderr
    text = log.read_text(encoding="utf-8")
    assert "PREFLIGHT_OK" in text and "mem_available_mib=" in text


def test_preflight_refuses_a_busy_machine_even_when_cool(tmp_path: Path):
    bin_dir = make_sensor_stubs(tmp_path / "bin", cpu_seq=[60], gpu=60)
    log = tmp_path / "thermal.log"
    with StatWriter(bin_dir / "proc_stat", [(60, 0.10)]):
        result = run_preflight(bin_dir, log, THERMAL_PREFLIGHT_WAIT_SECONDS="3")
    assert result.returncode == 2, result.stderr
    assert "PREFLIGHT_UNSTABLE" in log.read_text(encoding="utf-8")


def test_preflight_fails_closed_on_stale_counters(tmp_path: Path):
    bin_dir = make_sensor_stubs(tmp_path / "bin", cpu_seq=[60], gpu=60)  # constant proc_stat
    log = tmp_path / "thermal.log"
    result = run_preflight(bin_dir, log, THERMAL_PREFLIGHT_WAIT_SECONDS="3")
    assert result.returncode == 2, result.stderr
    text = log.read_text(encoding="utf-8")
    assert "PREFLIGHT_INVALID_IDLE_COUNTERS" in text and "PREFLIGHT_OK" not in text


def test_preflight_fails_closed_on_missing_or_garbage_counters(tmp_path: Path):
    bin_dir = make_sensor_stubs(tmp_path / "bin", cpu_seq=[60], gpu=60)
    for content in (None, "cpu  abc def\n", "intr 1 2 3\n"):
        stat = bin_dir / "proc_stat"
        extra = {}
        if content is None:
            extra["THERMAL_PROC_STAT"] = str(bin_dir / "does-not-exist")
        else:
            stat.write_text(content, encoding="utf-8")
        log = tmp_path / f"thermal-{abs(hash(content))}.log"
        result = run_preflight(bin_dir, log, THERMAL_PREFLIGHT_WAIT_SECONDS="3", **extra)
        assert result.returncode == 2, (content, result.stderr)
        assert "idle counters unreadable or invalid" in log.read_text(encoding="utf-8")


def test_idle_is_measured_over_the_whole_stable_window_not_just_the_last_interval(tmp_path: Path):
    # 3.5 s fully busy, then fully idle. The 2 s stable window only becomes clean after ~5.5 s, so
    # a run that is allowed 4 s must be refused even though the latest interval is entirely idle.
    phases = [(3.5, 0.0), (60, 1.0)]
    bin_dir = make_sensor_stubs(tmp_path / "bin", cpu_seq=[60], gpu=60)
    log = tmp_path / "thermal.log"
    with StatWriter(bin_dir / "proc_stat", phases):
        refused = run_preflight(bin_dir, log, THERMAL_PREFLIGHT_STABLE_SAMPLES="10",
                                THERMAL_PREFLIGHT_WAIT_SECONDS="4")
    assert refused.returncode == 2, refused.stderr
    assert "PREFLIGHT_OK" not in log.read_text(encoding="utf-8")
    log2 = tmp_path / "thermal2.log"
    with StatWriter(bin_dir / "proc_stat", phases):
        accepted = run_preflight(bin_dir, log2, THERMAL_PREFLIGHT_STABLE_SAMPLES="10",
                                 THERMAL_PREFLIGHT_WAIT_SECONDS="12")
    assert accepted.returncode == 0, accepted.stderr
    ok = next(ln for ln in log2.read_text().splitlines() if "PREFLIGHT_OK" in ln)
    assert int(ok.split("idle_pct=")[1].split()[0]) >= 70


def test_preflight_refuses_when_available_memory_is_insufficient(tmp_path: Path):
    bin_dir = make_sensor_stubs(tmp_path / "bin", cpu_seq=[60], gpu=60)
    (bin_dir / "meminfo").write_text("MemTotal: 30000000 kB\nMemAvailable: 6800000 kB\n")
    log = tmp_path / "thermal.log"
    with StatWriter(bin_dir / "proc_stat", [(60, 0.95)]):
        result = run_preflight(bin_dir, log, THERMAL_PREFLIGHT_WAIT_SECONDS="3",
                               THERMAL_MIN_AVAILABLE_MEM_MIB="12288")
    assert result.returncode == 2, result.stderr
    text = log.read_text(encoding="utf-8")
    assert "mem_available_mib=6640" in text and "PREFLIGHT_OK" not in text


def test_preflight_refuses_when_memory_cannot_be_read(tmp_path: Path):
    bin_dir = make_sensor_stubs(tmp_path / "bin", cpu_seq=[60], gpu=60)
    (bin_dir / "meminfo").write_text("MemTotal: 30000000 kB\n")
    log = tmp_path / "thermal.log"
    with StatWriter(bin_dir / "proc_stat", [(60, 0.95)]):
        result = run_preflight(bin_dir, log, THERMAL_PREFLIGHT_WAIT_SECONDS="5")
    assert result.returncode == 2
    assert "MemAvailable unreadable" in log.read_text(encoding="utf-8")


def test_sensor_loss_during_run_stops_the_worker(tmp_path: Path):
    bin_dir = make_sensor_stubs(tmp_path / "bin", cpu_seq=[60], gpu=60)
    (bin_dir / "nvidia-smi").write_text("#!/bin/bash\nexit 1\n", encoding="utf-8")
    log = tmp_path / "thermal.log"
    worker = subprocess.Popen(["sleep", "60"])
    try:
        result = run_guard(bin_dir, worker.pid, log, timeout=30)
        worker.wait(timeout=10)
    finally:
        if worker.poll() is None:
            worker.kill()
    assert result.returncode == 2
    assert "BLOCKED: required CPU/GPU thermal sensor unavailable" in log.read_text(encoding="utf-8")


def test_implausible_sensor_value_is_a_sensor_failure_not_a_safe_reading(tmp_path: Path):
    bin_dir = make_sensor_stubs(tmp_path / "bin", cpu_seq=[0], gpu=60)  # 0 C is not a real reading
    log = tmp_path / "thermal.log"
    worker = subprocess.Popen(["sleep", "60"])
    try:
        result = run_guard(bin_dir, worker.pid, log, timeout=30)
        worker.wait(timeout=10)
    finally:
        if worker.poll() is None:
            worker.kill()
    assert result.returncode == 2


def test_rapid_rise_between_samples_hits_the_raw_critical_stop(tmp_path: Path):
    bin_dir = make_sensor_stubs(tmp_path / "bin", cpu_seq=[55, 58, 94], gpu=60)
    log = tmp_path / "thermal.log"
    worker = subprocess.Popen(["sleep", "60"])
    try:
        result = run_guard(bin_dir, worker.pid, log, timeout=30)
        worker.wait(timeout=10)
    finally:
        if worker.poll() is None:
            worker.kill()
    text = log.read_text(encoding="utf-8")
    assert result.returncode == 3 and "CRITICAL_THERMAL_STOP cpu=94" in text
    assert "PAUSE" not in text  # the raw reading, not the smoothed median, decided


def test_worker_that_ignores_term_is_killed(tmp_path: Path):
    bin_dir = make_sensor_stubs(tmp_path / "bin", cpu_seq=[60, 95], gpu=60)
    log = tmp_path / "thermal.log"
    worker = subprocess.Popen(["bash", "-c", "trap '' TERM; while :; do sleep 0.2; done"])
    try:
        time.sleep(0.5)
        result = run_guard(bin_dir, worker.pid, log, timeout=30, THERMAL_KILL_GRACE_SECONDS="1")
        worker.wait(timeout=10)
    finally:
        if worker.poll() is None:
            worker.kill()
    assert result.returncode == 3
    assert "KILL_ESCALATION" in log.read_text(encoding="utf-8")
    assert not process_alive(worker.pid)


def test_k10temp_sysfs_reading_uses_the_hottest_of_tctl_and_tdie(tmp_path: Path):
    hwmon = tmp_path / "hwmon" / "hwmon3"
    hwmon.mkdir(parents=True)
    (hwmon / "name").write_text("k10temp\n")
    for index, (label, milli) in enumerate((("Tctl", 71000), ("Tdie", 68000)), start=1):
        (hwmon / f"temp{index}_label").write_text(label + "\n")
        (hwmon / f"temp{index}_input").write_text(f"{milli}\n")
    other = tmp_path / "hwmon" / "hwmon5"
    other.mkdir()
    (other / "name").write_text("amdgpu\n")
    (other / "temp1_label").write_text("Tctl\n")
    (other / "temp1_input").write_text("99000\n")  # a different chip is never mistaken for the CPU
    out = subprocess.run(
        ["bash", "-c", f". {SCRIPTS}/thermal_sensors.sh; read_cpu_c"],
        env={**os.environ, "THERMAL_HWMON_ROOT": str(tmp_path / "hwmon")},
        capture_output=True, text=True, check=False,
    )
    assert out.stdout.strip() == "71"
