#!/bin/bash
# Internal hybrid executor called by scripts/run_stickman_local.py (no env editing).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
FORK="$(cd "$HERE/../.." && pwd)"
DIALOGUE="${1:?dialogue JSON required}"
OUT="${2:?output directory required}"
PROFILE="${3:-$HERE/profile-gtx1650ti-hybrid.env}"
LOGDIR="${4:-$OUT}"
set -a
. "$PROFILE"
set +a
# Derive identities from actual code and installed model, not stale manual pins.
export VIBEVOICE_SOURCE_REVISION="$(git -C "$FORK" rev-parse HEAD)"
export VIBEVOICE_MODEL_REVISION="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["revision"])' "$VIBEVOICE_MODEL_MANIFEST")"
PY="${VIBEVOICE_PYTHON:-$HOME/.local/share/stickman-vibevoice/venv-rebuild/bin/python}"
if [ ! -x "$PY" ]; then
  PY="$FORK/.venv/bin/python"
fi
if [ ! -x "$PY" ]; then
  echo "BLOCKED: no existing VibeVoice runtime Python; see scripts/local-hybrid/rebuild_venv.sh" >&2
  exit 2
fi
mkdir -p "$LOGDIR" "$OUT"
# Gate before model load: never start on a machine that is already too hot.
if ! "$HERE/thermal_preflight.sh" "$LOGDIR/thermal.log"; then
  echo "exit=2" >> "$LOGDIR/synthesis.log"
  exit 2
fi
# Reversible, process-scoped intensity limits. Everything is tied to the worker's own process
# tree and ends with it; nothing host-wide, privileged or persistent is changed.
#  1. one named user-scope cgroup with a sustained CPU quota (percent of one core);
#  2. affinity to a fixed set of CPUs (default: one hardware thread on each of two physical cores);
#  3. OMP/MKL thread counts from the profile (and torch.set_num_threads in synthesize_local.py).
# Fail closed: a configured limit that cannot be applied blocks the run instead of starting uncapped.
UNIT="vv-synth-$$"
LAUNCH=()
if [ -n "${THERMAL_CPU_QUOTA:-}" ]; then
  if ! systemd-run --user --scope -q -p "CPUQuota=$THERMAL_CPU_QUOTA" true >/dev/null 2>&1; then
    echo "BLOCKED: CPU quota $THERMAL_CPU_QUOTA cannot be applied (systemd-run --user unavailable)" >> "$LOGDIR/thermal.log"
    echo "exit=2" >> "$LOGDIR/synthesis.log"
    exit 2
  fi
  LAUNCH=(systemd-run --user --scope -q --unit="$UNIT" -p "CPUQuota=$THERMAL_CPU_QUOTA" --)
fi
if [ -n "${THERMAL_CPU_AFFINITY:-}" ]; then
  if ! taskset -c "$THERMAL_CPU_AFFINITY" true >/dev/null 2>&1; then
    echo "BLOCKED: CPU affinity $THERMAL_CPU_AFFINITY cannot be applied (taskset unavailable)" >> "$LOGDIR/thermal.log"
    echo "exit=2" >> "$LOGDIR/synthesis.log"
    exit 2
  fi
  LAUNCH+=(taskset -c "$THERMAL_CPU_AFFINITY")
fi
"${LAUNCH[@]}" "$PY" "$HERE/synthesize_local.py" --dialogue "$DIALOGUE" --out-dir "$OUT" > "$LOGDIR/synthesis.log" 2>&1 &
PID=$!
echo "$PID" > "$LOGDIR/synthesis.pid"
"$HERE/thermal_guard.sh" "$PID" "$LOGDIR/thermal.log" &
GUARD=$!
residual_workers() {  # the scope owns the whole worker tree, including any orphaned child
  if [ -n "${THERMAL_CPU_QUOTA:-}" ] && systemctl --user is-active --quiet "$UNIT.scope" 2>/dev/null; then
    systemctl --user kill --signal=SIGKILL "$UNIT.scope" 2>/dev/null || true
    sleep 1
    echo "$(date -u +%FT%TZ) RESIDUAL_WORKERS_KILLED unit=$UNIT" >> "$LOGDIR/thermal.log"
    return 0
  fi
  echo "$(date -u +%FT%TZ) RESIDUAL_WORKERS=0" >> "$LOGDIR/thermal.log"
}
cleanup() {
  kill -CONT "$PID" 2>/dev/null || true
  kill "$GUARD" 2>/dev/null || true
  residual_workers
}
on_signal() {
  kill -CONT "$PID" 2>/dev/null || true
  kill -TERM "$PID" 2>/dev/null || true
  exit 143
}
trap cleanup EXIT
trap on_signal INT TERM
set +e
# The worker must never run unsupervised: if the guard exits while the worker is alive (crash or
# kill), the worker is stopped and the run fails.
wait -n "$PID" "$GUARD"
if kill -0 "$PID" 2>/dev/null; then
  echo "$(date -u +%FT%TZ) GUARD_LOST: watchdog exited while the worker was running; stopping worker" >> "$LOGDIR/thermal.log"
  kill -CONT "$PID" 2>/dev/null || true
  kill -TERM "$PID" 2>/dev/null || true
  sleep 3
  kill -KILL "$PID" 2>/dev/null || true
fi
wait "$PID"
RC=$?
wait "$GUARD" 2>/dev/null
set -e
if [ "$RC" -eq 0 ] && grep -qE 'CRITICAL_THERMAL_STOP|THERMAL_PAUSE_TIMEOUT|BLOCKED:|GUARD_LOST' "$LOGDIR/thermal.log"; then
  RC=1
fi
echo "exit=$RC" >> "$LOGDIR/synthesis.log"
exit "$RC"
