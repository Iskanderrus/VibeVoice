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
#  1. one named user-scope cgroup with a sustained CPU quota (percent of one core). The scope is
#     mandatory: it is the owned identity used for whole-tree signalling and for cleanup proof;
#  2. affinity to a fixed set of CPUs (default: one hardware thread on each of two physical cores);
#  3. OMP/MKL thread counts from the profile (and torch.set_num_threads in synthesize_local.py).
# scope_worker.sh verifies all three inside the launched process before the model is touched.
# Fail closed: a limit that cannot be applied blocks the run instead of starting uncapped.
block() {
  echo "$(date -u +%FT%TZ) BLOCKED: $1" >> "$LOGDIR/thermal.log"
  echo "exit=2" >> "$LOGDIR/synthesis.log"
  exit 2
}
[ -n "${THERMAL_CPU_QUOTA:-}" ] || block "an owned CPU-quota scope is required (THERMAL_CPU_QUOTA unset)"
UNIT="vv-synth-$$"
systemd-run --user --scope -q -p "CPUQuota=$THERMAL_CPU_QUOTA" true >/dev/null 2>&1 \
  || block "CPU quota $THERMAL_CPU_QUOTA cannot be applied (systemd-run --user unavailable)"
LAUNCH=(systemd-run --user --scope -q --unit="$UNIT" -p "CPUQuota=$THERMAL_CPU_QUOTA" --)
if [ -n "${THERMAL_CPU_AFFINITY:-}" ]; then
  taskset -c "$THERMAL_CPU_AFFINITY" true >/dev/null 2>&1 \
    || block "CPU affinity $THERMAL_CPU_AFFINITY cannot be applied (taskset unavailable)"
  LAUNCH+=(taskset -c "$THERMAL_CPU_AFFINITY")
fi
export VV_EXPECT_UNIT="$UNIT" VV_EXPECT_QUOTA_PERCENT="$THERMAL_CPU_QUOTA" \
  VV_EXPECT_AFFINITY="${THERMAL_CPU_AFFINITY:-}" VV_THERMAL_LOG="$LOGDIR/thermal.log"
"${LAUNCH[@]}" "$HERE/scope_worker.sh" "$PY" "$HERE/synthesize_local.py" --dialogue "$DIALOGUE" --out-dir "$OUT" > "$LOGDIR/synthesis.log" 2>&1 &
PID=$!
echo "$PID" > "$LOGDIR/synthesis.pid"
THERMAL_SCOPE_UNIT="$UNIT" "$HERE/thermal_guard.sh" "$PID" "$LOGDIR/thermal.log" &
GUARD=$!

scope_state() {  # prints active|gone|unknown; "unknown" is never treated as "gone"
  local load active
  load="$(systemctl --user show -p LoadState --value "$UNIT.scope" 2>/dev/null)" || { echo unknown; return; }
  active="$(systemctl --user show -p ActiveState --value "$UNIT.scope" 2>/dev/null)" || { echo unknown; return; }
  case "$load/$active" in
    not-found/inactive) echo gone ;;
    loaded/inactive|loaded/failed) echo gone ;;
    loaded/active|loaded/activating|loaded/deactivating) echo active ;;
    *) echo unknown ;;
  esac
}
scope_procs() {  # number of processes still in the owned scope's cgroup; empty when unreadable
  local cg
  cg="$(systemctl --user show -p ControlGroup --value "$UNIT.scope" 2>/dev/null)" || return 0
  [ -n "$cg" ] || { echo 0; return; }
  if [ -r "${THERMAL_CGROUP_ROOT:-/sys/fs/cgroup}$cg/cgroup.procs" ]; then
    wc -l < "${THERMAL_CGROUP_ROOT:-/sys/fs/cgroup}$cg/cgroup.procs"
  else
    echo 0
  fi
}
# Positive proof that no worker is left: returns 0 only for "scope gone" or "killed and emptied".
# Sets CLEANUP_NOTE. A worker that was still present at this point counts as a residual even if
# it is killed here, so a clean-looking exit status can never hide a leftover process.
verify_cleanup() {
  local state i
  state="$(scope_state)"
  case "$state" in
    gone) CLEANUP_NOTE="RESIDUAL_WORKERS=0 scope=gone"; return 0 ;;
    unknown) CLEANUP_NOTE="CLEANUP_UNVERIFIED scope state unknown"; return 1 ;;
  esac
  systemctl --user kill --kill-whom=all --signal=SIGKILL "$UNIT.scope" 2>/dev/null \
    || { CLEANUP_NOTE="CLEANUP_UNVERIFIED scope kill failed"; return 1; }
  for i in 1 2 3 4 5 6 7 8 9 10; do
    sleep 0.5
    state="$(scope_state)"
    if [ "$state" = gone ] || { [ "$state" = active ] && [ "$(scope_procs)" = 0 ]; }; then
      CLEANUP_NOTE="RESIDUAL_WORKERS_KILLED scope emptied after kill"
      return 1
    fi
  done
  CLEANUP_NOTE="CLEANUP_UNVERIFIED scope still populated after kill"
  return 1
}
on_signal() {
  kill -CONT "$PID" 2>/dev/null || true
  kill -TERM "$PID" 2>/dev/null || true
  kill "$GUARD" 2>/dev/null || true
  systemctl --user kill --kill-whom=all --signal=SIGKILL "$UNIT.scope" 2>/dev/null || true
  exit 143
}
on_exit() {  # last resort on any abnormal exit: never leave the guard or the scope behind
  kill "$GUARD" 2>/dev/null || true
  if [ "$(scope_state)" != gone ]; then
    systemctl --user kill --kill-whom=all --signal=SIGKILL "$UNIT.scope" 2>/dev/null || true
  fi
}
trap on_exit EXIT
trap on_signal INT TERM
set +e
# The worker must never run unsupervised: if the guard exits while the worker is alive (crash or
# kill), the worker is stopped and the run fails.
FIRST=""
wait -n -p FIRST "$PID" "$GUARD"
FIRST_RC=$?
if [ "$FIRST" = "$GUARD" ]; then
  GUARD_RC=$FIRST_RC
  if kill -0 "$PID" 2>/dev/null; then
    echo "$(date -u +%FT%TZ) GUARD_LOST: watchdog exited (status $GUARD_RC) while the worker was running; stopping worker" >> "$LOGDIR/thermal.log"
    systemctl --user kill --kill-whom=all --signal=SIGCONT "$UNIT.scope" 2>/dev/null
    systemctl --user kill --kill-whom=all --signal=SIGTERM "$UNIT.scope" 2>/dev/null
    sleep 3
    systemctl --user kill --kill-whom=all --signal=SIGKILL "$UNIT.scope" 2>/dev/null
  fi
  wait "$PID"
  RC=$?
else
  RC=$FIRST_RC
  wait "$GUARD" 2>/dev/null
  GUARD_RC=$?
fi
CLEANUP_NOTE=""
verify_cleanup
CLEAN=$?
echo "$(date -u +%FT%TZ) $CLEANUP_NOTE" >> "$LOGDIR/thermal.log"
set -e
if [ "$RC" -eq 0 ]; then
  if [ "$CLEAN" -ne 0 ]; then
    echo "$(date -u +%FT%TZ) FAILED: worker exited 0 but cleanup is not verified clean" >> "$LOGDIR/thermal.log"
    RC=1
  elif [ "$GUARD_RC" -ne 0 ]; then
    echo "$(date -u +%FT%TZ) FAILED: watchdog reported a thermal/sensor stop (exit $GUARD_RC)" >> "$LOGDIR/thermal.log"
    RC=1
  elif grep -qE 'CRITICAL_THERMAL_STOP|THERMAL_PAUSE_TIMEOUT|BLOCKED:|GUARD_LOST' "$LOGDIR/thermal.log"; then
    RC=1
  fi
elif [ "$CLEAN" -ne 0 ]; then
  echo "$(date -u +%FT%TZ) FAILED: worker failed and cleanup is not verified clean" >> "$LOGDIR/thermal.log"
fi
echo "exit=$RC" >> "$LOGDIR/synthesis.log"
exit "$RC"
