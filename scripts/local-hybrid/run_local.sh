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
"$PY" "$HERE/synthesize_local.py" --dialogue "$DIALOGUE" --out-dir "$OUT" > "$LOGDIR/synthesis.log" 2>&1 &
PID=$!
echo "$PID" > "$LOGDIR/synthesis.pid"
"$HERE/thermal_guard.sh" "$PID" "$LOGDIR/thermal.log" &
GUARD=$!
cleanup() {
  kill -CONT "$PID" 2>/dev/null || true
  kill "$GUARD" 2>/dev/null || true
}
on_signal() {
  kill -CONT "$PID" 2>/dev/null || true
  kill -TERM "$PID" 2>/dev/null || true
  exit 143
}
trap cleanup EXIT
trap on_signal INT TERM
set +e
wait "$PID"
RC=$?
set -e
if [ "$RC" -eq 0 ] && grep -qE 'CRITICAL_THERMAL_STOP|THERMAL_PAUSE_TIMEOUT|BLOCKED:' "$LOGDIR/thermal.log"; then
  RC=1
fi
echo "exit=$RC" >> "$LOGDIR/synthesis.log"
exit "$RC"
