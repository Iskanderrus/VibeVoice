#!/bin/bash
# One command: load the hybrid profile, run synthesis, guard the process against heat.
# usage: run_local.sh <dialogue.json> <out-dir> [profile.env] [log-dir]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DIALOGUE="$1"; OUT="$2"
PROFILE="${3:-$HERE/profile-gtx1650ti-hybrid.env}"
LOGDIR="${4:-$OUT}"
PY="${VIBEVOICE_PYTHON:?set VIBEVOICE_PYTHON to the venv python (see rebuild_venv.sh)}"
set -a; . "$PROFILE"; set +a
mkdir -p "$LOGDIR"
"$PY" "$HERE/synthesize_local.py" --dialogue "$DIALOGUE" --out-dir "$OUT" > "$LOGDIR/synthesis.log" 2>&1 &
PID=$!
echo "$PID" > "$LOGDIR/synthesis.pid"
"$HERE/thermal_guard.sh" "$PID" "$LOGDIR/thermal.log" &
GUARD=$!
set +e; wait "$PID"; RC=$?; set -e
kill "$GUARD" 2>/dev/null || true
echo "exit=$RC" | tee -a "$LOGDIR/synthesis.log"
exit "$RC"
