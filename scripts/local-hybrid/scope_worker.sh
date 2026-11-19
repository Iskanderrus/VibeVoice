#!/bin/bash
# Runs INSIDE the owned systemd scope and taskset, before any heavyweight import or model load:
# proves the intended limits are actually in force for this very process (and so for every child
# it will spawn), then execs the worker so the PID is unchanged. Any mismatch blocks the run.
# Usage: scope_worker.sh <worker command...>
# Expected values come from the environment set by run_local.sh:
#   VV_EXPECT_UNIT (scope name, empty = no scope), VV_EXPECT_QUOTA_PERCENT,
#   VV_EXPECT_AFFINITY (cpu list), VV_THERMAL_LOG
set -u
fail() {
  echo "$(date -u +%FT%TZ) BLOCKED: limits not applied: $1" >> "${VV_THERMAL_LOG:-/dev/stderr}"
  echo "BLOCKED: limits not applied: $1" >&2
  exit 2
}
if [ -n "${VV_EXPECT_UNIT:-}" ]; then
  cg="$(sed -n 's/^0:://p' /proc/self/cgroup)"
  case "$cg" in *"/${VV_EXPECT_UNIT}.scope") ;; *) fail "process is not in scope ${VV_EXPECT_UNIT}.scope (cgroup=${cg:-none})" ;; esac
  if [ -n "${VV_EXPECT_QUOTA_PERCENT:-}" ]; then
    want="$(( ${VV_EXPECT_QUOTA_PERCENT%\%} * 1000 )) 100000"
    got="$(cat "${THERMAL_CGROUP_ROOT:-/sys/fs/cgroup}$cg/cpu.max" 2>/dev/null || true)"
    [ "$got" = "$want" ] || fail "cpu.max is '${got:-unreadable}', expected '$want'"
  fi
fi
allowed="$(sed -n 's/^Cpus_allowed_list:[[:space:]]*//p' /proc/self/status)"
if [ -n "${VV_EXPECT_AFFINITY:-}" ] && [ "$allowed" != "$VV_EXPECT_AFFINITY" ]; then
  fail "affinity is '$allowed', expected '$VV_EXPECT_AFFINITY'"
fi
count="$(python3 - "$allowed" <<'PY'
import sys
n = 0
for part in sys.argv[1].split(","):
    lo, _, hi = part.partition("-")
    n += (int(hi) - int(lo) + 1) if hi else 1
print(n)
PY
)"
for var in OMP_NUM_THREADS MKL_NUM_THREADS; do
  val="${!var:-}"
  case "$val" in ''|*[!0-9]*) fail "$var is unset or not a number" ;; esac
  [ "$val" -ge 1 ] && [ "$val" -le "$count" ] || fail "$var=$val exceeds the $count allowed CPUs"
done
echo "$(date -u +%FT%TZ) LIMITS_APPLIED scope=${VV_EXPECT_UNIT:-none} quota=${VV_EXPECT_QUOTA_PERCENT:-none} affinity=$allowed omp=$OMP_NUM_THREADS mkl=$MKL_NUM_THREADS" >> "${VV_THERMAL_LOG:-/dev/null}"
exec "$@"
