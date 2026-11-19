#!/bin/bash
# Shared thermal readers for the hybrid runner. Each prints an integer in °C, or nothing
# when the sensor is unavailable or implausible. Sourced by thermal_preflight.sh and
# thermal_guard.sh.
#
# CPU: the k10temp hwmon sensor is read directly from sysfs (microseconds, no fork), taking the
# hottest of Tctl/Tdie, so a fast rise is seen quickly; `sensors` is only a fallback. This
# CPU family (Ryzen 7 3750H) exposes Tctl only and no max/crit file, so no hardware limit is
# inferred from the sensor; the configured pause/resume/critical limits are the only authority.
# A reading outside 1..119 °C is treated as a sensor failure, never as a safe value.

THERMAL_HWMON_ROOT=${THERMAL_HWMON_ROOT:-/sys/class/hwmon}

_plausible_c() {
  case "$1" in ''|*[!0-9]*) return 1 ;; esac
  [ "$1" -ge 1 ] && [ "$1" -le 119 ]
}

read_cpu_c() {
  local best="" dir f v
  for dir in "$THERMAL_HWMON_ROOT"/hwmon*; do
    [ "$(cat "$dir/name" 2>/dev/null)" = "k10temp" ] || continue
    for f in "$dir"/temp*_input; do
      [ -r "$f" ] || continue
      case "$(cat "${f%_input}_label" 2>/dev/null)" in Tctl|Tdie) ;; *) continue ;; esac
      v="$(cat "$f" 2>/dev/null)" || continue
      case "$v" in ''|*[!0-9]*) continue ;; esac
      v=$((v / 1000))
      if [ -z "$best" ] || [ "$v" -gt "$best" ]; then best=$v; fi
    done
  done
  if [ -z "$best" ]; then
    best="$(sensors 2>/dev/null | awk '/Tctl:|Tdie:|Package id 0:|Core 0:/{for(i=1;i<=NF;i++)if($i~/\+[0-9]+(\.[0-9]+)?/){gsub(/[^0-9.]/,"",$i);print int($i);exit}}' || true)"
  fi
  if _plausible_c "$best"; then echo "$best"; fi
}

read_gpu_c() {
  local v
  v="$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -cd '0-9' || true)"
  if _plausible_c "$v"; then echo "$v"; fi
}
