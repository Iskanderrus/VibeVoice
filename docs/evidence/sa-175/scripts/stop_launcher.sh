#!/usr/bin/env bash
# Cancel a queued (not yet started) launcher script, named by basename in $1. Never touches a running synthesis.
for pid in $(pgrep -f "bash .*/$1"); do kill "$pid" && echo "cancelled queued launcher pid $pid ($1)"; done
