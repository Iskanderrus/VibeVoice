#!/usr/bin/env bash
# Make the 1.8-semitone listening copy (formants kept, loudness-matched) for every batch / whole-piece raw take that lacks one.
O=~/.local/share/stickman-vibevoice/output
r=$(python3 -c "print(round(2**(-1.8/12),5))")
for raw in $O/vv-batch-s*-seed*-2026????-??????.wav $O/vv-whole-seed*-2026????-??????.wav; do
  [ -f "$raw" ] || continue
  out="${raw%.wav}-down1.8st-ln16.wav"
  [ -f "$out" ] && continue
  ffmpeg -v error -y -i "$raw" -af "rubberband=pitch=$r:formant=preserved:pitchq=quality,loudnorm=I=-16:TP=-1.5:LRA=11" -ar 24000 -ac 1 -c:a pcm_s16le "$out" && echo "made $(basename "$out")"
done
