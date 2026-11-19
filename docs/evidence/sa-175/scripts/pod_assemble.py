"""Assemble the podcast: per-scene WAVs -> tempo-normalised, pause-clamped single WAV (same fix-2 logic as the narration)."""
import os, sys, numpy as np, soundfile as sf
sys.path.insert(0, "/tmp/claude-1002/vvpre")
import postprocess as pp

jobs = [l.strip() for l in open("/tmp/claude-1002/vvpre/pod_jobs.txt") if l.strip()]
scenes = [pp.load(f"{pp.OUT}/{j}.wav") for j in jobs]
rates = [pp.rate(w) for w in scenes]
target = float(np.median(rates))
print("jobs", jobs); print("scene durations", [round(len(w)/pp.SR, 1) for w in scenes])
print("scene rates", [round(r, 2) for r in rates], "target", round(target, 2))
fixed = []
for k, (w, r) in enumerate(zip(scenes, rates)):
    f = target / r
    if abs(f - 1) < pp.DEADBAND: f = 1.0
    f = float(np.clip(f, *pp.TEMPO_CLAMP))
    w2 = pp.atempo(w, f) if f != 1.0 else w
    fixed.append(pp.assemble(w2, pp.segments(w2)))
    print(f"scene{k+1}: tempo x{f:.3f} {len(w)/pp.SR:.1f}s -> {len(fixed[-1])/pp.SR:.1f}s")
parts = []
for k, w in enumerate(fixed):
    parts.append(w)
    if k < len(fixed) - 1: parts.append(np.zeros(int(pp.SCENE_GAP * pp.SR)))
final = np.concatenate(parts); final = final / max(1.0, np.abs(final).max() / 0.95)
out = f"{pp.OUT}/vv-blackswan-podcast-perscene-normalised.wav"
sf.write(out, final, pp.SR, subtype="PCM_16")
print(pp.report("podcast raw concat", np.concatenate([np.concatenate([w, np.zeros(int(0.8*pp.SR))]) for w in scenes])))
print(pp.report("podcast normalised", final)); print("wrote", out)
