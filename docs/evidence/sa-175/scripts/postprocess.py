"""Fix 2: per-scene tempo normalisation + pause clamping. Writes a new WAV; never touches the sources."""
import subprocess, sys, tempfile, os
import numpy as np, soundfile as sf
from scipy.signal import butter, filtfilt, find_peaks

OUT = os.path.expanduser("~/.local/share/stickman-vibevoice/output")
SR = 24000
GAP_MIN, GAP_MAX, SCENE_GAP = 0.35, 0.80, 0.80   # seconds
TEMPO_CLAMP, DEADBAND = (0.90, 1.10), 0.04

def load(p):
    w, sr = sf.read(p)
    assert sr == SR
    return w.mean(1) if w.ndim > 1 else w

def envelope(w):
    b, a = butter(4, [300 / (SR / 2), 3000 / (SR / 2)], "band")
    x = filtfilt(b, a, w)
    b2, a2 = butter(2, 8 / (SR / 2))
    env = np.clip(filtfilt(b2, a2, np.abs(x)), 1e-9, None)
    return env[:: SR // 100]            # 100 Hz

def rate(w):
    e = envelope(w); db = 20 * np.log10(e); sp = db > db.max() - 35
    pk, _ = find_peaks(e, prominence=0.08 * e.max(), distance=9)
    pk = pk[sp[pk]]
    return len(pk) / (sp.sum() / 100.0)   # syllable-like peaks per second of speech

def segments(w, min_gap=0.25):
    e = envelope(w); db = 20 * np.log10(e); sp = db > db.max() - 38
    n = len(sp); segs = []; i = 0; start = None
    while i < n:
        if sp[i]:
            if start is None: start = i
            j = i
            while j < n and sp[j]: j += 1
            i = j
        else:
            j = i
            while j < n and not sp[j]: j += 1
            if start is not None and (j - i) * 0.01 >= min_gap and j < n:
                segs.append((start, i)); start = None
            i = j
    if start is not None: segs.append((start, n))
    return [(a * 0.01, b * 0.01) for a, b in segs]

def atempo(w, f):
    with tempfile.TemporaryDirectory() as d:
        a, b = os.path.join(d, "a.wav"), os.path.join(d, "b.wav")
        sf.write(a, w, SR, subtype="PCM_16")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a, "-af", f"atempo={f:.4f}", b], check=True)
        return load(b)

def assemble(w, segs, pad=0.05):
    out = []
    for k, (a, b) in enumerate(segs):
        a2, b2 = max(0, a - pad), min(len(w) / SR, b + pad)
        out.append(w[int(a2 * SR): int(b2 * SR)])
        if k < len(segs) - 1:
            gap = segs[k + 1][0] - b
            out.append(np.zeros(int(np.clip(gap, GAP_MIN, GAP_MAX) * SR)))
    return np.concatenate(out)

def report(label, w):
    r = rate(w); s = segments(w)
    g = [s[i + 1][0] - s[i][1] for i in range(len(s) - 1)]
    return f"{label}: dur={len(w)/SR:.1f}s rate={r:.2f}/s gaps min={min(g):.2f} max={max(g):.2f} sd={np.std(g):.2f}"

if __name__ == "__main__":
    jobs = [l.strip() for l in open("/tmp/claude-1002/vvpre/scene_jobs.txt") if l.strip()]
    scenes = [load(f"{OUT}/{j}.wav") for j in jobs]
    rates = [rate(w) for w in scenes]
    target = float(np.median(rates))
    print("scene rates:", [round(r, 2) for r in rates], "target", round(target, 2))
    fixed = []
    for k, (w, r) in enumerate(zip(scenes, rates)):
        f = target / r
        if abs(f - 1) < DEADBAND: f = 1.0
        f = float(np.clip(f, *TEMPO_CLAMP))
        w2 = atempo(w, f) if f != 1.0 else w
        segs = segments(w2)
        fixed.append(assemble(w2, segs))
        print(f"scene{k+1}: tempo x{f:.3f}  {len(w)/SR:.1f}s -> {len(fixed[-1])/SR:.1f}s")
    parts = []
    for k, w in enumerate(fixed):
        parts.append(w)
        if k < len(fixed) - 1: parts.append(np.zeros(int(SCENE_GAP * SR)))
    final = np.concatenate(parts)
    final = final / max(1.0, np.abs(final).max() / 0.95)
    out = f"{OUT}/vv-blackswan-g1-perscene-normalised.wav"
    sf.write(out, final, SR, subtype="PCM_16")
    print(report("BEFORE (single pass)", load(f"{OUT}/vv-blackswan-g1c-20261119-175533.wav")))
    print(report("AFTER  (per-scene+norm)", final))
    print("wrote", out)
