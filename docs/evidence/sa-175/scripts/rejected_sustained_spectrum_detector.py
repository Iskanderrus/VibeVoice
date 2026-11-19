"""Detector for sustained non-speech sound (music/ambience drop-ins) in generated speech.

Rationale: speech changes its spectrum every ~100-250 ms (phonemes, pitch movement). Background music/ambience holds a stable
spectrum for >= 0.5 s. A drop-in is a run of frames that (a) is clearly above the digital/room floor, (b) has a
stable spectrum (adjacent-frame cosine similarity high) and (c) lasts at least MIN_RUN seconds.  Runs overlapping strong
speech (a sustained vowel) are rare at this length; speech-level frames are excluded by capping the level so only
quiet-to-moderate sound counts (music under/between speech sits well below the speech peak level).
Heuristic, calibrated on this project's takes; reports timestamps so a human can confirm.
"""
import numpy as np
from scipy.signal import stft

SR = 24000
NPERSEG, HOP = 768, 384          # 32 ms window, 16 ms hop
LEVEL_MIN, LEVEL_MAX = -58.0, -22.0   # dB re take peak frame: above floor, below speech body
SIM_MIN = 0.92                   # adjacent-frame spectral cosine similarity for "stable"
MIN_RUN = 0.50                   # seconds

def analyse(w, sr=SR):
    f, t, Z = stft(w, sr, nperseg=NPERSEG, noverlap=NPERSEG - HOP)
    M = np.abs(Z)
    band = (f >= 80) & (f <= 8000)
    Mb = np.log1p(M[band] * 50.0)
    db = 10 * np.log10((M[band] ** 2).sum(0) + 1e-12); db -= db.max()
    nrm = Mb / (np.linalg.norm(Mb, axis=0, keepdims=True) + 1e-9)
    sim = np.r_[0.0, (nrm[:, 1:] * nrm[:, :-1]).sum(0)]
    stable = (sim > SIM_MIN) & (db > LEVEL_MIN) & (db < LEVEL_MAX)
    runs, i, n = [], 0, len(stable)
    step = HOP / sr
    while i < n:
        if stable[i]:
            j = i
            while j < n and stable[j]: j += 1
            if (j - i) * step >= MIN_RUN:
                runs.append((round(float(t[i]), 2), round(float(t[j - 1]), 2), round(float(np.mean(db[i:j])), 1)))
            i = j
        else:
            i += 1
    total = sum(b - a for a, b, _ in runs)
    return {"runs": runs, "seconds": round(total, 2), "frac": round(total / (len(w) / sr), 3), "dur": round(len(w) / sr, 1)}

if __name__ == "__main__":
    import sys, soundfile as sf
    for p in sys.argv[1:]:
        w, sr = sf.read(p); w = w.mean(1) if w.ndim > 1 else w
        r = analyse(w, sr)
        print(p.split("/")[-1][:44], "dur", r["dur"], "sustained_s", r["seconds"], "frac", r["frac"], "runs", r["runs"][:8])
