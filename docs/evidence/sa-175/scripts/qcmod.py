import numpy as np, sys
sys.path.insert(0,'/tmp/claude-1002/vvpre')
import postprocess as pp
FLOOR_MAX, F0_BAND, F0_P90_MAX = -55.0, (95.0, 155.0), 215.0
MUSIC_FRAC_MAX = 0.15   # share of 2 s windows whose quietest 5% stays above -48 dBFS (clean refs 0.04-0.19; BGM ref 1.0)

def f0(x, sr, lo=70, hi=300):
    out = []; n = int(0.04 * sr); hop = int(0.02 * sr)
    for i in range(0, len(x) - n, hop):
        fr = x[i:i + n] - x[i:i + n].mean()
        if np.sqrt((fr ** 2).mean()) < 0.01: continue
        ac = np.correlate(fr, fr, "full")[n - 1:]; ac /= ac[0] + 1e-9
        a, b = int(sr / hi), int(sr / lo); k = a + np.argmax(ac[a:b])
        if ac[k] > 0.5: out.append(sr / k)
    return np.array(out)

def qc(path):
    w = pp.load(path); fr = int(0.02 * pp.SR); n = len(w) // fr
    db = 20 * np.log10(np.sqrt((w[:n * fr].reshape(n, fr) ** 2).mean(1)) + 1e-9)
    f = f0(w, pp.SR)
    r = {"floor_db": float(np.percentile(db, 10)), "f0_med": float(np.median(f)), "f0_p90": float(np.percentile(f, 90)),
         "dur": len(w) / pp.SR}
    win = [(i / 50.0, float(np.percentile(db[i:i + 100], 5))) for i in range(0, len(db) - 100, 50)]
    r["music_frac"] = float(np.mean([v > -48 for _, v in win])) if win else 0.0
    r["music_at"] = [round(t, 1) for t, v in win if v > -48][:12]
    r["pass"] = bool(r["music_frac"] <= MUSIC_FRAC_MAX and r["floor_db"] <= FLOOR_MAX and F0_BAND[0] <= r["f0_med"] <= F0_BAND[1] and r["f0_p90"] <= F0_P90_MAX)
    return r

