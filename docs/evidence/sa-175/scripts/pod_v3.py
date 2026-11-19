"""Podcast v2: explicit voices (RED=Frank, BLUE=Carter), per-scene QC with seed retry, tempo/pause normalisation, loudness.

Automatic after launch. Each take is a guarded run_stickman_local.py run (same thermal limits, boost off).
QC per take: background floor (10th percentile of 20 ms RMS) <= -55 dB, voice pitch median in [95,155] Hz and p90 <= 215 Hz.
Up to 3 seeds per scene; if none passes, the best take (lowest music share among those with pitch in band) is used and flagged.
"""
import json, os, subprocess, sys, time
import numpy as np, soundfile as sf
sys.path.insert(0, "/tmp/claude-1002/vvpre")
import postprocess as pp

REPO = os.path.expanduser("~/Projects/tools/VibeVoice-pr8")
PY = os.path.expanduser("~/.local/share/stickman-vibevoice/venv-rebuild/bin/python")
SCEN = "/tmp/claude-1002/-home-alex-study-Projects-products-content-factory-os/4c65935b-378e-49ca-9850-13dfb7ecbccd/scratchpad/podcast_v3"
RED = f"{REPO}/demo/voices/en-Frank_man.wav"
BLUE = f"{REPO}/demo/voices/en-Carter_man.wav"
SEEDS = [18427, 18428, 18429, 18430, 18431]
CFG = "1.6"
FLOOR_MAX, F0_BAND, F0_P90_MAX = -55.0, (95.0, 155.0), 215.0
MUSIC_FRAC_MAX = 0.15   # share of 2 s windows whose quietest 5% stays above -48 dBFS (clean refs 0.04-0.19; BGM ref 1.0)
T = time.strftime("%Y%m%d-%H%M%S")
LOG = "/tmp/claude-1002/vvpre/pod3.log"
def log(m):
    open(LOG, "a").write(f"{time.strftime('%H:%M:%S')} {m}\n")

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

def busy():
    return subprocess.run(["pgrep", "-f", "graphifyy/bin/python|dartaotruntime|design_audit"], capture_output=True).returncode == 0

def take(i, seed):
    for attempt in range(1, 7):
        while busy(): time.sleep(2)
        job = f"vv-pod3-scene{i}-s{seed}-{T}"
        log(f"scene={i} seed={seed} try={attempt} start {job}")
        p = subprocess.run([PY, f"{REPO}/scripts/run_stickman_local.py", "--foreground", "--script", f"{SCEN}/scene{i}.txt",
                            "--job-id", job, "--voice", f"RED={RED}", "--voice", f"BLUE={BLUE}", "--seed", str(seed), "--cfg-scale", CFG],
                           capture_output=True, text=True)
        log(f"scene={i} seed={seed} try={attempt} exit={p.returncode}")
        if p.returncode == 0: return job
        if "PREFLIGHT" not in (p.stdout + p.stderr): raise SystemExit(f"scene {i} failed non-preflight: {(p.stderr or p.stdout)[-300:]}")
    raise SystemExit(f"scene {i}: preflight never cleared")

chosen = []; report = []
for i in (1, 2, 3, 4):
    takes = []
    for seed in SEEDS:
        job = take(i, seed); q = qc(f"{pp.OUT}/{job}.wav"); q.update(job=job, seed=seed); takes.append(q)
        log(f"scene={i} seed={seed} QC {json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in q.items()})}")
        if q["pass"]: break
    best = next((t for t in takes if t["pass"]), None)
    if best is None:
        inband = [t for t in takes if F0_BAND[0] <= t["f0_med"] <= F0_BAND[1]] or takes
        best = min(inband, key=lambda t: (t["music_frac"], t["floor_db"])); best = dict(best, flagged=True)
    chosen.append(best); report.append({"scene": i, "takes": takes, "chosen": best["job"], "qc_met": best["pass"]})
    log(f"scene={i} chosen {best['job']} qc_met={best['pass']}")

scenes = [pp.load(f"{pp.OUT}/{c['job']}.wav") for c in chosen]
rates = [pp.rate(w) for w in scenes]; target = float(np.median(rates)); fixed = []
for k, (w, r) in enumerate(zip(scenes, rates)):
    f = target / r
    if abs(f - 1) < pp.DEADBAND: f = 1.0
    f = float(np.clip(f, *pp.TEMPO_CLAMP))
    w2 = pp.atempo(w, f) if f != 1.0 else w
    fixed.append(pp.assemble(w2, pp.segments(w2))); log(f"scene{k+1} rate={r:.2f} tempo={f:.3f}")
parts = []
for k, w in enumerate(fixed):
    parts.append(w)
    if k < len(fixed) - 1: parts.append(np.zeros(int(pp.SCENE_GAP * pp.SR)))
final = np.concatenate(parts)
tmp_in, tmp_out = f"{pp.OUT}/.pod2-pre.wav", f"{pp.OUT}/vv-blackswan-podcast-v3-frank-carter.wav"
sf.write(tmp_in, final / max(1.0, np.abs(final).max() / 0.95), pp.SR, subtype="PCM_16")
target_lufs, tp = -16.0, -1.5
m = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", tmp_in, "-af", f"loudnorm=I={target_lufs}:TP={tp}:LRA=11:print_format=json", "-f", "null", "-"],
                   capture_output=True, text=True).stderr
j = json.loads(m[m.rindex("{"):])
af = (f"loudnorm=I={target_lufs}:TP={tp}:LRA=11:measured_I={j['input_i']}:measured_TP={j['input_tp']}:"
      f"measured_LRA={j['input_lra']}:measured_thresh={j['input_thresh']}:offset={j['target_offset']}:linear=true")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp_in, "-af", af, "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", tmp_out], check=True)
os.remove(tmp_in)
chk = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", tmp_out, "-af", "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"],
                     capture_output=True, text=True).stderr
k = json.loads(chk[chk.rindex("{"):])
out = {"output": tmp_out, "report": report, "loudness_before": {"I": j["input_i"], "TP": j["input_tp"]}, "loudness_after": {"I": k["input_i"], "TP": k["input_tp"]},
       "dur": len(final) / pp.SR, "rates": rates}
json.dump(out, open("/tmp/claude-1002/vvpre/pod3_report.json", "w"), indent=1, default=float)
log("ALLDONE " + json.dumps({"dur": round(out["dur"], 1), "LUFS_before": j["input_i"], "LUFS_after": k["input_i"], "TP_after": k["input_tp"]}))
