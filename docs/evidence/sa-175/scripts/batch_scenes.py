"""Batch of takes for scenes 1, 3, 4 (original dialogue), seeds 18427-18431, using the recipe that produced the operator's favourite:
Frank = RED, Carter = BLUE, guidance 1.6, boost off, same guarded limits. Nothing is rejected; each take is measured (music share, floor,
pitch, gaps and pitch falls inside speaker lines) and gets a listening copy lowered 1.5 semitones (formants kept) and loudness-matched.
"""
import json, os, subprocess, sys, time
import numpy as np
sys.path.insert(0, "/tmp/claude-1002/vvpre")
import qcmod, postprocess as pp, tighten

REPO = os.path.expanduser("~/Projects/tools/VibeVoice-pr8")
PY = os.path.expanduser("~/.local/share/stickman-vibevoice/venv-rebuild/bin/python")
SCEN = "/tmp/claude-1002/-home-alex-study-Projects-products-content-factory-os/4c65935b-378e-49ca-9850-13dfb7ecbccd/scratchpad/podcast"
RED, BLUE = f"{REPO}/demo/voices/en-Frank_man.wav", f"{REPO}/demo/voices/en-Carter_man.wav"
SCENES, SEEDS, CFG, SHIFT = (1, 3, 4), (18427, 18428, 18429, 18430, 18431), "1.6", 1.5
LOG = "/tmp/claude-1002/vvpre/batch.log"
T = time.strftime("%Y%m%d-%H%M%S")
def log(m): open(LOG, "a").write(f"{time.strftime('%H:%M:%S')} {m}\n")
def busy(): return subprocess.run(["pgrep", "-f", "graphifyy/bin/python|dartaotruntime|design_audit"], capture_output=True).returncode == 0

def med(x): return float(np.median(x)) if len(x) else float("nan")
def line_breaks(w, scene):
    turns = [l.split(": ", 1) for l in open(f"{SCEN}/scene{scene}.txt").read().splitlines() if l.strip()]
    sylls = [sum(tighten.syl(x) for x in t[1].split()) for t in turns]
    segs = pp.segments(w, min_gap=0.15)
    if len(segs) < len(turns): return None
    out = {"BLUE": [], "RED": []}
    for ti, g in enumerate(tighten.assign(segs, sylls)):
        for a, b in zip(g[:-1], g[1:]):
            gap = segs[b][0] - segs[a][1]
            if gap < 0.15: continue
            e = qcmod.f0(w[int(max(segs[a][0], segs[a][1] - 0.35) * pp.SR): int((segs[a][1] - 0.12) * pp.SR)], pp.SR)
            l = qcmod.f0(w[int((segs[a][1] - 0.12) * pp.SR): int(segs[a][1] * pp.SR)], pp.SR)
            fall = 100 * (med(l) - med(e)) / med(e) if len(e) and len(l) else None
            out[turns[ti][0]].append({"turn": ti + 1, "gap": round(gap, 2), "fall_pct": None if fall is None or fall != fall else round(fall)})
    return out

def listening_copy(src, dst):
    r = round(2 ** (-SHIFT / 12), 5)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-af", f"rubberband=pitch={r}:formant=preserved:pitchq=quality,loudnorm=I=-16:TP=-1.5:LRA=11",
                    "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", dst], check=True)

results = []
for scene in SCENES:
    for seed in SEEDS:
        job = f"vv-batch-s{scene}-seed{seed}-{T}"
        for attempt in range(1, 7):
            while busy(): time.sleep(2)
            log(f"scene={scene} seed={seed} try={attempt} start {job}")
            p = subprocess.run([PY, f"{REPO}/scripts/run_stickman_local.py", "--foreground", "--script", f"{SCEN}/scene{scene}.txt", "--job-id", job,
                                "--voice", f"RED={RED}", "--voice", f"BLUE={BLUE}", "--seed", str(seed), "--cfg-scale", CFG], capture_output=True, text=True)
            log(f"scene={scene} seed={seed} try={attempt} exit={p.returncode}")
            if p.returncode == 0: break
            if "PREFLIGHT" not in (p.stdout + p.stderr): raise SystemExit(f"failed non-preflight: {(p.stderr or p.stdout)[-300:]}")
        else:
            raise SystemExit("preflight never cleared")
        wav = f"{pp.OUT}/{job}.wav"; q = qcmod.qc(wav); w = pp.load(wav); br = line_breaks(w, scene)
        listening_copy(wav, f"{pp.OUT}/{job}-down{SHIFT}st-ln16.wav")
        row = {"scene": scene, "seed": seed, "job": job, "dur": round(q["dur"], 1), "floor_db": round(q["floor_db"], 1), "music_frac": round(q["music_frac"], 2),
               "f0_med": round(q["f0_med"], 1), "f0_p90": round(q["f0_p90"], 1), "music_at": q["music_at"][:6], "breaks": br,
               "blue_break_free": (br is not None and not br["BLUE"])}
        results.append(row); log("RESULT " + json.dumps(row))
json.dump(results, open("/tmp/claude-1002/vvpre/batch_report.json", "w"), indent=1)
log("ALLDONE")
