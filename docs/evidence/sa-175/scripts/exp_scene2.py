"""Controlled comparison on ONE scene (scene 2, original dialogue text), Frank=RED / Carter=BLUE, boost off, same guarded limits.
Grid: cfg_scale x seed. The two takes already on disk (cfg 1.6, seeds 18427/18428) are the first two cells and are not re-run.
No take is rejected: every take is kept, measured, and given a loudness-matched copy (-16 LUFS) for fair listening.
"""
import json, os, subprocess, sys, time
import numpy as np
sys.path.insert(0, "/tmp/claude-1002/vvpre")
import qcmod, postprocess as pp

REPO = os.path.expanduser("~/Projects/tools/VibeVoice-pr8")
PY = os.path.expanduser("~/.local/share/stickman-vibevoice/venv-rebuild/bin/python")
SCRIPT = "/tmp/claude-1002/-home-alex-study-Projects-products-content-factory-os/4c65935b-378e-49ca-9850-13dfb7ecbccd/scratchpad/podcast/scene2.txt"
RED, BLUE = f"{REPO}/demo/voices/en-Frank_man.wav", f"{REPO}/demo/voices/en-Carter_man.wav"
GRID = [("1.3", 18427), ("1.3", 18428), ("1.3", 18429), ("2.0", 18427), ("2.0", 18428), ("2.0", 18429), ("1.6", 18429)]
LOG = "/tmp/claude-1002/vvpre/exp.log"
T = time.strftime("%Y%m%d-%H%M%S")
def log(m): open(LOG, "a").write(f"{time.strftime('%H:%M:%S')} {m}\n")
def busy(): return subprocess.run(["pgrep", "-f", "graphifyy/bin/python|dartaotruntime|design_audit"], capture_output=True).returncode == 0

def loud_copy(src, dst):
    m = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", src, "-af", "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"],
                       capture_output=True, text=True).stderr
    j = json.loads(m[m.rindex("{"):])
    af = (f"loudnorm=I=-16:TP=-1.5:LRA=11:measured_I={j['input_i']}:measured_TP={j['input_tp']}:measured_LRA={j['input_lra']}:"
          f"measured_thresh={j['input_thresh']}:offset={j['target_offset']}:linear=true")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-af", af, "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", dst], check=True)
    return float(j["input_i"])

results = []
for cfg, seed in GRID:
    job = f"vv-exp-s2-cfg{cfg}-s{seed}-{T}"
    for attempt in range(1, 7):
        while busy(): time.sleep(2)
        log(f"cfg={cfg} seed={seed} try={attempt} start {job}")
        p = subprocess.run([PY, f"{REPO}/scripts/run_stickman_local.py", "--foreground", "--script", SCRIPT, "--job-id", job,
                            "--voice", f"RED={RED}", "--voice", f"BLUE={BLUE}", "--seed", str(seed), "--cfg-scale", cfg], capture_output=True, text=True)
        log(f"cfg={cfg} seed={seed} try={attempt} exit={p.returncode}")
        if p.returncode == 0: break
        if "PREFLIGHT" not in (p.stdout + p.stderr): raise SystemExit(f"failed non-preflight: {(p.stderr or p.stdout)[-300:]}")
    else:
        raise SystemExit("preflight never cleared")
    wav = f"{pp.OUT}/{job}.wav"; q = qcmod.qc(wav)
    lufs = loud_copy(wav, f"{pp.OUT}/{job}-ln16.wav")
    row = {"cfg": cfg, "seed": seed, "job": job, "floor_db": round(q["floor_db"], 1), "music_frac": round(q["music_frac"], 2),
           "f0_med": round(q["f0_med"], 1), "f0_p90": round(q["f0_p90"], 1), "dur": round(q["dur"], 1), "lufs_raw": lufs, "music_at": q["music_at"][:6]}
    results.append(row); log("RESULT " + json.dumps(row))
json.dump(results, open("/tmp/claude-1002/vvpre/exp_report.json", "w"), indent=1)
log("ALLDONE")
