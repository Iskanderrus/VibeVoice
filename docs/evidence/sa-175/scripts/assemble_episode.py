"""Full episode for ONE seed and ONE shift, method C (resample + atempo), all four scenes from that same seed.
usage: assemble_episode.py <seed> <semitones>
Per scene: raw take -> asetrate/aresample/atempo (lowers pitch by <semitones>, speed restored) -> single-pass loudnorm (as the operator heard the scene files)
-> trim empty ends, 15 ms fades -> 0.7 s between scenes -> final two-pass loudnorm to -16 LUFS / -1.5 dBTP. No tempo or pause editing.
"""
import glob, hashlib, json, os, subprocess, sys
import numpy as np, soundfile as sf
sys.path.insert(0, "/tmp/claude-1002/vvpre")
import postprocess as pp

O = os.path.expanduser("~/.local/share/stickman-vibevoice/output")
seed, semis = int(sys.argv[1]), float(sys.argv[2])
def raw(scene):
    if scene == 2:
        c = {18428: "vv-pod2-scene2-s18428-20261119-202821.wav", 18429: "vv-exp-s2-cfg1.6-s18429-20261119-210421.wav"}.get(seed)
        if c is None: c = (glob.glob(f"{O}/vv-s2-seed{seed}-*.wav") or [None])[0]
        else: c = f"{O}/{c}"
    else:
        c = (glob.glob(f"{O}/vv-batch-s{scene}-seed{seed}-2026????-??????.wav") or [None])[0]
    if not c or not os.path.isfile(c): raise SystemExit(f"missing raw take: scene {scene} seed {seed}")
    return c

r = 2 ** (-semis / 12)
def shifted(src, dst):
    af = f"asetrate={int(24000*r)},aresample=24000,atempo={round(1/r,5)},loudnorm=I=-16:TP=-1.5:LRA=11"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-af", af, "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", dst], check=True)

GAP, FADE = 0.7, int(0.015 * pp.SR)
parts, marks, t, srcs = [], [], 0.0, []
for sc in (1, 2, 3, 4):
    src = raw(sc); tmp = f"{O}/.ep-scene{sc}.wav"; shifted(src, tmp); w = pp.load(tmp).copy(); os.remove(tmp)
    fr = int(0.01 * pp.SR); n = len(w) // fr
    lv = 20 * np.log10(np.sqrt((w[:n * fr].reshape(n, fr) ** 2).mean(1)) + 1e-9)
    act = np.where(lv > -50)[0]; a = max(0, (act[0] - 25) * fr); b = min(len(w), (act[-1] + 26) * fr); w = w[a:b]
    w[:FADE] *= np.linspace(0, 1, FADE); w[-FADE:] *= np.linspace(1, 0, FADE)
    marks.append((f"scene {sc}", round(t, 1))); srcs.append(os.path.basename(src)); parts.append(w); t += len(w) / pp.SR
    if sc < 4: parts.append(np.zeros(int(GAP * pp.SR))); t += GAP
ep = np.concatenate(parts); pre = f"{O}/.ep-pre-{seed}.wav"
sf.write(pre, ep / max(1.0, np.abs(ep).max() / 0.95), pp.SR, subtype="PCM_16")
m = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", pre, "-af", "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"], capture_output=True, text=True).stderr
j = json.loads(m[m.rindex("{"):])
af = (f"loudnorm=I=-16:TP=-1.5:LRA=11:measured_I={j['input_i']}:measured_TP={j['input_tp']}:measured_LRA={j['input_lra']}:"
      f"measured_thresh={j['input_thresh']}:offset={j['target_offset']}:linear=true")
out = f"{O}/vv-blackswan-podcast-episode-seed{seed}-down{semis:g}st-methodC.wav"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", pre, "-af", af, "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", out], check=True); os.remove(pre)
c = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", out, "-af", "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"], capture_output=True, text=True).stderr
k = json.loads(c[c.rindex("{"):]); w = pp.load(out)
print(f"seed {seed} @ {semis:g} st: {len(w)/pp.SR:.1f}s  LUFS {k['input_i']}  TP {k['input_tp']}  sha256 {hashlib.sha256(open(out,'rb').read()).hexdigest()}")
print("  scene starts:", marks); print("  sources:", srcs)
