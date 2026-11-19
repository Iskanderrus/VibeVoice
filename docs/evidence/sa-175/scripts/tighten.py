"""Tighten pauses inside a speaker's line, keep (and bound) the pauses between lines.

Without a speech recogniser, speech segments (split at gaps >= 0.15 s) are assigned to the known turns in order by dynamic
programming on estimated syllable counts. Gaps between segments of the SAME turn are capped (INTRA_MAX); gaps between turns
are clamped to [INTER_MIN, INTER_MAX]. The alignment is an estimate: the printed mapping lets a human check it.
usage: tighten.py in.wav scene.txt out.wav
"""
import re, sys
import numpy as np, soundfile as sf
sys.path.insert(0, "/tmp/claude-1002/vvpre")
import postprocess as pp

import os
INTRA_MAX, INTER_MIN, INTER_MAX, PAD, XF = float(os.environ.get("INTRA_MAX", "0.20")), 0.35, 0.65, 0.04, 0.010

def syl(word):
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w: return 0
    c = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and c > 1 and not w.endswith("le"): c -= 1
    return max(c, 1)

def assign(segs, sylls):
    """Monotonic partition of segments into len(sylls) non-empty groups minimising squared duration mismatch."""
    n, m = len(segs), len(sylls)
    dur = [b - a for a, b in segs]
    tau = sum(dur) / sum(sylls)
    pre = np.r_[0, np.cumsum(dur)]
    INF = 1e18
    best = [[INF] * (n + 1) for _ in range(m + 1)]; back = [[0] * (n + 1) for _ in range(m + 1)]
    best[0][0] = 0.0
    for t in range(1, m + 1):
        for j in range(t, n + 1):
            for i in range(t - 1, j):
                if best[t - 1][i] >= INF: continue
                cost = best[t - 1][i] + (pre[j] - pre[i] - sylls[t - 1] * tau) ** 2
                if cost < best[t][j]: best[t][j], back[t][j] = cost, i
    groups, j = [], n
    for t in range(m, 0, -1):
        i = back[t][j]; groups.append(list(range(i, j))); j = i
    return groups[::-1]

def main(src, scene, dst):
    w = pp.load(src)
    turns = [l.split(": ", 1) for l in open(scene).read().splitlines() if l.strip()]
    sylls = [sum(syl(x) for x in t[1].split()) for t in turns]
    segs = pp.segments(w, min_gap=0.15)
    groups = assign(segs, sylls)
    owner = {k: ti for ti, g in enumerate(groups) for k in g}
    out, edits = [], []
    for k, (a, b) in enumerate(segs):
        a2, b2 = max(0, a - PAD), min(len(w) / pp.SR, b + PAD)
        piece = w[int(a2 * pp.SR): int(b2 * pp.SR)].copy()
        if out:                                    # gap before this segment
            gap = a - segs[k - 1][1]
            same = owner[k] == owner[k - 1]
            new = min(gap, INTRA_MAX) if same else float(np.clip(gap, INTER_MIN, INTER_MAX))
            if abs(new - gap) > 0.02: edits.append((round(segs[k - 1][1], 1), "inside turn %d (%s)" % (owner[k] + 1, turns[owner[k]][0]) if same else "between turns", round(gap, 2), round(new, 2)))
            out.append(np.zeros(int(new * pp.SR)))
        out.append(piece)
    res = np.concatenate(out)
    sf.write(dst, res, pp.SR, subtype="PCM_16")
    print("turns", len(turns), "segments", len(segs), "grouping", [[k for k in g] for g in groups])
    for t, g in zip(turns, groups): print(f"  {t[0]:4s} segs {g[0]}-{g[-1]}  {t[1][:60]}")
    print("edits (at s, where, old gap -> new gap):"); [print("  ", e) for e in edits]
    print(f"duration {len(w)/pp.SR:.1f}s -> {len(res)/pp.SR:.1f}s")

if __name__ == "__main__":
    main(*sys.argv[1:4])
