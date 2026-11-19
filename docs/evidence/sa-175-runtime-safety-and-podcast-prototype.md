# SA-175 / CFOS-035 — runtime safety evidence and the two-host podcast prototype

**Status:** evidence record, not a qualification. Thermal and audio-quality qualification are **not** granted; Master decides on this evidence.
**Scope:** one laptop (Acer Aspire A715-41G: Ryzen 7 3750H, GTX 1650 Ti 4 GiB, 29 GiB RAM), the pinned VibeVoice-1.5B revision `d374386b2a51d8e05277a64d85b296c89ec52376`, hybrid route (`cuda_tokenizers_cpu_language_model`), pinned venv (torch 2.5.1+cu121, numpy 2.1.2, diffusers 0.33.1, accelerate 1.6.0, transformers 4.51.3).
**Dates:** 2026-11-19. Thresholds were never raised: CPU pause/resume 80/70 °C, GPU 75/65 °C, raw critical 90 °C.

All audio is private test material. No episode was published, nothing here is attested, and the underlying source is a copyrighted book (use beyond private testing is not cleared). The scripts in [`sa-175/scripts/`](sa-175/scripts/) are evidence of what was run, not product code: they contain session-specific absolute paths and are not exercised by CI.

## 1. Safety gaps closed in source (commits `94f1bce`..`d77b7ffe`, then `00ff115`)

Covered by model-free tests (64 passed across the thermal, quota and runner test files at `d77b7ffe` in the pinned venv; after `00ff115` the runner test file passes with 12 tests, two of them new; the other two files were not re-run while a take was in flight):
1. **Raw versus median:** pause on any raw reading at or above the limit, never only on the smoothed median.
2. **Idle counters:** preflight idle is computed from per-sample `/proc/stat` counters over the same window; stale or missing counters fail closed.
3. **Cleanup proof:** `verify_cleanup` succeeds only when the scope is gone or killed and emptied; an unknown state or surviving worker is a failure. Guard death is detected.
4. **Unqualified CPU/CUDA route:** blocked in the runner; hybrid is the only qualified device.
5. **Memory admission:** `MemAvailable` floor, `THERMAL_MIN_AVAILABLE_MEM_MIB=12288` — **provisional**, not a measured footprint; peak memory was not recorded.
6. **Reversible limits:** process-scoped `systemd-run --user --scope` with `CPUQuota`, `taskset` affinity and OMP/MKL thread budget, proven from inside the scope (`LIMITS_APPLIED`) before the model loads.

`00ff115` adds optional, validated runner controls (defaults unchanged): `--voice RED=<wav> --voice BLUE=<wav>` (complete pair required), `--seed` (0..2^31-1) and `--cfg-scale` (1.0..3.0); the dry-run and run state record them.

## 2. Thermal findings

| Setup | Result |
|---|---|
| Boost on, 150% quota, cores 0,2, 2 threads — 7.07 s smoke | completed; 11 pauses; CPU raw max 88 °C; GPU max 65 °C; generation 216 s |
| Same, full 233-word narration, first attempt | **stopped by `CRITICAL_THERMAL_STOP`** at raw 90 °C after 7 pauses; no audio; scope and workers cleaned up |
| Quota probe, **boost on** (model-free fp32 matmul burner) | even a 50% quota reached raw 85 °C within 11 s and self-stopped |
| Same probe, **boost off** | quota 50/75/100/125%: CPU max 53/54/54/56 °C over 120 s each, no reading at or above 75 °C |
| Thread probe, boost off (model-free) | 2/4/6/8 threads: CPU max 58–61 °C, GPU max 53–55 °C, burner throughput 1.0×/1.2×/1.7×/2.1× |
| **Real model**, scene 3, 6 threads (cores 0–5, 450%) vs 2 threads | **292.9 s vs 174.4 s**: more threads were slower; the model-free probe did not predict this |

Conclusions:
1. A CPU quota limits time on the CPU, not clock speed. With boost on, the Ryzen reaches the pause line within seconds even at a 50% quota. **CPU boost off** (`/sys/devices/system/cpu/cpufreq/boost` = 0) removed the pauses. It is a manual, root-owned, non-persistent operator setting that the code neither sets nor verifies; whether to require it is Master's decision.
2. Best measured real-model setup: boost off, **2 threads on cores 0,2, 150% quota**. Generation is roughly 10–11 s per second of audio. Six threads are slower for this model.
3. The GPU is the tightest limit: 70–73 °C against the 75 °C pause line in real runs (the model-free burner kept it at 53–55 °C and cannot test it).
4. Earlier failures: preflight blocks (fail-closed, as designed) when other development jobs (graphify, Dart, `design_audit`) kept the CPU at 75–90 °C; an unpinned venv (torch 2.13 / diffusers 0.39) produced "model returned no speech output" — the pinned venv resolved it, root cause not proven.
5. Fan control is not available from Linux on this model (no `pwm`/`fan` hwmon entries, no `platform_profile`).

Sample size is a handful of short runs on one machine: **no sustained-load qualification and no segment-length recommendation.**

## 3. Narration test on the CFOS-068 draft (233 words, four scenes, `DRAFT_UNATTESTED`)

- Single pass, boost off: 94.4 s audio, generation 1027 s, **0 pauses**, CPU max 72 °C, GPU max 73 °C.
- Rough speaking-rate proxy (syllable-like energy peaks per second of speech, 10 s windows): 3.4–4.6/s, a 29% window range; pauses 0.25–1.63 s. Per-scene generation plus tempo normalisation (within ±10%) and pause clamping (0.35–0.8 s) narrowed the range to 19%. The proxy undercounts syllables (236 peaks against about 340) and is only for comparing windows. No speech recogniser was available, so word-level pace and the transcript are unverified.

## 4. Two-host (RED/BLUE) podcast prototype — not a CFOS workflow run

CFOS supports only `single_narrator` today (the Stickman script writer states "No cast dialogue; current approved mode is single_narrator"; the Gate 1 producer has no speaker-turn support), so this is a prototype outside the CFOS stages. A native two-speaker format needs a youtube-channels owner policy plus producer and contract support, and is a separate ticket that is **not** built here.

Pipeline (each step scripted, no hand-edited text):
1. **Dialogue rewrite** (`dialogue_rewrite.py`): the CFOS Claude Code subscription adapter (`ClaudeCodeSubscriptionCreativeTextGenerator`, `claude-sonnet-5-5`, CLI 2.1.326, plan pro, billed cost 0; API key scrubbed from the child environment, tools/MCP off). Only the four scene narrations and the system prompt were sent off the device. Deterministic checks: speakers, 4–9 turns per scene, turns ≤ 45 words, no markup, no number absent from the source scene, length 0.9–1.35× the source, at most two calls.
2. **Per-scene synthesis**: one guarded `run_stickman_local.py --foreground` run per scene (same limits, boost off).
3. **Assembly**: per-scene tempo normalisation, pause clamping, scene gaps, peak limit, and (v3) loudness normalisation to −16 LUFS / −1.5 dBTP.

### Observations from the operator's listening
| Take | Operator feedback | Our metrics |
|---|---|---|
| v1 (RED = Carter, BLUE = Frank, seed 18427, `cfg_scale` 1.3, original dialogue) | unexpected music; voices changing, including American/British accents | both references are low male voices (median pitch 130/116 Hz), only 14 Hz apart; scene 1 quiet floor −41.7 dB |
| v2 scene 1 (Frank RED / Carter BLUE, `cfg_scale` 1.6, seed 18427) | voices stable, **music drop-ins remain** | music share 0.37; the old floor check (≤ −55 dB) passed it at −67.6 dB |
| v2 scene 2, seed 18428 | "quite good… could be louder" | music share 0.00, floor −81.8 dB, mean level −24.7 dB |
| v2 scene 2, seed 18427 | rejected by the automatic checks | music share 0.92, floor −39.6 dB, pitch p90 235 Hz |

Findings:
- The default RED/BLUE references were the bundled `en-Carter_man.wav` and `en-Frank_man.wav` (byte-identical). Two similar male voices give the model little to hold apart; accents drift with it. The operator chose RED = Frank, BLUE = Carter for the male/male format of the channel, and wants male/female and female/female pairs plus an American and a British variant. The bundled voices carry no accent metadata and accents cannot be measured here: a British variant needs a British reference clip from the operator.
- README FAQ Q2 states that background music is spontaneous, cannot be controlled, and is triggered by (1) music in the voice prompt, (2) intro-like words such as "Welcome to", "Hello", "However", (3) certain voices (Alice), and is less likely with the larger model (7B, not usable on this 4 GiB GPU). Q6 recommends repeated sampling. The 1.5B model does not offer a switch. Each scene is a cold start, which resembles a show intro; this fits trigger 2 but is a hypothesis.
- **Detector** (`pod_v3.py`): the share of 2 s windows whose quietest 5% of 20 ms frames stays above −48 dBFS; pass at ≤ 0.15. Calibration: clean bundled voices 0.04–0.19, the bundled music voice 1.00, v1 scene 1 0.79, v2 scene 1 0.37, v2 scene 2 seed 18427 0.92, the take rated good 0.00. A first sustained-spectrum detector (`rejected_sustained_spectrum_detector.py`) failed its positive control (no detection in the bundled music voice) and was discarded. Limits: tuned on a few files (one clean reference scores 0.19), and music under continuous speech that never fills a pause is not detectable this way.
- **Text rules** (`dialogue_rewrite.py`): documented triggers (welcome, hello, hi, hey, however) are banned anywhere; intro-like turn openers (So, Well, Okay, Now, Right, Alright, And, But) are banned as an extension that is a project hypothesis, not a documented trigger. The rewritten dialogue passed on its first attempt (283 words). The earlier dialogue had three "So …" scene openers.
- **Voice drift check:** per-take pitch median in 95–155 Hz and p90 ≤ 215 Hz. The two references differ by only 14 Hz, so this band guards gross drift (for example towards a female voice), not speaker identity; no speaker-embedding tool was available.
- **Loudness:** takes sit at about −25 dB mean volume; v3 normalises the assembled file to −16 LUFS.

## 5. Controlled comparisons, the break-in-a-line finding, and the selected takes

The v3 automatic-selection run was abandoned after the operator's ratings contradicted the checks: with thresholds tightened to floor ≤ −70 dB, music share ≤ 0.03, pitch median 95–135 Hz and p90 ≤ 190 Hz (from ≤ −55 / ≤ 0.15 / 95–155 / ≤ 215), the metrics still did not predict what the operator liked. The work moved to controlled comparisons of the **same scene** (scene 2 of the original dialogue, RED = Frank, BLUE = Carter, boost off, same guarded limits), changing one setting at a time, with loudness-matched copies for fair listening.

**Reproducibility.** The same seed, text, voices and settings reproduce a take **byte-for-byte** (two v3 scene-1 jobs with seed 18427 have the identical SHA-256). A good seed is therefore reusable, and seed is the main source of variation.

**Guidance × seed grid** (scene 2, original text; `scripts/exp_scene2.py`; nothing rejected):

| Guidance | Seed | Floor | Music share | Pitch median / p90 | Note |
|---|---|---|---|---|---|
| 1.3 | 18427 | −94.2 dB | 0.00 | 128 / 172 Hz | clean |
| 1.3 | 18428 | −80.7 dB | 0.00 | 157 / 210 Hz | pitch drifted high |
| 1.3 | 18429 | −50.7 dB | 0.20 | 128 / 209 Hz | music at 16–19 s |
| 1.6 | 18427 | −39.6 dB | 0.92 | 149 / 235 Hz | worst take |
| 1.6 | 18428 | −81.8 dB | 0.00 | 125 / 175 Hz | operator: best |
| 1.6 | 18429 | −73.4 dB | 0.08 | 130 / 200 Hz | mild music at 10, 16 s |
| 2.0 | 18427 | −56.4 dB | 0.21 | 135 / 233 Hz | music at 2–4 s, 13 s |
| 2.0 | 18428 | −84.5 dB | 0.00 | 136 / 179 Hz | operator: good |
| 2.0 | 18429 | −74.3 dB | 0.00 | 142 / 211 Hz | clean, higher pitch |

Guidance does not reliably help (clean or nearly clean: 2 of 3 at 1.3, 1 of 3 at 1.6, 2 of 3 at 2.0); seed dominates. Pitch medians of the takes ranged 125–157 Hz against references at 116 and 130 Hz.

**Lower, younger voices.** The model has no pitch control; the voice follows the reference clip, and takes drift upward from it. Two routes were tried, both with formants preserved (`ffmpeg` `rubberband=pitch=2^(−n/12):formant=preserved:pitchq=quality`):
1. **Post-shift of a finished take** (instant, both speakers move together): for the operator's chosen take, down 1.5 / 1.8 / 2 / 3 / 4 semitones gave pitch medians 127 / 124 / 123 / 115 / 110 Hz from 137 Hz. The operator found **1.5 semitones the best and most natural**.
2. **Lowered references** (the lowered clips are in `~/.local/share/stickman-vibevoice/refs-shifted/`, the originals untouched): scene 1 with the rewritten dialogue, guidance 1.6, both speakers' references lowered (`scripts/exp_pitch.py`). The unshifted take of the same scene and seed 18427 has a pitch median of 145 Hz (floor −57.6 dB, music share 0.12).

| References lowered | Seed | Floor | Music share | Pitch median / p90 |
|---|---|---|---|---|
| −2 semitones | 18427 | −83.7 dB | 0.04 | 107 / 141 Hz |
| −2 semitones | 18428 | −90.8 dB | 0.07 | 107 / 140 Hz |
| −3 semitones | 18427 | −62.1 dB | 0.12 | 114 / 188 Hz |
| −3 semitones | 18428 | −87.3 dB | 0.07 | 103 / 148 Hz |

The lowered references do move the generated pitch down by about 40 Hz and tend to give clean takes, but quality still varies by seed (the −3 / 18427 take is the weakest). **The operator still rated the post-shifted copies of the clean scene-2 take (1.5 and 1.8 semitones) as the best**, so model-generated lowered references are not adopted. A further test of lowered references on the clean scene-2 take (1.5/1.8/2 semitones, `exp_pitch2.py`) was queued and **cancelled before it started** because that decision made it moot; it has no results.
"Younger" is not controlled by pitch: age lives in the reference's timbre, so a young low voice needs a suitable reference clip (own, contracted or licensed voice, with consent).

**Breaks inside a speaker's line.** The operator heard a pause **and a tone movement** between "person" and "turns" in the second speaker's first line of the guidance-2.0 take. Measured: a 0.46 s gap at 4.3 s with a pitch fall of about 40% in the last 120 ms before it, i.e. the model gives the subject a clause-final fall and restarts. Cutting the gap (`scripts/tighten.py`: segments assigned to the known turns by syllable-count dynamic programming, then intra-turn gaps capped and inter-turn gaps clamped to 0.35–0.65 s) shortens the pause but cannot repair the intonation; the variants made from it are not the selected takes. Across the nine takes, **7 of the 8 measurable takes have a gap ≥ 0.15 s inside a second-speaker line**; the only one without is guidance 1.6, seed 18428 (one take had too few segments to measure). The line-to-audio matching is an estimate (no speech recogniser) and the pitch tracker is rough.

**Selected takes (operator: both "excellent")**, in `~/.local/share/stickman-vibevoice/output/`:

| File | SHA-256 | Duration | Loudness |
|---|---|---|---|
| `vv-pod2-scene2-s18428-down1.5st-formant-kept-ln16.wav` | `4a4d1d80d79965348cf70a5dd614769faef90a29de5eea53fb2b7054db8cb22d` | 27.2 s | −17.2 LUFS, −1.5 dBTP |
| `vv-pod2-scene2-s18428-down1.8st-formant-kept-ln16.wav` | `8c43cc97708c7834f89b8cbaf8dc3932d038c5255faa6872af1b88c9b30b60d2` | 27.2 s | −17.4 LUFS, −1.5 dBTP |

Recipe, both files: source take `vv-pod2-scene2-s18428-20261119-202821.wav` (SHA-256 `9aa5283cce15e88e7cc6e1f497ed302612a0f8b6a5bb0bd118480361295bd636`): scene 2 of the original dialogue (`scene2.txt` sha256 prefix `191b36f32bee851d`, 7 turns), RED = `demo/voices/en-Frank_man.wav`, BLUE = `demo/voices/en-Carter_man.wav`, seed 18428, `cfg_scale` 1.6, boost off, 2 threads on cores 0,2, 150% quota, hybrid route, runner with the uncommitted `--voice/--seed/--cfg-scale` change that became `00ff115`. Then `rubberband` pitch ratio 0.91700 (1.5 semitones) or 0.90125 (1.8 semitones) with formants preserved, then single-pass `loudnorm=I=-16:TP=-1.5:LRA=11` (measured result above). The source take has a measured pitch median of 125 Hz and the shifted files about 115 Hz by the rough tracker; the source take has no gap inside any second-speaker line. Loudness is about 1 LUFS under the −16 target because single-pass `loudnorm` was used for these copies.

## 5a. Batch for scenes 1, 3, 4, selection by ear, pitch method and per-seed episodes

**Batch** (`scripts/batch_scenes.py`): 15 takes, scenes 1, 3 and 4 of the original dialogue, seeds 18427–18431, Frank = RED, Carter = BLUE, guidance 1.6, boost off, same guarded limits, no pauses or critical stops. Nothing was rejected; each take was measured and given listening copies. "Gaps" counts gaps ≥ 0.15 s inside a line, by speaker.

| Scene | Seed | Floor | Music share | Pitch median / p90 | Gaps (BLUE / RED) |
|---|---|---|---|---|---|
| 1 | 18427 | −67.6 dB | 0.37 | 128 / 166 Hz | 5 / 0 |
| 1 | 18428 | −85.4 dB | 0.08 | 137 / 190 Hz | 4 / 3 |
| 1 | 18429 | −77.7 dB | 0.04 | 163 / 229 Hz | 4 / 1 |
| 1 | 18430 | −55.2 dB | 0.35 | 149 / 214 Hz | 2 / 0 |
| 1 | 18431 | −44.7 dB | 0.65 | 122 / 169 Hz | 1 / 0 |
| 3 | 18427 | −93.9 dB | 0.00 | 121 / 150 Hz | 7 / 1 |
| 3 | 18428 | −77.2 dB | 0.08 | 132 / 175 Hz | 2 / 1 |
| 3 | 18429 | −79.2 dB | 0.00 | 131 / 207 Hz | 4 / 1 |
| 3 | 18430 | −70.2 dB | 0.07 | 122 / 210 Hz | 8 / 1 |
| 3 | 18431 | −93.8 dB | 0.12 | 116 / 175 Hz | 4 / 1 |
| 4 | 18427 | −83.7 dB | 0.00 | 134 / 175 Hz | 0 / 0 |
| 4 | 18428 | −94.3 dB | 0.00 | 129 / 167 Hz | 4 / 0 |
| 4 | 18429 | −71.2 dB | 0.00 | 150 / 190 Hz | 2 / 0 |
| 4 | 18430 | −82.4 dB | 0.06 | 126 / 182 Hz | 2 / 0 |
| 4 | 18431 | −70.0 dB | 0.06 | 136 / 168 Hz | 1 / 2 |

Observations:
- The scene-1 seed 18427 take is identical to the earlier v2 scene-1 take (same floor, music share and pitch): same seed, same audio.
- Music appears around 4–6 s in all five scene-1 takes, at the point where the second speaker's reply begins, so it looks tied to the text at that position (or to the opening position of the episode), not to the seed. The operator considers music at the start of an episode possibly wanted.
- Only scene 4 / seed 18427 has no gap inside any line.
- **Operator selection by ear** (the numbers do not predict it): scene 1 = seed 18430 (floor −55.2 dB, music share 0.35, i.e. poor on the metrics), scene 2 = seed 18428, scene 3 = seed 18429 (four gaps in second-speaker lines, rated excellent), scene 4 = seed 18428 (four gaps, rated very good). The metrics above order what is worth hearing first; they do not decide.

**Pitch-lowering method.** `rubberband` with formants preserved was "metallic" to the operator; on scene 3 / seed 18429 the 1.5 and 1.8 semitone versions were "terrible" and 1.2 semitones "my absolute favourite" (the same shifts had sounded excellent on scene 2, so the tolerable shift depends on the take). Four methods were compared at 1.2 semitones on that take: formant-preserved (default window), formant-preserved with smooth transients and a long window, formants moving with the pitch, and **method C: resample then restore the speed** (`asetrate=24000·r, aresample=24000, atempo=1/r`, `r = 2^(−semitones/12)`, then `loudnorm=I=-16:TP=-1.5:LRA=11`). **Method C was rated "absolute best"**, and scene 2 / seed 18428 with method C at 1.8 semitones "WOW". Method C also moves the formants down slightly; why it avoids the metallic quality was not investigated (the phase-vocoder step is the suspected source of the artifact in the other methods, unmeasured).
Per-scene shifts chosen with method C: scene 1 seed 18430 at 1.5, scene 2 seed 18428 at 1.8, scene 3 seed 18429 at 1.2, scene 4 seed 18428 at 1.2 (the choices for scenes 1 and 4 were made on other methods and not yet confirmed with method C).

**Episodes, one seed each** (`scripts/assemble_episode.py <seed> <semitones>`): all four scenes of one seed, method C, scene files as the operator heard them (no tempo or pause editing), empty ends trimmed, 15 ms fades, 0.7 s between scenes, final two-pass `loudnorm` to −16 LUFS / −1.5 dBTP. Seed 18430 lacked scene 2 (the earlier scene-2 grid covered seeds 18427–18429), so it was generated separately (guidance 1.6, same voices).

| File (in `~/.local/share/stickman-vibevoice/output/`, `vv-blackswan-podcast-episode-…-methodC.wav`) | Length | SHA-256 |
|---|---|---|
| `seed18428-down1.2st` | 101.9 s | `27b574eb9a34451aa3d9d21dbf7ee5fc0fcac82b67d7eb7f67d6e3753d7daa39` |
| `seed18428-down1.8st` | 101.9 s | `f4e87d9970cb06d5207c2d4823be9f22cab5674b8975b2bebf027793b59755e5` |
| `seed18429-down1.2st` | 102.8 s | `d6519693d89e017d67904fa37fcacb2901752e71c2a544dbbe7833730a0ca6f7` |
| `seed18429-down1.5st` | 102.8 s | `8bd8e627034e0154af2307ad76e0aba119506550f9e872c5f160f866b1bb7007` |
| `seed18429-down1.8st` | 102.8 s | `77bec504c3cc926c95e4abe8b009d9082647d118e2248d8db82226b7e466be22` |
| `seed18430-down1.5st` | 107.6 s | `62f27c08a26afe9fae3b947e91434410a1566618224cea3547372cff4a0cdc58` |

An earlier episode mixing the four per-scene picks from different seeds (`…episode-selected-methodC.wav`, sha256 `d5d1925041548748c23b5e01ced1522cd672d9e5da18368c971e6a291440a502`, 106.8 s) was **not what the operator wanted** and is superseded. **The operator's only candidates are seed 18429 at 1.8 and seed 18428 at 1.8.**

**Whole-piece comparison** (`scripts/whole_piece.py`): the two candidate seeds generated as ONE take each (all 25 turns) instead of per scene, with the same recipe, to compare against the per-scene episodes. Seed 18428 started at 23:53:10 (expected about 18–30 minutes, no pauses at the time of writing). The operator asked to pause after it and continue the next morning, so the orchestrator was stopped and **seed 18429 has not been started**. Quality results and the method-C 1.8-semitone versions are pending and will be appended.

## 6. Source and runs on record

- Source for the batch, the seed-18430 scene-2 take and the whole-piece take: the committed runner (`00ff115`) at the then-current branch HEAD; provenance files record that HEAD. Source for the earlier runs: HEAD `d77b7ffe8b8be75a0fa2170c8647a9494cb68622` for the smoke, narration and v1 podcast runs. The v2 takes and the early v3 take ran with the runner change **uncommitted** (working-tree diff sha256 prefix `a08f0b6f08edb189`) on top of `d77b7ffe`; the commit `00ff115` carries exactly that change, so provenance files written after it show a newer HEAD.
- Output files are in `~/.local/share/stickman-vibevoice/output/` with provenance; per-run thermal logs in `…/runs/<job>/thermal.log`. Nothing was merged, and no audio was published or attached to the PR.

## 7. Not established

- No sustained-load thermal qualification; GPU margin 2 °C; boost-off is a manual non-persistent setting.
- Audio quality, speaker identity, accents, transcript fidelity and claim accuracy of the dialogue are unverified by tooling (no speech recogniser, no speaker-embedding model); judgements above come from the operator's listening. The automatic checks (music share, floor, pitch band, break detector) do **not** predict the operator's ratings reliably and are heuristics tuned on a handful of takes.
- All four scenes now have operator-selected takes and six per-seed episodes exist, but none has been checked by tooling for joins or voice consistency across scenes (scene 1 sits about 15–25 Hz higher in pitch than the others); the whole-piece comparison against the per-scene approach is not complete.
- Music is not treated as a defect by the operator: it can be wanted at the start or on key ideas. The model cannot produce it on request (README FAQ Q2); a licensed music bed mixed in post-production is the controlled alternative and no such file is available.
- The RAM floor is provisional; peak memory was not recorded.

## 8. Follow-ups (not started)

- Ticket and owner-policy decision for a native two-speaker (and male/female, female/female) format in CFOS.
- Per-turn synthesis with exact pacing control; a speech-recognition check for transcript and alignment; a speaker-embedding check for voice identity.
- A British-accent reference clip for the British variant.
- Decide whether to require or verify boost-off in the runner.
- Add the in-line break test (gap ≥ 0.2 s with a pitch fall ≥ 15% before it) to take selection, and a dialogue rule against long subjects before the verb.
- Finish the whole-piece comparison (seed 18429 not started) and decide per-scene versus whole-piece generation.
- Confirm scenes 1 and 4 with method C by listening; check the pitch of scene 1 against the other scenes.
