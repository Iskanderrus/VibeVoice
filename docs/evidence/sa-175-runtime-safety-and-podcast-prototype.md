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

## 5. v3 run (in progress when this was written)

Frank RED / Carter BLUE, `cfg_scale` 1.6, the rewritten dialogue, up to 5 seeds per scene, checks: music share ≤ 0.15, floor ≤ −55 dB, pitch band. If no seed passes, the best take is used and flagged. Results will be appended to this file.

## 6. Source and runs on record

- Source for the runs: HEAD `d77b7ffe8b8be75a0fa2170c8647a9494cb68622` for the smoke, narration and v1 podcast runs. The v2 takes and the early v3 take ran with the runner change **uncommitted** (working-tree diff sha256 prefix `a08f0b6f08edb189`) on top of `d77b7ffe`; the commit `00ff115` carries exactly that change, so provenance files written after it show a newer HEAD.
- Output files are in `~/.local/share/stickman-vibevoice/output/` with provenance; per-run thermal logs in `…/runs/<job>/thermal.log`. Nothing was merged, and no audio was published or attached to the PR.

## 7. Not established

- No sustained-load thermal qualification; GPU margin 2 °C; boost-off is a manual non-persistent setting.
- Audio quality, speaker identity, accents, transcript fidelity and claim accuracy of the dialogue are unverified by tooling (no speech recogniser, no speaker-embedding model); judgements above come from the operator's listening.
- The RAM floor is provisional; peak memory was not recorded.

## 8. Follow-ups (not started)

- Ticket and owner-policy decision for a native two-speaker (and male/female, female/female) format in CFOS.
- Per-turn synthesis with exact pacing control; a speech-recognition check for transcript and alignment; a speaker-embedding check for voice identity.
- A British-accent reference clip for the British variant.
- Decide whether to require or verify boost-off in the runner.
