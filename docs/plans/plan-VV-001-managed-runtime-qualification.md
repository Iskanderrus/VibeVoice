# VV-001 — Managed VibeVoice Runtime Qualification

- **Local issue:** none; GitHub Issues are disabled in this fork
- **External executable tickets:** Stickman SA-175 / #336; CFOS-035 / content-factory-os#41
- **ADRs:** 0001, 0002
- **Current source baseline:** `Iskanderrus/VibeVoice` main currently contains the implemented isolated service; production always pins an exact commit rather than `main`
- **Current status:** service implemented; target-hardware/model smoke and end-to-end Stickman qualification remain
- **Portfolio timing:** CFOS `docs/implementation-order.md` only

## Goal

Produce an auditable qualification record proving that one exact owned-fork commit + exact `vibevoice/VibeVoice-1.5B` revision can safely serve the governed multi-speaker TTS capability on the target hardware and can be consumed by Stickman without dependency/runtime ambiguity.

## Data/evidence to own

The fork owns a qualification evidence package or report containing:

- source repository and exact commit SHA;
- service-contract/schema version;
- exact model repository/revision and local manifest digest;
- Docker image/build identity sufficient to prove the source revision baked into the image;
- target hardware/device/offload configuration;
- speaker/reference file SHA-256 values used by the smoke;
- input/dialogue hash and bounded generation controls/seed;
- load/synthesis/unload timestamps and durations;
- observed generation RTF and available RAM/VRAM observations;
- output WAV metadata and SHA-256;
- lifecycle/failure/cancellation result;
- resource-reclamation observation;
- service-contract test revision/result;
- qualification decision/status and reviewer/operator evidence;
- separate rights/disclosure/publication-eligibility references or explicit unknown state.

No model weights, private reference audio or generated production audio are committed to Git.

## Phase 1 — freeze runtime tuple

1. Select the exact fork commit to qualify.
2. Select/download the exact `vibevoice/VibeVoice-1.5B` model revision.
3. Generate/validate `.stickman-model.json` beside local weights.
4. Record expected source/model revisions in environment/build configuration.
5. Verify upstream/community repository is not used as a runtime/build dependency.

Exit: exact source+model tuple is reproducibly identifiable before model load.

## Phase 2 — model-free contract qualification

Run the existing fork service contract/compile tests for the selected commit and verify:

- owned-source enforcement;
- exact source-revision enforcement;
- 1.5B-only model policy;
- local model-manifest validation logic;
- request/speaker/turn/reference-hash validation;
- lifecycle and single-job serialization;
- cancellation/error mapping;
- deterministic result metadata and output-hash contract.

Exit: code/service contract is qualified independently of heavyweight model execution.

## Phase 3 — target-hardware load/synthesis/unload smoke

**Scope boundary:** This is an internal VibeVoice model/hardware test, not an approval of TSoBW cast voices for publication. Pending RED/BLUE production voice rights/identity evidence must not block load or smoke synthesis and must not trigger a rights-origin questionnaire. Use local operator-authorized test references without deliberate impersonation of non-consenting identifiable individuals. Hash paths/files as an implementation step, keep outputs private, and do not alter the production cast profile. Local Agent owns routine worktree/build/device/FP16 and test fixes autonomously. Production distribution remains a separate decision.


On the target machine using the approved constrained-GPU/offload profile:

1. start the isolated service;
2. verify `/health`, `/ready`, `/capabilities` identities;
3. load the exact model;
4. record load duration and observed resource state;
5. synthesize a representative 30–60 second **private technical** two-speaker exchange using available local test WAV references; compute their SHA-256 automatically and do not require channel cast/publication approval;
6. verify valid non-empty 24 kHz WAV and output hash;
7. record synthesis wall time/RTF and observed resource metrics;
8. unload;
9. verify resources are reclaimable sufficiently for the next local model stage;
10. exercise at least one cancellation/failure cleanup path if safe/practical.

Exit: one real exact tuple proves the required runtime lifecycle on target hardware.

## Phase 4 — publish qualification result to consumers

Record a compact versioned qualification result consumed by CFOS-035/SA-175:

- `qualified` — selectable for the governed capability under stated limits;
- `preview_only` — technically available but not yet approved for normal production/publication;
- `hold` — more evidence/work required;
- `rejected` — tuple is not selectable.

Include limitations such as language, max speaker count, concurrency, device/offload mode and timeout envelope. Treat rights/publication eligibility as an independent policy dimension.

## Phase 5 — Stickman consumer proof

With SA-175:

- capability discovery matches the qualified profile;
- source/model mismatch fails before synthesis;
- exact approved turns/references reach the service unchanged;
- Stickman performs transcript/alignment/mastering/QC after generation;
- failure paths unload/release resources;
- SA-189 execution result records the exact fork/model/service/device/reference/output identities.

## Regression and safety checks

- no moving `main` is accepted as a production source identity;
- upstream/community VibeVoice source is rejected for the governed capability;
- wrong model/7B manifest is rejected;
- service cannot silently download a different model when local-only mode is required;
- weights/reference/generated audio remain outside Git and image layers;
- fork changes do not weaken source/model/provenance/disclosure safeguards;
- no provider selection or fallback policy is implemented inside the fork.

## Definition of done

CFOS-035 has a complete qualification record for one exact fork+1.5B tuple and Stickman SA-175 can consume it end-to-end. A later fork/model revision requires a new qualification identity/evidence rather than mutating the historical result.


## Binding operator thermal/quality policy — 2026-11-16

**No additional hardware.** Use only the existing GTX 1650 Ti 4 GiB workstation and owned, exactly pinned VibeVoice-1.5B. Previous evidence: CPU FP32 21.33 s actual WAV / 428 s generation, about 24 GB RAM; the hybrid CPU FP32 language model + CUDA FP16 speech-components route produced 65.07 s actual 24 kHz WAV / 775 s generation and full Whisper match, operator rated its audio excellent. This confirms **function and sound quality**, **not thermal safety**: field CPU peaked 92°C and the later controlled hybrid run on this PR spiked to raw 94°C after smoothed resume, despite 1 Hz monitoring, 90°C hard stop and 15/15 focused test PASS (PR #8 comment 6641846361). No qualifying WAV was emitted from that newer run. GPU-only 4 GiB generation OOMs.

**Approved bounded implementation/qualification (replaces the older generic Phase 3 30–60 second mandate):**
1. Before any model-backed retry, inspect exact CPU family and available `k10temp` Tctl/Tdie labels and `temp*_max`/`temp*_crit`, actual cooling, governor/boost state and prior thermal logs. A reported Tctl value may differ from physical die temperature; neither 94°C=safe nor 94°C=damaging may be asserted without hardware information. Do **not** increase the existing 90°C software hard stop to turn a failing run green. If no trustworthy temperature measurement, stop instead of guessing.
2. Use reversible **process-scoped** CPU intensity restrictions (start `OMP_NUM_THREADS=2` and `MKL_NUM_THREADS=2`; reduce concurrency/affinity or process quota where necessary), restore on exit; no permanent privileged OS/firmware settings. Thermal control initial thresholds: CPU pause/resume **80°C / 70°C**, GPU **75°C / 65°C**, CPU critical stop **90°C provisional**. Monitor transient spikes closely: the existing 1 Hz median pause was insufficient. Cool before load, serialize jobs and avoid overlapping model-heavy processes; stop/cleanup on abrupt spike.
3. Make **one short real model-backed** guarded synthesis on the exact corrected PR code and pinned weights, rather than re-running 60 s only to satisfy a checklist. Record model manifest, device map/dtype, CPU/GPU temperature traces and maxima, actual load/generation time/RTF, memory usage, WAV SHA/metadata, Whisper match, speaker identity/listening quality and unload/reclamation. The hybrid route is preferred for auditory quality **only when this run proves a safe profile**. If not safe, keep it experimental and evaluate **guarded CPU FP32** with its own short measured qualification; the old CPU WAV had no sufficient thermal proof. No silent different-TTS fallback.
4. After a guarded safe route exists, for the first **approved SCRIPT** use serial synthesis chunks aligned to semantic/sentence boundaries, **initially 10–25 s where the measured temperature permits**, adjust shorter as needed, cool between jobs and never concatenate over missing/transposed audio. Preserve exact text/version, speaker/voice reference hashes, pauses/prosody, source/model/runtime profile, and each segment SHA; retry only a failed segment after rechecking approvals and safety, never resynthesize good segments on resume. Test final join/clicks/loudness, complete transcript and operator listening on genuine media.
5. Fail `BLOCKED_AUDIO_RUNTIME` on unsafe/unavailable/unknown hardware, missing sensors, model mismatch, drifted source, or incomplete WAV; never state that a quality-PASS sample equals thermal PASS. VibeVoice handles only synthesis; Stickman SA-175/SA-202 owns orchestration/mastering/QC; CFOS owns human AUDIO approval and independent stage authorization.

**Acceptance:** short guarded exact-source WAV with stable temperatures and no critical spikes, proper teardown, and no perceptible quality regression against the operator-accepted 65.07 s hybrid sample. Then qualify the next real approved-script segmented run without duplicate stress loops or new tickets. No model download, code change, live machine control or provider action is authorized merely by this documentation amendment.
