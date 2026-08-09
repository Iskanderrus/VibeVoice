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

On the target machine using the approved constrained-GPU/offload profile:

1. start the isolated service;
2. verify `/health`, `/ready`, `/capabilities` identities;
3. load the exact model;
4. record load duration and observed resource state;
5. synthesize a representative 30–60 second two-speaker RED/BLUE exchange with approved references;
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
