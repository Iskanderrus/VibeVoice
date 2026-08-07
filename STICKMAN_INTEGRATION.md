# Stickman Automation integration contract

- **Status:** Service implemented; target-hardware/model smoke pending
- **Consumer issue:** [Iskanderrus/stickman-automation#336](https://github.com/Iskanderrus/stickman-automation/issues/336)
- **Channel policy:** [Iskanderrus/youtube-channels#13](https://github.com/Iskanderrus/youtube-channels/issues/13)
- **Runtime source:** `Iskanderrus/VibeVoice` at an exact commit
- **Target model:** `vibevoice/VibeVoice-1.5B` at an exact revision
- **Normal route:** local/offline RED/BLUE multi-speaker dialogue; 7B is excluded

The fork now contains the complete isolated HTTP service described by the original integration plan. Operational details are maintained in [`STICKMAN_SERVICE.md`](./STICKMAN_SERVICE.md).

## Authority boundary

This repository owns:

- the isolated VibeVoice Python/model runtime;
- model load/unload and one-job serialization;
- the private HTTP contract;
- source/model identity enforcement;
- reference path/hash checks;
- deterministic request metadata and WAV output lineage;
- Docker packaging, resource cleanup and service contract tests.

Stickman Automation owns:

- provider selection and orchestration;
- GPU lease/other-local-model coordination;
- narration/performance chunk policy;
- Whisper alignment, mastering and content/audio QC;
- fallback policy;
- rights, disclosure, human approval and publication.

YouTube Channels owns persistent RED/BLUE character voice identity and allowed dialogue policy.

## Non-negotiable source policy

1. Production runtime source is **only** `Iskanderrus/VibeVoice`.
2. A production image uses an exact 40-character commit SHA; no moving `main`.
3. `vibevoice-community/VibeVoice` is upstream provenance only and is never a runtime/build dependency.
4. Automatic upstream synchronization is prohibited.
5. Only `vibevoice/VibeVoice-1.5B` is accepted by this service; 7B is rejected by static policy.
6. Model weights, generated audio and private/reference voices stay outside Git and image layers.
7. Safety/provenance/disclosure mechanisms are not deliberately removed.

## Implemented service contract

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Process liveness |
| `GET` | `/ready` | Lifecycle/readiness/failure state |
| `GET` | `/capabilities` | Pinned source/model identity and capabilities |
| `POST` | `/load` | Load the pinned 1.5B model |
| `POST` | `/synthesize-dialogue` | Generate one ordered multi-speaker performance |
| `POST` | `/cancel/{job_id}` | Cooperative cancellation |
| `POST` | `/unload` | Release model/runtime resources |

Lifecycle:

```text
unloaded -> loading -> ready -> generating -> ready
                         |
                         +-> unloading -> unloaded

runtime/model failure -> failed
```

Only one heavyweight operation may execute at once.

## Reproducibility and local-weight identity

The service requires:

- `VIBEVOICE_SOURCE_REVISION=<exact fork commit>`;
- `VIBEVOICE_MODEL_REVISION=<exact model revision>`;
- the image-baked source revision to match runtime configuration;
- a `.stickman-model.json` file beside the local weights;
- that manifest to identify exactly `vibevoice/VibeVoice-1.5B` and the configured revision.

The default is `VIBEVOICE_LOCAL_FILES_ONLY=true`; the service therefore fails closed rather than downloading a moving or different model.

## Request/output guarantees

A dialogue request:

- contains 1–4 explicit speaker bindings;
- contains ordered approved turns;
- is English-only for the initial governed route;
- supplies expected SHA-256 for every WAV reference;
- may request only bounded supported generation controls;
- is rejected if source/model identity differs from the running service.

The service:

- maps speaker bindings deterministically to native `Speaker 1`, `Speaker 2`, etc.;
- does not rewrite or extend approved dialogue;
- writes to a unique temporary WAV and atomically promotes it;
- refuses silent overwrite;
- requires a valid non-empty 24 kHz WAV;
- returns source/model revision, device mode, seed, generation controls, reference hashes, output hash and timing.

The result is **not** publication approval. Stickman performs the downstream QA and governance gates.

## Target hardware behavior

The intended laptop route is constrained-GPU execution with CPU/system-RAM offload:

```text
VIBEVOICE_CPU_OFFLOAD=true
VIBEVOICE_CUDA_MAX_MEMORY_MB=3400
VIBEVOICE_CPU_MAX_MEMORY_GB=24
VIBEVOICE_MAX_CONCURRENT_JOBS=1
```

These are conservative starting limits, not benchmark claims.

## Validation status

Implemented model-free checks cover:

- owned-source enforcement;
- 1.5B-only policy;
- source revision enforcement;
- model-manifest revision enforcement;
- schema and speaker-turn validation;
- reference path/symlink and SHA-256 enforcement;
- lifecycle/load/generate/unload;
- single-operation serialization and cancellation;
- deterministic output/hash metadata;
- stable HTTP validation errors.

The remaining acceptance step requires the user's pinned 1.5B weights and target laptop:

1. build the service image from an exact fork commit;
2. load the model using the approved offload mode;
3. synthesize a 30–60 second two-speaker exchange;
4. record load time, generation RTF and peak observed RAM/VRAM;
5. unload and confirm resources are reclaimable by the next local model stage.

That hardware smoke is intentionally separate from the Stickman Automation implementation slot.
