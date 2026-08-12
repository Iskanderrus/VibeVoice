# Stickman Automation integration contract

- **Status:** Service code/contract implemented and hardened; target-hardware/model smoke pending
- **Consumer issue:** [Iskanderrus/stickman-automation#336](https://github.com/Iskanderrus/stickman-automation/issues/336)
- **Channel policy:** [Iskanderrus/youtube-channels#13](https://github.com/Iskanderrus/youtube-channels/issues/13)
- **Runtime source:** `Iskanderrus/VibeVoice` at an exact commit
- **Target model:** `vibevoice/VibeVoice-1.5B` at an exact 40-character model commit
- **Normal route:** local/offline RED/BLUE multi-speaker dialogue; 7B is excluded

Operational details are maintained in [`STICKMAN_SERVICE.md`](./STICKMAN_SERVICE.md).

## Authority boundary

This repository owns:

- the isolated VibeVoice Python/model runtime;
- model load/unload and one-job serialization;
- the private HTTP contract;
- source/model identity and model-artifact integrity enforcement;
- reference path/hash/staging checks;
- deterministic request metadata and WAV output lineage;
- Docker packaging, cleanup and service contract tests.

Stickman Automation owns provider selection/orchestration, GPU lease coordination, performance-chunk policy, Whisper alignment, mastering/QC, fallback, rights/disclosure, human approval and publication.

YouTube Channels owns persistent RED/BLUE character voice identity and allowed dialogue policy.

## Non-negotiable source and model policy

1. Production runtime source is **only** `Iskanderrus/VibeVoice`.
2. Governed builds run through `scripts/build_stickman_service.sh` from a clean checkout whose `origin` is our fork and whose `HEAD` equals fetched `origin/main`.
3. The resulting image bakes both the exact source commit and exact model commit; Compose does not override either identity when the container starts.
4. `vibevoice-community/VibeVoice` is upstream provenance only and is never a runtime/build dependency.
5. Automatic upstream synchronization is prohibited.
6. Only `vibevoice/VibeVoice-1.5B` is accepted; 7B is rejected by static policy.
7. Source and model revisions must be exact 40-character commit SHAs.
8. `VIBEVOICE_LOCAL_FILES_ONLY=true` and `VIBEVOICE_REQUIRE_MODEL_MANIFEST=true` are mandatory and cannot be disabled through configuration.
9. The externally mounted model manifest and model files must match the image-baked model commit.
10. Model weights, generated audio and private/reference voices stay outside Git and image layers.
11. Safety/provenance/disclosure mechanisms are not deliberately removed.

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

The private service exposes no Gradio UI, FastAPI docs UI or OpenAPI endpoint.

Lifecycle:

```text
unloaded -> loading -> ready -> generating -> ready
                         |          |
                         |          +-- cancel/timeout -> unload -> unloaded
                         +-> unloading -> unloaded

runtime/model/cleanup failure -> failed
```

Only one heavyweight operation may execute at once. Cancellation and timeout are enforced inside the manager as well as at the HTTP boundary; interrupted inference is not reused without a fresh load.

## Reproducibility and local-weight identity

The governed image bakes:

- `VIBEVOICE_SOURCE_REVISION=<exact owned-fork commit>`;
- `VIBEVOICE_MODEL_REVISION=<exact 40-character model commit>`;
- the approved source repository and model repository identities.

At runtime the service additionally requires:

- `.stickman-model.json` inside the mounted model directory;
- that manifest to identify exactly `vibevoice/VibeVoice-1.5B` and the same model commit baked into the image;
- every inventoried local model artifact to carry its path, byte size and SHA-256;
- `config.json` and at least one model weight file to be present.

Before importing Torch or allocating model memory, model artifact paths are resolved under the configured model root and every recorded artifact has its size and SHA-256 re-verified. Symlink/path escapes, missing files, truncation and same-size corruption are rejected. No remote model fallback or runtime revision override is allowed.

## Request/output guarantees

A dialogue request:

- contains 1–4 explicit speaker bindings;
- contains ordered approved turns;
- is English-only for the initial governed route;
- supplies expected SHA-256 for every WAV reference;
- uses safe speaker IDs and cannot inject native `Speaker N:` control lines through turn text;
- is bounded in turn count and total dialogue size;
- may request only bounded supported generation controls;
- is rejected if expected source/model identity differs from the running service.

Before inference, each reference is resolved under the approved root, size-bounded, copied into a private staging area, SHA-256 checked and WAV-validated. The model consumes the staged copy.

The service:

- maps speaker bindings deterministically to native `Speaker 1`, `Speaker 2`, etc.;
- does not rewrite or extend approved dialogue;
- writes to a unique temporary WAV;
- validates WAV metadata and computes its hash before publication;
- atomically publishes with no-clobber semantics;
- returns source/model revision, device mode, seed, generation controls, reference hashes, output hash and timing.

The result is **not** publication approval. Stickman performs downstream QA and governance gates.

## Target hardware behavior

The intended laptop route is a GTX 1650 Ti 4 GB with CPU/system-RAM offload:

```text
VIBEVOICE_CPU_OFFLOAD=true
VIBEVOICE_CUDA_MAX_MEMORY_MB=3400
VIBEVOICE_CPU_MAX_MEMORY_GB=24
VIBEVOICE_MAX_CONCURRENT_JOBS=1
```

Auto dtype uses FP16 on CUDA devices without BF16 support. FlashAttention 2 is not selected on pre-Ampere GPUs such as the GTX 1650 Ti; SDPA is used instead. These remain conservative starting settings, not benchmark claims.

## Validation boundary

Model-free CI covers source/model policy, non-bypassable local provenance, exact commit pinning, cryptographic model-manifest/artifact integrity, dialogue/control validation, reference root/hash/WAV/staging behavior, lifecycle/concurrency, timeout/cancellation, no-clobber output publication, constrained-GPU dtype/attention selection, private API surface and stable HTTP errors. CI also verifies that runtime Compose configuration does not need or inject source/model revision overrides.

The final hardware acceptance step requires the pinned 1.5B weights and target laptop:

1. create the model manifest from the exact downloaded model commit;
2. build the service image from the reviewed owned-fork `main` commit while supplying that same model commit;
3. start the container without overriding the image-baked source/model identities;
4. load the model using the approved offload mode;
5. synthesize a 30–60 second two-speaker exchange;
6. record load time, generation RTF and peak observed RAM/VRAM;
7. unload and confirm resources are reclaimable by the next local model stage;
8. record the accepted Docker image digest together with the external model-manifest SHA-256 for the future Stickman integration.

That hardware smoke is intentionally separate from the currently active Stickman Automation implementation slot.
