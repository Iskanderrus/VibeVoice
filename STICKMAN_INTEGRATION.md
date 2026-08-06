# Stickman Automation integration plan

- **Status:** Proposed
- **Date:** 2026-08-06
- **Consumer issue:** [Iskanderrus/stickman-automation#336](https://github.com/Iskanderrus/stickman-automation/issues/336)
- **Channel-policy issue:** [Iskanderrus/youtube-channels#13](https://github.com/Iskanderrus/youtube-channels/issues/13)
- **Target model:** `vibevoice/VibeVoice-1.5B` at an exact pinned model revision
- **Normal target path:** local/offline RED/BLUE multi-speaker dialogue; 7B is out of scope for the target laptop

## Purpose

This fork is the owned, reviewable runtime source for an optional VibeVoice 1.5B service consumed by Stickman Automation. The integration must isolate VibeVoice dependencies from Stickman, expose a narrow local API, preserve exact source/model lineage and support safe model loading/unloading on constrained hardware.

This document is an implementation handoff. It does not declare the model, reference voices or generated output cleared for monetized publication. Stickman’s existing rights, evidence, disclosure and publication gates remain authoritative.

## Repository policy

1. Production integration uses an exact reviewed commit from this fork, never a moving `main` reference.
2. Upstream changes are fetched and reviewed deliberately; no production container runs `git pull` or follows upstream automatically.
3. Model weights, generated audio and private/reference voice recordings are never committed to Git.
4. Local Stickman-specific changes live on reviewed branches and are tagged after benchmark acceptance.
5. Upstream licence and attribution files remain intact.
6. The integration does not remove provenance, watermark, disclosure or other safety mechanisms.

## Service boundary

The fork will provide a local HTTP service rather than exposing the Gradio demo.

Required endpoints:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Process is alive; does not imply model readiness |
| `GET` | `/ready` | Current lifecycle state and whether synthesis is available |
| `GET` | `/capabilities` | Model/runtime capabilities and pinned revisions |
| `POST` | `/load` | Load the configured 1.5B model in an approved device/offload mode |
| `POST` | `/synthesize-dialogue` | Generate one bounded ordered multi-speaker performance |
| `POST` | `/unload` | Release model/runtime resources |

The service listens on a private/local interface or an internal Compose network. Public Gradio sharing and unauthenticated internet exposure are not part of this integration.

## Lifecycle state machine

```text
unloaded
  -> loading
  -> ready
  -> generating
  -> ready
  -> unloading
  -> unloaded
```

Failures transition to a stable `failed` state with a machine-readable reason and a bounded cleanup attempt. Supported error classes include:

- invalid request;
- model/revision absent;
- model load failure;
- GPU out of memory;
- generation timeout;
- generation failure;
- cancellation/client disconnect;
- invalid output;
- unload/cleanup failure.

Only one synthesis request runs at a time. The service may queue one bounded request or reject concurrent requests explicitly; it must not run multiple model generations concurrently on the target machine.

## Configuration

Initial configuration surface:

```text
VIBEVOICE_MODEL_PATH=/models/VibeVoice-1.5B
VIBEVOICE_MODEL_REPOSITORY=vibevoice/VibeVoice-1.5B
VIBEVOICE_MODEL_REVISION=<exact model revision>
VIBEVOICE_SOURCE_REVISION=<exact fork commit>
VIBEVOICE_DEVICE=auto
VIBEVOICE_DTYPE=auto
VIBEVOICE_CPU_OFFLOAD=true
VIBEVOICE_IDLE_UNLOAD_SECONDS=300
VIBEVOICE_MAX_CONCURRENT_JOBS=1
VIBEVOICE_OUTPUT_DIR=/shared/output
VIBEVOICE_REFERENCE_DIR=/shared/references
```

The target 4 GB GPU laptop is expected to require CPU/offload or another validated low-memory mode. A configuration that cannot load safely must fail clearly rather than silently selecting 7B, downloading a different model revision or changing quality-critical parameters.

## Dialogue request

A request represents one approved semantic/performance chunk, not a complete episode.

Example shape:

```json
{
  "job_id": "family-001-long-segment-03-attempt-01",
  "model_revision": "<expected revision>",
  "language": "en",
  "seed": 18427,
  "speakers": [
    {
      "speaker_id": "RED",
      "reference_path": "/shared/references/red.wav",
      "reference_sha256": "<hash>"
    },
    {
      "speaker_id": "BLUE",
      "reference_path": "/shared/references/blue.wav",
      "reference_sha256": "<hash>"
    }
  ],
  "turns": [
    {
      "speaker_id": "RED",
      "text": "Everyone says consistency is the secret. Are they all wrong?"
    },
    {
      "speaker_id": "BLUE",
      "text": "Not entirely. Consistency matters only when each attempt produces useful feedback."
    }
  ],
  "generation": {
    "cfg_scale": null,
    "inference_steps": null
  }
}
```

Requirements:

- turns remain ordered;
- every turn speaker exists in the speaker bindings;
- paths resolve only under configured read-only reference roots;
- expected source/model revisions must match the running service;
- text is treated as approved content and is not rewritten or extended;
- unsupported controls are rejected rather than ignored silently;
- references are validated by path policy and declared hash before use.

## Result contract

A successful response includes:

```json
{
  "job_id": "family-001-long-segment-03-attempt-01",
  "status": "completed",
  "output_path": "/shared/output/family-001/segment-03/attempt-01.wav",
  "duration_seconds": 31.42,
  "source_revision": "<fork commit>",
  "model_repository": "vibevoice/VibeVoice-1.5B",
  "model_revision": "<model revision>",
  "device_mode": "cuda_cpu_offload",
  "seed": 18427,
  "generation": {},
  "speaker_reference_hashes": {
    "RED": "<hash>",
    "BLUE": "<hash>"
  },
  "audio_sha256": "<hash>",
  "timings": {
    "load_seconds": 0.0,
    "generation_seconds": 0.0
  }
}
```

The service does not declare the audio approved. Stickman performs transcript alignment, mastering, speaker/text quality checks, rights/disclosure verification and human review.

## Candidate implementation layout

```text
stickman_service/
  __init__.py
  api.py
  schemas.py
  settings.py
  model_manager.py
  synthesis.py
  errors.py
  hashing.py
  tests/
Dockerfile.stickman
requirements-stickman.lock
STICKMAN_INTEGRATION.md
```

The exact layout may change to fit the codebase, but the API, isolation, revision and lifecycle requirements remain fixed.

## Docker requirements

The image contains:

- this fork at the pinned source revision;
- locked runtime dependencies;
- the local API wrapper;
- health/readiness checks;
- required audio utilities.

The image does not contain:

- model weights;
- generated audio;
- reference recordings;
- Stickman source;
- credentials;
- the host Docker socket.

Runtime mounts:

- persistent model cache/weights;
- read-only approved reference directory;
- writable shared output directory.

Use a non-root runtime where practical. The service must handle SIGTERM and client cancellation with bounded model/resource cleanup.

## Validation plan

### Unit

- schema validation and turn ordering;
- configured-root path enforcement;
- revision mismatch rejection;
- lifecycle transitions;
- stable error mapping;
- output/reference hashing;
- idle-unload behaviour.

### Service contract

- health versus readiness distinction;
- unloaded request handling;
- explicit load/unload;
- concurrent request rejection/serialization;
- cancellation and timeout;
- OOM mapping;
- deterministic metadata shape.

### Model smoke

Using the pinned 1.5B model on the target laptop:

1. load in approved offload mode;
2. synthesize a 30–60 second two-speaker RED/BLUE exchange;
3. write WAV to the shared output mount;
4. return complete revision/hash metadata;
5. unload;
6. demonstrate that the next Stickman local model stage can acquire resources.

Record load time, generation real-time factor, peak observed RAM/VRAM, failure rate and unload/reclaim time.

## Acceptance criteria

- The service runs without installing VibeVoice dependencies into Stickman.
- Exact fork and model revisions are reported and enforced.
- A bounded two-speaker WAV is generated through the API.
- Model weights/reference audio remain outside Git and image layers.
- One-job serialization and cleanup work under failure.
- Explicit unload returns the service to an unloaded state suitable for the next local model stage.
- No public sharing endpoint is enabled by default.
- Stickman can reject or approve the result independently using its own QA and rights gates.

## Non-goals

- VibeVoice 7B support on the normal target-laptop route;
- fine-tuning or unrestricted voice cloning;
- dialogue generation or improvisation;
- Stickman provider routing, GUI, mastering, transcription, publication or analytics;
- automatic upstream synchronization;
- removal of safety/provenance features.

## Upgrade process

For any upstream or model upgrade:

1. fetch without merging;
2. review source and dependency diffs;
3. build a separate test image;
4. run unit/contract tests;
5. run the frozen RED/BLUE benchmark corpus;
6. compare quality, resource use and failures against the approved version;
7. review rights/model-card changes;
8. approve or reject explicitly;
9. create a new immutable fork tag and update Stickman configuration through review.

The previous accepted tag remains available for rollback.
