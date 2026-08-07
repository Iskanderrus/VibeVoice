# Stickman VibeVoice service

This repository (`Iskanderrus/VibeVoice`) owns the local VibeVoice runtime used by Stickman Automation. The service is intentionally isolated from the Stickman Python environment and targets the 1.5B multi-speaker model only.

## Ownership invariants

- Runtime source repository: **`Iskanderrus/VibeVoice` only**.
- Model repository: **`vibevoice/VibeVoice-1.5B` only**.
- Source code and model revision are pinned separately.
- A production image is built with an exact 40-character fork commit SHA.
- Local weights must carry `.stickman-model.json` matching the configured model repository and revision.
- The service never downloads a different model when `VIBEVOICE_LOCAL_FILES_ONLY=true` (the default).
- 7B, automatic upstream synchronization, public Gradio sharing, fine-tuning and dialogue improvisation are outside this service.

The community repository may be consulted only as upstream provenance during deliberate review. It is not an authorized runtime/build source.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Process liveness |
| `GET` | `/ready` | Lifecycle state and synthesis readiness |
| `GET` | `/capabilities` | Pinned identities and capabilities |
| `POST` | `/load` | Load the configured 1.5B model |
| `POST` | `/synthesize-dialogue` | Generate one bounded multi-speaker WAV |
| `POST` | `/cancel/{job_id}` | Cooperative cancellation |
| `POST` | `/unload` | Release model/runtime resources |

The service has no public UI and should bind only to localhost or an internal Compose network.

## Lifecycle

```text
unloaded -> loading -> ready -> generating -> ready
                         |
                         +-> unloading -> unloaded

runtime/model failures -> failed -> load or unload
```

Exactly one heavyweight operation may run at a time. Generation uses the fork's native `stop_check_fn` hook for cooperative cancellation.

## Prepare the 1.5B weights

Download the model into a local directory and record the exact revision used. Then create the service manifest:

```bash
python scripts/create_stickman_model_manifest.py \
  --model-dir /absolute/path/to/VibeVoice-1.5B \
  --revision "<exact-model-revision>"
```

This writes:

```text
/absolute/path/to/VibeVoice-1.5B/.stickman-model.json
```

The container rejects missing/mismatched manifests before importing Torch or allocating model memory.

## Configure

```bash
cp .env.stickman.example .env.stickman
```

Set:

```text
VIBEVOICE_SOURCE_REVISION=<git rev-parse HEAD>
VIBEVOICE_MODEL_REVISION=<same revision stored in .stickman-model.json>
VIBEVOICE_MODEL_HOST_DIR=<absolute model directory>
VIBEVOICE_REFERENCE_HOST_DIR=<absolute approved references directory>
VIBEVOICE_OUTPUT_HOST_DIR=<absolute writable output directory>
```

Reference audio is mounted read-only. Each synthesis request must supply the expected SHA-256 for every speaker reference; path escapes and symlink escapes are rejected.

## Build and start

```bash
docker compose --env-file .env.stickman \
  -f docker-compose.stickman.yml build

docker compose --env-file .env.stickman \
  -f docker-compose.stickman.yml up -d
```

Check:

```bash
curl http://127.0.0.1:8765/health
curl http://127.0.0.1:8765/ready
curl http://127.0.0.1:8765/capabilities
```

Load explicitly:

```bash
curl -X POST http://127.0.0.1:8765/load \
  -H 'content-type: application/json' \
  -d '{
    "model_revision": "<exact-model-revision>",
    "source_revision": "<exact-fork-commit>"
  }'
```

## Target laptop defaults

The normal local target is a GTX 1650 Ti with 4 GB VRAM and system-RAM offload. Defaults therefore reserve headroom rather than trying to place the full 1.5B model on the GPU:

```text
VIBEVOICE_CPU_OFFLOAD=true
VIBEVOICE_CUDA_MAX_MEMORY_MB=3400
VIBEVOICE_CPU_MAX_MEMORY_GB=24
VIBEVOICE_MAX_CONCURRENT_JOBS=1
```

These values are starting limits, not claimed benchmark results. The first model smoke on the actual laptop must record peak VRAM/RAM, load time, real-time factor and resource-reclaim behavior.

## Dialogue request

The request is approved content. The service maps the ordered speaker bindings to native `Speaker 1`, `Speaker 2`, etc. It does not rewrite text.

```json
{
  "job_id": "family-001-long-segment-03-attempt-01",
  "model_revision": "<exact-model-revision>",
  "source_revision": "<exact-fork-commit>",
  "language": "en",
  "seed": 18427,
  "speakers": [
    {
      "speaker_id": "RED",
      "reference_path": "/shared/references/red.wav",
      "reference_sha256": "<sha256>"
    },
    {
      "speaker_id": "BLUE",
      "reference_path": "/shared/references/blue.wav",
      "reference_sha256": "<sha256>"
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
    "cfg_scale": 1.3,
    "inference_steps": 10,
    "disable_prefill": false,
    "max_length_times": 2.0
  }
}
```

A successful result contains the source/model identity, device mode, seed and generation controls, reference hashes, WAV SHA-256, duration, sample rate and timing. Stickman remains responsible for Whisper alignment, mastering, audio-content QC, rights/disclosure checks and human approval.

## Output behavior

- Output path is deterministic: `${VIBEVOICE_OUTPUT_DIR}/${job_id}.wav`.
- Existing output is never overwritten silently.
- Generation writes to a unique temporary file first and atomically renames it after WAV validation.
- Output must be a non-empty 24 kHz WAV.
- Runtime failures transition the service to `failed`; request/path/hash/revision validation failures leave a healthy loaded runtime ready.

## Resource lifecycle

`POST /unload` releases the model, runs garbage collection and clears CUDA caches when available. Idle unload is enabled by default after 300 seconds. SIGTERM/application shutdown requests cancellation and then bounded cleanup.

Do not run VibeVoice concurrently with another heavy local model on the target 4 GB GPU.

## Tests

The service contract suite is model-free:

```bash
python -m pip install \
  fastapi==0.116.1 httpx==0.28.1 pydantic==2.11.7 pytest==8.4.1
pytest -q tests/test_stickman_service.py
```

The suite covers source/model policy, baked revision checks, model-manifest checks, lifecycle, output lineage, reference path/hash enforcement, concurrency/cancellation and API validation.

The model-backed smoke test is deliberately separate because it requires the pinned 1.5B weights and target hardware.
