# Stickman VibeVoice service

This repository (`Iskanderrus/VibeVoice`) owns the local VibeVoice runtime used by Stickman Automation. The service is intentionally isolated from the Stickman Python environment and targets the 1.5B multi-speaker model only.

## Ownership invariants

- Runtime source repository: **`Iskanderrus/VibeVoice` only**.
- Model repository: **`vibevoice/VibeVoice-1.5B` only**.
- Source and model revisions must both be exact 40-character commit SHAs.
- Governed images are built from a clean checkout whose `origin` is our fork and whose `HEAD` equals the fetched `origin/main`.
- Both source and model revision SHAs are baked into the image and are not runtime-overridable; Compose does not replace either when the container starts.
- Local weights must carry `.stickman-model.json` matching the image-baked model repository/revision and inventorying each local model artifact by path, byte size and SHA-256.
- `VIBEVOICE_LOCAL_FILES_ONLY=true` and `VIBEVOICE_REQUIRE_MODEL_MANIFEST=true` are mandatory static policy, not optional development switches.
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

The service has no public UI, documentation UI or OpenAPI endpoint, and the supplied Compose configuration publishes it only on `127.0.0.1`.

## Lifecycle

```text
unloaded -> loading -> ready -> generating -> ready
                         |          |
                         |          +-- cancel/timeout -> unload -> unloaded
                         +-> unloading -> unloaded

runtime/model failures -> failed -> explicit load or unload
```

Exactly one heavyweight operation may run at a time. Timeout is enforced inside the manager as well as at the HTTP boundary. Cancellation is cooperative through the fork's native `stop_check_fn`. A cancelled or timed-out generation unloads the runtime before another generation is accepted; this prevents reuse of a partially interrupted inference state.

## Prepare the 1.5B weights

Download the model into a local directory at an exact Hugging Face commit. Then create the service manifest:

```bash
python scripts/create_stickman_model_manifest.py \
  --model-dir /absolute/path/to/VibeVoice-1.5B \
  --revision "<40-character-model-commit>"
```

This writes:

```text
/absolute/path/to/VibeVoice-1.5B/.stickman-model.json
```

The manifest records the approved repository/revision plus every local model artifact path, byte size and SHA-256. It requires `config.json` and at least one model weight file. At load time the service re-resolves every artifact under the model root and verifies its size and SHA-256 before importing Torch or allocating model memory. Missing, redirected, root-escaping, truncated or same-size corrupted artifacts are rejected.

## Configure

```bash
cp .env.stickman.example .env.stickman
```

Set at minimum for the governed **build**:

```text
VIBEVOICE_MODEL_REVISION=<same 40-character commit stored in .stickman-model.json>
VIBEVOICE_MODEL_HOST_DIR=<absolute model directory>
VIBEVOICE_REFERENCE_HOST_DIR=<absolute approved references directory>
VIBEVOICE_OUTPUT_HOST_DIR=<absolute writable output directory>
```

Do **not** manually choose `VIBEVOICE_SOURCE_REVISION` for a governed build. The build helper derives it from the clean owned checkout and verifies it against `origin/main`. The helper supplies both exact source and model revisions to the Docker build; both identities are baked into the image and remain authoritative when Compose starts the already-built container.

Reference audio is mounted read-only. Each synthesis request supplies the expected SHA-256 for every speaker reference. Before inference the service resolves the path below the approved root, bounds the file size, copies it to a private staging directory, hashes the staged copy and validates its WAV metadata. This closes the mutable-bind-mount TOCTOU window between validation and model consumption.

## Build and start

Build through the governed helper:

```bash
set -a
. ./.env.stickman
set +a
bash scripts/build_stickman_service.sh
```

The helper refuses to build when:

- `origin` is not `Iskanderrus/VibeVoice`;
- the working tree is dirty;
- `HEAD` is not the exact fetched `origin/main` commit;
- the model revision is not an exact 40-character commit;
- required host paths are absent or not absolute.

The Dockerfile additionally rejects all-zero placeholder source/model revisions during an actual build.

Then start the already-built service. Revision variables are not needed to establish runtime identity; they come from the image. The host mount variables are still required:

```bash
docker compose --env-file .env.stickman \
  -f docker-compose.stickman.yml up -d --no-build
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
    "model_revision": "<40-character-model-commit>",
    "source_revision": "<40-character-owned-fork-commit>"
  }'
```

The `/load` values are expected identities, not overrides; mismatches are rejected.

## Target laptop defaults

The normal local target is a GTX 1650 Ti with 4 GB VRAM and system-RAM offload. Auto mode therefore:

- uses FP16 on CUDA devices that do not support BF16;
- uses BF16 only when CUDA reports BF16 support;
- avoids FlashAttention 2 on pre-Ampere CUDA devices such as the GTX 1650 Ti and uses SDPA instead;
- reserves GPU headroom and permits CPU/system-RAM offload.

Defaults:

```text
VIBEVOICE_CPU_OFFLOAD=true
VIBEVOICE_CUDA_MAX_MEMORY_MB=3400
VIBEVOICE_CPU_MAX_MEMORY_GB=24
VIBEVOICE_MAX_CONCURRENT_JOBS=1
```

These values are starting limits, not benchmark results. The first model smoke on the actual laptop must record peak VRAM/RAM, load time, real-time factor and resource-reclaim behavior.

## Dialogue request

The request is approved content. The service maps the ordered speaker bindings to native `Speaker 1`, `Speaker 2`, etc. It never rewrites or improvises text. Because `Speaker N:` is VibeVoice control syntax, turn text containing native speaker-control lines is rejected. Speaker IDs are also restricted to a safe identifier grammar.

```json
{
  "job_id": "family-001-long-segment-03-attempt-01",
  "model_revision": "<40-character-model-commit>",
  "source_revision": "<40-character-owned-fork-commit>",
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
- Generation writes to a unique temporary WAV first.
- WAV validation, SHA-256 and response metadata are completed before publication.
- Publication uses an atomic no-clobber hard link in the same output filesystem; a late competing destination causes `OUTPUT_EXISTS` rather than overwrite.
- Output must be a non-empty, mono, 24 kHz WAV with an approved sample width.
- Reference/path/hash/output-exists request failures leave a healthy loaded runtime ready.
- Model/generation/output failures fail closed and unload or mark the service failed as appropriate.

## Resource lifecycle

`POST /unload` releases the model, runs garbage collection and clears CUDA caches when available. Idle unload is enabled by default after 300 seconds and runs only from the `ready` state. SIGTERM/application shutdown requests cancellation and then bounded cleanup.

Do not run VibeVoice concurrently with another heavy local model on the target 4 GB GPU.

## Tests

The service contract suite is model-free:

```bash
python -m pip install \
  fastapi==0.116.1 httpx==0.28.1 pydantic==2.11.7 pytest==8.4.1
python -m pytest -q tests/test_stickman_service.py
```

CI also compiles the service/tooling, syntax-checks the governed build helper, verifies the owned-fork build guard, validates runtime Compose without source/model revision overrides, exercises the cryptographic model-manifest generator, and runs the contract regression suite.

The model-backed smoke test is deliberately separate because it requires the pinned 1.5B weights and target hardware. Code/contract merge readiness does not claim hardware acceptance; the image used by Stickman should be recorded by digest only after that smoke passes.
