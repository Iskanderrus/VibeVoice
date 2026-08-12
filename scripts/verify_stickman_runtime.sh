#!/usr/bin/env bash
set -euo pipefail

EXPECTED_TOKENIZER_REPOSITORY="Qwen/Qwen2.5-1.5B"
COMMIT_RE='^[0-9a-fA-F]{40}$'
CHECK_ONLY=false

if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY=true
elif [[ $# -gt 0 ]]; then
  echo "usage: $0 [--check-only]" >&2
  exit 2
fi

model_root="${VIBEVOICE_MODEL_PATH:-/models/VibeVoice-1.5B}"
hf_home="${HF_HOME:-${model_root}/.hf-cache}"
tokenizer_repository="${VIBEVOICE_TOKENIZER_REPOSITORY:-}"
tokenizer_revision="${VIBEVOICE_TOKENIZER_REVISION:-}"

if [[ "$tokenizer_repository" != "$EXPECTED_TOKENIZER_REPOSITORY" ]]; then
  echo "error: tokenizer repository must be ${EXPECTED_TOKENIZER_REPOSITORY}; got: ${tokenizer_repository:-<missing>}" >&2
  exit 1
fi
if [[ ! "$tokenizer_revision" =~ $COMMIT_RE ]] || [[ "$tokenizer_revision" == "0000000000000000000000000000000000000000" ]]; then
  echo "error: VIBEVOICE_TOKENIZER_REVISION must be an exact non-zero 40-character commit SHA" >&2
  exit 1
fi

cache_repo="${hf_home}/hub/models--Qwen--Qwen2.5-1.5B"
ref_file="${cache_repo}/refs/main"
if [[ ! -f "$ref_file" ]]; then
  echo "error: pinned local Qwen tokenizer cache ref is missing: $ref_file" >&2
  exit 1
fi
cached_revision="$(tr -d '\r\n' < "$ref_file")"
if [[ "$cached_revision" != "$tokenizer_revision" ]]; then
  echo "error: local Qwen tokenizer cache revision does not match image identity" >&2
  echo "image: $tokenizer_revision" >&2
  echo "cache: $cached_revision" >&2
  exit 1
fi

snapshot="${cache_repo}/snapshots/${tokenizer_revision}"
if [[ ! -d "$snapshot" ]]; then
  echo "error: pinned Qwen tokenizer snapshot is missing: $snapshot" >&2
  exit 1
fi
for filename in tokenizer.json tokenizer_config.json merges.txt vocab.json; do
  if [[ ! -r "$snapshot/$filename" ]]; then
    echo "error: required Qwen tokenizer artifact is missing or unreadable: $snapshot/$filename" >&2
    exit 1
  fi
done

if $CHECK_ONLY; then
  printf 'tokenizer=%s@%s\n' "$tokenizer_repository" "$tokenizer_revision"
  exit 0
fi

exec python -m stickman_service
