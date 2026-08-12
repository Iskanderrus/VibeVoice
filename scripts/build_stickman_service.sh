#!/usr/bin/env bash
set -euo pipefail

EXPECTED_REPO="Iskanderrus/VibeVoice"
MODEL_REPO="vibevoice/VibeVoice-1.5B"
TOKENIZER_REPO="Qwen/Qwen2.5-1.5B"
COMMIT_RE='^[0-9a-fA-F]{40}$'
CHECK_ONLY=false

if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY=true
elif [[ $# -gt 0 ]]; then
  echo "usage: $0 [--check-only]" >&2
  exit 2
fi

repo_root="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "error: build must run from a Git checkout" >&2
  exit 1
}
cd "$repo_root"

origin="$(git config --get remote.origin.url || true)"
case "$origin" in
  "https://github.com/${EXPECTED_REPO}"|\
  "https://github.com/${EXPECTED_REPO}.git"|\
  "git@github.com:${EXPECTED_REPO}.git"|\
  "ssh://git@github.com/${EXPECTED_REPO}.git")
    ;;
  *)
    echo "error: origin must be the owned fork ${EXPECTED_REPO}; got: ${origin:-<missing>}" >&2
    exit 1
    ;;
esac

if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "error: working tree must be clean before building the governed image" >&2
  git status --short >&2
  exit 1
fi

head_sha="$(git rev-parse HEAD)"
if [[ ! "$head_sha" =~ $COMMIT_RE ]]; then
  echo "error: could not resolve an exact source commit" >&2
  exit 1
fi
if [[ -n "${VIBEVOICE_SOURCE_REVISION:-}" && "${VIBEVOICE_SOURCE_REVISION,,}" != "${head_sha,,}" ]]; then
  echo "error: VIBEVOICE_SOURCE_REVISION does not match checked-out HEAD" >&2
  exit 1
fi
export VIBEVOICE_SOURCE_REVISION="$head_sha"

if [[ ! "${VIBEVOICE_MODEL_REVISION:-}" =~ $COMMIT_RE ]]; then
  echo "error: VIBEVOICE_MODEL_REVISION must be an exact 40-character model commit SHA" >&2
  exit 1
fi
if [[ ! "${VIBEVOICE_TOKENIZER_REVISION:-}" =~ $COMMIT_RE ]]; then
  echo "error: VIBEVOICE_TOKENIZER_REVISION must be an exact 40-character tokenizer commit SHA" >&2
  exit 1
fi
export VIBEVOICE_MODEL_REPOSITORY="$MODEL_REPO"
export VIBEVOICE_TOKENIZER_REPOSITORY="$TOKENIZER_REPO"

printf 'owned_source=%s@%s\n' "$EXPECTED_REPO" "$VIBEVOICE_SOURCE_REVISION"
printf 'model=%s@%s\n' "$MODEL_REPO" "$VIBEVOICE_MODEL_REVISION"
printf 'tokenizer=%s@%s\n' "$TOKENIZER_REPO" "$VIBEVOICE_TOKENIZER_REVISION"

if $CHECK_ONLY; then
  exit 0
fi

# A governed image must be built from the exact reviewed commit currently published
# on our fork's main branch. This prevents an arbitrary local history from merely
# renaming its origin to the owned repository and claiming an approved source SHA.
git fetch --quiet --no-tags origin main
origin_main_sha="$(git rev-parse refs/remotes/origin/main)"
if [[ "$head_sha" != "$origin_main_sha" ]]; then
  echo "error: governed builds require HEAD == origin/main" >&2
  echo "HEAD:        $head_sha" >&2
  echo "origin/main: $origin_main_sha" >&2
  exit 1
fi

for name in VIBEVOICE_MODEL_HOST_DIR VIBEVOICE_REFERENCE_HOST_DIR VIBEVOICE_OUTPUT_HOST_DIR; do
  value="${!name:-}"
  if [[ -z "$value" || "$value" != /* ]]; then
    echo "error: $name must be set to an absolute host path" >&2
    exit 1
  fi
done
if [[ ! -d "$VIBEVOICE_MODEL_HOST_DIR" ]]; then
  echo "error: VIBEVOICE_MODEL_HOST_DIR does not exist: $VIBEVOICE_MODEL_HOST_DIR" >&2
  exit 1
fi
if [[ ! -d "$VIBEVOICE_REFERENCE_HOST_DIR" ]]; then
  echo "error: VIBEVOICE_REFERENCE_HOST_DIR does not exist: $VIBEVOICE_REFERENCE_HOST_DIR" >&2
  exit 1
fi
mkdir -p "$VIBEVOICE_OUTPUT_HOST_DIR"

tokenizer_cache="${VIBEVOICE_MODEL_HOST_DIR}/.hf-cache/hub/models--Qwen--Qwen2.5-1.5B"
tokenizer_ref="${tokenizer_cache}/refs/main"
if [[ ! -f "$tokenizer_ref" ]]; then
  echo "error: pinned local Qwen tokenizer cache ref is missing: $tokenizer_ref" >&2
  exit 1
fi
cached_tokenizer_revision="$(tr -d '\r\n' < "$tokenizer_ref")"
if [[ "$cached_tokenizer_revision" != "$VIBEVOICE_TOKENIZER_REVISION" ]]; then
  echo "error: local Qwen tokenizer cache revision does not match VIBEVOICE_TOKENIZER_REVISION" >&2
  echo "configured: $VIBEVOICE_TOKENIZER_REVISION" >&2
  echo "cache:      $cached_tokenizer_revision" >&2
  exit 1
fi
tokenizer_snapshot="${tokenizer_cache}/snapshots/${VIBEVOICE_TOKENIZER_REVISION}"
for filename in tokenizer.json tokenizer_config.json merges.txt vocab.json; do
  if [[ ! -r "$tokenizer_snapshot/$filename" ]]; then
    echo "error: required Qwen tokenizer artifact is missing or unreadable: $tokenizer_snapshot/$filename" >&2
    exit 1
  fi
done

exec docker compose -f docker-compose.stickman.yml build vibevoice
