# ADR 0001: Managed Runtime Authority and Owned-Fork Source Policy

- **Status:** Accepted
- **Date:** 2026-08-09
- **Decision owners:** VibeVoice fork / Stickman production architecture
- **Related:** Stickman ADR 0030; SA-175; CFOS ADR 0027/0030

## Context

`Iskanderrus/VibeVoice` is a fork of the community VibeVoice repository with portfolio-specific isolation, source/model pinning, Docker packaging, service lifecycle and output-lineage controls. Treating upstream/community VibeVoice as interchangeable with this fork would break reproducibility and could bypass the controls added specifically for Stickman production.

At the same time, this repository is not a content/business bounded context. It must not own provider selection, dialogue strategy, channel identity, publication approval or CFOS workflow state.

## Decision

`Iskanderrus/VibeVoice` is a **managed production-runtime repository** under the CFOS umbrella.

For the governed VibeVoice capability:

1. production runtime source is **only** `Iskanderrus/VibeVoice`;
2. every production build binds an exact 40-character source commit SHA;
3. moving `main` is not a production identity;
4. `vibevoice-community/VibeVoice` is upstream provenance only and is never a runtime/build dependency for the governed capability;
5. automatic upstream synchronization is prohibited;
6. initial governed model is only `vibevoice/VibeVoice-1.5B` at an exact immutable revision/manifest;
7. 7B is excluded from the normal target-laptop route unless a future ADR changes that policy;
8. model weights, private/reference voices and generated audio remain outside Git/image layers;
9. source/model mismatch fails closed before synthesis.

## Repository authority

This fork owns only runtime/service execution facts and controls:

- isolated Python/model dependency environment;
- HTTP service schema and lifecycle;
- load/unload/synthesis/cancellation behavior;
- exact source/model identity enforcement;
- local model-manifest validation;
- speaker-reference path/hash validation;
- one-job serialization and resource cleanup;
- deterministic request metadata;
- generated WAV validity/hash/timing/runtime lineage;
- Docker packaging and fork-side contract tests.

It does **not** own:

- provider selection/routing/fallback (Stickman under approved CFOS capability policy);
- RED/BLUE cast/voice/dialogue identity (YouTube Channels);
- dialogue/narration/editorial intent (Content Strategy/YouTube Channels/Stickman bounded execution structures);
- market discovery, offer design or CFOS workflow/approval;
- transcript alignment/mastering/final audio QC/publication packet (Stickman);
- commercial/publication rights clearance simply because the code/model executes successfully.

## Consequences

The fork can evolve independently from upstream while production remains reproducible. Upstream commits may be reviewed and manually ported through normal Git review, but no production run silently follows upstream or moving branch state.

## Review triggers

Review if the owned fork is replaced by a different governed runtime repository, a new model family/revision policy is introduced, or the service must own additional runtime facts that materially alter the current boundary.
