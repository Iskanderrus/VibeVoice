# ADR 0002: Conditional Capability Qualification and Exact Execution Lineage

- **Status:** Accepted
- **Date:** 2026-08-09
- **Decision owners:** VibeVoice fork / CFOS provider architecture
- **Related:** CFOS-029, CFOS-035, CFOS-024; Stickman SA-175, SA-189

## Context

VibeVoice is useful for contextual multi-speaker RED/BLUE dialogue, but it is not required for every content family. Making it a mandatory CFOS critical-path dependency would couple the portfolio to one optional TTS implementation. Making it freely selectable without qualification would allow an untested source/model/hardware tuple into a sealed production plan.

## Decision

VibeVoice is a **conditional provider capability**.

A CFOS planning context may select the governed VibeVoice capability only when an exact runtime profile has passed CFOS-035 qualification. The profile binds at minimum:

- capability/runtime profile ID/version/digest;
- repository `Iskanderrus/VibeVoice`;
- exact fork source revision;
- model repository `vibevoice/VibeVoice-1.5B`;
- exact model revision/manifest digest;
- service-contract version/capability response;
- target hardware/device/offload profile used for qualification;
- qualification status/evidence and validity/revocation state;
- supported language/speaker/control/concurrency limits.

If VibeVoice is not selected, this qualification does not block CFOS-007 or another qualified TTS route.

If VibeVoice **is** selected, missing/stale/revoked/wrong-source/wrong-model qualification blocks the plan or producer call before synthesis. Neither CFOS nor Stickman may silently substitute upstream/community VibeVoice, a moving branch, another model, or an unqualified revision.

## Qualification evidence

The first governed qualification requires:

- fork-side service contract tests for the exact source revision;
- image/runtime source-revision self-check;
- exact local model-manifest validation;
- target-hardware load;
- one 30–60 second two-speaker synthesis;
- load time, generation wall time/RTF and observed RAM/VRAM where measurable;
- deterministic output WAV/hash metadata;
- unload/resource reclamation sufficient for the next local model stage;
- explicit result such as `qualified`, `preview_only`, `hold` or `rejected`;
- rights/disclosure evidence state recorded separately from technical success.

## Execution lineage

When selected, Stickman SA-189 must record the actual source/model/service/device/reference/generation/output identities returned by the runtime. Fallback to another TTS provider creates explicit changed execution provenance and must be allowed by the pinned policy; it never pretends to be the original VibeVoice execution.

## Consequences

CFOS remains provider-flexible while VibeVoice stays rigorously reproducible when used. Qualification can be refreshed for new fork/model/runtime revisions without making the fork a global dependency.
