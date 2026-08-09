# VibeVoice fork architecture decisions

These ADRs define the portfolio-specific runtime boundary of `Iskanderrus/VibeVoice`. They do not make this fork a platform/domain strategy authority.

Portfolio implementation timing is controlled only by `Iskanderrus/content-factory-os/docs/implementation-order.md`.

| ADR | Decision |
|---|---|
| [0001](0001-managed-runtime-authority-and-source-policy.md) | The owned fork is the only governed VibeVoice runtime source and owns only isolated runtime/service execution facts |
| [0002](0002-conditional-capability-qualification-and-lineage.md) | VibeVoice is a conditional provider capability; exact fork/model/runtime qualification is required before a sealed CFOS plan may select it |
