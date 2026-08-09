# docs-manifest — Iskanderrus/VibeVoice managed runtime

```yaml
version: 1
project: VibeVoice
role: managed_production_runtime
portfolio: content-factory-os
categories:
  adr:
    path: docs/decisions
    index: docs/decisions/README.md
    authoritative_for: runtime_source_service_boundary
  plan:
    path: docs/plans
    index: docs/plans/README.md
    authoritative_for: local_runtime_qualification_scope
  implementation_order:
    path: docs/implementation-order.md
    authoritative: false
    note: CFOS docs/implementation-order.md is the sole scheduling authority
external_tickets:
  stickman_integration: Iskanderrus/stickman-automation#336
  cfos_qualification: Iskanderrus/content-factory-os#41
```
