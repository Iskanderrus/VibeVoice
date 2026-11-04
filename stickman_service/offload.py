"""Device-map construction for CUDA + CPU offload of the 1.5B model.

Why this module exists (see issue for the full analysis):

* ``device_map="auto"`` maps child modules only. Top-level scalar buffers such as
  ``model.speech_scaling_factor`` / ``model.speech_bias_factor`` get no device and
  transformers raises ``... doesn't have any device set``.
* Offloading the acoustic/semantic tokenizers breaks code that reads their
  parameters directly (for example ``block.gamma`` inside ``forward_features``),
  which then remain on the ``meta`` device. Those modules are therefore kept
  resident on the GPU; only the language model is offloaded.
"""

from __future__ import annotations

import gc
from typing import Any, Iterable

# Modules that must stay on the GPU because offload breaks their direct
# parameter access. Keys match the names used by transformers' device_map lookup.
RESIDENT_MODULE_PREFIXES: tuple[str, ...] = (
    "model.acoustic_tokenizer",
    "model.semantic_tokenizer",
    "model.acoustic_connector",
    "model.semantic_connector",
    "model.prediction_head",
    "model.speech_scaling_factor",
    "model.speech_bias_factor",
)


def build_offload_device_map(
    model: Any,
    *,
    gpu_budget_mib: int,
    cpu_gib: int,
    no_split_module_classes: Iterable[str] = (),
    resident_prefixes: tuple[str, ...] = RESIDENT_MODULE_PREFIXES,
) -> dict[str, Any]:
    """Return a complete device_map for ``model`` (ideally built on meta tensors).

    ``gpu_budget_mib`` is the GPU share left for offloadable modules; the resident
    prefixes are added on GPU on top of it, so callers must reserve their size.
    """
    from accelerate.utils import infer_auto_device_map

    if gpu_budget_mib < 0:
        raise ValueError("gpu_budget_mib must be non-negative")
    device_map = infer_auto_device_map(
        model,
        max_memory={0: f"{gpu_budget_mib}MiB", "cpu": f"{cpu_gib}GiB"},
        no_split_module_classes=list(no_split_module_classes),
    )
    for key in list(device_map):
        if key.startswith(resident_prefixes):
            device_map[key] = 0
    for prefix in resident_prefixes:
        device_map.setdefault(prefix, 0)
    return dict(device_map)


def build_empty_vibevoice_model(source: str, model_cls: Any, config_cls: Any) -> Any:
    """Instantiate the model on the meta device (no weights allocated)."""
    from accelerate import init_empty_weights

    config = config_cls.from_pretrained(source, local_files_only=True)
    with init_empty_weights():
        model = model_cls(config)
    model.tie_weights()
    return model


def release(model: Any) -> None:
    del model
    gc.collect()

