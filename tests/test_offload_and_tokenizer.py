"""Focused tests for the 4 GiB offload device map and the offline tokenizer pin.

Behaviour locked here (see the issue for the field evidence):
* every parameter and buffer of the model is covered by the device map;
* resident modules (tokenizers, connectors, prediction head, scalar speech
  buffers) are always placed on GPU, so offload cannot strand them on `meta`;
* the offline tokenizer revision defaults to the verified cache revision.
"""

from __future__ import annotations

import re

import pytest
import torch
from torch import nn

from stickman_service.offload import RESIDENT_MODULE_PREFIXES, build_offload_device_map
from stickman_service.settings import DEFAULT_TOKENIZER_REVISION, Settings


class _Inner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = nn.Sequential(*[nn.Linear(64, 64) for _ in range(4)])
        self.acoustic_tokenizer = nn.Module()
        self.acoustic_tokenizer.encoder = nn.Module()
        self.acoustic_tokenizer.encoder.stages = nn.Sequential(
            nn.Linear(64, 64), nn.Linear(64, 64)
        )
        self.prediction_head = nn.Linear(64, 64)
        self.register_buffer("speech_scaling_factor", torch.tensor(float("nan")))
        self.register_buffer("speech_bias_factor", torch.tensor(float("nan")))


class _ToyVibeVoice(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Inner()
        self.lm_head = nn.Linear(64, 64)


def _toy_on_meta() -> nn.Module:
    from accelerate import init_empty_weights

    with init_empty_weights():
        return _ToyVibeVoice()


def _device_for(name: str, device_map: dict):
    # Same lookup transformers uses: keys sorted in reverse, joined as alternation.
    pattern = "|".join(re.escape(k) for k in sorted(device_map, reverse=True))
    match = re.search(pattern, name)
    return None if match is None else device_map[match.group()]


def _mapped(name: str, device_map: dict) -> bool:
    return _device_for(name, device_map) is not None


def _all_tensor_names(model: nn.Module) -> list[str]:
    return [n for n, _ in model.named_parameters()] + [n for n, _ in model.named_buffers()]


def test_every_parameter_and_buffer_is_covered_by_device_map() -> None:
    model = _toy_on_meta()
    device_map = build_offload_device_map(model, gpu_budget_mib=64, cpu_gib=1)

    unmapped = [n for n in _all_tensor_names(model) if not _mapped(n, device_map)]

    assert unmapped == []


def test_speech_scalar_buffers_have_an_explicit_device() -> None:
    model = _toy_on_meta()
    device_map = build_offload_device_map(model, gpu_budget_mib=64, cpu_gib=1)

    assert "model.speech_scaling_factor" in device_map
    assert "model.speech_bias_factor" in device_map


def test_resident_modules_stay_on_gpu_even_when_language_model_is_offloaded() -> None:
    model = _toy_on_meta()
    # Zero GPU budget forces the offloadable language model out of VRAM.
    device_map = build_offload_device_map(model, gpu_budget_mib=0, cpu_gib=1)

    resident_names = [
        n for n in _all_tensor_names(model) if n.startswith(RESIDENT_MODULE_PREFIXES)
    ]
    assert resident_names, "the toy model must contain resident tensors"
    assert all(_device_for(n, device_map) == 0 for n in resident_names)

    language_model_names = [n for n in _all_tensor_names(model) if "language_model" in n]
    assert language_model_names
    assert all(_device_for(n, device_map) != 0 for n in language_model_names), (
        "with zero budget the language model must be offloaded"
    )


def test_negative_gpu_budget_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_offload_device_map(_toy_on_meta(), gpu_budget_mib=-1, cpu_gib=1)


def test_default_tokenizer_revision_is_the_verified_cache_revision(monkeypatch) -> None:
    monkeypatch.delenv("VIBEVOICE_TOKENIZER_REVISION", raising=False)

    assert DEFAULT_TOKENIZER_REVISION == "8faed761d45a263340a0528343f099c05c9a4323"
    assert Settings.from_env().tokenizer_revision == DEFAULT_TOKENIZER_REVISION


def test_tokenizer_revision_can_be_overridden_and_must_be_a_commit(monkeypatch) -> None:
    from stickman_service.errors import SourcePolicyError

    monkeypatch.setenv("VIBEVOICE_TOKENIZER_REVISION", "a" * 40)
    assert Settings.from_env().tokenizer_revision == "a" * 40

    bad = Settings(tokenizer_revision="main")
    with pytest.raises(SourcePolicyError):
        bad.validate_static_policy()
