"""Tests for the mixed precision utilities.

These helpers operate on torch modules and tensors rather than on CoNeX
behaviors, so the tests exercise them directly. Everything here runs on CPU,
with the CUDA-only paths covered by asserting the CPU fallbacks.
"""

import pytest
import torch

from conex import (
    MixedPrecisionManager,
    PrecisionConfig,
    PrecisionContext,
    PrecisionMode,
    convert_to_precision,
)


class TestPrecisionConfig:
    def test_full_precision_is_float32(self):
        assert PrecisionConfig(mode=PrecisionMode.FULL).dtype is torch.float32

    def test_half_precision_is_float16(self):
        assert PrecisionConfig(mode=PrecisionMode.HALF).dtype is torch.float16

    def test_bfloat16_sets_its_own_autocast_dtype(self):
        config = PrecisionConfig(mode=PrecisionMode.BFLOAT16)
        assert config.dtype is torch.bfloat16
        assert config.autocast_dtype is torch.bfloat16

    def test_bfloat16_disables_the_scaler(self):
        # bfloat16 has the dynamic range of float32, so loss scaling is moot
        assert PrecisionConfig(mode=PrecisionMode.BFLOAT16).scaler_enabled is False

    def test_mixed_precision_follows_the_autocast_dtype(self):
        config = PrecisionConfig(mode=PrecisionMode.MIXED, autocast_dtype=torch.bfloat16)
        assert config.dtype is torch.bfloat16

    def test_defaults_keep_sensitive_ops_in_fp32(self):
        assert "softmax" in PrecisionConfig().ops_to_keep_fp32


class TestMixedPrecisionManager:
    def test_defaults_to_full_precision_on_cpu(self):
        manager = MixedPrecisionManager(device=torch.device("cpu"))
        assert manager.config.mode is PrecisionMode.FULL
        assert manager.scaler is None

    def test_no_scaler_without_cuda(self):
        manager = MixedPrecisionManager(
            PrecisionConfig(mode=PrecisionMode.MIXED), device=torch.device("cpu")
        )
        assert manager.scaler is None

    def test_autocast_is_a_context_manager(self):
        manager = MixedPrecisionManager(device=torch.device("cpu"))
        with manager.autocast():
            result = torch.ones(4) * 2
        assert torch.allclose(result, torch.full((4,), 2.0))

    def test_cast_tensor_honours_full_precision(self):
        manager = MixedPrecisionManager(device=torch.device("cpu"))
        assert manager.cast_tensor(torch.ones(4, dtype=torch.float16)).dtype is torch.float32

    def test_cast_tensor_uses_the_configured_dtype(self):
        manager = MixedPrecisionManager(
            PrecisionConfig(mode=PrecisionMode.BFLOAT16), device=torch.device("cpu")
        )
        assert manager.cast_tensor(torch.ones(4)).dtype is torch.bfloat16

    def test_force_fp32_overrides_the_config(self):
        manager = MixedPrecisionManager(
            PrecisionConfig(mode=PrecisionMode.BFLOAT16), device=torch.device("cpu")
        )
        assert manager.cast_tensor(torch.ones(4), force_fp32=True).dtype is torch.float32

    def test_scale_loss_is_a_passthrough_without_a_scaler(self):
        manager = MixedPrecisionManager(device=torch.device("cpu"))
        loss = torch.tensor(2.5)
        assert manager.scale_loss(loss) is loss

    def test_step_falls_through_to_the_optimizer(self):
        manager = MixedPrecisionManager(device=torch.device("cpu"))
        weight = torch.nn.Parameter(torch.ones(2))
        optimizer = torch.optim.SGD([weight], lr=0.1)
        weight.grad = torch.ones(2)

        manager.step(optimizer)

        assert torch.allclose(weight.detach(), torch.full((2,), 0.9))

    def test_unscale_is_a_noop_without_a_scaler(self):
        manager = MixedPrecisionManager(device=torch.device("cpu"))
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(2))], lr=0.1)
        manager.unscale_gradients(optimizer)


class TestConvertToPrecision:
    def test_full_precision_returns_a_float32_module(self):
        module = torch.nn.Linear(4, 2).half()
        converted = convert_to_precision(module, PrecisionConfig(mode=PrecisionMode.FULL))
        assert converted.weight.dtype is torch.float32

    def test_module_is_cast_to_the_configured_dtype(self):
        module = torch.nn.Linear(4, 2)
        converted = convert_to_precision(
            module, PrecisionConfig(mode=PrecisionMode.BFLOAT16)
        )
        assert converted.weight.dtype is torch.bfloat16

    def test_named_layers_can_be_kept_in_fp32(self):
        module = torch.nn.Sequential()
        module.add_module("keep", torch.nn.Linear(4, 2))
        module.add_module("cast", torch.nn.Linear(2, 2))

        convert_to_precision(
            module, PrecisionConfig(mode=PrecisionMode.BFLOAT16), keep_fp32_layers={"keep"}
        )

        assert module.keep.weight.dtype is torch.float32
        assert module.cast.weight.dtype is torch.bfloat16


class TestPrecisionContext:
    def test_default_dtype_is_restored_on_exit(self):
        original = torch.get_default_dtype()
        with PrecisionContext(torch.float64):
            assert torch.get_default_dtype() is torch.float64
        assert torch.get_default_dtype() is original

    def test_tensors_created_inside_use_the_context_dtype(self):
        with PrecisionContext(torch.float64):
            assert torch.zeros(2).dtype is torch.float64

    def test_dtype_is_restored_after_an_exception(self):
        original = torch.get_default_dtype()
        with pytest.raises(RuntimeError):
            with PrecisionContext(torch.float64):
                raise RuntimeError("boom")
        assert torch.get_default_dtype() is original
