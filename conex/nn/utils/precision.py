"""
Author: Soroush Mohammaddeimi <soroushdeimi@gmail.com>


Mixed precision utilities for CoNeX.

This module provides utilities for mixed precision training, allowing networks
to use lower precision (float16/bfloat16) for certain operations to improve
performance while maintaining accuracy where needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Set, Callable, Any, TypeVar
import torch
from torch import Tensor

T = TypeVar("T", bound=torch.nn.Module)


class PrecisionMode(Enum):
    """Precision modes for neural computations."""

    FULL = "full"  # float32
    HALF = "half"  # float16
    BFLOAT16 = "bfloat16"  # bfloat16 (better dynamic range)
    MIXED = "mixed"  # automatic mixed precision


@dataclass
class PrecisionConfig:
    """Configuration for mixed precision training.

    Attributes:
        mode: The precision mode to use.
        autocast_dtype: The dtype to use for autocasting (float16 or bfloat16).
        scaler_enabled: Whether to use gradient scaler (for float16).
        ops_to_keep_fp32: Set of operation names to keep in float32.
        layers_to_keep_fp32: Set of layer indices to keep in float32.
        enabled: Whether mixed precision is enabled.
    """

    mode: PrecisionMode = PrecisionMode.FULL
    autocast_dtype: torch.dtype = torch.float16
    scaler_enabled: bool = True
    ops_to_keep_fp32: Set[str] = field(default_factory=lambda: {"softmax", "log_softmax", "layer_norm"})
    layers_to_keep_fp32: Set[int] = field(default_factory=set)
    enabled: bool = True

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.mode == PrecisionMode.BFLOAT16:
            self.autocast_dtype = torch.bfloat16
            # bfloat16 doesn't need gradient scaling due to better dynamic range
            self.scaler_enabled = False

    @property
    def dtype(self) -> torch.dtype:
        """Get the primary dtype for this precision mode."""
        if self.mode == PrecisionMode.FULL:
            return torch.float32
        elif self.mode == PrecisionMode.HALF:
            return torch.float16
        elif self.mode == PrecisionMode.BFLOAT16:
            return torch.bfloat16
        else:  # MIXED
            return self.autocast_dtype


class MixedPrecisionManager:
    """Manager for mixed precision operations in spiking neural networks.

    This class handles the complexity of mixed precision training, including:
    - Automatic casting of tensors to appropriate precision
    - Gradient scaling for numerical stability
    - Selective precision for sensitive operations

    Example:
        >>> config = PrecisionConfig(mode=PrecisionMode.MIXED)
        >>> manager = MixedPrecisionManager(config, torch.device("cuda"), torch.amp.GradScaler("cuda"))
        >>> with manager.autocast():
        ...     output = model(input)
        >>> manager.scale_loss(loss).backward()
        >>> manager.step(optimizer)
    """

    def __init__(
        self,
        config: Optional[PrecisionConfig] = None,
        device: Optional[torch.device] = None,
        scaler: Optional[torch.amp.GradScaler] = None,
    ) -> None:
        """Initialize the mixed precision manager.

        Args:
            config: Precision configuration. Defaults to full precision.
            device: Target device. cpu if not provided.
        """
        self.config = config or PrecisionConfig()
        self.device = device or torch.device("cpu")
        self._scaler= scaler or None

    @property
    def scaler(self) -> Optional[torch.amp.GradScaler]:
        """Get the gradient scaler if available."""
        return self._scaler

    def autocast(self) -> torch.amp.autocast:
        """Get an autocast context manager for mixed precision.

        Returns:
            Context manager for automatic mixed precision.
        """
        if self.config.mode == PrecisionMode.FULL or not self.config.enabled:
            # Return a no-op context manager for full precision
            return torch.amp.autocast(self.device.type, enabled=False)

        return torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.config.autocast_dtype,
            enabled=True,
        )

    def cast_tensor(
        self,
        tensor: Tensor,
        force_fp32: bool = False,
    ) -> Tensor:
        """Cast a tensor to the appropriate precision.

        Args:
            tensor: Input tensor to cast.
            force_fp32: If True, always cast to float32.

        Returns:
            Tensor cast to the appropriate dtype.
        """
        if force_fp32 or self.config.mode == PrecisionMode.FULL:
            return tensor.float()
        return tensor.to(self.config.dtype)

    def scale_loss(self, loss: Tensor) -> Tensor:
        """Scale loss for gradient stability in mixed precision.

        Args:
            loss: The loss tensor to scale.

        Returns:
            Scaled loss tensor.
        """
        if self._scaler is not None:
            return self._scaler.scale(loss)
        return loss

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Perform an optimizer step with gradient unscaling.

        Args:
            optimizer: The optimizer to step.
        """
        if self._scaler is not None:
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            optimizer.step()

    def unscale_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        """Unscale gradients before gradient clipping.

        Args:
            optimizer: The optimizer whose gradients to unscale.
        """
        if self._scaler is not None:
            self._scaler.unscale_(optimizer)


class PrecisionContext:
    """Context manager for precision-sensitive operations.

    Use this to temporarily switch precision for specific operations.

    Example:
        >>> with PrecisionContext(torch.float32):
        ...     # This will run in float32
        ...     sensitive_op(x)
    """

    def __init__(self, dtype: torch.dtype) -> None:
        """Initialize the precision context.

        Args:
            dtype: The dtype to use within this context.
        """
        self.dtype = dtype
        self._prev_dtype: Optional[torch.dtype] = None

    def __enter__(self) -> "PrecisionContext":
        """Enter the precision context."""
        self._prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.dtype)
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit the precision context."""
        if self._prev_dtype is not None:
            torch.set_default_dtype(self._prev_dtype)


def convert_to_precision(
    module: T,
    config: PrecisionConfig,
    keep_fp32_layers: Optional[Set[str]] = None,
) -> T:
    """Convert a module to the specified precision.

    Args:
        module: The module to convert.
        config: Precision configuration.
        keep_fp32_layers: Set of layer names to keep in float32.

    Returns:
        The module with converted precision.
    """
    if config.mode == PrecisionMode.FULL:
        return module.float()

    keep_fp32 = keep_fp32_layers or set()
    dtype = config.dtype

    def convert_layer(name: str, layer: torch.nn.Module) -> None:
        if name in keep_fp32:
            layer.float()
        else:
            layer.to(dtype)

    for name, layer in module.named_modules():
        convert_layer(name, layer)

    return module
