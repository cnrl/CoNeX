"""
Author: Soroush Mohammaddeimi <soroushdeimi@gmail.com>


Predictive synapses for top-down and bottom-up connections.

Implements specialized synapse behaviors for predictive coding:
- Top-down prediction synapses (feedback connections)
- Bottom-up error synapses (feedforward connections)
- Lateral prediction synapses (same-level context)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch import Tensor
import torch.nn.functional as F
from pymonntorch import Behavior, SynapseGroup

from ...neurons.numenta.predictive_coding import PredictiveCodingConfig


class PredictiveSynapseInit(Behavior):
    """Initialize synapses for predictive coding connections.
    
    Sets up weight matrices and state variables for predictive
    coding synaptic connections.
    
    Args:
        config: PredictiveCodingConfig.
        connection_type: Type of connection ("feedforward", "feedback", "lateral").
        init_scale: Scale for weight initialization.
    """
    
    def __init__(
        self,
        config: Optional[PredictiveCodingConfig] = None,
        connection_type: str = "feedforward",
        init_scale: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            config=config or PredictiveCodingConfig(),
            connection_type=connection_type,
            init_scale=init_scale,
            **kwargs,
        )
    
    def initialize(self, synapses: SynapseGroup) -> None:
        """Initialize predictive synapse state."""
        self.config = self.parameter("config")
        self.connection_type = self.parameter("connection_type", "feedforward")
        init_scale = self.parameter("init_scale", 0.1)
        
        src_size = synapses.src.size
        dst_size = synapses.dst.size
        device = synapses.network.device
        dtype = synapses.network.def_dtype
        
        # Weight matrix
        synapses.weights = torch.randn(
            src_size, dst_size, dtype=dtype, device=device
        ) * init_scale
        
        # Eligibility traces for learning
        synapses.eligibility = torch.zeros(
            src_size, dst_size, dtype=dtype, device=device
        )
        
        # Connection type flag
        synapses.is_feedforward = self.connection_type == "feedforward"
        synapses.is_feedback = self.connection_type == "feedback"
        synapses.is_lateral = self.connection_type == "lateral"
        
        # Prediction-specific state
        if synapses.is_feedback:
            synapses.prediction_output = torch.zeros(
                dst_size, dtype=dtype, device=device
            )
        
        if synapses.is_feedforward:
            synapses.error_output = torch.zeros(
                dst_size, dtype=dtype, device=device
            )

            # FeedforwardErrorSynapse deposits into dst.bottom_up_error and
            # otherwise falls back to dst.I, which only exists when a dendrite
            # behavior built it. Make sure the destination has the buffer.
            if not hasattr(synapses.dst, "bottom_up_error"):
                synapses.dst.bottom_up_error = torch.zeros(
                    dst_size, dtype=dtype, device=device
                )


class FeedbackPredictionSynapse(Behavior):
    """Synapse that transmits predictions from higher to lower layers.
    
    Implements top-down feedback connections in predictive coding.
    Higher layers send their activity as predictions about what
    lower layers should be representing.
    
    Args:
        config: PredictiveCodingConfig.
        delay: Synaptic delay in timesteps.
    """
    
    def __init__(
        self,
        config: Optional[PredictiveCodingConfig] = None,
        delay: int = 1,
        **kwargs,
    ):
        super().__init__(
            config=config or PredictiveCodingConfig(),
            delay=delay,
            **kwargs,
        )
    
    def initialize(self, synapses: SynapseGroup) -> None:
        """Initialize feedback synapse."""
        self.config = self.parameter("config")
        self.delay = self.parameter("delay", 1)
        
        dst_size = synapses.dst.size
        device = synapses.network.device
        dtype = synapses.network.def_dtype
        
        # Delayed prediction buffer
        if self.delay > 1:
            synapses.prediction_buffer = torch.zeros(
                self.delay, dst_size, dtype=dtype, device=device
            )
            synapses.buffer_idx = 0
    
    def forward(self, synapses: SynapseGroup) -> None:
        """Transmit prediction from source to destination."""
        # Get source activity (higher layer)
        src_activity = synapses.src.activity
        
        # Compute prediction via weight matrix
        prediction = torch.matmul(synapses.weights.T, src_activity)
        
        # Apply delay if specified
        if self.delay > 1:
            idx = synapses.buffer_idx % self.delay
            old_prediction = synapses.prediction_buffer[idx].clone()
            synapses.prediction_buffer[idx] = prediction
            synapses.buffer_idx += 1
            prediction = old_prediction
        
        synapses.prediction_output = prediction
        
        # Set prediction in destination layer
        if hasattr(synapses.dst, "top_down_prediction"):
            synapses.dst.top_down_prediction = prediction
        if hasattr(synapses.dst, "prediction"):
            synapses.dst.prediction = prediction


class FeedforwardErrorSynapse(Behavior):
    """Synapse that transmits prediction errors from lower to higher layers.
    
    Implements bottom-up feedforward connections in predictive coding.
    Lower layers send prediction errors (weighted by precision) to
    higher layers to update their representations.
    
    Args:
        config: PredictiveCodingConfig.
        precision_weighted: Whether to weight errors by precision.
    """
    
    def __init__(
        self,
        config: Optional[PredictiveCodingConfig] = None,
        precision_weighted: bool = True,
        **kwargs,
    ):
        super().__init__(
            config=config or PredictiveCodingConfig(),
            precision_weighted=precision_weighted,
            **kwargs,
        )
    
    def initialize(self, synapses: SynapseGroup) -> None:
        """Initialize feedforward error synapse."""
        self.config = self.parameter("config")
        self.precision_weighted = self.parameter("precision_weighted", True)
    
    def forward(self, synapses: SynapseGroup) -> None:
        """Transmit prediction error from source to destination."""
        # Get prediction error from source (lower layer)
        error = getattr(
            synapses.src, "prediction_error",
            synapses.src.activity
        )
        
        # Apply precision weighting
        if self.precision_weighted:
            precision = getattr(
                synapses.src, "precision",
                torch.ones_like(error)
            )
            error = error * precision
        
        # Transmit through weights
        output = torch.matmul(synapses.weights.T, error)
        
        synapses.error_output = output
        
        # Add to destination layer's input
        if hasattr(synapses.dst, "bottom_up_error"):
            synapses.dst.bottom_up_error = output
        else:
            # Add to general input
            synapses.dst.I += output


class LateralPredictionSynapse(Behavior):
    """Lateral connections for contextual prediction.
    
    Same-level lateral connections that provide contextual
    information for predictions within a layer.
    
    Args:
        config: PredictiveCodingConfig.
        inhibitory: Whether connection is inhibitory.
    """
    
    def __init__(
        self,
        config: Optional[PredictiveCodingConfig] = None,
        inhibitory: bool = False,
        **kwargs,
    ):
        super().__init__(
            config=config or PredictiveCodingConfig(),
            inhibitory=inhibitory,
            **kwargs,
        )
    
    def initialize(self, synapses: SynapseGroup) -> None:
        """Initialize lateral synapse."""
        self.config = self.parameter("config")
        self.inhibitory = self.parameter("inhibitory", False)
    
    def forward(self, synapses: SynapseGroup) -> None:
        """Transmit lateral context."""
        src_activity = synapses.src.activity
        
        # Lateral connection
        lateral_input = torch.matmul(synapses.weights.T, src_activity)
        
        # Apply sign for inhibition
        if self.inhibitory:
            lateral_input = -torch.abs(lateral_input)
        
        # Add to context
        if hasattr(synapses.dst, "lateral_context"):
            synapses.dst.lateral_context = lateral_input
        else:
            synapses.dst.I += lateral_input


class PredictiveSynapseLearning(Behavior):
    """Learning rule for predictive coding synapses.
    
    Updates synaptic weights to minimize prediction error.
    Different rules for feedforward vs feedback connections.
    
    Args:
        config: PredictiveCodingConfig.
        learning_type: Type of learning ("error_driven", "hebbian", "combined").
    """
    
    def __init__(
        self,
        config: Optional[PredictiveCodingConfig] = None,
        learning_type: str = "error_driven",
        **kwargs,
    ):
        super().__init__(
            config=config or PredictiveCodingConfig(),
            learning_type=learning_type,
            **kwargs,
        )
    
    def initialize(self, synapses: SynapseGroup) -> None:
        """Initialize learning state."""
        self.config = self.parameter("config")
        self.learning_type = self.parameter("learning_type", "error_driven")
    
    def forward(self, synapses: SynapseGroup) -> None:
        """Update weights based on prediction errors."""
        cfg = self.config
        dt = synapses.network.dt
        lr = cfg.learning_rate * dt
        
        src_activity = synapses.src.activity
        dst_error = getattr(
            synapses.dst, "prediction_error",
            torch.zeros(synapses.dst.size)
        )
        
        if self.learning_type == "error_driven":
            # Gradient descent on prediction error
            delta_w = torch.outer(src_activity, dst_error)
            
        elif self.learning_type == "hebbian":
            # Hebbian learning
            dst_activity = synapses.dst.activity
            delta_w = torch.outer(src_activity, dst_activity)
            
        else:  # combined
            # Error-driven + Hebbian
            dst_activity = synapses.dst.activity
            delta_w = (
                torch.outer(src_activity, dst_error) +
                0.1 * torch.outer(src_activity, dst_activity)
            )
        
        # Update eligibility trace
        synapses.eligibility = 0.9 * synapses.eligibility + 0.1 * delta_w
        
        # Apply update with eligibility
        synapses.weights = synapses.weights + lr * synapses.eligibility


class PredictiveConnection:
    """Factory for creating predictive coding connections.
    
    Creates properly configured feedforward or feedback connections
    between layers in a predictive coding hierarchy.
    
    Example:
        >>> conn = PredictiveConnection.feedback(higher_layer, lower_layer)
        >>> syn = Synapsis(conn, behaviors=conn.behaviors)
    """
    
    @staticmethod
    def feedback(
        src_layer,
        dst_layer,
        config: Optional[PredictiveCodingConfig] = None,
        with_learning: bool = True,
    ) -> dict:
        """Create feedback (top-down prediction) connection.
        
        Args:
            src_layer: Source layer (higher in hierarchy).
            dst_layer: Destination layer (lower in hierarchy).
            config: Configuration.
            with_learning: Include learning behavior.
        
        Returns:
            Configuration dict with behaviors.
        """
        config = config or PredictiveCodingConfig()
        behaviors = {
            100: PredictiveSynapseInit(config=config, connection_type="feedback"),
            110: FeedbackPredictionSynapse(config=config),
        }
        
        if with_learning:
            behaviors[120] = PredictiveSynapseLearning(
                config=config, learning_type="error_driven"
            )
        
        return {
            "src": src_layer,
            "dst": dst_layer,
            "behaviors": behaviors,
        }
    
    @staticmethod
    def feedforward(
        src_layer,
        dst_layer,
        config: Optional[PredictiveCodingConfig] = None,
        with_learning: bool = True,
    ) -> dict:
        """Create feedforward (bottom-up error) connection.
        
        Args:
            src_layer: Source layer (lower in hierarchy).
            dst_layer: Destination layer (higher in hierarchy).
            config: Configuration.
            with_learning: Include learning behavior.
        
        Returns:
            Configuration dict with behaviors.
        """
        config = config or PredictiveCodingConfig()
        behaviors = {
            100: PredictiveSynapseInit(config=config, connection_type="feedforward"),
            110: FeedforwardErrorSynapse(config=config),
        }
        
        if with_learning:
            behaviors[120] = PredictiveSynapseLearning(
                config=config, learning_type="error_driven"
            )
        
        return {
            "src": src_layer,
            "dst": dst_layer,
            "behaviors": behaviors,
        }
    
    @staticmethod
    def lateral(
        layer,
        config: Optional[PredictiveCodingConfig] = None,
        inhibitory: bool = False,
    ) -> dict:
        """Create lateral (same-level context) connection.
        
        Args:
            layer: The layer to add lateral connections to.
            config: Configuration.
            inhibitory: Whether connections are inhibitory.
        
        Returns:
            Configuration dict with behaviors.
        """
        config = config or PredictiveCodingConfig()
        behaviors = {
            100: PredictiveSynapseInit(config=config, connection_type="lateral"),
            110: LateralPredictionSynapse(config=config, inhibitory=inhibitory),
        }
        
        return {
            "src": layer,
            "dst": layer,
            "behaviors": behaviors,
        }
