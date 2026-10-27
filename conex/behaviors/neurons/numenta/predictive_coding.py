"""
Author: Soroush Mohammaddeimi <soroushdeimi@gmail.com>


Predictive Coding Engine for CoNeX.

Implements predictive processing based on Karl Friston's Free Energy Principle
and hierarchical predictive coding theory. Key concepts:

1. **Prediction Units**: Neurons that generate predictions about expected input
2. **Error Units**: Neurons that compute mismatch between prediction and reality
3. **Precision Weighting**: Uncertainty estimation that modulates error influence
4. **Hierarchical Processing**: Only prediction errors propagate up the hierarchy

References:
- Rao & Ballard (1999): Predictive coding in visual cortex
- Friston (2005): Free Energy Principle
- Bastos et al. (2012): Canonical microcircuits for predictive coding
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Callable, Dict, Any, List
from enum import Enum
import torch
from torch import Tensor
import torch.nn.functional as F
from pymonntorch import Behavior, NeuronGroup


class PredictionType(Enum):
    """Types of predictions in the hierarchy."""
    
    FEEDFORWARD = "feedforward"  # Bottom-up sensory
    FEEDBACK = "feedback"  # Top-down prediction
    LATERAL = "lateral"  # Same-level context


@dataclass
class PredictiveCodingConfig:
    """Configuration for predictive coding computations.
    
    Attributes:
        prediction_tau: Time constant for prediction updates (ms).
        error_tau: Time constant for error unit dynamics (ms).
        precision_tau: Time constant for precision learning (ms).
        learning_rate: Base learning rate for prediction weights.
        precision_learning_rate: Learning rate for precision estimation.
        min_precision: Minimum precision value (prevents division by zero).
        max_precision: Maximum precision value (prevents instability).
        error_nonlinearity: Nonlinearity applied to prediction errors.
        use_sparse_errors: Only propagate errors above threshold.
        error_threshold: Threshold for sparse error propagation.
        temporal_smoothing: Exponential smoothing for predictions.
    """
    
    prediction_tau: float = 20.0
    error_tau: float = 10.0
    precision_tau: float = 100.0
    learning_rate: float = 0.01
    precision_learning_rate: float = 0.001
    min_precision: float = 0.01
    max_precision: float = 100.0
    error_nonlinearity: str = "relu"  # "relu", "linear", "tanh"
    use_sparse_errors: bool = True
    error_threshold: float = 0.1
    temporal_smoothing: float = 0.9


class PredictionUnit(Behavior):
    """Neurons that generate predictions about expected input.
    
    These units receive top-down input from higher hierarchical levels
    and generate predictions about what the bottom-up input should be.
    
    Args:
        config: PredictiveCodingConfig for this unit.
        prediction_dim: Dimensionality of predictions.
        context_dim: Dimensionality of contextual input (optional).
        use_temporal: Whether to use temporal predictions.
        
    Attributes:
        prediction: Current prediction tensor.
        prediction_history: History of recent predictions.
        context: Current contextual information.
    """
    
    def __init__(
        self,
        config: Optional[PredictiveCodingConfig] = None,
        prediction_dim: Optional[int] = None,
        context_dim: Optional[int] = None,
        use_temporal: bool = True,
        **kwargs,
    ):
        super().__init__(
            config=config or PredictiveCodingConfig(),
            prediction_dim=prediction_dim,
            context_dim=context_dim,
            use_temporal=use_temporal,
            **kwargs,
        )
    
    def initialize(self, neurons: NeuronGroup) -> None:
        """Initialize prediction unit state."""
        self.config = self.parameter("config")
        self.use_temporal = self.parameter("use_temporal", True)
        
        # Determine dimensions
        n_neurons = neurons.size
        pred_dim = self.parameter("prediction_dim", n_neurons)
        ctx_dim = self.parameter("context_dim", 0)
        
        device = neurons.network.device
        dtype = neurons.network.def_dtype
        
        # Activity is the level's own representation. It is set by the caller
        # or by an incoming synapse, so allocate it if nothing has yet.
        if not hasattr(neurons, "activity"):
            neurons.activity = torch.zeros(n_neurons, dtype=dtype, device=device)

        # Initialize prediction state
        neurons.prediction = torch.zeros(n_neurons, dtype=dtype, device=device)
        neurons.prediction_target = torch.zeros(n_neurons, dtype=dtype, device=device)
        
        # Temporal prediction buffer
        if self.use_temporal:
            neurons.prediction_history = torch.zeros(
                10, n_neurons, dtype=dtype, device=device
            )
            neurons.history_idx = 0
        
        # Context integration
        if ctx_dim > 0:
            neurons.context = torch.zeros(ctx_dim, dtype=dtype, device=device)
            neurons.context_weights = torch.randn(
                ctx_dim, n_neurons, dtype=dtype, device=device
            ) * 0.1
        
        # Prediction weights (learned)
        neurons.prediction_weights = torch.eye(
            n_neurons, dtype=dtype, device=device
        )
        
        # Learning state
        neurons.prediction_lr = self.config.learning_rate
    
    def forward(self, neurons: NeuronGroup) -> None:
        """Update predictions based on top-down input."""
        cfg = self.config
        dt = neurons.network.dt
        tau = cfg.prediction_tau
        alpha = dt / tau
        
        # Get top-down input if available
        top_down = getattr(neurons, "top_down_input", None)
        if top_down is not None:
            # Generate prediction from top-down context
            new_prediction = torch.matmul(
                neurons.prediction_weights.T, top_down
            )
        else:
            # Self-prediction (autoregressive)
            new_prediction = torch.matmul(
                neurons.prediction_weights.T, neurons.activity
            )
        
        # Temporal smoothing
        neurons.prediction = (
            cfg.temporal_smoothing * neurons.prediction +
            (1 - cfg.temporal_smoothing) * new_prediction
        )
        
        # Store in history for temporal predictions
        if self.use_temporal:
            idx = neurons.history_idx % 10
            neurons.prediction_history[idx] = neurons.prediction.clone()
            neurons.history_idx += 1


class ErrorUnit(Behavior):
    """Neurons that compute prediction errors.
    
    Error units compare predictions with actual sensory input and
    compute the mismatch. Only significant errors are propagated
    up the hierarchy for efficient processing.
    
    Args:
        config: PredictiveCodingConfig for error computation.
        precision_weighted: Whether to weight errors by precision.
        
    Attributes:
        error: Current prediction error.
        error_magnitude: Absolute error magnitude.
        precision: Current precision estimate.
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
    
    def initialize(self, neurons: NeuronGroup) -> None:
        """Initialize error unit state."""
        self.config = self.parameter("config")
        self.precision_weighted = self.parameter("precision_weighted", True)
        
        n_neurons = neurons.size
        device = neurons.network.device
        dtype = neurons.network.def_dtype
        
        if not hasattr(neurons, "activity"):
            neurons.activity = torch.zeros(n_neurons, dtype=dtype, device=device)

        # Error state
        neurons.prediction_error = torch.zeros(n_neurons, dtype=dtype, device=device)
        neurons.error_magnitude = torch.zeros(n_neurons, dtype=dtype, device=device)
        neurons.signed_error = torch.zeros(n_neurons, dtype=dtype, device=device)

        # Where a feedforward synapse from the level below deposits its error.
        # Without it FeedforwardErrorSynapse falls back to neurons.I, which
        # only exists when a dendrite behavior has built it.
        neurons.bottom_up_error = torch.zeros(n_neurons, dtype=dtype, device=device)
        
        # Precision (inverse variance) - initialized to 1
        neurons.precision = torch.ones(n_neurons, dtype=dtype, device=device)
        neurons.precision_estimate = torch.ones(n_neurons, dtype=dtype, device=device)
        
        # Error history for precision learning
        neurons.error_variance = torch.ones(n_neurons, dtype=dtype, device=device)
        neurons.error_ema = torch.zeros(n_neurons, dtype=dtype, device=device)
    
    def forward(self, neurons: NeuronGroup) -> None:
        """Compute prediction error."""
        cfg = self.config
        
        # Get prediction and actual input
        prediction = getattr(neurons, "prediction", torch.zeros_like(neurons.activity))
        actual = neurons.activity
        
        # Compute signed error
        neurons.signed_error = actual - prediction
        
        # Apply nonlinearity
        if cfg.error_nonlinearity == "relu":
            neurons.prediction_error = F.relu(neurons.signed_error)
        elif cfg.error_nonlinearity == "tanh":
            neurons.prediction_error = torch.tanh(neurons.signed_error)
        else:  # linear
            neurons.prediction_error = neurons.signed_error
        
        # Compute error magnitude
        neurons.error_magnitude = torch.abs(neurons.signed_error)
        
        # Precision weighting
        if self.precision_weighted:
            neurons.prediction_error = neurons.prediction_error * neurons.precision
        
        # Sparse error propagation
        if cfg.use_sparse_errors:
            mask = neurons.error_magnitude > cfg.error_threshold
            neurons.prediction_error = neurons.prediction_error * mask.float()
        
        # Update precision estimate (online variance estimation)
        self._update_precision(neurons)
    
    def _update_precision(self, neurons: NeuronGroup) -> None:
        """Update precision estimate based on error history."""
        cfg = self.config
        dt = neurons.network.dt
        alpha = dt / cfg.precision_tau
        
        # Exponential moving average of squared error
        error_sq = neurons.signed_error ** 2
        neurons.error_variance = (
            (1 - alpha) * neurons.error_variance + alpha * error_sq
        )
        
        # Precision = inverse variance (clamped)
        neurons.precision_estimate = 1.0 / (neurons.error_variance + 1e-8)
        neurons.precision = torch.clamp(
            neurons.precision_estimate,
            cfg.min_precision,
            cfg.max_precision,
        )


class PrecisionWeighting(Behavior):
    """Adaptive precision weighting for prediction errors.
    
    Implements precision (inverse variance) estimation that modulates
    the influence of prediction errors. High precision = reliable signal,
    low precision = noisy/uncertain signal.
    
    This is a key component of the Free Energy Principle - the brain
    optimizes precision-weighted prediction errors.
    
    Args:
        config: PredictiveCodingConfig for precision parameters.
        learnable: Whether precision is learned or fixed.
        attention_modulated: Whether attention can modulate precision.
    """
    
    def __init__(
        self,
        config: Optional[PredictiveCodingConfig] = None,
        learnable: bool = True,
        attention_modulated: bool = True,
        **kwargs,
    ):
        super().__init__(
            config=config or PredictiveCodingConfig(),
            learnable=learnable,
            attention_modulated=attention_modulated,
            **kwargs,
        )
    
    def initialize(self, neurons: NeuronGroup) -> None:
        """Initialize precision weighting state."""
        self.config = self.parameter("config")
        self.learnable = self.parameter("learnable", True)
        self.attention_modulated = self.parameter("attention_modulated", True)
        
        n_neurons = neurons.size
        device = neurons.network.device
        dtype = neurons.network.def_dtype
        
        # Precision components
        neurons.sensory_precision = torch.ones(n_neurons, dtype=dtype, device=device)
        neurons.prior_precision = torch.ones(n_neurons, dtype=dtype, device=device)
        neurons.attention_gain = torch.ones(n_neurons, dtype=dtype, device=device)
        
        # Combined precision
        neurons.effective_precision = torch.ones(n_neurons, dtype=dtype, device=device)
        
        # Learning state
        if self.learnable:
            neurons.precision_grad = torch.zeros(n_neurons, dtype=dtype, device=device)
    
    def forward(self, neurons: NeuronGroup) -> None:
        """Update precision weights."""
        cfg = self.config
        
        # Compute effective precision
        neurons.effective_precision = (
            neurons.sensory_precision * neurons.prior_precision
        )
        
        # Attention modulation (multiplicative gain)
        if self.attention_modulated:
            attention = getattr(neurons, "attention", None)
            if attention is not None:
                neurons.attention_gain = 1.0 + attention
                neurons.effective_precision = (
                    neurons.effective_precision * neurons.attention_gain
                )
        
        # Clamp to valid range
        neurons.effective_precision = torch.clamp(
            neurons.effective_precision,
            cfg.min_precision,
            cfg.max_precision,
        )
        
        # Update precision in error computation
        if hasattr(neurons, "precision"):
            neurons.precision = neurons.effective_precision
        
        # Learn precision if enabled
        if self.learnable:
            self._learn_precision(neurons)
    
    def _learn_precision(self, neurons: NeuronGroup) -> None:
        """Learn precision from prediction error statistics."""
        cfg = self.config
        dt = neurons.network.dt
        lr = cfg.precision_learning_rate * dt
        
        if not hasattr(neurons, "signed_error"):
            return
        
        # Precision gradient: minimize free energy
        # F = 0.5 * (precision * error^2 - log(precision))
        # dF/d_precision = 0.5 * (error^2 - 1/precision)
        error_sq = neurons.signed_error ** 2
        neurons.precision_grad = 0.5 * (
            error_sq - 1.0 / (neurons.sensory_precision + 1e-8)
        )
        
        # Update sensory precision (gradient descent on free energy)
        neurons.sensory_precision = neurons.sensory_precision - lr * neurons.precision_grad
        neurons.sensory_precision = torch.clamp(
            neurons.sensory_precision,
            cfg.min_precision,
            cfg.max_precision,
        )


class TopDownPrediction(Behavior):
    """Generates top-down predictions from higher hierarchical levels.
    
    Higher cortical areas send predictions down to lower areas,
    which then compute the mismatch with actual sensory input.
    
    Args:
        config: PredictiveCodingConfig.
        source_layer: Name of the source layer for predictions.
        transform: Optional transformation applied to predictions.
    """
    
    def __init__(
        self,
        config: Optional[PredictiveCodingConfig] = None,
        source_layer: Optional[str] = None,
        transform: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            config=config or PredictiveCodingConfig(),
            source_layer=source_layer,
            transform=transform,
            **kwargs,
        )
    
    def initialize(self, neurons: NeuronGroup) -> None:
        """Initialize top-down prediction state."""
        self.config = self.parameter("config")
        self.source_layer = self.parameter("source_layer", None)
        self.transform = self.parameter("transform", None)
        
        n_neurons = neurons.size
        device = neurons.network.device
        dtype = neurons.network.def_dtype
        
        # Top-down prediction state
        neurons.top_down_input = torch.zeros(n_neurons, dtype=dtype, device=device)
        neurons.top_down_prediction = torch.zeros(n_neurons, dtype=dtype, device=device)
        
        # Prediction weight matrix (learned via error minimization)
        neurons.top_down_weights = torch.eye(
            n_neurons, dtype=dtype, device=device
        ) * 0.5
    
    def forward(self, neurons: NeuronGroup) -> None:
        """Generate top-down prediction."""
        cfg = self.config
        
        # Get source layer activity
        if self.source_layer is not None:
            # Look up source layer in network
            source = neurons.network.get(self.source_layer, None)
            if source is not None:
                source_activity = getattr(source, "activity", None)
                if source_activity is not None:
                    neurons.top_down_input = source_activity
        
        # Generate prediction via learned weights
        neurons.top_down_prediction = torch.matmul(
            neurons.top_down_weights.T, neurons.top_down_input
        )
        
        # Apply transformation if specified
        if self.transform == "sigmoid":
            neurons.top_down_prediction = torch.sigmoid(neurons.top_down_prediction)
        elif self.transform == "tanh":
            neurons.top_down_prediction = torch.tanh(neurons.top_down_prediction)
        
        # Set as prediction target for error computation
        neurons.prediction = neurons.top_down_prediction


class PredictiveCodingLearning(Behavior):
    """Learning rule for predictive coding networks.
    
    Updates prediction weights to minimize prediction error.
    This implements gradient descent on the free energy functional.
    
    Args:
        config: PredictiveCodingConfig.
        weight_decay: L2 regularization strength.
        use_hebbian: Use Hebbian component in addition to error gradient.
    """
    
    def __init__(
        self,
        config: Optional[PredictiveCodingConfig] = None,
        weight_decay: float = 0.0001,
        use_hebbian: bool = True,
        **kwargs,
    ):
        super().__init__(
            config=config or PredictiveCodingConfig(),
            weight_decay=weight_decay,
            use_hebbian=use_hebbian,
            **kwargs,
        )
    
    def initialize(self, neurons: NeuronGroup) -> None:
        """Initialize learning state."""
        self.config = self.parameter("config")
        self.weight_decay = self.parameter("weight_decay", 0.0001)
        self.use_hebbian = self.parameter("use_hebbian", True)
    
    def forward(self, neurons: NeuronGroup) -> None:
        """Update prediction weights based on errors."""
        cfg = self.config
        dt = neurons.network.dt
        lr = cfg.learning_rate * dt
        
        if not hasattr(neurons, "prediction_error"):
            return
        
        error = neurons.prediction_error
        
        # Update prediction weights.
        # PredictionUnit computes prediction = prediction_weights.T @ source,
        # where source is the top-down input when one is present. Descending
        # 0.5*||activity - prediction||^2 therefore gives outer(source, error).
        if hasattr(neurons, "prediction_weights"):
            activity = neurons.activity
            source = getattr(neurons, "top_down_input", None)
            if source is None:
                source = activity

            delta_w = torch.outer(source, error)

            # Hebbian component (activity correlation)
            if self.use_hebbian:
                hebbian = torch.outer(activity, activity) * 0.1
                delta_w = delta_w + hebbian
            
            # Weight decay
            delta_w = delta_w - self.weight_decay * neurons.prediction_weights
            
            # Apply update
            neurons.prediction_weights = neurons.prediction_weights + lr * delta_w
        
        # Update top-down weights.
        # TopDownPrediction computes prediction = top_down_weights.T @ top_down,
        # so descending 0.5*||activity - prediction||^2 gives outer(top_down, error).
        if hasattr(neurons, "top_down_weights"):
            top_down = neurons.top_down_input
            delta_td = torch.outer(top_down, error)
            neurons.top_down_weights = (
                neurons.top_down_weights + lr * delta_td
            )


class FreeEnergyMinimization(Behavior):
    """Tracks the Gaussian surprise of a predictive coding level.

    The Free Energy Principle states that biological systems minimize a
    quantity that bounds surprise. This behavior computes the Gaussian form
    of that bound for a single level:

        F = 0.5 * sum(precision * error^2) - 0.5 * sum(log(precision))

    Up to an additive constant this is the negative log likelihood of the
    prediction error under a Gaussian with the given precision, i.e. surprise
    itself. It is not the full variational decomposition into an accuracy term
    and a KL complexity term, because there is no prior over the precision to
    diverge from. `accuracy_term` and `complexity_term` name the two halves of
    the expression above for convenience, not those quantities in the
    variational sense.

    One consequence is worth knowing: with the error at zero the second term
    is all that is left, and since the precision update is a maximum-likelihood
    estimate with fixed point `precision = 1 / error^2`, precision keeps
    climbing to `max_precision` and F keeps falling. A prior over the precision
    would give it a finite resting point.

    Args:
        config: PredictiveCodingConfig.
        track_components: Whether to track the two halves separately.
    """
    
    def __init__(
        self,
        config: Optional[PredictiveCodingConfig] = None,
        track_components: bool = True,
        **kwargs,
    ):
        super().__init__(
            config=config or PredictiveCodingConfig(),
            track_components=track_components,
            **kwargs,
        )
    
    def initialize(self, neurons: NeuronGroup) -> None:
        """Initialize free energy tracking."""
        self.config = self.parameter("config")
        self.track_components = self.parameter("track_components", True)
        
        device = neurons.network.device
        dtype = neurons.network.def_dtype
        
        # Free energy state
        neurons.free_energy = torch.tensor(0.0, dtype=dtype, device=device)
        neurons.free_energy_history = []
        
        if self.track_components:
            neurons.accuracy_term = torch.tensor(0.0, dtype=dtype, device=device)
            neurons.complexity_term = torch.tensor(0.0, dtype=dtype, device=device)
    
    def forward(self, neurons: NeuronGroup) -> None:
        """Compute free energy."""
        if not hasattr(neurons, "signed_error"):
            return
        
        error = neurons.signed_error
        precision = getattr(neurons, "precision", torch.ones_like(error))
        
        # F = 0.5 * sum(precision * error^2) - 0.5 * sum(log(precision))

        # Precision-weighted squared error: the data-fit half.
        accuracy = 0.5 * torch.sum(precision * error ** 2)

        # Log-normaliser of the Gaussian likelihood. Not a KL divergence --
        # nothing here places a prior over the precision to diverge from.
        complexity = -0.5 * torch.sum(torch.log(precision + 1e-8))
        
        # Total free energy
        neurons.free_energy = accuracy + complexity
        
        if self.track_components:
            neurons.accuracy_term = accuracy
            neurons.complexity_term = complexity
        
        # Track history
        neurons.free_energy_history.append(neurons.free_energy.item())
        
        # Keep only recent history
        if len(neurons.free_energy_history) > 1000:
            neurons.free_energy_history = neurons.free_energy_history[-1000:]


class HierarchicalPredictiveCoding(Behavior):
    """Orchestrates predictive coding across hierarchical layers.
    
    Manages the flow of predictions (top-down) and errors (bottom-up)
    across multiple layers of a predictive coding hierarchy.
    
    Args:
        config: PredictiveCodingConfig.
        n_levels: Number of hierarchical levels.
        level_sizes: Size of each level (list).
    """
    
    def __init__(
        self,
        config: Optional[PredictiveCodingConfig] = None,
        n_levels: int = 3,
        level_sizes: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(
            config=config or PredictiveCodingConfig(),
            n_levels=n_levels,
            level_sizes=level_sizes,
            **kwargs,
        )
    
    def initialize(self, neurons: NeuronGroup) -> None:
        """Initialize hierarchical predictive coding."""
        self.config = self.parameter("config")
        self.n_levels = self.parameter("n_levels", 3)
        level_sizes = self.parameter("level_sizes", None)
        
        n_neurons = neurons.size
        device = neurons.network.device
        dtype = neurons.network.def_dtype
        
        # Default level sizes (decreasing)
        if level_sizes is None:
            level_sizes = [n_neurons // (2 ** i) for i in range(self.n_levels)]
        
        self.level_sizes = level_sizes
        
        # State for each level
        neurons.level_predictions = []
        neurons.level_errors = []
        neurons.level_precisions = []
        
        for size in level_sizes:
            neurons.level_predictions.append(
                torch.zeros(size, dtype=dtype, device=device)
            )
            neurons.level_errors.append(
                torch.zeros(size, dtype=dtype, device=device)
            )
            neurons.level_precisions.append(
                torch.ones(size, dtype=dtype, device=device)
            )
        
        # Inter-level weights
        neurons.feedforward_weights = []
        neurons.feedback_weights = []
        
        for i in range(len(level_sizes) - 1):
            # Bottom-up (feedforward) weights
            neurons.feedforward_weights.append(
                torch.randn(
                    level_sizes[i], level_sizes[i + 1],
                    dtype=dtype, device=device
                ) * 0.1
            )
            # Top-down (feedback) weights
            neurons.feedback_weights.append(
                torch.randn(
                    level_sizes[i + 1], level_sizes[i],
                    dtype=dtype, device=device
                ) * 0.1
            )
    
    def forward(self, neurons: NeuronGroup) -> None:
        """Process hierarchy: predictions down, errors up."""
        cfg = self.config
        
        # Get sensory input (bottom level)
        sensory = neurons.activity[:self.level_sizes[0]]
        
        # Bottom-up pass: compute errors at each level
        current_input = sensory
        for i in range(len(self.level_sizes) - 1):
            # Get prediction from level above
            prediction = neurons.level_predictions[i]
            
            # Compute error
            error = current_input - prediction
            neurons.level_errors[i] = error
            
            # Propagate error up (precision-weighted)
            precision = neurons.level_precisions[i]
            weighted_error = error * precision
            
            # Transform to next level
            current_input = torch.matmul(
                neurons.feedforward_weights[i].T, weighted_error
            )
        
        # Top-down pass: generate predictions
        for i in range(len(self.level_sizes) - 2, -1, -1):
            # Get activity from level above
            if i < len(self.level_sizes) - 1:
                above = neurons.level_predictions[i + 1]
            else:
                above = current_input
            
            # Generate prediction for level below
            prediction = torch.matmul(
                neurons.feedback_weights[i].T, above
            )
            neurons.level_predictions[i] = prediction


# Convenience function to create predictive coding layer
def create_predictive_layer(
    size: int,
    config: Optional[PredictiveCodingConfig] = None,
    with_precision: bool = True,
    with_learning: bool = True,
    with_free_energy: bool = True,
) -> Dict[int, Behavior]:
    """Create a complete predictive coding layer with all components.
    
    Args:
        size: Number of neurons in the layer.
        config: Configuration for predictive coding.
        with_precision: Include precision weighting.
        with_learning: Include learning behavior.
        with_free_energy: Include free energy tracking.
    
    Returns:
        Dictionary of behaviors for the layer.
    """
    config = config or PredictiveCodingConfig()
    behaviors = {}
    priority = 100
    
    # Prediction units
    behaviors[priority] = PredictionUnit(config=config)
    priority += 10
    
    # Error units
    behaviors[priority] = ErrorUnit(config=config, precision_weighted=with_precision)
    priority += 10
    
    # Precision weighting
    if with_precision:
        behaviors[priority] = PrecisionWeighting(config=config)
        priority += 10
    
    # Learning
    if with_learning:
        behaviors[priority] = PredictiveCodingLearning(config=config)
        priority += 10
    
    # Free energy tracking
    if with_free_energy:
        behaviors[priority] = FreeEnergyMinimization(config=config)
        priority += 10
    
    return behaviors
