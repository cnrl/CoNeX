"""
Author: Soroush Mohammaddeimi <soroushdeimi@gmail.com>


Active Dendrite Mechanisms.

Based on:
- Major, G., Larkum, M.E., Bhalla, U.S. (2013). "Active Dendrites". 
  Annual Review of Neuroscience.
- Hawkins, J., Ahmad, S. (2016). "Why Neurons Have Thousands of Synapses, 
  A Theory of Sequence Memory in Neocortex". Frontiers in Neural Circuits.
- Larkum, M.E., Nevian, T., et al. (2009). "Synaptic Integration in Tuft 
  Dendrites of Layer 5 Pyramidal Neurons". Science.

Active dendrites are fundamental to the Thousand Brains Theory because:
1. They allow context-dependent processing (predictions via distal dendrites)
2. They implement non-linear integration (dendritic spikes)
3. They provide independent computational subunits (dendritic segments)

This module implements biologically realistic active dendrite models
including NMDA spikes, calcium dynamics, and segment-based learning.
"""

from __future__ import annotations

from typing import Literal
from dataclasses import dataclass

import torch
from pymonntorch import Behavior


@dataclass
class DendriticSegmentConfig:
    """Configuration for a dendritic segment."""
    
    n_synapses_per_segment: int = 20
    activation_threshold: int = 10
    learning_threshold: int = 8
    permanence_increment: float = 0.1
    permanence_decrement: float = 0.02
    initial_permanence: float = 0.21
    connected_permanence: float = 0.2
    max_new_synapses: int = 5


class DendriticSegment(Behavior):
    """
    Dendritic Segment with HTM-style learning.
    
    A dendritic segment is a portion of dendrite that can recognize
    a specific pattern of input through its synapses. When enough
    synapses on a segment are active, the segment becomes active
    and can:
    - Depolarize the cell (making it predictive)
    - Cause a dendritic spike (non-linear amplification)
    
    Learning:
    - Active segments strengthen synapses from active inputs
    - Inactive segments can grow new synapses to active inputs
    - Synapses decay when inputs are inactive during segment activation
    
    Args:
        n_segments: Number of dendritic segments per neuron.
        n_cells: Number of cells (neurons) with segments.
        config: DendriticSegmentConfig with learning parameters.
    """
    
    def __init__(
        self,
        n_segments: int = 128,
        n_cells: int = 256,
        *,
        config: DendriticSegmentConfig | None = None,
        **kwargs,
    ):
        config = config or DendriticSegmentConfig()
        params = {
            "n_segments": n_segments,
            "n_cells": n_cells,
            "n_synapses_per_segment": config.n_synapses_per_segment,
            "activation_threshold": config.activation_threshold,
            "learning_threshold": config.learning_threshold,
            "permanence_increment": config.permanence_increment,
            "permanence_decrement": config.permanence_decrement,
            "initial_permanence": config.initial_permanence,
            "connected_permanence": config.connected_permanence,
            "max_new_synapses": config.max_new_synapses,
        }
        params.update(kwargs)
        super().__init__(**params)
    
    def initialize(self, neurons):
        """Initialize dendritic segment structures."""
        self.n_segments = self.parameter("n_segments", 128)
        self.n_cells = self.parameter("n_cells", 256)
        self.n_synapses = self.parameter("n_synapses_per_segment", 20)
        self.activation_threshold = self.parameter("activation_threshold", 10)
        self.learning_threshold = self.parameter("learning_threshold", 8)
        self.perm_inc = self.parameter("permanence_increment", 0.1)
        self.perm_dec = self.parameter("permanence_decrement", 0.02)
        self.initial_perm = self.parameter("initial_permanence", 0.21)
        self.connected_perm = self.parameter("connected_permanence", 0.2)
        self.max_new_synapses = self.parameter("max_new_synapses", 5)
        
        device = neurons.device
        dtype = neurons.network.def_dtype
        
        total_segments = self.n_cells * self.n_segments
        
        # Synapse permanences: values in [0, 1] indicate connection strength
        # Synapse is "connected" if permanence >= connected_permanence
        neurons.segment_permanences = torch.zeros(
            total_segments, self.n_synapses, device=device, dtype=dtype
        )
        
        # Synapse target indices: which presynaptic cell each synapse connects to
        # -1 indicates no connection
        neurons.segment_targets = torch.full(
            (total_segments, self.n_synapses), -1, device=device, dtype=torch.long
        )
        
        # Segment activation (number of active connected synapses)
        neurons.segment_activation = torch.zeros(
            total_segments, device=device, dtype=dtype
        )
        
        # Segment active state (activation >= threshold)
        neurons.segment_active = torch.zeros(
            total_segments, device=device, dtype=torch.bool
        )
        
        # Cell depolarization from active segments
        neurons.depolarization = torch.zeros(
            self.n_cells, device=device, dtype=dtype
        )
        
        # Track which segments matched (for learning)
        neurons.matching_segments = torch.zeros(
            total_segments, device=device, dtype=torch.bool
        )
        
        neurons.dendritic_segment = self
    
    def compute_segment_activation(
        self, neurons, presynaptic_activity: torch.Tensor
    ) -> None:
        """
        Compute activation of each segment based on presynaptic activity.
        
        A segment's activation is the count of connected synapses whose
        presynaptic cells are active.
        
        Args:
            neurons: The neuron group.
            presynaptic_activity: Boolean tensor of active presynaptic cells.
        """
        # Get connected synapses (permanence >= threshold)
        connected = neurons.segment_permanences >= self.connected_perm
        
        # Check if targets are valid and active
        valid_targets = neurons.segment_targets >= 0
        
        # Get activity of target cells
        # Handle -1 indices by clamping
        safe_targets = neurons.segment_targets.clamp(min=0)
        target_active = presynaptic_activity[safe_targets]
        
        # Only count synapses that are connected, valid, and have active targets
        active_synapses = connected & valid_targets & target_active
        
        # Sum active synapses per segment
        neurons.segment_activation = active_synapses.sum(dim=1).to(
            neurons.network.def_dtype
        )
        
        # Determine active and matching segments
        neurons.segment_active = neurons.segment_activation >= self.activation_threshold
        neurons.matching_segments = neurons.segment_activation >= self.learning_threshold
    
    def compute_depolarization(self, neurons) -> None:
        """
        Compute cell depolarization from active dendritic segments.
        
        Each cell's depolarization is determined by its most active segment.
        This implements the predictive state in HTM.
        """
        # Reshape to (n_cells, n_segments_per_cell)
        activation_per_cell = neurons.segment_activation.view(
            self.n_cells, self.n_segments
        )
        
        # Max activation across segments for each cell
        max_activation, _ = activation_per_cell.max(dim=1)
        
        # Normalize to [0, 1] based on threshold
        neurons.depolarization = (max_activation / self.activation_threshold).clamp(max=1.0)
    
    def learn_on_segments(
        self,
        neurons,
        active_cells: torch.Tensor,
        winner_cells: torch.Tensor,
        presynaptic_activity: torch.Tensor,
    ) -> None:
        """
        Apply learning to dendritic segments.
        
        Learning rules:
        1. Strengthen synapses on active segments from active inputs
        2. Weaken synapses on active segments from inactive inputs
        3. Grow new synapses on matching segments to active inputs
        
        Args:
            neurons: The neuron group.
            active_cells: Boolean tensor of cells that became active.
            winner_cells: Boolean tensor of cells chosen as "winners".
            presynaptic_activity: Boolean tensor of active presynaptic cells.
        """
        # Find segments belonging to winner cells that matched
        cell_indices = torch.arange(self.n_cells, device=neurons.device)
        segment_to_cell = cell_indices.repeat_interleave(self.n_segments)
        
        winner_segments = winner_cells[segment_to_cell]
        segments_to_learn = neurons.matching_segments & winner_segments
        
        if not segments_to_learn.any():
            return
        
        # Get valid targets and their activity
        valid_targets = neurons.segment_targets >= 0
        safe_targets = neurons.segment_targets.clamp(min=0)
        target_active = presynaptic_activity[safe_targets]
        
        # Strengthen active synapses, weaken inactive ones
        active_synapse_mask = segments_to_learn.unsqueeze(1) & valid_targets & target_active
        inactive_synapse_mask = segments_to_learn.unsqueeze(1) & valid_targets & ~target_active
        
        neurons.segment_permanences[active_synapse_mask] += self.perm_inc
        neurons.segment_permanences[inactive_synapse_mask] -= self.perm_dec
        
        # Clip permanences to [0, 1]
        neurons.segment_permanences.clamp_(0, 1)
        
        # Grow new synapses (simplified version)
        self._grow_synapses(neurons, segments_to_learn, presynaptic_activity)
    
    def _grow_synapses(
        self,
        neurons,
        segments_to_grow: torch.Tensor,
        presynaptic_activity: torch.Tensor,
    ) -> None:
        """
        Grow new synapses on segments that need them.
        
        New synapses are grown to active presynaptic cells that don't
        already have connections.
        """
        # Find segments with room for new synapses
        unconnected = neurons.segment_targets < 0
        room_for_synapses = unconnected.any(dim=1)
        
        grow_mask = segments_to_grow & room_for_synapses
        segment_indices = grow_mask.nonzero(as_tuple=True)[0]
        
        # Get active presynaptic cell indices
        active_cells = presynaptic_activity.nonzero(as_tuple=True)[0]
        
        if len(active_cells) == 0:
            return
        
        for seg_idx in segment_indices:
            # Find empty synapse slots
            empty_slots = (neurons.segment_targets[seg_idx] < 0).nonzero(as_tuple=True)[0]
            n_to_grow = min(len(empty_slots), self.max_new_synapses, len(active_cells))
            
            if n_to_grow == 0:
                continue
            
            # Randomly select presynaptic cells
            perm = torch.randperm(len(active_cells), device=neurons.device)[:n_to_grow]
            new_targets = active_cells[perm]
            
            # Assign new synapses
            slots_to_use = empty_slots[:n_to_grow]
            neurons.segment_targets[seg_idx, slots_to_use] = new_targets
            neurons.segment_permanences[seg_idx, slots_to_use] = self.initial_perm
    
    def forward(self, neurons):
        """
        Update dendritic segment state.
        
        Computes segment activation and cell depolarization based on
        afferent synapse activity.
        """
        # Get presynaptic activity (from connected synapses)
        if hasattr(neurons, "afferent_activity"):
            self.compute_segment_activation(neurons, neurons.afferent_activity)
            self.compute_depolarization(neurons)


class ActiveDendriteComputation(Behavior):
    """
    Active Dendrite Computation with NMDA-like nonlinearity.
    
    Implements the non-linear integration properties of active dendrites:
    - Subthreshold: Linear summation of inputs
    - Superthreshold: Non-linear amplification (dendritic spike)
    
    The NMDA receptor provides voltage-dependent amplification:
        g_NMDA(V) = g_max * s / (1 + [Mg2+]/K * exp(-V/V_half))
    
    This creates a sigmoidal activation function that amplifies
    coincident inputs (many synapses active together).
    
    Args:
        nmda_threshold: Threshold for NMDA nonlinearity.
        nmda_gain: Amplification factor for superthreshold inputs.
        integration_tau: Time constant for dendritic integration.
        calcium_tau: Time constant for calcium dynamics.
        backprop_strength: Strength of backpropagating action potentials.
    """
    
    def __init__(
        self,
        *,
        nmda_threshold: float = 0.5,
        nmda_gain: float = 2.0,
        integration_tau: float = 10.0,
        calcium_tau: float = 20.0,
        backprop_strength: float = 0.3,
        **kwargs,
    ):
        super().__init__(
            nmda_threshold=nmda_threshold,
            nmda_gain=nmda_gain,
            integration_tau=integration_tau,
            calcium_tau=calcium_tau,
            backprop_strength=backprop_strength,
            **kwargs,
        )
    
    def initialize(self, neurons):
        """Initialize active dendrite computation."""
        self.nmda_threshold = self.parameter("nmda_threshold", 0.5)
        self.nmda_gain = self.parameter("nmda_gain", 2.0)
        self.integration_tau = self.parameter("integration_tau", 10.0)
        self.calcium_tau = self.parameter("calcium_tau", 20.0)
        self.backprop_strength = self.parameter("backprop_strength", 0.3)
        
        device = neurons.device
        dtype = neurons.network.def_dtype
        
        # Dendritic potential for each dendrite type
        neurons.proximal_potential = neurons.vector(0.0)
        neurons.distal_potential = neurons.vector(0.0)
        neurons.apical_potential = neurons.vector(0.0)
        
        # Calcium concentration (for plateau potentials and learning)
        neurons.dendritic_calcium = neurons.vector(0.0)
        
        # Dendritic spike indicator
        neurons.dendritic_spike = neurons.vector(dtype=torch.bool)
        
        neurons.active_dendrite = self
    
    def nmda_nonlinearity(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply NMDA-like sigmoidal nonlinearity.
        
        Below threshold: output ≈ input
        Above threshold: output amplified by gain
        
        Uses soft threshold: f(x) = x + gain * sigmoid((x - threshold) * steepness)
        """
        steepness = 10.0  # Controls sharpness of transition
        amplification = self.nmda_gain * torch.sigmoid(
            (x - self.nmda_threshold) * steepness
        )
        return x + x * amplification
    
    def compute_dendritic_integration(
        self,
        neurons,
        proximal_input: torch.Tensor,
        distal_input: torch.Tensor | None = None,
        apical_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute integrated dendritic input with active properties.
        
        Proximal dendrites: Fast integration, linear (feedforward input)
        Distal dendrites: NMDA nonlinearity (contextual/prediction)
        Apical dendrites: Plateau potentials (feedback/attention)
        
        Args:
            neurons: The neuron group.
            proximal_input: Feedforward input current.
            distal_input: Contextual/lateral input.
            apical_input: Feedback/top-down input.
            
        Returns:
            Total dendritic current to soma.
        """
        dt = neurons.network.dt
        
        # Proximal: linear integration with fast time constant
        neurons.proximal_potential += (
            (-neurons.proximal_potential + proximal_input)
            * dt / (self.integration_tau / 2)
        )
        
        # Distal: NMDA nonlinearity (context-dependent gating)
        if distal_input is not None:
            neurons.distal_potential += (
                (-neurons.distal_potential + distal_input)
                * dt / self.integration_tau
            )
        
        distal_contribution = self.nmda_nonlinearity(neurons.distal_potential)
        
        # Apical: plateau potential dynamics (attention/feedback)
        if apical_input is not None:
            neurons.apical_potential += (
                (-neurons.apical_potential + apical_input)
                * dt / (self.integration_tau * 2)
            )
        
        apical_contribution = self.nmda_nonlinearity(neurons.apical_potential)
        
        # Detect dendritic spikes
        neurons.dendritic_spike = (
            (distal_contribution > self.nmda_threshold * 2) |
            (apical_contribution > self.nmda_threshold * 2)
        )
        
        # Update calcium (for learning and plateau potentials)
        calcium_influx = neurons.dendritic_spike.float() * 0.5
        neurons.dendritic_calcium += (
            (-neurons.dendritic_calcium + calcium_influx)
            * dt / self.calcium_tau
        )
        
        # Total current: proximal is direct, distal/apical modulate
        total_current = neurons.proximal_potential * (
            1.0 + distal_contribution + apical_contribution
        )
        
        return total_current
    
    def apply_backprop(self, neurons, somatic_spike: torch.Tensor) -> None:
        """
        Apply backpropagating action potential effects.
        
        When the soma fires, the action potential backpropagates into
        dendrites, which is important for:
        - Coincidence detection with dendritic input
        - Learning (dendritic calcium influx)
        
        Args:
            neurons: The neuron group.
            somatic_spike: Boolean tensor of cells that fired.
        """
        backprop_effect = somatic_spike.float() * self.backprop_strength
        
        # Boost calcium on backpropagation
        neurons.dendritic_calcium += backprop_effect * 0.3
        
        # Can trigger dendritic spikes if combined with synaptic input
        combined = neurons.distal_potential + backprop_effect
        neurons.dendritic_spike |= combined > self.nmda_threshold * 1.5
    
    def forward(self, neurons):
        """
        Update dendritic integration each timestep.
        """
        # Get inputs from afferent synapses
        proximal = getattr(neurons, "I_proximal", neurons.vector(0.0))
        distal = getattr(neurons, "I_distal", None)
        apical = getattr(neurons, "I_apical", None)
        
        # Compute integrated current
        neurons.I = self.compute_dendritic_integration(
            neurons, proximal, distal, apical
        )
        
        # Apply backprop if neuron spiked
        if hasattr(neurons, "spikes"):
            self.apply_backprop(neurons, neurons.spikes)


class ContextualPrediction(Behavior):
    """
    Contextual Prediction using distal dendrites.
    
    In the Thousand Brains Theory, neurons use distal dendrites to
    receive context (from other neurons in the same layer) that predicts
    when they should become active. This implements:
    
    1. Predictive state: Depolarized but not firing
    2. Prediction verification: Predicted cells fire first
    3. Prediction error: Unpredicted cells burst
    
    Args:
        depolarization_threshold: Threshold for predictive state.
        prediction_window: Time window for prediction validity.
        burst_threshold: Threshold for burst firing on surprise.
    """
    
    def __init__(
        self,
        *,
        depolarization_threshold: float = 0.5,
        prediction_window: int = 10,
        burst_threshold: float = 0.3,
        **kwargs,
    ):
        super().__init__(
            depolarization_threshold=depolarization_threshold,
            prediction_window=prediction_window,
            burst_threshold=burst_threshold,
            **kwargs,
        )
    
    def initialize(self, neurons):
        """Initialize contextual prediction state."""
        self.depol_threshold = self.parameter("depolarization_threshold", 0.5)
        self.pred_window = self.parameter("prediction_window", 10)
        self.burst_threshold = self.parameter("burst_threshold", 0.3)
        
        device = neurons.device
        dtype = neurons.network.def_dtype
        
        # Predictive state (depolarized by context)
        neurons.predictive = neurons.vector(dtype=torch.bool)
        
        # Prediction age (timesteps since prediction started)
        neurons.prediction_age = neurons.vector(dtype=torch.long)
        
        # Burst indicator (unpredicted activation)
        neurons.bursting = neurons.vector(dtype=torch.bool)
        
        neurons.contextual_prediction = self
    
    def update_predictions(
        self, neurons, context_input: torch.Tensor
    ) -> None:
        """
        Update predictive state based on contextual input.
        
        Neurons become predictive when their distal dendrites receive
        sufficient input from the current context.
        
        Args:
            neurons: The neuron group.
            context_input: Contextual input to distal dendrites.
        """
        # Get depolarization from dendritic segments if available
        if hasattr(neurons, "depolarization"):
            depol = neurons.depolarization
        else:
            depol = context_input
        
        # Neurons become predictive if depolarization exceeds threshold
        newly_predictive = depol >= self.depol_threshold
        
        # Update prediction age
        neurons.prediction_age[newly_predictive] = 0
        neurons.prediction_age[neurons.predictive & ~newly_predictive] += 1
        
        # Expire old predictions
        expired = neurons.prediction_age >= self.pred_window
        neurons.predictive = (neurons.predictive | newly_predictive) & ~expired
    
    def process_activation(
        self, neurons, feedforward_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process feedforward activation considering predictions.
        
        - Predicted cells: Normal activation
        - Unpredicted cells: Burst activation (prediction error signal)
        
        Args:
            neurons: The neuron group.
            feedforward_input: Feedforward input driving activation.
            
        Returns:
            Tuple of (activation, burst) tensors.
        """
        # Which cells would activate from feedforward input alone
        would_activate = feedforward_input >= self.burst_threshold
        
        # Predicted cells activate normally
        predicted_activation = would_activate & neurons.predictive
        
        # Unpredicted cells burst (stronger activation, learning signal)
        bursting = would_activate & ~neurons.predictive
        neurons.bursting = bursting
        
        # Burst cells get amplified activation
        activation = feedforward_input.clone()
        activation[bursting] *= 1.5  # Burst amplification
        
        return activation, bursting
    
    def forward(self, neurons):
        """
        Update contextual prediction state.
        """
        # Get context input (distal dendrite current)
        context = getattr(neurons, "I_distal", None)
        if context is not None:
            self.update_predictions(neurons, context)
