"""
Author: Soroush Mohammaddeimi <soroushdeimi@gmail.com>


Spatial Pooler - Input encoding layer for HTM.

Based on:
- Cui, Y., Ahmad, S., Hawkins, J. (2017). "The HTM Spatial Pooler—A Neocortical
  Algorithm for Online Sparse Distributed Coding". Frontiers in Computational Neuroscience.
- Hawkins, J., Ahmad, S., Cui, Y. (2017). "A Theory of How Columns in the Neocortex
  Enable Learning the Structure of the World". Frontiers in Neural Circuits.

The Spatial Pooler:
1. Encodes input as Sparse Distributed Representations (SDRs)
2. Maintains stable representations despite noise
3. Learns to represent semantic similarity through overlap
4. Implements competitive learning with homeostatic boosting

This is the input layer that feeds into Temporal Memory.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from pymonntorch import Behavior


class SpatialPoolerConfig(NamedTuple):
    """Configuration for Spatial Pooler."""
    
    input_size: int = 784  # e.g., 28x28 image
    n_columns: int = 2048
    potential_radius: int = 16
    potential_pct: float = 0.5
    global_inhibition: bool = True
    local_area_density: float = 0.02
    stimulus_threshold: int = 0
    syn_perm_inactive_dec: float = 0.008
    syn_perm_active_inc: float = 0.05
    syn_perm_connected: float = 0.1
    min_pct_overlap_duty_cycles: float = 0.001
    duty_cycle_period: int = 1000
    boost_strength: float = 0.0
    seed: int = -1


class SpatialPooler(Behavior):
    """
    Spatial Pooler implementing SDR encoding.
    
    The Spatial Pooler takes an input pattern and produces a sparse
    distributed representation (SDR) as output. Each column has a
    potential pool of input connections and competes to become active.
    
    Key properties:
    - Output sparsity: Fixed percentage of columns active (typically 2%)
    - Stability: Similar inputs produce similar outputs
    - Learning: Adapts to input statistics
    - Homeostasis: Boosting ensures all columns participate
    
    Args:
        config: SpatialPoolerConfig with algorithm parameters.
    """
    
    def __init__(self, config: SpatialPoolerConfig | None = None, **kwargs):
        config = config or SpatialPoolerConfig()
        params = {
            "input_size": config.input_size,
            "n_columns": config.n_columns,
            "potential_radius": config.potential_radius,
            "potential_pct": config.potential_pct,
            "global_inhibition": config.global_inhibition,
            "local_area_density": config.local_area_density,
            "stimulus_threshold": config.stimulus_threshold,
            "syn_perm_inactive_dec": config.syn_perm_inactive_dec,
            "syn_perm_active_inc": config.syn_perm_active_inc,
            "syn_perm_connected": config.syn_perm_connected,
            "min_pct_overlap_duty": config.min_pct_overlap_duty_cycles,
            "duty_cycle_period": config.duty_cycle_period,
            "boost_strength": config.boost_strength,
            "seed": config.seed,
        }
        params.update(kwargs)
        super().__init__(**params)
    
    def initialize(self, neurons):
        """Initialize Spatial Pooler state."""
        self.input_size = self.parameter("input_size", 784)
        self.n_columns = self.parameter("n_columns", 2048)
        self.potential_radius = self.parameter("potential_radius", 16)
        self.potential_pct = self.parameter("potential_pct", 0.5)
        self.global_inhibition = self.parameter("global_inhibition", True)
        self.local_area_density = self.parameter("local_area_density", 0.02)
        self.stimulus_threshold = self.parameter("stimulus_threshold", 0)
        self.perm_dec = self.parameter("syn_perm_inactive_dec", 0.008)
        self.perm_inc = self.parameter("syn_perm_active_inc", 0.05)
        self.connected_perm = self.parameter("syn_perm_connected", 0.1)
        self.min_duty_pct = self.parameter("min_pct_overlap_duty", 0.001)
        self.duty_period = self.parameter("duty_cycle_period", 1000)
        self.boost_strength = self.parameter("boost_strength", 0.0)
        seed = self.parameter("seed", -1)
        
        if seed >= 0:
            torch.manual_seed(seed)
        
        device = neurons.device
        dtype = neurons.network.def_dtype
        
        # Number of active columns
        self.n_active = max(1, int(self.n_columns * self.local_area_density))
        
        # Potential connections (which inputs each column can connect to)
        # Using sparse representation for large input spaces
        self._init_potential_connections(neurons)
        
        # Permanence values for each potential connection
        neurons.sp_permanences = torch.rand(
            self.n_columns, self.n_potential, device=device, dtype=dtype
        ) * 0.2 + self.connected_perm - 0.1
        
        # Boost factors for each column
        neurons.sp_boost_factors = torch.ones(
            self.n_columns, device=device, dtype=dtype
        )
        
        # Duty cycles (moving average of activity)
        neurons.sp_overlap_duty = torch.zeros(
            self.n_columns, device=device, dtype=dtype
        )
        neurons.sp_active_duty = torch.zeros(
            self.n_columns, device=device, dtype=dtype
        )
        
        # Output
        neurons.sp_active_columns = torch.zeros(
            self.n_columns, device=device, dtype=torch.bool
        )
        neurons.sp_overlap = torch.zeros(
            self.n_columns, device=device, dtype=dtype
        )
        
        # Iteration counter
        neurons.sp_iteration = 0
        
        neurons.spatial_pooler = self
    
    def _init_potential_connections(self, neurons) -> None:
        """Initialize potential connections for each column."""
        device = neurons.device
        
        # Number of potential inputs per column
        self.n_potential = max(1, int(self.input_size * self.potential_pct))
        
        # Each column connects to a random subset of inputs
        neurons.sp_potential_inputs = torch.zeros(
            self.n_columns, self.n_potential, device=device, dtype=torch.long
        )
        
        for col in range(self.n_columns):
            # Random sample of input indices
            perm = torch.randperm(self.input_size, device=device)[:self.n_potential]
            neurons.sp_potential_inputs[col] = perm
    
    def compute(
        self,
        neurons,
        input_sdr: torch.Tensor,
        learn: bool = True,
    ) -> torch.Tensor:
        """
        Compute Spatial Pooler output.
        
        Args:
            neurons: The neuron group.
            input_sdr: Input pattern (boolean or binary tensor).
            learn: Whether to apply learning.
            
        Returns:
            Active column indices.
        """
        neurons.sp_iteration += 1
        
        # Phase 1: Compute overlap
        overlap = self._compute_overlap(neurons, input_sdr)
        
        # Phase 2: Apply boosting
        boosted_overlap = overlap * neurons.sp_boost_factors
        
        # Phase 3: Inhibition (competition)
        active_columns = self._inhibit(neurons, boosted_overlap)
        neurons.sp_active_columns = active_columns
        
        # Phase 4: Learning
        if learn:
            self._learn(neurons, input_sdr)
            self._update_duty_cycles(neurons)
            self._update_boost_factors(neurons)
        
        return active_columns
    
    def _compute_overlap(
        self, neurons, input_sdr: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute overlap score for each column.
        
        Overlap = number of connected synapses with active inputs.
        """
        # Get connected synapses
        connected = neurons.sp_permanences >= self.connected_perm
        
        # Get input values at potential positions
        input_at_potential = input_sdr[neurons.sp_potential_inputs]
        
        # Count active connected synapses
        overlap = (connected & (input_at_potential > 0)).sum(dim=1).to(
            neurons.network.def_dtype
        )
        
        # Apply stimulus threshold
        overlap[overlap < self.stimulus_threshold] = 0
        
        neurons.sp_overlap = overlap
        return overlap
    
    def _inhibit(
        self, neurons, overlap: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply inhibition to select winning columns.
        
        With global inhibition: Select top-k columns globally.
        With local inhibition: Select top-k within each local region.
        """
        active = torch.zeros(self.n_columns, device=neurons.device, dtype=torch.bool)
        
        if self.global_inhibition:
            # Global winner-take-all
            if overlap.sum() > 0:
                _, top_indices = torch.topk(overlap, self.n_active)
                active[top_indices] = True
        else:
            # Local inhibition (simplified version)
            # In full implementation, would use topology
            _, top_indices = torch.topk(overlap, self.n_active)
            active[top_indices] = True
        
        return active
    
    def _learn(self, neurons, input_sdr: torch.Tensor) -> None:
        """
        Apply learning to active columns.
        
        For active columns:
        - Increase permanence for active inputs
        - Decrease permanence for inactive inputs
        """
        input_at_potential = input_sdr[neurons.sp_potential_inputs]
        
        for col_idx in neurons.sp_active_columns.nonzero(as_tuple=True)[0]:
            active_inputs = input_at_potential[col_idx] > 0
            
            # Increment active, decrement inactive
            neurons.sp_permanences[col_idx, active_inputs] += self.perm_inc
            neurons.sp_permanences[col_idx, ~active_inputs] -= self.perm_dec
        
        # Clip permanences
        neurons.sp_permanences.clamp_(0, 1)
    
    def _update_duty_cycles(self, neurons) -> None:
        """Update moving average duty cycles."""
        period = min(neurons.sp_iteration, self.duty_period)
        alpha = 1.0 / period
        
        # Update overlap duty (columns with non-zero overlap)
        overlap_active = neurons.sp_overlap > 0
        neurons.sp_overlap_duty = (
            (1 - alpha) * neurons.sp_overlap_duty + alpha * overlap_active.float()
        )
        
        # Update active duty
        neurons.sp_active_duty = (
            (1 - alpha) * neurons.sp_active_duty
            + alpha * neurons.sp_active_columns.float()
        )
    
    def _update_boost_factors(self, neurons) -> None:
        """Update boost factors based on duty cycles."""
        if self.boost_strength <= 0:
            return
        
        # Target duty cycle
        target_duty = self.local_area_density
        
        # Boost columns with below-target activity
        neurons.sp_boost_factors = torch.exp(
            self.boost_strength * (target_duty - neurons.sp_active_duty)
        )
    
    def forward(self, neurons):
        """
        Process input through Spatial Pooler.
        
        Expects neurons.sp_input to be set with input pattern.
        """
        if hasattr(neurons, "sp_input"):
            self.compute(neurons, neurons.sp_input, learn=True)
