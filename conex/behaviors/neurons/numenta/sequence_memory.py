"""
Author: Soroush Mohammaddeimi <soroushdeimi@gmail.com>


Sequence Memory and Temporal Memory.

Based on:
- Hawkins, J., Ahmad, S. (2016). "Why Neurons Have Thousands of Synapses, 
  A Theory of Sequence Memory in Neocortex". Frontiers in Neural Circuits.
- Ahmad, S., Hawkins, J. (2016). "How do neurons operate on sparse distributed
  representations? A mathematical theory of sparsity, neurons and active dendrites".
  arXiv:1601.00720.
- Cui, Y., Ahmad, S., Hawkins, J. (2017). "The HTM Spatial Pooler—A Neocortical
  Algorithm for Online Sparse Distributed Coding". Frontiers in Computational Neuroscience.

Temporal Memory is the core algorithm that enables:
1. Learning sequences of patterns
2. Predicting future inputs
3. Detecting anomalies (prediction errors)

Key concepts:
- Minicolumns: Groups of cells that share feedforward receptive fields
- Cells within a column: Represent the same input in different contexts
- Predictive state: Cells depolarized by context predict next input
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from pymonntorch import Behavior


class TemporalMemoryConfig(NamedTuple):
    """Configuration for Temporal Memory."""
    
    n_columns: int = 2048
    cells_per_column: int = 32
    activation_threshold: int = 13
    learning_threshold: int = 10
    initial_permanence: float = 0.21
    connected_permanence: float = 0.5
    permanence_increment: float = 0.1
    permanence_decrement: float = 0.1
    predicted_decrement: float = 0.0
    max_segments_per_cell: int = 255
    max_synapses_per_segment: int = 255
    max_new_synapses: int = 20


class TemporalMemory(Behavior):
    """
    Temporal Memory - HTM's sequence learning algorithm.
    
    Temporal Memory learns and recalls sequences of sparse distributed
    patterns. Each column has multiple cells that learn to recognize
    specific sequential contexts. When a pattern matches predictions,
    only predicted cells activate. When a pattern is unexpected,
    all cells in active columns activate (burst).
    
    Algorithm phases per timestep:
    1. Activate: Determine which cells become active
    2. Depolarize: Determine which cells become predictive
    3. Learn: Update segment connections
    
    Args:
        config: TemporalMemoryConfig with algorithm parameters.
    """
    
    def __init__(self, config: TemporalMemoryConfig | None = None, **kwargs):
        config = config or TemporalMemoryConfig()
        params = {
            "n_columns": config.n_columns,
            "cells_per_column": config.cells_per_column,
            "activation_threshold": config.activation_threshold,
            "learning_threshold": config.learning_threshold,
            "initial_permanence": config.initial_permanence,
            "connected_permanence": config.connected_permanence,
            "permanence_increment": config.permanence_increment,
            "permanence_decrement": config.permanence_decrement,
            "predicted_decrement": config.predicted_decrement,
            "max_segments": config.max_segments_per_cell,
            "max_synapses": config.max_synapses_per_segment,
            "max_new_synapses": config.max_new_synapses,
        }
        params.update(kwargs)
        super().__init__(**params)
    
    def initialize(self, neurons):
        """Initialize Temporal Memory state."""
        self.n_columns = self.parameter("n_columns", 2048)
        self.cells_per_column = self.parameter("cells_per_column", 32)
        self.n_cells = self.n_columns * self.cells_per_column
        
        self.activation_threshold = self.parameter("activation_threshold", 13)
        self.learning_threshold = self.parameter("learning_threshold", 10)
        self.initial_perm = self.parameter("initial_permanence", 0.21)
        self.connected_perm = self.parameter("connected_permanence", 0.5)
        self.perm_inc = self.parameter("permanence_increment", 0.1)
        self.perm_dec = self.parameter("permanence_decrement", 0.1)
        self.predicted_dec = self.parameter("predicted_decrement", 0.0)
        self.max_segments = self.parameter("max_segments", 255)
        self.max_synapses = self.parameter("max_synapses", 255)
        self.max_new_synapses = self.parameter("max_new_synapses", 20)
        
        device = neurons.device
        dtype = neurons.network.def_dtype
        
        # Cell states
        neurons.tm_active_cells = torch.zeros(self.n_cells, device=device, dtype=torch.bool)
        neurons.tm_predictive_cells = torch.zeros(self.n_cells, device=device, dtype=torch.bool)
        neurons.tm_winner_cells = torch.zeros(self.n_cells, device=device, dtype=torch.bool)
        
        # Column states
        neurons.tm_active_columns = torch.zeros(self.n_columns, device=device, dtype=torch.bool)
        neurons.tm_bursting_columns = torch.zeros(self.n_columns, device=device, dtype=torch.bool)
        
        # Previous states (for learning)
        neurons.tm_prev_active_cells = torch.zeros(self.n_cells, device=device, dtype=torch.bool)
        neurons.tm_prev_winner_cells = torch.zeros(self.n_cells, device=device, dtype=torch.bool)
        
        # Segments: stored as sparse structure
        # Using lists for flexibility (can convert to tensors for GPU)
        neurons.tm_segments = SegmentStorage(
            self.n_cells,
            self.max_segments,
            self.max_synapses,
            device=device,
            dtype=dtype,
        )
        
        # Metrics
        neurons.tm_anomaly_score = torch.tensor(0.0, device=device, dtype=dtype)
        neurons.tm_prediction_accuracy = torch.tensor(0.0, device=device, dtype=dtype)
        
        neurons.temporal_memory = self
    
    def compute(
        self,
        neurons,
        active_columns: torch.Tensor,
        learn: bool = True,
    ) -> None:
        """
        Run one timestep of the Temporal Memory algorithm.
        
        Args:
            neurons: The neuron group.
            active_columns: Boolean tensor of active columns from spatial pooler.
            learn: Whether to apply learning.
        """
        # Store previous state
        neurons.tm_prev_active_cells = neurons.tm_active_cells.clone()
        neurons.tm_prev_winner_cells = neurons.tm_winner_cells.clone()
        
        # Update active columns
        neurons.tm_active_columns = active_columns
        
        # Phase 1: Activate cells
        self._activate_cells(neurons)
        
        # Phase 2: Compute predictions for next timestep
        self._compute_predictions(neurons)
        
        # Phase 3: Learning
        if learn:
            self._learn(neurons)
        
        # Compute metrics
        self._compute_anomaly(neurons)
    
    def _activate_cells(self, neurons) -> None:
        """
        Determine which cells become active.
        
        For each active column:
        - If any cells were predicted, activate only those cells
        - Otherwise, activate all cells (burst)
        """
        device = neurons.device
        
        # Reset active and winner cells
        neurons.tm_active_cells.zero_()
        neurons.tm_winner_cells.zero_()
        neurons.tm_bursting_columns.zero_()
        
        # Get cell indices for each column
        for col_idx in neurons.tm_active_columns.nonzero(as_tuple=True)[0]:
            col_start = col_idx * self.cells_per_column
            col_end = col_start + self.cells_per_column
            col_cells = slice(col_start, col_end)
            
            # Check for predicted cells in this column
            predicted_in_col = neurons.tm_predictive_cells[col_cells]
            
            if predicted_in_col.any():
                # Activate only predicted cells
                neurons.tm_active_cells[col_cells] = predicted_in_col
                
                # Winner is the predicted cell with best matching segment
                winner_idx = self._get_best_matching_cell(
                    neurons, col_start, col_end, neurons.tm_prev_active_cells
                )
                if winner_idx >= 0:
                    neurons.tm_winner_cells[winner_idx] = True
            else:
                # Burst: activate all cells in column
                neurons.tm_active_cells[col_start:col_end] = True
                neurons.tm_bursting_columns[col_idx] = True
                
                # Winner is the cell with best matching segment (or random if none)
                winner_idx = self._get_best_matching_cell(
                    neurons, col_start, col_end, neurons.tm_prev_active_cells
                )
                if winner_idx < 0:
                    # No matching segment, pick least used cell
                    winner_idx = self._get_least_used_cell(neurons, col_start, col_end)
                neurons.tm_winner_cells[winner_idx] = True
    
    def _get_best_matching_cell(
        self,
        neurons,
        col_start: int,
        col_end: int,
        active_cells: torch.Tensor,
    ) -> int:
        """Find cell with best matching segment to active cells."""
        best_cell = -1
        best_score = 0
        
        for cell_idx in range(col_start, col_end):
            score = neurons.tm_segments.get_best_segment_score(
                cell_idx, active_cells, self.learning_threshold
            )
            if score > best_score:
                best_score = score
                best_cell = cell_idx
        
        return best_cell
    
    def _get_least_used_cell(
        self, neurons, col_start: int, col_end: int
    ) -> int:
        """Find cell with fewest segments."""
        min_segments = float("inf")
        least_used = col_start
        
        for cell_idx in range(col_start, col_end):
            n_segs = neurons.tm_segments.count_segments(cell_idx)
            if n_segs < min_segments:
                min_segments = n_segs
                least_used = cell_idx
        
        return least_used
    
    def _compute_predictions(self, neurons) -> None:
        """
        Compute which cells are predicted to be active next.
        
        A cell is predicted if any of its segments is active
        (has enough active synapses to active cells).
        """
        neurons.tm_predictive_cells.zero_()
        
        for cell_idx in range(self.n_cells):
            if neurons.tm_segments.has_active_segment(
                cell_idx,
                neurons.tm_active_cells,
                self.activation_threshold,
                self.connected_perm,
            ):
                neurons.tm_predictive_cells[cell_idx] = True
    
    def _learn(self, neurons) -> None:
        """
        Apply learning to segments.
        
        For each winner cell:
        - Reinforce active segment synapses from previously active cells
        - Punish segment synapses from previously inactive cells
        - Grow new synapses to previously active cells
        """
        prev_active = neurons.tm_prev_active_cells
        prev_winners = neurons.tm_prev_winner_cells
        
        for cell_idx in neurons.tm_winner_cells.nonzero(as_tuple=True)[0]:
            cell_idx = cell_idx.item()
            
            # Find learning segment (active or best matching)
            segment_idx = neurons.tm_segments.get_active_or_matching_segment(
                cell_idx,
                prev_active,
                self.activation_threshold,
                self.learning_threshold,
                self.connected_perm,
            )
            
            if segment_idx >= 0:
                # Learn on existing segment
                neurons.tm_segments.adapt_segment(
                    cell_idx,
                    segment_idx,
                    prev_active,
                    self.perm_inc,
                    self.perm_dec,
                )
                
                # Grow new synapses if needed
                neurons.tm_segments.grow_synapses(
                    cell_idx,
                    segment_idx,
                    prev_winners,
                    self.initial_perm,
                    self.max_new_synapses,
                    self.max_synapses,
                )
            else:
                # Create new segment
                neurons.tm_segments.create_segment(
                    cell_idx,
                    prev_winners,
                    self.initial_perm,
                    self.max_new_synapses,
                    self.max_segments,
                )
        
        # Punish segments that predicted incorrectly
        if self.predicted_dec > 0:
            self._punish_predicted_columns(neurons, prev_active)
    
    def _punish_predicted_columns(
        self, neurons, prev_active_cells: torch.Tensor
    ) -> None:
        """Punish segments that caused incorrect predictions."""
        for col_idx in range(self.n_columns):
            if neurons.tm_active_columns[col_idx]:
                continue  # Column is active, no punishment
            
            col_start = col_idx * self.cells_per_column
            col_end = col_start + self.cells_per_column
            
            for cell_idx in range(col_start, col_end):
                if neurons.tm_predictive_cells[cell_idx]:
                    # This cell was predicted but column didn't activate
                    neurons.tm_segments.punish_segments(
                        cell_idx,
                        prev_active_cells,
                        self.predicted_dec,
                        self.activation_threshold,
                        self.connected_perm,
                    )
    
    def _compute_anomaly(self, neurons) -> None:
        """
        Compute anomaly score.
        
        Anomaly = fraction of active columns that were not predicted.
        High anomaly indicates novel or unexpected input.
        """
        n_active = neurons.tm_active_columns.sum()
        if n_active == 0:
            neurons.tm_anomaly_score.fill_(0)
            return
        
        n_bursting = neurons.tm_bursting_columns.sum()
        neurons.tm_anomaly_score = n_bursting / n_active
        neurons.tm_prediction_accuracy = 1.0 - neurons.tm_anomaly_score
    
    def forward(self, neurons):
        """
        Process input through Temporal Memory.
        
        Expects neurons.tm_input_columns to be set with active column indices.
        """
        if hasattr(neurons, "tm_input_columns"):
            self.compute(neurons, neurons.tm_input_columns, learn=True)


class SegmentStorage:
    """
    Storage for dendritic segments and their synapses.
    
    Implements efficient storage and operations for the segment-synapse
    structure used in Temporal Memory.
    """
    
    def __init__(
        self,
        n_cells: int,
        max_segments: int,
        max_synapses: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.n_cells = n_cells
        self.max_segments = max_segments
        self.max_synapses = max_synapses
        self.device = device
        self.dtype = dtype
        
        # Per-cell segment count
        self.segment_counts = torch.zeros(n_cells, device=device, dtype=torch.long)
        
        # Segment data: (cell_idx, segment_idx) -> synapse data
        # Using dictionary for flexibility; could optimize with tensors
        self.segments: dict[tuple[int, int], dict] = {}
    
    def _get_segment(self, cell_idx: int, segment_idx: int) -> dict | None:
        """Get segment data or None if not exists."""
        return self.segments.get((cell_idx, segment_idx))
    
    def count_segments(self, cell_idx: int) -> int:
        """Count segments for a cell."""
        return self.segment_counts[cell_idx].item()
    
    def has_active_segment(
        self,
        cell_idx: int,
        active_cells: torch.Tensor,
        threshold: int,
        connected_perm: float,
    ) -> bool:
        """Check if cell has an active segment."""
        for seg_idx in range(self.count_segments(cell_idx)):
            seg = self._get_segment(cell_idx, seg_idx)
            if seg is None:
                continue
            
            # Count connected synapses to active cells
            connected = seg["permanences"] >= connected_perm
            targets_active = active_cells[seg["targets"]]
            n_active = (connected & targets_active).sum()
            
            if n_active >= threshold:
                return True
        
        return False
    
    def get_best_segment_score(
        self,
        cell_idx: int,
        active_cells: torch.Tensor,
        threshold: int,
    ) -> int:
        """Get score of best matching segment."""
        best_score = 0
        
        for seg_idx in range(self.count_segments(cell_idx)):
            seg = self._get_segment(cell_idx, seg_idx)
            if seg is None:
                continue
            
            targets_active = active_cells[seg["targets"]]
            score = targets_active.sum().item()
            
            if score > best_score:
                best_score = score
        
        return best_score
    
    def get_active_or_matching_segment(
        self,
        cell_idx: int,
        active_cells: torch.Tensor,
        activation_threshold: int,
        learning_threshold: int,
        connected_perm: float,
    ) -> int:
        """Get index of active or best matching segment."""
        best_segment = -1
        best_score = 0
        
        for seg_idx in range(self.count_segments(cell_idx)):
            seg = self._get_segment(cell_idx, seg_idx)
            if seg is None:
                continue
            
            # Check for active segment
            connected = seg["permanences"] >= connected_perm
            targets_active = active_cells[seg["targets"]]
            n_connected_active = (connected & targets_active).sum().item()
            
            if n_connected_active >= activation_threshold:
                return seg_idx
            
            # Check for matching segment
            n_active = targets_active.sum().item()
            if n_active >= learning_threshold and n_active > best_score:
                best_score = n_active
                best_segment = seg_idx
        
        return best_segment
    
    def adapt_segment(
        self,
        cell_idx: int,
        segment_idx: int,
        active_cells: torch.Tensor,
        perm_inc: float,
        perm_dec: float,
    ) -> None:
        """Adapt segment permanences based on active cells."""
        seg = self._get_segment(cell_idx, segment_idx)
        if seg is None:
            return
        
        targets_active = active_cells[seg["targets"]]
        
        # Increment for active, decrement for inactive
        seg["permanences"][targets_active] += perm_inc
        seg["permanences"][~targets_active] -= perm_dec
        
        # Clip to [0, 1]
        seg["permanences"].clamp_(0, 1)
    
    def grow_synapses(
        self,
        cell_idx: int,
        segment_idx: int,
        winner_cells: torch.Tensor,
        initial_perm: float,
        max_new: int,
        max_synapses: int,
    ) -> None:
        """Grow new synapses to winner cells."""
        seg = self._get_segment(cell_idx, segment_idx)
        if seg is None:
            return
        
        current_targets = set(seg["targets"].tolist())
        n_current = len(current_targets)
        
        if n_current >= max_synapses:
            return
        
        # Find winner cells not already connected
        candidates = []
        for idx in winner_cells.nonzero(as_tuple=True)[0]:
            if idx.item() not in current_targets:
                candidates.append(idx.item())
        
        if not candidates:
            return
        
        # Randomly select new targets
        n_new = min(len(candidates), max_new, max_synapses - n_current)
        perm = torch.randperm(len(candidates))[:n_new]
        new_targets = [candidates[i] for i in perm]
        
        # Add new synapses
        new_targets_t = torch.tensor(new_targets, device=self.device, dtype=torch.long)
        new_perms = torch.full((n_new,), initial_perm, device=self.device, dtype=self.dtype)
        
        seg["targets"] = torch.cat([seg["targets"], new_targets_t])
        seg["permanences"] = torch.cat([seg["permanences"], new_perms])
    
    def create_segment(
        self,
        cell_idx: int,
        winner_cells: torch.Tensor,
        initial_perm: float,
        max_new: int,
        max_segments: int,
    ) -> int:
        """Create a new segment on a cell."""
        n_segments = self.count_segments(cell_idx)
        
        if n_segments >= max_segments:
            return -1
        
        # Find winner cells to connect to
        candidates = winner_cells.nonzero(as_tuple=True)[0].tolist()
        
        if not candidates:
            return -1
        
        n_synapses = min(len(candidates), max_new)
        perm = torch.randperm(len(candidates))[:n_synapses]
        targets = [candidates[i] for i in perm]
        
        # Create segment
        segment_idx = n_segments
        self.segments[(cell_idx, segment_idx)] = {
            "targets": torch.tensor(targets, device=self.device, dtype=torch.long),
            "permanences": torch.full(
                (n_synapses,), initial_perm, device=self.device, dtype=self.dtype
            ),
        }
        self.segment_counts[cell_idx] += 1
        
        return segment_idx
    
    def punish_segments(
        self,
        cell_idx: int,
        active_cells: torch.Tensor,
        decrement: float,
        threshold: int,
        connected_perm: float,
    ) -> None:
        """Punish segments that caused false predictions."""
        for seg_idx in range(self.count_segments(cell_idx)):
            seg = self._get_segment(cell_idx, seg_idx)
            if seg is None:
                continue
            
            connected = seg["permanences"] >= connected_perm
            targets_active = active_cells[seg["targets"]]
            n_active = (connected & targets_active).sum()
            
            if n_active >= threshold:
                # This segment was active (caused prediction)
                seg["permanences"][targets_active] -= decrement
                seg["permanences"].clamp_(0, 1)
