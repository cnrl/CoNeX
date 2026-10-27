"""
Author: Soroush Mohammaddeimi <soroushdeimi@gmail.com>


Voting and Consensus Mechanisms for Cortical Columns.

Based on:
- Hawkins, J., Ahmad, S. (2016). "Why Neurons Have Thousands of Synapses, 
  A Theory of Sequence Memory in Neocortex". Frontiers in Neural Circuits.
- Hawkins, J., Lewis, M., et al. (2019). "A Framework for Intelligence and 
  Cortical Function Based on Grid Cells in the Neocortex". Frontiers in Neural Circuits.
- Lewis, M., et al. (2019). "Locations in the Neocortex: A Theory of Sensorimotor
  Object Recognition Using Cortical Grid Cells". Frontiers in Neural Circuits.

In the Thousand Brains Theory, each cortical column builds its own model of
objects. Columns communicate laterally to reach consensus on object identity
through a voting mechanism using sparse distributed representations (SDRs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from pymonntorch import Behavior

if TYPE_CHECKING:
    from pymonntorch import NeuronGroup, Network


class ColumnVoting(Behavior):
    """
    Voting mechanism for inter-column consensus.
    
    Each cortical column casts "votes" for object identities based on its
    local observations. Columns share votes through lateral connections
    and iteratively converge to a consensus.
    
    The voting mechanism uses sparse distributed representations (SDRs):
    - Each column maintains a set of possible object representations
    - Votes are SDR patterns representing object hypotheses
    - Consensus is reached when columns agree on a common SDR
    
    Key properties of SDR voting:
    - Robust to noise (partial overlap still meaningful)
    - Allows uncertainty (multiple hypotheses)
    - Naturally handles novel vs. familiar objects
    
    Args:
        n_objects: Maximum number of object representations.
        representation_size: Size of each object's SDR.
        sparsity: Target sparsity for SDRs.
        voting_threshold: Overlap threshold for vote agreement.
        decay_rate: Decay rate for vote accumulation.
        min_votes_for_consensus: Minimum votes needed for consensus.
    """
    
    def __init__(
        self,
        n_objects: int = 100,
        representation_size: int = 2048,
        *,
        sparsity: float = 0.02,
        voting_threshold: float = 0.5,
        decay_rate: float = 0.1,
        min_votes_for_consensus: int = 3,
        **kwargs,
    ):
        super().__init__(
            n_objects=n_objects,
            representation_size=representation_size,
            sparsity=sparsity,
            voting_threshold=voting_threshold,
            decay_rate=decay_rate,
            min_votes_for_consensus=min_votes_for_consensus,
            **kwargs,
        )
    
    def initialize(self, neurons):
        """Initialize voting mechanism state."""
        self.n_objects = self.parameter("n_objects", 100)
        self.representation_size = self.parameter("representation_size", 2048)
        self.sparsity = self.parameter("sparsity", 0.02)
        self.voting_threshold = self.parameter("voting_threshold", 0.5)
        self.decay_rate = self.parameter("decay_rate", 0.1)
        self.min_votes = self.parameter("min_votes_for_consensus", 3)
        
        device = neurons.device
        dtype = neurons.network.def_dtype
        
        self.n_active = max(1, int(self.representation_size * self.sparsity))
        
        # Vote accumulator for each possible object
        neurons.vote_accumulator = torch.zeros(
            self.n_objects, device=device, dtype=dtype
        )
        
        # Current object hypotheses (indices of possible objects)
        neurons.object_hypotheses = torch.zeros(
            self.n_objects, device=device, dtype=torch.bool
        )
        
        # Confidence in each hypothesis
        neurons.hypothesis_confidence = torch.zeros(
            self.n_objects, device=device, dtype=dtype
        )
        
        # Consensus state
        neurons.consensus_reached = False
        neurons.consensus_object = -1
        
        # Vote history for temporal integration
        neurons.vote_history = []
        
        neurons.voting_module = self
    
    def cast_vote(
        self,
        neurons,
        local_representation: torch.Tensor,
        object_memories: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cast a vote based on local column representation.
        
        Compares the local representation to stored object memories
        and generates votes for matching objects.
        
        Args:
            neurons: The neuron group (column).
            local_representation: Current SDR from this column.
            object_memories: Stored object SDRs, shape (n_objects, representation_size).
            
        Returns:
            Vote tensor of shape (n_objects,) with vote strengths.
        """
        # Compute overlap between local representation and each object memory
        # Using SDR overlap: |A ∩ B| / |A|
        local_active = local_representation > 0
        n_local_active = local_active.sum()
        
        if n_local_active == 0:
            return torch.zeros(self.n_objects, device=neurons.device)
        
        # Compute overlap with each object
        overlaps = (object_memories * local_active.unsqueeze(0)).sum(dim=1)
        votes = overlaps / n_local_active
        
        # Threshold votes
        votes = votes * (votes >= self.voting_threshold)
        
        return votes
    
    def receive_votes(
        self,
        neurons,
        incoming_votes: torch.Tensor,
        source_confidence: float = 1.0,
    ) -> None:
        """
        Receive and accumulate votes from other columns.
        
        Args:
            neurons: The neuron group (column).
            incoming_votes: Votes from another column, shape (n_objects,).
            source_confidence: Confidence weight for the source column.
        """
        # Accumulate votes with confidence weighting
        neurons.vote_accumulator = (
            neurons.vote_accumulator * (1 - self.decay_rate)
            + incoming_votes * source_confidence
        )
        
        # Update hypotheses based on accumulated votes
        neurons.object_hypotheses = neurons.vote_accumulator > 0
        neurons.hypothesis_confidence = neurons.vote_accumulator.clone()
    
    def check_consensus(self, neurons, n_columns: int) -> bool:
        """
        Check if consensus has been reached.
        
        Consensus is reached when:
        1. One object has significantly more votes than others
        2. The vote count exceeds minimum threshold
        
        Args:
            neurons: The neuron group (column).
            n_columns: Total number of participating columns.
            
        Returns:
            True if consensus is reached.
        """
        if neurons.vote_accumulator.sum() == 0:
            neurons.consensus_reached = False
            return False
        
        # Find top two vote counts
        sorted_votes, sorted_indices = torch.sort(
            neurons.vote_accumulator, descending=True
        )
        
        top_votes = sorted_votes[0]
        second_votes = sorted_votes[1] if len(sorted_votes) > 1 else 0
        
        # Check consensus criteria
        min_vote_threshold = self.min_votes
        margin_threshold = 1.5  # Top must be 1.5x second
        
        consensus = (
            top_votes >= min_vote_threshold
            and (second_votes == 0 or top_votes / second_votes >= margin_threshold)
        )
        
        neurons.consensus_reached = consensus
        if consensus:
            neurons.consensus_object = sorted_indices[0].item()
        else:
            neurons.consensus_object = -1
        
        return consensus
    
    def reset_voting(self, neurons) -> None:
        """Reset voting state for new inference."""
        neurons.vote_accumulator.zero_()
        neurons.object_hypotheses.zero_()
        neurons.hypothesis_confidence.zero_()
        neurons.consensus_reached = False
        neurons.consensus_object = -1
        neurons.vote_history.clear()
    
    def forward(self, neurons):
        """
        Update voting state.
        
        This is called each time step to decay old votes and
        check for consensus.
        """
        # Apply decay to vote accumulator
        neurons.vote_accumulator = neurons.vote_accumulator * (1 - self.decay_rate)


class ConsensusNetwork(Behavior):
    """
    Network-level behavior for coordinating consensus across columns.
    
    This behavior runs at the network level and orchestrates the
    voting process across all cortical columns.
    
    Args:
        convergence_threshold: Threshold for declaring network-wide consensus.
        max_iterations: Maximum voting iterations before timeout.
        broadcast_interval: Timesteps between vote broadcasts.
    """
    
    def __init__(
        self,
        *,
        convergence_threshold: float = 0.8,
        max_iterations: int = 20,
        broadcast_interval: int = 1,
        **kwargs,
    ):
        super().__init__(
            convergence_threshold=convergence_threshold,
            max_iterations=max_iterations,
            broadcast_interval=broadcast_interval,
            **kwargs,
        )
    
    def initialize(self, network):
        """Initialize consensus network."""
        self.convergence_threshold = self.parameter("convergence_threshold", 0.8)
        self.max_iterations = self.parameter("max_iterations", 20)
        self.broadcast_interval = self.parameter("broadcast_interval", 1)
        
        network.consensus_iteration = 0
        network.network_consensus = False
        network.consensus_network = self
    
    def broadcast_votes(self, network, columns: list) -> None:
        """
        Broadcast votes between all columns.
        
        Implements lateral communication between cortical columns
        for consensus building.
        
        Args:
            network: The network object.
            columns: List of column neuron groups with voting modules.
        """
        n_columns = len(columns)
        if n_columns < 2:
            return
        
        # Collect all votes
        all_votes = []
        for col in columns:
            if hasattr(col, "voting_module") and hasattr(col, "hypothesis_confidence"):
                all_votes.append(col.hypothesis_confidence.clone())
        
        if not all_votes:
            return
        
        # Average votes and broadcast back (simple scheme)
        # More sophisticated schemes could use topology or attention
        avg_votes = torch.stack(all_votes).mean(dim=0)
        
        for col in columns:
            if hasattr(col, "voting_module"):
                col.voting_module.receive_votes(col, avg_votes)
    
    def check_network_consensus(self, network, columns: list) -> bool:
        """
        Check if network-wide consensus has been reached.
        
        Args:
            network: The network object.
            columns: List of column neuron groups.
            
        Returns:
            True if sufficient columns agree on object identity.
        """
        if not columns:
            return False
        
        # Count columns that have reached consensus
        consensus_objects = []
        for col in columns:
            if hasattr(col, "consensus_reached") and col.consensus_reached:
                consensus_objects.append(col.consensus_object)
        
        if not consensus_objects:
            return False
        
        # Check if majority agree on same object
        from collections import Counter
        vote_counts = Counter(consensus_objects)
        most_common_object, count = vote_counts.most_common(1)[0]
        
        agreement_ratio = count / len(columns)
        network.network_consensus = agreement_ratio >= self.convergence_threshold
        
        return network.network_consensus
    
    def forward(self, network):
        """Update consensus state at network level."""
        network.consensus_iteration += 1


class SDROverlap(Behavior):
    """
    Utility behavior for computing SDR overlap metrics.
    
    Sparse Distributed Representations are fundamental to the
    Thousand Brains Theory. This behavior provides methods for
    computing overlap, which is used for pattern matching and voting.
    
    SDR properties:
    - Fixed sparsity (typically 2%)
    - High dimensional (typically 2048 bits)
    - Semantic similarity = bit overlap
    
    Args:
        similarity_threshold: Minimum overlap ratio for similarity.
    """
    
    def __init__(self, *, similarity_threshold: float = 0.3, **kwargs):
        super().__init__(similarity_threshold=similarity_threshold, **kwargs)
    
    def initialize(self, neurons):
        """Initialize SDR overlap behavior."""
        self.similarity_threshold = self.parameter("similarity_threshold", 0.3)
        neurons.sdr_overlap = self
    
    @staticmethod
    def compute_overlap(sdr_a: torch.Tensor, sdr_b: torch.Tensor) -> int:
        """
        Compute raw overlap (number of shared active bits).
        
        Args:
            sdr_a: First SDR (boolean or binary tensor).
            sdr_b: Second SDR (boolean or binary tensor).
            
        Returns:
            Number of overlapping active bits.
        """
        return ((sdr_a > 0) & (sdr_b > 0)).sum().item()
    
    @staticmethod
    def compute_overlap_ratio(sdr_a: torch.Tensor, sdr_b: torch.Tensor) -> float:
        """
        Compute overlap ratio (overlap / min active bits).
        
        Args:
            sdr_a: First SDR.
            sdr_b: Second SDR.
            
        Returns:
            Overlap ratio in [0, 1].
        """
        active_a = (sdr_a > 0).sum()
        active_b = (sdr_b > 0).sum()
        
        if active_a == 0 or active_b == 0:
            return 0.0
        
        overlap = ((sdr_a > 0) & (sdr_b > 0)).sum()
        return (overlap / min(active_a, active_b)).item()
    
    @staticmethod
    def compute_jaccard(sdr_a: torch.Tensor, sdr_b: torch.Tensor) -> float:
        """
        Compute Jaccard similarity (overlap / union).
        
        Args:
            sdr_a: First SDR.
            sdr_b: Second SDR.
            
        Returns:
            Jaccard similarity in [0, 1].
        """
        active_a = sdr_a > 0
        active_b = sdr_b > 0
        
        intersection = (active_a & active_b).sum()
        union = (active_a | active_b).sum()
        
        if union == 0:
            return 0.0
        
        return (intersection / union).item()
    
    def are_similar(self, sdr_a: torch.Tensor, sdr_b: torch.Tensor) -> bool:
        """
        Check if two SDRs are similar based on overlap threshold.
        
        Args:
            sdr_a: First SDR.
            sdr_b: Second SDR.
            
        Returns:
            True if overlap ratio exceeds threshold.
        """
        return self.compute_overlap_ratio(sdr_a, sdr_b) >= self.similarity_threshold
    
    @staticmethod
    def union(sdr_a: torch.Tensor, sdr_b: torch.Tensor) -> torch.Tensor:
        """
        Compute union of two SDRs.
        
        Args:
            sdr_a: First SDR.
            sdr_b: Second SDR.
            
        Returns:
            Union SDR.
        """
        return ((sdr_a > 0) | (sdr_b > 0)).to(sdr_a.dtype)
    
    @staticmethod
    def intersection(sdr_a: torch.Tensor, sdr_b: torch.Tensor) -> torch.Tensor:
        """
        Compute intersection of two SDRs.
        
        Args:
            sdr_a: First SDR.
            sdr_b: Second SDR.
            
        Returns:
            Intersection SDR.
        """
        return ((sdr_a > 0) & (sdr_b > 0)).to(sdr_a.dtype)
    
    def forward(self, neurons):
        """No-op for utility behavior."""
        pass
