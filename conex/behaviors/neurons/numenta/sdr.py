"""
Author: Soroush Mohammaddeimi <soroushdeimi@gmail.com>


Sparse Distributed Representation (SDR) Operations.

Based on:
- Ahmad, S., Hawkins, J. (2016). "How do neurons operate on sparse distributed
  representations? A mathematical theory of sparsity, neurons and active dendrites".
- Hawkins, J., Ahmad, S. (2016). "Why Neurons Have Thousands of Synapses".
- Numenta SDR research and implementations.

SDRs are the fundamental data structure in HTM and the Thousand Brains Theory:
- High dimensional (typically 2048 bits)
- Sparse (typically 2% active)
- Semantic similarity encoded as overlap

Key properties:
1. Fault tolerance: Partial matches still work
2. Capacity: Can store huge number of unique patterns
3. Union: Can represent multiple items simultaneously
4. Comparison: Simple overlap measure for similarity
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from pymonntorch import Behavior


@dataclass(frozen=True)
class SDRConfig:
    """Configuration for SDR operations."""
    
    size: int = 2048
    sparsity: float = 0.02
    
    @property
    def n_active(self) -> int:
        """Number of active bits."""
        return max(1, int(self.size * self.sparsity))


class SDR:
    """
    Sparse Distributed Representation.
    
    Represents a sparse binary pattern with efficient operations
    for overlap, union, and encoding.
    
    Can be constructed from:
    - Dense tensor (boolean or binary)
    - Sparse indices
    - Random generation
    
    Args:
        size: Total number of bits.
        indices: Active bit indices (optional).
        dense: Dense representation (optional).
        device: Torch device.
        dtype: Torch dtype for dense operations.
    """
    
    def __init__(
        self,
        size: int,
        indices: torch.Tensor | None = None,
        dense: torch.Tensor | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.size = size
        self.device = device
        self.dtype = dtype
        
        # An SDR is a set of active bits. Duplicates would inflate n_active and
        # break overlap(), which counts repeats in the sorted concatenation.
        if indices is not None:
            self._indices = torch.unique(indices.to(device=device, dtype=torch.long))
        elif dense is not None:
            self._indices = (dense > 0).nonzero(as_tuple=True)[0].to(device)
        else:
            self._indices = torch.tensor([], device=device, dtype=torch.long)
    
    @property
    def indices(self) -> torch.Tensor:
        """Active bit indices."""
        return self._indices
    
    @property
    def n_active(self) -> int:
        """Number of active bits."""
        return len(self._indices)
    
    @property
    def sparsity(self) -> float:
        """Fraction of active bits."""
        return self.n_active / self.size if self.size > 0 else 0.0
    
    def to_dense(self) -> torch.Tensor:
        """Convert to dense boolean tensor."""
        dense = torch.zeros(self.size, device=self.device, dtype=torch.bool)
        if self.n_active > 0:
            dense[self._indices] = True
        return dense
    
    def to_float(self) -> torch.Tensor:
        """Convert to dense float tensor."""
        return self.to_dense().to(self.dtype)
    
    @classmethod
    def from_dense(
        cls,
        dense: torch.Tensor,
        size: int | None = None,
    ) -> "SDR":
        """Create SDR from dense tensor."""
        size = size or len(dense)
        return cls(size=size, dense=dense, device=dense.device)
    
    @classmethod
    def random(
        cls,
        size: int,
        n_active: int,
        device: torch.device | str = "cpu",
    ) -> "SDR":
        """Create random SDR with specified number of active bits."""
        indices = torch.randperm(size, device=device)[:n_active]
        return cls(size=size, indices=indices, device=device)
    
    @classmethod
    def from_config(
        cls,
        config: SDRConfig,
        device: torch.device | str = "cpu",
    ) -> "SDR":
        """Create random SDR from configuration."""
        return cls.random(config.size, config.n_active, device)
    
    def overlap(self, other: "SDR") -> int:
        """
        Compute overlap (number of shared active bits).
        
        Args:
            other: Another SDR.
            
        Returns:
            Number of overlapping active bits.
        """
        if self.n_active == 0 or other.n_active == 0:
            return 0
        
        # Use set intersection via sorting
        combined = torch.cat([self._indices, other._indices])
        sorted_combined, _ = torch.sort(combined)
        duplicates = sorted_combined[1:] == sorted_combined[:-1]
        return duplicates.sum().item()
    
    def overlap_score(self, other: "SDR") -> float:
        """
        Compute overlap score (overlap / min active).
        
        This is the standard HTM similarity measure.
        Ranges from 0 (no overlap) to 1 (complete overlap).
        """
        if self.n_active == 0 or other.n_active == 0:
            return 0.0
        
        return self.overlap(other) / min(self.n_active, other.n_active)
    
    def jaccard(self, other: "SDR") -> float:
        """
        Compute Jaccard similarity (overlap / union size).
        
        Ranges from 0 (no overlap) to 1 (identical).
        """
        overlap = self.overlap(other)
        union_size = self.n_active + other.n_active - overlap
        
        if union_size == 0:
            return 0.0
        
        return overlap / union_size
    
    def union(self, other: "SDR") -> "SDR":
        """
        Compute union of two SDRs.
        
        Active bits in either SDR are active in result.
        """
        combined = torch.cat([self._indices, other._indices])
        unique = torch.unique(combined)
        return SDR(size=self.size, indices=unique, device=self.device)
    
    def intersection(self, other: "SDR") -> "SDR":
        """
        Compute intersection of two SDRs.
        
        Only bits active in both SDRs are active in result.
        """
        if self.n_active == 0 or other.n_active == 0:
            return SDR(size=self.size, device=self.device)
        
        # Find common elements
        combined = torch.cat([self._indices, other._indices])
        sorted_combined, sorted_indices = torch.sort(combined)
        duplicates = sorted_combined[1:] == sorted_combined[:-1]
        
        # Get the actual duplicate values
        dup_positions = duplicates.nonzero(as_tuple=True)[0]
        common = sorted_combined[dup_positions]
        
        return SDR(size=self.size, indices=common, device=self.device)
    
    def difference(self, other: "SDR") -> "SDR":
        """
        Compute difference (self - other).
        
        Bits active in self but not in other.
        """
        if self.n_active == 0:
            return SDR(size=self.size, device=self.device)
        if other.n_active == 0:
            return SDR(size=self.size, indices=self._indices.clone(), device=self.device)
        
        # Create mask for self indices that are not in other
        mask = torch.ones(self.n_active, device=self.device, dtype=torch.bool)
        for i, idx in enumerate(self._indices):
            if (other._indices == idx).any():
                mask[i] = False
        
        return SDR(size=self.size, indices=self._indices[mask], device=self.device)
    
    def subsample(self, n: int) -> "SDR":
        """
        Randomly subsample to n active bits.
        
        If n >= n_active, returns copy of self.
        """
        if n >= self.n_active:
            return SDR(size=self.size, indices=self._indices.clone(), device=self.device)
        
        perm = torch.randperm(self.n_active, device=self.device)[:n]
        return SDR(size=self.size, indices=self._indices[perm], device=self.device)
    
    def match_threshold(self, other: "SDR", theta: int) -> bool:
        """
        Check if overlap meets threshold.
        
        Used for pattern matching in HTM.
        
        Args:
            other: SDR to compare against.
            theta: Minimum overlap threshold.
            
        Returns:
            True if overlap >= theta.
        """
        return self.overlap(other) >= theta
    
    def __repr__(self) -> str:
        return f"SDR(size={self.size}, n_active={self.n_active}, sparsity={self.sparsity:.4f})"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SDR):
            return False
        if self.size != other.size or self.n_active != other.n_active:
            return False
        return self.overlap(other) == self.n_active
    
    def __hash__(self) -> int:
        return hash((self.size, tuple(sorted(self._indices.tolist()))))


class SDREncoder(Behavior):
    """
    SDR Encoder for converting inputs to sparse representations.
    
    Implements several encoding schemes:
    - Scalar: Encode scalar values as SDRs
    - Category: Encode categorical values
    - Random: Random hash-based encoding
    
    Args:
        size: SDR size.
        n_active: Number of active bits.
        encoder_type: Type of encoder ("scalar", "category", "random").
        min_val: Minimum value for scalar encoding.
        max_val: Maximum value for scalar encoding.
        n_categories: Number of categories for category encoding.
    """
    
    def __init__(
        self,
        size: int = 2048,
        n_active: int = 41,
        *,
        encoder_type: Literal["scalar", "category", "random"] = "scalar",
        min_val: float = 0.0,
        max_val: float = 1.0,
        n_categories: int = 10,
        **kwargs,
    ):
        super().__init__(
            size=size,
            n_active=n_active,
            encoder_type=encoder_type,
            min_val=min_val,
            max_val=max_val,
            n_categories=n_categories,
            **kwargs,
        )
    
    def initialize(self, neurons):
        """Initialize encoder."""
        self.size = self.parameter("size", 2048)
        self.n_active = self.parameter("n_active", 41)
        self.encoder_type = self.parameter("encoder_type", "scalar")
        self.min_val = self.parameter("min_val", 0.0)
        self.max_val = self.parameter("max_val", 1.0)
        self.n_categories = self.parameter("n_categories", 10)
        
        device = neurons.device
        
        # For category encoding, pre-generate random SDRs
        if self.encoder_type == "category":
            self.category_sdrs = [
                SDR.random(self.size, self.n_active, device)
                for _ in range(self.n_categories)
            ]
        
        neurons.sdr_encoder = self
    
    def encode_scalar(
        self,
        value: float,
        device: torch.device | str = "cpu",
    ) -> SDR:
        """
        Encode scalar value as SDR.
        
        Uses a simple bucket encoding where the value determines
        the center of the active bits.
        """
        # Normalize to [0, 1]
        normalized = (value - self.min_val) / (self.max_val - self.min_val)
        normalized = max(0.0, min(1.0, normalized))
        
        # Determine center position
        center = int(normalized * (self.size - self.n_active))
        
        # Generate indices around center
        indices = torch.arange(center, center + self.n_active, device=device)
        indices = indices % self.size  # Wrap around
        
        return SDR(size=self.size, indices=indices, device=device)
    
    def encode_category(
        self,
        category: int,
        device: torch.device | str = "cpu",
    ) -> SDR:
        """
        Encode categorical value as SDR.
        
        Each category maps to a pre-defined random SDR.
        """
        if category < 0 or category >= self.n_categories:
            raise ValueError(f"Category {category} out of range [0, {self.n_categories})")
        
        return self.category_sdrs[category]
    
    def encode_random(
        self,
        seed: int,
        device: torch.device | str = "cpu",
    ) -> SDR:
        """
        Encode value using random hash.
        
        Same seed always produces same SDR.
        """
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        indices = torch.randperm(self.size, generator=generator, device=device)[:self.n_active]
        return SDR(size=self.size, indices=indices, device=device)
    
    def encode(
        self,
        value: float | int,
        device: torch.device | str = "cpu",
    ) -> SDR:
        """
        Encode value using configured encoder type.
        """
        if self.encoder_type == "scalar":
            return self.encode_scalar(float(value), device)
        elif self.encoder_type == "category":
            return self.encode_category(int(value), device)
        elif self.encoder_type == "random":
            return self.encode_random(int(value), device)
        else:
            raise ValueError(f"Unknown encoder type: {self.encoder_type}")
    
    def forward(self, neurons):
        """No-op for encoder behavior."""
        pass


class SDRClassifier(Behavior):
    """
    SDR-based classifier for pattern recognition.
    
    Learns to associate SDR patterns with class labels using
    simple Hebbian-style learning.
    
    Args:
        n_classes: Number of output classes.
        sdr_size: Size of input SDRs.
        learning_rate: Learning rate for weight updates.
    """
    
    def __init__(
        self,
        n_classes: int = 10,
        sdr_size: int = 2048,
        *,
        learning_rate: float = 0.01,
        **kwargs,
    ):
        super().__init__(
            n_classes=n_classes,
            sdr_size=sdr_size,
            learning_rate=learning_rate,
            **kwargs,
        )
    
    def initialize(self, neurons):
        """Initialize classifier weights."""
        self.n_classes = self.parameter("n_classes", 10)
        self.sdr_size = self.parameter("sdr_size", 2048)
        self.learning_rate = self.parameter("learning_rate", 0.01)
        
        device = neurons.device
        dtype = neurons.network.def_dtype
        
        # Weight matrix: each class has weights over SDR bits
        neurons.classifier_weights = torch.zeros(
            self.n_classes, self.sdr_size, device=device, dtype=dtype
        )
        
        # Class counts for normalization
        neurons.class_counts = torch.zeros(
            self.n_classes, device=device, dtype=dtype
        )
        
        neurons.sdr_classifier = self
    
    def learn(
        self,
        neurons,
        sdr: SDR | torch.Tensor,
        label: int,
    ) -> None:
        """
        Learn association between SDR and label.
        
        Args:
            neurons: The neuron group.
            sdr: Input SDR or dense tensor.
            label: Class label.
        """
        if isinstance(sdr, SDR):
            indices = sdr.indices
        else:
            indices = (sdr > 0).nonzero(as_tuple=True)[0]
        
        # Increment weights for active bits in this class
        neurons.classifier_weights[label, indices] += self.learning_rate
        neurons.class_counts[label] += 1
    
    def infer(
        self,
        neurons,
        sdr: SDR | torch.Tensor,
    ) -> tuple[int, torch.Tensor]:
        """
        Infer class from SDR.
        
        Args:
            neurons: The neuron group.
            sdr: Input SDR or dense tensor.
            
        Returns:
            Tuple of (predicted class, class scores).
        """
        if isinstance(sdr, SDR):
            dense = sdr.to_float()
        else:
            dense = sdr.to(neurons.network.def_dtype)
        
        # Compute scores for each class
        scores = torch.matmul(neurons.classifier_weights, dense)
        
        # Normalize by class counts
        normalized_scores = scores / (neurons.class_counts + 1e-6)
        
        predicted = normalized_scores.argmax().item()
        
        return predicted, normalized_scores
    
    def forward(self, neurons):
        """No-op for classifier behavior."""
        pass


class UnionSDR:
    """
    Union of multiple SDRs representing multiple items.
    
    In HTM, unions are used to represent uncertainty or
    multiple active concepts simultaneously.
    
    Note: Unions get denser as more SDRs are added.
    The union property degrades when sparsity exceeds ~40%.
    """
    
    def __init__(
        self,
        size: int,
        device: torch.device | str = "cpu",
    ):
        self.size = size
        self.device = device
        self._indices = torch.tensor([], device=device, dtype=torch.long)
        self._members: list[SDR] = []
    
    def add(self, sdr: SDR) -> None:
        """Add an SDR to the union."""
        self._members.append(sdr)
        combined = torch.cat([self._indices, sdr.indices])
        self._indices = torch.unique(combined)
    
    def remove(self, sdr: SDR) -> None:
        """
        Remove an SDR from the union.
        
        Note: This recomputes the union from remaining members.
        """
        self._members = [m for m in self._members if m != sdr]
        self._recompute_union()
    
    def _recompute_union(self) -> None:
        """Recompute union from members."""
        if not self._members:
            self._indices = torch.tensor([], device=self.device, dtype=torch.long)
            return
        
        all_indices = torch.cat([m.indices for m in self._members])
        self._indices = torch.unique(all_indices)
    
    def contains(self, sdr: SDR, threshold: float = 0.5) -> bool:
        """
        Check if union contains an SDR.
        
        Uses overlap score to determine membership.
        """
        union_sdr = SDR(size=self.size, indices=self._indices, device=self.device)
        return union_sdr.overlap_score(sdr) >= threshold
    
    def to_sdr(self) -> SDR:
        """Convert union to SDR."""
        return SDR(size=self.size, indices=self._indices.clone(), device=self.device)
    
    @property
    def n_active(self) -> int:
        """Number of active bits in union."""
        return len(self._indices)
    
    @property
    def sparsity(self) -> float:
        """Sparsity of union."""
        return self.n_active / self.size if self.size > 0 else 0.0
    
    @property
    def n_members(self) -> int:
        """Number of SDRs in union."""
        return len(self._members)
    
    def __repr__(self) -> str:
        return f"UnionSDR(size={self.size}, n_members={self.n_members}, n_active={self.n_active})"
