"""
Author: Soroush Mohammaddeimi <soroushdeimi@gmail.com>


Grid Cell and Location Signal Modules.

Based on:
- Moser, E.I., Moser, M.B. (2008). "A metric for space". Hippocampus.
- Hawkins, J., Lewis, M., et al. (2019). "A Framework for Intelligence and Cortical 
  Function Based on Grid Cells in the Neocortex". Frontiers in Neural Circuits.
- Numenta's displacement cell and grid cell research.

Grid cells provide allocentric (world-centered) reference frames that allow
cortical columns to learn complete models of objects through movement.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from pymonntorch import Behavior


class GridCellModule(Behavior):
    """
    Grid Cell Module implementing hexagonal firing patterns.
    
    Grid cells fire when an agent is at vertices of a hexagonal lattice
    tiling an environment. Each module has a characteristic:
    - Scale (spacing between firing fields)
    - Orientation (rotation of the grid pattern)
    - Phase (offset of the grid pattern)
    
    In the Thousand Brains Theory, grid cells provide the reference frame
    that allows each cortical column to represent object locations.
    
    The firing rate follows:
        r(x) = sum_{i=1}^{3} cos(k_i · (x - x_0))
    
    where k_i are wave vectors at 60° angles.
    
    Args:
        n_modules: Number of grid cell modules with different scales.
        cells_per_module: Number of grid cells per module.
        scale_range: Tuple of (min_scale, max_scale) for grid spacing.
        orientation_range: Tuple of (min_angle, max_angle) in radians.
        learning_rate: Rate for path integration error correction.
        noise_std: Standard deviation of velocity noise for robustness.
    """
    
    def __init__(
        self,
        n_modules: int = 4,
        cells_per_module: int = 40,
        *,
        scale_range: tuple[float, float] = (0.5, 2.0),
        orientation_range: tuple[float, float] = (0.0, math.pi / 3),
        learning_rate: float = 0.01,
        noise_std: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            n_modules=n_modules,
            cells_per_module=cells_per_module,
            scale_range=scale_range,
            orientation_range=orientation_range,
            learning_rate=learning_rate,
            noise_std=noise_std,
            **kwargs,
        )
    
    def initialize(self, neurons):
        """Initialize grid cell parameters and state."""
        self.n_modules = self.parameter("n_modules", 4)
        self.cells_per_module = self.parameter("cells_per_module", 40)
        self.scale_range = self.parameter("scale_range", (0.5, 2.0))
        self.orientation_range = self.parameter("orientation_range", (0.0, math.pi / 3))
        self.learning_rate = self.parameter("learning_rate", 0.01)
        self.noise_std = self.parameter("noise_std", 0.0)
        
        # Check for user-provided scales and orientations
        user_scales = self.parameter("scales", None)
        user_orientations = self.parameter("orientations", None)
        
        n_cells = self.n_modules * self.cells_per_module
        device = neurons.device
        dtype = neurons.network.def_dtype
        
        # Initialize scales (use user-provided or generate logarithmically spaced)
        if user_scales is not None:
            scales = torch.tensor(user_scales, device=device, dtype=dtype)
        else:
            scales = torch.logspace(
                math.log10(self.scale_range[0]),
                math.log10(self.scale_range[1]),
                self.n_modules,
                device=device,
                dtype=dtype,
            )
        self.scales = scales.repeat_interleave(self.cells_per_module)
        
        # Initialize orientations (use user-provided or generate uniform within range)
        if user_orientations is not None:
            orientations = torch.tensor(user_orientations, device=device, dtype=dtype)
            # Convert degrees to radians if values seem to be in degrees
            if orientations.max() > 2 * math.pi:
                orientations = orientations * (math.pi / 180.0)
        else:
            orientations = torch.linspace(
                self.orientation_range[0],
                self.orientation_range[1],
                self.n_modules,
                device=device,
                dtype=dtype,
            )
        self.orientations = orientations.repeat_interleave(self.cells_per_module)
        
        # Initialize phases (random offsets for each cell)
        # Phase is 2D for spatial representation
        neurons.grid_phase = torch.rand(
            n_cells, 2, device=device, dtype=dtype
        ) * 2 * math.pi
        
        # Precompute wave vectors for hexagonal pattern (3 directions at 60°)
        # k_i = (2π/λ) * [cos(θ + i*60°), sin(θ + i*60°)]
        self._precompute_wave_vectors(neurons)
        
        # Current estimated position (for path integration)
        neurons.grid_position = torch.zeros(2, device=device, dtype=dtype)
        
        # Grid cell activations
        neurons.grid_activation = torch.zeros(n_cells, device=device, dtype=dtype)
        
        # Per-module phases for external access (reshaped view)
        neurons.grid_phases = neurons.grid_phase.view(self.n_modules, self.cells_per_module, 2)
        
        # Store reference to module
        neurons.grid_cell_module = self
    
    def encode_position(self, neurons, position: torch.Tensor) -> torch.Tensor:
        """Alias for compute_activation for cleaner API."""
        return self.compute_activation(neurons, position)
    
    def _precompute_wave_vectors(self, neurons):
        """Precompute wave vectors for efficient firing rate calculation."""
        device = neurons.device
        dtype = neurons.network.def_dtype
        n_cells = self.n_modules * self.cells_per_module
        
        # Wave vectors: 3 directions per cell at 60° intervals
        self.wave_vectors = torch.zeros(n_cells, 3, 2, device=device, dtype=dtype)
        
        for i in range(3):
            angle = self.orientations + i * (math.pi / 3)
            wavelength = self.scales
            k_magnitude = 2 * math.pi / wavelength
            
            self.wave_vectors[:, i, 0] = k_magnitude * torch.cos(angle)
            self.wave_vectors[:, i, 1] = k_magnitude * torch.sin(angle)
    
    def compute_activation(self, neurons, position: torch.Tensor) -> torch.Tensor:
        """
        Compute grid cell activation given a position.
        
        Uses the sum of three cosines at 60° angles to create hexagonal pattern:
            activation = (1/3) * sum_{i=1}^{3} cos(k_i · (x - phase))
        
        Args:
            neurons: The neuron group.
            position: 2D position tensor of shape (2,) or (batch, 2).
            
        Returns:
            Activation tensor of shape (n_cells,) or (batch, n_cells).
        """
        # Ensure position is 2D
        if position.dim() == 1:
            position = position.unsqueeze(0)
        
        batch_size = position.shape[0]
        n_cells = self.n_modules * self.cells_per_module
        
        # Compute phase-adjusted position: x - phase
        # phase: (n_cells, 2), position: (batch, 2)
        adjusted_pos = position.unsqueeze(1) - neurons.grid_phase.unsqueeze(0)
        # adjusted_pos: (batch, n_cells, 2)
        
        # Compute dot product with wave vectors
        # wave_vectors: (n_cells, 3, 2)
        # We need: k_i · adjusted_pos for each cell and each of 3 directions
        dot_products = torch.einsum(
            "bcd,cnd->bcn", adjusted_pos, self.wave_vectors
        )
        # dot_products: (batch, n_cells, 3)
        
        # Compute sum of cosines and normalize
        activation = torch.cos(dot_products).sum(dim=-1) / 3.0
        # activation: (batch, n_cells)
        
        # Shift to [0, 1] range: (cos + 1) / 2 gives [0, 1]
        activation = (activation + 1) / 2
        
        if batch_size == 1:
            return activation.squeeze(0)
        return activation
    
    def path_integrate(
        self, neurons, velocity: torch.Tensor, dt: float | None = None
    ) -> None:
        """
        Update position estimate through path integration.
        
        Grid cells maintain position through dead reckoning by integrating
        velocity signals. This is the basis for allocentric reference frames.
        
        Args:
            neurons: The neuron group.
            velocity: 2D velocity tensor of shape (2,).
            dt: Time step (uses network dt if None).
        """
        if dt is None:
            dt = neurons.network.dt
        
        # Add noise for robustness (optional)
        if self.noise_std > 0:
            noise = torch.randn_like(velocity) * self.noise_std
            velocity = velocity + noise
        
        # Update position estimate
        neurons.grid_position = neurons.grid_position + velocity * dt
    
    def anchor_to_input(
        self, neurons, sensory_activation: torch.Tensor, learning_rate: float | None = None
    ) -> None:
        """
        Anchor grid cell representation to sensory input.
        
        This implements error correction when sensory input provides
        a reliable location signal, correcting path integration drift.
        
        Args:
            neurons: The neuron group.
            sensory_activation: Sensory-driven location activation.
            learning_rate: Learning rate (uses default if None).
        """
        if learning_rate is None:
            learning_rate = self.learning_rate
        
        # Compute error between current activation and sensory-driven activation
        error = sensory_activation - neurons.grid_activation
        
        # Update phases to reduce error (Hebbian-like learning)
        # This is a simplified version; full implementation would use
        # competitive learning or attractor dynamics
        phase_update = learning_rate * error.unsqueeze(-1) * neurons.grid_position
        neurons.grid_phase = neurons.grid_phase + phase_update
        
        # Wrap phases to [0, 2π]
        neurons.grid_phase = neurons.grid_phase % (2 * math.pi)
    
    def forward(self, neurons):
        """
        Update grid cell activations based on current position.
        
        In a full implementation, this would receive velocity input
        for path integration. Here we compute activation from stored position.
        """
        neurons.grid_activation = self.compute_activation(
            neurons, neurons.grid_position
        )


class DisplacementCellModule(Behavior):
    """
    Displacement Cell Module for encoding movements between locations.
    
    Displacement cells encode the transformation (movement) between
    two locations. They are crucial for:
    - Learning object structure through movement
    - Predicting sensory input after movement
    - Composing reference frame transformations
    
    Based on Numenta's research on how cortical columns use displacement
    to link features at different locations on an object.
    
    Args:
        n_cells: Number of displacement cells.
        max_displacement: Maximum displacement magnitude to encode.
        sigma: Width of displacement tuning curves.
    """
    
    def __init__(
        self,
        n_cells: int = 100,
        *,
        max_displacement: float = 10.0,
        sigma: float = 0.5,
        **kwargs,
    ):
        super().__init__(
            n_cells=n_cells,
            max_displacement=max_displacement,
            sigma=sigma,
            **kwargs,
        )
    
    def initialize(self, neurons):
        """Initialize displacement cell parameters."""
        self.n_cells = self.parameter("n_cells", 100)
        self.max_displacement = self.parameter("max_displacement", 10.0)
        self.sigma = self.parameter("sigma", 0.5)
        
        # Alternative parameterization via directions/distances
        n_directions = self.parameter("n_directions", None)
        n_distances = self.parameter("n_distances", None)
        max_distance = self.parameter("max_distance", None)
        
        if n_directions is not None and n_distances is not None:
            self.n_cells = n_directions * n_distances
        if max_distance is not None:
            self.max_displacement = max_distance
        
        device = neurons.device
        dtype = neurons.network.def_dtype
        
        # Each cell has a preferred displacement (uniformly distributed)
        # Using polar coordinates: (magnitude, direction)
        if n_directions is not None and n_distances is not None:
            n_magnitudes = n_distances
            n_dirs = n_directions
        else:
            n_magnitudes = int(math.sqrt(self.n_cells))
            n_dirs = self.n_cells // n_magnitudes
        
        magnitudes = torch.linspace(
            0, self.max_displacement, n_magnitudes, device=device, dtype=dtype
        )
        directions = torch.linspace(
            0, 2 * math.pi, n_dirs + 1, device=device, dtype=dtype
        )[:-1]  # Exclude 2*pi (same as 0)
        
        # Create grid of preferred displacements
        mag_grid, dir_grid = torch.meshgrid(magnitudes, directions, indexing="ij")
        
        # Convert to Cartesian
        self.preferred_dx = (mag_grid * torch.cos(dir_grid)).flatten()[:self.n_cells]
        self.preferred_dy = (mag_grid * torch.sin(dir_grid)).flatten()[:self.n_cells]
        
        # Current displacement activation
        neurons.displacement_activation = torch.zeros(
            self.n_cells, device=device, dtype=dtype
        )
        
        # Previous location for computing displacement
        neurons.previous_location = None
        
        neurons.displacement_cell_module = self
        neurons.displacement_module = self  # Alias for convenience
    
    def encode_displacement(
        self, neurons, displacement: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode a displacement vector as a population activation pattern.
        
        Uses Gaussian tuning curves centered on each cell's preferred displacement.
        
        Args:
            neurons: The neuron group.
            displacement: 2D displacement vector (dx, dy).
            
        Returns:
            Activation pattern over displacement cells.
        """
        dx, dy = displacement[0], displacement[1]
        
        # Compute distance from each cell's preferred displacement
        dist_sq = (self.preferred_dx - dx) ** 2 + (self.preferred_dy - dy) ** 2
        
        # Gaussian activation
        activation = torch.exp(-dist_sq / (2 * self.sigma ** 2))
        
        return activation
    
    def decode_displacement(
        self, neurons, activation: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode displacement vector from population activation.
        
        Uses population vector decoding (weighted average of preferred displacements).
        
        Args:
            neurons: The neuron group.
            activation: Activation pattern over displacement cells.
            
        Returns:
            Decoded 2D displacement vector (dx, dy).
        """
        # Normalize activation to sum to 1
        activation_sum = activation.sum()
        if activation_sum < 1e-8:
            return torch.zeros(2, device=activation.device)
        
        weights = activation / activation_sum
        
        # Population vector decoding
        dx = (weights * self.preferred_dx).sum()
        dy = (weights * self.preferred_dy).sum()
        
        return torch.stack([dx, dy])

    def compute_displacement(
        self, neurons, from_location: torch.Tensor, to_location: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute and encode displacement between two locations.
        
        Args:
            neurons: The neuron group.
            from_location: Starting location (2D).
            to_location: Ending location (2D).
            
        Returns:
            Displacement activation pattern.
        """
        displacement = to_location - from_location
        return self.encode_displacement(neurons, displacement)
    
    def forward(self, neurons):
        """
        Update displacement cell activation based on movement.
        
        Computes displacement from previous to current location
        if both are available.
        """
        if hasattr(neurons, "grid_position"):
            current_location = neurons.grid_position
            
            if neurons.previous_location is not None:
                displacement = current_location - neurons.previous_location
                neurons.displacement_activation = self.encode_displacement(
                    neurons, displacement
                )
            
            neurons.previous_location = current_location.clone()


class LocationEncoder(Behavior):
    """
    Location Encoder that combines grid cells and sensory input.
    
    This module implements the location layer of a cortical column,
    combining:
    - Path-integrated location from grid cells
    - Sensory-derived location signals
    - Object-specific location representations
    
    Based on the Thousand Brains Theory's concept of how each
    cortical column maintains its own reference frame.
    
    Args:
        location_size: Size of the location representation.
        grid_weight: Weight for grid cell input (vs sensory).
        sparsity: Target sparsity for location representation.
    """
    
    def __init__(
        self,
        location_size: int = 256,
        *,
        grid_weight: float = 0.5,
        sparsity: float = 0.02,
        **kwargs,
    ):
        super().__init__(
            location_size=location_size,
            grid_weight=grid_weight,
            sparsity=sparsity,
            **kwargs,
        )
    
    def initialize(self, neurons):
        """Initialize location encoder."""
        self.location_size = self.parameter("location_size", 256)
        self.grid_weight = self.parameter("grid_weight", 0.5)
        self.sparsity = self.parameter("sparsity", 0.02)
        
        device = neurons.device
        dtype = neurons.network.def_dtype
        
        # Location representation
        neurons.location_representation = torch.zeros(
            self.location_size, device=device, dtype=dtype
        )
        
        # Active location indices (sparse representation)
        self.n_active = max(1, int(self.location_size * self.sparsity))
        neurons.active_locations = torch.zeros(
            self.n_active, device=device, dtype=torch.long
        )
        
        neurons.location_encoder = self
    
    def encode_location(
        self,
        neurons,
        grid_activation: torch.Tensor | None = None,
        sensory_location: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Encode location from grid cells and/or sensory input.
        
        Args:
            neurons: The neuron group.
            grid_activation: Grid cell activation pattern.
            sensory_location: Sensory-derived location signal.
            
        Returns:
            Sparse location representation.
        """
        device = neurons.device
        dtype = neurons.network.def_dtype
        
        combined = torch.zeros(self.location_size, device=device, dtype=dtype)
        
        if grid_activation is not None:
            # Project grid activation to location space
            # In a full implementation, this would be a learned projection
            grid_proj = torch.zeros(self.location_size, device=device, dtype=dtype)
            n_grid = min(len(grid_activation), self.location_size)
            grid_proj[:n_grid] = grid_activation[:n_grid]
            combined = combined + self.grid_weight * grid_proj
        
        if sensory_location is not None:
            combined = combined + (1 - self.grid_weight) * sensory_location
        
        # Apply winner-take-all for sparse representation
        if combined.sum() > 0:
            _, top_indices = torch.topk(combined, self.n_active)
            neurons.active_locations = top_indices
            
            sparse_rep = torch.zeros_like(combined)
            sparse_rep[top_indices] = 1.0
            return sparse_rep
        
        return combined
    
    def forward(self, neurons):
        """Update location representation."""
        grid_act = getattr(neurons, "grid_activation", None)
        sensory_loc = getattr(neurons, "sensory_location", None)
        
        neurons.location_representation = self.encode_location(
            neurons, grid_act, sensory_loc
        )
