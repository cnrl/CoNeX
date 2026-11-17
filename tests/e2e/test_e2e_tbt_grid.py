"""End-to-end tests for grid cells, displacement cells and location encoding.

Each file carries its own small network builder so it runs on its own.
"""

import math

import pytest
import torch

from conex import GridCellModule, DisplacementCellModule, LocationEncoder, TimeResolution
from pymonntorch import Network, NeuronGroup


def build_network(seed=0):
    torch.manual_seed(seed)
    net = Network(behavior={1: TimeResolution(dt=1.0)}, dtype=torch.float32, device="cpu")
    return net


def make_group(net, size, behavior, tag="grid"):
    g = NeuronGroup(net=net, size=size, behavior={100: behavior}, tag=tag)
    net.initialize()
    return g


N_MODULES = 4
CELLS_PER_MODULE = 20
N_CELLS = N_MODULES * CELLS_PER_MODULE


@pytest.fixture
def grid():
    net = build_network(seed=5)
    return make_group(net, N_CELLS, GridCellModule(n_modules=N_MODULES,
                                                   cells_per_module=CELLS_PER_MODULE,
                                                   scale_range=(0.5, 2.0)))


class TestGridCellModule:
    def test_state_shapes(self, grid):
        assert grid.grid_phase.shape == (N_CELLS, 2)
        assert grid.grid_activation.shape == (N_CELLS,)
        assert grid.grid_phases.shape == (N_MODULES, CELLS_PER_MODULE, 2)
        assert grid.grid_cell_module.wave_vectors.shape == (N_CELLS, 3, 2)

    def test_activation_bounded_and_deterministic(self, grid):
        a = grid.grid_cell_module.encode_position(grid, torch.tensor([1.7, -0.4]))
        assert a.shape == (N_CELLS,)
        assert a.min() >= 0.0 and a.max() <= 1.0
        assert torch.allclose(a, grid.grid_cell_module.encode_position(grid, torch.tensor([1.7, -0.4])))

    def test_lattice_translation_is_periodic(self, grid):
        """A full lattice translation leaves a module's code unchanged.

        The three wave vectors sit at 60 degrees, so the shortest translation
        with k_i . s in 2*pi*Z for all three is 2*lambda/sqrt(3) along theta+30.
        """
        module = grid.grid_cell_module
        scale = module.scales[0].item()
        angle = module.orientations[0].item() + math.pi / 6
        length = 2 * scale / math.sqrt(3)
        step = torch.tensor([length * math.cos(angle), length * math.sin(angle)])
        origin = module.encode_position(grid, torch.zeros(2))[:CELLS_PER_MODULE]
        shifted = module.encode_position(grid, step)[:CELLS_PER_MODULE]
        assert torch.allclose(origin, shifted, atol=1e-4)

    def test_half_lattice_step_changes_code(self, grid):
        module = grid.grid_cell_module
        scale = module.scales[0].item()
        angle = module.orientations[0].item() + math.pi / 6
        length = scale / math.sqrt(3)
        step = torch.tensor([length * math.cos(angle), length * math.sin(angle)])
        origin = module.encode_position(grid, torch.zeros(2))[:CELLS_PER_MODULE]
        shifted = module.encode_position(grid, step)[:CELLS_PER_MODULE]
        assert not torch.allclose(origin, shifted, atol=1e-2)

    def test_batched_positions(self, grid):
        positions = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        assert grid.grid_cell_module.encode_position(grid, positions).shape == (3, N_CELLS)

    def test_path_integration_accumulates(self, grid):
        grid.grid_cell_module.path_integrate(grid, torch.tensor([1.0, 0.0]), dt=0.5)
        grid.grid_cell_module.path_integrate(grid, torch.tensor([0.0, 2.0]), dt=0.5)
        assert torch.allclose(grid.grid_position, torch.tensor([0.5, 1.0]))

    def test_path_integration_uses_network_dt(self, grid):
        grid.grid_cell_module.path_integrate(grid, torch.tensor([2.0, 0.0]))
        assert torch.allclose(grid.grid_position, torch.tensor([2.0, 0.0]))  # dt = 1.0

    def test_forward_updates_activation(self, grid):
        grid.grid_position = torch.tensor([1.25, -0.75])
        grid.grid_cell_module.forward(grid)
        expected = grid.grid_cell_module.encode_position(grid, grid.grid_position)
        assert torch.allclose(grid.grid_activation, expected)

    def test_anchor_keeps_phase_in_range(self, grid):
        grid.grid_position = torch.tensor([3.0, 4.0])
        grid.grid_cell_module.forward(grid)
        grid.grid_cell_module.anchor_to_input(grid, torch.rand(N_CELLS))
        assert grid.grid_phase.min() >= 0.0
        assert grid.grid_phase.max() <= 2 * math.pi

    def test_user_scales_and_orientations(self):
        net = build_network(seed=6)
        g = make_group(net, N_CELLS, GridCellModule(n_modules=2, cells_per_module=10,
                                                    scales=[1.0, 3.0], orientations=[0.0, 0.5]))
        scales = g.grid_cell_module.scales.view(2, 10)[:, 0]
        assert torch.allclose(scales, torch.tensor([1.0, 3.0]))


@pytest.fixture
def displacement():
    net = build_network(seed=7)
    return make_group(net, 100, DisplacementCellModule(n_cells=100, max_displacement=10.0, sigma=0.5))


class TestDisplacementCellModule:
    def test_encode_decode_roundtrip(self, displacement):
        module = displacement.displacement_cell_module
        target = torch.tensor([3.0, -2.0])
        activation = module.encode_displacement(displacement, target)
        assert activation.shape == (100,)
        decoded = module.decode_displacement(displacement, activation)
        assert torch.allclose(decoded, target, atol=0.5)  # sigma = 0.5

    def test_encode_zero_displacement(self, displacement):
        module = displacement.displacement_cell_module
        activation = module.encode_displacement(displacement, torch.zeros(2))
        assert activation.max() <= 1.0
        assert activation.max() > 0.0  # some cell prefers near-zero

    def test_decode_empty_activation_is_zero(self, displacement):
        module = displacement.displacement_cell_module
        decoded = module.decode_displacement(displacement, torch.zeros(100))
        assert torch.allclose(decoded, torch.zeros(2))

    def test_compute_displacement_between_locations(self, displacement):
        module = displacement.displacement_cell_module
        act = module.compute_displacement(displacement, torch.tensor([1.0, 1.0]),
                                          torch.tensor([4.0, 3.0]))
        decoded = module.decode_displacement(displacement, act)
        assert torch.allclose(decoded, torch.tensor([3.0, 2.0]), atol=0.5)

    def test_forward_tracks_grid_position(self, displacement):
        displacement.grid_position = torch.zeros(2)
        displacement.displacement_cell_module.forward(displacement)
        assert displacement.previous_location is not None
        displacement.grid_position = torch.tensor([1.0, 0.0])
        displacement.displacement_cell_module.forward(displacement)
        assert displacement.displacement_activation.max() > 0.0


@pytest.fixture
def location():
    net = build_network(seed=8)
    return make_group(net, 256, LocationEncoder(location_size=256, grid_weight=0.5, sparsity=0.02))


class TestLocationEncoder:
    def test_sparse_representation_from_grid(self, location):
        enc = location.location_encoder
        grid_act = torch.rand(256)
        rep = enc.encode_location(location, grid_activation=grid_act)
        # n_active = int(256 * 0.02) = 5 active bits
        assert rep.sum().item() == 5
        assert location.active_locations.shape == (5,)

    def test_no_input_gives_zero(self, location):
        rep = location.location_encoder.encode_location(location)
        assert rep.sum().item() == 0.0

    def test_forward_uses_grid_activation(self, location):
        location.grid_activation = torch.rand(256)
        location.location_encoder.forward(location)
        assert location.location_representation.sum().item() == 5

    def test_sensory_only_input(self, location):
        enc = location.location_encoder
        rep = enc.encode_location(location, sensory_location=torch.rand(256))
        assert rep.sum().item() == 5
