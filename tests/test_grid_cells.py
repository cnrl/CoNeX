"""Tests for grid cell and location encoding behaviors."""

import math

import pytest
import torch

from conex import GridCellModule


N_MODULES = 4
CELLS_PER_MODULE = 25
N_CELLS = N_MODULES * CELLS_PER_MODULE


@pytest.fixture
def grid(make_group):
    return make_group(
        N_CELLS,
        GridCellModule(
            n_modules=N_MODULES,
            cells_per_module=CELLS_PER_MODULE,
            scale_range=(0.5, 2.0),
        ),
        tag="grid",
    )


def test_state_shapes(grid):
    assert grid.grid_phase.shape == (N_CELLS, 2)
    assert grid.grid_activation.shape == (N_CELLS,)
    assert grid.grid_phases.shape == (N_MODULES, CELLS_PER_MODULE, 2)
    assert grid.grid_cell_module.wave_vectors.shape == (N_CELLS, 3, 2)


def test_modules_have_distinct_scales(grid):
    scales = grid.grid_cell_module.scales.view(N_MODULES, CELLS_PER_MODULE)[:, 0]
    assert scales.unique().numel() == N_MODULES
    assert scales.min().item() == pytest.approx(0.5, rel=1e-5)
    assert scales.max().item() == pytest.approx(2.0, rel=1e-5)
    assert torch.all(scales.diff() > 0)


def test_activation_is_bounded(grid):
    activation = grid.grid_cell_module.encode_position(grid, torch.tensor([1.7, -0.4]))
    assert activation.shape == (N_CELLS,)
    assert activation.min() >= 0.0
    assert activation.max() <= 1.0


def test_encode_position_is_deterministic(grid):
    position = torch.tensor([2.0, 3.0])
    first = grid.grid_cell_module.encode_position(grid, position).clone()
    second = grid.grid_cell_module.encode_position(grid, position)
    assert torch.allclose(first, second)


def test_different_positions_give_different_codes(grid):
    a = grid.grid_cell_module.encode_position(grid, torch.tensor([0.0, 0.0]))
    b = grid.grid_cell_module.encode_position(grid, torch.tensor([0.37, 0.71]))
    assert not torch.allclose(a, b)


def test_batched_positions(grid):
    positions = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    assert grid.grid_cell_module.encode_position(grid, positions).shape == (3, N_CELLS)


def test_activation_is_periodic_on_the_hexagonal_lattice(grid):
    """A lattice translation leaves a module's code unchanged.

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


def test_half_lattice_step_changes_the_code(grid):
    module = grid.grid_cell_module
    scale = module.scales[0].item()
    angle = module.orientations[0].item() + math.pi / 6
    length = scale / math.sqrt(3)
    step = torch.tensor([length * math.cos(angle), length * math.sin(angle)])

    origin = module.encode_position(grid, torch.zeros(2))[:CELLS_PER_MODULE]
    shifted = module.encode_position(grid, step)[:CELLS_PER_MODULE]
    assert not torch.allclose(origin, shifted, atol=1e-2)


def test_path_integration_accumulates_velocity(grid):
    grid.grid_cell_module.path_integrate(grid, torch.tensor([1.0, 0.0]), dt=0.5)
    grid.grid_cell_module.path_integrate(grid, torch.tensor([0.0, 2.0]), dt=0.5)
    assert torch.allclose(grid.grid_position, torch.tensor([0.5, 1.0]))


def test_forward_updates_activation_from_position(grid):
    grid.grid_position = torch.tensor([1.25, -0.75])
    grid.grid_cell_module.forward(grid)
    expected = grid.grid_cell_module.encode_position(grid, grid.grid_position)
    assert torch.allclose(grid.grid_activation, expected)


def test_anchor_keeps_phase_in_range(grid):
    grid.grid_position = torch.tensor([3.0, 4.0])
    grid.grid_cell_module.forward(grid)
    target = torch.rand(N_CELLS)
    grid.grid_cell_module.anchor_to_input(grid, target)
    assert grid.grid_phase.min() >= 0.0
    assert grid.grid_phase.max() <= 2 * math.pi
