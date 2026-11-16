"""End-to-end combination: grid cells + location encoding + sequence memory.

Path integration on a known trajectory, a location-recognition check, and a
temporal memory that learns the trajectory as a sequence.

Each file carries its own small network builder so it runs on its own.
"""

import pytest
import torch

from pymonntorch import Network, NeuronGroup

from conex import (
    GridCellModule,
    LocationEncoder,
    LocationSetter,
    TemporalMemory,
    TemporalMemoryConfig,
    TimeResolution,
    Layer,
    prioritize_behaviors,
    NeuronAxon,
)


def build_network(seed=0):
    torch.manual_seed(seed)
    return Network(behavior={1: TimeResolution(dt=1.0)}, dtype=torch.float32, device="cpu")


N_MODULES = 4
CELLS_PER_MODULE = 20
N_CELLS = N_MODULES * CELLS_PER_MODULE
TM_CFG = TemporalMemoryConfig(
    n_columns=40, cells_per_column=4, activation_threshold=2,
    learning_threshold=1, max_new_synapses=5,
)


def build_grid(seed=5):
    """Grid cells, a location encoder and a temporal memory in one network."""
    net = build_network(seed)
    grid = NeuronGroup(net=net, size=N_CELLS, behavior={100: GridCellModule(
        n_modules=N_MODULES, cells_per_module=CELLS_PER_MODULE,
        scale_range=(0.5, 2.0))}, tag="grid")
    loc = NeuronGroup(net=net, size=256, behavior={100: LocationEncoder(
        location_size=256, grid_weight=0.5, sparsity=0.02)}, tag="loc")
    tm = NeuronGroup(net=net, size=TM_CFG.n_columns * TM_CFG.cells_per_column,
                     behavior={100: TemporalMemory(config=TM_CFG)}, tag="tm")
    net.initialize()
    return net, grid, loc, tm


def column_mask(location_rep):
    """Fold a location SDR into a 40-column boolean mask for the memory."""
    mask = torch.zeros(40, dtype=torch.bool)
    idx = (location_rep > 0).nonzero(as_tuple=True)[0]
    mask[idx % 40] = True
    return mask


class TestComboGrid:
    def test_path_integration_reaches_known_position(self):
        """Two steps +x then two steps +y land the agent at (2, 2)."""
        _, grid, _, _ = build_grid()
        for v in [(1.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 1.0)]:
            grid.grid_cell_module.path_integrate(grid, torch.tensor(v), dt=1.0)
        assert torch.allclose(grid.grid_position, torch.tensor([2.0, 2.0]))

    def test_location_recognition_same_vs_new_position(self):
        """The same position gives the same location SDR; a new one differs."""
        _, grid, loc, _ = build_grid()
        for v in [(1.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 1.0)]:
            grid.grid_cell_module.path_integrate(grid, torch.tensor(v), dt=1.0)
        grid.grid_cell_module.forward(grid)
        rep_final = loc.location_encoder.encode_location(loc, grid_activation=grid.grid_activation)
        assert rep_final.sum().item() == 5  # n_active = int(256 * 0.02)

        # re-encoding the same position reproduces the same SDR
        grid.grid_cell_module.forward(grid)
        rep_again = loc.location_encoder.encode_location(loc, grid_activation=grid.grid_activation)
        assert torch.equal(rep_final, rep_again)

        # a new position yields a different SDR (only partial overlap)
        grid.grid_cell_module.path_integrate(grid, torch.tensor([5.0, 0.0]), dt=1.0)
        grid.grid_cell_module.forward(grid)
        rep_new = loc.location_encoder.encode_location(loc, grid_activation=grid.grid_activation)
        overlap = ((rep_final > 0) & (rep_new > 0)).sum().item()
        assert overlap < rep_final.sum().item()

    def test_sequence_memory_learns_trajectory(self):
        """A temporal memory learns the location sequence and its error falls."""
        _, grid, loc, tm = build_grid()
        reps = []
        for v in [(1.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 1.0)]:
            grid.grid_cell_module.path_integrate(grid, torch.tensor(v), dt=1.0)
            grid.grid_cell_module.forward(grid)
            reps.append(column_mask(
                loc.location_encoder.encode_location(loc, grid_activation=grid.grid_activation)))
        first = None
        for _ in range(12):
            scores = []
            for mask in reps:
                tm.temporal_memory.compute(tm, mask, learn=True)
                scores.append(tm.tm_anomaly_score.item())
            if first is None:
                first = sum(scores)
        assert sum(scores) < first
        assert tm.tm_prediction_accuracy.item() > 0.0

    def test_location_setter_feeds_layer_loc(self):
        """LocationSetter copies the layer's loc into the population spikes."""
        net = build_network(seed=9)
        ng = NeuronGroup(net=net, size=4, tag="loc",
                         behavior=prioritize_behaviors([LocationSetter(), NeuronAxon()]))
        layer = Layer(net=net, neurongroups=[ng], tag="L")
        net.initialize()
        layer.loc = torch.tensor([False, True, False, True])
        net.simulate_iterations(1)
        assert ng.spikes.tolist() == [False, True, False, True]
