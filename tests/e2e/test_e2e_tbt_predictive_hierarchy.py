"""End-to-end tests for the predictive coding hierarchy structures.

Local build_network helper: the shared tests/e2e/conftest.py is owned by
another worker and not yet merged, so each file carries its own builder.
"""

import torch

from conex import (
    HierarchyLevel,
    PredictiveHierarchy,
    PredictiveCorticalColumn,
    PredictiveCodingConfig,
    TimeResolution,
)
from pymonntorch import Network


def build_network(seed=0):
    torch.manual_seed(seed)
    net = Network(behavior={1: TimeResolution(dt=1.0)}, dtype=torch.float32, device="cpu")
    return net


def linear_config(**overrides):
    settings = dict(error_nonlinearity="linear", use_sparse_errors=False)
    settings.update(overrides)
    return PredictiveCodingConfig(**settings)


class TestHierarchyLevel:
    def test_dataclass_fields(self):
        level = HierarchyLevel(size=32, name="sensory")
        assert level.size == 32
        assert level.name == "sensory"
        assert level.layer is None
        assert level.population is None
        assert level.config is None


class TestPredictiveHierarchy:
    def test_builds_levels_and_connections(self):
        net = build_network(seed=1)
        h = PredictiveHierarchy(net, level_sizes=[16, 8, 4],
                                level_names=["sensory", "mid", "top"],
                                config=linear_config())
        assert h.n_levels == 3
        assert len(h.feedforward_synapses) == 2
        assert len(h.feedback_synapses) == 2
        assert h.levels[0].population.size == 16
        assert h.levels[2].population.size == 4

    def test_default_names(self):
        net = build_network(seed=2)
        h = PredictiveHierarchy(net, level_sizes=[8, 4], config=linear_config())
        assert [level.name for level in h.levels] == ["level_0", "level_1"]

    def test_set_input_and_getters(self):
        net = build_network(seed=3)
        h = PredictiveHierarchy(net, level_sizes=[16, 8], config=linear_config())
        h.set_input(torch.rand(16))
        assert h.levels[0].population.activity.shape == (16,)
        assert h.get_predictions(0).shape == (16,)
        assert h.get_errors(0).shape == (16,)
        assert h.get_precision(0).shape == (16,)
        assert isinstance(h.get_free_energy(0), float)

    def test_runs_for_n_steps_and_reports_free_energy(self):
        net = build_network(seed=4)
        h = PredictiveHierarchy(net, level_sizes=[16, 8, 4], config=linear_config())
        h.set_input(torch.rand(16))
        net.initialize()
        net.simulate_iterations(5)
        total = h.get_total_free_energy()
        assert isinstance(total, float)
        # each level produced a finite free energy
        for i in range(h.n_levels):
            assert torch.isfinite(torch.tensor(h.get_free_energy(i)))

    def test_repr(self):
        net = build_network(seed=5)
        h = PredictiveHierarchy(net, level_sizes=[16, 8],
                                level_names=["a", "b"], config=linear_config())
        assert "a(16)" in repr(h)
        assert "b(8)" in repr(h)

    def test_build_immediately_false(self):
        net = build_network(seed=6)
        h = PredictiveHierarchy(net, level_sizes=[8, 4], config=linear_config(),
                                build_immediately=False)
        assert h.levels[0].population is None
        h.build()
        assert h.levels[0].population is not None


class TestPredictiveCorticalColumn:
    def test_builds_canonical_layers(self):
        net = build_network(seed=7)
        col = PredictiveCorticalColumn(net, column_size=20, config=linear_config(), name="c1")
        assert col.layer_sizes["L4"] == 20
        assert col.layer_sizes["L23"] == 30  # 20 * 1.5
        assert col.layer_sizes["L5"] == 16  # 20 * 0.8
        assert col.layer_sizes["L6"] == 10  # 20 * 0.5
        assert set(col.populations) == {"L4", "L23", "L5", "L6"}
        assert set(col.synapses) == {"L4_to_L23", "L23_to_L4", "L23_to_L5", "L6_to_L4"}

    def test_set_input_and_getters(self):
        net = build_network(seed=8)
        col = PredictiveCorticalColumn(net, column_size=20, config=linear_config())
        col.set_input(torch.rand(20))
        assert col.populations["L4"].activity.shape == (20,)
        assert col.get_prediction().shape == (30,)
        assert col.get_output().shape == (16,)
        assert col.get_error().shape == (20,)
        assert isinstance(col.get_free_energy(), float)

    def test_runs_for_n_steps(self):
        net = build_network(seed=9)
        col = PredictiveCorticalColumn(net, column_size=16, config=linear_config())
        col.set_input(torch.rand(16))
        net.initialize()
        net.simulate_iterations(4)
        assert torch.isfinite(torch.tensor(col.get_free_energy()))
        assert col.get_prediction().shape == (24,)

    def test_repr(self):
        net = build_network(seed=10)
        col = PredictiveCorticalColumn(net, column_size=12, config=linear_config(), name="c2")
        assert "c2" in repr(col)
        assert "size=12" in repr(col)
