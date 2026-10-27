"""Tests for the predictive coding synapses and hierarchy structures."""

import pytest
import torch
from pymonntorch import Network, NeuronGroup, SynapseGroup

from conex import (
    ErrorUnit,
    FeedbackPredictionSynapse,
    FeedforwardErrorSynapse,
    LateralPredictionSynapse,
    PredictionUnit,
    PredictiveCodingConfig,
    PredictiveCorticalColumn,
    PredictiveHierarchy,
    PredictiveSynapseInit,
    PredictiveSynapseLearning,
    TimeResolution,
)


SRC_SIZE = 6
DST_SIZE = 4


def linear_config(**overrides):
    settings = dict(error_nonlinearity="linear", use_sparse_errors=False)
    settings.update(overrides)
    return PredictiveCodingConfig(**settings)


@pytest.fixture
def pair():
    """Two populations joined by a synapse group, behaviors added per test."""

    def build(synapse_behaviors, src_behaviors=None, dst_behaviors=None):
        net = Network(
            behavior={1: TimeResolution(dt=1.0)}, dtype=torch.float32, device="cpu"
        )
        src = NeuronGroup(
            net=net, size=SRC_SIZE, behavior=src_behaviors or {}, tag="src"
        )
        dst = NeuronGroup(
            net=net, size=DST_SIZE, behavior=dst_behaviors or {}, tag="dst"
        )
        synapses = SynapseGroup(
            net=net, src=src, dst=dst, behavior=synapse_behaviors, tag="syn"
        )
        net.initialize()
        return net, src, dst, synapses

    return build


class TestPredictiveSynapseInit:
    def test_weight_shape_matches_the_populations(self, pair):
        cfg = linear_config()
        _, _, _, syn = pair({100: PredictiveSynapseInit(config=cfg)})
        assert syn.weights.shape == (SRC_SIZE, DST_SIZE)
        assert syn.eligibility.shape == (SRC_SIZE, DST_SIZE)

    def test_connection_type_flags(self, pair):
        cfg = linear_config()
        for kind, expected in [
            ("feedforward", ("is_feedforward", True)),
            ("feedback", ("is_feedback", True)),
            ("lateral", ("is_lateral", True)),
        ]:
            _, _, _, syn = pair(
                {100: PredictiveSynapseInit(config=cfg, connection_type=kind)}
            )
            assert getattr(syn, expected[0]) is expected[1]

    def test_feedforward_allocates_the_destination_error_buffer(self, pair):
        cfg = linear_config()
        _, _, dst, syn = pair(
            {100: PredictiveSynapseInit(config=cfg, connection_type="feedforward")}
        )
        assert syn.error_output.shape == (DST_SIZE,)
        assert dst.bottom_up_error.shape == (DST_SIZE,)

    def test_feedback_allocates_the_prediction_buffer(self, pair):
        cfg = linear_config()
        _, _, _, syn = pair(
            {100: PredictiveSynapseInit(config=cfg, connection_type="feedback")}
        )
        assert syn.prediction_output.shape == (DST_SIZE,)

    def test_init_scale_controls_weight_magnitude(self, pair):
        cfg = linear_config()
        _, _, _, small = pair(
            {100: PredictiveSynapseInit(config=cfg, init_scale=0.01)}
        )
        _, _, _, large = pair(
            {100: PredictiveSynapseInit(config=cfg, init_scale=1.0)}
        )
        assert large.weights.abs().mean() > small.weights.abs().mean()


class TestFeedbackPredictionSynapse:
    def test_prediction_reaches_the_destination(self, pair):
        cfg = linear_config()
        _, src, dst, syn = pair(
            {
                100: PredictiveSynapseInit(config=cfg, connection_type="feedback"),
                110: FeedbackPredictionSynapse(config=cfg),
            },
            dst_behaviors={100: PredictionUnit(config=cfg)},
        )
        src.activity = torch.ones(SRC_SIZE)

        syn.behavior[110].forward(syn)

        # PredictionUnit provides `prediction`; `top_down_prediction` only
        # exists when TopDownPrediction is also attached.
        assert syn.prediction_output.shape == (DST_SIZE,)
        assert torch.allclose(dst.prediction, syn.prediction_output)

    def test_zero_source_activity_predicts_nothing(self, pair):
        cfg = linear_config()
        _, src, _, syn = pair(
            {
                100: PredictiveSynapseInit(config=cfg, connection_type="feedback"),
                110: FeedbackPredictionSynapse(config=cfg),
            },
            dst_behaviors={100: PredictionUnit(config=cfg)},
        )
        src.activity = torch.zeros(SRC_SIZE)
        syn.behavior[110].forward(syn)
        assert torch.all(syn.prediction_output == 0)


class TestFeedforwardErrorSynapse:
    def test_error_is_forwarded_into_the_buffer(self, pair):
        cfg = linear_config()
        _, src, dst, syn = pair(
            {
                100: PredictiveSynapseInit(config=cfg, connection_type="feedforward"),
                110: FeedforwardErrorSynapse(config=cfg, precision_weighted=False),
            },
            src_behaviors={100: ErrorUnit(config=cfg, precision_weighted=False)},
        )
        src.prediction_error = torch.ones(SRC_SIZE)

        syn.behavior[110].forward(syn)

        assert dst.bottom_up_error.shape == (DST_SIZE,)
        assert torch.allclose(dst.bottom_up_error, syn.error_output)

    def test_precision_weighting_scales_the_error(self, pair):
        cfg = linear_config()

        def run(precision_weighted, precision):
            _, src, dst, syn = pair(
                {
                    100: PredictiveSynapseInit(
                        config=cfg, connection_type="feedforward"
                    ),
                    110: FeedforwardErrorSynapse(
                        config=cfg, precision_weighted=precision_weighted
                    ),
                },
                src_behaviors={100: ErrorUnit(config=cfg, precision_weighted=False)},
            )
            syn.weights = torch.ones(SRC_SIZE, DST_SIZE)
            src.prediction_error = torch.ones(SRC_SIZE)
            src.precision = torch.full((SRC_SIZE,), precision)
            syn.behavior[110].forward(syn)
            return dst.bottom_up_error.clone()

        plain = run(False, 1.0)
        scaled = run(True, 3.0)
        assert torch.allclose(scaled, plain * 3.0)


class TestLateralPredictionSynapse:
    def test_lateral_context_is_written(self, pair):
        cfg = linear_config()
        _, src, dst, syn = pair(
            {
                100: PredictiveSynapseInit(config=cfg, connection_type="lateral"),
                110: LateralPredictionSynapse(config=cfg),
            },
        )
        src.activity = torch.ones(SRC_SIZE)
        dst.lateral_context = torch.zeros(DST_SIZE)

        syn.behavior[110].forward(syn)

        assert dst.lateral_context.shape == (DST_SIZE,)

    def test_inhibitory_context_is_never_positive(self, pair):
        cfg = linear_config()
        _, src, dst, syn = pair(
            {
                100: PredictiveSynapseInit(config=cfg, connection_type="lateral"),
                110: LateralPredictionSynapse(config=cfg, inhibitory=True),
            },
        )
        src.activity = torch.ones(SRC_SIZE)
        dst.lateral_context = torch.zeros(DST_SIZE)

        syn.behavior[110].forward(syn)

        assert torch.all(dst.lateral_context <= 0)


class TestPredictiveSynapseLearning:
    def test_error_driven_update_changes_weights(self, pair):
        cfg = linear_config(learning_rate=0.1)
        _, src, dst, syn = pair(
            {
                100: PredictiveSynapseInit(config=cfg, connection_type="feedforward"),
                120: PredictiveSynapseLearning(config=cfg, learning_type="error_driven"),
            },
            dst_behaviors={100: ErrorUnit(config=cfg, precision_weighted=False)},
        )
        src.activity = torch.ones(SRC_SIZE)
        dst.prediction_error = torch.ones(DST_SIZE)
        before = syn.weights.clone()

        syn.behavior[120].forward(syn)

        assert not torch.equal(before, syn.weights)

    def test_no_activity_leaves_weights_alone(self, pair):
        cfg = linear_config(learning_rate=0.1)
        _, src, dst, syn = pair(
            {
                100: PredictiveSynapseInit(config=cfg, connection_type="feedforward"),
                120: PredictiveSynapseLearning(config=cfg),
            },
            dst_behaviors={100: ErrorUnit(config=cfg, precision_weighted=False)},
        )
        src.activity = torch.zeros(SRC_SIZE)
        dst.prediction_error = torch.zeros(DST_SIZE)
        before = syn.weights.clone()

        syn.behavior[120].forward(syn)

        assert torch.equal(before, syn.weights)

    def test_eligibility_accumulates_over_steps(self, pair):
        cfg = linear_config(learning_rate=0.1)
        _, src, dst, syn = pair(
            {
                100: PredictiveSynapseInit(config=cfg, connection_type="feedforward"),
                120: PredictiveSynapseLearning(config=cfg),
            },
            dst_behaviors={100: ErrorUnit(config=cfg, precision_weighted=False)},
        )
        src.activity = torch.ones(SRC_SIZE)
        dst.prediction_error = torch.ones(DST_SIZE)

        syn.behavior[120].forward(syn)
        first = syn.eligibility.abs().mean().item()
        syn.behavior[120].forward(syn)
        second = syn.eligibility.abs().mean().item()

        assert second > first

    def test_hebbian_uses_destination_activity(self, pair):
        cfg = linear_config(learning_rate=0.1)
        _, src, dst, syn = pair(
            {
                100: PredictiveSynapseInit(config=cfg, connection_type="feedforward"),
                120: PredictiveSynapseLearning(config=cfg, learning_type="hebbian"),
            },
        )
        src.activity = torch.ones(SRC_SIZE)
        dst.activity = torch.ones(DST_SIZE)
        before = syn.weights.clone()

        syn.behavior[120].forward(syn)

        assert not torch.equal(before, syn.weights)


class TestPredictiveHierarchy:
    @pytest.fixture
    def hierarchy(self):
        net = Network(
            behavior={1: TimeResolution(dt=1.0)}, dtype=torch.float32, device="cpu"
        )
        built = PredictiveHierarchy(
            net, level_sizes=[8, 4, 2], level_names=["sensory", "mid", "top"]
        )
        net.initialize()
        return net, built

    def test_levels_and_connections_are_built(self, hierarchy):
        _, built = hierarchy
        assert built.n_levels == 3
        assert all(level.population is not None for level in built.levels)
        assert all(level.layer is not None for level in built.levels)
        assert len(built.feedforward_synapses) == 2
        assert len(built.feedback_synapses) == 2

    def test_population_sizes_follow_level_sizes(self, hierarchy):
        _, built = hierarchy
        assert [level.population.size for level in built.levels] == [8, 4, 2]

    def test_set_input_reaches_the_bottom_population(self, hierarchy):
        _, built = hierarchy
        data = torch.rand(8)
        built.set_input(data)
        assert torch.allclose(built.levels[0].population.activity, data)

    def test_simulation_runs_and_reports(self, hierarchy):
        net, built = hierarchy
        built.set_input(torch.rand(8))
        net.simulate_iterations(5)

        assert built.get_predictions(0).shape == (8,)
        assert built.get_errors(0).shape == (8,)
        assert built.get_precision(0).shape == (8,)
        assert isinstance(built.get_total_free_energy(), float)

    def test_free_energy_is_summed_across_levels(self, hierarchy):
        net, built = hierarchy
        built.set_input(torch.rand(8))
        net.simulate_iterations(3)
        total = built.get_total_free_energy()
        parts = sum(built.get_free_energy(i) for i in range(built.n_levels))
        assert total == pytest.approx(parts)

    def test_repr_lists_the_levels(self, hierarchy):
        _, built = hierarchy
        text = repr(built)
        assert "sensory" in text and "top" in text


class TestPredictiveCorticalColumn:
    @pytest.fixture
    def column(self):
        net = Network(
            behavior={1: TimeResolution(dt=1.0)}, dtype=torch.float32, device="cpu"
        )
        built = PredictiveCorticalColumn(net, column_size=8)
        net.initialize()
        return net, built

    def test_all_four_layers_exist(self, column):
        _, built = column
        assert sorted(built.populations) == ["L23", "L4", "L5", "L6"]
        for layer in (built.L4, built.L23, built.L5, built.L6):
            assert layer is not None

    def test_layer_sizes_follow_cortical_proportions(self, column):
        _, built = column
        assert built.populations["L4"].size == 8
        assert built.populations["L23"].size == 12
        assert built.populations["L5"].size == 6
        assert built.populations["L6"].size == 4

    def test_canonical_connections_are_present(self, column):
        _, built = column
        assert sorted(built.synapses) == [
            "L23_to_L4",
            "L23_to_L5",
            "L4_to_L23",
            "L6_to_L4",
        ]

    def test_simulation_runs(self, column):
        net, built = column
        built.set_input(torch.rand(8))
        net.simulate_iterations(5)

        assert built.get_prediction().shape == (12,)
        assert built.get_error().shape == (8,)
        assert built.get_output().shape == (6,)
