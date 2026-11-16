"""End-to-end tests for the predictive coding behaviors.

Each file carries its own small network builder so it runs on its own.
"""

import pytest
import torch

from conex import (
    PredictionType,
    PredictiveCodingConfig,
    PredictionUnit,
    ErrorUnit,
    PrecisionWeighting,
    TopDownPrediction,
    PredictiveCodingLearning,
    FreeEnergyMinimization,
    HierarchicalPredictiveCoding,
    create_predictive_layer,
    TimeResolution,
)
from pymonntorch import Network, NeuronGroup


def build_network(seed=0):
    torch.manual_seed(seed)
    net = Network(behavior={1: TimeResolution(dt=1.0)}, dtype=torch.float32, device="cpu")
    return net


def make_group(net, size, behavior, tag="pc"):
    g = NeuronGroup(net=net, size=size, behavior={100: behavior}, tag=tag)
    net.initialize()
    return g, behavior


SIZE = 16


def linear_config(**overrides):
    settings = dict(error_nonlinearity="linear", use_sparse_errors=False)
    settings.update(overrides)
    return PredictiveCodingConfig(**settings)


class TestPredictionUnit:
    def test_state_allocated(self):
        net = build_network()
        g, beh = make_group(net, SIZE, PredictionUnit(config=linear_config()))
        assert g.prediction.shape == (SIZE,)
        assert g.prediction_history.shape == (10, SIZE)
        assert g.prediction_weights.shape == (SIZE, SIZE)

    def test_self_prediction_follows_activity(self):
        # smoothing 0 so the new prediction is taken in full
        net = build_network()
        g, beh = make_group(net, SIZE, PredictionUnit(config=linear_config(temporal_smoothing=0.0)))
        g.activity = torch.full((SIZE,), 2.0)
        beh.forward(g)
        # prediction_weights is the identity, so prediction tracks activity
        assert torch.allclose(g.prediction, torch.full((SIZE,), 2.0))

    def test_temporal_smoothing_lags(self):
        cfg = linear_config(temporal_smoothing=0.9)
        net = build_network()
        g, beh = make_group(net, SIZE, PredictionUnit(config=cfg))
        g.activity = torch.full((SIZE,), 1.0)
        beh.forward(g)
        # 90% of the old (zero) prediction is kept
        assert torch.allclose(g.prediction, torch.full((SIZE,), 0.1))

    def test_history_wraps_at_ten(self):
        net = build_network()
        g, beh = make_group(net, SIZE, PredictionUnit(config=linear_config()))
        for i in range(12):
            g.activity = torch.full((SIZE,), float(i))
            beh.forward(g)
        assert g.history_idx == 12
        assert g.prediction_history[2].abs().sum() > 0  # slot 2 was written at i=2


class TestErrorUnit:
    def test_error_is_actual_minus_prediction(self):
        net = build_network()
        g, beh = make_group(net, SIZE, ErrorUnit(config=linear_config(), precision_weighted=False))
        g.activity = torch.full((SIZE,), 3.0)
        g.prediction = torch.full((SIZE,), 1.0)
        beh.forward(g)
        assert torch.allclose(g.signed_error, torch.full((SIZE,), 2.0))
        assert torch.allclose(g.error_magnitude, torch.full((SIZE,), 2.0))

    def test_perfect_prediction_gives_zero_error(self):
        net = build_network()
        g, beh = make_group(net, SIZE, ErrorUnit(config=linear_config(), precision_weighted=False))
        g.activity = torch.rand(SIZE)
        g.prediction = g.activity.clone()
        beh.forward(g)
        assert torch.allclose(g.signed_error, torch.zeros(SIZE))

    def test_relu_drops_negative_error(self):
        cfg = PredictiveCodingConfig(error_nonlinearity="relu", use_sparse_errors=False)
        net = build_network()
        g, beh = make_group(net, SIZE, ErrorUnit(config=cfg, precision_weighted=False))
        g.activity = torch.zeros(SIZE)
        g.prediction = torch.ones(SIZE)
        beh.forward(g)
        assert torch.all(g.prediction_error == 0)

    def test_sparse_errors_are_thresholded(self):
        cfg = PredictiveCodingConfig(error_nonlinearity="linear",
                                     use_sparse_errors=True, error_threshold=0.5)
        net = build_network()
        g, beh = make_group(net, SIZE, ErrorUnit(config=cfg, precision_weighted=False))
        g.activity = torch.cat([torch.full((8,), 0.1), torch.full((8,), 1.0)])
        g.prediction = torch.zeros(SIZE)
        beh.forward(g)
        assert torch.all(g.prediction_error[:8] == 0)
        assert torch.all(g.prediction_error[8:] > 0)

    def test_precision_weighting_scales_error(self):
        net = build_network()
        g, beh = make_group(net, SIZE, ErrorUnit(config=linear_config(), precision_weighted=True))
        g.activity = torch.full((SIZE,), 2.0)
        g.prediction = torch.zeros(SIZE)
        g.precision = torch.full((SIZE,), 3.0)
        beh.forward(g)
        assert torch.allclose(g.prediction_error, torch.full((SIZE,), 6.0))

    def test_precision_clamped_to_config_range(self):
        cfg = linear_config(min_precision=0.05, max_precision=4.0)
        net = build_network()
        g, beh = make_group(net, SIZE, ErrorUnit(config=cfg, precision_weighted=False))
        g.activity = torch.full((SIZE,), 100.0)
        g.prediction = torch.zeros(SIZE)
        for _ in range(20):
            beh.forward(g)
        assert g.precision.min() >= 0.05
        assert g.precision.max() <= 4.0


class TestPrecisionWeighting:
    def test_effective_precision_is_product(self):
        net = build_network()
        beh = PrecisionWeighting(config=linear_config(),
                                 learnable=False, attention_modulated=False)
        g, beh = make_group(net, SIZE, beh)
        g.sensory_precision = torch.full((SIZE,), 2.0)
        g.prior_precision = torch.full((SIZE,), 3.0)
        beh.forward(g)
        assert torch.allclose(g.effective_precision, torch.full((SIZE,), 6.0))

    def test_attention_scales_precision(self):
        net = build_network()
        beh = PrecisionWeighting(config=linear_config(),
                                 learnable=False, attention_modulated=True)
        g, beh = make_group(net, SIZE, beh)
        g.attention = torch.ones(SIZE)
        beh.forward(g)
        assert torch.allclose(g.effective_precision, torch.full((SIZE,), 2.0))

    def test_precision_written_back_to_error_state(self):
        net = build_network()
        beh = PrecisionWeighting(config=linear_config(),
                                 learnable=False, attention_modulated=False)
        g, beh = make_group(net, SIZE, beh)
        g.precision = torch.zeros(SIZE)
        beh.forward(g)
        assert torch.allclose(g.precision, torch.ones(SIZE))


class TestTopDownPrediction:
    def test_prediction_from_top_down_input(self):
        net = build_network()
        g, beh = make_group(net, SIZE, TopDownPrediction(config=linear_config()))
        g.top_down_input = torch.full((SIZE,), 2.0)
        beh.forward(g)
        # weights are 0.5 * identity
        assert torch.allclose(g.top_down_prediction, torch.full((SIZE,), 1.0))
        assert torch.allclose(g.prediction, g.top_down_prediction)

    def test_sigmoid_transform(self):
        net = build_network()
        g, beh = make_group(net, SIZE, TopDownPrediction(config=linear_config(), transform="sigmoid"))
        g.top_down_input = torch.zeros(SIZE)
        beh.forward(g)
        # weights are 0.5 * identity, so prediction is 0 and sigmoid(0) = 0.5
        assert torch.allclose(g.top_down_prediction, torch.full((SIZE,), 0.5))
        g.top_down_input = torch.full((SIZE,), 10.0)
        beh.forward(g)
        # sigmoid(5.0) = 0.9933
        assert torch.allclose(g.top_down_prediction, torch.full((SIZE,), 0.9933), atol=1e-3)


class TestPredictiveCodingLearning:
    def test_weights_move_on_error(self):
        net = build_network()
        beh = PredictiveCodingLearning(config=linear_config(), use_hebbian=False)
        g, beh = make_group(net, SIZE, beh)
        # prediction_weights is normally allocated by PredictionUnit
        g.prediction_weights = torch.eye(SIZE)
        g.activity = torch.full((SIZE,), 1.0)
        g.prediction_error = torch.full((SIZE,), 1.0)
        before = g.prediction_weights.clone()
        beh.forward(g)
        assert not torch.equal(before, g.prediction_weights)

    def test_no_update_without_error_state(self):
        net = build_network()
        g, beh = make_group(net, SIZE, PredictiveCodingLearning(config=linear_config()))
        # no prediction_weights / prediction_error attributes -> no-op
        beh.forward(g)
        assert not hasattr(g, "prediction_weights")


class TestFreeEnergyMinimization:
    def test_free_energy_matches_closed_form(self):
        net = build_network()
        g, beh = make_group(net, SIZE, FreeEnergyMinimization(config=linear_config()))
        g.signed_error = torch.full((SIZE,), 2.0)
        g.precision = torch.full((SIZE,), 1.0)
        beh.forward(g)
        # F = 0.5 * sum(precision * e^2) - 0.5 * sum(log(precision))
        expected = 0.5 * SIZE * 4.0 - 0.5 * SIZE * 0.0
        assert g.free_energy.item() == pytest.approx(expected)

    def test_no_error_is_noop(self):
        net = build_network()
        g, beh = make_group(net, SIZE, FreeEnergyMinimization(config=linear_config()))
        beh.forward(g)
        assert g.free_energy.item() == 0.0

    def test_history_capped_at_1000(self):
        net = build_network()
        g, beh = make_group(net, SIZE, FreeEnergyMinimization(config=linear_config()))
        g.signed_error = torch.zeros(SIZE)
        g.precision = torch.ones(SIZE)
        for _ in range(1005):
            beh.forward(g)
        assert len(g.free_energy_history) == 1000


class TestHierarchicalPredictiveCoding:
    def test_level_states_allocated(self):
        net = build_network()
        beh = HierarchicalPredictiveCoding(config=linear_config(),
                                           n_levels=3, level_sizes=[16, 8, 4])
        g, beh = make_group(net, 32, beh)
        assert beh.level_sizes == [16, 8, 4]
        assert len(g.level_predictions) == 3
        assert g.level_predictions[0].shape == (16,)
        assert g.feedforward_weights[0].shape == (16, 8)
        assert g.feedback_weights[0].shape == (8, 16)

    def test_forward_propagates_error_and_prediction(self):
        net = build_network()
        beh = HierarchicalPredictiveCoding(config=linear_config(),
                                           n_levels=3, level_sizes=[16, 8, 4])
        g, beh = make_group(net, 32, beh)
        g.activity = torch.zeros(32)
        g.activity[:16] = 1.0  # sensory input at the bottom level
        beh.forward(g)
        assert g.level_errors[0].abs().sum() > 0
        assert g.level_predictions[0].shape == (16,)

    def test_default_level_sizes(self):
        net = build_network()
        g, beh = make_group(net, 32, HierarchicalPredictiveCoding(config=linear_config(), n_levels=3))
        # default sizes are n_neurons // 2**i
        assert beh.level_sizes == [32, 16, 8]


class TestCreatePredictiveLayer:
    def test_returns_full_behavior_stack(self):
        behaviors = create_predictive_layer(SIZE, config=linear_config())
        names = [b.__class__.__name__ for b in behaviors.values()]
        assert names == ["PredictionUnit", "ErrorUnit", "PrecisionWeighting",
                         "PredictiveCodingLearning", "FreeEnergyMinimization"]

    def test_optional_components(self):
        behaviors = create_predictive_layer(SIZE, config=linear_config(),
                                            with_precision=False, with_learning=False,
                                            with_free_energy=False)
        names = [b.__class__.__name__ for b in behaviors.values()]
        assert names == ["PredictionUnit", "ErrorUnit"]

    def test_layer_runs_end_to_end(self):
        net = build_network(seed=9)
        g = NeuronGroup(net=net, size=SIZE, behavior=create_predictive_layer(SIZE, config=linear_config()), tag="full")
        net.initialize()
        g.activity = torch.rand(SIZE)
        net.simulate_iterations(5)
        assert g.free_energy.item() >= 0.0
        assert g.prediction.shape == (SIZE,)


class TestPredictionType:
    def test_values(self):
        assert PredictionType.FEEDFORWARD.value == "feedforward"
        assert PredictionType.FEEDBACK.value == "feedback"
        assert PredictionType.LATERAL.value == "lateral"
