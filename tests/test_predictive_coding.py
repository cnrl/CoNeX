"""Tests for the predictive coding behaviors.

These behaviors read ``neurons.activity`` and are driven by explicit forward()
calls rather than by the spiking pipeline, so the tests set that state directly.
"""

import pytest
import torch

from conex import (
    ErrorUnit,
    FreeEnergyMinimization,
    PrecisionWeighting,
    PredictionUnit,
    PredictiveCodingConfig,
    PredictiveCodingLearning,
    TopDownPrediction,
)


SIZE = 16


@pytest.fixture
def build(make_group):
    """Attach one behavior to a fresh group and return (neurons, behavior)."""

    def _build(behavior):
        neurons = make_group(SIZE, behavior, tag="pc")
        neurons.activity = torch.zeros(SIZE)
        return neurons, behavior

    return _build


def linear_config(**overrides):
    settings = dict(error_nonlinearity="linear", use_sparse_errors=False)
    settings.update(overrides)
    return PredictiveCodingConfig(**settings)


class TestErrorUnit:
    def test_error_is_actual_minus_prediction(self, build):
        neurons, unit = build(ErrorUnit(config=linear_config(), precision_weighted=False))
        neurons.activity = torch.full((SIZE,), 3.0)
        neurons.prediction = torch.full((SIZE,), 1.0)
        unit.forward(neurons)
        assert torch.allclose(neurons.signed_error, torch.full((SIZE,), 2.0))

    def test_perfect_prediction_gives_zero_error(self, build):
        neurons, unit = build(ErrorUnit(config=linear_config(), precision_weighted=False))
        neurons.activity = torch.rand(SIZE)
        neurons.prediction = neurons.activity.clone()
        unit.forward(neurons)
        assert torch.allclose(neurons.signed_error, torch.zeros(SIZE))

    def test_error_magnitude_is_absolute(self, build):
        neurons, unit = build(ErrorUnit(config=linear_config(), precision_weighted=False))
        neurons.activity = torch.zeros(SIZE)
        neurons.prediction = torch.full((SIZE,), 2.0)
        unit.forward(neurons)
        assert torch.allclose(neurons.signed_error, torch.full((SIZE,), -2.0))
        assert torch.allclose(neurons.error_magnitude, torch.full((SIZE,), 2.0))

    def test_relu_nonlinearity_drops_negative_error(self, build):
        config = PredictiveCodingConfig(error_nonlinearity="relu", use_sparse_errors=False)
        neurons, unit = build(ErrorUnit(config=config, precision_weighted=False))
        neurons.activity = torch.zeros(SIZE)
        neurons.prediction = torch.ones(SIZE)
        unit.forward(neurons)
        assert torch.all(neurons.prediction_error == 0)

    def test_sparse_errors_are_thresholded(self, build):
        config = PredictiveCodingConfig(
            error_nonlinearity="linear", use_sparse_errors=True, error_threshold=0.5
        )
        neurons, unit = build(ErrorUnit(config=config, precision_weighted=False))
        neurons.activity = torch.cat([torch.full((8,), 0.1), torch.full((8,), 1.0)])
        neurons.prediction = torch.zeros(SIZE)
        unit.forward(neurons)
        assert torch.all(neurons.prediction_error[:8] == 0)
        assert torch.all(neurons.prediction_error[8:] > 0)

    def test_precision_rises_when_error_is_consistent(self, build):
        neurons, unit = build(ErrorUnit(config=linear_config(), precision_weighted=False))
        neurons.activity = torch.zeros(SIZE)
        neurons.prediction = torch.zeros(SIZE)
        for _ in range(50):
            unit.forward(neurons)
        # zero error -> variance collapses -> precision saturates at max
        assert torch.all(neurons.precision > 1.0)

    def test_precision_is_clamped_to_config_range(self, build):
        config = linear_config(min_precision=0.05, max_precision=4.0)
        neurons, unit = build(ErrorUnit(config=config, precision_weighted=False))
        neurons.activity = torch.full((SIZE,), 100.0)
        neurons.prediction = torch.zeros(SIZE)
        for _ in range(20):
            unit.forward(neurons)
        assert neurons.precision.min() >= 0.05
        assert neurons.precision.max() <= 4.0


class TestPrecisionWeighting:
    def test_effective_precision_is_product_of_components(self, build):
        neurons, unit = build(
            PrecisionWeighting(config=linear_config(), learnable=False, attention_modulated=False)
        )
        neurons.sensory_precision = torch.full((SIZE,), 2.0)
        neurons.prior_precision = torch.full((SIZE,), 3.0)
        unit.forward(neurons)
        assert torch.allclose(neurons.effective_precision, torch.full((SIZE,), 6.0))

    def test_attention_scales_precision(self, build):
        neurons, unit = build(
            PrecisionWeighting(config=linear_config(), learnable=False, attention_modulated=True)
        )
        neurons.attention = torch.ones(SIZE)
        unit.forward(neurons)
        assert torch.allclose(neurons.effective_precision, torch.full((SIZE,), 2.0))

    def test_precision_clamped_to_range(self, build):
        config = linear_config(min_precision=0.5, max_precision=1.5)
        neurons, unit = build(
            PrecisionWeighting(config=config, learnable=False, attention_modulated=False)
        )
        neurons.sensory_precision = torch.full((SIZE,), 100.0)
        neurons.prior_precision = torch.full((SIZE,), 100.0)
        unit.forward(neurons)
        assert torch.all(neurons.effective_precision == 1.5)


class TestPredictionUnit:
    def test_prediction_state_is_allocated(self, build):
        neurons, _ = build(PredictionUnit(config=linear_config()))
        assert neurons.prediction.shape == (SIZE,)
        assert neurons.prediction_weights.shape == (SIZE, SIZE)
        assert neurons.prediction_history.shape == (10, SIZE)

    def test_smoothing_moves_prediction_towards_target(self, build):
        config = linear_config(temporal_smoothing=0.5)
        neurons, unit = build(PredictionUnit(config=config))
        neurons.top_down_input = torch.ones(SIZE)
        unit.forward(neurons)
        # identity weights, so target is the top-down input itself
        assert torch.allclose(neurons.prediction, torch.full((SIZE,), 0.5))
        unit.forward(neurons)
        assert torch.allclose(neurons.prediction, torch.full((SIZE,), 0.75))

    def test_history_is_written(self, build):
        neurons, unit = build(PredictionUnit(config=linear_config()))
        neurons.top_down_input = torch.ones(SIZE)
        unit.forward(neurons)
        assert neurons.history_idx == 1
        assert torch.allclose(neurons.prediction_history[0], neurons.prediction)


class TestTopDownPrediction:
    def test_sets_prediction_from_top_down_input(self, build):
        neurons, unit = build(TopDownPrediction(config=linear_config()))
        neurons.top_down_input = torch.ones(SIZE)
        unit.forward(neurons)
        # weights are 0.5 * identity
        assert torch.allclose(neurons.prediction, torch.full((SIZE,), 0.5))
        assert torch.allclose(neurons.prediction, neurons.top_down_prediction)

    def test_sigmoid_transform_bounds_output(self, build):
        neurons, unit = build(TopDownPrediction(config=linear_config(), transform="sigmoid"))
        neurons.top_down_input = torch.full((SIZE,), 50.0)
        unit.forward(neurons)
        assert torch.all(neurons.top_down_prediction > 0.5)
        assert torch.all(neurons.top_down_prediction <= 1.0)


class TestPredictiveCodingLearning:
    """The top-down update must follow prediction = top_down_weights.T @ top_down.

    An update built from outer(error, top_down) rather than outer(top_down, error)
    is transposed: it stalls at a non-zero error instead of converging.
    """

    def learning_chain(self, make_group, config):
        neurons = make_group(SIZE, TopDownPrediction(config=config), tag="pc")
        neurons.activity = torch.zeros(SIZE)
        error_unit = ErrorUnit(config=config, precision_weighted=False)
        learning = PredictiveCodingLearning(config=config, use_hebbian=False)
        for behavior in (error_unit, learning):
            behavior.initialize(neurons)
        return neurons, [neurons.behavior[100], error_unit, learning]

    def test_error_converges_to_zero(self, make_group):
        torch.manual_seed(3)
        config = linear_config(learning_rate=0.05)
        neurons, chain = self.learning_chain(make_group, config)
        neurons.activity = torch.rand(SIZE)
        neurons.top_down_input = torch.rand(SIZE)

        for _ in range(60):
            for behavior in chain:
                behavior.forward(neurons)

        assert neurons.error_magnitude.mean().item() < 1e-4

    def test_error_decreases_monotonically(self, make_group):
        torch.manual_seed(3)
        config = linear_config(learning_rate=0.05)
        neurons, chain = self.learning_chain(make_group, config)
        neurons.activity = torch.rand(SIZE)
        neurons.top_down_input = torch.rand(SIZE)

        errors = []
        for _ in range(10):
            for behavior in chain:
                behavior.forward(neurons)
            errors.append(neurons.error_magnitude.mean().item())

        assert all(b < a for a, b in zip(errors, errors[1:]))

    def test_no_update_without_error_state(self, make_group):
        config = linear_config()
        neurons = make_group(SIZE, TopDownPrediction(config=config), tag="pc")
        learning = PredictiveCodingLearning(config=config)
        learning.initialize(neurons)
        before = neurons.top_down_weights.clone()
        learning.forward(neurons)
        assert torch.equal(before, neurons.top_down_weights)


class TestFreeEnergy:
    def test_zero_error_with_unit_precision_gives_zero_free_energy(self, build):
        neurons, unit = build(FreeEnergyMinimization(config=linear_config()))
        neurons.signed_error = torch.zeros(SIZE)
        neurons.precision = torch.ones(SIZE)
        unit.forward(neurons)
        assert neurons.free_energy.item() == pytest.approx(0.0, abs=1e-5)

    def test_free_energy_grows_with_error(self, build):
        neurons, unit = build(FreeEnergyMinimization(config=linear_config()))
        neurons.precision = torch.ones(SIZE)
        neurons.signed_error = torch.full((SIZE,), 1.0)
        unit.forward(neurons)
        small = neurons.free_energy.item()
        neurons.signed_error = torch.full((SIZE,), 5.0)
        unit.forward(neurons)
        assert neurons.free_energy.item() > small

    def test_accuracy_term_matches_formula(self, build):
        neurons, unit = build(FreeEnergyMinimization(config=linear_config()))
        neurons.precision = torch.full((SIZE,), 2.0)
        neurons.signed_error = torch.full((SIZE,), 3.0)
        unit.forward(neurons)
        assert neurons.accuracy_term.item() == pytest.approx(0.5 * SIZE * 2.0 * 9.0)

    def test_history_accumulates(self, build):
        neurons, unit = build(FreeEnergyMinimization(config=linear_config()))
        neurons.signed_error = torch.ones(SIZE)
        neurons.precision = torch.ones(SIZE)
        for _ in range(3):
            unit.forward(neurons)
        assert len(neurons.free_energy_history) == 3

    def test_no_error_state_is_a_noop(self, build):
        neurons, unit = build(FreeEnergyMinimization(config=linear_config()))
        unit.forward(neurons)
        assert neurons.free_energy_history == []
