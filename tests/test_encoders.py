"""Tests for SDR encoding, classification, and location coding."""

import math

import pytest
import torch

from conex import (
    DisplacementCellModule,
    LocationEncoder,
    SDR,
    SDRClassifier,
    SDREncoder,
)


SIZE = 64
N_ACTIVE = 8


@pytest.fixture
def scalar_encoder(make_group):
    return make_group(
        SIZE,
        SDREncoder(
            size=SIZE, n_active=N_ACTIVE, encoder_type="scalar", min_val=0.0, max_val=1.0
        ),
        tag="encoder",
    )


class TestSDREncoder:
    def test_scalar_encoding_has_the_requested_sparsity(self, scalar_encoder):
        code = scalar_encoder.sdr_encoder.encode_scalar(0.5)
        assert isinstance(code, SDR)
        assert code.n_active == N_ACTIVE

    def test_scalar_encoding_is_deterministic(self, scalar_encoder):
        first = scalar_encoder.sdr_encoder.encode_scalar(0.3)
        second = scalar_encoder.sdr_encoder.encode_scalar(0.3)
        assert first == second

    def test_nearby_values_overlap_more_than_distant_ones(self, scalar_encoder):
        encode = scalar_encoder.sdr_encoder.encode_scalar
        base = encode(0.5)
        near = encode(0.52)
        far = encode(0.0)
        assert base.overlap(near) > base.overlap(far)

    def test_values_are_clamped_to_the_configured_range(self, scalar_encoder):
        encode = scalar_encoder.sdr_encoder.encode_scalar
        assert encode(-5.0) == encode(0.0)
        assert encode(5.0) == encode(1.0)

    def test_category_encoding_is_stable_and_distinct(self, make_group):
        neurons = make_group(
            SIZE,
            SDREncoder(size=SIZE, n_active=N_ACTIVE, encoder_type="category", n_categories=4),
            tag="cat",
        )
        encode = neurons.sdr_encoder.encode_category
        assert encode(1) == encode(1)
        assert encode(1) != encode(2)

    def test_out_of_range_category_is_rejected(self, make_group):
        neurons = make_group(
            SIZE,
            SDREncoder(size=SIZE, n_active=N_ACTIVE, encoder_type="category", n_categories=3),
            tag="cat",
        )
        with pytest.raises(ValueError, match="out of range"):
            neurons.sdr_encoder.encode_category(7)

    def test_random_encoding_is_seed_stable(self, scalar_encoder):
        encode = scalar_encoder.sdr_encoder.encode_random
        assert encode(42) == encode(42)
        assert encode(42) != encode(43)
        assert encode(42).n_active == N_ACTIVE


@pytest.fixture
def classifier(make_group):
    return make_group(
        SIZE, SDRClassifier(n_classes=3, sdr_size=SIZE), tag="classifier"
    )


class TestSDRClassifier:
    def test_state_shapes(self, classifier):
        assert classifier.classifier_weights.shape == (3, SIZE)
        assert classifier.class_counts.shape == (3,)

    def test_learning_records_the_example(self, classifier):
        pattern = SDR(size=SIZE, indices=torch.arange(0, 8))
        classifier.sdr_classifier.learn(classifier, pattern, label=1)
        assert classifier.class_counts[1] == 1
        assert classifier.classifier_weights[1, :8].sum() > 0

    def test_learned_patterns_are_recovered(self, classifier):
        patterns = {
            0: SDR(size=SIZE, indices=torch.arange(0, 8)),
            1: SDR(size=SIZE, indices=torch.arange(20, 28)),
            2: SDR(size=SIZE, indices=torch.arange(40, 48)),
        }
        for label, pattern in patterns.items():
            for _ in range(5):
                classifier.sdr_classifier.learn(classifier, pattern, label)

        for label, pattern in patterns.items():
            predicted, scores = classifier.sdr_classifier.infer(classifier, pattern)
            assert predicted == label
            assert scores.shape == (3,)

    def test_dense_tensors_are_accepted(self, classifier):
        dense = torch.zeros(SIZE)
        dense[:8] = 1.0
        classifier.sdr_classifier.learn(classifier, dense, label=2)
        predicted, _ = classifier.sdr_classifier.infer(classifier, dense)
        assert predicted == 2

    def test_untrained_classifier_does_not_crash(self, classifier):
        predicted, scores = classifier.sdr_classifier.infer(
            classifier, SDR(size=SIZE, indices=torch.arange(0, 8))
        )
        assert 0 <= predicted < 3


N_MODULES = 3
CELLS_PER_MODULE = 8
N_CELLS = N_MODULES * CELLS_PER_MODULE


N_DISPLACEMENT_CELLS = 25


@pytest.fixture
def displacement(make_group):
    return make_group(
        N_DISPLACEMENT_CELLS,
        DisplacementCellModule(n_cells=N_DISPLACEMENT_CELLS, max_displacement=4.0),
        tag="displacement",
    )


class TestDisplacementCellModule:
    def test_state_is_allocated(self, displacement):
        assert displacement.displacement_activation.shape == (N_DISPLACEMENT_CELLS,)
        assert displacement.previous_location is None

    def test_encoding_a_displacement_is_bounded(self, displacement):
        code = displacement.displacement_module.encode_displacement(
            displacement, torch.tensor([1.0, 0.5])
        )
        assert code.shape == (N_DISPLACEMENT_CELLS,)
        assert torch.isfinite(code).all()
        assert code.min() >= 0.0

    def test_same_displacement_gives_the_same_code(self, displacement):
        encode = displacement.displacement_module.encode_displacement
        step = torch.tensor([0.7, -0.3])
        assert torch.allclose(encode(displacement, step), encode(displacement, step))

    def test_different_displacements_differ(self, displacement):
        encode = displacement.displacement_module.encode_displacement
        a = encode(displacement, torch.tensor([1.0, 0.0]))
        b = encode(displacement, torch.tensor([0.0, 1.0]))
        assert not torch.allclose(a, b)

    def test_compute_displacement_encodes_the_difference(self, displacement):
        module = displacement.displacement_module
        moved = module.compute_displacement(
            displacement, torch.tensor([1.0, 1.0]), torch.tensor([3.0, 4.0])
        )
        direct = module.encode_displacement(displacement, torch.tensor([2.0, 3.0]))
        assert torch.allclose(moved, direct)

    def test_forward_tracks_movement_from_grid_position(self, displacement):
        displacement.grid_position = torch.tensor([0.0, 0.0])
        displacement.displacement_module.forward(displacement)
        assert displacement.previous_location is not None

        displacement.grid_position = torch.tensor([1.0, 0.0])
        displacement.displacement_module.forward(displacement)
        assert displacement.displacement_activation.sum() > 0


@pytest.fixture
def location(make_group):
    return make_group(N_CELLS, LocationEncoder(), tag="location")


class TestLocationEncoder:
    def test_state_is_allocated(self, location):
        assert location.location_representation.shape[0] > 0
        assert location.active_locations.numel() > 0

    def test_encoding_is_deterministic(self, location):
        encode = location.location_encoder.encode_location
        position = torch.tensor([1.5, -0.5])
        assert torch.allclose(encode(location, position), encode(location, position))

    def test_different_positions_give_different_codes(self, location):
        encode = location.location_encoder.encode_location
        a = encode(location, torch.tensor([0.0, 0.0]))
        b = encode(location, torch.tensor([2.3, 1.7]))
        assert not torch.allclose(a, b)

    def test_forward_refreshes_the_representation(self, location):
        location.location_encoder.forward(location)
        assert torch.isfinite(location.location_representation).all()
