"""Tests for the active dendrite behaviors."""

import pytest
import torch

from conex import (
    ActiveDendriteComputation,
    ContextualPrediction,
    DendriticSegment,
    DendriticSegmentConfig,
)


N_CELLS = 8
N_SEGMENTS = 2
N_SYNAPSES = 4


@pytest.fixture
def segments(make_group):
    config = DendriticSegmentConfig(
        n_synapses_per_segment=N_SYNAPSES,
        activation_threshold=2,
        learning_threshold=1,
        connected_permanence=0.5,
        initial_permanence=0.6,
        permanence_increment=0.1,
        permanence_decrement=0.05,
        max_new_synapses=2,
    )
    neurons = make_group(
        N_CELLS,
        DendriticSegment(n_segments=N_SEGMENTS, n_cells=N_CELLS, config=config),
        tag="segments",
    )
    return neurons


def wire_segment(neurons, segment, targets, permanence=0.9):
    """Point one segment's synapses at specific presynaptic cells."""
    for slot, target in enumerate(targets):
        neurons.segment_targets[segment, slot] = target
        neurons.segment_permanences[segment, slot] = permanence


class TestDendriticSegment:
    def test_state_shapes(self, segments):
        total = N_CELLS * N_SEGMENTS
        assert segments.segment_permanences.shape == (total, N_SYNAPSES)
        assert segments.segment_targets.shape == (total, N_SYNAPSES)
        assert segments.segment_activation.shape == (total,)
        assert segments.depolarization.shape == (N_CELLS,)

    def test_targets_start_unassigned(self, segments):
        assert torch.all(segments.segment_targets == -1)

    def test_segment_counts_active_connected_synapses(self, segments):
        wire_segment(segments, 0, [1, 2, 3])
        active = torch.zeros(N_CELLS, dtype=torch.bool)
        active[[1, 2, 3]] = True

        segments.dendritic_segment.compute_segment_activation(segments, active)

        assert segments.segment_activation[0].item() == 3
        assert segments.segment_active[0]

    def test_unconnected_synapses_do_not_count(self, segments):
        wire_segment(segments, 0, [1, 2, 3], permanence=0.1)  # below connected_perm
        active = torch.zeros(N_CELLS, dtype=torch.bool)
        active[[1, 2, 3]] = True

        segments.dendritic_segment.compute_segment_activation(segments, active)

        assert segments.segment_activation[0].item() == 0

    def test_inactive_presynaptic_cells_do_not_count(self, segments):
        wire_segment(segments, 0, [1, 2, 3])
        segments.dendritic_segment.compute_segment_activation(
            segments, torch.zeros(N_CELLS, dtype=torch.bool)
        )
        assert segments.segment_activation[0].item() == 0

    def test_below_threshold_is_matching_but_not_active(self, segments):
        wire_segment(segments, 0, [1])
        active = torch.zeros(N_CELLS, dtype=torch.bool)
        active[1] = True

        segments.dendritic_segment.compute_segment_activation(segments, active)

        assert not segments.segment_active[0]
        assert segments.matching_segments[0]

    def test_depolarization_follows_the_best_segment(self, segments):
        wire_segment(segments, 0, [1, 2])  # segment 0 belongs to cell 0
        active = torch.zeros(N_CELLS, dtype=torch.bool)
        active[[1, 2]] = True

        segments.dendritic_segment.compute_segment_activation(segments, active)
        segments.dendritic_segment.compute_depolarization(segments)

        assert segments.depolarization[0].item() == pytest.approx(1.0)
        assert segments.depolarization[1:].sum().item() == 0

    def test_depolarization_is_clamped_to_one(self, segments):
        wire_segment(segments, 0, [1, 2, 3, 4])  # four active, threshold is two
        active = torch.zeros(N_CELLS, dtype=torch.bool)
        active[[1, 2, 3, 4]] = True

        segments.dendritic_segment.compute_segment_activation(segments, active)
        segments.dendritic_segment.compute_depolarization(segments)

        assert segments.depolarization.max().item() <= 1.0

    def test_forward_runs_without_presynaptic_state(self, segments):
        segments.dendritic_segment.forward(segments)


@pytest.fixture
def dendrite(make_group):
    return make_group(
        N_CELLS,
        ActiveDendriteComputation(nmda_threshold=0.5, nmda_gain=2.0),
        tag="dendrite",
    )


class TestActiveDendriteComputation:
    def test_state_is_allocated(self, dendrite):
        for name in (
            "proximal_potential",
            "distal_potential",
            "apical_potential",
            "dendritic_calcium",
            "dendritic_spike",
        ):
            assert getattr(dendrite, name).shape == (N_CELLS,)

    def test_nmda_is_monotonic_and_superlinear(self, dendrite):
        module = dendrite.active_dendrite
        low = module.nmda_nonlinearity(torch.tensor([0.1]))
        high = module.nmda_nonlinearity(torch.tensor([1.0]))
        assert high > low
        # above threshold the gain more than doubles the input
        assert high.item() > 2.0

    def test_nmda_leaves_zero_at_zero(self, dendrite):
        assert dendrite.active_dendrite.nmda_nonlinearity(torch.zeros(3)).sum() == 0

    def test_proximal_input_drives_current(self, dendrite):
        current = dendrite.active_dendrite.compute_dendritic_integration(
            dendrite, proximal_input=torch.ones(N_CELLS)
        )
        assert current.shape == (N_CELLS,)
        assert torch.all(dendrite.proximal_potential > 0)
        assert torch.all(current > 0)

    def test_distal_input_amplifies_the_same_proximal_drive(self, dendrite):
        plain = dendrite.active_dendrite.compute_dendritic_integration(
            dendrite, proximal_input=torch.ones(N_CELLS)
        ).clone()

        dendrite.proximal_potential.zero_()
        dendrite.distal_potential.zero_()
        for _ in range(20):
            boosted = dendrite.active_dendrite.compute_dendritic_integration(
                dendrite,
                proximal_input=torch.ones(N_CELLS),
                distal_input=torch.full((N_CELLS,), 3.0),
            )

        assert torch.all(boosted > plain)

    def test_strong_distal_input_triggers_dendritic_spikes(self, dendrite):
        for _ in range(30):
            dendrite.active_dendrite.compute_dendritic_integration(
                dendrite,
                proximal_input=torch.zeros(N_CELLS),
                distal_input=torch.full((N_CELLS,), 5.0),
            )
        assert torch.all(dendrite.dendritic_spike)

    def test_no_spike_without_distal_or_apical_drive(self, dendrite):
        for _ in range(30):
            dendrite.active_dendrite.compute_dendritic_integration(
                dendrite, proximal_input=torch.ones(N_CELLS)
            )
        assert not dendrite.dendritic_spike.any()

    def test_backprop_raises_calcium(self, dendrite):
        before = dendrite.dendritic_calcium.clone()
        spikes = torch.zeros(N_CELLS, dtype=torch.bool)
        spikes[[0, 1]] = True

        dendrite.active_dendrite.apply_backprop(dendrite, spikes)

        assert torch.all(dendrite.dendritic_calcium[[0, 1]] > before[[0, 1]])
        assert torch.equal(dendrite.dendritic_calcium[2:], before[2:])

    def test_forward_writes_the_somatic_current(self, dendrite):
        dendrite.active_dendrite.forward(dendrite)
        assert dendrite.I.shape == (N_CELLS,)


@pytest.fixture
def context(make_group):
    return make_group(
        N_CELLS,
        ContextualPrediction(
            depolarization_threshold=0.5, prediction_window=3, burst_threshold=0.3
        ),
        tag="context",
    )


class TestContextualPrediction:
    def test_state_is_allocated(self, context):
        assert context.predictive.shape == (N_CELLS,)
        assert context.prediction_age.shape == (N_CELLS,)
        assert not context.predictive.any()

    def test_strong_context_makes_cells_predictive(self, context):
        drive = torch.zeros(N_CELLS)
        drive[[0, 1]] = 0.9

        context.contextual_prediction.update_predictions(context, drive)

        assert context.predictive[[0, 1]].all()
        assert not context.predictive[2:].any()

    def test_weak_context_leaves_cells_unpredicted(self, context):
        context.contextual_prediction.update_predictions(context, torch.full((N_CELLS,), 0.2))
        assert not context.predictive.any()

    def test_predictions_expire_after_the_window(self, context):
        drive = torch.zeros(N_CELLS)
        drive[0] = 0.9
        context.contextual_prediction.update_predictions(context, drive)
        assert context.predictive[0]

        for _ in range(4):
            context.contextual_prediction.update_predictions(context, torch.zeros(N_CELLS))

        assert not context.predictive[0]

    def test_unpredicted_activation_bursts(self, context):
        activation, bursting = context.contextual_prediction.process_activation(
            context, torch.ones(N_CELLS)
        )
        assert bursting.all()
        assert context.bursting.all()

    def test_predicted_cells_do_not_burst(self, context):
        drive = torch.zeros(N_CELLS)
        drive[[0, 1]] = 0.9
        context.contextual_prediction.update_predictions(context, drive)

        _, bursting = context.contextual_prediction.process_activation(
            context, torch.ones(N_CELLS)
        )

        assert not bursting[[0, 1]].any()
        assert bursting[2:].all()

    def test_bursting_cells_get_amplified(self, context):
        plain = torch.ones(N_CELLS)
        activation, bursting = context.contextual_prediction.process_activation(context, plain)
        assert torch.all(activation[bursting] > plain[bursting])

    def test_input_below_burst_threshold_does_nothing(self, context):
        _, bursting = context.contextual_prediction.process_activation(
            context, torch.full((N_CELLS,), 0.1)
        )
        assert not bursting.any()
