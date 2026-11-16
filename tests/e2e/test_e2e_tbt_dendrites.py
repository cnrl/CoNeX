"""End-to-end tests for the active dendrite behaviors.

Local build_network helper: the shared tests/e2e/conftest.py is owned by
another worker and not yet merged, so each file carries its own builder.
"""

import pytest
import torch

from conex import (
    DendriticSegment,
    DendriticSegmentConfig,
    ActiveDendriteComputation,
    ContextualPrediction,
    TimeResolution,
)
from pymonntorch import Network, NeuronGroup


def build_network(seed=0):
    torch.manual_seed(seed)
    net = Network(behavior={1: TimeResolution(dt=1.0)}, dtype=torch.float32, device="cpu")
    return net


def make_group(net, size, behavior, tag="dend"):
    g = NeuronGroup(net=net, size=size, behavior={100: behavior}, tag=tag)
    net.initialize()
    return g


N_CELLS = 8
N_SEGMENTS = 4
N_SYNAPSES = 20
# presynaptic population the segments point into; must exceed the target indices
N_PRE = 32


@pytest.fixture
def segment_group():
    cfg = DendriticSegmentConfig(
        n_synapses_per_segment=N_SYNAPSES,
        activation_threshold=10,
        learning_threshold=8,
        connected_permanence=0.2,
    )
    net = build_network(seed=2)
    g = make_group(net, N_CELLS, DendriticSegment(n_segments=N_SEGMENTS, n_cells=N_CELLS, config=cfg))
    return g


def wire_segment(g, cell=0, seg=0, n_active=12):
    """Give one segment connected synapses to cells 0..n_active-1."""
    idx = cell * N_SEGMENTS + seg
    g.segment_permanences[idx, :n_active] = 0.5
    g.segment_targets[idx, :n_active] = torch.arange(n_active)


class TestDendriticSegment:
    def test_state_shapes(self, segment_group):
        g = segment_group
        assert g.segment_permanences.shape == (N_CELLS * N_SEGMENTS, N_SYNAPSES)
        assert g.segment_targets.shape == (N_CELLS * N_SEGMENTS, N_SYNAPSES)
        assert (g.segment_targets == -1).all()  # no connections yet

    def test_connected_synapses_drive_activation(self, segment_group):
        g = segment_group
        wire_segment(g, n_active=12)
        presyn = torch.zeros(N_PRE, dtype=torch.bool)
        presyn[:12] = True
        g.dendritic_segment.compute_segment_activation(g, presyn)
        assert g.segment_activation[0].item() == 12
        assert g.segment_active[0]  # 12 >= activation_threshold 10
        assert g.matching_segments[0]  # 12 >= learning_threshold 8

    def test_unconnected_synapses_do_not_count(self, segment_group):
        g = segment_group
        wire_segment(g, n_active=12)
        g.segment_permanences[0, :12] = 0.1  # below connected_permanence 0.2
        presyn = torch.zeros(N_PRE, dtype=torch.bool)
        presyn[:12] = True
        g.dendritic_segment.compute_segment_activation(g, presyn)
        assert g.segment_activation[0].item() == 0
        assert not g.segment_active[0]

    def test_invalid_targets_are_ignored(self, segment_group):
        g = segment_group
        wire_segment(g, n_active=12)
        g.segment_targets[0, :12] = -1
        presyn = torch.zeros(N_PRE, dtype=torch.bool)
        presyn[:12] = True
        g.dendritic_segment.compute_segment_activation(g, presyn)
        assert g.segment_activation[0].item() == 0

    def test_depolarization_is_max_segment_over_threshold(self, segment_group):
        g = segment_group
        wire_segment(g, cell=1, seg=2, n_active=12)
        presyn = torch.zeros(N_PRE, dtype=torch.bool)
        presyn[:12] = True
        g.dendritic_segment.compute_segment_activation(g, presyn)
        g.dendritic_segment.compute_depolarization(g)
        assert g.depolarization[1].item() == pytest.approx(1.0)  # 12/10 clamped
        assert g.depolarization[0].item() == 0.0

    def test_learning_strengthened_and_grows(self, segment_group):
        g = segment_group
        wire_segment(g, n_active=12)
        presyn = torch.zeros(N_PRE, dtype=torch.bool)
        presyn[:12] = True
        g.dendritic_segment.compute_segment_activation(g, presyn)
        # winner_cells is indexed by presynaptic cell, so it has N_PRE entries
        winners = torch.zeros(N_PRE, dtype=torch.bool)
        winners[0] = True
        before = g.segment_permanences[0, :12].clone()
        g.dendritic_segment.learn_on_segments(g, winners, winners, presyn)
        # active targets strengthened, empty slots may grow to active cells
        assert g.segment_permanences[0, :12].min() >= before.min()
        assert g.segment_permanences[0].max() <= 1.0

    def test_no_learning_without_matching_winner(self, segment_group):
        g = segment_group
        wire_segment(g, n_active=12)
        presyn = torch.zeros(N_PRE, dtype=torch.bool)
        presyn[:12] = True
        g.dendritic_segment.compute_segment_activation(g, presyn)
        before = g.segment_permanences.clone()
        winners = torch.zeros(N_PRE, dtype=torch.bool)  # no winner
        g.dendritic_segment.learn_on_segments(g, winners, winners, presyn)
        assert torch.equal(before, g.segment_permanences)


@pytest.fixture
def ad_group():
    net = build_network(seed=3)
    g = make_group(net, N_CELLS, ActiveDendriteComputation(nmda_threshold=0.5, nmda_gain=2.0))
    return g


class TestActiveDendriteComputation:
    def test_nmda_nonlinearity_amplifies_above_threshold(self, ad_group):
        g = ad_group
        beh = g.active_dendrite
        low = beh.nmda_nonlinearity(torch.full((N_CELLS,), 0.1))
        high = beh.nmda_nonlinearity(torch.full((N_CELLS,), 1.0))
        assert (low < 0.1 * 1.5).all()  # subthreshold: barely amplified
        assert (high > 1.0 * 2.0).all()  # superthreshold: gain kicks in

    def test_integration_produces_current_and_spike(self, ad_group):
        g = ad_group
        beh = g.active_dendrite
        # the distal potential starts at 0 and only moves by dt/tau per step,
        # so set it directly to test the superthreshold branch
        g.distal_potential = torch.full((N_CELLS,), 1.0)
        total = beh.compute_dendritic_integration(g, torch.full((N_CELLS,), 1.0),
                                                  torch.full((N_CELLS,), 1.0))
        assert total.shape == (N_CELLS,)
        assert g.dendritic_spike.any()  # strong distal input triggers a spike
        assert g.dendritic_calcium.min() >= 0.0

    def test_integration_without_distal(self, ad_group):
        g = ad_group
        beh = g.active_dendrite
        total = beh.compute_dendritic_integration(g, torch.full((N_CELLS,), 0.5))
        assert total.shape == (N_CELLS,)
        assert not g.dendritic_spike.any()

    def test_backprop_raises_calcium(self, ad_group):
        g = ad_group
        beh = g.active_dendrite
        beh.compute_dendritic_integration(g, torch.zeros(N_CELLS))
        before = g.dendritic_calcium.clone()
        spikes = torch.zeros(N_CELLS, dtype=torch.bool)
        spikes[0] = True
        beh.apply_backprop(g, spikes)
        assert g.dendritic_calcium[0] > before[0]


@pytest.fixture
def ctx_group():
    net = build_network(seed=4)
    beh = ContextualPrediction(
        depolarization_threshold=0.5, prediction_window=3, burst_threshold=0.3)
    return make_group(net, N_CELLS, beh)


class TestContextualPrediction:
    def test_predictive_state_from_depolarization(self, ctx_group):
        g = ctx_group
        g.depolarization = torch.full((N_CELLS,), 0.9)
        g.contextual_prediction.update_predictions(g, torch.zeros(N_CELLS))
        assert g.predictive.all()
        assert (g.prediction_age == 0).all()

    def test_predictions_expire_after_window(self, ctx_group):
        g = ctx_group
        g.depolarization = torch.full((N_CELLS,), 0.9)
        g.contextual_prediction.update_predictions(g, torch.zeros(N_CELLS))
        # keep depolarizing to hold the prediction, then drop it
        g.depolarization = torch.full((N_CELLS,), 0.9)
        g.contextual_prediction.update_predictions(g, torch.zeros(N_CELLS))
        g.depolarization = torch.zeros(N_CELLS)
        for _ in range(4):  # window is 3
            g.contextual_prediction.update_predictions(g, torch.zeros(N_CELLS))
        assert not g.predictive.any()

    def test_unpredicted_activation_bursts(self, ctx_group):
        g = ctx_group
        g.contextual_prediction.update_predictions(g, torch.zeros(N_CELLS))  # nothing predictive
        ff = torch.full((N_CELLS,), 0.8)
        activation, bursting = g.contextual_prediction.process_activation(g, ff)
        assert bursting.all()
        assert torch.allclose(activation, ff * 1.5)  # burst amplification

    def test_predicted_activation_is_not_amplified(self, ctx_group):
        g = ctx_group
        g.depolarization = torch.full((N_CELLS,), 0.9)
        g.contextual_prediction.update_predictions(g, torch.zeros(N_CELLS))
        ff = torch.full((N_CELLS,), 0.8)
        activation, bursting = g.contextual_prediction.process_activation(g, ff)
        assert not bursting.any()
        assert torch.allclose(activation, ff)

    def test_below_burst_threshold_does_not_activate(self, ctx_group):
        g = ctx_group
        g.contextual_prediction.update_predictions(g, torch.zeros(N_CELLS))
        ff = torch.full((N_CELLS,), 0.1)  # below burst_threshold 0.3
        activation, bursting = g.contextual_prediction.process_activation(g, ff)
        assert not bursting.any()
        assert torch.allclose(activation, ff)
