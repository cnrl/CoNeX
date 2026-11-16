"""End-to-end tests for the predictive synapse behaviors.

Local build_network helper: the shared tests/e2e/conftest.py is owned by
another worker and not yet merged, so each file carries its own builder.
"""

import torch

from conex import (
    PredictiveSynapseInit,
    FeedbackPredictionSynapse,
    FeedforwardErrorSynapse,
    LateralPredictionSynapse,
    PredictiveSynapseLearning,
    PredictiveConnection,
    PredictiveCodingConfig,
    TimeResolution,
)
from pymonntorch import Network, NeuronGroup, SynapseGroup


def build_network(seed=0):
    torch.manual_seed(seed)
    net = Network(behavior={1: TimeResolution(dt=1.0)}, dtype=torch.float32, device="cpu")
    return net


def make_group(net, size, tag="pop"):
    g = NeuronGroup(net=net, size=size, behavior={}, tag=tag)
    return g


def make_synapse(net, src, dst, behavior, tag="syn"):
    s = SynapseGroup(net=net, src=src, dst=dst, behavior=behavior, tag=tag)
    net.initialize()
    return s, behavior


def cfg(**overrides):
    settings = dict(error_nonlinearity="linear", use_sparse_errors=False)
    settings.update(overrides)
    return PredictiveCodingConfig(**settings)


SRC, DST = 8, 6


class TestPredictiveSynapseInit:
    def test_feedforward_state(self):
        net = build_network(seed=1)
        src, dst = make_group(net, SRC, "src"), make_group(net, DST, "dst")
        beh = {100: PredictiveSynapseInit(config=cfg(),
                                          connection_type="feedforward")}
        s, beh = make_synapse(net, src, dst, beh)
        assert s.weights.shape == (SRC, DST)
        assert s.eligibility.shape == (SRC, DST)
        assert s.is_feedforward and not s.is_feedback and not s.is_lateral
        assert s.error_output.shape == (DST,)
        assert dst.bottom_up_error.shape == (DST,)  # buffer created for dst

    def test_feedback_state(self):
        net = build_network(seed=2)
        src, dst = make_group(net, SRC, "src"), make_group(net, DST, "dst")
        beh = {100: PredictiveSynapseInit(config=cfg(),
                                          connection_type="feedback")}
        s, beh = make_synapse(net, src, dst, beh)
        assert s.is_feedback
        assert s.prediction_output.shape == (DST,)

    def test_lateral_state(self):
        net = build_network(seed=3)
        src, dst = make_group(net, SRC, "src"), make_group(net, DST, "dst")
        beh = {100: PredictiveSynapseInit(config=cfg(),
                                          connection_type="lateral")}
        s, beh = make_synapse(net, src, dst, beh)
        assert s.is_lateral


class TestFeedbackPredictionSynapse:
    def test_prediction_reaches_destination(self):
        net = build_network(seed=4)
        src, dst = make_group(net, SRC, "src"), make_group(net, DST, "dst")
        s, beh = make_synapse(net, src, dst, {
            100: PredictiveSynapseInit(config=cfg(), connection_type="feedback"),
            110: FeedbackPredictionSynapse(config=cfg()),
        })
        dst.prediction = torch.zeros(DST)  # PredictionUnit allocates this in a real hierarchy
        src.activity = torch.full((SRC,), 2.0)
        beh[110].forward(s)
        expected = torch.matmul(s.weights.T, src.activity)
        assert torch.allclose(s.prediction_output, expected)
        assert torch.allclose(dst.prediction, expected)

    def test_delayed_prediction_lags(self):
        net = build_network(seed=5)
        src, dst = make_group(net, SRC, "src"), make_group(net, DST, "dst")
        s, beh = make_synapse(net, src, dst, {
            100: PredictiveSynapseInit(config=cfg(), connection_type="feedback"),
            110: FeedbackPredictionSynapse(config=cfg(), delay=2),
        })
        assert s.prediction_buffer.shape == (2, DST)
        # the ring buffer lags by `delay` steps, so the first two outputs are
        # the stale zero slots and the third emits step 1's prediction
        src.activity = torch.full((SRC,), 1.0)
        beh[110].forward(s)
        assert torch.allclose(s.prediction_output, torch.zeros(DST))
        src.activity = torch.full((SRC,), 2.0)
        beh[110].forward(s)
        assert torch.allclose(s.prediction_output, torch.zeros(DST))
        src.activity = torch.full((SRC,), 3.0)
        beh[110].forward(s)
        assert torch.allclose(s.prediction_output, torch.matmul(s.weights.T, torch.full((SRC,), 1.0)))


class TestFeedforwardErrorSynapse:
    def test_error_reaches_destination(self):
        net = build_network(seed=6)
        src, dst = make_group(net, SRC, "src"), make_group(net, DST, "dst")
        s, beh = make_synapse(net, src, dst, {
            100: PredictiveSynapseInit(config=cfg(), connection_type="feedforward"),
            110: FeedforwardErrorSynapse(config=cfg(), precision_weighted=False),
        })
        src.activity = torch.zeros(SRC)  # getattr default is evaluated eagerly
        src.prediction_error = torch.full((SRC,), 3.0)
        beh[110].forward(s)
        expected = torch.matmul(s.weights.T, src.prediction_error)
        assert torch.allclose(s.error_output, expected)
        assert torch.allclose(dst.bottom_up_error, expected)

    def test_precision_weighting(self):
        net = build_network(seed=7)
        src, dst = make_group(net, SRC, "src"), make_group(net, DST, "dst")
        s, beh = make_synapse(net, src, dst, {
            100: PredictiveSynapseInit(config=cfg(), connection_type="feedforward"),
            110: FeedforwardErrorSynapse(config=cfg(), precision_weighted=True),
        })
        src.activity = torch.zeros(SRC)
        src.prediction_error = torch.full((SRC,), 2.0)
        src.precision = torch.full((SRC,), 0.5)
        beh[110].forward(s)
        expected = torch.matmul(s.weights.T, torch.full((SRC,), 1.0))
        assert torch.allclose(s.error_output, expected)

    def test_falls_back_to_activity_without_error(self):
        net = build_network(seed=8)
        src, dst = make_group(net, SRC, "src"), make_group(net, DST, "dst")
        s, beh = make_synapse(net, src, dst, {
            100: PredictiveSynapseInit(config=cfg(), connection_type="feedforward"),
            110: FeedforwardErrorSynapse(config=cfg(), precision_weighted=False),
        })
        src.activity = torch.full((SRC,), 1.0)
        beh[110].forward(s)
        assert torch.allclose(s.error_output, torch.matmul(s.weights.T, src.activity))


class TestLateralPredictionSynapse:
    def test_lateral_context(self):
        net = build_network(seed=9)
        src, dst = make_group(net, SRC, "src"), make_group(net, DST, "dst")
        s, beh = make_synapse(net, src, dst, {
            100: PredictiveSynapseInit(config=cfg(), connection_type="lateral"),
            110: LateralPredictionSynapse(config=cfg()),
        })
        dst.lateral_context = torch.zeros(DST)  # buffer the lateral input lands in
        src.activity = torch.full((SRC,), 1.0)
        beh[110].forward(s)
        assert torch.allclose(s.dst.lateral_context, torch.matmul(s.weights.T, src.activity))

    def test_inhibitory_is_nonpositive(self):
        net = build_network(seed=10)
        src, dst = make_group(net, SRC, "src"), make_group(net, DST, "dst")
        s, beh = make_synapse(net, src, dst, {
            100: PredictiveSynapseInit(config=cfg(), connection_type="lateral"),
            110: LateralPredictionSynapse(config=cfg(), inhibitory=True),
        })
        dst.lateral_context = torch.zeros(DST)
        src.activity = torch.full((SRC,), 1.0)
        beh[110].forward(s)
        assert (s.dst.lateral_context <= 0).all()


class TestPredictiveSynapseLearning:
    def test_error_driven_updates_weights(self):
        net = build_network(seed=11)
        src, dst = make_group(net, SRC, "src"), make_group(net, DST, "dst")
        s, beh = make_synapse(net, src, dst, {
            100: PredictiveSynapseInit(config=cfg(), connection_type="feedforward"),
            120: PredictiveSynapseLearning(config=cfg(), learning_type="error_driven"),
        })
        src.activity = torch.full((SRC,), 1.0)
        dst.prediction_error = torch.full((DST,), 1.0)
        before = s.weights.clone()
        beh[120].forward(s)
        assert not torch.equal(before, s.weights)
        assert s.eligibility.abs().sum() > 0

    def test_hebbian_uses_activity(self):
        net = build_network(seed=12)
        src, dst = make_group(net, SRC, "src"), make_group(net, DST, "dst")
        s, beh = make_synapse(net, src, dst, {
            100: PredictiveSynapseInit(config=cfg(), connection_type="feedforward"),
            120: PredictiveSynapseLearning(config=cfg(), learning_type="hebbian"),
        })
        src.activity = torch.full((SRC,), 1.0)
        dst.activity = torch.full((DST,), 1.0)
        before = s.weights.clone()
        beh[120].forward(s)
        assert not torch.equal(before, s.weights)


class TestPredictiveConnection:
    def test_feedback_factory(self):
        net = build_network(seed=13)
        src, dst = make_group(net, SRC, "src"), make_group(net, DST, "dst")
        conn = PredictiveConnection.feedback(src, dst, config=cfg())
        assert conn["src"] is src and conn["dst"] is dst
        assert {b.__class__.__name__ for b in conn["behaviors"].values()} == {
            "PredictiveSynapseInit", "FeedbackPredictionSynapse", "PredictiveSynapseLearning"}

    def test_feedforward_factory(self):
        net = build_network(seed=14)
        src, dst = make_group(net, SRC, "src"), make_group(net, DST, "dst")
        conn = PredictiveConnection.feedforward(src, dst, config=cfg(), with_learning=False)
        assert {b.__class__.__name__ for b in conn["behaviors"].values()} == {
            "PredictiveSynapseInit", "FeedforwardErrorSynapse"}

    def test_lateral_factory(self):
        net = build_network(seed=15)
        layer = make_group(net, SRC, "layer")
        conn = PredictiveConnection.lateral(layer, config=cfg(), inhibitory=True)
        assert conn["src"] is layer and conn["dst"] is layer
        # inhibitory is bound in initialize, so check the constructor kwarg
        assert conn["behaviors"][110].init_kwargs["inhibitory"] is True

    def test_factory_behaviors_run_on_real_synapse(self):
        net = build_network(seed=16)
        src, dst = make_group(net, SRC, "src"), make_group(net, DST, "dst")
        conn = PredictiveConnection.feedforward(src, dst, config=cfg())
        s, beh = make_synapse(net, src, dst, conn["behaviors"])
        src.activity = torch.full((SRC,), 1.0)
        src.prediction_error = torch.full((SRC,), 0.5)
        net.simulate_iterations(2)
        assert dst.bottom_up_error.shape == (DST,)
        assert s.error_output.abs().sum() > 0
