"""Regression tests for the local 2D STDP weight update.

Local2dSTDP previously read ``pre_spike`` when building the post-synaptic
spike term. These tests pin the LTP term to ``post_spike``.
"""

from types import SimpleNamespace

import pytest
import torch

from conex import Local2dSTDP


SRC_SHAPE = (1, 4, 4)  # 16 pre-synaptic neurons
DST_SHAPE = (2, 2, 2)  # 8 post-synaptic neurons
KERNEL_SHAPE = (2, 2, 2, 1, 3, 3)
WEIGHT_SHAPE = (2, 4, 9)  # (out_channel, out_size, connection_size)


def make_synapse(pre_spike, post_spike, pre_trace, post_trace):
    return SimpleNamespace(
        network=SimpleNamespace(def_dtype=torch.float32),
        src_shape=SRC_SHAPE,
        dst_shape=DST_SHAPE,
        kernel_shape=KERNEL_SHAPE,
        stride=1,
        padding=0,
        weights=torch.zeros(WEIGHT_SHAPE),
        pre_spike=pre_spike,
        post_spike=post_spike,
        pre_trace=pre_trace,
        post_trace=post_trace,
    )


@pytest.fixture
def rule():
    learning = Local2dSTDP(a_plus=1.0, a_minus=1.0)
    return learning


def test_ltp_term_follows_post_spike(rule):
    """Only rows whose post-synaptic neuron spiked may receive potentiation."""
    post_spike = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
    synapse = make_synapse(
        pre_spike=torch.ones(16),
        post_spike=post_spike,
        pre_trace=torch.ones(16),
        post_trace=torch.zeros(8),  # kills the depression term
    )
    rule.initialize(synapse)

    dw = rule.compute_dw(synapse)

    assert dw.shape == WEIGHT_SHAPE
    expected = post_spike.view(2, 4, 1).expand(WEIGHT_SHAPE)
    assert torch.allclose(dw, expected)


def test_silent_post_neurons_get_no_potentiation(rule):
    synapse = make_synapse(
        pre_spike=torch.ones(16),
        post_spike=torch.zeros(8),
        pre_trace=torch.ones(16),
        post_trace=torch.zeros(8),
    )
    rule.initialize(synapse)
    assert torch.all(rule.compute_dw(synapse) == 0)


def test_pre_spike_alone_does_not_potentiate(rule):
    """Pre-synaptic activity without post-synaptic spikes must not raise weights."""
    synapse = make_synapse(
        pre_spike=torch.ones(16),
        post_spike=torch.zeros(8),
        pre_trace=torch.ones(16),
        post_trace=torch.ones(8),
    )
    rule.initialize(synapse)
    assert torch.all(rule.compute_dw(synapse) <= 0)


def test_depression_term_follows_post_trace(rule):
    synapse = make_synapse(
        pre_spike=torch.ones(16),
        post_spike=torch.zeros(8),
        pre_trace=torch.zeros(16),
        post_trace=torch.ones(8),
    )
    rule.initialize(synapse)
    dw = rule.compute_dw(synapse)
    assert torch.all(dw == -1.0)


def test_update_is_zero_without_activity(rule):
    synapse = make_synapse(
        pre_spike=torch.zeros(16),
        post_spike=torch.zeros(8),
        pre_trace=torch.zeros(16),
        post_trace=torch.zeros(8),
    )
    rule.initialize(synapse)
    assert torch.all(rule.compute_dw(synapse) == 0)


def test_forward_applies_update_to_weights(rule):
    synapse = make_synapse(
        pre_spike=torch.ones(16),
        post_spike=torch.ones(8),
        pre_trace=torch.ones(16),
        post_trace=torch.zeros(8),
    )
    rule.initialize(synapse)
    rule.forward(synapse)
    assert torch.all(synapse.weights == 1.0)
