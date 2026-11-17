"""Regression tests for the One2One learning rules (fix-b, BUG 1).

A one-to-one connection has one weight per neuron pair, so its weight must be
1D. With a 2D weight the elementwise update row-broadcasts (matching no
sensible rule) and One2OneDendriticInput emits a 2D I that crashes the
network step. The rules now fail early in initialize() with a ValueError that
names the rule and the offending weight shape.
"""

import pytest
import torch
from pymonntorch import NeuronGroup, SynapseGroup

from conex import prioritize_behaviors
from conex.behaviors.neurons import (
    Fire,
    LIF,
    NeuronAxon,
    SimpleDendriteComputation,
    SimpleDendriteStructure,
)
from conex.behaviors.synapses import (
    One2OneDendriticInput,
    One2OneRSTDP,
    One2OneSTDP,
    One2OneiSTDP,
    PostSpikeCatcher,
    PostTrace,
    PreSpikeCatcher,
    PreTrace,
    SynapseInit,
    WeightInitializer,
)

from .conftest import make_network


def lif_group(net, tag, size=2, max_delay=1):
    return NeuronGroup(
        net=net,
        size=size,
        tag=tag,
        behavior=prioritize_behaviors(
            [
                SimpleDendriteStructure(),
                SimpleDendriteComputation(),
                LIF(R=5.0, threshold=1.0, tau=10.0, v_reset=0.0, v_rest=0.0),
                Fire(),
                NeuronAxon(max_delay=max_delay),
            ]
        ),
    )


def one2one_pair(net, rule, tag="o2o", weight_shape=None, with_traces=True):
    src = lif_group(net, tag + "_s", size=2)
    dst = lif_group(net, tag + "_d", size=2)
    behaviors = [
        SynapseInit(),
        WeightInitializer(mode=0.5, weight_shape=weight_shape),
        PreSpikeCatcher(),
        PostSpikeCatcher(),
    ]
    if with_traces:
        behaviors += [PreTrace(tau_s=10.0), PostTrace(tau_s=10.0)]
    behaviors.append(rule)
    syn = SynapseGroup(
        net=net,
        src=src,
        dst=dst,
        tag="Proximal",
        behavior=prioritize_behaviors(behaviors),
    )
    return src, dst, syn


# --- 2D weight must fail early ---------------------------------------------


@pytest.mark.parametrize(
    "rule",
    [
        One2OneSTDP(a_plus=0.1, a_minus=0.01),
        One2OneiSTDP(rho=0.5, alpha=1.0, lr=1.0, is_inhibitory=True),
        One2OneRSTDP(a_plus=0.1, a_minus=0.01, tau_c=10.0),
    ],
    ids=["One2OneSTDP", "One2OneiSTDP", "One2OneRSTDP"],
)
def test_one2one_rejects_2d_weight(rule):
    """A default 2D weight raises a ValueError naming the rule and shape."""
    net = make_network(dt=1.0)
    _, _, syn = one2one_pair(net, rule, weight_shape=None)
    with pytest.raises(ValueError) as exc:
        net.initialize()
    msg = str(exc.value)
    assert type(rule).__name__ in msg
    assert "(2, 2)" in msg


# --- 1D weight runs and learns ---------------------------------------------


def test_one2one_rstdp_1d_rewarded_pairing_potentiates():
    """With a 1D weight, a rewarded co-firing pairing potentiates its weight."""
    net = make_network(dt=1.0)
    net.dopamine_concentration = 1.0
    src = lif_group(net, "s", size=2)
    dst = lif_group(net, "d", size=2)
    syn = SynapseGroup(
        net=net,
        src=src,
        dst=dst,
        tag="Proximal",
        behavior=prioritize_behaviors(
            [
                SynapseInit(),
                WeightInitializer(mode=0.5, weight_shape=(2,)),
                One2OneDendriticInput(current_coef=1.0),
                PreSpikeCatcher(),
                PostSpikeCatcher(),
                PreTrace(tau_s=10.0),
                PostTrace(tau_s=10.0),
                One2OneRSTDP(a_plus=0.1, a_minus=0.01, tau_c=10.0),
            ]
        ),
    )
    net.initialize()
    assert syn.weights.dim() == 1
    w0 = syn.weights[0].item()
    for _ in range(5):
        src.v = torch.tensor([2.0, 0.0])
        dst.v = torch.tensor([2.0, 0.0])
        net.simulate_iterations(1)
    assert syn.weights[0].item() > w0
    # the unpaired neuron's weight is untouched
    assert syn.weights[1].item() == 0.5


def test_one2one_rstdp_1d_zero_dopamine_no_change():
    """With dopamine=0 the eligibility trace decays but weights never move."""
    net = make_network(dt=1.0)
    net.dopamine_concentration = 0.0
    src = lif_group(net, "s", size=2)
    dst = lif_group(net, "d", size=2)
    syn = SynapseGroup(
        net=net,
        src=src,
        dst=dst,
        tag="Proximal",
        behavior=prioritize_behaviors(
            [
                SynapseInit(),
                WeightInitializer(mode=0.5, weight_shape=(2,)),
                One2OneDendriticInput(current_coef=1.0),
                PreSpikeCatcher(),
                PostSpikeCatcher(),
                PreTrace(tau_s=10.0),
                PostTrace(tau_s=10.0),
                One2OneRSTDP(a_plus=0.1, a_minus=0.01, tau_c=10.0),
            ]
        ),
    )
    net.initialize()
    w0 = syn.weights.clone()
    for _ in range(5):
        src.v = torch.tensor([2.0, 0.0])
        dst.v = torch.tensor([2.0, 0.0])
        net.simulate_iterations(1)
    assert torch.allclose(syn.weights, w0)
