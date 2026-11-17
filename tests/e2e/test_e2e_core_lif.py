"""E2E tests for the LIF / ELIF / AELIF neuron dynamics."""

import torch
from pymonntorch import NeuronGroup

from conex import prioritize_behaviors
from conex.behaviors.neurons import (
    AELIF,
    ELIF,
    Fire,
    LIF,
    NeuronAxon,
    SimpleDendriteComputation,
    SimpleDendriteStructure,
)

from .conftest import make_network


def lif_group(net, tag, **lif_kwargs):
    """A single LIF population wired with dendrite + fire + axon."""
    return NeuronGroup(
        net=net,
        size=1,
        tag=tag,
        behavior=prioritize_behaviors(
            [
                SimpleDendriteStructure(),
                SimpleDendriteComputation(),
                LIF(**lif_kwargs),
                Fire(),
                NeuronAxon(),
            ]
        ),
    )


def test_lif_integrates_and_fires():
    """v=1.5, v_rest=0, tau=10, dt=1: v += (0-1.5)*1/10 = 1.35 >= 1.0 -> spike,
    reset to 0."""
    net = make_network(dt=1.0)
    n = lif_group(
        net,
        "lif",
        R=5.0,
        threshold=1.0,
        tau=10.0,
        v_reset=0.0,
        v_rest=0.0,
    )
    net.initialize()
    n.v[0] = 1.5
    net.simulate_iterations(1)
    assert n.spikes[0].item() is True
    assert n.v[0].item() == 0.0


def test_lif_no_fire_below_threshold():
    """v=0.9, v_rest=0, tau=10: v += (0-0.9)/10 = 0.81 < 1.0, no spike."""
    net = make_network(dt=1.0)
    n = lif_group(
        net,
        "lif",
        R=5.0,
        threshold=1.0,
        tau=10.0,
        v_reset=0.0,
        v_rest=0.0,
    )
    net.initialize()
    n.v[0] = 0.9
    net.simulate_iterations(1)
    assert n.spikes[0].item() is False
    assert n.v[0].item() == torch.tensor(0.81).item()


def test_lif_current_drives_voltage():
    """I=1.0, R=5, v_rest=0, v=0, tau=10, dt=1: v += (0 + 5*1.0)*1/10 = 0.5.
    No dendrite computation here, so the manually-set I survives to LIF."""
    net = make_network(dt=1.0)
    n = NeuronGroup(
        net=net,
        size=1,
        tag="lif",
        behavior=prioritize_behaviors(
            [
                LIF(R=5.0, threshold=1.0, tau=10.0, v_reset=0.0, v_rest=0.0),
                Fire(),
                NeuronAxon(),
            ]
        ),
    )
    net.initialize()
    n.I = torch.tensor([1.0])  # noqa: E741  (library attribute name)
    net.simulate_iterations(1)
    assert n.spikes[0].item() is False
    assert n.v[0].item() == torch.tensor(0.5).item()


def test_elif_exponential_boost():
    """ELIF adds delta*exp((v-theta_rh)/delta) to the leak. With v=theta_rh the
    boost is exactly delta, so the step is larger than a plain LIF there."""
    net = make_network(dt=1.0)
    n = NeuronGroup(
        net=net,
        size=1,
        tag="elif",
        behavior=prioritize_behaviors(
            [
                SimpleDendriteStructure(),
                SimpleDendriteComputation(),
                ELIF(
                    R=5.0,
                    threshold=1.0,
                    tau=10.0,
                    v_reset=0.0,
                    v_rest=0.0,
                    delta=2.0,
                    theta_rh=0.0,
                ),
                Fire(),
                NeuronAxon(),
            ]
        ),
    )
    net.initialize()
    n.v[0] = 0.0  # v == theta_rh, so the exponential term is delta = 2.0
    net.simulate_iterations(1)
    # F(u) = (0 - 0) + 2.0*exp(0) = 2.0; v += 2.0*1/10 = 0.2 (vs 0.0 for LIF)
    assert n.v[0].item() == torch.tensor(0.2).item()


def test_aelif_adaptation_after_spike():
    """AELIF adds an adaptation current omega that grows on a spike. After one
    spike omega > 0, which lowers the voltage relative to a plain LIF."""
    net = make_network(dt=1.0)
    n = NeuronGroup(
        net=net,
        size=1,
        tag="aelif",
        behavior=prioritize_behaviors(
            [
                SimpleDendriteStructure(),
                SimpleDendriteComputation(),
                AELIF(
                    R=5.0,
                    threshold=1.0,
                    tau=10.0,
                    v_reset=0.0,
                    v_rest=0.0,
                    delta=2.0,
                    theta_rh=0.0,
                    alpha=0.1,
                    beta=0.5,
                    w_tau=10.0,
                ),
                Fire(),
                NeuronAxon(),
            ]
        ),
    )
    net.initialize()
    n.v[0] = 1.5  # forces a spike so the adaptation current is triggered
    net.simulate_iterations(1)
    assert n.spikes[0].item() is True
    # spike_adaptation = beta*w_tau*spikes = 0.5*10*1 = 5.0
    # domega = 5.0/10 = 0.5
    assert n.omega[0].item() == torch.tensor(0.5).item()
