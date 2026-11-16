"""E2E tests for synapse specs: init, delays, weights, catchers, traces."""

import pytest
import torch
from pymonntorch import NeuronGroup, SynapseGroup

from conex import TimeResolution, prioritize_behaviors
from conex.behaviors.neurons import (
    Fire,
    LIF,
    NeuronAxon,
    SimpleDendriteComputation,
    SimpleDendriteStructure,
)
from conex.behaviors.synapses import (
    DelayInitializer,
    PostSpikeCatcher,
    PostTrace,
    PreSpikeCatcher,
    PreTrace,
    SimpleDendriticInput,
    SynapseInit,
    WeightClip,
    WeightInitializer,
    WeightNormalization,
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


def pair(net, tag, syn_beh, src_size=3, dst_size=2, max_delay=1):
    src = lif_group(net, tag + "_s", size=src_size, max_delay=max_delay)
    dst = lif_group(net, tag + "_d", size=dst_size, max_delay=max_delay)
    syn = SynapseGroup(
        net=net, src=src, dst=dst, tag="Proximal", behavior=prioritize_behaviors(syn_beh)
    )
    return src, dst, syn


# --- SynapseInit ----------------------------------------------------------


def test_synapse_init_delays():
    """SynapseInit allocates zero src/dst delay tensors sized to the populations."""
    net = make_network(dt=1.0)
    _, _, syn = pair(net, "si", [SynapseInit()])
    net.initialize()
    assert syn.src_delay.tolist() == [0, 0, 0]
    assert syn.dst_delay.tolist() == [0, 0]


# --- DelayInitializer -----------------------------------------------------


def test_delay_initializer_mode():
    """mode=1 fills every src delay with 1."""
    net = make_network(dt=1.0)
    _, _, syn = pair(
        net,
        "di",
        [
            SynapseInit(),
            DelayInitializer(mode=1),
            SimpleDendriticInput(current_coef=1.0),
            PreSpikeCatcher(),
        ],
        max_delay=2,
    )
    net.initialize()
    assert syn.src_delay.tolist() == [1, 1, 1]


def test_delay_initializer_offset_and_scale():
    """mode=1, scale=2, offset=1 -> every delay is 1*2+1 = 3."""
    net = make_network(dt=1.0)
    _, _, syn = pair(
        net,
        "di",
        [
            SynapseInit(),
            DelayInitializer(mode=1, scale=2, offset=1),
            SimpleDendriticInput(current_coef=1.0),
            PreSpikeCatcher(),
        ],
        max_delay=4,
    )
    net.initialize()
    assert syn.src_delay.tolist() == [3, 3, 3]


# --- WeightInitializer ----------------------------------------------------


def test_weight_initializer_mode():
    """mode=0.3 fills the whole weight matrix with 0.3."""
    net = make_network(dt=1.0)
    _, _, syn = pair(net, "wi", [SynapseInit(), WeightInitializer(mode=0.3)])
    net.initialize()
    assert syn.weights.shape == (3, 2)
    assert torch.allclose(syn.weights, torch.full((3, 2), 0.3))


def test_weight_initializer_scale_offset():
    """mode=1, scale=2, offset=0.5 -> every weight is 1*2+0.5 = 2.5."""
    net = make_network(dt=1.0)
    _, _, syn = pair(
        net, "wi", [SynapseInit(), WeightInitializer(mode=1, scale=2, offset=0.5)]
    )
    net.initialize()
    assert torch.allclose(syn.weights, torch.full((3, 2), 2.5))


def test_weight_initializer_sparse_density():
    """density=0.5 with true_sparsity=True builds a sparse matrix with ~half the nnz."""
    net = make_network(dt=1.0)
    _, _, syn = pair(
        net,
        "wi",
        [SynapseInit(), WeightInitializer(mode=1.0, density=0.5, true_sparsity=True)],
    )
    net.initialize()
    # 3x2 = 6 entries, density 0.5 -> 3 non-zero, stored in sparse_csc layout
    assert syn.weights.layout == torch.sparse_csc
    assert syn.weights._nnz() == 3


# --- WeightNormalization / WeightClip -------------------------------------


def test_weight_normalization():
    """After one step each destination column sums to norm=1."""
    net = make_network(dt=1.0)
    _, _, syn = pair(
        net,
        "wn",
        [
            SynapseInit(),
            WeightInitializer(mode=0.5),
            SimpleDendriticInput(current_coef=1.0),
            PreSpikeCatcher(),
            WeightNormalization(norm=1),
        ],
    )
    net.initialize()
    net.simulate_iterations(1)
    assert torch.allclose(syn.weights.sum(dim=0), torch.ones(2))


def test_weight_clip():
    """Weights are clipped into [w_min, w_max] each step."""
    net = make_network(dt=1.0)
    _, _, syn = pair(
        net,
        "wc",
        [
            SynapseInit(),
            WeightInitializer(mode=0.5),
            SimpleDendriticInput(current_coef=1.0),
            PreSpikeCatcher(),
            WeightClip(w_min=0.0, w_max=0.2),
        ],
    )
    net.initialize()
    net.simulate_iterations(1)
    assert syn.weights.max().item() <= 0.2 + 1e-6
    assert syn.weights.min().item() >= 0.0


def test_weight_clip_bad_interval():
    """w_min >= w_max raises on initialize."""
    net = make_network(dt=1.0)
    _, _, syn = pair(
        net,
        "wc",
        [
            SynapseInit(),
            WeightInitializer(mode=0.5),
            SimpleDendriticInput(current_coef=1.0),
            PreSpikeCatcher(),
            WeightClip(w_min=1.0, w_max=0.0),
        ],
    )
    with pytest.raises(AssertionError):
        net.initialize()


# --- Spike catchers and traces --------------------------------------------


def test_spike_catchers():
    """Pre/PostSpikeCatcher copy the (delayed) spikes of src and dst."""
    net = make_network(dt=1.0)
    src, dst, syn = pair(
        net,
        "sc",
        [
            SynapseInit(),
            WeightInitializer(mode=1.0),
            SimpleDendriticInput(current_coef=1.0),
            PreSpikeCatcher(),
            PostSpikeCatcher(),
        ],
        src_size=2,
        dst_size=2,
    )
    net.initialize()
    src.v = torch.tensor([2.0, 0.0])
    dst.v = torch.tensor([0.0, 2.0])
    net.simulate_iterations(1)
    assert syn.pre_spike.tolist() == [True, False]
    assert syn.post_spike.tolist() == [False, True]


def test_traces_decay():
    """trace += spike; trace -= trace/tau*dt. With tau=10, dt=1, one spike:
    1.0 - 1.0/10 = 0.9."""
    net = make_network(dt=1.0)
    src, dst, syn = pair(
        net,
        "tr",
        [
            SynapseInit(),
            WeightInitializer(mode=1.0),
            SimpleDendriticInput(current_coef=1.0),
            PreSpikeCatcher(),
            PostSpikeCatcher(),
            PreTrace(tau_s=10.0),
            PostTrace(tau_s=10.0),
        ],
        src_size=2,
        dst_size=2,
    )
    net.initialize()
    src.v = torch.tensor([2.0, 0.0])
    dst.v = torch.tensor([2.0, 0.0])
    net.simulate_iterations(1)
    assert syn.pre_trace[0].item() == pytest.approx(0.9, abs=1e-5)
    assert syn.post_trace[0].item() == pytest.approx(0.9, abs=1e-5)
    assert syn.pre_trace[1].item() == 0.0
