"""E2E tests for the dendritic input behaviors (spikes -> current)."""

import pytest
import torch
from pymonntorch import Network, NeuronDimension, NeuronGroup, SynapseGroup

from conex import TimeResolution, prioritize_behaviors
from conex.behaviors.neurons import (
    Fire,
    LIF,
    NeuronAxon,
    SimpleDendriteComputation,
    SimpleDendriteStructure,
)
from conex.behaviors.synapses import (
    AveragePool2D,
    Conv2dDendriticInput,
    LateralDendriticInput,
    Local2dDendriticInput,
    One2OneDendriticInput,
    PreSpikeCatcher,
    SimpleDendriticInput,
    SparseDendriticInput,
    SynapseInit,
    WeightInitializer,
)

from .conftest import make_network, seed


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


def _sxd_network():
    """A network in SxD (transposed) synapse mode, required by the dense/sparse
    dendritic inputs."""
    seed()
    return Network(
        behavior={1: TimeResolution(dt=1.0)},
        dtype=torch.float32,
        device="cpu",
        synapse_mode="SxD",
    )


# --- SimpleDendriticInput (full network) ----------------------------------


def test_simple_dendritic_input_current():
    """SxD mode: I = sum of weights of the fired pre-synaptic neurons.
    Weights all 1.0, src neurons 0 and 2 fire -> each dst gets 1.0+1.0 = 2.0."""
    net = _sxd_network()
    src = lif_group(net, "s", size=3)
    dst = lif_group(net, "d", size=2)
    syn = SynapseGroup(
        net=net,
        src=src,
        dst=dst,
        tag="Proximal",
        behavior=prioritize_behaviors(
            [
                SynapseInit(),
                WeightInitializer(mode=1.0),
                SimpleDendriticInput(current_coef=1.0),
                PreSpikeCatcher(),
            ]
        ),
    )
    net.initialize()
    syn.pre_spike = torch.tensor([True, False, True])
    syn.behavior[180].forward(syn)
    assert syn.I.tolist() == [2.0, 2.0]


# --- One2OneDendriticInput ------------------------------------------------


def test_one2one_dendritic_input():
    """I = pre_spike * weights (elementwise). pre=[1,0], w=[2,3] -> [2,0]."""
    net = make_network(dt=1.0)
    src = lif_group(net, "s", size=2)
    dst = lif_group(net, "d", size=2)
    o2o = One2OneDendriticInput(current_coef=1.0)
    syn = SynapseGroup(
        net=net,
        src=src,
        dst=dst,
        tag="Proximal",
        behavior=prioritize_behaviors(
            [SynapseInit(), WeightInitializer(mode=2.0), o2o, PreSpikeCatcher()]
        ),
    )
    net.initialize()
    syn.weights = torch.tensor([2.0, 3.0])
    syn.pre_spike = torch.tensor([1.0, 0.0])
    o2o.forward(syn)
    assert syn.I.tolist() == [2.0, 0.0]


def test_one2one_size_mismatch():
    """src and dst sizes must be equal for a one-to-one connection."""
    net = make_network(dt=1.0)
    src = lif_group(net, "s", size=3)
    dst = lif_group(net, "d", size=2)
    SynapseGroup(
        net=net,
        src=src,
        dst=dst,
        tag="Proximal",
        behavior=prioritize_behaviors(
            [
                SynapseInit(),
                WeightInitializer(mode=2.0),
                One2OneDendriticInput(),
                PreSpikeCatcher(),
            ]
        ),
    )
    with pytest.raises(RuntimeError):
        net.initialize()


# --- Inhibitory sign ------------------------------------------------------


def test_dendritic_input_inhibitory_sign():
    """A pre-synaptic group tagged 'inh' flips the current sign: I = -w*pre."""
    net = make_network(dt=1.0)
    src = NeuronGroup(
        net=net,
        size=2,
        tag="s,inh",
        behavior=prioritize_behaviors(
            [
                SimpleDendriteStructure(),
                SimpleDendriteComputation(),
                LIF(R=5.0, threshold=1.0, tau=10.0, v_reset=0.0, v_rest=0.0),
                Fire(),
                NeuronAxon(),
            ]
        ),
    )
    dst = lif_group(net, "d", size=2)
    o2o = One2OneDendriticInput(current_coef=1.0)
    syn = SynapseGroup(
        net=net,
        src=src,
        dst=dst,
        tag="Proximal",
        behavior=prioritize_behaviors(
            [SynapseInit(), WeightInitializer(mode=2.0), o2o, PreSpikeCatcher()]
        ),
    )
    net.initialize()
    syn.weights = torch.tensor([2.0, 3.0])
    syn.pre_spike = torch.tensor([1.0, 0.0])
    o2o.forward(syn)
    assert syn.I.tolist() == [-2.0, 0.0]


# --- SparseDendriticInput -------------------------------------------------


def test_sparse_dendritic_input():
    """SxD mode: I = pre_spike @ weights. pre=[1,1,0], w=ones(3x2) -> [2,2]."""
    net = _sxd_network()
    src = lif_group(net, "s", size=3)
    dst = lif_group(net, "d", size=2)
    sp = SparseDendriticInput(current_coef=1.0)
    syn = SynapseGroup(
        net=net,
        src=src,
        dst=dst,
        tag="Proximal",
        behavior=prioritize_behaviors(
            [SynapseInit(), WeightInitializer(mode=1.0), sp, PreSpikeCatcher()]
        ),
    )
    net.initialize()
    syn.pre_spike = torch.tensor([1.0, 1.0, 0.0])
    sp.forward(syn)
    assert syn.I.tolist() == [2.0, 2.0]


# --- Conv2dDendriticInput -------------------------------------------------


def test_conv2d_dendritic_input():
    """3x3 kernel of ones over a 4x4 all-spike input -> each of the 2x2 outputs
    sums 9 ones."""
    net = make_network(dt=1.0)
    src = lif_group(net, "s", size=NeuronDimension(depth=1, height=4, width=4))
    dst = lif_group(net, "d", size=NeuronDimension(depth=1, height=2, width=2))
    conv = Conv2dDendriticInput(current_coef=1.0, stride=1, padding=0)
    syn = SynapseGroup(
        net=net,
        src=src,
        dst=dst,
        tag="Proximal",
        behavior=prioritize_behaviors(
            [
                SynapseInit(),
                WeightInitializer(mode=1.0, weight_shape=(1, 1, 3, 3)),
                conv,
                PreSpikeCatcher(),
            ]
        ),
    )
    net.initialize()
    syn.pre_spike = torch.ones(16)
    conv.forward(syn)
    assert syn.I.tolist() == [9.0, 9.0, 9.0, 9.0]


def test_conv2d_bad_channels():
    """Weight input channels must match the source depth."""
    net = make_network(dt=1.0)
    src = lif_group(net, "s", size=NeuronDimension(depth=1, height=4, width=4))
    dst = lif_group(net, "d", size=NeuronDimension(depth=1, height=2, width=2))
    SynapseGroup(
        net=net,
        src=src,
        dst=dst,
        tag="Proximal",
        behavior=prioritize_behaviors(
            [
                SynapseInit(),
                WeightInitializer(mode=1.0, weight_shape=(1, 2, 3, 3)),  # 2 in-channels
                Conv2dDendriticInput(current_coef=1.0),
                PreSpikeCatcher(),
            ]
        ),
    )
    with pytest.raises(RuntimeError):
        net.initialize()


# --- AveragePool2D --------------------------------------------------------


def test_average_pool2d():
    """adaptive_avg_pool2d of a 4x4 all-spike input to 2x2 -> each output is 1.0."""
    net = make_network(dt=1.0)
    src = lif_group(net, "s", size=NeuronDimension(depth=1, height=4, width=4))
    dst = lif_group(net, "d", size=NeuronDimension(depth=1, height=2, width=2))
    pool = AveragePool2D(current_coef=1.0)
    syn = SynapseGroup(
        net=net,
        src=src,
        dst=dst,
        tag="Proximal",
        behavior=prioritize_behaviors(
            [
                SynapseInit(),
                WeightInitializer(mode=1.0, weight_shape=(1, 1, 3, 3)),
                pool,
                PreSpikeCatcher(),
            ]
        ),
    )
    net.initialize()
    syn.pre_spike = torch.ones(16)
    pool.forward(syn)
    assert syn.I.tolist() == [1.0, 1.0, 1.0, 1.0]


# --- LateralDendriticInput ------------------------------------------------


def test_lateral_dendritic_input():
    """A 3x3 lateral kernel of ones over a 4x4 all-spike input: corners sum 4,
    edges 6, centers 9 (conv3d with same padding)."""
    net = make_network(dt=1.0)
    pop = lif_group(net, "s", size=NeuronDimension(depth=1, height=4, width=4))
    lat = LateralDendriticInput(current_coef=1.0)
    syn = SynapseGroup(
        net=net,
        src=pop,
        dst=pop,
        tag="Proximal",
        behavior=prioritize_behaviors(
            [
                SynapseInit(),
                WeightInitializer(mode=1.0, weight_shape=(1, 1, 1, 3, 3)),
                lat,
                PreSpikeCatcher(),
            ]
        ),
    )
    net.initialize()
    syn.pre_spike = torch.ones(16)
    lat.forward(syn)
    assert syn.I.tolist() == [4, 6, 6, 4, 6, 9, 9, 6, 6, 9, 9, 6, 4, 6, 6, 4]


# --- Local2dDendriticInput ------------------------------------------------


def test_local2d_dendritic_input():
    """A 3x3 local kernel of ones over a 4x4 all-spike input -> each of the 2x2
    outputs sums 9 ones."""
    net = make_network(dt=1.0)
    src = lif_group(net, "s", size=NeuronDimension(depth=1, height=4, width=4))
    dst = lif_group(net, "d", size=NeuronDimension(depth=1, height=2, width=2))
    loc = Local2dDendriticInput(current_coef=1.0, stride=1, padding=0)
    syn = SynapseGroup(
        net=net,
        src=src,
        dst=dst,
        tag="Proximal",
        behavior=prioritize_behaviors(
            [
                SynapseInit(),
                WeightInitializer(
                    mode=1.0, weight_shape=(1, 4, 9), kernel_shape=(1, 2, 2, 1, 3, 3)
                ),
                loc,
                PreSpikeCatcher(),
            ]
        ),
    )
    net.initialize()
    syn.pre_spike = torch.ones(16)
    loc.forward(syn)
    assert syn.I.tolist() == [9.0, 9.0, 9.0, 9.0]
