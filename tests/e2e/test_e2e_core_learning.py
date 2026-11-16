"""E2E tests for the STDP / iSTDP / RSTDP learning rules and weight bounds."""

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
    BOUNDS,
    One2OneSTDP,
    PostSpikeCatcher,
    PostTrace,
    PreSpikeCatcher,
    PreTrace,
    SimpleSTDP,
    SimpleiSTDP,
    SynapseInit,
    WeightInitializer,
    hard_bound,
    no_bound,
    soft_bound,
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


def stdp_pair(
    net, rule, tag="stdp", src_size=2, dst_size=2, with_traces=False
):
    """A 2x2 pair wired with the given rule (and traces if requested)."""
    from conex.behaviors.synapses import (
        PostSpikeCatcher,
        PostTrace,
        PreSpikeCatcher,
        PreTrace,
    )

    src = lif_group(net, tag + "_s", size=src_size)
    dst = lif_group(net, tag + "_d", size=dst_size)
    behaviors = [SynapseInit(), WeightInitializer(mode=0.5)]
    if with_traces:
        behaviors += [
            PreSpikeCatcher(),
            PostSpikeCatcher(),
            PreTrace(tau_s=10.0),
            PostTrace(tau_s=10.0),
        ]
    behaviors.append(rule)
    syn = SynapseGroup(
        net=net,
        src=src,
        dst=dst,
        tag="Proximal",
        behavior=prioritize_behaviors(behaviors),
    )
    return src, dst, syn


# --- Weight bound functions -----------------------------------------------


def test_bound_functions():
    """soft/hard/no bound return the documented values for a weight."""
    assert soft_bound(0.5, 0.0, 1.0) == 0.25  # (0.5-0)*(1-0.5)
    assert hard_bound(0.5, 0.0, 1.0) == 1
    assert hard_bound(1.5, 0.0, 1.0) == 0
    assert no_bound(0.5, 0.0, 1.0) == 1
    assert BOUNDS["soft_bound"] is soft_bound


# --- SimpleSTDP -----------------------------------------------------------


def test_simple_stdp_ltp():
    """LTP term: dw = outer(pre_trace, post_spike) * a_plus.
    pre_trace=[1,0], post_spike=[1,0], a_plus=0.1 -> [[0.1,0],[0,0]]."""
    net = make_network(dt=1.0)
    rule = SimpleSTDP(a_plus=0.1, a_minus=0.01)
    _, _, syn = stdp_pair(net, rule)
    net.initialize()
    syn.pre_trace = torch.tensor([1.0, 0.0])
    syn.post_spike = torch.tensor([1.0, 0.0])
    syn.pre_spike = torch.tensor([0.0, 0.0])
    syn.post_trace = torch.tensor([0.0, 0.0])
    dw = rule.compute_dw(syn)
    assert torch.allclose(dw, torch.tensor([[0.1, 0.0], [0.0, 0.0]]))


def test_simple_stdp_ltd():
    """LTD term: dw = -outer(pre_spike, post_trace) * a_minus.
    pre_spike=[1,0], post_trace=[1,0], a_minus=0.01 -> [[-0.01,0],[0,0]]."""
    net = make_network(dt=1.0)
    rule = SimpleSTDP(a_plus=0.1, a_minus=0.01)
    _, _, syn = stdp_pair(net, rule)
    net.initialize()
    syn.pre_spike = torch.tensor([1.0, 0.0])
    syn.post_trace = torch.tensor([1.0, 0.0])
    syn.pre_trace = torch.tensor([0.0, 0.0])
    syn.post_spike = torch.tensor([0.0, 0.0])
    dw = rule.compute_dw(syn)
    assert torch.allclose(dw, torch.tensor([[-0.01, 0.0], [0.0, 0.0]]))


def test_simple_stdp_full_network_potentiates():
    """Repeated co-firing of src0 and dst0 potentiates weight (0,0)."""
    from conex.behaviors.synapses import (
        PostSpikeCatcher,
        PostTrace,
        PreSpikeCatcher,
        PreTrace,
        SimpleDendriticInput,
    )

    net = make_network(dt=1.0)
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
                WeightInitializer(mode=0.5),
                SimpleDendriticInput(current_coef=1.0),
                PreSpikeCatcher(),
                PostSpikeCatcher(),
                PreTrace(tau_s=10.0),
                PostTrace(tau_s=10.0),
                SimpleSTDP(a_plus=0.1, a_minus=0.01),
            ]
        ),
    )
    net.initialize()
    w0 = syn.weights[0, 0].item()
    for _ in range(3):
        src.v = torch.tensor([2.0, 0.0])
        dst.v = torch.tensor([2.0, 0.0])
        net.simulate_iterations(1)
    assert syn.weights[0, 0].item() > w0


# --- One2OneSTDP ----------------------------------------------------------


def test_one2one_stdp():
    """Elementwise STDP. pre_trace=[1,0], post_spike=[1,0], a_plus=0.1
    -> dw = [0.1, 0] (the LTD term is zero with pre_spike=[0,0])."""
    net = make_network(dt=1.0)
    rule = One2OneSTDP(a_plus=0.1, a_minus=0.01)
    _, _, syn = stdp_pair(net, rule)
    net.initialize()
    syn.pre_trace = torch.tensor([1.0, 0.0])
    syn.post_spike = torch.tensor([1.0, 0.0])
    syn.pre_spike = torch.tensor([0.0, 0.0])
    syn.post_trace = torch.tensor([0.0, 0.0])
    dw = rule.compute_dw(syn)
    assert torch.allclose(dw, torch.tensor([0.1, 0.0]))


# --- SimpleiSTDP ----------------------------------------------------------


def test_simple_istdp():
    """iSTDP: dw = lr * (outer(pre_spike, (alpha-post_trace)*sign)
    + outer(pre_trace, post_spike)). is_inhibitory -> sign = 1-2 = -1.
    alpha=1, post_trace=[0,0] -> (alpha-post_trace)*sign = [-1,-1];
    pre_spike=[1,0], pre_trace=[0,0], post_spike=[0,0], lr=1
    -> outer([1,0],[-1,-1]) = [[-1,-1],[0,0]]."""
    net = make_network(dt=1.0)
    rule = SimpleiSTDP(rho=0.5, alpha=1.0, lr=1.0, is_inhibitory=True)
    _, _, syn = stdp_pair(net, rule, with_traces=True)
    net.initialize()
    syn.pre_spike = torch.tensor([1.0, 0.0])
    syn.post_trace = torch.tensor([0.0, 0.0])
    syn.pre_trace = torch.tensor([0.0, 0.0])
    syn.post_spike = torch.tensor([0.0, 0.0])
    dw = rule.compute_dw(syn)
    assert torch.allclose(dw, torch.tensor([[-1.0, -1.0], [0.0, 0.0]]))


# --- One2OneiSTDP ---------------------------------------------------------


def test_one2one_istdp():
    """Elementwise iSTDP: dw = lr*(pre_spike*(alpha-post_trace)*sign +
    pre_trace*post_spike). sign=-1, alpha=1, post_trace=0, pre_spike=[1,0],
    lr=1 -> [-1, 0]."""
    from conex.behaviors.synapses import One2OneiSTDP

    net = make_network(dt=1.0)
    rule = One2OneiSTDP(rho=0.5, alpha=1.0, lr=1.0, is_inhibitory=True)
    _, _, syn = stdp_pair(net, rule, with_traces=True)
    net.initialize()
    syn.pre_spike = torch.tensor([1.0, 0.0])
    syn.post_trace = torch.tensor([0.0, 0.0])
    syn.pre_trace = torch.tensor([0.0, 0.0])
    syn.post_spike = torch.tensor([0.0, 0.0])
    dw = rule.compute_dw(syn)
    assert torch.allclose(dw, torch.tensor([-1.0, 0.0]))


# --- SparseSTDP / SparseiSTDP ---------------------------------------------


def _sparse_pair(net, rule, tag="sp", include_rule=True):
    from conex.behaviors.synapses import (
        PostSpikeCatcher,
        PostTrace,
        PreSpikeCatcher,
        PreTrace,
    )

    src = lif_group(net, tag + "_s", size=2)
    dst = lif_group(net, tag + "_d", size=2)
    behaviors = [
        SynapseInit(),
        WeightInitializer(mode=0.5, density=1.0, true_sparsity=True),
        PreSpikeCatcher(),
        PostSpikeCatcher(),
        PreTrace(tau_s=10.0),
        PostTrace(tau_s=10.0),
    ]
    if include_rule:
        behaviors.append(rule)
    syn = SynapseGroup(
        net=net,
        src=src,
        dst=dst,
        tag="Proximal",
        behavior=prioritize_behaviors(behaviors),
    )
    return src, dst, syn


def _set_coo(syn):
    """Give the synapse a coalesced COO weight and per-entry src/dst indices,
    which the sparse rules read (WeightInitializer leaves a CSC tensor)."""
    idx = torch.tensor([[0, 1, 0, 1], [0, 0, 1, 1]])
    syn.weights = torch.sparse_coo_tensor(
        idx, torch.full((4,), 0.5), (2, 2)
    ).coalesce()
    syn.src_idx = torch.tensor([0, 1, 0, 1])
    syn.dst_idx = torch.tensor([0, 0, 1, 1])


def test_sparse_stdp():
    """SparseSTDP LTP on the (src0,dst0) entry: pre_trace[src0]=1,
    post_spike[dst0]=1, a_plus=0.1 -> first nnz entry 0.1, rest 0."""
    from conex.behaviors.synapses import SparseSTDP

    net = make_network(dt=1.0)
    rule = SparseSTDP(a_plus=0.1, a_minus=0.01)
    _, _, syn = _sparse_pair(net, rule)
    net.initialize()
    _set_coo(syn)
    syn.pre_trace = torch.tensor([1.0, 0.0])
    syn.post_spike = torch.tensor([1.0, 0.0])
    syn.pre_spike = torch.tensor([0.0, 0.0])
    syn.post_trace = torch.tensor([0.0, 0.0])
    dw = rule.compute_dw(syn)
    assert torch.allclose(dw, torch.tensor([0.1, 0.0, 0.0, 0.0]))


def test_sparse_istdp():
    """SparseiSTDP on the (src0,dst0) and (src1,dst1) entries.
    pre_spike=[1,0], post_trace=[0,0], sign=-1, lr=1 -> [-1, 0, -1, 0]."""
    from conex.behaviors.synapses import SparseiSTDP

    net = make_network(dt=1.0)
    rule = SparseiSTDP(rho=0.5, alpha=1.0, lr=1.0, is_inhibitory=True)
    _, _, syn = _sparse_pair(net, rule)
    net.initialize()
    _set_coo(syn)
    syn.pre_spike = torch.tensor([1.0, 0.0])
    syn.post_trace = torch.tensor([0.0, 0.0])
    syn.pre_trace = torch.tensor([0.0, 0.0])
    syn.post_spike = torch.tensor([0.0, 0.0])
    dw = rule.compute_dw(syn)
    assert torch.allclose(dw, torch.tensor([-1.0, 0.0, -1.0, 0.0]))


# --- Conv2dSTDP / Local2dSTDP ---------------------------------------------


def test_conv2d_stdp():
    """Conv2dSTDP LTP: pre_trace=ones(16), post_spike=ones(4), a_plus=0.1.
    Each of the 4 dst neurons sees 9 pre neurons -> dw_plus per kernel = 9*0.1,
    divided by weight_divisor (2*2=4) -> total sum 36*0.1/4 = 0.9."""
    from conex.behaviors.synapses import Conv2dDendriticInput, Conv2dSTDP
    from pymonntorch import NeuronDimension

    net = make_network(dt=1.0)
    src = lif_group(net, "s", size=NeuronDimension(depth=1, height=4, width=4))
    dst = lif_group(net, "d", size=NeuronDimension(depth=1, height=2, width=2))
    rule = Conv2dSTDP(a_plus=0.1, a_minus=0.01)
    syn = SynapseGroup(
        net=net,
        src=src,
        dst=dst,
        tag="Proximal",
        behavior=prioritize_behaviors(
            [
                SynapseInit(),
                WeightInitializer(mode=0.5, weight_shape=(1, 1, 3, 3)),
                Conv2dDendriticInput(current_coef=1.0),
                PreSpikeCatcher(),
                PostSpikeCatcher(),
                PreTrace(tau_s=10.0),
                PostTrace(tau_s=10.0),
                rule,
            ]
        ),
    )
    net.initialize()
    syn.pre_trace = torch.ones(16)
    syn.post_spike = torch.ones(4)
    syn.pre_spike = torch.zeros(16)
    syn.post_trace = torch.zeros(4)
    dw = rule.compute_dw(syn)
    assert dw.shape == (1, 1, 3, 3)
    assert dw.sum().item() == pytest.approx(0.9, abs=1e-4)


def test_local2d_stdp():
    """Local2dSTDP LTP: pre_trace=ones(16), post_spike=ones(4), a_plus=0.1.
    Each of the 4 dst neurons sees 9 pre neurons -> dw_plus = 9*0.1 per neuron,
    total sum 36*0.1 = 3.6."""
    from conex.behaviors.synapses import Local2dDendriticInput, Local2dSTDP
    from pymonntorch import NeuronDimension

    net = make_network(dt=1.0)
    src = lif_group(net, "s", size=NeuronDimension(depth=1, height=4, width=4))
    dst = lif_group(net, "d", size=NeuronDimension(depth=1, height=2, width=2))
    rule = Local2dSTDP(a_plus=0.1, a_minus=0.01)
    syn = SynapseGroup(
        net=net,
        src=src,
        dst=dst,
        tag="Proximal",
        behavior=prioritize_behaviors(
            [
                SynapseInit(),
                WeightInitializer(
                    mode=0.5,
                    weight_shape=(1, 4, 9),
                    kernel_shape=(1, 2, 2, 1, 3, 3),
                ),
                Local2dDendriticInput(current_coef=1.0),
                PreSpikeCatcher(),
                PostSpikeCatcher(),
                PreTrace(tau_s=10.0),
                PostTrace(tau_s=10.0),
                rule,
            ]
        ),
    )
    net.initialize()
    syn.pre_trace = torch.ones(16)
    syn.post_spike = torch.ones(4)
    syn.pre_spike = torch.zeros(16)
    syn.post_trace = torch.zeros(4)
    dw = rule.compute_dw(syn)
    assert dw.shape == (1, 4, 9)
    assert dw.sum().item() == pytest.approx(3.6, abs=1e-4)


# --- RSTDP (reward-modulated) ---------------------------------------------


def _rstdp_pair(net, rule, tag="rs"):
    from conex.behaviors.synapses import (
        PostSpikeCatcher,
        PostTrace,
        PreSpikeCatcher,
        PreTrace,
    )

    net.dopamine_concentration = 1.0
    src = lif_group(net, tag + "_s", size=2)
    dst = lif_group(net, tag + "_d", size=2)
    syn = SynapseGroup(
        net=net,
        src=src,
        dst=dst,
        tag="Proximal",
        behavior=prioritize_behaviors(
            [
                SynapseInit(),
                WeightInitializer(mode=0.5),
                PreSpikeCatcher(),
                PostSpikeCatcher(),
                PreTrace(tau_s=10.0),
                PostTrace(tau_s=10.0),
                rule,
            ]
        ),
    )
    return src, dst, syn


def test_simple_rstdp_dopamine():
    """RSTDP accumulates an eligibility trace c and scales the weight update by
    dopamine. With dopamine=1, pre_trace=[1,0], post_spike=[1,0], a_plus=0.1:
    c[0,0]=0.1, weight[0,0] 0.5 -> 0.6."""
    from conex.behaviors.synapses import SimpleRSTDP

    net = make_network(dt=1.0)
    rule = SimpleRSTDP(a_plus=0.1, a_minus=0.01, tau_c=10.0)
    _, _, syn = _rstdp_pair(net, rule)
    net.initialize()
    w0 = syn.weights[0, 0].item()
    syn.pre_trace = torch.tensor([1.0, 0.0])
    syn.post_spike = torch.tensor([1.0, 0.0])
    syn.pre_spike = torch.tensor([0.0, 0.0])
    syn.post_trace = torch.tensor([0.0, 0.0])
    rule.forward(syn)
    assert syn.c[0, 0].item() == pytest.approx(0.1, abs=1e-5)
    assert syn.weights[0, 0].item() == pytest.approx(w0 + 0.1, abs=1e-5)


def test_one2one_rstdp():
    """One2OneRSTDP compute_dw is elementwise STDP: pre_trace=[1,0],
    post_spike=[1,0], a_plus=0.1 -> [0.1, 0]. (Its forward is inherited from
    SimpleRSTDP and expects a 2D weight, so the update is checked via
    compute_dw, the part specific to the one-to-one rule.)"""
    from conex.behaviors.synapses import One2OneRSTDP

    net = make_network(dt=1.0)
    rule = One2OneRSTDP(a_plus=0.1, a_minus=0.01, tau_c=10.0)
    _, _, syn = _rstdp_pair(net, rule)
    net.initialize()
    syn.pre_trace = torch.tensor([1.0, 0.0])
    syn.post_spike = torch.tensor([1.0, 0.0])
    syn.pre_spike = torch.tensor([0.0, 0.0])
    syn.post_trace = torch.tensor([0.0, 0.0])
    dw = rule.compute_dw(syn)
    assert torch.allclose(dw, torch.tensor([0.1, 0.0]))


def test_sparse_rstdp():
    """SparseRSTDP: eligibility on the (src0,dst0) nnz entry, dopamine-scaled.
    The rule is initialized by hand on a COO weight because WeightInitializer
    leaves a CSC tensor whose _nnz() is unsupported on CPU.
    """
    from conex.behaviors.synapses import SparseRSTDP

    net = make_network(dt=1.0)
    net.dopamine_concentration = 1.0
    rule = SparseRSTDP(a_plus=0.1, a_minus=0.01, tau_c=10.0)
    _, _, syn = _sparse_pair(net, rule, include_rule=False)
    net.initialize()
    _set_coo(syn)
    rule.initialize(syn)
    syn.pre_trace = torch.tensor([1.0, 0.0])
    syn.post_spike = torch.tensor([1.0, 0.0])
    syn.pre_spike = torch.tensor([0.0, 0.0])
    syn.post_trace = torch.tensor([0.0, 0.0])
    rule.forward(syn)
    assert syn.c[0].item() == pytest.approx(0.1, abs=1e-5)


def test_conv2d_rstdp():
    """Conv2dRSTDP: eligibility trace c has the kernel shape and is non-zero
    where pre and post activity coincide."""
    from conex.behaviors.synapses import Conv2dDendriticInput, Conv2dRSTDP
    from pymonntorch import NeuronDimension

    net = make_network(dt=1.0)
    net.dopamine_concentration = 1.0
    src = lif_group(net, "s", size=NeuronDimension(depth=1, height=4, width=4))
    dst = lif_group(net, "d", size=NeuronDimension(depth=1, height=2, width=2))
    rule = Conv2dRSTDP(a_plus=0.1, a_minus=0.01, tau_c=10.0)
    syn = SynapseGroup(
        net=net,
        src=src,
        dst=dst,
        tag="Proximal",
        behavior=prioritize_behaviors(
            [
                SynapseInit(),
                WeightInitializer(mode=0.5, weight_shape=(1, 1, 3, 3)),
                Conv2dDendriticInput(current_coef=1.0),
                PreSpikeCatcher(),
                PostSpikeCatcher(),
                PreTrace(tau_s=10.0),
                PostTrace(tau_s=10.0),
                rule,
            ]
        ),
    )
    net.initialize()
    syn.pre_trace = torch.ones(16)
    syn.post_spike = torch.ones(4)
    syn.pre_spike = torch.zeros(16)
    syn.post_trace = torch.zeros(4)
    rule.forward(syn)
    assert syn.c.shape == (1, 1, 3, 3)
    assert syn.c.sum().item() > 0


def test_local2d_rstdp():
    """Local2dRSTDP: eligibility trace c has the weight shape and is non-zero
    where pre and post activity coincide."""
    from conex.behaviors.synapses import Local2dDendriticInput, Local2dRSTDP
    from pymonntorch import NeuronDimension

    net = make_network(dt=1.0)
    net.dopamine_concentration = 1.0
    src = lif_group(net, "s", size=NeuronDimension(depth=1, height=4, width=4))
    dst = lif_group(net, "d", size=NeuronDimension(depth=1, height=2, width=2))
    rule = Local2dRSTDP(a_plus=0.1, a_minus=0.01, tau_c=10.0)
    syn = SynapseGroup(
        net=net,
        src=src,
        dst=dst,
        tag="Proximal",
        behavior=prioritize_behaviors(
            [
                SynapseInit(),
                WeightInitializer(
                    mode=0.5,
                    weight_shape=(1, 4, 9),
                    kernel_shape=(1, 2, 2, 1, 3, 3),
                ),
                Local2dDendriticInput(current_coef=1.0),
                PreSpikeCatcher(),
                PostSpikeCatcher(),
                PreTrace(tau_s=10.0),
                PostTrace(tau_s=10.0),
                rule,
            ]
        ),
    )
    net.initialize()
    syn.pre_trace = torch.ones(16)
    syn.post_spike = torch.ones(4)
    syn.pre_spike = torch.zeros(16)
    syn.post_trace = torch.zeros(4)
    rule.forward(syn)
    assert syn.c.shape == (1, 4, 9)
    assert syn.c.sum().item() > 0
