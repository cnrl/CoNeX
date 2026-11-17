"""Regression tests for the sparse STDP family (fix-b, BUG 2).

WeightInitializer used to convert sparse weights to CSC, which breaks the
scalar scale on torch 1.x and the .values()/_nnz() access the sparse rules
need. The fix keeps the weights in COO layout, which works on both torch 1.x
and 2.x.
"""

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
    PostSpikeCatcher,
    PostTrace,
    PreSpikeCatcher,
    PreTrace,
    SparseSTDP,
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


def sparse_pair(net, rule, tag="sp", src_size=2, dst_size=2, density=0.5):
    src = lif_group(net, tag + "_s", size=src_size)
    dst = lif_group(net, tag + "_d", size=dst_size)
    syn = SynapseGroup(
        net=net,
        src=src,
        dst=dst,
        tag="Proximal",
        behavior=prioritize_behaviors(
            [
                SynapseInit(),
                WeightInitializer(mode=0.5, density=density, true_sparsity=True),
                PreSpikeCatcher(),
                PostSpikeCatcher(),
                PreTrace(tau_s=10.0),
                PostTrace(tau_s=10.0),
                rule,
            ]
        ),
    )
    return src, dst, syn


def test_sparse_init_coo_density():
    """density=0.5, true_sparsity=True -> COO weights with the expected nnz.

    A 2x2 matrix has 4 entries; density 0.5 -> 2 non-zero. The layout must be
    COO (not CSC) so the sparse rules can read .values() on both torch 1.x and
    2.x, and the scalar scale/offset must have applied without error.
    """
    net = make_network(dt=1.0)
    rule = SparseSTDP(a_plus=0.1, a_minus=0.01)
    _, _, syn = sparse_pair(net, rule, density=0.5)
    net.initialize()
    assert syn.weights.layout == torch.sparse_coo
    assert syn.weights._nnz() == 2
    # scale=1, offset=0, mode=0.5 -> every stored value is 0.5
    assert torch.allclose(syn.weights.values(), torch.full((2,), 0.5))
    # index vectors line up with the stored entries
    assert syn.src_idx.numel() == 2
    assert syn.dst_idx.numel() == 2
    assert torch.all(syn.src_idx < 2)
    assert torch.all(syn.dst_idx < 2)


def test_sparse_stdp_runs_and_stays_in_bounds():
    """SparseSTDP runs end to end for N steps; weights stay within [w_min, w_max].

    The rule is driven with LTP-favoring spikes (pre trace high, post spike on)
    so the weights climb toward w_max. hard_bound caps the update at the bound,
    so after enough steps the weights must not exceed w_max and never drop
    below w_min.
    """
    net = make_network(dt=1.0)
    rule = SparseSTDP(
        a_plus=0.1,
        a_minus=0.01,
        w_min=0.0,
        w_max=1.0,
        positive_bound="hard_bound",
        negative_bound="hard_bound",
    )
    _, _, syn = sparse_pair(net, rule, density=0.5)
    net.initialize()
    nnz = syn.weights._nnz()
    assert nnz > 0

    # Drive LTP on every stored entry: pre trace high, post spike on, no LTD.
    for _ in range(20):
        syn.pre_trace = torch.ones(2)
        syn.post_spike = torch.ones(2)
        syn.pre_spike = torch.zeros(2)
        syn.post_trace = torch.zeros(2)
        rule.forward(syn)

    vals = syn.weights.values()
    assert vals.numel() == nnz
    # hard_bound gates the update at the bound, but float32 accumulation can
    # push a weight marginally past w_max (e.g. 0.9 + 0.1 -> 1.0000001), so
    # allow a small numerical tolerance around the bounds.
    tol = 1e-5
    assert bool((vals >= -tol).all()), f"weights below w_min: {vals.tolist()}"
    assert bool((vals <= 1.0 + tol).all()), f"weights above w_max: {vals.tolist()}"
    # LTP should have pushed at least one weight up from its 0.5 init
    assert bool((vals > 0.5).any()), f"no weight potentiated: {vals.tolist()}"
