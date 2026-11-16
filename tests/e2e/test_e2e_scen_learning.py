"""E2E learning scenarios: several modules working together in one running network.

Each scenario builds a small network from the public API and asserts on what it
LEARNs (selectivity, reward-modulated growth, label-wise spike statistics,
reproducibility), not on a single function. Inputs are Poisson-encoded so the
networks are genuinely stochastic and seed-dependent.

Reference style: end-to-end learning checks, seeded reproducibility, tiny sizes.
"""

import pytest
import torch
from pymonntorch import NeuronGroup, SynapseGroup
from torch.utils.data import DataLoader, TensorDataset

from conex import (
    ActivityBaseHomeostasis,
    Dopamine,
    Fire,
    InputLayer,
    KWTA,
    LIF,
    Neocortex,
    NeuronAxon,
    Payoff,
    Poisson,
    PostSpikeCatcher,
    PostTrace,
    PreSpikeCatcher,
    PreTrace,
    SimpleDendriteComputation,
    SimpleDendriteStructure,
    SimpleDendriticInput,
    SimpleRSTDP,
    SimpleSTDP,
    SynapseInit,
    WeightClip,
    WeightInitializer,
    prioritize_behaviors,
)

SEED = 1234
OTHER_SEED = 999


def make_net(net_beh=None):
    """A fresh CPU Neocortex (dt=1.0) carrying optional network behaviors."""
    return Neocortex(dt=1.0, behavior=net_beh or {})


def input_group(net, tag, size=4):
    """A spiking input population: axon only, spikes set externally each step."""
    return NeuronGroup(
        net=net, size=size, tag=tag,
        behavior=prioritize_behaviors([NeuronAxon(max_delay=1)]),
    )


def lif_group(net, tag, size=4, homeo=False, kwta=False):
    """A LIF output population with optional homeostasis and KWTA competition."""
    beh = [
        SimpleDendriteStructure(),
        SimpleDendriteComputation(),
        LIF(R=5.0, threshold=1.0, tau=10.0, v_reset=0.0, v_rest=0.0),
    ]
    if homeo:
        beh.append(ActivityBaseHomeostasis(activity_rate=1, window_size=10, updating_rate=0.05))
    if kwta:
        beh.append(KWTA(k=1))
    beh += [Fire(), NeuronAxon(max_delay=1)]
    return NeuronGroup(net=net, size=size, tag=tag, behavior=prioritize_behaviors(beh))


def stdp_syn(net, src, dst, rule, clip=True):
    """A Proximal synapse wired with the given learning rule (and clip if set)."""
    beh = [
        SynapseInit(),
        WeightInitializer(mode=0.5),
        SimpleDendriticInput(current_coef=1.0),
        PreSpikeCatcher(),
        PostSpikeCatcher(),
        PreTrace(tau_s=10.0),
        PostTrace(tau_s=10.0),
        rule,
    ]
    if clip:
        beh.append(WeightClip(w_min=0.0, w_max=1.0))
    return SynapseGroup(net=net, src=src, dst=dst, tag="Proximal",
                        behavior=prioritize_behaviors(beh))


def poisson(pattern, enc):
    """Encode a float intensity vector into a bool spike vector for one step."""
    return enc(pattern).view(-1)


# --- Scenario 1: competitive STDP ------------------------------------------


def build_competitive(seedv=SEED):
    """Poisson input -> LIF+KWTA+homeostasis output, SimpleSTDP + WeightClip.

    Two input patterns (A: channels 0-1, B: channels 2-3) are presented
    alternately. An asymmetric initial weight bias lets one output neuron win
    each pattern; STDP + competition sharpen that selectivity.
    """
    torch.manual_seed(seedv)
    net = make_net()
    inp = input_group(net, "in", 4)
    out = lif_group(net, "out", 2, homeo=True, kwta=True)
    syn = stdp_syn(net, inp, out, SimpleSTDP(a_plus=0.002, a_minus=0.0002))
    net.initialize()
    # asymmetric init: out0 biased toward A, out1 toward B
    syn.weights = torch.tensor([[0.8, 0.2], [0.8, 0.2], [0.2, 0.8], [0.2, 0.8]])
    return net, inp, out, syn


def run_competitive(seedv=SEED, steps=200):
    """Run the competitive network, returning per-pattern firing rates and weights."""
    net, inp, out, syn = build_competitive(seedv)
    enc = Poisson(time_window=1, ratio=1.0)
    A = torch.tensor([1.0, 1.0, 0.0, 0.0])
    B = torch.tensor([0.0, 0.0, 1.0, 1.0])
    rates = {0: torch.zeros(2), 1: torch.zeros(2)}
    for step in range(steps):
        inp.spikes = poisson(A if step % 2 == 0 else B, enc)
        net.simulate_iterations(1)
        lab = 0 if step % 2 == 0 else 1
        rates[lab] += out.spikes.float()
    half = steps // 2
    return rates[0] / half, rates[1] / half, syn.weights


def test_competitive_stdp_selectivity():
    """After learning, each output neuron prefers one input pattern (selectivity)."""
    rA, rB, w = run_competitive()
    # each output neuron fires more for its preferred pattern than the other
    sel0 = abs(rA[0].item() - rB[0].item())
    sel1 = abs(rA[1].item() - rB[1].item())
    assert sel0 >= 0.2, f"out0 selectivity too low: {sel0}"
    assert sel1 >= 0.2, f"out1 selectivity too low: {sel1}"
    # STDP potentiated the preferred connections to the clip bound (a_plus=0
    # would leave them below ~0.8), so the weights show pattern selectivity
    out0_pref = w[:2, 0].sum().item() - w[2:, 0].sum().item()  # A->out0 vs B->out0
    out1_pref = w[2:, 1].sum().item() - w[:2, 1].sum().item()  # B->out1 vs A->out1
    assert out0_pref > 0.5, f"out0 weights should prefer A: {out0_pref}"
    assert out1_pref > 0.5, f"out1 weights should prefer B: {out1_pref}"
    assert w.max().item() >= 0.99, "preferred weights should potentiate to the clip bound"
    # weights stay inside the clip bounds
    assert w.min().item() >= 0.0
    assert w.max().item() <= 1.0


def test_competitive_stdp_weights_bounded():
    """WeightClip keeps every weight in [0, 1] throughout learning."""
    _, _, w = run_competitive(steps=100)
    assert w.min().item() >= 0.0
    assert w.max().item() <= 1.0


# --- Scenario 2: reward-modulated STDP -------------------------------------


def build_rstdp(seedv=SEED):
    """Poisson input -> LIF output, SimpleRSTDP + WeightClip, with Payoff/Dopamine.

    SimpleRSTDP (not One2OneRSTDP, which is known broken) accumulates an
    eligibility trace scaled by the network dopamine concentration.
    """
    torch.manual_seed(seedv)
    net = make_net({
        100: Payoff(initial_payoff=0.0),
        120: Dopamine(tau_dopamine=10.0, initial_dopamine_concentration=0.0),
    })
    inp = input_group(net, "in", 4)
    out = lif_group(net, "out", 2)
    syn = stdp_syn(net, inp, out, SimpleRSTDP(a_plus=0.0002, a_minus=0.00002, tau_c=10.0))
    net.initialize()
    return net, inp, out, syn


def run_rstdp(seedv=SEED):
    """Reward pattern A, let dopamine decay, then present unrewarded pattern B.

    Returns the total weight growth for the A-connected and B-connected rows.
    """
    net, inp, out, syn = build_rstdp(seedv)
    w0 = syn.weights.clone()
    enc = Poisson(time_window=1, ratio=1.0)
    A = torch.tensor([1.0, 1.0, 0.0, 0.0])
    B = torch.tensor([0.0, 0.0, 1.0, 1.0])
    inp.spikes = torch.zeros(4)
    for _ in range(50):  # baseline, no reward
        net.simulate_iterations(1)
    for _ in range(30):  # reward + pattern A
        net.payoff = 0.1
        inp.spikes = poisson(A, enc)
        net.simulate_iterations(1)
    for _ in range(100):  # dopamine decays, no input
        net.payoff = 0.0
        inp.spikes = torch.zeros(4)
        net.simulate_iterations(1)
    for _ in range(30):  # no reward + pattern B
        net.payoff = 0.0
        inp.spikes = poisson(B, enc)
        net.simulate_iterations(1)
    gA = syn.weights[:2].sum().item() - w0[:2].sum().item()
    gB = syn.weights[2:].sum().item() - w0[2:].sum().item()
    return gA, gB, syn.weights


def test_rstdp_rewarded_grows_more():
    """The rewarded pattern's weights grow more than the unrewarded one's."""
    gA, gB, w = run_rstdp()
    assert gA > gB, f"rewarded growth {gA} should exceed unrewarded {gB}"
    assert gA > 0.0, "rewarded pattern should potentiate"
    assert w.min().item() >= 0.0
    assert w.max().item() <= 1.0


def test_rstdp_dopamine_decays():
    """Dopamine rises with reward and decays back toward zero after it stops."""
    net, inp, out, syn = build_rstdp()
    enc = Poisson(time_window=1, ratio=1.0)
    A = torch.tensor([1.0, 1.0, 0.0, 0.0])
    inp.spikes = torch.zeros(4)
    for _ in range(30):
        net.payoff = 0.1
        inp.spikes = poisson(A, enc)
        net.simulate_iterations(1)
    peak = net.dopamine_concentration
    for _ in range(100):
        net.payoff = 0.0
        inp.spikes = torch.zeros(4)
        net.simulate_iterations(1)
    assert peak > 0.0, "dopamine should rise under reward"
    assert net.dopamine_concentration < peak, "dopamine should decay after reward"


# --- Scenario 3: dataset to network ----------------------------------------


def build_dataset(seedv=SEED):
    """Poisson-encoded synthetic dataset -> InputLayer -> LIF readout.

    Label 0 instances are strong on channels 0-1, label 1 on channels 2-3, so
    the encoded spike statistics differ by label.
    """
    torch.manual_seed(seedv)
    net = make_net()
    raw = torch.tensor([[0.9, 0.1, 0.1, 0.1]] * 8 + [[0.1, 0.1, 0.9, 0.1]] * 8)
    labels = torch.tensor([0] * 8 + [1] * 8)
    enc = Poisson(time_window=1, ratio=3.0)
    spikes = enc(raw).view(16, 4)
    dl = DataLoader(TensorDataset(spikes, labels), batch_size=16)
    il = InputLayer(
        net=net, input_dataloader=dl, sensory_size=4, instance_duration=1,
        have_location=False, have_label=True, sensory_data_dim=1,
        output_ports={"data_out": (None, [("sensory_pop", {})])},
    )
    out = lif_group(net, "out", 2)
    syn = stdp_syn(net, il.sensory_pop, out, SimpleSTDP(a_plus=0.002, a_minus=0.0002))
    net.initialize()
    return net, il, out, syn, spikes


def test_dataset_spike_statistics_labelwise():
    """The Poisson encoder produces label-wise different spike statistics."""
    _, _, _, _, spikes = build_dataset()
    r0 = spikes[:8].float().mean(0)  # label 0: channels 0-1 strong
    r1 = spikes[8:].float().mean(0)  # label 1: channels 2-3 strong
    # label 0 fires channel 0 more than channel 2; label 1 the reverse
    assert r0[0].item() > r0[2].item(), "label0 should be strong on ch0"
    assert r1[2].item() > r1[0].item(), "label1 should be strong on ch2"
    # the two labels differ in their spike statistics
    assert not torch.allclose(r0, r1), "labels should have different spike stats"


def test_dataset_readout_learns():
    """The readout weights move as the network runs on the dataset."""
    net, il, out, syn, _ = build_dataset()
    w0 = syn.weights.clone()
    for _ in range(16):
        net.simulate_iterations(1)
    assert not torch.equal(w0, syn.weights), "readout weights should learn"
    assert syn.weights.min().item() >= 0.0
    assert syn.weights.max().item() <= 1.0


# --- Scenario 4: reproducibility matrix ------------------------------------


def test_repro_competitive_same_seed_identical():
    """Same seed run twice in-process gives bit-identical weights."""
    _, _, w1 = run_competitive(SEED)
    _, _, w2 = run_competitive(SEED)
    assert torch.equal(w1, w2)


def test_repro_competitive_diff_seed_differs():
    """A different seed gives different weights (stochastic Poisson input)."""
    _, _, w1 = run_competitive(SEED)
    _, _, w2 = run_competitive(OTHER_SEED)
    assert not torch.equal(w1, w2)


def test_repro_rstdp_same_seed_identical():
    """Same seed run twice in-process gives bit-identical weights."""
    gA1, gB1, w1 = run_rstdp(SEED)
    gA2, gB2, w2 = run_rstdp(SEED)
    assert torch.equal(w1, w2)
    assert gA1 == gA2 and gB1 == gB2


def test_repro_rstdp_diff_seed_differs():
    """A different seed gives different weights (stochastic Poisson input)."""
    _, _, w1 = run_rstdp(SEED)
    _, _, w2 = run_rstdp(OTHER_SEED)
    assert not torch.equal(w1, w2)


def test_repro_dataset_same_seed_identical():
    """Same seed run twice in-process gives bit-identical readout weights."""
    net1, _, _, syn1, _ = build_dataset(SEED)
    for _ in range(16):
        net1.simulate_iterations(1)
    net2, _, _, syn2, _ = build_dataset(SEED)
    for _ in range(16):
        net2.simulate_iterations(1)
    assert torch.equal(syn1.weights, syn2.weights)


def test_repro_dataset_diff_seed_differs():
    """A different seed gives different readout weights (stochastic encoding)."""
    net1, _, _, syn1, _ = build_dataset(SEED)
    for _ in range(16):
        net1.simulate_iterations(1)
    net2, _, _, syn2, _ = build_dataset(OTHER_SEED)
    for _ in range(16):
        net2.simulate_iterations(1)
    assert not torch.equal(syn1.weights, syn2.weights)


# --- Failure / edge cases ---------------------------------------------------


def test_edge_zero_input_no_learning():
    """With zero input, no pre/post spikes fire, so weights stay unchanged."""
    torch.manual_seed(SEED)
    net = make_net()
    inp = input_group(net, "in", 4)
    out = lif_group(net, "out", 2)
    syn = stdp_syn(net, inp, out, SimpleSTDP(a_plus=0.002, a_minus=0.0002))
    net.initialize()
    w0 = syn.weights.clone()
    inp.spikes = torch.zeros(4)
    for _ in range(50):
        net.simulate_iterations(1)
    assert torch.equal(w0, syn.weights), "zero input should not change weights"


def test_edge_saturated_weights_stay_bounded():
    """Weights already at the clip bound stay inside [0, 1] under strong input."""
    torch.manual_seed(SEED)
    net = make_net()
    inp = input_group(net, "in", 4)
    out = lif_group(net, "out", 2)
    syn = stdp_syn(net, inp, out, SimpleSTDP(a_plus=0.5, a_minus=0.0002))
    net.initialize()
    syn.weights = torch.ones(4, 2)  # saturated at w_max
    inp.spikes = torch.ones(4)
    for _ in range(50):
        net.simulate_iterations(1)
    assert syn.weights.max().item() <= 1.0
    assert syn.weights.min().item() >= 0.0


def test_edge_wrong_input_layer_shape_raises():
    """A sensory_size that does not match the data's last dim raises on run."""
    torch.manual_seed(SEED)
    net = make_net()
    data = torch.zeros(4, 4, dtype=torch.bool)  # last dim 4
    dl = DataLoader(TensorDataset(data), batch_size=4)
    InputLayer(
        net=net, input_dataloader=dl, sensory_size=3, instance_duration=1,
        have_location=False, have_label=False, sensory_data_dim=1,
        output_ports={"data_out": (None, [("sensory_pop", {})])},
    )
    net.initialize()
    with pytest.raises(RuntimeError):
        net.simulate_iterations(1)


def test_edge_wrong_weight_shape_raises():
    """A WeightInitializer weight matrix with the wrong shape raises on run."""
    torch.manual_seed(SEED)
    net = make_net()
    src = lif_group(net, "s", 4)
    dst = lif_group(net, "d", 2)
    SynapseGroup(
        net=net, src=src, dst=dst, tag="Proximal",
        behavior=prioritize_behaviors([
            SynapseInit(),
            WeightInitializer(weights=torch.zeros(3, 2)),  # should be 4x2
            SimpleDendriticInput(current_coef=1.0),
            PreSpikeCatcher(),
        ]),
    )
    net.initialize()
    with pytest.raises(IndexError):
        net.simulate_iterations(1)
