"""Shared scenario registry for the golden-number tests.

Each scenario is a pure function of (seed, dtype): it seeds torch, builds a
small network, runs a fixed protocol and returns a dict of named tensors.
The generator (generate.py) and both test files import from here so the
protocol is defined exactly once.
"""

import torch
from pymonntorch import Network, NeuronGroup, SynapseGroup

from conex import (
    ColumnVoting,
    ErrorUnit,
    GridCellModule,
    PrecisionWeighting,
    PredictiveCodingConfig,
    SpatialPooler,
    SpatialPoolerConfig,
    TemporalMemory,
    TemporalMemoryConfig,
    TimeResolution,
    prioritize_behaviors,
)
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
    SimpleDendriticInput,
    SimpleSTDP,
    SynapseInit,
    WeightInitializer,
)

SP_CFG = SpatialPoolerConfig(
    input_size=60, n_columns=40, potential_pct=0.5, local_area_density=0.05, seed=7
)
TM_CFG = TemporalMemoryConfig(
    n_columns=40, cells_per_column=4, activation_threshold=2,
    learning_threshold=1, max_new_synapses=5,
)


def build_network(seed, dtype):
    torch.manual_seed(seed)
    return Network(behavior={1: TimeResolution(dt=1.0)}, dtype=dtype, device="cpu")


def make_group(net, size, behavior, tag):
    g = NeuronGroup(net=net, size=size, behavior={100: behavior}, tag=tag)
    net.initialize()
    return g


def lif_group(net, tag, size=2):
    return NeuronGroup(
        net=net, size=size, tag=tag,
        behavior=prioritize_behaviors([
            SimpleDendriteStructure(),
            SimpleDendriteComputation(),
            LIF(R=5.0, threshold=1.0, tau=10.0, v_reset=0.0, v_rest=0.0),
            Fire(),
            NeuronAxon(max_delay=1),
        ]),
    )


def scenario_lif(seed, dtype):
    """One LIF neuron driven through a fixed voltage sequence."""
    net = build_network(seed, dtype)
    n = lif_group(net, "lif", size=1)
    net.initialize()
    spikes, voltages = [], []
    for v in (1.5, 0.9, 0.0, 1.2, 0.0):
        n.v = torch.full((1,), v, dtype=dtype)
        net.simulate_iterations(1)
        spikes.append(bool(n.spikes[0].item()))
        voltages.append(n.v[0].item())
    return {"spikes": spikes, "voltages": voltages}


def scenario_stdp(seed, dtype):
    """A 2x2 STDP pair co-firing for 3 steps; record the weight change."""
    net = build_network(seed, dtype)
    src = lif_group(net, "s")
    dst = lif_group(net, "d")
    syn = SynapseGroup(
        net=net, src=src, dst=dst, tag="Proximal",
        behavior=prioritize_behaviors([
            SynapseInit(),
            WeightInitializer(mode=0.5),
            SimpleDendriticInput(current_coef=1.0),
            PreSpikeCatcher(),
            PostSpikeCatcher(),
            PreTrace(tau_s=10.0),
            PostTrace(tau_s=10.0),
            SimpleSTDP(a_plus=0.1, a_minus=0.01),
        ]),
    )
    net.initialize()
    w0 = syn.weights[0, 0].item()
    for _ in range(3):
        src.v = torch.tensor([2.0, 0.0], dtype=dtype)
        dst.v = torch.tensor([2.0, 0.0], dtype=dtype)
        net.simulate_iterations(1)
    return {"w0": w0, "w3": syn.weights[0, 0].item(),
            "delta": syn.weights[0, 0].item() - w0}


def scenario_spatial_pooler(seed, dtype):
    """SP output for a fixed 35-of-60 input pattern (config seed = 7 + seed)."""
    net = build_network(seed, dtype)
    sp = make_group(net, SP_CFG.n_columns,
                    SpatialPooler(config=SP_CFG._replace(seed=7 + seed)), tag="sp")
    x = torch.zeros(60, dtype=dtype)
    x[5:40] = 1.0
    out = sp.spatial_pooler.compute(sp, x, learn=False)
    return {"active": int(out.sum().item()),
            "indices": sorted(int(i) for i in out.nonzero().flatten().tolist())}


def _columns(n, *active, dtype=torch.float32):
    c = torch.zeros(n, dtype=torch.bool)
    c[list(active)] = True
    return c


def scenario_temporal_memory(seed, dtype):
    """TM over a repeated 3-step sequence; record anomaly scores per pass."""
    net = build_network(seed, dtype)
    tm = make_group(net, TM_CFG.n_columns * TM_CFG.cells_per_column,
                    TemporalMemory(config=TM_CFG), tag="tm")
    seq = [_columns(40, 1, 2), _columns(40, 5, 6), _columns(40, 9, 10)]
    passes = []
    for _ in range(15):
        scores = []
        for step in seq:
            tm.temporal_memory.compute(tm, step, learn=True)
            scores.append(tm.tm_anomaly_score.item())
        passes.append(scores)
    return {"passes": passes,
            "accuracy": tm.tm_prediction_accuracy.item()}


def scenario_grid_cell(seed, dtype):
    """Grid lattice: module-0 activations at the origin and at one lattice step."""
    net = build_network(seed, dtype)
    g = make_group(net, 40, GridCellModule(n_modules=4, cells_per_module=10,
                                           scale_range=(0.5, 2.0)), tag="grid")
    module = g.grid_cell_module
    origin = module.encode_position(g, torch.zeros(2, dtype=dtype))[:10]
    # one full lattice translation: 2*lambda/sqrt(3) along orientation+30 deg
    import math
    scale = module.scales[0].item()
    angle = module.orientations[0].item() + math.pi / 6
    length = 2 * scale / math.sqrt(3)
    step = torch.tensor([length * math.cos(angle), length * math.sin(angle)],
                        dtype=dtype)
    shifted = module.encode_position(g, step)[:10]
    return {"origin": origin.tolist(), "shifted": shifted.tolist(),
            "max_diff": (origin - shifted).abs().max().item()}


def scenario_voting(seed, dtype):
    """Column voting: three full-match votes for object 1, then consensus."""
    net = build_network(seed, dtype)
    col = make_group(net, 32, ColumnVoting(
        n_objects=5, representation_size=32, voting_threshold=0.5,
        decay_rate=0.1, min_votes_for_consensus=2), tag="col")
    memories = torch.zeros(5, 32, dtype=dtype)
    for obj in range(5):
        memories[obj, obj * 4:(obj + 1) * 4] = 1.0
    votes = col.voting_module.cast_vote(col, memories[1].clone(), memories)
    for _ in range(3):
        col.voting_module.receive_votes(col, votes)
    reached = bool(col.voting_module.check_consensus(col, n_columns=3))
    return {"votes": votes.tolist(), "reached": reached,
            "object": int(col.consensus_object),
            "accumulator": col.vote_accumulator.tolist()}


def scenario_predictive_error(seed, dtype):
    """ErrorUnit: activity 2.0, prediction 0.5, precision 3.0, weighted."""
    net = build_network(seed, dtype)
    cfg = PredictiveCodingConfig(error_nonlinearity="linear", use_sparse_errors=False)
    g = make_group(net, 16, ErrorUnit(config=cfg, precision_weighted=True), tag="pc")
    g.activity = torch.full((16,), 2.0, dtype=dtype)
    g.prediction = torch.full((16,), 0.5, dtype=dtype)
    g.precision = torch.full((16,), 3.0, dtype=dtype)
    g.behavior[100].forward(g)
    return {"signed": g.signed_error.tolist(),
            "weighted": g.prediction_error.tolist()}


def scenario_precision(seed, dtype):
    """PrecisionWeighting: effective = sensory * prior, then written back."""
    net = build_network(seed, dtype)
    cfg = PredictiveCodingConfig(error_nonlinearity="linear", use_sparse_errors=False)
    g = make_group(net, 16, PrecisionWeighting(config=cfg, learnable=False,
                                               attention_modulated=False), tag="pw")
    g.sensory_precision = torch.full((16,), 2.0, dtype=dtype)
    g.prior_precision = torch.full((16,), 3.0, dtype=dtype)
    g.behavior[100].forward(g)
    return {"effective": g.effective_precision.tolist()}


SCENARIOS = {
    "lif": scenario_lif,
    "stdp": scenario_stdp,
    "spatial_pooler": scenario_spatial_pooler,
    "temporal_memory": scenario_temporal_memory,
    "grid_cell": scenario_grid_cell,
    "voting": scenario_voting,
    "predictive_error": scenario_predictive_error,
    "precision": scenario_precision,
}
