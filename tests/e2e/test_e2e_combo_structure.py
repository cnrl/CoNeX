"""End-to-end combination: structure round trips and precision.

Build a structure, save it to JSON and load it back (and replicate it), run
both for N steps with the same seed, and assert identical state tensors.
Then compare a float32 vs a float64 network (results close, dtype as declared).

Each file carries its own small network builder so it runs on its own.
"""

import os

import pytest
import torch
from pymonntorch import NeuronGroup, SynapseGroup

from conex import (
    Neocortex,
    CorticalLayer,
    prioritize_behaviors,
    SimpleDendriteStructure,
    SimpleDendriteComputation,
    LIF,
    NeuronAxon,
    SynapseInit,
    WeightInitializer,
    SimpleDendriticInput,
    PreSpikeCatcher,
    save_structure,
    save_structure_dict_to_json,
    load_structure_dict_from_json,
    create_structure_from_dict,
    replicate,
    PrecisionMode,
    PrecisionConfig,
)


def pop_behavior():
    return prioritize_behaviors([
        SimpleDendriteStructure(),
        SimpleDendriteComputation(),
        LIF(R=1.0, threshold=1.0, tau=1.0, v_reset=0.0, v_rest=0.0),
        NeuronAxon(),
    ])


def syn_behavior():
    # deterministic init so a round trip is bit-for-bit comparable
    return prioritize_behaviors([
        SynapseInit(),
        WeightInitializer(mode="ones"),
        SimpleDendriticInput(current_coef=0.1),
        PreSpikeCatcher(),
    ])


def build_layer(net):
    exc = NeuronGroup(net=net, size=8, tag="exc", behavior=pop_behavior())
    inh = NeuronGroup(net=net, size=4, tag="inh", behavior=pop_behavior())
    syn = SynapseGroup(net=net, src=exc, dst=inh, tag="s", behavior=syn_behavior())
    return CorticalLayer(net=net, excitatory_neurongroup=exc,
                         inhibitory_neurongroup=inh, synapsegroups=[syn], tag="l4")


def build_and_run(seed=42, steps=5, dtype=torch.float32):
    torch.manual_seed(seed)
    net = Neocortex(dt=1.0, dtype=dtype)
    layer = build_layer(net)
    net.initialize()
    net.simulate_iterations(steps)
    return net, layer


class TestStructureRoundTrip:
    def test_json_round_trip_gives_identical_state(self, tmp_path):
        """Save -> load -> rebuild -> run: state tensors match the original."""
        net1, layer1 = build_and_run(seed=42, steps=5)

        path = str(tmp_path / "layer.json")
        save_structure_dict_to_json(save_structure(layer1), path)
        loaded = load_structure_dict_from_json(path)
        assert loaded["class"] is type(layer1)

        net2, layer2 = build_and_run(seed=42, steps=0)  # fresh net, same seed
        layer2 = create_structure_from_dict(net=net2, structure_dict=loaded)
        net2.initialize()
        net2.simulate_iterations(5)

        assert layer2.exc_pop.size == layer1.exc_pop.size == 8
        assert layer2.inh_pop.size == layer1.inh_pop.size == 4
        assert torch.equal(layer1.synapsegroups[0].weights, layer2.synapsegroups[0].weights)
        assert torch.equal(layer1.exc_pop.v, layer2.exc_pop.v)
        assert torch.equal(layer1.exc_pop.spikes, layer2.exc_pop.spikes)

    def test_replicate_gives_identical_state(self):
        """replicate() rebuilds the structure on a new net with the same state."""
        net1, layer1 = build_and_run(seed=42, steps=5)
        net2 = Neocortex(dt=1.0)
        layer2, _ = replicate(layer1, net2)
        net2.initialize()
        net2.simulate_iterations(5)
        assert torch.equal(layer1.synapsegroups[0].weights, layer2.synapsegroups[0].weights)
        assert torch.equal(layer1.exc_pop.v, layer2.exc_pop.v)

    def test_json_file_is_written(self, tmp_path):
        _, layer = build_and_run(seed=1, steps=0)
        path = str(tmp_path / "layer.json")
        save_structure_dict_to_json(save_structure(layer), path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


class TestPrecisionRoundTrip:
    def test_float32_vs_float64_close_and_dtype_as_declared(self):
        """Same structure in float32 and float64: results close, dtype as set."""
        net32, l32 = build_and_run(seed=42, steps=5, dtype=torch.float32)
        net64, l64 = build_and_run(seed=42, steps=5, dtype=torch.float64)

        # dtype is exactly what was declared on the network
        assert net32.def_dtype == torch.float32
        assert net64.def_dtype == torch.float64
        assert l32.exc_pop.v.dtype == torch.float32
        assert l64.exc_pop.v.dtype == torch.float64

        # with a deterministic init the only difference is 32-vs-64 arithmetic
        w32 = l32.synapsegroups[0].weights.double()
        w64 = l64.synapsegroups[0].weights.double()
        assert torch.allclose(w32, w64, atol=1e-4)
        assert torch.allclose(l32.exc_pop.v.double(), l64.exc_pop.v.double(), atol=1e-4)
        assert torch.equal(l32.exc_pop.spikes, l64.exc_pop.spikes)

    def test_precision_config_full_maps_to_float32(self):
        assert PrecisionConfig(mode=PrecisionMode.FULL).dtype == torch.float32
