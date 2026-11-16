"""End-to-end tests for the conex.nn utilities: priority, precision, replication.

Each file carries its own small network builder so it runs on its own.
"""

import json

import pytest
import torch

from pymonntorch import NeuronGroup, SynapseGroup, Behavior

from conex import (
    NETWORK_PRIORITIES,
    NEURON_PRIORITIES,
    LAYER_PRIORITIES,
    SYNAPSE_PRIORITIES,
    ALL_PRIORITIES,
    prioritize_behaviors,
    PrecisionMode,
    PrecisionConfig,
    MixedPrecisionManager,
    PrecisionContext,
    convert_to_precision,
    behaviors_to_list,
    build_behavior_dict,
    save_structure,
    create_structure_from_dict,
    replicate,
    save_structure_dict_to_json,
    load_structure_dict_from_json,
    get_all_required_structures,
    save_ports,
    object_hook,
    ExtraCallableJSONEncoder,
    ELEMENTAL_STRUCTURE,
    Neocortex,
    SimpleDendriteStructure,
    SimpleDendriteComputation,
    LIF,
    NeuronAxon,
    SynapseInit,
    WeightInitializer,
    SimpleDendriticInput,
    PreSpikeCatcher,
    Port,
)


def build_network(seed=0):
    torch.manual_seed(seed)
    return Neocortex(dt=1.0)


def pop_behavior():
    return prioritize_behaviors([
        SimpleDendriteStructure(),
        SimpleDendriteComputation(),
        LIF(R=1.0, threshold=1.0, tau=1.0, v_reset=0.0, v_rest=0.0),
        NeuronAxon(),
    ])


class TestPriority:
    def test_all_priorities_merge_the_parts(self):
        assert set(ALL_PRIORITIES) == (
            set(NETWORK_PRIORITIES) | set(NEURON_PRIORITIES)
            | set(LAYER_PRIORITIES) | set(SYNAPSE_PRIORITIES))

    def test_known_behavior_priorities(self):
        assert NEURON_PRIORITIES["LIF"] == 260
        assert NEURON_PRIORITIES["Fire"] == 340
        assert SYNAPSE_PRIORITIES["SynapseInit"] == 2

    def test_prioritize_maps_to_registered(self):
        result = prioritize_behaviors([
            NeuronAxon(), SimpleDendriteStructure(),
            LIF(R=1.0, threshold=1.0, tau=1.0, v_reset=0.0, v_rest=0.0)])
        assert {k: v.__class__.__name__ for k, v in result.items()} == {
            380: "NeuronAxon", 220: "SimpleDendriteStructure", 260: "LIF"}

    def test_prioritize_rejects_unknown(self):
        class NotRegistered(Behavior):
            pass
        with pytest.raises(KeyError, match="NotRegistered"):
            prioritize_behaviors([NotRegistered()])


class TestPrecision:
    def test_mode_dtype_mapping(self):
        assert PrecisionConfig(mode=PrecisionMode.FULL).dtype == torch.float32
        assert PrecisionConfig(mode=PrecisionMode.HALF).dtype == torch.float16
        assert PrecisionConfig(mode=PrecisionMode.BFLOAT16).dtype == torch.bfloat16

    def test_bfloat16_disables_scaler(self):
        cfg = PrecisionConfig(mode=PrecisionMode.BFLOAT16)
        assert cfg.autocast_dtype == torch.bfloat16
        assert cfg.scaler_enabled is False

    def test_manager_cast_tensor(self):
        mgr = MixedPrecisionManager(PrecisionConfig(mode=PrecisionMode.HALF))
        t = torch.rand(4)
        assert mgr.cast_tensor(t).dtype == torch.float16
        assert mgr.cast_tensor(t, force_fp32=True).dtype == torch.float32

    def test_manager_full_precision_noop_autocast(self):
        mgr = MixedPrecisionManager(PrecisionConfig(mode=PrecisionMode.FULL))
        with mgr.autocast():
            pass  # no-op context, must not raise

    def test_manager_scale_loss_without_scaler(self):
        mgr = MixedPrecisionManager(PrecisionConfig())
        loss = torch.tensor(2.0)
        assert torch.equal(mgr.scale_loss(loss), loss)

    def test_precision_context_restores_dtype(self):
        original = torch.get_default_dtype()
        with PrecisionContext(torch.float64):
            assert torch.get_default_dtype() == torch.float64
        assert torch.get_default_dtype() == original

    def test_convert_to_precision(self):
        module = torch.nn.Linear(4, 4)
        out = convert_to_precision(module, PrecisionConfig(mode=PrecisionMode.HALF))
        assert out.weight.dtype == torch.float16

    def test_convert_keeps_fp32_layers(self):
        module = torch.nn.Sequential(torch.nn.Linear(4, 4))
        out = convert_to_precision(module, PrecisionConfig(mode=PrecisionMode.HALF),
                                   keep_fp32_layers={"0"})
        assert out[0].weight.dtype == torch.float32


class TestReplication:
    def _build_layer(self, net):
        exc = NeuronGroup(net=net, size=8, tag="exc", behavior=pop_behavior())
        inh = NeuronGroup(net=net, size=4, tag="inh", behavior=pop_behavior())
        syn = SynapseGroup(net=net, src=exc, dst=inh, tag="s",
                           behavior=prioritize_behaviors([
                               SynapseInit(), WeightInitializer(mode="randn"),
                               SimpleDendriticInput(current_coef=0.1), PreSpikeCatcher()]))
        from conex import CorticalLayer
        return CorticalLayer(net=net, excitatory_neurongroup=exc,
                             inhibitory_neurongroup=inh, synapsegroups=[syn], tag="l4")

    def test_behaviors_roundtrip(self):
        beh = {220: SimpleDendriteStructure(), 260: LIF(R=1.0, threshold=1.0, tau=1.0,
                                                        v_reset=0.0, v_rest=0.0)}
        lst = behaviors_to_list(beh)
        rebuilt = build_behavior_dict(lst)
        assert set(rebuilt) == {220, 260}
        assert rebuilt[220].__class__ is SimpleDendriteStructure
        assert rebuilt[260].__class__ is LIF

    def test_save_structure_captures_layer(self):
        net = build_network(seed=1)
        layer = self._build_layer(net)
        d = save_structure(layer)
        assert d["class"] is type(layer)
        assert "excitatory_neurongroup" in d
        assert "built_structures" in d

    def test_replicate_rebuilds_on_new_network(self):
        net = build_network(seed=2)
        layer = self._build_layer(net)
        net.initialize()
        new_net = build_network(seed=2)
        new_layer, _ = replicate(layer, new_net)
        assert new_layer.__class__ is type(layer)
        assert new_layer.exc_pop.size == 8
        assert new_layer.inh_pop.size == 4
        assert len(new_layer.synapsegroups) == 1

    def test_create_structure_from_dict_rebuilds_neurongroup(self):
        # a bare NeuronGroup has no required_helper, so save_structure can't
        # serialize it on its own; the NeuronGroup branch of
        # create_structure_from_dict is reached via a parent's built_structures.
        # Exercise that branch directly with the dict shape save_structure emits.
        net = build_network(seed=6)
        d = {
            "class": NeuronGroup,
            "size": 5,
            "tag": "p",
            "behavior": behaviors_to_list({220: SimpleDendriteStructure()}),
        }
        rebuilt = create_structure_from_dict(net, d)
        assert rebuilt.__class__ is NeuronGroup
        assert rebuilt.size == 5
        assert 220 in rebuilt.behavior

    def test_get_all_required_structures(self):
        net = build_network(seed=3)
        layer = self._build_layer(net)
        required = get_all_required_structures(layer)
        assert layer.exc_pop in required
        assert layer.inh_pop in required
        assert layer.synapsegroups[0] in required

    def test_json_roundtrip(self, tmp_path):
        net = build_network(seed=4)
        layer = self._build_layer(net)
        d = save_structure(layer)
        path = str(tmp_path / "layer.json")
        save_structure_dict_to_json(d, path)
        loaded = load_structure_dict_from_json(path)
        assert loaded["class"] is type(layer)
        assert "built_structures" in loaded

    def test_save_ports_serializes(self):
        # a port with no behavior hits a pre-existing bug (behaviors_to_list
        # on None), so give it one to exercise the working path
        net = build_network(seed=5)
        pop = NeuronGroup(net=net, size=4, tag="p", behavior=pop_behavior())
        ports = {"in": (None, [Port(object=pop, label=None,
                                    behavior={220: SimpleDendriteStructure()})])}
        saved = save_ports(ports, save_behavior_tag=True, save_behavior_priority=True)
        assert saved["in"][0] is None
        obj, label, beh_list = saved["in"][1][0]
        assert obj is pop
        assert label is None
        assert beh_list[0]["class"] is SimpleDendriteStructure

    def test_elemental_structure_list(self):
        assert NeuronGroup in ELEMENTAL_STRUCTURE
        assert SynapseGroup in ELEMENTAL_STRUCTURE

    def test_json_encoder_handles_callables(self):
        encoded = json.dumps({"f": torch.sum}, cls=ExtraCallableJSONEncoder)
        assert "python_callable" in encoded

    def test_object_hook_restores_callable(self):
        raw = {"f": ["python_callable", "torch", "sum"]}
        result = object_hook(raw)
        assert result["f"] is torch.sum
