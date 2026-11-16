"""End-to-end tests for the CoNeX network structures.

Local build_network helper: the shared tests/e2e/conftest.py is owned by
another worker and not yet merged, so each file carries its own builder.

OutputLayer is NOT covered: its __get_ng calls NeuronGroup(size, behavior, net,
tag) positionally, but pymonntorch 0.1.4's signature is (size, net, behavior,
tag), so `net` receives the behavior dict and construction raises
AttributeError. This is a pre-existing bug in conex/nn/structure/io_layer.py
that no existing test exercises, and it is outside this task's file ownership.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from pymonntorch import NeuronGroup, SynapseGroup, NeuronDimension

from conex import (
    Neocortex,
    Container,
    Layer,
    CorticalLayer,
    Port,
    Synapsis,
    CorticalLayerConnection,
    CorticalColumn,
    InputLayer,
    SimpleDendriteStructure,
    SimpleDendriteComputation,
    LIF,
    NeuronAxon,
    SynapseInit,
    WeightInitializer,
    SimpleDendriticInput,
    PreSpikeCatcher,
    prioritize_behaviors,
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


def syn_behavior():
    return prioritize_behaviors([
        SynapseInit(),
        WeightInitializer(mode="randn"),
        SimpleDendriticInput(current_coef=0.1),
        PreSpikeCatcher(),
    ])


def make_pop(net, size, tag):
    return NeuronGroup(net=net, size=size, tag=tag, behavior=pop_behavior())


class TestNeocortex:
    def test_constructs_with_time_resolution(self):
        net = build_network()
        net.initialize()
        assert net.dt == 1.0
        assert net.def_dtype == torch.float32
        assert net.input_layers == []
        assert net.cortical_columns == []

    def test_custom_dt(self):
        net = Neocortex(dt=2.5)
        net.initialize()
        assert net.dt == 2.5


class TestLayer:
    def test_wraps_neurongroups(self):
        net = build_network()
        p1 = make_pop(net, 6, "p1")
        p2 = make_pop(net, 4, "p2")
        layer = Layer(net=net, neurongroups=[p1, p2], tag="L")
        assert layer.neurongroups == [p1, p2]
        assert len(layer.sub_structures) == 2

    def test_empty_layer(self):
        net = build_network()
        layer = Layer(net=net, tag="empty")
        assert layer.neurongroups == []
        assert layer.synapsegroups == []


class TestCorticalLayer:
    def test_exc_and_inh_populations(self):
        net = build_network()
        exc = make_pop(net, 8, "exc")
        inh = make_pop(net, 4, "inh")
        syn = SynapseGroup(net=net, src=exc, dst=inh, tag="s", behavior=syn_behavior())
        layer = CorticalLayer(net=net, excitatory_neurongroup=exc,
                              inhibitory_neurongroup=inh, synapsegroups=[syn], tag="l4")
        assert layer.exc_pop is exc
        assert layer.inh_pop is inh
        assert layer.synapsegroups == [syn]

    def test_runs_spiking_pipeline(self):
        net = build_network(seed=1)
        exc = make_pop(net, 8, "exc")
        inh = make_pop(net, 4, "inh")
        syn = SynapseGroup(net=net, src=exc, dst=inh, tag="s", behavior=syn_behavior())
        CorticalLayer(net=net, excitatory_neurongroup=exc,
                      inhibitory_neurongroup=inh, synapsegroups=[syn], tag="l4")
        net.initialize()
        net.simulate_iterations(3)
        assert exc.spikes.shape[0] == 8
        assert inh.spikes.shape[0] == 4


class TestPort:
    def test_named_tuple_fields(self):
        net = build_network()
        pop = make_pop(net, 4, "p")
        port = Port(object=pop, label=None)
        assert port.object is pop
        assert port.label is None
        assert port.behavior is None

    def test_with_behavior(self):
        net = build_network()
        pop = make_pop(net, 4, "p")
        port = Port(object=pop, label="x", behavior={100: SimpleDendriteStructure()})
        assert port.label == "x"
        assert 100 in port.behavior


class TestCorticalLayerConnection:
    def test_creates_synapse_between_layers(self):
        net = build_network()
        l4 = _two_pop_layer(net, "l4")
        l2 = _two_pop_layer(net, "l2")
        clc = CorticalLayerConnection(
            net=net, src=l4, dst=l2,
            connections=[("exc_pop", "exc_pop", syn_behavior(), "l4l2")])
        assert len(clc.synapses) == 1
        assert clc.synapses[0].src is l4.exc_pop
        assert clc.synapses[0].dst is l2.exc_pop

    def test_deferred_connect(self):
        net = build_network()
        l4 = _two_pop_layer(net, "l4")
        l2 = _two_pop_layer(net, "l2")
        clc = CorticalLayerConnection(
            net=net, connections=[("exc_pop", "exc_pop", syn_behavior(), "c")])
        assert clc.synapses == []  # nothing until both ends are connected
        clc.connect_src(l4)
        assert clc.synapses == []  # still missing dst
        clc.connect_dst(l2)
        assert len(clc.synapses) == 1


class TestCorticalColumn:
    def test_holds_layers_and_connections(self):
        net = build_network()
        l4 = _two_pop_layer(net, "l4")
        l2 = _two_pop_layer(net, "l2")
        clc = CorticalLayerConnection(
            net=net, connections=[("exc_pop", "exc_pop", syn_behavior(), "l4l2")])
        col = CorticalColumn(net=net, layers={"l4": l4, "l2": l2},
                             layer_connections=[("l4", "l2", clc)], tag="col")
        assert set(col.layers) == {"l4", "l2"}
        assert clc.src is l4 and clc.dst is l2  # wired by create_layer_connections
        assert len(clc.synapses) == 1

    def test_column_runs_end_to_end(self):
        net = build_network(seed=2)
        l4 = _two_pop_layer(net, "l4")
        l2 = _two_pop_layer(net, "l2")
        clc = CorticalLayerConnection(
            net=net, connections=[("exc_pop", "exc_pop", syn_behavior(), "l4l2")])
        CorticalColumn(net=net, layers={"l4": l4, "l2": l2},
                       layer_connections=[("l4", "l2", clc)], tag="col")
        net.initialize()
        net.simulate_iterations(3)
        assert l4.exc_pop.spikes.shape[0] == 8
        assert l2.exc_pop.spikes.shape[0] == 8


class TestSynapsis:
    def test_creates_synapse_between_ports(self):
        net = build_network()
        il = _input_layer(net)
        pop = make_pop(net, 16, "p")
        layer = Layer(net=net, neurongroups=[pop], tag="L",
                      input_ports={"in": (None, [Port(object=pop, label=None)])})
        syn = Synapsis(net=net, src=il, dst=layer, input_port="data_out",
                       output_port="in", synapsis_behavior=syn_behavior(),
                       synaptic_tag="in")
        assert len(syn.synapses) == 1
        assert syn.synapses[0].src is il.sensory_pop
        assert syn.synapses[0].dst is pop

    def test_no_synapse_without_matching_port(self):
        net = build_network()
        il = _input_layer(net)
        pop = make_pop(net, 16, "p")
        layer = Layer(net=net, neurongroups=[pop], tag="L")  # no input_ports
        syn = Synapsis(net=net, src=il, dst=layer, input_port="data_out",
                       output_port="missing", synapsis_behavior=syn_behavior())
        assert syn.synapses == []

    def test_input_to_column_pipeline(self):
        net = build_network(seed=3)
        il = _input_layer(net)
        pop = make_pop(net, 16, "p")
        layer = Layer(net=net, neurongroups=[pop], tag="L",
                      input_ports={"in": (None, [Port(object=pop, label=None)])})
        Synapsis(net=net, src=il, dst=layer, input_port="data_out",
                 output_port="in", synapsis_behavior=syn_behavior(), synaptic_tag="in")
        net.initialize()
        net.simulate_iterations(2)
        # the input layer delivered a sensory pattern
        assert il.x.shape == (16,)


class TestInputLayer:
    def test_builds_sensory_population(self):
        net = build_network()
        il = _input_layer(net)
        assert il.sensory_pop.size == 16
        assert "Sensory" in il.sensory_pop.tags

    def test_delivers_data_during_simulation(self):
        net = build_network(seed=4)
        il = _input_layer(net)
        net.initialize()
        net.simulate_iterations(2)
        # x holds the raw (float) sensory pattern delivered by the dataloader
        assert il.x.shape == (16,)
        assert il.x.dtype == torch.float32

    def test_location_population(self):
        net = build_network()
        data = torch.rand(4, 1, 4, 4)
        loc = torch.rand(4, 2)
        labels = torch.tensor([0, 1, 0, 1])
        dl = DataLoader(TensorDataset(data, loc, labels), batch_size=2)
        il = InputLayer(net=net, input_dataloader=dl,
                        sensory_size=NeuronDimension(depth=1, height=4, width=4),
                        location_size=2, have_location=True, instance_duration=2,
                        output_ports={"data_out": (None, [("sensory_pop", {})])})
        assert il.location_pop.size == 2


class TestContainer:
    def test_adds_sub_structures(self):
        net = build_network()
        p1 = make_pop(net, 4, "a")
        p2 = make_pop(net, 4, "b")
        c = Container(net=net, sub_structures=[p1, p2], tag="c")
        assert c.sub_structures == [p1, p2]
        assert c.required_helper() == [p1, p2]

    def test_ports_helper_roundtrip(self):
        net = build_network()
        pop = make_pop(net, 4, "p")
        ports = {"in": (None, [Port(object=pop, label=None)])}
        # list form: index into the structure list
        result = Container.ports_helper(ports, [pop])
        assert result["in"][1][0][0] == 0  # pop is at index 0


def _two_pop_layer(net, tag):
    exc = make_pop(net, 8, tag + "exc")
    inh = make_pop(net, 4, tag + "inh")
    return CorticalLayer(net=net, excitatory_neurongroup=exc,
                         inhibitory_neurongroup=inh, tag=tag)


def _input_layer(net):
    data = torch.rand(4, 1, 4, 4)
    labels = torch.tensor([0, 1, 0, 1])
    dl = DataLoader(TensorDataset(data, labels), batch_size=2)
    return InputLayer(net=net, input_dataloader=dl,
                      sensory_size=NeuronDimension(depth=1, height=4, width=4),
                      instance_duration=2, have_location=False,
                      output_ports={"data_out": (None, [("sensory_pop", {})])})
