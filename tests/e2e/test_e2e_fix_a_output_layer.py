"""Regression tests for OutputLayer construction (fix-a, bug 1).

OutputLayer.__get_ng called NeuronGroup(size, behavior, net, tag) positionally,
but pymonntorch 0.1.4's signature is (size, net, behavior, tag), so `net`
received the behavior dict and construction raised AttributeError.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from pymonntorch import NeuronDimension

from conex import (
    Neocortex,
    InputLayer,
    OutputLayer,
    Synapsis,
    Port,
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


def _input_layer(net):
    data = torch.rand(4, 1, 4, 4)
    labels = torch.tensor([0, 1, 0, 1])
    dl = DataLoader(TensorDataset(data, labels), batch_size=2)
    return InputLayer(net=net, input_dataloader=dl,
                      sensory_size=NeuronDimension(depth=1, height=4, width=4),
                      instance_duration=2, have_location=False,
                      output_ports={"data_out": (None, [("sensory_pop", {})])})


class TestOutputLayer:
    def test_builds_representation_and_motor_populations(self):
        net = build_network()
        ol = OutputLayer(net=net, representation_size=6, motor_size=3,
                         tag="out")
        assert ol.representation_pop.size == 6
        assert ol.motor_pop.size == 3
        assert "Representation" in ol.representation_pop.tags
        assert "Motor" in ol.motor_pop.tags
        assert ol in net.output_layers

    def test_runs_and_produces_output(self):
        net = build_network(seed=7)
        il = _input_layer(net)
        # the representation pop needs the full spiking pipeline to emit spikes
        repr_beh = prioritize_behaviors([
            SimpleDendriteComputation(),
            LIF(R=1.0, threshold=1.0, tau=1.0, v_reset=0.0, v_rest=0.0),
            NeuronAxon(),
        ])
        ol = OutputLayer(net=net, representation_size=16,
                         representation_user_defined=repr_beh,
                         input_ports={"in": (None, [("representation_pop", {})])})
        Synapsis(net=net, src=il, dst=ol, input_port="data_out",
                 output_port="in", synapsis_behavior=syn_behavior(),
                 synaptic_tag="in")
        net.initialize()
        net.simulate_iterations(3)
        # the input layer delivered spikes and the output layer produced a
        # spikes buffer of the right size
        assert il.sensory_pop.spikes.any()
        assert ol.representation_pop.spikes.shape[0] == 16

    def test_user_defined_behavior_reaches_population(self):
        net = build_network(seed=8)
        beh = {240: SimpleDendriteComputation()}
        ol = OutputLayer(net=net, representation_size=4,
                         representation_user_defined=beh)
        assert ol.representation_pop.behavior[240].__class__ is SimpleDendriteComputation
