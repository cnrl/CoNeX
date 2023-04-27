"""
Module of input and output neuronal populations.
"""

from pymonntorch import NeuronGroup, TaggableObject

from conex.behaviours.neurons.sensory.dataset import SpikeNdDataset
from conex.behaviours.neurons.specs import NeuronAxon, NeuronDendrite, SpikeTrace
from conex.nn.timestamps import NEURON_TIMESTAMPS


# TODO: Discuss whether location (motor) and sensory (representation) populations need to be defined as (distinct) subclasses of NeuronGroup
# TODO: Define spike analysis behaviors for output neuron groups

class InputLayer(TaggableObject):
    def __init__(
        self,
        net,
        sensory_dataloader=None,
        location_dataloader=None,
        sensory_size=None,
        location_size=None,
        sensory_trace=None,
        location_trace=None,
        sensory_data_dim=2,
        location_data_dim=2,
        tag=None,
        sensory_tag=None,
        location_tag=None,
        sensory_config={},
        location_config={}
    ):
        super().__init__(tag=tag, device=net.device)
        self.network = net

        self.sensory_dataloader = sensory_dataloader
        self.location_dataloader = location_dataloader

        if sensory_dataloader is not None:
            self.sensory_neurons = self.__get_ng(
                net,
                sensory_size,
                sensory_dataloader,
                sensory_tag,
                sensory_trace,
                sensory_data_dim,
                sensory_config
            )
            self.sensory_neurons.add_tag("Sensory")

        if location_dataloader is not None:
            self.location_neurons = self.__get_ng(
                net,
                location_size,
                location_dataloader,
                location_tag,
                location_trace,
                location_data_dim,
                location_config
            )
            self.location_neurons.add_tag("Location")

        self.add_tag(self.__class__.__name__)

    def connect(self, cortical_column, config={}):
        pass

    def __get_ng(self, net, size, dataloader, tag, trace, data_dim, config):
        behavior = {
            NEURON_TIMESTAMPS["Fire"]: SpikeNdDataset(
                dataloader=dataloader, N=data_dim
            ),
            NEURON_TIMESTAMPS["NeuronAxon"]: NeuronAxon(),
        }

        if trace is not None:
            behavior[NEURON_TIMESTAMPS["Trace"]] = SpikeTrace(tau_s=trace)

        behavior.update(config)  # TODO: should be made compatible with new config setup later

        return NeuronGroup(size, behavior, net, tag)
    
    @property
    def iteration(self):
        return self.network.iteration


class OutputLayer(TaggableObject):
    def __init__(
        self,
        net,
        representation_size=None,
        motor_size=None,
        representation_trace=None,
        motor_trace=None,
        tag=None,
        representation_tag=None,
        motor_tag=None,
    ):
        super().__init__(tag=tag, device=net.device)
        self.network = net

        self.representation_neurons = self.__get_ng(
            net,
            representation_size,
            representation_tag,
            representation_trace,
        )
        self.representation_neurons.add_tag("Representation")

        self.motor_neurons = self.__get_ng(
            net,
            motor_size,
            motor_tag,
            motor_trace,
        )
        self.motor_neurons.add_tag("Motor")

        self.add_tag(self.__class__.__name__)

    def __get_ng(self, net, size, tag, trace, **dendrite_params):
        behavior = {
            NEURON_TIMESTAMPS["NeuronDendrite"]: NeuronDendrite(**dendrite_params),
        }

        if trace is not None:
            behavior[NEURON_TIMESTAMPS["Trace"]] = SpikeTrace(tau_s=trace)

        return NeuronGroup(size, behavior, net, tag)
