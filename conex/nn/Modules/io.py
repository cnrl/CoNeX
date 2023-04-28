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
        location_config={},
    ):
        super().__init__(tag=tag, device=net.device)
        self.network = net

        self.sensory_dataloader = sensory_dataloader
        self.location_dataloader = location_dataloader

        if sensory_dataloader is not None:
            self.sensory_pop = self.__get_ng(
                net,
                sensory_size,
                sensory_dataloader,
                sensory_tag,
                sensory_trace,
                sensory_data_dim,
                sensory_config,
            )
            self.sensory_pop.add_tag("Sensory")

        if location_dataloader is not None:
            self.location_pop = self.__get_ng(
                net,
                location_size,
                location_dataloader,
                location_tag,
                location_trace,
                location_data_dim,
                location_config,
            )
            self.location_pop.add_tag("Location")

        self.add_tag(self.__class__.__name__)

    def connect(
        self,
        cortical_column,
        sensory_L4_syn_config=None,
        sensory_L6_syn_config=None,
        location_L6_syn_config=None,
    ):
        synapses = {}

        if sensory_L4_syn_config:
            if hasattr(self, "sensory_pop") and hasattr(cortical_column, "L4"):
                synapses[
                    "sensory_L4_synapse"
                ] = cortical_column._add_synaptic_connection(
                    self.sensory_pop, cortical_column.L4, sensory_L4_syn_config
                )

        if sensory_L6_syn_config:
            if hasattr(self, "sensory_pop") and hasattr(cortical_column, "L6"):
                synapses[
                    "sensory_L6_synapse"
                ] = cortical_column._add_synaptic_connection(
                    self.sensory_pop, cortical_column.L6, sensory_L6_syn_config
                )

        if location_L6_syn_config:
            if hasattr(self, "location_pop") and hasattr(cortical_column, "L6"):
                synapses[
                    "location_L6_synapse"
                ] = cortical_column._add_synaptic_connection(
                    self.location_pop, cortical_column.L6, sensory_L6_syn_config
                )

        return synapses

    def __get_ng(self, net, size, dataloader, tag, trace, data_dim, config):
        behavior = {
            NEURON_TIMESTAMPS["Fire"]: SpikeNdDataset(
                dataloader=dataloader, N=data_dim
            ),
            NEURON_TIMESTAMPS["NeuronAxon"]: NeuronAxon(),
        }

        if trace is not None:
            behavior[NEURON_TIMESTAMPS["Trace"]] = SpikeTrace(tau_s=trace)

        behavior.update(
            config
        )  # TODO: should be made compatible with new config setup later

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

        self.representation_pop = self.__get_ng(
            net,
            representation_size,
            representation_tag,
            representation_trace,
        )
        self.representation_pop.add_tag("Representation")

        self.motor_pop = self.__get_ng(
            net,
            motor_size,
            motor_tag,
            motor_trace,
        )
        self.motor_pop.add_tag("Motor")

        self.add_tag(self.__class__.__name__)

    def __get_ng(self, net, size, tag, trace, **dendrite_params):
        behavior = {
            NEURON_TIMESTAMPS["NeuronDendrite"]: NeuronDendrite(**dendrite_params),
        }

        if trace is not None:
            behavior[NEURON_TIMESTAMPS["Trace"]] = SpikeTrace(tau_s=trace)

        return NeuronGroup(size, behavior, net, tag)
