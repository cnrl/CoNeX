"""
Module of input and output neuronal populations.
"""

from pymonntorch import NeuronGroup, TaggableObject

from conex.behaviours.neurons.sensory.dataset import SpikeNdDataset
from conex.behaviours.neurons.specs import NeuronAxon, NeuronDendrite, SpikeTrace
from conex.nn.timestamps import NEURON_TIMESTAMPS


# TODO: Discuss whether location (motor) and sensory (representation) populations need to be defined as (distinct) subclasses of NeuronGroup
# TODO: Define spike analysis behaviors for output neuron groups


class InputLayer(NetworkObject):
    def __init__(
        self,
        net,
        input_dataloader=None,
        have_sensory=True,
        have_location=False,
        have_label=True,
        sensory_size=None,
        location_size=None,
        sensory_trace=None,
        location_trace=None,
        sensory_data_dim=2,
        location_data_dim=2,
        tag=None,
        behavior=None,
        sensory_tag=None,
        location_tag=None,
        sensory_user_defined=None,
        location_user_defined=None,
    ):
        self.network = net
        net.input_layers.append(self)
        behavior = {} if behavior is None else behavior

        if have_sensory:
            self.sensory_pop = self.__get_ng(
                net,
                sensory_size,
                sensory_tag,
                sensory_trace,
                sensory_data_dim,
                sensory_user_defined,
            )
            self.sensory_pop.add_tag("Sensory")
            self.sensory_pop.layer = self

        if have_location:
            self.location_pop = self.__get_ng(
                net,
                location_size,
                location_tag,
                location_trace,
                location_data_dim,
                location_user_defined,
            )
            self.location_pop.add_tag("Location")
            self.location_pop.layer = self

        
        super().__init__(tag=tag, behavior=behavior, device=net.device)
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

    def __get_ng(self, net, size, tag, trace, data_dim, user_defined=None):
        behavior = {
            NEURON_TIMESTAMPS["Fire"]: SpikeNdDataset(
                dataloader=dataloader, N=data_dim
            ),
            NEURON_TIMESTAMPS["NeuronAxon"]: NeuronAxon(),
        }

        if trace is not None:
            behavior[NEURON_TIMESTAMPS["Trace"]] = SpikeTrace(tau_s=trace)

        if user_defined is not None:
            behavior.update(user_defined)

        return NeuronGroup(size, behavior, net, tag)


class OutputLayer(NetworkObject):
    def __init__(
        self,
        net,
        representation_size=None,
        motor_size=None,
        representation_trace=None,
        motor_trace=None,
        representation_dendrite_params=None,
        motor_dendrite_params=None,
        tag=None,
        behavior=None,
        representation_tag=None,
        motor_tag=None,
        representation_user_defined=None,
        motor_user_defined=None,
    ):
        behavior = {} if behavior is None else behavior
        super().__init__(tag=tag, behavior=behavior, device=net.device)
        self.network = net
        net.output_layers.append(self)

        self.representation_pop = self.__get_ng(
            net,
            representation_size,
            representation_tag,
            representation_trace,
            representation_dendrite_params,
            representation_user_defined,
        )
        self.representation_pop.add_tag("Representation")

        self.motor_pop = self.__get_ng(
            net,
            motor_size,
            motor_tag,
            motor_trace,
            motor_dendrite_params,
            motor_user_defined,
        )
        self.motor_pop.add_tag("Motor")

        self.add_tag(self.__class__.__name__)

    def __get_ng(self, net, size, tag, trace, dendrite_params, user_defined):
        dendrite_params = dendrite_params if dendrite_params is not None else {}
        behavior = {
            NEURON_TIMESTAMPS["NeuronDendrite"]: NeuronDendrite(**dendrite_params),
        }

        if trace is not None:
            behavior[NEURON_TIMESTAMPS["Trace"]] = SpikeTrace(tau_s=trace)

        if user_defined is not None:
            behavior.update(user_defined)

        return NeuronGroup(size, behavior, net, tag)
