"""
Module of input and output neuronal populations.
"""

from pymonntorch import NeuronGroup, NetworkObject

from conex.behaviors.neurons.setters import LocationSetter, SensorySetter
from conex.behaviors.neurons.dendrite import SimpleDendriteStructure
from conex.behaviors.neurons.axon import NeuronAxon
from conex.behaviors.neurons.specs import SpikeTrace
from conex.behaviors.layer.dataset import SpikeNdDataset

from conex.nn.priorities import NEURON_PRIORITIES, LAYER_PRIORITIES

# TODO: Define spike analysis behaviors for output neuron groups
# TODO: Docstring


class InputLayer(NetworkObject):
    def __init__(
        self,
        net,
        input_dataloader,
        have_sensory=True,
        have_location=False,
        have_label=True,
        sensory_size=None,
        location_size=None,
        sensory_axon=NeuronAxon,
        sensory_axon_params=None,
        location_axon=NeuronAxon,
        location_axon_params=None,
        silent_interval=0,
        instance_duration=0,
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
        behavior = {} if behavior is None else behavior

        if LAYER_PRIORITIES["InputDataset"] not in behavior:
            behavior[LAYER_PRIORITIES["InputDataset"]] = SpikeNdDataset(
                dataloader=input_dataloader,
                ndim_sensory=sensory_data_dim,
                ndim_location=location_data_dim,
                have_location=have_location,
                have_sensory=have_sensory,
                have_label=have_label,
                silent_interval=silent_interval,
                instance_duration=instance_duration,
            )

        super().__init__(tag=tag, network=net, behavior=behavior, device=net.device)
        self.add_tag(self.__class__.__name__)

        self.network = net
        net.input_layers.append(self)

        sensory_tag = "Sensory" if sensory_tag is None else "Sensory," + sensory_tag

        if have_sensory and sensory_size is not None:
            self.sensory_pop = self.__get_ng(
                net=net,
                size=sensory_size,
                tag=sensory_tag,
                trace=sensory_trace,
                setter=SensorySetter,
                axon=sensory_axon,
                axon_params=sensory_axon_params,
                user_defined=sensory_user_defined,
            )

            self.sensory_pop.layer = self

        location_tag = (
            "Location" if location_tag is None else "Location," + location_tag
        )

        if have_location and location_size is not None:
            self.location_pop = self.__get_ng(
                net=net,
                size=location_size,
                tag=location_tag,
                trace=location_trace,
                setter=LocationSetter,
                axon=location_axon,
                axon_params=location_axon_params,
                user_defined=location_user_defined,
            )

            self.location_pop.layer = self

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

    def __get_ng(
        self,
        net,
        size,
        tag,
        trace,
        setter,
        axon=NeuronAxon,
        axon_params=None,
        user_defined=None,
    ):
        behavior = {
            NEURON_PRIORITIES["Fire"]: setter(),
        }

        if axon:
            params = axon_params if axon_params is not None else {}
            behavior[NEURON_PRIORITIES["NeuronAxon"]] = axon(**params)

        if trace is not None:
            behavior[NEURON_PRIORITIES["Trace"]] = SpikeTrace(tau_s=trace)

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
        representation_dendrite_structure=SimpleDendriteStructure,
        representation_dendrite_structure_params=None,
        motor_dendrite_structure=SimpleDendriteStructure,
        motor_dendrite_structure_params=None,
        tag=None,
        behavior=None,
        representation_tag=None,
        motor_tag=None,
        representation_user_defined=None,
        motor_user_defined=None,
    ):
        behavior = {} if behavior is None else behavior
        super().__init__(tag=tag, network=net, behavior=behavior, device=net.device)
        self.network = net
        net.output_layers.append(self)

        representation_tag = (
            "Representation"
            if representation_tag is None
            else "Representation," + representation_tag
        )

        if representation_size is not None:
            self.representation_pop = self.__get_ng(
                net=net,
                size=representation_size,
                tag=representation_tag,
                trace=representation_trace,
                dendrite_structure=representation_dendrite_structure,
                dendrite_structure_params=representation_dendrite_structure_params,
                user_defined=representation_user_defined,
            )

            self.representation_pop.layer = self

        motor_tag = "Motor" if motor_tag is None else "Motor," + motor_tag

        if motor_size is not None:
            self.motor_pop = self.__get_ng(
                net=net,
                size=motor_size,
                tag=motor_tag,
                trace=motor_trace,
                dendrite_structure=motor_dendrite_structure,
                dendrite_structure_params=motor_dendrite_structure_params,
                user_defined=motor_user_defined,
            )

            self.motor_pop.layer = self

        self.add_tag(self.__class__.__name__)

    def __get_ng(
        self,
        net,
        size,
        tag,
        trace=None,
        dendrite_structure=SimpleDendriteStructure,
        dendrite_structure_params=None,
        user_defined=None,
    ):
        dendrite_structure_params = (
            dendrite_structure_params if dendrite_structure_params is not None else {}
        )
        behavior = {}

        if dendrite_structure:
            behavior[NEURON_PRIORITIES["DendriteStructure"]] = dendrite_structure(
                **dendrite_structure_params
            )

        if trace is not None:
            behavior[NEURON_PRIORITIES["Trace"]] = SpikeTrace(tau_s=trace)

        if user_defined is not None:
            behavior.update(user_defined)

        return NeuronGroup(size, behavior, net, tag)
