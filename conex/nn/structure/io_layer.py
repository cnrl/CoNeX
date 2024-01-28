from pymonntorch import NetworkObject, Network, NeuronDimension, Behavior, NeuronGroup
from conex.behaviors.neurons.axon import NeuronAxon
from conex.behaviors.neurons.setters import SensorySetter, LocationSetter
from conex.behaviors.neurons.specs import SpikeTrace
from conex.behaviors.neurons.dendrite import SimpleDendriteStructure
from torch.utils.data.dataloader import DataLoader
from typing import Union, Dict, Callable
from conex.nn.priority import LAYER_PRIORITIES, NEURON_PRIORITIES
from conex.behaviors.layer.dataset import SpikeNdDataset


class InputLayer(NetworkObject):
    def __init__(
        self,
        net: Network,
        input_dataloader: DataLoader,
        have_sensory: bool = True,
        have_location: bool = False,
        have_label: bool = True,
        sensory_size: Union[int, NeuronDimension] = None,
        location_size: Union[int, NeuronDimension] = None,
        sensory_axon: Behavior = NeuronAxon,
        sensory_axon_params: dict = None,
        location_axon: Behavior = NeuronAxon,
        location_axon_params: dict = None,
        silent_interval: int = 0,
        instance_duration: int = 0,
        sensory_trace: float = None,
        location_trace: float = None,
        sensory_data_dim: int = 2,
        location_data_dim: int = 2,
        tag: str = None,
        behavior: Dict[int, Behavior] = None,
        sensory_tag: str = None,
        location_tag: str = None,
        sensory_user_defined: Dict[int, Behavior] = None,
        location_user_defined: Dict[int, Behavior] = None,
    ):
        behavior = {} if behavior is None else behavior

        if LAYER_PRIORITIES["SpikeNdDataset"] not in behavior:
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
            self.add_sub_structure(self.sensory_pop)

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
            self.add_sub_structure(self.location_pop)

    def __get_ng(
        self,
        net: Network,
        size: Union[int, NeuronDimension],
        tag: Union[str, None],
        trace: Union[float, None],
        setter: Callable,
        axon: Behavior = NeuronAxon,
        axon_params: dict = None,
        user_defined: Dict[int, Behavior] = None,
    ) -> NeuronGroup:
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

        return NeuronGroup(size=size, behavior=behavior, net=net, tag=tag)

    def __repr__(self) -> str:
        result = (
            self.__class__.__name__
            + "("
            + f"Sensory Population {self.sensory_pop.tags[0](self.sensory_pop.size)}"
            if hasattr(self, "sensory_pop")
            else ""
            + f"Location Population {self.location_pop.tags[0](self.location_pop.size)}"
            if hasattr(self, "location_pop")
            else "" + "){"
        )
        for k in sorted(list(self.behavior.keys())):
            result += str(k) + ":" + str(self.behavior[k])
        return result + "}"


class OutputLayer(NetworkObject):
    def __init__(
        self,
        net: Network,
        representation_size: Union[int, NeuronDimension] = None,
        motor_size: Union[int, NeuronDimension] = None,
        representation_trace: Union[float, None] = None,
        motor_trace: Union[float, None] = None,
        representation_dendrite_structure: Callable = SimpleDendriteStructure,
        representation_dendrite_structure_params: dict = None,
        motor_dendrite_structure: Callable = SimpleDendriteStructure,
        motor_dendrite_structure_params: dict = None,
        tag: str = None,
        behavior: Dict[int, Behavior] = None,
        representation_tag: str = None,
        motor_tag: str = None,
        representation_user_defined: Dict[int, Behavior] = None,
        motor_user_defined: Dict[int, Behavior] = None,
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
            self.add_sub_structure(self.representation_pop)

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
            self.add_sub_structure(self.motor_pop)

        self.add_tag(self.__class__.__name__)

    def __get_ng(
        self,
        net: Network,
        size: Union[int, NeuronDimension],
        tag: Union[str, None],
        trace: Union[float, None] = None,
        dendrite_structure: Callable = SimpleDendriteStructure,
        dendrite_structure_params: dict = None,
        user_defined: Dict[int, Behavior] = None,
    ) -> NeuronGroup:
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

    def __repr__(self) -> str:
        result = (
            self.__class__.__name__
            + "("
            + f"representation Population {self.representation_pop.tags[0](self.representation_pop.size)}"
            if hasattr(self, "representation_pop")
            else "" + f"Motor Population {self.motor_pop.tags[0](self.motor_pop.size)}"
            if hasattr(self, "motor_pop")
            else "" + "){"
        )
        for k in sorted(list(self.behavior.keys())):
            result += str(k) + ":" + str(self.behavior[k])
        return result + "}"
