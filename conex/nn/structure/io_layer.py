from pymonntorch import NetworkObject, Network, NeuronDimension, Behavior, NeuronGroup
from conex.behaviors.neurons.axon import NeuronAxon
from conex.behaviors.neurons.setters import SensorySetter, LocationSetter
from conex.behaviors.neurons.dendrite import SimpleDendriteStructure
from torch.utils.data.dataloader import DataLoader
from typing import Union, Dict, Callable, Tuple, List
from conex.nn.priority import LAYER_PRIORITIES, NEURON_PRIORITIES
from conex.behaviors.layer.dataset import SpikeNdDataset
from conex.nn.structure.port import Port


class InputLayer(NetworkObject):
    """A sample input layer.

    Args:
        input_dataloader (torch dataloader): The dataloader of input layer.
        have_sensory (bool): Whether dataloader return sensory data or not.
        have_location (bool): Whether dataloader location data or not.
        have_label (bool): Whether dataloader label or not.
        sensory_size (int or behavior): The size of each sensory neurongroup.
        location_size (int or behavior): The size of each location neurongroup.
        sensory_axon (behavior): The behavior class for axon paradigm of sensory neurongroup.
        sensory_axon_params (dict): Parameters for axon class of sensory neurongroup.
        location_axon (behavior): The behavior class for axon paradigm of location neurongroup.
        location_axon_params (dict): Parameters for axon class of location neurongroup.
        silent_interval (int): Empty interval between two samples.
        instance_duration (int): Each sample duraiton
        sensory_data_dim (int): The number of dimension of sensory data.
        location_data_dim (int): The number of dimension of location data.
        behavior (dict): The behavior for the InputLayer itself.
        tag (str): tag of the InputLayer divided by ",".
        sensory_tag (str): tag of the sensory population divided by ",".
        location_tag (ste): tag of the location population divided by ",".
        sensory_user_defined (dict): Additional behavior for sensory neurongroup.
        location_user_defined (dict): Additional behavior for location neurongroup.
    """

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
        sensory_data_dim: int = 2,
        location_data_dim: int = 2,
        tag: str = None,
        behavior: Dict[int, Behavior] = None,
        sensory_tag: str = None,
        location_tag: str = None,
        output_ports: Dict[
            str, Tuple[Union[Tuple, None], List[Tuple[str, Dict[int, Behavior]]]]
        ] = None,
        sensory_user_defined: Dict[int, Behavior] = None,
        location_user_defined: Dict[int, Behavior] = None,
    ):
        behavior = {} if behavior is None else behavior

        if LAYER_PRIORITIES["SpikeNdDataset"] not in behavior:
            behavior[LAYER_PRIORITIES["SpikeNdDataset"]] = SpikeNdDataset(
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
                setter=LocationSetter,
                axon=location_axon,
                axon_params=location_axon_params,
                user_defined=location_user_defined,
            )

            self.location_pop.layer = self
            self.add_sub_structure(self.location_pop)

        self.output_ports = _create_port(output_ports, self)
        self.input_ports = {}

    def __get_ng(
        self,
        net: Network,
        size: Union[int, NeuronDimension],
        tag: Union[str, None],
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

        if user_defined is not None:
            behavior.update(user_defined)

        return NeuronGroup(size=size, behavior=behavior, net=net, tag=tag)

    def __repr__(self) -> str:
        result = (
            self.__class__.__name__
            + "("
            + f"Sensory Population {self.sensory_pop.tags[0]}({self.sensory_pop.size})"
            if hasattr(self, "sensory_pop")
            else ""
            + f"Location Population {self.location_pop.tags[0]}({self.location_pop.size})"
            if hasattr(self, "location_pop")
            else "" + "){"
        )
        for k in sorted(list(self.behavior.keys())):
            result += str(k) + ":" + str(self.behavior[k])
        return result + "}"


class OutputLayer(NetworkObject):
    """A sample output layer.

    Args:
        representation_size (int or behavior): The size of each representation neurongroup.
        motor_size (int or behavior): The size of each motor neurongroup.
        representation_dendrite_structure (Callable): Dendrite structure for representation population.
        representation_dendrite_structure_params (dict): The parameters for dendrite structure of representation population.
        motor_dendrite_structure (Callable): Dendrite structure for motor population.
        motor_dendrite_structure_params (dict): The parameters for dendrite structure of representation population.
        behavior (dict): The behavior for the InputLayer itself.
        tag (str): tag of the InputLayer divided by ",".
        representation_tag (str): tag of the representation population divided by ",".
        motor_tag (str): tag of the motor population divided by ",".
        representation_user_defined (dict): Additional behavior for representation neurongroup.
        motor_user_defined (dict): Additional behavior for motor neurongroup.
    """

    def __init__(
        self,
        net: Network,
        representation_size: Union[int, NeuronDimension] = None,
        motor_size: Union[int, NeuronDimension] = None,
        representation_dendrite_structure: Callable = SimpleDendriteStructure,
        representation_dendrite_structure_params: dict = None,
        motor_dendrite_structure: Callable = SimpleDendriteStructure,
        motor_dendrite_structure_params: dict = None,
        tag: str = None,
        behavior: Dict[int, Behavior] = None,
        representation_tag: str = None,
        motor_tag: str = None,
        input_ports: Dict[
            str, Tuple[Union[Tuple, None], List[Tuple[str, Dict[int, Behavior]]]]
        ] = None,
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
                dendrite_structure=motor_dendrite_structure,
                dendrite_structure_params=motor_dendrite_structure_params,
                user_defined=motor_user_defined,
            )

            self.motor_pop.layer = self
            self.add_sub_structure(self.motor_pop)

        self.add_tag(self.__class__.__name__)

        self.input_ports = _create_port(input_ports, self)
        self.output_ports = {}

    def __get_ng(
        self,
        net: Network,
        size: Union[int, NeuronDimension],
        tag: Union[str, None],
        dendrite_structure: Callable = SimpleDendriteStructure,
        dendrite_structure_params: dict = None,
        user_defined: Dict[int, Behavior] = None,
    ) -> NeuronGroup:
        dendrite_structure_params = (
            dendrite_structure_params if dendrite_structure_params is not None else {}
        )
        behavior = {}

        if dendrite_structure:
            behavior[NEURON_PRIORITIES["SimpleDendriteStructure"]] = dendrite_structure(
                **dendrite_structure_params
            )

        if user_defined is not None:
            behavior.update(user_defined)

        return NeuronGroup(size, behavior, net, tag)

    def __repr__(self) -> str:
        result = (
            self.__class__.__name__
            + "("
            + f"representation Population {self.representation_pop.tags[0]}({self.representation_pop.size})"
            if hasattr(self, "representation_pop")
            else ""
            + f"Motor Population {self.motor_pop.tags[0]}({self.motor_pop.size})"
            if hasattr(self, "motor_pop")
            else "" + "){"
        )
        for k in sorted(list(self.behavior.keys())):
            result += str(k) + ":" + str(self.behavior[k])
        return result + "}"


def _create_port(
    port_dict: Dict[
        str, Tuple[Union[Tuple, None], List[Tuple[str, Dict[int, Behavior]]]]
    ],
    obj: Union[InputLayer, OutputLayer],
):
    result = {}
    if port_dict is not None:
        result = {
            k: (v[0], [Port(object=getattr(obj, sp[0]), behavior=sp[1]) for sp in v[1]])
            for k, v in port_dict.items()
        }
    return result
