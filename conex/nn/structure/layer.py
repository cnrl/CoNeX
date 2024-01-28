from .container import Container
from typing import Dict, Union
from pymonntorch import Network, NeuronGroup, SynapseGroup, Behavior, NetworkObject
from .port import Port
import torch


class Layer(Container):
    def __init__(
        self,
        net: Network,
        neurongroups: list[NeuronGroup] = None,
        synapsegroups: list[SynapseGroup] = None,
        input_ports: Dict[str, list[Port]] = None,
        output_ports: Dict[str, list[Port]] = None,
        behavior: Dict[int, Behavior] = None,
        tag: str = None,
        device: Union[torch.device, int, str] = None,
    ):
        self.neurongroups = neurongroups
        self.synapsegroups = synapsegroups
        super().__init__(
            net=net,
            sub_structures=[*neurongroups, *synapsegroups],
            input_ports=input_ports,
            output_ports=output_ports,
            behavior=behavior,
            tag=tag,
            device=device,
        )

    def __repr__(self) -> str:
        result = (
            self.__class__.__name__
            + str(self.tags)
            + f"(SubStructures({len(self.sub_structures)}):"
            + "NeuronGroups"
            + str([value.tags[0] for value in self.neurongroups])
            + "SynapseGroups"
            + str([value.tags[0] for value in self.synapsegroups])
            + "){"
        )
        for k in sorted(list(self.behavior.keys())):
            result += str(k) + ":" + str(self.behavior[k])
        return result + "}"

    def save_helper(self, all_structures: list[NetworkObject]) -> dict:
        """A function to help saving the structures. into a dictionary.

        If tag and behavior parameters are not provided, they be handled by higher saving paradigm.
        Network should be excluded.

        Args:
            all_structures (list): a list containing the output of required_helper which are structures required to make same instance.
        """
        result_parameters = {
            "input_ports": Container.ports_helper(self.input_ports, all_structures),
            "output_ports": Container.ports_helper(self.output_ports, all_structures),
            "neurongroups": [
                all_structures.index(struc) for struc in self.neurongroups
            ],
            "synapsegroups": [
                all_structures.index(struc) for struc in self.synapsegroups
            ],
        }
        return result_parameters

    @staticmethod
    def build_helper(
        parameter_dic: dict, built_structures: dict[int, NetworkObject]
    ) -> dict:
        """Function to edit the parameter dictionary into acceptable argument of the class constructor.

        Note: behavior can also be edited in this function as thay later will be constructed.
        """
        parameter_dic["neurongroups"] = [
            built_structures[idx] for idx in parameter_dic["neurongroups"]
        ]
        parameter_dic["synapsegroups"] = [
            built_structures[idx] for idx in parameter_dic["synapsegroups"]
        ]
        parameter_dic["input_ports"] = Container.ports_helper(
            parameter_dic["input_ports"], built_structures
        )
        parameter_dic["output_ports"] = Container.ports_helper(
            parameter_dic["output_ports"], built_structures
        )
        return parameter_dic


class CorticalLayer(Layer):
    def __init__(
        self,
        net: Network,
        excitatory_neurongroup: NeuronGroup = None,
        inhibitory_neurongroup: NeuronGroup = None,
        synapsegroups: list[SynapseGroup] = None,
        input_ports: Dict[str, list[Port]] = None,
        output_ports: Dict[str, list[Port]] = None,
        behavior: Dict[int, Behavior] = None,
        tag: str = None,
        device: Union[torch.device, int, str] = None,
    ):
        self.exc_pop = excitatory_neurongroup
        self.inh_pop = inhibitory_neurongroup
        super().__init__(
            net=net,
            neurongroups=[self.exc_pop, self.inh_pop],
            synapsegroups=synapsegroups,
            input_ports=input_ports,
            output_ports=output_ports,
            behavior=behavior,
            tag=tag,
            device=device,
        )

    def save_helper(self, all_structures: list[NetworkObject]) -> dict:
        """A function to help saving the structures. into a dictionary.

        If tag and behavior parameters are not provided, they be handled by higher saving paradigm.
        Network should be excluded.

        Args:
            all_structures (list): a list containing the output of required_helper which are structures required to make same instance.
        """
        result_parameters = {
            "input_ports": Container.ports_helper(self.input_ports, all_structures),
            "output_ports": Container.ports_helper(self.output_ports, all_structures),
            "excitatory_neurongroup": all_structures.index(self.exc_pop),
            "inhibitory_neurongroup": all_structures.index(self.inh_pop),
            "synapsegroups": [
                all_structures.index(struc) for struc in self.synapsegroups
            ],
        }
        return result_parameters

    @staticmethod
    def build_helper(
        parameter_dic: dict, built_structures: dict[int, NetworkObject]
    ) -> dict:
        """Function to edit the parameter dictionary into acceptable argument of the class constructor.

        Note: behavior can also be edited in this function as thay later will be constructed.
        """
        parameter_dic["excitatory_neurongroup"] = built_structures[
            parameter_dic["excitatory_neurongroup"]
        ]
        parameter_dic["inhibitory_neurongroup"] = built_structures[
            parameter_dic["inhibitory_neurongroup"]
        ]
        parameter_dic["synapsegroups"] = [
            built_structures[idx] for idx in parameter_dic["synapsegroups"]
        ]
        parameter_dic["input_ports"] = Container.ports_helper(
            parameter_dic["input_ports"], built_structures
        )
        parameter_dic["output_ports"] = Container.ports_helper(
            parameter_dic["output_ports"], built_structures
        )
        return parameter_dic
