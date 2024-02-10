"""
Implementation of Base Container.
"""

from pymonntorch import NetworkObject, Network, Behavior
from .port import Port
from typing import Dict, Union, List, Tuple
import torch


class Container(NetworkObject):
    """The container Structute.

    Note: All the children should override required_helper, save_helper and build_helper.

    Args:
        net (Network): The network of the layer.
        sub_structures (list of NetworkObjects): The list of NetworkObject to be part of the container.
        input_ports (dictionary): a dictionary of lables into the list of ports.
        output_ports (dictionary): a dictionary of lables into the list of ports.
        behavior (dictionary): A dictionary of keys and behaviors attached to the container.
        tag (str): tag of the container divided by ",".
        device (device): device of the structure. defaults to the netowrk device.
    """

    def __init__(
        self,
        net: Network,
        sub_structures: List[NetworkObject],
        input_ports: Dict[str, Tuple[dict, List[Port]]] = None,
        output_ports: Dict[str, Tuple[dict, List[Port]]] = None,
        behavior: Dict[int, Behavior] = None,
        tag: str = None,
        device: Union[torch.device, int, str] = None,
    ):
        super().__init__(network=net, tag=tag, behavior=behavior, device=device)
        self.add_sub_structures(sub_structures)
        self.input_ports = input_ports if input_ports is not None else {}
        self.output_ports = output_ports if output_ports is not None else {}

    def add_sub_structures(self, structure_list: List[NetworkObject]):
        for struc in structure_list:
            self.add_sub_structure(struc)

    def __repr__(self) -> str:
        result = (
            self.__class__.__name__
            + str(self.tags)
            + f"(SubStructures({len(self.sub_structures)}):"
            + str([value.tags[0] for value in self.sub_structures])
            + "){"
        )
        for k in sorted(list(self.behavior.keys())):
            result += str(k) + ":" + str(self.behavior[k])
        return result + "}"

    def required_helper(self) -> List[NetworkObject]:
        """A function to find required structures.

        This function should return a list of structures required in time of creating the instance.
        """
        return self.sub_structures

    def save_helper(self, all_structures: List[NetworkObject]) -> dict:
        """A function to help saving the structures. into a dictionary.

        If tag and behavior parameters are not provided, they be handled by higher saving paradigm.
        Network should be excluded.

        Args:
            all_structures (list): a list containing the output of required_helper which are structures required to make same instance.
        """
        result_parameters = {
            "input_ports": Container.ports_helper(self.input_ports, all_structures),
            "output_ports": Container.ports_helper(self.output_ports, all_structures),
            "sub_structures": [
                all_structures.index(struc) for struc in self.sub_structures
            ],
        }
        return result_parameters

    @staticmethod
    def ports_helper(
        ports: Dict[str, List[Port]], helper_struc: Union[list, dict]
    ) -> dict:
        """Transforms the port dictionary to use objects or indexes.

        Args:
            ports (dictionary): The dictionary of desired ports to transform.
            helper_struc (list or dictionary): dictionary of saved structures or a list of structures.
        """
        if isinstance(helper_struc, dict):
            result = {
                k: (v[0], [(helper_struc[x[0]], x[1], x[2]) for x in v[1]])
                for k, v in ports.items()
            }
        else:
            result = {
                k: (
                    v[0],
                    [(helper_struc.index(x[0]), x[1], x[2]) for x in v[1]],
                )
                for k, v in ports.items()
            }
        return result

    @staticmethod
    def build_helper(
        parameter_dic: dict, built_structures: Dict[int, NetworkObject]
    ) -> dict:
        """Function to edit the parameter dictionary into acceptable argument of the class constructor.

        Note: behavior can also be edited in this function as thay later will be constructed.
        """
        parameter_dic["sub_structures"] = [
            built_structures[idx] for idx in parameter_dic["sub_structures"]
        ]
        parameter_dic["input_ports"] = Container.ports_helper(
            parameter_dic["input_ports"], built_structures
        )
        parameter_dic["output_ports"] = Container.ports_helper(
            parameter_dic["output_ports"], built_structures
        )
        return parameter_dic
