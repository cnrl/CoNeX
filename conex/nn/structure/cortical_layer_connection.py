from .container import Container
from typing import Dict, Union, List
from pymonntorch import Network, Behavior, SynapseGroup, NetworkObject
import torch
import copy
from conex.nn.utils.replication import behaviors_to_list, build_behavior_dict


class CorticalLayerConnection(Container):
    """Connection scheme between two layers.

    Args:
        src (NetworkObject): The source network object such as a layer with neurongroups as attribute.
        dst (NetworkObject): The destination network object such as a layer with neurongroups as attribute.
        connections: (list): The list of connection created between two objects. Each connection is defined with quaduple of source neurongroup's attribute name, destinations' attribute name, behavior dictionary, and synapse tag.
        behavior (dict): The behavior for the CorticalLayerConnection itself.
        tag (str): tag of the CorticalLayerConnection divided by ",".
        device (device): device of the structure. Defaults to the netowrk device.
    """

    def __init__(
        self,
        net: Network,
        src: NetworkObject = None,
        dst: NetworkObject = None,
        connections: List[List[str, str, Dict[int, Behavior], str]] = None,
        behavior: Dict[int, Behavior] = None,
        tag: str = None,
        device: Union[torch.device, int, str] = None,
    ):
        self.src = src
        self.dst = dst
        self.connections = connections

        super().__init__(
            net=net, sub_structures=[], behavior=behavior, tag=tag, device=device
        )

        self.synapses = self.sub_structures
        if self.src is not None and self.dst is not None:
            self.create_synapses(self)

    def connect_src(self, src: NetworkObject):
        if self.src is None and src is not None:
            self.src = src
        if self.src is not None and self.dst is not None:
            self.create_synapses()

    def connect_dst(self, dst: NetworkObject):
        if self.dst is None and dst is not None:
            self.dst = dst
        if self.dst is not None and self.dst is not None:
            self.create_synapses()

    def create_synapses(self):
        if not self.synapses:
            for connection in self.connections:
                src_str, dst_str, beh, tag = connection
                src = getattr(self.src, src_str)
                dst = getattr(self.dst, dst_str)
                self.synapses.append(
                    SynapseGroup(
                        net=self.net,
                        src=src,
                        dst=dst,
                        behavior=copy.deepcopy(beh),
                        tag=tag,
                    )
                )

    def __repr__(self) -> str:
        result = (
            self.__class__.__name__
            + str(self.tags)
            + "(Synapses:"
            + str([value.tags[0] for value in self.synapses])
            + "){"
        )
        for k in sorted(list(self.behavior.keys())):
            result += str(k) + ":" + str(self.behavior[k])
        return result + "}"

    def required_helper(self) -> List[NetworkObject]:
        """A function to find required structures.

        This function should return a list of structures required in time of creating the instance.
        """
        return []

    def save_helper(self, all_structures: List[NetworkObject]) -> dict:
        """A function to help saving the structures. into a dictionary.

        If tag and behavior parameters are not provided, they be handled by higher saving paradigm.
        Network should be excluded.

        Args:
            all_structures (list): a list containing the output of required_helper which are structures required to make same instance.
        """
        result_parameters = {
            "src": all_structures.index(self.src)
            if self.src in all_structures
            else None,
            "dst": all_structures.index(self.dst)
            if self.dst in all_structures
            else None,
            "connections": [
                [x[0], x[1], behaviors_to_list(x[2]), x[3]] for x in self.connections
            ],
        }
        return result_parameters

    @staticmethod
    def build_helper(
        parameter_dic: dict, built_structures: Dict[int, NetworkObject]
    ) -> dict:
        """Function to edit the parameter dictionary into acceptable argument of the class constructor.

        Note: behavior can also be edited in this function as thay later will be constructed.
        """
        parameter_dic["src"] = (
            built_structures[parameter_dic["src"]]
            if parameter_dic["src"] is not None
            else None
        )
        parameter_dic["dst"] = (
            built_structures[parameter_dic["dst"]]
            if parameter_dic["dst"] is not None
            else None
        )
        parameter_dic["synapsis_behavior"] = [
            [x[0], x[1], build_behavior_dict(x[2]), x[3]]
            for x in built_structures["synapsis_behavior"]
        ]
        return parameter_dic
