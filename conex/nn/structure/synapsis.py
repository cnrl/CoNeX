from .container import Container
from typing import Union, Dict, List
from pymonntorch import Network, Behavior, NetworkObject, SynapseGroup, NeuronGroup
import torch
import copy
from conex.nn.utils.replication import behaviors_to_list, build_behavior_dict


class Synapsis(Container):
    """Synapsis structure

    This structure connects two ports, and connects all the possible synapsegroups between them.

    Args:
        net (Network): The network of the layer.
        src (NetworkObject): The source of synpsis object.
        dst (NetworkObject): The destination of synpsis object.
        input_port (str): The input label of source port.
        output_port (str): The output label of destination port.
        synapsis_behavior (dict): the behavior of synapses created.
        synapstic_tag (str): The tag of the synapses created.
        behavior (dictionary): A dictionary of keys and behaviors attached to the container.
        tag (str): tag of the container divided by ",".
        device (device): device of the structure. defaults to the netowrk device.
    """

    def __init__(
        self,
        net: Network,
        src: NetworkObject = None,
        dst: NetworkObject = None,
        input_port: str = None,
        output_port: str = None,
        synapsis_behavior: Dict[int, Behavior] = None,
        synaptic_tag: str = None,
        behavior: Dict[int, Behavior] = None,
        tag: str = None,
        device: Union[torch.device, int, str] = None,
    ):
        self.input_port = input_port
        self.output_port = output_port
        self.synapsis_behavior = synapsis_behavior
        self.synaptic_tag = synaptic_tag
        self.src = src
        self.dst = dst

        super().__init__(
            net=net, sub_structures=[], behavior=behavior, tag=tag, device=device
        )

        self.synapses = self.sub_structures
        if self.src is not None and self.dst is not None:
            self.create_synapses(self)

    def connect_src(self, src: NetworkObject):
        if self.src is None and src is not None:
            self.src = src
        if self.dst is not None:
            self.create_synapses()

    def connect_dst(self, dst: NetworkObject):
        if self.dst is None and dst is not None:
            self.dst = dst
        if self.src is not None:
            self.create_synapses()

    @staticmethod
    def _port2ng(
        label: Union[str, None], object: NetworkObject, src_port: bool = True
    ) -> List[List[NeuronGroup, Dict[int, Behavior]]]:
        if isinstance(object, NeuronGroup):
            return [[object, {}]]
        elif isinstance(object, Container):
            result = []
            search_port = object.output_ports if src_port else object.input_ports
            for port in search_port[label]:
                port_result = Synapsis._port2ng(port.label, port.object)
                port_result = [
                    [x[0], x[1].update(copy.deepcopy(port.behavior))]
                    for x in port_result
                ]
                result.extend(port_result)
            return result
        else:
            return []

    def create_synapses(self):
        if not self.synapses:
            src_ng = Synapsis._port2ng(self.input_port, self.src)
            dst_ng = Synapsis._port2ng(self.output_port, self.dst)
            for x in src_ng:
                for y in dst_ng:
                    self.synapses.append(
                        SynapseGroup(
                            net=self.net,
                            src=x[0],
                            dst=y[0],
                            behavior={**x[1], **y[1], **self.synapsis_behavior},
                            tag=self.synaptic_tag,
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
            "input_port": self.input_port,
            "output_port": self.output_port,
            "src": all_structures.index(self.src)
            if self.src in all_structures
            else None,
            "dst": all_structures.index(self.dst)
            if self.dst in all_structures
            else None,
            "synapsis_behavior": behaviors_to_list(self.synapsis_behavior),
            "synaptic_tag": self.synaptic_tag,
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
        parameter_dic["synapsis_behavior"] = build_behavior_dict(
            built_structures["synapsis_behavior"]
        )
        return parameter_dic
