from .container import Container
from pymonntorch import Network, Behavior, NetworkObject
from .layer import CorticalLayer
from typing import Union, Dict, List, Tuple
from .cortical_layer_connection import CorticalLayerConnection
from .port import Port
import torch


class CorticalColumn(Container):
    """The Implementation of CorticalColumn.

    Args:
        net (Network): The network of the Cortical Column.
        layers (dictionary): a dictionary with key as layers' name and value as corticallayers which will be cortical column's layers.
        layer_connections (list): a list of tuple with values as (source layer name, destination layer name, corticallayerconnection). to connect inner layers.
        input_ports (dictionary): a dictionary of lables into the list of ports.
        output_ports (dictionary): a dictionary of lables into the list of ports.
        behavior (dictionary): a dictionary of keys and behaviors attached to the container.
        tag (str): tag of the container divided by ",".
        device (device): device of the structure. defaults to the netowrk device.
    """

    def __init__(
        self,
        net: Network,
        layers: Dict[str, CorticalLayer] = None,
        layer_connections: List[Tuple[str, str, CorticalLayerConnection]] = None,
        input_ports: Dict[str, Tuple[dict, List[Port]]] = None,
        output_ports: Dict[str, Tuple[dict, List[Port]]] = None,
        behavior: Dict[int, Behavior] = None,
        tag: str = None,
        device: Union[torch.device, int, str] = None,
    ):
        self.layers = layers
        self.layer_connections = layer_connections
        self.create_layer_connections()
        super().__init__(
            net=net,
            sub_structures=list(self.layers.values())
            + [x[2] for x in layer_connections],
            input_ports=input_ports,
            output_ports=output_ports,
            behavior=behavior,
            tag=tag,
            device=device,
        )

    def create_layer_connections(self):
        for x in self.layer_connections:
            src_str, dst_str, clc = x
            if clc.src is None and clc.dst is None:
                clc.connect_src(self.layers[src_str])
                clc.connect_dst(self.layers[dst_str])

    def __repr__(self) -> str:
        tag_str = self.tags[0] if self.tags else "untagged"
        layers = [x for x in self.sub_structures if isinstance(x, CorticalLayer)]
        connections = [x for x in self.sub_structures if isinstance(x, CorticalLayerConnection)]
        layer_tags = [x.tags[0] if x.tags else "?" for x in layers]
        conn_tags = [x.tags[0] if x.tags else "?" for x in connections]
        result = f"{self.__class__.__name__}[{tag_str}](layers={layer_tags}, connections={conn_tags}){{"
        behaviors = [f"{k}:{self.behavior[k].__class__.__name__}" for k in sorted(self.behavior.keys())]
        result += ", ".join(behaviors)
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
            "layer_tags": list(self.layers.keys()),
            "layers": [
                all_structures.index(self.layers[layer_tag])
                for layer_tag in self.layers.keys()
            ],
            "layers_connections": [
                (x[0], x[1], all_structures.index(x[2])) for x in self.layer_connections
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
        parameter_dic["layers"] = {
            layer_tag: built_structures[idx]
            for layer_tag, idx in zip(
                parameter_dic.get("layer_tags", []),
                parameter_dic["layers"]
            )
        }
        parameter_dic["layer_connections"] = [
            (conn[0], conn[1], built_structures[conn[2]])
            for conn in parameter_dic["layers_connections"]
        ]
        parameter_dic["input_ports"] = Container.ports_helper(
            parameter_dic["input_ports"], built_structures
        )
        parameter_dic["output_ports"] = Container.ports_helper(
            parameter_dic["output_ports"], built_structures
        )
        return parameter_dic
