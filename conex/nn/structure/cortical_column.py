from .container import Container
from pymonntorch import Network, Behavior, NetworkObject
from .layer import CorticalLayer
from typing import Union, Dict
from .cortical_layer_connection import CorticalLayerConnection
from .port import Port
import torch


class CorticalColumn(Container):
    def __init__(
        self,
        net: Network,
        layers: Dict[str, CorticalLayer] = None,
        layer_connections: list[tuple[str, str, CorticalLayerConnection]] = None,
        input_ports: Dict[str, list[Port]] = None,
        output_ports: Dict[str, list[Port]] = None,
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
        result = (
            self.__class__.__name__
            + str(self.tags)
            + f"(SubStructures({len(self.sub_structures)}):"
            + f"CorticalLayers({sum([isinstance(x, CorticalLayer) for x in self.sub_structures])}):"
            + str(
                [x.tags[0] for x in self.sub_structures if isinstance(x, CorticalLayer)]
            )
            + f"CorticalLayerConnections({sum([isinstance(x, CorticalLayerConnection) for x in self.sub_structures])}):"
            + str(
                [
                    x.tags[0]
                    for x in self.sub_structures
                    if isinstance(x, CorticalLayerConnection)
                ]
            )
            + "){"
        )
        for k in sorted(list(self.behavior.keys())):
            result += str(k) + ":" + str(self.behavior[k])
        return result + "}"

    def required_helper(self) -> list[NetworkObject]:
        """A function to find required structures.

        This function should return a list of structures required in time of creating the instance.
        """
        return self.sub_structures

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
            "layers": [
                all_structures.index(struc) for struc in self.sub_structures if isinstance(struc, CorticalLayer)
            ],
            "layers_connections": [
                (x[0],x[1],all_structures.index(x[2])) for x in self.layer_connections
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
        parameter_dic["layer"] = [
            built_structures[idx] for idx in parameter_dic["sub_structures"]
        ]
        parameter_dic["layer_connections"] = [
            built_structures[idx] for idx in parameter_dic["sub_structures"]
        ]
        parameter_dic["input_ports"] = Container.ports_helper(
            parameter_dic["input_ports"], built_structures
        )
        parameter_dic["output_ports"] = Container.ports_helper(
            parameter_dic["output_ports"], built_structures
        )
        return parameter_dic
