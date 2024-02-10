from pymonntorch import Network, Behavior
from conex.nn.priority import NETWORK_PRIORITIES
from conex.behaviors.network.time_resolution import TimeResolution
from typing import Union, Dict
import torch


class Neocortex(Network):
    """Neocortex.

    Neocortex structure is the network equivalent. it also contains the all the structures..

    Args:
        dt (float): The time resolution of the simulation.
        behavior (dictionary): a dictionary of keys and behaviors attached to the Layer.
        dtype (torch dtype): The precision of floating point compuation.
        device (device): device of the Layer. defaults to the netowrk device.
        synapse_mode (str): The synapse structure in simulation, possible values: "SxD", "DxS".
        index (bool): Add the index vector to neuron populations.
        tag (str): tag of the Layer divided by ",".
    """

    def __init__(
        self,
        dt: Union[float, None] = 1,
        behavior: Dict[int, Behavior] = None,
        dtype: torch.dtype = torch.float32,
        device: Union[torch.device, int, str] = "cpu",
        synapse_mode: str = "SxD",
        index: bool = False,
        tag: str = None,
    ):
        behavior = behavior if behavior is not None else {}
        behavior = {
            **behavior,
            NETWORK_PRIORITIES["TimeResolution"]: TimeResolution(dt=dt),
        }
        tag = tag if tag is not None else "Neocortex"
        self.input_layers = []
        self.output_layers = []
        self.cortical_columns = []
        super().__init__(
            tag=tag,
            behavior=behavior,
            dtype=dtype,
            device=device,
            synapse_mode=synapse_mode,
            index=index,
        )
