from pymonntorch import Network, Behavior
from conex.nn.priority import NETWORK_PRIORITIES
from conex.behaviors.network.time_resolution import TimeResolution
from typing import Union, Dict
import torch


class Neocortex(Network):
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
        behavior = {
            **behavior,
            NETWORK_PRIORITIES["TimeResolution"]: TimeResolution(dt=dt),
        }
        tag = tag if tag is not None else "Neocortex"
        super().__init__(
            tag=tag,
            behavior=behavior,
            dtype=dtype,
            device=device,
            synapse_mode=synapse_mode,
            index=index,
        )
