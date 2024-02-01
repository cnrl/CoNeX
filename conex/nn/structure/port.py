from typing import Dict
from dataclasses import dataclass
from pymonntorch import NetworkObject, Behavior


@dataclass
class Port:
    """Container's Port dataclass

    Args:
        object (NetowrkObject): Inner container Object that port should connect to.
        label (str or None): Port port name of the inner object or None to connect to the object itself.
        behavior (Behavior): Behaviors that connections will attain while using the port.
    """

    object: NetworkObject
    label: str or None
    behavior: Dict[int, Behavior]
