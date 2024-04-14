from typing import Dict, NamedTuple
from pymonntorch import NetworkObject, Behavior


class Port(NamedTuple):
    """Container's Port dataclass

    Args:
        object (NetowrkObject): Inner container Object that port should connect to.
        label (str or None): Port port name of the inner object or None to connect to the object itself.
        behavior (Behavior): Behaviors that connections will attain while using the port.
    """

    object: NetworkObject
    label: str or None = None
    behavior: Dict[int, Behavior] = None
