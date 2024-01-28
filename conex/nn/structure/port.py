from typing import Dict
from dataclasses import dataclass
from pymonntorch import NetworkObject, Behavior

@dataclass
class Port:
    object: NetworkObject
    label: str or None
    behavior: Dict[int, Behavior]
