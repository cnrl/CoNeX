"""Utilities for CoNeX neural networks."""

from .replication import *
from .precision import (
    PrecisionMode,
    PrecisionConfig,
    MixedPrecisionManager,
    PrecisionContext,
    convert_to_precision,
)

__all__ = [
    # Replication
    "ELEMENTAL_STRUCTURE",
    "object_hook",
    "ExtraCallableJSONEncoder",
    "get_all_required_structures",
    "save_ports",
    "build_ports",
    "save_structure",
    "behaviors_to_list",
    "build_behavior_dict",
    "create_structure_from_dict",
    "replicate",
    "save_structure_dict_to_json",
    "load_structure_dict_from_json",
    # Precision
    "PrecisionMode",
    "PrecisionConfig",
    "MixedPrecisionManager",
    "PrecisionContext",
    "convert_to_precision",
]
