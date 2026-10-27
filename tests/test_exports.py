"""Tests for the public API surface and behavior priority registration."""

import pkgutil
import importlib
import inspect

import pytest
from pymonntorch import Behavior

import conex
from conex.nn.priority import ALL_PRIORITIES, prioritize_behaviors


# Names that were importable from the top-level package before __all__ existed.
LEGACY_API = [
    "LIF",
    "ELIF",
    "AELIF",
    "Fire",
    "KWTA",
    "InherentNoise",
    "NeuronAxon",
    "SimpleDendriteStructure",
    "SimpleDendriteComputation",
    "SimpleSTDP",
    "Conv2dSTDP",
    "Local2dSTDP",
    "WeightInitializer",
    "SynapseInit",
    "PreTrace",
    "PostTrace",
    "SimpleDendriticInput",
    "CorticalLayerConnection",
    "replicate",
    "save_structure",
    "create_structure_from_dict",
    "save_structure_dict_to_json",
    "load_structure_dict_from_json",
]

NEW_API = [
    "SDR",
    "SDREncoder",
    "GridCellModule",
    "TemporalMemory",
    "SpatialPooler",
    "ColumnVoting",
    "ConsensusNetwork",
    "SDROverlap",
    "PredictionUnit",
    "ErrorUnit",
    "PredictiveHierarchy",
    "MixedPrecisionManager",
]


def all_behavior_classes():
    """Every Behavior subclass defined under the conex package."""
    found = {}
    for mod in pkgutil.walk_packages(conex.__path__, prefix="conex."):
        module = importlib.import_module(mod.name)
        for name, obj in vars(module).items():
            if not inspect.isclass(obj) or not issubclass(obj, Behavior):
                continue
            if obj is Behavior or obj.__module__ != mod.name:
                continue
            found[name] = obj
    return found


@pytest.mark.parametrize("name", LEGACY_API)
def test_legacy_names_still_exported(name):
    assert hasattr(conex, name)
    assert name in conex.__all__


@pytest.mark.parametrize("name", NEW_API)
def test_new_names_exported(name):
    assert hasattr(conex, name)
    assert name in conex.__all__


def test_all_entries_resolve():
    missing = [n for n in conex.__all__ if not hasattr(conex, n)]
    assert missing == []


def test_all_has_no_duplicates():
    assert len(conex.__all__) == len(set(conex.__all__))


def test_star_import_matches_all():
    ns = {}
    exec("from conex import *", ns)
    ns.pop("__builtins__", None)
    assert set(ns) == set(conex.__all__)


def test_every_behavior_has_a_priority():
    unregistered = sorted(n for n in all_behavior_classes() if n not in ALL_PRIORITIES)
    assert unregistered == []


def test_every_behavior_is_exported():
    unexported = sorted(n for n in all_behavior_classes() if n not in conex.__all__)
    assert unexported == []


def test_prioritize_behaviors_maps_to_registered_priority():
    result = prioritize_behaviors(
        [conex.NeuronAxon(), conex.SimpleDendriteStructure(), conex.Fire()]
    )
    assert {k: v.__class__.__name__ for k, v in result.items()} == {
        380: "NeuronAxon",
        220: "SimpleDendriteStructure",
        340: "Fire",
    }


def test_prioritize_behaviors_rejects_unknown():
    class NotRegistered(Behavior):
        pass

    with pytest.raises(KeyError, match="NotRegistered"):
        prioritize_behaviors([NotRegistered()])
