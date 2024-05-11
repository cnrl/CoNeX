from typing import Dict, List

from pymonntorch import Behavior

"""
This file specifies the priorities for different behaviors.
"""

# TODO add priority as class property

NETWORK_PRIORITIES = {
    "TimeResolution": 1,
    "Payoff": 100,
    "Dopamine": 120,
}

NEURON_PRIORITIES = {
    "SimpleDendriteStructure": 220,
    "SimpleDendriteComputation": 240,
    "NeuronDynamic": 260,
    "LIF": 260,
    "ELIF": 260,
    "AELIF": 260,
    "InherentNoise": 280,
    "KWTA": 300,
    "Fire": 340,
    "LocationSetter": 340,
    "SensorySetter": 340,
    "NeuronAxon": 380,
    "ActivityBaseHomeostasis": 341,
    "VoltageBaseHomeostasis": 301,
}

LAYER_PRIORITIES = {"SpikeNdDataset": 320}

SYNAPSE_PRIORITIES = {
    "SynapseInit": 2,
    "WeightInitializer": 3,
    "DelayInitializer": 4,
    "AveragePool2D": 180,
    "BaseDendriticInput": 180,
    "Conv2dDendriticInput": 180,
    "LateralDendriticInput": 180,
    "Local2dDendriticInput": 180,
    "One2OneDendriticInput": 180,
    "SimpleDendriticInput": 180,
    "SparseDendriticInput": 180,
    "CurrentNormalization": 200,
    "PreSpikeCatcher": 420,
    "PostSpikeCatcher": 440,
    "PreTrace": 460,
    "PostTrace": 480,
    "BaseLearning": 480,
    "Conv2dRSTDP": 480,
    "Conv2dSTDP": 480,
    "Local2dRSTDP": 480,
    "Local2dSTDP": 480,
    "One2OneRSTDP": 480,
    "One2OneSTDP": 480,
    "One2OneiSTDP": 480,
    "SimpleRSTDP": 480,
    "SimpleSTDP": 480,
    "SimpleiSTDP": 480,
    "SparseRSTDP": 480,
    "SparseSTDP": 480,
    "SparseiSTDP": 480,
    "LearningRule": 480,
    "WeightNormalization": 500,
    "WeightClip": 520,
}

ALL_PRIORITIES = {
    **NETWORK_PRIORITIES,
    **NEURON_PRIORITIES,
    **LAYER_PRIORITIES,
    **SYNAPSE_PRIORITIES,
}


def prioritize_behaviors(behavior: List[Behavior]) -> Dict[int, Behavior]:
    result = {ALL_PRIORITIES[x.__class__.__name__]: x for x in behavior}
    return result
