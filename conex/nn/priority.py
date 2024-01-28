from pymonntorch import Behavior
from typing import Dict

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
    "SpikeTrace": 360,
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
    "BaseLearning": 400,
    "Conv2dRSTDP": 400,
    "Conv2dSTDP": 400,
    "Local2dRSTDP": 400,
    "Local2dSTDP": 400,
    "One2OneRSTDP": 400,
    "One2OneSTDP": 400,
    "One2OneiSTDP": 400,
    "SimpleRSTDP": 400,
    "SimpleSTDP": 400,
    "SimpleiSTDP": 400,
    "SparseRSTDP": 400,
    "SparseSTDP": 400,
    "SparseiSTDP": 400,
    "LearningRule": 400,
    "WeightNormalization": 420,
    "WeightClip": 440,
}

ALL_PRIORITIES = (
    NETWORK_PRIORITIES | NEURON_PRIORITIES | LAYER_PRIORITIES | SYNAPSE_PRIORITIES
)


def prioritize_behaviors(behavior: list[Behavior]) -> Dict[int, Behavior]:
    result = {ALL_PRIORITIES[x.__class__.__name__]: x for x in behavior}
    return result
