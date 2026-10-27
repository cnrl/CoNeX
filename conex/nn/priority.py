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
    "SDROverlap": 130,
    "ColumnVoting": 140,
    "ConsensusNetwork": 160,
}

NEURON_PRIORITIES = {
    "SDREncoder": 200,
    "SpatialPooler": 205,
    "TemporalMemory": 210,
    "GridCellModule": 215,
    "DisplacementCellModule": 216,
    "LocationEncoder": 217,
    "SimpleDendriteStructure": 220,
    "DendriticSegment": 225,
    "TopDownPrediction": 230,
    "PredictionUnit": 232,
    "ErrorUnit": 234,
    "PrecisionWeighting": 236,
    "FreeEnergyMinimization": 238,
    "HierarchicalPredictiveCoding": 239,
    "SimpleDendriteComputation": 240,
    "ActiveDendriteComputation": 240,
    "ContextualPrediction": 250,
    "NeuronDynamic": 260,
    "LIF": 260,
    "ELIF": 260,
    "AELIF": 260,
    "InherentNoise": 280,
    "KWTA": 300,
    "VoltageBaseHomeostasis": 301,
    "Fire": 340,
    "LocationSetter": 340,
    "SensorySetter": 340,
    "ActivityBaseHomeostasis": 341,
    "PredictiveCodingLearning": 350,
    "SDRClassifier": 360,
    "NeuronAxon": 380,
}

LAYER_PRIORITIES = {"SpikeNdDataset": 320}

SYNAPSE_PRIORITIES = {
    "SynapseInit": 2,
    "WeightInitializer": 3,
    "DelayInitializer": 4,
    "PredictiveSynapseInit": 5,
    "AveragePool2D": 180,
    "BaseDendriticInput": 180,
    "Conv2dDendriticInput": 180,
    "FeedbackPredictionSynapse": 180,
    "FeedforwardErrorSynapse": 180,
    "LateralDendriticInput": 180,
    "LateralPredictionSynapse": 180,
    "Local2dDendriticInput": 180,
    "One2OneDendriticInput": 180,
    "SimpleDendriticInput": 180,
    "SparseDendriticInput": 180,
    "CurrentNormalization": 200,
    "PreSpikeCatcher": 420,
    "PostSpikeCatcher": 440,
    "PreTrace": 460,
    "PostTrace": 480,
    "BaseLearning": 500,
    "Conv2dRSTDP": 500,
    "Conv2dSTDP": 500,
    "Local2dRSTDP": 500,
    "Local2dSTDP": 500,
    "One2OneRSTDP": 500,
    "One2OneSTDP": 500,
    "One2OneiSTDP": 500,
    "PredictiveSynapseLearning": 500,
    "SimpleRSTDP": 500,
    "SimpleSTDP": 500,
    "SimpleiSTDP": 500,
    "SparseRSTDP": 500,
    "SparseSTDP": 500,
    "SparseiSTDP": 500,
    "LearningRule": 500,
    "WeightNormalization": 520,
    "WeightClip": 540,
}

ALL_PRIORITIES = {
    **NETWORK_PRIORITIES,
    **NEURON_PRIORITIES,
    **LAYER_PRIORITIES,
    **SYNAPSE_PRIORITIES,
}

__all__ = [
    "NETWORK_PRIORITIES",
    "NEURON_PRIORITIES",
    "LAYER_PRIORITIES",
    "SYNAPSE_PRIORITIES",
    "ALL_PRIORITIES",
    "prioritize_behaviors",
]


def prioritize_behaviors(behavior: List[Behavior]) -> Dict[int, Behavior]:
    result = {}
    for b in behavior:
        name = b.__class__.__name__
        if name not in ALL_PRIORITIES:
            raise KeyError(f"{name} has no priority in conex.nn.priority")
        result[ALL_PRIORITIES[name]] = b
    return result
