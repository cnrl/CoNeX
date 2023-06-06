"""
This file specifies the priorities for different behaviors.
"""
NETWORK_PRIORITIES = {
    "TimeResolution": 1,
    "Payoff": 100,
    "NeuroModulator": 120,
}

NEURON_PRIORITIES = {
    "DendriteStructure": 220,
    "DendriteComputation": 240,
    "NeuronDynamic": 260,
    "DirectNoise": 280,
    "KWTA": 300,
    "Fire": 340,
    "Trace": 360,
    "NeuronAxon": 380,
}

LAYER_PRIORITIES = {"InputDataset": 320}

SYNAPSE_PRIORITIES = {
    "Init": 2,
    "WeightInit": 3,
    "SrcDelayInit": 4,
    "DstDelayInit": 5,
    "DendriticInput": 180,
    "CurrentNormalization": 200,
    "LearningRule": 400,
    "WeightNormalization": 420,
    "WeightClip": 440,
}
