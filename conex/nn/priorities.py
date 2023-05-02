"""
This file specifies the priorities for different behaviors.
"""
NETWORK_PRIORITIES = {
    "TimeResolution": 1,
    "Payoff": 100,
    "NeuroModulator": 120,
}

NEURON_PRIORITIES = {
    "NeuronDendrite": 200,
    "NeuronDynamic": 220,
    "DirectNoise": 240,
    "KWTA": 260,
    "Fire": 300,
    "Trace": 320,
    "NeuronAxon": 340,
}

LAYER_PRIORITIES = {"InputDataset": 280}

SYNAPSE_PRIORITIES = {
    "Init": 2,
    "WeightInit": 3,
    "SrcDelayInit": 4,
    "DstDelayInit": 5,
    "DendriticInput": 180,
    "LearningRule": 360,
    "WeightNormalization": 380,
    "WeightClip": 400,
}
