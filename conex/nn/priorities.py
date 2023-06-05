"""
This file specifies the priorities for different behaviors.
"""
NETWORK_PRIORITIES = {
    "TimeResolution": 1,
    "Payoff": 100,
    "NeuroModulator": 120,
}

NEURON_PRIORITIES = {
    "NeuronDendrite": 220,
    "NeuronDynamic": 240,
    "DirectNoise": 260,
    "KWTA": 280,
    "Fire": 320,
    "Trace": 340,
    "NeuronAxon": 360,
}

LAYER_PRIORITIES = {"InputDataset": 300}

SYNAPSE_PRIORITIES = {
    "Init": 2,
    "WeightInit": 3,
    "SrcDelayInit": 4,
    "DstDelayInit": 5,
    "DendriticInput": 180,
    "CurrentNormalization": 200,
    "LearningRule": 380,
    "WeightNormalization": 400,
    "WeightClip": 420,
}
