"""
This file specifies the priorities for different behaviors.
"""
NETWORK_PRIORITIES = {
    "TimeResolution": 1,
    "Payoff": 101,
    "NeuroModulator": 150,
}

NEURON_PRIORITIES = {
    "NeuronDendrite": 310,
    "NeuronDynamic": 320,
    "DirectNoise": 330,
    "KWTA": 340,
    "Fire": 360,
    "Trace": 380,
    "NeuronAxon": 390,
}

LAYER_PRIORITIES = {"InputDataset": 350}

SYNAPSE_PRIORITIES = {
    "Init": 101,
    "WeightInit": 102,
    "SrcDelayInit": 103,
    "DstDelayInit": 104,
    "DendriticInput": 250,
    "LearningRule": 420,
    "WeightNormalization": 430,
    "WeightClip": 440,
}
