"""
This file specifies the priorities for different behaviors.
"""
NETWORK_PRIORITIES = {
    "TimeResolution": 1,
    "Payoff": 2,
    "NeuroModulator": 100,
}

NEURON_PRIORITIES = {
    "NeuronDendrite": 260,
    "NeuronDynamic": 270,
    "DirectNoise": 280,
    "KWTA": 290,
    "Fire": 300,
    "Trace": 320,
    "NeuronAxon": 330,
}

LAYER_PRIORITIES = {"InputDataset": 350}

SYNAPSE_PRIORITIES = {
    "Init": 2,
    "WeightInit": 3,
    "SrcDelayInit": 4,
    "DstDelayInit": 5,
    "DendriticInput": 200,
    "LearningRule": 360,
    "WeightNormalization": 370,
    "WeightClip": 380,
}
