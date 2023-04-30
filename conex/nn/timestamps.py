"""
This file specifies the timestamp for different behaviors.
"""
NETWORK_TIMESTAMPS = {
    "TimeStep": 1,
    "Reward": 101,
    "NeuroModulator": 150,
}

NEURON_TIMESTAMPS = {
    "NeuronDendrite": 310,
    "NeuronDynamic": 320,
    "DirectNoise": 330,
    "KWTA": 340,
    "Fire": 360,
    "Trace": 380,
    "NeuronAxon": 390,
}

LAYER_TIMESTAMPS = {"InputDataset": 350}

SYNAPSE_TIMESTAMPS = {
    "Init": 101,
    "WeightInit": 102,
    "SrcDelayInit": 103,
    "DstDelayInit": 104,
    "DendriticInput": 250,
    "LearningRule": 420,
    "WeightNormalization": 430,
    "WeightClip": 4430,
}
