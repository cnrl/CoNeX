"""
Structure of spiking neural populations.
"""

import warnings
from CorticalColumn.behaviours.neurons.lif_neurons import LIF
from CorticalColumn.behaviours.neurons.specs import (
    KWTA,
    Fire,
    InherentNoise,
    NeuronAxon,
    NeuronDendrite,
    SpikeTrace,
)
from pymonntorch import NeuronGroup

import torch


class SpikingNeuronGroup(NeuronGroup):
    def __init__(
        self,
        size,
        net,
        neuron_type=LIF,
        kwta=1,
        kwta_dim=None,
        tau_trace=None,
        max_delay=None,
        noise_function=None,
        tag=None,
        color=None,
        dendrite_params={},
        neuron_params={},
    ):
        if tag is None and net is not None:
            tag = "SpikingNeuronGroup_" + str(len(net.NeuronGroups) + 1)

        behavior = {
            31: NeuronDendrite(**dendrite_params),
            32: neuron_type(**neuron_params),
            34: KWTA(k=kwta, dimension=kwta_dim),
            35: Fire(),
            38: NeuronAxon(max_delay=max_delay),
        }

        if tau_trace:
            behavior[37] = SpikeTrace(tau_s=tau_trace)

        if noise_function:
            behavior[33] = InherentNoise(noise_function)

        super().__init__(size, behavior, net, tag, color)

        if not hasattr(self, "v"):
            if hasattr(self, "v_rest"):
                warnings.warn(
                    "Spiking neuron behavior lacks attribute v. Adding the attribute..."
                )
                self.v = self.v_rest * self.get_neuron_vec(mode="ones()")
            else:
                raise AttributeError("Spiking neuron lacks attribute v.")

        if not hasattr(self, "spikes"):
            warnings.warn(
                "Spiking neuron behavior lacks attribute spike. Adding the attribute..."
            )
            if hasattr(self, "init_s"):
                if isinstance(self.init_s, torch.Tensor):
                    self.spikes = self.init_s
            self.spikes = self.get_neuron_vec(mode="zeros()")
