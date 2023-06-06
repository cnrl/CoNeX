"""
General specifications needed for spiking neurons.
"""

from pymonntorch import Behavior
import torch


class InherentNoise(Behavior):
    """
    Applies noisy voltage to neurons in the population.

    Args:
        mode (str): Mode to be used in initialize the tensor. Accepts similar values to Pymonntorch's `tensor` function. Defaults to "rand".
        scale (float): Scale factor to multiply to the tensor. Default is 1.0.
        offset (function): An offset value to be added to the tensor. Default is 0.0.
    """

    def initialize(self, neurons):
        self.mode = self.parameter("mode", "rand")
        self.scale = self.parameter("scale", 1)
        self.offset = self.parameter("offset", 0)

    def forward(self, neurons):
        neurons.v += neurons.vector(mode=self.mode, scale=self.scale) + self.offset


class SpikeTrace(Behavior):
    """
    Calculates the spike trace.

    Note : should be placed after Fire behavior.

    Args:
        tau_s (float): decay term for spike trace. The default is None.
    """

    def initialize(self, neurons):
        """
        Sets the trace attribute for the neural population.
        """
        self.tau_s = self.parameter("tau_s", None, required=True)
        neurons.trace = neurons.vector(0.0)

    def forward(self, neurons):
        """
        Calculates the spike trace of each neuron by adding current spike and decaying the trace so far.
        """
        neurons.trace += neurons.spikes
        neurons.trace -= (neurons.trace / self.tau_s) * neurons.network.dt


class Fire(Behavior):
    """
    Asks neurons to Fire.
    """

    def forward(self, neurons):
        neurons.spiking_neuron.Fire(neurons)


class KWTA(Behavior):
    """
    KWTA behavior of spiking neurons:

    if v >= threshold then v = v_reset and all other spiked neurons are inhibited.

    Note: Population should be built by NeuronDimension.
    and firing behavior should be added too.

    Args:
        k (int): number of winners.
        dimension (int, optional): K-WTA on specific dimension. defaults to None.
    """

    def initialize(self, neurons):
        self.k = self.parameter("k", None, required=True)
        self.dimension = self.parameter("dimension", None)
        self.shape = (1, 1, neurons.size)
        if hasattr(neurons, "depth"):
            self.shape = (neurons.depth, neurons.height, neurons.width)

    def forward(self, neurons):
        will_spike = neurons.v >= neurons.threshold
        will_spike_v = will_spike * (neurons.v - neurons.threshold)

        dim = 0
        if self.dimension is not None:
            will_spike_v = will_spike_v.view(self.shape)
            will_spike = will_spike.view(self.shape)
            dim = self.dimension

        if (will_spike.sum(axis=dim) <= self.k).all():
            return

        k_values, k_winners_indices = torch.topk(
            will_spike_v, self.k, dim=dim, sorted=False
        )
        min_values = k_values.min(dim=0).values
        winners = will_spike_v >= min_values.expand(will_spike_v.size())
        ignored = will_spike * (~winners)

        neurons.v[ignored.view((-1,))] = neurons.v_reset
