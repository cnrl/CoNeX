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

    def __init__(self, *args, mode="rand", scale=1, offset=0, **kwargs):
        super().__init__(*args, mode=mode, scale=scale, offset=offset, **kwargs)

    def initialize(self, neurons):
        self.mode = self.parameter("mode", "rand")
        self.scale = self.parameter("scale", 1)
        self.offset = self.parameter("offset", 0)

    def forward(self, neurons):
        neurons.v += neurons.vector(mode=self.mode, scale=self.scale) + self.offset




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

    def __init__(self, k, *args, dimension=None, **kwargs):
        super().__init__(*args, k=k, dimension=dimension, **kwargs)

    def initialize(self, neurons):
        self.k = self.parameter("k", None, required=True)
        self.dimension = self.parameter("dimension", None)
        self.shape = (neurons.size, 1, 1)
        if hasattr(neurons, "depth"):
            self.shape = (neurons.depth, neurons.height, neurons.width)

    def forward(self, neurons):
        will_spike = neurons.v >= neurons.threshold
        v_values = neurons.v

        dim = 0
        if self.dimension is not None:
            v_values = v_values.view(self.shape)
            will_spike = will_spike.view(self.shape)
            dim = self.dimension

        if (will_spike.sum(axis=dim) <= self.k).all():
            return

        _, k_winners_indices = torch.topk(
            v_values, self.k, dim=dim, sorted=False
        )

        ignored = will_spike
        ignored.scatter_(dim, k_winners_indices, False)

        neurons.v[ignored.view((-1,))] = neurons.v_reset
