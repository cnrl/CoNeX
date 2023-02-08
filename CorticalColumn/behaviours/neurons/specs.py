"""
General specifications needed for spiking neurons.
"""

from pymonntorch import Behavior
import torch

# TODO inhibition of KWTA, how should it be???
# TODO adaptive neuorns will KWTA, What behaviour should I expect???

class Fire(Behavior):
    """
    Asks neurons to Fire.
    """
    def new_iteration(self, neurons):
        neurons.Fire(neurons)

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

    def set_variables(self, neurons):
        self.k = self.get_init_attr("k", None)
        self.dimension = self.get_init_attr('dimension', None)
        self.shape = (neurons.depht, neurons.height, neurons.width)

    def new_iteration(self, neurons):
        will_spike = (neurons.v >= neurons.threshold)

        will_spike_v = (will_spike * (neurons.v - neurons.threshold))

        if self.dimension:
            will_spike_v = will_spike_v.reshape(self.shape)
            will_spike = will_spike.reshape(self.shape)
        else:
            self.dimension = 0
        
        if (will_spike.sum(axis=self.dimension) <= self.k).all():
            return

        k_values, k_winners_indices = torch.topk(will_spike_v, self.k+1, dim=self.dimension, sorted=False)
        min_values = k_values.min(dim = 0).values
        winners = will_spike_v > min_values.expand(will_spike_v.size())
        ignored = will_spike * (~winners)

        neurons.v[ignored.reshape((-1,))] = neurons.v_reset