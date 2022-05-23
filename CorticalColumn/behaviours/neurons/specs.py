"""
General specifications needed for spiking neurons.
"""

from PymoNNto import Behaviour


class Fire(Behaviour):
    """
    Basic firing behavior of spiking neurons:

    if v >= threshold then v = v_reset.
    """

    def new_iteration(self, neurons):
        neurons.spikes = neurons.v >= neurons.threshold
        neurons.v[neurons.spikes] = neurons.v_reset
