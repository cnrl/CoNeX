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


class SpikeTrace(Behaviour):
    def set_variables(self, neurons):
        self.set_init_attrs_as_variables(neurons)
        neurons.trace = neurons.get_neuron_vec(mode="zeros()")

    def new_iteration(self, neurons):
        dx = -neurons.trace / neurons.tau_s + neurons.spikes
        neurons.trace += dx
