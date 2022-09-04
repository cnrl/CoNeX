"""
General specifications needed for spiking neurons.
"""

from PymoNNto import Behaviour
# TODO: add KWTA behavior.

class Fire(Behaviour):
    """
    Basic firing behavior of spiking neurons:

    if v >= threshold then v = v_reset.
    """

    def new_iteration(self, neurons):
        neurons.spikes = neurons.v >= neurons.threshold
        neurons.v[neurons.spikes] = neurons.v_reset
        neurons.effective_spikes = neurons.spikes.copy()


class SpikeTrace(Behaviour):
    def set_variables(self, neurons):
        self.set_init_attrs_as_variables(neurons)
        self.add_tag("trace")
        t = neurons.afferent_synapses.max_delay
        neurons.traces = neurons.get_neuron_vec_buffer(t)

    def new_iteration(self, neurons):
        dx = -neurons.traces / neurons.tau_s + neurons.spikes
        x = neurons.traces[:, 0] + dx
        neurons.buffer_roll(neurons.traces, x)


class SpikeHistory(Behaviour):
    def set_variables(self, neurons):
        self.add_tag("history")
        t = neurons.afferent_synapses.max_delay
        neurons.spike_history = neurons.get_neuron_vec_buffer(t)

    def new_iteration(self, neurons):
        neurons.buffer_roll(neurons.spike_history, neurons.spikes)
