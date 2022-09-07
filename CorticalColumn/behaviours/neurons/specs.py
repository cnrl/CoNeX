"""
General specifications needed for spiking neurons.
"""

import numpy as np
from PymoNNto import Behaviour


class Fire(Behaviour):
    """
    Basic firing behavior of spiking neurons:

    if v >= threshold then v = v_reset.
    """

    def new_iteration(self, neurons):
        neurons.spikes = neurons.v >= neurons.threshold
        neurons.v[neurons.spikes] = neurons.v_reset
        neurons.effective_spikes = neurons.spikes.copy()


class KWTA(Behaviour):
    """
    KWTA behavior of spiking neurons:

    if v >= threshold then v = v_reset and all other neurons are inhibited.
    """

    def __calc_inhibited(self, neurons, winners_masked):
        """
        Calculates the inhibition rate for each neuron.
        """
        max_sub_threshold = neurons.v[neurons.v < neurons.threshold].max()
        val = neurons.threshold - max_sub_threshold
        sup_threshold = winners_masked[
            (~winners_masked.mask) & (winners_masked >= neurons.threshold)
        ]
        return (
            val * (sup_threshold - sup_threshold.min()) / sup_threshold.ptp()
            - max_sub_threshold
        )

    def set_variables(self, neurons):
        self.k = self.get_init_attr("k", 10)

    def new_iteration(self, neurons):
        will_spike = neurons.v * (neurons.v >= neurons.threshold)

        if will_spike.sum() <= self.k:
            return

        kw = np.argpartition(will_spike, -self.k)[-self.k :]
        winners_masked = np.ma.array(neurons.v, mask=False)
        winners_masked.mask[kw] = True
        winners_masked[
            (~winners_masked.mask) & (winners_masked >= neurons.threshold)
        ] = self.__calc_inhibited(neurons, winners_masked)


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
