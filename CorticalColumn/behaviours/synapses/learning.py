"""
Learning rules.
"""

import numpy as np
from PymoNNto import Behaviour


class STDP(Behaviour):
    def set_variables(self, synapses):
        self.get_init_attr("a_plus", 1.0, synapses)
        self.get_init_attr("a_minus", 1.0, synapses)

    def compute(self, synapses):
        if "history" in synapses.src.tags:
            d = synapses.get_delay_as_index()

            src_spikes = synapses.src.spike_history.repeat(synapses.dst.size)
            src_spikes = src_spikes * (1 - d[1]) + np.roll(src_spikes, 1, axis=2) * d[1]

            src_trace = synapses.src.traces.repeat(synapses.dst.size)
            src_trace = src_trace * (1 - d[1]) + np.roll(src_trace, 1, axis=2) * d[1]

            grid = np.indices((synapses.dst.size, synapses.src.size))

            src_spikes = src_spikes[grid[0], grid[1], d[0]]
            src_trace = src_trace[grid[0], grid[1], d[0]]
        else:
            src_spikes = synapses.src.spikes
            src_trace = synapses.src.traces[:, 0]

        dw_plus = self.a_plus * synapses.dst.spikes * src_trace
        dw_minus = self.a_minus * synapses.dst.trace * src_spikes
        return dw_plus - dw_minus

    def new_iteration(self, synapses):
        dw = self.compute(synapses)
        indices = self.get_topology()
        self.weights[indices] += dw[indices]


class RSTDP(STDP):
    def set_variables(self, synapses):
        self.get_init_attr("tau_c", 1000.0, synapses, required=True)
        synapses.c = synapses.get_mat(mode="zeros()")

    def new_iteration(self, synapses):
        stdp = self.compute(synapses)
        indices = self.get_topology()
        dc = -synapses.c[indices] / self.tau_c + stdp[indices]
        synapses.c[indices] += dc
        synapses.weights[indices] += (
            synapses.c[indices] * synapses.network.dopamine_concentration
        )
