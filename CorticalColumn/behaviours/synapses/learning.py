"""
Learning rules.
"""

from PymoNNto import Behaviour


class STDP(Behaviour):
    def set_variables(self, synapses):
        self.get_init_attr("a_plus", 1.0, synapses)
        self.get_init_attr("a_minus", 1.0, synapses)

    def new_iteration(self, synapses):
        if "history" in synapses.src.tags:
            src_spikes = synapses.src.spike_history[:, synapses.delays]
            src_trace = synapses.src.traces[:, synapses.delays]
        else:
            src_spikes = synapses.src.spikes
            src_trace = synapses.src.traces[:, 0]
        dw_plus = self.a_plus * src_trace * synapses.dst.spikes
        dw_minus = self.a_minus * synapses.dst.trace * src_spikes
        self.weights += dw_plus - dw_minus


class RSTDP(STDP):
    def set_variables(self, synapses):
        self.get_init_attr("tau_c", 1000.0, synapses, required=True)
        synapses.c = synapses.get_mat(mode="zeros()")

    def new_iteration(self, synapses):
        stdp = super().new_iteration(synapses)
        dc = -synapses.c / self.tau_c + stdp
        synapses.c += dc
        synapses.weights += synapses.c * synapses.network.dopamine_concentration
