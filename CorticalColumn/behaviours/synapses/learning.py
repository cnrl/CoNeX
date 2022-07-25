"""
Learning rules.
"""

import numba
import numpy as np
from PymoNNto import Behaviour

from CorticalColumn.nn.Modules.topological_connections import (
    ConvSynapseGroup,
    SparseSynapseGroup,
)


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

    @numba.jit(nopython=True, parallel=True)
    def new_iteration(self, synapses):
        if isinstance(synapses, SparseSynapseGroup):
            # TODO: compare efficiency with the case of working with sub-synapses
            dw = self.compute(synapses)
            indices = self.get_topology()
            self.weights[indices] += dw[indices]

        elif isinstance(synapses, ConvSynapseGroup):
            dw = synapses.get_synapse_mat(mode="zeros")

            src_grid = np.indices((synapses.src.x, synapses.src.y, synapses.src.z))
            dst_grid = np.indices((synapses.dst.x, synapses.dst.y, synapses.dst.z))

            dim = np.prod(synapses.receptive_field)
            new_src = (np.arange(dim) == src_grid[..., None] - 1).astype(int)
            new_dst = np.repeat(dst_grid[:, :, :, np.newaxis], dim, axis=-1)

            for i in numba.prange(dim):
                x = i // synapses.receptive_field[0]
                yz = i % synapses.receptive_field[0]
                y = yz // synapses.receptive_field[1]
                z = yz % synapses.receptive_field[1]
                z = z // synapses.receptive_field[2]

                subsyn = synapses.get_sub_synapse_group(new_src[i], new_dst[i])
                subsyn.delay = synapses.delay[x, y, z]
                subsyn.weights = synapses.weights[x, y, z]

                dw[x, y, z] = self.compute(subsyn).sum()

                del subsyn

            del src_grid, dst_grid, new_src, new_dst

            self.weights += dw


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
