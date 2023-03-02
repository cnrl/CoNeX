"""
Learning rules.
"""

from pymonntorch import Behavior

import torch
import torch.nn.functional as F 

# TODO do we need padding? WE DO
# TODO super class

# Super class
class STDP(Behavior):
    pass

class SimpleSTDP(Behavior):
    def set_variables(self, synapse):
        self.a_plus = self.get_init_attr('a_plus', None)
        self.a_minus = self.get_init_attr('a_minus', None)

    def get_spike_and_trace(self, synapse):
        src_spike = synapse.src.axon.get_spike(synapse.src, synapse.src_delay)
        dst_spike = synapse.dst.axon.get_spike(synapse.dst, synapse.dst_delay)

        src_spike_trace = synapse.src.axon.get_spike_trace(synapse.src, synapse.src_delay)
        dst_spike_trace = synapse.dst.axon.get_spike_trace(synapse.dst, synapse.dst_delay)

        return src_spike, dst_spike, src_spike_trace, dst_spike_trace

    def forward(self, synapse):
        src_spike, dst_spike, src_spike_trace, dst_spike_trace = self.get_spike_and_trace(synapse)

        dw_plus = torch.outer(dst_spike_trace, src_spike) * self.a_plus
        dw_minus = torch.outer(dst_spike, src_spike_trace) * self.a_minus

        synapse.weights += dw_plus - dw_minus

class Conv2dSTDP(SimpleSTDP):
    def set_variables(self, synapse):
        super().set_variables(synapse)
        self.weight_divisor = synapse.dst_shape[2] * synapse.dst_shape[1]

    def forward(self, synapse):
        src_spike, dst_spike, src_spike_trace, dst_spike_trace = self.get_spike_and_trace(synapse)

        src_spike = src_spike.reshape(synapse.src_shape).to(torch.float32)
        src_spike = F.unfold(src_spike, kernel_size=synapse.weights.size()[-2:], stride = synapse.stride)
        src_spike = src_spike.expand(synapse.dst_shape[0], *src_spike.shape)

        dst_spike_trace = dst_spike_trace.reshape((synapse.dst_shape[0], -1, 1))

        dw_minus = torch.bmm(src_spike, dst_spike_trace).reshape(synapse.weights.shape)
        
        src_spike_trace = src_spike_trace.reshape(synapse.src_shape)
        src_spike_trace = F.unfold(src_spike_trace, kernel_size=synapse.weights.size()[-2:], stride = synapse.stride)
        src_spike_trace = src_spike_trace.expand(synapse.dst_shape[0], *src_spike_trace.shape)

        dst_spike = dst_spike.reshape((synapse.dst_shape[0], -1, 1)).to(torch.float32)
        
        dw_plus = torch.bmm(src_spike_trace, dst_spike).reshape(synapse.weights.shape)

        synapse.weights += (dw_plus * self.a_plus - dw_minus * self.a_minus) / self.weight_divisor

class Local2dSTDP(SimpleSTDP):
    def forward(self, synapse):
        src_spike, dst_spike, src_spike_trace, dst_spike_trace = self.get_spike_and_trace(synapse)

        src_spike = src_spike.reshape(synapse.src_shape).to(torch.float32)
        src_spike = F.unfold(src_spike, kernel_size=synapse.kernel_shape[-2:], stride = synapse.stride)
        src_spike = src_spike.transpose(0, 1)
        src_spike = src_spike.expand(synapse.dst_shape[0], *src_spike.shape)

        dst_spike_trace = dst_spike_trace.reshape((synapse.dst_shape[0], -1, 1))
        dst_spike_trace = dst_spike_trace.expand(synapse.weights.shape)

        dw_minus = dst_spike_trace * src_spike
        
        src_spike_trace = src_spike_trace.reshape(synapse.src_shape)
        src_spike_trace = F.unfold(src_spike_trace, kernel_size=synapse.kernel_shape[-2:], stride = synapse.stride)
        src_spike_trace = src_spike_trace.transpose(0, 1)
        src_spike_trace = src_spike_trace.expand(synapse.dst_shape[0], *src_spike_trace.shape)

        dst_spike = dst_spike.reshape((synapse.dst_shape[0], -1, 1)).to(torch.float32)
        dst_spike = dst_spike.expand(synapse.weights.shape)

        dw_plus = dst_spike * src_spike_trace

        synapse.weights += (dw_plus * self.a_plus - dw_minus * self.a_minus)

"""
class STDP_behaviour(Behaviour):
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
    def forward(self, synapses):
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


class RSTDP_behaviour(STDP_behaviour):
    def set_variables(self, synapses):
        self.get_init_attr("tau_c", 1000.0, synapses, required=True)
        synapses.c = synapses.get_mat(mode="zeros()")

    def forward(self, synapses):
        stdp = self.compute(synapses)
        indices = self.get_topology()
        dc = -synapses.c[indices] / self.tau_c + stdp[indices]
        synapses.c[indices] += dc
        synapses.weights[indices] += (
            synapses.c[indices] * synapses.network.dopamine_concentration
        )
"""