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

    def compute_dw(self, synapse):
        src_spike, dst_spike, src_spike_trace, dst_spike_trace = self.get_spike_and_trace(synapse)

        dw_plus = torch.outer(dst_spike_trace, src_spike) * self.a_plus
        dw_minus = torch.outer(dst_spike, src_spike_trace) * self.a_minus

        return dw_plus - dw_minus

    def forward(self, synapse):
        synapse.weights += self.compute_dw(synapse)


class Conv2dSTDP(SimpleSTDP):
    def set_variables(self, synapse):
        super().set_variables(synapse)
        self.weight_divisor = synapse.dst_shape[2] * synapse.dst_shape[1]

    def compute_dw(self, synapse):
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

        return (dw_plus * self.a_plus - dw_minus * self.a_minus) / self.weight_divisor


class Local2dSTDP(SimpleSTDP):
    def compute_dw(self, synapse):
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

        return (dw_plus * self.a_plus - dw_minus * self.a_minus)


class SimpleRSTDP(SimpleSTDP):
    def set_variables(self, synapse):
        super().set_variables(synapse)
        self.tau_c = self.get_init_attr("tau_c", None)
        mode = self.get_init_attr('init_c_mode', 0)
        synapse.c = synapse._get_mat(mode=mode, dim=synapse.weights.shape)

    def forward(self, synapse):
        computed_stdp = self.compute_dw(synapse)
        synapse.c += (synapse.c / self.tau_c) + computed_stdp
        synapse.weights += synapse.c * synapse.network.dopamine_concentration


class Conv2dRSTDP(Conv2dSTDP, SimpleRSTDP):
    pass


class Local2dRSTDP(Local2dSTDP, SimpleRSTDP):
    pass
