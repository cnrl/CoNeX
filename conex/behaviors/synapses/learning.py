"""
Learning rules.
"""

from pymonntorch import Behavior

import torch
import torch.nn.functional as F

# TODO add boundings function


class SimpleSTDP(Behavior):
    """
    Spike-Timing Dependent Plasticity (STDP) rule for simple connections.

    Note: The implementation uses local variables (spike trace).

    Args:
        a_plus (float): Coefficient for the positive weight change. The default is None.
        a_minus (float): Coefficient for the negative weight change. The default is None.
    """

    def initialize(self, synapse):
        self.a_plus = self.parameter("a_plus", None, required=True)
        self.a_minus = self.parameter("a_minus", None, required=True)

        self.def_dtype = (
            torch.float32
            if not hasattr(synapse.network, "def_dtype")
            else synapse.network.def_dtype
        )

    def get_spike_and_trace(self, synapse):
        src_spike = synapse.src.axon.get_spike(synapse.src, synapse.src_delay)
        dst_spike = synapse.dst.axon.get_spike(synapse.dst, synapse.dst_delay)

        src_spike_trace = synapse.src.axon.get_spike_trace(
            synapse.src, synapse.src_delay
        )
        dst_spike_trace = synapse.dst.axon.get_spike_trace(
            synapse.dst, synapse.dst_delay
        )

        return src_spike, dst_spike, src_spike_trace, dst_spike_trace

    def compute_dw(self, synapse):
        (
            src_spike,
            dst_spike,
            src_spike_trace,
            dst_spike_trace,
        ) = self.get_spike_and_trace(synapse)

        dw_minus = torch.outer(src_spike, dst_spike_trace) * self.a_minus
        dw_plus = torch.outer(src_spike_trace, dst_spike) * self.a_plus

        return dw_plus - dw_minus

    def forward(self, synapse):
        synapse.weights += self.compute_dw(synapse)


class Conv2dSTDP(SimpleSTDP):
    """
    Spike-Timing Dependent Plasticity (STDP) rule for 2D convolutional connections.

    Note: The implementation uses local variables (spike trace).

    Args:
        a_plus (float): Coefficient for the positive weight change. The default is None.
        a_minus (float): Coefficient for the negative weight change. The default is None.
    """

    def initialize(self, synapse):
        super().initialize(synapse)
        self.weight_divisor = synapse.dst_shape[2] * synapse.dst_shape[1]

    def compute_dw(self, synapse):
        (
            src_spike,
            dst_spike,
            src_spike_trace,
            dst_spike_trace,
        ) = self.get_spike_and_trace(synapse)

        src_spike = src_spike.view(synapse.src_shape).to(self.def_dtype)
        src_spike = F.unfold(
            src_spike,
            kernel_size=synapse.weights.size()[-2:],
            stride=synapse.stride,
            padding=synapse.padding,
        )
        src_spike = src_spike.expand(synapse.dst_shape[0], *src_spike.shape)

        dst_spike_trace = dst_spike_trace.view((synapse.dst_shape[0], -1, 1))

        dw_minus = torch.bmm(src_spike, dst_spike_trace).view(synapse.weights.shape)

        src_spike_trace = src_spike_trace.view(synapse.src_shape)
        src_spike_trace = F.unfold(
            src_spike_trace,
            kernel_size=synapse.weights.size()[-2:],
            stride=synapse.stride,
            padding=synapse.padding,
        )
        src_spike_trace = src_spike_trace.expand(
            synapse.dst_shape[0], *src_spike_trace.shape
        )

        dst_spike = dst_spike.view((synapse.dst_shape[0], -1, 1)).to(self.def_dtype)

        dw_plus = torch.bmm(src_spike_trace, dst_spike).view(synapse.weights.shape)

        return (dw_plus * self.a_plus - dw_minus * self.a_minus) / self.weight_divisor


class Local2dSTDP(SimpleSTDP):
    """
    Spike-Timing Dependent Plasticity (STDP) rule for 2D local connections.
    """

    def compute_dw(self, synapse):
        (
            src_spike,
            dst_spike,
            src_spike_trace,
            dst_spike_trace,
        ) = self.get_spike_and_trace(synapse)

        src_spike = src_spike.view(synapse.src_shape).to(self.def_dtype)
        src_spike = F.unfold(
            src_spike,
            kernel_size=synapse.kernel_shape[-2:],
            stride=synapse.stride,
            padding=synapse.padding,
        )
        src_spike = src_spike.transpose(0, 1)
        src_spike = src_spike.expand(synapse.dst_shape[0], *src_spike.shape)

        dst_spike_trace = dst_spike_trace.view((synapse.dst_shape[0], -1, 1))
        dst_spike_trace = dst_spike_trace.expand(synapse.weights.shape)

        dw_minus = dst_spike_trace * src_spike

        src_spike_trace = src_spike_trace.view(synapse.src_shape)
        src_spike_trace = F.unfold(
            src_spike_trace,
            kernel_size=synapse.kernel_shape[-2:],
            stride=synapse.stride,
            padding=synapse.padding,
        )
        src_spike_trace = src_spike_trace.transpose(0, 1)
        src_spike_trace = src_spike_trace.expand(
            synapse.dst_shape[0], *src_spike_trace.shape
        )

        dst_spike = dst_spike.view((synapse.dst_shape[0], -1, 1)).to(self.def_dtype)
        dst_spike = dst_spike.expand(synapse.weights.shape)

        dw_plus = dst_spike * src_spike_trace

        return dw_plus * self.a_plus - dw_minus * self.a_minus


class SimpleRSTDP(SimpleSTDP):
    """
    Reward-modulated Spike-Timing Dependent Plasticity (RSTDP) rule for simple connections.

    Note: The implementation uses local variables (spike trace).

    Args:
        a_plus (float): Coefficient for the positive weight change. The default is None.
        a_minus (float): Coefficient for the negative weight change. The default is None.
        tau_c (float): Time constant for the eligibility trace. The default is None.
        init_c_mode (int): Initialization mode for the eligibility trace. The default is 0.
    """

    def initialize(self, synapse):
        super().initialize(synapse)
        self.tau_c = self.parameter("tau_c", None, required=True)
        mode = self.parameter("init_c_mode", 0)
        synapse.c = synapse._get_mat(mode=mode, dim=synapse.weights.shape)

    def forward(self, synapse):
        computed_stdp = self.compute_dw(synapse)
        synapse.c += (synapse.c / self.tau_c) + computed_stdp
        synapse.weights += synapse.c * synapse.network.dopamine_concentration


class Conv2dRSTDP(Conv2dSTDP, SimpleRSTDP):
    """
    Reward-modulated Spike-Timing Dependent Plasticity (RSTDP) rule for 2D convolutional connections.

    Note: The implementation uses local variables (spike trace).

    Args:
        a_plus (float): Coefficient for the positive weight change. The default is None.
        a_minus (float): Coefficient for the negative weight change. The default is None.
        tau_c (float): Time constant for the eligibility trace. The default is None.
        init_c_mode (int): Initialization mode for the eligibility trace. The default is 0.
    """

    pass


class Local2dRSTDP(Local2dSTDP, SimpleRSTDP):
    """
    Reward-modulated Spike-Timing Dependent Plasticity (RSTDP) rule for 2D local connections.

    Note: The implementation uses local variables (spike trace).

    Args:
        a_plus (float): Coefficient for the positive weight change. The default is None.
        a_minus (float): Coefficient for the negative weight change. The default is None.
        tau_c (float): Time constant for the eligibility trace. The default is None.
        init_c_mode (int): Initialization mode for the eligibility trace. The default is 0.
    """

    pass
