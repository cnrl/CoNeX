"""
Learning rules.
"""

from pymonntorch import Behavior

import torch
import torch.nn.functional as F

from conex.behaviors.synapses.specs import PreTrace, PostTrace

# TODO docstring for bound functions


def soft_bound(w, w_min, w_max):
    return (w - w_min) * (w_max - w)


def hard_bound(w, w_min, w_max):
    return (w > w_min) * (w < w_max)


def no_bound(w, w_min, w_max):
    return 1


BOUNDS = {"soft_bound": soft_bound, "hard_bound": hard_bound, "no_bound": no_bound}


class BaseLearning(Behavior):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_tag("weight_learning")

    def compute_dw(self, synapse):
        ...

    def forward(self, synapse):
        synapse.weights += self.compute_dw(synapse)


class SimpleSTDP(BaseLearning):
    """
    Spike-Timing Dependent Plasticity (STDP) rule for simple connections.

    Note: The implementation uses local variables (spike trace).

    Args:
        w_min (float): Minimum for weights. The default is 0.0.
        w_max (float): Maximum for weights. The default is 1.0.
        a_plus (float): Coefficient for the positive weight change. The default is None.
        a_minus (float): Coefficient for the negative weight change. The default is None.
        positive_bound (str or function): Bounding mechanism for positive learning. Accepting "no_bound", "hard_bound" and "soft_bound". The default is "no_bound". "weights", "w_min" and "w_max" pass as arguments for a bounding function.
        negative_bound (str or function): Bounding mechanism for negative learning. Accepting "no_bound", "hard_bound" and "soft_bound". The default is "no_bound". "weights", "w_min" and "w_max" pass as arguments for a bounding function.
    """

    def __init__(
        self,
        a_plus,
        a_minus,
        *args,
        w_min=0.0,
        w_max=1.0,
        positive_bound=None,
        negative_bound=None,
        **kwargs,
    ):
        super().__init__(
            *args,
            a_plus=a_plus,
            a_minus=a_minus,
            w_min=w_min,
            w_max=w_max,
            positive_bound=positive_bound,
            negative_bound=negative_bound,
            **kwargs,
        )

    def initialize(self, synapse):
        self.w_min = self.parameter("w_min", 0.0)
        self.w_max = self.parameter("w_max", 1.0)
        self.a_plus = self.parameter("a_plus", None, required=True)
        self.a_minus = self.parameter("a_minus", None, required=True)
        self.p_bound = self.parameter("positive_bound", "no_bound")
        self.n_bound = self.parameter("negative_bound", "no_bound")

        self.p_bound = (
            BOUNDS[self.p_bound] if isinstance(self.p_bound, str) else self.p_bound
        )
        self.n_bound = (
            BOUNDS[self.n_bound] if isinstance(self.n_bound, str) else self.n_bound
        )

        self.def_dtype = (
            torch.float32
            if not hasattr(synapse.network, "def_dtype")
            else synapse.network.def_dtype
        )

    def compute_dw(self, synapse):
        dw_minus = (
            torch.outer(synapse.pre_spike, synapse.post_trace)
            * self.a_minus
            * self.n_bound(synapse.weights, self.w_min, self.w_max)
        )
        dw_plus = (
            torch.outer(synapse.pre_trace, synapse.post_spike)
            * self.a_plus
            * self.p_bound(synapse.weights, self.w_min, self.w_max)
        )

        return dw_plus - dw_minus


class SparseSTDP(SimpleSTDP):
    """
    Spike-Timing Dependent Plasticity (STDP) rule for sparse connections.

    Note: The implementation uses local variables (spike trace).

    Args:
        w_min (float): Minimum for weights. The default is 0.0.
        w_max (float): Maximum for weights. The default is 1.0.
        a_plus (float): Coefficient for the positive weight change. The default is None.
        a_minus (float): Coefficient for the negative weight change. The default is None.
        positive_bound (str or function): Bounding mechanism for positive learning. Accepting "no_bound", "hard_bound" and "soft_bound". The default is "no_bound". "weights", "w_min" and "w_max" pass as arguments for a bounding function.
        negative_bound (str or function): Bounding mechanism for negative learning. Accepting "no_bound", "hard_bound" and "soft_bound". The default is "no_bound". "weights", "w_min" and "w_max" pass as arguments for a bounding function.
    """

    def compute_dw(self, synapse):
        weight_data = synapse.weights.values()[:]
        dw_minus = (
            synapse.pre_spike[synapse.src_idx]
            * synapse.post_trace[synapse.dst_idx]
            * self.a_minus
            * self.n_bound(weight_data, self.w_min, self.w_max)
        )
        dw_plus = (
            synapse.pre_trace[synapse.src_idx]
            * synapse.post_spike[synapse.dst_idx]
            * self.a_plus
            * self.p_bound(weight_data, self.w_min, self.w_max)
        )

        return dw_plus - dw_minus

    def forward(self, synapse):
        synapse.weights.values()[:] += self.compute_dw(synapse)


class One2OneSTDP(SimpleSTDP):
    """
    Spike-Timing Dependent Plasticity (STDP) rule for One 2 One connections.

    Note: The implementation uses local variables (spike trace).

    Args:
        w_min (float): Minimum for weights. The default is 0.0.
        w_max (float): Maximum for weights. The default is 1.0.
        a_plus (float): Coefficient for the positive weight change. The default is None.
        a_minus (float): Coefficient for the negative weight change. The default is None.
        positive_bound (str or function): Bounding mechanism for positive learning. Accepting "no_bound", "hard_bound" and "soft_bound". The default is "no_bound". "weights", "w_min" and "w_max" pass as arguments for a bounding function.
        negative_bound (str or function): Bounding mechanism for negative learning. Accepting "no_bound", "hard_bound" and "soft_bound". The default is "no_bound". "weights", "w_min" and "w_max" pass as arguments for a bounding function.
    """

    def compute_dw(self, synapse):
        dw_minus = (
            synapse.pre_spike
            * synapse.post_trace
            * self.a_minus
            * self.n_bound(synapse.weights, self.w_min, self.w_max)
        )
        dw_plus = (
            synapse.pre_trace
            * synapse.post_spike
            * self.a_plus
            * self.p_bound(synapse.weights, self.w_min, self.w_max)
        )

        return dw_plus - dw_minus


class SimpleiSTDP(BaseLearning):
    """
    Implementation of symmetric inhibitory Spike-Time Dependent Plasticity (iSTDP).
    DOI: 10.1126/science.1211095

    Note: The implementation uses local variables (spike trace).
          The implementation assumes that tau is in milliseconds.

    Args:
        lr (float): Learning rate. The Default is 1e-5.
        rho (float): Constant that determines the fire rate of target neurons.
        alpha (float): Manual constant for target trace, which replace rho value.
    """

    def __init__(
        self,
        *args,
        rho=None,
        alpha=None,
        lr=1e-5,
        is_inhibitory=True,
        **kwargs,
    ):
        super().__init__(
            *args, lr=lr, rho=rho, alpha=alpha, is_inhibitory=is_inhibitory, **kwargs
        )

    def initialize(self, synapse):
        self.lr = self.parameter("lr", 1e-5)
        self.rho = self.parameter("rho", None)
        self.change_sign = 1 - self.parameter("is_inhibitory", True) * 2
        self.alpha = self.parameter("alpha", None)

        # messy till I move trace to synapse.
        pre_tau = [
            synapse.behavior[key_behavior]
            for key_behavior in synapse.behavior
            if isinstance(synapse.behavior[key_behavior], PreTrace)
        ][0].tau_s
        post_tau = [
            synapse.behavior[key_behavior]
            for key_behavior in synapse.behavior
            if isinstance(synapse.behavior[key_behavior], PostTrace)
        ][0].tau_s

        assert (
            pre_tau == post_tau
        ), "for Symmetric iSTDP, pre and post trace decay should be equal."

        if self.alpha is None:
            self.alpha = 2 * self.rho * pre_tau / 1000

    def compute_dw(self, synapse):
        pre_spike_changes = torch.outer(
            synapse.pre_spike, (self.alpha - synapse.post_trace) * self.change_sign
        )
        post_spike_changes = torch.outer(synapse.pre_trace, synapse.post_spike)
        return self.lr * (pre_spike_changes + post_spike_changes)


class One2OneiSTDP(SimpleiSTDP):
    """
    Implementation of symmetric inhibitory Spike-Time Dependent Plasticity (iSTDP) for One 2 One Connections.
    DOI: 10.1126/science.1211095

    Note: The implementation uses local variables (spike trace).
          The implementation assumes that tau is in milliseconds.

    Args:
        lr (float): Learning rate. The Default is 1e-5.
        rho (float): Constant that determines the fire rate of target neurons.
        alpha (float): Manual constant for target trace, which replace rho value.
    """

    def compute_dw(self, synapse):
        pre_spike_changes = (
            synapse.pre_spike * (self.alpha - synapse.post_trace) * self.change_sign
        )
        post_spike_changes = synapse.pre_trace * synapse.post_spike
        return self.lr * (pre_spike_changes + post_spike_changes)


class SparseiSTDP(SimpleiSTDP):
    """
    Implementation of symmetric inhibitory Spike-Time Dependent Plasticity (iSTDP) for Sparse Connections.
    DOI: 10.1126/science.1211095

    Note: The implementation uses local variables (spike trace).
          The implementation assumes that tau is in milliseconds.

    Args:
        lr (float): Learning rate. The Default is 1e-5.
        rho (float): Constant that determines the fire rate of target neurons.
        alpha (float): Manual constant for target trace, which replace rho value.
    """

    def compute_dw(self, synapse):
        pre_spike_changes = (
            synapse.pre_spike[synapse.src_idx]
            * (self.alpha - synapse.post_trace)[synapse.dst_idx]
            * self.change_sign
        )
        post_spike_changes = (
            synapse.pre_trace[synapse.src_idx] * synapse.post_spike[synapse.dst_idx]
        )
        return self.lr * (pre_spike_changes + post_spike_changes)

    def forward(self, synapse):
        synapse.weights.values()[:] += self.compute_dw(synapse)


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
        src_spike = synapse.pre_spike.view(synapse.src_shape).to(self.def_dtype)
        src_spike = F.unfold(
            src_spike,
            kernel_size=synapse.weights.size()[-2:],
            stride=synapse.stride,
            padding=synapse.padding,
        )
        src_spike = src_spike.expand(synapse.dst_shape[0], *src_spike.shape)

        dst_spike_trace = synapse.post_trace.view((synapse.dst_shape[0], -1, 1))

        dw_minus = torch.bmm(src_spike, dst_spike_trace).view(
            synapse.weights.shape
        ) * self.n_bound(synapse.weights, self.w_min, self.w_max)

        src_spike_trace = synapse.pre_trace.view(synapse.src_shape)
        src_spike_trace = F.unfold(
            src_spike_trace,
            kernel_size=synapse.weights.size()[-2:],
            stride=synapse.stride,
            padding=synapse.padding,
        )
        src_spike_trace = src_spike_trace.expand(
            synapse.dst_shape[0], *src_spike_trace.shape
        )

        dst_spike = synapse.post_spike.view((synapse.dst_shape[0], -1, 1)).to(
            self.def_dtype
        )

        dw_plus = torch.bmm(src_spike_trace, dst_spike).view(
            synapse.weights.shape
        ) * self.p_bound(synapse.weights, self.w_min, self.w_max)

        return (dw_plus * self.a_plus - dw_minus * self.a_minus) / self.weight_divisor


class Local2dSTDP(SimpleSTDP):
    """
    Spike-Timing Dependent Plasticity (STDP) rule for 2D local connections.
    """

    def compute_dw(self, synapse):
        src_spike = synapse.pre_spike.view(synapse.src_shape).to(self.def_dtype)
        src_spike = F.unfold(
            src_spike,
            kernel_size=synapse.kernel_shape[-2:],
            stride=synapse.stride,
            padding=synapse.padding,
        )
        src_spike = src_spike.transpose(0, 1)
        src_spike = src_spike.expand(synapse.dst_shape[0], *src_spike.shape)

        dst_spike_trace = synapse.post_trace.view((synapse.dst_shape[0], -1, 1))
        dst_spike_trace = dst_spike_trace.expand(synapse.weights.shape)

        dw_minus = (
            dst_spike_trace
            * src_spike
            * self.n_bound(synapse.weights, self.w_min, self.w_max)
        )

        src_spike_trace = synapse.pre_trace.view(synapse.src_shape)
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

        dst_spike = synapse.pre_spike.view((synapse.dst_shape[0], -1, 1)).to(
            self.def_dtype
        )
        dst_spike = dst_spike.expand(synapse.weights.shape)

        dw_plus = (
            dst_spike
            * src_spike_trace
            * self.p_bound(synapse.weights, self.w_min, self.w_max)
        )

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

    def __init__(
        self,
        a_plus,
        a_minus,
        tau_c,
        *args,
        init_c_mode=0,
        w_min=0.0,
        w_max=1.0,
        positive_bound=None,
        negative_bound=None,
        **kwargs,
    ):
        super().__init__(
            *args,
            a_plus=a_plus,
            a_minus=a_minus,
            tau_c=tau_c,
            init_c_mode=init_c_mode,
            w_min=w_min,
            w_max=w_max,
            positive_bound=positive_bound,
            negative_bound=negative_bound,
            **kwargs,
        )

    def initialize(self, synapse):
        super().initialize(synapse)
        self.tau_c = self.parameter("tau_c", None, required=True)
        mode = self.parameter("init_c_mode", 0)
        synapse.c = synapse.tensor(mode=mode, dim=synapse.weights.shape)

    def forward(self, synapse):
        computed_stdp = self.compute_dw(synapse)
        synapse.c += (-synapse.c / self.tau_c) + computed_stdp
        synapse.weights += synapse.c * synapse.network.dopamine_concentration


class One2OneRSTDP(One2OneSTDP, SimpleRSTDP):
    """
    Reward-modulated Spike-Timing Dependent Plasticity (RSTDP) rule for One 2 One connections.

    Note: The implementation uses local variables (spike trace).

    Args:
        a_plus (float): Coefficient for the positive weight change. The default is None.
        a_minus (float): Coefficient for the negative weight change. The default is None.
        tau_c (float): Time constant for the eligibility trace. The default is None.
        init_c_mode (int): Initialization mode for the eligibility trace. The default is 0.
    """

    pass


class SparseRSTDP(SparseSTDP):
    """
    Reward-modulated Spike-Timing Dependent Plasticity (RSTDP) rule for sparse connections.

    Note: The implementation uses local variables (spike trace).

    Args:
        a_plus (float): Coefficient for the positive weight change. The default is None.
        a_minus (float): Coefficient for the negative weight change. The default is None.
        tau_c (float): Time constant for the eligibility trace. The default is None.
        init_c_mode (int): Initialization mode for the eligibility trace. The default is 0.
    """

    def __init__(
        self,
        a_plus,
        a_minus,
        tau_c,
        *args,
        init_c_mode=0,
        w_min=0.0,
        w_max=1.0,
        positive_bound=None,
        negative_bound=None,
        **kwargs,
    ):
        super().__init__(
            *args,
            a_plus=a_plus,
            a_minus=a_minus,
            tau_c=tau_c,
            init_c_mode=init_c_mode,
            w_min=w_min,
            w_max=w_max,
            positive_bound=positive_bound,
            negative_bound=negative_bound,
            **kwargs,
        )

    def initialize(self, synapse):
        super().initialize(synapse)
        self.tau_c = self.parameter("tau_c", None, required=True)
        mode = self.parameter("init_c_mode", 0)
        synapse.c = synapse.tensor(mode=mode, dim=(synapse.weights._nnz(),))

    def forward(self, synapse):
        computed_stdp = self.compute_dw(synapse)
        synapse.c += (-synapse.c / self.tau_c) + computed_stdp
        synapse.weights.values()[:] += (
            synapse.c * synapse.network.dopamine_concentration
        )


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
