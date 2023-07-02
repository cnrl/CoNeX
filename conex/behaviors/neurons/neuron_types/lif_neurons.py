"""
Leaky Integrate-and-Fire variants.

the dynamics can be represented by:
tau*dv/dt = F(u) + RI(u).
"""

# TODO multi-adaptation paradigm

from pymonntorch import Behavior
import torch


class LIF(Behavior):
    """
    The neural dynamics of LIF is defined by:

    F(u) = v_rest - v,
    RI(u) = R*I.

    We assume that the input to the neuron is current-based.

    Note: at least one Input mechanism  should be added to the behaviors of the population.
          and Fire method should be called by other behaviors.

    Args:
        tau (float): time constant of voltage decay.
        R (float): the resistance of the membrane potential.
        threshold (float): the threshold of neurons to initiate spike.
        v_reset (float): immediate membrane potential after a spike.
        v_rest (float): neuron membrane potential in absent of input.
    """

    def __init__(
        self,
        R,
        threshold,
        tau,
        v_reset,
        v_rest,
        *args,
        init_v=None,
        init_s=None,
        **kwargs
    ):
        super().__init__(
            *args,
            R=R,
            tau=tau,
            threshold=threshold,
            v_reset=v_reset,
            v_rest=v_rest,
            init_v=init_v,
            init_s=init_s,
            **kwargs
        )

    def initialize(self, neurons):
        """
        Set neuron attributes. and adds Fire function as attribute to population.

        Args:
            neurons (NeuronGroup): the neural population.
        """
        self.add_tag(self.__class__.__name__)

        neurons.R = self.parameter("R", None, required=True)
        neurons.tau = self.parameter("tau", None, required=True)
        neurons.threshold = self.parameter("threshold", None, required=True)
        neurons.v_reset = self.parameter("v_reset", None, required=True)
        neurons.v_rest = self.parameter("v_rest", None, required=True)

        neurons.v = self.parameter("init_v", neurons.vector())
        neurons.spikes = self.parameter("init_s", neurons.v >= neurons.threshold)

        neurons.spiking_neuron = self

    def _RIu(self, neurons):
        """
        Part of neuron dynamic for voltage-dependent input resistance and internal currents.
        """
        return neurons.R * neurons.I

    def _Fu(self, neurons):
        """
        Leakage dynamic
        """
        return neurons.v_rest - neurons.v

    def Fire(self, neurons):
        """
        Basic firing behavior of spiking neurons:

        if v >= threshold then v = v_reset.
        """
        neurons.spikes = neurons.v >= neurons.threshold
        neurons.v[neurons.spikes] = neurons.v_reset

    def forward(self, neurons):
        """
        Single step of dynamics.

        Args:
            neurons (NeuronGroup): the neural population.
        """
        neurons.v += (
            (self._Fu(neurons) + self._RIu(neurons)) * neurons.network.dt / neurons.tau
        )


class ELIF(LIF):
    """
    The neural dynamics of Exponential LIF is defined by:

    F(u) = v_rest - v + delta * exp((v - theta_rh) / delta),
    RI(u) = R*I,

    We assume that the input to the neuron is current-based.

    Note: at least one Input mechanism and one Firing mechanism should be added to the behaviors of the population

    Args:
        tau (float): time constant of voltage decay.
        R (float): the resistance of the membrane potential.
        threshold (float): the threshold of neurons to initiate spike.
        v_reset (float): immediate membrane potential after a spike.
        v_rest (float): neuron membrane potential in absent of input.
        delta (float): the constant defining the sharpness of exponential curve.
        theta_rh (float): The boosting threshold. (rheobase)
    """

    def __init__(
        self,
        R,
        threshold,
        tau,
        v_reset,
        v_rest,
        delta,
        theta_rh,
        *args,
        init_v=None,
        init_s=None,
        **kwargs
    ):
        super().__init__(
            *args,
            R=R,
            tau=tau,
            threshold=threshold,
            v_reset=v_reset,
            v_rest=v_rest,
            delta=delta,
            theta_rh=theta_rh,
            init_v=init_v,
            init_s=init_s,
            **kwargs
        )

    def initialize(self, neurons):
        """
        Set neuron attributes.

        Args:
            neurons (NeuronGroup): the neural population.
        """
        super().initialize(neurons)

        neurons.delta = self.parameter("delta", None, required=True)
        neurons.theta_rh = self.parameter("theta_rh", None, required=True)

    def _Fu(self, neurons):
        """
        Leakage dynamic
        """
        return super()._Fu(neurons) + neurons.delta * torch.exp(
            (neurons.v - neurons.theta_rh) / neurons.delta
        )


class AELIF(ELIF):
    """
    The neural dynamics of Adaptive Exponential LIF is defined by:

    tau_a*d(omega)/dt = alpha*(v - v_rest) - omega + beta*tau_a*spikes,
    F(u) = v_rest - v + delta * exp((v - theta_rh) / delta),
    RI(u) = R*I - R*omega,

    We assume that the input to the neuron is current-based.

    Note: at least one Input mechanism and one Firing mechanism should be added to the behaviors of the population

    Args:
        tau (float): time constant of voltage decay.
        R (float): the resistance of the membrane potential.
        threshold (float): the threshold of neurons to initiate spike.
        v_reset (float): immediate membrane potential after a spike.
        v_rest (float): neuron membrane potential in absent of input.
        delta (float): the constant defining the sharpness of exponential curve.
        theta_rh (float): The boosting threshold.
        alpha (float): subthreshold adaptation parameter.
        beta (float): spike-triggered adaptation parameter.
        w_tau (float): time constant of adaptation decay.
    """

    def __init__(
        self,
        R,
        threshold,
        tau,
        v_reset,
        v_rest,
        delta,
        theta_rh,
        alpha,
        beta,
        w_tau,
        *args,
        init_v=None,
        init_s=None,
        omega=None,
        **kwargs
    ):
        super().__init__(
            *args,
            R=R,
            tau=tau,
            threshold=threshold,
            v_reset=v_reset,
            v_rest=v_rest,
            delta=delta,
            theta_rh=theta_rh,
            alpha=alpha,
            beta=beta,
            w_tau=w_tau,
            init_v=init_v,
            init_s=init_s,
            omega=omega,
            **kwargs
        )

    def initialize(self, neurons):
        """
        Set neuron attributes.

        Args:
            neurons (NeuronGroup): the neural population.
        """
        super().initialize(neurons)

        neurons.alpha = self.parameter("alpha", None, required=True)
        neurons.beta = self.parameter("beta", None, required=True)
        neurons.w_tau = self.parameter("w_tau", None, required=True)

        neurons.omega = self.parameter("omega", neurons.vector())

    def _RIu(self, neurons):
        """
        Part of neuron dynamic for voltage-dependent input resistance and internal currents.
        """
        return super()._RIu(neurons) - neurons.R * neurons.omega

    def domega_dt(self, neurons):
        """
        Single step adaptation dynamics of AELIF neurons.

        Args:
            neurons (NeuronGroup): the neural population.
        """
        spike_adaptation = neurons.beta * neurons.w_tau * neurons.spikes
        sub_thresh_adaptation = neurons.alpha * (neurons.v - neurons.v_rest)
        return (
            (sub_thresh_adaptation - neurons.omega + spike_adaptation)
            * neurons.network.dt
            / neurons.w_tau
        )

    def Fire(self, neurons):
        """
        Basic firing behavior of spiking neurons:

        if v >= threshold then v = v_reset.

        and it do the adaptation.
        """
        super().Fire(neurons)
        neurons.omega += self.domega_dt(neurons)
