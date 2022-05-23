"""
Leaky Integrate-and-Fire variants.
"""

import numpy as np
from PymoNNto import Behaviour


class LIF(Behaviour):
    """
    The neural dynamics of LIF is defined by:

    tau*dv/dt = v_rest - v + R*I.

    We assume that the input to the neuron is current-based.

    Args:
        tau (float): time constant of voltage decay.
        r (float): the resistance of the membrane potential.
    """

    def set_variables(self, neurons):
        """
        Set neuron attributes.

        Args:
            neurons (NeuronGroup): the neural population.
        """
        self.add_tag("LIF")
        self.set_init_attrs_as_variables(neurons)

    def _dv_dt(self, neurons):
        delta_v = neurons.v_rest - neurons.v
        current = (
            neurons.proximal_input_current
            + neurons.basal_input_current
            + neurons.apical_input_current
        )
        ri = neurons.r * current
        dv_dt = delta_v + ri

        if neurons.noisy_current:
            dv_dt += neurons.noisy_current

        return dv_dt

    def new_iteration(self, neurons):
        """
        Firing behavior of LIF neurons.

        Args:
            neurons (NeuronGroup): the neural population.
        """
        neurons.v += self._dv_dt(neurons) / neurons.tau


class ELIF(LIF):
    """
    The neural dynamics of Exponential LIF is defined by:

    F(u) = delta * exp((v - theta_rh) / delta),
    tau*dv/dt = v_rest - v + F(u) + R*I.

    We assume that the input to the neuron is current-based.

    Args:
        tau (float): time constant of voltage decay.
        r (float): the resistance of the membrane potential.
        delta (float): the constant defining the sharpness of exponential curve.
        theta_rh (float): The boosting threshold.
    """

    def set_variables(self, neurons):
        """
        Set neuron attributes.

        Args:
            neurons (NeuronGroup): the neural population.
        """
        self.add_tag("ELIF")
        self.set_init_attrs_as_variables(neurons)

    def _dv_dt(self, neurons):
        """
        Single step voltage dynamics of exponential LIF neurons.

        Args:
            neurons (NeuronGroup): the neural population.
        """
        simple = super()._dv_dt(neurons)
        v_m = (neurons.v - neurons.theta_rh) / neurons.delta
        nonlinear_f = neurons.delta * np.exp(v_m)
        return simple + nonlinear_f

    def new_iteration(self, neurons):
        """
        Single step of LIF dynamics.

        Args:
            neurons (NeuronGroup): the neural population.
        """
        neurons.v += self._dv_dt(neurons) / neurons.tau


class AELIF(ELIF):
    """
    The neural dynamics of Adaptive Exponential LIF is defined by:

    F(u) = delta * exp((v - theta_rh) / delta),
    tau*dv/dt = v_rest - v + F(u) + R*I - R*omega,
    tau_a*d(omega)/dt = alpha*(v - v_rest) - omega + beta*tau_a*spikes.

    We assume that the input to the neuron is current-based.

    Args:
        tau (float): time constant of voltage decay.
        r (float): the resistance of the membrane potential.
        delta (float): the constant defining the sharpness of exponential curve.
        theta_rh (float): The boosting threshold.
        alpha (float): subthreshold adaptation parameter.
        beta (float): spike-triggered adaptation parameter.
    """

    def set_variables(self, neurons):
        """
        Set neuron attributes.

        Args:
            neurons (NeuronGroup): the neural population.
        """
        self.add_tag("AELIF")
        self.set_init_attrs_as_variables(neurons)

        neurons.omega = neurons.get_neuron_vec(mode="zeros()")

    def _domega_dt(self, neurons):
        """
        Single step adaptation dynamics of AELIF neurons.

        Args:
            neurons (NeuronGroup): the neural population.
        """
        spike_adaptation = neurons.beta * neurons.tau_a * neurons.spikes
        sub_thresh_adaptation = neurons.alpha * (neurons.v - neurons.v_rest)
        return sub_thresh_adaptation + spike_adaptation - neurons.omega

    def new_iteration(self, neurons):
        """
        Single step of LIF dynamics.

        Args:
            neurons (NeuronGroup): the neural population.
        """
        dv_dt = self._dv_dt(neurons) - neurons.r * neurons.omega
        neurons.v += dv_dt / neurons.tau
