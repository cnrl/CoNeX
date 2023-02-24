"""
Leaky Integrate-and-Fire variants.

the dynamics can be represented by:
tau*dv/dt = F(u) + RI(u).
"""

# TODO initialization
# TODO multi-adaptation paradime

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
        v_rest (flaot): neuron membrane potential in absent of input. 
    """

    def set_variables(self, neurons):
        """
        Set neuron attributes. and adds Fire fucntion as attribute to population.

        Args:
            neurons (NeuronGroup): the neural population.
        """
        self.add_tag(self.__class__.__name__)

        neurons.R = self.get_init_attr('R', None)
        neurons.tau = self.get_init_attr('tau', None)
        neurons.threshold = self.get_init_attr('threshold', None)
        neurons.v_reset = self.get_init_attr('v_reset', None)
        neurons.v_rest = self.get_init_attr('v_rest', None)

        neurons.v = self.get_init_attr('init_v', neurons.get_neuron_vec())
        neurons.spikes = self.get_init_attr('init_s', neurons.v >= neurons.threshold)

        neurons.Fire = self.Fire

    def _RIu(self, neurons):
        """
        Part of neuron dynamic for voltage-dependent input resistance and internal currents.
        """
        return neurons.R * neurons.I

    def _Fu(self, neurons):
        """
        Leakage dynamic
        """
        return -1 * (neurons.v - neurons.v_rest)

    @classmethod
    def Fire(cls, neurons):
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
        neurons.v += (self._Fu(neurons) + self._RIu(neurons)) * neurons.network.dt / neurons.tau


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
        v_rest (flaot): neuron membrane potential in absent of input. 
        delta (float): the constant defining the sharpness of exponential curve.
        theta_rh (float): The boosting threshold. (rheobase)
    """

    def set_variables(self, neurons):
        """
        Set neuron attributes.

        Args:
            neurons (NeuronGroup): the neural population.
        """
        super().set_variables(neurons)

        neurons.delta = self.get_init_attr('delta', None)
        neurons.theta_rh = self.get_init_attr('theta_rh', None)

    def _Fu(self, neurons):
        """
        Leakage dynamic
        """
        return super()._Fu(neurons) + neurons.delta * torch.exp((neurons.u - neurons.theta_rh) / neurons.delta)


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
        v_rest (flaot): neuron membrane potential in absent of input.
        delta (float): the constant defining the sharpness of exponential curve.
        theta_rh (float): The boosting threshold.
        alpha (float): subthreshold adaptation parameter.
        beta (float): spike-triggered adaptation parameter.
        w_tau (flaot): time constant of adaptation decay.
    """

    def set_variables(self, neurons):
        """
        Set neuron attributes.

        Args:
            neurons (NeuronGroup): the neural population.
        """
        super().set_variables(neurons)
        
        neurons.alpha = self.get_init_attr('alpha', None)
        neurons.beta = self.get_init_attr('beta', None)
        neurons.w_tau = self.get_init_attr('w_tau', None)

        neurons.omega = self.get_init_attr('omega', neurons.get_neuron_vec())

    def _RIu(self, neurons):
        """
        Part of neuron dynamic for voltage-dependent input resistance and internal currents.
        """
        return -1 * (neurons.R * neurons.omega) + super()._RIu(neurons)

    @classmethod
    def domega_dt(cls, neurons):
        """
        Single step adaptation dynamics of AELIF neurons.

        Args:
            neurons (NeuronGroup): the neural population.
        """
        spike_adaptation = neurons.beta * neurons.w_tau * neurons.spikes
        sub_thresh_adaptation = neurons.alpha * (neurons.v - neurons.v_rest)
        return (sub_thresh_adaptation - neurons.omega + spike_adaptation) * neurons.network.dt / neurons.w_tau

    @classmethod
    def Fire(cls, neurons):
        """
        Basic firing behavior of spiking neurons:

        if v >= threshold then v = v_reset.
        
        and it do the adaptation.
        """
        super().Fire(neurons)
        neurons.omega += cls.domega_dt(neurons)