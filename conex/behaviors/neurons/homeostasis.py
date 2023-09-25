import torch
from pymonntorch import Behavior


class ActivityBaseHomeostasis(Behavior):
    """
    Homeostasis Based on the target activity of neurons.

    Note: Threshold of neurons should be a population size tensor.

    Args:
        window_size (int): The simulation steps to accumulate spikes.
        activity_rate (int):  The expected number of spikes in a window.
        updating_rate (float): A scaler to change update effect with.
        decay_rate (float): A scaler to change updating_rate after each applied homeostasis. The default is 1.0
    """

    def __init__(
        self, activity_rate, window_size, updating_rate, *args, decay_rate=1.0, **kwargs
    ):
        super().__init__(
            *args,
            activity_rate,
            window_size,
            updating_rate,
            decay_rate=decay_rate,
            **kwargs
        )

    def initialize(self, neurons):
        activity_rate = self.parameter("activity_rate", required=True)
        self.window_size = self.parameter("window_size", required=True)
        self.updating_rate = self.parameter("updating_rate", required=True)
        self.decay_rate = self.parameter("decay_rate", 1.0)

        self.firing_reward = 1
        self.non_firing_penalty = -activity_rate / (self.window_size - activity_rate)

        self.activities = neurons.vector(mode="zeros")

    def forward(self, neurons):
        add_activitiies = torch.where(
            neurons.spikes, self.firing_reward, self.non_firing_penalty
        )

        self.activities += add_activitiies

        if (neurons.iteration % self.window_size) == 0:
            change = -self.activities * self.updating_rate
            neurons.threshold -= change
            self.activities.fill_(0)
            self.updating_rate *= self.decay_rate


class VoltageBaseHomeostasis(Behavior):
    """
    Homeostasis base on the voltage rate of Neurons.

    Args:
        target_voltage (float): The expected voltage of neuron. Defaults to None.
        max_ta (float): The desired maximum voltage for a neuron. If not provided, the value of target_voltage is used.
        min_ta (float): The desired minimum voltage for a neuron. If not provided, the value of target_voltage is used.
        eta_ip (flaot): The updating speed of the homeostasis process. The default is 0.001.

    """

    def __init__(
        self,
        *args,
        target_voltage=None,
        max_ta=None,
        min_ta=None,
        eta_ip=0.001,
        **kwargs
    ):
        super().__init__(
            *args,
            target_voltage=target_voltage,
            max_ta=max_ta,
            min_ta=min_ta,
            eta_ip=eta_ip,
            **kwargs
        )

    def initialize(self, neurons):
        target_act = self.parameter("target_voltage")
        self.max_ta = self.parameter("max_ta", target_act)
        self.min_ta = self.parameter("min_ta", target_act)
        self.adj_strength = self.parameter("eta_ip", 0.001)

        neurons.exhaustion = neurons.vector()

    def forward(self, neurons):
        greater = (neurons.v > self.max_ta) * (neurons.v - self.max_ta)
        smaller = (neurons.v < self.min_ta) * (neurons.v - self.min_ta)

        change = (greater + smaller) * self.adj_strength

        neurons.exhaustion += change
        neurons.v -= neurons.exhaustion
