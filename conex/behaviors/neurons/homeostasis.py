import torch
from pymonntorch import Behavior

# TODO init, doc, test


class ActivityBaseHomeostasis(Behavior):
    def initialize(self, neurons):
        activity_rate = self.parameter("activity_rate", 5)
        self.window_size = self.parameter("window_size", 50)
        self.updating_rate = self.parameter("updating_rate", 8)
        self.decay_rate = self.parameter("decay_rate", 0.9)

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
    def initialize(self, neurons):
        target_act = self.parameter("target_voltage", 0.05)
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
