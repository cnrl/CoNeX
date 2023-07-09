import torch
from pymonntorch import Behavior

class Voltage_base_Homeostasis(Behavior):

    def initialize(self, neurons):
        self.add_tag('Voltage_base_Homeostasis')

        target_act = self.parameter('target_voltage', 0.05, neurons)
        self.max_ta = self.parameter('max_ta', target_act, neurons)
        self.min_ta = self.parameter('min_ta', target_act, neurons)
        self.adj_strength = -self.parameter('eta_ip', 0.001, neurons)

        neurons.exhaustion = neurons.vector()

    def forward(self, neurons):
        greater = ((neurons.v > self.max_ta) * -1).type(torch.float32)
        smaller = ((neurons.v < self.min_ta) * 1).type(torch.float32)

        greater *= neurons.v - self.max_ta
        smaller *= self.min_ta - neurons.v

        change = (greater + smaller) * self.adj_strength
        neurons.exhaustion += change
        neurons.v -= neurons.exhaustion