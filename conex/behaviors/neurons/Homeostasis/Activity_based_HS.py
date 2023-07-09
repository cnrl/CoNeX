import torch
from pymonntorch import Behavior
import numpy as np

class Activity_base_Homeostasis(Behavior):

    def initialize(self, neurons):
        self.add_tag('Activity_base_Homeostasis')

        self.window_size = self.parameter("window_size", 50, None)
        self.updating_rate = self.parameter("updating_rate", 8, None)
        self.activity_rate = self.parameter("activity_rate", 5, None)
        self.decay_rate = self.parameter("decay_rate", 0.9, None)
        
        self.activity_rate = np.ceil(self.activity_rate / neurons.size)

        self.non_firing_penalty = -self.activity_rate / (self.window_size - self.activity_rate)

        self.activities = neurons.vector(mode="zeros")
        self.exhaustion = neurons.vector(mode="zeros")

    def forward(self, neurons):
        global TH
        if self.activity_rate * neurons.size <= self.window_size:
            add_activitiies = torch.ones((neurons.spikes.shape))
            add_activitiies[(neurons.spikes != 1).nonzero(as_tuple=True)[0]] = self.non_firing_penalty
            self.activities += add_activitiies

            if (neurons.iteration % self.window_size) == 0:
                # print('act', self.activities)
                change = (-self.activities* self.updating_rate) * np.power(self.decay_rate, (neurons.iteration // self.window_size))
                # print('change', change)
                neurons.threshold -= change
                TH = neurons.threshold
                # print('th', neurons.threshold)
                self.activities *= 0
        else:
            "Error in the process of homeostasis"