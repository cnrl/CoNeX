"""
Helper behaviors to communicate with input and output layers
"""

from pymonntorch import Behavior
import torch


class SensorySetter(Behavior):
    """
    Gets input from Layer Object and sets as spike for the population.
    """

    def initialize(self, neurons):
        self.layer = neurons.layer
        neurons.spikes = neurons.vector(dtype=torch.bool)

    def forward(self, neurons):
        neurons.spikes = self.layer.x


class LocationSetter(Behavior):
    """
    Gets location from Layer Object and sets as spike for the population.
    """

    def initialize(self, neurons):
        self.layer = neurons.layer
        neurons.spikes = neurons.vector(dtype=torch.bool)

    def forward(self, neurons):
        neurons.spikes = self.layer.loc
