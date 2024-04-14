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
        self.layer = neurons.parent_structure
        neurons.spikes = neurons.vector(dtype=torch.bool)

    def forward(self, neurons):
        if self.layer.x is not None:
            neurons.spikes = self.layer.x
        else:
            neurons.spikes = neurons.vector("zeros", dtype=torch.bool)


class LocationSetter(Behavior):
    """
    Gets location from Layer Object and sets as spike for the population.
    """

    def initialize(self, neurons):
        self.layer = neurons.parent_structure
        neurons.spikes = neurons.vector(dtype=torch.bool)

    def forward(self, neurons):
        if self.layer.loc is not None:
            neurons.spikes = self.layer.loc
        else:
            neurons.spikes = neurons.vector("zeros", dtype=torch.bool)
