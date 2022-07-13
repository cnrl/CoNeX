"""
Synapse-related behaviors.
"""

import numpy as np
from PymoNNto import Behaviour


class WeightInitializer(Behaviour):
    def set_variables(self, synapses):
        init_mode = self.get_init_attr("mode", "random")
        synapses.weights = synapses.get_synapse_mat(mode=init_mode)


class DelayInitializer(Behaviour):
    def set_variables(self, synapses):
        init_mode = self.get_init_attr("mode", "random")
        synapses.delays = synapses.get_synapse_mat(mode=init_mode)


class WeightClip(Behaviour):
    """
    Clip the synaptic weights in a range.

    Args:
        w_min (float): minimum weight constraint.
        w_max (float): maximum weight constraint.
    """

    def set_variables(self, synapses):
        """
        Set weight constraint attributes to the synapses.

        Args:
            synapses (SynapseGroup): The synapses whose weight should be bound.
        """
        self.set_init_attrs_as_variables(synapses)

    def new_iteration(self, synapses):
        """
        Clip the synaptic weights in each time step.

        Args:
            synapses (SynapseGroup): The synapses whose weight should be bound.
        """
        synapses.weights = np.clip(synapses.weights, synapses.w_min, synapses.w_max)
