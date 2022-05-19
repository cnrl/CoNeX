"""
Implementation of SynapseGroup-related behaviors.
"""

import numpy as np
from PymoNNto import Behaviour


class DendriticInput(Behaviour):
    def set_variables(self, synapses):
        self.set_init_attrs_as_variables(synapses)

        self.current_coef = (
            1 if ("GABA" in synapses.src.tags) or ("inh" in synapses.src.tags) else -1
        )


class ProximalInput(DendriticInput):
    def set_variables(self, synapses):
        self.add_tag("Proximal")
        super().set_variables(synapses)

    def new_iteration(self, synapses):
        synapses.dst.proximal_input_current = self.current_coef * np.sum(
            self.strength, axis=-1
        )


class DistalInput(DendriticInput):
    def set_variables(self, synapses):
        self.add_tag("Distal")
        super().set_variables(synapses)

    def new_iteration(self, synapses):
        return (synapses.dst.v - synapses.dst.v_rest) * np.tanh(
            np.sum(self.strength, axis=-1)
        )


class ApicalInput(DistalInput):
    def set_variables(self, synapses):
        self.add_tag("Apical")
        super().set_variables(synapses)

    def new_iteration(self, synapses):
        update_values = super().new_iteration(synapses)
        synapses.apical_input_current = self.current_coef * update_values


class BasalInput(DistalInput):
    def set_variables(self, synapses):
        self.add_tag("Basal")
        super().set_variables(synapses)

    def new_iteration(self, synapses):
        update_values = super().new_iteration(synapses)
        synapses.basal_input_current = self.current_coef * update_values
