"""
Dendritic behaviors.
"""

import numpy as np
from PymoNNto import Behaviour


class DendriticInput(Behaviour):
    """
    Base dendrite behavior. It checks for excitatory/inhibitory attributes of
    pre-synaptic neurons and sets a coefficient, accordingly.
    """

    def set_variables(self, synapses):
        """
        Sets the current_coef to -1 if the pre-synaptic neurons are inhibitory.

        Args:
            synapses (SynapseGroup): Synapses on which the dendrites are defined.
        """
        self.set_init_attrs_as_variables(synapses)

        self.current_coef = (
            -1 if ("GABA" in synapses.src.tags) or ("inh" in synapses.src.tags) else 1
        )

    def new_iteration(self, synapses):
        if "history" in synapses.src.tags:
            return synapses.src.spike_history[:, synapses.delays]
        else:
            return synapses.src.spikes


class ProximalInput(DendriticInput):
    """
    Proximal dendrite behavior. It basically calculates a current accumulation based
    on synaptic weights.
    """

    def set_variables(self, synapses):
        """
        Sets the current_coef to -1 if the pre-synaptic neurons are inhibitory.

        Args:
            synapses (SynapseGroup): Synapses on which the dendrites are defined.
        """
        self.add_tag("Proximal")
        super().set_variables(synapses)

    def new_iteration(self, synapses):
        """
        Calculates the proximal input current by a summation over synaptic weights,
        weighted by current_coef.

        Args:
            synapses (SynapseGroup): Synapses on which the dendrites are defined.
        """
        effective_spikes = super().new_iteration(synapses)
        synapses.dst.proximal_input_current = self.current_coef * np.sum(
            self.weights * effective_spikes, axis=-1
        )


class DistalInput(DendriticInput):
    """
    Distal dendrite behavior. These dendrites increase the input current to the
    post-synaptic neurons by:

    (dst.v - dst.v_rest) * tanh(sum(weights*src.spikes)).
    """

    def set_variables(self, synapses):
        """
        Sets the current_coef to -1 if the pre-synaptic neurons are inhibitory.

        Args:
            synapses (SynapseGroup): Synapses on which the dendrites are defined.
        """
        self.add_tag("Distal")
        super().set_variables(synapses)

    def new_iteration(self, synapses):
        """
        Returns the current value to be added to the post-synaptic neurons.

        Args:
            synapses (SynapseGroup): Synapses on which the dendrites are defined.
        """
        effective_spikes = super().new_iteration(synapses)
        return (synapses.dst.v - synapses.dst.v_rest) * np.tanh(
            np.sum(self.weights * effective_spikes, axis=-1)
        )


class ApicalInput(DistalInput):
    """
    Apical dendrite behavior. These dendrites increase the input current to the
    post-synaptic neurons by:

    (dst.v - dst.v_rest) * tanh(sum(weights*src.spikes)).
    """

    def set_variables(self, synapses):
        """
        Sets the current_coef to -1 if the pre-synaptic neurons are inhibitory.

        Args:
            synapses (SynapseGroup): Synapses on which the dendrites are defined.
        """
        self.add_tag("Apical")
        super().set_variables(synapses)

    def new_iteration(self, synapses):
        """
        Sets the apical input current.

        Args:
            synapses (SynapseGroup): Synapses on which the dendrites are defined.
        """
        update_values = super().new_iteration(synapses)
        synapses.apical_input_current = self.current_coef * update_values


class BasalInput(DistalInput):
    """
    Basal dendrite behavior. These dendrites increase the input current to the
    post-synaptic neurons by:

    (dst.v - dst.v_rest) * tanh(sum(weights*src.spikes)).
    """

    def set_variables(self, synapses):
        """
        Sets the current_coef to -1 if the pre-synaptic neurons are inhibitory.

        Args:
            synapses (SynapseGroup): Synapses on which the dendrites are defined.
        """
        self.add_tag("Basal")
        super().set_variables(synapses)

    def new_iteration(self, synapses):
        """
        Sets the basal input current.

        Args:
            synapses (SynapseGroup): Synapses on which the dendrites are defined.
        """
        update_values = super().new_iteration(synapses)
        synapses.basal_input_current = self.current_coef * update_values
