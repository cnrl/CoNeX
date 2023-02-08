"""
Dendritic behaviors.
"""
from pymonntorch import Behavior
import torch

# TODO not priming neurons with over threshold potential.
# TODO lower than threshold nonPriming
# TODO Priming inhibtory neurons???? by inhibitory neurons

class DendriticInput(Behavior):
    """
    Base dendrite behavior. It checks for excitatory/inhibitory attributes 
    of pre-synaptic neurons and sets a coefficient, accordingly.

    Note: weights must be intialize by others behaviors.
          Also, Axon paradigm should be added to synapse beforehand.

    Args:
        current_coef (float): scaller coefficient that multiplys weights
    """

    def set_variables(self, synapses):
        """
        Sets the current_coef to -1 if the pre-synaptic neurons are inhibitory.

        Args:
            synapses (SynapseGroup): Synapses on which the dendrites are defined.
        """
        self.add_tag(self.__class__.__name__)

        self.current_coef = synapses.get_init_attr('current_coef', 1)
        self.current_type = (
        -1 if ("GABA" in synapses.src.tags) or ("inh" in synapses.src.tags) else 1
        )


class ProximalInput(DendriticInput):
    """
    Proximal dendrite behavior. It basically calculates a current accumulation based
    on synaptic weights.
    """

    def new_iteration(self, synapses):
        """
        Calculates the proximal input current by a summation over synaptic weights,
        weighted by current_coef.

        Args:
            synapses (SynapseGroup): Synapses on which the dendrites are defined.
        """
        synapses.Is['proximal'] = self.current_type * self.current_coef * torch.sum(
            self.weights[:, synapses.spikes], axis=1
            )


class NonPriming(DendriticInput):
    """
    Base Non Priming dendrite behavior. It checks for excitatory/inhibitory attributes 
    of pre-synaptic neurons and sets a coefficient, accordingly.
    These dendrites increase the input current to the post-synaptic neurons which have 
    membrane potentioal below the threshold by:

    (dst.threshold - dst.v) * tanh(sum(weights*src.spikes)).

    Note: weights must be intialize by others behaviors.
          Also, Axon paradigm should be added to synapse beforehand.

    Args:
        current_coef (float): scaller coefficient that multiplys weights
        provocativeness (float): the percentage these dendrites can stimulate 
    """

    def set_variables(self, synapses):
        """
        Sets the current_coef to -1 if the pre-synaptic neurons are inhibitory.

        Args:
            synapses (SynapseGroup): Synapses on which the dendrites are defined.
        """
        super().set_variables(synapses)
        self.provocativeness = synapses.get_init_attr('provocativeness', None)

    def _increase_potential(self, synapses):
        """
        Returns the current value to be added to the post-synaptic neurons.

        Args:
            synapses (SynapseGroup): Synapses on which the dendrites are defined.

        """
        provocative_limite = synapses.dst.v_rest + self.provocativeness * (synapses.dst.threshold - synapses.dst.v_rest) # TODO v_rest or v_reset ? stupid question anyway.
        dv = torch.clip(provocative_limite - synapses.dst.v, min=0)
        return  dv * torch.Tanh(torch.sum(self.weights[:, synapses.spikes], axis=1))


class ApicalInput(NonPriming):
    """
    Apical dendrite behavior. These dendrites increase the input current to the
    post-synaptic neurons by:

    (dst.threshold - dst.v) * tanh(sum(weights*src.spikes)).

    Note: weights must be intialize by others behaviors.
          Also, Axon paradigm should be added to synapse beforehand.
    """

    def new_iteration(self, synapses):
        """
        Sets the apical input current.

        Args:
            synapses (SynapseGroup): Synapses on which the dendrites are defined.
        """
        update_values = self._increase_potential(synapses)
        synapses.Is['apical'] = self.current_type * self.current_coef * update_values


class DistalInput(NonPriming):
    """
    Distal dendrite behavior. These dendrites increase the input current to the
    post-synaptic neurons by:

    (dst.threshold - dst.v) * tanh(sum(weights*src.spikes)).

    Note: weights must be intialize by others behaviors.
          Also, Axon paradigm should be added to synapse beforehand.
    """

    def new_iteration(self, synapses):
        """
        Sets the basal input current.

        Args:
            synapses (SynapseGroup): Synapses on which the dendrites are defined.
        """
        update_values = self._increase_potential(synapses)
        synapses.Is['basal'] = self.current_type * self.current_coef * update_values