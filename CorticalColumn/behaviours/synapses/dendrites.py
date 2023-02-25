"""
Dendritic behaviors.
"""
from pymonntorch import Behavior

import torch
import torch.nn.functional as F

# TODO not priming neurons with over threshold potential.
# TODO lower than threshold nonPriming
# TODO Priming inhibtory neurons???? by inhibitory neurons
# TODO Conv2d NonPriming

class SimpleDendriticInput(Behavior):
    """
    Base dendrite behavior. It checks for excitatory/inhibitory attributes 
    of pre-synaptic neurons and sets a coefficient, accordingly.

    Note: weights must be intialize by others behaviors.
          Also, Axon paradigm should be added to synapse beforehand.

    Args:
        current_coef (float): scaller coefficient that multiplys weights
    """

    def set_variables(self, synapse):
        """
        Sets the current_coef to -1 if the pre-synaptic neurons are inhibitory.

        Args:
            synapse (SynapseGroup): Synapses on which the dendrites are defined.
        """
        synapse.add_tag(self.__class__.__name__)
        self.current_coef = self.get_init_attr('current_coef', 1)
        self.connection_type = self.get_init_attr('connection_type', 'proximal_input')

        self.add_tag(self.connection_type)
        self.current_type = (
        -1 if ("GABA" in synapse.src.tags) or ("inh" in synapse.src.tags) else 1
        )

    def calculate_input(self, synapse):
        spikes = synapse.src.axon.get_spike(synapse.src, synapse.src_delay)
        return torch.sum(synapse.weights[:, spikes], axis=1)
    
    def forward(self, synapse):
        synapse.dst.__dict__[self.connection_type] += self.current_coef * self.current_type * self.calculate_input(synapse)

class Conv2dDendriteInput(SimpleDendriticInput):
    def set_variables(self, synapse):
        super().set_variables(synapse)
        
        synapse.stride = self.get_init_attr('stride', 1)

    def calculate_input(self, synapse):
        spikes = synapse.src.axon.get_spike(synapse.src, synapse.src_delay)
        spikes = spikes.reshape(synapse.src_shape)
        return F.conv2d(input = spikes.float(), weight = synapse.weights, stride = synapse.stride)

class Local2dDendriteInput(SimpleDendriticInput):

    # weight shape (o_n, p_h, p_w, p_c, w_h, w_w)
    # weight shape (out channel, 
    #               result_height * result_weight,
    #               inpit_channel * kernel_height * kernel_weight)

    def calculate_input(self, synapse):
        spikes = synapse.src.axon.get_spike(synapse.src, synapse.src_delay) # to.float()
        spikes = spikes.reshape(synapse.src_shape)
        spikes = spikes.unfold(kernel_size=synapse.weights.size()[-2:], stride = synapse.stride).transpose(1,2)
        I = (spikes * synapse.weights).sum(axis=-1) 
        return I.reshape((-1,))