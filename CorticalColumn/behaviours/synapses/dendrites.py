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
# TODO current delay 


class SimpleDendriticInput(Behavior):
    """
    Base dendrite behavior. It checks for excitatory/inhibitory attributes 
    of pre-synaptic neurons and sets a coefficient, accordingly.

    Note: weights must be intialize by others behaviors.
          Also, Axon paradigm should be added to the neurons.
          Connection type (Proximal, Distal, Apical) should be specified by the tag 
          of the synapse. and Dendrite behavior of the neurons group should access the 
          `I` of each synapse to apply them.

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

        self.current_type = (
        -1 if ("GABA" in synapse.src.tags) or ("inh" in synapse.src.tags) else 1
        )

        synapse.I = synapse.dst.get_neuron_vec(0)

    def calculate_input(self, synapse):
        spikes = synapse.src.axon.get_spike(synapse.src, synapse.src_delay)
        return torch.sum(synapse.weights[:, spikes], axis=1)
    
    def forward(self, synapse):
        synapse.I = self.current_coef * self.current_type * self.calculate_input(synapse)


class Conv2dDendriteInput(SimpleDendriticInput):
    """
    Weight shape = (out_channel, in_channel, kernel_height, kernel_width)
    """


    def set_variables(self, synapse):
        super().set_variables(synapse)
        
        synapse.stride = self.get_init_attr('stride', 1)
        synapse.padding = self.get_init_attr('padding', 0)

    def calculate_input(self, synapse):
        spikes = synapse.src.axon.get_spike(synapse.src, synapse.src_delay).to(torch.float32)
        spikes = spikes.reshape(synapse.src_shape)

        I = F.conv2d(input = spikes, weight = synapse.weights, stride = synapse.stride, padding = synapse.padding)
        
        # Alternative code that may have efficiency advantage
        # 
        # unfold_spikes = F.unfold(input=spikes, kernel_size=synapse.weights.shape[-2:], stride = synapse.stride, padding = synapse.padding)
        # I = (unfold_spikes.T.matmul(synapse.weights.view(synapse.weights.size(0), -1).T)).T
        
        return I.reshape((-1,))


class Local2dDendriteInput(Conv2dDendriteInput):
    """
    Weight shape = (out_channel, out_size, connection_size)
                    out_size = out_height * out_width, 
                    connection_size = input_channel * connection_height * connection_width
    """

    def calculate_input(self, synapse):
        spikes = synapse.src.axon.get_spike(synapse.src, synapse.src_delay).to(torch.float32)
        spikes = spikes.reshape(synapse.src_shape)
        spikes = F.unfold(spikes, kernel_size=synapse.kernel_shape[-2:], stride = synapse.stride, padding=synapse.padding).T
        I = (spikes * synapse.weights).sum(axis=-1) 
        return I.reshape((-1,))