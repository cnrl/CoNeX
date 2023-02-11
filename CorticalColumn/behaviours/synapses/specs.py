"""
Synapse-related behaviors.
"""

from pymonntorch import Behavior
import torch

# TODO stupid indexing for delay.
# TODO dendrite Delay is wrong.


class SynapseInit(Behavior):
    def set_variables(self, synapse):
        synapse.src_shape = (synapse.src.depht, synapse.src.height, synapse.src.width)
        synapse.dst_shape = (synapse.dst.depht, synapse.dst.height, synapse.dst.width)

        synapse.src_delay = None
        synapse.dst_delay = None

class WeightInitializer(Behavior):
    """
    Intialize the weights of synapse.

    Args:
        mode (str or number): string should be torch functions that fills a tensor like:
                              "random", "normal", "zeros", "ones", ... .
                              In number case the synapse weights will be filled with that number.
        weights (tensor): giving the weights directly.
    """

    def set_variables(self, synapse):
        init_mode = self.get_init_attr("mode", None)
        synapse.weights = self.get_init_attr('weights', None)

        if init_mode is not None and synapse.weights is None:
            synapse.weights = synapse.get_synapse_mat(mode=init_mode)

class DelayInitializer(Behavior):
    """
    Intialize the delay of axon entering the synapse or the dendrite conveying current.

    delays (Tensor(int), optional): a tensor of delay for each neuron of source or destination.
    
    Args:
        mode (str or number): string should be torch functions that fills a tensor like:
                              "random", "normal", "zeros", "ones", ... .
                              In number case the synapse delays will be filled with that number.
        offset (int): delay added to the all delays.
        scale (int): scales delay.
        weights (tensor): giving the delays directly.
        destination (boolean): True for destination neurons. defaults to False
    """
    def set_variables(self, synapse):
        """
        Makes index for the Synapse delay.

        Args:
            synapses (SynapseGroup): The synapses whose weight should be bound.
        """
        init_mode = self.get_init_attr("mode", None)
        delays = self.get_init_attr('delays', None)
        scale = self.get_init_attr('scale', 1)
        offset = self.get_init_attr('offset', 0)
        isDestination = self.get_init_attr('destination', False)

        neruons = synapse.src
        attribute = 'src'
        if isDestination:
            neruons = synapse.dst
            attribute = 'dst'

        if init_mode is not None and synapse.delays is None:
            delays = neruons.get_neuron_vec(mode=init_mode)
            delays *= scale
            delays += offset
        
        delays = delays.to(torch.long)
        synapse.__dict__[f'{attribute}_delay'] = (torch.arange(0, delays.size(0)).to(delays.get_device()), delays)



class WeightClip(Behavior):
    """
    Clip the synaptic weights in a range.

    Args:
        w_min (float): minimum weight constraint.
        w_max (float): maximum weight constraint.
    """

    def set_variables(self, synapse):
        """
        Set weight constraint attributes to the synapses.

        Args:
            synapses (SynapseGroup): The synapses whose weight should be bound.
        """
        self.w_min = self.get_init_attr('w_min', 0)
        self.w_max = self.get_init_attr('w_max', 1)


    def new_iteration(self, synapses):
        """
        Clip the synaptic weights in each time step.

        Args:
            synapses (SynapseGroup): The synapses whose weight should be bound.
        """
        synapse.weights = torch.clip(synapses.weights, self.w_min, self.w_max)