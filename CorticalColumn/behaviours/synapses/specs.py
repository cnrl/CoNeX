"""
Synapse-related behaviors.
"""

from pymonntorch import Behavior
import torch

# TODO stupid indexing for delay.
# TODO dendrite Delay is wrong.


class SynapseInit(Behavior):
    """
    This Behavior makes initial variable required for multiple behavior to use.
    WARNING: ``src_delay`` and ``dst_delay`` have equal delay for all their neurons.
    And should be initialized by other behaviors.
    """

    def set_variables(self, synapse):
        synapse.src_shape = (synapse.src.depth, synapse.src.height, synapse.src.width)
        synapse.dst_shape = (synapse.dst.depth, synapse.dst.height, synapse.dst.width)

        # TODO maybe this should be done with a NeuronInit Behavior?
        if not hasattr(synapse.src, 'index_vector'):
            synapse.src.index_vector = torch.arange(0, synapse.src.size).to(synapse.network.device)
        if not hasattr(synapse.dst, 'index_vector'):
            synapse.dst.index_vector = torch.arange(0, synapse.dst.size).to(synapse.network.device)

        synapse.src_delay = (torch.tensor(0).expand(synapse.src.size).to(synapse.network.device),
                             synapse.src.index_vector)
        synapse.dst_delay = (torch.tensor(0).expand(synapse.dst.size).to(synapse.network.device),
                             synapse.dst.index_vector)


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

        neurons = synapse.src
        attribute = 'src'
        if isDestination:
            neurons = synapse.dst
            attribute = 'dst'

        if init_mode is not None and delays is None:
            delays = neurons.get_neuron_vec(mode=init_mode)
            delays *= scale
            delays += offset
        
        delays = delays.to(torch.long)
        synapse.__dict__[f'{attribute}_delay'] = (delays, torch.arange(0, delays.size(0)).to(delays.device))


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
        synapse.weight_shape = self.get_init_attr('weight_shape', None)
        synapse.kernel_shape = self.get_init_attr('kernel_shape', None)

        if init_mode is not None and synapse.weights is None:
            if synapse.weight_shape is None:
                synapse.weights = synapse.get_synapse_mat(mode=init_mode)
            else:
                synapse.weights = synapse._get_mat(mode=init_mode, dim=synapse.weight_shape)


class WeightNormalization(Behavior):
    """
    This Behavior normalize weights in order to assure each destinatin neuron has
    sum of its weight equal to ``norm``. Supporting `Simple`, `Local2d`, 'Conv2d'.

    Args:
        norm (int): Desired sum of weights for each neuron.
    """
    def set_variables(self, syanpse):
        self.norm = self.get_init_attr('norm', 1)
        self.dims = [x for x in range(1, len(syanpse.weights.shape))]
        if len(syanpse.weights.shape) == 2:
            self.dims = [2]

    def forward(self, synapse):
        weights_sum = synapse.weights.sum(dim=self.dims, keepdim=True)
        weights_sum[weights_sum == 0] = 1
        synapse.weights *= self.norm / weights_sum


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

        assert self.w_min >= 0 and self.w_max < self.w_min, 'Invalid Interval for Weight Clip' 

    def forward(self, synapses):
        """
        Clip the synaptic weights in each time step.

        Args:
            synapses (SynapseGroup): The synapses whose weight should be bound.
        """
        synapses.weights = torch.clip(synapses.weights, self.w_min, self.w_max)
