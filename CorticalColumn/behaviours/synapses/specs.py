"""
Synapse-related behaviors.
"""

from pymonntorch import Behavior
import torch

# TODO stupid indexing for delay

class CurrentVariables(Behavior):
    """
    Makes a dictionary collecting different current.
    """
    def set_variables(self, synapses):
        synapses.Is = {}

class WeightInitializer(Behavior):
    """
    Intialize the weights of synapse.

    Args:
        mode (str or number): string should be torch functions that fills a tensor like:
                              "random", "normal", "zeros", "ones", ... .
                              In number case the synapse weights will be filled with that number.
        weights (tensor): giving the weights directly.
    """

    def set_variables(self, synapses):
        init_mode = self.get_init_attr("mode", None)
        synapses.weights = self.get_init_attr('weights', None)

        if init_mode is not None and synapses.weights is None:
            synapses.weights = synapses.get_synapse_mat(mode=init_mode)

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
        isDendrite (boolean): True for destination neurons. defaults to False
    """
    def set_variables(self, synapses):
        """
        Makes index for the Synapse delay.

        Args:
            synapses (SynapseGroup): The synapses whose weight should be bound.
        """
        init_mode = self.get_init_attr("mode", None)
        delays = self.get_init_attr('delays', None)
        scale = self.get_init_attr('scale', 1)
        offset = self.get_init_attr('offset', 0)
        isDendrite = self.get_init_attr('isDendrite', False)

        neruons = synapses.src
        attribute = 'axon'
        if isDendrite:
            neruons = synapses.dst
            attribute = 'dendrite'

        if init_mode is not None and synapses.delays is None:
            delays = neruons.get_neuron_vec(mode=init_mode)
            delays *= scale
            delays += offset
        
        delays = delays.to(torch.long)
        synapses.__dict__[f'{attribute}_delays_index'] = (torch.arange(0, delays.size(0)).to(delays.get_device()), delays)

class Axon(Behavior):
    """
    Paradigm to receive spikes.
    """

    def set_variables(self, synapse):
        """
        Makes history for the Synapse provided delay is present.

        Args:
            synapses (SynapseGroup): The synapses whose weight should be bound.
        """
        if synapses.hasattr('axon_delays_index'):
            synapses.axon_spikes = synapses.src.get_neuron_vec_buffer(self.axon_delays_index[1].max()+1).to(torch.bool)

    def new_iteration(self, synapse):
        """
        Recives spikes from axon of pre-synaptic neurons, applys the delay paradigm if existed.

        Args:
            synapses (SynapseGroup): Synapses on which the dendrites are defined.

        Returns:
            (Tensor): spikes entering synapse
        """
        if synapses.hasattr('axon_spikes'):
            synapses.buffer_roll(synapses.axon_spikes, synapses.src.spikes)
            synapse.spikes = synapses.axon_spikes[synapses.axon_delays_index]
        else:
            synapse.spikes = synapses.src.spikes

class DendriteAggregator(Behavior):
    """
    Paradigm to send current to destination neurons.
    Applys delay to dendrit if available.

    Args:
        I_tau (float): time constant current decay. 
    """

    def set_variables(self, synapse):
        """
        Makes history for the Synapse provided delay is present.

        Args:
            synapses (SynapseGroup): The synapses whose weight should be bound.
        """
        self.I_tau = self.get_init_attr('I_tau', None)

        if synapses.hasattr('dendrite_delays_index'):
            synapses.dendrite_current = synapses.dst.get_neuron_vec_buffer(self.dendrite_delays_index[1].max()+1)

    def new_iteration(self, synapse):
        """
        Recives spikes from axon of pre-synaptic neurons, applys the delay paradigm if existed.

        Args:
            synapses (SynapseGroup): Synapses on which the dendrites are defined.

        Returns:
            (Tensor): spikes entering synapse
        """
        sum_I = sum(synapse.Is.values())

        if synapses.hasattr('dendrite_current'):
            synapses.buffer_roll(synapses.dendrite_current, sum_I)
            dI_dt = synapses.dendrite_current[synapses.dendrite_delays_index]
        else:
            dI_dt = sum_I

        if self.I_tau is not None:
            synapse.dst.I -= synapse.dst.I / self.I_tau
        
        synapse.dst.I += sum_I

class WeightClip(Behavior):
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
        synapses.w_min = self.get_init_attr('w_min', 0)
        synapses.w_max = self.get_init_attr('w_max', 1)


    def new_iteration(self, synapses):
        """
        Clip the synaptic weights in each time step.

        Args:
            synapses (SynapseGroup): The synapses whose weight should be bound.
        """
        synapses.weights = torch.clip(synapses.weights, synapses.w_min, synapses.w_max)