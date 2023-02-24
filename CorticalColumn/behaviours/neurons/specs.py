"""
General specifications needed for spiking neurons.
"""

from pymonntorch import Behavior
import torch

# TODO inhibition of KWTA, how should it be???
# TODO adaptive neuorns will KWTA, What behaviour should I expect???

class SpikeTrace(Behavior):
    """
    calculates the spike trace.

    Note : should be placed after Fire behavior.

    Args:
        tau_s (float): ddcay term for spike trace
    """
    def set_variables(self, neurons):
        self.tau_s = self.get_init_attr('tau_s', None)
        neurons.trace = neurons.get_neuron_vec(0.0)

    def forward(self, neurons):
        neurons.trace -= neurons.trace / self.tau_s
        neurons.trace += neurons.spike 

class NeuronAxon(Behavior):
    """
    Propogate the spikes. and applys the delay mechanism.

    Note: should be added after fire and trace behavior.

    Args:
        max_delay (int): delay of all dendrit connected neurons.
        have_trace (boolean): wether to calculate trace or not. default False. 
    """
    def set_variables(self, neurons):
        self.max_delay = self.get_init_attr('max_delay',None)
        self.have_trace = self.get_init_attr('have_trace', False)

        if self.max_delay is not None:
            self.spike_history = neurons.get_neuron_vec_buffer(self.max_delay)
            if self.have_trace:
                self.trace_history = neurons.get_neuron_vec_buffer(self.max_delay)

        neurons.axon = self

    def get_spike(self, neurons, delay=None):
        if self.max_delay  is not None:
            return self.spike_history[delay]
        else:
            return neurons.spikes

    def get_spike_trace(self, neuorns, delay=None):
        if self.max_delay  is not None:
            return self.trace_history[delay]
        else:
            return neurons.trace
    
    def forward(self, neurons):
        if self.max_delay:
            self.spike_history.buffer_roll(neurons.spike)
            if have_trace:
                self.trace_history.buffer_roll(neurons.trace)
                # TODO should trace decay as it propagate throught the axon 

class NeuronDendrite(Behavior): # TODO seperation
    """
    Sums the different kind of dendrite entering the neurons.

    Args:
        Behavior (_type_): _description_
    """
    def set_variables(self, neurons):
        self.apical_provocativeness = self.get_init_attr('apical_provocativeness', None)
        self.distal_provocativeness = self.get_init_attr('distal_provocativeness', None)
        self.I_tau = self.get_init_attr('I_tau', None)

        neurons.I = 0
        neurons.apical_input = 0
        neurons.distal_input = 0
        neurons.proximal_input = 0
    
    def _calc_ratio(self, neurons, provocativeness):
        provocative_limite = neurons.v_rest + provocativeness * (neurons.threshold - neurons.v_rest)
        dv = torch.clip(provocative_limite - neurons.v, min=0)
        return dv

    def forward(self, neurons):
        if self.I_tau is not None:
            neurons.I -= neurons.I / self.I_tau
        
        neurons.I += neurons.proximal_input

        non_priming = neurons.get_neuron_vec(0.0)
        if self.apical_provocativeness is not None:
            non_priming += torch.Tanh(neurons.apical_input) * self._calc_ratio(neurons, self.apical_provocativeness)
        if self.distal_provocativeness is not None:
            non_priming += torch.Tanh(neurons.distal_input) * self._calc_ratio(neurons, self.distal_provocativeness)
        
        neurons.I += non_priming/neurons.R # TODO what to do ? (* tau)


        neurons.apical_input = 0
        neurons.distal_input = 0
        neurons.proximal_input = 0


class Fire(Behavior):
    """
    Asks neurons to Fire.
    """
    def forward(self, neurons):
        neurons.Fire(neurons)

class KWTA(Behavior):
    """
    KWTA behavior of spiking neurons:

    if v >= threshold then v = v_reset and all other spiked neurons are inhibited.

    Note: Population should be built by NeuronDimension.
    and firing behavior should be added too.

    Args:
        k (int): number of winners.
        dimension (int, optional): K-WTA on specific dimension. defaults to None.
    """

    def set_variables(self, neurons):
        self.k = self.get_init_attr("k", None)
        self.dimension = self.get_init_attr('dimension', None)
        self.shape = (neurons.depth, neurons.height, neurons.width)

    def forward(self, neurons):
        will_spike = (neurons.v >= neurons.threshold)

        will_spike_v = (will_spike * (neurons.v - neurons.threshold))

        if self.dimension:
            will_spike_v = will_spike_v.reshape(self.shape)
            will_spike = will_spike.reshape(self.shape)
        else:
            self.dimension = 0
        
        if (will_spike.sum(axis=self.dimension) <= self.k).all():
            return

        k_values, k_winners_indices = torch.topk(will_spike_v, self.k+1, dim=self.dimension, sorted=False)
        min_values = k_values.min(dim = 0).values
        winners = will_spike_v > min_values.expand(will_spike_v.size())
        ignored = will_spike * (~winners)

        neurons.v[ignored.reshape((-1,))] = neurons.v_reset