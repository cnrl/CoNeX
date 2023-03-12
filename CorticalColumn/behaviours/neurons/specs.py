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
        neurons.trace += neurons.spikes

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
        self.have_trace = self.get_init_attr('have_trace', hasattr(neurons, 'trace'))

        if self.max_delay is not None:
            self.spike_history = neurons.get_neuron_vec_buffer(self.max_delay)
            if self.have_trace:
                self.trace_history = neurons.get_neuron_vec_buffer(self.max_delay)

        neurons.axon = self

    def get_spike(self, neurons, delay=None):
        if self.max_delay is not None:
            if delay is not None:
                return self.spike_history[delay]
            else:
                return self.spike_history[0]
        else:
            return neurons.spikes

    def get_spike_trace(self, neuorns, delay=None):
        if self.max_delay is not None:
            if delay is not None:
                return self.trace_history[delay]
            else:
                return self.trace_history[0]
        else:
            return neuorns.trace
    
    def forward(self, neurons):
        if self.max_delay:
            self.spike_history = neurons.buffer_roll(mat=self.spike_history, new=neurons.spikes)
            if self.have_trace:
                self.trace_history = neurons.buffer_roll(mat=self.trace_history, new=neurons.trace)
                # TODO should trace decay as it propagate throught the axon? 
                

class NeuronDendrite(Behavior): # TODO seperation
    """
    Sums the different kind of dendrite entering the neurons.

    Args:
        Behavior (_type_): _description_
    """
    def set_variables(self, neurons):
        self.apical_provocativeness = self.get_init_attr('apical_provocativeness', None)
        self.distal_provocativeness = self.get_init_attr('distal_provocativeness', None)
        self.proximal_max_delay = self.get_init_attr('Proximal_max_delay', None)
        self.apical_max_delay = self.get_init_attr('Apical_max_delay', None)
        self.distal_max_delay = self.get_init_attr('Distal_max_delay', None)
        self.I_tau = self.get_init_attr('I_tau', None)

        neurons.I = neurons.get_neuron_vec()

        neurons.apical_input = 0
        if self.apical_provocativeness is not None:
            if self.apical_max_delay is not None:
                neurons.apical_input = neurons.get_neuron_vec_buffer(self.apical_max_delay)
            else:
                neurons.apical_input = neurons.get_neuron_vec()

        neurons.distal_input = 0
        if self.distal_provocativeness is not None:
            if self.distal_max_delay is not None:
                neurons.distal_input = neurons.get_neuron_vec_buffer(self.distal_max_delay)
            else:
                neurons.distal_input = neurons.get_neuron_vec() 
            
        neurons.proximal_input = neurons.get_neuron_vec()
        if self.proximal_max_delay is not None:
            neurons.proximal_input = neurons.get_neuron_vec_buffer(self.proximal_max_delay)
    
    def _calc_ratio(self, neurons, provocativeness):
        provocative_limite = neurons.v_rest + provocativeness * (neurons.threshold - neurons.v_rest)
        dv = torch.clip(provocative_limite - neurons.v, min=0)
        return dv

    def _add_proximal(self, neurons, synapse):
        if self.proximal_max_delay is not None:
            if synapse.dst_delay is not None:
                neurons.proximal_input[synapse.dst_delay] += synapse.I
            else:
                neurons.proximal_input[0] += synapse.I
        else:
            neurons.proximal_input += synapse.I
    
    def _add_apical(self, neurons, synapse):
        if self.apical_max_delay is not None:
            if synapse.dst_delay is not None:
                neurons.apical_input[synapse.ds_tdelay] += synapse.I
            else:
                neurons.apical_input[0] += synapse.I
        else:
            neurons.apical_input += synapse.I

    def _add_distal(self, neurons, synapse):
        if self.distal_max_delay is not None:
            if synapse.dst_delay is not None:
                neurons.distal_input[synapse.dst_delay] += synapse.I
            else:
                neurons.distal_input[0] += synapse.I
        else:
            neurons.distal_input += synapse.I

    def forward(self, neurons):
        if self.I_tau is not None:
            neurons.I -= neurons.I / self.I_tau
        
        for synapse in neurons.afferent_synapses.get('Proximal', []):
            self._add_proximal(neurons, synapse)
        for synapse in neurons.afferent_synapses.get('Distal', []):
            self._add_distal(neurons, synapse)
        for synapse in neurons.afferent_synapses.get('Apical', []):
            self._add_apical(neurons, synapse)
        
        neurons.I += neurons.proximal_input[0] if self.proximal_max_delay is not None else neurons.proximal_input
        apical_input = neurons.apical_input[0] if self.apical_max_delay is not None else neurons.apical_input
        distal_input = neurons.distal_input[0] if self.distal_max_delay is not None else neurons.distal_input

        non_priming = neurons.get_neuron_vec(0.0)
        if self.apical_provocativeness is not None:
            non_priming += torch.Tanh(apical_input) * self._calc_ratio(neurons, self.apical_provocativeness)
        if self.distal_provocativeness is not None:
            non_priming += torch.Tanh(distal_input) * self._calc_ratio(neurons, self.distal_provocativeness)
        
        neurons.I += non_priming/neurons.R # TODO what to do ? (* tau)

        if self.apical_max_delay is not None:
            neurons.apical_input = neurons.buffer_roll(mat=neurons.apical_input, new=0, counter=True)
        elif neurons.apical_input != 0:
            neurons.apical_input.zero_()
        if self.distal_max_delay is not None:
            neurons.distal_input = neurons.buffer_roll(mat=neurons.distal_input, new=0, counter=True)
        elif neurons.distal_input != 0:
            neurons.distal_input.zero_()
        if self.proximal_max_delay is not None:
            neurons.proximal_input = neurons.buffer_roll(mat=neurons.proximal_input, new=0, counter=True)
        else:
            neurons.proximal_input.zero_()

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