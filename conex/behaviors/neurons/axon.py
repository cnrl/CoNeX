"""
Axon mechanisms for neurons. 
"""

from pymonntorch import Behavior
import torch


class NeuronAxon(Behavior):
    """
    Propagate the spikes and apply the delay mechanism.

    Note: should be added after fire and trace behavior.

    Args:
        max_delay (int): maximum delay of all dendrites connected to the neurons. This value determines the delay buffer size.
        proximal_min_delay (int): minimum delay of proximal dendrites. The default is 0.
        distal_min_delay (int): minimum delay of distal dendrites. The default is 0.
        apical_min_delay (int): minimum delay of apical dendrites. The default is 0.
        have_trace (boolean): wether to calculate trace or not. default False.
    """

    def initialize(self, neurons):
        self.max_delay = self.parameter("max_delay", 1)
        self.proximal_min_delay = self.parameter("proximal_min_delay", 0)
        self.distal_min_delay = self.parameter("distal_min_delay", 0)
        self.apical_min_delay = self.parameter("apical_min_delay", 0)
        self.have_trace = self.parameter("have_trace", hasattr(neurons, "trace"))

        self.spike_history = neurons.vector_buffer(self.max_delay, dtype=torch.bool)
        if self.have_trace:
            self.trace_history = neurons.vector_buffer(self.max_delay)

        neurons.axon = self

    def update_min_delay(self, neurons):
        if proximal_synapses := neurons.efferent_synapses.get("Proximal", []):
            self.proximal_min_delay = torch.cat(
                [synapse.src_delay for synapse in proximal_synapses]
            ).min()
        if distal_synapses := neurons.efferent_synapses.get("Distal", []):
            self.distal_min_delay = torch.cat(
                [synapse.src_delay for synapse in distal_synapses]
            ).min()
        if apical_synapses := neurons.efferent_synapses.get("Apical", []):
            self.apical_min_delay = torch.cat(
                [synapse.src_delay for synapse in apical_synapses]
            ).min()

    def get_spike(self, neurons, delay):
        return self.spike_history.gather(0, delay.unsqueeze(0)).squeeze(0)

    def get_spike_trace(self, neurons, delay):
        return self.trace_history.gather(0, delay.unsqueeze(0)).squeeze(0)

    def forward(self, neurons):
        self.spike_history = neurons.buffer_roll(
            mat=self.spike_history, new=neurons.spikes
        )
        if self.have_trace:
            self.trace_history = neurons.buffer_roll(
                mat=self.trace_history, new=neurons.trace
            )
