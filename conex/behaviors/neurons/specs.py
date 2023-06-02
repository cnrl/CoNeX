"""
General specifications needed for spiking neurons.
"""

from pymonntorch import Behavior
import torch


class InherentNoise(Behavior):
    """
    Applies noisy voltage to neurons in the population.

    Args:
        mode (str): Mode to be used in initialize the tensor. Accepts similar values to Pymonntorch's `tensor` function. Defaults to "rand".
        scale (float): Scale factor to multiply to the tensor. Default is 1.0.
        offset (function): An offset value to be added to the tensor. Default is 0.0.
    """

    def initialize(self, neurons):
        self.mode = self.parameter("mode", "rand")
        self.scale = self.parameter("scale", 1)
        self.offset = self.parameter("offset", 0)

    def forward(self, neurons):
        neurons.v += neurons.vector(mode=self.mode, scale=self.scale) + self.offset


class SpikeTrace(Behavior):
    """
    Calculates the spike trace.

    Note : should be placed after Fire behavior.

    Args:
        tau_s (float): decay term for spike trace. The default is None.
    """

    def initialize(self, neurons):
        """
        Sets the trace attribute for the neural population.
        """
        self.tau_s = self.parameter("tau_s", None, required=True)
        neurons.trace = neurons.vector(0.0)

    def forward(self, neurons):
        """
        Calculates the spike trace of each neuron by adding current spike and decaying the trace so far.
        """
        neurons.trace += neurons.spikes
        neurons.trace -= (neurons.trace / self.tau_s) * neurons.network.dt


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


class NeuronDendrite(Behavior):
    """
    Sums the different kind of dendrite entering the neurons.

    Args:
        apical_provocativeness (float): the strength of the apical dendrites. The default is None.
        distal_provocativeness (float): the strength of the distal dendrites. The default is None.
        proximal_max_delay (int): maximum delay of proximal dendrites. The default is 1.
        distal_max_delay (int): maximum delay of distal dendrites. The default is 1.
        apical_max_delay (int): maximum delay of distal dendrites. The default is `distal_max_delay + 1`.
        proximal_min_delay (int): minimum delay of proximal dendrites. The default is 0.
        distal_min_delay (int): minimum delay of distal dendrites. The default is 0.
        apical_min_delay (int): minimum delay of apical dendrites. The default is `distal_min_delay + 1`.
    """

    def initialize(self, neurons):
        self.apical_provocativeness = self.parameter("apical_provocativeness", None)
        self.distal_provocativeness = self.parameter("distal_provocativeness", None)
        self.proximal_max_delay = self.parameter("Proximal_max_delay", 1)
        self.distal_max_delay = self.parameter("Distal_max_delay", 1)
        self.apical_max_delay = self.parameter(
            "Apical_max_delay", self.distal_max_delay + 1
        )
        self.proximal_min_delay = self.parameter("proximal_min_delay", 0)
        if self.proximal_min_delay >= self.proximal_max_delay:
            raise ValueError(
                "proximal_min_delay should be smaller than proximal_max_delay"
            )
        self.distal_min_delay = self.parameter("distal_min_delay", 0)
        if self.distal_min_delay >= self.distal_max_delay:
            raise ValueError("distal_min_delay should be smaller than distal_max_delay")
        self.apical_min_delay = self.parameter(
            "apical_min_delay", self.distal_min_delay + 1
        )
        if self.apical_min_delay >= self.apical_max_delay:
            raise ValueError("apical_min_delay should be smaller than apical_max_delay")
        self.I_tau = self.parameter("I_tau", None)

        neurons.I = neurons.vector()

        neurons.apical_input = [0]
        if self.apical_provocativeness is not None:
            neurons.apical_input = neurons.vector_buffer(self.apical_max_delay)

        neurons.distal_input = [0]
        if self.distal_provocativeness is not None:
            neurons.distal_input = neurons.vector_buffer(self.distal_max_delay)

        neurons.proximal_input = neurons.vector_buffer(self.proximal_max_delay)

    def update_min_delay(self, neurons):
        if proximal_synapses := neurons.afferent_synapses.get("Proximal", []):
            self.proximal_min_delay = torch.cat(
                [synapse.dst_delay for synapse in proximal_synapses]
            ).min()
        if distal_synapses := neurons.afferent_synapses.get("Distal", []):
            self.distal_min_delay = torch.cat(
                [synapse.dst_delay for synapse in distal_synapses]
            ).min()
        if apical_synapses := neurons.afferent_synapses.get("Apical", []):
            self.apical_min_delay = torch.cat(
                [synapse.dst_delay for synapse in apical_synapses]
            ).min()

    def _calc_ratio(self, neurons, provocativeness):
        provocative_limit = neurons.v_rest + provocativeness * (
            neurons.threshold - neurons.v_rest
        )
        dv = torch.clip(provocative_limit - neurons.v, min=0)
        return dv

    def _add_proximal(self, neurons, synapse):
        neurons.proximal_input.scatter_add_(
            0, synapse.dst_delay.unsqueeze(0), synapse.I.unsqueeze(0)
        )

    def _add_apical(self, neurons, synapse):
        neurons.apical_input.scatter_add_(
            0, synapse.dst_delay.unsqueeze(0), synapse.I.unsqueeze(0)
        )

    def _add_distal(self, neurons, synapse):
        neurons.distal_input.scatter_add_(
            0, synapse.dst_delay.unsqueeze(0), synapse.I.unsqueeze(0)
        )

    def forward(self, neurons):
        if self.I_tau is not None:
            neurons.I -= neurons.I / self.I_tau
        else:
            neurons.I.fill_(0)

        for synapse in neurons.afferent_synapses.get("Proximal", []):
            self._add_proximal(neurons, synapse)
        for synapse in neurons.afferent_synapses.get("Distal", []):
            self._add_distal(neurons, synapse)
        for synapse in neurons.afferent_synapses.get("Apical", []):
            self._add_apical(neurons, synapse)

        neurons.I += neurons.proximal_input[0]
        apical_input = neurons.apical_input[0]
        distal_input = neurons.distal_input[0]

        non_priming_apical = (
            (
                torch.tanh(apical_input)
                * self._calc_ratio(neurons, self.apical_provocativeness)
            )
            if self.apical_provocativeness is not None
            else 0
        )
        non_priming_distal = (
            (
                torch.tanh(distal_input)
                * self._calc_ratio(neurons, self.distal_provocativeness)
            )
            if self.distal_provocativeness is not None
            else 0
        )

        neurons.I += (non_priming_apical + non_priming_distal) / getattr(
            neurons, "R", 1
        )

        if self.apical_provocativeness is not None:
            neurons.apical_input = neurons.buffer_roll(
                mat=neurons.apical_input, new=0, counter=True
            )

        if self.distal_provocativeness is not None:
            neurons.distal_input = neurons.buffer_roll(
                mat=neurons.distal_input, new=0, counter=True
            )

        neurons.proximal_input = neurons.buffer_roll(
            mat=neurons.proximal_input, new=0, counter=True
        )


class Fire(Behavior):
    """
    Asks neurons to Fire.
    """

    def forward(self, neurons):
        neurons.spiking_neuron.Fire(neurons)


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

    def initialize(self, neurons):
        self.k = self.parameter("k", None, required=True)
        self.dimension = self.parameter("dimension", None)
        self.shape = (1, 1, neurons.size)
        if hasattr(neurons, "depth"):
            self.shape = (neurons.depth, neurons.height, neurons.width)

    def forward(self, neurons):
        will_spike = neurons.v >= neurons.threshold
        will_spike_v = will_spike * (neurons.v - neurons.threshold)

        dim = 0
        if self.dimension is not None:
            will_spike_v = will_spike_v.view(self.shape)
            will_spike = will_spike.view(self.shape)
            dim = self.dimension

        if (will_spike.sum(axis=dim) <= self.k).all():
            return

        k_values, k_winners_indices = torch.topk(
            will_spike_v, self.k, dim=dim, sorted=False
        )
        min_values = k_values.min(dim=0).values
        winners = will_spike_v >= min_values.expand(will_spike_v.size())
        ignored = will_spike * (~winners)

        neurons.v[ignored.view((-1,))] = neurons.v_reset
