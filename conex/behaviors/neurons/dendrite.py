"""
Dendrite structure and computation variants.
"""

from pymonntorch import Behavior
import torch


class SimpleDendriteStructure(Behavior):
    """
    Defines the Structure of the dendrite. Gathers currents for the Computation Behavior.

    Args:
        proximal_max_delay (int): Maximum delay of proximal dendrites. The default is 1. Set this to 0 to discard Proximal dendrite.
        distal_max_delay (int): Maximum delay of distal dendrites. The default is 1. Set this to 0 to discard Distal dendrite.
        apical_max_delay (int): Maximum delay of distal dendrites. The default is `distal_max_delay + 1`. Set this to 0 to discard Apical dendrite.
        proximal_min_delay (int): Minimum delay of proximal dendrites. The default is 0.
        distal_min_delay (int): Minimum delay of distal dendrites. The default is 0.
        apical_min_delay (int): Minimum delay of apical dendrites. The default is `distal_min_delay + 1`.
    """

    def initialize(self, neurons):
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
        if self.distal_min_delay >= self.distal_max_delay and self.distal_max_delay > 0:
            raise ValueError("distal_min_delay should be smaller than distal_max_delay")
        self.apical_min_delay = self.parameter(
            "apical_min_delay", self.distal_min_delay + 1
        )
        if self.apical_min_delay >= self.apical_max_delay and self.apical_max_delay > 0:
            raise ValueError("apical_min_delay should be smaller than apical_max_delay")

        neurons.apical_input = [0]
        if self.apical_max_delay:
            neurons.apical_input = neurons.vector_buffer(self.apical_max_delay)

        neurons.distal_input = [0]
        if self.distal_max_delay:
            neurons.distal_input = neurons.vector_buffer(self.distal_max_delay)

        neurons.proximal_input = [0]
        if self.proximal_max_delay:
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
        if self.apical_max_delay:
            neurons.apical_input = neurons.buffer_roll(
                mat=neurons.apical_input, new=0, counter=True
            )

        if self.distal_max_delay:
            neurons.distal_input = neurons.buffer_roll(
                mat=neurons.distal_input, new=0, counter=True
            )

        if self.proximal_max_delay:
            neurons.proximal_input = neurons.buffer_roll(
                mat=neurons.proximal_input, new=0, counter=True
            )

        for synapse in neurons.afferent_synapses.get("Proximal", []):
            self._add_proximal(neurons, synapse)
        for synapse in neurons.afferent_synapses.get("Distal", []):
            self._add_distal(neurons, synapse)
        for synapse in neurons.afferent_synapses.get("Apical", []):
            self._add_apical(neurons, synapse)

        neurons.I_proximal = neurons.proximal_input[0]
        neurons.I_apical = neurons.apical_input[0]
        neurons.I_distal = neurons.distal_input[0]


class SimpleDendriteComputation(Behavior):
    """
    Sums the different kind of dendrite entering the neurons.

    Args:
        apical_provocativeness (float): The strength of the apical dendrites. The default is None.
        distal_provocativeness (float): The strength of the distal dendrites. The default is None.
        I_tau (float): Decaying factor to current. If None, at each step, current falls to zero.
    """

    def initialize(self, neurons):
        self.apical_provocativeness = self.parameter("apical_provocativeness", None)
        self.distal_provocativeness = self.parameter("distal_provocativeness", None)
        self.I_tau = self.parameter("I_tau", None)

        neurons.I = neurons.vector()

    def _calc_ratio(self, neurons, provocativeness):
        provocative_limit = neurons.v_rest + provocativeness * (
            neurons.threshold - neurons.v_rest
        )
        dv = torch.clip(provocative_limit - neurons.v, min=0)
        return dv

    def forward(self, neurons):
        if self.I_tau is not None:
            neurons.I -= neurons.I / self.I_tau
        else:
            neurons.I.fill_(0)

        non_priming_apical = (
            (
                torch.tanh(neurons.I_apical)
                * self._calc_ratio(neurons, self.apical_provocativeness)
            )
            if self.apical_provocativeness is not None
            else 0
        )
        non_priming_distal = (
            (
                torch.tanh(neurons.I_distal)
                * self._calc_ratio(neurons, self.distal_provocativeness)
            )
            if self.distal_provocativeness is not None
            else 0
        )

        neurons.I += neurons.I_proximal + (
            (non_priming_apical + non_priming_distal) / getattr(neurons, "R", 1)
        )
