"""
Structure of spiking neural populations.
"""

from PymoNNto import NeuronGroup

from CorticalColumn.behaviours import Fire


class SpikingNeuronGroup(NeuronGroup):
    def __init__(
        self,
        size,
        behaviour,
        net,
        tag=None,
        color=None,
        v_rest=0.0,
        v_reset=0.0,
        threshold=10.0,
        has_noise=True,
    ):
        assert 1 not in behaviour, "Behaviour index 1 is reserved for firing behaviour."

        if tag is None and net is not None:
            tag = "SpikingNeuronGroup_" + str(len(net.NeuronGroups) + 1)

        super().__init__(size, behaviour, net, tag, color)

        self.v_rest = v_rest * self.get_neuron_vec(mode="ones()")
        self.v_reset = v_reset * self.get_neuron_vec(mode="ones()")
        self.threshold = threshold * self.get_neuron_vec(mode="ones()")

        self.v = self.v_rest * self.get_neuron_vec(mode="ones()")
        self.spikes = self.get_neuron_vec(mode="zeros()")

        net.add_behaviours_to_object({1: Fire()}, self)

        if has_noise:
            self.noisy_current = self.get_neuron_vec(mode="uniform(0.0, 1.0)")
        else:
            self.noisy_current = None

        self.proximal_input_current = self.get_neuron_vec(mode="zeros()")
        self.basal_input_current = self.get_neuron_vec(mode="zeros()")
        self.apical_input_current = self.get_neuron_vec(mode="zeros()")
