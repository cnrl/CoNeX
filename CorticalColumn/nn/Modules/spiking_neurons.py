"""
Structure of spiking neural populations.
"""

from CorticalColumn.behaviours.neurons.lif_neurons import LIF
from CorticalColumn.behaviours.neurons import (
    KWTA,
    Fire,
    InherentNoise,
    NeuronAxon,
    NeuronDendrite,
    SpikeTrace,
)

from CorticalColumn.nn.timestamps import NEURON_TIMESTAMPS

from pymonntorch import NeuronGroup


class SpikingNeuronGroup(NeuronGroup):
    def __init__(
        self,
        size,
        net,
        neuron_type=LIF,
        kwta=None,
        kwta_dim=None,
        tau_trace=None,
        max_delay=1,
        noise_function=None,
        tag=None,
        # color=None,
        dendrite_params={},
        neuron_params={},
    ):
        if tag is None and net is not None:
            tag = "SpikingNeuronGroup_" + str(len(net.NeuronGroups) + 1)

        behavior = {
            NEURON_TIMESTAMPS['NeuronDendrite']: NeuronDendrite(**dendrite_params),
            NEURON_TIMESTAMPS['NeuronDynamic']: neuron_type(**neuron_params),
            NEURON_TIMESTAMPS['Fire']: Fire(),
            NEURON_TIMESTAMPS['NeuronAxon']: NeuronAxon(max_delay=max_delay),
        }

        if kwta is not None:
            behavior[NEURON_TIMESTAMPS['KWTA']] = KWTA(k=kwta, dimension=kwta_dim)

        if tau_trace:
            behavior[NEURON_TIMESTAMPS['Trace']] = SpikeTrace(tau_s=tau_trace)

        if noise_function:
            behavior[NEURON_TIMESTAMPS['DirectNoise']] = InherentNoise(noise_function)

        super().__init__(size, behavior, net, tag)


"""
        if not hasattr(self, "v"):
            if hasattr(self, "v_rest"):
                warnings.warn(
                    "Spiking neuron behavior lacks attribute v. Adding the attribute..."
                )
                self.v = self.v_rest * self.get_neuron_vec(mode="ones()")
            else:
                raise AttributeError("Spiking neuron lacks attribute v.")

        if not hasattr(self, "spikes"):
            warnings.warn(
                "Spiking neuron behavior lacks attribute spike. Adding the attribute..."
            )
            if hasattr(self, "init_s"):
                if isinstance(self.init_s, torch.Tensor):
                    self.spikes = self.init_s
            self.spikes = self.get_neuron_vec(mode="zeros()")
"""