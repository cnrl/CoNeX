"""
Structure of spiking neural populations.
"""

from CCSNN.behaviours.neurons.neuron_types.lif_neurons import LIF
from CCSNN.behaviours.neurons import neuron_types
from CCSNN.behaviours.neurons import (
    KWTA,
    Fire,
    InherentNoise,
    NeuronAxon,
    NeuronDendrite,
    SpikeTrace,
)

from CCSNN.nn.timestamps import NEURON_TIMESTAMPS

from pymonntorch import NeuronGroup


class SpikingNeuronGroup(NeuronGroup):
    """
    Simplifies defining spiking neural populations.

    Args:
        size (int or Behavior): The size or dimension of the population.
        net (Network): The network the population belongs to.
        neuron_type (str or Behavior): Type of neurons which defines the neural dynamics. The default is `LIF`.
        kwta (int):  If not None, enables k-winner-take-all (KWTA) mechanism and specifies the number of winners.
        kwta_dim (tuple): If KWTA is enabled, specifies the dimension of KWTA neighborhood.
        tau_trace (float): If not None, enables the spike trace for neurons and specifies its time constant.
        max_delay (int): Defines the maximum (buffer size of) axonal delay. The default is 1.
        noise_function (function): If not None, enables inherent noise in membrane potential and specifies its function.
        tag (str): The tag(s) of the population. If None, it is set to `"SpikingNeuronGroup{str(len(net.NeuronGroups) + 1)}"`.
        dendrite_params (dict): Parameters of the NeuronDendrite behavior.
        neuron_params (dict): Parameters of the specified neuron type.
    """

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
            tag = "SpikingNeuronGroup" + str(len(net.NeuronGroups) + 1)

        if isinstance(neuron_type, str):
            neuron_type = getattr(neuron_types, neuron_type)

        behavior = {
            NEURON_TIMESTAMPS["NeuronDendrite"]: NeuronDendrite(**dendrite_params),
            NEURON_TIMESTAMPS["NeuronDynamic"]: neuron_type(**neuron_params),
            NEURON_TIMESTAMPS["Fire"]: Fire(),
            NEURON_TIMESTAMPS["NeuronAxon"]: NeuronAxon(max_delay=max_delay),
        }

        if kwta is not None:
            behavior[NEURON_TIMESTAMPS["KWTA"]] = KWTA(k=kwta, dimension=kwta_dim)

        if tau_trace:
            behavior[NEURON_TIMESTAMPS["Trace"]] = SpikeTrace(tau_s=tau_trace)

        if noise_function:
            behavior[NEURON_TIMESTAMPS["DirectNoise"]] = InherentNoise(noise_function)

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
