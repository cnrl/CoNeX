"""
Structure of spiking neural populations.
"""

from conex.behaviours.neurons.neuron_types.lif_neurons import LIF
from conex.behaviours.neurons import neuron_types
from conex.behaviours.neurons import (
    KWTA,
    Fire,
    InherentNoise,
    NeuronAxon,
    NeuronDendrite,
    SpikeTrace,
)

from conex.nn.timestamps import NEURON_TIMESTAMPS

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
        neuron_params,
        neuron_type=LIF,
        kwta=None,
        kwta_dim=None,
        tau_trace=None,
        max_delay=None,
        noise_function=None,
        fire=False,
        tag=None,
        dendrite_params=None,
        user_defined=None,
    ):
        if tag is None and net is not None:
            tag = "SpikingNeuronGroup" + str(len(net.NeuronGroups) + 1)

        if isinstance(neuron_type, str):
            neuron_type = getattr(neuron_types, neuron_type)

        behavior = {NEURON_TIMESTAMPS["NeuronDynamic"]: neuron_type(**neuron_params)}

        if dendrite_params is not None:
            behavior[NEURON_TIMESTAMPS["NeuronDendrite"]] = NeuronDendrite(
                **dendrite_params
            )

        if kwta is not None:
            behavior[NEURON_TIMESTAMPS["KWTA"]] = KWTA(k=kwta, dimension=kwta_dim)

        if tau_trace:
            behavior[NEURON_TIMESTAMPS["Trace"]] = SpikeTrace(tau_s=tau_trace)

        if max_delay is not None:
            behavior[NEURON_TIMESTAMPS["NeuronAxon"]] = NeuronAxon(max_delay=max_delay)

        if noise_function:
            behavior[NEURON_TIMESTAMPS["DirectNoise"]] = InherentNoise(noise_function)

        if fire:
            behavior[NEURON_TIMESTAMPS["Fire"]] = Fire()

        if user_defined is not None:
            behavior.update(user_defined)

        super().__init__(size, behavior, net, tag)
