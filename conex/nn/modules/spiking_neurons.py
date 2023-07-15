"""
Structure of spiking neural populations.
"""

from conex.behaviors.neurons.neuron_types.lif_neurons import LIF
from conex.behaviors.neurons import neuron_types
from conex.behaviors.neurons import (
    KWTA,
    Fire,
    InherentNoise,
    NeuronAxon,
    SimpleDendriteStructure,
    SimpleDendriteComputation,
    SpikeTrace,
)

from conex.nn.priorities import NEURON_PRIORITIES

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
        axon (Behavior): Behavior class for axon dynamic of neuron group.
        max_delay (int): Defines the maximum (buffer size of) axonal delay. The default is 1.
        noise_params (dict): If not None, enables inherent noise in membrane potential and specifies its parameters.
        tag (str): The tag(s) of the population. If None, it is set to `"SpikingNeuronGroup{str(len(net.NeuronGroups) + 1)}"`.
        dendrite_structure (Behavior): Behavior class for defining the structure of dendrite.
        dendrite_structure_params (dict): Parameters of the dendrite_structure behavior.
        dendrite_computation (Behavior): Behavior class for dynamics of a the dendrite structure to compute the current entering the soma.
        dendrite_computation_params (dict): Parameters of the dendrite_computation behavior.
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
        axon=NeuronAxon,
        max_delay=1,
        noise_params=None,
        fire=True,
        tag=None,
        dendrite_structure=SimpleDendriteStructure,
        dendrite_structure_params=None,
        dendrite_computation=SimpleDendriteComputation,
        dendrite_computation_params=None,
        user_defined=None,
    ):
        if tag is None and net is not None:
            tag = "SpikingNeuronGroup" + str(len(net.NeuronGroups) + 1)

        if isinstance(neuron_type, str):
            neuron_type = getattr(neuron_types, neuron_type)

        behavior = {NEURON_PRIORITIES["NeuronDynamic"]: neuron_type(**neuron_params)}

        if dendrite_structure is not None:
            dendrite_structure_params = (
                dendrite_structure_params
                if dendrite_structure_params is not None
                else {}
            )
            behavior[NEURON_PRIORITIES["DendriteStructure"]] = dendrite_structure(
                **dendrite_structure_params
            )

        if dendrite_computation is not None:
            dendrite_computation_params = (
                dendrite_computation_params
                if dendrite_computation_params is not None
                else {}
            )
            behavior[NEURON_PRIORITIES["DendriteComputation"]] = dendrite_computation(
                **dendrite_computation_params
            )

        if kwta is not None:
            behavior[NEURON_PRIORITIES["KWTA"]] = KWTA(k=kwta, dimension=kwta_dim)

        if tau_trace:
            behavior[NEURON_PRIORITIES["Trace"]] = SpikeTrace(tau_s=tau_trace)

        if max_delay is not None:
            behavior[NEURON_PRIORITIES["NeuronAxon"]] = axon(max_delay=max_delay)

        if noise_params:
            behavior[NEURON_PRIORITIES["DirectNoise"]] = InherentNoise(**noise_params)

        if fire:
            behavior[NEURON_PRIORITIES["Fire"]] = Fire()

        if user_defined is not None:
            behavior.update(user_defined)
        
        self.dendrites = {}

        super().__init__(size, behavior, net, tag)
