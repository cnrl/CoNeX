from conex.behaviors.network.time_resolution import TimeResolution
from pymonntorch import Network

from conex.nn.priorities import NETWORK_PRIORITIES


class Neocortex(Network):
    """
    A subclass Network that enables defining cortical connections.

    Args:
        dt (float): The time resolution. Default is 1.
        payoff (Payoff): If not None, enables reinforcement learning with a payoff function defined by an instance of Payoff class.
        neuromodulators (list): List of Neuromodulators used in the network.
        device (str): Device on which the network and its components are located. The default is "cpu".

    Usage:
        net = Neocortex()

        # define neural population that holds processed sensory input spikes
        sensory_inp = NeuronGroup(size=28*28, net=net, behavior={...})

        # define neural network responsible for generating output
        output = SpikingNeuronGroup(
            10,
            net,
            neuron_type=LIF,
            kwta=1,
            dendrite_params={
                "proximal_max_delay": 2
            },
            neuron_params={
                "tau": 20,
                "R": 1,
                "threshold": -55,
                "v_rest": -70,
                "v_reset": -72
            }
        )

        # define a cortical column by specifying the configuration of its layers and their connections
        cc1 = CorticalColumn(
            net,
            sensory_inp,
            representation_layer=output,
            L2_3_config={
                "exc_pop_config": {
                    "neuron_type": AELIF,
                    "kwta": 5,
                    "dendrite_params": {...},
                    "neuron_params": {...},
                },
                "inh_pop_config": {
                    "neuron_type": ELIF,
                    "neuron_params": {...},
                    "dendrite_params": {...},
                },
                "exc_inh_config": {
                    "structure": "Simple",
                    "learning_rule": None,
                    "weight_init_params": {
                        "mode": "randn",
                    }
                }
                "inh_exc_config": {
                    "structure": "Simple",
                    "learning_rule": None,
                    "weight_init_params": {
                        "mode": "randn",
                    }
                }
            }
            L4_config={...},
            L4_L2_3_config={
                "exc_exc": {
                    "structure": "Conv2d",
                    "learning_rule": "STDP",
                    "weight_init_params": {
                        "mode": "randn",
                    }
                }
            }
            sensory_L4_config={...},
            L2_3_representation_config={...},
        )

        # The same can go here for creating another cortical column
        cc2 = CorticalColumn(...)

        # specify the connectivity configurations of the two columns
        # cc1 -> cc2
        cc1.connect(
            cc2,
            L2_3_L2_3_config={
                "exc_exc": {...},
                "inh_exc": {...}
            }
        )
        # cc2 -> cc1
        cc2.connect(
            cc1,
            L2_3_L2_3_config={
                "exc_exc": {...},
                "inh_exc": {...}
            }
        )

        net.initialize()

        # Now you can simulate your network using net.simulate_iterations(...)
    """

    def __init__(self, dt=1, payoff=None, neuromodulators=None, settings={}):
        behavior = {NETWORK_PRIORITIES["TimeResolution"]: TimeResolution(dt=dt)}
        if payoff:
            behavior[NETWORK_PRIORITIES["Payoff"]] = payoff

        if neuromodulators is not None:
            for i, neuromodulator in enumerate(neuromodulators):
                behavior[NETWORK_PRIORITIES["NeuroModulator"] + i] = neuromodulator

        super().__init__(tag="Neocortex", behavior=behavior, settings=settings)
        self.dt = dt
        self.columns = []
        self.input_layers = []
        self.output_layers = []
        self.inter_column_synapses = []

    def initialize(self, info=True, storage_manager=None):
        """
        Initializes the network by saving inter-column synapses as well as other components of the network.

        Args:
            info (bool): If true, prints information about the network.
            storage_manager (StorageManager): Storage manager to use for the network.
        """
        for syn in self.SynapseGroups:
            if hasattr(syn, "Apical"):
                self.inter_column_synapses.append(syn)

        super().initialize(info=info, storage_manager=storage_manager)

    def connect_columns(
        self,
        L2_3_L2_3_config=None,
        L2_3_L4_config=None,
        L5_L5_config=None,
        L5_L6_config=None,
    ):
        """
        Makes connections between all column in the network.

        Note: In the config dicts, the key is the name of synapse between the populations in the corresponding layers
              and the values are the synaptic config dicts.

        Args:
            L2_3_L2_3_config (dict): Adds the synaptic connections from L2/3 of a to L2/3 of the other with the specified configurations.
            L2_3_L4_config (dict): Adds the synaptic connections from L2/3 of a column to L4 of the other with the specified configurations.
            L5_L5_config (dict): Adds the synaptic connections from L5 of a column to L5 of the other with the specified configurations.
            L6_L6_config (dict): Adds the synaptic connections from L6 of a column to L6 of the other with the specified configurations.
        """
        synapses = {}

        for i, col_i in enumerate(self.columns):
            for j, col_j in enumerate(self.columns[i:], i):
                col_syn = col_i.connect(
                    col_j, L2_3_L2_3_config, L2_3_L4_config, L5_L5_config, L5_L6_config
                )
                synapses.update(col_syn)

        return synapses
