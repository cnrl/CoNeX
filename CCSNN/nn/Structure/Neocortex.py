from CCSNN.behaviours.network.timestep import TimeStep
from pymonntorch import Network

from CCSNN.nn.timestamps import NETWORK_TIMESTAMPS


class Neocortex(Network):
    """
    A subclass Network that enables defining cortical connections.

    Args:
        dt (float): The timestep. Default is 1.
        reward (Reward): If not None, enables reinforcement learning with a reward function defined by an instance of Reward class.
        neuromodulators (list): List of Neuromodulators used in the network.
        device (str): Device on which the network and its components are located. The default is "cpu".
    """

    def __init__(self, dt=1, reward=None, neuromodulators=[], device="cpu"):
        behavior = {NETWORK_TIMESTAMPS["TimeStep"]: TimeStep(dt=dt)}
        if reward:
            behavior[NETWORK_TIMESTAMPS["Reward"]] = reward
        for i, neuromodulator in enumerate(neuromodulators):
            behavior[NETWORK_TIMESTAMPS["NeuroModulator"] + i] = neuromodulator

        super().__init__(tag="Neocortex", behavior=behavior, device=device)
        self.dt = dt
        self.columns = []
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
