from CorticalColumn.behaviours.network.timestep import TimeStep
from pymonntorch import Network
# TODO: inter-column delay should be 1 unit more than intra-column delay.
class Neocortex(Network):
    def __init__(self, dt=1, reward=None, neuromodulators=[], device="cpu"):
        behavior = {1  : TimeStep(dt=dt)}
        if reward:
            behavior[20] = reward
        for i, neuromodulator in enumerate(neuromodulators):
            behavior[21+i] = neuromodulator

        super().__init__(tag="Neocortex", behavior=behavior, device=device)
        self.dt = dt
        self.columns = []
        self.inter_column_synapses = []

    def initialize(self, info=True, storage_manager=None):
        for syn in self.SynapseGroups:
            if hasattr(syn, "Apical"):
                self.inter_column_synapses.append(syn)

        super().initialize(info=info, storage_manager=storage_manager)
