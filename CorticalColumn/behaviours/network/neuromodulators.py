"""
Network-wide neuromodulators.
"""

from PymoNNto import Behaviour


class Dopamine(Behaviour):
    """
    Compute extracellular dopamine concentration.

    Note: Reward behavior should be defined prior to Dopamine behavior
    while defining the network.

    Args:
        tau_dopamine (float): Dopamine decay time constant.
    """

    def set_variables(self, network):
        """
        Set initial dopamine concentration value based on initial reward value.

        Args:
            network (Network): Network object.
        """
        self.add_tag("Dopamine")
        self.set_init_attrs_as_variables(network)

        network.dopamine_concentration = network.reward

    def new_iteration(self, network):
        """
        Compute extracellular dopamine concentration at each time step by:

        dd/dt = -d/tau_d + reward(t).

        Args:
            network (Network): Network object.
        """
        dd_dt = (
            -(network.dopamine_concentration / network.tau_dopamine) + network.reward
        )
        network.dopamine_concentration += dd_dt
