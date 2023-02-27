"""
Network-wide neuromodulators.
"""

from pymonntorch import Behavior


class Dopamine(Behavior):
    """
    Compute extracellular dopamine concentration.

    Note: Reward behavior should be defined prior to Dopamine behavior
    while defining the network.

    Args:
        tau_dopamine (float): Dopamine decay time constant.
        initial_dopamine_concentration (float, optional): Initial dopamine concentration
    """

    def set_variables(self, network):
        """
        Set initial dopamine concentration value based on initial reward value.

        Args:
            network (Network): Network object.
        """
        self.add_tag("Dopamine")

        network.tau_dopamine = self.get_init_attr("tau_dopamine", 0.0)
        network.dopamine_concentration = self.get_init_attr("initial_dopamine_concentration", network.reward) 

    def forward(self, network):
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
