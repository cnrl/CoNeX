"""
Network-wide neuromodulators.
"""

from pymonntorch import Behavior


class Dopamine(Behavior):
    """
    Compute extracellular dopamine concentration.

    Note: Payoff behavior should be defined prior to Dopamine behavior
    while defining the network.

    Args:
        tau_dopamine (float): Dopamine decay time constant.
        initial_dopamine_concentration (float, optional): Initial dopamine concentration
    """

    def initialize(self, network):
        """
        Set initial dopamine concentration value based on initial payoff value.

        Args:
            network (Network): Network object.
        """
        self.add_tag("Dopamine")

        network.tau_dopamine = self.parameter("tau_dopamine", 0.0)
        network.dopamine_concentration = self.parameter(
            "initial_dopamine_concentration", network.payoff
        )

    def forward(self, network):
        """
        Compute extracellular dopamine concentration at each time step by:

        dd/dt = -d/tau_d + payoff(t).

        Args:
            network (Network): Network object.
        """
        dd_dt = (
            -(network.dopamine_concentration /
              network.tau_dopamine) + network.payoff
        )
        network.dopamine_concentration += dd_dt
