"""
Payoff definition base.
"""

from pymonntorch import Behavior


class Payoff(Behavior):
    """
    Base behavior class to define the payoff (reward/punishment) function. Define the desired payoff
    function by inheriting this class and defining `forward` abstract
    method per se. You will set `network.payoff` with the payoff value at the
    time step.

    Args:
        initial_payoff (float): Initial reward/punishment value. Default is 0.0.
    """

    def initialize(self, network):
        """
        Initialize network's payoff with `initial_payoff`.

        Args:
            network (Network): Network object.
        """
        network.payoff = self.parameter("initial_payoff", 0.0, network)
