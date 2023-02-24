"""
Reward definition base.
"""

from pymonntorch import Behavior


class Reward(Behavior):
    """
    Base behavior class to define reward function. Define the desired reward
    function by inheriting this class and defining `forward` abstract
    method per se. You will set `network.reward` with the reward value at the
    time step.

    Args:
        initial_reward (float): Initial reward value. Default is 0.0.
    """

    def set_variables(self, network):
        """
        Initialize network's reward with `initial_reward`.

        Args:
            network (Network): Network object.
        """
        network.reward = self.get_init_attr("initial_reward", 0.0, network)
