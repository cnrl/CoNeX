from pymonntorch import Behavior


class TimeStep(Behavior):
    """
    The behavior that sets universal dt for the network. by each iteration,
    time advances as much as dt.

    Args:
        dt (float): Initial iteration timestep. Default is 1
    """

    def set_variables(self, network):
        """
        Initialize network's timestep with `dt`.

        Args:
            network (Network): Network object.
        """
        network.dt = self.get_init_attr("dt", 1)
