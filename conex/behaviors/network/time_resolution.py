from pymonntorch import Behavior


class TimeResolution(Behavior):
    """
    The behavior that sets universal dt for the network. by each iteration,
    time advances as much as dt.

    Args:
        dt (float): Initial iteration time resolution. Default is 1
    """

    def __init__(self, *args, dt=1, **kwargs):
        super().__init__(*args, dt=dt, **kwargs)

    def initialize(self, network):
        """
        Initialize network's time resolution with `dt`.

        Args:
            network (Network): Network object.
        """
        network.dt = self.parameter("dt", 1)
