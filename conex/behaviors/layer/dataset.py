import torch

"""
Behaviors to load datasets 
"""

from pymonntorch import Behavior
import torch


class SpikeNdDataset(Behavior):
    """
    This behavior ease loading dataset as spikes for `InputLayer`.

    Args:
        dataloader (Dataloader): A pytorch dataloader kind returning up to a triole of (sensory, location, label).
        ndim_sensory (int): Sensory's number of dimension refering to a single instance.
        ndim_location (int): Location's number of dimension refering to a single instance.
        have_location (bool): Whether dataloader returns location input.
        have_sensory (bool): Whether dataloader returns sensory input.
        have_label (bool): Whether dataloader returns label of input.
        silent_interval (int): The interval of silent activity between two different input.
        instance_duration (int): The duration of each instance of input with same target value.
        loop (bool): If True, dataloader repeats.
    """

    def __init__(
        self,
        dataloader,
        instance_duration,
        *args,
        ndim_sensory=2,
        ndim_location=2,
        have_location=False,
        have_sensory=True,
        have_label=True,
        silent_interval=0,
        loop=True,
        **kwargs
    ):
        super().__init__(
            *args,
            dataloader=dataloader,
            ndim_sensory=ndim_sensory,
            ndim_location=ndim_location,
            have_location=have_location,
            have_sensory=have_sensory,
            have_label=have_label,
            silent_interval=silent_interval,
            instance_duration=instance_duration,
            loop=loop,
            **kwargs
        )

    def initialize(self, layer):
        self.dataloader = self.parameter("dataloader", None, required=True)
        self.sensory_dimension = self.parameter("ndim_sensory", 2)
        self.location_dimension = self.parameter("ndim_location", 2)
        self.have_location = self.parameter("have_location", False)
        self.have_sensory = self.parameter("have_sensory", True)
        self.have_label = self.parameter("have_label", True)
        self.silent_interval = self.parameter("silent_interval", 0)
        self.each_instance = self.parameter("instance_duration", 0, required=True)
        self.loop = self.parameter("loop", True)

        self.data_generator = self._get_data()
        self.device = layer.device
        self.new_data = False
        self.silent_iteration = 0

    def _get_data(self):
        while self.loop:
            for batch in self.dataloader:
                batch_x = batch[0] if self.have_sensory else None
                batch_loc = batch[self.have_sensory] if self.have_location else None
                batch_y = batch[-1] if self.have_label else None

                if batch_x is not None:
                    batch_x = batch_x.to(self.device)
                    batch_x = batch_x.view(
                        (-1, *batch_x.shape[-self.sensory_dimension :])
                    )
                    num_instance = batch_x.size(0)

                if batch_loc is not None:
                    batch_loc = batch_loc.to(self.device)
                    batch_loc = batch_loc.view(
                        (-1, *batch_loc.shape[-self.location_dimension :])
                    )
                    num_instance = batch_loc.size(0)
                    if batch_x:
                        assert (
                            batch_x.size(0) == num_instance
                        ), "sensory and location should have same number of instances."

                if batch_y is not None:
                    batch_y = batch_y.to(self.device)
                    self.each_instance = num_instance // torch.numel(batch_y)

                for i in range(num_instance):
                    x = batch_x[i].view((-1,)) if batch_x is not None else None
                    loc = batch_loc[i].view((-1,)) if batch_loc is not None else None
                    y = (
                        batch_y[i // self.each_instance]
                        if batch_y is not None
                        else None
                    )
                    if i % self.each_instance == self.each_instance - 1:
                        self.new_data = True
                    yield x, loc, y

    def forward(self, layer):
        if self.silent_interval and self.new_data:
            if self.silent_iteration == 0:
                layer.x = (
                    layer.tensor(mode="zeros", dtype=torch.bool, dim=layer.x.shape)
                    if layer.x is not None
                    else None
                )
                layer.loc = (
                    layer.tensor(mode="zeros", dtype=torch.bool, dim=layer.loc.shape)
                    if layer.loc is not None
                    else None
                )

            self.silent_iteration += 1

            if self.silent_iteration == self.silent_interval:
                self.new_data = False
                self.silent_iteration = 0
            return

        layer.x, layer.loc, layer.y = next(self.data_generator)
