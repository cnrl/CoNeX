"""
Behaviors to load datasets 
"""

from pymonntorch import Behavior
import torch


class SpikeNdDataset(Behavior):
    """
    This behavior ease loading dataset as spikes for `InputLayer`.
    """

    def initialize(self, layer):
        self.dataloader = self.parameter("dataloader", None, required=True)
        self.sensory_dimension = self.parameter("ndim_sensory", 2)
        self.location_dimension = self.parameter("ndim_location", 2)
        self.have_location = self.parameter("have_location", False)
        self.have_sensory = self.parameter("have_sensory", True)
        self.have_label = self.parameter("have_label", True)
        self.loop = self.parameter("loop", True)

        self.data_generator = self._get_data()
        self.device = layer.device

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
                    each_instance = num_instance // torch.numel(batch_y)

                for i in range(num_instance):
                    x = batch_x[i].view((-1,)) if batch_x is not None else None
                    loc = batch_loc[i].view((-1,)) if batch_loc is not None else None
                    y = batch_y[i // each_instance] if batch_y is not None else None
                    yield x, loc, y

    def forward(self, layer):
        layer.x, layer.loc, layer.y = next(self.data_generator)
