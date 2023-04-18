"""
Behaviors to load datasets 
"""

from pymonntorch import Behavior
import torch


class SpikeNdDataset(Behavior):
    """
    This behavior ease loading dataset as spikes for `NeuronGroup`. suitable for images
    """

    def set_variables(self, neurons):
        self.dataloader = self.get_init_attr('dataloader', None, required=True)
        self.dimension = self.get_init_attr('N', 2)
        self.loop = self.get_init_attr('loop', True)

        self.data_generator = self._get_data()
        self.device = neurons.device
        neurons.spikes = neurons.get_neuron_vec(dtype=torch.bool)

    def _get_data(self):
        while self.loop:
            for batch in self.dataloader:
                batch_x, batch_y = batch
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_x = batch_x.view((-1, *batch_x.shape[-self.dimension:]))

                each_instance = batch_x.size(0) // torch.numel(batch_y)

                for i, x in enumerate(batch_x):
                    yield x, batch_y[i//each_instance]

    def forward(self, neurons):
        x, y = next(self.data_generator)
        neurons.spikes = x.view((-1,))
        neurons.network.y_target = y