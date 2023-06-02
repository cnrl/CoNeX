"""
Synapse-related behaviors.
"""

from pymonntorch import Behavior
import torch


class SynapseInit(Behavior):
    """
    This Behavior makes initial variable required for multiple behavior to use.

    **WARNING:** ``src_delay`` and ``dst_delay`` have equal delay for all of their neurons.
    And should be initialized by other behaviors.
    """

    def initialize(self, synapse):
        synapse.src_shape = (1, 1, synapse.src.size)
        if hasattr(synapse.src, "depth"):
            synapse.src_shape = (
                synapse.src.depth,
                synapse.src.height,
                synapse.src.width,
            )
        synapse.dst_shape = (1, 1, synapse.dst.size)
        if hasattr(synapse.dst, "depth"):
            synapse.dst_shape = (
                synapse.dst.depth,
                synapse.dst.height,
                synapse.dst.width,
            )

        synapse.src_delay = synapse.tensor(
            mode="zeros", dim=(1,), dtype=torch.long
        ).expand(synapse.src.size)
        synapse.dst_delay = synapse.tensor(
            mode="zeros", dim=(1,), dtype=torch.long
        ).expand(synapse.dst.size)


class DelayInitializer(Behavior):
    """
    Initialize the delay of axon entering the synapse or the dendrite conveying current.

    delays (Tensor(int), optional): a tensor of delay for each neuron of source or destination.

    Args:
        mode (str or number): string should be torch functions that fills a tensor like:
                              "random", "normal", "zeros", "ones", ... .
                              In number case the synapse delays will be filled with that number.
        offset (int): delay added to the all delays.
        scale (int): scales delay.
        weights (tensor): giving the delays directly.
        destination (boolean): True for destination neurons. defaults to False.
    """

    def initialize(self, synapse):
        """
        Makes index for the Synapse delay.

        Args:
            synapses (SynapseGroup): The synapses whose weight should be bound.
        """
        init_mode = self.parameter("mode", None)
        delays = self.parameter("delays", None)
        scale = self.parameter("scale", 1)
        offset = self.parameter("offset", 0)
        isDestination = self.parameter("destination", False)

        neurons = synapse.src
        attribute = "src"
        if isDestination:
            neurons = synapse.dst
            attribute = "dst"

        if init_mode is not None and delays is None:
            delays = neurons.vector(mode=init_mode)
            delays *= scale
            delays += offset

        setattr(synapse, f"{attribute}_delay", delays.to(torch.long))


class WeightInitializer(Behavior):
    """
    Initialize the weights of synapse.

    Note: Either `mode` or `weights` should be not None or else, the weight matrix will be None.

    Args:
        mode (str or number): string should be torch functions that fills a tensor like:
                              "random", "normal", "zeros", "ones", ... .
                              In number case the synapse weights will be filled with that number.
        weights (tensor): Optional parameter to specify the weights matrix directly.
        weight_shape (tuple): Optional parameter to specify the shape of the weights matrix.
        kernel_shape (tuple): Optional parameter to specify the shape of the kernel.
    """

    def initialize(self, synapse):
        init_mode = self.parameter("mode", None)
        synapse.weights = self.parameter("weights", None)
        synapse.weight_shape = self.parameter("weight_shape", None)
        synapse.kernel_shape = self.parameter("kernel_shape", None)

        if init_mode is not None and synapse.weights is None:
            if synapse.weight_shape is None:
                synapse.weights = synapse.matrix(mode=init_mode)
            else:
                synapse.weights = synapse.tensor(
                    mode=init_mode, dim=synapse.weight_shape
                )


class WeightNormalization(Behavior):
    """
    This Behavior normalize weights in order to assure each destination neuron has
    sum of its weight equal to ``norm``. Supporting `Simple`, `Local2d`, 'Conv2d'.

    Args:
        norm (int): Desired sum of weights for each neuron.
    """

    def initialize(self, synapse):
        self.norm = self.parameter("norm", 1)
        self.dims = [x for x in range(1, len(synapse.weights.shape))]
        if len(synapse.weights.shape) == 1:
            self.dims = [0]
        if len(synapse.weights.shape) == 2:
            self.dims = [2]

    def forward(self, synapse):
        weights_sum = synapse.weights.sum(dim=self.dims, keepdim=True)
        weights_sum[weights_sum == 0] = 1
        synapse.weights *= self.norm / weights_sum


class WeightClip(Behavior):
    """
    Clip the synaptic weights in a range.

    Args:
        w_min (float): minimum weight constraint.
        w_max (float): maximum weight constraint.
    """

    def initialize(self, synapse):
        """
        Set weight constraint attributes to the synapses.

        Args:
            synapses (SynapseGroup): The synapses whose weight should be bound.
        """
        self.w_min = self.parameter("w_min", 0)
        self.w_max = self.parameter("w_max", 1)

        assert 0 <= self.w_min < self.w_max, "Invalid Interval for Weight Clip"

    def forward(self, synapses):
        """
        Clip the synaptic weights in each time step.

        Args:
            synapses (SynapseGroup): The synapses whose weight should be bound.
        """
        synapses.weights = torch.clip(synapses.weights, self.w_min, self.w_max)
