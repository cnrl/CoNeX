"""
Synapse-related behaviors.
"""

from pymonntorch import Behavior
import random
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

    def __init__(
        self,
        *args,
        mode=None,
        scale=1,
        offset=0,
        destination=False,
        delays=None,
        **kwargs,
    ):
        super().__init__(
            *args,
            mode=mode,
            scale=scale,
            offset=offset,
            destination=destination,
            delays=delays,
            **kwargs,
        )

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
        scale (float): Scaling factor to apply on the weight.
        offset (float): An offset to add to the weight.
        function (callable): A function to apply on weight.
        density (flaot): The sparsity of weights. default is one.
        true_sparsity (bool) : If false, weights are created but have zero value. Defaults to True.
        weights (tensor): Optional parameter to specify the weights matrix directly.
        weight_shape (tuple): Optional parameter to specify the shape of the weights tensor.
        kernel_shape (tuple): Optional parameter to specify the shape of the kernel.
    """

    def __init__(
        self,
        *args,
        mode=None,
        scale=1,
        offset=0,
        function=None,
        density=1,
        true_sparsity=True,
        weight_shape=None,
        kernel_shape=None,
        weights=None,
        **kwargs,
    ):
        super().__init__(
            *args,
            mode=mode,
            scale=scale,
            offset=offset,
            function=function,
            density=density,
            true_sparsity=true_sparsity,
            weight_shape=weight_shape,
            kernel_shape=kernel_shape,
            weights=weights,
            **kwargs,
        )

    def initialize(self, synapse):
        init_mode = self.parameter("mode", None)
        scale = self.parameter("scale", 1)
        offset = self.parameter("offset", 0)
        function = self.parameter("function", None)
        density = self.parameter("density", 1)
        true_sparsity = self.parameter("true_sparsity", True)
        synapse.weights = self.parameter("weights", None)
        weight_shape = self.parameter("weight_shape", None)
        synapse.kernel_shape = self.parameter("kernel_shape", None)

        weight_shape = (
            weight_shape if weight_shape is not None else synapse.matrix_dim()
        )
        if init_mode is not None and synapse.weights is None:
            if not true_sparsity or density == 1:
                synapse.weights = synapse.tensor(
                    mode=init_mode, dim=weight_shape, density=density
                )
            else:
                n_row, n_col = synapse.matrix_dim()
                nnz = int(n_row * n_col * density)
                both_indices = torch.tensor(
                    random.sample(range(n_row * n_col), nnz), device=synapse.device
                )  # TODO pytorch alternative
                dst_idx = both_indices % n_col
                src_idx = both_indices // n_col
                indices = torch.stack([src_idx, dst_idx])
                values = synapse.tensor(mode=init_mode, dim=(nnz,))
                synapse.weights = torch.sparse_coo_tensor(
                    indices, values, synapse.matrix_dim()
                )
                synapse.weights = synapse.weights.coalesce()
                synapse.weights = synapse.weights.to_sparse_csc()
                synapse.dst_idx = torch.arange(
                    n_col, device=synapse.device
                ).repeat_interleave(synapse.weights.ccol_indices().diff())
                synapse.src_idx = synapse.weights.row_indices()

            if function is not None:
                synapse.weights = function(synapse.weights)

            synapse.weights = synapse.weights * scale
            if synapse.weights.layout != torch.strided:  # Pytorch should fix is_sparse
                synapse.weights.values()[:] += offset
            else:
                synapse.weights += offset


class WeightNormalization(Behavior):
    """
    This Behavior normalize weights in order to assure each destination neuron has
    sum of its weight equal to ``norm``. Supporting `Simple`, `Local2d`, 'Conv2d'.

    Args:
        norm (int): Desired sum of weights for each neuron.
    """

    def __init__(self, *args, norm=1, **kwargs):
        super().__init__(*args, norm=norm, **kwargs)

    def initialize(self, synapse):
        self.norm = self.parameter("norm", 1)
        self.dims = [x for x in range(1, len(synapse.weights.shape))]
        if len(synapse.weights.shape) == 2:
            self.dims = [0]
        if len(synapse.weights.shape) == 3:
            self.dims = [2]

    def forward(self, synapse):
        weights_sum = synapse.weights.sum(dim=self.dims, keepdim=True)
        weights_sum[weights_sum == 0] = 1
        synapse.weights *= self.norm / weights_sum


class CurrentNormalization(Behavior):
    """
    This Behavior normalize Current in order to assure each destination neuron
    maximum input current is eight equal to ``norm``. Supporting `Simple`, `Local2d`, 'Conv2d'.

    Args:
        norm (int): Desired maximum of current for each neuron.
    """

    def __init__(self, *args, norm=1, **kwargs):
        super().__init__(*args, norm=norm, **kwargs)

    def initialize(self, synapse):
        self.norm = self.parameter("norm", 1)
        self.dims = [x for x in range(1, len(synapse.weights.shape))]
        if len(synapse.weights.shape) == 2:
            self.dims = [0]
        if len(synapse.weights.shape) == 3:
            self.dims = [2]

    def forward(self, synapse):
        weights_sum = synapse.weights.sum(dim=self.dims).view(
            -1,
        )
        weights_sum[weights_sum == 0] = 1
        normalized = self.norm / weights_sum
        synapse.I *= normalized.repeat_interleave(
            synapse.I.numel() // normalized.numel()
        )


class WeightClip(Behavior):
    """
    Clip the synaptic weights in a range.

    Args:
        w_min (float): Minimum weight constraint.
        w_max (float): Maximum weight constraint.
    """

    def __init__(self, *args, w_min=0, w_max=1, **kwargs):
        super().__init__(*args, w_min=w_min, w_max=w_max, **kwargs)

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


class SrcSpikeCatcher(Behavior):
    """
    Get the spikes from pre synaptic neuron group and set as src_spike attribute for the synapse group.

    Note: Axon should be added to pre synaptice neuron group
    """

    def forward(self, synapse):
        synapse.pre_spike = synapse.src.axon.get_spike(synapse.src, synapse.src_delay)


class DstSpikeCatcher(Behavior):
    """
    Get the spikes from post synaptic neuron group and set as dst_spike attribute for the synapse group.

    Note: Axon should be added to post synaptice neuron group
    """

    def forward(self, synapse):
        synapse.post_spike = synapse.dst.axon.get_spike(synapse.dst, synapse.dst_delay)


class PreTrace(Behavior):
    """
    Calculates the pre synaptic spike trace.

    Note : should be placed after spike catcher behavior.

    Args:
        tau_s (float): decay term for spike trace. The default is None.
        spike_scale (float): the increase effect of spikes on the trace.
    """

    def __init__(self, tau_s, *args, spike_scale=1.0, **kwargs):
        super().__init__(*args, tau_s=tau_s, spike_scale=spike_scale, **kwargs)

    def initialize(self, synapse):
        """
        Sets the trace attribute for the neural population.
        """
        self.tau_s = self.parameter("tau_s", None, required=True)
        self.spike_scale = self.parameter("spike_scale", 1.0)
        synapse.pre_trace = synapse.src.vector(0.0)

    def forward(self, synapse):
        """
        Calculates the spike trace of each neuron by adding current spike and decaying the trace so far.
        """
        synapse.pre_trace += synapse.src_spikes * self.spike_scale
        synapse.pre_trace -= (synapse.pre_trace / self.tau_s) * synapse.network.dt


class PostTrace(Behavior):
    """
    Calculates the post synaptic spike trace.

    Note : should be placed after spike catcher behavior.

    Args:
        tau_s (float): decay term for spike trace. The default is None.
        spike_scale (float): the increase effect of spikes on the trace.
    """

    def __init__(self, tau_s, *args, spike_scale=1.0, **kwargs):
        super().__init__(*args, tau_s=tau_s, spike_scale=spike_scale, **kwargs)

    def initialize(self, synapse):
        """
        Sets the trace attribute for the neural population.
        """
        self.tau_s = self.parameter("tau_s", None, required=True)
        self.spike_scale = self.parameter("spike_scale", 1.0)
        synapse.post_trace = synapse.src.vector(0.0)

    def forward(self, synapse):
        """
        Calculates the spike trace of each neuron by adding current spike and decaying the trace so far.
        """
        synapse.post_trace += synapse.dst_spikes * self.spike_scale
        synapse.post_trace -= (synapse.post_trace / self.tau_s) * synapse.network.dt
