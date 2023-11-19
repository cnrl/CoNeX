"""
Dendritic behaviors.
"""
from pymonntorch import Behavior

import torch
import torch.nn.functional as F


# TODO not priming neurons with over threshold potential.
# TODO lower than threshold nonPriming
# TODO Priming inhibitory neurons???? by inhibitory neurons


class BaseDendriticInput(Behavior):
    """
    Base behavior for turning pre-synaptic spikes to post-synaptic current. It checks for excitatory/inhibitory attributes
    of pre-synaptic neurons and sets a coefficient accordingly.

    Note: weights must be initialize by others behaviors.
          Also, Axon paradigm should be added to the neurons.
          Connection type (Proximal, Distal, Apical) should be specified by the tag
          of the synapse. and Dendrite behavior of the neurons group should access the
          `I` of each synapse to apply them.

    Args:
        current_coef (float): Scalar coefficient that multiplies weights.
    """

    def __init__(self, *args, current_coef=1, **kwargs):
        super().__init__(*args, current_coef=current_coef, **kwargs)

    def initialize(self, synapse):
        """
        Sets the current_type to -1 if the pre-synaptic neurons are inhibitory.

        Args:
            current_coef (float): Strength of the synapse.
        """
        synapse.add_tag(self.__class__.__name__)
        self.current_coef = self.parameter("current_coef", 1)

        self.current_type = (
            -1 if ("GABA" in synapse.src.tags) or ("inh" in synapse.src.tags) else 1
        )

        self.def_dtype = synapse.def_dtype
        synapse.I = synapse.dst.vector(0)

    def calculate_input(self, synapse):
        ...

    def forward(self, synapse):
        synapse.I = (
            self.current_coef * self.current_type * self.calculate_input(synapse)
        )


class SimpleDendriticInput(BaseDendriticInput):
    """
    Fully connected dendrite behavior. It checks for excitatory/inhibitory attributes
    of pre-synaptic neurons and sets a coefficient, accordingly.

    Note: weights must be initialize by others behaviors.
          Also, Axon paradigm should be added to the neurons.
          Connection type (Proximal, Distal, Apical) should be specified by the tag
          of the synapse. and Dendrite behavior of the neurons group should access the
          `I` of each synapse to apply them.

    Args:
        current_coef (float): Scalar coefficient that multiplies weights.
    """

    def initialize(self, synapse):
        super().initialize(synapse)

        if not synapse.network.transposed_synapse_matrix_mode:
            raise RuntimeError(f"Network should've made with SxD mode for synapses")

    def calculate_input(self, synapse):
        spikes = synapse.src.axon.get_spike(synapse.src, synapse.src_delay)
        return torch.sum(synapse.weights[spikes], axis=0)


class AveragePool2D(BaseDendriticInput):
    """
    Average Pooling on Source population spikes.

    Note: Axon paradigm should be added to the neurons.
          Connection type (Proximal, Distal, Apical) should be specified by the tag
          of the synapse. and Dendrite behavior of the neurons group should access the
          `I` of each synapse to apply them.

    Args:
        current_coef (float): Scalar coefficient that multiplies weights.
    """

    def initialize(self, synapse):
        super().initialize(synapse)
        self.output_shape = (synapse.dst.height, synapse.dst.width)

        if synapse.src.depth != synapse.dst.depth:
            raise RuntimeError(
                f"For pooling, source({synapse.src.depht}) and destionation({synapse.dst.depth}) should have same depth."
            )

    def calculate_input(self, synapse):
        spikes = synapse.src.axon.get_spike(synapse.src, synapse.src_delay)
        spikes = spikes.view(synapse.src_shape).to(self.def_dtype)
        I = F.adaptive_avg_pool2d(spikes, self.output_shape)
        return I.view((-1,))


class LateralDendriticInput(BaseDendriticInput):
    """
    Lateral dendrite behavior.

    Note: weight shape = (1, 1, kernel_depth, kernel_height, kernel_width)
          weights must be initialize by others behaviors.
          Also, Axon paradigm should be added to the neurons.
          Connection type (Proximal, Distal, Apical) should be specified by the tag
          of the synapse. and Dendrite behavior of the neurons group should access the
          `I` of each synapse to apply them.

    Args:
        current_coef (float): Scalar coefficient that multiplies weights.
        inhibitory (bool or None): If None, connection type respect the NeuronGroup type. if True, the effect in inhibitory and False is excitatory.
    """

    def __init__(self, *args, current_coef=1, inhibitory=None, **kwargs):
        super().__init__(
            *args, current_coef=current_coef, inhibitory=inhibitory, **kwargs
        )

    def initialize(self, synapse):
        super().initialize(synapse)
        ctype = self.parameter("inhibitory", None)

        self.padding = tuple(((synapse.weights.shape[i] - 1) // 2) for i in range(2, 5))
        if ctype is not None:
            self.current_type = ctype * -2 + 1

        if synapse.src != synapse.dst:
            raise RuntimeError(
                f"Synapse {synapse.src.tags[0]}=>{synapse.dst.tags[0]}: For lateral connection src and dst neuron group should be same"
            )

        if not synapse.weights.numel() % 2:
            raise RuntimeError(
                f"Synapse {synapse.src.tags[0]}=>{synapse.dst.tags[0]}: For lateral connection weight should not have any even size dimension. {synapse.weights.shape}"
            )

    def calculate_input(self, synapse):
        spikes = synapse.src.axon.get_spike(synapse.src, synapse.src_delay).to(
            self.def_dtype
        )
        spikes = spikes.view(1, *synapse.src_shape)

        I = F.conv3d(input=spikes, weight=synapse.weights, padding=self.padding)
        return I.view((-1,))


class Conv2dDendriticInput(BaseDendriticInput):
    """
    2D convolutional dendrite behavior. It checks for excitatory/inhibitory attributes
    of pre-synaptic neurons and sets a coefficient, accordingly.

    Note: Weight shape = (out_channel, in_channel, kernel_height, kernel_width)
          weights must be initialize by others behaviors.
          Also, Axon paradigm should be added to the neurons.
          Connection type (Proximal, Distal, Apical) should be specified by the tag
          of the synapse. and Dendrite behavior of the neurons group should access the
          `I` of each synapse to apply them.

    Args:
        current_coef (float): Scalar coefficient that multiplies weights.
        stride (int): Stride of the convolution. The default is 1.
        padding (int): Padding added to both sides of the input. The default is 0.
    """

    def __init__(self, *args, current_coef=1, stride=1, padding=0, **kwargs):
        super().__init__(
            *args, current_coef=current_coef, stride=stride, padding=padding, **kwargs
        )

    def initialize(self, synapse):
        super().initialize(synapse)

        synapse.stride = self.parameter("stride", 1)
        synapse.padding = self.parameter("padding", 0)

        padding = (
            synapse.padding
            if isinstance(synapse.padding, tuple)
            else (synapse.padding, synapse.padding)
        )
        stride = (
            synapse.stride
            if isinstance(synapse.stride, tuple)
            else (synapse.stride, synapse.stride)
        )

        if synapse.src.depth != synapse.weights.size(1):
            raise RuntimeError(
                f"Synapse {synapse.src.tags[0]}=>{synapse.dst.tags[0]}: Convolution's weight input channel size({synapse.weights.size(1)}) should be same as the depht of source neurongroup ({synapse.src.tags[0]}: {synapse.src.depth})."
            )

        if synapse.dst.depth != synapse.weights.size(0):
            raise RuntimeError(
                f"Synapse {synapse.src.tags[0]}=>{synapse.dst.tags[0]}: Convolution's weight output channel size({synapse.weights.size(0)}) should be same as the depht of destination neurongroup ({synapse.dst.tags[0]}: {synapse.dst.depth})."
            )

        if not (
            synapse.dst.height
            <= (
                (
                    (synapse.src.height + 2 * padding[0] - synapse.weights.size(2))
                    / stride[0]
                )
                + 1
            )
            < synapse.dst.height + 1
        ):
            raise RuntimeError(
                f"Synapse {synapse.src.tags[0]}=>{synapse.dst.tags[0]}: Convolution's height size is not constistent."
            )

        if not (
            synapse.dst.width
            <= (
                (
                    (synapse.src.width + 2 * padding[1] - synapse.weights.size(3))
                    / stride[1]
                )
                + 1
            )
            < synapse.dst.width + 1
        ):
            raise RuntimeError(
                f"Synapse {synapse.src.tags[0]}=>{synapse.dst.tags[0]}: Convolution's width size is not constistent."
            )

    def calculate_input(self, synapse):
        spikes = synapse.src.axon.get_spike(synapse.src, synapse.src_delay).to(
            self.def_dtype
        )
        spikes = spikes.view(synapse.src_shape)

        I = F.conv2d(
            input=spikes,
            weight=synapse.weights,
            stride=synapse.stride,
            padding=synapse.padding,
        )

        # Alternative code that may have efficiency advantage
        #
        # unfold_spikes = F.unfold(input=spikes, kernel_size=synapse.weights.shape[-2:], stride = synapse.stride, padding = synapse.padding)
        # I = (unfold_spikes.T.matmul(synapse.weights.view(synapse.weights.size(0), -1).T)).T

        return I.view((-1,))


class Local2dDendriticInput(BaseDendriticInput):
    """
    2D local dendrite behavior. It checks for excitatory/inhibitory attributes
    of pre-synaptic neurons and sets a coefficient, accordingly.

    Note: Weight shape = (out_channel, out_size, connection_size);
          where out_size = out_height * out_width,
          and connection_size = input_channel * connection_height * connection_width.
          Kernel shape = (out_channel, out_height, out_width, in_channel, connection_height, connection_width)
          weights must be initialize by others behaviors.
          Also, Axon paradigm should be added to the neurons.
          Connection type (Proximal, Distal, Apical) should be specified by the tag
          of the synapse. and Dendrite behavior of the neurons group should access the
          `I` of each synapse to apply them.

    Args:
        current_coef (float): Scalar coefficient that multiplies weights.
        stride (int): Stride of the convolution. The default is 1.
        padding (int): Padding added to both sides of the input. The default is 0.
    """

    def __init__(self, *args, current_coef=1, stride=1, padding=0, **kwargs):
        super().__init__(
            *args, current_coef=current_coef, stride=stride, padding=padding, **kwargs
        )

    def initialize(self, synapse):
        super().initialize(synapse)

        synapse.stride = self.parameter("stride", 1)
        synapse.padding = self.parameter("padding", 0)

        padding = (
            synapse.padding
            if isinstance(synapse.padding, tuple)
            else (synapse.padding, synapse.padding)
        )
        stride = (
            synapse.stride
            if isinstance(synapse.stride, tuple)
            else (synapse.stride, synapse.stride)
        )

        if (
            synapse.kernel_shape[0] != synapse.weights.size(0)
            or synapse.kernel_shape[1] * synapse.kernel_shape[2]
            != synapse.weights.size(1)
            or synapse.kernel_shape[3]
            * synapse.kernel_shape[4]
            * synapse.kernel_shape[5]
            != synapse.weights.size(2)
        ):
            raise RuntimeError(
                f"Synapse {synapse.src.tags[0]}=>{synapse.dst.tags[0]}: Local connetion's weight shape({synapse.weights.shape}) is not consitant with its logical shape({synapse.kernel_shape})."
            )

        if synapse.src.depth != synapse.kernel_shape[3]:
            raise RuntimeError(
                f"Synapse {synapse.src.tags[0]}=>{synapse.dst.tags[0]}: Local connetion's weight input channel size({synapse.kernel_shape[3]}) should be same as the depht of source neurongroup ({synapse.src.tags[0]}: {synapse.src.depth})."
            )

        if synapse.dst.depth != synapse.kernel_shape[0]:
            raise RuntimeError(
                f"Synapse {synapse.src.tags[0]}=>{synapse.dst.tags[0]}: Local connetion's weight output channel size({synapse.kernel_shape[0]}) should be same as the depht of destination neurongroup ({synapse.dst.tags[0]}: {synapse.dst.depth})."
            )

        if (
            not (
                synapse.dst.height
                <= (
                    (
                        (synapse.src.height + 2 * padding[0] - synapse.kernel_shape[4])
                        / stride[0]
                    )
                    + 1
                )
                < synapse.dst.height + 1
            )
            or synapse.kernel_shape[1] != synapse.dst.height
        ):
            raise RuntimeError(
                f"Synapse {synapse.src.tags[0]}=>{synapse.dst.tags[0]}: Local connetion's height size is not constistent."
            )

        if (
            not (
                synapse.dst.width
                <= (
                    (
                        (synapse.src.width + 2 * padding[1] - synapse.kernel_shape[5])
                        / stride[1]
                    )
                    + 1
                )
                < synapse.dst.width + 1
            )
            or synapse.kernel_shape[2] != synapse.dst.width
        ):
            raise RuntimeError(
                f"Synapse {synapse.src.tags[0]}=>{synapse.dst.tags[0]}: Local connetion's width size is not constistent."
            )

    def calculate_input(self, synapse):
        spikes = synapse.src.axon.get_spike(synapse.src, synapse.src_delay).to(
            self.def_dtype
        )
        spikes = spikes.view(synapse.src_shape)
        spikes = F.unfold(
            spikes,
            kernel_size=synapse.kernel_shape[-2:],
            stride=synapse.stride,
            padding=synapse.padding,
        ).T

        I = (spikes * synapse.weights).sum(axis=-1)
        return I.view((-1,))
