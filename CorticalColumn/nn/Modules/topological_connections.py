"""
Structured SynapseGroup connection schemes.
"""
# TODO: check the whole thing once again for the sake of consistency

import numpy as np
import scipy.sparse as sp
from pymonntorch import SynapseGroup

from CorticalColumn.behaviours.synapses.specs import DelayInitializer, WeightInitializer


# TODO: add a method to return the mask for synapse subgroups

class TopologicalSynapseGroup(SynapseGroup):
    def __init__(self, src, dst, net, max_delay=1.0, tag=None, behaviour={}):
        assert max_delay >= 1.0, f"Invalid value for max_delay: {max_delay}."
        assert net is not None, "net cannot be None."

        if tag is None:
            tag = f"StructuredSynapseGroup_{len(net.synapseGroups) + 1}"

        super().__init__(src, dst, net, tag, behaviour)

        self.min_delay = 1
        self.max_delay = max_delay

    def get_topology(self):
        """
        returns who is connected to who
        """
        raise NotImplementedError

    def get_delay_as_index(self):
        """
        returns [exponent, fraction] of delay
        """
        raise NotImplementedError

    def get_masks(self, src_inds=None, dst_inds=None):
        """
        returns two ndarrays, each of which is a binary mask or
        indices matrix for a population
        """
        if src_inds is None and dst_inds is None:
            return self.get_topology()

    def broadcast(self):
        raise NotImplementedError


class SparseSynapseGroup(TopologicalSynapseGroup):
    def __init__(
        self,
        src,
        dst,
        net,
        density,
        weight_init_mode="normal(0.5, 0.5)",
        delay_init_mode="zeros()",
        max_delay=1,
        tag=None,
        behaviour={},
    ):
        super().__init__(src, dst, net, max_delay, tag, behaviour)

        self.add_tag("sparse")

        self.density = density

        self.__topology = self._create_topology()

        self.add_behaviours_to_object(
            {1: WeightInitializer(init_mode=weight_init_mode)}, self
        )
        self.add_behaviours_to_object(
            {2: DelayInitializer(init_mode=delay_init_mode)}, self
        )

    def _create_topology(self):
        indices = np.meshgrid((range(self.dst.size), range(self.src.size)))
        coords = np.array(
            list(zip(*np.stack(indices).T))
        )  # (post.size,pre.size,2) array

        coords = (
            coords.view([(f"f{i}", coords.dtype) for i in range(coords.shape[-1])])[
                ..., 0
            ]
            .astype("O")
            .flatten()
        )  # array of tuples

        random_coords = np.random.choice(coords, int(self.density * indices[0].size))
        return list(zip(*random_coords))  # [(row_indices, col_indices)]

    def get_delay_as_index(self):
        d_int = np.trunc(self.delay)
        d_fraction = self.delay - d_int
        return d_int, d_fraction

    def get_topology(self):
        return self.__topology

    def broadcast(self):
        # TODO
        pass

    def get_synapse_mat(
        self,
        mode="uniform",
        scale=None,
        density=None,
        only_enabled=True,
        clone_along_first_axis=False,
        plot=False,
        kwargs={},
        args=[],
    ):
        assert not clone_along_first_axis

        result = self._get_mat(
            mode=mode,
            dim=(self.get_synapse_mat_dim()),
            scale=scale,
            density=density,
            plot=plot,
            kwargs=kwargs,
            args=args,
        )

        mask = np.ones((self.get_synapse_mat_dim()), dtype=bool)
        mask[self.__topology] = False
        result[mask] = 0.0

        if only_enabled:
            result *= self.enabled

        result = sp.dok_array(result, dtype=np.float64)

        del mask
        import gc

        gc.collect()

        return result


class StructuredSynapseGroup(TopologicalSynapseGroup):
    def __init__(
        self,
        src,
        dst,
        net,
        receptive_field,
        stride=1,
        padding=False,
        max_delay=1,
        tag=None,
        behaviour={},
    ):
        super().__init__(src, dst, net, max_delay, tag, behaviour)

        if isinstance(receptive_field, int):
            self.receptive_field = (receptive_field, 1, 1)
        if len(receptive_field) == 1:
            self.receptive_field = (receptive_field[0], 1, 1)
        if len(receptive_field) == 2:
            self.receptive_field = (receptive_field[0], receptive_field[1], 1)
        else:
            self.receptive_field = receptive_field

        self.stride = stride
        self.padding = padding

    def get_synapse_mat_dim(self):
        raise NotImplementedError


class LocalSynapseGroup(StructuredSynapseGroup):
    def __init__(
        self,
        src,
        dst,
        net,
        receptive_field,
        stride=1,
        padding=False,
        weight_init_mode="normal(0.5, 0.5)",
        delay_init_mode="zeros()",
        max_delay=1,
        tag=None,
        behaviour={},
    ):
        super().__init__(
            src, dst, net, receptive_field, stride, padding, max_delay, tag, behaviour
        )

        self.add_tag("local")

        self.add_behaviours_to_object(
            {1: WeightInitializer(init_mode=weight_init_mode)}, self
        )
        self.add_behaviours_to_object(
            {2: DelayInitializer(init_mode=delay_init_mode)}, self
        )

    def get_topology(self):
        # TODO
        x_range = (
            self.dst.x * np.arange(0, self.receptive_field[0], dtype=int)
            + self.dst.x * self.stride
        ).ravel()
        y_range = (
            self.dst.y * np.arange(0, self.receptive_field[1], dtype=int)
            + self.dst.y * self.stride
        ).ravel()
        z_range = (
            self.dst.z * np.arange(0, self.receptive_field[2], dtype=int)
            + self.dst.z * self.stride
        ).ravel()

        return x_range, y_range, z_range

    def get_delay_as_index(self):
        # TODO
        pass

    def broadcast(self):
        # TODO
        pass

    def get_synapse_mat_dim(self):
        # TODO
        return np.prod(self.receptive_field), self.dst.size


class ConvSynapseGroup(StructuredSynapseGroup):
    def __init__(
        self,
        src,
        dst,
        net,
        receptive_field,
        stride=1,
        padding=False,
        weight_init_mode="normal(0.5, 0.5)",
        delay_init_mode="zeros()",
        max_delay=1,
        tag=None,
        behaviour={},
    ):
        super().__init__(
            src,
            dst,
            net,
            receptive_field,
            stride,
            padding,
            max_delay,
            tag,
            behaviour,
        )

        self.add_tag("conv")

        self.add_behaviours_to_object(
            {1: WeightInitializer(init_mode=weight_init_mode)}, self
        )
        self.add_behaviours_to_object(
            {2: DelayInitializer(init_mode=delay_init_mode)}, self
        )

    def get_topology(self):
        # TODO
        x_range = (
            self.dst.x * np.arange(0, self.receptive_field[0], dtype=int)
            + self.dst.x * self.stride
        ).ravel()
        y_range = (
            self.dst.y * np.arange(0, self.receptive_field[1], dtype=int)
            + self.dst.y * self.stride
        ).ravel()
        z_range = (
            self.dst.z * np.arange(0, self.receptive_field[2], dtype=int)
            + self.dst.z * self.stride
        ).ravel()

        return x_range, y_range, z_range

    def get_delay_as_index(self):
        d_int = np.trunc(self.delay)
        d_fraction = self.delay - d_int
        return d_int, d_fraction

    def broadcast(self):
        # TODO
        pass

    def get_synapse_mat_dim(self):
        return self.receptive_field
