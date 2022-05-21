"""
Convolution SynapseGroup connection scheme.
"""

import numpy as np
import scipy.sparse as sp
from PymoNNto import SynapseGroup


class ConvolutionSynapseGroup(SynapseGroup):
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
        tag=None,
        behaviour={},
    ):
        if tag is None and net is not None:
            tag = "ConvolutionSynapseGroup_" + str(len(net.SynapseGroups) + 1)

        super().__init__(src, dst, net, tag, behaviour)

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
        self.topology = self._get_topology()

        self.weights = self.get_synapse_mat(mode=weight_init_mode)
        self.delays = self.get_synapse_mat(mode=delay_init_mode)

    def _get_topology(self):
        mask = sp.lil_array((self.post.size, self.src.size), dtype=bool)
        for i in range(self.post.size):
            x = self.post.x[i]
            y = self.post.y[i]
            z = self.post.z[i]
            for k in range(self.receptive_field[2]):
                for j in range(self.receptive_field[1]):
                    ind = (
                        x * self.stride
                        + (y + j) * self.post.height
                        + (z + k) * self.post.depth
                    )
                    mask[i, ind : ind + self.receptive_field[0]] = True
        return mask.tobsr(blocksize=(1, self.receptive_field[1]), copy=False)

    def _get_mat(
        self, mode, dim, scale=None, density=None, plot=False, kwargs={}, args=[]
    ):
        result = sp.dok_array((self.post.size, self.pre.size))

        if mode == "random" or mode == "rand" or mode == "rnd":
            mode = "uniform"

        if type(mode) == int or type(mode) == float:
            mode = "ones()*" + str(mode)

        if "(" not in mode and ")" not in mode:
            mode += "()"

        if mode not in self._mat_eval_dict:
            if "zeros" in mode or "ones" in mode:
                a1 = "shape=dim"
            else:
                a1 = "size=dim"
            if "()" in mode:  # no arguments => no comma
                ev_str = mode.replace(")", "*args," + a1 + ",**kwargs)")
            else:
                if args != []:
                    print(
                        "Warning: args cannot be used when arguments are passed as strings"
                    )
                ev_str = mode.replace(")", "," + a1 + ",**kwargs)")

            self._mat_eval_dict[mode] = compile(ev_str, "<string>", "eval")

        result[self.topology[0], self.topology[1]] = eval(self._mat_eval_dict[mode])

        if scale is not None:
            result *= scale

        if plot:
            import matplotlib.pyplot as plt

            plt.hist(result.flatten(), bins=30)
            plt.show()

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
        result = self._get_mat(
            mode=mode,
            dim=int(self.density * self.post.size * self.pre.size),
            scale=scale,
            density=density,
            plot=plot,
            kwargs=kwargs,
            args=args,
        )

        if only_enabled:
            result *= self.enabled

        return result.astype(np.float64)
