"""
Structured SynapseGroup connection schemes.
"""

from CorticalColumn.behaviours.synapses import learning, dendrites
from pymonntorch import SynapseGroup

from CorticalColumn.behaviours.synapses.specs import (
    DelayInitializer,
    SynapseInit,
    WeightClip,
    WeightInitializer,
)

# TODO: define delay range
# TODO: should we add (read) structure and learning rule to (from) tags?


class StructuredSynapseGroup(SynapseGroup):
    def __init__(
        self,
        src,
        dst,
        net,
        structure="Simple",
        learning_rule="STDP",
        weight_init_mode=None,
        delay_init_mode=None,
        w_min=0.0,
        w_max=1.0,
        weight_norm=None,
        tag=None,
        delay_params={},
        weight_init_params={},
        structure_params={},
        learning_params={},
    ):
        assert net is not None, "net cannot be None."

        if tag is None:
            tag = f"StructuredSynapseGroup_{len(net.synapseGroups) + 1}"

        behavior = {
            12: SynapseInit(),
            13: WeightInitializer(mode=weight_init_mode, **weight_init_params),
            14: DelayInitializer(mode=delay_init_mode, **delay_params),
            51: WeightNormalization(norm=weight_norm),
            52: WeightClip(w_min=w_min, w_max=w_max),
        }

        learning_rule = structure + learning_rule
        behavior[50] = getattr(learning, learning_rule)(**learning_params)

        structure += "DendriteInput"
        behavior[27] = getattr(dendrites, structure)(**structure_params)

        super().__init__(src, dst, net, tag, behavior)
