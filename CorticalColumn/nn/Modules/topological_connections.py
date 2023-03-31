"""
Structured SynapseGroup connection schemes.
"""

from CorticalColumn.behaviours.synapses import learning, dendrites
from pymonntorch import SynapseGroup

from CorticalColumn.behaviours.synapses import (
    DelayInitializer,
    SynapseInit,
    WeightClip,
    WeightInitializer,
    WeightNormalization,
)

from CorticalColumn.nn.timestamps import SYNAPSE_TIMESTAMPS

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
        src_delay_init_mode=None,
        dst_delay_init_mode=None,
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
            SYNAPSE_TIMESTAMPS['Init']: SynapseInit(),
            SYNAPSE_TIMESTAMPS['WeightInit']: WeightInitializer(mode=weight_init_mode, **weight_init_params),
            SYNAPSE_TIMESTAMPS['WeightClip']: WeightClip(w_min=w_min, w_max=w_max),
        }

        if src_delay_init_mode is not None:
            SYNAPSE_TIMESTAMPS['SrcDelayInit']: DelayInitializer(mode=src_delay_init_mode, **delay_params)

        if dst_delay_init_mode is not None:
            SYNAPSE_TIMESTAMPS['DstDelayInit']: DelayInitializer(mode=dst_delay_init_mode, **delay_params)

        if weight_norm is not None:
            SYNAPSE_TIMESTAMPS['WeightNormalization']: WeightNormalization(norm=weight_norm)

        if learning_rule is not None:
            learning_rule = structure + learning_rule
            behavior[SYNAPSE_TIMESTAMPS['LearningRule']] = getattr(learning, learning_rule)(**learning_params)

        structure += "DendriteInput"
        behavior[SYNAPSE_TIMESTAMPS['DendriteInput']] = getattr(dendrites, structure)(**structure_params)

        super().__init__(src, dst, net, tag, behavior)
