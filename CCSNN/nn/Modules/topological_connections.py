"""
Structured SynapseGroup connection schemes.
"""

from CCSNN.behaviours.synapses import learning, dendrites
from pymonntorch import SynapseGroup

from CCSNN.behaviours.synapses import (
    DelayInitializer,
    SynapseInit,
    WeightClip,
    WeightInitializer,
    WeightNormalization,
)

from CCSNN.nn.timestamps import SYNAPSE_TIMESTAMPS

# TODO: should we add (read) structure and learning rule to (from) tags?


class StructuredSynapseGroup(SynapseGroup):
    """
    Simplifies defining synaptic connections with structures.

    Args:
        src (NeuronGroup): The source (pre-synaptic) population.
        dst (NeuronGroup): The destination (post-synaptic) population.
        net (Network): The network the synapses belongs to.
        structure (str): Type of synaptic structure. Valid values are:
                            "Simple", "Conv2d", "Local2d"
        learning_rule (str or Behavior): The learning rule to be applied on the synapses.
        src_delay_init_mode (str or numeric): If not None, initializes delay for source neurons' axons.
                                                The string should be torch functions that fills a tensor like:
                                                "random", "normal", "zeros", "ones", ... .
                                                In numeric case the pre-synaptic delays will be filled with that number.
        dst_delay_init_mode (str or numeric): If not None, initializes delay for destination neurons' dendrites.
                                                The string should be torch functions that fills a tensor like:
                                                "random", "normal", "zeros", "ones", ... .
                                                In numeric case the post-synaptic delays will be filled with that number.
        w_min (float): The minimum possible weight. The default is 0.0.
        w_max (float): The maximum possible weight. The default is 1.0.
        weight_norm (float): If not None, enables wight normalization with the specified norm factor.
        tag (str): The tag(s) of the synapses.
        delay_init_params (dict): Parameters (other than `mode`) for `DelayInitializer`.
        weight_init_params (dict): Parameters for `WeightInitializer`.
        structure_params (dict): Parameters for the defined DendriticInput structure.
        learning_params (dict): Parameters for the defined learning rule.
    """

    def __init__(
        self,
        src,
        dst,
        net,
        structure="Simple",
        learning_rule=None,
        src_delay_init_mode=None,
        dst_delay_init_mode=None,
        w_min=0.0,
        w_max=1.0,
        weight_norm=None,
        tag=None,
        delay_init_params={},
        weight_init_params={},
        structure_params={},
        learning_params={},
    ):
        assert net is not None, "net cannot be None."

        if tag is None:
            tag = f"StructuredSynapseGroup_{len(net.SynapseGroups) + 1}"

        behavior = {
            SYNAPSE_TIMESTAMPS["Init"]: SynapseInit(),
            SYNAPSE_TIMESTAMPS["WeightInit"]: WeightInitializer(**weight_init_params),
            SYNAPSE_TIMESTAMPS["WeightClip"]: WeightClip(w_min=w_min, w_max=w_max),
        }

        if src_delay_init_mode is not None:
            SYNAPSE_TIMESTAMPS["SrcDelayInit"]: DelayInitializer(
                mode=src_delay_init_mode, **delay_init_params
            )

        if dst_delay_init_mode is not None:
            SYNAPSE_TIMESTAMPS["DstDelayInit"]: DelayInitializer(
                mode=dst_delay_init_mode, **delay_init_params
            )

        if weight_norm is not None:
            SYNAPSE_TIMESTAMPS["WeightNormalization"]: WeightNormalization(
                norm=weight_norm
            )

        if learning_rule is not None and isinstance(learning_rule, str):
            learning_rule = getattr(learning, structure + learning_rule)
            behavior[SYNAPSE_TIMESTAMPS["LearningRule"]] = learning_rule(**learning_params)

        structure += "DendriticInput"
        behavior[SYNAPSE_TIMESTAMPS["DendriticInput"]] = getattr(dendrites, structure)(
            **structure_params
        )

        super().__init__(src, dst, net, tag, behavior)
