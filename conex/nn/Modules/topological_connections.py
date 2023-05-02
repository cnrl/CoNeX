"""
Structured SynapseGroup connection schemes.
"""

from conex.behaviors.synapses import learning, dendrites
from pymonntorch import SynapseGroup

from conex.behaviors.synapses import (
    DelayInitializer,
    SynapseInit,
    WeightClip,
    WeightInitializer,
    WeightNormalization,
)

from conex.nn.priorities import SYNAPSE_PRIORITIES

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
        weight_init_params,
        structure,
        structure_params,
        learning_rule=None,
        learning_params=None,
        src_delay_init_mode=None,
        src_delay_init_params=None,
        dst_delay_init_mode=None,
        dst_delay_init_params=None,
        w_interval=None,
        weight_norm=None,
        tag=None,
        user_defined=None,
    ):
        assert net is not None, "net cannot be None."

        if tag is None:
            tag = f"StructuredSynapseGroup_{len(net.SynapseGroups) + 1}"

        behavior = {
            SYNAPSE_PRIORITIES["Init"]: SynapseInit(),
            SYNAPSE_PRIORITIES["WeightInit"]: WeightInitializer(**weight_init_params),
        }

        if w_interval is not None:
            behavior[SYNAPSE_PRIORITIES["WeightClip"]] = WeightClip(
                w_min=w_interval[0], w_max=w_interval[1]
            )

        if src_delay_init_mode is not None:
            behavior[SYNAPSE_PRIORITIES["SrcDelayInit"]] = DelayInitializer(
                mode=src_delay_init_mode, **src_delay_init_params
            )

        if dst_delay_init_mode is not None:
            behavior[SYNAPSE_PRIORITIES["DstDelayInit"]] = DelayInitializer(
                mode=dst_delay_init_mode, **dst_delay_init_params
            )

        if weight_norm is not None:
            behavior[SYNAPSE_PRIORITIES["WeightNormalization"]] = WeightNormalization(
                norm=weight_norm
            )

        if learning_rule is not None:
            if isinstance(learning_rule, str) and isinstance(structure, str):
                learning_rule = getattr(learning, structure + learning_rule)
                behavior[SYNAPSE_PRIORITIES["LearningRule"]] = learning_rule(
                    **learning_params
                )
            else:
                behavior[SYNAPSE_PRIORITIES["LearningRule"]] = learning_rule(
                    **learning_params
                )

        if isinstance(structure, str):
            structure += "DendriticInput"
            behavior[SYNAPSE_PRIORITIES["DendriticInput"]] = getattr(
                dendrites, structure
            )(**structure_params)
        else:
            behavior[SYNAPSE_PRIORITIES["DendriticInput"]] = structure(
                **structure_params
            )

        if user_defined is not None:
            behavior.update(user_defined)

        super().__init__(src, dst, net, tag, behavior)
