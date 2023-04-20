"""
Implementation of Cortical layers.
"""

import warnings
from CCSNN.nn.Modules.spiking_neurons import SpikingNeuronGroup
from CCSNN.nn.Modules.topological_connections import StructuredSynapseGroup
from pymonntorch import NeuronGroup, SynapseGroup, TaggableObject


class Layer(TaggableObject):
    """
    Base class to create a cortical layer.

    Note: 1) All config dicts are designed for SpikingNeuronGroup/StructuredSynapseGroup. To define a NeuronGroup/SynapseGroup
            with your behaviors of favor, add "user_defined" key to your config dict with value `True` and specify the parameters of
            the NeuronGroup/SynapseGroup (excluding the network) as key-values in the config dict.

          2) The "size" key is mandatory for a neural population config dict.

    Args:
        tag (str): The name of the layer. Valid values are:
                    "L2_3", "L4", "L5", "L6".
        net (Neocortex): The cortical network the layer belongs to.
        cortical_column (CorticalColumn): The cortical column the layer belongs to.
        exc_pop_config (dict): If not None, defines an excitatory population in the layer with the given config.
        inh_pop_config (dict): If not None, defines an inhibitory population in the layer with the given config.
        exc_exc_config (dict): If not None, defines an exc -> exc synaptic connection within the layer with the given config.
        exc_inh_config (dict): If not None, defines an exc -> inh synaptic connection within the layer with the given config.
        inh_exc_config (dict): If not None, defines an inh -> exc synaptic connection within the layer with the given config.
        inh_inh_config (dict): If not None, defines an inh -> inh synaptic connection within the layer with the given config.
    """

    def __init__(
        self,
        tag,
        net,
        cortical_column,
        exc_pop_config=None,
        inh_pop_config=None,
        exc_exc_config=None,
        exc_inh_config=None,
        inh_exc_config=None,
        inh_inh_config=None,
    ):
        super().__init__(f"{cortical_column.tags[0]}_{tag}", device=net.device)
        self.network = net
        self.cortical_column = cortical_column

        self.exc_pop = self._create_neural_population(
            net, exc_pop_config, self.tags[0] + "_exc"
        )
        if self.exc_pop and "exc" not in self.exc_pop.tags:
            self.exc_pop.add_tag("exc")

        self.inh_pop = self._create_neural_population(
            net, inh_pop_config, self.tags[0] + "_inh"
        )
        if self.inh_pop and "exc" not in self.inh_pop.tags:
            self.inh_pop.add_tag("inh")

        if self.inh_pop is None and self.exc_pop is None:
            raise RuntimeError(
                f"No proper neural population defined for {self.tags[0]}"
            )

        if self.exc_pop:
            self.exc_exc_syn = self._create_synaptic_connection(
                self.exc_pop, self.exc_pop, net, exc_exc_config
            )

        if self.inh_pop:
            self.inh_inh_syn = self._create_synaptic_connection(
                self.inh_pop, self.inh_pop, net, inh_inh_config
            )

        if self.inh_pop and self.exc_pop:
            self.exc_inh_syn = self._create_synaptic_connection(
                self.exc_pop, self.inh_pop, net, exc_inh_config
            )

            self.inh_exc_syn = self._create_synaptic_connection(
                self.inh_pop, self.exc_pop, net, inh_exc_config
            )

            if self.inh_exc_syn is None and self.exc_inh_syn is None:
                raise RuntimeError(
                    f"No connection between Excitatory and Inhibitory populations in {self.tags[0]}"
                )

    @staticmethod
    def _create_neural_population(net, config, tag):
        if isinstance(config, dict):
            if not config.get("user_defined", False):
                if config.get("tag", None) is None:
                    config["tag"] = tag
                else:
                    if isinstance(config["tag"], str):
                        config["tag"] = tag + "," + config["tag"]
                    else:
                        config["tag"].insert(0, tag)
                return SpikingNeuronGroup(net=net, **config)
            else:
                return NeuronGroup(net=net, **config)
        else:
            warnings.warn(f"No proper neural population is defined in {tag}.")
            return None

    @staticmethod
    def _create_synaptic_connection(src, dst, net, config):
        if isinstance(config, dict):
            if not config.get("user_defined", False):
                syn = StructuredSynapseGroup(src, dst, net, **config)
                syn.tags.insert(0, Layer._create_synaptic_tag(src.tags[0], dst.tags[0]))
            else:
                syn = SynapseGroup(src, dst, net, config)
            syn.add_tag("Proximal")
            return syn
        else:
            warnings.warn(
                f"No synaptic connection from {src.tags[0]} to {dst.tags[0]}."
            )
            return None

    @staticmethod
    def _create_synaptic_tag(src_tag, dst_tag):
        src_tag = list(filter(None, src_tag.split("_")))
        dst_tag = list(filter(None, dst_tag.split("_")))

        if len(src_tag) < 3 or len(dst_tag) < 3:
            return f"{'_'.join(src_tag)} => {'_'.join(dst_tag)}"

        src_col, src_layer, src_pop = src_tag[0], "_".join(src_tag[1:-1]), src_tag[-1]
        dst_col, dst_layer, dst_pop = dst_tag[0], "_".join(dst_tag[1:-1]), dst_tag[-1]

        if src_col == dst_col:
            prefix = src_col
            if src_layer == dst_layer:
                prefix = f"{prefix}_{src_layer}"
                return f"{prefix}: {src_pop} => {dst_pop}"
            else:
                return f"{prefix}: {src_layer}_{src_pop} => {dst_layer}_{dst_pop}"
        else:
            return f"{src_tag} => {dst_tag}"
