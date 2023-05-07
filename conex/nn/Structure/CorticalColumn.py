import warnings
from conex.nn.Modules.io import OutputLayer
from conex.nn.Structure.Layers import Layer
from conex.nn.Modules.topological_connections import StructuredSynapseGroup

from pymonntorch import SynapseGroup, NeuronGroup, TaggableObject

# TODO: handle multi-scale
# TODO: str.removesuffix needs python 3.9


class CorticalColumn(TaggableObject):
    """
    Base class to define a single cortical column.

    Note: 1) All layer configurations are designed for SpikingNeuronGroup. To define a NeuronGroup with your behaviors of favor,
            add "user_defined" key to your config dict with value `True` and specify the parameters of the NeuronGroup (excluding
            the network) as key-values in the config dict.

          2) In the layer connections config dicts, the key is the name of synapse between the populations in the corresponding layers
            and the values are the synaptic config dicts.

    Args:
        net (Neocortex): The cortical network the column belongs to.
        L2_3_config (dict): If not None, adds L2/3 with the specified configurations to the column.
        L4_config (dict): If not None, adds L4 with the specified configurations to the column.
        L5_config (dict): If not None, adds L5 with the specified configurations to the column.
        L6_config (dict): If not None, adds L6 with the specified configurations to the column.
        L6_L4_syn_config (dict): If not None, adds the synaptic connections from L6 to L4 with the specified configurations.
        L4_L2_3_syn_config (dict): If not None, adds the synaptic connections from L4 to L2/3 with the specified configurations.
        L2_3_L5_syn_config (dict): If not None, adds the synaptic connections from L2/3 to L5 with the specified configurations.
        L5_L2_3_syn_config (dict): If not None, adds the synaptic connections from L5 to L2/3 with the specified configurations.
        L5_L6_syn_config (dict): If not None, adds the synaptic connections from L5 to L6 with the specified configurations.
    """

    def __init__(
        self,
        net,
        L2_3_config=None,
        L4_config=None,
        L5_config=None,
        L6_config=None,
        L6_L4_syn_config=None,
        L4_L2_3_syn_config=None,
        L2_3_L5_syn_config=None,
        L5_L2_3_syn_config=None,
        L5_L6_syn_config=None,
    ):
        super().__init__(f"CorticalColumn{len(net.columns)}", device=net.device)
        self.network = net

        self.L2_3 = self._create_layer(net, L2_3_config, "L2_3")
        self.L4 = self._create_layer(net, L4_config, "L4")
        self.L5 = self._create_layer(net, L5_config, "L5")
        self.L6 = self._create_layer(net, L6_config, "L6")

        if self.L6 is None and self.L4 is None:
            raise RuntimeError(
                "At least one of L4 and L6 must be defined in a cortical column."
            )
        if (
            self.L2_3 is None
            and self.L4 is None
            and self.L5 is None
            and self.L6 is None
        ):
            raise RuntimeError("No layers defined in the cortical column.")

        self.L6_L4_synapses = self._add_synaptic_connection(
            self.L6, self.L4, L6_L4_syn_config
        )

        self.L4_L2_3_synapses = self._add_synaptic_connection(
            self.L4, self.L2_3, L4_L2_3_syn_config
        )

        self.L2_3_L5_synapses = self._add_synaptic_connection(
            self.L2_3, self.L5, L2_3_L5_syn_config
        )

        self.L5_L2_3_synapses = self._add_synaptic_connection(
            self.L5, self.L2_3, L5_L2_3_syn_config
        )

        self.L5_L6_synapses = self._add_synaptic_connection(
            self.L5, self.L6, L5_L6_syn_config
        )

        self.network.columns.append(self)

    def _create_layer(self, net, config, name):
        if config:
            return Layer(name, net, self, **config)
        else:
            return None

    @classmethod
    def _add_synaptic_connection(cls, src, dst, config):
        # src and dst can either be Layer or NeuronGroup
        if src is None or dst is None:
            return {}

        net = src.network

        if not isinstance(config, dict):
            raise ValueError("Synaptic connection config must be a dictionary.")

        synapses = {}
        for key in config:
            src_tag = src.tags[0]
            if pop_tag := config[key].get("src_pop"):
                src_tag = src_tag + "_" + pop_tag.removesuffix("_pop")
            dst_tag = dst.tags[0]
            if pop_tag := config[key].get("dst_pop"):
                dst_tag = dst_tag + "_" + pop_tag.removesuffix("_pop")
            tag = f"{src_tag} => {dst_tag}"

            if isinstance(config[key], dict):
                if isinstance(src, NeuronGroup):
                    src_pop = src
                else:
                    src_pop = getattr(src, config[key].pop("src_pop"))

                if isinstance(dst, NeuronGroup):
                    dst_pop = dst
                else:
                    dst_pop = getattr(dst, config[key].pop("dst_pop"))

                synapses[key] = StructuredSynapseGroup(
                    src=src_pop, dst=dst_pop, net=net, **config[key]
                )

                if tag in synapses[key].tags:
                    synapses[key].tags.remove(tag)
                synapses[key].tags.insert(0, tag)
                synapses[key].add_tag(key)

                if not (
                    config[key].get("tag", None)
                    and any(
                        connection
                        in list(map(str.strip, config[key]["tag"].split(",")))
                        for connection in ["Proximal", "Distal", "Apical"]
                    )
                ):
                    if hasattr(dst, "cortical_column"):
                        if src.cortical_column == dst.cortical_column:
                            if "L4" in src_tag and "L2_3" in dst_tag:
                                synapses[key].add_tag("Proximal")
                            else:
                                synapses[key].add_tag("Distal")
                        else:
                            synapses[key].add_tag("Apical")
                    elif isinstance(dst, OutputLayer):
                        synapses[key].add_tag("Proximal")
                    else:
                        raise ValueError(f"Invalid destination object: {type(dst_pop)}")
            else:
                warnings.warn(
                    f"Ignoring connection {key} from {src.tags[0]} to {dst.tags[0]}..."
                )
        return synapses

    def connect_column(
        self,
        cortical_column,
        L2_3_L2_3_config=None,
        L2_3_L4_config=None,
        L5_L5_config=None,
        L5_L6_config=None,
    ):
        """
        Makes connections between current cortical column and another one.

        Note: In the config dicts, the key is the name of synapse between the populations in the corresponding layers
                and the values are the synaptic config dicts.

        Args:
            cortical_column (CorticalColumn): The column to connect to.
            L2_3_L2_3_config (dict): Adds the synaptic connections from L2/3 of current column to L2/3 of the other with the specified configurations.
            L2_3_L4_config (dict): Adds the synaptic connections from L2/3 of current column to L4 of the other with the specified configurations.
            L5_L5_config (dict): Adds the synaptic connections from L5 of current column to L5 of the other with the specified configurations.
            L6_L6_config (dict): Adds the synaptic connections from L6 of current column to L6 of the other with the specified configurations.
        """
        synapses = {}
        all_empty = True

        if L2_3_L2_3_config:
            tag = self.tags[0] + "_" + "L2_3 => " + cortical_column.tags[0] + "_L2_3"
            synapses[tag] = self._add_synaptic_connection(
                self.L2_3, cortical_column.L2_3, L2_3_L2_3_config
            )
            all_empty *= synapses[tag] == {}

        if L2_3_L4_config:
            tag = self.tags[0] + "_" + "L2_3 => " + cortical_column.tags[0] + "_L4"
            synapses[tag] = self._add_synaptic_connection(
                self.L2_3, cortical_column.L4, L2_3_L4_config
            )
            all_empty *= synapses[tag] == {}

        if L5_L5_config:
            tag = self.tags[0] + "_" + "L5 => " + cortical_column.tags[0] + "_L5"
            synapses[tag] = self._add_synaptic_connection(
                self.L5, cortical_column.L5, L5_L5_config
            )
            all_empty *= synapses[tag] == {}

        if L5_L6_config:
            tag = self.tags[0] + "_" + "L5 => " + cortical_column.tags[0] + "_L6"
            synapses[tag] = self._add_synaptic_connection(
                self.L5, cortical_column.L6, L5_L6_config
            )
            all_empty *= synapses[tag] == {}

        if all_empty:
            raise RuntimeError(
                f"No synaptic connections formed between {self.tags[0]}, {cortical_column.tags[0]}"
            )

        return synapses

    def connect2output(
        self, other, L2_3_representation_syn_config=None, L5_motor_syn_config=None
    ):
        synapses = {}
        if L2_3_representation_syn_config:
            if hasattr(other, "representation_pop") and hasattr(self, "L2_3"):
                synapses["L2_3_representation_synapse"] = self._add_synaptic_connection(
                    self.L2_3, other.representation_pop, L2_3_representation_syn_config
                )

        if L5_motor_syn_config:
            if hasattr(other, "motor_pop") and hasattr(self, "L5"):
                synapses["L5_motor_synapse"] = self._add_synaptic_connection(
                    self.L5, other.motor_pop, L5_motor_syn_config
                )

        return synapses

    def connect(self, other, **kwargs):
        if isinstance(other, CorticalColumn):
            return self.connect_column(other, **kwargs)
        elif isinstance(other, OutputLayer):
            return self.connect2output(other, **kwargs)
        else:
            raise RuntimeError(f"Not supported object {other} to connect.")
