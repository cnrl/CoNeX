import warnings
from CorticalColumn.nn.Structure.Layers import Layer
from CorticalColumn.nn.Modules.topological_connections import StructuredSynapseGroup

from pymonntorch import SynapseGroup, NeuronGroup, TaggableObject

# TODO: handle multi-scale


class CorticalColumn(TaggableObject):
    def __init__(
        self,
        net,
        sensory_layer,
        location_layer=None,
        representation_layer=None,
        motor_layer=None,
        L2_3_config=None,
        L4_config=None,
        L5_config=None,
        L6_config=None,
        sensory_L4_syn_config=None,
        sensory_L6_syn_config=None,
        location_L6_syn_config=None,
        L6_L4_syn_config=None,
        L4_L2_3_syn_config=None,
        L2_3_L5_syn_config=None,
        L5_L2_3_syn_config=None,
        L5_L6_syn_config=None,
        L2_3_representation_syn_config=None,
        L5_motor_syn_config=None,
    ):
        super().__init__("CorticalColumn" + len(net.cortical_column), device=net.device)
        self.network = net

        self.sensory_layer = sensory_layer
        self.location_layer = location_layer
        self.representation_layer = representation_layer
        self.motor_layer = motor_layer

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

        self.sensory_L4_synapses = self._add_synaptic_connection(
            self.sensory_layer, self.L4, sensory_L4_syn_config
        )

        self.sensory_L6_synapses = self._add_synaptic_connection(
            self.sensory_layer, self.L6, sensory_L6_syn_config
        )

        self.location_L6_synapses = self._add_synaptic_connection(
            self.location_layer, self.L6, location_L6_syn_config
        )

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

        self.L2_3_representation_synapses = self._add_synaptic_connection(
            self.L2_3, self.representation_layer, L2_3_representation_syn_config
        )

        self.L5_motor_synapses = self._add_synaptic_connection(
            self.L5, self.motor_layer, L5_motor_syn_config
        )

        self.network.columns.append(self)

    def _create_layer(self, net, config, name):
        if config:
            return Layer(name, net, self, **config)
        else:
            return None

    @classmethod
    def _add_synaptic_connection(cls, src, dst, config):
        if src is None or dst is None:
            return {}

        net = src.network

        if not isinstance(config, dict):
            raise ValueError("Synaptic connection config must be a dictionary.")

        synapses = {}
        for key in config:
            tag = src.tags[0] + " => " + dst.tags[0] + " : " + key
            if isinstance(config[key], dict):
                if isinstance(src, NeuronGroup):
                    src_pop = src
                else:
                    src_pop = getattr(src, config[key]["src"])

                if isinstance(dst, NeuronGroup):
                    dst_pop = dst
                else:
                    dst_pop = getattr(dst, config[key]["dst"])

                if not config[key].get("user_defined", False):
                    synapses[key] = StructuredSynapseGroup(
                        src=src_pop, dst=dst_pop, net=net, **config[key]
                    )
                else:
                    synapses[key] = SynapseGroup(
                        src=src_pop, dst=dst_pop, net=net, **config[key]
                    )

                synapses[key].tags.insert(0, tag)
                try:
                    if src_pop.cortical_column == dst_pop.cortical_column:
                        synapses[key].add_tag("Distal")
                    else:
                        synapses[key].add_tag("Apical")
                except AttributeError:
                    synapses[key].add_tag("Distal")
            elif isinstance(config[key], SynapseGroup) and config[key].network == net:
                synapses[key] = config[key]

                synapses[key].tags.insert(0, tag)
                synapses[key].add_tag("Distal")
            else:
                warnings.warn(
                    f"Ignoring connection {key} from {src.tags[0]} to {dst.tags[0]}..."
                )
        return synapses

    def connect(
        self,
        cortical_column,
        L2_3_L2_3_config=None,
        L2_3_L4_config=None,
        L5_L5_config=None,
        L5_L6_config=None,
    ):
        synapses = {}
        tag = self.tags[0] + "_" + "L2_3 => " + cortical_column.tags[0] + "_L2_3"
        synapses[tag] = self._add_synaptic_connection(
            self.L2_3, cortical_column.L2_3, L2_3_L2_3_config
        )
        all_empty = synapses[tag] == {}

        tag = self.tags[0] + "_" + "L2_3 => " + cortical_column.tags[0] + "_L4"
        synapses[tag] = self._add_synaptic_connection(
            self.L2_3, cortical_column.L4, L2_3_L4_config
        )
        all_empty *= synapses[tag] == {}

        tag = self.tags[0] + "_" + "L5 => " + cortical_column.tags[0] + "_L5"
        synapses[tag] = self._add_synaptic_connection(
            self.L5, cortical_column.L5, L5_L5_config
        )
        all_empty *= synapses[tag] == {}

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
