import warnings
from CorticalColumn.nn.Layers import Layer
from CorticalColumn.nn.Modules.topological_connections import StructuredSynapseGroup

from pymonntorch import SynapseGroup

# TODO: handle multi-scale

class CorticalColumn:
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
        self.network = net

        self.sensory_layer = sensory_layer
        self.location_layer = location_layer
        self.representation_layer = representation_layer
        self.motor_layer = motor_layer

        self.L2_3 = self._create_layer(net, L2_3_config)
        self.L4 = self._create_layer(net, L4_config)
        self.L5 = self._create_layer(net, L5_config)
        self.L6 = self._create_layer(net, L6_config)

        if self.L6 is None and self.L4 is None:
            raise RuntimeError("At least one of L4 and L6 must be defined in a cortical column.")
        if self.L2_3 is None and self.L4 is None and self.L5 is None and self.L6 is None:
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

    @classmethod
    def _create_layer(cls, net, config):
        if config:
            return Layer(net, **config)
        else:
            return None

    @classmethod
    def _add_synaptic_connection(cls, src, dst, config):  # TODO
        net = src.network
        if isinstance(config, dict):
            if not config.get("user_defined", False):
                syn_type = StructuredSynapseGroup
            else:
                syn_type = SynapseGroup
        else:
            warnings.warn(f"No synaptic connection from {src} to {dst} in {cls.__name__}")
            return None
        return syn_type(src, dst, net, config)

    def add_layer(self, layer, name):
        if name not in ["L2_3", "L4", "L5", "L6"]:
            raise AttributeError("Invalid cortical layer name:", name)
        if not isinstance(layer, Layer):
            raise ValueError("Argument layer must be of type CCSNN.nn.Layer.")
        if hasattr(self, name):
            warnings.warn(f"{name} is being redefined...")
        setattr(self, name, layer)
