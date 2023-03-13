"""
Implementation of Cortical layers.
"""

# TODO: Complete implementation of each layer, including the undefined methods and attributes.


import warnings
from CorticalColumn.nn.Modules.spiking_neurons import SpikingNeuronGroup
from CorticalColumn.nn.Modules.topological_connections import StructuredSynapseGroup
from pymonntorch import NeuronGroup, SynapseGroup


class Layer:
    def __init__(
        self,
        net,
        exc_pop_config=None,
        inh_pop_config=None,
        exc_exc_config=None,
        exc_inh_config=None,
        inh_exc_config=None,
        inh_inh_config=None,
    ):
        self.network = net
        
        self.exc_pop = self._create_neural_population(net, exc_pop_config)
        if self.exc_pop and "exc" not in self.exc_pop.tags:
                self.exc_pop.add_tag("exc")

        self.inh_pop = self._create_neural_population(net, inh_pop_config)
        if self.inh_pop and "exc" not in self.inh_pop.tags:
                self.inh_pop.add_tag("inh")

        if self.inh_pop is None and self.exc_pop is None:
            raise RuntimeError(f"No proper neural population defined for {self.__class__.__name__}")
        
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
                raise RuntimeError(f"No connection between Excitatory and Inhibitory populations in {self.__class__.__name__}")
            
    @classmethod
    def _create_neural_population(cls, net, config):
        if  isinstance(config, dict):
            if not config.get("user_defined", False):
                return SpikingNeuronGroup(net=net, **config)
            else:
                return NeuronGroup(net=net, **config)
        else:
            warnings.warn(f"No proper neural population is defined in {cls.__name__}.")
            return None
        
    @classmethod
    def _create_synaptic_connection(cls, src, dst, net, config):
        if isinstance(config, dict):
            if not config.get("user_defined", False):
                syn = StructuredSynapseGroup(src, dst, net, config)
            else:
                syn = SynapseGroup(src, dst, net, config)
            syn.add_tag("Proximal")
            return syn
        else:
            warnings.warn(f"No synaptic connection from {src} to {dst} in {cls.__name__}")
            return None
