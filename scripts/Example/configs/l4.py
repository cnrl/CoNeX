from CCSNN.behaviours.neurons.neuron_types.lif_neurons import LIF, ELIF
from pymonntorch import *

l4 = {"exc_pop_config": {"size": NeuronDimension(depth=4, height=25, width=25),
                                  "neuron_type": LIF,
                                  "dendrite_params": {"distal_provocativeness": 0.5},
                                  "tau_trace": 2.7,
                                  "neuron_params": {"R":20, 
                                                    "tau":25, 
                                                    "threshold":-37, 
                                                    "v_reset":-75, 
                                                    "v_rest":-65, 
                                                    },
                                  },
               }
