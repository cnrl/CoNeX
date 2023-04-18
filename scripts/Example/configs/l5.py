from CCSNN.behaviours.neurons.neuron_types.lif_neurons import ELIF
from pymonntorch import *

l5 = {"exc_pop_config": {"size": NeuronDimension(depth=3, height=10, width=10),
                                  "neuron_type": ELIF,
                                  "dendrite_params": {"distal_provocativeness": 0.5},
                                  "tau_trace": 2.7,
                                  "neuron_params": {"R":20, 
                                                    "tau":25, 
                                                    "threshold":-37, 
                                                    "v_reset":-75, 
                                                    "v_rest":-65, 
                                                    "delta": 0.8, 
                                                    "theta_rh":-42,
                                                    },
                                  },
                "inh_pop_config": {"size": NeuronDimension(depth=3, height=5, width=5),
                                  "neuron_type": ELIF,
                                  "dendrite_params": {"distal_provocativeness": 0.5},
                                  "tau_trace": 2.7,
                                  "neuron_params": {"R":20, 
                                                    "tau":25, 
                                                    "threshold":-37, 
                                                    "v_reset":-75, 
                                                    "v_rest":-65, 
                                                    "delta": 0.8, 
                                                    "theta_rh":-42,
                                                    },
                                  },
                "exc_inh_config": {"structure": "Simple",
                                   "learning_rule": None,
                                   "weight_init_params": {"mode": "uniform"},
                                   "structure_params": {"current_coef": 3},
                                  },
                "inh_exc_config": {"structure": "Simple",
                                   "learning_rule": None,
                                   "weight_init_params": {"mode": "uniform"},
                                   "structure_params": {"current_coef": 10}
                                  },
               }