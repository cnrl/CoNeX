from conex.behaviors.neurons.neuron_types.lif_neurons import LIF, ELIF
from conex.nn.Config.layer_config import LayerConfig
from pymonntorch import *


class l4(LayerConfig):
    exc_size = (4, 25, 25)
    exc_neuron_params = {
        "R": 20,
        "tau": 25,
        "threshold": -37,
        "v_reset": -75,
        "v_rest": -65,
    }
    exc_neuron_type = LIF
    exc_tau_trace = 2.7
    exc_fire = True
    exc_dendrite_params = {"distal_provocativeness": 0.5}

    inh_size = (4, 5, 5)
    inh_neuron_params = {
        "R": 20,
        "tau": 25,
        "threshold": -37,
        "v_reset": -75,
        "v_rest": -65,
        "delta": 0.8,
        "theta_rh": -42,
    }
    inh_neuron_type = ELIF
    inh_tau_trace = 2.7
    inh_fire = True
    inh_dendrite_params = {"distal_provocativeness": 0.5}

    exc_exc_weight_init_params = {"mode": "uniform"}
    exc_exc_structure = "Simple"
    exc_exc_structure_params = {"current_coef": 3}

    exc_inh_weight_init_params = {"mode": "uniform"}
    exc_inh_structure = "Simple"
    exc_inh_structure_params = {"current_coef": 3}

    inh_exc_weight_init_params = {"mode": "uniform"}
    inh_exc_structure = "Simple"
    inh_exc_structure_params = {"current_coef": 3}

    inh_inh_weight_init_params = {"mode": "uniform"}
    inh_inh_structure = "Simple"
    inh_inh_structure_params = {"current_coef": 3}