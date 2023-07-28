import random
from collections import namedtuple

from pymonntorch import *
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import MNIST

from conex import *
from conex.behaviors.network.neuromodulators import Dopamine
from conex.behaviors.network.payoff import Payoff
from conex.behaviors.neurons.neuron_types.lif_neurons import LIF
from conex.behaviors.synapses.specs import CurrentNormalization
from conex.helpers.transforms.encoders import Poisson
from conex.helpers.transforms.misc import *
from conex.nn.config.connection_config import (
    Layer2LayerConnectionConfig,
    Pop2LayerConnectionConfig,
)
from conex.nn.config.layer_config import LayerConfig
from visualisation import plot_conv_weight, visualise_network_structure


def to_namedtuple(name, dictionary):
    return namedtuple(name, dictionary)(**dictionary)


# parameters
NUM_ITERATION = 1_000
config = to_namedtuple('Configuration', {
    "device": 'cuda' if torch.cuda.is_available() else "cpu",
    # RuntimeError: "slow_conv2d_cpu" not implemented for 'Half'
    # DTYPE = torch.float16
    "dtype": torch.float32,
    "synapse_mode": SxD,
    "dt": 1,
    "poisson_time": 5,
    "poisson_ratio": 15 / 30,
    "silent_ratio": 1,
    "sensory_size_height": 28,
    "sensory_size_width": 28,
    "sensory_trace_tau_s": 10,
    "num_iteration": NUM_ITERATION,
    "batch_size": NUM_ITERATION // 5,
    "initial_dopamine_concentration": 0.0,
    "tau_dopamine": 2,

    "random_seed": 1998,
    "dataset_root": "~/Temp/MNIST/",
    # labels for Mnist are in (0,1..8). use None to disable feature
    "subset_selected_labels": [0, 1, 5],
    # NOTE: number of classes must is the number of output neurons in the last l4 layer
    # prefely this number must match the subset_selected_labels and your pytorch dataset.targets
    "num_classes": 3,
    "dataset_hxw": (26, 26),
    "presenting_timing_window": 15,
})


# behaviour
class AccuracyMetric(Behavior):
    def initialize(self, neurons):
        self.time_window = self.parameter('time_window', None)
        # NOTE: neurons.depth is the number of classes which network had recorded.
        self.accumulated_winner_classes = torch.zeros((neurons.depth, 1))
        self.cc_input_layer = neurons.network.input_layers[0]
        # save the behaviour in the network
        neurons.network.metric_ = self
        self.output = []

    def forward(self, neurons):
        current_timestep_winner_classes = torch.sum(
                neurons.spikes.view((neurons.depth, -1)), axis=(1,)
        )
        self.accumulated_winner_classes += torch.argmax(current_timestep_winner_classes)

        if neurons.iteration % self.time_window == 0:
            winner_class = torch.argmax(self.accumulated_winner_classes)
            self.output.append((winner_class, self.cc_input_layer.y))
            self.accumulated_winner_classes = torch.zeros((neurons.depth, 1))




class L4Configurator(LayerConfig):
    h, w = config.dataset_hxw
    exc_size = (config.num_classes, h, w)
    exc_neuron_params = {
        "R": 1,
        "tau": 5,
        "threshold": 1,
        "v_reset": 0,
        "v_rest": 0,
    }
    exc_neuron_type = LIF
    exc_tau_trace = 10
    exc_fire = True
    exc_dendrite_computation_params = {
        "distal_provocativeness": 0.0,
        "apical_provocativeness": 0.0,
    }

    exc_user_defined_behaviors_class = {805: AccuracyMetric}
    exc_user_defined_behaviors_params = {805: {'time_window': config.presenting_timing_window}}

    exc_kwta = 1
    exc_kwta_dim = 0
    exc_noise_params = {"scale": 0.0}
    exc_exc_weight_init_params = {"mode": "uniform"}
    exc_exc_structure = "Simple"
    exc_exc_structure_params = {"current_coef": 0}
    exc_exc_tag = "Distal"


class L23Configurator(LayerConfig):
    exc_size = (4, 10, 10)
    exc_neuron_params = {
        "R": 1,
        "tau": 10,
        "threshold": 1,
        "v_reset": 0,
        "v_rest": 0,
    }
    exc_neuron_type = LIF
    exc_tau_trace = 20
    exc_fire = True
    exc_dendrite_computation_params = {"distal_provocativeness": 0.35}
    exc_kwta = 1

    exc_exc_weight_init_params = {"mode": "uniform"}
    exc_exc_structure = "Simple"
    exc_exc_structure_params = {"current_coef": 0}
    exc_exc_tag = "Distal"


# synapses
class SensoryL4Configurator(Pop2LayerConnectionConfig):
    pop_2_exc_weight_init_params = {
        "mode": "uniform",
        "weight_shape": [4, 1, 3, 3],
        "scale": 0.5,
        "offset": 0.37,
    }
    pop_2_exc_structure = "Conv2d"
    pop_2_exc_structure_params = {"current_coef": 6}
    pop_2_exc_dst_pop = "exc_pop"
    pop_2_exc_tag = "Proximal"
    pop_2_exc_learning_rule = "RSTDP"
    pop_2_exc_learning_params = {
        "a_plus": 0.0001,
        "a_minus": 0.0001,
        "tau_c": 2,
        "positive_bound": "soft_bound",
        "negative_bound": "soft_bound",
    }
    pop_2_exc_user_defined_behaviors_class = {
        181: CurrentNormalization,
    }
    pop_2_exc_user_defined_behaviors_params = {
        181: {"norm": 1},
    }


class L4L23Configurator(Layer2LayerConnectionConfig):
    exc_exc_weight_init_params = {"mode": "uniform"}
    exc_exc_structure = "Simple"
    exc_exc_structure_params = {"current_coef": 0}
    exc_exc_src_pop = "exc_pop"
    exc_exc_dst_pop = "exc_pop"


class L23L4Configurator(Layer2LayerConnectionConfig):
    exc_exc_weight_init_params = {"mode": "uniform"}
    exc_exc_structure = "Simple"
    exc_exc_structure_params = {"current_coef": 0}
    exc_exc_src_pop = "exc_pop"
    exc_exc_dst_pop = "exc_pop"
    exc_exc_tag = "Apical"


# Data preparation
torch.manual_seed(config.random_seed)
random.seed(config.random_seed)
torch.use_deterministic_algorithms(True)
np.random.seed(config.random_seed)

dataset = MNIST(
        root=config.dataset_root,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            SqueezeTransform(dim=0),
            Poisson(time_window=config.poisson_time, ratio=config.poisson_ratio),
        ]),
)

# Select subset of dataset
if config.subset_selected_labels is not None:
    indices = [idx for idx, target in enumerate(dataset.targets) if target in config.subset_selected_labels]
    dataset = Subset(dataset, indices)
dataloader = DataLoader(dataset, batch_size=16, drop_last=True)


# Reward Behavior
class CustomPayoff(Payoff):
    def initialize(self, network):
        super().initialize(network)
        self.rewarding_pop = network.columns[0].L2_3.exc_pop
        self.inp_layer = network.input_layers[0]

    def forward(self, network):
        winning_class = torch.argmax(
                torch.sum(self.rewarding_pop.spikes.reshape(self.rewarding_pop.depth, -1), axis=1)
        )

        if hasattr(self.inp_layer, 'y'):
            if self.inp_layer.y == config.CLASSES[winning_class]:
                network.payoff += 0.001
            else:
                network.payoff += -0.001


# Neocortex Network
net = Neocortex(
        dt=config.dt,
        payoff=CustomPayoff(),
        neuromodulators=[
            Dopamine(
                    tau_dopamine=config.TAU_DOPAMINE,
                    initial_dopamine_concentration=config.initial_dopamine_concentration
            )
        ],
        settings={
            "device": config.device,
            "dtype": config.dtype,
            "synapse_mode": config.synapse_mode,
        }
)

input_layer = InputLayer(
        net=net,
        have_label=True,
        input_dataloader=dataloader,
        sensory_trace=config.sensory_trace_tau_s,
        instance_duration=config.poisson_time,
        silent_interval=config.poisson_time * config.silent_ratio,
        sensory_size=NeuronDimension(
                depth=1,
                height=config.sensory_size_height,
                width=config.sensory_size_width
        )
)

cc1 = CorticalColumn(
        net,
        L4_config=L4Configurator().make(),
        L2_3_config=L23Configurator().make(),
        L4_L2_3_syn_config=L4L23Configurator().make(),
        L2_3_L4_syn_config=L23L4Configurator().make(),
)

# connect sensory to column
input_layer.connect(
        cc1,
        sensory_L4_syn_config=SensoryL4Configurator().make(),
)


def callback(x, net):
    plot_conv_weight(net.SynapseGroups[4].weights)


if __name__ == '__main__':
    net.initialize()
    visualise_network_structure(net)
    net.simulate_iterations(
            num_iteration,
            batch_size=config.batch_size,
            batch_progress_update_func=callback
    )

    # for accuracy
    print(net.metric_.output)
