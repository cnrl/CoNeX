from pymonntorch import NeuronGroup, SynapseGroup, NeuronDimension

from conex import (
    CorticalLayer,
    CorticalColumn,
    CorticalLayerConnection,
    Synapsis,
    Neocortex,
    InputLayer,
    replicate,
    prioritize_behaviors,
    Port,
    save_structure,
    create_structure_from_dict,
    save_structure_dict_to_json,
    load_structure_dict_from_json,
)

from conex.behaviors.neurons import (
    SimpleDendriteStructure,
    SimpleDendriteComputation,
    LIF,
    NeuronAxon,
)
from conex.behaviors.synapses import (
    SynapseInit,
    PreTrace,
    PostTrace,
    PreSpikeCatcher,
    PostSpikeCatcher,
    WeightInitializer,
    SimpleDendriticInput,
    SimpleSTDP,
    Conv2dDendriticInput,
    Conv2dSTDP,
)

from conex.helpers.transforms.encoders import Poisson
from conex.helpers.transforms.misc import SqueezeTransform

from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

import torch

from visualisation import visualize_network_structure

##################################################
# parameters
##################################################
DEVICE = "cpu"
DTYPE = torch.float32
DT = 1

POISSON_TIME = 30
POISSON_RATIO = 5 / 30
MNIST_ROOT = "~/Temp/MNIST/"
SENSORY_SIZE_HEIGHT = 28
SENSORY_SIZE_WIDTH = 28  # MNIST's image size

# Layer 4
L4_EXC_DEPTH = 4
L4_EXC_HEIGHT = 24
L4_EXC_WIDTH = 24
L4_EXC_R = 5.0
L4_EXC_THRESHOLD = 0.0
L4_EXC_TAU = 10.0
L4_EXC_V_RESET = 0.0
L4_EXC_V_REST = 0.0

L4_INH_SIZE = 576
L4_INH_R = 5.0
L4_INH_THRESHOLD = 0.0
L4_INH_TAU = 10.0
L4_INH_V_RESET = 0.0
L4_INH_V_REST = 0.0
L4_INH_TRACE_TAU = 10.0

L4_EXC_EXC_MODE = "random"
L4_EXC_EXC_COEF = 1
L4_EXC_INH_MODE = "random"
L4_EXC_INH_COEF = 1
L4_INH_INH_MODE = "random"
L4_INH_INH_COEF = 1
L4_INH_EXC_MODE = "random"
L4_INH_EXC_COEF = 1

L2_EXC_SIZE = 400
L2_INH_SIZE = 100

L4_L2_MODE = "random"
L4_L2_COEF = 1
L4_L2_A_PLUS = 0.01
L4_L2_A_MINUS = 0.002


INP_CC_MODE = "random"
INP_CC_WEIGHT_SHAPE = (4, 1, 5, 5)
INP_CC_COEF = 1
INP_CC_A_PLUS = 0.01
INP_CC_A_MINUS = 0.002

L4_EXC_L23_EXC_PRE_TRACE = 10.0
L4_EXC_L23_EXC_POST_TRACE = 10.0


SENSORY_L4_PRE_TRACE = 10.0
SENSORY_L4_POST_TRACE = 10.0


##################################################
# making dataloader
##################################################
transformation = transforms.Compose(
    [
        transforms.ToTensor(),
        SqueezeTransform(dim=0),
        Poisson(time_window=POISSON_TIME, ratio=POISSON_RATIO),
    ]
)

dataset = MNIST(root=MNIST_ROOT, train=True, download=True, transform=transformation)

dl = DataLoader(dataset, batch_size=16)
##################################################


##################################################
# initializing neocortex
##################################################
net = Neocortex(dt=DT, device=DEVICE, dtype=DTYPE)
##################################################


##################################################
# input layer
##################################################
input_layer = InputLayer(
    net=net,
    input_dataloader=dl,
    sensory_size=NeuronDimension(
        depth=1, height=SENSORY_SIZE_HEIGHT, width=SENSORY_SIZE_WIDTH
    ),
    instance_duration=POISSON_TIME,
    output_ports={"data_out": (None, [("sensory_pop", {})])},
)

##################################################


##################################################
# making a cortical Layer
##################################################
pop_exc = NeuronGroup(
    net=net,
    size=NeuronDimension(depth=L4_EXC_DEPTH, height=L4_EXC_HEIGHT, width=L4_EXC_WIDTH),
    behavior=prioritize_behaviors(
        [
            SimpleDendriteStructure(),
            SimpleDendriteComputation(),
            LIF(
                R=L4_EXC_R,
                threshold=L4_EXC_THRESHOLD,
                tau=L4_EXC_TAU,
                v_reset=L4_EXC_V_RESET,
                v_rest=L4_EXC_V_REST,
            ),
            NeuronAxon(),
        ]
    ),
)

pop_inh = NeuronGroup(
    net=net,
    size=L4_INH_SIZE,
    tag="inh",
    behavior=prioritize_behaviors(
        [
            SimpleDendriteStructure(),
            SimpleDendriteComputation(),
            LIF(
                R=L4_INH_R,
                threshold=L4_INH_THRESHOLD,
                tau=L4_INH_TAU,
                v_reset=L4_INH_V_RESET,
                v_rest=L4_INH_V_REST,
            ),
            NeuronAxon(),
        ]
    ),
)

syn_exc_exc = SynapseGroup(
    net=net,
    src=pop_exc,
    dst=pop_exc,
    tag="Proximal",
    behavior=prioritize_behaviors(
        [
            SynapseInit(),
            WeightInitializer(mode=L4_EXC_EXC_MODE),
            SimpleDendriticInput(current_coef=L4_EXC_EXC_COEF),
            PreSpikeCatcher(),
        ]
    ),
)

syn_exc_inh = SynapseGroup(
    net=net,
    src=pop_exc,
    dst=pop_inh,
    tag="Proximal",
    behavior=prioritize_behaviors(
        [
            SynapseInit(),
            WeightInitializer(mode=L4_EXC_INH_MODE),
            SimpleDendriticInput(current_coef=L4_EXC_INH_COEF),
            PreSpikeCatcher(),
        ]
    ),
)

syn_inh_exc = SynapseGroup(
    net=net,
    src=pop_inh,
    dst=pop_exc,
    tag="Proximal,inh",
    behavior=prioritize_behaviors(
        [
            SynapseInit(),
            WeightInitializer(mode=L4_INH_EXC_MODE),
            SimpleDendriticInput(current_coef=L4_INH_EXC_COEF),
            PreSpikeCatcher(),
        ]
    ),
)

syn_inh_inh = SynapseGroup(
    net=net,
    src=pop_inh,
    dst=pop_inh,
    tag="Proximal,inh",
    behavior=prioritize_behaviors(
        [
            SynapseInit(),
            WeightInitializer(mode=L4_INH_INH_MODE),
            SimpleDendriticInput(current_coef=L4_INH_INH_COEF),
            PreSpikeCatcher(),
        ]
    ),
)

layer_l4 = CorticalLayer(
    net=net,
    excitatory_neurongroup=pop_exc,
    inhibitory_neurongroup=pop_inh,
    synapsegroups=[syn_exc_exc, syn_exc_inh, syn_inh_inh, syn_inh_exc],
    input_ports={
        "input": (
            None,
            [Port(object=pop_exc, label=None)],
        )
    },
)
##################################################


##################################################
# making layer 2 form layer 4
##################################################
l4_dict = save_structure(
    layer_l4,
    save_device=True,
    built_structures=None,
    save_structure_tag=True,
    save_behavior_tag=True,
    save_behavior_priority=True,
    all_structures_required=None,
)

save_structure_dict_to_json(l4_dict, "layer4.json")
l2_dict = load_structure_dict_from_json("layer4.json")

# updateing port
l2_dict["output_ports"] = {"output": (None, [(0, None, None)])}

# removing neurondimension
l2_dict["built_structures"][0]["behavior"] = [
    beh_save
    for beh_save in l2_dict["built_structures"][0]["behavior"]
    if beh_save["key"] != 0
]
l2_dict["built_structures"][0]["size"] = L2_EXC_SIZE
l2_dict["built_structures"][1]["size"] = L2_INH_SIZE


layer_l2 = create_structure_from_dict(
    net=net, structure_dict=l2_dict, built_structures=None
)
##################################################

##################################################
# replicating for layer 5
##################################################
layer_l5, _ = replicate(layer_l2, net)
##################################################

##################################################
# cortical connection l4_l2
##################################################
cortical_connection_l4_l2 = CorticalLayerConnection(
    net=net,
    src=layer_l4,
    dst=layer_l2,
    connections=[
        (
            "exc_pop",
            "exc_pop",
            prioritize_behaviors(
                [
                    SynapseInit(),
                    WeightInitializer(mode=L4_L2_MODE),
                    SimpleDendriticInput(current_coef=L4_L2_COEF),
                    SimpleSTDP(a_plus=L4_L2_A_PLUS, a_minus=L4_L2_A_MINUS),
                    PreSpikeCatcher(),
                    PostSpikeCatcher(),
                    PreTrace(tau_s=L4_EXC_L23_EXC_PRE_TRACE),
                    PostTrace(tau_s=L4_EXC_L23_EXC_POST_TRACE),
                ]
            ),
            "Proximal",
        )
    ],
)
##################################################

##################################################
# cortical connection l2_l5
##################################################
cortical_connection_l2_l5, _ = replicate(cortical_connection_l4_l2, net)
cortical_connection_l2_l5.connect_src(layer_l2)
cortical_connection_l2_l5.connect_dst(layer_l5)
##################################################


##################################################
# Cortical column
##################################################
cc1 = CorticalColumn(
    net=net,
    layers={"l4": layer_l4, "l2": layer_l2, "l5": layer_l5},
    layer_connections=[
        ("l4", "l2", cortical_connection_l4_l2),
        ("l2", "l5", cortical_connection_l2_l5),
    ],
    input_ports={"input": (None, [Port(object=layer_l4, label="input")])},
)
##################################################

##################################################
# Connection input layer with ports
##################################################
synapsis_input_cc1 = Synapsis(
    net=net,
    src=input_layer,
    dst=cc1,
    input_port="data_out",
    output_port="input",
    synapsis_behavior=prioritize_behaviors(
        [
            SynapseInit(),
            WeightInitializer(
                mode=INP_CC_MODE,
                weight_shape=INP_CC_WEIGHT_SHAPE,
                kernel_shape=INP_CC_WEIGHT_SHAPE,
            ),
            Conv2dDendriticInput(current_coef=INP_CC_COEF),
            Conv2dSTDP(a_plus=INP_CC_A_PLUS, a_minus=INP_CC_A_MINUS),
            PreSpikeCatcher(),
            PostSpikeCatcher(),
            PreTrace(tau_s=SENSORY_L4_PRE_TRACE),
            PostTrace(tau_s=SENSORY_L4_POST_TRACE),
        ]
    ),
    synaptic_tag="Proximal",
)
##################################################

net.initialize()
net.simulate_iterations(100)

visualize_network_structure(net)
