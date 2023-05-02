from pymonntorch import *

from conex import *
from conex.nn.priorities import NEURON_PRIORITIES

from conex.helpers.encoders import Poisson
from conex.helpers.transforms import *

from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

import torch

# parameters
POISSON_TIME = 30
POISSON_RATIO = 5 / 30
MNIST_ROOT = "~/Temp/MNIST/"
SENSORY_SIZE_HEIGHT = 28
SENSORY_SIZE_WIDTH = 28  # MNIST's image size
SENSORY_TRACE_TAU_S = 2.7
DEVICE = "cuda"
DTYPE = torch.float16

from configs.l2_3 import *
from configs.l4 import *
from configs.l5 import *
from configs.sens_l4 import *
from configs.l2_3_l5 import *
from configs.l4_l2_3 import *
from configs.l5_l2_3 import *


#############################
# making dataloader
#############################
transformation = transforms.Compose(
    [
        transforms.ToTensor(),
        SqueezeTransform(dim=0),
        Poisson(time_window=POISSON_TIME, ratio=POISSON_RATIO),
    ]
)

dataset = MNIST(root=MNIST_ROOT, train=True, download=True, transform=transformation)

dl = DataLoader(dataset, batch_size=16)
#############################


#############################
# initializing neocortex
#############################
net = Neocortex(settings={"device": DEVICE, "dtype": DTYPE})
#############################


#############################
# input layer
#############################
input_layer = InputLayer(
    net=net,
    input_dataloader=dl,
    sensory_size=NeuronDimension(
        depth=1, height=SENSORY_SIZE_HEIGHT, width=SENSORY_SIZE_WIDTH
    ),
    sensory_trace=SENSORY_TRACE_TAU_S,
)

#############################


#############################
# making cortical column
#############################
cc1 = CorticalColumn(
    net,
    L2_3_config=l2_3().make(),
    L4_config=l4().make(),
    L5_config=l5().make(),
    L6_config=None,
    L6_L4_syn_config=None,
    L4_L2_3_syn_config=l4_l2_3().make(),
    L2_3_L5_syn_config=l2_3_l5().make(),
    L5_L2_3_syn_config=l5_l2_3().make(),
)
#############################


#############################
# connect sensory to column
#############################
input_layer.connect(
    cc1,
    sensory_L4_syn_config=sens_l4().make(),
    sensory_L6_syn_config=None,
    location_L6_syn_config=None,
)
#############################


net.initialize()

net.simulate_iterations(100)
