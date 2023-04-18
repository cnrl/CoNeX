import json
from pymonntorch import *

from CCSNN import *
from CCSNN.nn.timestamps import NEURON_TIMESTAMPS

from CCSNN.helpers.transformers import *

from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

import torch

# parameters
POISSON_TIME = 30
POISSON_RATIO = 5/30
MNIST_ROOT = "~/Temp/MNIST/"
SENSORY_SIZE_HEIGHT = 28
SENSORY_SIZE_WIDTH = 28  # MNIST's image size
SENSORY_TRACE_TAU_S = 2.7
DEVICE = 'cuda'
DTYPE = torch.float16

from configs.l2_3 import*
from configs.l4 import*
from configs.l5 import*
from configs.sens_l4 import*
from configs.l2_3_l5 import*
from configs.l4_l2_3 import*
from configs.l5_l2_3 import*



#############################
# making dataloader
#############################
transformation = transforms.Compose([transforms.ToTensor(),
                                     transform_squeeze(dim=0),
                                     poisson(timesteps=POISSON_TIME, ratio=POISSON_RATIO)])

dataset = MNIST(root=MNIST_ROOT, train=True,
                download=True, transform=transformation)

dl = DataLoader(dataset, batch_size=16)
#############################


#############################
# initializing neocortex
#############################
net = Neocortex(settings={"device": DEVICE, "dtype": DTYPE})
#############################


#############################
# sensory layer
#############################
sensory_layer = NeuronGroup(net=net,
                            tag="exh, Sensory",
                            size=NeuronDimension(depth=1, height=SENSORY_SIZE_HEIGHT, width=SENSORY_SIZE_WIDTH),
                            behavior={NEURON_TIMESTAMPS["Fire"]: SpikeNdDataset(dataloader=dl),
                                      NEURON_TIMESTAMPS["Trace"]: SpikeTrace(tau_s=SENSORY_TRACE_TAU_S),
                                      NEURON_TIMESTAMPS["NeuronAxon"]: NeuronAxon()})
#############################


#############################
# making cortical column
#############################
cc1 = CorticalColumn(net,
                     sensory_layer,
                     location_layer=None,
                     representation_layer=None,
                     motor_layer=None,
                     L2_3_config=l2_3,
                     L4_config=l4,
                     L5_config=l5,
                     L6_config=None,
                     sensory_L4_syn_config=sens_l4,
                     sensory_L6_syn_config=None,
                     location_L6_syn_config=None,
                     L6_L4_syn_config=None,
                     L4_L2_3_syn_config=l4_l2_3,
                     L2_3_L5_syn_config=l2_3_l5,
                     L5_L2_3_syn_config=l5_l2_3,
                     L5_L6_syn_config=None,
                     L2_3_representation_syn_config=None,
                     L5_motor_syn_config=None,
                     )
#############################


net.initialize()

net.simulate_iterations(100)
