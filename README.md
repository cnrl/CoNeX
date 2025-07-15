# CoNeX: Cortical network for everything <img align="right" src="./assets/logo.jpg" width="150px" >

[![PyPI version](https://img.shields.io/pypi/v/cnrl-conex?style=flat-square&color=orange)](https://pypi.org/project/cnrl-conex/)
[![Python versions](https://img.shields.io/pypi/pyversions/cnrl-conex?style=flat-square&color=crimson)](https://pypi.org/project/cnrl-conex/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/cnrl-conex?style=flat-square&color=blue)](https://pypi.org/project/cnrl-conex/)
[![License](https://img.shields.io/pypi/l/cnrl-conex?style=flat-square&color=darkgreen)](https://github.com/cnrl/CoNeX/blob/master/LICENSE)
[![Contributions welcome](https://img.shields.io/badge/Contributions-welcome-brightgreen.svg?style=flat-square&color=tomato)](https://github.com/cnrl/CoNeX/blob/master/CONTRIBUTING.rst)
[![GitHub last commit](https://img.shields.io/github/last-commit/cnrl/CoNeX?style=flat-square&color=blue)](https://github.com/cnrl/CoNeX/commits/master)


**CoNeX** is a powerful and flexible Python framework for building, simulating, and experimenting with spiking neural networks, with a focus on modeling cortical structures. Built on `PyTorch` and `Pymonntorch`, CoNeX provides a high-level API to design complex network architectures inspired by the brain's neocortex, such as cortical layers and columns. 

The framework is designed for researchers and developers in computational neuroscience to easily create and manage intricate models, define custom neuronal behaviors, and apply various learning rules like Spike-Timing Dependent Plasticity (STDP).

-----

## Core Features

  * **Modular Architecture:** Build complex networks from reusable components like `Layers`, `CorticalColumns`, and `Synapses`.
  * **Biologically Plausible Models:** Includes built-in neuron models such as Leaky Integrate-and-Fire (LIF), Exponential LIF (ELIF), and Adaptive ELIF (AELIF). 
  * **Advanced Learning Rules:** Implement various forms of STDP, including Reward-modulated STDP (RSTDP) and inhibitory STDP (iSTDP), for different connection types (dense, sparse, convolutional). 
  * **Flexible Connectivity:** Supports numerous connection styles, including one-to-one, sparse, convolutional, and local connections. 
  * **Structured Network Building:** Easily define hierarchical structures like `CorticalLayer` and `CorticalColumn` to model brain-like architectures. 
  * **Customizable Behaviors:** Extend the framework with custom behaviors for neurons, synapses, and network dynamics.
  * **Data Handling & Pre-processing:** Includes helper utilities for creating datasets, encoding sensory information (e.g., Poisson encoding), and applying transformations. 

-----

## Project Structure

The CoNeX project is organized into several key directories. Here is a simplified view of the module hierarchy:

```
conex-cnrl/
├── conex/                    # Core library source code
│   ├── behaviors/            # Defines behaviors for network components
│   │   ├── layer/
│   │   ├── network/
│   │   ├── neurons/
│   │   └── synapses/
│   ├── helpers/              # Utility functions for data, filters, and transforms
│   │   └── transforms/
│   └── nn/                   # Core neural network structures and utilities
│       ├── structure/
│       └── utils/
├── docs/                     # Documentation files
├── Example/                  # Example usage scripts and notebooks
│   ├── helpers/
│   ├── test/
│   └── visualization/
├── tests/                    # Unit tests for the project
├── CONTRIBUTING.rst          # Guidelines for contributors
├── LICENSE                   # Project's MIT License
└── setup.py                  # Installation script
```

-----

## Modules and Functionality

### `conex.behaviors`

This module contains the building blocks for defining the dynamics of the network. Each behavior is a class that can be attached to a network object (like a `NeuronGroup` or `SynapseGroup`) to confer specific functionality.

  * **`network`**: Defines network-wide dynamics.
      * `TimeResolution`: Sets the simulation time step (`dt`). 
      * `Payoff`: A base class for defining reward/punishment signals. 
      * `Dopamine`: Models the effect of dopamine as a neuromodulator, influenced by the payoff signal. 
  * **`neurons`**: Defines the behavior of individual neurons.
      * `neuron_types`: Includes LIF, ELIF, and AELIF neuron models.  These models define the fundamental voltage dynamics of a neuron.
      * `dendrite`: Models dendritic compartments (proximal, distal, apical) and computes the input current `I` by summing their contributions. 
      * `axon`: Propagates spikes from a neuron to its connected synapses, handling transmission delays. 
      * `homeostasis`: Implements mechanisms to maintain stable network activity, either by regulating firing rates or membrane voltage. 
  * **`synapses`**: Defines the behavior of synapses.
      * `dendrites`: Determines how pre-synaptic spikes are converted into post-synaptic current for various connection types (e.g., `SimpleDendriticInput`, `Conv2dDendriticInput`). 
      * `learning`: Implements synaptic plasticity rules like STDP and RSTDP.  These rules modify synaptic weights based on the timing of pre- and post-synaptic spikes, and in the case of RSTDP, a global reward signal (dopamine).
      * `specs`: Includes essential synaptic mechanisms like weight initialization, spike catching, and calculating spike traces used in learning rules. 

### `conex.nn`

This module provides the tools to construct and manage the network architecture.

  * **`structure`**: Contains classes for building the network.
      * `Neocortex`: The main network class that holds all other structures. 
      * `Container`: A base class for creating hierarchical network objects that contain other objects. 
      * `Port`: An interface for connecting different network structures in a modular way. 
      * `Layer` / `CorticalLayer`: A container for neuron groups (e.g., excitatory and inhibitory populations) and their local connections. 
      * `CorticalColumn`: A container that groups multiple `CorticalLayer` objects to form a column-like structure. 
      * `Synapsis`: A structure that connects two `Ports`, automatically creating the required `SynapseGroup` objects between them. 
      * `InputLayer` / `OutputLayer`: Specialized layers for interfacing with data. 
  * **`utils`**: Provides utilities for network management.
      * `replication`: Functions to save, load, and replicate network structures, enabling easy reuse of complex architectures. 

### `conex.helpers`

This module contains utilities to assist with data preparation and processing.

  * **`transforms`**:
      * `encoders`: Functions to convert raw data (like images) into spike trains, such as `Poisson` and `Intensity2Latency` encoding. 
      * `masks`: Transformers that can occlude or isolate parts of an input, useful for attention experiments. 
  * **`filters`**: Implementations of common visual filters like Difference of Gaussians (`DoGFilter`) and `GaborFilter`. 

-----

## How to Use CoNeX

### Installation

Install CoNeX directly from PyPI:

```bash
pip install cnrl-conex
```

<details>
    <summary>Or, for the latest development version, install from GitHub</summary>

```bash
pip install git+https://github.com/cnrl/conex
```
</details>


### Building a Simple Network

Here's a conceptual example of how to build a network with an input layer and a cortical column, inspired by the project's examples. 

```python
import torch
from pymonntorch import NeuronGroup, NeuronDimension
from conex import Neocortex, InputLayer, CorticalLayer, CorticalColumn, Synapsis, Port, prioritize_behaviors
from conex.behaviors.neurons import LIF, NeuronAxon, SimpleDendriteStructure
from conex.behaviors.synapses import WeightInitializer, SimpleDendriticInput, PreSpikeCatcher, SynapseInit
from torch.utils.data import DataLoader

# 1. Initialize the Neocortex (the main network object)
net = Neocortex(dt=1.0, device="cpu", dtype=torch.float32)

# Dummy dataloader for the example
dummy_dataloader = DataLoader([torch.rand(28, 28) for _ in range(10)])

# 2. Create an Input Layer to feed data into the network
input_layer = InputLayer(
    net=net,
    input_dataloader=dummy_dataloader,
    sensory_size=NeuronDimension(height=28, width=28),
    instance_duration=30, # ms
    output_ports={"data_out": (None, [("sensory_pop", {})])},
)

# 3. Create a Cortical Layer
# Define an excitatory population of neurons
exc_neurons = NeuronGroup(
    net=net,
    size=100,
    behavior=prioritize_behaviors([
        LIF(R=5.0, tau=10.0, threshold=-50.0, v_reset=-70.0, v_rest=-65.0),
        NeuronAxon(),
        SimpleDendriteStructure(),
    ]),
)

# Create the layer itself, defining its input port
my_layer = CorticalLayer(
    net=net,
    excitatory_neurongroup=exc_neurons,
    input_ports={"input": (None, [Port(object=exc_neurons, label=None)])},
)

# 4. Create a Cortical Column to house the layer
my_column = CorticalColumn(
    net=net,
    layers={"my_layer": my_layer},
    input_ports={"input": (None, [Port(object=my_layer, label="input")])},
)

# 5. Connect the Input Layer to the Cortical Column using Synapsis
# This will connect the 'data_out' port of the input to the 'input' port of the column
data_connection = Synapsis(
    net=net,
    src=input_layer,
    dst=my_column,
    input_port="data_out",
    output_port="input",
    synapsis_behavior=prioritize_behaviors([
        SynapseInit(),
        WeightInitializer(mode="random"),
        SimpleDendriticInput(current_coef=1.0),
        PreSpikeCatcher(),
    ]),
    synaptic_tag="Proximal",
)

# 6. Initialize and run the simulation
net.initialize()
net.simulate_iterations(100) # Simulate for 100 timesteps

print("Simulation finished!")
```

### Defining a Custom Behavior

You can easily extend CoNeX with your own custom logic by creating new behavior classes. A behavior must inherit from `pymonntorch.Behavior` and can implement `initialize` and `forward` methods.

```python
from pymonntorch import Behavior
import torch

class MyCustomNeuronBehavior(Behavior):
    """A custom behavior that adds a random voltage boost to neurons on every step."""

    def initialize(self, neurons):
        # This method is called once when the network is initialized.
        # You can define parameters here.
        self.boost_strength = self.parameter("boost_strength", 1.0)
        print(f"Initialized MyCustomNeuronBehavior with boost_strength={self.boost_strength}")

    def forward(self, neurons):
        # This method is called on every simulation timestep.
        # 'neurons' is the NeuronGroup object this behavior is attached to.
        random_boost = torch.rand(neurons.size, device=neurons.device) * self.boost_strength
        neurons.v += random_boost

# How to use it:
# custom_behavior = MyCustomNeuronBehavior(boost_strength=0.5)
#
# my_neurons = NeuronGroup(
#     net=net,
#     size=100,
#     behavior={
#         ...
#         999: custom_behavior, # Add with a unique priority key
#     }
# )
```

-----

## Project To-Dos and Future Work

The project has several areas marked for future improvement and development. Contributions in these areas are highly welcome\!

  * **Refactoring and Documentation:**
      * Add comprehensive docstrings for bound functions in learning rules. 
      * Consider refactoring behavior priority assignment to be a class property instead of using a dictionary. 
  * **Enhanced Neuron Models:**
      * Implement a multi-adaptation paradigm for LIF neurons. 
  * **Advanced Synaptic Dynamics:**
      * Improve the dendrite model to prevent priming of neurons that are already above their voltage threshold. 
      * Explore the effects of priming inhibitory neurons. 
  * **Optimization:**
      * Find a pure PyTorch alternative for `random.sample` in sparse weight initialization for better performance on GPU. 
  * **General Notes for Implementation:**
      * The `Payoff` behavior should be defined before the `Dopamine` behavior in the network's behavior dictionary. 
      * Many synaptic behaviors require that an `Axon` behavior is added to the pre-synaptic neuron group. 
      * Learning behaviors (`STDP`, etc.) often require `PreSpikeCatcher`, `PostSpikeCatcher`, `PreTrace`, and `PostTrace` to be present in the synapse's behaviors. 

-----

## ❤️ How to Contribute

Contributions are welcome and greatly appreciated\! Every little bit helps, and credit will always be given.  You can contribute in many ways, from reporting bugs to writing documentation and implementing new features.

1.  **Report Bugs:** If you find a bug, please report it on the [GitHub Issues](https://github.com/cnrl/CoNeX/issues) page. Include your OS, local setup details, and steps to reproduce the bug. 
2.  **Fix Bugs or Implement Features:** Look through the GitHub issues for anything tagged with "bug" or "enhancement" and "help wanted." 
3.  **Write Documentation:** The project could always use more documentation, whether in official docs, docstrings, or blog posts. 
4.  **Submit Feedback:** The best way to send feedback or propose a new feature is to open an issue on GitHub. 

### Getting Started with Development

Ready to contribute code? Here’s how to set up `CoNeX` for local development.

1.  **Fork the repository** on GitHub. 
2.  **Clone your fork** locally: 
    ```bash
    git clone git@github.com:your_name_here/CoNeX.git
    ```
3.  **Set up a virtual environment** and install the project in development mode: 
    ```bash
    cd CoNeX/
    python -m venv venv
    source venv/bin/activate
    pip install -e .
    ```
4.  **Create a new branch** for your feature or bugfix. 
5.  **Make your changes.** When you're done, run the tests and linter: 
    ```bash
    pip install flake8 pytest
    flake8 conex tests
    pytest
    ```
6.  **Commit your changes** and push them to your fork. 
7.  **Submit a pull request** through the GitHub website.  Please ensure your PR includes tests for new functionality and that the documentation is updated. 
