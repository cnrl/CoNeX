# Thousand Brains Theory Support

CoNeX implements key components from Jeff Hawkins' **Thousand Brains Theory** and Numenta's **Hierarchical Temporal Memory (HTM)** research. These components enable building brain-like systems that learn through movement, use reference frames, and reach consensus across multiple models.

They all live under a `numenta` subpackage, so they stay grouped and clearly
separated from CoNeX's core behaviors.

## Key Components

| Component | Module | Description |
|-----------|--------|-------------|
| **Grid Cells** | `neurons.numenta.grid_cells` | Hexagonal firing patterns for allocentric (world-centered) reference frames |
| **Displacement Cells** | `neurons.numenta.grid_cells` | Encode movements between locations for learning object structure |
| **Active Dendrites** | `neurons.numenta.active_dendrites` | NMDA-based dendritic segments with nonlinear integration and contextual prediction |
| **Temporal Memory** | `neurons.numenta.sequence_memory` | HTM sequence learning algorithm with predictive cells and burst detection |
| **Spatial Pooler** | `neurons.numenta.spatial_pooler` | SDR encoding with competitive learning and homeostatic boosting |
| **SDR Operations** | `neurons.numenta.sdr` | Overlap, union, intersection, encoding, and classification for sparse patterns |
| **Column Voting** | `network.numenta.voting` | Inter-column consensus mechanism for object recognition |
| **Predictive Coding** | `neurons.numenta.predictive_coding` | Prediction and error units, precision weighting, free energy tracking |
| **Predictive Synapses** | `synapses.numenta.predictive` | Feedback prediction, feedforward error and lateral context connections |
| **Predictive Hierarchy** | `nn.structure.numenta.predictive_hierarchy` | Multi-level hierarchy builder and canonical cortical microcircuit |

## Running

The scripts import `conex` as an installed package. From a source checkout,
either install it in editable mode:

```bash
pip install -e .
python Example/numenta/htm_sequence_learning.py
```

or point `PYTHONPATH` at the repository root:

```bash
PYTHONPATH=. python Example/numenta/htm_sequence_learning.py
```

## `htm_sequence_learning.py`

The full HTM pipeline: symbols are encoded as sparse distributed
representations, the Spatial Pooler turns them into a stable set of active
columns, and Temporal Memory learns which element follows which.

The anomaly score starts at 1.0 (nothing is predicted yet) and falls to zero
once the sequence `A B C D` is learned:

```
epoch  mean anomaly  bursting columns
    0         1.000              10.0
    4         0.275               2.8
    7         0.000               0.0
   11         0.000               0.0
```

Then an unexpected symbol is injected. Note that the surprise costs *two*
steps, not one — breaking the context leaves the following element unpredicted
as well:

```
symbol  anomaly  bursting columns
     A    0.000                 0
     B    0.000                 0
     X    1.000                10
     D    1.000                10
```

## `predictive_coding.py`

One population wired as a predictive coding level:

```
TopDownPrediction -> ErrorUnit -> PrecisionWeighting
    -> FreeEnergyMinimization -> PredictiveCodingLearning
```

The higher level carries an unrelated random pattern, so the top-down weights
have to learn the mapping onto the sensory input:

```
 step   mean |error|   mean precision   free energy
    0       0.33209           1.0000        2.6553
    2       0.06951           1.0090       -0.0257
   16       0.00000           1.0762       -1.1754
   39       0.00000           1.1784       -2.6270
```

Read that in two phases. Up to about step 16 the error falls because the
weights are learning, and the free energy falls with it. After that the error
is already zero and the free energy keeps dropping for a different reason.

The quantity being tracked is

```
F = 0.5 * sum(precision * error^2)  -  0.5 * sum(log(precision))
```

Once the error is zero the first term vanishes and `F` is nothing but the
precision term. The numbers bear this out: `-0.5 * 32 * log(1.1784) = -2.6265`,
which is the free energy reported at step 39.

That second phase is a known degeneracy rather than better prediction. The
precision update is a maximum-likelihood estimate whose fixed point is
`precision = 1 / error^2`, so with zero error there is no finite fixed point and
precision climbs until it hits `max_precision`. With the default clamp of 100
the free energy bottoms out at `-0.5 * 32 * log(100)`, roughly -73.7. Putting a
prior on the precision would give it somewhere to settle; the current
implementation has none.

Perturbing the sensory input makes the error jump again, which is the surprise
signal a full hierarchy would propagate upwards.

## Quick reference

### Grid cells

```python
import torch
from pymonntorch import Network, NeuronGroup

from conex import GridCellModule, TimeResolution

net = Network(behavior={1: TimeResolution(dt=1.0)}, dtype=torch.float32, device="cpu")

# Create a grid cell population with 4 modules
neurons = NeuronGroup(
    net=net,
    size=400,
    behavior={
        215: GridCellModule(
            n_modules=4,
            cells_per_module=100,
            scales=[40.0, 50.0, 70.0, 100.0],
        )
    },
    tag="grid_cells",
)
net.initialize()

# Encode a position
position = torch.tensor([50.0, 50.0])
activation = neurons.grid_cell_module.encode_position(neurons, position)
```

### Temporal memory

```python
from conex import TemporalMemory, TemporalMemoryConfig

config = TemporalMemoryConfig(
    n_columns=2048,
    cells_per_column=32,
    activation_threshold=13,
)

neurons = NeuronGroup(
    net=net,
    size=2048 * 32,
    behavior={210: TemporalMemory(config=config)},
    tag="temporal_memory",
)
net.initialize()

# Process a sequence of sparse column activations
for active_columns in sequence:
    neurons.temporal_memory.compute(neurons, active_columns, learn=True)
```

## A note on the API

These behaviors are driven by explicit `forward()` and `compute()` calls and
read state the caller sets (`activity`, `sp_input`, `tm_input_columns`) rather
than reading `neurons.spikes` and writing `neurons.I`. They sit alongside the
spiking pipeline rather than inside it, which is why both scripts step them by
hand instead of calling `net.simulate_iterations()`.
