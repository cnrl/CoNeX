"""Predictive coding: learning to predict away the error.

Wires the predictive coding behaviors onto one population and gives it a
sensory pattern it cannot explain yet:

    TopDownPrediction -> ErrorUnit -> PrecisionWeighting
        -> FreeEnergyMinimization -> PredictiveCodingLearning

The higher level carries an unrelated random pattern, so the top-down weights
have to learn the mapping onto the sensory input. As they do, the prediction
error and the free energy both fall. Perturbing the sensory input afterwards
makes the error jump again, which is the "surprise" signal a real hierarchy
would propagate upwards.

These behaviors are driven by explicit forward() calls and read
``neurons.activity`` rather than the spiking pipeline, so the loop sets that
state directly.

Run:
    python Example/numenta/predictive_coding.py
"""

import torch
from pymonntorch import Network, NeuronGroup

from conex import (
    ErrorUnit,
    FreeEnergyMinimization,
    PrecisionWeighting,
    PredictiveCodingConfig,
    PredictiveCodingLearning,
    TimeResolution,
    TopDownPrediction,
)


SIZE = 32
STEPS = 40
SEED = 7


def build_network(config):
    net = Network(behavior={1: TimeResolution(dt=1.0)}, dtype=torch.float32, device="cpu")

    # Keys follow conex.nn.priority: prediction -> error -> precision -> learning.
    behaviors = {
        230: TopDownPrediction(config=config),
        234: ErrorUnit(config=config, precision_weighted=False),
        236: PrecisionWeighting(config=config, learnable=True, attention_modulated=False),
        238: FreeEnergyMinimization(config=config, track_components=True),
        350: PredictiveCodingLearning(config=config, use_hebbian=False),
    }
    neurons = NeuronGroup(net=net, size=SIZE, behavior=behaviors, tag="level")
    net.initialize()
    return net, neurons


def report(neurons, label):
    print(
        f"{label:>5}   {neurons.error_magnitude.mean():>11.5f}   "
        f"{neurons.precision.mean():>14.4f}   {neurons.free_energy:>11.4f}"
    )


def main():
    torch.manual_seed(SEED)

    config = PredictiveCodingConfig(
        error_nonlinearity="linear",
        use_sparse_errors=False,
        learning_rate=0.05,
        precision_tau=20.0,
        precision_learning_rate=0.01,
    )
    net, neurons = build_network(config)

    sensory = torch.rand(SIZE)
    higher_level = torch.rand(SIZE)  # unrelated pattern, mapping must be learned

    neurons.activity = sensory
    neurons.top_down_input = higher_level

    order = [neurons.behavior[k] for k in sorted(neurons.behavior)]

    def advance():
        for behavior in order:
            behavior.forward(neurons)

    print(f"{SIZE} units, learning rate {config.learning_rate}\n")
    print(" step   mean |error|   mean precision   free energy")
    for i in range(STEPS):
        advance()
        if i < 3 or i % 8 == 0 or i == STEPS - 1:
            report(neurons, i)

    print("\nSensory input perturbed -- the level is surprised again:")
    print(" step   mean |error|   mean precision   free energy")
    neurons.activity = sensory + torch.randn(SIZE) * 0.4
    for i in range(6):
        advance()
        report(neurons, i)

    print(
        f"\nfree energy = accuracy {neurons.accuracy_term:.4f} "
        f"+ complexity {neurons.complexity_term:.4f}"
    )


if __name__ == "__main__":
    main()
