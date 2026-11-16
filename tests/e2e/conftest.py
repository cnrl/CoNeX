"""Shared helpers for e2e tests: a seeded network builder and a run loop.

Each test seeds torch and builds its own network so a test run twice gives
identical numbers. Keep this file to small shared helpers only.
"""

import torch
from pymonntorch import Network

from conex import TimeResolution

SEED = 1234


def seed():
    """Reset the torch RNG so a test is deterministic across runs."""
    torch.manual_seed(SEED)


def make_network(dt=1.0, dtype=torch.float32):
    """A fresh CPU network carrying a time resolution (behaviors read net.dt)."""
    seed()
    return Network(
        behavior={1: TimeResolution(dt=dt)}, dtype=dtype, device="cpu"
    )


def run(network, steps):
    """Advance the network ``steps`` iterations and return it."""
    network.simulate_iterations(steps)
    return network
