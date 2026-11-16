"""E2E tests for network-level behaviors and the spike dataset loader."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from pymonntorch import Network

from conex import Dopamine, Payoff, TimeResolution
from conex.behaviors.layer.dataset import SpikeNdDataset
from conex.nn.structure.layer import Layer

from .conftest import make_network, seed


# --- TimeResolution -------------------------------------------------------


def test_time_resolution_sets_dt():
    """TimeResolution stores the network dt used by every behavior."""
    net = make_network(dt=2.5)
    net.initialize()
    assert net.dt == 2.5


# --- Payoff ----------------------------------------------------------------


def test_payoff_initial():
    """Payoff sets network.payoff to the initial value."""
    net = Network(
        behavior={1: TimeResolution(dt=1.0), 100: Payoff(initial_payoff=2.5)},
        dtype=torch.float32,
        device="cpu",
    )
    net.initialize()
    assert net.payoff == 2.5


def test_payoff_default_zero():
    """With no initial_payoff the payoff defaults to 0.0."""
    net = Network(
        behavior={1: TimeResolution(dt=1.0), 100: Payoff()},
        dtype=torch.float32,
        device="cpu",
    )
    net.initialize()
    assert net.payoff == 0.0


# --- Dopamine --------------------------------------------------------------


def test_dopamine_decay():
    """d += -d/tau + payoff. With payoff=0, d0=1.0, tau=2: d -> 1.0 - 0.5 = 0.5."""
    net = Network(
        behavior={
            1: TimeResolution(dt=1.0),
            100: Payoff(initial_payoff=0.0),
            120: Dopamine(tau_dopamine=2.0, initial_dopamine_concentration=1.0),
        },
        dtype=torch.float32,
        device="cpu",
    )
    net.initialize()
    assert net.dopamine_concentration == 1.0
    net.simulate_iterations(1)
    assert net.dopamine_concentration == pytest.approx(0.5, abs=1e-6)


def test_dopamine_with_payoff():
    """d += -d/tau + payoff. With payoff=1.0, d0=1.0, tau=2: d -> 1.0 - 0.5 + 1.0 = 1.5."""
    net = Network(
        behavior={
            1: TimeResolution(dt=1.0),
            100: Payoff(initial_payoff=1.0),
            120: Dopamine(tau_dopamine=2.0, initial_dopamine_concentration=1.0),
        },
        dtype=torch.float32,
        device="cpu",
    )
    net.initialize()
    net.simulate_iterations(1)
    assert net.dopamine_concentration == pytest.approx(1.5, abs=1e-6)


# --- SpikeNdDataset --------------------------------------------------------


def _dataset_layer(net, data, **kwargs):
    dl = DataLoader(TensorDataset(data), batch_size=data.size(0))
    layer = Layer(
        net=net,
        tag="L",
        behavior={320: SpikeNdDataset(dataloader=dl, **kwargs)},
    )
    return layer


def test_spike_nd_dataset_feeds_layer():
    """The dataset yields each instance's sensory vector as layer.x in order."""
    net = make_network(dt=1.0)
    data = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.bool)
    layer = _dataset_layer(
        net,
        data,
        instance_duration=1,
        ndim_sensory=1,
        have_location=False,
        have_label=False,
    )
    net.initialize()
    net.simulate_iterations(1)
    assert layer.x.tolist() == [True, False, True, False]
    net.simulate_iterations(1)
    assert layer.x.tolist() == [False, True, False, True]


def test_spike_nd_dataset_silent_interval():
    """With silent_interval=1, a silent (all-zero) step separates two instances."""
    net = make_network(dt=1.0)
    data = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.bool)
    layer = _dataset_layer(
        net,
        data,
        instance_duration=1,
        ndim_sensory=1,
        have_location=False,
        have_label=False,
        silent_interval=1,
    )
    net.initialize()
    net.simulate_iterations(1)  # instance 1
    assert layer.x.tolist() == [True, False, True, False]
    net.simulate_iterations(1)  # silent gap
    assert layer.x.tolist() == [False, False, False, False]
    net.simulate_iterations(1)  # instance 2
    assert layer.x.tolist() == [False, True, False, True]
