"""E2E tests for axon, setters, homeostasis, dendrite and neuron specs."""

import pytest
import torch
from pymonntorch import NeuronGroup

from conex import TimeResolution, prioritize_behaviors
from conex.behaviors.neurons import (
    ActivityBaseHomeostasis,
    Fire,
    InherentNoise,
    KWTA,
    LIF,
    LocationSetter,
    NeuronAxon,
    SensorySetter,
    SimpleDendriteComputation,
    SimpleDendriteStructure,
    VoltageBaseHomeostasis,
)
from conex.nn.structure.layer import Layer

from .conftest import make_network


def lif_group(net, tag, size=1, R=5.0, threshold=1.0, tau=10.0, v_reset=0.0, v_rest=0.0):
    return NeuronGroup(
        net=net,
        size=size,
        tag=tag,
        behavior=prioritize_behaviors(
            [
                SimpleDendriteStructure(),
                SimpleDendriteComputation(),
                LIF(R=R, threshold=threshold, tau=tau, v_reset=v_reset, v_rest=v_rest),
                Fire(),
                NeuronAxon(),
            ]
        ),
    )


# --- Axon -----------------------------------------------------------------


def test_axon_delay_buffer_tracks_spikes():
    """max_delay=2 keeps the last two spike vectors; get_spike reads by delay."""
    net = make_network(dt=1.0)
    n = NeuronGroup(
        net=net,
        size=2,
        tag="axon",
        behavior=prioritize_behaviors(
            [
                SimpleDendriteStructure(),
                SimpleDendriteComputation(),
                LIF(R=5.0, threshold=1.0, tau=10.0, v_reset=0.0, v_rest=0.0),
                Fire(),
                NeuronAxon(max_delay=2),
            ]
        ),
    )
    net.initialize()
    assert n.axon.spike_history.shape == (2, 2)
    n.v = torch.tensor([2.0, 0.0])  # neuron 0 fires
    net.simulate_iterations(1)
    n.v = torch.tensor([0.0, 0.0])  # both silent
    net.simulate_iterations(1)
    # buffer[0] is the latest (silent), buffer[1] is the previous (neuron 0 fired)
    assert n.axon.get_spike(n, torch.tensor([0, 0])).tolist() == [False, False]
    assert n.axon.get_spike(n, torch.tensor([1, 1])).tolist() == [True, False]


def test_axon_update_min_delay_from_efferent():
    """update_min_delay reads the minimum src_delay of efferent synapses."""
    from conex.behaviors.synapses import (
        PreSpikeCatcher,
        SimpleDendriticInput,
        SynapseInit,
        WeightInitializer,
    )
    from pymonntorch import SynapseGroup

    net = make_network(dt=1.0)
    src = lif_group(net, "src", size=2)
    dst = lif_group(net, "dst", size=2)
    syn = SynapseGroup(
        net=net,
        src=src,
        dst=dst,
        tag="Proximal",
        behavior=prioritize_behaviors(
            [
                SynapseInit(),
                WeightInitializer(mode=0.5),
                SimpleDendriticInput(current_coef=1.0),
                PreSpikeCatcher(),
            ]
        ),
    )
    net.initialize()
    # set a non-zero src delay so the minimum is meaningful
    syn.src_delay = torch.tensor([3, 1])
    src.axon.update_min_delay(src)
    assert src.axon.proximal_min_delay.item() == 1


# --- Setters --------------------------------------------------------------


def _setter_layer(net, tag, setter):
    ng = NeuronGroup(
        net=net, size=4, tag=tag, behavior=prioritize_behaviors([setter(), NeuronAxon()])
    )
    layer = Layer(net=net, neurongroups=[ng], tag="L")
    net.initialize()
    return layer, ng


def test_sensory_setter_copies_layer_x():
    net = make_network(dt=1.0)
    layer, ng = _setter_layer(net, "sens", SensorySetter)
    assert ng.parent_structure is layer
    layer.x = torch.tensor([True, False, True, False])
    net.simulate_iterations(1)
    assert ng.spikes.tolist() == [True, False, True, False]


def test_sensory_setter_zero_when_no_input():
    net = make_network(dt=1.0)
    layer, ng = _setter_layer(net, "sens", SensorySetter)
    layer.x = None
    net.simulate_iterations(1)
    assert ng.spikes.tolist() == [False, False, False, False]


def test_location_setter_copies_layer_loc():
    net = make_network(dt=1.0)
    layer, ng = _setter_layer(net, "loc", LocationSetter)
    layer.loc = torch.tensor([False, True, False, True])
    net.simulate_iterations(1)
    assert ng.spikes.tolist() == [False, True, False, True]


def test_location_setter_zero_when_no_input():
    net = make_network(dt=1.0)
    layer, ng = _setter_layer(net, "loc", LocationSetter)
    layer.loc = None
    net.simulate_iterations(1)
    assert ng.spikes.tolist() == [False, False, False, False]


# --- Homeostasis ----------------------------------------------------------


def test_activity_homeostasis_shifts_threshold():
    """Fire both steps of a window of 2: activities = 1 + 1 = 2, so
    threshold -= -2*1.0 -> 1.0 + 2.0 = 3.0."""
    net = make_network(dt=1.0)
    n = NeuronGroup(
        net=net,
        size=1,
        tag="act",
        behavior=prioritize_behaviors(
            [
                SimpleDendriteStructure(),
                SimpleDendriteComputation(),
                LIF(
                    R=5.0,
                    threshold=torch.tensor([1.0]),
                    tau=10.0,
                    v_reset=0.0,
                    v_rest=0.0,
                ),
                Fire(),
                ActivityBaseHomeostasis(activity_rate=1, window_size=2, updating_rate=1.0),
                NeuronAxon(),
            ]
        ),
    )
    net.initialize()
    n.v = torch.tensor([5.0])
    net.simulate_iterations(1)
    n.v = torch.tensor([5.0])
    net.simulate_iterations(1)
    assert n.threshold.tolist() == [3.0]


def test_activity_homeostasis_window_reset():
    """activities is zeroed at the window boundary, so a second window starts fresh."""
    net = make_network(dt=1.0)
    n = NeuronGroup(
        net=net,
        size=1,
        tag="act",
        behavior=prioritize_behaviors(
            [
                SimpleDendriteStructure(),
                SimpleDendriteComputation(),
                LIF(
                    R=5.0,
                    threshold=torch.tensor([1.0]),
                    tau=10.0,
                    v_reset=0.0,
                    v_rest=0.0,
                ),
                Fire(),
                ActivityBaseHomeostasis(activity_rate=1, window_size=2, updating_rate=1.0),
                NeuronAxon(),
            ]
        ),
    )
    net.initialize()
    beh = n.behavior[341]
    n.v = torch.tensor([5.0])
    net.simulate_iterations(2)  # one full window
    assert beh.activities.tolist() == [0.0]


def test_voltage_homeostasis_pushes_v_toward_target():
    """v=1.0, target=0.5, eta=0.1: change=(1-0.5)*0.1=0.05, v -= 0.05 -> 0.95."""
    net = make_network(dt=1.0)
    vh = VoltageBaseHomeostasis(target_voltage=0.5, eta_ip=0.1)
    n = NeuronGroup(
        net=net,
        size=1,
        tag="volt",
        behavior=prioritize_behaviors(
            [
                SimpleDendriteStructure(),
                SimpleDendriteComputation(),
                LIF(R=5.0, threshold=1.0, tau=10.0, v_reset=0.0, v_rest=0.0),
                Fire(),
                vh,
                NeuronAxon(),
            ]
        ),
    )
    net.initialize()
    n.v = torch.tensor([1.0])
    vh.forward(n)
    assert n.v[0].item() == pytest.approx(0.95, abs=1e-5)


def test_voltage_homeostasis_no_change_in_band():
    """v inside [min_ta, max_ta] produces no exhaustion change."""
    net = make_network(dt=1.0)
    vh = VoltageBaseHomeostasis(target_voltage=0.5, eta_ip=0.1)
    n = NeuronGroup(
        net=net,
        size=1,
        tag="volt",
        behavior=prioritize_behaviors(
            [
                SimpleDendriteStructure(),
                SimpleDendriteComputation(),
                LIF(R=5.0, threshold=1.0, tau=10.0, v_reset=0.0, v_rest=0.0),
                Fire(),
                vh,
                NeuronAxon(),
            ]
        ),
    )
    net.initialize()
    n.v = torch.tensor([0.5])  # exactly at target, within band
    vh.forward(n)
    assert n.exhaustion[0].item() == 0.0
    assert n.v[0].item() == pytest.approx(0.5, abs=1e-6)


# --- Dendrite structure / computation -------------------------------------


def test_dendrite_structure_buffers():
    """initialize allocates a per-delay buffer for each dendrite type."""
    net = make_network(dt=1.0)
    n = NeuronGroup(
        net=net,
        size=2,
        tag="dend",
        behavior=prioritize_behaviors(
            [
                SimpleDendriteStructure(
                    Proximal_max_delay=2, Distal_max_delay=3, Apical_max_delay=4
                ),
                SimpleDendriteComputation(),
                LIF(R=5.0, threshold=1.0, tau=10.0, v_reset=0.0, v_rest=0.0),
                Fire(),
                NeuronAxon(),
            ]
        ),
    )
    net.initialize()
    assert n.proximal_input.shape == (2, 2)
    assert n.distal_input.shape == (3, 2)
    assert n.apical_input.shape == (4, 2)


def test_dendrite_structure_bad_min_delay():
    """proximal_min_delay >= proximal_max_delay raises ValueError on initialize."""
    net = make_network(dt=1.0)
    n = NeuronGroup(
        net=net,
        size=2,
        tag="dend",
        behavior=prioritize_behaviors(
            [
                SimpleDendriteStructure(
                    Proximal_max_delay=1, proximal_min_delay=1
                ),
                SimpleDendriteComputation(),
                LIF(R=5.0, threshold=1.0, tau=10.0, v_reset=0.0, v_rest=0.0),
                Fire(),
                NeuronAxon(),
            ]
        ),
    )
    with pytest.raises(ValueError):
        net.initialize()


def test_dendrite_structure_gathers_current():
    """A fired pre-synaptic neuron with weight 1.0 delivers I_proximal=1.0 to dst
    one step after the spike (dendritic input runs before the spike catcher)."""
    from conex.behaviors.synapses import (
        PreSpikeCatcher,
        SimpleDendriticInput,
        SynapseInit,
        WeightInitializer,
    )
    from pymonntorch import SynapseGroup

    net = make_network(dt=1.0)
    src = lif_group(net, "src", size=2)
    dst = lif_group(net, "dst", size=2)
    SynapseGroup(
        net=net,
        src=src,
        dst=dst,
        tag="Proximal",
        behavior=prioritize_behaviors(
            [
                SynapseInit(),
                WeightInitializer(mode=1.0),
                SimpleDendriticInput(current_coef=1.0),
                PreSpikeCatcher(),
            ]
        ),
    )
    net.initialize()
    src.v = torch.tensor([2.0, 0.0])  # src neuron 0 fires
    net.simulate_iterations(1)
    assert dst.I_proximal.tolist() == [0.0, 0.0]
    net.simulate_iterations(1)
    assert dst.I_proximal.tolist() == [1.0, 1.0]


def test_dendrite_computation_current_decay():
    """I_tau=2 decays I by I/2 each step: 1.0 -> 0.5 with no new proximal input."""
    net = make_network(dt=1.0)
    n = NeuronGroup(
        net=net,
        size=1,
        tag="decay",
        behavior=prioritize_behaviors(
            [
                SimpleDendriteStructure(),
                SimpleDendriteComputation(I_tau=2.0),
                LIF(R=5.0, threshold=1.0, tau=10.0, v_reset=0.0, v_rest=0.0),
                Fire(),
                NeuronAxon(),
            ]
        ),
    )
    net.initialize()
    n.I = torch.tensor([1.0])
    net.simulate_iterations(1)
    assert n.I[0].item() == pytest.approx(0.5, abs=1e-6)


def test_dendrite_computation_distal_provocativeness():
    """distal_provocativeness=0.5, v=0, v_rest=0, threshold=1:
    ratio = clip(0.5*(1-0)-0, 0) = 0.5; I += tau*tanh(1)*0.5/R = 10*0.7616*0.5/5."""
    net = make_network(dt=1.0)
    comp = SimpleDendriteComputation(distal_provocativeness=0.5)
    n = NeuronGroup(
        net=net,
        size=1,
        tag="prov",
        behavior=prioritize_behaviors(
            [
                SimpleDendriteStructure(),
                comp,
                LIF(R=5.0, threshold=1.0, tau=10.0, v_reset=0.0, v_rest=0.0),
                Fire(),
                NeuronAxon(),
            ]
        ),
    )
    net.initialize()
    n.v = torch.tensor([0.0])
    n.I = torch.tensor([0.0])
    n.I_proximal = torch.tensor([0.0])
    n.I_apical = torch.tensor([0.0])
    n.I_distal = torch.tensor([1.0])
    comp.forward(n)
    expected = 10.0 * torch.tanh(torch.tensor(1.0)).item() * 0.5 / 5.0
    assert n.I[0].item() == pytest.approx(expected, abs=1e-5)


# --- Specs: InherentNoise, Fire, KWTA -------------------------------------


def test_inherent_noise_adds_offset():
    """mode='zeros' adds exactly the offset to v: 0.0 + 0.25 = 0.25."""
    net = make_network(dt=1.0)
    n = NeuronGroup(
        net=net,
        size=1,
        tag="noise",
        behavior=prioritize_behaviors(
            [
                SimpleDendriteStructure(),
                SimpleDendriteComputation(),
                LIF(R=5.0, threshold=1.0, tau=10.0, v_reset=0.0, v_rest=0.0),
                Fire(),
                InherentNoise(mode="zeros", scale=1.0, offset=0.25),
                NeuronAxon(),
            ]
        ),
    )
    net.initialize()
    n.v = torch.tensor([0.0])
    net.simulate_iterations(1)
    assert n.v[0].item() == pytest.approx(0.25, abs=1e-6)


def test_inherent_noise_zero_scale():
    """scale=0 with mode='rand' adds nothing: v is unchanged by the noise term."""
    net = make_network(dt=1.0)
    n = NeuronGroup(
        net=net,
        size=1,
        tag="noise",
        behavior=prioritize_behaviors(
            [
                SimpleDendriteStructure(),
                SimpleDendriteComputation(),
                LIF(R=5.0, threshold=1.0, tau=10.0, v_reset=0.0, v_rest=0.0),
                Fire(),
                InherentNoise(mode="rand", scale=0.0, offset=0.0),
                NeuronAxon(),
            ]
        ),
    )
    net.initialize()
    n.v = torch.tensor([0.3])
    net.simulate_iterations(1)
    # LIF leak: 0.3 + (0-0.3)*1/10 = 0.27, noise adds 0
    assert n.v[0].item() == pytest.approx(0.27, abs=1e-5)


def test_fire_resets_spiking_neuron():
    """Fire sets spikes = v >= threshold and resets spiking neurons to v_reset."""
    net = make_network(dt=1.0)
    n = lif_group(
        net, "fire", R=5.0, threshold=1.0, tau=10.0, v_reset=0.0, v_rest=0.0
    )
    net.initialize()
    n.v = torch.tensor([1.5, 0.5])
    n.behavior[340].forward(n)  # Fire
    assert n.spikes.tolist() == [True, False]
    assert n.v.tolist() == [0.0, 0.5]


def test_kwta_inhibits_losers():
    """k=1 with three over-threshold neurons: only the top keeps v, rest reset."""
    net = make_network(dt=1.0)
    n = NeuronGroup(
        net=net,
        size=3,
        tag="kwta",
        behavior=prioritize_behaviors(
            [
                SimpleDendriteStructure(),
                SimpleDendriteComputation(),
                LIF(R=5.0, threshold=1.0, tau=10.0, v_reset=0.0, v_rest=0.0),
                KWTA(k=1),
                Fire(),
                NeuronAxon(),
            ]
        ),
    )
    net.initialize()
    n.v = torch.tensor([1.5, 2.0, 1.2])
    n.behavior[300].forward(n)  # KWTA
    assert n.v.tolist() == [0.0, 2.0, 0.0]


def test_kwta_no_inhibition_under_k():
    """When the number of over-threshold neurons is <= k, nothing is reset."""
    net = make_network(dt=1.0)
    n = NeuronGroup(
        net=net,
        size=3,
        tag="kwta",
        behavior=prioritize_behaviors(
            [
                SimpleDendriteStructure(),
                SimpleDendriteComputation(),
                LIF(R=5.0, threshold=1.0, tau=10.0, v_reset=0.0, v_rest=0.0),
                KWTA(k=3),
                Fire(),
                NeuronAxon(),
            ]
        ),
    )
    net.initialize()
    n.v = torch.tensor([1.5, 2.0, 1.2])
    n.behavior[300].forward(n)  # KWTA
    assert n.v.tolist() == pytest.approx([1.5, 2.0, 1.2])
