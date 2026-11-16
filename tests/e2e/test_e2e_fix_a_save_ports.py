"""Regression tests for save_ports with a port whose behavior is None (fix-a, bug 2).

save_ports called behaviors_to_list(None) for such a port, which crashed with
AttributeError. The minimal correct serialization of "no behavior" is the
empty list: build_ports rebuilds it via build_behavior_dict([]) == {}.
"""

import torch

from pymonntorch import NeuronGroup

from conex import (
    Neocortex,
    Port,
    SimpleDendriteStructure,
    save_ports,
    build_ports,
)


def build_network(seed=0):
    torch.manual_seed(seed)
    return Neocortex(dt=1.0)


class TestSavePortsNoneBehavior:
    def test_save_ports_with_none_behavior(self):
        net = build_network(seed=9)
        pop = NeuronGroup(net=net, size=4, tag="p", behavior={})
        ports = {"in": (None, [Port(object=pop, label=None)])}
        saved = save_ports(ports, save_behavior_tag=True, save_behavior_priority=True)
        assert saved["in"][0] is None
        obj, label, beh_list = saved["in"][1][0]
        assert obj is pop
        assert label is None
        assert beh_list == []

    def test_round_trip_with_behavior(self):
        net = build_network(seed=10)
        pop = NeuronGroup(net=net, size=4, tag="p", behavior={})
        ports = {"in": (None, [Port(object=pop, label=None,
                                    behavior={220: SimpleDendriteStructure()})])}
        rebuilt = build_ports(save_ports(ports, True, True))
        port = rebuilt["in"][1][0]
        assert port.object is pop
        assert set(port.behavior) == {220}
        assert port.behavior[220].__class__ is SimpleDendriteStructure

    def test_round_trip_without_behavior(self):
        net = build_network(seed=11)
        pop = NeuronGroup(net=net, size=4, tag="p", behavior={})
        ports = {"in": (None, [Port(object=pop, label=None)])}
        rebuilt = build_ports(save_ports(ports, True, True))
        port = rebuilt["in"][1][0]
        assert port.object is pop
        assert port.behavior == {}
