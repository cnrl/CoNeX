"""End-to-end combination: two predictive cortical columns plus two voting
columns connected through a ConsensusNetwork in one network.

Asserts the consensus agrees with the columns on a shared input and degrades
with a conflicting one.

Each file carries its own small network builder so it runs on its own.
"""

import pytest
import torch

from pymonntorch import Network, NeuronGroup

from conex import (
    PredictiveCorticalColumn,
    PredictiveCodingConfig,
    ColumnVoting,
    ConsensusNetwork,
    TimeResolution,
)


def build_network(seed=0):
    torch.manual_seed(seed)
    return Network(
        behavior={1: TimeResolution(dt=1.0), 160: ConsensusNetwork(max_iterations=5)},
        dtype=torch.float32, device="cpu",
    )


N_OBJECTS = 5
REP = 32


def object_memories():
    """Five distinct object SDRs, four active bits each, no overlap."""
    memories = torch.zeros(N_OBJECTS, REP)
    for obj in range(N_OBJECTS):
        memories[obj, obj * 4:(obj + 1) * 4] = 1.0
    return memories


def build_columns(seed=2):
    """Two predictive columns and two voting columns in one network."""
    net = build_network(seed)
    cfg = PredictiveCodingConfig(error_nonlinearity="linear", use_sparse_errors=False)
    colA = PredictiveCorticalColumn(net, column_size=16, config=cfg, name="A")
    colB = PredictiveCorticalColumn(net, column_size=16, config=cfg, name="B")
    vA = NeuronGroup(net=net, size=REP, behavior={100: ColumnVoting(
        n_objects=N_OBJECTS, representation_size=REP, voting_threshold=0.5,
        decay_rate=0.1, min_votes_for_consensus=1)}, tag="vA")
    vB = NeuronGroup(net=net, size=REP, behavior={100: ColumnVoting(
        n_objects=N_OBJECTS, representation_size=REP, voting_threshold=0.5,
        decay_rate=0.1, min_votes_for_consensus=1)}, tag="vB")
    net.initialize()
    return net, [colA, colB], [vA, vB]


def cast_and_check(col, representation, memories, n_columns):
    votes = col.voting_module.cast_vote(col, representation, memories)
    for _ in range(3):
        col.voting_module.receive_votes(col, votes)
    col.voting_module.check_consensus(col, n_columns=n_columns)


class TestComboColumns:
    def test_consensus_agrees_on_shared_input(self):
        """Both columns see the same object -> network consensus on that object."""
        net, _, columns = build_columns()
        memories = object_memories()
        shared = memories[2].clone()
        for col in columns:
            cast_and_check(col, shared, memories, n_columns=len(columns))
        assert columns[0].consensus_object == 2
        assert columns[1].consensus_object == 2
        assert net.consensus_network.check_network_consensus(net, columns)
        assert net.network_consensus

    def test_consensus_degrades_with_conflicting_input(self):
        """Columns see different objects -> no network-wide consensus."""
        net, _, columns = build_columns()
        memories = object_memories()
        cast_and_check(columns[0], memories[0].clone(), memories, n_columns=2)
        cast_and_check(columns[1], memories[3].clone(), memories, n_columns=2)
        # each column still resolves its own (conflicting) object
        assert columns[0].consensus_object == 0
        assert columns[1].consensus_object == 3
        assert not net.consensus_network.check_network_consensus(net, columns)
        assert not net.network_consensus

    def test_predictive_columns_run_together(self):
        """Both predictive columns run in the same network and stay finite."""
        net, cols, _ = build_columns()
        cols[0].set_input(torch.rand(16))
        cols[1].set_input(torch.rand(16))
        net.simulate_iterations(4)
        for col in cols:
            assert torch.isfinite(torch.tensor(col.get_free_energy()))
            assert col.get_prediction().shape == (24,)
            assert col.get_output().shape == (12,)

    def test_consensus_iteration_advances(self):
        net, _, _ = build_columns()
        start = net.consensus_iteration
        net.simulate_iterations(3)
        assert net.consensus_iteration == start + 3

    def test_reset_clears_conflicting_state(self):
        net, _, columns = build_columns()
        memories = object_memories()
        cast_and_check(columns[0], memories[0].clone(), memories, n_columns=2)
        columns[0].voting_module.reset_voting(columns[0])
        assert columns[0].consensus_object == -1
        assert not columns[0].consensus_reached
