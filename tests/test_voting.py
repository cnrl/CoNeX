"""Tests for inter-column voting and consensus."""

import pytest
import torch
from pymonntorch import Network

from conex import ColumnVoting, ConsensusNetwork, SDROverlap, TimeResolution


N_OBJECTS = 5
REPRESENTATION_SIZE = 32


def sdr(size, *active):
    x = torch.zeros(size)
    x[list(active)] = 1.0
    return x


def object_memories():
    """Five distinct object SDRs, four active bits each, no overlap."""
    memories = torch.zeros(N_OBJECTS, REPRESENTATION_SIZE)
    for obj in range(N_OBJECTS):
        memories[obj, obj * 4:(obj + 1) * 4] = 1.0
    return memories


@pytest.fixture
def column(make_group):
    return make_group(
        REPRESENTATION_SIZE,
        ColumnVoting(
            n_objects=N_OBJECTS,
            representation_size=REPRESENTATION_SIZE,
            voting_threshold=0.5,
            decay_rate=0.1,
            min_votes_for_consensus=2,
        ),
        tag="column",
    )


class TestColumnVoting:
    def test_state_is_allocated(self, column):
        assert column.vote_accumulator.shape == (N_OBJECTS,)
        assert column.object_hypotheses.shape == (N_OBJECTS,)
        assert column.consensus_object == -1
        assert column.consensus_reached is False

    def test_matching_object_gets_the_vote(self, column):
        memories = object_memories()
        votes = column.voting_module.cast_vote(column, memories[2].clone(), memories)
        assert votes.argmax().item() == 2
        assert votes[2] > 0
        assert votes[[0, 1, 3, 4]].sum() == 0

    def test_partial_match_still_votes(self, column):
        memories = object_memories()
        partial = sdr(REPRESENTATION_SIZE, 8, 9, 10)  # 3 of object 2's 4 bits
        votes = column.voting_module.cast_vote(column, partial, memories)
        assert votes[2] == pytest.approx(1.0)

    def test_vote_below_threshold_is_suppressed(self, column):
        memories = object_memories()
        # one bit of object 0, three bits belonging to nothing
        weak = sdr(REPRESENTATION_SIZE, 0, 28, 29, 30)
        votes = column.voting_module.cast_vote(column, weak, memories)
        assert votes[0] == 0

    def test_empty_representation_casts_no_vote(self, column):
        votes = column.voting_module.cast_vote(
            column, torch.zeros(REPRESENTATION_SIZE), object_memories()
        )
        assert votes.sum() == 0

    def test_received_votes_accumulate(self, column):
        incoming = torch.zeros(N_OBJECTS)
        incoming[1] = 3.0
        column.voting_module.receive_votes(column, incoming)
        assert column.vote_accumulator[1] > 0
        assert column.object_hypotheses[1]
        assert not column.object_hypotheses[0]

    def test_confidence_scales_incoming_votes(self, column):
        incoming = torch.zeros(N_OBJECTS)
        incoming[1] = 4.0
        column.voting_module.receive_votes(column, incoming, source_confidence=0.5)
        strong = column.vote_accumulator[1].item()
        column.voting_module.reset_voting(column)
        column.voting_module.receive_votes(column, incoming, source_confidence=1.0)
        assert column.vote_accumulator[1].item() > strong

    def test_consensus_needs_a_clear_winner(self, column):
        votes = torch.zeros(N_OBJECTS)
        votes[1] = 5.0
        column.voting_module.receive_votes(column, votes)
        assert column.voting_module.check_consensus(column, n_columns=3)
        assert column.consensus_object == 1

    def test_no_consensus_when_two_objects_tie(self, column):
        votes = torch.zeros(N_OBJECTS)
        votes[1] = 5.0
        votes[3] = 5.0
        column.voting_module.receive_votes(column, votes)
        assert not column.voting_module.check_consensus(column, n_columns=3)
        assert column.consensus_object == -1

    def test_no_consensus_without_votes(self, column):
        assert not column.voting_module.check_consensus(column, n_columns=3)

    def test_reset_clears_state(self, column):
        votes = torch.zeros(N_OBJECTS)
        votes[1] = 5.0
        column.voting_module.receive_votes(column, votes)
        column.voting_module.check_consensus(column, n_columns=3)
        column.voting_module.reset_voting(column)
        assert column.vote_accumulator.sum() == 0
        assert column.consensus_object == -1
        assert column.consensus_reached is False

    def test_forward_decays_votes(self, column):
        votes = torch.zeros(N_OBJECTS)
        votes[1] = 5.0
        column.voting_module.receive_votes(column, votes)
        before = column.vote_accumulator[1].item()
        column.voting_module.forward(column)
        assert column.vote_accumulator[1].item() < before


class TestConsensusNetwork:
    @pytest.fixture
    def network_with_columns(self, make_group):
        net = Network(
            behavior={1: TimeResolution(dt=1.0), 160: ConsensusNetwork(max_iterations=5)},
            dtype=torch.float32,
            device="cpu",
        )
        columns = [
            make_group(
                REPRESENTATION_SIZE,
                ColumnVoting(
                    n_objects=N_OBJECTS,
                    representation_size=REPRESENTATION_SIZE,
                    min_votes_for_consensus=1,
                ),
                tag=f"col{i}",
                net=net,
            )
            for i in range(3)
        ]
        net.initialize()
        return net, columns

    def test_iteration_counter_advances(self, network_with_columns):
        net, _ = network_with_columns
        start = net.consensus_iteration
        net.simulate_iterations(3)
        assert net.consensus_iteration == start + 3

    def test_broadcast_spreads_one_column_vote_to_all(self, network_with_columns):
        net, columns = network_with_columns
        votes = torch.zeros(N_OBJECTS)
        votes[2] = 6.0
        columns[0].voting_module.receive_votes(columns[0], votes)

        net.consensus_network.broadcast_votes(net, columns)

        for col in columns:
            assert col.vote_accumulator[2] > 0

    def test_broadcast_is_a_noop_for_a_single_column(self, network_with_columns):
        net, columns = network_with_columns
        before = columns[0].vote_accumulator.clone()
        net.consensus_network.broadcast_votes(net, columns[:1])
        assert torch.equal(before, columns[0].vote_accumulator)

    def test_network_consensus_when_columns_agree(self, network_with_columns):
        net, columns = network_with_columns
        votes = torch.zeros(N_OBJECTS)
        votes[2] = 6.0
        for col in columns:
            col.voting_module.receive_votes(col, votes)
            col.voting_module.check_consensus(col, n_columns=len(columns))

        assert net.consensus_network.check_network_consensus(net, columns)
        assert net.network_consensus

    def test_no_network_consensus_when_columns_disagree(self, network_with_columns):
        net, columns = network_with_columns
        for i, col in enumerate(columns):
            votes = torch.zeros(N_OBJECTS)
            votes[i] = 6.0
            col.voting_module.receive_votes(col, votes)
            col.voting_module.check_consensus(col, n_columns=len(columns))

        assert not net.consensus_network.check_network_consensus(net, columns)

    def test_no_consensus_without_columns(self, network_with_columns):
        net, _ = network_with_columns
        assert not net.consensus_network.check_network_consensus(net, [])


class TestSDROverlap:
    def test_raw_overlap(self):
        a = sdr(32, 1, 2, 3, 4)
        b = sdr(32, 3, 4, 5)
        assert SDROverlap.compute_overlap(a, b) == 2

    def test_overlap_of_disjoint_sdrs(self):
        assert SDROverlap.compute_overlap(sdr(32, 0, 1), sdr(32, 8, 9)) == 0

    def test_overlap_with_itself(self):
        a = sdr(32, 1, 2, 3)
        assert SDROverlap.compute_overlap(a, a) == 3

    def test_behavior_is_reachable_from_the_group(self, make_group):
        neurons = make_group(32, SDROverlap(similarity_threshold=0.4), tag="ov")
        assert neurons.sdr_overlap.similarity_threshold == 0.4
