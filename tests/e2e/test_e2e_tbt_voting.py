"""End-to-end tests for inter-column voting and consensus.

Each file carries its own small network builder so it runs on its own.
"""

import pytest
import torch

from conex import ColumnVoting, ConsensusNetwork, SDROverlap, TimeResolution
from pymonntorch import Network, NeuronGroup


def build_network(seed=0, extra_behavior=None):
    torch.manual_seed(seed)
    behavior = {1: TimeResolution(dt=1.0)}
    if extra_behavior:
        behavior.update(extra_behavior)
    net = Network(behavior=behavior, dtype=torch.float32, device="cpu")
    return net


def make_group(net, size, behavior, tag="col"):
    g = NeuronGroup(net=net, size=size, behavior={100: behavior}, tag=tag)
    net.initialize()
    return g


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
def column():
    net = build_network(seed=1)
    return make_group(net, REPRESENTATION_SIZE, ColumnVoting(
        n_objects=N_OBJECTS, representation_size=REPRESENTATION_SIZE,
        voting_threshold=0.5, decay_rate=0.1, min_votes_for_consensus=2))


class TestColumnVoting:
    def test_state_allocated(self, column):
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

    def test_vote_below_threshold_is_suppressed(self, column):
        memories = object_memories()
        weak = sdr(REPRESENTATION_SIZE, 0, 28, 29, 30)  # 1 of object 0's 4 bits
        votes = column.voting_module.cast_vote(column, weak, memories)
        assert votes[0] == 0

    def test_empty_representation_casts_no_vote(self, column):
        votes = column.voting_module.cast_vote(column, torch.zeros(REPRESENTATION_SIZE),
                                               object_memories())
        assert votes.sum() == 0

    def test_receive_and_consensus_pipeline(self, column):
        memories = object_memories()
        votes = column.voting_module.cast_vote(column, memories[1].clone(), memories)
        # a full match casts a vote of 1.0; the accumulator decays by 10% per
        # receive, so three receives (1.0, 1.9, 2.71) clear min_votes=2
        for _ in range(3):
            column.voting_module.receive_votes(column, votes)
        assert column.voting_module.check_consensus(column, n_columns=3)
        assert column.consensus_object == 1

    def test_no_consensus_on_tie(self, column):
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
    def network_with_columns(self):
        net = build_network(seed=2, extra_behavior={160: ConsensusNetwork(max_iterations=5)})
        columns = [
            make_group(net, REPRESENTATION_SIZE, ColumnVoting(
                n_objects=N_OBJECTS, representation_size=REPRESENTATION_SIZE,
                min_votes_for_consensus=1), tag=f"col{i}")
            for i in range(3)
        ]
        return net, columns

    def test_iteration_counter_advances(self, network_with_columns):
        net, _ = network_with_columns
        start = net.consensus_iteration
        net.simulate_iterations(3)
        assert net.consensus_iteration == start + 3

    def test_broadcast_spreads_votes(self, network_with_columns):
        net, columns = network_with_columns
        votes = torch.zeros(N_OBJECTS)
        votes[2] = 6.0
        columns[0].voting_module.receive_votes(columns[0], votes)
        net.consensus_network.broadcast_votes(net, columns)
        for col in columns:
            assert col.vote_accumulator[2] > 0

    def test_broadcast_noop_single_column(self, network_with_columns):
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

    def test_no_network_consensus_when_disagree(self, network_with_columns):
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

    def test_overlap_ratio_and_jaccard(self):
        a = sdr(32, 1, 2, 3, 4)
        b = sdr(32, 3, 4, 5, 6)
        assert SDROverlap.compute_overlap_ratio(a, b) == pytest.approx(2 / 4)
        assert SDROverlap.compute_jaccard(a, b) == pytest.approx(2 / 6)

    def test_empty_sdr_ratio(self):
        assert SDROverlap.compute_overlap_ratio(torch.zeros(32), sdr(32, 1)) == 0.0
        assert SDROverlap.compute_jaccard(torch.zeros(32), torch.zeros(32)) == 0.0

    def test_union_and_intersection(self):
        a = sdr(32, 1, 2, 3)
        b = sdr(32, 3, 4)
        assert SDROverlap.union(a, b).sum().item() == 4
        assert SDROverlap.intersection(a, b).sum().item() == 1

    def test_are_similar_uses_threshold(self):
        # similarity_threshold is bound in initialize, so set it directly here
        beh = SDROverlap(similarity_threshold=0.3)
        beh.similarity_threshold = 0.3
        a = sdr(32, 1, 2, 3, 4)
        b = sdr(32, 3, 4, 5, 6)
        assert beh.are_similar(a, b)  # ratio 0.5 >= 0.3
        strict = SDROverlap(similarity_threshold=0.6)
        strict.similarity_threshold = 0.6
        assert not strict.are_similar(a, b)
