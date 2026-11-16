"""End-to-end HTM pipeline: encoder -> spatial pooler -> temporal memory.

Each file carries its own small network builder so it runs on its own.
"""

import pytest
import torch

from conex import (
    SpatialPooler,
    SpatialPoolerConfig,
    TemporalMemory,
    TemporalMemoryConfig,
    TimeResolution,
)
from pymonntorch import Network, NeuronGroup


def build_network(seed=0):
    torch.manual_seed(seed)
    net = Network(behavior={1: TimeResolution(dt=1.0)}, dtype=torch.float32, device="cpu")
    return net


def make_group(net, size, behavior, tag="htm"):
    g = NeuronGroup(net=net, size=size, behavior={100: behavior}, tag=tag)
    net.initialize()
    return g


def pattern(size, *active):
    x = torch.zeros(size)
    x[list(active)] = 1.0
    return x


def columns(n_columns, *active):
    c = torch.zeros(n_columns, dtype=torch.bool)
    c[list(active)] = True
    return c


SP_CFG = SpatialPoolerConfig(
    input_size=60, n_columns=40, potential_pct=0.5, local_area_density=0.05, seed=7
)
TM_CFG = TemporalMemoryConfig(
    n_columns=40, cells_per_column=4, activation_threshold=2,
    learning_threshold=1, max_new_synapses=5,
)


@pytest.fixture
def pipeline():
    """One network carrying both a spatial pooler and a temporal memory."""
    net = build_network(seed=11)
    sp = make_group(net, SP_CFG.n_columns, SpatialPooler(config=SP_CFG), tag="sp")
    tm = make_group(net, TM_CFG.n_columns * TM_CFG.cells_per_column,
                    TemporalMemory(config=TM_CFG), tag="tm")
    return net, sp, tm


class TestSpatialPoolerE2E:
    def test_fixed_output_sparsity(self, pipeline):
        _, sp, _ = pipeline
        out = sp.spatial_pooler.compute(sp, pattern(60, *range(0, 30)))
        assert out.sum().item() == sp.spatial_pooler.n_active == 2

    def test_deterministic_with_same_seed(self):
        outs = []
        for _ in range(2):
            net = build_network(seed=7)
            sp = make_group(net, SP_CFG.n_columns, SpatialPooler(config=SP_CFG), tag="sp")
            x = pattern(60, *range(5, 40))
            outs.append(sp.spatial_pooler.compute(sp, x, learn=False).clone())
        assert torch.equal(outs[0], outs[1])

    def test_different_seeds_differ(self):
        # SpatialPooler.initialize calls torch.manual_seed(config.seed), so the
        # config seed, not the network seed, drives the random potential pools.
        outs = []
        for seed in (1, 2):
            cfg = SP_CFG._replace(seed=seed)
            net = build_network(seed=seed)
            sp = make_group(net, SP_CFG.n_columns, SpatialPooler(config=cfg), tag="sp")
            x = pattern(60, *range(5, 40))
            outs.append(sp.spatial_pooler.compute(sp, x, learn=False).clone())
        assert not torch.equal(outs[0], outs[1])

    def test_empty_input_activates_nothing(self, pipeline):
        _, sp, _ = pipeline
        out = sp.spatial_pooler.compute(sp, torch.zeros(60), learn=False)
        assert out.sum().item() == 0

    def test_wrong_input_size_raises(self, pipeline):
        _, sp, _ = pipeline
        with pytest.raises(IndexError):
            sp.spatial_pooler.compute(sp, pattern(10, 0, 1), learn=False)

    def test_permanences_stay_in_unit_interval(self, pipeline):
        _, sp, _ = pipeline
        x = pattern(60, *range(0, 30))
        for _ in range(15):
            sp.spatial_pooler.compute(sp, x, learn=True)
        assert sp.sp_permanences.min() >= 0.0
        assert sp.sp_permanences.max() <= 1.0

    def test_forward_uses_sp_input(self, pipeline):
        _, sp, _ = pipeline
        sp.sp_input = pattern(60, *range(0, 30))
        sp.spatial_pooler.forward(sp)
        assert sp.sp_active_columns.sum().item() == 2


class TestSpatialPoolerTieBreak:
    """Tied overlaps must resolve to the lowest column indices, on any platform."""

    def _tied_pooler(self, global_inhibition=True):
        cfg = SP_CFG._replace(global_inhibition=global_inhibition)
        net = build_network(seed=3)
        sp = make_group(net, SP_CFG.n_columns, SpatialPooler(config=cfg), tag="sp")
        # identical potential pools and permanences -> every column ties
        sp.sp_potential_inputs[:] = torch.arange(sp.spatial_pooler.n_potential)
        sp.sp_permanences[:] = 0.5
        return net, sp

    def test_tied_overlaps_pick_lowest_indices(self):
        _, sp = self._tied_pooler()
        out = sp.spatial_pooler.compute(sp, pattern(60, 0, 1, 2, 3, 4), learn=False)
        assert out.nonzero(as_tuple=True)[0].tolist() == [0, 1]

    def test_tied_overlaps_deterministic_across_runs_and_call_order(self):
        def fresh():
            _, sp = self._tied_pooler()
            return sp.spatial_pooler.compute(
                sp, pattern(60, 0, 1, 2, 3, 4), learn=False
            ).clone()

        a = fresh()
        b = fresh()
        assert torch.equal(a, b)
        # same tied input after other calls must give the same winners
        _, sp = self._tied_pooler()
        sp.spatial_pooler.compute(sp, pattern(60, 10, 11), learn=False)
        sp.spatial_pooler.compute(sp, pattern(60, 20, 21, 22), learn=False)
        c = sp.spatial_pooler.compute(sp, pattern(60, 0, 1, 2, 3, 4), learn=False)
        assert torch.equal(a, c)

    def test_matches_python_reference(self):
        _, sp = self._tied_pooler()
        out = sp.spatial_pooler.compute(sp, pattern(60, 0, 1, 2, 3, 4), learn=False)
        overlap = sp.sp_overlap.tolist()
        n_active = sp.spatial_pooler.n_active
        # reference rule: highest score first, ties to the lowest index
        expected = [
            i
            for i, _ in sorted(enumerate(overlap), key=lambda t: (-t[1], t[0]))[:n_active]
        ]
        assert out.nonzero(as_tuple=True)[0].tolist() == sorted(expected)

    def test_local_inhibition_ties_pick_lowest_indices(self):
        _, sp = self._tied_pooler(global_inhibition=False)
        out = sp.spatial_pooler.compute(sp, pattern(60, 0, 1, 2, 3, 4), learn=False)
        assert out.nonzero(as_tuple=True)[0].tolist() == [0, 1]


class TestTemporalMemoryE2E:
    def test_burst_then_prediction(self, pipeline):
        _, _, tm = pipeline
        tm.temporal_memory.compute(tm, columns(40, 3), learn=False)
        assert tm.tm_bursting_columns[3]
        assert tm.tm_active_cells[12:16].all()  # all 4 cells burst

    def test_anomaly_is_one_for_novel_input(self, pipeline):
        _, _, tm = pipeline
        tm.temporal_memory.compute(tm, columns(40, 3, 9), learn=True)
        assert tm.tm_anomaly_score.item() == pytest.approx(1.0)

    def test_repeated_sequence_improves_prediction(self, pipeline):
        """The core e2e claim: over a repeated sequence, bursts fall."""
        _, _, tm = pipeline
        seq = [columns(40, 1, 2), columns(40, 5, 6), columns(40, 9, 10)]
        first_pass = []
        for _ in range(15):
            scores = []
            for step in seq:
                tm.temporal_memory.compute(tm, step, learn=True)
                scores.append(tm.tm_anomaly_score.item())
            if not first_pass:
                first_pass = scores
        assert sum(scores) < sum(first_pass)
        assert tm.tm_prediction_accuracy.item() > 0.0

    def test_learn_false_keeps_segments_empty(self, pipeline):
        _, _, tm = pipeline
        tm.temporal_memory.compute(tm, columns(40, 1, 2), learn=False)
        total = sum(tm.tm_segments.count_segments(c) for c in range(tm.size))
        assert total == 0

    def test_segment_storage_direct_ops(self, pipeline):
        _, _, tm = pipeline
        segs = tm.tm_segments
        winners = torch.zeros(tm.size, dtype=torch.bool)
        winners[[5, 6, 7]] = True  # winner_cells is a boolean mask, not indices
        idx = segs.create_segment(0, winners, 0.21, 20, 255)
        assert idx == 0
        assert segs.count_segments(0) == 1
        active = torch.zeros(tm.size, dtype=torch.bool)
        active[[5, 6, 7]] = True
        assert segs.has_active_segment(0, active, threshold=2, connected_perm=0.2)
        assert not segs.has_active_segment(0, torch.zeros_like(active), 2, 0.2)

    def test_segment_storage_max_segments(self, pipeline):
        _, _, tm = pipeline
        segs = tm.tm_segments
        winners = torch.zeros(tm.size, dtype=torch.bool)
        winners[2] = True
        for _ in range(3):
            assert segs.create_segment(1, winners, 0.21, 5, 3) >= 0
        assert segs.create_segment(1, winners, 0.21, 5, 3) == -1

    def test_segment_storage_create_without_winners(self, pipeline):
        _, _, tm = pipeline
        assert tm.tm_segments.create_segment(2, torch.zeros(160, dtype=torch.bool),
                                             0.21, 5, 255) == -1


class TestFullPipeline:
    def test_encoder_to_pooler_to_memory(self, pipeline):
        """SDR input -> pooler -> memory, run for several steps."""
        net, sp, tm = pipeline
        for step in range(6):
            x = pattern(60, *range(step * 5, step * 5 + 20))
            active = sp.spatial_pooler.compute(sp, x, learn=True)
            tm.temporal_memory.compute(tm, active, learn=True)
        assert tm.tm_anomaly_score.item() <= 1.0
        assert tm.tm_prediction_accuracy.item() >= 0.0
        # segments were grown by the memory
        assert any(tm.tm_segments.count_segments(c) > 0 for c in range(tm.size))

    def test_pipeline_deterministic_with_same_seed(self):
        def run(seed):
            net = build_network(seed=seed)
            sp = make_group(net, SP_CFG.n_columns, SpatialPooler(config=SP_CFG), tag="sp")
            tm = make_group(net, TM_CFG.n_columns * TM_CFG.cells_per_column,
                            TemporalMemory(config=TM_CFG), tag="tm")
            for step in range(4):
                x = pattern(60, *range(step * 5, step * 5 + 20))
                active = sp.spatial_pooler.compute(sp, x, learn=True)
                tm.temporal_memory.compute(tm, active, learn=True)
            return tm.tm_anomaly_score.item(), tm.tm_prediction_accuracy.item()

        assert run(5) == run(5)
