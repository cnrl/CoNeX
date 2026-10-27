"""Tests for the HTM Spatial Pooler and Temporal Memory."""

import pytest
import torch

from conex import (
    SpatialPooler,
    SpatialPoolerConfig,
    TemporalMemory,
    TemporalMemoryConfig,
)


@pytest.fixture
def pooler(make_group):
    config = SpatialPoolerConfig(
        input_size=100,
        n_columns=200,
        potential_pct=0.5,
        local_area_density=0.05,
        seed=7,
    )
    return make_group(config.n_columns, SpatialPooler(config=config))


@pytest.fixture
def memory(make_group):
    config = TemporalMemoryConfig(
        n_columns=20,
        cells_per_column=4,
        activation_threshold=2,
        learning_threshold=1,
        max_new_synapses=5,
    )
    return make_group(
        config.n_columns * config.cells_per_column, TemporalMemory(config=config)
    )


def pattern(size, *active):
    x = torch.zeros(size)
    x[list(active)] = 1.0
    return x


def columns(n_columns, *active):
    c = torch.zeros(n_columns, dtype=torch.bool)
    c[list(active)] = True
    return c


class TestSpatialPooler:
    def test_output_sparsity_is_fixed(self, pooler):
        active = pooler.spatial_pooler.compute(pooler, pattern(100, *range(0, 40)))
        assert active.sum().item() == pooler.spatial_pooler.n_active == 10

    def test_same_input_gives_same_output(self, pooler):
        x = pattern(100, *range(10, 50))
        first = pooler.spatial_pooler.compute(pooler, x, learn=False).clone()
        second = pooler.spatial_pooler.compute(pooler, x, learn=False)
        assert torch.equal(first, second)

    def test_different_inputs_give_different_outputs(self, pooler):
        a = pooler.spatial_pooler.compute(pooler, pattern(100, *range(0, 30)), learn=False).clone()
        b = pooler.spatial_pooler.compute(pooler, pattern(100, *range(60, 90)), learn=False)
        assert not torch.equal(a, b)

    def test_empty_input_activates_nothing(self, pooler):
        active = pooler.spatial_pooler.compute(pooler, torch.zeros(100), learn=False)
        assert active.sum().item() == 0

    def test_learning_changes_permanences(self, pooler):
        before = pooler.sp_permanences.clone()
        pooler.spatial_pooler.compute(pooler, pattern(100, *range(0, 40)), learn=True)
        assert not torch.equal(before, pooler.sp_permanences)

    def test_permanences_stay_in_range(self, pooler):
        x = pattern(100, *range(0, 40))
        for _ in range(20):
            pooler.spatial_pooler.compute(pooler, x, learn=True)
        assert pooler.sp_permanences.min() >= 0.0
        assert pooler.sp_permanences.max() <= 1.0

    def test_repeated_input_stabilises_output(self, pooler):
        x = pattern(100, *range(20, 60))
        for _ in range(10):
            pooler.spatial_pooler.compute(pooler, x, learn=True)
        settled = pooler.spatial_pooler.compute(pooler, x, learn=True).clone()
        again = pooler.spatial_pooler.compute(pooler, x, learn=True)
        assert torch.equal(settled, again)

    def test_forward_uses_sp_input(self, pooler):
        pooler.sp_input = pattern(100, *range(0, 40))
        pooler.spatial_pooler.forward(pooler)
        assert pooler.sp_active_columns.sum().item() == 10


class TestTemporalMemory:
    def test_unpredicted_column_bursts(self, memory):
        memory.temporal_memory.compute(memory, columns(20, 1, 5), learn=False)
        assert memory.tm_bursting_columns[[1, 5]].all()
        # bursting means every cell in the column is active
        assert memory.tm_active_cells[4:8].all()

    def test_inactive_columns_have_no_active_cells(self, memory):
        memory.temporal_memory.compute(memory, columns(20, 1), learn=False)
        assert not memory.tm_active_cells[8:].any()

    def test_one_winner_cell_per_active_column(self, memory):
        memory.temporal_memory.compute(memory, columns(20, 2, 7, 11), learn=False)
        assert memory.tm_winner_cells.sum().item() == 3

    def test_anomaly_is_one_for_novel_input(self, memory):
        memory.temporal_memory.compute(memory, columns(20, 3, 9), learn=True)
        assert memory.tm_anomaly_score.item() == pytest.approx(1.0)

    def test_learning_creates_segments(self, memory):
        memory.temporal_memory.compute(memory, columns(20, 1, 2), learn=True)
        memory.temporal_memory.compute(memory, columns(20, 5, 6), learn=True)
        total = sum(memory.tm_segments.count_segments(c) for c in range(memory.size))
        assert total > 0

    def test_repeated_sequence_reduces_anomaly(self, memory):
        sequence = [columns(20, 1, 2), columns(20, 5, 6), columns(20, 9, 10)]
        first_pass = []
        for _ in range(15):
            scores = []
            for step in sequence:
                memory.temporal_memory.compute(memory, step, learn=True)
                scores.append(memory.tm_anomaly_score.item())
            if not first_pass:
                first_pass = scores
        assert sum(scores) < sum(first_pass)

    def test_state_resets_between_timesteps(self, memory):
        memory.temporal_memory.compute(memory, columns(20, 1), learn=False)
        first = memory.tm_active_cells.clone()
        memory.temporal_memory.compute(memory, columns(20, 15), learn=False)
        assert not torch.equal(first, memory.tm_active_cells)
        assert torch.equal(memory.tm_prev_active_cells, first)
