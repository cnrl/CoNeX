"""End-to-end combination: encoder -> SDR -> spatial pooler -> temporal memory
-> predictive coding hierarchy, all in one running network on a repeating
sequence.

Each file carries its own small network builder so it runs on its own.
"""

import pytest
import torch

from pymonntorch import Network, NeuronGroup

from conex import (
    SDREncoder,
    SpatialPooler,
    SpatialPoolerConfig,
    TemporalMemory,
    TemporalMemoryConfig,
    PredictiveHierarchy,
    PredictiveCodingConfig,
    TimeResolution,
)


def build_network(seed=0):
    torch.manual_seed(seed)
    return Network(behavior={1: TimeResolution(dt=1.0)}, dtype=torch.float32, device="cpu")


SP_CFG = SpatialPoolerConfig(
    input_size=60, n_columns=40, potential_pct=0.5, local_area_density=0.05, seed=7
)
TM_CFG = TemporalMemoryConfig(
    n_columns=40, cells_per_column=4, activation_threshold=2,
    learning_threshold=1, max_new_synapses=5,
)
SEQ = [0.0, 0.25, 0.5, 0.75, 1.0]


def build_pipeline(seed=11):
    """One network carrying encoder, pooler, memory and a 2-level hierarchy."""
    net = build_network(seed)
    enc = NeuronGroup(
        net=net, size=60,
        behavior={100: SDREncoder(size=60, n_active=12, encoder_type="scalar",
                                  min_val=0.0, max_val=1.0)}, tag="enc")
    sp = NeuronGroup(net=net, size=SP_CFG.n_columns,
                     behavior={100: SpatialPooler(config=SP_CFG)}, tag="sp")
    tm = NeuronGroup(net=net, size=TM_CFG.n_columns * TM_CFG.cells_per_column,
                     behavior={100: TemporalMemory(config=TM_CFG)}, tag="tm")
    h = PredictiveHierarchy(net, level_sizes=[16, 8],
                            config=PredictiveCodingConfig(
                                error_nonlinearity="linear", use_sparse_errors=False))
    net.initialize()
    return net, enc, sp, tm, h


def step(net, enc, sp, tm, h, value):
    """Push one scalar through the whole chain and return the two error signals."""
    dense = enc.sdr_encoder.encode_scalar(value).to_float()
    active = sp.spatial_pooler.compute(sp, dense, learn=True)
    tm.temporal_memory.compute(tm, active, learn=True)
    h.set_input(dense[:16])
    net.simulate_iterations(1)
    return tm.tm_anomaly_score.item(), h.get_errors(0).abs().sum().item()


class TestComboHtmPipeline:
    def test_prediction_error_falls_over_repeated_sequence(self):
        """The core claim: over a repeated sequence, prediction error falls."""
        net, enc, sp, tm, h = build_pipeline()
        first_anom = first_herr = None
        for _ in range(15):
            anoms, herrs = [], []
            for v in SEQ:
                a, e = step(net, enc, sp, tm, h, v)
                anoms.append(a)
                herrs.append(e)
            if first_anom is None:
                first_anom, first_herr = sum(anoms), sum(herrs)
        assert sum(anoms) < first_anom          # temporal memory learns the sequence
        assert sum(herrs) < first_herr          # hierarchy error also falls
        assert tm.tm_prediction_accuracy.item() > 0.0

    def test_sdr_sparsity_stays_inside_bounds(self):
        """Encoder output is exactly n_active bits; pooler output is fixed."""
        net, enc, sp, tm, h = build_pipeline()
        for v in SEQ:
            sdr = enc.sdr_encoder.encode_scalar(v)
            assert sdr.n_active == 12
            assert 0.0 <= sdr.sparsity <= 1.0
            active = sp.spatial_pooler.compute(sp, sdr.to_float(), learn=True)
            # pooler keeps a fixed number of columns active (local_area_density)
            assert active.sum().item() == sp.spatial_pooler.n_active == 2

    def test_anomaly_reaches_zero_when_sequence_is_learned(self):
        net, enc, sp, tm, h = build_pipeline()
        for _ in range(15):
            for v in SEQ:
                step(net, enc, sp, tm, h, v)
        # a fully learned repeating sequence is no longer anomalous
        assert tm.tm_anomaly_score.item() == pytest.approx(0.0)

    def test_pipeline_is_deterministic_with_same_seed(self):
        def run(seed):
            net, enc, sp, tm, h = build_pipeline(seed)
            out = []
            for v in SEQ:
                a, e = step(net, enc, sp, tm, h, v)
                out.append((a, e))
            return out
        assert run(5) == run(5)

    def test_novel_input_is_anomalous_after_learning(self):
        """A value outside the learned sequence spikes the anomaly score."""
        net, enc, sp, tm, h = build_pipeline()
        for _ in range(15):
            for v in SEQ:
                step(net, enc, sp, tm, h, v)
        # 0.125 sits between two learned values -> a new pooler pattern
        a, _ = step(net, enc, sp, tm, h, 0.125)
        assert a > 0.0
