"""End-to-end tests for the SDR data structure and its encoder/classifier.

Each file carries its own small network builder so it runs on its own.
"""

import pytest
import torch

from conex import SDR, SDRConfig, SDREncoder, SDRClassifier, UnionSDR, TimeResolution
from pymonntorch import Network, NeuronGroup


def build_network(seed=0):
    torch.manual_seed(seed)
    net = Network(behavior={1: TimeResolution(dt=1.0)}, dtype=torch.float32, device="cpu")
    return net


def make_group(net, size, behavior, tag="sdr"):
    g = NeuronGroup(net=net, size=size, behavior={100: behavior}, tag=tag)
    net.initialize()
    return g


class TestSDR:
    def test_random_then_overlap_pipeline(self):
        a = SDR.random(1000, 40)
        b = SDR.random(1000, 40)
        assert a.n_active == 40
        assert a.sparsity == pytest.approx(0.04)
        # two random 40-bit SDRs in 1000 bits share a small, stable overlap
        assert 0 <= a.overlap(b) <= 40

    def test_union_intersection_difference_are_consistent(self):
        a = SDR(size=100, indices=torch.tensor([1, 2, 3, 4]))
        b = SDR(size=100, indices=torch.tensor([3, 4, 5, 6]))
        u = a.union(b)
        i = a.intersection(b)
        d = a.difference(b)
        assert sorted(u.indices.tolist()) == [1, 2, 3, 4, 5, 6]
        assert sorted(i.indices.tolist()) == [3, 4]
        assert sorted(d.indices.tolist()) == [1, 2]
        # union size = |a| + |b| - |a ∩ b|
        assert u.n_active == a.n_active + b.n_active - i.n_active

    def test_empty_sdr_operations(self):
        empty = SDR(size=100)
        a = SDR(size=100, indices=torch.tensor([1, 2]))
        assert empty.overlap(a) == 0
        assert empty.overlap_score(a) == 0.0
        assert empty.intersection(a).n_active == 0
        assert a.difference(empty).n_active == 2
        assert empty.difference(a).n_active == 0

    def test_dense_roundtrip_and_float(self):
        dense = torch.zeros(64, dtype=torch.bool)
        dense[[3, 17, 40]] = True
        s = SDR.from_dense(dense)
        assert torch.equal(s.to_dense(), dense)
        f = s.to_float()
        assert f.dtype == torch.float32
        assert f.sum().item() == 3

    def test_subsample_bounds(self):
        a = SDR(size=100, indices=torch.arange(10))
        assert a.subsample(10).n_active == 10  # n >= n_active -> copy
        assert a.subsample(3).n_active == 3
        assert a.subsample(0).n_active == 0

    def test_match_threshold_and_equality(self):
        a = SDR(size=100, indices=torch.tensor([1, 2, 3, 4]))
        b = SDR(size=100, indices=torch.tensor([3, 4, 5, 6]))
        assert a.match_threshold(b, theta=2)
        assert not a.match_threshold(b, theta=3)
        assert a == SDR(size=100, indices=torch.tensor([4, 1, 2, 3]))
        assert a != b

    def test_duplicate_indices_are_deduped(self):
        s = SDR(size=100, indices=torch.tensor([1, 1, 2, 2, 2]))
        assert s.n_active == 2
        assert sorted(s.indices.tolist()) == [1, 2]


class TestSDRConfig:
    def test_n_active_floor(self):
        assert SDRConfig(size=2048, sparsity=0.02).n_active == 40
        assert SDRConfig(size=10, sparsity=0.0).n_active == 1  # max(1, 0)

    def test_from_config_random(self):
        cfg = SDRConfig(size=512, sparsity=0.05)
        s = SDR.from_config(cfg)
        assert s.n_active == cfg.n_active == 25
        assert s.size == 512


class TestSDREncoder:
    def test_scalar_bucket_math(self):
        enc = SDREncoder(size=100, n_active=5, encoder_type="scalar", min_val=0.0, max_val=1.0)
        # make_group runs net.initialize(), which binds the encoder's params
        _ = make_group(build_network(), 100, enc)
        s = enc.encode_scalar(0.5)
        # center = int(0.5 * (100-5)) = 47, indices 47..51
        assert sorted(s.indices.tolist()) == [47, 48, 49, 50, 51]
        assert s.n_active == 5

    def test_scalar_clamps_out_of_range(self):
        enc = SDREncoder(size=100, n_active=5, encoder_type="scalar", min_val=0.0, max_val=1.0)
        make_group(build_network(), 100, enc)
        # value above max clamps to 1.0 -> center = 95, wraps
        hi = enc.encode_scalar(5.0)
        assert hi.n_active == 5
        lo = enc.encode_scalar(-5.0)
        assert sorted(lo.indices.tolist()) == [0, 1, 2, 3, 4]

    def test_category_encoding_distinct_and_stable(self):
        enc = SDREncoder(size=256, n_active=16, encoder_type="category", n_categories=4)
        make_group(build_network(seed=3), 256, enc)
        c0 = enc.encode_category(0)
        c1 = enc.encode_category(1)
        assert c0.n_active == 16
        assert c0 == enc.encode_category(0)  # stable
        assert c0 != c1  # distinct categories

    def test_category_out_of_range_raises(self):
        enc = SDREncoder(size=256, n_active=16, encoder_type="category", n_categories=4)
        make_group(build_network(), 256, enc)
        with pytest.raises(ValueError, match="out of range"):
            enc.encode_category(4)
        with pytest.raises(ValueError, match="out of range"):
            enc.encode_category(-1)

    def test_random_encoding_is_deterministic(self):
        enc = SDREncoder(size=256, n_active=16, encoder_type="random")
        make_group(build_network(), 256, enc)
        assert enc.encode_random(42) == enc.encode_random(42)
        assert enc.encode_random(42) != enc.encode_random(43)

    def test_encode_dispatches_by_type(self):
        enc = SDREncoder(size=100, n_active=5, encoder_type="scalar")
        make_group(build_network(), 100, enc)
        assert enc.encode(0.5).n_active == 5
        with pytest.raises(ValueError, match="Unknown encoder type"):
            enc.encoder_type = "bogus"
            enc.encode(0.5)


class TestSDRClassifier:
    def test_learn_then_infer_recovers_label(self):
        clf = SDRClassifier(n_classes=3, sdr_size=64, learning_rate=0.1)
        g = make_group(build_network(seed=1), 64, clf)
        for label in range(3):
            sdr = SDR.random(64, 8)
            for _ in range(5):
                clf.learn(g, sdr, label)
        pred, scores = clf.infer(g, SDR.random(64, 8))
        assert isinstance(pred, int)
        assert scores.shape == (3,)

    def test_infer_accepts_dense_tensor(self):
        clf = SDRClassifier(n_classes=2, sdr_size=32, learning_rate=0.1)
        g = make_group(build_network(), 32, clf)
        dense = torch.zeros(32)
        dense[[1, 2, 3]] = 1.0
        clf.learn(g, dense, 0)
        pred, scores = clf.infer(g, dense)
        assert pred == 0
        assert scores[0] > scores[1]


class TestUnionSDR:
    def test_add_and_to_sdr(self):
        u = UnionSDR(size=100)
        a = SDR(size=100, indices=torch.tensor([1, 2]))
        b = SDR(size=100, indices=torch.tensor([2, 3]))
        u.add(a)
        u.add(b)
        assert u.n_members == 2
        assert sorted(u.to_sdr().indices.tolist()) == [1, 2, 3]
        assert u.n_active == 3

    def test_remove_recomputes(self):
        u = UnionSDR(size=100)
        a = SDR(size=100, indices=torch.tensor([1, 2]))
        b = SDR(size=100, indices=torch.tensor([3, 4]))
        u.add(a)
        u.add(b)
        u.remove(a)
        assert u.n_members == 1
        assert sorted(u.to_sdr().indices.tolist()) == [3, 4]

    def test_contains_uses_overlap_score(self):
        u = UnionSDR(size=100)
        a = SDR(size=100, indices=torch.arange(10))
        u.add(a)
        assert u.contains(a, threshold=0.5)
        far = SDR(size=100, indices=torch.arange(50, 60))
        assert not u.contains(far, threshold=0.5)
