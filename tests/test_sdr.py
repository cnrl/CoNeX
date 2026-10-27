"""Tests for Sparse Distributed Representation operations."""

import torch

from conex import SDR, SDRConfig, UnionSDR


def sdr(size, *idx):
    return SDR(size=size, indices=torch.tensor(idx, dtype=torch.long))


def test_config_active_bits():
    assert SDRConfig(size=2048, sparsity=0.02).n_active == 40
    assert SDRConfig(size=10, sparsity=0.0).n_active == 1


def test_empty_sdr():
    empty = SDR(size=100)
    assert empty.n_active == 0
    assert empty.sparsity == 0.0
    assert not empty.to_dense().any()


def test_dense_roundtrip():
    dense = torch.zeros(64, dtype=torch.bool)
    dense[[3, 17, 40]] = True
    restored = SDR.from_dense(dense)
    assert restored.n_active == 3
    assert torch.equal(restored.to_dense(), dense)


def test_sparsity():
    assert sdr(100, 0, 1).sparsity == 0.02


def test_random_has_requested_active_bits():
    s = SDR.random(size=1000, n_active=25)
    assert s.n_active == 25
    assert s.indices.unique().numel() == 25


def test_overlap():
    a = sdr(100, 1, 2, 3, 4)
    b = sdr(100, 3, 4, 5, 6)
    assert a.overlap(b) == 2
    assert b.overlap(a) == 2


def test_overlap_disjoint_and_identical():
    a = sdr(100, 1, 2, 3)
    assert a.overlap(sdr(100, 7, 8, 9)) == 0
    assert a.overlap(a) == 3
    assert a.overlap(SDR(size=100)) == 0


def test_overlap_score():
    a = sdr(100, 1, 2, 3, 4)
    b = sdr(100, 3, 4)
    assert a.overlap_score(b) == 1.0
    assert SDR(size=100).overlap_score(a) == 0.0


def test_jaccard():
    a = sdr(100, 1, 2, 3, 4)
    b = sdr(100, 3, 4, 5, 6)
    assert a.jaccard(b) == 2 / 6
    assert a.jaccard(a) == 1.0


def test_union():
    result = sdr(100, 1, 2, 3).union(sdr(100, 3, 4))
    assert sorted(result.indices.tolist()) == [1, 2, 3, 4]


def test_intersection():
    result = sdr(100, 1, 2, 3, 4).intersection(sdr(100, 3, 4, 5))
    assert sorted(result.indices.tolist()) == [3, 4]


def test_intersection_empty():
    assert sdr(100, 1, 2).intersection(sdr(100, 8, 9)).n_active == 0


def test_difference():
    result = sdr(100, 1, 2, 3, 4).difference(sdr(100, 3, 4))
    assert sorted(result.indices.tolist()) == [1, 2]


def test_subsample():
    a = SDR.random(size=500, n_active=50)
    assert a.subsample(10).n_active == 10
    assert a.subsample(99).n_active == 50


def test_match_threshold():
    a = sdr(100, 1, 2, 3, 4)
    b = sdr(100, 2, 3, 4)
    assert a.match_threshold(b, 3)
    assert not a.match_threshold(b, 4)


def test_equality_and_hash():
    a = sdr(100, 1, 2, 3)
    assert a == sdr(100, 3, 2, 1)
    assert a != sdr(100, 1, 2, 4)
    assert a != "not an sdr"
    assert hash(a) == hash(sdr(100, 3, 2, 1))


def test_union_sdr_accumulates():
    u = UnionSDR(size=100)
    u.add(sdr(100, 1, 2))
    u.add(sdr(100, 3))
    assert u.n_members == 2
    assert sorted(u.to_sdr().indices.tolist()) == [1, 2, 3]


def test_union_sdr_contains_member():
    member = sdr(100, 1, 2, 3, 4)
    u = UnionSDR(size=100)
    u.add(member)
    u.add(sdr(100, 50, 51))
    assert u.contains(member)
    assert not u.contains(sdr(100, 80, 81, 82, 83))


def test_union_sdr_remove():
    u = UnionSDR(size=100)
    u.add(sdr(100, 1, 2))
    u.add(sdr(100, 8, 9))
    u.remove(sdr(100, 8, 9))
    assert u.n_members == 1
    assert sorted(u.to_sdr().indices.tolist()) == [1, 2]


def test_similar_inputs_overlap_more_than_dissimilar():
    base = SDR.random(size=1000, n_active=40)
    near = SDR(size=1000, indices=torch.cat([base.indices[:35], torch.tensor([991, 992, 993, 994, 995])]))
    far = SDR.random(size=1000, n_active=40)
    assert base.overlap(near) > base.overlap(far)
