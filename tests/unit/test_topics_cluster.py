from __future__ import annotations

import numpy as np

from contextd.topics.cluster import Cluster, cluster_vectors, pca, split_oversize


def _blobs(seed: int = 1, per: int = 12, dim: int = 16) -> tuple[list[list[float]], list[int]]:
    rng = np.random.default_rng(seed)
    centres = [np.eye(dim)[i] * 5.0 for i in range(3)]
    vectors: list[list[float]] = []
    labels: list[int] = []
    for label, c in enumerate(centres):
        for _ in range(per):
            vectors.append([float(v) for v in c + rng.normal(scale=0.2, size=dim)])
            labels.append(label)
    return vectors, labels


def test_pca_shape_and_variance_order() -> None:
    x = np.random.default_rng(0).normal(size=(20, 8))
    p = pca(x, 3)
    assert p.shape == (20, 3)
    variances = p.var(axis=0)
    assert variances[0] >= variances[1] >= variances[2]
    assert pca(x, 100).shape[1] == 8


def test_bic_recovers_three_blobs_deterministically() -> None:
    vectors, labels = _blobs()
    a = cluster_vectors(vectors, min_members=3, seed=7)
    b = cluster_vectors(vectors, min_members=3, seed=7)
    assert len(a) == 3
    assert [c.members for c in a] == [c.members for c in b]
    for c in a:
        assert len({labels[i] for i in c.members if c.probabilities[c.members.index(i)] > 0.5}) == 1
    covered = {i for c in a for i in c.members}
    assert covered == set(range(len(vectors)))
    assert all(len(c.centroid) == 16 for c in a)


def test_small_inputs_form_one_cluster() -> None:
    assert cluster_vectors([]) == []
    out = cluster_vectors([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], min_members=3)
    assert len(out) == 1 and out[0].members == [0, 1, 2]


def test_soft_assignment_allows_multi_membership() -> None:
    vectors, _ = _blobs(per=10)
    # A point half-way between blob 0 and blob 1 should get a foot in both.
    mid = [(a + b) / 2 for a, b in zip(vectors[0], vectors[10], strict=True)]
    vectors.append(mid)
    out = cluster_vectors(vectors, min_members=3, soft_threshold=0.05, seed=3)
    memberships = [c for c in out if len(vectors) - 1 in c.members]
    assert len(memberships) >= 1
    assert len(out) == 3


def test_split_oversize_reclusters_big_clusters() -> None:
    vectors, _ = _blobs(per=10)
    one = [Cluster(list(range(len(vectors))), [1.0] * len(vectors))]
    tokens = [100] * len(vectors)
    out = split_oversize(
        vectors,
        tokens,
        one,
        max_cluster_tokens=1500,
        min_members=3,
        soft_threshold=0.1,
        pca_dims=8,
        seed=0,
    )
    assert len(out) == 3
    assert {i for c in out for i in c.members} == set(range(len(vectors)))
    kept = split_oversize(
        vectors,
        tokens,
        one,
        max_cluster_tokens=10_000,
        min_members=3,
        soft_threshold=0.1,
        pca_dims=8,
        seed=0,
    )
    assert kept == one
