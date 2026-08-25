"""Soft clustering of embedding vectors: PCA → GMM with BIC model selection.

The RAPTOR recipe (UMAP → GMM with BIC → soft assignment → re-cluster
oversize clusters) reduced to what numpy alone can do well: a PCA
projection replaces UMAP (deterministic, no extra dependency; adequate for
the tens-to-thousands of section vectors a corpus yields), and a
diagonal-covariance Gaussian mixture fitted by EM with k-means++ seeding
replaces scikit-learn's ``GaussianMixture``. The number of components is
chosen by BIC over ``1..max_components``.

Everything here is deterministic for a given ``seed`` and independent of the
graph, so it is unit-testable with synthetic blobs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass
class Cluster:
    members: list[int]
    """Indices into the input vector list."""
    probabilities: list[float]
    """Responsibility of this cluster for each member (aligned with ``members``)."""
    centroid: list[float] = field(default_factory=list)


def _normalise(x: FloatArray) -> FloatArray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return x / norms


def pca(x: FloatArray, dims: int) -> FloatArray:
    """Project rows of ``x`` onto the top ``dims`` principal components."""
    n, d = x.shape
    k = max(1, min(dims, n - 1, d))
    centred = x - x.mean(axis=0, keepdims=True)
    # Economy SVD: rows are samples, so V^T holds the component directions.
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return np.asarray(centred @ vt[:k].T, dtype=np.float64)


def _kmeanspp(x: FloatArray, k: int, rng: np.random.Generator) -> FloatArray:
    n = x.shape[0]
    centres = [x[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min(((x[:, None, :] - np.array(centres)[None, :, :]) ** 2).sum(axis=2), axis=1)
        total = d2.sum()
        if total <= 0.0:
            centres.append(x[rng.integers(n)])
            continue
        centres.append(x[rng.choice(n, p=d2 / total)])
    return np.array(centres)


def _log_gauss_diag(x: FloatArray, mean: FloatArray, var: FloatArray) -> FloatArray:
    """Log density of every row of ``x`` under a diagonal Gaussian."""
    d = x.shape[1]
    diff2 = (x - mean) ** 2
    dens = -0.5 * (d * math.log(2 * math.pi) + np.log(var).sum() + (diff2 / var).sum(axis=1))
    return np.asarray(dens, dtype=np.float64)


@dataclass
class _Gmm:
    weights: FloatArray
    means: FloatArray
    variances: FloatArray
    log_likelihood: float
    responsibilities: FloatArray

    @property
    def k(self) -> int:
        return int(self.weights.shape[0])


def _fit_gmm(x: FloatArray, k: int, rng: np.random.Generator, *, iters: int = 100) -> _Gmm:
    n, d = x.shape
    means = _kmeanspp(x, k, rng)
    variances = np.full((k, d), x.var(axis=0).mean() + 1e-6)
    weights = np.full(k, 1.0 / k)
    log_r = np.zeros((n, k))
    prev_ll = -math.inf
    ll = -math.inf
    for _ in range(iters):
        # E-step
        for j in range(k):
            log_r[:, j] = math.log(max(weights[j], 1e-12)) + _log_gauss_diag(
                x, means[j], variances[j]
            )
        log_norm = np.logaddexp.reduce(log_r, axis=1)
        ll = float(log_norm.sum())
        r = np.exp(log_r - log_norm[:, None])
        # M-step
        nk = r.sum(axis=0) + 1e-9
        weights = nk / n
        means = (r.T @ x) / nk[:, None]
        for j in range(k):
            diff2 = (x - means[j]) ** 2
            variances[j] = (r[:, j][:, None] * diff2).sum(axis=0) / nk[j] + 1e-6
        if abs(ll - prev_ll) < 1e-6:
            break
        prev_ll = ll
    return _Gmm(weights, means, variances, ll, r)


def _bic(model: _Gmm, n: int, d: int) -> float:
    params = model.k * (2 * d) + (model.k - 1)
    return -2.0 * model.log_likelihood + params * math.log(max(n, 2))


def cluster_vectors(
    vectors: list[list[float]],
    *,
    min_members: int = 3,
    soft_threshold: float = 0.1,
    pca_dims: int = 32,
    seed: int = 0,
    max_components: int = 50,
) -> list[Cluster]:
    """Soft-cluster ``vectors``; every input index lands in at least one cluster.

    Returns clusters in descending size order. Fewer than ``2 * min_members``
    inputs yield a single cluster (nothing to separate). Clusters smaller
    than ``min_members`` are dissolved, their members re-assigned to the
    remaining cluster with the highest responsibility.
    """
    n = len(vectors)
    if n == 0:
        return []
    if n < 2 * min_members:
        return [Cluster(list(range(n)), [1.0] * n, _centroid(vectors, list(range(n))))]
    x = pca(_normalise(np.asarray(vectors, dtype=np.float64)), pca_dims)
    rng = np.random.default_rng(seed)
    k_max = max(1, min(max_components, n // min_members))
    best: _Gmm | None = None
    best_bic = math.inf
    for k in range(1, k_max + 1):
        model = _fit_gmm(x, k, rng)
        score = _bic(model, n, x.shape[1])
        if score < best_bic:
            best, best_bic = model, score
    assert best is not None
    r = best.responsibilities
    argmax = r.argmax(axis=1)
    sizes = np.bincount(argmax, minlength=best.k)
    alive = [j for j in range(best.k) if sizes[j] >= min_members] or [int(sizes.argmax())]
    clusters: list[Cluster] = []
    for j in alive:
        members = [i for i in range(n) if r[i, j] >= soft_threshold or argmax[i] == j]
        if argmax_rescued := [
            i
            for i in range(n)
            if argmax[i] not in alive and i not in members and _best_alive(r[i], alive) == j
        ]:
            members.extend(argmax_rescued)
        members.sort()
        clusters.append(
            Cluster(members, [float(r[i, j]) for i in members], _centroid(vectors, members))
        )
    clusters.sort(key=lambda c: -len(c.members))
    return clusters


def _best_alive(row: FloatArray, alive: list[int]) -> int:
    return max(alive, key=lambda j: float(row[j]))


def _centroid(vectors: list[list[float]], members: list[int]) -> list[float]:
    if not members:
        return []
    arr = np.asarray([vectors[i] for i in members], dtype=np.float64)
    c = arr.mean(axis=0)
    norm = float(np.linalg.norm(c))
    return [float(v) for v in (c / norm if norm > 0 else c)]


def split_oversize(
    vectors: list[list[float]],
    token_counts: list[int],
    clusters: list[Cluster],
    *,
    max_cluster_tokens: int,
    min_members: int,
    soft_threshold: float,
    pca_dims: int,
    seed: int,
    depth: int = 0,
) -> list[Cluster]:
    """Re-cluster any cluster whose members exceed ``max_cluster_tokens``
    (RAPTOR's ``max_length_in_cluster``), recursing at most three levels."""
    out: list[Cluster] = []
    for c in clusters:
        total = sum(token_counts[i] for i in c.members)
        if total <= max_cluster_tokens or len(c.members) < 2 * min_members or depth >= 3:
            out.append(c)
            continue
        sub = cluster_vectors(
            [vectors[i] for i in c.members],
            min_members=min_members,
            soft_threshold=soft_threshold,
            pca_dims=pca_dims,
            seed=seed + depth + 1,
        )
        if len(sub) <= 1:
            out.append(c)
            continue
        remapped = [
            Cluster(
                [c.members[i] for i in s.members],
                s.probabilities,
                _centroid(vectors, [c.members[i] for i in s.members]),
            )
            for s in sub
        ]
        out.extend(
            split_oversize(
                vectors,
                token_counts,
                remapped,
                max_cluster_tokens=max_cluster_tokens,
                min_members=min_members,
                soft_threshold=soft_threshold,
                pca_dims=pca_dims,
                seed=seed,
                depth=depth + 1,
            )
        )
    return out
