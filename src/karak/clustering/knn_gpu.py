"""Distance-weighted k-NN label voting on the GPU (CuPy).

Matches ``sklearn.neighbors.KNeighborsClassifier(weights="distance")`` as
``assign_noise_pixels`` uses it: Euclidean distance, each neighbor weighted
by 1/distance (a query with zero-distance neighbors counts only those), and
the lowest label wins a tied vote.

A brute-force CuPy kernel keeps each query's ``k + extra`` nearest candidates
in registers while the reference set streams through shared memory. The
candidates' distances are then recomputed in float64, as sklearn's tree
does, re-ranked, and the top ``k`` vote. The extra candidates absorb rank
swaps between the float32 search and the float64 distances. Large ``k`` or
many features fall back to cuML's brute-force NearestNeighbors for the
search, with the same re-rank and vote.

CuPy is imported lazily: this module loads without the cuda extra.
"""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

# Register and shared-memory limits of the kernel; beyond them, use cuML.
MAX_KERNEL_CANDIDATES = 32
MAX_KERNEL_FEATURES = 48
_THREADS = 256
# Batches of queries take this fraction of the free device memory.
_MEMORY_FRACTION = 0.5

_KERNEL = r"""
extern "C" __global__ void knn_candidates(
    const float* __restrict__ ref, const int n_ref,
    const float* __restrict__ query, const int n_query,
    float* __restrict__ out_dist, int* __restrict__ out_idx)
{
    const int D = %(D)d, K = %(K)d, T = %(T)d;
    __shared__ float tile[T * D];
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    float x[D];
    for (int j = 0; j < D; ++j) x[j] = (q < n_query) ? query[(size_t)q * D + j] : 0.f;
    float best_d[K];
    int best_i[K];
    for (int j = 0; j < K; ++j) { best_d[j] = 3.4e38f; best_i[j] = -1; }
    for (int base = 0; base < n_ref; base += T) {
        const int n = min(T, n_ref - base);
        __syncthreads();
        for (int t = threadIdx.x; t < n * D; t += blockDim.x)
            tile[t] = ref[(size_t)base * D + t];
        __syncthreads();
        for (int r = 0; r < n; ++r) {
            float s = 0.f;
            #pragma unroll
            for (int j = 0; j < D; ++j) {
                const float d = x[j] - tile[r * D + j];
                s = fmaf(d, d, s);
            }
            // strict < keeps the lower reference index on equal distances
            if (s < best_d[K - 1]) {
                int p = K - 1;
                while (p > 0 && best_d[p - 1] > s) {
                    best_d[p] = best_d[p - 1]; best_i[p] = best_i[p - 1]; --p;
                }
                best_d[p] = s; best_i[p] = base + r;
            }
        }
    }
    if (q < n_query)
        for (int j = 0; j < K; ++j) {
            out_dist[(size_t)q * K + j] = best_d[j];
            out_idx[(size_t)q * K + j] = best_i[j];
        }
}
"""


def n_candidates(k: int, n_ref: int) -> int:
    """Candidates kept per query before the float64 re-rank."""
    return min(n_ref, k + max(3, k // 2))


@lru_cache(maxsize=None)
def _kernel(n_features: int, n_cand: int):
    import cupy as cp

    source = _KERNEL % {"D": n_features, "K": n_cand, "T": _THREADS}
    return cp.RawKernel(source, "knn_candidates")


def _uses_kernel(n_features: int, n_cand: int) -> bool:
    return n_cand <= MAX_KERNEL_CANDIDATES and n_features <= MAX_KERNEL_FEATURES


def batch_rows(n_features: int, n_cand: int, n_classes: int, free_bytes: int) -> int:
    """Queries per batch that fit the device-memory budget."""
    per_query = n_cand * (16 * n_features + 16) + 8 * n_classes + 64
    return max(1024, int(free_bytes * _MEMORY_FRACTION) // per_query)


def _candidates(ref, queries, n_cand: int, nn=None):
    """(n_query, n_cand) candidate indices into ``ref``, nearest first."""
    import cupy as cp

    n_query, n_features = queries.shape
    if nn is not None:
        _, idx = nn.kneighbors(queries)
        return cp.asarray(idx).astype(cp.int32)
    dist = cp.empty((n_query, n_cand), cp.float32)
    idx = cp.empty((n_query, n_cand), cp.int32)
    _kernel(n_features, n_cand)(
        ((n_query + _THREADS - 1) // _THREADS,), (_THREADS,),
        (ref, cp.int32(ref.shape[0]), queries, cp.int32(n_query), dist, idx),
    )
    return idx


def vote(distances, neighbor_classes, n_classes: int):
    """sklearn's distance-weighted vote: class index per row.

    ``distances`` (n, k) float64, nearest first; ``neighbor_classes`` (n, k)
    class indices. Rows with a zero distance weight only those neighbors.
    argmax picks the lowest class index on a tie, as sklearn does.
    """
    import cupy as cp

    zero = distances == 0
    any_zero = zero.any(axis=1, keepdims=True)
    weights = cp.where(any_zero, zero.astype(cp.float64),
                       1.0 / cp.where(zero, 1.0, distances))
    rows = cp.arange(distances.shape[0])
    score = cp.zeros((distances.shape[0], n_classes), cp.float64)
    for j in range(distances.shape[1]):   # one neighbor rank at a time: no
        score[rows, neighbor_classes[:, j]] += weights[:, j]   # duplicate cells
    return cp.argmax(score, axis=1)


def knn_vote_cuda(ref_features, ref_labels, query_features, k: int):
    """Label of each query by distance-weighted k-NN over the reference set.

    All arrays are CuPy (features float32, C-contiguous; labels int). Returns
    int32 labels on the device.
    """
    import cupy as cp

    ref = cp.ascontiguousarray(ref_features, dtype=cp.float32)
    queries = cp.ascontiguousarray(query_features, dtype=cp.float32)
    classes, ref_class = cp.unique(ref_labels, return_inverse=True)
    ref_class = ref_class.astype(cp.int32).ravel()
    n_ref, n_features = ref.shape
    k = min(k, n_ref)
    n_cand = n_candidates(k, n_ref)

    nn = None
    if not _uses_kernel(n_features, n_cand):
        from cuml.neighbors import NearestNeighbors

        logger.info("k-NN on the device: cuML brute force (%d candidates, "
                    "%d features)", n_cand, n_features)
        nn = NearestNeighbors(n_neighbors=n_cand, algorithm="brute").fit(ref)

    free, _ = cp.cuda.Device().mem_info
    rows = batch_rows(n_features, n_cand, classes.size, free)
    out = cp.empty(queries.shape[0], cp.int32)
    for start in range(0, queries.shape[0], rows):
        q = queries[start:start + rows]
        idx = _candidates(ref, q, n_cand, nn)
        # float64 distances of the candidates, then the k nearest of them
        diff = ref[idx].astype(cp.float64) - q[:, None, :].astype(cp.float64)
        dist = cp.sqrt((diff * diff).sum(axis=2))
        order = cp.argsort(dist, axis=1)[:, :k]   # CuPy sorts are stable
        dist = cp.take_along_axis(dist, order, axis=1)
        idx = cp.take_along_axis(idx, order, axis=1)
        out[start:start + rows] = classes[vote(dist, ref_class[idx], classes.size)]
    return out
