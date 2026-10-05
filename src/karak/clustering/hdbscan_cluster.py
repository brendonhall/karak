"""HDBSCAN clustering on PCA-reduced mineral pixel features.

Clusters mineral pixels to identify major mineral phases. Supports
subsampling for large images with approximate_predict for remaining pixels.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import hdbscan
import numpy as np

if TYPE_CHECKING:
    from karak.core_params import HDBSCANConfig

logger = logging.getLogger(__name__)

# approximate_predict on cuML holds about min_samples * 16 bytes per
# predicted point; batches take this fraction of the free device memory
_PREDICT_MEMORY_FRACTION = 0.25


def subsample_indices(n: int, subsample_n: int | None,
                      random_state: int) -> np.ndarray | None:
    """Indices of the pixels HDBSCAN is fitted on, or None to fit on all.

    One host RNG draw, so the cpu and cuda paths fit the same subsample.
    """
    if subsample_n is None or subsample_n >= n:
        return None
    rng = np.random.default_rng(random_state)
    return rng.choice(n, size=subsample_n, replace=False)


def predict_batch_rows(min_samples: int, free_bytes: int) -> int:
    """Rows per cuML approximate_predict call that fit the memory budget."""
    per_row = max(1, min_samples) * 16
    return max(1024, int(free_bytes * _PREDICT_MEMORY_FRACTION) // per_row)


def _cuml_predict(model, features, batch_rows: int):
    """approximate_predict over ``features`` in batches; labels and
    probabilities as CuPy arrays (int32, float32). Results do not depend
    on the batch size."""
    import cupy as cp
    from cuml.cluster.hdbscan import approximate_predict

    labels, probabilities = [], []
    for start in range(0, features.shape[0], batch_rows):
        lab, prob = approximate_predict(model, features[start:start + batch_rows])
        labels.append(cp.asarray(lab).astype(cp.int32))
        probabilities.append(cp.asarray(prob).astype(cp.float32))
    return cp.concatenate(labels), cp.concatenate(probabilities)


def _run_cuml(pca_features, config, min_samples: int):
    """cuML HDBSCAN: fit on all points, or on the same subsample as the cpu
    path and then approximate_predict every point in batches."""
    from karak.accel import get_array_module
    from karak.errors import StageError

    cp = get_array_module("cuda")  # raises StageError without CUDA
    from cuml.cluster import HDBSCAN as CumlHDBSCAN

    features = cp.asarray(pca_features, dtype=cp.float32)
    n = features.shape[0]
    fit_idx = subsample_indices(n, config.subsample_n, config.random_state)
    model = CumlHDBSCAN(
        min_cluster_size=config.min_cluster_size,
        min_samples=min_samples,
        prediction_data=fit_idx is not None,
    )
    try:
        if fit_idx is None:
            logger.info("Fitting cuML HDBSCAN on all %d mineral pixels "
                        "(min_cluster_size=%d, min_samples=%d)",
                        n, config.min_cluster_size, min_samples)
            model.fit(features)
            return (cp.asarray(model.labels_).astype(cp.int32),
                    cp.asarray(model.probabilities_).astype(cp.float32), model)
        logger.info("Fitting cuML HDBSCAN on %d/%d subsampled pixels "
                    "(min_cluster_size=%d, min_samples=%d)",
                    fit_idx.size, n, config.min_cluster_size, min_samples)
        model.fit(features[cp.asarray(fit_idx)])
        free, _ = cp.cuda.Device().mem_info
        batch_rows = predict_batch_rows(min_samples, free)
        labels, probabilities = _cuml_predict(model, features, batch_rows)
        return labels, probabilities, model
    except (MemoryError, cp.cuda.memory.OutOfMemoryError) as exc:
        raise StageError(_cuml_memory_hint(n, config, min_samples, exc)) from exc
    except RuntimeError as exc:   # RMM reports std::bad_alloc this way
        if "bad_alloc" not in str(exc) and "out of memory" not in str(exc).lower():
            raise
        raise StageError(_cuml_memory_hint(n, config, min_samples, exc)) from exc


def _cuml_memory_hint(n: int, config, min_samples: int, exc) -> str:
    fitted = config.subsample_n if config.subsample_n and config.subsample_n < n else n
    return (
        f"cuML HDBSCAN ran out of device memory fitting {fitted:,} of {n:,} "
        f"pixels with min_samples={min_samples} ({exc}). Memory grows with "
        "the fitted pixel count times min_samples: lower subsample_n, or "
        "run this node on the cpu (--set NODE.device=cpu)."
    )


# The cpu fit (hdbscan package) holds about this many bytes per fitted
# point and min_samples neighbor: measured 31.3 to 33.0 on NWA 4587 features
# (200 k to 782 k points, min_samples 250 and 1000).
_FIT_BYTES_PER_NEIGHBOR = 32
# Refuse a fit whose estimate exceeds this share of the available memory.
_FIT_MEMORY_FRACTION = 0.8

# approximate_predict queries 2 * min_samples neighbors per point at once
# (an 8-byte distance and an 8-byte index each); chunks keep all workers'
# queries together under this many bytes.
_PREDICT_MEMORY_BYTES = 2 << 30
# Fewer rows than this per worker: predict serially (pool start-up costs more)
_MIN_ROWS_PER_PREDICT_WORKER = 20_000

_pool_clusterer = None   # set in each prediction pool worker


def _init_predict_worker(clusterer) -> None:
    global _pool_clusterer
    _pool_clusterer = clusterer


def _predict_chunk(points: np.ndarray):
    return hdbscan.approximate_predict(_pool_clusterer, points)


def predict_chunk_rows(min_samples: int, workers: int) -> int:
    """Rows per approximate_predict call so that ``workers`` concurrent
    calls stay within the memory budget."""
    per_row = 2 * max(1, min_samples) * 16
    return max(1000, _PREDICT_MEMORY_BYTES // max(1, workers) // per_row)


def approximate_predict_parallel(clusterer, points: np.ndarray, workers: int):
    """``hdbscan.approximate_predict`` over ``points`` in memory-bounded
    chunks, in ``workers`` processes. Each point's label and probability
    depend only on the fitted model and that point, so the result equals
    one serial call exactly, without its (n, 2 * min_samples) query arrays
    (about 25 GB for 782 k points at min_samples 1000)."""
    min_samples = clusterer.min_samples or clusterer.min_cluster_size
    if workers > 1 and len(points) >= 2 * _MIN_ROWS_PER_PREDICT_WORKER:
        workers = min(workers, len(points) // _MIN_ROWS_PER_PREDICT_WORKER)
    else:
        workers = 1
    rows = predict_chunk_rows(min_samples, workers)
    chunks = [points[i:i + rows] for i in range(0, len(points), rows)]
    if workers == 1:
        parts = [hdbscan.approximate_predict(clusterer, c) for c in chunks]
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("forkserver"),
            initializer=_init_predict_worker, initargs=(clusterer,),
        ) as pool:
            parts = list(pool.map(_predict_chunk, chunks))
    return (np.concatenate([labels for labels, _ in parts]),
            np.concatenate([probs for _, probs in parts]))


def fit_memory_bytes(n_fit: int, min_samples: int) -> int:
    """Estimated peak memory of a cpu HDBSCAN fit on ``n_fit`` points."""
    return _FIT_BYTES_PER_NEIGHBOR * n_fit * max(1, min_samples)


def check_fit_memory(n_fit: int, n_total: int, min_samples: int,
                     available: int | None) -> None:
    """StageError when a cpu fit would need more than 80 % of ``available``
    bytes (no check when the available memory is unknown), before the
    hdbscan package allocates it and the machine starts to swap."""
    if available is None:
        return
    need = fit_memory_bytes(n_fit, min_samples)
    if need <= _FIT_MEMORY_FRACTION * available:
        return
    from karak.errors import StageError

    raise StageError(
        f"HDBSCAN on the cpu would need about {need / 1e9:.0f} GB to fit "
        f"{n_fit:,} of {n_total:,} pixels with min_samples={min_samples} "
        f"({available / 1e9:.0f} GB available). Set subsample_n (see "
        "'HDBSCAN settings for full-scale runs' in the user guide), lower "
        "min_samples, or run on a downsampled cube."
    )


def run_hdbscan(
    pca_features: np.ndarray,
    config: HDBSCANConfig,
    device: str = "cpu",
    core_dist_n_jobs: int | None = None,
    predict_workers: int = 1,
) -> tuple[np.ndarray, np.ndarray, hdbscan.HDBSCAN]:
    """Run HDBSCAN on PCA-reduced mineral pixel features.

    If config.subsample_n is set and fewer than the total number of pixels,
    HDBSCAN is fitted on a random subsample and every pixel is then
    assigned via approximate_predict. Both devices fit the same subsample
    (``subsample_indices``); on cuda the prediction runs in batches sized
    from the free device memory.

    Parameters
    ----------
    pca_features : np.ndarray or cupy.ndarray
        (N_mineral, n_components) PCA-transformed mineral pixel features.
        A CuPy array is accepted only with ``device="cuda"``.
    config : HDBSCANConfig
        HDBSCAN configuration parameters.
    device : str
        Device for clustering: "cpu" (hdbscan package) or "cuda" (cuML).
    core_dist_n_jobs : int or None
        Forwarded to ``hdbscan.HDBSCAN`` when not None, to cap joblib's
        core-distance parallelism (e.g. when called from a worker pool
        where each process should stay single-threaded internally). None
        leaves hdbscan's own default (4 jobs) untouched.
    predict_workers : int
        Processes for the cpu ``approximate_predict`` after a subsample fit
        (``approximate_predict_parallel``, memory-bounded chunks). 1 =
        serial. Results are identical for any count.

    Returns
    -------
    labels : np.ndarray
        (N_mineral,) int32 cluster labels. -1 = noise/unclassified.
        On cuda, a CuPy array.
    probabilities : np.ndarray
        (N_mineral,) float32 membership probabilities [0, 1].
        On cuda, a CuPy array.
    clusterer : hdbscan.HDBSCAN
        Fitted HDBSCAN model (for approximate_predict if needed later).
    """
    from karak.accel import is_device_array

    if device == "cpu" and is_device_array(pca_features):
        from karak.errors import StageError
        raise StageError(
            "device='cpu' received a device array; the executor places "
            "inputs on the stage's device, so run this stage with device='cuda'"
        )
    if device == "cuda":
        min_samples = (config.min_samples if config.min_samples is not None
                       else config.min_cluster_size)
        return _run_cuml(pca_features, config, min_samples)

    n_mineral = pca_features.shape[0]
    min_samples = config.min_samples if config.min_samples is not None else config.min_cluster_size

    fit_idx = subsample_indices(n_mineral, config.subsample_n, config.random_state)
    from karak.memory import available_host_memory

    check_fit_memory(n_mineral if fit_idx is None else fit_idx.size, n_mineral,
                     min_samples, available_host_memory())

    hdbscan_kwargs: dict = {}
    if core_dist_n_jobs is not None:
        hdbscan_kwargs["core_dist_n_jobs"] = core_dist_n_jobs

    if fit_idx is not None:
        # Subsample fitting
        n_fit = fit_idx.size
        fit_features = pca_features[fit_idx]

        logger.info(
            "Fitting HDBSCAN on %d/%d subsampled pixels "
            "(min_cluster_size=%d, min_samples=%d)",
            n_fit, n_mineral, config.min_cluster_size, min_samples,
        )

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=config.min_cluster_size,
            min_samples=min_samples,
            prediction_data=True,
            **hdbscan_kwargs,
        )
        clusterer.fit(fit_features)

        # Assign all pixels (including fitted ones) via approximate_predict
        # for consistency
        labels_all, probs_all = approximate_predict_parallel(
            clusterer, pca_features, predict_workers)
        labels = labels_all.astype(np.int32)
        probabilities = probs_all.astype(np.float32)

    else:
        # Fit on all mineral pixels
        logger.info(
            "Fitting HDBSCAN on all %d mineral pixels "
            "(min_cluster_size=%d, min_samples=%d)",
            n_mineral, config.min_cluster_size, min_samples,
        )

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=config.min_cluster_size,
            min_samples=min_samples,
            prediction_data=True,
            **hdbscan_kwargs,
        )
        clusterer.fit(pca_features)

        labels = clusterer.labels_.astype(np.int32)
        probabilities = clusterer.probabilities_.astype(np.float32)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    noise_pct = 100.0 * n_noise / n_mineral

    logger.info(
        "HDBSCAN result: %d clusters, %d noise pixels (%.1f%%)",
        n_clusters, n_noise, noise_pct,
    )

    return labels, probabilities, clusterer


def compute_cluster_stats(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    """Compute summary statistics for HDBSCAN clustering results.

    Parameters
    ----------
    labels : np.ndarray
        (N_mineral,) int32 cluster labels (-1 = noise).
    probabilities : np.ndarray
        (N_mineral,) float32 membership probabilities.

    Returns
    -------
    dict
        Summary statistics including n_clusters, n_noise, noise_pct,
        per-cluster pixel counts, and mean probabilities.
    """
    n_total = len(labels)
    unique_labels = sorted(set(labels))
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = int(np.sum(labels == -1))
    noise_pct = 100.0 * n_noise / n_total if n_total > 0 else 0.0

    cluster_info = {}
    for lbl in unique_labels:
        if lbl == -1:
            continue
        mask = labels == lbl
        cluster_info[int(lbl)] = {
            "n_pixels": int(np.sum(mask)),
            "pct": 100.0 * np.sum(mask) / n_total,
            "mean_prob": float(np.mean(probabilities[mask])),
        }

    return {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_pct": noise_pct,
        "n_total": n_total,
        "clusters": cluster_info,
    }
