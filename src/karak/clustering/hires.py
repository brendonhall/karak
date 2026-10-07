"""Sub-phase split at a higher resolution than the working labels.

The working labels (one per mineral pixel at the pipeline's downsampled
resolution) select a region; a cube at full resolution supplies the
feature; one GMM fit classifies every full-resolution pixel of the region;
the working labels take the majority of their children.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def resolution_ratio(working_shape: tuple[int, int],
                     hires_shape: tuple[int, int]) -> int:
    """Hires pixels per working pixel, the same along both axes. The hires
    image may exceed ``working * ratio`` by up to ``ratio - 1`` pixels (odd
    source dimensions); anything else is an error."""
    (H, W), (H1, W1) = working_shape, hires_shape
    ds = max(1, int(round(H1 / H)))
    for size, hi in ((H, H1), (W, W1)):
        if not 0 <= hi - size * ds < ds:
            raise ValueError(
                f"hires shape {hires_shape} is not an integer multiple of the "
                f"working shape {working_shape} (ratio {H1 / H:.3f} x {W1 / W:.3f})")
    return ds


def _feature_image(cube_hi, element_names, feature) -> tuple[np.ndarray, np.ndarray]:
    """(values, defined) over the whole hires image for one feature token."""
    from karak.clustering.refinement import _channel_index
    from karak.clustering.features import parse_feature

    kind, names = parse_feature(feature)
    if kind == "bse":
        raise ValueError("split_hires takes a channel or a ratio, not BSE")
    if kind == "ratio":
        a = cube_hi[:, :, _channel_index(element_names, names[0])]
        b = cube_hi[:, :, _channel_index(element_names, names[1])]
        denom = a + b
        defined = denom > 0
        values = np.zeros(a.shape, dtype=np.float32)
        np.divide(a, denom, out=values, where=defined)
        return values, defined
    values = cube_hi[:, :, _channel_index(element_names, names[0])]
    return values.astype(np.float32), np.ones(values.shape, dtype=bool)


def hires_split(
    labels: np.ndarray,
    mineral_indices: np.ndarray,
    image_shape: tuple[int, int],
    cube_hi: np.ndarray,
    element_names_hi: list[str],
    target_phases: list[int],
    feature: str,
    *,
    n_components: int,
    subsample_n: int | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, list[int], dict]:
    """Classify every hires pixel of the target phases with one GMM on
    ``feature``; return ``(working_labels, hires_image, new_labels, info)``.

    New labels are ``max(labels) + 1`` onwards, ordered by ascending
    component mean of the feature. A working pixel takes the majority label
    of its defined children; a tie goes to the label of the child at
    ``(r * ds, c * ds)`` when that label is among the tied labels, else to
    the smallest tied label; a pixel
    with no defined child keeps its parent label.
    """
    from sklearn.mixture import GaussianMixture

    from karak.clustering.noise_assign import labels_to_image

    H, W = image_shape
    H1, W1 = cube_hi.shape[:2]
    ds = resolution_ratio((H, W), (H1, W1))

    label_img = labels_to_image(labels, mineral_indices, (H, W), background_value=-1)
    region = np.isin(label_img, np.asarray(target_phases))
    rep = np.repeat(np.repeat(region, ds, axis=0), ds, axis=1)
    region_hi = np.zeros((H1, W1), dtype=bool)
    region_hi[:rep.shape[0], :rep.shape[1]] = rep[:H1, :W1]

    values, defined = _feature_image(cube_hi, element_names_hi, feature)
    use = region_hi & defined
    n_undefined = int((region_hi & ~defined).sum())
    sample = values[use]
    method = f"GMM {n_components}-component on {feature} at {ds}x resolution"
    if sample.size < n_components * 10:
        raise ValueError(f"hires_split: only {sample.size} defined pixels in phases "
                         f"{target_phases}, need {n_components * 10}")

    rng = np.random.default_rng(random_state)
    fit = sample
    if subsample_n is not None and sample.size > subsample_n:
        fit = sample[rng.choice(sample.size, subsample_n, replace=False)]
    gmm = GaussianMixture(n_components=n_components, random_state=random_state,
                          n_init=5).fit(fit.reshape(-1, 1))
    comp = gmm.predict(sample.reshape(-1, 1))
    order = np.argsort(gmm.means_.ravel(), kind="stable")
    rank = np.empty(n_components, dtype=np.int64)
    rank[order] = np.arange(n_components)
    first = int(labels.max()) + 1
    new_labels = [first + i for i in range(n_components)]

    hires = np.full((H1, W1), -1, dtype=np.int16)
    hires[use] = (first + rank[comp]).astype(np.int16)

    # Majority vote per working pixel over its ds x ds children.
    padded = np.full((H * ds, W * ds), -1, dtype=np.int16)
    padded[:H1, :W1] = hires[:H * ds, :W * ds]
    blocks = padded.reshape(H, ds, W, ds)
    counts = np.stack([(blocks == lab).sum(axis=(1, 3)) for lab in new_labels])  # (k, H, W)
    best = counts.max(axis=0)
    winner = np.asarray(new_labels)[counts.argmax(axis=0)]          # smallest on ties
    tied = (counts == best[None]).sum(axis=0) > 1
    top_left = padded[0::ds, 0::ds]
    tl_idx = np.clip(top_left - first, 0, len(new_labels) - 1)
    tl_count = np.take_along_axis(counts, tl_idx[None], 0)[0]
    use_tl = tied & (top_left >= 0) & (tl_count == best)
    winner = np.where(use_tl, top_left, winner)
    result_img = np.where(region & (best > 0), winner, label_img)

    out = labels.copy()
    rows, cols = mineral_indices[:, 0], mineral_indices[:, 1]
    out[:] = result_img[rows, cols]

    info = {
        "method": method,
        "n_pixels": {lab: int((hires == lab).sum()) for lab in new_labels},
        "n_pixels_working": {lab: int((out == lab).sum()) for lab in new_labels},
        "component_means": {first + int(rank[c]): {feature: float(gmm.means_[c, 0])}
                            for c in range(n_components)},
        "n_undefined": n_undefined,
        "ratio": ds,
    }
    logger.info("%s in phases %s: %s hires px, %d undefined", method, target_phases,
                info["n_pixels"], n_undefined)
    return out, hires, new_labels, info
