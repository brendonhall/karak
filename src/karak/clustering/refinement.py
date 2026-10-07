"""Post-clustering phase refinement: olivine extraction and pyroxene GMM split.

Applied after kNN noise reassignment to split composite phases
(e.g., pyroxene → olivine + pigeonite + augite) using threshold-based
extraction and Gaussian Mixture Models.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from karak.core_params import RefinementConfig

logger = logging.getLogger(__name__)


def _extract_olivine(
    cleaned_labels: np.ndarray,
    denoised_cube: np.ndarray,
    mineral_indices: np.ndarray,
    element_names: list[str],
    target_phase: int,
    fe_threshold: float,
    ca_threshold: float,
) -> tuple[np.ndarray, int]:
    """Extract olivine pixels from target phase using Fe/Ca thresholds.

    Parameters
    ----------
    cleaned_labels : (N,) int array of cluster labels for mineral pixels.
    denoised_cube : (H,W,C) float32 denoised intensity cube.
    mineral_indices : (N,2) int array of (row, col) for mineral pixels.
    element_names : list of element channel names.
    target_phase : cluster label to extract from.
    fe_threshold : minimum Fe-K intensity for olivine.
    ca_threshold : maximum Ca intensity for olivine.

    Returns
    -------
    updated_labels : (N,) int array with olivine pixels re-labeled.
    olivine_label : the new label assigned to olivine pixels (-1 if none extracted).
    """
    # Find element indices
    fe_idx = None
    ca_idx = None
    for i, name in enumerate(element_names):
        if name.lower() in ("fe-k", "fe_k", "fe"):
            fe_idx = i
        if name.lower() == "ca":
            ca_idx = i

    if fe_idx is None or ca_idx is None:
        logger.warning(
            "Cannot extract olivine: Fe-K (found=%s) or Ca (found=%s) not in element_names",
            fe_idx is not None, ca_idx is not None,
        )
        return cleaned_labels, -1

    # Get pixel locations for target phase
    phase_mask = cleaned_labels == target_phase
    n_phase = int(phase_mask.sum())
    if n_phase == 0:
        logger.warning("Target phase %d has no pixels", target_phase)
        return cleaned_labels, -1

    rows = mineral_indices[phase_mask, 0]
    cols = mineral_indices[phase_mask, 1]

    # Extract denoised Fe and Ca values
    fe_vals = denoised_cube[rows, cols, fe_idx]
    ca_vals = denoised_cube[rows, cols, ca_idx]

    # Apply thresholds
    olivine_mask_local = (fe_vals > fe_threshold) & (ca_vals < ca_threshold)
    n_olivine = int(olivine_mask_local.sum())

    if n_olivine == 0:
        logger.info("No olivine pixels found (Fe>%.2f & Ca<%.2f)", fe_threshold, ca_threshold)
        return cleaned_labels, -1

    # Assign new label (max existing + 1)
    new_label = int(cleaned_labels.max()) + 1
    updated = cleaned_labels.copy()
    phase_indices = np.where(phase_mask)[0]
    updated[phase_indices[olivine_mask_local]] = new_label

    pct = 100 * n_olivine / n_phase
    logger.info(
        "Olivine extraction: %d pixels (%.1f%% of phase %d) -> label %d",
        n_olivine, pct, target_phase, new_label,
    )

    return updated, new_label


_OPERATORS = {
    "<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal,
}


def _channel_index(element_names: list[str], channel: str) -> int:
    lowered = [n.lower() for n in element_names]
    if channel.lower() not in lowered:
        raise ValueError(f"channel {channel!r} not in the cube: {list(element_names)}")
    return lowered.index(channel.lower())


def threshold_split(
    cleaned_labels: np.ndarray,
    denoised_cube: np.ndarray,
    mineral_indices: np.ndarray,
    element_names: list[str],
    target_phase: int,
    rules: list[tuple[str, str, float]],
) -> tuple[np.ndarray, int, int]:
    """Move the target-phase pixels that satisfy every rule to a new label.

    ``rules`` are ``(channel, operator, value)`` with operators ``<``,
    ``<=``, ``>``, ``>=`` on the denoised cube. Returns ``(labels,
    new_label, n_moved)``; ``new_label`` is ``max(labels) + 1``, or -1 when
    no pixel moves (labels come back as an unchanged copy).
    """
    phase_mask = cleaned_labels == target_phase
    rows, cols = mineral_indices[phase_mask, 0], mineral_indices[phase_mask, 1]
    keep = np.ones(int(phase_mask.sum()), dtype=bool)
    for channel, op, value in rules:
        values = denoised_cube[rows, cols, _channel_index(element_names, channel)]
        keep &= _OPERATORS[op](values, value)
    n_moved = int(keep.sum())
    updated = cleaned_labels.copy()
    if n_moved == 0:
        logger.info("threshold split of phase %d: no pixel satisfies %s", target_phase, rules)
        return updated, -1, 0
    new_label = int(cleaned_labels.max()) + 1
    updated[np.where(phase_mask)[0][keep]] = new_label
    logger.info("threshold split of phase %d: %d pixels -> label %d (%s)",
                target_phase, n_moved, new_label, rules)
    return updated, new_label, n_moved


def _gmm_split(
    cleaned_labels: np.ndarray,
    denoised_cube: np.ndarray,
    bse: np.ndarray,
    mineral_indices: np.ndarray,
    element_names: list[str],
    target_phase: int,
    n_components: int,
    features: list[str],
    bse_weight: float,
    subsample_n: int | None,
    random_state: int,
) -> tuple[np.ndarray, list[int]]:
    """Split target phase into sub-phases using Gaussian Mixture Model.

    Parameters
    ----------
    cleaned_labels : (N,) int array of cluster labels for mineral pixels.
    denoised_cube : (H,W,C) float32 denoised intensity cube.
    bse : (H,W) float32 BSE image.
    mineral_indices : (N,2) int array of (row, col) for mineral pixels.
    element_names : list of element channel names.
    target_phase : cluster label to split.
    n_components : number of GMM components.
    features : list of feature names (element names + 'BSE').
    bse_weight : weight multiplier for BSE feature.
    subsample_n : max pixels to fit GMM on (None = all).
    random_state : random seed.

    Returns
    -------
    updated_labels : (N,) int array with split phase pixels re-labeled.
    new_labels : list of new label values assigned to split sub-phases.
    """
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    phase_mask = cleaned_labels == target_phase
    n_phase = int(phase_mask.sum())
    if n_phase < n_components * 10:
        logger.warning(
            "Phase %d has only %d pixels, too few for %d-component GMM",
            target_phase, n_phase, n_components,
        )
        return cleaned_labels, []

    rows = mineral_indices[phase_mask, 0]
    cols = mineral_indices[phase_mask, 1]

    # Build feature matrix
    feature_cols = []
    feature_names_used = []

    # Build element name -> index map
    elem_map = {name.lower(): i for i, name in enumerate(element_names)}

    # Check if Ca and Mg both present for ratio feature
    has_ca = "ca" in elem_map
    has_mg = "mg" in elem_map

    for feat in features:
        if feat.upper() == "BSE":
            bse_vals = bse[rows, cols].astype(np.float32)
            feature_cols.append(bse_vals)
            feature_names_used.append("BSE")
        else:
            idx = elem_map.get(feat.lower())
            if idx is None:
                logger.warning("Feature '%s' not in element_names, skipping", feat)
                continue
            feature_cols.append(denoised_cube[rows, cols, idx])
            feature_names_used.append(feat)

    # Add Ca/(Ca+Mg) ratio if both present
    if has_ca and has_mg:
        ca_vals = denoised_cube[rows, cols, elem_map["ca"]]
        mg_vals = denoised_cube[rows, cols, elem_map["mg"]]
        denom = ca_vals + mg_vals
        ratio = np.where(denom > 0, ca_vals / denom, 0.0)
        feature_cols.append(ratio.astype(np.float32))
        feature_names_used.append("Ca/(Ca+Mg)")

    if len(feature_cols) < 2:
        logger.warning("Only %d features available, need at least 2 for GMM", len(feature_cols))
        return cleaned_labels, []

    X = np.column_stack(feature_cols).astype(np.float32)

    # Z-score normalize within this phase
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply BSE weight
    if bse_weight != 1.0:
        for i, name in enumerate(feature_names_used):
            if name == "BSE":
                X_scaled[:, i] *= bse_weight

    logger.info(
        "GMM split: %d pixels, %d features (%s), %d components",
        n_phase, X_scaled.shape[1], ", ".join(feature_names_used), n_components,
    )

    # Subsample for fitting if needed
    rng = np.random.default_rng(random_state)
    if subsample_n is not None and n_phase > subsample_n:
        fit_idx = rng.choice(n_phase, subsample_n, replace=False)
        X_fit = X_scaled[fit_idx]
        logger.info("  Subsampled %d -> %d for GMM fitting", n_phase, subsample_n)
    else:
        X_fit = X_scaled

    # Fit GMM
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=random_state,
        n_init=5,
    )
    gmm.fit(X_fit)

    # Predict all pixels
    sub_labels = gmm.predict(X_scaled)
    proba = gmm.predict_proba(X_scaled)

    # Log component sizes and confidence
    for comp in range(n_components):
        comp_mask = sub_labels == comp
        n_comp = int(comp_mask.sum())
        mean_conf = float(proba[comp_mask, comp].mean()) if n_comp > 0 else 0.0
        pct = 100 * n_comp / n_phase
        logger.info(
            "  Component %d: %d pixels (%.1f%%), mean confidence: %.3f",
            comp, n_comp, pct, mean_conf,
        )

    # Assign new labels: keep the largest component as the original label,
    # assign new labels to smaller components
    comp_sizes = [(int((sub_labels == c).sum()), c) for c in range(n_components)]
    comp_sizes.sort(reverse=True)  # largest first

    updated = cleaned_labels.copy()
    phase_indices = np.where(phase_mask)[0]
    new_labels = []

    next_label = int(cleaned_labels.max()) + 1

    for rank, (size, comp) in enumerate(comp_sizes):
        comp_pixel_mask = sub_labels == comp
        if rank == 0:
            # Largest component keeps original label
            logger.info(
                "  Component %d (largest, %d px) -> keeps label %d",
                comp, size, target_phase,
            )
        else:
            # Smaller components get new labels
            updated[phase_indices[comp_pixel_mask]] = next_label
            new_labels.append(next_label)
            logger.info(
                "  Component %d (%d px) -> new label %d",
                comp, size, next_label,
            )
            next_label += 1

    return updated, new_labels


def refine_phases(
    cleaned_labels: np.ndarray,
    denoised_cube: np.ndarray,
    bse: np.ndarray,
    mineral_indices: np.ndarray,
    element_names: list[str],
    config: RefinementConfig,
) -> np.ndarray:
    """Apply post-clustering refinement to split composite phases.

    Runs olivine extraction first (if enabled), then GMM split on the
    remaining target phase pixels (if enabled).

    Parameters
    ----------
    cleaned_labels : (N,) int array of cluster labels for mineral pixels.
    denoised_cube : (H,W,C) float32 denoised intensity cube.
    bse : (H,W) float32 BSE image.
    mineral_indices : (N,2) int array of (row, col) for mineral pixels.
    element_names : list of element channel names.
    config : RefinementConfig with olivine and gmm_split settings.

    Returns
    -------
    refined_labels : (N,) int array with refined cluster labels.
    """
    if not config.enabled:
        return cleaned_labels

    labels = cleaned_labels
    target = config.target_phase

    # Verify target phase exists
    if not np.any(labels == target):
        logger.warning("Target phase %d not found in labels, skipping refinement", target)
        return labels

    n_target = int(np.sum(labels == target))
    logger.info(
        "Phase refinement: target phase %d (%d pixels)",
        target, n_target,
    )

    # Step 1: Olivine extraction
    if config.olivine.enabled:
        labels, olivine_label = _extract_olivine(
            labels,
            denoised_cube,
            mineral_indices,
            element_names,
            target_phase=target,
            fe_threshold=config.olivine.fe_threshold,
            ca_threshold=config.olivine.ca_threshold,
        )
        if olivine_label >= 0:
            n_remaining = int(np.sum(labels == target))
            logger.info(
                "  After olivine extraction: %d pixels remain in phase %d",
                n_remaining, target,
            )

    # Step 2: GMM split on remaining target phase pixels
    if config.gmm_split.enabled:
        labels, new_labels = _gmm_split(
            labels,
            denoised_cube,
            bse,
            mineral_indices,
            element_names,
            target_phase=target,
            n_components=config.gmm_split.n_components,
            features=config.gmm_split.features,
            bse_weight=config.gmm_split.bse_weight,
            subsample_n=config.gmm_split.subsample_n,
            random_state=config.gmm_split.random_state,
        )
        if new_labels:
            logger.info(
                "  GMM split created %d new labels: %s",
                len(new_labels), new_labels,
            )

    n_phases = len(set(labels.tolist()) - {-1})
    logger.info("Phase refinement complete: %d total phases", n_phases)

    return labels


def _feature_matrix(
    denoised_cube, bse, rows, cols, element_names, features,
) -> tuple[np.ndarray, list[str]]:
    """Columns for the feature tokens (channel, BSE, A/(A+B)) at the given
    pixels; ValueError names an unknown channel or a missing BSE image."""
    from karak.clustering.features import parse_feature

    columns, used = [], []
    for token in features:
        kind, names = parse_feature(token)
        if kind == "bse":
            if bse is None:
                raise ValueError("feature 'BSE' needs the bse input")
            columns.append(bse[rows, cols].astype(np.float32))
        elif kind == "ratio":
            a = denoised_cube[rows, cols, _channel_index(element_names, names[0])]
            b = denoised_cube[rows, cols, _channel_index(element_names, names[1])]
            denom = a + b
            columns.append(np.where(denom > 0, a / np.where(denom > 0, denom, 1), 0.0)
                           .astype(np.float32))
        else:
            columns.append(denoised_cube[rows, cols, _channel_index(element_names, names[0])])
        used.append(token.strip())
    return np.column_stack(columns).astype(np.float32), used


def gmm_split(
    cleaned_labels: np.ndarray,
    denoised_cube: np.ndarray,
    bse: np.ndarray | None,
    mineral_indices: np.ndarray,
    element_names: list[str],
    target_phase: int,
    features: list[str],
    *,
    n_components: int,
    bse_weight: float,
    subsample_n: int | None,
    random_state: int,
    keep_parent: bool,
    order_by: str,
) -> tuple[np.ndarray, list[int], dict]:
    """Split the target phase with a GMM on z-scored features.

    ``keep_parent`` True: the largest component keeps the parent label and
    the others get new labels, ordered by ascending mean of ``order_by``
    (or by size, descending, when ``order_by`` is empty). False: every
    component gets a new label in that order and the parent empties.
    Returns ``(labels, new_labels, info)``.
    """
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    phase_mask = cleaned_labels == target_phase
    n_phase = int(phase_mask.sum())
    method = f"GMM {n_components}-component on {'+'.join(t.strip() for t in features)}"
    if n_phase < n_components * 10:
        logger.warning("GMM split of phase %d: only %d pixels, need %d; skipped",
                       target_phase, n_phase, n_components * 10)
        return cleaned_labels.copy(), [], {"method": method, "skipped":
                                            f"{n_phase} pixels < {n_components * 10}"}
    rows, cols = mineral_indices[phase_mask, 0], mineral_indices[phase_mask, 1]
    X, used = _feature_matrix(denoised_cube, bse, rows, cols, element_names, features)
    X_scaled = StandardScaler().fit_transform(X)
    for i, name in enumerate(used):
        if name.upper() == "BSE" and bse_weight != 1.0:
            X_scaled[:, i] *= bse_weight

    rng = np.random.default_rng(random_state)
    X_fit = X_scaled
    if subsample_n is not None and n_phase > subsample_n:
        X_fit = X_scaled[rng.choice(n_phase, subsample_n, replace=False)]
    gmm = GaussianMixture(n_components=n_components, covariance_type="full",
                          random_state=random_state, n_init=5).fit(X_fit)
    comp = gmm.predict(X_scaled)
    sizes = np.bincount(comp, minlength=n_components)

    if order_by.strip():
        if order_by.strip() not in used:
            raise ValueError(f"order_by {order_by!r} is not one of the features {used}")
        column = used.index(order_by.strip())
        means = np.array([X[comp == c, column].mean() if sizes[c] else np.inf
                          for c in range(n_components)])
        order = [int(c) for c in np.argsort(means, kind="stable")]
    else:
        order = [int(c) for c in np.argsort(-sizes, kind="stable")]

    updated = cleaned_labels.copy()
    phase_indices = np.where(phase_mask)[0]
    next_label = int(cleaned_labels.max()) + 1
    new_labels: list[int] = []
    assigned: dict[int, int] = {}
    if keep_parent:
        largest = int(np.argmax(sizes))
        assigned[largest] = target_phase
        order = [c for c in order if c != largest]
    for c in order:
        if sizes[c] == 0:
            continue
        assigned[c] = next_label
        new_labels.append(next_label)
        next_label += 1
    for c, label in assigned.items():
        updated[phase_indices[comp == c]] = label

    info = {
        "method": method,
        "n_pixels": {label: int(sizes[c]) for c, label in assigned.items()
                     if label != target_phase},
        "component_means": {
            label: {name: float(X[comp == c, i].mean()) for i, name in enumerate(used)}
            for c, label in assigned.items()
        },
    }
    logger.info("%s of phase %d: %s", method, target_phase,
                {label: int(sizes[c]) for c, label in assigned.items()})
    return updated, new_labels, info
