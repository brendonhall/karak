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
    from karak.config import RefinementConfig

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
