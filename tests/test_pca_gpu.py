"""PCA on the device: a CuPy port of sklearn's covariance path, exact tier."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import PCA

from conftest import make_synthetic_scene, pca_cfg
from karak.accel import cuda_available
from karak.clustering.pca import _covariance_pca, fit_pca
from karak.stages.pca import PCAStage


def _spectra():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(2000, 4)).astype(np.float32)
    mix = np.array([[1, 0.5, 0, 0], [0, 1, 0.3, 0], [0, 0, 1, 0.2], [0.1, 0, 0, 1]], np.float32)
    return base @ mix


def test_pca_declares_a_device_param():
    names = [p.name for p in PCAStage.PARAMS]
    assert names[-1] == "device"
    assert PCAStage.PARAMS[-1].choices == ("cpu", "cuda")


def test_covariance_pca_matches_sklearn_on_numpy():
    """The port is checked against sklearn on the CPU, so CI covers it."""
    spectra = _spectra()
    components, mean, evr = _covariance_pca(spectra, 4, np)
    ref = PCA(n_components=4, random_state=42).fit(spectra)
    np.testing.assert_allclose(mean, ref.mean_, atol=1e-6)
    np.testing.assert_allclose(evr, ref.explained_variance_ratio_, atol=1e-6)
    np.testing.assert_allclose(components, ref.components_, atol=1e-5)   # signs included
    ours = (spectra - mean) @ components.T
    np.testing.assert_allclose(ours, ref.transform(spectra), atol=1e-4)


def test_fit_pca_on_numpy_still_returns_an_sklearn_model():
    cube = make_synthetic_scene()
    mask = cube.sum(axis=-1) > 0
    model, features, idx = fit_pca(cube, mask, pca_cfg(n_components=None, subsample_fraction=None, random_state=42))
    assert isinstance(model, PCA)
    assert features.dtype == np.float32


@pytest.mark.skipif(not cuda_available(), reason="no CUDA")
@pytest.mark.parametrize("subsample", [None, 0.5])
def test_fit_pca_on_the_device_matches_cpu(subsample):
    """Exact against sklearn in float64; close to the float32 CPU path.

    sklearn fits float32 input in float32. On this scene components 2 and 3
    have near-equal eigenvalues (about 4.0e-4 and 3.8e-4 against 0.18), so
    float32 rounding turns them by up to 4e-3 and moves the features by up
    to 3.4e-4. The device path works in float64, so it is compared exactly
    with sklearn run on float64 input, and loosely with the CPU path.
    """
    import cupy as cp

    from karak.clustering.pca import PCAModel

    cube = make_synthetic_scene()
    mask = cube.sum(axis=-1) > 0
    cfg = pca_cfg(n_components=None, subsample_fraction=subsample, random_state=42)
    model, features, idx = fit_pca(cube, mask, cfg)
    gmodel, gfeatures, gidx = fit_pca(cp.asarray(cube), cp.asarray(mask), cfg)
    assert isinstance(gmodel, PCAModel)
    assert isinstance(gmodel.explained_variance_ratio_, np.ndarray)
    assert isinstance(gfeatures, cp.ndarray) and gfeatures.dtype == cp.float32
    assert isinstance(gidx, cp.ndarray) and gidx.dtype == cp.int32
    np.testing.assert_array_equal(cp.asnumpy(gidx), idx)

    spectra = cube[mask].astype(np.float64)
    fit = spectra
    if subsample is not None:
        rng = np.random.default_rng(42)
        fit = spectra[rng.choice(len(spectra), int(len(spectra) * subsample), replace=False)]
    ref = PCA(n_components=spectra.shape[1], random_state=42).fit(fit)
    np.testing.assert_allclose(gmodel.components_, ref.components_, atol=1e-9)
    np.testing.assert_allclose(gmodel.explained_variance_ratio_,
                               ref.explained_variance_ratio_, atol=1e-12)
    np.testing.assert_allclose(cp.asnumpy(gfeatures), ref.transform(spectra), atol=1e-6)

    np.testing.assert_allclose(gmodel.explained_variance_ratio_,
                               model.explained_variance_ratio_, atol=1e-5)
    np.testing.assert_allclose(cp.asnumpy(gfeatures), features, atol=1e-3)
