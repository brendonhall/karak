"""End-to-end: builtin `global` flow on the synthetic scene via the executor."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from conftest import make_synthetic_scene
from karak.flow.builtins import builtin_flow, override_params
from karak.flow.executor import run


@pytest.fixture(scope="module")
def scene_dir(tmp_path_factory):
    """Synthetic scene rendered as palette PNGs + grayscale SEM."""
    import imageio.v2 as imageio
    import matplotlib

    import karak.io.loaders as loaders

    tmp_path = tmp_path_factory.mktemp("scene")

    # Isolate the LUT cache for the whole module
    orig_dir, orig_mem = loaders._CACHE_DIR, loaders._FULL_LUT_CACHE
    loaders._CACHE_DIR = tmp_path / "lut_cache"
    loaders._FULL_LUT_CACHE = {}

    palette = (
        np.array(
            [matplotlib.colormaps["jet"](s)[:3] for s in np.linspace(0, 1, 64)]
        ) * 255
    ).astype(np.uint8)
    lut_path = tmp_path / "palette.npy"
    np.save(lut_path, palette)

    cube = make_synthetic_scene()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for i, element in enumerate(("A", "B", "C")):
        indices = np.round(cube[:, :, i] * 63).astype(np.uint8)
        imageio.imwrite(data_dir / f"s-01-{element}.png", palette[indices])
    imageio.imwrite(
        data_dir / "s-01-SEM.png", np.zeros((64, 64), dtype=np.uint8)
    )

    yield data_dir, f"lut:{lut_path}"

    loaders._CACHE_DIR, loaders._FULL_LUT_CACHE = orig_dir, orig_mem


def _test_graph(scene_dir):
    data_dir, colormap = scene_dir
    return override_params(builtin_flow("global"), {
        "src.colormap": colormap,
        "src.exclude_elements": "",
        "src.header_trim_px": 0,
        "src.downsample_factor": 1,
        "msk.min_object_size": 10,
        "pca.n_components": 3,
        "pca.random_state": 0,
        "hdb.min_cluster_size": 100,
        "hdb.random_state": 0,
    })


def test_global_flow_end_to_end(tmp_path, scene_dir):
    data_dir, _ = scene_dir
    out_base = str(tmp_path / "result")
    summary = run(
        _test_graph(scene_dir),
        input_path=str(data_dir),
        out_base=out_base,
        work_dir=str(tmp_path / "work"),
    )

    assert all(not entry["cached"] for entry in summary.values())

    with h5py.File(out_base + ".h5", "r") as fh:
        assert set(fh["raw"]) == {"A", "B", "C"}
        assert fh["clusters"].attrs["n_clusters"] == 2
        assert fh["clusters"].attrs["clustering_strategy"] == "global"
        assert "pipeline_config" in fh.attrs
    figures = list((tmp_path / "result" / "figures").glob("*.png"))
    assert len(figures) >= 6

    # Warm rerun: every producer cached, sinks re-run
    summary2 = run(
        _test_graph(scene_dir),
        input_path=str(data_dir),
        out_base=out_base,
        work_dir=str(tmp_path / "work"),
    )
    producers = {"src", "msk", "dn", "nrm", "pca", "hdb", "knn", "stats", "fp"}
    for node_id in producers:
        assert summary2[node_id]["cached"] is True, node_id
    assert summary2["exp"]["cached"] is False


def test_no_qc_skips_figure_sinks(tmp_path, scene_dir):
    data_dir, _ = scene_dir
    summary = run(
        _test_graph(scene_dir),
        input_path=str(data_dir),
        out_base=str(tmp_path / "result"),
        work_dir=str(tmp_path / "work"),
        skip_types={
            "qc_mask", "qc_denoise", "qc_normalize", "qc_scree",
            "qc_phase_map", "qc_cluster_summary", "qc_fingerprints",
            "qc_tiled", "qc_named_phase_map",
        },
    )
    assert summary["qc_mask"].get("skipped") is True
    assert not (tmp_path / "result" / "figures").exists()
