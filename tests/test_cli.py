"""Tests for the karak CLI: flow subcommands + legacy config path."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
import yaml

from conftest import make_synthetic_scene
from karak.cli.main import main


def test_schema_subcommand(capsys):
    assert main(["schema"]) == 0
    palette = json.loads(capsys.readouterr().out)
    assert any(entry["id"] == "load_elements" for entry in palette)


def test_validate_subcommand(capsys):
    assert main(["validate", "--builtin", "global"]) == 0


def test_entry_point_reexport():
    from karak.cli import runner

    assert runner.main is main


@pytest.fixture(scope="module")
def legacy_setup(tmp_path_factory):
    """Scene PNGs + a legacy YAML config pointing at them."""
    import imageio.v2 as imageio
    import matplotlib

    import karak.io.loaders as loaders

    tmp_path = tmp_path_factory.mktemp("legacy")
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

    config = {
        "input_dir": str(data_dir),
        "hdf5_output": str(tmp_path / "out.h5"),
        "figure_dir": str(tmp_path / "figures"),
        "exclude_elements": [],
        "loader": {"colormap": f"lut:{lut_path}"},
        "downsample": {"header_trim_px": 0, "downsample_factor": 1},
        "mask": {"min_object_size": 10},
        "cluster": {
            "pca": {"n_components": 3, "random_state": 0},
            "hdbscan": {"min_cluster_size": 100, "random_state": 0},
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config))

    yield tmp_path, config_path

    loaders._CACHE_DIR, loaders._FULL_LUT_CACHE = orig_dir, orig_mem


def test_legacy_config_runs_through_flow(legacy_setup, capsys):
    tmp_path, config_path = legacy_setup
    assert main(["-c", str(config_path)]) == 0

    with h5py.File(tmp_path / "out.h5", "r") as fh:
        assert set(fh["raw"]) == {"A", "B", "C"}
        assert fh["clusters"].attrs["n_clusters"] == 2
    assert list((tmp_path / "figures").glob("*.png"))


def test_legacy_emit_flow(legacy_setup, capsys):
    _, config_path = legacy_setup
    assert main(["-c", str(config_path), "--emit-flow"]) == 0
    flow = json.loads(capsys.readouterr().out)
    assert {n["type"] for n in flow["nodes"]} >= {"load_elements", "export_h5"}


def test_from_stage_warns_deprecated(legacy_setup, capsys):
    _, config_path = legacy_setup
    assert main(["-c", str(config_path), "--from-stage", "denoise"]) == 0
    assert "deprecated" in capsys.readouterr().err.lower()
