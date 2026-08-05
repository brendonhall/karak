"""Config round-trip tests: save/load YAML preserves values."""

from __future__ import annotations

from pathlib import Path

from karak.config import (
    DenoiseConfig,
    DownsampleConfig,
    HDBSCANConfig,
    LoaderConfig,
    MaskConfig,
    PCAConfig,
    PipelineConfig,
    load_config,
    save_config,
)


def _make_config(tmp_path: Path) -> PipelineConfig:
    """Build a config with non-default values and absolute paths."""
    return PipelineConfig(
        input_dir=str(tmp_path / "maps"),
        hdf5_output=str(tmp_path / "out.h5"),
        figure_dir=str(tmp_path / "figs"),
        exclude_elements=["Fe-L", "Ni"],
        bse_channel="BSE",
        loader=LoaderConfig(
            file_glob="*.bmp",
            filename_pattern="Map_{sample}_eds_{element}.bmp",
            colormap="cmap:viridis",
        ),
        downsample=DownsampleConfig(
            header_trim_px=50,
            bottom_trim_px=10,
            downsample_factor=4,
        ),
        mask=MaskConfig(
            min_object_size=64,
            valid_mask_path=str(tmp_path / "valid.csv"),
        ),
        denoise=DenoiseConfig(
            method="anisotropic_diffusion",
            niter=7,
            kappa=25.0,
        ),
    )


def test_roundtrip_preserves_values(tmp_path):
    cfg = _make_config(tmp_path)
    cfg.cluster.strategy = "tiled"
    cfg.cluster.pca = PCAConfig(n_components=7, random_state=7)
    cfg.cluster.hdbscan = HDBSCANConfig(
        min_cluster_size=777, subsample_n=1234, noise_reassign_k=3
    )
    cfg.cluster.tiled.tile_size = 256
    cfg.cluster.rare_phase.enabled = True

    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    loaded = load_config(path)

    # All paths in the original are absolute, so load_config's path
    # resolution is the identity and full equality must hold.
    assert loaded.model_dump() == cfg.model_dump()


def test_defaults_roundtrip(tmp_path):
    cfg = PipelineConfig(
        input_dir=str(tmp_path),
        hdf5_output=str(tmp_path / "p.h5"),
        figure_dir=str(tmp_path / "figs"),
    )
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded.model_dump() == cfg.model_dump()


def test_relative_paths_resolved_against_config_dir(tmp_path):
    """Relative paths in the YAML resolve relative to the config file."""
    cfg_dir = tmp_path / "proj"
    cfg_dir.mkdir()
    path = cfg_dir / "config.yaml"
    path.write_text(
        "input_dir: maps\n"
        "hdf5_output: out.h5\n"
        "figure_dir: figs\n"
        "mask:\n"
        "  valid_mask_path: mask/valid.csv\n"
    )
    loaded = load_config(path)
    assert loaded.input_dir == str((cfg_dir / "maps").resolve())
    assert loaded.hdf5_output == str((cfg_dir / "out.h5").resolve())
    assert loaded.mask.valid_mask_path == str(
        (cfg_dir / "mask" / "valid.csv").resolve()
    )
