"""Core argument bundles carry no defaults: stages fill every field."""

from __future__ import annotations

import dataclasses

import pytest

from karak import core_params

BUNDLES = [
    core_params.DownsampleConfig, core_params.LoaderConfig,
    core_params.DenoiseConfig, core_params.PCAConfig,
    core_params.HDBSCANConfig, core_params.TiledConfig,
    core_params.RarePhaseConfig, core_params.OlivineExtractionConfig,
    core_params.GMMSplitConfig, core_params.RefinementConfig,
]


@pytest.mark.parametrize("bundle", BUNDLES, ids=lambda b: b.__name__)
def test_bundles_have_no_default_values(bundle):
    for f in dataclasses.fields(bundle):
        assert f.default is dataclasses.MISSING, f"{bundle.__name__}.{f.name}"
        assert f.default_factory is dataclasses.MISSING, f"{bundle.__name__}.{f.name}"
    with pytest.raises(TypeError):
        bundle()


def test_stage_builders_fill_every_field_from_the_stage_params():
    from karak.stages.cluster import HdbscanTiledStage, hdbscan_config, tiled_config
    from karak.stages.denoise import DenoiseStage, denoise_config
    from karak.stages.load import LoadElementsStage, downsample_config, loader_config
    from karak.stages.pca import PCAStage, pca_config
    from karak.stages.rare_phase import RarePhaseStage, rare_phase_config

    load = LoadElementsStage.template()
    assert downsample_config(load).downsample_factor == load["downsample_factor"]
    assert loader_config(load).colormap == load["colormap"]
    assert denoise_config(DenoiseStage.template()).method == "bilateral"
    assert pca_config(PCAStage.template()).random_state == 42
    tiled = HdbscanTiledStage.template()
    assert hdbscan_config(tiled).min_samples is None       # 0 = automatic
    assert tiled_config(tiled).merge_threshold == tiled["merge_threshold"]
    rare = RarePhaseStage.template()
    assert rare_phase_config(rare).merge_threshold == rare["merge_threshold"]


def test_rare_phase_merge_threshold_is_explicit():
    from karak.stages.rare_phase import RarePhaseStage

    # 0 used to mean "reuse the tiled threshold" but silently took 0.92
    # from code; now the flow states the value and 0 is out of range.
    assert RarePhaseStage.template()["merge_threshold"] == 0.92
    with pytest.raises(ValueError, match="merge_threshold"):
        RarePhaseStage.coerce_params({"merge_threshold": 0.0})
