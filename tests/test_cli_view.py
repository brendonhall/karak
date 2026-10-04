"""`karak view`: find cached outputs and turn them into napari layers."""

from __future__ import annotations

import os

import numpy as np
import pytest

import karak.cli.view as view
from karak.cli.view import find_cache_files, layer_specs, pick_outputs, view_main
from karak.flow.cache import store_payload
from karak.stages.payloads import (
    BseImage,
    ClusterStats,
    ElementCube,
    Labels,
    LabelState,
    MaskSet,
    PCAFeatures,
    Space,
)


def _cube(names=("Al", "Fe-K", "Si"), factor=2, trim=0, space=Space.RAW):
    return ElementCube(
        pixels=np.zeros((4, 5, len(names)), np.float32),
        element_names=tuple(names),
        space=space,
        downsample_factor=factor,
        header_trim_px=trim,
    )


def _bse():
    return BseImage(pixels=np.zeros((4, 5), np.float32))


def _masks(valid=True):
    mineral = np.zeros((4, 5), bool)
    mineral[1:3, 1:4] = True
    return MaskSet(mineral_mask=mineral,
                   valid_mask=np.ones((4, 5), bool) if valid else None)


def _store(cache, recipe, port, payload, mtime, upstream=None):
    path = store_payload(recipe, port, payload, cache, upstream=upstream)
    os.utime(path, (mtime, mtime))
    return path


def test_find_from_out_base(tmp_path):
    cache = tmp_path / "output" / "work" / "cache"
    _store(cache, "aaa", "cube", _cube(), 1000)
    files = find_cache_files(tmp_path / "output" / "run")
    assert [(f.recipe, f.port, f.payload_type) for f in files] == [
        ("aaa", "cube", "element_cube")
    ]


def test_find_from_work_dir_cache_dir_and_single_file(tmp_path):
    cache = tmp_path / "work" / "cache"
    path = _store(cache, "aaa", "cube", _cube(), 1000)
    for target in (tmp_path / "work", cache, path):
        assert len(find_cache_files(target)) == 1


def test_find_raises_when_nothing_is_there(tmp_path):
    with pytest.raises(FileNotFoundError, match="no cached outputs"):
        find_cache_files(tmp_path / "nowhere" / "run")


def test_pick_newest_cube_and_the_bse_from_the_same_step(tmp_path):
    cache = tmp_path / "work" / "cache"
    _store(cache, "old", "cube", _cube(), 1000)
    _store(cache, "old", "bse", _bse(), 1000)
    _store(cache, "new", "cube", _cube(), 2000)
    _store(cache, "new", "bse", _bse(), 1500)
    _store(cache, "zzz", "bse", _bse(), 3000)
    _store(cache, "st", "stats", ClusterStats(stats={}), 4000)
    picked = pick_outputs(find_cache_files(cache))
    assert picked.cube.recipe == "new"
    assert picked.bse.recipe == "new"
    assert picked.masks is None
    assert [f.recipe for f in picked.other_cubes] == ["old"]


def test_pick_newest_masks_of_the_selected_cube_from_a_cache_scan(tmp_path):
    cache = tmp_path / "work" / "cache"
    _store(cache, "c", "cube", _cube(), 1000)
    _store(cache, "m-old", "masks", _masks(), 1500, upstream={"cube": "c"})
    _store(cache, "m-new", "masks", _masks(), 2500, upstream={"cube": "c"})
    picked = pick_outputs(find_cache_files(cache))
    assert picked.masks.recipe == "m-new"


def test_pick_ignores_masks_of_another_cube_in_a_cache_scan(tmp_path):
    # Sample A ran at factor 1, then B at factor 2, then A again with new
    # mask parameters: A's load is a cache hit (old mtime) while A's new
    # masks are the newest file in the cache.
    cache = tmp_path / "work" / "cache"
    _store(cache, "a", "cube", _cube(factor=1), 1000)
    _store(cache, "a-m1", "masks", _masks(), 1000, upstream={"cube": "a"})
    _store(cache, "b", "cube", _cube(factor=2), 2000)
    _store(cache, "b-m", "masks", _masks(), 2000, upstream={"cube": "b"})
    _store(cache, "a-m2", "masks", _masks(), 3000, upstream={"cube": "a"})
    picked = pick_outputs(find_cache_files(cache))
    assert picked.cube.recipe == "b"
    assert picked.masks.recipe == "b-m"


def test_pick_no_masks_when_none_link_to_the_selected_cube(tmp_path):
    cache = tmp_path / "work" / "cache"
    _store(cache, "b", "cube", _cube(), 2000)
    _store(cache, "a-m", "masks", _masks(), 3000, upstream={"cube": "a"})
    _store(cache, "old", "masks", _masks(), 4000)   # written before upstream links
    assert pick_outputs(find_cache_files(cache)).masks is None


def test_pick_without_a_cube_raises(tmp_path):
    cache = tmp_path / "work" / "cache"
    _store(cache, "b", "bse", _bse(), 1000)
    with pytest.raises(ValueError, match="no ElementCube"):
        pick_outputs(find_cache_files(cache))


def test_layer_specs_names_visibility_and_placement():
    specs = layer_specs(_cube(factor=2, trim=100), _bse(), show=("Fe-K",))
    assert [s.name for s in specs] == ["BSE", "Al", "Fe-K", "Si"]
    assert [s.visible for s in specs] == [True, False, True, False]
    for spec in specs:
        # a downsampled pixel covers a 2x2 block; its center sits at +0.5
        assert spec.scale == (2, 2)
        assert spec.translate == (100.5, 0.5)
        assert spec.contrast_limits == (0.0, 1.0)
    assert specs[2].data.shape == (4, 5)


def test_layer_specs_show_first_element_when_requested_one_is_missing():
    specs = layer_specs(_cube(), None, show=("Zr",))
    assert [s.name for s in specs] == ["Al", "Fe-K", "Si"]
    assert [s.visible for s in specs] == [True, False, False]


def test_view_without_napari_prints_install_hint(tmp_path, monkeypatch, capsys):
    _store(tmp_path / "work" / "cache", "a", "cube", _cube(), 1000)
    monkeypatch.setattr(view, "_napari_available", lambda: False)
    assert view_main([str(tmp_path / "work")]) == 1
    assert "uv sync --extra view" in capsys.readouterr().err


def test_view_opens_layers_and_mask(tmp_path, monkeypatch, capsys):
    cache = tmp_path / "work" / "cache"
    _store(cache, "a", "cube", _cube(), 1000)
    _store(cache, "a", "bse", _bse(), 1000)
    mask = tmp_path / "Valid_mask.csv"
    mask.write_text(
        "index,shape-type,vertex-index,axis-0,axis-1\n"
        "0,polygon,0,0.0,0.0\n"
        "0,polygon,1,0.0,8.0\n"
        "0,polygon,2,6.0,8.0\n"
        "1,path,0,1.0,1.0\n"
        "1,path,1,2.0,2.0\n"
    )
    opened = {}
    monkeypatch.setattr(view, "_napari_available", lambda: True)
    monkeypatch.setattr(view, "open_viewer",
                        lambda specs, shapes: opened.update(specs=specs, shapes=shapes))
    rc = view_main([str(tmp_path / "work"), "--show", "Al,Si", "--mask", str(mask)])
    assert rc == 0
    visible = [s.name for s in opened["specs"] if s.visible]
    assert visible == ["BSE", "Al", "Si"]
    assert [kind for kind, _ in opened["shapes"]] == ["polygon", "path"]
    np.testing.assert_array_equal(
        opened["shapes"][0][1], [[0.0, 0.0], [0.0, 8.0], [6.0, 8.0]]
    )
    out = capsys.readouterr().out
    assert "ElementCube 4×5×3" in out


def _write_record(out_base, outputs, masks=None, denoised=None,
                  normalized=None, features=None, labels=None, cleaned=None):
    """A minimal run record whose src node lists the given cache files,
    plus msk, dn, nrm, pca, hdb and knn nodes for the files given."""
    import json

    def node(stage, files):
        return {"type": stage, "status": "ran",
                "outputs": {port: {"file": str(path), "summary": ""}
                            for port, path in files.items()}}

    nodes = {"src": node("load_elements", outputs)}
    if masks is not None:
        nodes["msk"] = node("mask", {"masks": masks})
    if denoised is not None:
        nodes["dn"] = node("denoise", {"cube": denoised})
    if normalized is not None:
        nodes["nrm"] = node("normalize", {"cube": normalized})
    if features is not None:
        nodes["pca"] = node("pca", {"features": features})
    if labels is not None:
        nodes["hdb"] = node("hdbscan_global", {"labels": labels})
    if cleaned is not None:
        nodes["knn"] = node("noise_assign", {"labels": cleaned})
    run_dir = out_base / "runs" / "2026-09-29T14-05-12Z"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({
        "status": "ok", "nodes": nodes,
    }))
    (out_base / "runs" / "latest").symlink_to(run_dir.name)
    return run_dir


def test_view_uses_the_latest_run_record_over_newer_cache_files(tmp_path, capsys):
    from karak.cli.view import find_run_outputs

    cache = tmp_path / "output" / "work" / "cache"
    rec_cube = _store(cache, "rec", "cube", _cube(), 1000)
    rec_bse = _store(cache, "rec", "bse", _bse(), 1000)
    _store(cache, "newer", "cube", _cube(), 5000)   # not from the recorded run
    out_base = tmp_path / "output" / "run"
    _write_record(out_base, {"cube": rec_cube, "bse": rec_bse})

    picked = pick_outputs(find_run_outputs(out_base), newest_first=False)
    assert (picked.cube.recipe, picked.bse.recipe) == ("rec", "rec")
    assert picked.other_cubes == []


def test_view_picks_the_masks_of_the_recorded_run(tmp_path):
    from karak.cli.view import find_run_outputs

    cache = tmp_path / "output" / "work" / "cache"
    rec_cube = _store(cache, "rec", "cube", _cube(), 1000)
    rec_masks = _store(cache, "recm", "masks", _masks(), 1000)
    _store(cache, "newer", "masks", _masks(), 5000)   # not from the recorded run
    out_base = tmp_path / "output" / "run"
    _write_record(out_base, {"cube": rec_cube}, masks=rec_masks)
    picked = pick_outputs(find_run_outputs(out_base), newest_first=False)
    assert picked.masks.recipe == "recm"


def test_view_falls_back_to_the_cache_when_record_files_are_gone(tmp_path):
    from karak.cli.view import find_run_outputs

    cache = tmp_path / "output" / "work" / "cache"
    gone = cache / "gone__cube.h5"
    _store(cache, "aaa", "cube", _cube(), 1000)
    out_base = tmp_path / "output" / "run"
    _write_record(out_base, {"cube": gone})
    assert find_run_outputs(out_base) is None


def test_view_without_a_record_returns_none(tmp_path):
    from karak.cli.view import find_run_outputs

    assert find_run_outputs(tmp_path / "output" / "run") is None


def test_view_main_reports_the_record_it_opened(tmp_path, monkeypatch, capsys):
    cache = tmp_path / "output" / "work" / "cache"
    rec_cube = _store(cache, "rec", "cube", _cube(), 1000)
    out_base = tmp_path / "output" / "run"
    run_dir = _write_record(out_base, {"cube": rec_cube})
    monkeypatch.setattr(view, "_napari_available", lambda: True)
    monkeypatch.setattr(view, "open_viewer", lambda specs, shapes: None)
    assert view_main([str(out_base)]) == 0
    assert f"run record: {run_dir}" in capsys.readouterr().out


def test_layer_specs_add_the_masks_as_labels_layers():
    cube = _cube(factor=2, trim=3)
    specs = layer_specs(cube, None, _masks())
    by_name = {s.name: s for s in specs}
    mineral, valid = by_name["mineral mask"], by_name["valid mask"]
    assert (mineral.kind, mineral.visible) == ("labels", True)
    assert (valid.kind, valid.visible) == ("labels", False)
    assert mineral.data.dtype == np.uint8
    assert (mineral.scale, mineral.translate) == (by_name["Al"].scale,
                                                  by_name["Al"].translate)
    assert by_name["Al"].kind == "image"
    assert [s.name for s in specs][-2:] == ["mineral mask", "valid mask"]


def test_layer_specs_skip_the_valid_mask_when_there_is_none():
    names = [s.name for s in layer_specs(_cube(), None, _masks(valid=False))]
    assert "mineral mask" in names
    assert "valid mask" not in names


def test_layer_specs_without_masks_add_no_labels_layers():
    assert all(s.kind == "image" for s in layer_specs(_cube(), _bse()))


def test_view_main_opens_the_masks_of_a_run(tmp_path, monkeypatch, capsys):
    cache = tmp_path / "work" / "cache"
    _store(cache, "a", "cube", _cube(), 1000)
    _store(cache, "m", "masks", _masks(), 1000, upstream={"cube": "a"})
    opened = {}
    monkeypatch.setattr(view, "_napari_available", lambda: True)
    monkeypatch.setattr(view, "open_viewer",
                        lambda specs, shapes: opened.update(specs=specs))
    assert view_main([str(tmp_path / "work")]) == 0
    labels = [s.name for s in opened["specs"] if s.kind == "labels"]
    assert labels == ["mineral mask", "valid mask"]
    assert "MaskSet mineral 30.0% · valid 100.0%" in capsys.readouterr().out


def test_open_viewer_adds_labels_layers_with_add_labels(monkeypatch):
    import sys
    import types

    calls = []

    class Viewer:
        def __init__(self, title):
            pass

        def add_image(self, data, **kw):
            calls.append(("image", kw["name"]))

        def add_labels(self, data, **kw):
            calls.append(("labels", kw["name"], kw["visible"]))

    fake = types.SimpleNamespace(Viewer=Viewer, run=lambda: None)
    monkeypatch.setitem(sys.modules, "napari", fake)
    specs = layer_specs(_cube(), None, _masks(valid=False))
    view.open_viewer(specs, [])
    assert ("labels", "mineral mask", True) in calls
    assert ("image", "Al") in calls


def _denoised():
    return _cube(space=Space.DENOISED)


def test_pick_denoised_cube_linked_to_the_raw_cube_from_a_cache_scan(tmp_path):
    cache = tmp_path / "work" / "cache"
    _store(cache, "raw", "cube", _cube(), 1000)
    _store(cache, "dn-old", "cube", _denoised(), 1500, upstream={"cube": "raw", "masks": "m"})
    _store(cache, "dn-new", "cube", _denoised(), 2500, upstream={"cube": "raw", "masks": "m"})
    picked = pick_outputs(find_cache_files(cache))
    assert picked.cube.recipe == "raw"
    assert picked.denoised.recipe == "dn-new"
    assert [e.recipe for e in picked.other_cubes] == ["dn-old"]


def test_pick_no_denoised_cube_when_none_links_to_the_raw_cube(tmp_path):
    cache = tmp_path / "work" / "cache"
    _store(cache, "raw", "cube", _cube(), 3000)
    _store(cache, "dn-other", "cube", _denoised(), 4000, upstream={"cube": "other"})
    picked = pick_outputs(find_cache_files(cache))
    assert picked.cube.recipe == "raw"
    assert picked.denoised is None
    assert [e.recipe for e in picked.other_cubes] == ["dn-other"]


def test_pick_denoised_cube_of_the_recorded_run(tmp_path):
    from karak.cli.view import find_run_outputs

    cache = tmp_path / "output" / "work" / "cache"
    raw = _store(cache, "raw", "cube", _cube(), 1000)
    dn = _store(cache, "dn", "cube", _denoised(), 1000, upstream={"cube": "raw"})
    out_base = tmp_path / "output" / "run"
    _write_record(out_base, {"cube": raw}, denoised=dn)
    picked = pick_outputs(find_run_outputs(out_base), newest_first=False)
    assert (picked.cube.recipe, picked.denoised.recipe) == ("raw", "dn")


def test_layer_specs_add_prefixed_denoised_layers_after_the_raw_ones():
    specs = layer_specs(_cube(), _bse(), _masks(), _denoised(), show=("Si",))
    names = [s.name for s in specs]
    assert names == ["BSE", "Al", "Fe-K", "Si", "dn: Al", "dn: Fe-K", "dn: Si",
                     "mineral mask", "valid mask"]
    visible = [s.name for s in specs if s.visible]
    assert visible == ["BSE", "Si", "dn: Si", "mineral mask"]
    by_name = {s.name: s for s in specs}
    assert by_name["dn: Si"].kind == "image"
    assert (by_name["dn: Si"].scale, by_name["dn: Si"].translate) == (
        by_name["Si"].scale, by_name["Si"].translate)


def test_view_main_opens_the_denoised_cube(tmp_path, monkeypatch, capsys):
    cache = tmp_path / "work" / "cache"
    _store(cache, "raw", "cube", _cube(), 1000)
    _store(cache, "dn", "cube", _denoised(), 2000, upstream={"cube": "raw"})
    opened = {}
    monkeypatch.setattr(view, "_napari_available", lambda: True)
    monkeypatch.setattr(view, "open_viewer",
                        lambda specs, shapes: opened.update(specs=specs))
    assert view_main([str(tmp_path / "work")]) == 0
    assert "dn: Fe-K" in [s.name for s in opened["specs"]]
    out = capsys.readouterr().out
    assert "space=denoised" in out
    assert "other cube" not in out


# --- review fixes: legacy cache files and coherent mask/denoise pairs ------

def test_legacy_cache_scan_picks_the_raw_cube_by_space(tmp_path):
    """Files written before the upstream attribute: the newer denoised cube
    must not be mistaken for the load step's cube."""
    cache = tmp_path / "work" / "cache"
    _store(cache, "raw", "cube", _cube(), 1000)
    _store(cache, "dn", "cube", _denoised(), 2000)            # no upstream
    picked = pick_outputs(find_cache_files(cache))
    assert picked.cube.recipe == "raw"
    assert picked.denoised is None                          # no provable link
    assert [e.recipe for e in picked.other_cubes] == ["dn"]


def test_legacy_run_record_keeps_the_denoised_cube(tmp_path):
    from karak.cli.view import find_run_outputs

    cache = tmp_path / "output" / "work" / "cache"
    raw = _store(cache, "raw", "cube", _cube(), 1000)
    dn = _store(cache, "dn", "cube", _denoised(), 1000)       # no upstream
    out_base = tmp_path / "output" / "run"
    _write_record(out_base, {"cube": raw}, denoised=dn)
    picked = pick_outputs(find_run_outputs(out_base), newest_first=False)
    assert (picked.cube.recipe, picked.denoised.recipe) == ("raw", "dn")


def test_scan_pairs_the_denoised_cube_with_the_mask_it_consumed(tmp_path):
    """Mask M1 -> denoise D1, then a new mask M2 with the rerun interrupted
    before denoise: the viewer must not overlay M2 on D1."""
    cache = tmp_path / "work" / "cache"
    _store(cache, "raw", "cube", _cube(), 1000)
    _store(cache, "m1", "masks", _masks(), 1500, upstream={"cube": "raw"})
    _store(cache, "d1", "cube", _denoised(), 2000, upstream={"cube": "raw", "masks": "m1"})
    _store(cache, "m2", "masks", _masks(), 3000, upstream={"cube": "raw"})
    picked = pick_outputs(find_cache_files(cache))
    assert (picked.denoised.recipe, picked.masks.recipe) == ("d1", "m1")


def test_scan_without_a_denoised_cube_shows_the_newest_mask(tmp_path):
    cache = tmp_path / "work" / "cache"
    _store(cache, "raw", "cube", _cube(), 1000)
    _store(cache, "m1", "masks", _masks(), 1500, upstream={"cube": "raw"})
    _store(cache, "m2", "masks", _masks(), 3000, upstream={"cube": "raw"})
    picked = pick_outputs(find_cache_files(cache))
    assert picked.denoised is None and picked.masks.recipe == "m2"


def test_scan_omits_the_mask_when_the_denoised_cube_names_a_missing_one(tmp_path):
    cache = tmp_path / "work" / "cache"
    _store(cache, "raw", "cube", _cube(), 1000)
    _store(cache, "m2", "masks", _masks(), 3000, upstream={"cube": "raw"})
    _store(cache, "d1", "cube", _denoised(), 2000, upstream={"cube": "raw", "masks": "gone"})
    picked = pick_outputs(find_cache_files(cache))
    assert picked.denoised.recipe == "d1"
    assert picked.masks is None
def _normalized(fill=0.0):
    cube = _cube(space=Space.NORMALIZED)
    rng = np.random.default_rng(3)
    pixels = rng.normal(0.0, 1.0, cube.pixels.shape).astype(np.float32)
    pixels[0, 0, :] = 50.0          # one outlier per channel
    return cube.replace(pixels=pixels)


def test_pick_normalized_cube_linked_to_the_denoised_cube_from_a_cache_scan(tmp_path):
    cache = tmp_path / "work" / "cache"
    _store(cache, "raw", "cube", _cube(), 1000)
    _store(cache, "dn", "cube", _denoised(), 2000, upstream={"cube": "raw"})
    _store(cache, "nrm", "cube", _normalized(), 3000, upstream={"cube": "dn", "masks": "m"})
    picked = pick_outputs(find_cache_files(cache))
    assert (picked.cube.recipe, picked.denoised.recipe) == ("raw", "dn")
    assert picked.normalized.recipe == "nrm"
    assert picked.other_cubes == []


def test_pick_no_normalized_cube_without_a_denoised_one(tmp_path):
    cache = tmp_path / "work" / "cache"
    _store(cache, "raw", "cube", _cube(), 1000)
    _store(cache, "nrm", "cube", _normalized(), 3000, upstream={"cube": "other"})
    picked = pick_outputs(find_cache_files(cache))
    assert picked.denoised is None and picked.normalized is None


def test_normalized_layers_use_contrast_limits_from_the_data():
    specs = layer_specs(_cube(), None, None, _denoised(), _normalized(), show=("Si",))
    names = [s.name for s in specs]
    assert names[-3:] == ["nrm: Al", "nrm: Fe-K", "nrm: Si"]
    by_name = {s.name: s for s in specs}
    lo, hi = by_name["nrm: Si"].contrast_limits
    assert lo < 0 < hi
    assert hi < 50.0                      # the outlier does not set the range
    assert by_name["nrm: Si"].visible and not by_name["nrm: Al"].visible
    assert by_name["Si"].contrast_limits == (0.0, 1.0)


def test_normalized_contrast_limits_ignore_the_zeroed_background():
    cube = _normalized()
    pixels = cube.pixels.copy()
    pixels[:2] = 0.0                       # half the image is non-mineral
    specs = layer_specs(_cube(), None, None, None, cube.replace(pixels=pixels))
    lo, hi = {s.name: s for s in specs}["nrm: Al"].contrast_limits
    assert lo < -0.1 and hi > 0.1


def _features(n_kept=2, image_shape=(4, 5)):
    rows, cols = np.nonzero(_masks().mineral_mask)
    scores = np.stack([np.arange(rows.size, dtype=np.float32) + 1.0,
                       -np.arange(rows.size, dtype=np.float32) - 1.0,
                       np.full(rows.size, 0.5, np.float32)], axis=1)
    scores[0, 0] = 100.0                  # one outlier on PC1
    return PCAFeatures(
        features=scores[:, :n_kept],
        mineral_indices=np.stack([rows, cols], axis=1).astype(np.int32),
        image_shape=image_shape,
        explained_variance_ratio=np.array([0.6, 0.3, 0.1]),
        n_kept=n_kept,
    )


def test_pick_pca_features_linked_to_the_normalized_cube_from_a_cache_scan(tmp_path):
    cache = tmp_path / "work" / "cache"
    _store(cache, "raw", "cube", _cube(), 1000)
    _store(cache, "dn", "cube", _denoised(), 2000, upstream={"cube": "raw"})
    _store(cache, "nrm", "cube", _normalized(), 3000, upstream={"cube": "dn"})
    _store(cache, "pca", "features", _features(), 4000,
           upstream={"cube": "nrm", "masks": "m"})
    _store(cache, "other", "features", _features(), 5000,
           upstream={"cube": "another-nrm", "masks": "m"})
    picked = pick_outputs(find_cache_files(cache))
    assert picked.normalized.recipe == "nrm"
    assert picked.features.recipe == "pca"


def test_pick_no_pca_features_without_a_normalized_cube(tmp_path):
    cache = tmp_path / "work" / "cache"
    _store(cache, "raw", "cube", _cube(), 1000)
    _store(cache, "dn", "cube", _denoised(), 2000, upstream={"cube": "raw"})
    _store(cache, "pca", "features", _features(), 4000, upstream={"cube": "nrm"})
    picked = pick_outputs(find_cache_files(cache))
    assert picked.normalized is None and picked.features is None


def test_pick_pca_features_of_the_recorded_run(tmp_path):
    from karak.cli.view import find_run_outputs

    cache = tmp_path / "output" / "work" / "cache"
    rec_cube = _store(cache, "rec", "cube", _cube(), 1000)
    rec_dn = _store(cache, "recd", "cube", _denoised(), 1000)
    rec_nrm = _store(cache, "recn", "cube", _normalized(), 1000)
    rec_pca = _store(cache, "recp", "features", _features(), 1000)
    _store(cache, "newer", "features", _features(), 5000)   # not from the run
    out_base = tmp_path / "output" / "run"
    _write_record(out_base, {"cube": rec_cube}, denoised=rec_dn,
                  normalized=rec_nrm, features=rec_pca)
    picked = pick_outputs(find_run_outputs(out_base), newest_first=False)
    assert picked.features.recipe == "recp"


def test_pca_layers_scatter_the_kept_components_into_the_image():
    cube = _cube(factor=2, trim=3)
    features = _features(n_kept=2)
    specs = layer_specs(cube, None, _masks(), None, _normalized(), features)
    by_name = {s.name: s for s in specs}
    names = [s.name for s in specs]
    assert names[names.index("pca: PC1"):] == [
        "pca: PC1", "pca: PC2", "mineral mask", "valid mask"]
    pc2 = by_name["pca: PC2"]
    rows, cols = features.mineral_indices.T
    np.testing.assert_array_equal(pc2.data[rows, cols], features.features[:, 1])
    outside = ~_masks().mineral_mask
    assert (pc2.data[outside] == 0).all()
    assert pc2.data.shape == (4, 5) and pc2.data.dtype == np.float32
    assert not pc2.visible
    assert (pc2.scale, pc2.translate) == (by_name["Al"].scale, by_name["Al"].translate)
    lo, hi = by_name["pca: PC1"].contrast_limits
    assert 1.0 <= lo < hi < 100.0         # the outlier does not set the range


def test_pca_layers_skipped_when_the_image_shape_differs():
    specs = layer_specs(_cube(), None, None, None, None,
                        _features(image_shape=(8, 10)))
    assert not any(s.name.startswith("pca:") for s in specs)


def test_pca_summary_lists_the_kept_variance():
    from karak.cli.view import pca_summary

    assert pca_summary(_features(n_kept=2)) == (
        "pca: 2 components kept (90.0% variance): PC1 60.0%, PC2 30.0%")


def test_view_main_prints_the_pca_summary(tmp_path, monkeypatch, capsys):
    cache = tmp_path / "work" / "cache"
    _store(cache, "raw", "cube", _cube(), 1000)
    _store(cache, "dn", "cube", _denoised(), 2000, upstream={"cube": "raw"})
    _store(cache, "nrm", "cube", _normalized(), 3000, upstream={"cube": "dn"})
    _store(cache, "pca", "features", _features(), 4000, upstream={"cube": "nrm"})
    opened = {}
    monkeypatch.setattr(view, "_napari_available", lambda: True)
    monkeypatch.setattr(view, "open_viewer",
                        lambda specs, shapes: opened.update(specs=specs))
    assert view_main([str(cache)]) == 0
    assert "pca: 2 components kept (90.0% variance)" in capsys.readouterr().out
    assert [s.name for s in opened["specs"]][-2:] == ["pca: PC1", "pca: PC2"]


def _labels(state=LabelState.RAW, image_shape=(4, 5)):
    """Raw HDBSCAN labels on the six mineral pixels of _masks(): phases 0
    and 1, with two noise pixels."""
    rows, cols = np.nonzero(_masks().mineral_mask)
    return Labels(
        labels=np.array([0, 0, -1, 1, 1, -1], np.int32),
        probabilities=np.array([0.9, 0.8, 0.0, 0.7, 0.6, 0.0], np.float32),
        mineral_indices=np.stack([rows, cols], axis=1).astype(np.int32),
        image_shape=image_shape,
        state=state,
    )


def _store_chain(cache):
    _store(cache, "raw", "cube", _cube(), 1000)
    _store(cache, "dn", "cube", _denoised(), 2000, upstream={"cube": "raw"})
    _store(cache, "nrm", "cube", _normalized(), 3000, upstream={"cube": "dn"})
    _store(cache, "pca", "features", _features(), 4000, upstream={"cube": "nrm"})


def test_pick_raw_labels_linked_to_the_pca_features_from_a_cache_scan(tmp_path):
    cache = tmp_path / "work" / "cache"
    _store_chain(cache)
    _store(cache, "hdb", "labels", _labels(), 5000, upstream={"features": "pca"})
    _store(cache, "other", "labels", _labels(), 6000,
           upstream={"features": "another-pca"})
    _store(cache, "knn", "labels", _labels(LabelState.CLEANED), 7000,
           upstream={"features": "pca", "labels": "hdb"})
    picked = pick_outputs(find_cache_files(cache))
    assert picked.features.recipe == "pca"
    assert picked.labels.recipe == "hdb"


def test_pick_no_labels_without_pca_features(tmp_path):
    cache = tmp_path / "work" / "cache"
    _store(cache, "raw", "cube", _cube(), 1000)
    _store(cache, "dn", "cube", _denoised(), 2000, upstream={"cube": "raw"})
    _store(cache, "hdb", "labels", _labels(), 5000, upstream={"features": "pca"})
    picked = pick_outputs(find_cache_files(cache))
    assert picked.features is None and picked.labels is None


def test_pick_labels_of_the_recorded_run(tmp_path):
    from karak.cli.view import find_run_outputs

    cache = tmp_path / "output" / "work" / "cache"
    rec = {name: _store(cache, f"rec{name}", port, payload, 1000)
           for name, port, payload in [
               ("c", "cube", _cube()), ("d", "cube", _denoised()),
               ("n", "cube", _normalized()), ("p", "features", _features()),
               ("h", "labels", _labels())]}
    _store(cache, "newer", "labels", _labels(), 5000)   # not from the run
    out_base = tmp_path / "output" / "run"
    _write_record(out_base, {"cube": rec["c"]}, denoised=rec["d"],
                  normalized=rec["n"], features=rec["p"], labels=rec["h"])
    picked = pick_outputs(find_run_outputs(out_base), newest_first=False)
    assert picked.labels.recipe == "rech"


def test_hdb_layers_show_phases_noise_and_probability():
    cube = _cube(factor=2, trim=3)
    labels = _labels()
    specs = layer_specs(cube, None, _masks(), None, None, None, labels)
    by_name = {s.name: s for s in specs}
    names = [s.name for s in specs]
    assert names[names.index("hdb: phases"):] == [
        "hdb: phases", "hdb: noise", "hdb: probability",
        "mineral mask", "valid mask"]
    rows, cols = labels.mineral_indices.T
    phases = by_name["hdb: phases"]
    assert phases.kind == "labels" and phases.visible
    np.testing.assert_array_equal(phases.data[rows, cols], [1, 1, 0, 2, 2, 0])
    assert (phases.data[~_masks().mineral_mask] == 0).all()
    noise = by_name["hdb: noise"]
    assert noise.kind == "labels" and not noise.visible
    np.testing.assert_array_equal(noise.data[rows, cols], [0, 0, 1, 0, 0, 1])
    prob = by_name["hdb: probability"]
    assert prob.kind == "image" and not prob.visible
    assert prob.contrast_limits == (0.0, 1.0)
    np.testing.assert_allclose(prob.data[rows, cols], labels.probabilities)
    assert (phases.scale, phases.translate) == (by_name["Al"].scale,
                                                by_name["Al"].translate)


def test_hdb_layers_skipped_when_the_image_shape_differs():
    specs = layer_specs(_cube(), None, None, None, None, None,
                        _labels(image_shape=(8, 10)))
    assert not any(s.name.startswith("hdb:") for s in specs)


def test_hdb_summary_counts_phases_and_noise():
    from karak.cli.view import hdb_summary

    assert hdb_summary(_labels()) == (
        "hdb: 2 phases, 2 noise (33.3%); pixels per phase: 0 2, 1 2")


def test_view_main_prints_the_hdb_summary(tmp_path, monkeypatch, capsys):
    cache = tmp_path / "work" / "cache"
    _store_chain(cache)
    _store(cache, "hdb", "labels", _labels(), 5000, upstream={"features": "pca"})
    opened = {}
    monkeypatch.setattr(view, "_napari_available", lambda: True)
    monkeypatch.setattr(view, "open_viewer",
                        lambda specs, shapes: opened.update(specs=specs))
    assert view_main([str(cache)]) == 0
    assert "hdb: 2 phases, 2 noise (33.3%)" in capsys.readouterr().out
    assert [s.name for s in opened["specs"]][-3:] == [
        "hdb: phases", "hdb: noise", "hdb: probability"]


def _cleaned(image_shape=(4, 5)):
    """The noise pixels of _labels() reassigned: one to phase 0, one to 1."""
    raw = _labels(image_shape=image_shape)
    return raw.replace(labels=np.array([0, 0, 0, 1, 1, 1], np.int32),
                       state=LabelState.CLEANED)


def test_pick_cleaned_labels_linked_to_the_raw_labels_from_a_cache_scan(tmp_path):
    cache = tmp_path / "work" / "cache"
    _store_chain(cache)
    _store(cache, "hdb", "labels", _labels(), 5000, upstream={"features": "pca"})
    _store(cache, "knn", "labels", _cleaned(), 6000,
           upstream={"features": "pca", "labels": "hdb"})
    _store(cache, "other", "labels", _cleaned(), 7000,
           upstream={"features": "pca", "labels": "another-hdb"})
    picked = pick_outputs(find_cache_files(cache))
    assert picked.labels.recipe == "hdb"
    assert picked.cleaned.recipe == "knn"


def test_pick_no_cleaned_labels_without_raw_labels(tmp_path):
    cache = tmp_path / "work" / "cache"
    _store_chain(cache)
    _store(cache, "knn", "labels", _cleaned(), 6000,
           upstream={"features": "pca", "labels": "hdb"})
    picked = pick_outputs(find_cache_files(cache))
    assert picked.labels is None and picked.cleaned is None


def test_pick_cleaned_labels_of_the_recorded_run(tmp_path):
    from karak.cli.view import find_run_outputs

    cache = tmp_path / "output" / "work" / "cache"
    rec = {name: _store(cache, f"rec{name}", port, payload, 1000)
           for name, port, payload in [
               ("c", "cube", _cube()), ("d", "cube", _denoised()),
               ("n", "cube", _normalized()), ("p", "features", _features()),
               ("h", "labels", _labels()), ("k", "labels", _cleaned())]}
    _store(cache, "newer", "labels", _cleaned(), 5000)   # not from the run
    out_base = tmp_path / "output" / "run"
    _write_record(out_base, {"cube": rec["c"]}, denoised=rec["d"],
                  normalized=rec["n"], features=rec["p"], labels=rec["h"],
                  cleaned=rec["k"])
    picked = pick_outputs(find_run_outputs(out_base), newest_first=False)
    assert (picked.labels.recipe, picked.cleaned.recipe) == ("rech", "reck")


def test_knn_layer_shows_every_mineral_pixel_with_a_phase():
    cleaned = _cleaned()
    specs = layer_specs(_cube(factor=2, trim=3), None, _masks(), None, None,
                        None, _labels(), cleaned)
    by_name = {s.name: s for s in specs}
    names = [s.name for s in specs]
    assert names[names.index("hdb: phases"):] == [
        "hdb: phases", "hdb: noise", "hdb: probability", "knn: phases",
        "mineral mask", "valid mask"]
    knn = by_name["knn: phases"]
    assert knn.kind == "labels" and not knn.visible
    rows, cols = cleaned.mineral_indices.T
    np.testing.assert_array_equal(knn.data[rows, cols], [1, 1, 1, 2, 2, 2])
    assert (knn.data[~_masks().mineral_mask] == 0).all()
    assert (knn.scale, knn.translate) == (by_name["hdb: phases"].scale,
                                          by_name["hdb: phases"].translate)


def test_knn_layer_skipped_when_the_image_shape_differs():
    specs = layer_specs(_cube(), None, None, None, None, None, None,
                        _cleaned(image_shape=(8, 10)))
    assert not any(s.name.startswith("knn:") for s in specs)


def test_knn_summary_counts_the_reassigned_pixels():
    from karak.cli.view import knn_summary

    assert knn_summary(_cleaned(), _labels()) == (
        "knn: 2 phases; pixels per phase: 0 3 (+1), 1 3 (+1)")
    assert knn_summary(_cleaned()) == "knn: 2 phases; pixels per phase: 0 3, 1 3"


def test_view_main_prints_the_knn_summary(tmp_path, monkeypatch, capsys):
    cache = tmp_path / "work" / "cache"
    _store_chain(cache)
    _store(cache, "hdb", "labels", _labels(), 5000, upstream={"features": "pca"})
    _store(cache, "knn", "labels", _cleaned(), 6000,
           upstream={"features": "pca", "labels": "hdb"})
    opened = {}
    monkeypatch.setattr(view, "_napari_available", lambda: True)
    monkeypatch.setattr(view, "open_viewer",
                        lambda specs, shapes: opened.update(specs=specs))
    assert view_main([str(cache)]) == 0
    assert "knn: 2 phases; pixels per phase: 0 3 (+1), 1 3 (+1)" in capsys.readouterr().out
    assert [s.name for s in opened["specs"]][-1] == "knn: phases"
