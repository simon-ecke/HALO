"""Golden tests: the core/ matching, tier, unmatched and baseline stages reproduce the old notebook tables.

Every stage is fed with the OLD upstream tables of outputs/site155_all/tables/ and must give the OLD
downstream table. Tests that need a point cloud or a raster are marked slow.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "outputs" / "site155_all"
TABLES = OLD / "tables"

pytestmark = pytest.mark.skipif(not (TABLES / "field_trees_detection.csv").is_file(), reason="old 155 tables not available")

# Excel must be read before geopandas / rasterio read any file in this process: read it at import time.
try:
    from core.config import load_config
    from core.matching import load_plots

    _CFG = load_config(ROOT / "config" / "site_155.yaml")
    _PLOTS, _CORNER_RMSE = load_plots(_CFG)
except Exception as exc:                       # pragma: no cover - depends on the data drive
    _CFG, _PLOTS, _CORNER_RMSE, _LOAD_ERROR = None, None, None, exc
else:
    _LOAD_ERROR = None

needs_corners = pytest.mark.skipif(_PLOTS is None, reason=f"plot corners not readable: {_LOAD_ERROR}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def old_table(name, **kwargs):
    return pd.read_csv(TABLES / name, dtype={"tree_id": str}, **kwargs)


def assert_tables_equal(new, old, tol=1e-6, label=""):
    """Same columns in the same order; numbers to `tol`, everything else exact (NaN equals NaN)."""
    new = pd.read_csv(io.StringIO(new.to_csv(index=False)), dtype={"tree_id": str})     # as written to and read from file
    old = old.reset_index(drop=True)
    assert list(new.columns) == list(old.columns), f"{label}: columns differ: {list(new.columns)} vs {list(old.columns)}"
    assert len(new) == len(old), f"{label}: {len(new)} rows, old {len(old)}"
    for col in old.columns:
        a, b = new[col], old[col]
        if pd.api.types.is_bool_dtype(b) or pd.api.types.is_bool_dtype(a) or b.dtype == object or pd.api.types.is_string_dtype(b):
            sa = a.astype(object).where(a.notna(), None).map(lambda v: v if v is None else str(v))
            sb = b.astype(object).where(b.notna(), None).map(lambda v: v if v is None else str(v))
            bad = ~((sa == sb) | (sa.isna() & sb.isna()))
            assert not bad.any(), f"{label}.{col}: {int(bad.sum())} rows differ, first: {sa[bad].iloc[0]!r} vs {sb[bad].iloc[0]!r}"
        else:
            x, y = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
            both_nan = np.isnan(x) & np.isnan(y)
            diff = np.where(both_nan, 0.0, np.abs(x - y))
            assert not np.isnan(diff).any(), f"{label}.{col}: NaN in one table only ({int(np.isnan(diff).sum())} rows)"
            assert diff.max() <= tol, f"{label}.{col}: {int((diff > tol).sum())} rows differ, largest {diff.max():.3g}"


@pytest.fixture(scope="module")
def old_detection():
    return old_table("field_trees_detection.csv")


@pytest.fixture(scope="module")
def old_instances():
    import geopandas as gpd

    return gpd.read_file(TABLES / "lidar_instances.gpkg", layer="instance_base")


@pytest.fixture(scope="module")
def inventory(old_detection):
    """The inventory columns of the old detection table (everything before the matching columns)."""
    return old_detection.loc[:, :"dbh_class"].copy()


@pytest.fixture(scope="module")
def matching_tables(inventory, old_instances):
    from core.matching import compute_matching

    m = _CFG.matching
    return compute_matching(
        inventory, old_instances, _PLOTS, _CORNER_RMSE,
        max_distance_m=m.max_distance_m, removed_threshold=m.removed_threshold, removal_penalty_m=m.removal_penalty_m,
        high_confidence_distance_m=m.high_confidence_distance_m, height_gate_rel=m.height_gate_rel,
        gone_above=m.gone_above, standing_below=m.standing_below)


# ---------------------------------------------------------------------------
# Stage 4: matching and detection classes
# ---------------------------------------------------------------------------

@needs_corners
def test_instance_and_detection_exact(matching_tables, old_detection):
    new = matching_tables["table"]
    assert len(new) == len(old_detection) == 1602
    assert (new["tree_id"].to_numpy() == old_detection["tree_id"].to_numpy()).all()
    new_inst = new["instance"].astype("float64").to_numpy()
    old_inst = old_detection["instance"].astype("float64").to_numpy()
    assert np.array_equal(new_inst, old_inst, equal_nan=True)
    assert (new["detection"].to_numpy() == old_detection["detection"].to_numpy()).all()


@needs_corners
def test_field_trees_detection(matching_tables, old_detection):
    assert_tables_equal(matching_tables["table"], old_detection, label="field_trees_detection")


@needs_corners
def test_match_distance_only(matching_tables):
    new = matching_tables["felled"].copy()
    new["lidar"] = new["lidar"].astype(str)
    assert_tables_equal(new, old_table("match_distance_only.csv"), label="match_distance_only")


@needs_corners
def test_detection_per_parzelle(matching_tables):
    new = matching_tables["per_plot"].reset_index()
    assert_tables_equal(new, pd.read_csv(TABLES / "detection_per_parzelle.csv"), label="detection_per_parzelle")


@needs_corners
def test_detection_gpkg_layers(matching_tables, tmp_path):
    import geopandas as gpd
    import pyogrio

    from core.matching import write_detection_gpkg

    path = tmp_path / "detection_all_trees.gpkg"
    write_detection_gpkg(path, matching_tables["table"], matching_tables["inst"], matching_tables["per_plot"], _PLOTS,
                         _CFG.site.crs)
    old_path = TABLES / "detection_all_trees.gpkg"
    assert sorted(n for n, _ in pyogrio.list_layers(path)) == sorted(n for n, _ in pyogrio.list_layers(old_path))
    for layer in ["field_trees", "lidar_instances", "stem_to_base", "parzellen"]:
        assert len(gpd.read_file(path, layer=layer)) == len(gpd.read_file(old_path, layer=layer)), layer


@needs_corners
def test_inventory_fallbacks(inventory):
    """Block and DBH class are derived from the inventory and the config when the inventory lacks them."""
    from core.matching import add_block, add_dbh_class

    bare = inventory.drop(columns=["block", "dbh_class"])
    again = add_dbh_class(add_block(bare), _CFG.cleaning.dbh_plausible_mm, _CFG.cleaning.dbh_class_edges_mm)
    assert (again["block"] == inventory["block"]).all()
    assert (again["dbh_class"].astype(str) == inventory["dbh_class"]).all()


# ---------------------------------------------------------------------------
# Stage 10: LiDAR instances without a field tree
# ---------------------------------------------------------------------------

@needs_corners
def test_unmatched_instances(old_detection, old_instances):
    """Stem evidence comes from the old table; the rest of the evidence and the readings are recomputed."""
    from core.matching import CLASSES, instances_in_plots, unmatched_instances

    old = pd.read_csv(TABLES / "lidar_instances_without_field_tree.csv")
    result = old_detection.copy()
    result["detection"] = pd.Categorical(result["detection"], categories=CLASSES)
    m = _CFG.matching
    inst, in_plot = instances_in_plots(old_instances, _PLOTS, result, m.removed_threshold)
    stems = old[["instance", "stem_slices", "stem_offset_m", "stem_instance", "stem_instance_share"]].rename(
        columns={"instance": "tree_id"})
    new = unmatched_instances(
        inst, in_plot, result, _PLOTS, stems, max_distance_m=m.max_distance_m,
        high_confidence_distance_m=m.high_confidence_distance_m, min_stem_slices=_CFG.structure.min_stem_slices,
        edge_m=_CFG.structure.edge_m, figure_count=m.unmatched_figure_count, figure_seed=m.unmatched_figure_seed)
    assert_tables_equal(new, old, label="lidar_instances_without_field_tree")


# ---------------------------------------------------------------------------
# Stage 9: tiers
# ---------------------------------------------------------------------------

def test_height_gate_matches_old_tier(old_detection):
    """The gate columns of the old automatic tier; confidence differs only where the column profile changed it."""
    from core.tiers import add_height_gate, matches_from_detection

    old = old_table("match_tiers_auto.csv")
    gated = add_height_gate(matches_from_detection(old_detection), old_detection, _CFG.matching.height_gate_rel)
    assert_tables_equal(gated[["tree_id", "field_height"]], old[["tree_id", "field_height"]], label="field_height")
    new_gate = gated["height_gate_ok"].astype(object).where(gated["height_gate_ok"].notna(), None)
    old_gate = old["height_gate_ok"].astype(object).where(old["height_gate_ok"].notna(), None)
    assert (new_gate.map(str) == old_gate.map(str)).all()
    untouched = old["method"].isin(["sat_instance"])
    assert (gated.loc[untouched, "confidence"].to_numpy() == old.loc[untouched, "confidence"].to_numpy()).all()



def test_final_tier_equals_auto_without_manual_input():
    from core.tiers import final_tier

    old_auto = old_table("match_tiers_auto.csv")
    old_final = old_table("match_tiers.csv")
    tiers = final_tier(old_auto, None)
    assert_tables_equal(tiers, old_final, label="match_tiers")
    assert (tiers["confidence"] == tiers["auto_confidence"]).all() and (tiers["method"] == tiers["auto_method"]).all()


def test_overrides_and_manual_segments_keep_their_method():
    import geopandas as gpd
    from shapely.geometry import Point

    from core.tiers import final_tier

    auto = old_table("match_tiers_auto.csv").iloc[:6].copy()
    instances = gpd.GeoDataFrame({"instance": [1, 2], "height": [20.0, 21.0], "x_top": [1.0, 2.0], "y_top": [1.0, 2.0],
                                  "x_base": [1.0, 2.0], "y_base": [1.0, 2.0], "removed_fraction": [0.0, 0.0],
                                  "lean_m": [0.1, 0.2], "crown_base_m": [5.0, 6.0]},
                                 geometry=[Point(1, 1), Point(2, 2)])
    ids = auto["tree_id"].tolist()
    overrides = pd.DataFrame({"tree_id": [ids[0], ids[1]], "instance": [2, 0]})
    manual = pd.DataFrame({"tree_id": [ids[2]], "lidar_height": [12.5], "x_top": [1.0], "y_top": [2.0], "x_base": [3.0],
                           "y_base": [4.0], "lean_m": [0.3], "n_points": [10], "file": ["tree_x.laz"]})
    out = final_tier(auto, instances, overrides, manual)
    assert out.loc[0, ["confidence", "method"]].tolist() == ["manual", "sat_instance"]
    assert out.loc[1, ["confidence", "method"]].tolist() == ["manual", "none"]
    assert out.loc[2, ["confidence", "method"]].tolist() == ["manual", "manual_segmentation"] or \
        out.loc[2, ["method", "confidence"]].tolist() == ["manual_segmentation", "manual"]
    assert out.loc[2, "lidar_height"] == 12.5
    # the untouched automatic result stays in the auto_* columns
    assert (out["auto_confidence"].to_numpy() == auto["confidence"].to_numpy()).all()
    assert (out["auto_method"].to_numpy() == auto["method"].to_numpy()).all()


@pytest.mark.slow
@needs_corners
def test_automatic_tier_with_cloud(old_detection, inventory):
    """Height gate and column-profile fallback on the old clipped cloud reproduce match_tiers_auto.csv."""
    from core.pointcloud import add_height_above_ground, load_instance_cloud
    from core.tiers import automatic_tier, matches_from_detection
    from modules.field_tree_matching import flag_removed_points            # test only: the old removed-point flag

    cp = _CFG.matching.column_profile
    before = load_instance_cloud(OLD / "subset" / "sat_before_aoi.laz", _CFG.paths.sat_before.instance_dim)
    after = load_instance_cloud(OLD / "subset" / "sat_after_aoi.laz", _CFG.paths.sat_after.instance_dim)
    dtm = OLD / "rasters" / "dtm_lidar_after.tif"
    before = add_height_above_ground(before, dtm)
    after = add_height_above_ground(after, dtm)
    min_x, min_y, max_x, max_y = _PLOTS.total_bounds
    buf = _CFG.grid.aoi_buffer_m
    removed = flag_removed_points(before, after, (min_x - buf, min_y - buf, max_x + buf, max_y + buf),
                                  voxel_m=_CFG.grid.removed_voxel_m)
    del after
    auto = automatic_tier(matches_from_detection(old_detection), old_detection, before, removed,
                          height_gate_rel=_CFG.matching.height_gate_rel, radius_m=cp.radius_m, max_gap_m=cp.max_gap_m,
                          min_height_m=cp.min_height_m, min_points=cp.min_points)
    old = old_table("match_tiers_auto.csv")
    new_inst = auto["instance"].astype("float64").to_numpy()
    assert np.array_equal(new_inst, old["instance"].astype("float64").to_numpy(), equal_nan=True)
    assert (auto["confidence"].to_numpy() == old["confidence"].to_numpy()).all()
    assert (auto["method"].to_numpy() == old["method"].to_numpy()).all()
    assert_tables_equal(auto, old, label="match_tiers_auto")


# ---------------------------------------------------------------------------
# Stage 11: baseline
# ---------------------------------------------------------------------------

@pytest.mark.slow
@needs_corners
def test_baseline_tops_and_matches(old_instances, inventory):
    import rasterio

    from core.baseline import baseline_matches, baseline_tops

    with rasterio.open(OLD / "rasters" / "chm_lidar_before.tif") as src:
        chm = src.read(1).astype(np.float32)
        chm[chm == src.nodata] = np.nan
        transform = src.transform
    b = _CFG.baseline
    windows = [(w.name, w.diameter_m, w.get("per_height_m", 0.0)) for w in b.windows]
    tops = baseline_tops(chm, transform, windows, old_instances, _PLOTS, min_height_m=b.min_height_m,
                         smoothing_m=b.smoothing_m)
    old_tops = pd.read_csv(TABLES / "chm_baseline_tops.csv")
    assert tops.groupby("method", sort=False).size().to_dict() == old_tops.groupby("method", sort=False).size().to_dict()
    assert_tables_equal(tops, old_tops, label="chm_baseline_tops")

    matches = baseline_matches(tops, inventory[["tree_id", "utm_x", "utm_y", "removed_2026"]],
                               top_match_max_m=b.top_match_max_m, top_found_distance_m=b.top_found_distance_m,
                               control_shift_m=_CFG.matching.control_shift_m)
    assert_tables_equal(matches, old_table("chm_baseline_matches.csv"), label="chm_baseline_matches")
