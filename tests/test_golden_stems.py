"""Golden test of stages 5-8: the core functions reproduce the notebook tables of outputs/site155_all.

Each stage is fed with the OLD upstream tables (and for stems and DBH the old clipped before-harvest
cloud) and must give the OLD downstream table: categorical columns exact, numbers to 1e-6.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.config import load_config

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "outputs" / "site155_all"
TABLES = OLD / "tables"
CFG = load_config(ROOT / "config" / "site_155.yaml")

# Excel is read at import (collection) time, before any test reads a file with geopandas or rasterio.
try:
    from core.causes import plot_extent, plot_polygons, read_corners
    CORNERS = read_corners(CFG)
except Exception as exc:   # input data not reachable
    CORNERS = None
    CORNERS_ERROR = str(exc)

need_corners = pytest.mark.skipif(CORNERS is None, reason="plot corner sheet not readable")
need_old = pytest.mark.skipif(not (TABLES / "field_trees_detection.csv").exists(), reason="old tables missing")


def old(name):
    return pd.read_csv(TABLES / name, dtype={"tree_id": str})


def assert_table(new, expected, columns=None, tol=1e-6, col_tol=None):
    """Same rows in the same order; numbers close, everything else equal as text."""
    columns = list(columns if columns is not None else expected.columns)
    assert len(new) == len(expected)
    problems = []
    col_tol = col_tol or {}
    for col in columns:
        a, b = new[col].reset_index(drop=True), expected[col].reset_index(drop=True)
        tol_col = col_tol.get(col, tol)
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            a, b = a.astype(float).to_numpy(), b.astype(float).to_numpy()
            same_nan = np.isnan(a) == np.isnan(b)
            close = np.isclose(a, b, rtol=0, atol=tol_col, equal_nan=True) | (np.abs(a - b) <= tol_col * np.abs(b))
            bad = ~(same_nan & close)
            if bad.any():
                problems.append(f"{col}: {bad.sum()} rows differ, max abs diff {np.nanmax(np.abs(a - b)[bad]):.3g}")
        else:
            sa = a.astype(object).where(a.notna(), "").astype(str)
            sb = b.astype(object).where(b.notna(), "").astype(str)
            bad = sa != sb
            if bad.any():
                problems.append(f"{col}: {bad.sum()} rows differ, e.g. {sa[bad].iloc[0]!r} vs {sb[bad].iloc[0]!r}")
    assert not problems, "; ".join(problems)


@pytest.fixture(scope="module")
def trees():
    return old("field_trees_detection.csv")


@pytest.fixture(scope="module")
def chm():
    from core.causes import read_chm
    return read_chm(OLD / "rasters" / "chm_lidar_before.tif")


# --- stage 5 ----------------------------------------------------------------

@need_old
def test_structure_per_tree(trees, chm):
    from core.causes import structure_per_tree

    struct, _ = structure_per_tree(
        trees, *chm, dbh_plausible_mm=CFG.cleaning.dbh_plausible_mm,
        competition_radius_m=CFG.structure.competition_radius_m, chm_stem_radius_m=CFG.structure.chm_stem_radius_m,
        chm_resolution_m=CFG.grid.chm_resolution_m, no_canopy_deficit_m=CFG.structure.no_canopy_deficit_m,
        high_confidence_distance_m=CFG.matching.high_confidence_distance_m,
        gone_above=CFG.matching.gone_above, standing_below=CFG.matching.standing_below)
    expected = old("structure_per_tree.csv")
    assert len(expected) == 1602
    assert_table(struct, expected)


@need_old
@need_corners
def test_canopy_per_parzelle(chm):
    from core.causes import canopy_per_parzelle

    new = canopy_per_parzelle(*chm, plot_polygons(CORNERS, CFG.site.crs), CFG.structure.gap_height_m)
    assert_table(new, pd.read_csv(TABLES / "canopy_per_parzelle.csv"))


# --- stage 6 (profile from the old stem tables) ------------------------------

@need_old
@need_corners
def test_tree_profile_and_missed(trees):
    from core.dbh import missed_trees, tree_profile

    width, depth = plot_extent(CORNERS)
    area = pd.read_csv(TABLES / "canopy_per_parzelle.csv").set_index("parzelle")["area_m2"]
    profile = tree_profile(
        old("structure_per_tree.csv"), trees, old("stem_form_per_tree.csv"), area, width, depth,
        dbh_plausible_mm=CFG.cleaning.dbh_plausible_mm, standing_height_dm_above=CFG.cleaning.standing_height_dm_above,
        competition_radius_m=CFG.structure.competition_radius_m, species_groups=CFG.structure.species_groups.to_dict())
    expected = old("tree_profile_all.csv")
    assert len(expected) == 1602
    assert_table(profile, expected)
    missed = missed_trees(profile)
    assert_table(missed, old("missed_trees_profile.csv"))


# --- stage 7 -----------------------------------------------------------------

@need_old
def test_match_causes(trees):
    import geopandas as gpd

    from core.causes import match_causes_table

    instances = gpd.read_file(TABLES / "lidar_instances.gpkg", layer="instance_base")
    out = match_causes_table(
        old("tree_profile_all.csv"), trees, instances, old("column_structure_per_tree.csv"),
        high_confidence_distance_m=CFG.matching.high_confidence_distance_m, gone_above=CFG.matching.gone_above,
        standing_below=CFG.matching.standing_below, no_canopy_deficit_m=CFG.structure.no_canopy_deficit_m,
        competition_radius_m=CFG.structure.competition_radius_m, instance_radius_m=CFG.structure.instance_radius_m,
        dbh_plausible_mm=CFG.cleaning.dbh_plausible_mm)
    expected = old("match_causes.csv")
    assert len(expected) == 1602
    # chm_at_stem_m is stored as float32 text (8 digits) in the old profile, the notebook used the unrounded value
    assert_table(out, expected, col_tol={"overtopped_by_m": 5e-6})


def test_cause_texts():
    from core.causes import CAUSES, cause_explanations

    texts = cause_explanations(CFG.structure.no_canopy_deficit_m)
    assert set(texts) == set(CAUSES)
    assert "more than 6 m below" in texts["no tree at the position on the flight day"]


# --- stages 6 and 8 from the point cloud -------------------------------------

@pytest.fixture(scope="module")
def cloud():
    from core.pointcloud import add_height_above_ground, load_instance_cloud

    points = load_instance_cloud(OLD / "subset" / "sat_before_aoi.laz", instance_dim=CFG.paths.sat_before.instance_dim)
    return add_height_above_ground(points, OLD / "rasters" / "dtm_lidar_after.tif")


@pytest.mark.slow
@need_old
def test_column_structure(cloud, trees):
    from core.dbh import column_structure, field_positions

    assert_table(column_structure(cloud, field_positions(trees)), old("column_structure_per_tree.csv"))


@pytest.mark.slow
@need_old
def test_stem_form(cloud, trees):
    from core.dbh import stem_form_table

    expected = old("stem_form_per_tree.csv")
    assert len(expected) == 1602
    assert_table(stem_form_table(cloud, trees), expected)


@pytest.mark.slow
@need_old
def test_lidar_dbh(cloud, trees):
    from core.dbh import dbh_table

    new = dbh_table(cloud, trees, breast_height_m=CFG.dbh.breast_height_m, slice_half_height_m=CFG.dbh.slice_half_height_m,
                    search_radius_m=CFG.dbh.search_radius_m, ellipse_axis_ratio=CFG.dbh.ellipse_axis_ratio)
    expected = old("height_dbh_all_trees.csv")
    assert len(expected) == 1602
    columns = ["tree_id", "dbh_centre_from", "dbh_lidar_mm", "x_stem", "y_stem", "n_slice", "n_inliers", "n_interior",
               "rmse_mm", "arc_coverage", "dbh_quality", "dbh_ellipse_mm", "dbh_ellipse_area_mm", "ellipse_a_mm",
               "ellipse_b_mm", "ellipse_axis_ratio", "ellipse_theta", "ellipse_cx", "ellipse_cy", "ellipse_rmse_mm"]
    assert_table(new, expected, columns)
