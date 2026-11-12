"""Golden tests: the core/ stages inventory, orientation and instances reproduce the notebook tables.

The inventory fixture runs first and reads Excel before any test reads a GeoPackage (Excel after
geopandas crashes in this environment).
"""

import copy

import numpy as np
import pandas as pd
import pytest

from core.config import Config, load_config

from conftest import ROOT, SITE_155

OLD = ROOT / "outputs" / "site155_all" / "tables"
if not OLD.is_dir() or not all(p.is_file() for p in load_config(SITE_155).paths.field_tables):
    pytest.skip("field data and reference tables of site 155 are not available", allow_module_level=True)


@pytest.fixture(scope="module")
def inventory_cfg(tmp_path_factory):
    """Config of site 155 writing to a temporary folder; the inventory stage has run."""
    from core import cache
    from core.io_field import load_corners, run_inventory

    raw = copy.deepcopy(load_config(SITE_155).raw)
    raw["paths"]["output_dir"] = str(tmp_path_factory.mktemp("site155"))
    cfg = Config(raw)
    cache.set_force(True)
    try:
        run_inventory(cfg)
    finally:
        cache.set_force(False)
    cfg._corners = load_corners(cfg)          # noqa: SLF001  (the corners are read from the CSV written above)
    return cfg


def _assert_frames_equal(new, old, tol=1e-6):
    assert list(new.columns) == list(old.columns)
    assert len(new) == len(old)
    for col in old.columns:
        a, b = new[col], old[col]
        if a.dtype == bool or b.dtype == bool:
            assert a.astype(bool).tolist() == b.astype(bool).tolist(), col
        elif not (pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b)):
            assert a.fillna("").astype(str).tolist() == b.fillna("").astype(str).tolist(), col
        else:
            np.testing.assert_allclose(a.to_numpy(float), b.to_numpy(float), rtol=0, atol=tol, equal_nan=True, err_msg=col)


def test_inventory_equals_notebook(inventory_cfg):
    old = pd.read_csv(OLD / "field_trees_detection.csv", dtype={"tree_id": str})
    old = old[list(old.columns[: list(old.columns).index("dbh_class") + 1])]
    new = pd.read_csv(inventory_cfg.table("field_trees.csv"), dtype={"tree_id": str})
    assert len(new) == 1602
    assert new["tree_id"].tolist() == old["tree_id"].tolist()
    _assert_frames_equal(new, old)


def test_orientation_equals_notebook(inventory_cfg):
    import geopandas as gpd

    from core.coords import check_orientation, orientation_check

    old_inst = gpd.read_file(OLD / "lidar_instances.gpkg", layer="instance_base")
    trees = pd.read_csv(inventory_cfg.table("field_trees.csv"), dtype={"tree_id": str})
    new = orientation_check(trees, old_inst[["x_base", "y_base"]].to_numpy(), inventory_cfg._corners)  # noqa: SLF001
    old = pd.read_csv(OLD / "orientation_check.csv").set_index("parzelle")
    assert new.index.tolist() == old.index.tolist()
    assert list(new.columns) == list(old.columns)
    assert new["best"].tolist() == old["best"].tolist()
    assert new["trees"].tolist() == old["trees"].tolist()
    np.testing.assert_allclose(new.drop(columns=["best"]).to_numpy(float), old.drop(columns=["best"]).to_numpy(float),
                               rtol=0, atol=1e-9)
    check_orientation(new, inventory_cfg.plots.y_mirrored)       # agrees with plots.y_mirrored


@pytest.mark.slow
def test_instances_equal_notebook(site155_path):
    """Full run on the real clouds (about 15 minutes the first time, cached under outputs/app/site155/work)."""
    import geopandas as gpd

    from core.instances import instances_path, run_instances

    cfg = load_config(site155_path)
    run_instances(cfg)
    new = gpd.read_file(instances_path(cfg), layer="instance_base")
    old = gpd.read_file(OLD / "lidar_instances.gpkg", layer="instance_base")
    assert sorted(new["instance"]) == sorted(old["instance"])
    new, old = new.set_index("instance").sort_index(), old.set_index("instance").sort_index()
    for col in ["x_base", "y_base", "x_top", "y_top", "z_top", "height"]:
        np.testing.assert_allclose(new[col].to_numpy(float), old[col].to_numpy(float), rtol=0, atol=1e-3, err_msg=col)
    np.testing.assert_allclose(new["removed_fraction"].to_numpy(float), old["removed_fraction"].to_numpy(float),
                               rtol=0, atol=1e-3, equal_nan=True, err_msg="removed_fraction")

    new_d = pd.read_csv(cfg.table("point_density_per_parzelle.csv"))
    old_d = pd.read_csv(OLD / "point_density_per_parzelle.csv")
    assert list(new_d.columns) == list(old_d.columns)
    np.testing.assert_allclose(new_d.to_numpy(float), old_d.to_numpy(float), rtol=1e-6, atol=1e-6)
