"""End-to-end regression: tables of the new pipeline (outputs/app/site155/tables) equal the old tables
(outputs/site155_all/tables) on all shared columns and all rows, and the 219 golden statistics hold on the new tables.

The test only compares files; run `python scripts/run_pipeline.py config/site_155.yaml` and
`python scripts/run_stats.py config/site_155.yaml --write-tables` first.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core import stats
import test_golden_stats as gs      # tests/ is on sys.path (pytest rootdir import mode)

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "outputs" / "site155_all" / "tables"
NEW = ROOT / "outputs" / "app" / "site155" / "tables"

# Pipeline stage tables (CSV; the GPKG layers are compared on their attribute columns)
TABLES = [
    "field_trees.csv", "orientation_check.csv", "match_distance_only.csv", "field_trees_detection.csv",
    "detection_per_parzelle.csv", "structure_per_tree.csv", "canopy_per_parzelle.csv", "column_structure_per_tree.csv",
    "stem_form_per_tree.csv", "tree_profile_all.csv", "missed_trees_profile.csv", "match_causes.csv",
    "height_dbh_all_trees.csv", "match_tiers_auto.csv", "match_tiers.csv", "lidar_instances_without_field_tree.csv",
    "chm_baseline_tops.csv", "chm_baseline_matches.csv", "point_density_per_parzelle.csv", "chance_control_matches.csv",
]
GPKG = ["lidar_instances.gpkg"]

# Columns that exist only in the old tables because their part is not ported (photogrammetry, three-algorithm
# comparison, stem-seeded refinement, March flights). Anything else missing in a new table fails the test.
EXPECTED_GAPS = {
    "height_dbh_all_trees.csv": {"photo_height", "photo_dbh", "photo_found", "refined", "march"},   # prefixes, see below
}
GAP_PREFIXES = ("photo", "visible_from_above", "refine", "march", "algo", "segmentation_comparison", "tsl", "lidar360", "treeiso")

TOL = {"overtopped_by_m": 5e-6}
DEFAULT_TOL = 1e-6
KEYS = ("tree_id", "instance", "parzelle", "top_id")


def _is_gap(table, column):
    return column in EXPECTED_GAPS.get(table, set()) or column.lower().startswith(GAP_PREFIXES)


def _read(path):
    if path.suffix == ".gpkg":
        import geopandas as gpd
        df = gpd.read_file(path, layer="instance_base")
        return pd.DataFrame(df.drop(columns="geometry"))
    return pd.read_csv(path, dtype={"tree_id": str})


def _align(old, new, name):
    """Return old and new with the same row order (by the first unique shared key column, else by position)."""
    for key in KEYS:
        if key in old.columns and key in new.columns and old[key].is_unique and new[key].is_unique:
            assert set(old[key].dropna()) == set(new[key].dropna()), f"{name}: row keys ({key}) differ"
            return old.sort_values(key).reset_index(drop=True), new.sort_values(key).reset_index(drop=True)
    assert len(old) == len(new), f"{name}: {len(old)} old rows, {len(new)} new rows"
    return old.reset_index(drop=True), new.reset_index(drop=True)


def _compare_column(name, col, o, n):
    tol = TOL.get(col, DEFAULT_TOL)
    if pd.api.types.is_numeric_dtype(o) and pd.api.types.is_numeric_dtype(n):
        o, n = o.astype(float).to_numpy(), n.astype(float).to_numpy()
        bad = ~(np.isclose(o, n, rtol=0, atol=tol, equal_nan=True))
        if bad.any():
            diff = np.nanmax(np.abs(o[bad] - n[bad])) if np.isfinite(o[bad] - n[bad]).any() else float("nan")
            return f"{name}.{col}: {bad.sum()} of {len(o)} rows differ, largest difference {diff:.3g}"
    else:
        o, n = o.astype("string").fillna("<NA>"), n.astype("string").fillna("<NA>")
        bad = (o != n).to_numpy()
        if bad.any():
            i = int(np.flatnonzero(bad)[0])
            return f"{name}.{col}: {bad.sum()} of {len(o)} rows differ, e.g. old {o.iloc[i]!r} new {n.iloc[i]!r}"


@pytest.mark.slow
@pytest.mark.parametrize("name", TABLES + GPKG)
def test_table_equals_old(name):
    if not (OLD / name).exists():
        pytest.skip(f"{name} does not exist in the old tables")
    assert (NEW / name).exists(), f"{name} was not written by the pipeline"
    old, new = _read(OLD / name), _read(NEW / name)
    missing = [c for c in old.columns if c not in new.columns and not _is_gap(name, c)]
    assert not missing, f"{name}: columns missing in the new table that are not expected gaps: {missing}"
    old, new = _align(old, new, name)
    problems = [m for m in (_compare_column(name, c, old[c], new[c]) for c in old.columns if c in new.columns) if m]
    assert not problems, "\n".join(problems)


@pytest.mark.slow
def test_golden_statistics_on_new_tables():
    computed, _ = stats.compute_all(stats.load_tables(NEW), **gs.PARAMS)
    entries = {(e["id"], e["subset"]): e for e in computed}
    failures = []
    for g in gs.GOLDEN:
        e = entries.get((g["id"], g["subset"]))
        got = None if e is None else e[g["field"]]
        if got is None:
            failures.append(f"{g['id']}|{g['subset']}|{g['field']}: missing")
        elif g["kind"] == "count":
            if got != g["expected"]:
                failures.append(f"{g['id']}|{g['subset']}|{g['field']}: old {g['expected']} new {got}")
        elif not math.isclose(got, g["expected"], rel_tol=g["tol"].get("rel", 0), abs_tol=g["tol"].get("abs", 0)):
            failures.append(f"{g['id']}|{g['subset']}|{g['field']}: old {g['expected']!r} new {got!r}")
    assert not failures, f"{len(failures)} of {len(gs.GOLDEN)} golden statistics differ:\n" + "\n".join(failures[:30])
