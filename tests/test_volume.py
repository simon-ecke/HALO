"""Synthetic tests for core.volume (no project data)."""
import copy
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import volume as V  # noqa: E402
from core.config import Config, load_config  # noqa: E402

FF = pd.DataFrame({"species": ["A", "B", "default"], "form_factor": [0.5, 0.4, 0.45]})
PT = pd.DataFrame({"species": ["A", "default"], "a": [0.0001, 0.0002], "b": [2.0, 2.0], "c": [1.0, 1.0]})
KW = dict(dbh_plausible_mm=(70, 1000), min_trees_per_species=5, breast_height_m=1.3, diameter_unit="cm")
SENS = {"height_m": 1, "dbh_mm": 10, "form_factor": 0.02}
CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "site_155.yaml")


def ff():
    return V.METHODS["form_factor"], V.METHODS["form_factor"].prepare(FF)


def stand(n=40, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.uniform(150, 600, n)
    h = 1.3 + (d / 10 / (1.5 + 0.2 * d / 10)) ** 2 * rng.uniform(0.95, 1.05, n)
    return pd.DataFrame({"tree_id": [f"1_{i}" for i in range(n)], "parzelle": 1, "species": ["A"] * 30 + ["B"] * (n - 30),
                         "dbh_field_mm": d, "dbh_uls_mm": d * rng.uniform(0.95, 1.05, n), "height_uls_m": h,
                         "detection": "detected", "identity_class": "confirmed", "removed_2026": False})


def run(trees, method_name="form_factor"):
    m = V.METHODS[method_name]
    params = m.prepare(FF if method_name == "form_factor" else PT)
    return V.compute_volume_table(trees, method=m, params=params, **KW)


def entries(table, curves, **over):
    m, p = ff()
    kw = dict(method=m, params=p, diameter_unit="cm", areas_ha=pd.Series({1: 0.5}), ratio={"n": 10, "median": 1.0},
              plausible_mm=(70, 1000), class_edges_mm=[0, 300, 2000], min_trees=5, n_boot=200, seed=0, sens=SENS)
    kw.update(over)
    return V.entries_from_table(table, curves, **kw)


def test_known_tree_form_factor():
    tr = stand().iloc[:1].copy()
    tr[["dbh_field_mm", "dbh_uls_mm", "height_uls_m"]] = [400.0, 380.0, 30.0]
    tr["species"] = "A"
    table, _ = V.compute_volume_table(tr, method=ff()[0], params=ff()[1], **{**KW, "min_trees_per_species": 1})
    assert math.isclose(table.wood_volume_field_m3.iloc[0], math.pi / 4 * 0.4 ** 2 * 30 * 0.5, rel_tol=1e-12)   # 1.88496 m3
    assert math.isclose(table.wood_volume_uls_m3.iloc[0], math.pi / 4 * 0.38 ** 2 * 30 * 0.5, rel_tol=1e-12)
    assert table.height_source.iloc[0] == "uls"


def test_default_row_for_unknown_species():
    tr = stand().iloc[:1].copy()
    tr["species"] = "Z"
    tr[["dbh_field_mm", "height_uls_m"]] = [200.0, 20.0]
    table, _ = V.compute_volume_table(tr, method=ff()[0], params=ff()[1], **{**KW, "min_trees_per_species": 1})
    assert math.isclose(table.wood_volume_field_m3.iloc[0], math.pi / 4 * 0.04 * 20 * 0.45)


def test_parameter_table_method():
    tr = stand().iloc[:1].copy()
    tr[["species", "dbh_field_mm", "dbh_uls_mm", "height_uls_m"]] = ["A", 400.0, 400.0, 25.0]
    m = V.METHODS["parameter_table"]
    table, _ = V.compute_volume_table(tr, method=m, params=m.prepare(PT), **{**KW, "min_trees_per_species": 1})
    assert math.isclose(table.wood_volume_field_m3.iloc[0], 0.0001 * 40.0 ** 2 * 25.0)    # d in cm
    assert table.volume_method.iloc[0] == "parameter_table"


def test_sum_of_trees_equals_reported_totals():
    t = stand()
    t.loc[3, "height_uls_m"] = np.nan                                  # one unmatched tree
    table, curves = run(t)
    by = {e["id"]: e for e in entries(table, curves)}
    assert math.isclose(by["volume.total.field_with_modelled.stand"]["value"], table.wood_volume_field_m3.sum())
    assert math.isclose(by["volume.total.uls.stand"]["value"], table.wood_volume_uls_m3.sum())
    assert math.isclose(by["volume.total.field_without_modelled.stand"]["value"],
                        table.loc[table.height_source == "uls", "wood_volume_field_m3"].sum())
    assert math.isclose(by["volume.per_ha.uls.stand"]["value"], table.wood_volume_uls_m3.sum() / 0.5)
    sp = sum(by[f"volume.total.uls.{s}"]["value"] for s in ("a", "b"))
    assert math.isclose(sp, by["volume.total.uls.stand"]["value"])
    gap = sum(by[k]["value"] for k in ("volume.gap.trees_missed", "volume.gap.diameter_difference", "volume.gap.uls_only_trees"))
    assert math.isclose(gap, by["volume.gap.total"]["value"], abs_tol=1e-9)
    assert all("provisional" in e["statement"] for e in by.values() if e["id"].startswith("volume.total"))
    assert all(set(e) == {"id", "value", "unit", "n", "ci_low", "ci_high", "subset", "statement"} for e in by.values())


def test_field_and_uls_columns_do_not_mix():
    t = stand()
    base, _ = run(t)
    t2 = t.copy()
    t2["dbh_uls_mm"] = t2["dbh_uls_mm"] * 1.2
    changed, _ = run(t2)
    assert np.allclose(base.wood_volume_field_m3, changed.wood_volume_field_m3)
    assert not np.allclose(base.wood_volume_uls_m3, changed.wood_volume_uls_m3)
    t3 = t.copy()
    t3["dbh_field_mm"] = t3["dbh_field_mm"] * 1.2
    changed, _ = run(t3)
    assert np.allclose(base.wood_volume_uls_m3, changed.wood_volume_uls_m3)
    assert not np.allclose(base.wood_volume_field_m3, changed.wood_volume_field_m3)


def test_implausible_field_diameter_gives_no_field_volume():
    t = stand()
    t.loc[0, "dbh_field_mm"] = 5933.0
    t.loc[1, "dbh_field_mm"] = np.nan
    t.loc[2, "dbh_field_mm"] = 40.0
    table, curves = run(t)
    assert not table.loc[:2, "dbh_field_valid"].any()
    assert table.loc[:2, "wood_volume_field_m3"].isna().all()
    assert table.loc[3:, "dbh_field_valid"].all()
    assert table.loc[0, "wood_volume_uls_m3"] > 0                     # the ULS side is not touched
    ex = next(e for e in entries(table, curves) if e["id"] == "volume.field_excluded")
    assert ex["value"] == 3 and all(f"1_{i}" in ex["statement"] for i in range(3))


def test_unmatched_tree_gets_modelled_height_and_no_uls_volume():
    t = stand()
    t.loc[5, "height_uls_m"] = np.nan
    m, p = ff()
    table, curves = V.compute_volume_table(t, method=m, params=p, **{**KW, "min_trees_per_species": 15})
    r = table.loc[5]
    assert r.height_source == "modelled" and np.isfinite(r.height_used_m) and r.height_used_m > 1.3
    assert np.isnan(r.wood_volume_uls_m3) and r.wood_volume_field_m3 > 0
    assert set(curves.scope) == {"pooled", "species"} and list(curves[curves.scope == "species"].species) == ["A"]   # B has < 15 trees
    assert (curves.loo_rmse_m < 3).all()
    t.loc[35, "height_uls_m"] = np.nan                                 # species B is modelled from the pooled curve
    assert V.compute_volume_table(t, method=m, params=p, **{**KW, "min_trees_per_species": 15})[0].loc[35, "height_source"] == "modelled"


def test_method_in_config_changes_result(tmp_path):
    raw = copy.deepcopy(load_config(CONFIG).raw)
    pt = tmp_path / "pt.csv"
    PT.to_csv(pt, index=False)
    t = stand()
    out = {}
    for name, params in (("form_factor", raw["volume"]["params_file"]), ("parameter_table", str(pt))):
        r = copy.deepcopy(raw)
        r["volume"].update(method=name, params_file=params)
        m, p, unit = V.method_from_cfg(Config(r))
        out[name] = V.compute_volume_table(t, method=m, params=p, **{**KW, "diameter_unit": unit})[0]
    assert out["form_factor"].volume_method.iloc[0] == "form_factor"
    assert out["parameter_table"].volume_method.iloc[0] == "parameter_table"
    assert not np.allclose(out["form_factor"].wood_volume_field_m3, out["parameter_table"].wood_volume_field_m3)
    r = copy.deepcopy(raw)
    r["volume"]["method"] = "nope"
    try:
        V.method_from_cfg(Config(r))
        assert False, "unknown method must be refused"
    except ValueError:
        pass


def test_dbh_ratio_check_detects_diameter_and_circumference():
    t = stand()
    assert abs(V.dbh_ratio_check(t, (70, 1000))["median"] - 1.0) < 0.1
    t["dbh_field_mm"] = t["dbh_field_mm"] * math.pi
    assert V.dbh_ratio_check(t, (70, 5000))["median"] > 2.5
