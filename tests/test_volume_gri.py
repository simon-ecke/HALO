"""Synthetic tests for core.volume_gri (no project data)."""
import copy
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core import volume_gri as G  # noqa: E402
from core.config import Config, load_config  # noqa: E402

CONFIG = os.path.join(ROOT, "config", "site_155.yaml")
REFERENCE = os.path.join(ROOT, "tests", "data", "v_gri_reference.csv")
COEFFICIENT_FILE = os.path.join(ROOT, "config", "volume_gri_coefficients.csv")
CODES = {"A": 1, "B": 2}
KW = dict(species_codes=CODES, dbh_plausible_mm=(70, 1000), min_trees_per_species=5,
          standing_height_dm_above=100, breast_height_m=1.3)


def coefficients() -> pd.DataFrame:
    """Species 1: form height = 0.5 * h (only c11 and c12 set). Species 2: a full random matrix."""
    rng = np.random.default_rng(7)
    rows = []
    c1 = np.zeros((3, 3))
    c1[0, 0], c1[0, 1] = math.log(0.5), 1.0
    c2 = rng.normal(0, 0.05, (3, 3))
    c2[0, 0] = -0.3
    c2[0, 1] = 0.9
    for code, name, c in ((1, "A", c1), (2, "B", c2)):
        rows.append({"species_code": code, "species_name": name, **{f"c{i + 1}{j + 1}": c[i, j] for i in range(3) for j in range(3)}})
    return pd.DataFrame(rows)


def slow_gri(c, d_cm, h_m):
    """Independent loop implementation of the formula."""
    if math.isnan(d_cm) or math.isnan(h_m):
        return float("nan")
    if d_cm < 6.1 or h_m < 1.3:
        return 0.0
    left = [1.0, math.log(d_cm), math.log(d_cm) ** 2]
    right = [1.0, math.log(h_m), math.log(h_m) ** 2]
    s = sum(left[i] * c[i][j] * right[j] for i in range(3) for j in range(3))
    return max(0.0, math.pi / 40000 * d_cm ** 2 * math.exp(s))


def matrix(coef, code):
    row = coef[coef.species_code == code].iloc[0]
    return [[row[f"c{i}{j}"] for j in (1, 2, 3)] for i in (1, 2, 3)]


def stand(n=40, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.uniform(150, 600, n)
    h = 1.3 + (d / 10 / (1.5 + 0.2 * d / 10)) ** 2 * rng.uniform(0.95, 1.05, n)
    return pd.DataFrame({"tree_id": [f"1_{i}" for i in range(n)], "parzelle": 1, "species": ["A"] * 30 + ["B"] * (n - 30),
                         "dbh_field_mm": d, "dbh_uls_mm": d * rng.uniform(0.95, 1.05, n), "height_uls_m": h,
                         "height_standing_sheet": h * 0.8,            # field heights differ from the ULS heights
                         "detection": "detected", "identity_class": "confirmed", "removed_2026": False})


def run(trees, **over):
    return G.compute_gri_table(trees, coefficients=coefficients(), **{**KW, **over})


def entries(table, curves, **over):
    coef = coefficients()
    ev = lambda sp, d, h: G.gri_volume(G.species_to_code(pd.Series(sp), CODES, coef).to_numpy(), d, h, coef)
    kw = dict(evaluate=ev, areas_ha=pd.Series({1: 0.5}), class_edges_mm=[0, 300, 2000], min_trees=5, n_boot=200, seed=0)
    kw.update(over)
    return G.entries_from_table(table, curves, **kw)


def test_known_value():
    v = G.gri_volume([1], [400.0], [30.0], coefficients())
    assert math.isclose(v[0], math.pi / 40000 * 40.0 ** 2 * 0.5 * 30.0, rel_tol=1e-12)


def test_full_matrix_against_slow_loop():
    coef = coefficients()
    rng = np.random.default_rng(1)
    d_mm, h = rng.uniform(70, 900, 200), rng.uniform(3, 45, 200)
    codes = rng.choice([1, 2], 200)
    got = G.gri_volume(codes, d_mm, h, coef)
    expected = [slow_gri(matrix(coef, c), d / 10, hh) for c, d, hh in zip(codes, d_mm, h)]
    assert np.allclose(got, expected, rtol=1e-12)


def test_zero_and_nan_rules():
    coef = coefficients()
    v = G.gri_volume([1, 1, 1, 1, 1], [60.0, 61.0, 300.0, np.nan, 300.0], [20.0, 20.0, 1.2, 20.0, np.nan], coef)
    assert v[0] == 0.0                      # d = 6.0 cm < 6.1 cm
    assert v[1] > 0                         # d = 6.1 cm is the first diameter with a volume
    assert v[2] == 0.0                      # h < 1.3 m
    assert np.isnan(v[3]) and np.isnan(v[4])


def test_species_without_code_gives_nan_and_is_counted():
    t = stand()
    t.loc[0, "species"] = "Z"                                 # no code
    table, curves = run(t, species_codes={"A": 1})            # B has no code either
    assert table["gri_code"].iloc[0] != table["gri_code"].iloc[0]
    assert table.loc[table.species.isin(["Z", "B"]), "volume_gri_field_m3"].isna().all()
    assert table.loc[table.species == "A", "volume_gri_field_m3"].iloc[1:].notna().all()
    e = {x["id"]: x for x in entries(table, curves)}
    assert e["volume_gri.species_without_code"]["value"] == int((t.species != "A").sum())
    assert "Z" in e["volume_gri.species_without_code"]["statement"]
    code_without_coefficients = G.species_to_code(pd.Series(["A"]), {"A": 99}, coefficients())
    assert code_without_coefficients.isna().all()


def test_practice_volume_uses_field_height_curve_not_uls_height():
    table, curves = run(stand())
    assert not np.allclose(table.height_practice_m, table.height_used_m, atol=0.5)
    coef = coefficients()
    expected = G.gri_volume(table.gri_code, table.dbh_field_mm, table.height_practice_m, coef)
    assert np.allclose(table.volume_gri_practice_m3, expected)
    a = table[table.species == "A"]
    assert (a.volume_gri_practice_m3 < a.volume_gri_field_m3).mean() > 0.9     # field heights are 20 % lower
    # the practice height follows the field heights, so a change of the ULS height leaves it unchanged
    t2 = stand()
    t2["height_uls_m"] = t2["height_uls_m"] * 1.3
    table2, _ = run(t2)
    assert np.allclose(table.height_practice_m, table2.height_practice_m)
    assert np.allclose(table.volume_gri_practice_m3, table2.volume_gri_practice_m3)


def test_decimetre_correction_of_standing_height():
    s = pd.Series([31.6, 316.0, 100.0, 101.0, np.nan])
    out = G.standing_height_m(s, 100)
    assert np.allclose(out.to_numpy(), [31.6, 31.6, 100.0, 10.1, np.nan], equal_nan=True)
    t = stand()
    t.loc[0, "height_standing_sheet"] = 316.0
    table, _ = run(t)
    assert table.loc[0, "height_field_m"] == pytest.approx(31.6)


def test_modelled_height_for_unmatched_tree_and_uls_volume_empty():
    t = stand()
    t.loc[5, "height_uls_m"] = np.nan
    table, _ = run(t)
    r = table.loc[5]
    assert r.height_source == "modelled" and r.height_used_m > 1.3
    assert np.isnan(r.volume_gri_uls_m3) and r.volume_gri_field_m3 > 0
    t.loc[2, "dbh_field_mm"] = 5933.0
    assert np.isnan(run(t)[0].loc[2, "volume_gri_field_m3"])


def test_totals_equal_sums_and_gap_adds_up():
    t = stand()
    t.loc[3, "height_uls_m"] = np.nan
    table, curves = run(t)
    e = {x["id"]: x for x in entries(table, curves)}
    assert math.isclose(e["volume_gri.total.field_with_modelled.stand"]["value"], table.volume_gri_field_m3.sum())
    assert math.isclose(e["volume_gri.total.uls.stand"]["value"], table.volume_gri_uls_m3.sum())
    assert math.isclose(e["volume_gri.total.practice.stand"]["value"], table.volume_gri_practice_m3.sum())
    assert math.isclose(e["volume_gri.per_ha.practice.stand"]["value"], table.volume_gri_practice_m3.sum() / 0.5)
    assert math.isclose(e["volume_gri.total.field_without_modelled.stand"]["value"],
                        table.loc[table.height_source == "uls", "volume_gri_field_m3"].sum())
    assert math.isclose(sum(e[f"volume_gri.total.uls.{s}"]["value"] for s in ("a", "b")), e["volume_gri.total.uls.stand"]["value"])
    gap = sum(e[k]["value"] for k in ("volume_gri.gap.trees_missed", "volume_gri.gap.diameter_difference", "volume_gri.gap.uls_only_trees"))
    assert math.isclose(gap, e["volume_gri.gap.total"]["value"], abs_tol=1e-9)
    assert all(x["statement"].startswith("With the GRI volume functions") for x in e.values() if x["id"].startswith("volume_gri.total"))
    assert not any("provisional" in x["statement"] for x in e.values())
    assert all(set(x) == {"id", "value", "unit", "n", "ci_low", "ci_high", "subset", "statement"} for x in e.values())
    assert "volume_gri.practice_vs_field.rmse" in e and "volume_gri.height_curve_field.pooled.loo_rmse" in e


def test_missing_coefficient_file_or_codes_skips_cleanly(tmp_path, capsys):
    raw = copy.deepcopy(load_config(CONFIG).raw)
    raw["volume_gri"].update(params_file=str(tmp_path / "absent.csv"), species_codes={"A": 1})
    cfg = Config(raw)
    assert G.run_volume_gri(cfg) is None
    assert "absent.csv" in capsys.readouterr().out
    assert G.compute_entries(cfg) == []
    coef = tmp_path / "coef.csv"
    coefficients().to_csv(coef, index=False)
    raw["volume_gri"].update(params_file=str(coef), species_codes={})
    cfg = Config(raw)
    assert G.run_volume_gri(cfg) is None
    assert "species_codes" in capsys.readouterr().out
    assert G.compute_entries(cfg) == []


def test_against_r_reference():
    if not os.path.isfile(REFERENCE) or not os.path.isfile(COEFFICIENT_FILE):
        pytest.skip("reference values or coefficient file not present")
    ref = pd.read_csv(REFERENCE)
    got = G.gri_volume(ref["species_code"], ref["dbh_cm"] * 10.0, ref["height_m"], G.load_coefficients(COEFFICIENT_FILE))
    assert np.allclose(got, ref["v_gri_m3"], rtol=1e-9, atol=0)
