"""Tests for core.increment and the harvest volume of core.volume_gri (ports checked against ForestElementsR values)."""
import copy
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core import increment as I  # noqa: E402
from core import volume_gri as G  # noqa: E402
from core.config import Config, load_config  # noqa: E402

CONFIG = os.path.join(ROOT, "config", "site_155.yaml")
DATA = os.path.join(ROOT, "tests", "data")
YIELD_TABLES = os.path.join(ROOT, "config", "fer_yield_tables.json")
GROWTH = os.path.join(ROOT, "config", "fer_gnfi3_parameters.csv")
REDUCTION = os.path.join(ROOT, "config", "fer_harvest_reduction.csv")


def need(*paths):
    if not all(os.path.isfile(p) for p in paths):
        pytest.skip("exported ForestElementsR data or reference values not present")


def same(got, expected, rtol=1e-8):
    got, expected = np.asarray(got, dtype=float), np.asarray(expected, dtype=float)
    assert np.array_equal(np.isnan(got), np.isnan(expected))
    ok = ~np.isnan(expected)
    assert np.allclose(got[ok], expected[ok], rtol=rtol, atol=1e-12)


# ---------------------------------------------------------------------------
# Ports against the reference values of ForestElementsR
# ---------------------------------------------------------------------------

def test_yield_table_functions_against_reference():
    ref_path = os.path.join(DATA, "fer_ytable_reference.csv")
    need(ref_path, YIELD_TABLES)
    tables = I.load_yield_tables(YIELD_TABLES)
    ref = pd.read_csv(ref_path)
    got, flags_age, flags_si = [], [], []
    for r in ref.itertuples():
        t = tables[r.species_code]
        if r.fn == "site_index":
            v = I.site_index(t, r.age, r.x)
            si_flag = I.si_outside_table(t, v) if np.isfinite(v) else False
        else:
            v = {"lookup_pai": lambda: I.ytable_lookup(t, r.age, r.x, "pai_m3_ha_yr"),
                 "lookup_mai": lambda: I.ytable_lookup(t, r.age, r.x, "mai_m3_ha_yr"),
                 "lookup_ba": lambda: I.ytable_lookup(t, r.age, r.x, "ba_m2_ha"),
                 "stocking_level": lambda: I.stocking_level(t, r.ba_m2_ha, r.age, r.x)}[r.fn]()
            si_flag = I.si_outside_table(t, r.x)
        got.append(v)
        flags_age.append(I.age_outside_table(t, r.age))
        flags_si.append(si_flag)
    same(got, ref["result"])
    assert list(map(int, flags_age)) == ref["age_clamped"].tolist()
    assert list(map(int, flags_si)) == ref["si_extrapolated"].tolist()
    assert ref["age_clamped"].sum() > 0 and ref["si_extrapolated"].sum() > 0       # the grid does leave the tables
    assert (ref["age_clamped"] == 0).sum() > 0 and (ref["si_extrapolated"] == 0).sum() > 0


def test_growth_functions_against_reference():
    ref_path = os.path.join(DATA, "fer_gnfi3_reference.csv")
    need(ref_path, GROWTH)
    params = I.load_growth_parameters(GROWTH)
    ref = pd.read_csv(ref_path)
    for fn, f in (("d", I.d_age_gnfi3), ("h", I.h_age_gnfi3)):
        r = ref[ref["fn"] == fn]
        assert len(r) > 100
        same(f(r["species_code"], r["age"], r["size_known"], r["age_known"], params), r["result"])


def test_harvest_reduction_against_reference():
    ref_path = os.path.join(DATA, "fer_efm_reference.csv")
    need(ref_path, REDUCTION)
    factors = G.load_harvest_factors(REDUCTION)
    ref = pd.read_csv(ref_path)
    same(G.harvest_volume(ref["species_code"], ref["v_gri_m3"], factors), ref["efm_m3"])


# ---------------------------------------------------------------------------
# Rules of the ports on a small constructed table
# ---------------------------------------------------------------------------

def small_table() -> I.YieldTable:
    ages = np.array([20.0, 40.0, 60.0, 80.0])
    si = np.array([1.0, 2.0, 3.0])
    h = np.array([[8, 7, np.nan], [16, 14, 12], [22, 19, 16], [26, 22, 18]], dtype=float)
    ba = np.array([[10, 8, np.nan], [24, 20, 16], [34, 28, 22], [40, 33, 26]], dtype=float)
    pai = np.array([[12, 9, np.nan], [14, 11, 8], [12, 9, 6], [10, 7, 4]], dtype=float)
    return I.YieldTable(1, "synthetic", si, ages, {"h_q_m": h, "ba_m2_ha": ba, "pai_m3_ha_yr": pai, "mai_m3_ha_yr": pai * 0.7})


def test_interpolation_extrapolation_and_clamping():
    t = small_table()
    assert I.ytable_lookup(t, 50, 1.5, "ba_m2_ha") == pytest.approx((34 + 24) / 2 * 0.5 + (28 + 20) / 2 * 0.5)
    assert I.ytable_lookup(t, 40, 2.0, "pai_m3_ha_yr") == pytest.approx(11.0)
    assert I.ytable_lookup(t, 200, 2.0, "pai_m3_ha_yr") == I.ytable_lookup(t, 80, 2.0, "pai_m3_ha_yr")     # age is clamped
    assert I.age_outside_table(t, 200) and not I.age_outside_table(t, 60)
    # site index 0.5 lies outside: linear extrapolation through the two nearest site indexes (1 and 2)
    assert I.ytable_lookup(t, 40, 0.5, "pai_m3_ha_yr") == pytest.approx(14 + 0.5 * 3)
    assert I.si_outside_table(t, 0.5) and not I.si_outside_table(t, 3.0)
    # inverse lookup: h_q = 17.5 at age 40 lies halfway between SI 2 (14) and SI 1 (16) ... nearest two are 16 and 14
    assert I.site_index(t, 40, 15.0) == pytest.approx(1.5)
    assert I.site_index(t, 40, 20.0) == pytest.approx(1.0 - 4 / 2)                                         # extrapolated
    assert I.stocking_level(t, 20.0, 40, 2.0) == pytest.approx(1.0)
    assert I.stocking_level(t, -5.0, 40, 2.0) == 0.0
    with pytest.raises(ValueError):
        I.site_index(t, 40, 15.0, "d_q_cm")


def test_growth_function_keeps_known_size_at_known_age():
    params = pd.DataFrame({"d_alpha": [0.1], "d_beta": [0.05], "d_gamma": [500.0], "h_alpha": [-0.03], "h_beta": [0.6],
                           "h_gamma": [85.0]}, index=pd.Index([1], name="species_code"))
    assert I.d_age_gnfi3([1], 60, 30.0, 60, params)[0] == pytest.approx(30.0)
    assert I.h_age_gnfi3([1], 70, 25.0, 60, params)[0] > 25.0
    assert np.isnan(I.h_age_gnfi3([99], 70, 25.0, 60, params)[0]) and np.isnan(I.d_age_gnfi3([np.nan], 70, 25.0, 60, params)[0])


# ---------------------------------------------------------------------------
# Efm = volume x factor
# ---------------------------------------------------------------------------

def coefficients() -> pd.DataFrame:
    row = {f"c{i}{j}": 0.0 for i in (1, 2, 3) for j in (1, 2, 3)}
    row.update(species_code=1, species_name="A", c11=math.log(0.5), c12=1.0)       # form height = 0.5 h
    return pd.DataFrame([row])


def test_efm_is_volume_times_species_factor():
    factors = pd.Series({1: 0.81, 3: 0.79})
    v = G.harvest_volume([1, 3, 7, np.nan, 1], [2.0, 1.0, 5.0, 3.0, np.nan], factors)
    assert v[0] == pytest.approx(1.62) and v[1] == pytest.approx(0.79)
    assert np.isnan(v[2]) and np.isnan(v[3]) and np.isnan(v[4])
    rng = np.random.default_rng(0)
    d = rng.uniform(150, 500, 20)
    t = pd.DataFrame({"tree_id": range(20), "parzelle": 1, "species": "A", "dbh_field_mm": d, "dbh_uls_mm": d,
                      "height_uls_m": 1.3 + (d / 10 / (1.5 + 0.2 * d / 10)) ** 2, "height_standing_sheet": 20.0})
    kw = dict(coefficients=coefficients(), species_codes={"A": 1}, dbh_plausible_mm=(70, 1000), min_trees_per_species=5,
              standing_height_dm_above=100)
    table, _ = G.compute_gri_table(t, harvest_factors=factors, **kw)
    for k in ("field", "uls", "practice"):
        assert np.allclose(table[f"efm_gri_{k}_m3"], table[f"volume_gri_{k}_m3"] * 0.81)
    plain, _ = G.compute_gri_table(t, **kw)
    assert not any(c.startswith("efm_") for c in plain.columns)                      # no factors, no Efm columns
    assert np.allclose(plain["volume_gri_field_m3"], table["volume_gri_field_m3"])    # the standing volumes are unchanged
    s = G.summary_per_parzelle(table, pd.Series({1: 0.5}))
    assert s["efm_gri_field_sum_m3"].iloc[0] == pytest.approx(table["volume_gri_field_m3"].sum() * 0.81)
    assert s["efm_gri_field_sum_m3_per_ha"].iloc[0] == pytest.approx(table["volume_gri_field_m3"].sum() * 0.81 / 0.5)
    e = {x["id"]: x for x in G.entries_from_table(table, _, evaluate=lambda sp, d_, h_: np.zeros(len(h_)), areas_ha=pd.Series({1: 0.5}),
                                                  class_edges_mm=[0, 2000], min_trees=5, n_boot=100, seed=0)}
    assert e["volume_gri.efm.total.field_with_modelled.stand"]["value"] == pytest.approx(table["efm_gri_field_m3"].sum())
    assert e["volume_gri.efm.per_ha.practice.stand"]["value"] == pytest.approx(table["efm_gri_practice_m3"].sum() / 0.5)
    assert "(Efm o. R.)" in e["volume_gri.efm.total.uls.stand"]["statement"]
    assert "v_red_harvest_ubark" in e["volume_gri.efm.total.uls.stand"]["statement"]


# ---------------------------------------------------------------------------
# Measured increment (periodic increment of two surveys)
# ---------------------------------------------------------------------------

CURVES = pd.DataFrame([{"scope": "pooled", "species": "", "form": "naeslund_cm", "n": 50, "a": 1.0, "b": 0.2}])
CURVES_2020 = pd.DataFrame([{"scope": "pooled", "species": "", "form": "naeslund_cm", "n": 50, "a": 2.0, "b": 0.2}])
SEASONS = {2025: 5.0, 2026: 5.0}
AREAS = pd.Series({1: 0.1, 2: 0.2})


def curve_h(d_mm, a=1.0):
    d = d_mm / 10.0
    return 1.3 + (d / (a + 0.2 * d)) ** 2


def vol(d_mm, a=1.0):
    return math.pi / 40000 * (d_mm / 10.0) ** 2 * 0.5 * curve_h(d_mm, a)


def survey_trees():
    """Parzelle 1 (survey 2025): two survivors, one tree removed earlier, one missing, one with an absurd diameter
    change. Parzelle 2 (survey 2026): one survivor."""
    all_2020 = pd.DataFrame({
        "tree_id": ["1_1", "1_2", "1_3", "1_4", "1_5", "2_1"], "parzelle": ["1", "1", "1", "1", "1", "2"], "species": "A",
        "dbh_2020_mm": [280.0, 300.0, 250.0, 260.0, 200.0, 340.0], "height_2020_m": np.nan,
        "removed_earlier": [False, False, True, False, False, False]})
    now = pd.DataFrame({"tree_id": ["1_1", "1_2", "1_5", "2_1"], "dbh_field_mm": [300.0, 330.0, 400.0, 350.0]})
    return all_2020, now


def per_tree_table(all_2020=None, now=None) -> pd.DataFrame:
    if all_2020 is None:
        all_2020, now = survey_trees()
    return I.tree_table(all_2020, now, CURVES_2020, CURVES, coefficients(), {"A": 1}, dbh_plausible_mm=(70, 1000),
                        dbh_change_plausible_mm=(-10, 150), survey_years={1: 2025, 2: 2026}, breast=1.3)


def test_tree_table_status_and_volumes():
    t = per_tree_table().set_index("tree_id")
    assert t["status"].to_dict() == {"1_1": "survivor", "1_2": "survivor", "1_3": "removed_earlier", "1_4": "missing",
                                     "1_5": "excluded_dbh_change", "2_1": "survivor"}
    assert t.loc["1_1", "v_2020_m3"] == pytest.approx(vol(280, a=2.0))           # the 2020 curve at the 2020 diameter
    assert t.loc["1_1", "v_now_m3"] == pytest.approx(vol(300, a=1.0))            # the current curve at the current diameter
    assert t.loc["1_1", "dbh_change_mm"] == 20 and t.loc["1_1", "survey_year"] == 2025 and t.loc["2_1", "survey_year"] == 2026
    assert t.loc["1_3", "v_2020_m3"] == pytest.approx(vol(250, a=2.0)) and np.isnan(t.loc["1_3", "v_now_m3"])
    assert np.isnan(t.loc["1_4", "v_now_m3"]) and t.loc["1_4", "v_2020_m3"] > 0
    assert t.loc["1_5", "dbh_change_mm"] == 200                                   # reported, but excluded from the ledger
    # an implausible diameter at either date, and a tree without a 2020 diameter, are classified before the change test
    all_2020, now = survey_trees()
    all_2020.loc[0, "dbh_2020_mm"] = 5.0
    all_2020.loc[1, "dbh_2020_mm"] = np.nan
    now.loc[now.tree_id == "1_5", "dbh_field_mm"] = 5933.0
    t = per_tree_table(all_2020, now).set_index("tree_id")
    assert t.loc["1_1", "status"] == "excluded_implausible_dbh" and t.loc["1_2", "status"] == "no_2020_dbh"
    assert t.loc["1_5", "status"] == "excluded_implausible_dbh"


@pytest.mark.parametrize("share", [0.0, 0.5])
def test_ledger_identity(share):
    t = per_tree_table()
    s = I.ledger_table(t, AREAS, SEASONS, share).set_index("parzelle")
    r = s.loc[1]
    surv_old, surv_now = vol(280, 2.0) + vol(300, 2.0), vol(300) + vol(330)
    removed_2020 = vol(250, 2.0) + vol(260, 2.0)
    rel = (surv_now - surv_old) / surv_old
    assert (r.n_survivors, r.n_removed_earlier, r.n_missing, r.n_excluded) == (2, 1, 1, 1)
    assert r.v_alive_t1_m3 == pytest.approx(surv_old + removed_2020)              # the excluded tree is in neither sum
    assert r.v_alive_t2_m3 == pytest.approx(surv_now)
    assert r.v_removed_2020_m3 == pytest.approx(removed_2020)
    assert r.v_removed_m3 == pytest.approx(removed_2020 * (1 + share * rel))
    assert r.survivor_rel_growth == pytest.approx(rel)
    assert r.increment_m3 == pytest.approx(r.v_alive_t2_m3 + r.v_removed_m3 - r.v_alive_t1_m3)
    assert r.increment_m3 == pytest.approx(surv_now - surv_old + share * removed_2020 * rel)
    assert r.pai_measured_m3_ha_yr == pytest.approx(r.increment_m3 / 5 / 0.1)
    assert r.dbh_increment_mm_yr == pytest.approx((20 + 30) / 2 / 5)
    st = s.loc["stand"]
    assert st.increment_m3 == pytest.approx(s.loc[1, "increment_m3"] + s.loc[2, "increment_m3"])
    assert st.area_ha == pytest.approx(0.3) and st.n_survivors == 3
    assert st.pai_measured_m3_ha_yr == pytest.approx(st.increment_m3 / 5 / 0.3)


def test_pai_can_be_recomputed_from_the_table_columns():
    t = per_tree_table()
    for share, seasons in ((0.0, SEASONS), (0.5, SEASONS), (1.0, {2025: 6.0, 2026: 6.0})):
        s = I.ledger_table(t, AREAS, seasons, share)
        for _, r in s.iterrows():
            rel = 0.0 if not np.isfinite(r.survivor_rel_growth) else r.survivor_rel_growth
            again = (r.survivor_growth_m3 + share * r.v_removed_2020_m3 * rel) / r.growing_seasons / r.area_ha
            assert r.pai_measured_m3_ha_yr == pytest.approx(again)
    # the table of one setting carries the pai of another: share 0.5 from the columns of the share 0 table
    base = I.ledger_table(t, AREAS, SEASONS, 0.0).set_index("parzelle")
    other = I.ledger_table(t, AREAS, SEASONS, 0.5).set_index("parzelle")
    for p in (1, 2, "stand"):
        r = base.loc[p]
        assert other.loc[p, "pai_measured_m3_ha_yr"] == pytest.approx(
            (r.survivor_growth_m3 + 0.5 * r.v_removed_2020_m3 * r.survivor_rel_growth) / r.growing_seasons / r.area_ha)


def test_per_survey_curves_give_a_larger_increment_than_one_static_curve():
    t = per_tree_table()
    s = I.ledger_table(t, AREAS, SEASONS, 0.0).set_index("parzelle")
    assert (s["pai_measured_m3_ha_yr"] > s["pai_static_curve_m3_ha_yr"]).all()    # 2020 heights lower at equal diameter
    surv = t[t.status == "survivor"]
    static = (surv["v_now_m3"] - surv["v_static_2020_m3"]).sum() / 5 / 0.3
    assert s.loc["stand", "pai_static_curve_m3_ha_yr"] == pytest.approx(static)
    assert t.loc[t.tree_id == "1_1", "v_static_2020_m3"].iloc[0] == pytest.approx(vol(280, a=1.0))
    same = I.tree_table(*survey_trees(), CURVES, CURVES, coefficients(), {"A": 1}, dbh_plausible_mm=(70, 1000),
                        dbh_change_plausible_mm=(-10, 150), survey_years={1: 2025, 2: 2026}, breast=1.3)
    e = I.ledger_table(same, AREAS, SEASONS, 0.0)
    assert np.allclose(e["pai_measured_m3_ha_yr"], e["pai_static_curve_m3_ha_yr"])  # identical curves: no difference


def test_growing_seasons_are_applied_per_survey_year():
    t = per_tree_table()
    s = I.ledger_table(t, AREAS, {2025: 5.0, 2026: 6.0}, 0.0).set_index("parzelle")
    assert s.loc[1, "growing_seasons"] == 5 and s.loc[2, "growing_seasons"] == 6 and np.isnan(s.loc["stand", "growing_seasons"])
    assert s.loc[2, "pai_measured_m3_ha_yr"] == pytest.approx(s.loc[2, "increment_m3"] / 6 / 0.2)
    assert s.loc["stand", "pai_measured_m3_ha_yr"] == pytest.approx(
        (s.loc[1, "increment_m3"] / 5 + s.loc[2, "increment_m3"] / 6) / 0.3)
    assert s.loc[2, "dbh_increment_mm_yr"] == pytest.approx(10 / 6)
    assert I.bootstrap_pai(t, AREAS, {2025: 5.0, 2026: 6.0}, 0.5, 200, 0)[0] > 0


def test_height_2020_dm_rule_and_bounds():
    h = pd.Series([20.4, 104.0, 1.0, 70.0, np.nan])
    got = I.clean_height_2020(h, 100, (2.0, 60.0))
    assert got.tolist()[:2] == [20.4, 10.4] and got.isna().tolist() == [False, False, True, True, True]
    rng = np.random.default_rng(1)
    d = rng.uniform(200, 500, 40)
    a = pd.DataFrame({"species": "A", "dbh_2020_mm": d, "height_2020_m": curve_h(d, 2.0)})
    a.loc[0, "height_2020_m"] = 90.0                                              # not a decimetre value, above 60 m
    curves, n_dropped = I.curves_2020(a, dbh_plausible_mm=(70, 1000), dm_above=100, bounds_m=(2.0, 60.0), min_trees=100, breast=1.3)
    assert n_dropped == 1
    assert curves.loc[0, "n"] == 39 and curves.loc[0, "a"] == pytest.approx(2.0, rel=0.02)


def test_2020_survey_keeps_trees_of_earlier_thinnings():
    from core import io_field
    sheet = pd.DataFrame({"Parzelle": [1, 1, 2, np.nan], "Nr": [1, 2, 1, 9], "art": ["A", "A", "B", "B"], "d20": [280, 250, 300, 1],
                          "h20": [24.0, 22.0, np.nan, 5.0], "removal": [np.nan, 1, np.nan, np.nan]})
    cols = {"plot": "Parzelle", "nr": "Nr", "species": "art", "dbh_2020_mm": "d20", "height_2020_m": "h20"}
    out = io_field.survey_2020_frame(sheet, cols, {}, "removal")
    assert list(out.columns) == io_field.SURVEY_2020_COLUMNS
    assert out["tree_id"].tolist() == ["1_1", "1_2", "2_1"]                       # the row without a plot is dropped
    assert out["removed_earlier"].tolist() == [False, True, False]
    assert out["dbh_2020_mm"].tolist() == [280, 250, 300] and out["height_2020_m"].isna().tolist() == [False, False, True]
    assert not io_field.survey_2020_frame(sheet, cols)["removed_earlier"].any()   # no flag column: nobody is flagged


def test_measured_entries_state_assumptions():
    t = per_tree_table()
    s = I.ledger_table(t, AREAS, SEASONS, 0.5)
    e = {x["id"]: x for x in I.measured_entries(t, s, CURVES_2020.assign(loo_rmse_m=1.5), areas_ha=AREAS, seasons=SEASONS, share=0.5,
                                                n_dropped_heights=2, n_boot=200, seed=0)}
    pai = e["increment.measured.stand.pai"]
    assert pai["value"] == pytest.approx(s.loc[s.parzelle == "stand", "pai_measured_m3_ha_yr"].iloc[0])
    assert pai["ci_low"] is not None and pai["ci_low"] < pai["value"] < pai["ci_high"]
    for text in ("V_alive(t2) + V_removed - V_alive(t1)", "5 growing seasons between 2020 and 2025", "50 %",
                 "neither a removal year nor a diameter at removal is recorded", "no ingrowth is recorded"):
        assert text in pai["statement"]
    assert e["increment.measured.trees"]["value"] == 3
    assert e["increment.measured.removed_earlier"]["value"] == 1
    assert e["increment.measured.excluded"]["value"] == 1 and "1_5" in e["increment.measured.excluded"]["statement"]
    for k in ("pai_static_curve", "dbh", "v_alive_t1_per_ha", "v_alive_t2_per_ha", "v_removed_per_ha"):
        assert f"increment.measured.stand.{k}" in e
    assert e["increment.measured.stand.v_alive_t1_per_ha"]["value"] == pytest.approx(s.loc[s.parzelle == "stand", "v_alive_t1_m3"].iloc[0] / 0.3)


# ---------------------------------------------------------------------------
# Age-dependent methods and the stand row
# ---------------------------------------------------------------------------

def stand_inputs():
    rng = np.random.default_rng(3)
    n = 30
    d = rng.uniform(200, 500, n)
    h = curve_h(d)
    trees = pd.DataFrame({"tree_id": range(n), "parzelle": [1] * 10 + [2] * 20, "species": "A", "gri_code": 1.0,
                          "dbh_field_mm": d, "dbh_field_valid": True, "height_uls_m": h, "height_used_m": h})
    trees["volume_gri_field_m3"] = G.gri_volume(trees["gri_code"], d, h, coefficients())
    areas = pd.Series({1: 0.1, 2: 0.3})
    structure = I.stand_structure(trees, areas, CURVES, 1.3)
    growth = pd.DataFrame({"d_alpha": [0.1], "d_beta": [0.05], "d_gamma": [500.0], "h_alpha": [-0.03], "h_beta": [0.6],
                           "h_gamma": [85.0]}, index=pd.Index([1], name="species_code"))
    kw = dict(tables={1: small_table()}, coefficients=coefficients(), growth=growth, horizon=5.0, factors=pd.Series({1: 0.8}))
    return trees, structure, kw


def test_stand_structure():
    trees, structure, _ = stand_inputs()
    p1 = structure.loc[1]
    d = trees.loc[trees.parzelle == 1, "dbh_field_mm"]
    assert p1.main_species == "A" and p1.gri_code == 1
    assert p1.d_q_cm == pytest.approx(math.sqrt((d ** 2).mean()) / 10)
    assert p1.ba_m2_ha == pytest.approx((math.pi / 4 * (d / 1000) ** 2).sum() / 0.1)
    assert p1.h_q_m == pytest.approx(curve_h(p1.d_q_cm * 10))
    assert structure.loc["stand", "area_ha"] == pytest.approx(0.4)
    assert structure.loc["stand", "ba_m2_ha"] == pytest.approx(
        (structure.loc[1, "ba_m2_ha"] * 0.1 + structure.loc[2, "ba_m2_ha"] * 0.3) / 0.4)


def test_stand_row_is_area_weighted_and_tree_growth_sums_per_ha():
    trees, structure, kw = stand_inputs()
    out = I.by_age_table(structure, trees, [40, 60], **kw)
    assert list(out.columns) == I.BY_AGE_COLUMNS and len(out) == 6
    for age in (40, 60):
        g = out[out.age == age].set_index("parzelle")
        for col in ("pai_yield_table_m3_ha_yr", "pai_tree_growth_m3_ha_yr", "site_index", "stocking_level", "pai_table_m3_ha_yr"):
            assert g.loc["stand", col] == pytest.approx((g.loc[1, col] * 0.1 + g.loc[2, col] * 0.3) / 0.4)
        assert g.loc["stand", "n_trees_tree_growth"] == 30
        assert g.loc[1, "yield_table"] == "synthetic"
        assert g.loc[1, "pai_yield_table_m3_ha_yr"] == pytest.approx(g.loc[1, "pai_table_m3_ha_yr"] * min(g.loc[1, "stocking_level"], 1.0))
        assert g.loc[1, "pai_tree_growth_efm_m3_ha_yr"] == pytest.approx(g.loc[1, "pai_tree_growth_m3_ha_yr"] * 0.8)
    # the tree growth increment of a Parzelle is the summed per-tree volume change over the horizon, per ha
    inc = I.tree_growth_increment(trees, 40, 5.0, coefficients(), kw["growth"])
    g40 = out[(out.age == 40) & (out.parzelle == 2)].iloc[0]
    assert g40.pai_tree_growth_m3_ha_yr == pytest.approx(inc[trees.parzelle == 2].sum() / 0.3)
    assert (inc > 0).all()


def test_parzelle_age_override_changes_only_that_parzelle():
    trees, structure, kw = stand_inputs()
    same_age = I.evaluate_age(structure, trees, lambda p: 40.0, **kw).set_index("parzelle")
    mixed = I.evaluate_age(structure, trees, lambda p: 60.0 if p == 1 else 40.0, **kw).set_index("parzelle")
    assert mixed.loc[2, "pai_yield_table_m3_ha_yr"] == pytest.approx(same_age.loc[2, "pai_yield_table_m3_ha_yr"])
    assert mixed.loc[1, "age"] == 60 and np.isnan(mixed.loc["stand", "age"]) and same_age.loc["stand", "age"] == 40


def test_missing_exported_files_or_tables_skip_cleanly(tmp_path, capsys):
    raw = copy.deepcopy(load_config(CONFIG).raw)
    raw.setdefault("increment", {})["params_dir"] = str(tmp_path)
    cfg = Config(raw)
    assert I.run_increment(cfg) is None
    out = capsys.readouterr().out
    assert out.count("\n") == 1 and "fer_yield_tables.json" in out
    assert I.compute_entries(cfg) == []
    assert I.load_parameters(tmp_path)[0] is None
