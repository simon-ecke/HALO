"""Synthetic tests for core.analyses (no project data): error budget, sampling value, stand description, ledger."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import analyses as A  # noqa: E402


# --- 1. error budget -------------------------------------------------------

def test_error_budget_known_values():
    tape = pd.Series([20.0, 22.0, 24.0, 26.0, 28.0, 30.0])
    vertex = tape + [0.3, -0.3, 0.3, -0.3, 0.3, -0.3]          # RMS of the Vertex-tape differences = 0.3
    diff_uls = [0.2, -0.2, 0.2, 2.0, 0.2, -0.2]
    eb = A.error_budget(tape, vertex, tape + diff_uls, noise_band_sd=1.96, n_boot=200, seed=0)
    assert eb['n'] == 6
    assert np.isclose(eb['rmse_vertex'], 0.3)
    assert np.isclose(eb['band'], 1.96 * 0.3)
    assert eb['n_inside'] == 5 and np.isclose(eb['share_inside'], 5 / 6)       # only the 2.0 m difference is outside
    assert np.isclose(eb['rmse_uls'], np.sqrt(np.mean(np.square(diff_uls))))
    assert np.isclose(eb['rmse_diff'], eb['rmse_uls'] - eb['rmse_vertex'])


def test_error_budget_uses_only_trees_with_all_three_and_is_seeded():
    tape = pd.Series([10, 12, 14, 16, np.nan, 20.0])
    vertex = pd.Series([10.1, 12.1, np.nan, 16.1, 18.1, 20.1])
    uls = pd.Series([10.2, 12.2, 14.2, 16.2, 18.2, 20.2])
    a = A.error_budget(tape, vertex, uls, n_boot=100, seed=0)
    b = A.error_budget(tape, vertex, uls, n_boot=100, seed=0)
    assert a['n'] == 4 and a['n_tape'] == 5
    assert a['rmse_diff_ci'] == b['rmse_diff_ci']


def test_error_budget_verdict_never_claims_when_interval_includes_zero():
    rng = np.random.default_rng(3)
    tape = pd.Series(rng.uniform(20, 30, 14))
    vertex = tape + rng.normal(0, 0.4, 14)
    uls = tape + rng.normal(0, 0.4, 14)
    eb = A.error_budget(tape, vertex, uls, n_boot=500, seed=0)
    entries = A.error_budget_entries(eb)
    verdict = [e for e in entries if e['id'] == 'error_budget.verdict'][0]
    assert verdict['n'] == 14 and '14 trees' in verdict['statement']
    if eb['rmse_diff_ci'][0] <= 0 <= eb['rmse_diff_ci'][1]:
        assert 'cannot be told apart' in verdict['statement']
    assert all(e['n'] == 14 for e in entries)


# --- 3. sampling value -----------------------------------------------------

def test_sampling_value_spread_shrinks_and_matches_theory():
    h = pd.Series(np.random.default_rng(0).normal(29, 3, 1500))
    t = A.sampling_value(h, [5, 20, 100], draws=2000, seed=0)
    assert list(t['k']) == [5, 20, 100]
    assert t['estimate_sd_m'].is_monotonic_decreasing
    assert np.isclose(t['census_mean_m'].iloc[0], h.mean())
    expected = h.std(ddof=1) / np.sqrt(20) * np.sqrt(1 - 20 / 1500)
    assert abs(t.loc[t['k'] == 20, 'estimate_sd_m'].iloc[0] - expected) < 0.05
    assert t.loc[t['k'] == 100, 'share_within_1_0m'].iloc[0] > 0.99
    again = A.sampling_value(h, [5, 20, 100], draws=2000, seed=0)
    pd.testing.assert_frame_equal(t, again)


def test_sampling_value_k_above_census_uses_all_trees():
    t = A.sampling_value(pd.Series([20.0, 25.0, 30.0]), [10], draws=50, seed=0)
    assert t['k_used'].iloc[0] == 3 and t['estimate_sd_m'].iloc[0] == 0


# --- 4. stand description --------------------------------------------------

def _stand_frame():
    rows = []
    for plot, base in [('a', 20.0), ('b', 30.0)]:
        for i in range(30):
            rows.append({'parzelle': plot, 'species': 'x' if i % 2 else 'y', 'detection': 'detected' if i < 27 else 'not detected',
                         'lidar_height': base + i * 0.1 if i < 27 else np.nan, 'dbh_field_mm': 300 + 10 * i,
                         'dbh_lidar_mm': 305 + 10 * i, 'dbh_quality': 'ok'})
    return pd.DataFrame(rows)


def test_top_height_scales_to_plot_area():
    h = pd.Series(np.arange(1, 41, dtype=float))
    d = pd.Series(np.arange(1, 41, dtype=float))                 # thickest = tallest here
    top, n = A.top_height(h, d, 2500.0, trees_per_ha=100)       # 0.25 ha -> 25 trees
    assert n == 25 and np.isclose(top, np.mean(np.arange(16, 41)))
    top1, n1 = A.top_height(h, d, 50.0, trees_per_ha=100)       # tiny plot: at least one tree
    assert n1 == 1 and top1 == 40.0


def test_stand_per_parzelle():
    plots, species = A.stand_per_parzelle(_stand_frame(), pd.Series({'a': 1500.0, 'b': 1500.0}), trees_per_ha=100)
    p = plots.set_index('parzelle')
    assert p.loc['a', 'n_field_trees'] == 30 and p.loc['a', 'n_matched'] == 27
    assert np.isclose(p.loc['a', 'detection_rate'], 27 / 30)
    assert np.isclose(p.loc['a', 'dbh_bias_uls_minus_field_cm'], 0.5)
    assert p.loc['a', 'n_top_trees'] == 15 and p.loc['b', 'top_height_m'] > p.loc['a', 'top_height_m']
    assert np.isclose(p.loc['b', 'mean_height_m'], 30 + 0.1 * np.arange(27).mean())
    assert set(species['species']) == {'x', 'y'} and species['n'].sum() == 54


# --- 6. ledger -------------------------------------------------------------

def _ledger_frame():
    base = dict(parzelle=1, species='x', detection='detected', has_height=False, removed_2026=True, removal_consistent=True,
                removed_fraction=0.95, dbh_mm=400.0, remark=np.nan, height_massband=np.nan, height_vertex=np.nan, lidar_height=28.0,
                identity_class='confirmed', reasons='', cause='')
    rows = [dict(base, tree_id='t_ok'),
            dict(base, tree_id='t_nodbh', dbh_mm=np.nan),
            dict(base, tree_id='t_bigdbh', dbh_mm=5933.0),
            dict(base, tree_id='t_miss', detection='not detected', identity_class='unmatched', lidar_height=np.nan, cause='merged'),
            dict(base, tree_id='t_amb', identity_class='ambiguous', reasons='competitor as near'),
            dict(base, tree_id='t_rec', removal_consistent=False, removed_fraction=0.0),
            dict(base, tree_id='t_lying', has_height=True, height_massband=31.0, removed_fraction=0.0, lidar_height=29.0,
                 remark='2026; check', removal_consistent=False),
            dict(base, tree_id='t_routine', has_height=True, height_massband=28.0, remark='BAD!; 2026')]
    return pd.DataFrame(rows)


def test_ledger_finds_issues_from_the_data():
    led = A.ledger(_ledger_frame(), dbh_plausible_mm=[70, 1000], gone_above=0.8, standing_below=0.2,
                   routine_remark_tokens=['BAD', '2026', '!', ';'], height_diff_m=1.0, mirrored_plots=[7])
    got = {(r.tree_id, r.issue) for r in led.itertuples()}
    assert ('t_nodbh', 'dbh_missing') in got and ('t_bigdbh', 'dbh_implausible') in got
    assert ('t_miss', 'unmatched') in got and ('t_amb', 'identity_ambiguous') in got
    assert ('t_rec', 'harvest_record_disagrees') in got
    assert ('t_lying', 'sample_tree_stands_after_harvest') in got       # found through the data, no id in the code
    assert ('t_lying', 'sample_tree_remark') in got and ('t_lying', 'sample_tree_height_differs') in got
    assert not any(r.tree_id in ('t_ok', 't_routine') for r in led.itertuples())
    assert ('', 'plot_y_mirrored') in got
    assert list(led.columns) == ['tree_id', 'parzelle', 'issue', 'detail']


def test_routine_remarks():
    tok = ['BAD', '2026', '!', ';']
    assert A.is_routine_remark(np.nan, tok) and A.is_routine_remark('2026; BAD!', tok)
    assert not A.is_routine_remark('2026; Stumpfhöhe bereits addiert', tok)


def test_explanation_only_above_threshold_and_uses_cause_and_remark():
    r = pd.Series({'height_massband': 31.0, 'height_vertex': np.nan, 'height_uls': 28.0, 'cause': 'tree found, harvest record disagrees',
                   'identity_class': 'confirmed', 'tier_method': 'sat_instance', 'remark': 'Krone abgebrochen'})
    text = A.explain_difference(r, height_diff_m=1.0, routine_remark_tokens=['2026'])
    assert 'below the Maßband' in text and 'harvest record disagrees' in text and 'Krone abgebrochen' in text
    assert A.explain_difference(r.copy().set_axis(r.index).where(r.index != 'height_uls', 30.5), height_diff_m=1.0, routine_remark_tokens=[]) == ''
