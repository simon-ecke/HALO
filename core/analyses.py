"""Analyses that show what the ULS data add: error budget, identity subsets, sampling value,
stand description, detection model, data-quality ledger, sample-tree table, height variants.

Everything is a pure function (data in, DataFrame or dict out; thresholds as arguments). Only
`write_tables` writes files. `compute_entries(cfg, tables_dir)` returns entries in the shape of
stats.json: {id, value, unit, n, ci_low, ci_high, subset, statement}.

Rules kept throughout: the tape height (Massband) is the reference; Vertex and the mean of both are
evaluated separately; tree height never selects a match; no tree is dropped for its height.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from core import stats
from core.stats import _entry, agreement, SUBSETS

IDENTITY_SUBSETS = ('all_matched', 'detected_only', 'confirmed_only', 'confirmed_or_probable')
SUBSET_LABEL = {'all_matched': 'matched trees', 'detected_only': 'detected trees',
                'confirmed_only': 'trees with a confirmed identity',
                'confirmed_or_probable': 'trees with a confirmed or probable identity'}

# the statement builder of core.stats looks subset names up here
stats.SUBSET_TEXT.setdefault('confirmed_only', 'matched trees with a confirmed identity')
stats.SUBSET_TEXT.setdefault('confirmed_or_probable', 'matched trees with a confirmed or probable identity')


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _pct_ci(boot, level: float = 95.0):
    lo = (100 - level) / 2
    return float(np.percentile(boot, lo)), float(np.percentile(boot, 100 - lo))


def _rmse(e) -> float:
    e = np.asarray(e, dtype=float)
    return float(np.sqrt(np.mean(e ** 2)))


def _fmt_ci(lo, hi, digits=2, unit='m') -> str:
    return f"{lo:.{digits}f} to {hi:.{digits}f} {unit}".strip()


# ---------------------------------------------------------------------------
# 1. Error budget: ULS - tape against Vertex - tape
# ---------------------------------------------------------------------------

def error_budget(tape: pd.Series, vertex: pd.Series, uls: pd.Series, *, noise_band_sd: float = 1.96,
                 n_boot: int = 2000, seed: int = 0) -> dict:
    """Compare the ULS-tape differences with the Vertex-tape differences on the trees that have all three.

    The Vertex-tape differences are the terrestrial noise floor. `band` = noise_band_sd x the RMS of the
    Vertex-tape differences (centred on 0 because the tape is the reference); `share_inside` is the share of
    ULS-tape differences inside +/- band. Trees are resampled in pairs, so the intervals keep the pairing.
    `share_inside_all_tape` uses the same band on all trees that have a tape and a ULS height.
    """
    both = pd.concat([tape, vertex, uls], axis=1, keys=['t', 'v', 'u']).dropna()
    e_u, e_v = (both['u'] - both['t']).to_numpy(), (both['v'] - both['t']).to_numpy()
    n = len(both)
    out = {'n': n}
    tu = pd.concat([tape, uls], axis=1, keys=['t', 'u']).dropna()
    all_e = (tu['u'] - tu['t']).to_numpy()
    out['n_tape'] = len(tu)
    out['median_abs_uls'] = float(np.median(np.abs(e_u))) if n else np.nan
    out['median_abs_vertex'] = float(np.median(np.abs(e_v))) if n else np.nan
    if n < 3:
        return out
    band = noise_band_sd * _rmse(e_v)
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    bu, bv = e_u[idx], e_v[idx]
    rmse_u, rmse_v = np.sqrt((bu ** 2).mean(axis=1)), np.sqrt((bv ** 2).mean(axis=1))
    sd_u, sd_v = bu.std(axis=1, ddof=1), bv.std(axis=1, ddof=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = (sd_u ** 2) / (sd_v ** 2)
    ratio = ratio[np.isfinite(ratio)]
    out.update({
        'bias_uls': float(e_u.mean()), 'bias_vertex': float(e_v.mean()),
        'sd_uls': float(e_u.std(ddof=1)), 'sd_vertex': float(e_v.std(ddof=1)),
        'rmse_uls': _rmse(e_u), 'rmse_vertex': _rmse(e_v),
        'rmse_diff': _rmse(e_u) - _rmse(e_v), 'band': band,
        'share_inside': float(np.mean(np.abs(e_u) <= band)), 'n_inside': int((np.abs(e_u) <= band).sum()),
        'share_inside_all_tape': float(np.mean(np.abs(all_e) <= band)) if len(all_e) else np.nan,
        'n_inside_all_tape': int((np.abs(all_e) <= band).sum()),
    })
    out['rmse_uls_ci'], out['rmse_vertex_ci'] = _pct_ci(rmse_u), _pct_ci(rmse_v)
    out['rmse_diff_ci'] = _pct_ci(rmse_u - rmse_v)
    out['var_ratio'] = float(e_u.var(ddof=1) / e_v.var(ddof=1)) if e_v.var(ddof=1) > 0 else np.nan
    out['var_ratio_ci'] = _pct_ci(ratio) if len(ratio) else (np.nan, np.nan)
    out['share_inside_ci'] = _pct_ci(np.mean(np.abs(bu) <= band, axis=1))
    return out


def error_budget_entries(eb: dict) -> list[dict]:
    n = eb['n']
    if n < 3:
        return []
    ci = eb['rmse_diff_ci']
    if ci[0] > 0:
        verdict = (f"the ULS error is larger than the Vertex error (the interval of the difference, "
                   f"{_fmt_ci(*ci)}, lies above 0)")
    elif ci[1] < 0:
        verdict = (f"the ULS error is smaller than the Vertex error (the interval of the difference, "
                   f"{_fmt_ci(*ci)}, lies below 0)")
    else:
        verdict = (f"ULS and the Vertex cannot be told apart in RMSE (the interval of the difference, "
                   f"{_fmt_ci(*ci)}, includes 0), so no claim that ULS is better or worse than the terrestrial noise floor")
    typical = (f"; for the typical tree the median absolute difference is {eb['median_abs_uls']:.2f} m for ULS and "
               f"{eb['median_abs_vertex']:.2f} m for the Vertex")
    out = [
        _entry('error_budget.rmse_uls', eb['rmse_uls'], 'm', n, ci=eb['rmse_uls_ci'], subset='sample_trees_with_both',
               statement=f"On the {n} sample trees with Maßband, Vertex and ULS height, the ULS-minus-Maßband RMSE is "
                         f"{eb['rmse_uls']:.2f} m (95 % bootstrap interval {_fmt_ci(*eb['rmse_uls_ci'])}), bias {eb['bias_uls']:+.2f} m."),
        _entry('error_budget.rmse_vertex', eb['rmse_vertex'], 'm', n, ci=eb['rmse_vertex_ci'], subset='sample_trees_with_both',
               statement=f"On the same {n} trees the Vertex-minus-Maßband RMSE, the terrestrial noise floor, is "
                         f"{eb['rmse_vertex']:.2f} m (95 % bootstrap interval {_fmt_ci(*eb['rmse_vertex_ci'])}), bias {eb['bias_vertex']:+.2f} m."),
        _entry('error_budget.rmse_difference', eb['rmse_diff'], 'm', n, ci=eb['rmse_diff_ci'], subset='sample_trees_with_both',
               statement=f"The ULS RMSE minus the Vertex RMSE is {eb['rmse_diff']:+.2f} m on {n} trees (paired bootstrap interval {_fmt_ci(*eb['rmse_diff_ci'])})."),
        _entry('error_budget.variance_ratio', eb['var_ratio'], 'ratio', n, ci=eb['var_ratio_ci'], subset='sample_trees_with_both',
               statement=f"The variance of the ULS-Maßband differences is {eb['var_ratio']:.2f} times that of the Vertex-Maßband differences "
                         f"on {n} trees (95 % bootstrap interval {eb['var_ratio_ci'][0]:.2f} to {eb['var_ratio_ci'][1]:.2f}); 1 means equal spread."),
        _entry('error_budget.share_inside_noise', eb['share_inside'], 'share', n, ci=eb['share_inside_ci'], subset='sample_trees_with_both',
               statement=f"{eb['n_inside']} of {n} ULS-Maßband differences ({100 * eb['share_inside']:.0f} %) lie within +/- {eb['band']:.2f} m, "
                         f"the spread of the Vertex-Maßband differences (1.96 x their RMS)."),
        _entry('error_budget.verdict', eb['rmse_diff'], 'm', n, ci=eb['rmse_diff_ci'], subset='sample_trees_with_both',
               statement=f"Compared with the terrestrial noise floor, on {n} trees {verdict}; with this few trees only large differences can be shown{typical}."),
    ]
    return out


# ---------------------------------------------------------------------------
# 2. Accuracy for confirmed identity only
# ---------------------------------------------------------------------------

def identity_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """The two existing subsets plus the two identity subsets (identity_class from match_identity.csv)."""
    m = dict(stats.subset_masks(frame))
    cls = frame['identity_class']
    m['confirmed_only'] = cls == 'confirmed'
    m['confirmed_or_probable'] = cls.isin(['confirmed', 'probable'])
    return m


def identity_accuracy(everyone: pd.DataFrame) -> pd.DataFrame:
    """Height (three references, kept separate) and diameter accuracy for the four subsets."""
    rows = []
    samples = everyone[everyone['has_height']]
    refs = [('massband', 'height_massband'), ('vertex', 'height_vertex'), ('mean', 'height_felled_mean')]
    for subset, keep in identity_masks(samples).items():
        for ref, col in refs:
            d = samples[keep & samples[col].notna()]
            rows.append({'quantity': 'height', 'group': ref, 'subset': subset, 'unit': 'm',
                         **agreement(d[col], d['lidar_height'])})
    groups = {'sample trees': everyone[everyone['has_height']],
              'other harvested trees': everyone[everyone['removed_2026'] & ~everyone['has_height']],
              'all field trees': everyone}
    for group, g in groups.items():
        for subset, keep in identity_masks(g).items():
            d = g[keep & g['dbh_field_mm'].notna() & (g['dbh_quality'] == 'ok')]
            rows.append({'quantity': 'dbh', 'group': group, 'subset': subset, 'unit': 'cm',
                         **agreement(d['dbh_field_mm'] / 10, d['dbh_lidar_mm'] / 10)})
    return pd.DataFrame(rows)


def identity_accuracy_entries(table: pd.DataFrame) -> list[dict]:
    out = []
    for _, r in table.iterrows():
        if r['n'] < 2:
            continue
        sub = r['subset']
        if r['quantity'] == 'height':
            prefix = f"identity.height_{r['group']}.{sub}"
            label = f"ULS height against the {'Maßband' if r['group'] == 'massband' else r['group'].capitalize()} height"
        else:
            prefix = f"identity.dbh_{stats.slug(r['group'])}.{sub}"
            label = f"ULS against field DBH, {r['group']} (ok circle fits)"
        out.extend(stats._accuracy_entries(prefix, label, sub, r, r['unit']))
    # change in RMSE when only confirmed matches are kept; an improvement is claimed only if intervals are apart
    for (q, g), t in table.groupby(['quantity', 'group']):
        t = t.set_index('subset')
        if not {'all_matched', 'confirmed_only'} <= set(t.index):
            continue
        a, c = t.loc['all_matched'], t.loc['confirmed_only']
        if min(a['n'], c['n']) < 3:
            continue
        better = c['rmse_ci_high'] < a['rmse_ci_low']
        worse = c['rmse_ci_low'] > a['rmse_ci_high']
        text = ("the intervals are apart, so confirmed matches are more accurate" if better else
                "the intervals are apart, so confirmed matches are less accurate" if worse else
                "the 95 % intervals overlap, so no improvement is claimed")
        tag = f"{'height_' if q == 'height' else 'dbh_'}{stats.slug(g)}"
        out.append(_entry(f'identity.{tag}.rmse_change_confirmed', c['rmse'] - a['rmse'], t['unit'].iloc[0], int(c['n']),
                          subset='confirmed_only',
                          statement=f"{q.upper() if q == 'dbh' else 'Height'} ({g}): the RMSE is {c['rmse']:.2f} on {int(c['n'])} confirmed matches "
                                    f"against {a['rmse']:.2f} on {int(a['n'])} matched trees; {text}."))
    return out


# ---------------------------------------------------------------------------
# 3. Sampling value
# ---------------------------------------------------------------------------

def sampling_value(heights: pd.Series, ks: Sequence[int], *, draws: int = 2000, seed: int = 0,
                   within_m: Sequence[float] = (0.5, 1.0)) -> pd.DataFrame:
    """Stand mean height estimated from k randomly drawn trees (without replacement) against the census."""
    h = heights.dropna().to_numpy(dtype=float)
    N = len(h)
    rng = np.random.default_rng(seed)
    census = float(h.mean())
    rows = []
    for k in ks:
        kk = min(int(k), N)
        pick = rng.random((draws, N)).argpartition(kk - 1, axis=1)[:, :kk] if kk < N else np.tile(np.arange(N), (draws, 1))
        est = h[pick].mean(axis=1)
        row = {'k': int(k), 'k_used': kk, 'n_census': N, 'draws': draws, 'census_mean_m': census,
               'census_sd_m': float(h.std(ddof=1)), 'estimate_mean_m': float(est.mean()), 'estimate_sd_m': float(est.std(ddof=1)),
               'q025_m': float(np.percentile(est, 2.5)), 'q975_m': float(np.percentile(est, 97.5)),
               'halfwidth_95_m': float((np.percentile(est, 97.5) - np.percentile(est, 2.5)) / 2)}
        for w in within_m:
            row[f'share_within_{str(w).replace(".", "_")}m'] = float(np.mean(np.abs(est - census) <= w))
        rows.append(row)
    return pd.DataFrame(rows)


def sampling_value_entries(table: pd.DataFrame, sample_mean: float, n_sample: int, within_m: float = 1.0) -> list[dict]:
    out = []
    col = f'share_within_{str(within_m).replace(".", "_")}m'
    for _, r in table.iterrows():
        out.append(_entry(f"sampling.k{int(r['k'])}.sd", r['estimate_sd_m'], 'm', int(r['k_used']), ci=(r['q025_m'], r['q975_m']),
                          statement=f"A stand mean ULS height from {int(r['k_used'])} randomly drawn trees varies with an SD of {r['estimate_sd_m']:.2f} m "
                                    f"(95 % of {int(r['draws'])} draws between {r['q025_m']:.2f} and {r['q975_m']:.2f} m, "
                                    f"{100 * r[col]:.0f} % within {within_m:g} m of the census mean {r['census_mean_m']:.2f} m); "
                                    f"the ULS census of {int(r['n_census'])} matched trees has no sampling error."))
    c = float(table['census_mean_m'].iloc[0])
    out.append(_entry('sampling.sample_trees_mean_minus_census', sample_mean - c, 'm', n_sample,
                      statement=f"The {n_sample} actual sample trees have a mean ULS height {sample_mean - c:+.2f} m from the census mean {c:.2f} m; "
                                f"they are felled trees chosen by the crew, not a random draw, so this one value does not show a bias."))
    return out


# ---------------------------------------------------------------------------
# 4. Stand description from ULS
# ---------------------------------------------------------------------------

def top_height(heights: pd.Series, dbh: pd.Series, area_m2: float, *, trees_per_ha: int = 100) -> tuple[float, int]:
    """Mean height of the thickest trees: `trees_per_ha` per hectare scaled to the plot area (at least 1)."""
    d = pd.concat([heights, dbh], axis=1, keys=['h', 'd']).dropna()
    if d.empty or not area_m2 > 0:
        return np.nan, 0
    n_top = max(1, int(round(trees_per_ha * area_m2 / 10000.0)))
    top = d.sort_values('d', ascending=False).head(n_top)
    return float(top['h'].mean()), int(len(top))


def stand_per_parzelle(everyone: pd.DataFrame, areas: pd.Series, *, trees_per_ha: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-plot table and per-plot-and-species height distribution; ULS heights of all matched trees.

    `areas` maps the plot id to its area in m2 (from the surveyed corners).
    """
    matched = everyone['detection'] != 'not detected'
    rows, sp_rows = [], []
    for p, g in everyone.groupby('parzelle'):
        m = g[matched.loc[g.index]]
        ok = m[(m['dbh_quality'] == 'ok') & m['dbh_field_mm'].notna()]
        area = float(areas.get(str(p), areas.get(p, np.nan)))
        top, n_top = top_height(m['lidar_height'], m['dbh_field_mm'], area, trees_per_ha=trees_per_ha)
        h = m['lidar_height'].dropna()
        rows.append({
            'parzelle': p, 'area_m2': area, 'n_field_trees': len(g), 'n_matched': len(m),
            'n_detected': int((g['detection'] == 'detected').sum()),
            'detection_rate': float((g['detection'] == 'detected').mean()), 'matched_rate': float(len(m) / len(g)),
            'field_trees_per_ha': len(g) / area * 10000 if area > 0 else np.nan,
            'n_dbh_ok': len(ok), 'dbh_bias_uls_minus_field_cm': float((ok['dbh_lidar_mm'] - ok['dbh_field_mm']).mean() / 10) if len(ok) else np.nan,
            'mean_height_m': float(h.mean()) if len(h) else np.nan, 'sd_height_m': float(h.std(ddof=1)) if len(h) > 1 else np.nan,
            'p10_height_m': float(h.quantile(0.1)) if len(h) else np.nan, 'median_height_m': float(h.median()) if len(h) else np.nan,
            'p90_height_m': float(h.quantile(0.9)) if len(h) else np.nan,
            'top_height_m': top, 'n_top_trees': n_top, 'n_heights': len(h)})
        for sp, s in m.groupby(m['species'].fillna('unknown')):
            hs = s['lidar_height'].dropna()
            sp_rows.append({'parzelle': p, 'species': sp, 'n': len(hs), 'mean_height_m': hs.mean(),
                            'sd_height_m': hs.std(ddof=1) if len(hs) > 1 else np.nan, 'min_height_m': hs.min(),
                            'p10_height_m': hs.quantile(0.1), 'median_height_m': hs.median(),
                            'p90_height_m': hs.quantile(0.9), 'max_height_m': hs.max()})
    return pd.DataFrame(rows), pd.DataFrame(sp_rows)


def stand_entries(plots: pd.DataFrame, everyone: pd.DataFrame, trees_per_ha: int) -> list[dict]:
    matched = everyone[everyone['detection'] != 'not detected']
    h = matched['lidar_height'].dropna()
    n_pl = len(plots)
    th = plots['top_height_m']
    ok = matched[(matched['dbh_quality'] == 'ok') & matched['dbh_field_mm'].notna()]
    a = agreement(ok['dbh_field_mm'] / 10, ok['dbh_lidar_mm'] / 10)
    out = [
        _entry('stand.mean_height', h.mean(), 'm', len(h), ci=(h.quantile(0.025), h.quantile(0.975)), subset='all_matched',
               statement=f"The mean ULS height of the {len(h)} matched trees is {h.mean():.1f} m; the interval gives the 2.5 and 97.5 percentiles of single trees "
                         f"({h.quantile(0.025):.1f} to {h.quantile(0.975):.1f} m), not a confidence interval."),
        _entry('stand.top_height_median_over_plots', th.median(), 'm', n_pl, ci=(th.min(), th.max()), subset='all_matched',
               statement=f"The top height (mean ULS height of the {trees_per_ha} thickest trees per hectare, scaled to the plot area) has a median of "
                         f"{th.median():.1f} m over {n_pl} Parzellen, ranging from {th.min():.1f} to {th.max():.1f} m."),
        _entry('stand.mean_height_range_over_plots', plots['mean_height_m'].median(), 'm', n_pl,
               ci=(plots['mean_height_m'].min(), plots['mean_height_m'].max()), subset='all_matched',
               statement=f"The mean ULS height per Parzelle has a median of {plots['mean_height_m'].median():.1f} m over {n_pl} Parzellen "
                         f"(range {plots['mean_height_m'].min():.1f} to {plots['mean_height_m'].max():.1f} m)."),
        _entry('stand.detection_rate_range_over_plots', plots['detection_rate'].median(), 'share', n_pl,
               ci=(plots['detection_rate'].min(), plots['detection_rate'].max()), subset='detected_only',
               statement=f"The detection rate per Parzelle has a median of {100 * plots['detection_rate'].median():.0f} % "
                         f"(range {100 * plots['detection_rate'].min():.0f} to {100 * plots['detection_rate'].max():.0f} %) over {n_pl} Parzellen."),
    ]
    if a['n'] >= 2:
        out.append(_entry('stand.dbh_bias_all_matched', a['bias'], 'cm', a['n'], ci=(a['bias_ci_low'], a['bias_ci_high']), subset='all_matched',
                          statement=f"Over all {a['n']} matched trees with an ok circle fit the ULS diameter is {a['bias']:+.2f} cm from the field diameter "
                                    f"(95 % bootstrap interval {a['bias_ci_low']:.2f} to {a['bias_ci_high']:.2f} cm); the plot values are in stand_per_parzelle.csv."))
    return out


# ---------------------------------------------------------------------------
# 5. Detection model
# ---------------------------------------------------------------------------

def _logit_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=1e6, max_iter=500).fit(X, y).coef_[0]


def detection_model(everyone: pd.DataFrame, predictors: dict[str, str], *, class_edges_mm: Sequence[float],
                    size_column: str, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Logistic regression of detected (1) against not detected or uncertain (0).

    `predictors` maps a label to a column; each is standardised, so coefficients are log-odds per SD.
    Rows: kind 'coefficient' (value, ci_low, ci_high, n, n_events, odds_ratio_per_sd), kind 'rate_by_class'
    (value = detection rate, Wilson interval, n trees per diameter class) and kind 'correlation'.
    """
    cols = list(predictors.values())
    d = everyone[['detection'] + cols].dropna()
    y = (d['detection'] == 'detected').to_numpy(dtype=int)
    X = d[cols].to_numpy(dtype=float)
    X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)
    rows = []
    if y.min() != y.max():
        coef = _logit_fit(X, y)
        rng = np.random.default_rng(seed)
        boot = []
        for _ in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            if y[i].min() == y[i].max():
                continue
            boot.append(_logit_fit(X[i], y[i]))
        boot = np.array(boot)
        for j, label in enumerate(predictors):
            lo, hi = _pct_ci(boot[:, j])
            rows.append({'kind': 'coefficient', 'term': label, 'value': coef[j], 'ci_low': lo, 'ci_high': hi, 'n': len(y),
                         'n_events': int(y.sum()), 'odds_ratio_per_sd': float(np.exp(coef[j]))})
        corr = np.corrcoef(X.T)
        labels = list(predictors)
        for a in range(len(labels)):
            for b in range(a + 1, len(labels)):
                rows.append({'kind': 'correlation', 'term': f'{labels[a]} / {labels[b]}', 'value': corr[a, b], 'n': len(y)})
    sizes = everyone[size_column]
    cls = pd.cut(sizes, bins=list(class_edges_mm), right=False)
    for c, g in everyone.groupby(cls, observed=True):
        k, n = int((g['detection'] == 'detected').sum()), len(g)
        lo, hi = stats.wilson(k, n)
        rows.append({'kind': 'rate_by_class', 'term': f'{c.left:g}-{c.right:g} mm', 'value': k / n, 'ci_low': lo, 'ci_high': hi,
                     'n': n, 'n_events': k})
    return pd.DataFrame(rows)


def detection_model_entries(table: pd.DataFrame, phrases: dict[str, str]) -> list[dict]:
    """One entry and one plain sentence per predictor. `phrases` maps the label to a short description."""
    out = []
    for _, r in table[table['kind'] == 'coefficient'].iterrows():
        what = phrases.get(r['term'], r['term'])
        clear = r['ci_low'] > 0 or r['ci_high'] < 0
        if clear:
            direction = 'raises' if r['value'] > 0 else 'lowers'
            verdict = f"a larger {what} {direction} the chance that a tree is detected"
        else:
            verdict = f"no clear effect of {what} (the interval includes 0)"
        out.append(_entry(f"detection_model.{stats.slug(r['term'])}", r['value'], 'log-odds per SD', r['n'], ci=(r['ci_low'], r['ci_high']),
                          statement=f"On {int(r['n'])} field trees ({int(r['n_events'])} detected), one SD more {what} changes the log-odds of detection by "
                                    f"{r['value']:+.2f} (95 % bootstrap interval {r['ci_low']:+.2f} to {r['ci_high']:+.2f}, odds ratio {r['odds_ratio_per_sd']:.2f}): {verdict}; "
                                    f"the predictors overlap, so read each effect with the others held fixed."))
    return out


def detection_class_entries(table: pd.DataFrame) -> list[dict]:
    out = []
    for _, r in table[table['kind'] == 'rate_by_class'].iterrows():
        out.append(_entry(f"detection_model.rate_{stats.slug(r['term'])}", r['value'], 'share', r['n'], ci=(r['ci_low'], r['ci_high']),
                          subset='detected_only',
                          statement=f"In the diameter class {r['term']}, {int(r['n_events'])} of {int(r['n'])} field trees are detected ({100 * r['value']:.1f} %, Wilson interval "
                                    f"{100 * r['ci_low']:.1f} to {100 * r['ci_high']:.1f} %)."))
    return out


# ---------------------------------------------------------------------------
# 6. Data-quality ledger
# ---------------------------------------------------------------------------

def is_routine_remark(text, tokens: Sequence[str]) -> bool:
    """True when the remark holds nothing but the routine tokens (year, BAD, '!', ';')."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return True
    s = str(text)
    for t in tokens:
        s = s.replace(t, ' ')
    return s.strip() == ''


def ledger(everyone: pd.DataFrame, *, dbh_plausible_mm: Sequence[float], gone_above: float, standing_below: float,
           routine_remark_tokens: Sequence[str], height_diff_m: float, mirrored_plots: Sequence = ()) -> pd.DataFrame:
    """One row per issue: tree_id, parzelle, issue, detail. A tree can appear in several rows."""
    rows = []

    def add(tree, plot, issue, detail):
        rows.append({'tree_id': tree, 'parzelle': plot, 'issue': issue, 'detail': detail})

    lo, hi = dbh_plausible_mm
    state = pd.Series(stats.lidar_state(everyone['removed_fraction'].fillna(np.nan), gone_above, standing_below), index=everyone.index)
    for i, r in everyone.iterrows():
        t, p = r['tree_id'], r['parzelle']
        if pd.isna(r['dbh_mm']):
            add(t, p, 'dbh_missing', 'no field diameter recorded')
        elif not lo <= r['dbh_mm'] <= hi:
            add(t, p, 'dbh_implausible', f"field diameter {r['dbh_mm']:g} mm is outside {lo:g} to {hi:g} mm and is not used")
        if r['detection'] == 'not detected':
            add(t, p, 'unmatched', f"no LiDAR instance assigned; cause: {r.get('cause') or 'none given'}")
        if r.get('identity_class') in ('ambiguous', 'rejected'):
            add(t, p, f"identity_{r['identity_class']}", str(r.get('reasons') or ''))
        if r.get('removal_consistent') is False or (pd.notna(r.get('removal_consistent')) and not bool(r['removal_consistent'])):
            rec = 'marked for removal' if r['removed_2026'] else 'not marked for removal'
            add(t, p, 'harvest_record_disagrees', f"field record: {rec}; LiDAR: instance {state[i]} after the harvest "
                                                  f"(removed fraction {r['removed_fraction']:.2f})")
        if r['has_height']:
            if pd.notna(r['removed_fraction']) and r['removed_2026'] and state[i] == 'still standing':
                add(t, p, 'sample_tree_stands_after_harvest',
                    f"recorded as measured lying (Maßband {r['height_massband']}, Vertex {r['height_vertex']}), but a tree still stands at its position "
                    f"after the harvest (removed fraction {r['removed_fraction']:.2f}); needs a field check")
            if not is_routine_remark(r['remark'], routine_remark_tokens):
                add(t, p, 'sample_tree_remark', f"remark: {r['remark']}")
            ref = r['height_massband'] if pd.notna(r['height_massband']) else r['height_vertex']
            if pd.notna(ref) and pd.notna(r['lidar_height']) and abs(r['lidar_height'] - ref) > height_diff_m:
                add(t, p, 'sample_tree_height_differs', f"ULS {r['lidar_height']:.2f} m against {'Maßband' if pd.notna(r['height_massband']) else 'Vertex'} "
                                                       f"{ref:.2f} m ({r['lidar_height'] - ref:+.2f} m); flag only, the match is not changed")
    for p in mirrored_plots:
        n = int((everyone['parzelle'] == p).sum())
        add('', p, 'plot_y_mirrored', f"stem y coordinates of this Parzelle are mirrored before matching ({n} trees concerned)")
    return pd.DataFrame(rows, columns=['tree_id', 'parzelle', 'issue', 'detail'])


def ledger_entries(led: pd.DataFrame) -> list[dict]:
    out = []
    for issue, g in led.groupby('issue'):
        n_trees = int(g.loc[g['tree_id'] != '', 'tree_id'].nunique())
        out.append(_entry(f'ledger.{issue}', len(g), 'rows', n_trees or len(g),
                          statement=f"The ledger has {len(g)} '{issue}' entries"
                                    + (f" for {n_trees} trees." if n_trees else " (one per Parzelle).")))
    return out


# ---------------------------------------------------------------------------
# 7. Sample trees table
# ---------------------------------------------------------------------------

def explain_difference(r: pd.Series, *, height_diff_m: float, routine_remark_tokens: Sequence[str]) -> str:
    """Plain-English reason for a height difference above `height_diff_m`; built from cause, identity class and remark only."""
    ref_name, ref = ('the Maßband length', r['height_massband']) if pd.notna(r['height_massband']) else ('the Vertex length', r['height_vertex'])
    if pd.isna(ref) or pd.isna(r['height_uls']) or abs(r['height_uls'] - ref) <= height_diff_m:
        return ''
    d = r['height_uls'] - ref
    parts = [f"The ULS height is {abs(d):.1f} m {'above' if d > 0 else 'below'} {ref_name}."]
    cause = r.get('cause')
    if isinstance(cause, str) and cause:
        parts.append(f"The matching analysis records: {cause}.")
    cls = r.get('identity_class')
    if cls == 'confirmed':
        parts.append("The match itself is confirmed by position, so a wrong instance is unlikely; look at the measurement or the crown.")
    elif cls == 'probable':
        parts.append("The match is probable by position, but not confirmed.")
    elif cls in ('ambiguous', 'rejected'):
        parts.append(f"The identity of the match is {cls}, so the ULS instance may be another tree.")
    elif cls == 'unmatched':
        parts.append("No ULS instance is matched to this tree.")
    if r.get('tier_method') == 'none':
        parts.append("The final match table holds no instance for this tree (tier 'none'); the ULS height shown comes from the first-pass nearest-instance match, which the later steps did not keep.")
    elif r.get('tier_method') in ('manual_segmentation', 'column_profile'):
        parts.append(f"The height comes from a {str(r['tier_method']).replace('_', ' ')}, not from SegmentAnyTree.")
    if not is_routine_remark(r.get('remark'), routine_remark_tokens):
        parts.append(f"Field remark: {r['remark']}.")
    if len(parts) == 1:
        parts.append("The data hold no cause; check the field record and the tape reading.")
    return ' '.join(parts)


def sample_trees(everyone: pd.DataFrame, *, height_diff_m: float, routine_remark_tokens: Sequence[str]) -> pd.DataFrame:
    s = everyone[everyone['has_height']].copy()
    t = pd.DataFrame({
        'tree_id': s['tree_id'], 'parzelle': s['parzelle'], 'species': s['species'],
        'height_massband_m': s['height_massband'], 'height_vertex_m': s['height_vertex'], 'height_uls_m': s['lidar_height'],
        'diff_uls_minus_massband_m': s['lidar_height'] - s['height_massband'],
        'diff_vertex_minus_massband_m': s['height_vertex'] - s['height_massband'],
        'diff_uls_minus_vertex_m': s['lidar_height'] - s['height_vertex'],
        'identity_class': s['identity_class'], 'identity_score': s['identity_score'], 'tier_method': s['tier_method'],
        'detection': s['detection'].astype(str), 'cause': s['cause'], 'dbh_field_mm': s['dbh_field_mm'],
        'dbh_uls_mm': s['dbh_lidar_mm'], 'remark': s['remark']})
    tmp = t.rename(columns={'height_massband_m': 'height_massband', 'height_vertex_m': 'height_vertex', 'height_uls_m': 'height_uls'})
    t['explanation'] = tmp.apply(lambda r: explain_difference(r, height_diff_m=height_diff_m, routine_remark_tokens=routine_remark_tokens), axis=1)
    return t.reset_index(drop=True)


def sample_entries(t: pd.DataFrame, height_diff_m: float) -> list[dict]:
    n = len(t)
    k = int((t['explanation'] != '').sum())
    return [_entry('samples.differ_more_than_threshold', k, 'trees', n,
                   statement=f"{k} of the {n} sample trees differ from the tape (or Vertex, where no tape) by more than {height_diff_m:g} m; "
                             f"each has an explanation built from the cause, the identity class and the field remark.")]


# ---------------------------------------------------------------------------
# 8. Height definition variants (reads the cloud; see height_variants_table)
# ---------------------------------------------------------------------------

def height_variants_table(tape: pd.Series, variants: dict[str, pd.Series], *, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Each variant against the tape, with the leave-one-out RMSE and the RMSE change against the first variant.

    `variants` maps a name to heights on the same index as `tape`; the first entry is the reference variant.
    Leave-one-out: the RMSE of the variant on the other trees, per left-out tree, summarised by its range.
    """
    base_name = next(iter(variants))
    rows = []
    base_err = None
    for name, h in variants.items():
        d = pd.concat([tape, h], axis=1, keys=['t', 'h']).dropna()
        e = (d['h'] - d['t']).to_numpy()
        if base_name == name:
            base_err = pd.Series(e, index=d.index)
        a = agreement(d['t'], d['h'], n_boot, seed)
        n = len(e)
        loo = np.array([_rmse(np.delete(e, i)) for i in range(n)]) if n > 2 else np.array([np.nan])
        row = {'variant': name, 'n': n, 'bias_m': a['bias'], 'rmse_m': a['rmse'], 'median_m': a['median'], 'mad_m': a['mad'],
               'bias_ci_low': a['bias_ci_low'], 'bias_ci_high': a['bias_ci_high'], 'rmse_ci_low': a['rmse_ci_low'],
               'rmse_ci_high': a['rmse_ci_high'], 'loo_rmse_min_m': np.nanmin(loo), 'loo_rmse_max_m': np.nanmax(loo)}
        if base_err is not None and n > 2:
            ee = pd.Series(e, index=d.index)
            common = ee.index.intersection(base_err.index)
            idx = np.random.default_rng(seed).integers(0, len(common), size=(n_boot, len(common)))
            b = ee.loc[common].to_numpy(); r0 = base_err.loc[common].to_numpy()
            diff = np.sqrt((b[idx] ** 2).mean(axis=1)) - np.sqrt((r0[idx] ** 2).mean(axis=1))
            row['rmse_change_vs_first'] = _rmse(b) - _rmse(r0)
            row['rmse_change_ci_low'], row['rmse_change_ci_high'] = _pct_ci(diff)
            row['improvement_claimed'] = bool(row['rmse_change_ci_high'] < 0)
        rows.append(row)
    return pd.DataFrame(rows)


def read_instance_points(laz_path, instance_ids, instance_dim: str = 'PredInstance', chunk_size: int = 10_000_000) -> pd.DataFrame:
    """One streaming pass over a LAZ file; returns x, y, z, instance of the points of the wanted instances only."""
    import laspy
    wanted = np.asarray(sorted(set(int(i) for i in instance_ids)))
    parts = []
    with laspy.open(laz_path) as reader:
        for pts in reader.chunk_iterator(chunk_size):
            inst = np.asarray(pts[instance_dim]).astype(np.int64)
            m = np.isin(inst, wanted)
            if m.any():
                parts.append(pd.DataFrame({'x': np.asarray(pts.x)[m], 'y': np.asarray(pts.y)[m], 'z': np.asarray(pts.z)[m],
                                           'instance': inst[m]}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=['x', 'y', 'z', 'instance'])


def instance_height_variants(points: pd.DataFrame, ground: np.ndarray, chm: np.ndarray, chm_transform, *, percentile: float = 99.0) -> pd.DataFrame:
    """Per instance: highest point, percentile of the point heights, highest CHM cell that holds a point of the instance.

    `ground` is the terrain height under each point (same order as `points`); `chm` holds heights above ground.
    """
    pts = points.assign(h=points['z'].to_numpy(dtype=float) - ground)
    inv = ~chm_transform
    col, row = inv * (pts['x'].to_numpy(), pts['y'].to_numpy())
    r, c = np.floor(row).astype(int), np.floor(col).astype(int)
    inside = (r >= 0) & (r < chm.shape[0]) & (c >= 0) & (c < chm.shape[1])
    pts['chm'] = np.where(inside, chm[np.clip(r, 0, chm.shape[0] - 1), np.clip(c, 0, chm.shape[1] - 1)], np.nan)
    g = pts.groupby('instance')
    return pd.DataFrame({'highest_point': g['h'].max(), f'p{percentile:g}': g['h'].quantile(percentile / 100.0),
                         'chm_max_in_crown': g['chm'].max(), 'n_points': g.size()})


def height_variants_run(cfg, tables_dir=None, rasters_dir=None, max_minutes: Optional[float] = None) -> Optional[pd.DataFrame]:
    """Height variants on the sample trees with a tape height; one pass over the clipped cloud. Writes height_variants.csv.

    Returns None (and writes nothing) when the cloud, DTM or CHM is missing or the run exceeds `max_minutes`.
    """
    import time
    import rasterio
    from core.pointcloud import sample_raster_bilinear
    from core.tables import read_table
    a = cfg.analyses.height_variants
    limit = float(max_minutes if max_minutes is not None else a.max_minutes)
    t0 = time.time()
    rdir = Path(rasters_dir) if rasters_dir else cfg.output_dir / 'rasters'
    cloud = cfg.output_dir / 'work' / 'sat_before_aoi.laz'
    dtm, chm_p = rdir / 'dtm_lidar_after.tif', rdir / 'chm_lidar_before.tif'
    if not (cloud.exists() and dtm.exists() and chm_p.exists()):
        return None
    e = load_everyone(cfg, tables_dir)
    s = e[e['has_height'] & e['height_massband'].notna() & e['instance'].notna()]
    pts = read_instance_points(cloud, s['instance'].astype(int), cfg.paths.sat_before.instance_dim)
    if (time.time() - t0) / 60 > limit:
        return None
    ground = sample_raster_bilinear(dtm, pts['x'].to_numpy(), pts['y'].to_numpy())
    with rasterio.open(chm_p) as src:
        chm = src.read(1).astype(np.float32)
        chm[chm == src.nodata] = np.nan
        tr = src.transform
    per = instance_height_variants(pts, ground, chm, tr, percentile=float(a.percentile))
    s = s.set_index('tree_id')
    tape = s['height_massband']
    p_name = f"p{float(a.percentile):g}"
    variants = {'highest point of the instance': per['highest_point'].reindex(s['instance'].astype(int)).set_axis(s.index),
                f'{p_name} of the point heights': per[p_name].reindex(s['instance'].astype(int)).set_axis(s.index),
                'highest canopy-model cell in the crown': per['chm_max_in_crown'].reindex(s['instance'].astype(int)).set_axis(s.index)}
    table = height_variants_table(tape, variants, n_boot=int(cfg.analyses.bootstrap.n), seed=int(cfg.analyses.bootstrap.seed))
    cfg.tables_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(cfg.tables_dir / 'height_variants.csv', index=False, encoding='utf-8')
    return table


def height_variant_entries(t: pd.DataFrame) -> list[dict]:
    out = []
    for _, r in t.iterrows():
        if 'rmse_change_vs_first' not in r or pd.isna(r.get('rmse_change_vs_first')):
            continue
        if r['rmse_change_ci_low'] == r['rmse_change_ci_high']:     # the reference variant itself
            continue
        verdict = ("the interval of the RMSE change is below 0, an improvement" if r['improvement_claimed']
                   else "the interval of the RMSE change includes 0 or is above it, so no improvement is claimed")
        out.append(_entry(f"height_variant.{stats.slug(r['variant'])}.rmse_change", r['rmse_change_vs_first'], 'm', int(r['n']),
                          ci=(r['rmse_change_ci_low'], r['rmse_change_ci_high']), subset='sample_trees_with_tape',
                          statement=f"The height variant '{r['variant']}' changes the RMSE against the Maßband by {r['rmse_change_vs_first']:+.2f} m on {int(r['n'])} trees "
                                    f"(95 % bootstrap interval {r['rmse_change_ci_low']:+.2f} to {r['rmse_change_ci_high']:+.2f} m): {verdict}."))
    return out


# ---------------------------------------------------------------------------
# Loading and the two entry points
# ---------------------------------------------------------------------------

_MEMO: dict = {}


def load_everyone(cfg, tables_dir=None) -> pd.DataFrame:
    """One row per field tree: the frame of stats.build_everyone plus the identity columns and structure columns."""
    from core.tables import read_table
    sid = {'dtype': {'tree_id': str}}
    result = read_table(cfg, 'field_trees_detection.csv', tables_dir, **sid)
    values = read_table(cfg, 'height_dbh_all_trees.csv', tables_dir, **sid)
    tiers = read_table(cfg, 'match_tiers.csv', tables_dir, **sid)
    causes = read_table(cfg, 'match_causes.csv', tables_dir, **sid).fillna({'cause': '', 'attributed_to': ''})
    profile = read_table(cfg, 'tree_profile_all.csv', tables_dir, **sid)
    ident = read_table(cfg, 'match_identity.csv', tables_dir, **sid)
    r = stats.derive_field(result, standing_height_dm_above=cfg.cleaning.standing_height_dm_above,
                           dbh_plausible_mm=list(cfg.cleaning.dbh_plausible_mm))
    e = stats.build_everyone(r, values, tiers, causes, profile)
    return e.merge(ident[['tree_id', 'identity_score', 'identity_class', 'reasons']], on='tree_id', how='left') \
            .assign(identity_class=lambda d: d['identity_class'].fillna('unmatched'))


def plot_areas(cfg) -> pd.Series:
    """Plot area in m2 from the surveyed corners (read Excel first, geopandas after)."""
    from core import coords, io_field
    corners = io_field.load_corners(cfg)
    poly = coords.parzelle_polygons(corners, cfg.site.crs)
    return poly.set_index('parzelle')['area_m2']


def compute(cfg, tables_dir=None) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    key = (id(cfg), str(tables_dir))
    if key in _MEMO:
        return _MEMO[key]
    a = cfg.analyses
    n_boot, seed = int(a.bootstrap.n), int(a.bootstrap.seed)
    e = load_everyone(cfg, tables_dir)
    entries, tabs = [], {}

    s = e[e['has_height']]
    eb = error_budget(s['height_massband'], s['height_vertex'], s['lidar_height'], noise_band_sd=float(a.noise_band_sd),
                      n_boot=n_boot, seed=seed)
    entries += error_budget_entries(eb)
    for subset, keep in stats.subset_masks(s).items():
        d = s[keep & s['height_massband'].notna() & s['height_vertex'].notna() & s['lidar_height'].notna()]
        entries += stats._accuracy_entries('error_budget.uls_vs_massband', 'ULS height against the Maßband length (trees with Maßband, Vertex and ULS height)',
                                           subset, agreement(d['height_massband'], d['lidar_height'], n_boot, seed), 'm')
        entries += stats._accuracy_entries('error_budget.vertex_vs_massband', 'Vertex length against the Maßband length (the noise floor; same trees)',
                                           subset, agreement(d['height_massband'], d['height_vertex'], n_boot, seed), 'm')

    acc = identity_accuracy(e)
    tabs['accuracy_by_identity.csv'] = acc
    entries += identity_accuracy_entries(acc)

    matched = e[e['detection'] != 'not detected']
    sv = sampling_value(matched['lidar_height'], list(a.sampling.k), draws=int(a.sampling.draws), seed=int(a.sampling.seed))
    tabs['sampling_value.csv'] = sv
    sample_h = s['lidar_height'].dropna()
    entries += sampling_value_entries(sv, float(sample_h.mean()), len(sample_h), float(a.sampling.within_m))

    plots, species = stand_per_parzelle(e, plot_areas(cfg), trees_per_ha=int(a.top_height.trees_per_ha))
    tabs['stand_per_parzelle.csv'], tabs['stand_per_parzelle_species.csv'] = plots, species
    entries += stand_entries(plots, e, int(a.top_height.trees_per_ha))

    dm = a.detection_model
    preds = {'diameter': dm.size_column, 'competition': dm.competition_column, 'relative size': dm.relative_size_column}
    dmt = detection_model(e, preds, class_edges_mm=list(cfg.cleaning.dbh_class_edges_mm), size_column=dm.size_column,
                          n_boot=n_boot, seed=seed)
    tabs['detection_model.csv'] = dmt
    entries += detection_model_entries(dmt, {'diameter': 'field diameter', 'competition': 'local competition (Hegyi index)',
                                             'relative size': 'diameter relative to the neighbours'})
    entries += detection_class_entries(dmt)

    tokens = list(a.ledger.routine_remark_tokens)
    led = ledger(e, dbh_plausible_mm=list(cfg.cleaning.dbh_plausible_mm), gone_above=cfg.matching.gone_above,
                 standing_below=cfg.matching.standing_below, routine_remark_tokens=tokens,
                 height_diff_m=float(a.explain_height_diff_m), mirrored_plots=list(cfg.plots.y_mirrored))
    tabs['ledger.csv'] = led
    entries += ledger_entries(led)

    st = sample_trees(e, height_diff_m=float(a.explain_height_diff_m), routine_remark_tokens=tokens)
    tabs['sample_trees.csv'] = st
    entries += sample_entries(st, float(a.explain_height_diff_m))

    # height variants only when a result table from the cloud run exists (see height_variants_run); never recomputed here
    from core.tables import has_table, read_table
    if has_table(cfg, 'height_variants.csv', tables_dir):
        hv = read_table(cfg, 'height_variants.csv', tables_dir)
        entries += height_variant_entries(hv)

    _MEMO[key] = (entries, tabs)
    return entries, tabs


def compute_entries(cfg, tables_dir=None) -> list[dict]:
    """Entries for stats.json."""
    return compute(cfg, tables_dir)[0]


def write_tables(cfg, tables_dir=None, rasters_dir=None, height_variants: bool = False) -> list[Path]:
    """Write the analysis tables to cfg.tables_dir (UTF-8 CSV, no index); returns the paths."""
    _, tabs = compute(cfg, tables_dir)
    cfg.tables_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, df in tabs.items():
        p = cfg.tables_dir / name
        df.to_csv(p, index=False, encoding='utf-8')
        paths.append(p)
    if height_variants and height_variants_run(cfg, tables_dir, rasters_dir) is not None:
        paths.append(cfg.tables_dir / 'height_variants.csv')
    return paths
