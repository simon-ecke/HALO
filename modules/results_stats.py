"""Statistics on the per-tree tables of Versuch 155 (no point clouds, no rasters).

Used by the results notebook; the pipeline uses the few functions it needs to
write its per-Parzelle table.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from . import notebook_common as nc


# ---------------------------------------------------------------------------
# Agreement of an estimate with a reference
# ---------------------------------------------------------------------------

def agreement(reference: pd.Series, estimate: pd.Series, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """n, bias, SD, RMSE, MAE, limits of agreement, r, and robust measures; error = estimate - reference.

    median / mad describe the typical tree and are not moved by a few large errors
    (mad = median absolute deviation from the median, scaled to match an SD for
    normal errors). rmse_ci_low / rmse_ci_high is the 95 % bootstrap interval of the
    RMSE (trees resampled), within_1 the share of errors of at most one unit.
    """
    ok = reference.notna() & estimate.notna()
    ref = reference[ok].astype(float)
    err = (estimate[ok] - reference[ok]).astype(float)
    keys = ['bias', 'sd', 'rmse', 'mae', 'loa_low', 'loa_high', 'r', 'median', 'mad', 'max_abs',
            'rmse_ci_low', 'rmse_ci_high', 'within_1']
    if len(err) < 2:
        return {'n': int(len(err)), **{k: np.nan for k in keys}}

    sd = err.std(ddof=1)
    values = err.to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(values), size=(n_boot, len(values)))
    boot = np.sqrt((values[idx] ** 2).mean(axis=1))
    return {
        'n': int(len(err)), 'bias': err.mean(), 'sd': sd, 'rmse': float(np.sqrt((err ** 2).mean())),
        'mae': err.abs().mean(), 'loa_low': err.mean() - 1.96 * sd, 'loa_high': err.mean() + 1.96 * sd,
        'r': float(np.corrcoef(ref, ref + err)[0, 1]),
        'median': float(np.median(values)), 'mad': float(1.4826 * np.median(np.abs(values - np.median(values)))),
        'max_abs': float(np.abs(values).max()),
        'rmse_ci_low': float(np.percentile(boot, 2.5)), 'rmse_ci_high': float(np.percentile(boot, 97.5)),
        'within_1': float(np.mean(np.abs(values) <= 1.0)),
    }


def paired_comparison(trees: pd.DataFrame, references: Sequence[tuple[str, str]], label: str,
                      first: str = 'lidar_height', second: str = 'photo_height',
                      n_boot: int = 5000, seed: int = 0) -> list[dict[str, object]]:
    """Two sensors on identical trees: errors of each, RMSE difference with bootstrap interval, Wilcoxon test.

    `references` is a list of (column, label). The RMSE difference is first minus
    second; trees are resampled together, so the interval reflects the paired design.
    """
    from scipy.stats import wilcoxon

    rng = np.random.default_rng(seed)
    rows = []
    for ref_col, ref_label in references:
        d = trees[trees[ref_col].notna() & trees[first].notna() & trees[second].notna()]
        if len(d) < 2:
            continue
        e1, e2 = d[first] - d[ref_col], d[second] - d[ref_col]
        idx = rng.integers(0, len(d), size=(n_boot, len(d)))
        diff = np.sqrt((e1.to_numpy()[idx] ** 2).mean(1)) - np.sqrt((e2.to_numpy()[idx] ** 2).mean(1))
        p = wilcoxon(e1.abs(), e2.abs()).pvalue if len(d) > 5 else np.nan
        row = {'trees': label, 'reference': ref_label, 'n': len(d)}
        for prefix, err in [('lidar', e1), ('photo', e2)]:
            sd = err.std(ddof=1)
            row.update({f'{prefix}_bias': err.mean(), f'{prefix}_sd': sd, f'{prefix}_mae': err.abs().mean(),
                        f'{prefix}_rmse': float(np.sqrt((err ** 2).mean())),
                        f'{prefix}_loa_low': err.mean() - 1.96 * sd, f'{prefix}_loa_high': err.mean() + 1.96 * sd})
        row.update({'rmse_diff': diff.mean(), 'rmse_diff_ci_low': np.percentile(diff, 2.5),
                    'rmse_diff_ci_high': np.percentile(diff, 97.5), 'wilcoxon_p': p})
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Detection rates
# ---------------------------------------------------------------------------

def wilson(k, n, z: float = 1.96):
    """95 % Wilson interval of a share k / n (works on arrays)."""
    k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
    centre = (k + z * z / 2) / (n + z * z)
    half = z * np.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
    return centre - half, centre + half


def f1(recall: float, precision: float) -> float:
    return 2 * recall * precision / (recall + precision)


def rate_table(groups: pd.Series, table: pd.DataFrame) -> pd.DataFrame:
    """Counts per detection class and detection rates for a grouping of the field trees."""
    out = pd.crosstab(groups, table['detection'], dropna=False).reindex(columns=nc.CLASSES, fill_value=0)
    out.insert(0, 'field_trees', out.sum(axis=1))
    out['detection_rate'] = (out['detected'] / out['field_trees']).round(3)
    out['incl_uncertain'] = ((out['detected'] + out['uncertain']) / out['field_trees']).round(3)
    return out[out['field_trees'] > 0]


def detection_per_parzelle(result: pd.DataFrame, in_plot: pd.DataFrame, corner_rmse: pd.Series) -> pd.DataFrame:
    """Field trees per class, LiDAR instances and their count ratio, per Parzelle.

    `in_plot` are the LiDAR instances with their base inside a Parzelle (columns
    parzelle, instance, tree_id); `corner_rmse` is indexed by the Parzelle as string.
    """
    per_plot = rate_table(result['parzelle'].astype(int).rename('parzelle'), result)
    lidar_counts = in_plot.groupby(in_plot['parzelle'].astype(int)).agg(
        lidar_instances=('instance', 'size'), lidar_without_field_tree=('tree_id', lambda s: int(s.isna().sum())))
    per_plot = per_plot.join(lidar_counts).fillna({'lidar_instances': 0, 'lidar_without_field_tree': 0})
    per_plot[['lidar_instances', 'lidar_without_field_tree']] = per_plot[['lidar_instances', 'lidar_without_field_tree']].astype(int)
    per_plot['count_ratio'] = (per_plot['lidar_instances'] / per_plot['field_trees']).round(2)
    per_plot['corner_rmse_m'] = per_plot.index.map(lambda p: round(corner_rmse[str(p)], 2))
    return per_plot


# ---------------------------------------------------------------------------
# Harvest record against the LiDAR
# ---------------------------------------------------------------------------

LIDAR_STATES = ['gone', 'unclear', 'still standing', f'no instance within {nc.HIGH_CONFIDENCE_DISTANCE_M} m']
RECORD = {True: 'marked for removal 2026', False: 'not marked'}


def lidar_state(removed_fraction, has_instance=True) -> np.ndarray:
    """'gone' / 'still standing' / 'unclear' from the share of an instance's points gone after the harvest."""
    removed_fraction = np.asarray(removed_fraction, dtype=float)
    has_instance = np.broadcast_to(np.asarray(has_instance, dtype=bool), removed_fraction.shape)
    return np.select([~has_instance, removed_fraction > nc.GONE_ABOVE, removed_fraction < nc.STANDING_BELOW],
                     [LIDAR_STATES[3], 'gone', 'still standing'], default='unclear')


def record_vs_lidar(table: pd.DataFrame) -> pd.DataFrame:
    """Harvest record (rows) against the LiDAR state of the instance at the stem (columns)."""
    out = pd.crosstab(table['record'], table['lidar'], dropna=False).reindex(
        index=list(RECORD.values()), columns=LIDAR_STATES, fill_value=0)
    out['field_trees'] = out.sum(axis=1)
    return out


# ---------------------------------------------------------------------------
# Independent checks of a matching (algorithm comparison)
# ---------------------------------------------------------------------------

def harvest_check(matches: pd.DataFrame) -> dict[str, float]:
    """Found trees with a clear instance state: does the state agree with the harvest record?"""
    d = matches[matches['instance'].notna() & (matches['base_dist_m'] <= nc.FOUND_DISTANCE_M)]
    gone, standing = d['removed_fraction'] > nc.GONE_ABOVE, d['removed_fraction'] < nc.STANDING_BELOW
    clear = d[gone | standing]
    agree = (clear['removed_fraction'] > nc.GONE_ABOVE) == clear['removed_2026']
    return {'found': len(d), 'clear_state': len(clear), 'harvest_agree': int(agree.sum()),
            'harvest_disagree': int((~agree).sum()), 'harvest_agreement': agree.mean()}


def stem_check(matches: pd.DataFrame, stem_instance: pd.Series) -> dict[str, float]:
    """Is the matched instance the one that holds the LiDAR stem found at the field position?

    `stem_instance` is indexed by tree_id and holds only trees with a reliable stem.
    """
    d = matches.set_index('tree_id').loc[stem_instance.index]
    matched = d['instance'].notna()
    same = (d['instance'].astype('Float64') == stem_instance.astype('Float64')).fillna(False)
    return {'trees_with_lidar_stem': len(d), 'matched_to_stem_instance': int(same.sum()),
            'matched_to_other_instance': int((matched & ~same).sum()), 'not_matched': int((~matched).sum()),
            'stem_agreement': same.sum() / matched.sum()}


def auc(values: pd.Series, reference: pd.Series) -> tuple[float, float]:
    """Chance that a value of the first group is higher than one of the second (0.5 = no difference), and its p-value."""
    from scipy.stats import mannwhitneyu

    a, b = values.dropna(), reference.dropna()
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan
    test = mannwhitneyu(a, b)
    return float(test.statistic / (len(a) * len(b))), float(test.pvalue)
