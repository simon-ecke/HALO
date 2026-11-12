"""Statistics on the per-tree tables (no point clouds, no rasters).

Pure functions: DataFrames in, DataFrames or dicts out. No site-specific value
lives here; thresholds come in as keyword arguments (the stage runner reads
them from the config). Only `load_tables` touches the file system.

Rules that hold throughout:
- the Massband (tape on the felled tree) is the height reference; Vertex and the
  mean of both are evaluated separately and never replace it;
- every accuracy result is given for the subsets `all_matched` (detected plus
  uncertain) and `detected_only`, with bias, RMSE, median, MAD and a bootstrap CI;
- no tree is dropped because its height differs;
- matches with a manual method are a separate tier, not SegmentAnyTree results.
Errors are always estimate minus reference.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

CLASSES = ['detected', 'uncertain', 'not detected']
SUBSETS = ('all_matched', 'detected_only')
SUBSET_TEXT = {'all_matched': 'matched trees (detected and uncertain)', 'detected_only': 'detected trees'}
LIDAR_STATES = ['gone', 'unclear', 'still standing']
RECORD = {True: 'marked for removal 2026', False: 'not marked'}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_tables(tables_dir) -> dict[str, pd.DataFrame]:
    """Read the per-tree tables that the statistics need from one tables folder."""
    d = Path(tables_dir)
    str_id = {'tree_id': str}

    def read(name):
        return pd.read_csv(d / name, dtype=str_id)

    import geopandas as gpd      # imported late: only the instance layer needs it
    inst = gpd.read_file(d / 'detection_all_trees.gpkg', layer='lidar_instances')
    instances = pd.DataFrame(inst.drop(columns='geometry'))
    return {
        'result': read('field_trees_detection.csv'), 'values': read('height_dbh_all_trees.csv'),
        'tiers': read('match_tiers.csv'), 'causes': read('match_causes.csv').fillna({'cause': '', 'attributed_to': ''}),
        'profile': read('tree_profile_all.csv'), 'plain': read('match_distance_only.csv'),
        'control': read('chance_control_matches.csv'), 'base_tops': pd.read_csv(d / 'chm_baseline_tops.csv'),
        'base_match': read('chm_baseline_matches.csv'), 'instances': instances,
    }


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def slug(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', str(text).lower()).strip('_')


def wilson(k, n, z: float = 1.96):
    """95 % Wilson interval of a share k / n (works on arrays)."""
    k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
    centre = (k + z * z / 2) / (n + z * z)
    half = z * np.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
    return centre - half, centre + half


def f1(recall: float, precision: float) -> float:
    return 2 * recall * precision / (recall + precision)


def agreement(reference: pd.Series, estimate: pd.Series, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """n, bias, SD, RMSE, MAE, limits of agreement, r and robust measures; error = estimate - reference.

    median / mad describe the typical tree and are not moved by a few large errors
    (mad = median absolute deviation from the median, scaled to match an SD for
    normal errors). `<measure>_ci_low/high` are the 95 % bootstrap intervals (trees
    resampled; the same resamples for all measures; seed and count as in the notebook).
    """
    ok = reference.notna() & estimate.notna()
    ref = reference[ok].astype(float)
    err = (estimate[ok] - reference[ok]).astype(float)
    keys = ['bias', 'sd', 'rmse', 'mae', 'loa_low', 'loa_high', 'r', 'median', 'mad', 'max_abs',
            'rmse_ci_low', 'rmse_ci_high', 'within_1',
            'bias_ci_low', 'bias_ci_high', 'median_ci_low', 'median_ci_high', 'mad_ci_low', 'mad_ci_high']
    if len(err) < 2:
        return {'n': int(len(err)), **{k: np.nan for k in keys}}

    sd = err.std(ddof=1)
    values = err.to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(values), size=(n_boot, len(values)))
    sample = values[idx]
    boot_rmse = np.sqrt((sample ** 2).mean(axis=1))
    boot_bias = sample.mean(axis=1)
    boot_median = np.median(sample, axis=1)
    boot_mad = 1.4826 * np.median(np.abs(sample - boot_median[:, None]), axis=1)

    def ci(boot):
        return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    out = {
        'n': int(len(err)), 'bias': err.mean(), 'sd': sd, 'rmse': float(np.sqrt((err ** 2).mean())),
        'mae': err.abs().mean(), 'loa_low': err.mean() - 1.96 * sd, 'loa_high': err.mean() + 1.96 * sd,
        'r': float(np.corrcoef(ref, ref + err)[0, 1]),
        'median': float(np.median(values)), 'mad': float(1.4826 * np.median(np.abs(values - np.median(values)))),
        'max_abs': float(np.abs(values).max()), 'within_1': float(np.mean(np.abs(values) <= 1.0)),
    }
    out['rmse_ci_low'], out['rmse_ci_high'] = ci(boot_rmse)
    out['bias_ci_low'], out['bias_ci_high'] = ci(boot_bias)
    out['median_ci_low'], out['median_ci_high'] = ci(boot_median)
    out['mad_ci_low'], out['mad_ci_high'] = ci(boot_mad)
    return out


def rate_table(groups: pd.Series, table: pd.DataFrame) -> pd.DataFrame:
    """Counts per detection class and detection rates for a grouping of the field trees."""
    out = pd.crosstab(groups, table['detection'], dropna=False).reindex(columns=CLASSES, fill_value=0)
    out.insert(0, 'field_trees', out.sum(axis=1))
    out['detection_rate'] = (out['detected'] / out['field_trees']).round(3)
    out['incl_uncertain'] = ((out['detected'] + out['uncertain']) / out['field_trees']).round(3)
    return out[out['field_trees'] > 0]


def lidar_state(removed_fraction, gone_above: float, standing_below: float) -> np.ndarray:
    """'gone' / 'still standing' / 'unclear' from the share of an instance's points gone after the harvest."""
    removed_fraction = np.asarray(removed_fraction, dtype=float)
    return np.select([removed_fraction > gone_above, removed_fraction < standing_below],
                     ['gone', 'still standing'], default='unclear')


# ---------------------------------------------------------------------------
# Derivations (done inline in the notebook)
# ---------------------------------------------------------------------------

def derive_field(result: pd.DataFrame, *, standing_height_dm_above: float, dbh_plausible_mm: Sequence[float]) -> pd.DataFrame:
    """Add height_felled (= Massband, the reference), height_felled_mean, height_standing and dbh_field_mm."""
    r = result.copy()
    r['detection'] = pd.Categorical(r['detection'], categories=CLASSES)
    r['removal_consistent'] = r['removal_consistent'].astype('boolean')
    sheet = r['height_standing_sheet']
    r['height_standing'] = sheet.where(sheet <= standing_height_dm_above, sheet / 10)
    r['height_felled'] = r['height_massband']
    r['height_felled_mean'] = r[['height_vertex', 'height_massband']].mean(axis=1)
    r['dbh_field_mm'] = r['dbh_mm'].where(r['dbh_mm'].between(*dbh_plausible_mm))
    return r


def build_everyone(result: pd.DataFrame, values: pd.DataFrame, tiers: pd.DataFrame,
                   causes: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    """One row per field tree with the matching result, LiDAR DBH, tiers, cause and stand position."""
    tier_part = (tiers[['tree_id', 'auto_height', 'auto_confidence', 'auto_method', 'sat_height', 'height_gate_ok']]
                 .assign(tier_confidence=tiers['confidence'], tier_method=tiers['method'], tier_height=tiers['lidar_height']))
    e = (result.merge(values, on='tree_id').merge(tier_part, on='tree_id')
         .merge(causes[['tree_id', 'cause', 'attributed_to']], on='tree_id')
         .merge(profile[['tree_id', 'rel_dbh', 'nn_dist_m', 'hegyi']], on='tree_id'))
    if 'visible_from_above' not in e:       # photogrammetry is not ported: no tree counts as visible from above
        e['visible_from_above'] = False
    e['visible_from_above'] = e['visible_from_above'].fillna(False).astype(bool)
    return e


def in_plot_instances(instances: pd.DataFrame) -> pd.DataFrame:
    return instances[instances['in_plot'].astype(bool)]


# ---------------------------------------------------------------------------
# Section 1: data counts
# ---------------------------------------------------------------------------

def inventory_counts(result: pd.DataFrame) -> dict[str, object]:
    """Counts of trees, Parzellen, DBH and heights (`result` must have passed `derive_field`)."""
    species = result['species'].value_counts()
    both = int((result['height_vertex'].notna() & result['height_massband'].notna()).sum())
    return {
        'field_trees': len(result), 'parzellen': int(result['parzelle'].nunique()),
        'main_species': str(species.index[0]), 'main_species_share': float(species.iloc[0] / len(result)),
        'plausible_dbh': int(result['dbh_field_mm'].notna().sum()), 'sample_trees': int(result['has_height'].sum()),
        'massband': int(result['height_massband'].notna().sum()), 'vertex': int(result['height_vertex'].notna().sum()),
        'both_methods': both, 'standing_height': int(result['height_standing'].notna().sum()),
        'any_height': int((result['has_height'] | result['height_standing'].notna()).sum()),
    }


# ---------------------------------------------------------------------------
# Section 2.2: distance against the chance level
# ---------------------------------------------------------------------------

def _within(table: pd.DataFrame, d: float) -> pd.Series:
    ok = (table['base_dist_m'] <= d) & table['removal_consistent'].eq(True)
    return ok.groupby(table['shift']).mean() if 'shift' in table.columns else pd.Series([ok.mean()])


def match_distance(result: pd.DataFrame, control: pd.DataFrame, *, high_confidence_distance_m: float,
                   max_match_distance_m: float, thresholds: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 3.0)) -> dict[str, object]:
    """Stem-to-base distance of the matches, and the share matched within a distance for real and shifted stems."""
    control = control.copy()
    control['removal_consistent'] = control['removal_consistent'].astype('boolean')
    matched = result['base_dist_m'].dropna()
    curve = pd.DataFrame({
        'max_distance_m': list(thresholds),
        'field_trees': [_within(result, d).mean() for d in thresholds],
        'shifted_control': [_within(control, d).mean() for d in thresholds],
        'control_min': [_within(control, d).min() for d in thresholds],
        'control_max': [_within(control, d).max() for d in thresholds]})
    row = curve[np.isclose(curve['max_distance_m'], high_confidence_distance_m)].iloc[0]
    return {
        'median_m': float(matched.median()), 'p95_m': float(matched.quantile(0.95)), 'n_matched': int(len(matched)),
        'real_within': float(row['field_trees']), 'control_within': float(row['shifted_control']),
        'control_min': float(row['control_min']), 'control_max': float(row['control_max']),
        'control_any_instance': float((control['confidence'] != 'unmatched').mean()),
        'max_match_distance_m': max_match_distance_m, 'curve': curve,
    }


# ---------------------------------------------------------------------------
# Sections 3.1, 3.2: detection
# ---------------------------------------------------------------------------

def detection_overall(result: pd.DataFrame, in_plot: pd.DataFrame) -> dict[str, float]:
    """Counts and rates seen from the field side (recall) and the LiDAR side (precision)."""
    n_field = len(result)
    n_det = int((result['detection'] == 'detected').sum())
    n_unc = int((result['detection'] == 'uncertain').sum())
    n_lidar = len(in_plot)
    n_lidar_det = int((in_plot['detection'] == 'detected').sum())
    n_lidar_unc = int((in_plot['detection'] == 'uncertain').sum())
    prec_det, prec_all = n_lidar_det / n_lidar, (n_lidar_det + n_lidar_unc) / n_lidar
    return {
        'field_trees': n_field, 'lidar_instances': n_lidar, 'detected': n_det, 'uncertain': n_unc,
        'not_detected': n_field - n_det - n_unc, 'lidar_without_field_tree': int(in_plot['tree_id'].isna().sum()),
        'recall_detected': n_det / n_field, 'recall_all_matched': (n_det + n_unc) / n_field,
        'precision_detected': prec_det, 'precision_all_matched': prec_all,
        'f1_detected': f1(n_det / n_field, prec_det), 'f1_all_matched': f1((n_det + n_unc) / n_field, prec_all),
        'count_ratio': n_lidar / n_field, 'n_lidar_det': n_lidar_det,
    }


def detection_per_parzelle(result: pd.DataFrame, in_plot: pd.DataFrame) -> pd.DataFrame:
    """Field trees per class, LiDAR instances and their count ratio, per Parzelle."""
    per_plot = rate_table(result['parzelle'].astype(int).rename('parzelle'), result)
    counts = in_plot.groupby(in_plot['parzelle'].astype(int)).agg(
        lidar_instances=('instance', 'size'), lidar_without_field_tree=('tree_id', lambda s: int(s.isna().sum())))
    per_plot = per_plot.join(counts).fillna({'lidar_instances': 0, 'lidar_without_field_tree': 0})
    cols = ['lidar_instances', 'lidar_without_field_tree']
    per_plot[cols] = per_plot[cols].astype(int)
    per_plot['count_ratio'] = (per_plot['lidar_instances'] / per_plot['field_trees']).round(2)
    return per_plot


def detection_groups(result: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Detection rates by block (halves of the site) and by mirrored / not mirrored plot orientation."""
    return {'block': rate_table(result['block'], result),
            'y_mirrored': rate_table(result['y_mirrored'].astype(bool).map({True: 'mirrored', False: 'as_sheet'}).rename('orientation'), result)}


# ---------------------------------------------------------------------------
# Section 3.6: CHM baseline
# ---------------------------------------------------------------------------

def baseline_comparison(result: pd.DataFrame, profile: pd.DataFrame, causes: pd.DataFrame, control: pd.DataFrame,
                        base_tops: pd.DataFrame, base_match: pd.DataFrame, in_plot: pd.DataFrame, *,
                        high_confidence_distance_m: float, top_found_distance_m: float,
                        sat_tops_name: str = 'SegmentAnyTree instance tops',
                        sat_base_name: str = 'SegmentAnyTree, base matching') -> pd.DataFrame:
    """Local maxima on the CHM and SegmentAnyTree tops against SegmentAnyTree base matching, per method."""
    base_match = base_match.copy()
    base_match['found'] = base_match['top_dist_m'] <= top_found_distance_m
    methods = list(dict.fromkeys(base_tops['method']))
    methods = [m for m in methods if m != sat_tops_name] + ([sat_tops_name] if sat_tops_name in methods else [])
    n_lidar, n_lidar_det = len(in_plot), int((in_plot['detection'] == 'detected').sum())

    rel_fifth = pd.qcut(profile.set_index('tree_id')['rel_dbh'], 5, labels=False)
    over_fifth = pd.qcut(causes.set_index('tree_id')['overtopped_by_m'], 5, labels=False)
    ref = result[['tree_id', 'has_height', 'height_felled', 'height_standing', 'detection', 'lidar_height']].assign(
        rel_fifth=lambda d: d['tree_id'].map(rel_fifth), over_fifth=lambda d: d['tree_id'].map(over_fifth))
    sample_ref = ref['has_height'] & ref['height_felled'].notna()
    standing_ref = ~ref['has_height'] & ref['height_standing'].notna()

    def row(name, found, chance, n_units, n_matched_units, height):
        k, n = int(found.sum()), len(found)
        low, high = wilson(k, n)
        recall, precision = k / n, n_matched_units / n_units
        sa = agreement(ref.loc[sample_ref & found, 'height_felled'], height[sample_ref & found])
        st = agreement(ref.loc[standing_ref & found, 'height_standing'], height[standing_ref & found])
        return {'method': name, 'units': n_units, 'field_trees_found': k, 'found': recall, 'ci_low': float(low), 'ci_high': float(high),
                'chance': chance, 'precision': precision, 'F1': f1(recall, precision),
                'found_smallest_rel_dbh_fifth': found[ref['rel_fifth'] == 0].mean(),
                'found_largest_rel_dbh_fifth': found[ref['rel_fifth'] == 4].mean(),
                'found_most_overtopped_fifth': found[ref['over_fifth'] == 4].mean(),
                'massband_n': sa['n'], 'massband_rmse': sa['rmse'], 'massband_median': sa['median'],
                'standing_n': st['n'], 'standing_rmse': st['rmse'], 'standing_median': st['median']}

    rows = []
    for name in methods:
        m = base_match[base_match['method'] == name]
        real = ref[['tree_id']].merge(m[m['shift'] == 'none'], on='tree_id', how='left')
        found = real['found'].fillna(False).astype(bool)
        chance = m[m['shift'] != 'none'].groupby('shift')['found'].mean().mean()
        tops_in = base_tops[(base_tops['method'] == name) & base_tops['parzelle'].notna()]
        matched = int(tops_in['top_id'].isin(real.loc[found, 'top_id']).sum())
        rows.append(row(name, found, chance, len(tops_in), matched, real['top_height'].where(found)))

    control = control.copy()
    shifted = control.assign(found=(control['base_dist_m'] <= high_confidence_distance_m)
                             & control['removal_consistent'].astype('boolean').fillna(False).astype(bool))
    sat_detected = ref['detection'] == 'detected'
    rows.append(row(sat_base_name, sat_detected, shifted.groupby('shift')['found'].mean().mean(),
                    n_lidar, n_lidar_det, ref['lidar_height'].where(sat_detected)))
    return pd.DataFrame(rows).set_index('method')


# ---------------------------------------------------------------------------
# Section 4.1: causes
# ---------------------------------------------------------------------------

def cause_counts(causes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trees per (detection class, attribution, cause), and per attribution; only uncertain and not detected trees have a cause."""
    missed = causes[causes['cause'] != '']
    table = (missed.groupby(['detection', 'attributed_to', 'cause']).size().rename('trees').reset_index()
             .sort_values(['detection', 'trees'], ascending=[True, False]))
    attribution = pd.crosstab(missed['attributed_to'], missed['detection'])
    attribution['both'] = attribution.sum(axis=1)
    return table, attribution


# ---------------------------------------------------------------------------
# Section 6.1: harvest record against the LiDAR
# ---------------------------------------------------------------------------

def record_vs_lidar(plain: pd.DataFrame, no_instance_label: str) -> pd.DataFrame:
    """Harvest record (rows) against the LiDAR state of the instance at the stem (columns)."""
    states = LIDAR_STATES + [no_instance_label]
    out = pd.crosstab(plain['record'], plain['lidar'], dropna=False).reindex(
        index=list(RECORD.values()), columns=states, fill_value=0)
    out['field_trees'] = out.sum(axis=1)
    return out


def harvest_summary(plain: pd.DataFrame, in_plot: pd.DataFrame, *, gone_above: float, standing_below: float) -> dict[str, float]:
    """Does the harvest record agree with the LiDAR? Uses the distance-only assignment (an independent check)."""
    clear = plain[plain['lidar'].isin(['gone', 'still standing'])]
    marked, gone = clear['removed_2026'].astype(bool), clear['lidar'] == 'gone'
    state = pd.Series(lidar_state(in_plot['removed_fraction'], gone_above, standing_below), index=in_plot.index)
    no_tree = in_plot['tree_id'].isna()
    return {
        'field_trees': len(plain), 'marked_for_removal': int(plain['removed_2026'].sum()), 'with_clear_lidar_state': len(clear),
        'record_and_lidar_agree': int((marked == gone).sum()), 'agreement': float((marked == gone).mean()),
        'marked_and_gone': int((marked & gone).sum()), 'marked_but_standing': int((marked & ~gone).sum()),
        'not_marked_but_gone': int((~marked & gone).sum()), 'not_marked_and_standing': int((~marked & ~gone).sum()),
        'share_of_marked_found_gone': float(gone[marked].mean()), 'share_of_gone_that_are_marked': float(marked[gone].mean()),
        'lidar_instances_gone': int((state == 'gone').sum()), 'lidar_instances_gone_without_field_tree': int(((state == 'gone') & no_tree).sum()),
    }


# ---------------------------------------------------------------------------
# Section 7: height accuracy
# ---------------------------------------------------------------------------

def subset_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """The two subsets every accuracy is reported for."""
    return {'all_matched': frame['detection'] != 'not detected', 'detected_only': frame['detection'] == 'detected'}


def reference_sets(everyone: pd.DataFrame) -> list[tuple[str, pd.Series, str]]:
    """Trees with a field height: sample trees (Massband, the reference), and trees with a standing height (weaker)."""
    has_standing = everyone['height_standing'].notna() & ~everyone['has_height']
    return [('sample trees, Maßband length', everyone['has_height'] & everyone['height_felled'].notna(), 'height_felled'),
            ('other harvested trees, standing height', has_standing & everyone['removed_2026'], 'height_standing'),
            ('remaining stand, standing height', has_standing & ~everyone['removed_2026'], 'height_standing')]


def height_funnel(everyone: pd.DataFrame) -> pd.DataFrame:
    """From field tree to usable height: where trees drop out."""
    rows = []
    for label, selected, ref_col in reference_sets(everyone):
        d = everyone[selected]
        err = (d['lidar_height'] - d[ref_col])[d['detection'] == 'detected']
        rows.append({'reference set': label, 'trees with a field height': len(d),
                     'not detected': int((d['detection'] == 'not detected').sum()), 'uncertain': int((d['detection'] == 'uncertain').sum()),
                     'detected': int((d['detection'] == 'detected').sum()),
                     'detected, LiDAR within 2 m': int((err.abs() <= 2).sum()), 'detected, LiDAR within 1 m': int((err.abs() <= 1).sum())})
    out = pd.DataFrame(rows).set_index('reference set')
    out['share within 1 m, of all trees with a height'] = (out['detected, LiDAR within 1 m'] / out['trees with a field height']).round(3)
    return out


def height_accuracy_by_class(everyone: pd.DataFrame) -> pd.DataFrame:
    """LiDAR minus field height per reference set, for all_matched, detected_only and detected-and-visible-from-above."""
    rows = []
    for label, selected, ref_col in reference_sets(everyone):
        d = everyone[selected]
        masks = subset_masks(d)
        masks['detected_visible'] = masks['detected_only'] & d['visible_from_above']
        for subset, keep in masks.items():
            rows.append({'reference set': label, 'subset': subset, **agreement(d.loc[keep, ref_col], d.loc[keep, 'lidar_height'])})
    return pd.DataFrame(rows)


def height_accuracy_by_reference(everyone: pd.DataFrame) -> pd.DataFrame:
    """The three references separately on the sample trees: Massband (reference), Vertex, mean of both.

    `trees` is 'all with this measurement' or 'measured with both'. Never mixed, never replacing the reference.
    """
    m = everyone[everyone['has_height']]
    has_m, has_v = m['height_massband'].notna(), m['height_vertex'].notna()
    variants = [('massband', 'height_massband', has_m), ('mean', 'height_felled_mean', has_m | has_v), ('vertex', 'height_vertex', has_v)]
    rows = []
    for subset, keep in subset_masks(m).items():
        for ref, col, has in variants:
            for trees, sel in [('all_with_measurement', keep & has), ('measured_with_both', keep & has_m & has_v)]:
                rows.append({'subset': subset, 'reference': ref, 'trees': trees,
                             **agreement(m.loc[sel, col], m.loc[sel, 'lidar_height'])})
        # Vertex trees that have no tape length: not comparable with the reference, shown on their own
        sel = keep & has_v & ~has_m
        rows.append({'subset': subset, 'reference': 'vertex_without_massband', 'trees': 'all_with_measurement',
                     **agreement(m.loc[sel, 'height_vertex'], m.loc[sel, 'lidar_height'])})
    return pd.DataFrame(rows)


def field_instrument_agreement(everyone: pd.DataFrame) -> dict[str, float]:
    """Vertex minus Massband on the trees that have both: the noise floor of the terrestrial measurement."""
    m = everyone[everyone['has_height']]
    return agreement(m['height_massband'], m['height_vertex'])


def height_tiers(everyone: pd.DataFrame) -> pd.DataFrame:
    """LiDAR against each reference by result tier. Manual picks and manual segments are separate tiers, not SAT results."""
    measured = everyone[everyone['has_height']].copy()
    measured['field_mean'] = measured['height_felled_mean']
    is_seg = measured['tier_method'] == 'manual_segmentation'
    is_pick = (measured['tier_confidence'] == 'manual') & ~is_seg
    tiers = [
        ('SAT instance, all with an instance', measured, 'sat_height'),
        ('SAT instance, detected', measured[measured['detection'] == 'detected'], 'sat_height'),
        ('SAT instance, high confidence incl. height gate (biased)', measured[measured['auto_confidence'] == 'high'], 'sat_height'),
        ('Column profile fallback (not SAT)', measured[measured['auto_method'] == 'column_profile'], 'auto_height'),
        ('SAT instance picked manually', measured[is_pick], 'tier_height'),
        ('Manual segmentation (reference, not SAT)', measured[is_seg], 'tier_height'),
    ]
    rows = []
    for ref_col, label in [('height_vertex', 'Vertex'), ('height_massband', 'Maßband'), ('field_mean', 'mean of both')]:
        for tier, d, col in tiers:
            rows.append({'tier': tier, 'comparison': f'LiDAR vs {label}', **agreement(d[ref_col], d[col])})
    out = pd.DataFrame(rows)
    out.attrs['auto_confidence_counts'] = measured['auto_confidence'].value_counts().to_dict()
    out.attrs['manual_picked'], out.attrs['manual_segmented'] = int(is_pick.sum()), int(is_seg.sum())
    return out[out['n'] > 0]


def _rmse(e) -> float:
    return float(np.sqrt(np.mean(np.square(e)))) if len(e) else np.nan


def leave_one_out(everyone: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-out influence and jackknife standard error of the RMSE, sample trees against the Massband."""
    s = everyone[reference_sets(everyone)[0][1]]
    masks = {'all_matched': s['detection'] != 'not detected', 'detected_only': s['detection'] == 'detected',
             'detected_visible': (s['detection'] == 'detected') & s['visible_from_above']}
    influence_rows, jack_rows = [], []
    for subset, keep in masks.items():
        d = s[keep]
        if len(d) < 2:                  # e.g. detected_visible without photogrammetry
            continue
        e = (d['lidar_height'] - d['height_felled']).to_numpy()
        full = _rmse(e)
        loo = np.array([_rmse(np.delete(e, i)) for i in range(len(e))])
        med = np.array([np.median(np.delete(e, i)) for i in range(len(e))])
        se = float(np.sqrt((len(e) - 1) / len(e) * np.sum((loo - loo.mean()) ** 2)))
        jack_rows.append({'subset': subset, 'n': len(e), 'rmse': full, 'jackknife_se': se,
                          'rmse_without_largest_influence': loo.min(), 'median': float(np.median(e)),
                          'median_loo_min': med.min(), 'median_loo_max': med.max()})
        for i, tree_id in enumerate(d['tree_id']):
            influence_rows.append({'subset': subset, 'tree_id': tree_id, 'error': e[i], 'rmse_full': full,
                                   'rmse_without': loo[i], 'rmse_change': loo[i] - full})
    return pd.DataFrame(influence_rows), pd.DataFrame(jack_rows).set_index('subset')


# ---------------------------------------------------------------------------
# Section 8.1: LiDAR DBH
# ---------------------------------------------------------------------------

def dbh_groups(everyone: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The harvested trees, split into the sample trees and the other harvested trees."""
    trees = everyone[everyone['removed_2026'] | everyone['has_height']]
    return {'sample trees': trees[trees['has_height']], 'other harvested trees': trees[~trees['has_height']]}


def dbh_accuracy(everyone: pd.DataFrame, *, ellipse_axis_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """LiDAR minus field DBH (cm), trees with an ok circle fit, for both subsets and both shapes (circle, ellipse).

    Returns the accuracy table (group, subset, trees, shape) and the robust measures of the ok fits.
    """
    acc, robust = [], []
    for group, g in dbh_groups(everyone).items():
        for subset, keep in subset_masks(g).items():
            det = g[keep & g['dbh_field_mm'].notna()]
            ok = det[(det['dbh_quality'] == 'ok') & det['dbh_ellipse_mm'].notna()]
            at_limit = ok['ellipse_axis_ratio'] > ellipse_axis_ratio - 0.02
            for trees, d in [('ok fits', ok), ('ok fits, ellipse not at the axis-ratio limit', ok[~at_limit]),
                             ('ok and weak fits', det[det['dbh_quality'].isin(['ok', 'weak'])])]:
                for shape, col in [('circle', 'dbh_lidar_mm'), ('ellipse', 'dbh_ellipse_mm')]:
                    acc.append({'group': group, 'subset': subset, 'trees': trees, 'shape': shape,
                                **agreement(d['dbh_field_mm'] / 10, d[col] / 10)})
            for shape, col in [('circle', 'dbh_lidar_mm'), ('ellipse', 'dbh_ellipse_mm')]:
                err = (ok[col] - ok['dbh_field_mm']).dropna() / 10
                robust.append({'group': group, 'subset': subset, 'shape': shape, 'n': len(err),
                               'median_abs_error_cm': err.abs().median(), 'within_2cm': float(np.mean(err.abs() <= 2)),
                               'within_5cm': float(np.mean(err.abs() <= 5)), 'errors_above_10cm': int((err.abs() > 10).sum())})
    return pd.DataFrame(acc), pd.DataFrame(robust)


# ---------------------------------------------------------------------------
# Everything at once: entries for stats.json
# ---------------------------------------------------------------------------

def _entry(id_, value, unit, n, statement, subset=None, ci=(None, None)):
    def clean(v):
        if v is None:
            return None
        v = float(v)
        return None if np.isnan(v) else v
    return {'id': id_, 'value': clean(value), 'unit': unit, 'n': None if n is None else int(n),
            'ci_low': clean(ci[0]), 'ci_high': clean(ci[1]), 'subset': subset, 'statement': statement}


def _pct(v, digits=1):
    return f"{100 * v:.{digits}f}%"


def _accuracy_entries(prefix: str, label: str, subset: Optional[str], stats: dict, unit: str, digits: int = 2) -> list[dict]:
    """bias, RMSE, median and MAD of one comparison, each with its bootstrap interval."""
    n = stats['n']
    who = SUBSET_TEXT.get(subset, subset or 'trees')
    out = []
    if n < 2:
        return out
    f = f"{{:+.{digits}f}}"
    g = f"{{:.{digits}f}}"
    def interval(k):
        return f"{g.format(stats[k + '_ci_low'])} to {g.format(stats[k + '_ci_high'])} {unit}"
    out.append(_entry(f'{prefix}.bias', stats['bias'], unit, n, subset=subset, ci=(stats['bias_ci_low'], stats['bias_ci_high']),
        statement=f"{label}, on {n} {who}, the mean error is {f.format(stats['bias'])} {unit} (95 % bootstrap interval {interval('bias')}); a positive value means the first is higher."))
    out.append(_entry(f'{prefix}.rmse', stats['rmse'], unit, n, subset=subset, ci=(stats['rmse_ci_low'], stats['rmse_ci_high']),
        statement=f"{label}, on {n} {who}, the root-mean-square error is {g.format(stats['rmse'])} {unit} (95 % bootstrap interval {interval('rmse')}); a single large error can dominate it."))
    out.append(_entry(f'{prefix}.median', stats['median'], unit, n, subset=subset, ci=(stats['median_ci_low'], stats['median_ci_high']),
        statement=f"{label}, on {n} {who}, the median error is {f.format(stats['median'])} {unit} (95 % bootstrap interval {interval('median')}), the error of the typical tree."))
    out.append(_entry(f'{prefix}.mad', stats['mad'], unit, n, subset=subset, ci=(stats['mad_ci_low'], stats['mad_ci_high']),
        statement=f"{label}, on {n} {who}, the median absolute deviation of the errors is {g.format(stats['mad'])} {unit} (95 % bootstrap interval {interval('mad')}), a spread that a few large errors do not move."))
    return out


REFERENCE_TEXT = {'massband': 'ULS height against the Maßband length (the reference)',
                  'vertex': 'ULS height against the Vertex length (a separate check, not the reference)',
                  'mean': 'ULS height against the mean of Maßband and Vertex (a separate check, not the reference)'}


def compute_all(tables: dict[str, pd.DataFrame], *, high_confidence_distance_m: float, max_match_distance_m: float,
                gone_above: float, standing_below: float, dbh_plausible_mm: Sequence[float],
                standing_height_dm_above: float, top_found_distance_m: float,
                ellipse_axis_ratio: float) -> tuple[list[dict], dict[str, object]]:
    """All statistics. Returns the stats.json entries and the detail tables behind them."""
    result = derive_field(tables['result'], standing_height_dm_above=standing_height_dm_above,
                          dbh_plausible_mm=dbh_plausible_mm)
    inst = in_plot_instances(tables['instances'])
    everyone = build_everyone(result, tables['values'], tables['tiers'], tables['causes'], tables['profile'])
    entries: list[dict] = []
    add = entries.append

    # 1. data counts
    inv = inventory_counts(result)
    for key, text in [('field_trees', f"The inventory holds {inv['field_trees']} trees in {inv['parzellen']} Parzellen."),
                      ('plausible_dbh', f"{inv['plausible_dbh']} trees have a plausible field DBH."),
                      ('sample_trees', f"{inv['sample_trees']} sample trees were measured lying after felling."),
                      ('massband', f"{inv['massband']} trees have a Maßband length, which is the height reference."),
                      ('vertex', f"{inv['vertex']} trees have a Vertex length, measured on the lying tree."),
                      ('both_methods', f"{inv['both_methods']} trees have both a Maßband and a Vertex length."),
                      ('standing_height', f"{inv['standing_height']} trees have a height measured on the standing tree.")]:
        add(_entry(f'inventory.{key}', inv[key], 'trees', inv[key], text))
    add(_entry('inventory.parzellen', inv['parzellen'], 'Parzellen', inv['parzellen'], f"The trees stand in {inv['parzellen']} Parzellen."))
    add(_entry('inventory.main_species_share', inv['main_species_share'], 'share', inv['field_trees'],
               f"{_pct(inv['main_species_share'], 0)} of the trees are of the most frequent species ({inv['main_species']})."))

    # 2.2 match distance and chance level
    md = match_distance(result, tables['control'], high_confidence_distance_m=high_confidence_distance_m,
                        max_match_distance_m=max_match_distance_m)
    add(_entry('match.distance_median', md['median_m'], 'm', md['n_matched'],
               f"The median distance from a field stem to its matched LiDAR tree base is {md['median_m']:.2f} m."))
    add(_entry('match.distance_p95', md['p95_m'], 'm', md['n_matched'],
               f"95 % of the matched trees have the LiDAR base within {md['p95_m']:.2f} m of the field stem."))
    add(_entry('match.within_high_confidence_real', md['real_within'], 'share', len(result),
               f"{_pct(md['real_within'])} of the real stems have an instance base within {high_confidence_distance_m} m with agreeing harvest evidence."))
    add(_entry('match.within_high_confidence_control', md['control_within'], 'share', len(tables['control']),
               f"{_pct(md['control_within'])} of the shifted stems (the chance level) still get an instance base within {high_confidence_distance_m} m, "
               f"range {_pct(md['control_min'])} to {_pct(md['control_max'])} over the shift directions.",
               ci=(md['control_min'], md['control_max'])))
    add(_entry('match.control_any_instance', md['control_any_instance'], 'share', len(tables['control']),
               f"{_pct(md['control_any_instance'])} of the shifted stems get some instance within {max_match_distance_m:g} m."))

    # 3.1 detection overall
    ov = detection_overall(result, inst)
    n = ov['field_trees']
    for subset, k, rate in [('detected_only', ov['detected'], ov['recall_detected']),
                            ('all_matched', ov['detected'] + ov['uncertain'], ov['recall_all_matched'])]:
        lo, hi = wilson(k, n)
        add(_entry('detection.rate', rate, 'share', n, subset=subset, ci=(lo, hi),
                   statement=f"{k} of {n} field trees are found among the {SUBSET_TEXT[subset]} ({_pct(rate)}, 95 % Wilson interval {_pct(lo)} to {_pct(hi)})."))
    for subset, key in [('detected_only', 'precision_detected'), ('all_matched', 'precision_all_matched')]:
        add(_entry('detection.precision', ov[key], 'share', ov['lidar_instances'], subset=subset,
                   statement=f"{_pct(ov[key])} of the {ov['lidar_instances']} LiDAR instances inside the Parzellen belong to a field tree ({SUBSET_TEXT[subset]})."))
    for subset, key in [('detected_only', 'f1_detected'), ('all_matched', 'f1_all_matched')]:
        add(_entry('detection.f1', ov[key], 'score', n, subset=subset,
                   statement=f"The F1 score of detection (recall and precision combined) is {ov[key]:.2f} for the {SUBSET_TEXT[subset]}."))
    for key, text in [('detected', 'field trees are detected'), ('uncertain', 'field trees are uncertain'),
                      ('not_detected', 'field trees are not detected')]:
        add(_entry(f'detection.count_{key}', ov[key], 'trees', n, f"{ov[key]} of {n} {text}."))
    add(_entry('detection.lidar_instances', ov['lidar_instances'], 'instances', ov['lidar_instances'],
               f"{ov['lidar_instances']} LiDAR instances have their base inside a Parzelle (ratio to field trees {ov['count_ratio']:.2f})."))
    add(_entry('detection.lidar_without_field_tree', ov['lidar_without_field_tree'], 'instances', ov['lidar_instances'],
               f"{ov['lidar_without_field_tree']} of these instances have no field tree."))

    # 3.2 per Parzelle and by group
    per_plot = detection_per_parzelle(result, inst)
    rate = per_plot['detection_rate']
    add(_entry('detection.parzelle_rate_min', rate.min(), 'share', len(per_plot), subset='detected_only',
               statement=f"The lowest detection rate of a Parzelle is {_pct(rate.min(), 0)} (Parzelle {rate.idxmin()})."))
    add(_entry('detection.parzelle_rate_median', rate.median(), 'share', len(per_plot), subset='detected_only',
               statement=f"The median detection rate over the {len(per_plot)} Parzellen is {_pct(rate.median(), 0)}."))
    add(_entry('detection.parzelle_rate_max', rate.max(), 'share', len(per_plot), subset='detected_only',
               statement=f"The highest detection rate of a Parzelle is {_pct(rate.max(), 0)}."))
    ratio = per_plot['count_ratio']
    add(_entry('detection.parzelle_count_ratio_min', ratio.min(), 'ratio', len(per_plot),
               f"The smallest ratio of LiDAR instances to field trees in a Parzelle is {ratio.min():.2f}."))
    add(_entry('detection.parzelle_count_ratio_max', ratio.max(), 'ratio', len(per_plot),
               f"The largest ratio of LiDAR instances to field trees in a Parzelle is {ratio.max():.2f}."))
    groups = detection_groups(result)
    for kind, table in groups.items():
        for name, r in table.iterrows():
            add(_entry(f'detection.{kind}.{slug(name)}', r['detection_rate'], 'share', r['field_trees'], subset='detected_only',
                       statement=f"In the group '{name}' ({int(r['field_trees'])} trees) {_pct(r['detection_rate'])} of the field trees are detected."))

    # 3.6 baseline
    base = baseline_comparison(result, tables['profile'], tables['causes'], tables['control'], tables['base_tops'],
                               tables['base_match'], inst, high_confidence_distance_m=high_confidence_distance_m,
                               top_found_distance_m=top_found_distance_m)
    lm = base[~base.index.str.startswith('SegmentAnyTree')]
    best = lm['F1'].idxmax()
    picks = [('best_local_maxima', best)] + [(slug(m), m) for m in base.index if m.startswith('SegmentAnyTree')]
    for tag, name in picks:
        b = base.loc[name]
        add(_entry(f'baseline.{tag}.found', b['found'], 'share', len(result), ci=(b['ci_low'], b['ci_high']),
                   statement=f"'{name}' finds {_pct(b['found'])} of the field trees, against a chance level of {_pct(b['chance'])} for stems shifted by a fixed distance."))
        add(_entry(f'baseline.{tag}.chance', b['chance'], 'share', len(result),
                   f"For '{name}', {_pct(b['chance'])} of the shifted stems are found by chance."))
        add(_entry(f'baseline.{tag}.precision', b['precision'], 'share', b['units'],
                   f"{_pct(b['precision'])} of the {int(b['units'])} tops or instances of '{name}' in the Parzellen belong to a field tree."))
        add(_entry(f'baseline.{tag}.f1', b['F1'], 'score', len(result), f"The F1 score of '{name}' is {b['F1']:.2f}."))
        add(_entry(f'baseline.{tag}.found_smallest_fifth', b['found_smallest_rel_dbh_fifth'], 'share', None,
                   f"In the fifth of the trees with the smallest DBH relative to their Parzelle, '{name}' finds {_pct(b['found_smallest_rel_dbh_fifth'], 0)}."))
        add(_entry(f'baseline.{tag}.massband_rmse', b['massband_rmse'], 'm', b['massband_n'],
                   f"'{name}' has an RMSE of {b['massband_rmse']:.2f} m against the Maßband length on {int(b['massband_n'])} sample trees."))
        add(_entry(f'baseline.{tag}.massband_median', b['massband_median'], 'm', b['massband_n'],
                   f"'{name}' has a median error of {b['massband_median']:+.2f} m against the Maßband length on {int(b['massband_n'])} sample trees."))
    add(_entry('baseline.local_maxima_found_min', lm['found'].min(), 'share', len(result),
               f"The local-maxima windows find between {_pct(lm['found'].min())} and {_pct(lm['found'].max())} of the field trees."))
    add(_entry('baseline.local_maxima_found_max', lm['found'].max(), 'share', len(result),
               f"The best local-maxima window finds at most {_pct(lm['found'].max())} of the field trees."))

    # 4.1 causes
    cause_table, attribution = cause_counts(tables['causes'])
    for _, r in cause_table.iterrows():
        add(_entry(f"causes.{slug(r['detection'])}.{slug(r['cause'])}", r['trees'], 'trees', int((tables['causes']['detection'] == r['detection']).sum()),
                   subset=None, statement=f"{int(r['trees'])} of the trees classed '{r['detection']}' have the cause '{r['cause']}' (attributed to {r['attributed_to']})."))
    for name, r in attribution.iterrows():
        add(_entry(f'causes.attributed.{slug(name)}', r['both'], 'trees', len(result),
                   f"{int(r['both'])} uncertain or not detected trees are attributed to '{name}'."))

    # 6.1 harvest record against LiDAR
    hs = harvest_summary(tables['plain'], inst, gone_above=gone_above, standing_below=standing_below)
    lo, hi = wilson(hs['record_and_lidar_agree'], hs['with_clear_lidar_state'])
    add(_entry('harvest.agreement', hs['agreement'], 'share', hs['with_clear_lidar_state'], ci=(lo, hi),
               statement=f"For the {hs['with_clear_lidar_state']} trees with a clear LiDAR state, harvest record and LiDAR agree in {_pct(hs['agreement'])} of cases."))
    for key, text in [('marked_for_removal', 'field trees are marked for removal in the record'),
                      ('marked_but_standing', 'trees are marked for removal but still stand in the LiDAR'),
                      ('not_marked_but_gone', 'trees are not marked for removal but are gone in the LiDAR'),
                      ('lidar_instances_gone', 'LiDAR instances inside the Parzellen are gone after the harvest'),
                      ('lidar_instances_gone_without_field_tree', 'of the gone LiDAR instances have no field tree')]:
        add(_entry(f'harvest.{key}', hs[key], 'trees', hs['field_trees'], f"{hs[key]} {text}."))

    # 7.1 funnel
    funnel = height_funnel(everyone)
    for label, r in funnel.iterrows():
        tag = slug(label.split(',')[0])
        for key, col in [('with_height', 'trees with a field height'), ('detected', 'detected'), ('uncertain', 'uncertain'),
                         ('not_detected', 'not detected'), ('within_1m', 'detected, LiDAR within 1 m'),
                         ('within_2m', 'detected, LiDAR within 2 m')]:
            add(_entry(f'height.funnel.{tag}.{key}', r[col], 'trees', r['trees with a field height'],
                       f"Of the {int(r['trees with a field height'])} {label} trees, {int(r[col])} are counted as '{col}'."))

    # 7.2 to 7.6 height accuracy: three references x two subsets
    by_ref = height_accuracy_by_reference(everyone)
    sel = by_ref[(by_ref['trees'] == 'all_with_measurement') & (by_ref['reference'] != 'vertex_without_massband')]
    for _, r in sel.iterrows():
        entries.extend(_accuracy_entries(f"height.{r['reference']}", REFERENCE_TEXT[r['reference']], r['subset'], r, 'm'))
    # standing-height trees (a weaker reference)
    by_class = height_accuracy_by_class(everyone)
    for _, r in by_class[~by_class['reference set'].str.startswith('sample') & by_class['subset'].isin(SUBSETS)].iterrows():
        tag = 'standing_other_harvested' if r['reference set'].startswith('other') else 'standing_remaining_stand'
        entries.extend(_accuracy_entries(f'height.{tag}', f"ULS height against the standing-tree height of the {r['reference set'].split(',')[0]}",
                                         r['subset'], r, 'm'))
    fi = field_instrument_agreement(everyone)
    entries.extend(_accuracy_entries('height.vertex_vs_massband', 'Vertex length against the Maßband length (the terrestrial noise floor)', None, fi, 'm'))
    # leave-one-out
    influence, jack = leave_one_out(everyone)
    for subset in SUBSETS:
        j = jack.loc[subset]
        add(_entry('height.massband.jackknife_se', j['jackknife_se'], 'm', j['n'], subset=subset,
                   statement=f"The jackknife standard error of the RMSE against the Maßband is {j['jackknife_se']:.2f} m on {int(j['n'])} {SUBSET_TEXT[subset]}, so the RMSE rests on very few trees."))
        add(_entry('height.massband.rmse_without_largest_influence', j['rmse_without_largest_influence'], 'm', j['n'] - 1, subset=subset,
                   statement=f"Leaving out the single most influential tree, the RMSE against the Maßband is {j['rmse_without_largest_influence']:.2f} m (instead of {j['rmse']:.2f} m)."))
    # tiers
    tiers = height_tiers(everyone)
    add(_entry('height.manual_picked', tiers.attrs['manual_picked'], 'trees', int(everyone['has_height'].sum()),
               f"{tiers.attrs['manual_picked']} sample trees have a manually picked instance; they are a separate tier and not counted as SegmentAnyTree results."))
    add(_entry('height.manual_segmented', tiers.attrs['manual_segmented'], 'trees', int(everyone['has_height'].sum()),
               f"{tiers.attrs['manual_segmented']} sample trees were segmented by hand; they are a separate tier and not counted as SegmentAnyTree results."))

    # 8.1 DBH
    dbh, dbh_robust = dbh_accuracy(everyone, ellipse_axis_ratio=ellipse_axis_ratio)
    okc = dbh[(dbh['trees'] == 'ok fits') & (dbh['shape'] == 'circle')]
    for _, r in okc.iterrows():
        entries.extend(_accuracy_entries(f"dbh.{slug(r['group'])}", f"LiDAR DBH against field DBH ({r['group']}, ok circle fits)", r['subset'], r, 'cm', digits=2))
    for _, r in dbh_robust[dbh_robust['shape'] == 'circle'].iterrows():
        add(_entry(f"dbh.{slug(r['group'])}.within_2cm", r['within_2cm'], 'share', r['n'], subset=r['subset'],
                   statement=f"{_pct(r['within_2cm'], 0)} of the {int(r['n'])} {r['group']} with an ok circle fit have a LiDAR DBH within 2 cm of the field DBH."))
        add(_entry(f"dbh.{slug(r['group'])}.errors_above_10cm", r['errors_above_10cm'], 'trees', r['n'], subset=r['subset'],
                   statement=f"{int(r['errors_above_10cm'])} of the {int(r['n'])} {r['group']} with an ok circle fit have a DBH error above 10 cm."))

    details = {'detection_per_parzelle': per_plot, 'detection_overall': ov, 'baseline': base, 'cause_table': cause_table,
               'height_funnel': funnel, 'height_by_class': by_class, 'height_by_reference': by_ref, 'height_tiers': tiers,
               'influence': influence, 'jackknife': jack, 'dbh': dbh, 'dbh_robust': dbh_robust, 'match_distance': md,
               'harvest': hs, 'inventory': inv, 'record_vs_lidar': record_vs_lidar(
                   tables['plain'], [s for s in tables['plain']['lidar'].unique() if s not in LIDAR_STATES][0] if
                   [s for s in tables['plain']['lidar'].unique() if s not in LIDAR_STATES] else 'no instance')}
    return entries, details
