"""Stand structure at every field tree (stage 5) and the cause of an uncertain or missed tree (stage 7).

Pure functions take data and thresholds and return DataFrames. Only run_structure() and
run_causes() read inputs and write tables. All thresholds come from the site config.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

from core.cache import cached_table

# cause -> who or what it is attributed to
CAUSES = {
    'tree found, harvest record disagrees': 'field record or timing',
    'tree found, instance partly gone': 'SegmentAnyTree',
    'merged with a neighbour': 'SegmentAnyTree',
    'own instance, base far from the stem': 'SegmentAnyTree',
    'stem visible, not segmented': 'SegmentAnyTree',
    'stem not visible in the LiDAR': 'LiDAR coverage',
    'no tree at the position on the flight day': 'field record or timing',
    'instance 1.5-3 m away, not confirmed by a stem': 'unclear',
}
ATTRIBUTIONS = ['SegmentAnyTree', 'field record or timing', 'LiDAR coverage', 'unclear']


def cause_explanations(no_canopy_deficit_m: float) -> dict[str, str]:
    """Plain-language explanation of every cause (the canopy limit is a threshold of the site)."""
    return {
        'tree found, harvest record disagrees':
            'Instance base within 1.5 m, and the instance is clearly gone or clearly standing after the harvest, '
            'but the record says the opposite. The tree itself is matched.',
        'tree found, instance partly gone':
            'Instance base within 1.5 m, but 20-80 % of the instance is gone after the harvest: the instance holds '
            'points of a felled and of a standing tree.',
        'merged with a neighbour':
            "A stem stands at the field position, but its points belong to the instance of another field tree: "
            'two stems, one instance.',
        'own instance, base far from the stem':
            'A stem stands at the field position and belongs to an instance, but the base of that instance was '
            'placed more than 1.5 m away.',
        'stem visible, not segmented':
            'A stem stands at the field position, but its points belong to no tree instance.',
        'stem not visible in the LiDAR':
            'Canopy at the position, but no stem circle in any of the eight slices from 1.5 to 8.5 m.',
        'no tree at the position on the flight day':
            f'The canopy at the position is more than {no_canopy_deficit_m:.0f} m below the height a tree of this '
            'diameter should have: the tree did not stand on the flight day, or the position is wrong.',
        'instance 1.5-3 m away, not confirmed by a stem':
            'An instance was assigned at 1.5-3 m. No stem was found at the field position, or the stem belongs to no '
            'instance, so nothing confirms or contradicts the assignment.',
    }


# ---------------------------------------------------------------------------
# Inputs shared with core.dbh
# ---------------------------------------------------------------------------

def read_corners(cfg) -> pd.DataFrame:
    """Plot corner sheet with standard column names (plot, nr, x, y, utm_x, utm_y).

    Uses io_field.load_corners (CSV copy written by the inventory stage, else Excel: then call before geopandas
    or rasterio read any file in the process).
    """
    from core.io_field import load_corners

    return load_corners(cfg).rename(columns={'Parzelle': 'plot', 'Nr': 'nr', 'UTM_x': 'utm_x', 'UTM_y': 'utm_y'})


def plot_extent(corners: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Plot width (largest local x) and depth (largest local y) per plot, indexed by integer plot id."""
    c = corners.assign(plot=corners['plot'].astype(int))
    return c.groupby('plot')['x'].max(), c.groupby('plot')['y'].max()


def plot_polygons(corners: pd.DataFrame, crs: str | None = None):
    """Plot outlines (convex hull of the surveyed corners), indexed by integer plot id."""
    import geopandas as gpd
    from shapely.geometry import MultiPoint

    rows = []
    for plot, group in corners.groupby(corners['plot'].astype(int)):
        hull = MultiPoint(group[['utm_x', 'utm_y']].to_numpy(dtype=float)).convex_hull
        rows.append({'parzelle': int(plot), 'geometry': hull})
    return gpd.GeoDataFrame(rows, geometry='geometry', crs=crs).set_index('parzelle').sort_index()


def read_chm(path):
    """LiDAR canopy height model as float32 array (nodata -> NaN), its transform."""
    with rasterio.open(path) as src:
        chm = src.read(1).astype(np.float32)
        chm[chm == src.nodata] = np.nan
        return chm, src.transform


# ---------------------------------------------------------------------------
# Stage 5: stand structure
# ---------------------------------------------------------------------------

def height_curve(dbh, a, b, c):
    """Expected tree height (m) from DBH (mm)."""
    return 1.3 + a * (1 - np.exp(-b * dbh)) ** c


def structure_per_tree(trees: pd.DataFrame, chm: np.ndarray, chm_transform, *, dbh_plausible_mm,
                       competition_radius_m: float, chm_stem_radius_m: float, chm_resolution_m: float,
                       no_canopy_deficit_m: float, high_confidence_distance_m: float,
                       gone_above: float, standing_below: float):
    """Stand structure per field tree; returns (table with utm_x, utm_y still attached, height curve parameters).

    trees: field_trees_detection.csv. Per tree: DBH relative to the plot mean, nearest stem, Hegyi
    competition index, canopy height at the stem against the height a tree of that DBH should have
    (curve fitted to the detected trees), and the coarse first reading `case` of a miss.
    """
    struct = trees[['tree_id', 'parzelle', 'species', 'removed_2026', 'utm_x', 'utm_y', 'detection', 'uncertain_reason',
                    'base_dist_m', 'removed_fraction', 'runner_up_instance', 'lidar_height']].copy()
    struct['parzelle'] = struct['parzelle'].astype(int)
    struct['dbh_mm'] = trees['dbh_mm'].where(trees['dbh_mm'].between(*dbh_plausible_mm))
    struct['rel_dbh'] = struct['dbh_mm'] / struct.groupby('parzelle')['dbh_mm'].transform('mean')

    # Neighbourhood from the field stems of the whole site, so neighbours across a plot border count as well
    stem_xy = struct[['utm_x', 'utm_y']].to_numpy()
    stem_index = cKDTree(stem_xy)
    nn_dist, _ = stem_index.query(stem_xy, k=2)
    struct['nn_dist_m'] = nn_dist[:, 1]
    dbh_filled = struct['dbh_mm'].fillna(struct['dbh_mm'].median()).to_numpy()
    hegyi, larger_dist = [], []
    for i, neighbours in enumerate(stem_index.query_ball_point(stem_xy, competition_radius_m)):
        neighbours = np.array([j for j in neighbours if j != i], dtype=int)
        dist = np.hypot(*(stem_xy[neighbours] - stem_xy[i]).T) if len(neighbours) else np.array([])
        hegyi.append(float(np.sum(dbh_filled[neighbours] / dbh_filled[i] / np.maximum(dist, 0.3))))
        larger = dist[dbh_filled[neighbours] > dbh_filled[i]]
        larger_dist.append(larger.min() if len(larger) else np.nan)
    struct['hegyi'] = hegyi
    struct['larger_neighbour_dist_m'] = larger_dist   # NaN: no larger tree within the competition radius

    rows, cols = rasterio.transform.rowcol(chm_transform, struct['utm_x'].to_numpy(), struct['utm_y'].to_numpy())
    k = int(round(chm_stem_radius_m / chm_resolution_m))
    struct['chm_at_stem_m'] = np.array([np.nanmax(chm[max(r - k, 0):r + k + 1, max(c - k, 0):c + k + 1]) for r, c in zip(rows, cols)], dtype=np.float64)

    fit = struct[(struct['detection'] == 'detected') & struct['dbh_mm'].notna() & struct['lidar_height'].notna()]
    curve_params, _ = curve_fit(height_curve, fit['dbh_mm'], fit['lidar_height'], p0=[35, 0.005, 1], maxfev=20000)
    struct['expected_height_m'] = height_curve(struct['dbh_mm'].fillna(struct['dbh_mm'].median()), *curve_params)
    struct['canopy_deficit_m'] = struct['expected_height_m'] - struct['chm_at_stem_m']

    struct['case'] = ''
    nd = struct['detection'] == 'not detected'
    struct.loc[nd, 'case'] = np.select(
        [struct.loc[nd, 'canopy_deficit_m'] > no_canopy_deficit_m, struct.loc[nd, 'runner_up_instance'].notna()],
        ['no canopy at the stem', 'instance taken by a neighbour'], default='no instance base within 3 m')
    unc = struct['detection'] == 'uncertain'
    u = struct[unc]
    struct.loc[unc, 'case'] = np.select(
        [u['base_dist_m'] > high_confidence_distance_m,
         ~u['removed_2026'] & (u['removed_fraction'] > gone_above),
         u['removed_2026'] & (u['removed_fraction'] < standing_below)],
        ['base 1.5-3 m away', 'not marked, instance gone', 'marked, instance still standing'], default='instance partly gone')
    return struct, curve_params


def canopy_per_parzelle(chm: np.ndarray, chm_transform, plots, gap_height_m: float) -> pd.DataFrame:
    """Canopy top height (95th percentile), roughness and gap share per plot from the CHM.

    plots: GeoDataFrame indexed by integer plot id with the plot outline (see plot_polygons).
    """
    rows = []
    for parzelle, geom in plots.geometry.items():
        values = chm[geometry_mask([geom], chm.shape, chm_transform, invert=True)]
        values = values[np.isfinite(values)]
        rows.append({'parzelle': parzelle, 'area_m2': geom.area, 'canopy_top_height_m': np.percentile(values, 95),
                     'canopy_height_sd_m': values.std(), 'gap_share': (values < gap_height_m).mean()})
    return pd.DataFrame(rows)


def run_structure(cfg) -> None:
    """Write structure_per_tree.csv and canopy_per_parzelle.csv."""
    corners = read_corners(cfg)        # Excel first
    from core import instances         # owned by another module; imported late

    detection_path = cfg.table('field_trees_detection.csv')
    chm_file = instances.chm_path(cfg)
    memo = {}

    def chm():
        if 'chm' not in memo:
            memo['chm'] = read_chm(chm_file)
        return memo['chm']

    def compute_struct():
        trees = pd.read_csv(detection_path, dtype={'tree_id': str})
        struct, _ = structure_per_tree(
            trees, *chm(), dbh_plausible_mm=cfg.cleaning.dbh_plausible_mm,
            competition_radius_m=cfg.structure.competition_radius_m, chm_stem_radius_m=cfg.structure.chm_stem_radius_m,
            chm_resolution_m=cfg.grid.chm_resolution_m, no_canopy_deficit_m=cfg.structure.no_canopy_deficit_m,
            high_confidence_distance_m=cfg.matching.high_confidence_distance_m,
            gone_above=cfg.matching.gone_above, standing_below=cfg.matching.standing_below)
        return struct.drop(columns=['utm_x', 'utm_y'])

    def compute_canopy():
        return canopy_per_parzelle(*chm(), plot_polygons(corners, cfg.site.crs), cfg.structure.gap_height_m)

    cached_table(cfg, 'structure_per_tree.csv', ['structure', 'cleaning', 'matching', 'grid'],
                 [detection_path, chm_file], compute_struct)
    cached_table(cfg, 'canopy_per_parzelle.csv', ['structure', 'grid', 'field_columns'],
                 [chm_file, cfg.paths.plot_corners], compute_canopy)


# ---------------------------------------------------------------------------
# Stage 7: why a tree is uncertain or not detected
# ---------------------------------------------------------------------------

def classify_match_cause(profile: pd.DataFrame, *, high_confidence_distance_m: float, gone_above: float,
                         standing_below: float, no_canopy_deficit_m: float) -> pd.DataFrame:
    """One cause per uncertain or not detected field tree, by rules on the evidence in the profile.

    profile: one row per field tree with detection, base_dist_m, removed_fraction, canopy_deficit_m,
    stem_owner and instance (tables/tree_profile_all.csv).

    Returns tree_id, detection, cause, attributed_to and stem_confirms_instance (True: the stem found at
    the field position lies in the assigned instance; False: it lies elsewhere; NA: no stem found, or no
    instance assigned). Detected trees get an empty cause.
    """
    owner = profile['stem_owner'].astype(str)
    own = owner == 'own instance'
    other_tree = owner == 'instance of another field tree'
    other_instance = owner == 'instance without a field tree'
    unsegmented = owner == 'not in any instance'
    no_stem = owner == 'no stem at the position'

    far = profile['base_dist_m'] > high_confidence_distance_m
    clear_state = (profile['removed_fraction'] > gone_above) | (profile['removed_fraction'] < standing_below)
    not_detected = profile['detection'] == 'not detected'
    uncertain = profile['detection'] == 'uncertain'

    cause = pd.Series('', index=profile.index)
    nd_rules = [
        (profile['canopy_deficit_m'] > no_canopy_deficit_m, 'no tree at the position on the flight day'),
        (other_tree, 'merged with a neighbour'),
        (other_instance, 'own instance, base far from the stem'),
        (unsegmented, 'stem visible, not segmented'),
        (no_stem, 'stem not visible in the LiDAR'),
    ]
    unc_rules = [
        (other_tree, 'merged with a neighbour'),
        (far & (own | other_instance), 'own instance, base far from the stem'),
        (far, 'instance 1.5-3 m away, not confirmed by a stem'),
        (clear_state, 'tree found, harvest record disagrees'),
        (pd.Series(True, index=profile.index), 'tree found, instance partly gone'),
    ]
    for selected, rules in [(not_detected, nd_rules), (uncertain, unc_rules)]:
        open_rows = selected.copy()
        for condition, name in rules:
            hit = open_rows & condition
            cause[hit] = name
            open_rows &= ~hit
        cause[open_rows] = 'unclear'

    confirms = pd.Series(pd.NA, index=profile.index, dtype='boolean')
    has_instance = profile['instance'].notna() if 'instance' in profile.columns else profile['base_dist_m'].notna()
    confirms[has_instance & own] = True
    confirms[has_instance & (other_tree | other_instance | unsegmented)] = False

    return pd.DataFrame({
        'tree_id': profile['tree_id'], 'detection': profile['detection'].astype(str), 'cause': cause,
        'attributed_to': cause.map(CAUSES).fillna(cause.where(cause == 'unclear', '')),
        'stem_confirms_instance': confirms,
    })


def stand_conditions(trees: pd.DataFrame, instances: pd.DataFrame, *, radius_m: float, structure_radius_m: float,
                     dbh_plausible_mm) -> pd.DataFrame:
    """Neighbourhood of every field tree, from the stem map and the LiDAR instance table.

    trees: tree_id, utm_x, utm_y, dbh_mm. instances: x_base, y_base, height.

    Returns tree_id and
    - larger_share_5m: share of the field stems within `radius_m` with a larger DBH
    - instances_10m: LiDAR instances with their base within `structure_radius_m`
    - height_sd_10m, height_range_10m: spread of their heights
    - low_share_10m: share of them below two thirds of the tallest
    """
    xy = trees[['utm_x', 'utm_y']].to_numpy(dtype=float)
    dbh = trees['dbh_mm'].where(trees['dbh_mm'].between(*dbh_plausible_mm)).to_numpy(dtype=float)
    larger_share = np.full(len(trees), np.nan)
    for i, neighbours in enumerate(cKDTree(xy).query_ball_point(xy, radius_m)):
        neighbours = [j for j in neighbours if j != i and np.isfinite(dbh[j])]
        if neighbours and np.isfinite(dbh[i]):
            larger_share[i] = float(np.mean(dbh[neighbours] > dbh[i]))

    base_xy = instances[['x_base', 'y_base']].to_numpy(dtype=float)
    heights = instances['height'].to_numpy(dtype=float)
    rows = []
    for neighbours in cKDTree(base_xy).query_ball_point(xy, structure_radius_m):
        h = heights[neighbours]
        if len(h) >= 2:
            rows.append((len(h), float(h.std(ddof=1)), float(h.max() - h.min()), float(np.mean(h < 2 / 3 * h.max()))))
        else:
            rows.append((len(h), np.nan, np.nan, np.nan))
    out = pd.DataFrame(rows, columns=['instances_10m', 'height_sd_10m', 'height_range_10m', 'low_share_10m'])
    out.insert(0, 'larger_share_5m', larger_share)
    out.insert(0, 'tree_id', trees['tree_id'].to_numpy())
    return out


def match_causes_table(profile: pd.DataFrame, trees: pd.DataFrame, instances: pd.DataFrame, column: pd.DataFrame, *,
                       high_confidence_distance_m: float, gone_above: float, standing_below: float,
                       no_canopy_deficit_m: float, competition_radius_m: float, instance_radius_m: float,
                       dbh_plausible_mm) -> pd.DataFrame:
    """match_causes.csv: cause, stand conditions and point-column structure per tree."""
    causes = classify_match_cause(profile, high_confidence_distance_m=high_confidence_distance_m,
                                  gone_above=gone_above, standing_below=standing_below,
                                  no_canopy_deficit_m=no_canopy_deficit_m)
    conditions = stand_conditions(trees[['tree_id', 'utm_x', 'utm_y', 'dbh_mm']], instances, radius_m=competition_radius_m,
                                  structure_radius_m=instance_radius_m, dbh_plausible_mm=dbh_plausible_mm)
    causes = causes.merge(conditions, on='tree_id').merge(column, on='tree_id', how='left')
    # positive: the canopy at the stem is higher than this tree is expected to be, i.e. it stands under larger trees
    causes['overtopped_by_m'] = (profile['chm_at_stem_m'] - profile['expected_height_m']).to_numpy()
    return causes


def run_causes(cfg) -> None:
    """Write match_causes.csv."""
    import geopandas as gpd

    paths = [cfg.table(n) for n in ('field_trees_detection.csv', 'tree_profile_all.csv', 'column_structure_per_tree.csv',
                                     'lidar_instances.gpkg')]

    def compute():
        trees = pd.read_csv(paths[0], dtype={'tree_id': str})
        profile = pd.read_csv(paths[1], dtype={'tree_id': str})
        column = pd.read_csv(paths[2], dtype={'tree_id': str})
        instances = gpd.read_file(paths[3], layer='instance_base')
        out = match_causes_table(
            profile, trees, instances, column,
            high_confidence_distance_m=cfg.matching.high_confidence_distance_m, gone_above=cfg.matching.gone_above,
            standing_below=cfg.matching.standing_below, no_canopy_deficit_m=cfg.structure.no_canopy_deficit_m,
            competition_radius_m=cfg.structure.competition_radius_m, instance_radius_m=cfg.structure.instance_radius_m,
            dbh_plausible_mm=cfg.cleaning.dbh_plausible_mm)
        missed = out[out['cause'] != '']
        assert len(missed) == int((trees['detection'] != 'detected').sum())
        return out

    cached_table(cfg, 'match_causes.csv', ['structure', 'matching', 'cleaning'], paths, compute)
