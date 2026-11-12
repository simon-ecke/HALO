"""
coord_transform.py
==================

Transform tree coordinates from local plot coordinates to UTM.
Ported from 04_peakfinder's field_data.py, dropping ODBC dependency.

Functions
---------
- fit_parzelle_transforms(corners_df) -> dict
- find_ground_lines(corners_df) -> dict
- transform_trees_local_to_utm(trees_df, corners_df) -> DataFrame

"""

import numpy as np
import pandas as pd


def _parz_key(value):
    """Normalize a Parzelle id (1, 1.0, '1', ' 1') to the string '1'."""
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value).strip()


def fit_parzelle_transforms(corners_df):
    """
    Fit a least-squares affine transform local (x, y) -> UTM for each Parzelle.

    Uses the surveyed local coordinates of the corner points, so no assumption
    about which corner is the origin or how the plot is oriented is needed.

    Args:
        corners_df (pd.DataFrame): Columns [Parzelle, x, y, UTM_x, UTM_y],
                                   at least 3 non-collinear corners per Parzelle.

    Returns:
        dict: {parzelle_str: {'coef': (3, 2) array, 'rmse': float, 'n': int}}.
              UTM = [local_x, local_y, 1] @ coef. rmse is the corner residual in metres.
    """
    transforms = {}
    for parzelle_id, group in corners_df.groupby('Parzelle'):
        pts = group[['x', 'y', 'UTM_x', 'UTM_y']].apply(pd.to_numeric, errors='coerce').dropna()
        if len(pts) < 3:
            print(f"Warning: Parzelle {parzelle_id} has only {len(pts)} usable corners. Skipping.")
            continue

        design = np.column_stack([pts['x'], pts['y'], np.ones(len(pts))])
        target = pts[['UTM_x', 'UTM_y']].to_numpy(dtype=float)
        if np.linalg.matrix_rank(design) < 3:
            print(f"Warning: Corners of Parzelle {parzelle_id} are collinear. Skipping.")
            continue

        coef, *_ = np.linalg.lstsq(design, target, rcond=None)
        residuals = design @ coef - target
        rmse = float(np.sqrt((residuals ** 2).sum(axis=1).mean()))
        transforms[_parz_key(parzelle_id)] = {'coef': coef, 'rmse': rmse, 'n': len(pts)}

    return transforms


def find_ground_lines(corners_df):
    """
    Identify the lower-left and lower-right ground line for each Parzelle.

    Legacy heuristic for corner tables without local coordinates. It only works
    if the plot origin is the southernmost corner; prefer fit_parzelle_transforms().

    Args:
        corners_df (pd.DataFrame): Columns [Parzelle, Nr, UTM_x, UTM_y].
                                   Nr should be > 10000 (convention for corner points).

    Returns:
        dict: {parzelle_str: ((lower_left_x, lower_left_y), (lower_right_x, lower_right_y))}.
              Lower-left = min UTM_y; lower-right = max UTM_x (among the min-y points).
    """
    # Group by Parzelle
    grouped = corners_df.groupby('Parzelle')

    ground_lines = {}
    for parzelle_id, group in grouped:
        parz_str = _parz_key(parzelle_id)

        # Find lower-left: minimum UTM_y point
        min_y_idx = group['UTM_y'].idxmin()
        lower_left = (group.loc[min_y_idx, 'UTM_x'], group.loc[min_y_idx, 'UTM_y'])

        # Find lower-right: maximum UTM_x among all points
        max_x_idx = group['UTM_x'].idxmax()
        lower_right = (group.loc[max_x_idx, 'UTM_x'], group.loc[max_x_idx, 'UTM_y'])

        ground_lines[parz_str] = (lower_left, lower_right)

    return ground_lines


def transform_trees_local_to_utm(trees_df, corners_df):
    """
    Transform tree coordinates from local plot coords to UTM.

    Args:
        trees_df (pd.DataFrame): Columns [parzelle, nr, tree_id, local_x, local_y, ...].
                                 Output of field_data_excel.load_field_trees_with_heights().
        corners_df (pd.DataFrame): Columns [Parzelle, Nr, UTM_x, UTM_y], ideally also the
                                    local corner coordinates [x, y].
                                    Supply corner points for the Parzellen in trees_df.

    Returns:
        pd.DataFrame: Input trees_df + new columns [utm_x, utm_y].
    """
    result = trees_df.copy()
    result['utm_x'] = np.nan
    result['utm_y'] = np.nan

    # Preferred: corners carry their local coordinates -> fit the transform directly
    if {'x', 'y'}.issubset(corners_df.columns):
        transforms = fit_parzelle_transforms(corners_df)

        for parz_str, idx in result.groupby(result['parzelle'].map(_parz_key)).groups.items():
            if parz_str not in transforms:
                print(f"Warning: No corner points found for Parzelle {parz_str}. Skipping {len(idx)} trees.")
                continue

            local = result.loc[idx, ['local_x', 'local_y']].to_numpy(dtype=float)
            utm = np.column_stack([local, np.ones(len(local))]) @ transforms[parz_str]['coef']
            result.loc[idx, 'utm_x'] = utm[:, 0]
            result.loc[idx, 'utm_y'] = utm[:, 1]

        return result

    # Fallback: derive ground lines from UTM corner positions only
    ground_lines = find_ground_lines(corners_df)

    # Transform each tree
    for idx, row in result.iterrows():
        parz_str = _parz_key(row['parzelle'])

        if parz_str not in ground_lines:
            # No ground line for this parzelle; skip
            print(f"Warning: No ground line found for Parzelle {parz_str}. Skipping tree {row['tree_id']}.")
            continue

        (ll_x, ll_y), (lr_x, lr_y) = ground_lines[parz_str]

        # Ground vector and perpendicular
        ground_vector = np.array([lr_x - ll_x, lr_y - ll_y])
        ground_length = np.linalg.norm(ground_vector)
        if ground_length == 0:
            print(f"Warning: Zero-length ground line for Parzelle {parz_str}. Skipping tree {row['tree_id']}.")
            continue

        unit_ground_vector = ground_vector / ground_length
        perpendicular_vector = np.array([-unit_ground_vector[1], unit_ground_vector[0]])

        # Transform: local (x, y) -> UTM
        local_x = row['local_x']
        local_y = row['local_y']
        utm_x = ll_x + local_x * unit_ground_vector[0] + local_y * perpendicular_vector[0]
        utm_y = ll_y + local_x * unit_ground_vector[1] + local_y * perpendicular_vector[1]

        result.at[idx, 'utm_x'] = utm_x
        result.at[idx, 'utm_y'] = utm_y

    return result


def parzelle_polygons(corners_df, crs=None):
    """
    Plot outlines as polygons, built from the four surveyed corner points.

    Args:
        corners_df (pd.DataFrame): Columns [Parzelle, x, y, UTM_x, UTM_y].
        crs (str, optional): CRS of the UTM coordinates.

    Returns:
        gpd.GeoDataFrame: Columns [parzelle, area_m2, geometry], one row per Parzelle.
    """
    import geopandas as gpd
    from shapely.geometry import MultiPoint

    rows = []
    for parzelle_id, group in corners_df.groupby('Parzelle'):
        # Convex hull, so the corner order in the table does not matter
        hull = MultiPoint(group[['UTM_x', 'UTM_y']].to_numpy(dtype=float)).convex_hull
        rows.append({'parzelle': _parz_key(parzelle_id), 'area_m2': hull.area, 'geometry': hull})

    return gpd.GeoDataFrame(rows, geometry='geometry', crs=crs)


# ---------------------------------------------------------------------------
# Added in core/: y-mirror and orientation check
# ---------------------------------------------------------------------------

# Orientation variants of the local plot coordinates: name -> (mirror x, mirror y)
VARIANTS = {'as in sheet': (False, False), 'x mirrored': (True, False), 'y mirrored': (False, True), 'both mirrored': (True, True)}


class OrientationError(Exception):
    """The best orientation per plot disagrees with plots.y_mirrored in the config."""


def plot_extent(corners_df):
    """Plot width (max local x of the corners) and depth (max local y), each a Series indexed by Parzelle."""
    return corners_df.groupby('Parzelle')['x'].max(), corners_df.groupby('Parzelle')['y'].max()


def mirror_y(trees_df, corners_df, mirrored_plots):
    """Mirror local y (y -> plot depth - y) for the trees of the listed plots.

    Adds 'local_y_sheet' (y as in the sheet) and 'y_mirrored' (bool); 'local_y' holds the corrected value.
    """
    result = trees_df.copy()
    _, depth = plot_extent(corners_df)
    plot = result['parzelle'].astype(int)
    result['local_y_sheet'] = result['local_y']
    mirrored = plot.isin([int(p) for p in mirrored_plots])
    result.loc[mirrored, 'local_y'] = plot[mirrored].map(depth) - result.loc[mirrored, 'local_y_sheet']
    result['y_mirrored'] = mirrored
    return result


def orientation_check(trees_df, base_xy, corners_df, max_dist_m=1.0):
    """Per plot, the share of stems with an instance base within max_dist_m, for the four orientation variants.

    trees_df needs tree_id, parzelle, local_x and local_y_sheet; base_xy is an (n, 2) array of instance
    bases. Returns a DataFrame indexed by 'parzelle' with columns trees, one per variant, and best.
    This decides the orientation of a whole plot, never the position of single trees.
    """
    from scipy.spatial import cKDTree

    base_index = cKDTree(np.asarray(base_xy, dtype=float))
    width, depth = plot_extent(corners_df)
    rows = []
    for parzelle, trees in trees_df.groupby(trees_df['parzelle'].astype(int)):
        row = {'parzelle': parzelle, 'trees': len(trees)}
        for name, (flip_x, flip_y) in VARIANTS.items():
            variant = trees[['tree_id', 'parzelle']].assign(
                local_x=width[parzelle] - trees['local_x'] if flip_x else trees['local_x'],
                local_y=depth[parzelle] - trees['local_y_sheet'] if flip_y else trees['local_y_sheet'])
            utm = transform_trees_local_to_utm(variant, corners_df)
            dist, _ = base_index.query(utm[['utm_x', 'utm_y']].to_numpy())
            row[name] = round(float((dist <= max_dist_m).mean()), 2)
        rows.append(row)
    orientation = pd.DataFrame(rows).set_index('parzelle')
    orientation['best'] = orientation[list(VARIANTS)].idxmax(axis=1)
    return orientation


def check_orientation(orientation, y_mirrored):
    """Raise OrientationError if the best variant per plot is not the configured one."""
    other = sorted(orientation.index[~orientation['best'].isin(['as in sheet', 'y mirrored'])])
    if other:
        raise OrientationError(f"Plots {other} fit best with an orientation that is not handled "
                               f"(x mirrored or both mirrored); check the local coordinates of these plots.")
    found = sorted(int(p) for p in orientation.index[orientation['best'] == 'y mirrored'])
    expected = sorted(int(p) for p in y_mirrored if int(p) in set(int(i) for i in orientation.index))
    if found != expected:
        raise OrientationError(f"Orientation check finds y mirrored in plots {found}, but plots.y_mirrored gives "
                               f"{expected}. Update plots.y_mirrored in the config and re-run from the inventory stage.")


def run_orientation(cfg):
    """Write orientation_check.csv; raise OrientationError if it disagrees with plots.y_mirrored."""
    import geopandas as gpd

    from .io_field import load_corners
    from .instances import instances_path

    corners = load_corners(cfg)                       # before geopandas reads a file
    trees = pd.read_csv(cfg.table('field_trees.csv'), dtype={'tree_id': str})
    instances = gpd.read_file(instances_path(cfg), layer='instance_base')
    orientation = orientation_check(trees, instances[['x_base', 'y_base']].to_numpy(), corners)
    cfg.tables_dir.mkdir(parents=True, exist_ok=True)
    orientation.to_csv(cfg.table('orientation_check.csv'))
    check_orientation(orientation, cfg.plots.y_mirrored)
    n_sheet = int((orientation['best'] == 'as in sheet').sum())
    print(f"Orientation: sheet orientation best in {n_sheet} of {len(orientation)} plots; y mirrored in "
          f"{sorted(int(p) for p in orientation.index[orientation['best'] == 'y mirrored'])}, as configured.")
