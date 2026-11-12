"""Field inventory: read the Excel sheets named in the config, clean them, mirror y, transform to UTM.

Every sheet column name, unit factor and plot rule comes from ``cfg.field_columns`` and ``cfg.plots``.
Read Excel before geopandas or rasterio read any file in the same process (otherwise an access
violation). ``run_inventory`` therefore also writes the plot corners to ``<output_dir>/work/plot_corners.csv``;
later stages read that file through ``load_corners`` and never need Excel again.

Corner tables are returned with the column names the coordinate functions use:
``Parzelle, Nr, x, y, UTM_x, UTM_y`` (these are internal standard names, not sheet names).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import cache
from .coords import mirror_y, transform_trees_local_to_utm

# Output columns of the inventory table (field_trees.csv), in this order.
INVENTORY_COLUMNS = [
    "tree_id", "parzelle", "nr", "species", "local_x", "local_y", "dbh_mm", "dbh_2020_mm", "dbh_year",
    "removed_2026", "height_vertex", "height_massband", "height_standing_sheet", "crown_base_m", "remark",
    "source_file", "has_height", "local_y_sheet", "y_mirrored", "utm_x", "utm_y", "block", "dbh_class",
]

# Diameter classes (mm) of the report tables.
DBH_CLASS_EDGES = [0, 200, 300, 400, 500, 2000]
DBH_CLASS_LABELS = ["< 200 mm", "200-300 mm", "300-400 mm", "400-500 mm", "> 500 mm"]
DBH_CLASS_MISSING = "no / implausible DBH"

_CORNER_NAMES = {"plot": "Parzelle", "nr": "Nr", "x": "x", "y": "y", "utm_x": "UTM_x", "utm_y": "UTM_y"}


def work_dir(cfg) -> Path:
    return cfg.output_dir / "work"


def corners_csv(cfg) -> Path:
    return work_dir(cfg) / "plot_corners.csv"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _sheet(spec) -> int | str:
    sheet = spec.get("sheet", 0)
    return 0 if sheet is None else sheet


def _numeric(df: pd.DataFrame, column: str | None, factor: float = 1.0) -> pd.Series:
    """Numeric values of a sheet column times the unit factor; NaN everywhere if the column is not mapped."""
    if column is None or column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce") * factor


def _field_table_path(cfg, file_name: str) -> Path:
    for p in cfg.paths.field_tables:
        if Path(p).name == file_name:
            return Path(p)
    raise ValueError(f"field_columns.layouts: file '{file_name}' is not in paths.field_tables")


def load_layout(cfg, layout) -> pd.DataFrame:
    """One inventory sheet as a table with the standard column names (rows of earlier thinnings dropped)."""
    path = _field_table_path(cfg, layout.file)
    cols = layout.columns.to_dict()
    factors = layout.factors.to_dict() if "factors" in layout else {}
    df = pd.read_excel(path, sheet_name=_sheet(layout))
    for std in ("plot", "nr", "x", "y", "dbh_mm"):
        if cols[std] not in df.columns:
            raise ValueError(f"{path.name}: column '{cols[std]}' (standard name '{std}') not found")
    df = df[df[cols["plot"]].notna()].copy()
    block_plots = pd.to_numeric(df[cols["plot"]]).astype(int)
    block = f"Parzellen {block_plots.min()}-{block_plots.max()}"

    if "earlier_removal_column" in layout:
        earlier = pd.to_numeric(df[layout.earlier_removal_column], errors="coerce").notna()
        df = df[~earlier]

    def num(std):
        return _numeric(df, cols.get(std), float(factors.get(std, 1.0)))

    removal = layout.removal
    if removal.column not in df.columns:
        raise ValueError(f"{path.name}: removal column '{removal.column}' not found")
    if isinstance(removal.removed_value, str):
        removed = df[removal.column].astype("string").str.strip().eq(removal.removed_value).fillna(False).astype(bool)
    else:
        removed = pd.to_numeric(df[removal.column], errors="coerce").eq(removal.removed_value)

    plot = pd.to_numeric(df[cols["plot"]]).astype("Int64").astype(str)
    nr = pd.to_numeric(df[cols["nr"]]).astype("Int64").astype(str)
    remark_col = cols.get("remark")
    # Crown base measured on the felled tree from the cut, plus the stump height (both converted to m by their factors)
    crown_base = num("crown_base_cm") + num("stump_height_cm").fillna(0.0)
    return pd.DataFrame({
        "tree_id": plot + "_" + nr,
        "parzelle": plot,
        "nr": nr,
        "species": df[cols["species"]] if "species" in cols else np.nan,
        "local_x": num("x").astype(float),
        "local_y": num("y").astype(float),
        "dbh_mm": num("dbh_mm"),
        "dbh_2020_mm": num("dbh_2020_mm"),
        "dbh_year": int(layout.dbh_year),
        "removed_2026": removed,
        "height_vertex": num("height_vertex_m"),
        "height_massband": num("height_massband_m"),
        "height_standing_sheet": num("height_standing_sheet_m"),
        "crown_base_m": crown_base,
        "remark": df[remark_col].astype("string").str.strip() if remark_col in df.columns else pd.NA,
        "source_file": path.name,
        "block": block,
    })


def load_inventory(cfg) -> pd.DataFrame:
    """Every tree standing before the harvest, from all layouts. Columns: see INVENTORY_COLUMNS (without
    local_y_sheet, y_mirrored, utm_x, utm_y, dbh_class); plus ``block``."""
    parts = [load_layout(cfg, layout) for layout in cfg.field_columns.layouts]
    result = pd.concat(parts, ignore_index=True)
    result["has_height"] = result[["height_vertex", "height_massband"]].notna().any(axis=1)
    if result["tree_id"].duplicated().any():
        dups = result.loc[result["tree_id"].duplicated(keep=False), "tree_id"].unique().tolist()
        raise ValueError(f"Duplicate tree ids among standing trees: {dups}")
    return result


SURVEY_2020_COLUMNS = ["tree_id", "parzelle", "nr", "species", "dbh_2020_mm", "height_2020_m", "removed_earlier"]


def survey_2020_frame(df: pd.DataFrame, cols: dict, factors: dict | None = None,
                      earlier_column: str | None = None) -> pd.DataFrame:
    """Every tree of the 2020 survey of one sheet, with the standard column names. Pure function.

    Unlike `load_layout`, trees of earlier thinnings are kept and flagged in ``removed_earlier`` (any number in
    ``earlier_column``). Rows without a plot value are dropped. Heights are returned as typed in the sheet (converted
    to m by the unit factor only); the cleaning of decimetre values is left to the caller.
    """
    factors = factors or {}
    df = df[df[cols["plot"]].notna()].copy()

    def num(std):
        return _numeric(df, cols.get(std), float(factors.get(std, 1.0)))

    if earlier_column is not None and earlier_column in df.columns:
        earlier = pd.to_numeric(df[earlier_column], errors="coerce").notna()
    else:
        earlier = pd.Series(False, index=df.index)
    plot = pd.to_numeric(df[cols["plot"]]).astype("Int64").astype(str)
    nr = pd.to_numeric(df[cols["nr"]]).astype("Int64").astype(str)
    return pd.DataFrame({
        "tree_id": plot + "_" + nr,
        "parzelle": plot,
        "nr": nr,
        "species": df[cols["species"]] if "species" in cols else np.nan,
        "dbh_2020_mm": num("dbh_2020_mm"),
        "height_2020_m": num("height_2020_m"),
        "removed_earlier": earlier.astype(bool),
    })[SURVEY_2020_COLUMNS].reset_index(drop=True)


def load_survey_2020(cfg) -> pd.DataFrame:
    """All trees of the 2020 survey from every layout, including those removed in an earlier thinning.

    Columns: tree_id (as in the inventory), parzelle, nr, species, dbh_2020_mm, height_2020_m, removed_earlier.
    Reads Excel. The sheet column of the 2020 height is ``increment.height_2020_column`` (one name for all layouts, or a
    mapping layout name -> column).
    """
    height = (cfg.raw.get("increment") or {}).get("height_2020_column")
    parts = []
    for layout in cfg.field_columns.layouts:
        cols = layout.columns.to_dict()
        column = height.get(layout.name) if isinstance(height, dict) else height
        if column:
            cols["height_2020_m"] = column
        factors = layout.factors.to_dict() if "factors" in layout else {}
        earlier = layout.earlier_removal_column if "earlier_removal_column" in layout else None
        df = pd.read_excel(_field_table_path(cfg, layout.file), sheet_name=_sheet(layout))
        parts.append(survey_2020_frame(df, cols, factors, earlier))
    result = pd.concat(parts, ignore_index=True)
    # A tree of an earlier thinning may carry the number of a standing tree; its id then gets the suffix "_earlier"
    # so that the ids of the standing trees stay those of the inventory.
    standing_ids = set(result.loc[~result["removed_earlier"], "tree_id"])
    clash = result["removed_earlier"] & result["tree_id"].isin(standing_ids)
    result.loc[clash, "tree_id"] = result.loc[clash, "tree_id"] + "_earlier"
    if result["tree_id"].duplicated().any():
        dups = result.loc[result["tree_id"].duplicated(keep=False), "tree_id"].unique().tolist()
        raise ValueError(f"Duplicate tree ids in the 2020 survey: {dups}")
    return result


def read_corners_excel(cfg) -> pd.DataFrame:
    """Plot corner points from the Excel sheet (always reads Excel)."""
    spec = cfg.field_columns.corners
    cols = spec.columns.to_dict()
    factors = spec.factors.to_dict() if "factors" in spec else {}
    df = pd.read_excel(cfg.paths.plot_corners, sheet_name=_sheet(spec))
    out = pd.DataFrame({name: pd.to_numeric(df[cols[std]], errors="coerce") * float(factors.get(std, 1.0))
                        for std, name in _CORNER_NAMES.items()})
    out = out[out["Parzelle"].notna()].copy()
    out["Parzelle"] = out["Parzelle"].astype(int)
    return out.reset_index(drop=True)


def load_corners(cfg) -> pd.DataFrame:
    """Plot corner points with columns Parzelle, Nr, x, y, UTM_x, UTM_y.

    Reads ``<output_dir>/work/plot_corners.csv`` when it is newer than the Excel sheet (so no Excel is read
    after geopandas has been used); otherwise reads the Excel sheet.
    """
    src = cfg.paths.plot_corners
    cached = corners_csv(cfg)
    if cached.is_file() and Path(src).is_file() and cached.stat().st_mtime >= Path(src).stat().st_mtime:
        return pd.read_csv(cached)
    return read_corners_excel(cfg)


def load_sample_heights(cfg) -> pd.DataFrame:
    """Sample trees with tape and Vertex heights: columns plot, nr, x, y, height_vertex_m, height_massband_m
    (rows with at least one height), tree_id = '<plot>_<nr>'."""
    spec = cfg.field_columns.sample_heights
    cols = spec.columns.to_dict()
    factors = spec.factors.to_dict() if "factors" in spec else {}
    df = pd.read_excel(cfg.paths.sample_heights_table, sheet_name=_sheet(spec))
    out = pd.DataFrame({std: pd.to_numeric(df[col], errors="coerce") * float(factors.get(std, 1.0))
                        for std, col in cols.items()})
    out = out[out[["height_vertex_m", "height_massband_m"]].notna().any(axis=1)].copy()
    out["plot"] = out["plot"].astype(int)
    out["nr"] = out["nr"].astype(int)
    out.insert(0, "tree_id", out["plot"].astype(str) + "_" + out["nr"].astype(str))
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cleaning, mirror, transform
# ---------------------------------------------------------------------------

def build_inventory(field: pd.DataFrame, corners: pd.DataFrame, y_mirrored, dbh_plausible_mm) -> pd.DataFrame:
    """Mirror local y in the given plots, transform to UTM, add diameter class. Pure function."""
    field = mirror_y(field, corners, y_mirrored)
    field = transform_trees_local_to_utm(field, corners)
    if not field[["utm_x", "utm_y"]].notna().all().all():
        raise ValueError("Some trees have no UTM position (plot without corner points?)")

    dbh_ok = field["dbh_mm"].between(*dbh_plausible_mm)
    dbh_class = pd.cut(field["dbh_mm"].where(dbh_ok), DBH_CLASS_EDGES, labels=DBH_CLASS_LABELS)
    field["dbh_class"] = dbh_class.cat.add_categories(DBH_CLASS_MISSING).fillna(DBH_CLASS_MISSING).astype(str)
    return field[INVENTORY_COLUMNS]


def compute_inventory(cfg) -> pd.DataFrame:
    corners = load_corners(cfg)
    field = load_inventory(cfg)
    return build_inventory(field, corners, list(cfg.plots.y_mirrored), cfg.cleaning.dbh_plausible_mm)


def run_inventory(cfg) -> None:
    """Write field_trees.csv, field_trees_2020.csv (and work/plot_corners.csv for the later stages)."""
    # Always read the corners from Excel here and store them for the stages that run after geopandas.
    corners_csv(cfg).parent.mkdir(parents=True, exist_ok=True)
    read_corners_excel(cfg).to_csv(corners_csv(cfg), index=False)

    inputs = [*cfg.paths.field_tables, cfg.paths.plot_corners]
    cache.cached_table(cfg, "field_trees.csv", ["site", "field_columns", "plots", "cleaning"], inputs,
                       lambda: compute_inventory(cfg))
    cache.cached_table(cfg, "field_trees_2020.csv", ["site", "field_columns", "increment"], list(cfg.paths.field_tables),
                       lambda: load_survey_2020(cfg))
