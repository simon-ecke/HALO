"""Check a site config against the files it points to.

    python -m core.validate config\\site_155.yaml

Prints one plain line per problem, then "OK" or "N problems". Exit code 0 or 1.
Only headers are read: Excel header rows and LAZ headers, never point data.
"""

from __future__ import annotations

import sys
from pathlib import Path

from core.config import Config, ConfigError, read_raw, schema_problems


def _excel_header(path: Path, sheet):
    """Column names of a sheet (header row only)."""
    import pandas as pd

    return list(pd.read_excel(path, sheet_name=sheet, nrows=0).columns)


def _check_columns(problems, key, path, sheet, columns, headers):
    """Every column named in `columns` (standard name -> sheet column) must be in the sheet header."""
    try:
        if (path, sheet) not in headers:
            headers[(path, sheet)] = _excel_header(path, sheet)
        have = headers[(path, sheet)]
    except Exception as exc:  # unreadable workbook, missing sheet
        problems.append(f"{key}: cannot read sheet {sheet!r} of {path.name}: {exc}")
        return
    for std, col in columns.items():
        if col not in have:
            problems.append(f"{key}.{std}: column '{col}' not found in sheet {sheet!r} of {path.name}")


def validate(config_path) -> list[str]:
    """Return the list of problems (empty if the config and its inputs are fine)."""
    try:
        raw = read_raw(config_path)
    except ConfigError as exc:
        return [str(exc)]
    problems = schema_problems(raw)
    if problems:
        return problems  # the rest needs a structurally valid file

    cfg = Config(raw)
    p = cfg.paths
    exists_ok = set()

    def need_file(key, path):
        if Path(path).is_file():
            exists_ok.add(Path(path))
        else:
            problems.append(f"{key}: file not found: {path}")

    for i, f in enumerate(p.field_tables):
        need_file(f"paths.field_tables[{i}]", f)
    need_file("paths.sample_heights_table", p.sample_heights_table)
    need_file("paths.plot_corners", p.plot_corners)
    need_file("paths.sat_before.path", p.sat_before.path)
    if "sat_after" in p:
        need_file("paths.sat_after.path", p.sat_after.path)
    for key in ("ortho_before", "match_overrides"):
        if key in p:
            need_file(f"paths.{key}", p[key])
    if "manual_segments_dir" in p and not Path(p.manual_segments_dir).is_dir():
        problems.append(f"paths.manual_segments_dir: folder not found: {p.manual_segments_dir}")

    # Excel sheets first (before any geo library is imported)
    headers: dict = {}
    fc = raw["field_columns"]
    by_name = {Path(f).name: f for f in p.field_tables}
    for i, lay in enumerate(fc["layouts"]):
        key = f"field_columns.layouts[{i}]"
        path = by_name.get(lay["file"])
        if path is None:
            problems.append(f"{key}.file: '{lay['file']}' is not a file name in paths.field_tables")
            continue
        if path in exists_ok:
            cols = dict(lay["columns"])
            cols["removal.column"] = lay["removal"]["column"]
            if "earlier_removal_column" in lay:
                cols["earlier_removal_column"] = lay["earlier_removal_column"]
            _check_columns(problems, key + ".columns", path, lay.get("sheet", 0), cols, headers)
        for std in lay.get("factors", {}):
            if std not in lay["columns"]:
                problems.append(f"{key}.factors.{std}: no column is mapped to this standard name")
    for name in by_name:
        if name not in {lay["file"] for lay in fc["layouts"]}:
            problems.append(f"paths.field_tables: '{name}' has no entry in field_columns.layouts")

    sh = fc["sample_heights"]
    if Path(p.sample_heights_table) in exists_ok:
        _check_columns(problems, "field_columns.sample_heights.columns", Path(p.sample_heights_table),
                       sh.get("sheet", 0), sh["columns"], headers)

    corner_plots = None
    cc = fc["corners"]
    if Path(p.plot_corners) in exists_ok:
        n_before = len(problems)
        _check_columns(problems, "field_columns.corners.columns", Path(p.plot_corners), cc.get("sheet", 0),
                       cc["columns"], headers)
        if len(problems) == n_before:
            import pandas as pd

            col = pd.read_excel(p.plot_corners, sheet_name=cc.get("sheet", 0), usecols=[cc["columns"]["plot"]])
            corner_plots = set(pd.to_numeric(col.iloc[:, 0], errors="coerce").dropna().astype(int))
    if corner_plots is not None:
        for plot in raw["plots"]["y_mirrored"]:
            if plot not in corner_plots:
                problems.append(f"plots.y_mirrored: plot {plot} is not in the corners table")

    # CRS
    try:
        from pyproj import CRS

        CRS.from_user_input(raw["site"]["crs"])
    except Exception as exc:
        problems.append(f"site.crs: cannot parse '{raw['site']['crs']}': {exc}")

    # LAZ headers: the instance dimension must exist
    for key, entry in (("paths.sat_before", p.sat_before), ("paths.sat_after", p.get("sat_after"))):
        if entry is None or Path(entry.path) not in exists_ok:
            continue
        try:
            import laspy

            with laspy.open(str(entry.path)) as fh:
                dims = list(fh.header.point_format.dimension_names)
            if entry.instance_dim not in dims:
                problems.append(f"{key}.instance_dim: '{entry.instance_dim}' not in the LAZ header of "
                                f"{Path(entry.path).name} (has: {', '.join(dims)})")
        except Exception as exc:
            problems.append(f"{key}.path: cannot read LAZ header: {exc}")

    return problems


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m core.validate <config.yaml>")
        return 2
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    problems = validate(argv[0])
    for line in problems:
        print(line)
    print("OK" if not problems else f"{len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
