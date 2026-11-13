"""Statistics stage: per-tree tables -> stats.json.

    python scripts/run_stats.py config/site_155.yaml [--tables-dir DIR]

Reads the per-tree tables (default cfg.tables_dir) and writes stats.json to the config's output folder.
The file always goes there, also with --tables-dir (testing against another run), so the other run is
never touched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import analyses, stats           # noqa: E402
from core.config import load_config        # noqa: E402


def run(cfg, tables_dir=None, write_tables: bool = False, height_variants: bool = False,
        rasters_dir=None) -> Path:
    """Compute all statistics and write stats.json; returns its path."""
    tables = stats.load_tables(tables_dir or cfg.tables_dir)
    entries, _ = stats.compute_all(
        tables,
        high_confidence_distance_m=cfg.matching.high_confidence_distance_m,
        max_match_distance_m=cfg.matching.max_distance_m,
        gone_above=cfg.matching.gone_above, standing_below=cfg.matching.standing_below,
        dbh_plausible_mm=list(cfg.cleaning.dbh_plausible_mm),
        standing_height_dm_above=cfg.cleaning.standing_height_dm_above,
        top_found_distance_m=cfg.baseline.top_found_distance_m,
        ellipse_axis_ratio=cfg.dbh.ellipse_axis_ratio)
    entries = list(entries) + analyses.compute_entries(cfg, tables_dir)
    try:                                   # core/volume.py is written by another stage; optional until it exists
        from core import volume
    except ImportError:
        volume = None
    if volume is not None:
        entries += volume.compute_entries(cfg, tables_dir)
    try:                                   # GRI volume functions; no entries without coefficient file and species codes
        from core import volume_gri
    except ImportError:
        volume_gri = None
    if volume_gri is not None:
        entries += volume_gri.compute_entries(cfg, tables_dir)
    try:                                   # volume increment (ForestElementsR); no entries without the exported files
        from core import increment
    except ImportError:
        increment = None
    if increment is not None:
        entries += increment.compute_entries(cfg, tables_dir)
    if write_tables:
        paths = analyses.write_tables(cfg, tables_dir, rasters_dir=rasters_dir, height_variants=height_variants)
        print(f"{len(paths)} analysis tables written to {cfg.tables_dir}")
        if volume is not None and hasattr(volume, 'run_volume'):
            volume.run_volume(cfg, tables_dir)
        if volume_gri is not None:
            volume_gri.run_volume_gri(cfg, tables_dir)
        if increment is not None:
            increment.run_increment(cfg, tables_dir)
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'stats.json'
    path.write_text(json.dumps(entries, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f"{len(entries)} statistics written to {path}")
    return path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('config')
    parser.add_argument('--tables-dir', default=None, help='read the tables from this folder instead of cfg.tables_dir')
    parser.add_argument('--write-tables', action='store_true', help='also write the analysis (and volume) tables to cfg.tables_dir')
    parser.add_argument('--height-variants', action='store_true', help='with --write-tables: run the height-variant analysis (one pass over the cloud)')
    parser.add_argument('--rasters-dir', default=None, help='folder with dtm_lidar_after.tif and chm_lidar_before.tif (default cfg.output_dir/rasters)')
    args = parser.parse_args(argv)
    run(load_config(args.config), Path(args.tables_dir) if args.tables_dir else None, args.write_tables, args.height_variants,
        args.rasters_dir)


if __name__ == '__main__':
    main()
