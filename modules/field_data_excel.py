"""
field_data_excel.py
===================

Load and filter field measurement data from Excel (Vertex/Maßband heights).

Functions
---------
- load_field_trees_with_heights() -> DataFrame

"""

import pandas as pd
import os


def load_field_trees_with_heights(excel_path):
    """
    Load field trees from Excel, filter to rows with Vertex and/or Maßband heights.

    Args:
        excel_path (str): Path to 155_EBEe_mitVertexund Maßband.xlsx (or similar).

    Returns:
        pd.DataFrame: Columns [tree_id, parzelle, nr, local_x, local_y, height_vertex, height_massband].
                      tree_id = f"{parzelle}_{nr}".
                      Only rows where height_vertex or height_massband is non-null.
    """
    df = pd.read_excel(excel_path, sheet_name=0)

    # Column mapping (handle possible encoding issues in the file)
    vertex_col = 'Gesamthöhe_m_Vertex'
    massband_col = 'Gesamtlänge_m_Maßband'

    # Ensure columns exist
    if vertex_col not in df.columns:
        raise ValueError(f"Column '{vertex_col}' not found in Excel file.")
    if massband_col not in df.columns:
        raise ValueError(f"Column '{massband_col}' not found in Excel file.")

    # Filter to rows where at least one of Vertex or Massband is non-null
    mask = df[[vertex_col, massband_col]].notna().any(axis=1)
    df_subset = df[mask].copy()

    # Standardize columns
    # Parzelle/Nr are read as floats (the sheet has blank rows), so cast to int
    # first to get ids like '1_1009' instead of '1.0_1009.0'.
    df_subset['parzelle'] = pd.to_numeric(df_subset['Parzelle']).astype('Int64').astype(str)
    df_subset['nr'] = pd.to_numeric(df_subset['Nr']).astype('Int64').astype(str)
    df_subset['tree_id'] = df_subset['parzelle'] + '_' + df_subset['nr']
    df_subset['local_x'] = df_subset['x'].astype(float)
    df_subset['local_y'] = df_subset['y'].astype(float)
    df_subset['height_vertex'] = pd.to_numeric(df_subset[vertex_col], errors='coerce')
    df_subset['height_massband'] = pd.to_numeric(df_subset[massband_col], errors='coerce')

    # Select and return relevant columns
    result = df_subset[['tree_id', 'parzelle', 'nr', 'local_x', 'local_y', 'height_vertex', 'height_massband']].reset_index(drop=True)

    return result


def _first_column(df, candidates):
    """Return the first of `candidates` that is a column of df, else None."""
    for name in candidates:
        if name in df.columns:
            return name
    return None


def load_field_trees_inventory(excel_paths):
    """
    Load every tree standing before the 2026 harvest from one or more inventory files.

    Handles both layouts of the Versuch 155 sheets:
    - '155_EBEe_mitVertexund Maßband.xlsx' (Parzellen 1-26): diameter in
      'DurchmesserUmfang_2025', 2026 removal in 'removal_2026'.
    - '155_EBEe-P27-45_done_ausg 1.xlsx' (Parzellen 27-45): diameter in
      'DurchmesserUmfang_2026', 2026 removal in the column headed
      'removal Kronenansatz' (it is 1 exactly where the remark is 'BAD').

    Rows without a Parzelle (blank and totals rows) and trees with an entry in
    'removal' (removed in an earlier thinning) are dropped.

    Args:
        excel_paths (iterable): Paths of the inventory files.

    Returns:
        pd.DataFrame: Columns [tree_id, parzelle, nr, species, local_x, local_y, dbh_mm, dbh_2020_mm, dbh_year,
                      removed_2026, height_vertex, height_massband, height_standing_sheet, crown_base_m,
                      has_height, remark, source_file].
    """
    parts = []
    for path in excel_paths:
        df = pd.read_excel(path, sheet_name=0)
        df = df[df['Parzelle'].notna()].copy()

        earlier_removal = pd.to_numeric(df['removal'], errors='coerce').notna()
        df = df[~earlier_removal]

        dbh_col = _first_column(df, ['DurchmesserUmfang_2026', 'DurchmesserUmfang_2025'])
        removal_col = _first_column(df, ['removal_2026', 'removal Kronenansatz'])
        if dbh_col is None or removal_col is None:
            raise ValueError(f"{os.path.basename(str(path))}: no diameter or 2026 removal column found.")

        def numeric(col):
            if col in df.columns:
                return pd.to_numeric(df[col], errors='coerce')
            return pd.Series(float('nan'), index=df.index)

        parzelle = pd.to_numeric(df['Parzelle']).astype('Int64').astype(str)
        nr = pd.to_numeric(df['Nr']).astype('Int64').astype(str)
        parts.append(pd.DataFrame({
            'tree_id': parzelle + '_' + nr,
            'parzelle': parzelle,
            'nr': nr,
            'species': df['fe_art'],
            'local_x': df['x'].astype(float),
            'local_y': df['y'].astype(float),
            'dbh_mm': numeric(dbh_col),
            # Diameter of the 2020 inventory and the year of the current one, for the diameter growth
            'dbh_2020_mm': numeric('DurchmesserUmfang_mm_2020'),
            'dbh_year': int(dbh_col[-4:]),
            'removed_2026': numeric(removal_col).eq(1),
            'height_vertex': numeric('Gesamthöhe_m_Vertex'),
            'height_massband': numeric('Gesamtlänge_m_Maßband'),
            # Height measured on the standing tree (Höhe_2025 / Höhe_2026), as typed in the sheet
            'height_standing_sheet': numeric(_first_column(df, ['Höhe_2026', 'Höhe_2025'])),
            # Crown base measured on the felled tree from the cut, plus the stump height
            'crown_base_m': (numeric('Kronenansatz_cm') + numeric('Stumpfhöhe_cm').fillna(0.0)) / 100.0,
            'remark': df['Bemerkung'].astype('string').str.strip() if 'Bemerkung' in df.columns else pd.NA,
            'source_file': os.path.basename(str(path)),
        }))

    result = pd.concat(parts, ignore_index=True)
    result['has_height'] = result[['height_vertex', 'height_massband']].notna().any(axis=1)
    if result['tree_id'].duplicated().any():
        dups = result.loc[result['tree_id'].duplicated(keep=False), 'tree_id'].unique().tolist()
        raise ValueError(f"Duplicate tree ids among standing trees: {dups}")

    return result
