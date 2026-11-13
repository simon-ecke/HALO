"""Execute notebooks/04_dashboard.ipynb end to end (slow). The executed copy is not written back."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "04_dashboard.ipynb"


@pytest.mark.slow
def test_dashboard_notebook_runs(cfg155):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    if not (Path(cfg155.tables_dir) / "field_trees_detection.csv").exists():
        pytest.skip("pipeline tables of site 155 are not available")

    nb = nbformat.read(NOTEBOOK, as_version=4)
    kernel = nb.metadata.get("kernelspec", {}).get("name", "python3")
    client = nbclient.NotebookClient(nb, timeout=7200, kernel_name=kernel,
                                     resources={"metadata": {"path": str(NOTEBOOK.parent)}})
    client.execute()                                      # result stays in memory only

    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        errors = [o for o in cell.get("outputs", []) if o.get("output_type") == "error"]
        assert not errors, f"cell {i} failed: {errors[0].get('ename')}: {errors[0].get('evalue')}"

    dist = ROOT / "dist" / str(cfg155.site.id)
    assert (dist / "index.html").exists()
    assert (dist / "data.js").exists()
