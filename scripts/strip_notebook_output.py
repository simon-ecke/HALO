"""Git clean filter: remove outputs and execution counts from a notebook read on stdin.

The working copy keeps its outputs; only the version stored in git is stripped. Set up once per clone:

    git config filter.strip-output.clean "python scripts/strip_notebook_output.py"
    git config filter.strip-output.smudge cat

and .gitattributes assigns the filter to *.ipynb.
"""

import json
import sys


def strip(nb: dict) -> dict:
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        cell.get("metadata", {}).pop("execution", None)
    nb.get("metadata", {}).pop("widgets", None)
    return nb


def main() -> None:
    nb = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    text = json.dumps(strip(nb), indent=1, ensure_ascii=False) + "\n"
    sys.stdout.buffer.write(text.encode("utf-8"))


if __name__ == "__main__":
    main()
