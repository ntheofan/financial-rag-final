"""Remove execution state from source notebooks using only the standard library."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def clear_notebook(path: Path) -> tuple[int, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cleared_outputs = 0
    cleared_counts = 0

    for cell in payload.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        outputs = cell.get("outputs", [])
        cleared_outputs += len(outputs)
        cell["outputs"] = []
        if cell.get("execution_count") is not None:
            cleared_counts += 1
        cell["execution_count"] = None

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return cleared_outputs, cleared_counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "notebooks",
        nargs="*",
        type=Path,
        default=sorted((REPO_ROOT / "notebooks").glob("*.ipynb")),
    )
    args = parser.parse_args()

    for notebook in args.notebooks:
        path = notebook if notebook.is_absolute() else REPO_ROOT / notebook
        outputs, counts = clear_notebook(path)
        print(f"{path.name}: cleared {outputs} outputs and {counts} execution counts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
