"""Prepare datasets referenced by docs/README.md."""

from __future__ import annotations

import shutil
from pathlib import Path

SOURCE = Path("data/tests")
TARGET = Path("tests/fixtures")


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    for path in SOURCE.iterdir():
        if path.is_file():
            shutil.copy2(path, TARGET / path.name)
    print(f"Copied {len(list(SOURCE.iterdir()))} files into {TARGET}")


if __name__ == "__main__":
    main()
