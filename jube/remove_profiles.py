#!/usr/bin/env python3

import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <directory>")
        sys.exit(1)

    root_dir = Path(sys.argv[1]).resolve()

    if not root_dir.is_dir():
        print(f"Error: directory does not exist: {root_dir}")
        sys.exit(1)

    laab_count = 0
    lock_count = 0

    for path in root_dir.rglob("*.laab"):
        if path.is_file() or path.is_symlink():
            try:
                path.unlink()
                laab_count += 1
                print(f"Removed: {path}")
            except OSError as error:
                print(f"Failed to remove {path}: {error}")

    for path in root_dir.rglob("*.laab.lock"):
        if path.is_file() or path.is_symlink():
            try:
                path.unlink()
                lock_count += 1
            except OSError as error:
                print(f"Failed to remove {path}: {error}")

    print(
        f"\nRemoved {laab_count} .laab files and "
        f"{lock_count} .laab.lock files."
    )


if __name__ == "__main__":
    main()
