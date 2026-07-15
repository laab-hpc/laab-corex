#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <root_directory>")
        sys.exit(1)

    root_dir = Path(sys.argv[1]).resolve()

    if not root_dir.is_dir():
        print(f"Error: directory does not exist: {root_dir}")
        sys.exit(1)

    scripts = sorted(root_dir.rglob("work/laab_process.sh"))

    if not scripts:
        print("No laab_process.sh files found.")
        return

    for script in scripts:
        work_dir = script.parent

        print(f"\nRunning {script}")
        print(f"Working directory: {work_dir}")

        result = subprocess.run(
            ["bash", script.name],
            cwd=work_dir
        )

        if result.returncode != 0:
            print(
                f"Failed with return code {result.returncode}: "
                f"{script}"
            )


if __name__ == "__main__":
    main()