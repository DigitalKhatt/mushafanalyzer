#!/usr/bin/env python3
"""Verify that diagnostic output is controlled by --debug-images."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile


DEBUG_PAGE_FILES = (
    "debug_bin.png",
    "debug_overlay.png",
    "debug_component_insertions.png",
    "debug_reconstructed.png",
    "debug_geometry.png",
    "debug_reconstruction_difference.png",
    "component-insertion-gaps.json",
    "reconstruction-report.json",
)


def fail(message: str) -> None:
    raise SystemExit(f"debug output regression failed: {message}")


def run(cli: Path, input_root: Path, database: Path, output: Path,
        *extra: str) -> None:
    command = [
        str(cli),
        "--input", str(input_root),
        "--out", str(output),
        "--pages", "3",
        "--force",
        "--database", str(database),
        "--blurK", "0",
        "--morphK", "0",
        "--recognizer-cache", str(output / "recognizer-cache"),
        "--recognizer-source", "debug-output-test",
        *extra,
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        fail(f"mushaf_cli failed: {completed.stdout}\n{completed.stderr}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--database", required=True, type=Path)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="mushaf-debug-output-") as root:
        root_path = Path(root)
        no_debug = root_path / "no-debug"
        no_debug.mkdir(parents=True)
        (no_debug / "validation-contact-sheet.png").write_bytes(b"stale")
        stale_page = no_debug / "page003"
        stale_page.mkdir(parents=True)
        for filename in DEBUG_PAGE_FILES:
            (stale_page / filename).write_bytes(b"stale")

        run(args.cli, args.input, args.database, no_debug)
        for filename in DEBUG_PAGE_FILES:
            if (no_debug / "page003" / filename).exists():
                fail(f"debug artifact written without --debug-images: {filename}")
        if (no_debug / "validation-contact-sheet.png").exists():
            fail("stale validation contact sheet was not removed")

        with_debug = root_path / "with-debug"
        run(args.cli, args.input, args.database, with_debug, "--debug-images")
        for filename in DEBUG_PAGE_FILES:
            if not (with_debug / "page003" / filename).is_file():
                fail(f"debug artifact missing with --debug-images: {filename}")
        if not (with_debug / "validation-contact-sheet.png").is_file():
            fail("validation contact sheet missing with --debug-images")

    print("debug output opt-in verified")


if __name__ == "__main__":
    main()
