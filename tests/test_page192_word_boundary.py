#!/usr/bin/env python3
"""Regression for the wide initial-kaf body crossing page 192's word boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def fail(message: str) -> None:
    raise SystemExit(f"page-192 word-boundary regression failed: {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--database", required=True, type=Path)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="mushaf-page192-boundary-") as output:
        output_root = Path(output)
        command = [
            str(args.cli),
            "--input",
            str(args.input),
            "--out",
            str(output_root),
            "--pages",
            "192",
            "--database",
            str(args.database),
            "--analysis-only",
            "--no-debug-reconstructed",
            "--blurK",
            "0",
            "--morphK",
            "0",
            "--recognizer-cache",
            str(output_root / "recognizer-cache"),
            "--recognizer-source",
            "old",
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode != 0:
            fail(
                "mushaf_cli failed with exit code "
                f"{completed.returncode}: {completed.stderr}"
            )

        data = json.loads(
            (output_root / "page192" / "words.json").read_text(encoding="utf-8")
        )
        words = {
            word["wordId"]: word
            for line in data["lines"]
            for word in line["words"]
            if word["lineIndex"] == 4
        }
        first = words.get("l04_w01")
        second = words.get("l04_w02")
        if not first or not second:
            fail(f"missing target words: {sorted(words)}")

        if 290 in first["componentIndices"]:
            fail(f"component 290 leaked into l04_w01: {first['componentIndices']}")
        if 290 not in second["componentIndices"]:
            fail(f"component 290 missing from l04_w02: {second['componentIndices']}")
        if 308 not in first["componentIndices"]:
            fail(f"component 308 missing from l04_w01: {first['componentIndices']}")
        if 283 not in second["componentIndices"]:
            fail(f"component 283 missing from l04_w02: {second['componentIndices']}")
        if first["bboxGapToNextPx"] < 0 or first["spacingGapToNextPx"] < 0:
            fail(f"word boundary still overlaps: {first}")

        first_paw = second["paws"][0]
        if first_paw["bodyComponentIndices"] != [290]:
            fail(
                "initial kaf body was not selected: "
                f"{first_paw['bodyComponentIndices']}"
            )
        if 283 not in first_paw["markComponentIndices"]:
            fail(f"component 283 was not retained as a mark: {first_paw}")


if __name__ == "__main__":
    main()
