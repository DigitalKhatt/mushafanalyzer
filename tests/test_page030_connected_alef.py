#!/usr/bin/env python3
"""Regression for an Alef/body component fused at a word boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def fail(message: str) -> None:
    raise SystemExit(f"page-030 connected-Alef regression failed: {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--database", required=True, type=Path)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="mushaf-page030-connected-alef-") as output:
        output_root = Path(output)
        command = [
            str(args.cli),
            "--input",
            str(args.input),
            "--out",
            str(output_root),
            "--pages",
            "30",
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
            (output_root / "page030" / "words.json").read_text(encoding="utf-8")
        )
        words = {
            word["wordId"]: word
            for line in data["lines"]
            for word in line["words"]
        }

        connected = words.get("l04_w08")
        if not connected:
            fail("missing l04_w08")
        first, second = connected["paws"][:2]
        if first["bodyComponentIndices"]:
            fail(f"l04_w08 PAW 0 still owns a body: {first}")
        if not first.get("bodyConnectedToPreviousWord"):
            fail(f"l04_w08 boundary condition was not reported: {first}")
        if 375 not in second["bodyComponentIndices"]:
            fail(f"l04_w08 connected body was not handed to PAW 1: {second}")

        # A normal narrow Alef remains a body even when it follows another
        # word on the same line.
        control = words.get("l04_w11")
        if not control or control["paws"][0]["bodyComponentIndices"] != [392]:
            fail(f"normal Alef changed unexpectedly: {control}")


if __name__ == "__main__":
    main()
