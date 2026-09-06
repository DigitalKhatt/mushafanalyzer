#!/usr/bin/env python3
"""Regression for a standalone Alef merged toward the preceding word."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def fail(message: str) -> None:
    raise SystemExit(f"page-565 connected-Alef regression failed: {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--database", required=True, type=Path)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="mushaf-page565-connected-alef-") as output:
        output_root = Path(output)
        command = [
            str(args.cli),
            "--input",
            str(args.input),
            "--out",
            str(output_root),
            "--pages",
            "565",
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
            (output_root / "page565" / "words.json").read_text(encoding="utf-8")
        )
        words = {
            word["wordId"]: word
            for line in data["lines"]
            for word in line["words"]
            if word["lineIndex"] == 10
        }

        w01 = words.get("l10_w01")
        w02 = words.get("l10_w02")
        if not w01 or not w02:
            fail("missing l10_w01 or l10_w02")

        # Narrow, separable Alef remains a normal first PAW.
        if w01["paws"][0]["bodyComponentIndices"] != [921]:
            fail(f"l10_w01 Alef changed unexpectedly: {w01['paws'][0]}")

        first = w02["paws"][0]
        second = w02["paws"][1]
        if first["bodyComponentIndices"]:
            fail(f"connected Alef PAW still owns a body: {first}")
        if not first.get("bodyConnectedToPreviousWord"):
            fail(f"connected Alef condition was not reported: {first}")
        if 916 not in second["bodyComponentIndices"]:
            fail(f"connected body was not moved to PAW 1: {second}")
        if 908 in second["bodyComponentIndices"]:
            fail(f"mark-like component 908 was still used as the body: {second}")


if __name__ == "__main__":
    main()
