#!/usr/bin/env python3
"""Regression for the detached sajda bar on page 251."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def fail(message: str) -> None:
    raise SystemExit(f"sajda-bar regression failed: {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--database", required=True, type=Path)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="mushaf-sajda-bar-") as output:
        output_root = Path(output)
        command = [
            str(args.cli),
            "--input",
            str(args.input),
            "--out",
            str(output_root),
            "--pages",
            "251",
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
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode != 0:
            fail(
                "mushaf_cli failed with exit code "
                f"{completed.returncode}: {completed.stderr}"
            )

        words_path = output_root / "page251" / "words.json"
        data = json.loads(words_path.read_text(encoding="utf-8"))
        components = {
            component["componentIndex"]: component
            for component in data["components"]
        }
        bars = [
            component
            for component in components.values()
            if component.get("ignored")
            and component.get("lineIndex") == -1
            and component.get("h", 0) > 0
            and component.get("w", 0) / component["h"] >= 24.0
        ]
        if len(bars) != 1:
            fail(f"expected one ignored sajda bar, got {bars}")
        bar = bars[0]
        correction = (
            f"ignored-sajda-bar component={bar['componentIndex']} line=2"
        )
        if correction not in data.get("assignmentCorrections", []):
            fail(f"missing sajda-bar correction: {data.get('assignmentCorrections')}")

        word = next(
            word
            for line in data["lines"]
            for word in line["words"]
            if word["wordId"] == "l02_w04"
        )
        if bar["componentIndex"] in word["componentIndices"]:
            fail(
                "sajda bar leaked into l02_w04: "
                f"{word['componentIndices']}"
            )

        report = json.loads(
            (output_root / "validation-report.json").read_text(encoding="utf-8")
        )
        page = report["pages"][0]
        if page.get("ignoredInkPixels", 0) <= 0:
            fail(f"ignored ink was not reported: {page}")


if __name__ == "__main__":
    main()
