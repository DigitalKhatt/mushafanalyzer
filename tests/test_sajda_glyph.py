#!/usr/bin/env python3
"""Regression for the two-component U+06E9 sajda glyph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def fail(message: str) -> None:
    raise SystemExit(f"sajda-glyph regression failed: {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--database", required=True, type=Path)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="mushaf-sajda-glyph-") as output:
        output_root = Path(output)
        command = [
            str(args.cli),
            "--input",
            str(args.input),
            "--out",
            str(output_root),
            "--pages",
            "365",
            "--force",
            "--database",
            str(args.database),
            "--mark-template-root",
            str(args.input.parent.parent / "templates" / "oldmadina" / "mark-templates"),
            "--blurK",
            "0",
            "--morphK",
            "0",
            "--no-debug-reconstructed",
            "--recognizer-cache",
            str(output_root / "recognizer-cache"),
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode != 0:
            fail(
                "mushaf_cli failed with exit code "
                f"{completed.returncode}: {completed.stderr}"
            )

        page_dir = output_root / "page365"
        data = json.loads((page_dir / "words.json").read_text(encoding="utf-8"))
        components = {
            component["componentIndex"]: component
            for component in data["components"]
        }
        word = next(
            word
            for line in data["lines"]
            for word in line["words"]
            if word["wordId"] == "l06_w04"
        )
        paw = next(paw for paw in word["paws"] if paw["text"] == "ا۩")

        if len(paw["bodyComponentIndices"]) != 1:
            fail(f"expected one Alef body component, got {paw}")
        body_index = paw["bodyComponentIndices"][0]
        body = components[body_index]
        if body["w"] > 250 or body["h"] < 500:
            fail(f"the Alef body was not selected: component={body}")

        quranic_masks = [
            mark
            for mark in paw["markMasks"]
            if mark["className"] == "quranic_sign"
        ]
        if len(quranic_masks) != 1:
            fail(f"expected one quranic_sign mask, got {paw['markMasks']}")
        quranic_components = quranic_masks[0]["componentIndices"]
        if len(quranic_components) != 2:
            fail(f"expected two sajda components, got {quranic_masks}")
        if body_index in quranic_components:
            fail(f"Alef was merged into the sajda mask: {quranic_masks}")
        if set(quranic_components) != set(paw["markComponentIndices"]):
            fail(f"sajda components were not the complete mark range: {paw}")
        if not Path(quranic_masks[0]["path"]).is_file():
            fail(f"missing exported sajda mask: {quranic_masks[0]['path']}")


if __name__ == "__main__":
    main()
