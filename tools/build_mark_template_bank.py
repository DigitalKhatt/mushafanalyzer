#!/usr/bin/env python3
"""Build a ComponentRecognizer template bank from reviewed mark labels.

The output layout is:

    <out>/<visual-class>/<record-id>.png
    <out>/<visual-class>/<template-role>/<record-id>.png

The PNGs are the normalized component masks emitted by
``--export-component-dataset``.  Use several reviewed examples per class;
the recognizer compares a candidate with the best template in each class and
role. ``templateRole`` defaults to ``whole``; non-whole roles are written to
a subdirectory, for example ``kasratan/open``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


DEFAULT_CLASSES = (
    "fatha",
    "damma",
    "dot_above_one",
    "dot_above_two",
    "dot_above_three",
    "hamza_above",
    "hamza_below",
    "maddah",
    "small_alef",
    "sukun",
    "kasra",
    "kasratan",
    "fathatan",
    "dammatan",
    "shadda",
    "dot_below_one",
    "dot_below_two",
    "wasla",
    "waqf_jeem",
    "waqf_qaf",
    "waqf_sad",
    "waqf_lam",
    "waqf_meem",
    "waqf_three_dots",
)


def resolve_mask(path_text: str, annotations: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute() and path.is_file():
        return path
    candidates = (Path.cwd() / path, annotations.parent / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(path_text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--class",
        dest="classes",
        action="append",
        help=(
            "Class to export; repeatable (default: fatha, damma, "
            "dot above one/two/three, hamza above/below, maddah, small alef, "
            "sukun, kasra/tanwins, shadda, below dots, wasla, and waqf classes)"
        ),
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="Maximum templates per class, or 0 for all reviewed records",
    )
    parser.add_argument(
        "--role",
        dest="roles",
        action="append",
        help="Template role to export; repeatable (default: all roles)",
    )
    args = parser.parse_args()

    classes = set(args.classes or DEFAULT_CLASSES)
    roles = set(args.roles or [])
    records = [
        json.loads(line)
        for line in args.annotations.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    counts = {name: 0 for name in classes}
    copied = 0
    for record in records:
        class_name = record.get("visualClass")
        if not record.get("reviewed") or class_name not in classes:
            continue
        role = record.get("templateRole", "whole")
        if roles and role not in roles:
            continue
        if args.max_per_class and counts[class_name] >= args.max_per_class:
            continue
        source = resolve_mask(record["maskPath"], args.annotations)
        destination_dir = args.out / class_name
        if role != "whole":
            destination_dir /= role
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"{record['id']}.png"
        shutil.copyfile(source, destination)
        counts[class_name] += 1
        copied += 1

    if not copied:
        raise SystemExit("No reviewed templates were exported")
    for class_name in sorted(counts):
        print(f"{class_name}: {counts[class_name]}")


if __name__ == "__main__":
    main()
