#!/usr/bin/env python3
"""Select a minimal page-disjoint seed set containing small yeh."""

import argparse
import collections
import json
import pathlib
import sqlite3


def split_name(page):
    if page % 10 == 0:
        return "test"
    if page % 10 == 5:
        return "validation"
    return "train"


def image_exists(directory, page):
    for stem in (str(page), f"{page:03d}"):
        for extension in (".png", ".jpg", ".jpeg"):
            if (directory / f"{stem}{extension}").is_file():
                return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=pathlib.Path, required=True)
    parser.add_argument("--images", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--train-occurrences", type=int, default=80)
    parser.add_argument("--validation-occurrences", type=int, default=25)
    parser.add_argument("--test-occurrences", type=int, default=25)
    args = parser.parse_args()

    database = sqlite3.connect(args.database)
    rows = database.execute(
        """
        SELECT l.page, l.line, w.word_number_all, w.dk_v1
        FROM qpc_v1_layout l
        JOIN words w
          ON l.type = 'ayah'
         AND w.word_number_all BETWEEN l.range_start AND l.range_end
        WHERE instr(w.dk_v1, ?) > 0 OR instr(w.dk_v1, ?) > 0
        ORDER BY l.page, l.line, w.word_number_all
        """,
        ("\u06e6", "\u06e7"),
    ).fetchall()

    occurrences = collections.Counter(row[0] for row in rows)
    targets = {
        "train": args.train_occurrences,
        "validation": args.validation_occurrences,
        "test": args.test_occurrences,
    }
    selected = []
    unavailable = []
    for split, target in targets.items():
        total = 0
        candidates = sorted(
            (
                (page, count)
                for page, count in occurrences.items()
                if split_name(page) == split
            ),
            key=lambda item: (-item[1], item[0]),
        )
        for page, count in candidates:
            if not image_exists(args.images, page):
                unavailable.append(page)
                continue
            selected.append(
                {"page": page, "split": split, "occurrences": count}
            )
            total += count
            if total >= target:
                break
        if total < target:
            raise RuntimeError(
                f"Only {total}/{target} available {split} occurrences"
            )

    report = {
        "codepoints": ["U+06E6", "U+06E7"],
        "corpusOccurrences": len(rows),
        "corpusPages": len(occurrences),
        "targets": targets,
        "selectedOccurrences": sum(item["occurrences"] for item in selected),
        "selectedPages": selected,
        "pageArgument": ",".join(str(item["page"]) for item in selected),
        "unavailablePages": sorted(set(unavailable)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(report["pageArgument"])
    print(
        f"{len(selected)} pages, {report['selectedOccurrences']} occurrences; "
        f"report: {args.out}"
    )


if __name__ == "__main__":
    main()
