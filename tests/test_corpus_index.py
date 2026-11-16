#!/usr/bin/env python3
"""Integration regression for the letter/mark corpus index.

Builds a slice of an analyzed page through the real builder, then checks the
invariants that a downstream query depends on: every ink pixel is attributed
to exactly one letter, every letter that has ink has an outline, ids are
deterministic, the normalized frame round-trips, and a mark row means one mark.

    uv run tests/test_corpus_index.py \
        --analysis output/oldmedina2-analysis --page 3 --words 12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys
import tempfile

import cv2

CORPUS = Path(__file__).resolve().parents[1] / "tools" / "corpus"
sys.path.insert(0, str(CORPUS))

import build_corpus  # noqa: E402
from geometry import decode_mask_rle  # noqa: E402
from letter_classes import mark_parts, shape_class  # noqa: E402


def fail(message: str) -> None:
    raise SystemExit(f"corpus index regression failed: {message}")


def build(analysis: Path, page: int, templates: str, words: int,
          database: Path) -> sqlite3.Connection:
    options = {
        "min_confidence": 0.25,
        "skeleton_shapes": False,
        "logical_ligatures": False,
        # What the shipped default and both VS Code tasks use. With cut-based
        # attribution there are no join spans at all, so a test built that way
        # cannot see the kashida rules it is meant to be checking.
        "body_join_alignment": True,
        "cut_policy": "next-start",
        "templates": templates,
        "trace": True,
        "limit_words": words,
    }
    result = build_corpus.build_page(str(analysis), page, options)
    if result.errors:
        fail(f"builder reported errors: {result.errors[0]}")
    estimate = build_corpus.normalize_page(result)
    connection = build_corpus.open_database(database)
    build_corpus.write_page(connection, result, estimate, "test")
    connection.row_factory = sqlite3.Row
    return connection


def body_path(page_dir: Path, recorded: str) -> Path:
    """Resolve a PAW body image the way the segmenter does.

    ``bodyPath`` is an absolute path baked in by the C++ analyzer, so renaming
    the analysis directory breaks every one of them. segment_paw_letters falls
    back to the page's own subwordimgs; without the same fallback this check
    silently examines nothing.
    """
    if not recorded:
        return None
    direct = Path(recorded)
    if direct.is_absolute() and direct.is_file():
        return direct
    beside = page_dir / "subwordimgs" / direct.name
    return beside if beside.is_file() else None


def check_ink_conservation(connection: sqlite3.Connection, analysis: Path,
                           page: int) -> None:
    """Letters must partition their PAW's ink: no pixel lost, none counted twice.

    Connector ink counts here, which is the point: it belongs to the letter on
    its right, so a PAW's letters together still cover every pixel of its body.
    PAWs holding a fused group are excluded -- a template that consumed several
    letters without cutting leaves one body owned by no single letter, and that
    is a deliberate abstention rather than lost ink.
    """
    page_dir = analysis / f"page{page:03d}"
    words = {word["wordId"]: word
             for line in json.loads((page_dir / "words.json").read_text())["lines"]
             for word in line["words"]}
    rows = connection.execute(
        "SELECT p.paw_id, p.word_id, p.paw_index, SUM(l.ink_px) AS ink "
        "FROM paws p JOIN letters l ON l.paw_id = p.paw_id "
        "GROUP BY p.paw_id "
        "HAVING MAX(l.group_size) = 1").fetchall()
    if not rows:
        fail("no PAWs were indexed")
    checked = 0
    for row in rows:
        word_key = row["word_id"].split(":", 1)[1]
        metadata = (words[word_key].get("paws") or [])
        if row["paw_index"] >= len(metadata):
            continue
        body = body_path(page_dir, metadata[row["paw_index"]].get("bodyPath"))
        if body is None:
            continue
        image = cv2.imread(str(body), cv2.IMREAD_GRAYSCALE)
        ink = int((image > 127).sum())
        if ink != row["ink"]:
            fail(f"{row['paw_id']} attributes {row['ink']} px of {ink} px of ink")
        checked += 1
    if not checked:
        fail("no PAW body images were available to check")
    print(f"  ink conservation exact on {checked} PAWs")


def check_shapes(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
        "SELECT l.letter_id, l.ink_px, s.path_d, s.mask_rle, s.mask_w, s.mask_h "
        "FROM letters l JOIN letter_shape s USING (letter_id)").fetchall()
    if len(rows) != connection.execute(
            "SELECT COUNT(*) FROM letters").fetchone()[0]:
        fail("some letters have no letter_shape row")
    for row in rows:
        if not row["ink_px"]:
            continue
        if not row["path_d"]:
            fail(f"{row['letter_id']} has ink but no traced outline")
        mask = decode_mask_rle(row["mask_rle"], row["mask_w"], row["mask_h"])
        if int(mask.sum()) != row["ink_px"]:
            fail(f"{row['letter_id']} mask holds {int(mask.sum())} px, "
                 f"row says {row['ink_px']}")
    print(f"  {len(rows)} letters carry a mask and an outline that agree")


def check_kashida(connection: sqlite3.Connection) -> None:
    """A letter is measured with the connector leaving it, not without.

    The segmenter labels join ink as a state of its own, belonging to neither
    neighbour; the index charges it to the letter on its right. If that ever
    stops happening the widths silently become body-only again, which reads as
    a plausible number rather than as a failure.
    """
    rows = connection.execute(
        "SELECT letter_id, spine_extent_px, kashida_px, advance_width_px, "
        "       ink_px, group_size "
        "FROM letters WHERE ok = 1 AND kashida_px IS NOT NULL").fetchall()
    if not rows:
        fail("no letters carry a kashida measurement")
    joined = 0
    for row in rows:
        if row["kashida_px"] < 0:
            fail(f"{row['letter_id']} has a negative kashida")
        if row["kashida_px"] > row["spine_extent_px"] + 1e-6:
            fail(f"{row['letter_id']} kashida {row['kashida_px']:.1f} exceeds "
                 f"its span {row['spine_extent_px']:.1f}")
        if row["kashida_px"] > 0:
            joined += 1
            if row["group_size"] == 1 and not row["ink_px"]:
                fail(f"{row['letter_id']} has a kashida but no ink")
    if not joined:
        fail("no letter on this page owns a connector, which cannot be right")
    print(f"  {joined} of {len(rows)} letters include their left kashida")


def check_identifiers(connection: sqlite3.Connection, page: int) -> None:
    """Ids must be re-derivable, so a rebuild keeps its review decisions."""
    rows = connection.execute(
        "SELECT l.letter_id, l.letter_index, w.word_key, p.paw_index "
        "FROM letters l JOIN paws p USING (paw_id) "
        "JOIN words w ON w.word_id = l.word_id").fetchall()
    for row in rows:
        expected = build_corpus.letter_id(page, row["word_key"],
                                          row["paw_index"], row["letter_index"])
        if expected != row["letter_id"]:
            fail(f"id {row['letter_id']} does not re-derive as {expected}")
    unique = connection.execute(
        "SELECT COUNT(DISTINCT letter_id) FROM letters").fetchone()[0]
    if unique != len(rows):
        fail("letter ids are not unique")
    print(f"  {len(rows)} letter ids are unique and deterministic")


def check_frame(connection: sqlite3.Connection) -> None:
    """The normalized frame must map back onto the letter's own pixels.

    Origin at the entry point on the baseline, y up, x along the writing
    direction, scaled to qalam widths. Inverting that has to land the stored
    view box back on the mask's bounding box -- a sign flip or a wrong scale
    moves it a whole letter away.

    The two boxes agree closely but not exactly, in both directions, and
    neither gap is a frame error: the view box spans control points rather
    than on-curve points, so it runs slightly wide, while tiny speckle
    contours are dropped before tracing, so it can also run slightly narrow.
    Box overlap is therefore the honest test.
    """
    rows = connection.execute(
        "SELECT l.letter_id, l.bbox_x, l.bbox_y, l.bbox_w, l.bbox_h, "
        "s.view_x, s.view_y, s.view_w, s.view_h, s.entry_x, s.baseline_y, "
        "s.norm_scale FROM letters l JOIN letter_shape s USING (letter_id) "
        "WHERE s.path_d != '' AND l.bbox_x IS NOT NULL").fetchall()
    if not rows:
        fail("no traced letters to check the frame with")
    worst = 1.0
    culprit = ""
    for row in rows:
        scale = row["norm_scale"]
        right = row["entry_x"] - row["view_x"] * scale
        left = row["entry_x"] - (row["view_x"] + row["view_w"]) * scale
        bottom = row["baseline_y"] - row["view_y"] * scale
        top = row["baseline_y"] - (row["view_y"] + row["view_h"]) * scale
        mask = (row["bbox_x"], row["bbox_y"],
                row["bbox_x"] + row["bbox_w"], row["bbox_y"] + row["bbox_h"])
        overlap = (max(0.0, min(right, mask[2]) - max(left, mask[0]))
                   * max(0.0, min(bottom, mask[3]) - max(top, mask[1])))
        union = ((right - left) * (bottom - top)
                 + (mask[2] - mask[0]) * (mask[3] - mask[1]) - overlap)
        score = overlap / union if union > 0 else 0.0
        if score < worst:
            worst, culprit = score, row["letter_id"]
    if worst < 0.8:
        fail(f"{culprit}: outline box and mask box overlap only {worst:.2f}; "
             "the normalized frame is wrong")
    print(f"  frame round-trips: worst outline/mask box overlap {worst:.2f}")


def check_marks(connection: sqlite3.Connection, analysis: Path,
                page: int) -> None:
    """One row per mark instance, under whichever mask contract wrote the page.

    With per-instance masks the analyzer decides what one mark is and a mask
    entry is a row. With the older class-grouped masks a single-component
    class still has to be split, or two fathas over one PAW would be indexed
    as one mark spanning both.
    """
    page_dir = analysis / f"page{page:03d}"
    page_json = json.loads((page_dir / "words.json").read_text())
    mark_format = build_corpus.mark_mask_format(page_json)
    indexed = {row["word_id"].split(":", 1)[1]
               for row in connection.execute("SELECT DISTINCT word_id FROM marks")}
    expected: dict[str, int] = {}
    for line in page_json["lines"]:
        for word in line["words"]:
            if word["wordId"] not in indexed:
                continue
            for paw in word.get("paws") or []:
                for mask in paw.get("markMasks") or []:
                    name = mask.get("className") or "unknown"
                    count = len(mask.get("componentIndices") or [])
                    if not count:
                        continue
                    per_entry = (1 if mark_format == "instance"
                                 else (count if mark_parts(name) == 1 else 1))
                    expected[name] = expected.get(name, 0) + per_entry
    for name, count in expected.items():
        actual = connection.execute(
            "SELECT COUNT(*) FROM marks WHERE class = ?", (name,)).fetchone()[0]
        if actual != count:
            fail(f"class {name}: indexed {actual} marks, expected {count}")
    stored = connection.execute(
        "SELECT mark_format FROM pages").fetchone()[0]
    if stored != mark_format:
        fail(f"page row says mark_format {stored!r}, data says {mark_format!r}")
    total = connection.execute("SELECT COUNT(*) FROM marks").fetchone()[0]
    print(f"  {total} mark rows match the {mark_format} mask contract")


def check_mask_contracts() -> None:
    """Both mark-mask contracts, on synthetic pages.

    The per-instance masks are what `PageProcessor.cpp` writes now, so this
    has to hold before any page has been regenerated with them: two fathas
    over one PAW are two rows, and a dammatan's two components stay one.
    """
    components = [{"x": 10 * i, "y": 5, "w": 8, "h": 6, "area": 40}
                  for i in range(6)]

    def page_of(masks):
        return {"lines": [{"words": [{
            "wordId": "l00_w00", "x": 0, "y": 0,
            "paws": [{"pawIndex": 0, "markMasks": masks}]}]}]}

    instance = [
        {"className": "fatha", "componentIndices": [0],
         "path": "/x/l00_w00_paw00_fatha_00.png"},
        {"className": "fatha", "componentIndices": [1],
         "path": "/x/l00_w00_paw00_fatha_01.png"},
        {"className": "dammatan", "componentIndices": [2, 3],
         "path": "/x/l00_w00_paw00_dammatan_02.png"},
    ]
    legacy = [
        {"className": "fatha", "componentIndices": [0, 1],
         "path": "/x/l00_w00_paw00_fatha.png"},
        {"className": "dammatan", "componentIndices": [2, 3],
         "path": "/x/l00_w00_paw00_dammatan.png"},
    ]

    for masks, want_format, want_rows, want_grouping in (
            (instance, "instance", 3, "instance"),
            (legacy, "class-group", 3, None)):
        page_json = page_of(masks)
        found = build_corpus.mark_mask_format(page_json)
        if found != want_format:
            fail(f"mask format detected as {found!r}, expected {want_format!r}")
        build = build_corpus.PageBuild(page=1)
        rows = build_corpus.collect_marks(
            1, page_json["lines"][0]["words"][0], components, {}, "/x",
            found, build)
        if len(rows) != want_rows:
            fail(f"{want_format}: {len(rows)} mark rows, expected {want_rows}")
        fathas = [r for r in rows if r["class"] == "fatha"]
        if len(fathas) != 2:
            fail(f"{want_format}: two fathas over one PAW became "
                 f"{len(fathas)} rows")
        dammatan = [r for r in rows if r["class"] == "dammatan"]
        if len(dammatan) != 1 or dammatan[0]["component_count"] != 2:
            fail(f"{want_format}: dammatan should be one row of two components")
        if want_grouping and any(r["grouping"] != want_grouping for r in rows):
            fail(f"{want_format}: rows should be grouped as {want_grouping}")
        if len({r["mark_id"] for r in rows}) != len(rows):
            fail(f"{want_format}: mark ids are not unique")

    empty = page_of([{"className": "fatha", "componentIndices": [],
                      "path": "/x/l00_w00_paw00_fatha_00.png"}])
    build = build_corpus.PageBuild(page=1)
    rows = build_corpus.collect_marks(
        1, empty["lines"][0]["words"][0], components, {}, "/x", "instance",
        build)
    if rows or build.marks_unindexed != 1:
        fail("a mask with no component indices must be counted, not dropped "
             "silently")
    print("  both mark-mask contracts give one row per mark instance")


def check_classes() -> None:
    cases = [("ب", "initial", "senna"), ("ب", "final", "beh"),
             ("ن", "medial", "senna"), ("ن", "isolated", "noon"),
             ("ت", "initial", "senna"), ("ج", "medial", "hah")]
    for char, form, expected in cases:
        actual = shape_class(char, form)
        if actual != expected:
            fail(f"shape_class({char!r}, {form!r}) = {actual!r}, "
                 f"expected {expected!r}")
    print("  shape classes fold dotted variants and the senna tooth")


def check_idempotent(connection: sqlite3.Connection, analysis: Path, page: int,
                     templates: str, words: int) -> None:
    before = connection.execute("SELECT COUNT(*) FROM letters").fetchone()[0]
    options = {
        "min_confidence": 0.25, "skeleton_shapes": False,
        "logical_ligatures": False, "body_join_alignment": False,
        "cut_policy": "next-start", "templates": templates, "trace": False,
        "limit_words": words,
    }
    result = build_corpus.build_page(str(analysis), page, options)
    estimate = build_corpus.normalize_page(result)
    build_corpus.write_page(connection, result, estimate, "test-again")
    after = connection.execute("SELECT COUNT(*) FROM letters").fetchone()[0]
    if before != after:
        fail(f"rebuilding the page changed the row count: {before} -> {after}")
    print(f"  rebuilding the page leaves {after} letters, not duplicates")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path,
                        default=Path("output/oldmedina2-analysis"))
    parser.add_argument("--page", type=int, default=3)
    parser.add_argument("--words", type=int, default=12,
                        help="words per page to index; 0 indexes the page")
    parser.add_argument("--templates",
                        default=str(CORPUS.parent / ".." / "templates" / "oldmadina"
                                    / "skeleton_templates.json"))
    args = parser.parse_args()

    if not (args.analysis / f"page{args.page:03d}" / "words.json").is_file():
        fail(f"no analyzed page {args.page} under {args.analysis}")

    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / "corpus.sqlite"
        connection = build(args.analysis, args.page, args.templates,
                           args.words, database)
        check_classes()
        check_mask_contracts()
        check_ink_conservation(connection, args.analysis, args.page)
        check_shapes(connection)
        check_kashida(connection)
        check_identifiers(connection, args.page)
        check_frame(connection)
        check_marks(connection, args.analysis, args.page)
        check_idempotent(connection, args.analysis, args.page, args.templates,
                         args.words)
        connection.close()
    print("corpus index regression passed")


if __name__ == "__main__":
    main()
