#!/usr/bin/env python3
"""Build a searchable SQLite pilot of handwritten letter and mark instances."""

import argparse
import json
import pathlib
import sqlite3
import unicodedata

import cv2
import numpy as np


DUAL_JOIN = set("بتثجحخسشصضطظعغفقكلمنهيئى")
RIGHT_JOIN = set("آاٱأإدذرزوؤة")
BASES = DUAL_JOIN | RIGHT_JOIN | set("ءـ٠١٢٣٤٥٦٧٨٩۝۞")


def joining_form(bases, index):
    current = bases[index]
    joins_previous = index > 0 and bases[index - 1] in DUAL_JOIN | {"ـ"} and current != "ء"
    joins_next = index + 1 < len(bases) and current in DUAL_JOIN | {"ـ"} and bases[index + 1] != "ء"
    if joins_previous and joins_next:
        return "medial"
    if joins_previous:
        return "final"
    if joins_next:
        return "initial"
    return "isolated"


def resolve_path(raw, project):
    path = pathlib.Path(raw)
    if path.is_absolute():
        return path
    candidate = project / path
    return candidate if candidate.exists() else path.resolve()


def estimate_intervals(body, count):
    """Split RTL connected ink at approximately equal cumulative ink mass."""
    if count == 0:
        return []
    ink = body > 0
    projection = ink.sum(axis=0).astype(np.float64)
    occupied = np.flatnonzero(projection)
    if occupied.size == 0:
        width = body.shape[1]
        edges = np.linspace(width, 0, count + 1).round().astype(int)
        return [(min(edges[i], edges[i + 1]), max(edges[i], edges[i + 1]), 0.0)
                for i in range(count)]
    left, right = int(occupied[0]), int(occupied[-1]) + 1
    cumulative = np.cumsum(projection[left:right])
    total = cumulative[-1]
    ascending = [left]
    for part in range(1, count):
        target = total * part / count
        raw = left + int(np.searchsorted(cumulative, target))
        radius = max(3, (right - left) // max(8 * count, 1))
        lo, hi = max(left + 1, raw - radius), min(right - 1, raw + radius)
        raw = min(range(lo, hi + 1), key=lambda x: projection[x]) if lo <= hi else raw
        ascending.append(max(ascending[-1] + 1, raw))
    ascending.append(right)
    ascending = np.asarray(ascending)
    peak = max(float(projection.max()), 1.0)
    result = []
    for rtl_index in range(count):
        interval_index = count - rtl_index - 1
        x0, x1 = int(ascending[interval_index]), int(ascending[interval_index + 1])
        boundary_values = []
        if x0 > left:
            boundary_values.append(projection[x0] / peak)
        if x1 < right:
            boundary_values.append(projection[x1] / peak)
        confidence = 1.0 - float(np.mean(boundary_values)) if boundary_values else 0.65
        result.append((x0, x1, max(0.0, min(1.0, confidence))))
    return result


def schema(connection):
    connection.executescript("""
    PRAGMA journal_mode=WAL;
    DROP TABLE IF EXISTS letter_instances;
    DROP TABLE IF EXISTS mark_instances;
    CREATE TABLE letter_instances(
      id TEXT PRIMARY KEY, page INTEGER, line_index INTEGER, word_index INTEGER,
      word_id TEXT, character_index INTEGER, character TEXT, codepoint TEXT,
      joining_form TEXT, expected_text TEXT, x INTEGER, y INTEGER,
      width INTEGER, height INTEGER, ink_width INTEGER, confidence REAL,
      estimation_method TEXT, word_image_path TEXT, body_image_path TEXT,
      crop_path TEXT, overlay_path TEXT, review_status TEXT DEFAULT 'unreviewed',
      error_type TEXT DEFAULT '', note TEXT DEFAULT ''
    );
    CREATE INDEX letter_query ON letter_instances(character, joining_form, width);
    CREATE INDEX letter_confidence ON letter_instances(confidence);
    CREATE TABLE mark_instances(
      id TEXT PRIMARY KEY, page INTEGER, line_index INTEGER, word_index INTEGER,
      word_id TEXT, component_index INTEGER, expected_text TEXT,
      visual_class TEXT, suggested_class TEXT, confidence REAL,
      owning_character_index INTEGER, x INTEGER, y INTEGER, width INTEGER,
      height INTEGER, area INTEGER, mask_path TEXT, context_path TEXT,
      word_image_path TEXT, review_status TEXT, error_type TEXT DEFAULT '',
      note TEXT DEFAULT ''
    );
    CREATE INDEX mark_query ON mark_instances(visual_class, width);
    CREATE INDEX mark_suggestion ON mark_instances(suggested_class, confidence);
    """)


def load_annotations(path):
    if not path or not path.exists():
        return {}
    return {record["id"]: record for line in path.read_text().splitlines()
            if line.strip() for record in [json.loads(line)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--annotations", type=pathlib.Path)
    args = parser.parse_args()
    project = pathlib.Path(__file__).resolve().parents[1]
    assets = args.out.parent / "instance-assets"
    assets.mkdir(parents=True, exist_ok=True)
    annotations = load_annotations(args.annotations)
    connection = sqlite3.connect(args.out)
    schema(connection)
    letter_count = mark_count = 0
    for words_path in sorted(args.pages.glob("page*/words.json")):
        page_data = json.loads(words_path.read_text())
        page = page_data["page"]
        owners = {}
        for line in page_data["lines"]:
            for word in line["words"]:
                for component in word.get("componentIndices", []):
                    owners[component] = word
                text = word.get("expectedText", "")
                bases = [character for character in text if character in BASES]
                if not bases or not word.get("pngPath"):
                    continue
                word_path = resolve_path(word["pngPath"], project)
                body_path = words_path.parent / "subwordimgs" / f'{word["wordId"]}_body.png'
                body = cv2.imread(str(body_path), cv2.IMREAD_GRAYSCALE)
                word_image = cv2.imread(str(word_path), cv2.IMREAD_GRAYSCALE)
                if body is None or word_image is None:
                    continue
                intervals = estimate_intervals(body, len(bases))
                overlay = cv2.cvtColor(word_image, cv2.COLOR_GRAY2BGR)
                for character_index, (character, interval) in enumerate(zip(bases, intervals)):
                    x0, x1, confidence = interval
                    form = joining_form(bases, character_index)
                    instance_id = f"p{page:03d}_{word['wordId']}_ch{character_index:02d}"
                    crop_path = assets / f"{instance_id}.png"
                    overlay_path = assets / f"{instance_id}_context.png"
                    pad = 12
                    crop = word_image[:, max(0, x0 - pad):min(word_image.shape[1], x1 + pad)]
                    cv2.imwrite(str(crop_path), crop)
                    local_overlay = overlay.copy()
                    cv2.line(local_overlay, (x0, 0), (x0, overlay.shape[0] - 1), (0, 255, 0), 4)
                    cv2.line(local_overlay, (x1, 0), (x1, overlay.shape[0] - 1), (0, 255, 0), 4)
                    cv2.imwrite(str(overlay_path), local_overlay)
                    ink_columns = np.flatnonzero((body[:, x0:x1] > 0).any(axis=0))
                    ink_width = int(ink_columns[-1] - ink_columns[0] + 1) if ink_columns.size else 0
                    connection.execute("""INSERT INTO letter_instances VALUES
                      (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                        instance_id, page, word["lineIndex"], word["wordIndex"], word["wordId"],
                        character_index, character, f"U+{ord(character):04X}", form, text,
                        word["x"] + x0, word["y"], x1 - x0, word["h"], ink_width,
                        confidence, "rtl_equal_ink_mass_with_projection_valley",
                        str(word_path), str(body_path.resolve()), str(crop_path.resolve()),
                        str(overlay_path.resolve()), "unreviewed", "", ""))
                    letter_count += 1
        for component in page_data.get("components", []):
            owner = owners.get(component["componentIndex"])
            if owner is None or not owner.get("expectedText") or not component.get("maskPath"):
                continue
            record_id = f"p{page:03d}_c{component['componentIndex']:05d}"
            annotation = annotations.get(record_id, {})
            probabilities = dict(component.get("markProbabilities") or [])
            ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
            suggestion, confidence = ordered[0] if ordered else (None, 0.0)
            suggestion = annotation.get("suggestedVisualClass") or suggestion
            confidence = annotation.get(
                "autoAssignmentConfidence",
                annotation.get("classifierConfidence", confidence),
            )
            visual_class = annotation.get("visualClass")
            status = "reviewed" if annotation.get("reviewed") else "unreviewed"
            connection.execute("""INSERT INTO mark_instances VALUES
              (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                record_id, page, owner["lineIndex"], owner["wordIndex"], owner["wordId"],
                component["componentIndex"], owner["expectedText"], visual_class, suggestion,
                confidence, annotation.get("owningCharacterIndex"), component["x"], component["y"],
                component["w"], component["h"], component["area"],
                str(resolve_path(component["maskPath"], project)),
                str(resolve_path(component["contextPath"], project)),
                str(resolve_path(owner.get("pngPath", ""), project)), status, "",
                annotation.get("note", "")))
            mark_count += 1
    connection.commit()
    connection.close()
    print(f"Wrote {letter_count} letter and {mark_count} mark instances to {args.out}")


if __name__ == "__main__":
    main()
