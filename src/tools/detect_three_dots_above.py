#!/usr/bin/env python3
"""Detect one-component renderings of the three dots above ث and ش.

The text continues to expect the nominal two connected components. This tool
only emits an override when a scanned component closely matches a reviewed
one-component template. The resulting JSONL is consumed before word DP.
"""

import argparse
import json
import math
import pathlib
import shutil

import cv2
import numpy as np


def load_gray(path: str, root: pathlib.Path):
    candidate = pathlib.Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    return cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)


def binary_iou(left, right):
    left = left > 127
    right = right > 127
    union = np.count_nonzero(left | right)
    return float(np.count_nonzero(left & right)) / union if union else 0.0


def has_expected_three_dots(word):
    return any(
        mark["className"] == "dot_above_three" and mark["count"] > 0
        for mark in word.get("expectedMarks", [])
    )


def horizontal_distance(component, word):
    left = component["x"]
    right = left + component["w"]
    word_left = word["x"]
    word_right = word_left + word["w"]
    if left > word_right:
        return left - word_right
    if word_left > right:
        return word_left - right
    return 0


def make_contact_sheet(records, root, output):
    tile_width, tile_height, columns = 400, 190, 4
    rows = max(1, math.ceil(len(records) / columns))
    sheet = np.full((rows * tile_height, columns * tile_width, 3), 245, np.uint8)
    for index, record in enumerate(records):
        row, column = divmod(index, columns)
        x0, y0 = column * tile_width, row * tile_height
        context = load_gray(record["contextPath"], root)
        if context is not None:
            context = cv2.cvtColor(context, cv2.COLOR_GRAY2BGR)
            scale = min(380 / context.shape[1], 135 / context.shape[0])
            context = cv2.resize(context, None, fx=scale, fy=scale)
            sheet[y0:y0 + context.shape[0], x0:x0 + context.shape[1]] = context
        label = (
            f'{record["source"]} p{record["page"]:03d} '
            f'c{record["componentIndex"]} score={record["score"]:.3f} '
            f'-> {record["targetWordId"]}'
        )
        cv2.putText(sheet, label, (x0 + 4, y0 + 163),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1,
                    cv2.LINE_AA)
    cv2.imwrite(str(output), sheet)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=pathlib.Path, action="append", required=True)
    parser.add_argument("--templates", type=pathlib.Path, required=True,
                        help="Reviewed masks for one-component three dots")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--source", help="Scan-set name stored in the manifest")
    parser.add_argument("--minimum-iou", type=float, default=0.62)
    args = parser.parse_args()

    root = pathlib.Path.cwd()
    templates = [
        cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        for path in sorted(args.templates.glob("*.png"))
    ]
    templates = [image for image in templates if image is not None]
    if not templates:
        raise SystemExit(f"No PNG templates found in {args.templates}")

    records = []
    for pages_root in args.pages:
        source = args.source or pages_root.name
        for words_path in sorted(pages_root.glob("page*/words.json")):
            page = json.loads(words_path.read_text())
            components = page["components"]
            for line in page["lines"]:
                baseline = line["baseLine"]
                pitch = max(1, line["h"])
                line_components = [
                    component for component in components
                    if component.get("lineIndex") == line["lineIndex"]
                    and component["y"] + component["h"] <= baseline
                    and component.get("bodyProbability", 1.0) < 0.50
                ]
                for word in line["words"]:
                    if not has_expected_three_dots(word):
                        continue
                    candidates = []
                    for component in line_components:
                        if horizontal_distance(component, word) > 0.12 * pitch:
                            continue
                        mask = load_gray(component.get("maskPath", ""), root)
                        if mask is None:
                            continue
                        score = max(binary_iou(mask, template) for template in templates)
                        candidates.append((score, component))
                    if not candidates:
                        continue
                    score, component = max(candidates, key=lambda item: item[0])
                    if score < args.minimum_iou:
                        continue
                    records.append({
                        "source": source,
                        "page": page["page"],
                        "componentIndex": component["componentIndex"],
                        "componentParts": [component["componentIndex"]],
                        "connectedComponentCount": 1,
                        "score": float(score),
                        "targetLineIndex": line["lineIndex"],
                        "targetWordId": word["wordId"],
                        "targetExpectedText": word["expectedText"],
                        "maskPath": component["maskPath"],
                        "contextPath": component["contextPath"],
                        "reviewed": False,
                        "autoAcceptedCount": bool(
                            score >= max(0.72, args.minimum_iou)
                        ),
                    })

    records.sort(key=lambda record: (
        record["source"], record["page"], record["targetLineIndex"],
        record["targetWordId"]
    ))
    args.out.mkdir(parents=True, exist_ok=True)
    proposals = args.out / "proposals" / "connected_1"
    if proposals.exists():
        shutil.rmtree(proposals)
    proposals.mkdir(parents=True)
    for record in records:
        mask = load_gray(record["maskPath"], root)
        if mask is not None:
            cv2.imwrite(str(proposals / (
                f'{record["source"]}-p{record["page"]:03d}-'
                f'c{record["componentIndex"]:05d}.png'
            )), mask)

    manifest = args.out / "three-dots-above-detections.jsonl"
    with manifest.open("w") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    make_contact_sheet(records, root, args.out / "three-dots-above-contact-sheet.png")
    make_contact_sheet(
        [record for record in records if record["autoAcceptedCount"]],
        root,
        args.out / "three-dots-above-auto-accepted-contact-sheet.png",
    )
    print(json.dumps({
        "detections": len(records),
        "autoAccepted": sum(record["autoAcceptedCount"] for record in records),
        "manifest": str(manifest),
    }, indent=2))


if __name__ == "__main__":
    main()
