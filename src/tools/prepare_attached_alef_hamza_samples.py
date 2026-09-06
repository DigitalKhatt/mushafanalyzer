#!/usr/bin/env python3
"""Export review samples proposed as connected alef + hamza-above forms."""

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


def normalize(mask):
    output = np.zeros((96, 96), np.uint8)
    height, width = mask.shape
    scale = min(88.0 / max(1, width), 88.0 / max(1, height))
    resized = cv2.resize(mask, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_NEAREST)
    y = (96 - resized.shape[0]) // 2
    x = (96 - resized.shape[1]) // 2
    output[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return output


def contact_sheet(records, root, output):
    tile_width, tile_height, columns = 390, 205, 4
    rows = max(1, math.ceil(len(records) / columns))
    sheet = np.full((rows * tile_height, columns * tile_width, 3),
                    245, np.uint8)
    for index, record in enumerate(records):
        row, column = divmod(index, columns)
        x0, y0 = column * tile_width, row * tile_height
        context = load_gray(record["contextPath"], root)
        if context is not None:
            context = cv2.cvtColor(context, cv2.COLOR_GRAY2BGR)
            scale = min(370 / context.shape[1], 150 / context.shape[0])
            context = cv2.resize(context, None, fx=scale, fy=scale)
            sheet[y0:y0 + context.shape[0],
                  x0:x0 + context.shape[1]] = context
        label = (f'p{record["page"]:03d} c{record["componentIndex"]:05d} '
                 f'{record["probability"]:.3f} {record["wordId"]}')
        cv2.putText(sheet, label, (x0 + 4, y0 + 178),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1,
                    cv2.LINE_AA)
    cv2.imwrite(str(output), sheet)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=pathlib.Path, action="append",
                        required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--minimum-probability", type=float, default=0.82)
    args = parser.parse_args()

    root = pathlib.Path.cwd()
    records = []
    for pages_root in args.pages:
        for words_path in sorted(pages_root.glob("page*/words.json")):
            page = json.loads(words_path.read_text())
            owners = {}
            for line in page["lines"]:
                for word in line["words"]:
                    for component in word.get("componentIndices", []):
                        owners[component] = word
            for component in page["components"]:
                probability = component.get(
                    "attachedAlefHamzaAboveProbability", 0.0)
                owner = owners.get(component["componentIndex"])
                if probability < args.minimum_probability or owner is None:
                    continue
                if owner.get("recognizedAttachedAlefHamzaAboveCount", 0) == 0:
                    continue
                if not component.get("maskPath") or not component.get(
                        "contextPath"):
                    continue
                records.append({
                    "page": page["page"],
                    "componentIndex": component["componentIndex"],
                    "probability": probability,
                    "wordId": owner["wordId"],
                    "expectedText": owner["expectedText"],
                    "maskPath": component["maskPath"],
                    "contextPath": component["contextPath"],
                    "reviewed": False,
                })

    records.sort(key=lambda item: (item["page"], item["wordId"]))
    proposals = args.out / "proposals"
    contexts = args.out / "contexts"
    if proposals.exists():
        shutil.rmtree(proposals)
    if contexts.exists():
        shutil.rmtree(contexts)
    proposals.mkdir(parents=True)
    contexts.mkdir(parents=True)
    for record in records:
        stem = (f'p{record["page"]:03d}-'
                f'c{record["componentIndex"]:05d}-{record["wordId"]}')
        mask = load_gray(record["maskPath"], root)
        context = load_gray(record["contextPath"], root)
        if mask is not None:
            cv2.imwrite(str(proposals / f"{stem}.png"), normalize(mask))
        if context is not None:
            cv2.imwrite(str(contexts / f"{stem}.png"), context)

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "attached-alef-hamza-proposals.jsonl").open("w") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    contact_sheet(records, root, args.out / "contact-sheet.png")
    print(json.dumps({"proposals": len(records), "out": str(args.out)},
                     indent=2))


if __name__ == "__main__":
    main()
