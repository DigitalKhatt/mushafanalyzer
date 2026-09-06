#!/usr/bin/env python3
"""Render components whose DP word owner changed between two analyses."""

import argparse
import json
import pathlib

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PANEL_WIDTH = 1200
PANEL_HEIGHT = 420
IMAGE_TOP = 92
IMAGE_HEIGHT = PANEL_HEIGHT - IMAGE_TOP


def resolve_page_image(directory, page):
    for stem in (str(page), f"{page:03d}"):
        for suffix in (".jpg", ".jpeg", ".png"):
            path = directory / f"{stem}{suffix}"
            if path.exists():
                return path
    raise RuntimeError(f"No scan for page {page}")


def owners(data):
    result = {}
    for line in data["lines"]:
        for word in line["words"]:
            for component in word["componentIndices"]:
                result[component] = word
    return result


def rect(word):
    return np.array([word["x"], word["y"], word["w"], word["h"]])


def union_rect(left, right, component, image_width, image_height):
    boxes = [rect(left), rect(right), np.array([
        component["x"], component["y"], component["w"], component["h"]])]
    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[0] + box[2] for box in boxes)
    y1 = max(box[1] + box[3] for box in boxes)
    pad_x = max(180, int(0.08 * (x1 - x0)))
    pad_y = max(150, int(0.20 * (y1 - y0)))
    return (max(0, x0 - pad_x), max(0, y0 - pad_y),
            min(image_width, x1 + pad_x), min(image_height, y1 + pad_y))


def draw_view(source, crop, word, component, word_color):
    x0, y0, x1, y1 = crop
    view = cv2.cvtColor(source[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
    overlay = view.copy()
    cx = component["x"] - x0
    cy = component["y"] - y0
    cw = component["w"]
    ch = component["h"]
    cv2.rectangle(overlay, (cx, cy), (cx + cw, cy + ch), (0, 220, 255), -1)
    cv2.addWeighted(overlay, 0.28, view, 0.72, 0.0, view)
    wx = word["x"] - x0
    wy = word["y"] - y0
    cv2.rectangle(view, (wx, wy), (wx + word["w"], wy + word["h"]),
                  word_color, max(5, view.shape[1] // 500))
    cv2.rectangle(view, (cx, cy), (cx + cw, cy + ch), (0, 170, 255),
                  max(5, view.shape[1] // 500))
    scale = min((PANEL_WIDTH // 2 - 12) / view.shape[1],
                IMAGE_HEIGHT / view.shape[0])
    resized = cv2.resize(view, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
    canvas = np.full((IMAGE_HEIGHT, PANEL_WIDTH // 2, 3), 245, np.uint8)
    ox = (canvas.shape[1] - resized.shape[1]) // 2
    oy = (canvas.shape[0] - resized.shape[0]) // 2
    canvas[oy:oy + resized.shape[0], ox:ox + resized.shape[1]] = resized
    return canvas


def load_font(path, size):
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()


def add_header(panel, record, font, arabic_font):
    image = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    draw.text((12, 7),
              f"#{record['index']:02d}  p{record['page']:03d} "
              f"c{record['componentIndex']:05d}", fill=(15, 15, 15),
              font=font)
    draw.text((12, 39), f"OLD {record['oldWordId']}", fill=(210, 40, 40),
              font=font)
    draw.text((PANEL_WIDTH // 2 + 12, 39),
              f"NEW {record['newWordId']}", fill=(20, 145, 55), font=font)
    try:
        draw.text((PANEL_WIDTH // 2 - 12, 38), record["oldText"],
                  fill=(30, 30, 30), font=arabic_font, anchor="ra",
                  direction="rtl", language="ar")
        draw.text((PANEL_WIDTH - 12, 38), record["newText"],
                  fill=(30, 30, 30), font=arabic_font, anchor="ra",
                  direction="rtl", language="ar")
    except (KeyError, ValueError):
        pass
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed", type=pathlib.Path, required=True)
    parser.add_argument("--variable", type=pathlib.Path, required=True)
    parser.add_argument("--images", type=pathlib.Path, required=True)
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--last", type=int, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    records = []
    panels = []
    font = load_font("/System/Library/Fonts/Supplemental/Arial.ttf", 24)
    arabic_font = load_font("/System/Library/Fonts/GeezaPro.ttc", 27)
    for page in range(args.first, args.last + 1):
        fixed = json.loads(
            (args.fixed / f"page{page:03d}" / "words.json").read_text())
        variable = json.loads(
            (args.variable / f"page{page:03d}" / "words.json").read_text())
        fixed_owners = owners(fixed)
        variable_owners = owners(variable)
        changed = sorted(component for component in
                         fixed_owners.keys() | variable_owners.keys()
                         if fixed_owners.get(component, {}).get("wordId") !=
                         variable_owners.get(component, {}).get("wordId"))
        if not changed:
            continue
        scan = cv2.imread(str(resolve_page_image(args.images, page)),
                          cv2.IMREAD_GRAYSCALE)
        if scan is None:
            raise RuntimeError(f"Could not read scan for page {page}")
        crop_x = fixed["preprocessing"]["cropX"]
        crop_y = fixed["preprocessing"]["cropY"]
        for component_index in changed:
            old_word = fixed_owners[component_index]
            new_word = variable_owners[component_index]
            component = fixed["components"][component_index].copy()
            # Analyzer coordinates are relative to the cropped page.
            old_absolute = old_word.copy()
            new_absolute = new_word.copy()
            component_absolute = component.copy()
            for item in (old_absolute, new_absolute, component_absolute):
                item["x"] += crop_x
                item["y"] += crop_y
            crop = union_rect(old_absolute, new_absolute, component_absolute,
                              scan.shape[1], scan.shape[0])
            record = {
                "index": len(records) + 1,
                "page": page,
                "componentIndex": component_index,
                "oldWordId": old_word["wordId"],
                "oldText": old_word["expectedText"],
                "oldObserved": old_word["observedComponentCount"],
                "oldExpected": old_word["expectedComponentCount"],
                "newWordId": new_word["wordId"],
                "newText": new_word["expectedText"],
                "newObserved": new_word["observedComponentCount"],
                "newAllowed": new_word.get("allowedComponentCounts", []),
                "decision": "unreviewed",
                "note": "",
            }
            panel = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), 245, np.uint8)
            panel[IMAGE_TOP:, :PANEL_WIDTH // 2] = draw_view(
                scan, crop, old_absolute, component_absolute, (40, 40, 220))
            panel[IMAGE_TOP:, PANEL_WIDTH // 2:] = draw_view(
                scan, crop, new_absolute, component_absolute, (40, 170, 40))
            cv2.line(panel, (PANEL_WIDTH // 2, 0),
                     (PANEL_WIDTH // 2, PANEL_HEIGHT), (100, 100, 100), 2)
            panels.append(add_header(panel, record, font, arabic_font))
            records.append(record)

    columns = 2
    rows = (len(panels) + columns - 1) // columns
    sheet = np.full((rows * PANEL_HEIGHT, columns * PANEL_WIDTH, 3),
                    235, np.uint8)
    for index, panel in enumerate(panels):
        row, column = divmod(index, columns)
        sheet[row * PANEL_HEIGHT:(row + 1) * PANEL_HEIGHT,
              column * PANEL_WIDTH:(column + 1) * PANEL_WIDTH] = panel
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.out), sheet):
        raise RuntimeError(f"Could not write {args.out}")
    manifest = args.out.with_suffix(".json")
    manifest.write_text(json.dumps({
        "schemaVersion": 1,
        "fixedRoot": str(args.fixed),
        "variableRoot": str(args.variable),
        "legend": {
            "red": "fixed-count owner",
            "green": "variable-count owner",
            "yellow": "moved component",
        },
        "allowedDecisions": ["unreviewed", "better", "worse", "uncertain"],
        "records": records,
    }, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {len(records)} changes to {args.out} and {manifest}")


if __name__ == "__main__":
    main()
