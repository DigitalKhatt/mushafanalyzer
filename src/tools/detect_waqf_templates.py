#!/usr/bin/env python3
"""Detect high-confidence waqf components using reviewed scan templates.

The detector is deliberately expectation-guided: a shape match is reported only
when the current or an adjacent line contains a word that expects that waqf
class. Results are review proposals, never automatic training labels.
"""

import argparse
import json
import math
import pathlib
import shutil

import cv2
import numpy as np


def load_image(path, project_root):
    path = pathlib.Path(path)
    if not path.is_absolute():
        path = project_root / path
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def binary_iou(left, right):
    left = left > 127
    right = right > 127
    union = np.count_nonzero(left | right)
    return float(np.count_nonzero(left & right)) / union if union else 0.0


def horizontal_distance(component, word):
    component_left = component["x"]
    component_right = component_left + component["w"]
    word_left = word["x"]
    word_right = word_left + word["w"]
    if component_left > word_right:
        return component_left - word_right
    if word_left > component_right:
        return word_left - component_right
    return 0


def expected_waqf_words(page, class_name, component):
    candidates = []
    source_line = component.get("lineIndex", -1)
    for line in page["lines"]:
        if abs(line["lineIndex"] - source_line) > 1:
            continue
        if component["y"] + component["h"] > line["baseLine"]:
            continue
        for word in line["words"]:
            expected = {
                mark["className"] for mark in word.get("expectedMarks", [])
            }
            if class_name not in expected:
                continue
            distance = horizontal_distance(component, word)
            pitch = max(1, line["h"])
            if distance > 0.30 * pitch:
                continue
            deficit = (
                word["expectedComponentCount"]
                - word["observedComponentCount"]
            )
            candidates.append((distance, -deficit, line, word))
    return sorted(candidates, key=lambda item: (item[0], item[1]))


def find_jeem_dot(body, components):
    center_x = body["x"] + body["w"] / 2
    center_y = body["y"] + body["h"] / 2
    candidates = []
    for component in components:
        if component["componentIndex"] == body["componentIndex"]:
            continue
        if component.get("dotProbability", 0.0) < 0.55:
            continue
        dot_x = component["x"] + component["w"] / 2
        dot_y = component["y"] + component["h"] / 2
        if abs(dot_x - center_x) > 0.65 * body["w"]:
            continue
        if abs(dot_y - center_y) > 0.65 * body["h"]:
            continue
        if component["area"] > 0.25 * body["area"]:
            continue
        distance = math.hypot(
            (dot_x - center_x) / max(1, body["w"]),
            (dot_y - center_y) / max(1, body["h"]),
        )
        candidates.append((distance, component))
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1] if candidates else None


def find_qaf_dots(body, components):
    center_x = body["x"] + body["w"] / 2
    center_y = body["y"] + body["h"] / 2
    candidates = []
    for component in components:
        if component["componentIndex"] == body["componentIndex"]:
            continue
        if component.get("bodyProbability", 1.0) >= 0.50:
            continue
        if max(component.get("dotProbability", 0.0), component.get("ordinaryMarkProbability", 0.0)) < 0.45:
            continue
        dot_x = component["x"] + component["w"] / 2
        dot_y = component["y"] + component["h"] / 2
        if abs(dot_x - center_x) > 0.70 * body["w"]:
            continue
        if dot_y > center_y + 0.10 * body["h"]:
            continue
        if component["area"] > 0.30 * body["area"]:
            continue
        distance = math.hypot(
            (dot_x - center_x) / max(1, body["w"]),
            (dot_y - body["y"]) / max(1, body["h"]),
        )
        candidates.append((distance, component))
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1] if candidates else None


def make_contact_sheet(records, project_root, output):
    tile_width, tile_height = 420, 210
    columns = 4
    rows = max(1, math.ceil(len(records) / columns))
    sheet = np.full((rows * tile_height, columns * tile_width, 3), 245, np.uint8)
    for index, record in enumerate(records):
        row, column = divmod(index, columns)
        x0, y0 = column * tile_width, row * tile_height
        context = load_image(record["contextPath"], project_root)
        if context is not None:
            context = cv2.cvtColor(context, cv2.COLOR_GRAY2BGR)
            scale = min(400 / context.shape[1], 150 / context.shape[0])
            context = cv2.resize(context, None, fx=scale, fy=scale)
            sheet[y0:y0 + context.shape[0], x0:x0 + context.shape[1]] = context
        label = (
            f'{record["source"]} p{record["page"]:03d} '
            f'c{record["componentIndex"]} {record["className"]} '
            f'{record["score"]:.3f} -> {record["targetWordId"]}'
        )
        cv2.putText(sheet, label, (x0 + 4, y0 + 180), cv2.FONT_HERSHEY_SIMPLEX,
                    0.43, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.imwrite(str(output), sheet)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=pathlib.Path, action="append", required=True)
    parser.add_argument("--templates", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--minimum-iou", type=float, default=0.48)
    parser.add_argument("--minimum-margin", type=float, default=0.08)
    args = parser.parse_args()

    project_root = pathlib.Path.cwd()
    templates = {}
    for class_directory in sorted(args.templates.iterdir()):
        if not class_directory.is_dir():
            continue
        images = [
            cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            for path in sorted(class_directory.glob("*.png"))
        ]
        templates[class_directory.name] = [image for image in images if image is not None]

    records = []
    for pages_root in args.pages:
        source = pages_root.name
        for words_path in sorted(pages_root.glob("page*/words.json")):
            page = json.loads(words_path.read_text())
            components = page["components"]
            owners = {}
            for line in page["lines"]:
                for word in line["words"]:
                    for component_index in word.get("componentIndices", []):
                        owners[component_index] = word
            for component in components:
                if component.get("bodyProbability", 1.0) >= 0.50:
                    continue
                mask = load_image(component.get("maskPath", ""), project_root)
                if mask is None:
                    continue
                scored = []
                for class_name, class_templates in templates.items():
                    if not expected_waqf_words(page, class_name, component):
                        continue
                    score = max((binary_iou(mask, template) for template in class_templates), default=0.0)
                    scored.append((score, class_name))
                scored.sort(reverse=True)
                if not scored:
                    continue
                score, class_name = scored[0]
                runner_up = scored[1][0] if len(scored) > 1 else 0.0
                if score < args.minimum_iou or score - runner_up < args.minimum_margin:
                    continue
                targets = expected_waqf_words(page, class_name, component)
                if not targets:
                    continue
                _, _, target_line, target_word = targets[0]
                parts = [component["componentIndex"]]
                part_roles = ["whole"]
                if class_name == "waqf_jeem":
                    dot = find_jeem_dot(component, components)
                    if dot is None:
                        continue
                    parts.append(dot["componentIndex"])
                    part_roles = ["body", "dot"]
                elif class_name == "waqf_qaf":
                    dots = find_qaf_dots(component, components)
                    if dots is None:
                        continue
                    parts.append(dots["componentIndex"])
                    part_roles = ["body", "dots"]
                if any(
                    components[part]["y"] + components[part]["h"]
                    > target_line["baseLine"]
                    for part in parts
                ):
                    continue
                current_word = owners.get(component["componentIndex"])
                current_difference = (
                    current_word["observedComponentCount"]
                    - current_word["expectedComponentCount"]
                    if current_word else 0
                )
                record = {
                    "source": source,
                    "page": page["page"],
                    "componentIndex": component["componentIndex"],
                    "componentParts": parts,
                    "componentPartRoles": part_roles,
                    "partMaskPaths": [components[part]["maskPath"] for part in parts],
                    "partContextPaths": [components[part]["contextPath"] for part in parts],
                    "className": class_name,
                    "score": score,
                    "margin": score - runner_up,
                    "currentLineIndex": component.get("lineIndex", -1),
                    "currentWordId": current_word["wordId"] if current_word else "",
                    "currentComponentExcess": current_difference,
                    "targetLineIndex": target_line["lineIndex"],
                    "targetWordId": target_word["wordId"],
                    "targetExpectedText": target_word["expectedText"],
                    "targetComponentDeficit": target_word["expectedComponentCount"] - target_word["observedComponentCount"],
                    "maskPath": component["maskPath"],
                    "contextPath": component["contextPath"],
                    "reviewed": False,
                }
                record["autoAcceptedCorrection"] = bool(
                    record["currentLineIndex"] != record["targetLineIndex"]
                    and record["targetComponentDeficit"] == len(parts)
                    and current_difference >= len(parts)
                    and score >= 0.65
                    and score - runner_up >= args.minimum_margin
                )
                records.append(record)

    # A component in either adjacent line can resemble the expected mark. Keep
    # only the strongest proposal for each semantic waqf instance; otherwise a
    # single expected sign can produce two contradictory ownership proposals.
    unique = {}
    for record in records:
        key = (record["source"], record["page"], record["targetWordId"], record["className"])
        ownership_bonus = 0.05 if record["currentLineIndex"] == record["targetLineIndex"] else 0.0
        rank = record["score"] + ownership_bonus
        if key not in unique or rank > unique[key][0]:
            unique[key] = (rank, record)
    records = [item[1] for item in unique.values()]
    records.sort(key=lambda record: (record["source"], record["page"], record["targetLineIndex"], record["targetWordId"], record["className"]))

    args.out.mkdir(parents=True, exist_ok=True)
    samples_root = args.out / "samples"
    if samples_root.exists():
        shutil.rmtree(samples_root)
    for record in records:
        sample_paths = []
        stem = f'{record["source"]}-p{record["page"]:03d}-c{record["componentIndex"]:05d}'
        for part_index, (role, mask_path, context_path) in enumerate(zip(record["componentPartRoles"], record["partMaskPaths"], record["partContextPaths"]), 1):
            part_directory = samples_root / record["className"] / role
            masks_directory = part_directory / "masks"
            contexts_directory = part_directory / "contexts"
            masks_directory.mkdir(parents=True, exist_ok=True)
            contexts_directory.mkdir(parents=True, exist_ok=True)
            part_stem = f"{stem}-part{part_index}"
            sample_mask = masks_directory / f"{part_stem}.png"
            sample_context = contexts_directory / f"{part_stem}.png"
            mask = load_image(mask_path, project_root)
            context = load_image(context_path, project_root)
            if mask is not None:
                cv2.imwrite(str(sample_mask), mask)
            if context is not None:
                cv2.imwrite(str(sample_context), context)
            sample_paths.append({
                "role": role,
                "maskPath": str(sample_mask),
                "contextPath": str(sample_context),
            })
        record["samplePaths"] = sample_paths

    manifest = args.out / "waqf-detections.jsonl"
    with manifest.open("w") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    make_contact_sheet(records, project_root, args.out / "waqf-contact-sheet.png")
    print(json.dumps({"detections": len(records), "manifest": str(manifest)}, indent=2))


if __name__ == "__main__":
    main()
