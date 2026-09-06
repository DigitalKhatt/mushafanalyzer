#!/usr/bin/env python3
"""Collect and review small-yeh component prototypes from analyzer output.

This tool intentionally uses only the Python standard library and ImageMagick.
It bootstraps candidates from one reviewed seed, writes a ranked contact sheet,
and exports the measurements used by a future C++ classifier.
"""

import argparse
import json
import math
import pathlib
import re
import subprocess


SMALL_YEH = "\u06e6"
MOMENT_RE = re.compile(r"^\s+I([1-7]):\s+([-+0-9.eE]+)", re.MULTILINE)


def hu_moments(image, box):
    x, y, width, height = box
    command = [
        "magick",
        str(image),
        "-crop",
        f"{width}x{height}+{x}+{y}",
        "+repage",
        "-threshold",
        "50%",
        "-define",
        "identify:moments=true",
        "-verbose",
        "info:",
    ]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    values = {int(index): float(value) for index, value in MOMENT_RE.findall(result.stdout)}
    if len(values) != 7:
        raise RuntimeError(f"Could not read Hu moments for {image} {box}")
    return [values[index] for index in range(1, 8)]


def signed_log_moments(values):
    return [
        -math.copysign(math.log10(max(abs(value), 1.0e-30)), value)
        for value in values
    ]


def distance(first, second):
    # High-order Hu moments are useful but considerably less stable in scans.
    weights = [1.0, 0.5, 0.35, 0.25, 0.10, 0.10, 0.10]
    return math.sqrt(
        sum(weight * (left - right) ** 2 for weight, left, right in zip(weights, first, second))
    )


def geometry_features(data, component):
    line_index = component["lineIndex"]
    line = data["lines"][line_index]
    pitches = []
    if line_index > 0:
        pitches.append(line["baseLine"] - data["lines"][line_index - 1]["baseLine"])
    if line_index + 1 < len(data["lines"]):
        pitches.append(data["lines"][line_index + 1]["baseLine"] - line["baseLine"])
    pitch = max(1.0, sum(pitches) / len(pitches))
    return [
        component["w"] / pitch,
        component["h"] / pitch,
        component["area"] / (pitch * pitch),
        (component["y"] + 0.5 * component["h"] - line["baseLine"]) / pitch,
        component["area"] / max(1.0, component["w"] * component["h"]),
    ]


def sample_distance(first, second, geometry_weight):
    hu = distance(first["huLog"], second["huLog"])
    scales = [0.06, 0.06, 0.006, 0.12, 0.20]
    geometry = math.sqrt(
        sum(
            ((left - right) / scale) ** 2
            for left, right, scale in zip(
                first["geometry"], second["geometry"], scales
            )
        )
    )
    return hu + geometry_weight * geometry


def crop_candidate(page_image, component, destination):
    box = f"{component['w']}x{component['h']}+{component['x']}+{component['y']}"
    subprocess.run(
        [
            "magick",
            str(page_image),
            "-crop",
            box,
            "+repage",
            "-trim",
            "+repage",
            "-resize",
            "260x180",
            "-gravity",
            "center",
            "-background",
            "black",
            "-extent",
            "280x180",
            str(destination),
        ],
        check=True,
    )


def component_source(page_dir, data, component):
    index = component["componentIndex"]
    for line in data["lines"]:
        for word in line["words"]:
            if index not in word["componentIndices"]:
                continue
            image = page_dir / "wordimgs" / f"{word['wordId']}.png"
            box = [
                component["x"] - word["x"],
                component["y"] - word["y"],
                component["w"],
                component["h"],
            ]
            return image, box
    raise RuntimeError(
        f"Component {index} on page {data['page']} has no word owner"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--seed-page", type=int, default=560)
    parser.add_argument("--seed-component", type=int, default=309)
    parser.add_argument("--labels", type=pathlib.Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    records = {}
    for words_path in sorted(args.pages.glob("page*/words.json")):
        data = json.loads(words_path.read_text())
        if not data.get("components"):
            continue
        records[data["page"]] = (words_path.parent, data)
    if args.seed_page not in records:
        raise RuntimeError("Seed page is absent from the component dataset")

    seed_dir, seed_data = records[args.seed_page]
    seed = next(
        component
        for component in seed_data["components"]
        if component["componentIndex"] == args.seed_component
    )
    seed_image, seed_box = component_source(seed_dir, seed_data, seed)
    seed_features = signed_log_moments(hu_moments(seed_image, seed_box))

    selections = []
    for page, (page_dir, data) in sorted(records.items()):
        components = data["components"]
        for line in data["lines"]:
            for word in line["words"]:
                if SMALL_YEH not in word["expectedText"]:
                    continue
                margin = 450
                left = word["x"] - margin
                right = word["x"] + word["w"] + margin
                candidates = []
                for component in components:
                    if component["lineIndex"] != line["lineIndex"]:
                        continue
                    component_right = component["x"] + component["w"]
                    if component_right < left or component["x"] > right:
                        continue
                    if not (8000 <= component["area"] <= 70000):
                        continue
                    if not (80 <= component["w"] <= 600 and 70 <= component["h"] <= 500):
                        continue
                    image, box = component_source(page_dir, data, component)
                    features = signed_log_moments(hu_moments(image, box))
                    candidates.append((distance(seed_features, features), component, features))
                if not candidates:
                    continue
                candidates.sort(key=lambda item: item[0])
                for rank, (score, component, features) in enumerate(candidates[:3], start=1):
                    selections.append(
                        {
                            "page": page,
                            "wordId": word["wordId"],
                            "text": word["expectedText"],
                            "rank": rank,
                            "distance": score,
                            "componentIndex": component["componentIndex"],
                            "box": [component[key] for key in ("x", "y", "w", "h")],
                            "huLog": features,
                            "geometry": geometry_features(data, component),
                        }
                    )

    (args.out / "small-yeh-candidates.json").write_text(
        json.dumps(
            {
                "seed": {
                    "page": args.seed_page,
                    "componentIndex": args.seed_component,
                    "huLog": seed_features,
                },
                "candidates": selections,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if args.labels:
        labels = json.loads(args.labels.read_text())
        positive_indices = set(labels["positiveCandidateIndices"])
        if not positive_indices.issubset(range(len(selections))):
            raise RuntimeError("A reviewed candidate index is out of range")
        samples = [
            {
                **selection,
                "label": 1 if index in positive_indices else 0,
            }
            for index, selection in enumerate(selections)
        ]

        best_validation = None
        for geometry_weight in [0.0, 0.25, 0.5, 1.0, 2.0]:
            correct = 0
            evaluated = 0
            predictions = []
            for sample in samples:
                training = [
                    item for item in samples if item["page"] != sample["page"]
                ]
                positives = [item for item in training if item["label"] == 1]
                negatives = [item for item in training if item["label"] == 0]
                if not positives or not negatives:
                    continue
                positive_distance = min(
                    sample_distance(sample, item, geometry_weight)
                    for item in positives
                )
                negative_distance = min(
                    sample_distance(sample, item, geometry_weight)
                    for item in negatives
                )
                prediction = int(positive_distance < negative_distance)
                correct += prediction == sample["label"]
                evaluated += 1
                predictions.append(
                    {
                        "page": sample["page"],
                        "wordId": sample["wordId"],
                        "componentIndex": sample["componentIndex"],
                        "label": sample["label"],
                        "prediction": prediction,
                        "positiveDistance": positive_distance,
                        "negativeDistance": negative_distance,
                    }
                )
            candidate_validation = (correct, geometry_weight, evaluated, predictions)
            if best_validation is None or candidate_validation[:2] > best_validation[:2]:
                best_validation = candidate_validation
        correct, geometry_weight, evaluated, predictions = best_validation

        model = {
            "modelType": "hu-moment-nearest-prototype",
            "smallLetter": "U+06E6",
            "validation": {
                "method": "leave-one-page-out",
                "geometryWeight": geometry_weight,
                "correct": correct,
                "evaluated": evaluated,
                "accuracy": correct / evaluated if evaluated else 0.0,
            },
            "positivePrototypes": [
                item for item in samples if item["label"] == 1
            ],
            "negativePrototypes": [
                item for item in samples if item["label"] == 0
            ],
            "predictions": predictions,
        }
        group_correct = 0
        groups = {}
        for prediction in predictions:
            groups.setdefault(
                (prediction["page"], prediction["wordId"]), []
            ).append(prediction)
        for group in groups.values():
            selected = max(
                group,
                key=lambda item: item["negativeDistance"]
                - item["positiveDistance"],
            )
            group_correct += selected["label"] == 1
        model["validation"]["wordConstrainedCorrect"] = group_correct
        model["validation"]["wordConstrainedEvaluated"] = len(groups)
        model["validation"]["wordConstrainedAccuracy"] = (
            group_correct / len(groups) if groups else 0.0
        )
        holdout_pages = set(labels.get("holdoutPages", []))
        if holdout_pages:
            training = [item for item in samples if item["page"] not in holdout_pages]
            holdout = [item for item in samples if item["page"] in holdout_pages]
            positives = [item for item in training if item["label"] == 1]
            negatives = [item for item in training if item["label"] == 0]
            holdout_predictions = []
            for sample in holdout:
                positive_distance = min(
                    sample_distance(sample, item, geometry_weight)
                    for item in positives
                )
                negative_distance = min(
                    sample_distance(sample, item, geometry_weight)
                    for item in negatives
                )
                holdout_predictions.append(
                    {
                        **sample,
                        "prediction": int(positive_distance < negative_distance),
                        "positiveDistance": positive_distance,
                        "negativeDistance": negative_distance,
                    }
                )
            holdout_groups = {}
            for prediction in holdout_predictions:
                holdout_groups.setdefault(
                    (prediction["page"], prediction["wordId"]), []
                ).append(prediction)
            holdout_group_correct = 0
            for group in holdout_groups.values():
                selected = max(
                    group,
                    key=lambda item: item["negativeDistance"]
                    - item["positiveDistance"],
                )
                holdout_group_correct += selected["label"] == 1
            model["holdoutValidation"] = {
                "pages": sorted(holdout_pages),
                "correct": sum(
                    item["label"] == item["prediction"]
                    for item in holdout_predictions
                ),
                "evaluated": len(holdout_predictions),
                "wordConstrainedCorrect": holdout_group_correct,
                "wordConstrainedEvaluated": len(holdout_groups),
            }
        (args.out / "small-yeh-model.json").write_text(
            json.dumps(model, ensure_ascii=False, indent=2)
        )

    tiles = []
    for index, selection in enumerate(selections):
        page_dir, data = records[selection["page"]]
        component = next(
            item
            for item in data["components"]
            if item["componentIndex"] == selection["componentIndex"]
        )
        tile = args.out / f"tile-{index:03d}.png"
        image, box = component_source(page_dir, data, component)
        local_component = dict(component)
        (
            local_component["x"],
            local_component["y"],
            local_component["w"],
            local_component["h"],
        ) = box
        crop_candidate(
            image,
            local_component,
            tile,
        )
        tiles.append(tile)
    if tiles:
        rows = []
        for row_index in range(0, len(tiles), 3):
            row = args.out / f"row-{row_index // 3:03d}.png"
            subprocess.run(
                ["magick", *map(str, tiles[row_index : row_index + 3]), "+append", str(row)],
                check=True,
            )
            rows.append(row)
        subprocess.run(
            [
                "magick",
                *map(str, rows),
                "-append",
                str(args.out / "small-yeh-contact-sheet.png"),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
