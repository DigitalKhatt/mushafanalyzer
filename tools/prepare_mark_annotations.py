#!/usr/bin/env python3
"""Create an editable JSONL annotation manifest from schema-4 page outputs."""

import argparse
import json
import pathlib


def prediction(component):
    probabilities = component.get("markProbabilities") or {}
    if isinstance(probabilities, list):
        probabilities = dict(probabilities)
    ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    if not ordered:
        return None, 0.0, 0.0
    confidence = float(ordered[0][1])
    runner_up = float(ordered[1][1]) if len(ordered) > 1 else 0.0
    return ordered[0][0], confidence, confidence - runner_up


def review_metadata(component, owner, confidence_threshold, margin_threshold):
    predicted_class, confidence, margin = prediction(component)
    reasons = []
    expected_classes = {
        mark["className"] for mark in owner.get("expectedMarks", [])
    } if owner else set()

    if owner is None:
        reasons.append("unassigned_component")
    else:
        difference = (
            owner.get("observedComponentCount", 0)
            - owner.get("expectedComponentCount", 0)
        )
        if difference:
            reasons.append("word_component_count_mismatch")
        if owner.get("quality", {}).get("crossLineContamination"):
            reasons.append("possible_cross_line_component")

    if not predicted_class:
        reasons.append("no_classifier_prediction")
    else:
        if confidence < confidence_threshold:
            reasons.append("low_classifier_confidence")
        if margin < margin_threshold:
            reasons.append("ambiguous_classifier_prediction")
        if expected_classes and predicted_class not in expected_classes:
            reasons.append("prediction_not_expected_by_text")

    safe_auto_accept = (
        owner is not None
        and not reasons
        and predicted_class in expected_classes
        and confidence >= confidence_threshold
        and margin >= margin_threshold
    )
    priority = 0
    priority += 100 if "word_component_count_mismatch" in reasons else 0
    priority += 80 if "possible_cross_line_component" in reasons else 0
    priority += 60 if "prediction_not_expected_by_text" in reasons else 0
    priority += 40 if "no_classifier_prediction" in reasons else 0
    priority += round(30 * (1.0 - confidence))
    priority += round(20 * (1.0 - margin))
    return {
        "suggestedVisualClass": predicted_class,
        "classifierConfidence": confidence,
        "classifierMargin": margin,
        "reviewReasons": reasons,
        "reviewPriority": priority,
        "reviewStatus": "auto_accepted" if safe_auto_accept else "needs_review",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument(
        "--include-non-text-components",
        action="store_true",
        help="Include components owned by headings, frames, or empty-text lines",
    )
    parser.add_argument("--auto-accept-confidence", type=float, default=0.995)
    parser.add_argument("--auto-accept-margin", type=float, default=0.25)
    args = parser.parse_args()

    existing = {}
    if args.out.exists():
        existing = {
            record["id"]: record
            for line in args.out.read_text().splitlines()
            if line.strip()
            for record in [json.loads(line)]
        }

    records = []
    for words_path in sorted(args.pages.glob("page*/words.json")):
        page = json.loads(words_path.read_text())
        owners = {}
        for line in page["lines"]:
            for word in line["words"]:
                for component_index in word.get("componentIndices", []):
                    owners[component_index] = word
        for component in page.get("components", []):
            if not component.get("maskPath") or not component.get("contextPath"):
                continue
            owner = owners.get(component["componentIndex"])
            if (
                not args.include_non_text_components
                and (owner is None or not owner.get("expectedText"))
            ):
                continue
            page_number = page["page"]
            record_id = f"p{page_number:03d}_c{component['componentIndex']:05d}"
            record = {
                    "id": record_id,
                    "page": page_number,
                    "componentIndex": component["componentIndex"],
                    "wordId": owner["wordId"] if owner else "",
                    "expectedText": owner["expectedText"] if owner else "",
                    "expectedMarks": owner.get("expectedMarks", []) if owner else [],
                    "wordImagePath": owner.get("pngPath", "") if owner else "",
                    "wordGeometry": {
                        "x": owner.get("x", 0) if owner else 0,
                        "y": owner.get("y", 0) if owner else 0,
                        "width": owner.get("w", 0) if owner else 0,
                        "height": owner.get("h", 0) if owner else 0,
                    },
                    "maskPath": component["maskPath"],
                    "contextPath": component["contextPath"],
                    "geometry": {
                        "x": component["x"],
                        "y": component["y"],
                        "width": component["w"],
                        "height": component["h"],
                        "area": component["area"],
                    },
                    "visualClass": None,
                    "semanticClass": None,
                    "templateRole": "whole",
                    "instanceGroupId": "",
                    "partRole": "whole",
                    "partIndex": 1,
                    "partCount": 1,
                    "orientationVariant": "unknown",
                    "correctionAction": "keep",
                    "correctedLineIndex": component.get("lineIndex", -1),
                    "correctedWordId": owner["wordId"] if owner else "",
                    "owningCharacterIndex": None,
                    "mergedClasses": [],
                    "touchesBody": None,
                    "isNoise": None,
                    "reviewed": False,
                    "split": (
                        "test"
                        if page_number % 10 == 0
                        else "validation"
                        if page_number % 10 == 5
                        else "train"
                    ),
                }
            record.update(
                review_metadata(
                    component,
                    owner,
                    args.auto_accept_confidence,
                    args.auto_accept_margin,
                )
            )
            previous = existing.get(record_id)
            if previous:
                for key in (
                    "visualClass",
                    "semanticClass",
                    "templateRole",
                    "instanceGroupId",
                    "partRole",
                    "partIndex",
                    "partCount",
                    "orientationVariant",
                    "correctionAction",
                    "correctedLineIndex",
                    "correctedWordId",
                    "owningCharacterIndex",
                    "mergedClasses",
                    "touchesBody",
                    "isNoise",
                    "reviewed",
                    "note",
                ):
                    if key in previous:
                        record[key] = previous[key]
                if previous.get("reviewed"):
                    record["reviewStatus"] = "reviewed"
            records.append(record)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
