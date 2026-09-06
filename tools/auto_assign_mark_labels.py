#!/usr/bin/env python3
"""Propose mark labels using text inventory, RTL order, and baseline geometry.

The tool deliberately abstains unless a complete word has a unique,
geometry-compatible assignment. Proposals are not training labels until a
reviewer accepts them.
"""

import argparse
import collections
import json
import math
import pathlib
import unicodedata


COMBINING_CLASSES = {
    "\u064e": "fatha",
    "\u0650": "kasra",
    "\u064f": "damma",
    "\u064b": "fathatan",
    "\u08f0": "fathatan",
    "\u064d": "kasratan",
    "\u08f2": "kasratan",
    "\u064c": "dammatan",
    "\u08f1": "dammatan",
    "\u0651": "shadda",
    "\u0652": "sukun",
    "\u06e1": "sukun",
    "\u0653": "maddah",
    "\u0654": "hamza_above",
    "\u0655": "hamza_below",
    "\u0670": "small_alef",
    "\u06e6": "small_yeh",
    "\u06e7": "small_yeh",
    "\u06e5": "small_waw",
    "\u08f3": "small_waw",
    "\u06e2": "small_meem",
    "\u06ed": "small_meem",
    "\u06e8": "small_noon",
    "\u06dc": "small_seen",
    "\u06e3": "small_seen",
    "\u06d9": "waqf_lam",
    "\u06d7": "waqf_qaf",
    "\u06da": "waqf_jeem",
    "\u06d6": "waqf_sad",
    "\u06d8": "waqf_meem",
    "\u06db": "waqf_three_dots",
    "\u06df": "quranic_sign",
    "\u06e0": "quranic_sign",
    "\u06e9": "quranic_sign",
    "\u06ec": "quranic_sign",
}

DOT_BELOW_ONE = set("بج")
DOT_ABOVE_ONE = set("خذزضظغفن")
DOT_ABOVE_TWO = set("تقة")
DOT_BELOW_TWO = set("يى")
DOT_ABOVE_THREE = set("ثش")
BELOW_CLASSES = {
    "dot_below_one", "dot_below_two", "kasra", "kasratan", "hamza_below",
}
DOT_CLASSES = {
    "dot_above_one", "dot_below_one", "dot_above_two",
    "dot_below_two", "dot_above_three",
}
SMALL_CLASSES = {
    "small_alef", "small_yeh", "small_waw", "small_meem",
    "small_noon", "small_seen",
}


def expected_instances(text):
    bases = []
    instances = []
    current_base = -1
    for character in text:
        mark_class = COMBINING_CLASSES.get(character)
        if mark_class:
            if current_base < 0:
                return []
            instances.append((mark_class, current_base))
            continue
        if unicodedata.combining(character):
            return []
        current_base += 1
        bases.append(character)
        if character in DOT_BELOW_ONE:
            instances.append(("dot_below_one", current_base))
        elif character in DOT_ABOVE_ONE:
            instances.append(("dot_above_one", current_base))
        elif character in DOT_ABOVE_TWO:
            instances.append(("dot_above_two", current_base))
        elif character in DOT_BELOW_TWO:
            instances.append(("dot_below_two", current_base))
        elif character in DOT_ABOVE_THREE:
            instances.append(("dot_above_three", current_base))
        elif character == "\u0671":
            instances.append(("wasla", current_base))
    return instances, len(bases)


def component_cost(component, mark_class, anchor, base_count, baseline, word):
    center_x = component["x"] + component["w"] / 2
    center_y = component["y"] + component["h"] / 2
    word_width = max(1, word["w"])
    observed_position = (word["x"] + word_width - center_x) / word_width
    expected_position = anchor / max(1, base_count - 1)
    horizontal_error = abs(observed_position - expected_position)
    if horizontal_error > 0.48:
        return math.inf

    below = mark_class in BELOW_CLASSES
    clearance = 0.015 * max(word["h"], 1)
    if below and center_y < baseline - clearance:
        return math.inf
    if not below and center_y > baseline + clearance:
        return math.inf

    dot_probability = float(component.get("dotProbability", 0))
    ordinary_probability = float(component.get("ordinaryMarkProbability", 0))
    small_probability = float(component.get("smallLetterProbability", 0))
    if mark_class in DOT_CLASSES:
        # Two- and three-dot groups are often one connected contour and the
        # preliminary binary classifier may call them an ordinary mark. Text,
        # side of baseline, and RTL anchor order remain the hard evidence.
        type_probability = max(dot_probability, ordinary_probability * 0.55)
    elif mark_class in SMALL_CLASSES:
        type_probability = max(small_probability, ordinary_probability * 0.35)
    else:
        type_probability = ordinary_probability
    if type_probability < 0.20:
        return math.inf

    vertical_distance = abs(center_y - baseline) / max(word["h"], 1)
    return (
        2.5 * horizontal_error
        + 0.35 * vertical_distance
        - 0.30 * math.log(max(type_probability, 1e-6))
    )


def unique_class_arrangements(classes):
    counts = collections.Counter(classes)
    output = [None] * len(classes)

    def visit(index):
        if index == len(output):
            yield tuple(output)
            return
        for mark_class in sorted(counts):
            if counts[mark_class] == 0:
                continue
            counts[mark_class] -= 1
            output[index] = mark_class
            yield from visit(index + 1)
            counts[mark_class] += 1

    yield from visit(0)


def arrangement_cost(arrangement, components, instances_by_class, base_count,
                     baseline, word):
    total = 0.0
    for mark_class, expected in instances_by_class.items():
        candidate_indices = [
            index for index, assigned in enumerate(arrangement)
            if assigned == mark_class
        ]
        candidate_indices.sort(
            key=lambda index: components[index]["x"], reverse=True
        )
        anchors = sorted(anchor for _, anchor in expected)
        if len(candidate_indices) != len(anchors):
            return math.inf
        for candidate_index, anchor in zip(candidate_indices, anchors):
            cost = component_cost(
                components[candidate_index], mark_class, anchor, base_count,
                baseline, word,
            )
            if not math.isfinite(cost):
                return math.inf
            total += cost
    return total


def propose_word(word, line, components_by_id, maximum_marks, minimum_gap):
    if word["expectedComponentCount"] != word["observedComponentCount"]:
        return None, "component_count_mismatch"
    parsed = expected_instances(word["expectedText"])
    if not parsed:
        return None, "unsupported_text"
    instances, base_count = parsed
    if len(instances) != word["expectedMarkComponents"]:
        return None, "text_inventory_incomplete"
    if not instances or len(instances) > maximum_marks:
        return None, "mark_count_out_of_range"

    components = [components_by_id[index] for index in word["componentIndices"]]
    body = [component for component in components
            if component.get("bodyProbability", 0) >= 0.80]
    marks = [component for component in components
             if component.get("bodyProbability", 0) <= 0.20]
    if (
        len(body) != word["expectedBodyComponents"]
        or len(marks) != word["expectedMarkComponents"]
        or len(body) + len(marks) != len(components)
    ):
        return None, "ambiguous_body_mark_separation"

    marks.sort(key=lambda component: component["componentIndex"])
    by_class = collections.defaultdict(list)
    for instance in instances:
        by_class[instance[0]].append(instance)
    scored = []
    for arrangement in unique_class_arrangements([item[0] for item in instances]):
        cost = arrangement_cost(
            arrangement, marks, by_class, base_count, line["baseLine"], word
        )
        if math.isfinite(cost):
            scored.append((cost, arrangement))
    scored.sort(key=lambda item: item[0])
    if not scored:
        return None, "no_geometry_compatible_assignment"
    if len(scored) > 1 and scored[1][0] - scored[0][0] < minimum_gap:
        return None, "assignment_margin_too_small"

    confidence = 1.0 if len(scored) == 1 else 1.0 - math.exp(
        -(scored[1][0] - scored[0][0])
    )
    return [
        {
            "componentIndex": component["componentIndex"],
            "visualClass": mark_class,
            "assignmentCost": scored[0][0],
            "assignmentConfidence": confidence,
            "orientationVariant": (
                "vertical"
                if mark_class in ("dot_above_two", "dot_below_two")
                and component["h"] > 1.25 * component["w"]
                else "horizontal"
                if mark_class in ("dot_above_two", "dot_below_two")
                and component["w"] > 1.25 * component["h"]
                else "unknown"
            ),
        }
        for component, mark_class in zip(marks, scored[0][1])
    ], None


def propose_final_small_yeh(word, line, components_by_id, line_pitch):
    """Identify final U+06E6 from its stable Old Madinah geometry.

    U+06E6 follows the final base and is drawn as a compact disconnected glyph
    immediately to the left of the word body. It often intersects the baseline,
    so the preliminary classifier calls it body-like. Exact line and word
    component counts are required by the caller.
    """
    if "\u06e6" not in word["expectedText"]:
        return None, "no_final_small_yeh"
    if word["expectedText"].count("\u06e6") != 1:
        return None, "multiple_final_small_yeh"

    candidates = []
    pitch = max(1.0, line_pitch)
    for component_index in word["componentIndices"]:
        component = components_by_id[component_index]
        left = (component["x"] - word["x"]) / max(1, word["w"])
        width = component["w"] / pitch
        height = component["h"] / pitch
        area = component["area"] / (pitch * pitch)
        baseline_distance = abs(
            component["y"] + component["h"] / 2 - line["baseLine"]
        ) / pitch
        aspect = component["w"] / max(1, component["h"])
        if (
            left <= 0.08
            and 0.12 <= width <= 0.22
            and 0.07 <= height <= 0.15
            and 0.003 <= area <= 0.014
            and baseline_distance <= 0.10
            and 1.20 <= aspect <= 2.00
            and component.get("bodyProbability", 0.0) >= 0.75
        ):
            candidates.append(component)
    if len(candidates) != 1:
        return None, (
            "no_final_small_yeh_candidate"
            if not candidates
            else "multiple_final_small_yeh_candidates"
        )
    component = candidates[0]
    return [{
        "componentIndex": component["componentIndex"],
        "visualClass": "small_yeh",
        "assignmentCost": 0.0,
        # This is rule uniqueness, not a calibrated probability.
        "assignmentConfidence": 1.0,
        "orientationVariant": "unknown",
        "assignmentRule": "final_small_yeh_left_of_word",
    }], None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=pathlib.Path, required=True)
    parser.add_argument("--annotations", type=pathlib.Path, required=True)
    parser.add_argument("--maximum-marks", type=int, default=8)
    parser.add_argument("--minimum-gap", type=float, default=0.65)
    parser.add_argument(
        "--only-class",
        help="Write proposals only for this class (for example small_yeh)",
    )
    parser.add_argument(
        "--accept-proposals",
        action="store_true",
        help="Write unique proposals as reviewed labels after visual validation",
    )
    args = parser.parse_args()

    records = [
        json.loads(line) for line in args.annotations.read_text().splitlines()
        if line.strip()
    ]
    records_by_id = {record["id"]: record for record in records}
    summary = collections.Counter()

    for words_path in sorted(args.pages.glob("page*/words.json")):
        page = json.loads(words_path.read_text())
        components = {
            component["componentIndex"]: component
            for component in page["components"]
        }
        for line in page["lines"]:
            if not line["words"] or any(
                word["expectedComponentCount"] != word["observedComponentCount"]
                for word in line["words"]
            ):
                summary["line_component_count_mismatch"] += 1
                continue
            pitches = []
            line_index = line["lineIndex"]
            if line_index > 0:
                pitches.append(
                    line["baseLine"] - page["lines"][line_index - 1]["baseLine"]
                )
            if line_index + 1 < len(page["lines"]):
                pitches.append(
                    page["lines"][line_index + 1]["baseLine"] - line["baseLine"]
                )
            line_pitch = sum(pitches) / len(pitches) if pitches else 1.0
            for word in line["words"]:
                if args.only_class == "small_yeh":
                    if "\u06e6" not in word["expectedText"]:
                        continue
                    proposal, reason = propose_final_small_yeh(
                        word, line, components, line_pitch
                    )
                else:
                    proposal, reason = propose_word(
                        word, line, components, args.maximum_marks,
                        args.minimum_gap,
                    )
                if proposal is None:
                    summary[reason] += 1
                    continue
                proposal_id = f"p{page['page']:03d}_{word['wordId']}"
                written = 0
                for item in proposal:
                    if (
                        args.only_class
                        and item["visualClass"] != args.only_class
                    ):
                        continue
                    record_id = (
                        f"p{page['page']:03d}_c{item['componentIndex']:05d}"
                    )
                    record = records_by_id.get(record_id)
                    if record is None:
                        continue
                    if record.get("reviewed"):
                        summary[
                            "reviewed_proposal_correct"
                            if record.get("visualClass") == item["visualClass"]
                            else "reviewed_proposal_wrong"
                        ] += 1
                        continue
                    record["suggestedVisualClass"] = item["visualClass"]
                    record["autoAssignmentConfidence"] = item[
                        "assignmentConfidence"
                    ]
                    record["autoAssignmentCost"] = item["assignmentCost"]
                    record["autoAssignmentRule"] = item.get(
                        "assignmentRule", "constrained_text_geometry"
                    )
                    record["wordProposalId"] = proposal_id
                    if item["orientationVariant"] != "unknown":
                        record["orientationVariant"] = item[
                            "orientationVariant"
                        ]
                    if args.accept_proposals:
                        record["visualClass"] = item["visualClass"]
                        record["reviewed"] = True
                        record["reviewStatus"] = "reviewed"
                        record["note"] = (
                            record.get("note", "")
                            + " accepted unique "
                            + record["autoAssignmentRule"]
                        ).strip()
                    else:
                        record["reviewStatus"] = "proposed"
                    record["reviewReasons"] = ["constrained_text_geometry_proposal"]
                    summary["proposed_components"] += 1
                    written += 1
                if written:
                    summary["proposed_words"] += 1

    temporary = args.annotations.with_suffix(args.annotations.suffix + ".tmp")
    with temporary.open("w") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(args.annotations)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
