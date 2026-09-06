#!/usr/bin/env python3
"""Summarize word component-count mismatches from analyzer JSON output."""

import argparse
import collections
import json
import pathlib


def likely_causes(word, delta):
    text = word["expectedText"]
    marks = {item["className"] for item in word["expectedMarks"]}
    causes = []
    if "۝" in text:
        causes.append("verse_marker_component_model")
    if "۞" in text:
        causes.append("rub_el_hizb_grouped_with_word")
    if delta < 0:
        if "hamza_above" in marks or "hamza_below" in marks:
            causes.append("hamza_attached_or_touching")
        if "dot_above_three" in marks:
            causes.append("three_dots_variable_connectivity")
        if marks & {"fathatan", "kasratan", "dammatan"}:
            causes.append("tanwin_variable_connectivity")
        if marks & {"waqf_qaf", "waqf_jeem", "waqf_three_dots"}:
            causes.append("multi_component_waqf_variable_connectivity")
        if not causes:
            causes.append("connected_marks_or_body_paths")
    elif delta > 0 and not causes:
        causes.append("disconnected_body_or_extra_ink")
    return causes


def count_residual(word):
    observed = word["observedComponentCount"]
    allowed = word.get(
        "allowedComponentCounts",
        [word.get("effectiveExpectedComponentCount",
                  word["expectedComponentCount"])],
    )
    nearest = min(allowed, key=lambda count: abs(observed - count))
    return observed - nearest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=pathlib.Path, required=True)
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--last", type=int, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    lines = []
    word_delta_counts = collections.Counter()
    cause_counts = collections.Counter()
    pages_without_mismatch = []
    for page in range(args.first, args.last + 1):
        path = args.pages / f"page{page:03d}" / "words.json"
        data = json.loads(path.read_text())
        page_has_mismatch = False
        for line in data["lines"]:
            mismatches = []
            for word in line["words"]:
                delta = count_residual(word)
                if word.get("componentCountAccepted", delta == 0):
                    continue
                page_has_mismatch = True
                causes = likely_causes(word, delta)
                cause_counts.update(causes)
                word_delta_counts[delta] += 1
                mismatches.append({
                    "wordId": word["wordId"],
                    "text": word["expectedText"],
                    "expected": word["expectedComponentCount"],
                    "observed": word["observedComponentCount"],
                    "allowed": word.get(
                        "allowedComponentCounts",
                        [word["expectedComponentCount"]]),
                    "delta": delta,
                    "likelyCauses": causes,
                })
            if mismatches:
                line_delta = sum(count_residual(word)
                                 for word in line["words"])
                if line_delta == 0:
                    line_reason = "word_boundary_redistribution"
                elif line_delta < 0:
                    line_reason = "fewer_connected_components_than_font_paths"
                else:
                    line_reason = "extra_components_or_fragmented_body"
                lines.append({
                    "page": page,
                    "line": line["lineIndex"],
                    "lineDelta": line_delta,
                    "reasonClass": line_reason,
                    "words": mismatches,
                })
        if not page_has_mismatch:
            pages_without_mismatch.append(page)

    report = {
        "schemaVersion": 1,
        "pageRange": [args.first, args.last],
        "mismatchedLineCount": len(lines),
        "mismatchedWordCount": sum(word_delta_counts.values()),
        "pagesWithoutMismatch": pages_without_mismatch,
        "wordDeltaCounts": dict(sorted(word_delta_counts.items())),
        "likelyCauseCounts": cause_counts,
        "limitations": [
            "Likely causes are candidate explanations, not character-level recognition.",
            "A word can have more than one cause.",
            "Exact attribution requires mark/body component grouping.",
        ],
        "lines": lines,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"{len(lines)} mismatched lines, "
          f"{sum(word_delta_counts.values())} mismatched words")
    print(args.out)


if __name__ == "__main__":
    main()
