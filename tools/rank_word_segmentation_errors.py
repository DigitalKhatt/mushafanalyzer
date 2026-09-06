
#!/usr/bin/env python3
"""Rank likely word-boundary and cross-line segmentation errors."""

import argparse
import json
import pathlib


VARIABLE_CONNECTIVITY = {
    "dot_above_three",
    "fathatan",
    "kasratan",
    "dammatan",
    "hamza_above",
    "hamza_below",
    "waqf_qaf",
    "waqf_jeem",
    "waqf_three_dots",
}


def expected_count(word):
    return word.get(
        "effectiveExpectedComponentCount", word["expectedComponentCount"]
    )


def delta(word):
    return word["observedComponentCount"] - expected_count(word)


def summary(word):
    return {
        "wordId": word["wordId"],
        "text": word["expectedText"],
        "expected": expected_count(word),
        "observed": word["observedComponentCount"],
        "delta": delta(word),
        "components": word["componentIndices"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    candidates = []
    page_count = 0
    for words_path in sorted(args.pages.glob("**/page*/words.json")):
        page = json.loads(words_path.read_text())
        page_count += 1
        for line in page["lines"]:
            words = line["words"]
            line_delta = sum(delta(word) for word in words)
            for index in range(len(words) - 1):
                first, second = words[index], words[index + 1]
                if not first["expectedText"] or not second["expectedText"]:
                    continue
                first_delta, second_delta = delta(first), delta(second)
                if (
                    first_delta != 0
                    and second_delta != 0
                    and first_delta == -second_delta
                ):
                    candidates.append(
                        {
                            "score": 100 + 5 * abs(first_delta),
                            "kind": "adjacent_component_redistribution",
                            "page": page["page"],
                            "line": line["lineIndex"],
                            "lineDelta": line_delta,
                            "bboxGapPx": first["bboxGapToNextPx"],
                            "spacingGapPx": first["spacingGapToNextPx"],
                            "words": [summary(first), summary(second)],
                            "source": str(words_path),
                        }
                    )

            for word in words:
                if not word["expectedText"] or "۝" in word["expectedText"] or "۞" in word["expectedText"]:
                    continue
                quality = word["quality"]
                word_delta = delta(word)
                if quality["crossLineContamination"]:
                    candidates.append(
                        {
                            "score": 90 + 30 * quality["foreignLineInkFraction"],
                            "kind": "cross_line_contamination",
                            "page": page["page"],
                            "line": line["lineIndex"],
                            "lineDelta": line_delta,
                            "foreignLineInkFraction": quality[
                                "foreignLineInkFraction"
                            ],
                            "heightToLinePitch": quality["heightToLinePitch"],
                            "words": [summary(word)],
                            "source": str(words_path),
                        }
                    )
                if word_delta == 0:
                    continue
                marks = {
                    mark["className"] for mark in word.get("expectedMarks", [])
                }
                explained = word_delta < 0 and bool(
                    marks & VARIABLE_CONNECTIVITY
                )
                candidates.append(
                    {
                        "score": 35 + 5 * abs(word_delta) - (20 if explained else 0),
                        "kind": (
                            "variable_connectivity_candidate"
                            if explained
                            else "unexplained_component_mismatch"
                        ),
                        "page": page["page"],
                        "line": line["lineIndex"],
                        "lineDelta": line_delta,
                        "expectedMarks": sorted(marks),
                        "words": [summary(word)],
                        "source": str(words_path),
                    }
                )

    candidates.sort(
        key=lambda item: (-item["score"], item["page"], item["line"])
    )
    report = {
        "schemaVersion": 1,
        "analyzedPages": page_count,
        "candidateCount": len(candidates),
        "highPriorityCount": sum(item["score"] >= 90 for item in candidates),
        "candidates": candidates,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        f"{page_count} pages, {len(candidates)} candidates, "
        f"{report['highPriorityCount']} high priority"
    )
    print(args.out)


if __name__ == "__main__":
    main()
