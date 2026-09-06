#!/usr/bin/env python3
"""Data/API smoke test for the local segmentation review GUI."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from review_segmentation import HTML, SegmentationIndex  # noqa: E402


def fail(message: str) -> None:
    raise SystemExit(f"segmentation viewer regression failed: {message}")


def check_mark_group_scroll_guard() -> None:
    # Selecting a mark group rebuilds the right-hand main element. Keep this
    # source-level guard alongside the API smoke test because that replacement
    # must restore the independent result-list scroll position.
    required_fragments = (
        "async function renderMarks(resetResultScroll=false){",
        "const previousAside=document.querySelector('.app.global > aside');",
        "const previousAsideScrollTop=previousAside?.scrollTop||0;",
        "if(nextAside){nextAside.scrollTop=previousAsideScrollTop;",
        "const previousMain=document.querySelector('.app.global > main');",
        "const previousScrollTop=previousMain?.scrollTop||0;",
        "if(nextMain){nextMain.scrollTop=resetResultScroll?0:previousScrollTop;",
        "state.markOffset=0;renderMarks(true)",
    )
    for fragment in required_fragments:
        if fragment not in HTML:
            fail(f"mark-group scroll preservation is missing: {fragment}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    check_mark_group_scroll_guard()

    with tempfile.TemporaryDirectory(prefix="mushaf-segmentation-review-") as directory:
        index = SegmentationIndex(args.root, Path(directory) / "reviews.json")
        summary = index.validation_summary()
        if summary["pages"] == 0:
            fail("no pages were indexed")
        if summary["missingArtifacts"]:
            fail(f"missing artifacts: {summary['missingArtifacts'][:3]}")

        bootstrap = index.bootstrap()
        if len(bootstrap["pages"]) != summary["pages"]:
            fail("bootstrap page count differs from indexed page count")
        page = bootstrap["pages"][0]
        page_response = index.page_response(page["id"])
        if not page_response or not page_response["words"]:
            fail("page endpoint returned no words")
        word_key = page_response["words"][0]["wordKey"]
        word = index.word_response(word_key)
        if not word or not word["paws"]:
            fail("word endpoint returned no PAWs")
        if not all(paw["imageUrl"] and paw["bodyUrl"] and paw["marksUrl"]
                   for paw in word["paws"]):
            fail("word endpoint returned a PAW without image URLs")
        if any(not mask["imageUrl"]
               for paw in word["paws"] for mask in paw["markMasks"]):
            fail("word endpoint returned a mark without an image URL")

        marks = index.mark_response({"class": [""], "status": [""],
                                     "q": [""], "offset": ["0"]})
        paws = index.paw_response({"filter": ["all"], "status": [""], "q": [""]})
        if marks["total"] != summary["marks"]:
            fail("mark endpoint count differs from indexed mark count")
        if paws["total"] != summary["paws"]:
            fail("PAW endpoint count differs from indexed PAW count")

        mark_id = index.marks[0]["reviewId"]
        review = index.update_review("mark", mark_id, "error", "viewer smoke test")
        if review["status"] != "error":
            fail("review status was not stored")
        if index._review("mark", mark_id).get("note") != "viewer smoke test":
            fail("review note was not stored")

    print(summary)


if __name__ == "__main__":
    main()
