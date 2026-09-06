#!/usr/bin/env python3
"""Focused integration regression for page 3 word-PAW segmentation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


TARGETS = {
    "l00_w00": {"paws": 2, "body": 2, "marks": 5},
    "l00_w01": {"paws": 3, "body": 3, "marks": 8},
    "l00_w03": {"paws": 3, "body": 3, "marks": 5},
    "l00_w04": {"paws": 1, "body": 1, "marks": 6},
    "l00_w08": {"paws": 3, "body": 3, "marks": 8},
    "l07_w01": {"paws": 3, "body": 3, "marks": 7},
    "l09_w09": {"paws": 3, "body": 3, "marks": 10},
    "l01_w03": {"paws": 1, "body": 1, "marks": 5},
    "l01_w04": {"paws": 2, "body": 2, "marks": 4},
    "l01_w01": {"paws": 3, "body": 3, "marks": 9},
    "l01_w08": {"paws": 1, "body": 1, "marks": 6},
    "l00_w05": {"paws": 5, "body": 5, "marks": 10},
    "l00_w06": {"paws": 2, "body": 2, "marks": 2},
    "l02_w00": {"paws": 3, "body": 3, "marks": 8},
    "l06_w02": {"paws": 2, "body": 2, "marks": 6},
    "l07_w03": {"paws": 4, "body": 4, "marks": 7},
    "l12_w06": {"paws": 2, "body": 2, "marks": 10},
    "l13_w04": {"paws": 3, "body": 3, "marks": 9},
    "l01_w06": {"paws": 2, "body": 2, "marks": 7},
    "l13_w06": {"paws": 4, "body": 4, "marks": 9},
    "l08_w01": {"paws": 2, "body": 2, "marks": 7},
    # These words contain physical connected-component deficits. The test
    # counts observed components, not the larger text-derived path count.
    "l06_w07": {"paws": 4, "body": 3, "marks": 4},
    "l08_w06": {"paws": 3, "body": 3, "marks": 8},
}

# These are semantic marks whose scan representation can legitimately contain
# several disconnected physical components. All other classes must have one
# component per exported mark mask; repeated marks of the same class must be
# represented by repeated mask entries instead of one class union.
MULTI_COMPONENT_MARK_CLASSES = {
    "dot_above_three",
    "fathatan",
    "kasratan",
    "dammatan",
    "waqf_qaf",
    "waqf_jeem",
    "waqf_three_dots",
}

# Reviewed page-3 component IDs used to exercise the template recognizer in
# this integration test. Component 897 is the user-confirmed fatha in
# l13_w04; it is deliberately included as a fatha template even though the
# generic DP assigns it to hamza without template evidence. The small-alef
# examples are independent reviewed vertical small-alef forms; component
# 1008 is intentionally not used as a template because it is the regression
# target in l14_w03.
TEMPLATE_COMPONENTS = {
    "fatha": (2, 85, 140, 402, 469, 897, 898, 900),
    "damma": (94, 846, 893),
    # Reviewed singleton and connected-pair forms used by the dot-slot
    # ambiguity regression in l01_w03.
    "dot_above_one": (43, 107, 111, 804),
    "dot_above_two": (103, 112, 934),
    "hamza_above": (413, 483, 873),
    "hamza_below": (67, 587),
    # Reviewed maddah components from other PAWs on page 3. Keep the
    # l13_w04 components out of the bank so the same-base case remains a
    # held-out check.
    "maddah": (21, 121, 294, 486, 533, 602, 608, 613, 619, 680, 684, 750, 759),
    "small_alef": (119, 122, 127, 171, 315, 571, 717, 800, 946, 998),
    # Keep the two l01_w08 sukun components out of the bank.
    "sukun": (18, 33, 35, 40, 47, 49, 53, 63, 87, 109, 156, 167),
    # The reviewed qaf body is deliberately kept in the same bank as the
    # ordinary mark templates. The detached companion is selected by the
    # waqf recognizer's geometry rule.
    "waqf_qaf": (596,),
}


def fail(message: str) -> None:
    raise SystemExit(f"word-PAW regression failed: {message}")


def find_words(words_path: Path) -> dict[str, dict]:
    data = json.loads(words_path.read_text(encoding="utf-8"))
    if data.get("schemaVersion") != 8:
        fail(f"expected words.json schema version 8, got {data.get('schemaVersion')}")
    words = {
        word["wordId"]: word
        for line in data["lines"]
        for word in line["words"]
    }
    for word in words.values():
        if "pieces" in word:
            fail(f"{word['wordId']}: legacy pieces field is still present")
    return words


def check_template_scores(words_path: Path) -> None:
    data = json.loads(words_path.read_text(encoding="utf-8"))
    components = {
        component["componentIndex"]: component
        for component in data["components"]
    }
    for component_index, expected_class in (
        (897, "fatha"),
        (939, "hamza_above"),
        (1008, "small_alef"),
        (937, "maddah"),
        (949, "small_alef"),
        (113, "sukun"),
        (479, "sukun"),
        (958, "hamza_below"),
        (107, "dot_above_one"),
        (112, "dot_above_two"),
    ):
        scores = components[component_index].get(
            "templateMarkProbabilities", {}
        )
        competing = {
            "fatha": "hamza_above",
            "hamza_above": "fatha",
            "small_alef": "fatha",
            "maddah": "small_alef",
            "sukun": "hamza_above",
            "hamza_below": "fatha",
            "dot_above_one": "dot_above_two",
            "dot_above_two": "dot_above_one",
        }[expected_class]
        if scores.get(expected_class, 0.0) <= scores.get(competing, 0.0):
            fail(
                f"component {component_index}: expected template score for "
                f"{expected_class} to exceed {competing}, got {scores}"
            )


def mask_sets(paw: dict, class_name: str) -> list[set[int]]:
    values = [
        set(mask["componentIndices"])
        for mask in paw.get("markMasks", [])
        if mask.get("className") == class_name
    ]
    return sorted(values, key=lambda value: tuple(sorted(value)))


def check_word(word_id: str, word: dict, output_root: Path) -> None:
    expected = TARGETS[word_id]
    paws = word.get("paws", [])
    if len(paws) != expected["paws"]:
        fail(f"{word_id}: expected {expected['paws']} PAWs, got {len(paws)}")

    word_components = sorted(word["componentIndices"])
    paw_components = sorted(
        component
        for paw in paws
        for component in paw["componentIndices"]
    )
    if paw_components != word_components:
        fail(f"{word_id}: PAWs do not partition the word components")

    body_components = [
        component
        for paw in paws
        for component in paw["bodyComponentIndices"]
    ]
    mark_components = [
        component
        for paw in paws
        for component in paw["markComponentIndices"]
    ]
    if len(body_components) != expected["body"]:
        fail(f"{word_id}: expected {expected['body']} body components")
    if len(mark_components) != expected["marks"]:
        fail(f"{word_id}: expected {expected['marks']} mark components")
    if set(body_components) & set(mark_components):
        fail(f"{word_id}: component assigned to both body and marks")

    for paw in paws:
        if not str(paw.get("pawIndex", "")).isdigit():
            fail(f"{word_id}: PAW is missing pawIndex")
        for field in ("pngPath", "bodyPath", "marksPath"):
            path = Path(paw[field])
            if "_paw" not in path.name:
                fail(f"{word_id}: generated file is not named as a PAW: {path}")
            if not path.is_file():
                fail(f"{word_id}: missing generated PAW file {path}")
        for mark_mask in paw.get("markMasks", []):
            if not mark_mask.get("className"):
                fail(f"{word_id}: categorized mask is missing className")
            path = Path(mark_mask["path"])
            if not path.is_file():
                fail(f"{word_id}: missing categorized mark mask {path}")
        categorized_components = [
            component
            for mark_mask in paw.get("markMasks", [])
            for component in mark_mask["componentIndices"]
        ]
        if sorted(categorized_components) != sorted(
            paw["markComponentIndices"]
        ):
            fail(f"{word_id}: categorized masks do not partition PAW marks")
        if len(categorized_components) != len(set(categorized_components)):
            fail(f"{word_id}: component appears in multiple categorized masks")
        paths = [mask["path"] for mask in paw.get("markMasks", [])]
        if len(paths) != len(set(paths)):
            fail(f"{word_id}: mark mask paths are not unique")
        for mark_mask in paw.get("markMasks", []):
            if (mark_mask.get("className") not in MULTI_COMPONENT_MARK_CLASSES
                    and len(mark_mask["componentIndices"]) > 1):
                fail(
                    f"{word_id}: {mark_mask['className']} combines separate "
                    "mark components in one mask"
                )

    if word_id == "l00_w00":
        if 67 not in mark_components or 67 in body_components:
            fail("l00_w00: hamza-below component 67 is not a mark")
        paw = paws[0]
        categorized = {
            mark["className"]: set(mark["componentIndices"])
            for mark in paw["markMasks"]
        }
        if categorized.get("hamza_below") != {67}:
            fail("l00_w00: component 67 is not in the hamza_below mask")
        if categorized.get("kasra") != {71}:
            fail("l00_w00: component 71 is not in the kasra mask")
        paw = paws[1]
        categorized = {
            mark["className"]: set(mark["componentIndices"])
            for mark in paw["markMasks"]
        }
        # On نَّ, the scan stacks the dot nearest the body, then shadda,
        # then fatha. The two elongated marks have similar generic evidence;
        # the stack order must keep their assignments from being swapped.
        for mark_class, component in (
            ("dot_above_one", 43),
            ("shadda", 22),
            ("fatha", 2),
        ):
            if categorized.get(mark_class) != {component}:
                fail(
                    f"l00_w00: expected component {component} in "
                    f"{mark_class} mask, got {categorized.get(mark_class)}"
                )
    if word_id == "l00_w01":
        if 29 not in body_components or 29 in mark_components:
            fail("l00_w01: Alef component 29 is not a body")
        paw = paws[1]
        categorized = {
            mark["className"]: set(mark["componentIndices"])
            for mark in paw["markMasks"]
        }
        for mark_class, component in (
            ("shadda", 5),
            ("fatha", 0),
        ):
            if categorized.get(mark_class) != {component}:
                fail(
                    f"l00_w01: expected component {component} in "
                    f"{mark_class} mask, got {categorized.get(mark_class)}"
                )
    if word_id == "l00_w03":
        # U+064C is one textual dammatan mark but renders as two separate
        # damma-shaped components. Both components must stay in the
        # dammatan mask instead of sending the second one to `other`.
        paw = paws[2]
        categorized = {
            mark["className"]: set(mark["componentIndices"])
            for mark in paw["markMasks"]
        }
        if categorized.get("dammatan") != {19, 31}:
            fail(
                "l00_w03: dammatan must contain components 19 and 31, got "
                f"{categorized}"
            )
        if "other" in categorized:
            fail(
                "l00_w03: second damma was left in the other mask, got "
                f"{categorized}"
            )
    if word_id == "l00_w04":
        # The connected two-dot component is 70. Component 73 is the
        # diagonal kasra on the following base and must not fill the second
        # physical slot of dot_below_two.
        paw = paws[0]
        categorized = {
            mark["className"]: set(mark["componentIndices"])
            for mark in paw["markMasks"]
        }
        if categorized.get("dot_below_two") != {70}:
            fail(
                "l00_w04: dot_below_two must contain component 70, got "
                f"{categorized}"
            )
        if categorized.get("kasra") != {73}:
            fail(
                "l00_w04: kasra must contain component 73, got "
                f"{categorized}"
            )
    if word_id == "l01_w03":
        # Component 107 is the singleton dot above خ. Component 112 is the
        # connected pair above ت; the singleton must not fill the pair slot.
        paw = paws[0]
        categorized = {
            mark["className"]: set(mark["componentIndices"])
            for mark in paw["markMasks"]
        }
        if categorized.get("dot_above_one") != {107}:
            fail(
                "l01_w03: dot_above_one must contain component 107, got "
                f"{categorized}"
            )
        if categorized.get("dot_above_two") != {112}:
            fail(
                "l01_w03: dot_above_two must contain component 112, got "
                f"{categorized}"
            )
    if word_id == "l07_w01":
        # On تُفْسِدُ, the sukun (479) is the outer mark above ف, while the
        # singleton dot (507) is nearest to that body. Component 500 is the
        # connected two-dot mark above ت. Sukun must use the same outer/top
        # stack rank as fatha and damma so it cannot be assigned the
        # dot_above_two slot.
        paw = paws[0]
        categorized = {
            mark["className"]: set(mark["componentIndices"])
            for mark in paw["markMasks"]
        }
        for mark_class, components in (
            ("dot_above_two", {500}),
            ("sukun", {479}),
            ("dot_above_one", {507}),
            ("kasra", {523}),
        ):
            if categorized.get(mark_class) != components:
                fail(
                    f"l07_w01: expected {components} in {mark_class} mask, "
                    f"got {categorized.get(mark_class)}"
                )
        if mask_sets(paw, "damma") != [{468}, {480}]:
            fail(
                "l07_w01: dammas must be separate masks for components 468 "
                f"and 480, got {mask_sets(paw, 'damma')}"
            )
    if word_id == "l09_w09":
        # Waqf qaf (ۗ) is encoded by two physical paths: the main qaf-shaped
        # glyph and its detached pair of dots. Both components belong to the
        # same waqf_qaf mask; the dots must not fall through to `other`.
        paw = paws[2]
        categorized = {
            mark["className"]: set(mark["componentIndices"])
            for mark in paw["markMasks"]
        }
        if categorized.get("waqf_qaf") != {595, 596}:
            fail(
                "l09_w09: waqf_qaf must contain components 595 and 596, "
                f"got {categorized.get('waqf_qaf')}"
            )
        if categorized.get("damma") != {644}:
            fail(
                "l09_w09: damma must contain component 644, got "
                f"{categorized.get('damma')}"
            )
        if "other" in categorized:
            fail(
                "l09_w09: waqf-qaf companion was left in the other mask, "
                f"got {categorized}"
            )
    if word_id == "l01_w04":
        paw = paws[1]
        categorized = {
            mark["className"]: set(mark["componentIndices"])
            for mark in paw["markMasks"]
        }
        for mark_class, component in (
            ("shadda", 98),
            ("fatha", 85),
            ("damma", 94),
        ):
            if categorized.get(mark_class) != {component}:
                fail(
                    f"l01_w04: expected component {component} in "
                    f"{mark_class} mask, got {categorized.get(mark_class)}"
                )
    if word_id == "l01_w08":
        # Component 113 is the second compact sukun. Component 79 is the
        # larger waqf-sad glyph; their generic mark probabilities are similar
        # enough that the final-DP shape prior must keep them separate.
        paw = paws[0]
        categorized = {
            mark["className"]: set(mark["componentIndices"])
            for mark in paw["markMasks"]
        }
        if mask_sets(paw, "sukun") != [{102}, {113}]:
            fail(
                "l01_w08: sukun components 102 and 113 must be separate "
                f"masks, got {mask_sets(paw, 'sukun')}"
            )
        if mask_sets(paw, "kasra") != [{132}, {135}]:
            fail(
                "l01_w08: kasra components 132 and 135 must be separate "
                f"masks, got {mask_sets(paw, 'kasra')}"
            )
        if categorized.get("waqf_sad") != {79}:
            fail(
                "l01_w08: waqf_sad must contain component 79, got "
                f"{categorized}"
            )
    if word_id == "l01_w01":
        if 118 not in body_components or 128 in body_components:
            fail("l01_w01: real body component 118 is not the PAW body")
        if 128 not in mark_components:
            fail("l01_w01: kasra component 128 is not a PAW mark")
        paw = paws[1]
        categorized = {
            mark["className"]: set(mark["componentIndices"])
            for mark in paw["markMasks"]
        }
        for mark_class, component in (
            ("kasra", 128),
            ("damma", 80),
            ("dot_above_one", 111),
        ):
            if categorized.get(mark_class) != {component}:
                fail(
                    f"l01_w01: expected component {component} in "
                    f"{mark_class} mask, got {categorized.get(mark_class)}"
                )
    if word_id == "l06_w02":
        # For أَ, fatha is above hamza-above. The two components have
        # similar generic mark evidence, so the stack order must decide it.
        paw = paws[0]
        categorized = {
            mark["className"]: set(mark["componentIndices"])
            for mark in paw["markMasks"]
        }
        for mark_class, component in (
            ("fatha", 402),
            ("hamza_above", 413),
        ):
            if categorized.get(mark_class) != {component}:
                fail(
                    f"l06_w02: expected component {component} in "
                    f"{mark_class} mask, got {categorized.get(mark_class)}"
                )
    if word_id == "l00_w05":
        # The hamza is attached to the Alef body; the detached upper mark is
        # the fatha, not a second hamza-above component.
        paw = paws[1]
        categorized = {
            mark["className"]: set(mark["componentIndices"])
            for mark in paw["markMasks"]
        }
        if categorized.get("fatha") != {4} or "hamza_above" in categorized:
            fail(
                "l00_w05: attached Alef must leave component 4 as fatha, "
                f"got {categorized}"
            )
    if word_id == "l00_w06":
        paw = paws[0]
        categorized = {
            mark["className"]: set(mark["componentIndices"])
            for mark in paw["markMasks"]
        }
        if categorized.get("fatha") != {14} or "hamza_above" in categorized:
            fail(
                "l00_w06: attached Alef must leave component 14 as fatha, "
                f"got {categorized}"
            )
    if word_id == "l02_w00":
        paw = paws[0]
        categorized = {
            mark["className"]: set(mark["componentIndices"])
            for mark in paw["markMasks"]
        }
        if categorized.get("fatha") != {140} or "hamza_above" in categorized:
            fail(
                "l02_w00: attached Alef must leave component 140 as fatha, "
                f"got {categorized}"
            )
    if word_id == "l07_w03":
        paw = paws[1]
        categorized = {
            mark["className"]: set(mark["componentIndices"])
            for mark in paw["markMasks"]
        }
        for mark_class, component in (
            ("fatha", 469),
            ("hamza_above", 483),
        ):
            if categorized.get(mark_class) != {component}:
                fail(
                    f"l07_w03: expected component {component} in "
                    f"{mark_class} mask, got {categorized.get(mark_class)}"
                )
    if word_id == "l12_w06":
        # The same rule applies to damma: it must be above hamza-above.
        paw = paws[1]
        categorized = {
            mark["className"]: set(mark["componentIndices"])
            for mark in paw["markMasks"]
        }
        for mark_class, component in (
            ("damma", 846),
            ("hamza_above", 873),
        ):
            if categorized.get(mark_class) != {component}:
                fail(
                    f"l12_w06: expected component {component} in "
                    f"{mark_class} mask, got {categorized.get(mark_class)}"
                )
    if word_id == "l13_w04":
        paw = paws[0]
        categorized = {
            mark["className"]: set(mark["componentIndices"])
            for mark in paw["markMasks"]
        }
        if categorized.get("damma") != {893} or "hamza_above" in categorized:
            fail(
                "l13_w04: attached Alef must leave component 893 as damma, "
                f"got {categorized}"
            )
        # The final PAW has a fatha on the kaf. The detached contour above the
        # final kaf is the second glyph path of kaf, not hamza-above. The
        # expectation builder exposes that path as an Other slot so a
        # hamza-like template cannot force it into the hamza class.
        paw = paws[2]
        categorized = {
            mark["className"]: set(mark["componentIndices"])
            for mark in paw["markMasks"]
        }
        if mask_sets(paw, "fatha") != [{896}, {897}]:
            fail(
                "l13_w04: fatha components 896 and 897 must be separate "
                f"masks, got {mask_sets(paw, 'fatha')}"
            )
        if categorized.get("other") != {939} or "hamza_above" in categorized:
            fail(
                "l13_w04: Kaf contour must be an other component, got "
                f"{categorized}"
            )
        if categorized.get("hamza_below") != {958} or "dot_below_two" in categorized:
            fail(
                "l13_w04: component 958 must be hamza_below, got "
                f"{categorized}"
            )
    if word_id == "l14_w03":
        # The vertical component 1008 is a small alef. Components 985 and
        # 986 are the two diagonal fathas in this PAW. This case verifies
        # both positive small-alef evidence and fatha as its competitor.
        paw = paws[0]
        categorized = {
            mark["className"]: set(mark["componentIndices"])
            for mark in paw["markMasks"]
        }
        if categorized.get("small_alef") != {1008}:
            fail(
                "l14_w03: component 1008 must be small_alef, got "
                f"{categorized}"
            )
        if mask_sets(paw, "fatha") != [{985}, {986}]:
            fail(
                "l14_w03: fatha components 985 and 986 must be separate "
                f"masks, got {mask_sets(paw, 'fatha')}"
            )
    if word_id == "l02_w01":
        # The vertical component 175 is the small alef above sheen. The
        # other two upper components are the three-dot mark, represented by
        # one single-dot component and one connected pair.
        paw = paws[0]
        categorized = {
            mark["className"]: set(mark["componentIndices"])
            for mark in paw["markMasks"]
        }
        if categorized.get("small_alef") != {175}:
            fail(
                "l02_w01: component 175 must be small_alef, got "
                f"{categorized}"
            )
        if categorized.get("dot_above_three") != {164, 174}:
            fail(
                "l02_w01: sheen dots must be components 164 and 174, got "
                f"{categorized}"
            )
    if word_id == "l11_w07":
        # Same small-alef/three-dot ambiguity in a second scan location.
        paw = paws[0]
        categorized = {
            mark["className"]: set(mark["componentIndices"])
            for mark in paw["markMasks"]
        }
        if categorized.get("small_alef") != {798}:
            fail(
                "l11_w07: component 798 must be small_alef, got "
                f"{categorized}"
            )
        if categorized.get("dot_above_three") != {790, 799}:
            fail(
                "l11_w07: sheen dots must be components 790 and 799, got "
                f"{categorized}"
            )
        # Components 815 and 817 are connected pairs of dots below the
        # baseline, each belonging to a different mark. The paired-dot
        # geometry must keep them out of the kasra class; the other
        # below-baseline components are the three diagonal kasras.
        if mask_sets(paw, "dot_below_two") != [{815}, {817}]:
            fail(
                "l11_w07: two-dot marks 815 and 817 must be separate masks, "
                f"got {mask_sets(paw, 'dot_below_two')}"
            )
        if mask_sets(paw, "kasra") != [{816}, {818}, {822}]:
            fail(
                "l11_w07: kasras 816, 818, and 822 must be separate masks, "
                f"got {mask_sets(paw, 'kasra')}"
            )
    if word_id == "l13_w04":
        # Small alef and maddah can both be floating marks. Their reviewed
        # templates distinguish the horizontal maddah (937) from the
        # vertical small alef (949) in this same-base stack.
        paw = paws[2]
        categorized = {
            mark["className"]: set(mark["componentIndices"])
            for mark in paw["markMasks"]
        }
        if categorized.get("small_alef") != {949}:
            fail(
                "l13_w04: component 949 must be small_alef, got "
                f"{categorized}"
            )
        if categorized.get("maddah") != {937}:
            fail(
                "l13_w04: component 937 must be maddah, got "
                f"{categorized}"
            )
    if word_id == "l01_w06":
        # For qaf, the damma is above qaf's two dots. This disambiguates
        # the connected components that have nearly identical generic mark
        # probabilities.
        paw = paws[0]
        categorized = {
            mark["className"]: set(mark["componentIndices"])
            for mark in paw["markMasks"]
        }
        if categorized.get("dot_above_two") != {103}:
            fail(
                "l01_w06: qaf's dot_above_two must be component 103, got "
                f"{categorized.get('dot_above_two')}"
            )
        if mask_sets(paw, "damma") != [{76}, {77}]:
            fail(
                "l01_w06: dammas 76 and 77 must be separate masks, got "
                f"{mask_sets(paw, 'damma')}"
            )
    if word_id == "l13_w06":
        paw = paws[1]
        categorized = {
            mark["className"]: set(mark["componentIndices"])
            for mark in paw["markMasks"]
        }
        if categorized.get("dot_above_three") != {935, 942}:
            fail(
                "l13_w06: three dots must be components 935 and 942, got "
                f"{categorized.get('dot_above_three')}"
            )
        if categorized.get("dot_above_two") != {934}:
            fail(
                "l13_w06: two dots must be component 934, got "
                f"{categorized.get('dot_above_two')}"
            )
        if mask_sets(paw, "fatha") != [{898}, {900}]:
            fail(
                "l13_w06: fathas 898 and 900 must be separate masks, got "
                f"{mask_sets(paw, 'fatha')}"
            )
    if word_id == "l08_w01":
        paw = paws[0]
        if 561 not in paw["bodyComponentIndices"]:
            fail("l08_w01: Alef component 561 is not the PAW body")
        if 587 in paw["bodyComponentIndices"]:
            fail("l08_w01: hamza-below component 587 is the PAW body")
        categorized = {
            mark["className"]: set(mark["componentIndices"])
            for mark in paw["markMasks"]
        }
        if categorized.get("hamza_below") != {587}:
            fail("l08_w01: component 587 is not in the hamza_below mask")
        if categorized.get("kasra") != {593}:
            fail("l08_w01: component 593 is not in the kasra mask")
    if word_id == "l06_w07":
        # Component 436 is a fatha. It overlaps a connected body component
        # at the PAW boundary; it must remain a mark even when this PAW has
        # no independently separable body component.
        paw = paws[0]
        if 436 in paw["bodyComponentIndices"]:
            fail("l06_w07: fatha component 436 was used as a PAW body")
        if 436 not in paw["markComponentIndices"]:
            fail("l06_w07: fatha component 436 is missing from PAW marks")
    if word_id == "l08_w06":
        # The main body of the first connected PAW is component 577. The
        # surrounding marks must not replace it as the selected body.
        paw = paws[0]
        if paw["bodyComponentIndices"] != [577]:
            fail(
                "l08_w06: connected PAW body must be component 577, got "
                f"{paw['bodyComponentIndices']}"
            )
        for mark_component in (591, 546, 574, 542, 565, 554):
            if mark_component in paw["bodyComponentIndices"]:
                fail(
                    f"l08_w06: mark component {mark_component} was used "
                    "as the connected PAW body"
                )


def check_page18_waqf(words_path: Path) -> None:
    data = json.loads(words_path.read_text(encoding="utf-8"))
    words = {
        word["wordId"]: word
        for line in data["lines"]
        for word in line["words"]
    }
    previous_line_word = words.get("l10_w08")
    target_word = words.get("l11_w07")
    if previous_line_word is None or target_word is None:
        fail("page 18: missing qaf boundary regression word")
    qaf_masks = [
        mask
        for paw in target_word.get("paws", [])
        for mask in paw.get("markMasks", [])
        if mask.get("className") == "waqf_qaf"
    ]
    qaf_components = (
        set(qaf_masks[0]["componentIndices"]) if len(qaf_masks) == 1 else set()
    )
    if len(qaf_components) != 2:
        fail(
            "page 18: expected two-component waqf_qaf body/companion, got "
            f"{[mask.get('componentIndices') for mask in qaf_masks]}"
        )
    if qaf_components.intersection(previous_line_word["componentIndices"]):
        fail("page 18: a qaf component stayed in l10_w08")
    if not any(
        "waqf_qaf component=" in correction and "line=10->11" in correction
        for correction in data.get("assignmentCorrections", [])
    ):
        fail("page 18: qaf companion line correction was not recorded")

    edge_word = words.get("l11_w08")
    if edge_word is None:
        fail("page 18: missing edge-artifact regression word")
    if edge_word["x"] <= 0:
        fail(
            "page 18: l11_w08 still starts at the cropped page boundary "
            f"(x={edge_word['x']})"
        )
    if (
        edge_word["observedComponentCount"]
        != edge_word["expectedComponentCount"]
    ):
        fail(
            "page 18: l11_w08 still has an extra edge component "
            f"({edge_word['observedComponentCount']} observed, "
            f"{edge_word['expectedComponentCount']} expected)"
        )


def check_page530_kasratan(words_path: Path) -> None:
    data = json.loads(words_path.read_text(encoding="utf-8"))
    words = {
        word["wordId"]: word
        for line in data["lines"]
        for word in line["words"]
    }
    upper_word = words.get("l13_w04")
    lower_word = words.get("l14_w04")
    if upper_word is None or lower_word is None:
        fail("page 530: missing kasratan boundary regression words")

    kasratan_masks = [
        mask
        for paw in upper_word.get("paws", [])
        for mask in paw.get("markMasks", [])
        if mask.get("className") == "kasratan"
    ]
    if len(kasratan_masks) != 1 or set(
        kasratan_masks[0]["componentIndices"]
    ) != {1105, 1110}:
        fail(
            "page 530: l13_w04 kasratan must contain both bars 1105 and "
            f"1110, got {[mask.get('componentIndices') for mask in kasratan_masks]}"
        )
    if (
        upper_word["observedComponentCount"]
        != upper_word["expectedComponentCount"]
        or lower_word["observedComponentCount"]
        != lower_word["expectedComponentCount"]
    ):
        fail(
            "page 530: kasratan reassignment left a word component mismatch "
            f"({upper_word['observedComponentCount']}/"
            f"{upper_word['expectedComponentCount']}, "
            f"{lower_word['observedComponentCount']}/"
            f"{lower_word['expectedComponentCount']})"
        )
    components = {
        component["componentIndex"]: component
        for component in data["components"]
    }
    if any(components[index]["lineIndex"] != 13 for index in (1105, 1110)):
        fail("page 530: kasratan bars were not assigned to line 13")
    if not any(
        "kasratan-open component=1110" in correction
        and "line=14->13" in correction
        for correction in data.get("assignmentCorrections", [])
    ):
        fail("page 530: open-kasratan line correction was not recorded")


def check_special_template_cache(
    attached_cache: Path, three_dot_cache: Path
) -> None:
    page3 = json.loads(attached_cache.read_text(encoding="utf-8"))
    attached = {
        tuple(recognition["componentParts"]): recognition
        for recognition in page3.get("recognitions", [])
        if recognition.get("attachedAlefHamzaAboveProbability", 0.0) > 0.0
    }
    recognition = attached.get((34,))
    if recognition is None or recognition["score"] <= recognition[
        "attachedAlefHamzaAboveProbability"
    ]:
        fail(
            "page 3: canonical attached alef+hamza template was not applied "
            "to component 34"
        )
    if (832,) in attached:
        fail(
            "page 3: wasla component 832 was incorrectly accepted as "
            "attached alef+hamza"
        )

    three_dot_page = json.loads(three_dot_cache.read_text(encoding="utf-8"))
    if not any(
        recognition.get("componentParts") == [51]
        and recognition.get("threeDotsAboveComponentCount") == 1
        for recognition in three_dot_page.get("recognitions", [])
    ):
        fail("page 30: canonical connected-three-dot template was not applied")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--database", required=True, type=Path)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="mushaf-word-paw-test-") as output:
        output_root = Path(output)
        command = [
            str(args.cli),
            "--input",
            str(args.input),
            "--out",
            str(output_root),
            "--pages",
            "3",
            "--database",
            str(args.database),
            "--export-component-dataset",
            "--no-debug-reconstructed",
            "--recognizer-cache",
            str(output_root / "recognizer-cache"),
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode != 0:
            fail(
                "mushaf_cli failed with exit code "
                f"{completed.returncode}: {completed.stderr}"
            )

        subwordimgs = output_root / "page003" / "subwordimgs"
        aggregate_masks = sorted(
            list(subwordimgs.glob("*_dots.png"))
            + list(subwordimgs.glob("*_uncertain.png"))
            + [
                path
                for path in subwordimgs.glob("*_marks.png")
                if "_paw" not in path.stem
            ]
        )
        if aggregate_masks:
            fail(
                "obsolete word-level aggregate masks were generated: "
                f"{aggregate_masks}"
            )

        template_root = output_root / "mark-templates"
        component_masks = (
            output_root / "page003" / "component-dataset" / "masks"
        )
        for class_name, component_indices in TEMPLATE_COMPONENTS.items():
            destination = template_root / class_name
            destination.mkdir(parents=True, exist_ok=True)
            for component_index in component_indices:
                source = component_masks / f"c{component_index:05d}.png"
                if not source.is_file():
                    fail(f"missing template source {source}")
                shutil.copyfile(
                    source, destination / source.name
                )

        template_command = [
            str(args.cli),
            "--input",
            str(args.input),
            "--out",
            str(output_root),
            "--pages",
            "3",
            "--database",
            str(args.database),
            "--mark-template-root",
            str(template_root),
            "--no-debug-reconstructed",
            "--recognizer-cache",
            str(output_root / "recognizer-cache"),
            "--reset-recognizer-cache",
            "--force",
        ]
        completed = subprocess.run(
            template_command, text=True, capture_output=True
        )
        if completed.returncode != 0:
            fail(
                "template-enabled mushaf_cli failed with exit code "
                f"{completed.returncode}: {completed.stderr}"
            )

        words = find_words(output_root / "page003" / "words.json")
        check_template_scores(output_root / "page003" / "words.json")
        for word_id in TARGETS:
            if word_id not in words:
                fail(f"missing target word {word_id}")
            check_word(word_id, words[word_id], output_root)

        page18_root = output_root / "page18"
        page18_command = [
            str(args.cli),
            "--input",
            str(args.input),
            "--out",
            str(page18_root),
            "--pages",
            "18",
            "--database",
            str(args.database),
            "--mark-template-root",
            str(template_root),
            "--no-debug-reconstructed",
            "--recognizer-cache",
            str(page18_root / "recognizer-cache"),
            "--recognizer-source",
            "old",
            "--reset-recognizer-cache",
            "--blurK",
            "0",
            "--morphK",
            "0",
        ]
        completed = subprocess.run(
            page18_command, text=True, capture_output=True
        )
        if completed.returncode != 0:
            fail(
                "page-18 template regression failed with exit code "
                f"{completed.returncode}: {completed.stderr}"
            )
        check_page18_waqf(page18_root / "page018" / "words.json")

        canonical_root = Path(__file__).resolve().parents[1] / "templates" / "oldmadina" / "mark-templates"
        canonical_page530_root = output_root / "canonical-page530"
        canonical_page530_command = [
            str(args.cli),
            "--input",
            str(args.input),
            "--out",
            str(canonical_page530_root),
            "--pages",
            "530",
            "--database",
            str(args.database),
            "--mark-template-root",
            str(canonical_root),
            "--no-debug-reconstructed",
            "--recognizer-cache",
            str(canonical_page530_root / "recognizer-cache"),
            "--recognizer-source",
            "old",
            "--reset-recognizer-cache",
            "--blurK",
            "0",
            "--morphK",
            "0",
        ]
        completed = subprocess.run(
            canonical_page530_command, text=True, capture_output=True
        )
        if completed.returncode != 0:
            fail(
                "canonical page-530 kasratan regression failed with exit code "
                f"{completed.returncode}: {completed.stderr}"
            )
        check_page530_kasratan(
            canonical_page530_root / "page530" / "words.json"
        )

        canonical_page3_root = output_root / "canonical-page3"
        canonical_page3_command = [
            str(args.cli),
            "--input",
            str(args.input),
            "--out",
            str(canonical_page3_root),
            "--pages",
            "3",
            "--database",
            str(args.database),
            "--mark-template-root",
            str(canonical_root),
            "--no-debug-reconstructed",
            "--recognizer-cache",
            str(canonical_page3_root / "recognizer-cache"),
            "--recognizer-source",
            "old",
            "--reset-recognizer-cache",
            "--blurK",
            "0",
            "--morphK",
            "0",
        ]
        completed = subprocess.run(
            canonical_page3_command, text=True, capture_output=True
        )
        if completed.returncode != 0:
            fail(
                "canonical page-3 template regression failed with exit code "
                f"{completed.returncode}: {completed.stderr}"
            )

        canonical_page30_root = output_root / "canonical-page30"
        canonical_page30_command = [
            str(args.cli),
            "--input",
            str(args.input),
            "--out",
            str(canonical_page30_root),
            "--pages",
            "30",
            "--database",
            str(args.database),
            "--mark-template-root",
            str(canonical_root),
            "--no-debug-reconstructed",
            "--recognizer-cache",
            str(canonical_page30_root / "recognizer-cache"),
            "--recognizer-source",
            "old",
            "--reset-recognizer-cache",
            "--blurK",
            "0",
            "--morphK",
            "0",
        ]
        completed = subprocess.run(
            canonical_page30_command, text=True, capture_output=True
        )
        if completed.returncode != 0:
            fail(
                "canonical page-30 template regression failed with exit code "
                f"{completed.returncode}: {completed.stderr}"
            )
        check_special_template_cache(
            canonical_page3_root / "recognizer-cache" / "old-page003.json",
            canonical_page30_root / "recognizer-cache" / "old-page030.json",
        )


if __name__ == "__main__":
    main()
