#!/usr/bin/env python3
"""Regression for PAW body order when one body is nested in another."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def fail(message: str) -> None:
    raise SystemExit(f"PAW body-inversion regression failed: {message}")


def load_words(output_root: Path, page: int) -> dict[str, dict]:
    data = json.loads(
        (output_root / f"page{page:03d}" / "words.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        word["wordId"]: word
        for line in data["lines"]
        for word in line["words"]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--database", required=True, type=Path)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="mushaf-paw-body-inversion-") as output:
        output_root = Path(output)
        command = [
            str(args.cli),
            "--input",
            str(args.input),
            "--out",
            str(output_root),
            "--pages",
            "49,105,166,230,234,517,526,544,559,567,573,579,584,585,587,591",
            "--jobs",
            "2",
            "--database",
            str(args.database),
            "--analysis-only",
            "--no-debug-reconstructed",
            "--blurK",
            "0",
            "--morphK",
            "0",
            "--mark-template-root",
            "templates/oldmadina/mark-templates",
            "--recognizer-cache",
            str(output_root / "recognizer-cache"),
            "--recognizer-source",
            "old",
        ]
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            cwd=Path(__file__).resolve().parents[1],
        )
        if completed.returncode != 0:
            fail(
                "mushaf_cli failed with exit code "
                f"{completed.returncode}: {completed.stderr}"
            )

        page105 = load_words(output_root, 105)
        word105 = page105.get("l05_w03")
        if not word105:
            fail("missing page 105 word l05_w03")
        paw0, paw1 = word105["paws"][:2]
        if paw0["bodyComponentIndices"] != [384]:
            fail(f"page 105 PAW 0 body was not recovered: {paw0}")
        if paw1["bodyComponentIndices"] != [344]:
            fail(f"page 105 PAW 1 body was not retained: {paw1}")
        if paw0["markComponentIndices"] != [364]:
            fail(f"page 105 PAW 0 marks changed: {paw0}")
        if sorted(paw1["markComponentIndices"]) != [342, 346, 373, 381]:
            fail(f"page 105 PAW 1 marks changed: {paw1}")

        page049 = load_words(output_root, 49)
        word049 = page049.get("l10_w09")
        if not word049:
            fail("missing page 049 word l10_w09")
        paw0, paw1 = word049["paws"][:2]
        if paw0["bodyComponentIndices"] != [757]:
            fail(f"page 049 Alef body was not recovered: {paw0}")
        if paw1["bodyComponentIndices"] != [718]:
            fail(f"page 049 PAW 1 body was not retained: {paw1}")
        if paw0["markComponentIndices"] != [742]:
            fail(f"page 049 PAW 0 wasla changed: {paw0}")
        if 757 in paw1["markComponentIndices"]:
            fail(f"page 049 Alef was still classified as a mark: {paw1}")

        page587 = load_words(output_root, 587)
        word587 = page587.get("l04_w09")
        if not word587:
            fail("missing page 587 word l04_w09")
        paw0, paw1 = word587["paws"][:2]
        if paw0["bodyComponentIndices"] != [324]:
            fail(f"page 587 first PAW body was not recovered: {paw0}")
        if paw0["markComponentIndices"] != [301]:
            fail(f"page 587 first PAW mark was not recovered: {paw0}")
        if paw0["observedMarkComponents"] != 1:
            fail(f"page 587 first PAW mark count was not updated: {paw0}")
        if paw1["bodyComponentIndices"] != [282]:
            fail(f"page 587 PAW 1 body was not retained: {paw1}")
        if sorted(paw1["markComponentIndices"]) != [263, 276, 280, 294, 311, 328]:
            fail(f"page 587 PAW 1 marks changed: {paw1}")
        if 324 in paw1["markComponentIndices"]:
            fail(f"page 587 first PAW body was still classified as a mark: {paw1}")

        page230 = load_words(output_root, 230)
        word230 = page230.get("l02_w00")
        if not word230:
            fail("missing page 230 word l02_w00")
        paw1, paw2 = word230["paws"][1:3]
        if paw1["bodyComponentIndices"] != [211]:
            fail(f"page 230 first body was not kept before long Kaf: {paw1}")
        if paw2["bodyComponentIndices"] != [164]:
            fail(f"page 230 long Kaf body was not handed to PAW 2: {paw2}")
        if 164 in paw1["markComponentIndices"]:
            fail(f"page 230 long Kaf body remained in PAW 1: {paw1}")
        if 211 in paw2["markComponentIndices"]:
            fail(f"page 230 first PAW body leaked into PAW 2: {paw2}")

        page584 = load_words(output_root, 584)
        word584 = page584.get("l01_w05")
        if not word584:
            fail("missing page 584 word l01_w05")
        paw0, paw1 = word584["paws"][:2]
        if paw0["bodyComponentIndices"] != [169]:
            fail(f"page 584 first body was not kept before long Kaf: {paw0}")
        if paw1["bodyComponentIndices"] != [100]:
            fail(f"page 584 long Kaf body was not handed to PAW 1: {paw1}")
        if 100 in paw0["markComponentIndices"]:
            fail(f"page 584 long Kaf body remained in PAW 0: {paw0}")
        if 169 in paw1["markComponentIndices"]:
            fail(f"page 584 first PAW body leaked into PAW 1: {paw1}")

        def assert_adjacent_bodies(
            page: int,
            word_id: str,
            first_paw: int,
            expected_first: int,
            expected_second: int,
        ) -> None:
            word = load_words(output_root, page).get(word_id)
            if not word:
                fail(f"missing page {page:03d} word {word_id}")
            paws = word["paws"]
            if first_paw + 1 >= len(paws):
                fail(
                    f"page {page:03d} word {word_id} has no PAW pair "
                    f"{first_paw}/{first_paw + 1}: {paws}"
                )
            first, second = paws[first_paw : first_paw + 2]
            if first["bodyComponentIndices"] != [expected_first]:
                fail(
                    f"page {page:03d} word {word_id} PAW {first_paw} "
                    f"body was not recovered: {first}"
                )
            if second["bodyComponentIndices"] != [expected_second]:
                fail(
                    f"page {page:03d} word {word_id} PAW {first_paw + 1} "
                    f"body was not retained: {second}"
                )
            if expected_second in first["markComponentIndices"]:
                fail(
                    f"page {page:03d} word {word_id} long body leaked into "
                    f"PAW {first_paw} marks: {first}"
                )
            if expected_first in second["markComponentIndices"]:
                fail(
                    f"page {page:03d} word {word_id} first body leaked into "
                    f"PAW {first_paw + 1} marks: {second}"
                )

        assert_adjacent_bodies(559, "l09_w04", 0, 794, 732)
        assert_adjacent_bodies(517, "l03_w07", 0, 312, 254)
        assert_adjacent_bodies(234, "l09_w06", 0, 695, 644)
        assert_adjacent_bodies(544, "l11_w08", 0, 977, 916)
        assert_adjacent_bodies(584, "l12_w04", 0, 1068, 1012)
        assert_adjacent_bodies(567, "l03_w07", 0, 336, 273)
        assert_adjacent_bodies(585, "l04_w09", 0, 355, 296)
        assert_adjacent_bodies(591, "l14_w09", 1, 1067, 1013)
        assert_adjacent_bodies(166, "l04_w03", 3, 351, 342)

        page526 = load_words(output_root, 526)
        word526 = page526.get("l07_w07")
        if not word526:
            fail("missing page 526 word l07_w07")
        paws526 = word526["paws"]
        expected_bodies526 = [[554], [521], [547]]
        for index, expected in enumerate(expected_bodies526):
            if paws526[index]["bodyComponentIndices"] != expected:
                fail(
                    f"page 526 PAW {index} body order was not preserved by "
                    f"baseline ink: {paws526[index]}"
                )
        if paws526[0]["markComponentIndices"] != [497, 531]:
            fail(f"page 526 Zay marks changed: {paws526[0]}")
        if sorted(paws526[2]["markComponentIndices"]) != [490, 527]:
            fail(f"page 526 Ghayn marks changed: {paws526[2]}")

        page573 = load_words(output_root, 573)
        word573 = page573.get("l14_w05")
        if not word573:
            fail("missing page 573 word l14_w05")
        paws573 = word573["paws"]
        expected_bodies573 = [[1099], [1062], [1100]]
        for index, expected in enumerate(expected_bodies573):
            if paws573[index]["bodyComponentIndices"] != expected:
                fail(
                    f"page 573 PAW {index} attached Alef body assignment was "
                    f"not recovered: {paws573[index]}"
                )
        if 1062 in paws573[1]["markComponentIndices"]:
            fail(f"page 573 attached Alef remained a mark: {paws573[1]}")
        if 1100 in paws573[1]["markComponentIndices"]:
            fail(
                "page 573 following connected body leaked into PAW 1: "
                f"{paws573[1]}"
            )
        if paws573[1]["markComponentIndices"] != [1052]:
            fail(f"page 573 Alef fatha assignment changed: {paws573[1]}")

        page579 = load_words(output_root, 579)
        word579 = page579.get("l03_w07")
        if not word579:
            fail("missing page 579 word l03_w07")
        paws579 = word579["paws"]
        expected_bodies579 = [[280], [320], [292]]
        for index, expected in enumerate(expected_bodies579):
            if paws579[index]["bodyComponentIndices"] != expected:
                fail(
                    f"page 579 PAW {index} dot-below-two body assignment was "
                    f"not recovered: {paws579[index]}"
                )
        if paws579[1]["markComponentIndices"] != [330, 264, 286]:
            fail(f"page 579 PAW 1 marks changed: {paws579[1]}")
        if 330 in paws579[1]["bodyComponentIndices"]:
            fail(f"page 579 connected two-dot mark remained a body: {paws579[1]}")
        if 320 in paws579[2]["markComponentIndices"]:
            fail(f"page 579 PAW 1 body leaked into PAW 2: {paws579[2]}")
        page579_data = json.loads(
            (output_root / "page579" / "words.json").read_text(
                encoding="utf-8"
            )
        )
        component579 = next(
            component
            for component in page579_data["components"]
            if component["componentIndex"] == 330
        )
        if component579.get("templateMarkProbabilities", {}).get(
            "dot_below_two", 0.0
        ) < 0.72:
            fail(
                "page 579: connected two-dot template was not applied to "
                f"component 330: {component579}"
            )

        recognizer_cache = (
            output_root / "recognizer-cache" / "old-page573.json"
        )
        cache = json.loads(recognizer_cache.read_text(encoding="utf-8"))
        if cache.get("schemaVersion") != 6:
            fail(
                "page 573 recognizer cache was not recomputed with schema 6: "
                f"{cache.get('schemaVersion')}"
            )
        attached = [
            recognition
            for recognition in cache.get("recognitions", [])
            if recognition.get("componentParts") == [1062]
        ]
        if (
            not attached
            or attached[0].get("attachedAlefHamzaAboveProbability", 0.0)
            < 0.82
        ):
            fail(
                "page 573 attached Alef-with-hamza template fallback was not "
                f"recorded for component 1062: {attached}"
            )


if __name__ == "__main__":
    main()
