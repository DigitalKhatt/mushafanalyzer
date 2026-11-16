#!/usr/bin/env python3
"""Build the searchable letter/mark corpus index.

Runs the PAW letter segmenter over analyzed pages and records one row per
letter and per mark, with the measurements needed to answer questions like
"all initial beh sorted by width" across the whole Mushaf.

The segmenter is run **in process** rather than reading its JSON output,
because the per-pixel letter attribution (``PawResult.pixel_letter``) is a
numpy array that never reaches the JSON, and it is what makes a real letter
mask -- and therefore a traced letter outline -- possible.

Example::

    uv run tools/corpus/build_corpus.py \
        --analysis output/oldmedina2-analysis \
        --out output/corpus/corpus.sqlite --pages 3 --jobs 4

The build is idempotent per page: re-running ``--pages 300-350`` after a
segmenter change replaces exactly those pages. Instance ids are deterministic,
so an id that named a letter before a rebuild names the same letter after it.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass, field
import datetime as dt
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple
import uuid

# One process per core is this builder's whole parallel strategy, so the
# libraries underneath it must not each open a thread pool of their own: on a
# 14-core machine, `--jobs 12` otherwise means twelve OpenCV pools and twelve
# BLAS pools competing for the same cores. These have to be set before numpy
# is imported to take effect, and macOS starts pool workers with `spawn`,
# which re-imports this module -- so setting them here covers worker and
# parent alike. An explicit value from the environment wins.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                  "NUMEXPR_NUM_THREADS"):
  os.environ.setdefault(_variable, "1")

import cv2
import numpy as np

cv2.setNumThreads(1)

_HERE = Path(__file__).resolve().parent
_SEGMENTATION = _HERE.parents[0] / "paw_segmentation"
for _path in (str(_HERE), str(_SEGMENTATION)):
  if _path not in sys.path:
    sys.path.insert(0, _path)

import paw_width                                         # noqa: E402
import segment_paw_letters as seg                        # noqa: E402

from geometry import (                                   # noqa: E402
    DEFAULT_FIT,
    LetterFrame,
    advance_width,
    contours_to_path,
    encode_mask_rle,
    normalized,
    relative_path,
    trace_mask,
)
import letter_classes                                     # noqa: E402
from letter_classes import shape_class                    # noqa: E402

SCHEMA = _HERE / "schema.sql"


# ---------------------------------------------------------------------------
# identifiers
# ---------------------------------------------------------------------------

def page_id(page: int) -> str:
  return f"p{page:03d}"


def word_id(page: int, word_key: str) -> str:
  return f"{page_id(page)}:{word_key}"


def paw_id(page: int, word_key: str, paw_index: int) -> str:
  return f"{word_id(page, word_key)}:paw{paw_index}"


def letter_id(page: int, word_key: str, paw_index: int, letter_index: int) -> str:
  return f"{paw_id(page, word_key, paw_index)}:let{letter_index}"


def mark_id(page: int, word_key: str, paw_index: int, class_name: str,
            component: int) -> str:
  return f"{paw_id(page, word_key, paw_index)}:mk:{class_name}:{component}"


# ---------------------------------------------------------------------------
# geometry helpers over one segmentation result
# ---------------------------------------------------------------------------

def spine_point(medial, s: float) -> Optional[Tuple[float, float]]:
  """Word-space point at arclength ``s`` along a PAW's spine.

  The spine runs right to left, so s=0 is where the pen enters the PAW and
  s=spine_length is where it leaves -- which is exactly what an advance width
  is measured between.
  """
  if medial is None or len(getattr(medial, "spine", ())) == 0:
    return None
  spine_s = np.asarray(medial.spine_s, dtype=float)
  points = np.asarray(medial.vertices, dtype=float)[list(medial.spine)]
  if points.shape[0] == 0:
    return None
  s = float(np.clip(s, spine_s[0], spine_s[-1]))
  index = int(np.searchsorted(spine_s, s, side="left"))
  if index <= 0:
    x, y = points[0]
  elif index >= points.shape[0]:
    x, y = points[-1]
  else:
    span = spine_s[index] - spine_s[index - 1]
    t = 0.0 if span <= 0 else (s - spine_s[index - 1]) / span
    x, y = points[index - 1] + t * (points[index] - points[index - 1])
  return (float(x) - seg.PAD_PX, float(y) - seg.PAD_PX)


def letter_spans(paw) -> List[Tuple[float, float, int, float]]:
  """(entry_s, exit_s, group_size, kashida_s) per letter of a PAW.

  ``segment_groups`` already carries the body span: one group per letter
  normally, and one group spanning several letters where a template consumed
  them without an internal cut. Letters of such a group have no individual
  advance width, and saying so is better than inventing a boundary inside a
  shape the segmenter deliberately refused to cut.

  A letter owns the connector on its left, up to where the next body starts,
  so ``exit_s`` runs to the next group's ``a_s`` rather than stopping at this
  one's ``b_s``. That is what makes the width an advance: entry of this letter
  to entry of the next, kashida included. The last letter of a PAW has no
  connector and ends at its own body. ``kashida_s`` is how much of the span is
  connector, kept separately because "how long is the tooth" and "how long is
  the stroke leaving it" are different questions.
  """
  count = len(paw.letters)
  spans: List[Tuple[float, float, int, float]] = [
      (0.0, float(paw.spine_length), 1, 0.0) for _ in range(count)]
  groups = sorted(paw.segment_groups, key=lambda group: group.a_s)
  for position, group in enumerate(groups):
    size = group.letter_end - group.letter_start
    body_end = float(group.b_s)
    exit_s = body_end
    if position + 1 < len(groups):
      exit_s = max(body_end, float(groups[position + 1].a_s))
    for index in range(group.letter_start, min(group.letter_end, count)):
      spans[index] = (float(group.a_s), exit_s, size, exit_s - body_end)
  return spans


def page_word_count(analysis_root: str, page: int) -> int:
  """How many words the analyzer produced for a page, ornaments included."""
  words_json = os.path.join(analysis_root, f"page{page:03d}", "words.json")
  if not os.path.isfile(words_json):
    return 0
  with open(words_json, encoding="utf-8") as handle:
    page_json = json.load(handle)
  return sum(len(line.get("words", []) or [])
             for line in page_json.get("lines", []))


@dataclass
class PageBuild:
  """Everything one page contributes, before page-scale normalization."""
  page: int
  source_image: str = ""
  image_w: int = 0
  image_h: int = 0
  words: List[dict] = field(default_factory=list)
  paws: List[dict] = field(default_factory=list)
  letters: List[dict] = field(default_factory=list)
  shapes: List[dict] = field(default_factory=list)
  marks: List[dict] = field(default_factory=list)
  observations: List[dict] = field(default_factory=list)
  errors: List[str] = field(default_factory=list)
  # Base letters the text calls for, whether or not the segmenter located
  # them. A shortfall against len(letters) is the one failure that would make
  # a query under-report without saying so, so it is counted, not assumed.
  expected_letters: int = 0
  # Which mark-mask contract this page was written with; see mark_mask_format.
  mark_format: str = ""
  # Mask entries with ink but no usable component index, so no geometry.
  marks_unindexed: int = 0
  # Verse ornaments left out of the index; see is_ayah_word. Counted so a page
  # line still accounts for every word the analyzer produced.
  ayah_skipped: int = 0
  # The page was asked for but the analyzer has not produced it. Distinct
  # from a page that produced nothing: a skipped page leaves whatever the
  # index already holds for it alone.
  missing: bool = False


INSTANCE_MASK_NAME = re.compile(r"_\d\d\.png$")

# U+06DD END OF AYAH and U+06DE START OF RUB EL HIZB are the verse ornaments;
# an ayah word is one of them plus the Arabic-Indic digits it encircles. None
# of it is handwriting: there is no letter shape to search for and no width
# worth comparing against a real letter's, and the ornament's hole-riddled
# outline is expensive to trace. Skipping the word skips its digits too, which
# is the point -- they are numerals, not letters.
AYAH_MARKS = frozenset("۝۞")


def is_ayah_word(text: str) -> bool:
  return any(ch in AYAH_MARKS for ch in text)


def mark_mask_format(page_json: dict) -> str:
  """Which mark-mask contract this page was written with.

  `PageProcessor.cpp` now groups a PAW's mark components by their textual mark
  instance and writes one mask each, named `..._paw01_fatha_00.png`. Before
  that it wrote one mask per class per PAW, so two fathas over one PAW shared
  an entry. The trailing ordinal is the discriminator, and the choice has to
  be made per page because an analysis directory can hold output from both.
  """
  for line in page_json.get("lines", []):
    for word in line.get("words", []):
      for paw in word.get("paws") or []:
        for mask in paw.get("markMasks") or []:
          if INSTANCE_MASK_NAME.search(str(mask.get("path", "") or "")):
            return "instance"
  return "class-group"


def collect_marks(page: int, word: dict, page_components: Sequence[dict],
                  letters_by_paw: Dict[int, List[dict]],
                  analysis_root: str, mark_format: str,
                  build: "PageBuild") -> List[dict]:
  """Mark instance rows, owned by a letter where unambiguous.

  With per-instance masks the analyzer has already decided what one mark is,
  so an entry is a row and nothing here needs to guess. The class-based
  splitting is kept only for older output, where a class entry could hold
  several marks; see `letter_classes.MARK_PARTS`.
  """
  rows: List[dict] = []
  key = word["wordId"]
  for paw in word.get("paws", []) or []:
    paw_index = int(paw.get("pawIndex", 0))
    letters = letters_by_paw.get(paw_index, [])
    for mask in paw.get("markMasks", []) or []:
      indices = [int(i) for i in mask.get("componentIndices", []) or []
                 if 0 <= int(i) < len(page_components)]
      class_name = str(mask.get("className", "") or "") or "unknown"
      if not indices:
        # The mask has ink but no usable component index, so there is nothing
        # to measure it against. Counted rather than dropped in silence.
        build.marks_unindexed += 1
        continue
      if mark_format == "instance":
        groups = [(indices, "instance")]
      else:
        parts = letter_classes.mark_parts(class_name)
        if parts == 1:
          groups = [([index], "component") for index in indices]
        elif len(indices) <= parts:
          groups = [(indices, "class-parts")]
        else:
          groups = [(indices, "ambiguous")]

      for members, grouping in groups:
        boxes = [page_components[i] for i in members]
        x0 = min(int(c["x"]) for c in boxes)
        y0 = min(int(c["y"]) for c in boxes)
        x1 = max(int(c["x"]) + int(c["w"]) for c in boxes)
        y1 = max(int(c["y"]) + int(c["h"]) for c in boxes)
        local_x = x0 - int(word["x"])
        local_y = y0 - int(word["y"])
        center = local_x + (x1 - x0) / 2.0

        owner = None
        for letter in letters:
          if letter["bbox_x"] is None:
            continue
          if letter["bbox_x"] <= center < letter["bbox_x"] + letter["bbox_w"]:
            owner = letter["letter_id"]
            break

        rows.append({
            "mark_id": mark_id(page, key, paw_index, class_name, min(members)),
            "page": page,
            "word_id": word_id(page, key),
            "paw_id": paw_id(page, key, paw_index),
            "class": class_name,
            "component_indices": json.dumps(members),
            "component_count": len(members),
            "grouping": grouping,
            "x": local_x, "y": local_y, "w": x1 - x0, "h": y1 - y0,
            "page_x": x0, "page_y": y0,
            "area": sum(int(c.get("area", 0)) for c in boxes),
            "mask_path": relative_path(str(mask.get("path", "")), analysis_root),
            "owning_letter_id": owner,
            "confidence": None,
        })
  return rows


def build_page(analysis_root: str, page: int, options: dict,
               word_slice: Optional[Tuple[int, int]] = None) -> PageBuild:
  """Segment one page and collect every row it contributes.

  ``word_slice`` restricts the work to a half-open range of the page's words,
  counted over every word the analyzer produced, ayah ornaments included, so
  that the slices of a page partition it exactly and each one charges itself
  for what it skipped. `merge_shards` puts the pieces back together.
  """
  page_dir = os.path.join(analysis_root, f"page{page:03d}")
  build = PageBuild(page=page)
  words_json = os.path.join(page_dir, "words.json")
  if not os.path.isfile(words_json):
    # A page range is a request, not a promise: the analyzer has only run on
    # part of the Mushaf, and asking for 3-604 should index what exists.
    build.missing = True
    return build
  with open(words_json, encoding="utf-8") as handle:
    page_json = json.load(handle)

  build.source_image = relative_path(str(page_json.get("sourceImage", "")),
                                     analysis_root)
  build.image_w = int(page_json.get("imageWidth", 0))
  build.image_h = int(page_json.get("imageHeight", 0))
  components = page_json.get("components", []) or []
  build.mark_format = mark_mask_format(page_json)

  templates, elastic = seg.load_template_definitions(options["templates"])
  limit = options.get("limit_words") or 0
  seen = 0

  keep_ayah = options.get("ayah_words", False)
  first_word, last_word = word_slice if word_slice else (0, None)
  word_index = -1

  for line in page_json.get("lines", []):
    for word in line.get("words", []):
      word_index += 1
      if word_index < first_word or (last_word is not None
                                     and word_index >= last_word):
        continue
      if not keep_ayah and is_ayah_word(word.get("expectedText", "") or ""):
        # Skipped before the word budget is charged, so --limit-words still
        # means that many real words.
        build.ayah_skipped += 1
        continue
      if limit and seen >= limit:
        return build
      seen += 1
      key = word["wordId"]
      try:
        result = seg.run_one(
            page_dir, line, word, out_dir="", write_overlay=False,
            want_pixels=True,
            min_confidence=options["min_confidence"],
            skeleton_shapes=options["skeleton_shapes"],
            logical_ligatures=options["logical_ligatures"],
            skeleton_templates=templates,
            elastic_body_templates=elastic,
            body_join_alignment=options["body_join_alignment"],
            cut_policy=options["cut_policy"])
      except Exception as error:                       # keep the page going
        build.errors.append(f"{page_id(page)} {key}: {type(error).__name__}: {error}")
        continue
      if result is None:
        continue

      build.observations.extend(paw_width.page_width_observations(result))
      build.expected_letters += len(result.bases)
      baseline = float(result.baseline_y) - seg.PAD_PX
      build.words.append({
          "word_id": word_id(page, key),
          "page": page,
          "line_index": int(word.get("lineIndex", line.get("lineIndex", 0))),
          "word_index": int(word.get("wordIndex", 0)),
          "word_key": key,
          "text": word.get("expectedText", "") or result.text,
          "bases": "".join(result.bases),
          "x": int(word["x"]), "y": int(word["y"]),
          "w": int(word["w"]), "h": int(word["h"]),
          "baseline_y_page": float(line["baseLine"]),
          "expected_paws": result.expected_paws,
          "observed_paws": result.observed_paws,
          "abstain": int(bool(result.abstain)),
      })

      letters_by_paw: Dict[int, List[dict]] = {}
      word_letter_index = 0
      for paw in result.paws:
        meta = (word.get("paws") or [])
        paw_meta = meta[paw.paw_index] if paw.paw_index < len(meta) else {}
        build.paws.append({
            "paw_id": paw_id(page, key, paw.paw_index),
            "word_id": word_id(page, key),
            "page": page,
            "paw_index": paw.paw_index,
            "text": paw_meta.get("text", ""),
            "letters": "".join(paw.letters),
            "x": int(paw_meta.get("x", 0)), "y": int(paw_meta.get("y", 0)),
            "w": int(paw_meta.get("w", 0)), "h": int(paw_meta.get("h", 0)),
            "stroke_width": float(paw.stroke_width),
            "spine_length": float(paw.spine_length),
            "loops": int(paw.loops),
            "alignment_mode": "body-join" if paw.body_join_alignment else "cuts",
            "cut_policy": paw.cut_policy,
            "ok": int(bool(paw.ok)),
            "reason": paw.reason,
            "merged": int(bool(paw.merged)),
            "connected_to_previous_word":
                int(bool(getattr(paw, "connected_to_previous_word", False))),
            "low_confidence": int(bool(paw.low_confidence)),
            "confidence": float(paw.confidence),
            "margin": float(paw.margin),
        })

        rows = extract_letters(page, key, word, result, paw, baseline,
                               word_letter_index, options, build)
        letters_by_paw[paw.paw_index] = rows
        word_letter_index += len(paw.letters)

      build.marks.extend(collect_marks(page, word, components, letters_by_paw,
                                       analysis_root, build.mark_format,
                                       build))
  return build


# The segmenter labels the body of an uncut ligature -2 rather than giving it
# a letter index, because it will not invent a boundary inside a shape it
# deliberately refused to cut. Storing nothing for it is worse than choosing:
# the ink then leaves the index entirely, and 7597 letters -- every لا, في and
# كا -- have no box, no mask and no outline, so a word drawn from the corpus
# has a hole where its ligature should be. The whole ligature is charged to its
# first letter, which is where its entry point already is; the rest of the
# group stays empty so it is counted and drawn exactly once.
FUSED_BODY_LABEL = -2


def fused_group_owner(spans: Sequence[Tuple[float, float, int, float]]
                      ) -> Optional[int]:
  """Index of the letter that should carry the PAW's fused body, if any.

  Only when the PAW holds exactly one fused group: -2 does not say *which*
  group a pixel belongs to, so with two of them there is nothing to separate
  them by and the honest answer is to leave both as they were. Three PAWs in
  the corpus are in that position, against 4479 that are not.
  """
  owner, groups = None, 0
  previous: Optional[Tuple[float, float]] = None
  for index, (entry_s, exit_s, group_size, _) in enumerate(spans):
    if group_size <= 1:
      previous = None
      continue
    if previous != (entry_s, exit_s):
      groups += 1
      owner = index
    previous = (entry_s, exit_s)
  return owner if groups == 1 else None


@dataclass(frozen=True)
class PawInk:
  """A PAW's pixel labels, cropped to the ink they actually cover.

  The segmenter returns label maps the size of the whole padded word, of which
  any one PAW occupies a small corner -- a few hundred thousand ink pixels in
  three or four million. Every per-letter question below (which pixels are
  this letter's, where is its box, how much ink does it hold) used to scan
  that entire rectangle, five hundred times a page. Cropping once per PAW and
  carrying the offset gives the same answers off arrays an order of magnitude
  smaller.

  ``x`` and ``y`` are the crop's origin in padded-canvas coordinates, so word
  space is ``crop + (x, y) - PAD_PX``.
  """
  letter: np.ndarray                    # cropped pixel_letter
  join: Optional[np.ndarray]            # cropped pixel_join, where there is one
  x: int
  y: int


def paw_ink(paw) -> Optional[PawInk]:
  """Crop a PAW's label maps to their ink, or None if it has no labels."""
  if paw.pixel_letter is None:
    return None
  # Every label a letter mask can select -- a letter index, the fused-body -2,
  # or a join's -3 -- is something other than -1, so this box contains all of
  # them. Unattributed ink also survives, which costs nothing and keeps the
  # test to a single comparison.
  ys, xs = np.nonzero(paw.pixel_letter != -1)
  if xs.size == 0:
    return None
  y0, y1 = int(ys.min()), int(ys.max())
  x0, x1 = int(xs.min()), int(xs.max())
  return PawInk(
      letter=paw.pixel_letter[y0:y1 + 1, x0:x1 + 1],
      join=(None if paw.pixel_join is None
            else paw.pixel_join[y0:y1 + 1, x0:x1 + 1]),
      x=x0, y=y0)


def letter_mask(ink: PawInk, index: int, fused_owner: Optional[int] = None
                ) -> Optional[np.ndarray]:
  """One letter's ink in the PAW crop, its left kashida included.

  Under body-join alignment the connector is a state of its own rather than
  ink belonging to either neighbour. For measurement it has to belong to
  somebody: a letter is written together with the stroke that leaves it, and
  that stroke is part of what the letter costs on the line.
  """
  mask = ink.letter == index
  if fused_owner is not None and index == fused_owner:
    mask |= ink.letter == FUSED_BODY_LABEL
  if ink.join is not None:
    mask |= ink.join == index + 1
  return mask if mask.any() else None


def kashida_mask(ink: PawInk, index: int) -> Optional[np.ndarray]:
  """Just the connector leaving this letter, as the segmenter attributed it.

  ``attribute_body_join_pixels`` already decides which pixels are join rather
  than body, so the connector is known exactly here. Folding it into the
  letter mask and then re-deriving it from the letter's outline at query time
  -- which is what the browser used to do -- both costs a second per three
  hundred rows and cannot separate body ink that genuinely touches the
  connector. Keep it.
  """
  if ink.join is None:
    return None
  mask = ink.join == index + 1
  return mask if mask.any() else None


def mask_box(mask: np.ndarray, ink: PawInk
             ) -> Optional[Tuple[int, int, int, int]]:
  """Inclusive (x0, y0, x1, y1) in word space, undoing crop and padding."""
  ys, xs = np.nonzero(mask)
  if xs.size == 0:
    return None
  dx, dy = ink.x - seg.PAD_PX, ink.y - seg.PAD_PX
  return (int(xs.min()) + dx, int(ys.min()) + dy,
          int(xs.max()) + dx, int(ys.max()) + dy)


def extract_letters(page: int, key: str, word: dict, result, paw,
                    baseline: float, word_letter_index: int, options: dict,
                    build: PageBuild) -> List[dict]:
  """Rows and geometry for the letters of one PAW."""
  rows: List[dict] = []
  spans = letter_spans(paw)
  # A merged pair is one blob charged to its first letter, exactly as an
  # uncut ligature is -- but it reaches this point without a segment group
  # to say so, because nothing was ever aligned. Name the owner directly.
  fused_owner = 0 if paw.merged else fused_group_owner(spans)
  labels = paw_ink(paw)
  for index, char in enumerate(paw.letters):
    identifier = letter_id(page, key, paw.paw_index, index)
    form = paw.forms[index] if index < len(paw.forms) else ""
    mask = None if labels is None else letter_mask(labels, index, fused_owner)
    box = (mask_box(mask, labels) if mask is not None else
           (paw.letter_boxes[index] if index < len(paw.letter_boxes) else None))
    ink = (int(mask.sum()) if mask is not None else
           (int(paw.letter_ink[index]) if index < len(paw.letter_ink) else 0))
    entry_s, exit_s, group_size, kashida_s = spans[index]
    entry = spine_point(paw.medial, entry_s)
    exit_ = spine_point(paw.medial, exit_s)

    bbox_x = bbox_y = bbox_w = bbox_h = None
    height = above = below = None
    if box is not None:
      x0, y0, x1, y1 = box
      bbox_x, bbox_y = int(x0), int(y0)
      bbox_w, bbox_h = int(x1 - x0 + 1), int(y1 - y0 + 1)
      height = float(bbox_h)
      above = max(0.0, baseline - y0)
      below = max(0.0, y1 - baseline)

    shape = paw.letter_shape[index] if index < len(paw.letter_shape) else None
    teeth = paw.letter_teeth[index] if index < len(paw.letter_teeth) else (None, None)
    group = seg_logical_group(paw, index)

    row = {
        "letter_id": identifier,
        "paw_id": paw_id(page, key, paw.paw_index),
        "word_id": word_id(page, key),
        "page": page,
        "letter_index": index,
        "word_letter_index": word_letter_index + index,
        "char": char,
        "char_class": shape_class(char, form),
        "form": form,
        "prev_char": (result.bases[word_letter_index + index - 1]
                      if word_letter_index + index > 0
                      and word_letter_index + index - 1 < len(result.bases) else None),
        "next_char": (result.bases[word_letter_index + index + 1]
                      if word_letter_index + index + 1 < len(result.bases) else None),
        "logical_group": group,
        "template": template_for(paw, index),
        "group_size": group_size,
        "bbox_x": bbox_x, "bbox_y": bbox_y, "bbox_w": bbox_w, "bbox_h": bbox_h,
        "page_x": None if bbox_x is None else int(word["x"]) + bbox_x,
        "page_y": None if bbox_y is None else int(word["y"]) + bbox_y,
        "ink_px": ink,
        # A PAW that was never cut has no per-letter span: letter_spans seeds
        # every letter with the whole PAW and nothing overwrites it, so these
        # two would report the entire PAW as this one letter's -- 45.7 stroke
        # widths for the final Kaf of يستخفنك, seven letters wide, which beats
        # any real letter and lands at the top of a width sort. No measurement
        # is the honest answer. bbox_width_px is left alone: it comes from the
        # letter's own mask, not from the span.
        "advance_width_px": (advance_width(entry, exit_)
                             if paw.ok and group_size == 1 else None),
        # For a merged pair the box above is the joint blob's, not this
        # letter's: it spans both PAWs, their tails and whatever they overlap
        # by. Kept as geometry so the outline can be drawn, refused as a
        # measurement so it cannot be compared with a letter that was actually
        # cut. NULL sorts last, which is where an uncut pair belongs.
        "bbox_width_px": (None if bbox_w is None or paw.merged
                          else float(bbox_w)),
        "spine_extent_px": float(exit_s - entry_s) if paw.ok else None,
        "kashida_px": float(kashida_s),
        "height_px": None if paw.merged else height,
        "above_baseline_px": None if paw.merged else above,
        "below_baseline_px": None if paw.merged else below,
        "radius_px": (float(shape["radius"]) * float(paw.stroke_width)
                      if shape and paw.stroke_width else None),
        "teeth_expected": teeth[0],
        "teeth_observed": teeth[1],
        "confidence": float(paw.confidence),
        "low_confidence": int(bool(paw.low_confidence)),
        "ok": int(bool(paw.ok)),
    }
    rows.append(row)
    build.letters.append(row)
    # A merged pair has no spine, so no entry point, and `normalize_page`
    # traces nothing without one -- the outline would be stored as a mask and
    # never drawn. The frame only needs an origin, and the blob supplies one:
    # its own entry edge on the baseline, which is where an RTL letter starts.
    # This reaches `letter_shape.entry_x` but cannot become a measurement: the
    # pair's `exit_x` stays NULL, so the group-span fallback yields NULL, and
    # the only letters that could read it as a neighbour's entry are the pair's
    # own, which have no shape at all.
    shape_entry = entry
    if paw.merged and entry is None and box is not None:
      shape_entry = (float(box[2]), baseline)
    build.shapes.append(letter_geometry(
        labels, mask,
        None if labels is None else kashida_mask(labels, index),
        box, shape_entry, exit_, baseline, identifier, options))
  return rows


def seg_logical_group(paw, index: int) -> Optional[str]:
  for group in paw.segment_groups:
    if (group.letter_end - group.letter_start > 1
        and group.letter_start <= index < group.letter_end):
      return "".join(paw.letters[group.letter_start:group.letter_end])
  return None


def template_for(paw, index: int) -> Optional[str]:
  for group in paw.segment_groups:
    if group.letter_start <= index < group.letter_end and group.template:
      return str(group.template)
  return None


def letter_geometry(labels, full_mask, join_mask, box, entry, exit_,
                    baseline: float, identifier: str, options: dict) -> dict:
  """Trace one letter's mask and keep the mask itself.

  ``full_mask`` covers the whole cropped PAW (see `PawInk`) and already
  includes the letter's kashida; only the letter's own box is stored and
  traced.
  ``join_mask`` is that kashida on its own, traced into a second outline so
  the browser can paint it without re-deriving it.

  Cubics are kept in word-space pixels here; they are mapped into the
  normalized frame only once the page's nominal stroke width is known.
  """
  geometry = {
      "letter_id": identifier,
      "contours": [],
      "kashida_contours": [],
      "mask_rle": None, "mask_w": 0, "mask_h": 0, "mask_x": 0, "mask_y": 0,
      "entry_x": None if entry is None else entry[0],
      "entry_y": None if entry is None else entry[1],
      "exit_x": None if exit_ is None else exit_[0],
      "exit_y": None if exit_ is None else exit_[1],
      "baseline_y": baseline,
  }
  if box is None or full_mask is None:
    return geometry
  x0, y0, x1, y1 = box
  # box is in word space; full_mask is the PAW crop, so undo both offsets.
  dx, dy = seg.PAD_PX - labels.x, seg.PAD_PX - labels.y
  mask = full_mask[y0 + dy:y1 + dy + 1, x0 + dx:x1 + dx + 1]
  if not mask.any():
    return geometry
  geometry["mask_rle"] = encode_mask_rle(mask)
  geometry["mask_w"] = int(mask.shape[1])
  geometry["mask_h"] = int(mask.shape[0])
  geometry["mask_x"] = int(x0)
  geometry["mask_y"] = int(y0)
  if options["trace"]:
    geometry["contours"] = trace_mask(mask, (int(x0), int(y0)), cfg=DEFAULT_FIT)
    if join_mask is not None:
      join_box = mask_box(join_mask, labels)
      if join_box is not None:
        jx0, jy0, jx1, jy1 = join_box
        crop = join_mask[jy0 + dy:jy1 + dy + 1, jx0 + dx:jx1 + dx + 1]
        if crop.any():
          geometry["kashida_contours"] = trace_mask(
              crop, (int(jx0), int(jy0)), cfg=DEFAULT_FIT)
  return geometry


def merge_shards(shards: Sequence[PageBuild]) -> PageBuild:
  """Reassemble the slices of one page, in word order.

  Order is what makes a sharded build identical to an unsharded one: the rows
  come out in the same sequence, and `normalize_page` then runs once over the
  whole page, so the nominal stroke width -- and every measurement scaled by
  it -- is the page's, not a slice's.
  """
  if len(shards) == 1:
    return shards[0]
  merged = PageBuild(page=shards[0].page)
  merged.source_image = shards[0].source_image
  merged.image_w, merged.image_h = shards[0].image_w, shards[0].image_h
  merged.mark_format = shards[0].mark_format
  merged.missing = shards[0].missing
  for shard in shards:
    merged.words.extend(shard.words)
    merged.paws.extend(shard.paws)
    merged.letters.extend(shard.letters)
    merged.shapes.extend(shard.shapes)
    merged.marks.extend(shard.marks)
    merged.observations.extend(shard.observations)
    merged.errors.extend(shard.errors)
    merged.expected_letters += shard.expected_letters
    merged.marks_unindexed += shard.marks_unindexed
    merged.ayah_skipped += shard.ayah_skipped
  return merged


def normalize_page(build: PageBuild) -> dict:
  """Apply the page's nominal stroke width to every measurement it scales."""
  estimate = paw_width.estimate_page_nominal_width(build.observations)
  scale = estimate.get("nominalWidth")
  scale = float(scale) if scale else None

  for row in build.letters:
    row["adv_w_norm"] = normalized(row["advance_width_px"], scale)
    row["bbox_w_norm"] = normalized(row["bbox_width_px"], scale)
    row["extent_norm"] = normalized(row["spine_extent_px"], scale)
    row["kashida_norm"] = normalized(row["kashida_px"], scale)
    row["height_norm"] = normalized(row["height_px"], scale)
    row["above_norm"] = normalized(row["above_baseline_px"], scale)
    row["below_norm"] = normalized(row["below_baseline_px"], scale)
    row["radius_norm"] = normalized(row["radius_px"], scale)

  for row in build.marks:
    row["width_norm"] = normalized(row["w"], scale)
    row["height_norm"] = normalized(row["h"], scale)
    row["area_norm"] = (None if not scale
                        else float(row["area"]) / (scale * scale))

  for shape in build.shapes:
    contours = shape.pop("contours")
    kashida_contours = shape.pop("kashida_contours")
    entry_x = shape["entry_x"]
    shape["norm_scale"] = scale
    shape["contour_count"] = len(contours)
    shape["path_d"] = ""
    shape["kashida_path_d"] = ""
    shape["view_x"] = shape["view_y"] = shape["view_w"] = shape["view_h"] = None
    if not contours or not scale or entry_x is None:
      continue
    frame = LetterFrame(entry_x=entry_x, entry_y=shape["entry_y"],
                        baseline_y=shape["baseline_y"], scale=scale)
    path, box = contours_to_path(contours, frame)
    shape["path_d"] = path
    if box is not None:
      shape["view_x"], shape["view_y"], shape["view_w"], shape["view_h"] = box
    # The same frame, so the connector lands exactly on the letter it left.
    if kashida_contours:
      shape["kashida_path_d"] = contours_to_path(kashida_contours, frame)[0]

  return estimate


# ---------------------------------------------------------------------------
# database
# ---------------------------------------------------------------------------

def insert(connection: sqlite3.Connection, table: str, rows: Sequence[dict],
           run_id: Optional[str] = None) -> None:
  if not rows:
    return
  columns = list(rows[0])
  if run_id is not None:
    columns.append("run_id")
  quoted = ",".join(f'"{column}"' for column in columns)
  placeholders = ",".join("?" * len(columns))
  statement = f"INSERT OR REPLACE INTO {table} ({quoted}) VALUES ({placeholders})"
  connection.executemany(statement, [
      tuple(run_id if column == "run_id" else row.get(column)
            for column in columns)
      for row in rows])


def write_page(connection: sqlite3.Connection, build: PageBuild,
               estimate: dict, run_id: str) -> None:
  page = build.page
  with connection:
    connection.execute(
        "DELETE FROM letter_shape WHERE letter_id IN "
        "(SELECT letter_id FROM letters WHERE page = ?)", (page,))
    for table in ("marks", "letters", "paws", "words"):
      connection.execute(f"DELETE FROM {table} WHERE page = ?", (page,))
    connection.execute("DELETE FROM pages WHERE page = ?", (page,))

    insert(connection, "words", build.words)
    insert(connection, "paws", build.paws)
    insert(connection, "letters", build.letters, run_id)
    insert(connection, "letter_shape", build.shapes)
    insert(connection, "marks", build.marks, run_id)
    connection.execute(
        "INSERT INTO pages (page, source_image, image_w, image_h, "
        "nominal_stroke_px, nominal_stroke_mad, nominal_method, word_count, "
        "paw_count, letter_count, letters_expected, mark_count, mark_format, "
        "marks_unindexed, run_id) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (page, build.source_image, build.image_w, build.image_h,
         estimate.get("nominalWidth"), estimate.get("mad"),
         estimate.get("method"), len(build.words), len(build.paws),
         len(build.letters), build.expected_letters, len(build.marks),
         build.mark_format, build.marks_unindexed, run_id))


# Columns added to schema.sql after the first indexes were built. An existing
# corpus represents hours of segmentation, so it is migrated in place rather
# than being invalidated; SQLite's ADD COLUMN is O(1) and leaves old rows null.
MIGRATIONS = (
    ("pages", "letters_expected", "INTEGER"),
    ("pages", "mark_format", "TEXT"),
    ("pages", "marks_unindexed", "INTEGER"),
    ("marks", "component_count", "INTEGER"),
    ("marks", "grouping", "TEXT"),
    ("letters", "kashida_px", "REAL"),
    ("letters", "kashida_norm", "REAL"),
    ("letter_shape", "kashida_path_d", "TEXT"),
    ("paws", "connected_to_previous_word", "INTEGER"),
)


def migrate(connection: sqlite3.Connection) -> List[str]:
  applied: List[str] = []
  for table, column, kind in MIGRATIONS:
    existing = {row[1] for row in
                connection.execute(f"PRAGMA table_info({table})")}
    if existing and column not in existing:
      connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
      applied.append(f"{table}.{column}")
  if applied:
    connection.commit()
  return applied


def open_database(path: Path) -> sqlite3.Connection:
  path.parent.mkdir(parents=True, exist_ok=True)
  connection = sqlite3.connect(path)
  connection.executescript(SCHEMA.read_text(encoding="utf-8"))
  applied = migrate(connection)
  if applied:
    print(f"migrated existing index: added {', '.join(applied)}")
  return connection


def git_commit() -> str:
  try:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          cwd=str(_HERE), capture_output=True, text=True,
                          check=True).stdout.strip()
  except Exception:
    return ""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_pages(spec: str, analysis_root: str) -> List[int]:
  if spec in ("", "all"):
    root = Path(analysis_root)
    return sorted(int(child.name[4:]) for child in root.glob("page[0-9]*")
                  if child.is_dir() and child.name[4:].isdigit()
                  and (child / "words.json").is_file())
  pages: List[int] = []
  for part in spec.split(","):
    part = part.strip()
    if not part:
      continue
    if "-" in part:
      low, high = part.split("-", 1)
      pages.extend(range(int(low), int(high) + 1))
    else:
      pages.append(int(part))
  return sorted(set(pages))


def worker_threads(parallel: int) -> int:
  """How many threads one worker process may use for a threaded inner loop.

  The only such loop is the segmenter's nearest-neighbour query over a PAW's
  ink (`segment_paw_letters.QUERY_WORKERS`), which is a quarter of its time
  and about six times faster spread across cores. Spread it only as far as
  there are cores left over: with a process per core there are none, and with
  a single process -- the tight rebuild-one-page loop -- there is the whole
  machine. ``parallel`` is how many tasks actually run at once, which is not
  ``--jobs`` when there are fewer tasks than that.
  """
  return max(1, (os.cpu_count() or 1) // max(1, parallel))


def word_ranges(analysis_root: str, page: int, shards: int
                ) -> List[Optional[Tuple[int, int]]]:
  """Contiguous word ranges covering one page, or [None] for the whole page.

  Ranges are as even as the word count allows. A page with fewer words than
  requested slices yields fewer, rather than empty tasks.
  """
  if shards <= 1:
    return [None]
  count = page_word_count(analysis_root, page)
  if count <= 1:
    return [None]
  parts = min(shards, count)
  edges = [count * index // parts for index in range(parts + 1)]
  return [(edges[i], edges[i + 1]) for i in range(parts)]


def shards_per_page(pages: int, jobs: int, options: dict) -> int:
  """How many slices to cut each page into, to fill the pool.

  Rebuilding one page after a segmenter change is the loop this exists for:
  page-level parallelism leaves a single core on it however large `--jobs` is.
  Splitting the page's words instead fills the machine, and `merge_shards`
  plus a single `normalize_page` in the parent make the result identical.

  Slicing is not free: a slice cannot be normalized in its worker, because the
  page scale is not known until every slice is in, so its traced cubics travel
  back to the parent as cubics rather than as path strings. It therefore has
  to earn its place -- required here to at least double how much of the pool
  is busy, which rules out shaving the last couple of idle workers off an
  otherwise full run.

  Left alone under `--limit-words`, whose budget is per page and would be cut
  up along with it.
  """
  if pages <= 0 or jobs <= 1 or options.get("limit_words"):
    return 1
  if pages > jobs // 2:
    return 1
  return -(-jobs // pages)                              # ceil


def init_worker(threads: int) -> None:
  # Runs once per pool process. The environment variables above are already in
  # place via the module import; OpenCV's pool is per-process state that has to
  # be set after it loads.
  cv2.setNumThreads(1)
  seg.QUERY_WORKERS = threads


def page_task(arguments):
  """One unit of pool work: a whole page, or one slice of one.

  A whole page is normalized here, in the worker, so the traced cubics are
  turned into path strings before anything is sent back. A slice cannot be --
  the scale is a property of the page -- so it returns its rows raw and the
  parent normalizes after merging.
  """
  analysis_root, page, options, word_slice = arguments
  start = time.perf_counter()
  build = build_page(analysis_root, page, options, word_slice)
  if word_slice is not None:
    return build, None, time.perf_counter() - start
  return build, normalize_page(build), time.perf_counter() - start


def main() -> int:
  parser = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--analysis", required=True, help="analysis output root")
  parser.add_argument("--out", required=True, help="corpus sqlite path")
  parser.add_argument("--pages", default="all",
                      help="page list: 3, 3-20, 3,7,9, or all")
  parser.add_argument("--jobs", type=int, default=1)
  parser.add_argument("--resume", action="store_true",
                      help="skip pages already in the index, to continue an "
                           "interrupted build")
  parser.add_argument("--no-trace", action="store_true",
                      help="skip outline tracing; masks and metrics only")
  parser.add_argument("--limit-words", type=int, default=0,
                      help="stop after this many words per page (smoke tests)")
  # Off by default: the verse ornaments are the one class of word that is
  # guaranteed to hold nothing searchable, so indexing them only costs build
  # time and pads every failure count with words that were never letters.
  parser.add_argument("--ayah-words", default=False,
                      action=argparse.BooleanOptionalAction,
                      help="also index verse-number ornaments (default: skip)")
  parser.add_argument("--min-confidence", type=float,
                      default=seg.MIN_CUT_CONFIDENCE)
  parser.add_argument("--skeleton-shapes", action="store_true")
  parser.add_argument("--logical-ligatures", action="store_true")
  # On by default so the index agrees with review_paw_segmentation.py, which
  # always runs with it. Without it the segmenter is more permissive: it cuts
  # PAWs the viewer reports as unmatched, and those cuts land in the corpus as
  # ordinary letters, so a row can look segmented while the viewer says it is
  # not. Pass --no-body-join-alignment to index the permissive result instead.
  parser.add_argument("--body-join-alignment", default=True,
                      action=argparse.BooleanOptionalAction)
  parser.add_argument("--cut-policy",
                      choices=("current-end", "midpoint", "next-start", "none"),
                      default="next-start")
  parser.add_argument("--skeleton-templates", default=seg.DEFAULT_TEMPLATE_PATH)
  arguments = parser.parse_args()

  options = {
      "min_confidence": arguments.min_confidence,
      "skeleton_shapes": arguments.skeleton_shapes,
      "logical_ligatures": arguments.logical_ligatures,
      "body_join_alignment": arguments.body_join_alignment,
      "cut_policy": arguments.cut_policy,
      "templates": arguments.skeleton_templates,
      "trace": not arguments.no_trace,
      "limit_words": arguments.limit_words,
      "ayah_words": arguments.ayah_words,
  }
  analysis_root = os.path.abspath(arguments.analysis)
  pages = parse_pages(arguments.pages, analysis_root)
  if not pages:
    print("no pages matched", file=sys.stderr)
    return 1

  connection = open_database(Path(arguments.out))
  run_id = uuid.uuid4().hex[:12]
  connection.execute(
      "INSERT INTO runs (run_id, tool, argv, git_commit, created_at) "
      "VALUES (?,?,?,?,?)",
      (run_id, "build_corpus.py", " ".join(sys.argv), git_commit(),
       dt.datetime.now().isoformat(timespec="seconds")))
  connection.commit()

  kept = 0
  if arguments.resume:
    # write_page commits a page's rows and its pages entry in one transaction,
    # so an entry means that page finished. Anything interrupted mid-page
    # rolled back and is rebuilt.
    done = {row[0] for row in connection.execute("SELECT page FROM pages")}
    remaining = [page for page in pages if page not in done]
    kept = len(pages) - len(remaining)
    pages = remaining
    if kept:
      print(f"resuming: {kept} pages already indexed, {len(pages)} to go")
    if not pages:
      print("nothing left to build")
      connection.close()
      return 0

  # Page-major, so a page's slices arrive together and consecutively from
  # pool.map, which preserves submission order.
  shards = shards_per_page(len(pages), arguments.jobs, options)
  tasks = []
  shard_counts: List[int] = []
  for page in pages:
    ranges = word_ranges(analysis_root, page, shards)
    shard_counts.append(len(ranges))
    tasks.extend((analysis_root, page, options, span) for span in ranges)
  if shards > 1:
    print(f"one page at a time would leave {arguments.jobs - len(pages)} "
          f"of {arguments.jobs} workers idle; splitting each page into "
          f"{shards} slices")

  totals = {"letters": 0, "marks": 0, "words": 0, "errors": 0, "skipped": 0,
            "ayah": 0, "formats": set()}
  started = time.perf_counter()

  def record(build: PageBuild, estimate: dict, elapsed: float) -> None:
    if build.missing:
      totals["skipped"] += 1
      print(f"{page_id(build.page)}  skipped, not analyzed", flush=True)
      return
    write_page(connection, build, estimate, run_id)
    totals["letters"] += len(build.letters)
    totals["marks"] += len(build.marks)
    totals["words"] += len(build.words)
    totals["errors"] += len(build.errors)
    totals["ayah"] += build.ayah_skipped
    for message in build.errors[:5]:
      print(f"  ! {message}", file=sys.stderr)
    width = estimate.get("nominalWidth")
    qalam = f"{width:.1f}px" if width else "n/a"
    missing = build.expected_letters - len(build.letters)
    notes = "".join([
        f"  ({missing} letters unlocated)" if missing else "",
        f"  ({build.marks_unindexed} marks without components)"
        if build.marks_unindexed else "",
        "  [legacy class-grouped marks]"
        if build.mark_format == "class-group" else "",
        f"  ({build.ayah_skipped} ayah words skipped)"
        if build.ayah_skipped else "",
    ])
    totals["formats"].add(build.mark_format)
    print(f"{page_id(build.page)}  words {len(build.words):4d}  "
          f"letters {len(build.letters):5d}  marks {len(build.marks):5d}  "
          f"qalam {qalam:>8}  {elapsed:.1f}s{notes}", flush=True)

  threads = worker_threads(min(arguments.jobs, len(tasks)))

  def collect(results) -> None:
    """Group each page's slices back together, then record the page once."""
    pending: List[PageBuild] = []
    elapsed = 0.0
    counts = iter(shard_counts)
    wanted = next(counts)
    for build, estimate, took in results:
      if wanted == 1:
        record(build, estimate, took)
      else:
        pending.append(build)
        elapsed += took
        if len(pending) == wanted:
          # The sum of the slices, so the printed per-page cost stays
          # comparable with an unsharded run's. It is deliberately not the
          # wall time the page took, which the slices ran concurrently for.
          merged = merge_shards(pending)
          record(merged, normalize_page(merged), elapsed)
          pending, elapsed = [], 0.0
        else:
          continue
      wanted = next(counts, 1)

  if arguments.jobs > 1:
    with concurrent.futures.ProcessPoolExecutor(
        arguments.jobs, initializer=init_worker, initargs=(threads,)) as pool:
      collect(pool.map(page_task, tasks))
  else:
    seg.QUERY_WORKERS = threads
    collect(page_task(task) for task in tasks)

  connection.execute("ANALYZE")
  connection.commit()
  connection.close()

  elapsed = time.perf_counter() - started
  size_mb = Path(arguments.out).stat().st_size / 1e6
  notes = "".join([
      f", {totals['skipped']} not analyzed" if totals["skipped"] else "",
      f", {kept} already indexed" if kept else "",
      f", {totals['ayah']} ayah words skipped" if totals["ayah"] else "",
  ])
  print(f"\n{len(pages) - totals['skipped']} pages built{notes}, "
        f"{totals['words']} words, {totals['letters']} letters, "
        f"{totals['marks']} marks, {totals['errors']} errors "
        f"in {elapsed:.1f}s -> {arguments.out} ({size_mb:.1f} MB)")
  formats = totals["formats"] - {""}
  if formats == {"class-group"}:
    print("marks came from legacy class-grouped masks: an entry can hold "
          "several marks of a class, so per-mark widths are approximate. "
          "Rebuild mushaf_cli and re-run it for per-instance masks.")
  elif len(formats) > 1:
    print(f"WARNING: this analysis directory mixes mark formats {sorted(formats)}; "
          "pages written before the per-instance change are not comparable "
          "with the rest. See pages.mark_format.", file=sys.stderr)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
