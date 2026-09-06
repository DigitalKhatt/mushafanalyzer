#!/usr/bin/env python3
"""Bootstrap per-(letter, joining form) shape priors from the segmenter's own
confident output -- Fix 1b in the session's segmentation work.

The width-uniformity prior in segment_paw_letters.py cannot tell a genuine
letter from a plausible-looking wrong one: on a flat connector (a shoulder
join, e.g. the yeh before a final meem) every cut position scores alike
geometrically, so only knowing what the *claimed letter* should look like
can decide it. TOOTH_COUNT (Fix 1a) supplies that for a handful of letters
from script knowledge alone; this script estimates it for every letter from
data, using no external labels:

  1. Segment every word with whatever shape priors currently exist (none, on
     the first iteration).
  2. Keep only ACCEPTED letter instances: ok, not merged, not low-confidence.
     Confidence and margin are computed *before* any shape prior exists (see
     segment_paw_letters.py), so this selection is not circular -- the first
     iteration's training set owes nothing to what it's about to produce.
  3. Group by "letter|form"; also pool into "letter|*" across forms, for
     letters too sparse to support a per-form estimate.
  4. Fit a robust per-feature median/MAD over {extent, radius, teeth}
     (segment_paw_letters.SHAPE_FEATURES) and write it out.
  5. Optionally repeat: re-segment with the fitted priors, re-fit from the
     (hopefully larger, now-disambiguated) accepted set.

Usage:

  tools/paw_segmentation/train_letter_shape_priors.py \
      --analysis output/oldmedina2-analysis --page 3 \
      --out output/paw-segmentation/letter_shape_priors.json --iterations 3
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np

import segment_paw_letters as S


def fit_priors(instances: Dict[str, Dict[str, List[float]]]) -> dict:
  """instances: class_key -> feature -> list of observed values."""
  priors = {}
  for key, feats in instances.items():
    n = len(next(iter(feats.values()), []))
    if n == 0:
      continue
    cls = {"n": n}
    for feat in S.SHAPE_FEATURES:
      vals = np.asarray(feats[feat], dtype=float)
      median = float(np.median(vals))
      mad = float(np.median(np.abs(vals - median)))
      cls[feat] = {"median": median, "mad": mad}
    priors[key] = cls
  return priors


def collect_instances(page_dir: str, page_json: dict, shape_priors) -> tuple:
  """Runs the segmenter once over the page. Returns (instances, stats)."""
  instances: Dict[str, Dict[str, List[float]]] = defaultdict(
      lambda: {feat: [] for feat in S.SHAPE_FEATURES})
  n_words = n_paws = n_multi = n_accepted_paws = n_accepted_letters = 0

  for line, word in S.iter_words(page_json):
    result = S.run_one(page_dir, line, word, out_dir="", write_overlay=False,
                       shape_priors=shape_priors)
    if result is None:
      continue
    n_words += 1
    for pr in result.paws:
      if not pr.letters:
        continue
      n_paws += 1
      if len(pr.letters) < 2:
        continue
      n_multi += 1
      accepted = pr.ok and not pr.merged and not pr.low_confidence
      if not accepted:
        continue
      n_accepted_paws += 1
      for letter, form, shape in zip(pr.letters, pr.forms, pr.letter_shape):
        for key in (f"{letter}|{form}", f"{letter}|*"):
          for feat in S.SHAPE_FEATURES:
            instances[key][feat].append(shape[feat])
        n_accepted_letters += 1

  stats = dict(words=n_words, paws=n_paws, multiLetterPaws=n_multi,
              acceptedPaws=n_accepted_paws, acceptedLetters=n_accepted_letters)
  return instances, stats


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--analysis", required=True)
  ap.add_argument("--page", type=int, action="append", required=True,
                  help="repeatable; e.g. --page 3 --page 5")
  ap.add_argument("--out", required=True, help="priors JSON to write")
  ap.add_argument("--iterations", type=int, default=3,
                  help="run-fit-rerun cycles; 1 = bootstrap only, no reuse")
  ap.add_argument("--seed-priors", help="existing priors JSON to start from "
                  "instead of an empty prior (iteration 0)")
  args = ap.parse_args()

  pages = [S.load_page(args.analysis, p) for p in args.page]

  shape_priors = None
  if args.seed_priors:
    with open(args.seed_priors, encoding="utf-8") as fh:
      shape_priors = json.load(fh)

  os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
  prev_accepted = None
  for it in range(args.iterations):
    merged: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {feat: [] for feat in S.SHAPE_FEATURES})
    totals = dict(words=0, paws=0, multiLetterPaws=0, acceptedPaws=0,
                 acceptedLetters=0)
    for page_dir, page_json in pages:
      instances, stats = collect_instances(page_dir, page_json, shape_priors)
      for key, feats in instances.items():
        for feat in S.SHAPE_FEATURES:
          merged[key][feat].extend(feats[feat])
      for k in totals:
        totals[k] += stats[k]

    shape_priors = fit_priors(merged)
    per_form = sum(1 for k, c in shape_priors.items()
                  if "|*" not in k and c["n"] >= S.MIN_CLASS_SAMPLES)
    pooled = sum(1 for k, c in shape_priors.items()
                if k.endswith("|*") and c["n"] >= S.MIN_CLASS_SAMPLES)
    delta = "" if prev_accepted is None else f" ({totals['acceptedPaws']-prev_accepted:+d})"
    print(f"iteration {it}: {totals['multiLetterPaws']} multi-letter PAWs, "
          f"{totals['acceptedPaws']} accepted{delta}, "
          f"{totals['acceptedLetters']} accepted letters -- "
          f"{per_form} (letter,form) classes and {pooled} pooled classes "
          f"reach {S.MIN_CLASS_SAMPLES}+ samples")
    prev_accepted = totals["acceptedPaws"]

    with open(args.out, "w", encoding="utf-8") as fh:
      json.dump(shape_priors, fh, ensure_ascii=False, indent=2, sort_keys=True)

  print(f"wrote {args.out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
