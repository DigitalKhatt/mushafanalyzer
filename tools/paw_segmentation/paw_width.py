"""Optional local/page stroke-width measurements for PAW segmentation.

The segmenter keeps its PAW-wide stroke width as the authoritative scale.
This module contains the experimental measurements that describe local and
page-wide width variation, so they can be enabled, disabled, or revised
without mixing them into the medial-axis and cut-score implementation.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks


# Page-level nominal-width diagnostics. These values are deliberately used
# only to measure the page; the existing per-PAW scale still drives the
# segmentation thresholds. A clean horizontal spine sample is a better proxy
# for the full qalam width than an arbitrary PAW median.
NOMINAL_HORIZONTAL_MIN = 0.88  # |tangent.x| / |tangent|; nearly horizontal
NOMINAL_BASELINE_TOL_W = 1.00   # keep samples close to the writing line
NOMINAL_BRANCH_EXCLUDE_W = 0.25 # leave a small gap around graph junctions
NOMINAL_SAMPLE_STEP_W = 0.10    # resample radii uniformly in spine arclength
NOMINAL_MIN_SAMPLES = 5         # minimum samples for one PAW observation
NOMINAL_MAD_SIGMA = 3.0         # robust page-level outlier rejection

LOCAL_WIDTH_SMOOTH_W = 0.75     # suppress one-sample neck/junction noise
LOCAL_ENTRY_MAX_SCALE = 0.95    # only add a local entry below the PAW scale
BRANCH_LOCAL_MIN_SCALE = 0.95   # never let a narrow local neck lower the floor
BRANCH_LOCAL_MAX_SCALE = 1.15   # allow a genuinely wide local join to be stricter


def horizontal_nominal_width_samples(medial, baseline_y: float,
                                     reference_width: float) -> np.ndarray:
  """Collect clean-ish full-width samples from a PAW's spine.

  The current ``2 * median(medial radius)`` is a useful per-PAW scale, but a
  long oblique or bowl-shaped body can dominate it. For the page reference we
  instead sample the spine uniformly in arclength and keep only samples that
  are nearly horizontal, near the writing line, and away from graph
  junctions. The result is diagnostic evidence only; it does not change the
  segmentation scale.

  ``2 * radius`` is measured normal to the spine, so unlike a vertical pixel
  extent it is not reduced merely by the stroke's obliqueness.
  """
  spine = np.asarray(medial.spine, dtype=int)
  spine_s = np.asarray(medial.spine_s, dtype=float)
  if spine.size < 3 or spine_s.size != spine.size or spine_s[-1] <= 0.0:
    return np.empty(0, dtype=float)

  scale = max(float(reference_width), 1.0)
  step = max(1.0, NOMINAL_SAMPLE_STEP_W * scale)
  grid = np.arange(0.0, spine_s[-1] + 0.5 * step, step)
  if grid.size < 3:
    return np.empty(0, dtype=float)

  vertices = medial.vertices[spine]
  x = np.interp(grid, spine_s, vertices[:, 0])
  y = np.interp(grid, spine_s, vertices[:, 1])
  width = np.interp(grid, spine_s, 2.0 * medial.radii[spine])
  dx = np.gradient(x, grid)
  dy = np.gradient(y, grid)
  tangent_norm = np.hypot(dx, dy)
  horizontal = (tangent_norm > 1e-9
                ) & (np.abs(dx) / np.maximum(tangent_norm, 1e-9)
                    >= NOMINAL_HORIZONTAL_MIN)
  near_baseline = np.abs(y - float(baseline_y)) <= (
      NOMINAL_BASELINE_TOL_W * scale)

  # Exclude both surviving degree-3 vertices and the junctions where a short
  # branch was pruned. A small arclength gap prevents the thickest part of a
  # join from becoming the page's nominal single-stroke width.
  degree: Dict[int, int] = {}
  for a, b in medial.edges:
    degree[a] = degree.get(a, 0) + 1
    degree[b] = degree.get(b, 0) + 1
  pruned_vertices = {branch.anchor for branch in medial.pruned_branches}
  junction_s = [
      float(spine_s[i]) for i, vertex in enumerate(spine)
      if degree.get(int(vertex), 0) >= 3 or int(vertex) in pruned_vertices
  ]
  away_from_junction = np.ones(grid.shape, dtype=bool)
  if junction_s:
    distance = np.min(np.abs(grid[:, None] - np.asarray(junction_s)[None, :]),
                      axis=1)
    away_from_junction = distance > NOMINAL_BRANCH_EXCLUDE_W * scale

  interior = (grid >= 0.5 * step) & (grid <= spine_s[-1] - 0.5 * step)
  valid = (horizontal & near_baseline & away_from_junction & interior
           & np.isfinite(width) & (width > 0.0))
  return width[valid]


def local_width_profile(profile: dict, reference_width: float) -> np.ndarray:
  """Return a smoothed normal-width estimate at every profile sample.

  ``2 * profile['r']`` is the local width of the medial core, but a single
  narrow neck or a graph junction is not a useful denominator for a body
  entry. A short median filter gives the nearby connector/body scale while
  retaining real calligraphic variation. This is intentionally not clipped
  to the PAW-wide reference: tight ligatures are allowed to be narrower.
  """
  raw = 2.0 * np.asarray(profile["r"], dtype=float)
  if raw.size == 0:
    return raw
  ds = max(float(profile.get("ds", 1.0)), 1e-6)
  scale = max(float(reference_width), 1.0)
  size = max(3, int(round(LOCAL_WIDTH_SMOOTH_W * scale / ds)))
  if size % 2 == 0:
    size += 1
  if size >= raw.size:
    size = raw.size if raw.size % 2 else raw.size - 1
  if size < 3:
    return np.maximum(raw, 1e-6)
  smoothed = median_filter(raw, size=size, mode="nearest")
  return np.maximum(smoothed, 1e-6)


def boundary_trace(profile: dict, stroke_width: float,
                   prominence_w: float = 0.05,
                   separation_w: float = 0.4) -> dict:
  """Trace the center and upper boundary of every normal cross-section.

  A small tooth may not create a medial-axis branch. The upper endpoint of
  the cut chord still moves upward, however, so its locally detrended signal
  provides a topology-independent tooth diagnostic.
  The default prominence is intentionally modest: this is a review signal,
  not a production segmentation threshold.
  """
  chords = np.asarray(profile.get("chords", []), dtype=float)
  grid = np.asarray(profile.get("grid", []), dtype=float)
  if chords.ndim != 3 or chords.shape[1:] != (2, 2) or grid.size != len(chords):
    empty = np.empty((0, 2), dtype=float)
    return {
        "center": empty,
        "upper": empty,
        "trend": np.empty(0, dtype=float),
        "signal": np.empty(0, dtype=float),
        "indices": np.empty(0, dtype=int),
        "positions": np.empty(0, dtype=float),
    }

  center = chords.mean(axis=1)
  upper_index = np.argmin(chords[:, :, 1], axis=1)
  upper = chords[np.arange(len(chords)), upper_index]
  upper_y = upper[:, 1]

  ds = max(float(profile.get("ds", 1.0)), 1e-6)
  scale = max(float(stroke_width), 1.0)
  window = max(3, int(round(2.0 * scale / ds)))
  if window % 2 == 0:
    window += 1
  if window >= upper_y.size:
    window = upper_y.size if upper_y.size % 2 else upper_y.size - 1
  trend = (median_filter(upper_y, size=window, mode="nearest")
           if window >= 3 else upper_y.copy())
  signal = trend - upper_y
  indices, _ = find_peaks(
      signal,
      prominence=max(0.0, float(prominence_w)) * scale,
      distance=max(1, int(round(float(separation_w) * scale / ds))),
  )
  return {
      "center": center,
      "upper": upper,
      "trend": trend,
      "signal": signal,
      "indices": indices,
      "positions": grid[indices],
  }


def page_width_observations(result) -> List[dict]:
  """Return compact per-PAW width observations for page aggregation."""
  observations = []
  baseline_y = float(getattr(result, "baseline_y", 0.0))
  for pr in result.paws:
    stroke_width = float(getattr(pr, "stroke_width", 0.0))
    medial = getattr(pr, "medial", None)
    samples = (horizontal_nominal_width_samples(
        medial, baseline_y, stroke_width)
               if medial is not None and stroke_width > 0.0
               else np.empty(0, dtype=float))
    observations.append({
        "wordId": result.word_id,
        "paw": pr.paw_index,
        "letters": "".join(pr.letters),
        "strokeWidth": stroke_width,
        "horizontalWidth": float(np.median(samples)) if samples.size else None,
        "horizontalSamples": int(samples.size),
    })
  return observations


def estimate_page_nominal_width(observations: Sequence[dict]) -> dict:
  """Estimate a robust page reference from equal-weight PAW observations.

  Each PAW contributes one median of its clean horizontal samples, so a long
  PAW cannot dominate merely by having more spine vertices. The existing
  per-PAW median is retained as a fallback when a page has no qualifying
  horizontal observations.
  """
  horizontal = [
      float(item["horizontalWidth"])
      for item in observations
      if item.get("horizontalWidth") is not None
      and int(item.get("horizontalSamples", 0)) >= NOMINAL_MIN_SAMPLES
      and float(item["horizontalWidth"]) > 0.0
  ]
  fallback = [
      float(item["strokeWidth"])
      for item in observations
      if float(item.get("strokeWidth", 0.0)) > 0.0
  ]
  values = np.asarray(horizontal or fallback, dtype=float)
  method = "horizontal_baseline_paw_medians" if horizontal else (
      "paw_spine_medians_fallback" if fallback else "unavailable")
  if values.size == 0:
    return {
        "nominalWidth": None,
        "method": method,
        "pawCount": 0,
        "pixelSampleCount": 0,
        "inlierPawCount": 0,
        "outlierPawCount": 0,
        "rawMedian": None,
        "mad": None,
        "q10": None, "q25": None, "q50": None,
        "q75": None, "q90": None,
    }

  raw_median = float(np.median(values))
  mad = float(np.median(np.abs(values - raw_median)))
  # The relative floor prevents a zero/very small MAD from making the
  # estimate reject every naturally varying calligraphic width.
  tolerance = max(NOMINAL_MAD_SIGMA * 1.4826 * mad,
                 0.08 * max(raw_median, 1.0))
  inlier = np.abs(values - raw_median) <= tolerance
  if not inlier.any():
    inlier = np.ones(values.shape, dtype=bool)
  nominal = float(np.median(values[inlier]))
  pixel_count = int(sum(
      int(item.get("horizontalSamples", 0)) for item in observations
      if item.get("horizontalWidth") is not None
      and int(item.get("horizontalSamples", 0)) >= NOMINAL_MIN_SAMPLES
  ))

  return {
      "nominalWidth": nominal,
      "method": method,
      "pawCount": int(values.size),
      "pixelSampleCount": pixel_count,
      "inlierPawCount": int(inlier.sum()),
      "outlierPawCount": int((~inlier).sum()),
      "rawMedian": raw_median,
      "mad": mad,
      "q10": float(np.quantile(values, 0.10)),
      "q25": float(np.quantile(values, 0.25)),
      "q50": float(np.quantile(values, 0.50)),
      "q75": float(np.quantile(values, 0.75)),
      "q90": float(np.quantile(values, 0.90)),
  }


def annotate_page_width(payload: dict, estimate: dict) -> None:
  """Attach page-scale diagnostics to one serialized word result."""
  width = estimate.get("nominalWidth")
  payload["pageNominalWidth"] = round(width, 2) if width else None
  payload["pageNominalWidthMethod"] = estimate.get("method")
  payload["pageNominalWidthPawCount"] = estimate.get("pawCount", 0)
  if not width:
    for paw in payload.get("paws", []):
      paw["strokeWidthRatio"] = None
      paw["horizontalWidthRatio"] = None
    return
  for paw in payload.get("paws", []):
    stroke = float(paw.get("strokeWidth", 0.0))
    horizontal = paw.get("nominalWidthSampleMedian")
    paw["strokeWidthRatio"] = round(stroke / width, 3) if stroke else None
    paw["horizontalWidthRatio"] = (
        round(float(horizontal) / width, 3)
        if horizontal is not None else None)


def collect_page_diagnostics(analysis_root: str, page: int) -> dict:
  """Segment one analyzed page and return width-only diagnostic data.

  The segmenter is imported lazily so this module remains a standalone CLI
  and ``segment_paw_letters.py`` does not need to import it back.
  """
  import segment_paw_letters as segmenter

  page_dir = Path(analysis_root) / f"page{page:03d}"
  with (page_dir / "words.json").open(encoding="utf-8") as fh:
    page_json = json.load(fh)

  observations = []
  for line in page_json["lines"]:
    for word in line["words"]:
      body_path = page_dir / "subwordimgs" / f"{word['wordId']}_body.png"
      if not body_path.is_file():
        continue
      result = segmenter.segment_word(
          str(body_path),
          word.get("expectedText", ""),
          float(line["baseLine"]) - float(word["y"]),
          want_pixels=False,
          paws_meta=word.get("paws"),
          page_dir=str(page_dir))
      result.word_id = word["wordId"]
      observations.extend(page_width_observations(result))

  estimate = estimate_page_nominal_width(observations)
  return {
      "page": page,
      **estimate,
      "observations": observations,
  }


def main() -> int:
  parser = argparse.ArgumentParser(
      description="Generate page-level PAW stroke-width diagnostics.")
  parser.add_argument("--analysis", required=True,
                      help="analysis output root containing pageNNN/")
  parser.add_argument("--page", type=int, required=True)
  parser.add_argument(
      "--out", default="output/page_width_diagnostics.json",
      help="output JSON path")
  args = parser.parse_args()

  diagnostics = collect_page_diagnostics(args.analysis, args.page)
  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  with out_path.open("w", encoding="utf-8") as fh:
    json.dump(diagnostics, fh, ensure_ascii=False, indent=2)

  estimate = diagnostics
  nominal = estimate.get("nominalWidth")
  print(f"PAWs: {len(estimate['observations'])}")
  print(f"nominal width: {nominal:.2f}" if nominal else
        "nominal width: unavailable")
  print(f"wrote: {os.path.abspath(out_path)}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
