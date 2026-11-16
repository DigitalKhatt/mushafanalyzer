"""
Diagnostic pass over the No-PoTrace tracing pipeline for one page.

For every word glyph, re-traces it (reusing the exact production call chain)
and computes cheap per-contour red-flag metrics that correlate with visible
tracing defects, without requiring a human to scan every glyph:

  - n_curves            : cubic count in the contour (very high => likely zigzag/noise)
  - max_handle_ratio     : longest control-handle length / chord length of its curve
                            (very large => overshoot/cusp risk)
  - min_curve_chord      : shortest chord among curves (near-zero => degenerate/spike)
  - max_turn_deg         : largest turning angle between consecutive curve tangents
                            at a join (large + non-corner => faceted/kinked join)
  - self_intersections   : approximate self-intersection count of the sampled
                            outline (bowls/loops that cross themselves)

Words are ranked by a combined suspicion score so the worst offenders can be
reviewed first (e.g. in VectorFitViewer or by cropping their PNG + overlay).

Usage:
  uv run tools/contour_metrics_report.py output/oldmedina2-analysis/page003 --top 40
"""
import argparse
import itertools
import json
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np

from png_to_ttf_no_potrace import (
    read_bw_grayscale,
    pad_image,
    find_contours_with_hierarchy,
    contour_to_beziers,
)
from bezier_fit import FitConfig

Point = Tuple[float, float]
Cubic = Tuple[Point, Point, Point, Point]


def bez3(p0, p1, p2, p3, t):
  u = 1.0 - t
  b0, b1, b2, b3 = u*u*u, 3*u*u*t, 3*u*t*t, t*t*t
  return (p0[0]*b0+p1[0]*b1+p2[0]*b2+p3[0]*b3,
          p0[1]*b0+p1[1]*b1+p2[1]*b2+p3[1]*b3)


def sample_cubic(c: Cubic, n=12):
  return [bez3(*c, t) for t in np.linspace(0, 1, n)]


def seg_intersect(p1, p2, p3, p4):
  def cross(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
  d1 = cross(p3, p4, p1)
  d2 = cross(p3, p4, p2)
  d3 = cross(p1, p2, p3)
  d4 = cross(p1, p2, p4)
  if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
     ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
    return True
  return False


def count_self_intersections(poly: List[Point]) -> int:
  n = len(poly)
  if n < 8:
    return 0
  count = 0
  # skip adjacent segments; O(n^2) is fine for per-glyph polygons (~small after sampling)
  for i in range(n):
    a1, a2 = poly[i], poly[(i+1) % n]
    for j in range(i+2, n):
      if i == 0 and j == n-1:
        continue
      b1, b2 = poly[j], poly[(j+1) % n]
      if seg_intersect(a1, a2, b1, b2):
        count += 1
  return count


def angle_deg(u, v):
  nu, nv = math.hypot(*u), math.hypot(*v)
  if nu < 1e-9 or nv < 1e-9:
    return 0.0
  d = (u[0]*v[0]+u[1]*v[1]) / (nu*nv)
  return math.degrees(math.acos(max(-1.0, min(1.0, d))))


def contour_metrics(cubics: List[Cubic]) -> dict:
  if not cubics:
    return {}
  handle_ratios = []
  chords = []
  for (p0, p1, p2, p3) in cubics:
    chord = math.hypot(p3[0]-p0[0], p3[1]-p0[1])
    h1 = math.hypot(p1[0]-p0[0], p1[1]-p0[1])
    h2 = math.hypot(p2[0]-p3[0], p2[1]-p3[1])
    chords.append(chord)
    if chord > 1e-6:
      handle_ratios.append(max(h1, h2) / chord)

  max_turn = 0.0
  m = len(cubics)
  for i in range(m):
    c_prev = cubics[i-1]
    c_cur = cubics[i]
    t_in = (c_prev[3][0]-c_prev[2][0], c_prev[3][1]-c_prev[2][1])
    t_out = (c_cur[1][0]-c_cur[0][0], c_cur[1][1]-c_cur[0][1])
    max_turn = max(max_turn, angle_deg(t_in, t_out))

  poly = list(itertools.chain.from_iterable(sample_cubic(c, 10) for c in cubics))
  n_self_int = count_self_intersections(poly)

  return {
      "n_curves": len(cubics),
      "max_handle_ratio": max(handle_ratios) if handle_ratios else 0.0,
      "min_curve_chord": min(chords) if chords else 0.0,
      "max_turn_deg": max_turn,
      "self_intersections": n_self_int,
  }


def score(m: dict) -> float:
  if not m:
    return 0.0
  s = 0.0
  s += max(0, m["n_curves"] - 25) * 2.0
  # max_handle_rel=1.0 in fit_single_cubic_2 clamps ratio to <=1.0, so a
  # threshold above 1.0 here can never fire. A clamped ratio on its own is
  # usually benign (a short chord legitimately needing a long relative
  # handle to bend within a tight span) -- verified page-wide on page003:
  # 50/50 clamped curves stayed within fit tolerance. Keep this as a weak
  # signal only, not a defect indicator by itself.
  s += max(0, m["max_handle_ratio"] - 0.95) * 3.0
  s += max(0, 30 - m["max_turn_deg"]) * 0.0  # placeholder (kinks handled below)
  if m["max_turn_deg"] > 60:
    s += (m["max_turn_deg"] - 60) * 1.5
  s += m["self_intersections"] * 40.0
  if 0 < m["min_curve_chord"] < 2.0:
    s += (2.0 - m["min_curve_chord"]) * 20.0
  return s


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("page_dir", type=str, help="e.g. output/oldmedina2-analysis/page003")
  ap.add_argument("--top", type=int, default=30)
  ap.add_argument("--fit_err_px", type=float, default=8)
  ap.add_argument("--min_arc_height", type=float, default=6)
  ap.add_argument("--reparam_iters", type=int, default=5)
  ap.add_argument("--max_fit_error", type=float, default=9)
  ap.add_argument("--pad_px", type=int, default=24)
  args = ap.parse_args()

  page_dir = Path(args.page_dir)
  json_words = page_dir / "words.json"
  data = json.loads(json_words.read_text(encoding="utf-8"))

  cfg = FitConfig(
      tol=args.fit_err_px,
      min_arc_height=args.min_arc_height,
      reparam_iters=args.reparam_iters,
      straight_seg_tol=args.max_fit_error,
      retry_without_inflection_check=True,
      use_naive_split_only=False,
  )

  results = []

  for line in data.get("lines", []):
    for w in line.get("words", []):
      png_path = Path(w["pngPath"])
      wid = str(w["wordId"])
      if not png_path.exists():
        continue
      bw = read_bw_grayscale(png_path)
      bw = pad_image(bw, args.pad_px)
      nodes = find_contours_with_hierarchy(bw)
      word_worst = {"n_curves": 0, "max_handle_ratio": 0.0,
                    "min_curve_chord": 1e9, "max_turn_deg": 0.0,
                    "self_intersections": 0}
      total_score = 0.0
      n_contours = 0
      for n in nodes:
        if len(n.pts) < 12:
          continue
        bez = contour_to_beziers(
            pts_np=n.pts, is_hole=n.is_hole,
            cfg=cfg,
        )
        if not bez.cubics:
          continue
        n_contours += 1
        m = contour_metrics(bez.cubics)
        total_score += score(m)
        word_worst["n_curves"] = max(word_worst["n_curves"], m["n_curves"])
        word_worst["max_handle_ratio"] = max(word_worst["max_handle_ratio"], m["max_handle_ratio"])
        word_worst["min_curve_chord"] = min(word_worst["min_curve_chord"], m["min_curve_chord"])
        word_worst["max_turn_deg"] = max(word_worst["max_turn_deg"], m["max_turn_deg"])
        word_worst["self_intersections"] += m["self_intersections"]

      if n_contours == 0:
        continue
      results.append((total_score, wid, str(png_path), n_contours, word_worst))

  results.sort(key=lambda r: r[0], reverse=True)

  print(f"{'score':>8} {'wordId':<10} {'contours':>8} {'curves':>7} {'handle':>7} "
        f"{'minchord':>9} {'turn':>6} {'selfint':>8}  png")
  for total_score, wid, png_path, n_contours, wm in results[:args.top]:
    print(f"{total_score:8.1f} {wid:<10} {n_contours:8d} {wm['n_curves']:7d} "
          f"{wm['max_handle_ratio']:7.2f} {wm['min_curve_chord']:9.2f} "
          f"{wm['max_turn_deg']:6.1f} {wm['self_intersections']:8d}  {png_path}")

  print(f"\nTotal words analyzed: {len(results)}")


if __name__ == "__main__":
  main()
