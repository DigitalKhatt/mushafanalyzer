"""
Scans every contour on a page for cubics whose control-handle length hit the
max_handle_rel=1.0 clamp in fit_single_cubic_2, and reports each one's
chord length and actual fit error (max_error_against_poly), to check
whether a clamped handle correlates with a real visual defect (large error
on a long chord) or is just the expected/benign shape of a tight, short
sub-arc (small chord, low error).

Usage:
  .venv/bin/python tools/scan_clamped_handles.py output/oldmedina2-analysis/page003 --top 20
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from png_to_ttf_no_potrace import read_bw_grayscale, pad_image, find_contours_with_hierarchy, ensure_orientation
from segement_extrama import fit_closed_by_extrema_segments, _wrap_pts
from bezier_fit import FitConfig, max_error_against_poly


def v_dist(a, b):
  return math.hypot(a[0] - b[0], a[1] - b[1])


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("page_dir")
  ap.add_argument("--ratio_thresh", type=float, default=0.95)
  ap.add_argument("--top", type=int, default=20)
  ap.add_argument("--pad_px", type=int, default=24)
  args = ap.parse_args()

  page_dir = Path(args.page_dir)
  data = json.loads((page_dir / "words.json").read_text(encoding="utf-8"))

  rows = []
  for line in data.get("lines", []):
    for w in line.get("words", []):
      png_path = Path(w["pngPath"])
      if not png_path.exists():
        continue
      bw = read_bw_grayscale(png_path)
      bw = pad_image(bw, args.pad_px)
      nodes = find_contours_with_hierarchy(bw)
      for node in nodes:
        if len(node.pts) < 12:
          continue
        pts_np = ensure_orientation(node.pts.astype(np.float64), want_ccw=not node.is_hole)
        pts_closed = [(float(x), float(y)) for x, y in pts_np.tolist()]
        cfg = FitConfig(tol=8.0, min_arc_height=6.0, reparam_iters=5, straight_seg_tol=9.0)
        cubics, spans, _, _, _ = fit_closed_by_extrema_segments(pts_closed, cfg)
        n = len(pts_closed)
        pts_ring = _wrap_pts(pts_closed)
        for c, sp in zip(cubics, spans):
          p0, p1, p2, p3 = c
          chord = v_dist(p0, p3)
          if chord < 1e-6:
            continue
          ratio = max(v_dist(p1, p0), v_dist(p2, p3)) / chord
          if ratio < args.ratio_thresh:
            continue
          a, b = sp[0] % n, sp[1] % n
          pts_span = pts_ring[a:b+1] if a <= b else pts_ring[a:] + pts_ring[:b+1]
          err = max_error_against_poly(pts_span, c) if len(pts_span) >= 2 else -1
          rows.append((err, chord, ratio, str(w["wordId"]), str(png_path)))

  rows.sort(key=lambda r: r[0], reverse=True)
  print(f"{'err_px':>7} {'chord_px':>9} {'ratio':>6}  wordId  png")
  for err, chord, ratio, wid, png in rows[:args.top]:
    print(f"{err:7.2f} {chord:9.1f} {ratio:6.3f}  {wid:<10}  {png}")

  print(f"\n{len(rows)} clamped (ratio>={args.ratio_thresh}) curves total")
  if rows:
    errs = [r[0] for r in rows]
    chords = [r[1] for r in rows]
    print(f"err: min={min(errs):.2f} median={sorted(errs)[len(errs)//2]:.2f} max={max(errs):.2f}")
    print(f"chord: min={min(chords):.1f} median={sorted(chords)[len(chords)//2]:.1f} max={max(chords):.1f}")
    n_over_tol = sum(1 for e in errs if e > 8.0)
    print(f"clamped curves that ALSO exceed tol=8px: {n_over_tol}/{len(rows)}")


if __name__ == "__main__":
  main()
