"""
Before/after demo for the "half rectangle" elongated-orthogonal-extrema-pair
fix in segement_extrama.py (find_elongated_orthogonal_split).

Renders three states of the same tall stroke (l00_w00.png, contour index 4):
  A) naive       -- no chord-error floor check, no elongated-pair split
                    (the original bug: single bowed cubic, ~60px error)
  B) shipped     -- today's production behavior: chord-error floor check on
                    (from the earlier session fix), elongated-pair split off
                    (silently recovers via generic recursive splitting)
  C) new fix     -- elongated-pair split on: deliberately splits at the
                    geometric elbow instead of relying on generic recursion

Usage:
  .venv/bin/python tools/demo_elongated_split_fix.py
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from png_to_ttf_no_potrace import read_bw_grayscale, pad_image, find_contours_with_hierarchy, ensure_orientation
from segement_extrama import (
    detect_browse_extrema_closed_typed, extract_arc_points_closed,
    tangent_from_extrema_pair_typed, fit_closed_by_extrema_segments, _wrap_pts,
)
from bezier_fit import FitConfig, fit_cubics_open_with_spans, max_error_against_poly

PNG_PATH = "output/oldmedina2-analysis/page003/wordimgs/l00_w00.png"
CONTOUR_INDEX = 5
PARAMS = dict(tol=8.0, min_arc_height=6.0, reparam_iters=5, straight_seg_tol=9.0)
XLIM = (500, 800)
YLIM = (1000, 200)


def bez3(p0, p1, p2, p3, t):
  u = 1.0 - t
  b0, b1, b2, b3 = u*u*u, 3*u*u*t, 3*u*t*t, t*t*t
  return (p0[0]*b0+p1[0]*b1+p2[0]*b2+p3[0]*b3,
          p0[1]*b0+p1[1]*b1+p2[1]*b2+p3[1]*b3)


def sample_cubic(c, n=60):
  return np.array([bez3(*c, t) for t in np.linspace(0, 1, n)])


def gather_span_points(pts_ring, i0, i1):
  n = len(pts_ring)
  if i1 - i0 >= n:
    return None
  a, b = i0 % n, i1 % n
  if a <= b:
    return pts_ring[a:b+1]
  return pts_ring[a:] + pts_ring[:b+1]


def load_contour():
  bw = read_bw_grayscale(PNG_PATH)
  bw = pad_image(bw, 24)
  nodes = find_contours_with_hierarchy(bw)
  node = nodes[CONTOUR_INDEX]
  pts_np = ensure_orientation(node.pts.astype(np.float64), want_ccw=not node.is_hole)
  pts_closed = [(float(x), float(y)) for x, y in pts_np.tolist()]
  return bw, pts_closed


def render(ax, bw, pts_closed, cubics, spans, title):
  ax.imshow(bw, cmap="gray", alpha=0.35, origin="upper")
  ax.set_aspect("equal")
  ax.invert_yaxis()
  pts_ring = _wrap_pts(pts_closed)

  worst_err = 0.0
  for c, sp in zip(cubics, spans):
    pts_span = gather_span_points(pts_ring, sp[0], sp[1])
    err = max_error_against_poly(pts_span, c) if pts_span and len(pts_span) >= 2 else -1.0
    worst_err = max(worst_err, err)
    color = "cyan" if err <= PARAMS["tol"] else ("orange" if err <= 1.5*PARAMS["tol"] else "red")
    pts_curve = sample_cubic(c, 60)
    ax.plot(pts_curve[:, 0], pts_curve[:, 1], "-", color=color, linewidth=2.0)
    ax.plot([c[0][0], c[1][0]], [c[0][1], c[1][1]], "-", color="yellow", linewidth=0.6, alpha=0.8)
    ax.plot([c[3][0], c[2][0]], [c[3][1], c[2][1]], "-", color="yellow", linewidth=0.6, alpha=0.8)
    ax.plot(c[0][0], c[0][1], "o", color="lime", markersize=4)

  ax.set_xlim(XLIM)
  ax.set_ylim(YLIM)
  ax.set_title(f"{title}\nn_curves={len(cubics)}  worst_err={worst_err:.2f}px (tol={PARAMS['tol']}px)")
  return worst_err


def main():
  bw, pts_closed = load_contour()
  pts = _wrap_pts(pts_closed)
  n = len(pts)

  # --- State A: naive (reproduce the original arc-level bug directly) ---
  extrema_typed = detect_browse_extrema_closed_typed(pts, min_arc_height=PARAMS["min_arc_height"])
  a, a_type = extrema_typed[1]   # (75, 'xmin')
  b, b_type = extrema_typed[2]   # (716, 'ymax')
  arc_pts, i0 = extract_arc_points_closed(pts_closed, a, b)
  t0, t1 = tangent_from_extrema_pair_typed(pts_closed, a, a_type, b, b_type, arc_pts)
  naive_cfg = FitConfig(
      tol=PARAMS["tol"], reparam_iters=PARAMS["reparam_iters"],
  )
  naive_curves, naive_spans = fit_cubics_open_with_spans(
      arc_pts, naive_cfg, start_tangent=t0, end_tangent=t1, i0=i0,
  )

  # --- State B: shipped today (floor check on, elongated split off) ---
  shipped_cfg = FitConfig(
      tol=PARAMS["tol"], min_arc_height=PARAMS["min_arc_height"],
      reparam_iters=PARAMS["reparam_iters"], straight_seg_tol=PARAMS["straight_seg_tol"],
      enable_elongated_split=False,
  )
  shipped_curves, shipped_spans, _, _, _ = fit_closed_by_extrema_segments(pts_closed, shipped_cfg)

  # --- State C: new fix (elongated split on) ---
  fixed_cfg = FitConfig(
      tol=PARAMS["tol"], min_arc_height=PARAMS["min_arc_height"],
      reparam_iters=PARAMS["reparam_iters"], straight_seg_tol=PARAMS["straight_seg_tol"],
      enable_elongated_split=True,
  )
  fixed_curves, fixed_spans, _, _, _ = fit_closed_by_extrema_segments(pts_closed, fixed_cfg)

  fig, axes = plt.subplots(1, 3, figsize=(21, 9), dpi=140)
  render(axes[0], bw, pts_closed, naive_curves, naive_spans,
         "A) naive (pre-fix): single cubic")
  render(axes[1], bw, pts_closed, shipped_curves, shipped_spans,
         "B) shipped today: generic recursive split")
  render(axes[2], bw, pts_closed, fixed_curves, fixed_spans,
         "C) new fix: deliberate elongated-pair split at the elbow")
  fig.tight_layout()
  out_path = "output/eval_harness/elongated_split_before_after.png"
  fig.savefig(out_path)
  print("Wrote", out_path)

  for name, curves in [("A naive", naive_curves), ("B shipped", shipped_curves), ("C new fix", fixed_curves)]:
    print(f"{name}: n_curves(this render)={len(curves)}")
  print(f"B total curves (whole contour): {len(shipped_curves)}")
  print(f"C total curves (whole contour): {len(fixed_curves)}")


if __name__ == "__main__":
  main()
