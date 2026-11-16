"""
Fixed evaluation harness for the No-PoTrace tracing pipeline
(tools/png_to_ttf_no_potrace.py -> tools/segement_extrama.py -> tools/bezier_fit.py).

Purpose (see "Outline tracing, and how to debug it" in README.md):
runs the ACTIVE production call chain on a fixed, reproducible set of test
contours chosen to cover distinct shape categories (long smooth outer
contours, narrow joining strokes, bowls/holes, decorative ornament texture,
near-straight tall strokes, a noise-hole regression case), and for each one
saves:

  - a visual overlay PNG: source pixels, typed extrema, fitted cubics +
    control handles (color-coded by whether each curve's fit error exceeds
    tolerance), single-scale accepted inflection points, and the positions
    the currently-unused multiscale persistent-inflection detector
    (inflections_dp.detect_real_inflections_multiscale_dp) would have
    picked instead, for direct comparison
  - per-case JSON metrics: topology, per-stage timings, single-scale vs
    multiscale inflection counts/positions with persistence scores, exact
    cubic inflection roots, per-curve max fit error, curve count and
    handle-ratio stats

This does NOT change production tracing behavior or parameters. It exists
so future parameter/algorithm changes (in particular swapping in the
multiscale inflection detector) can be judged by a numeric diff across a
fixed case set instead of by eyeballing overlays one word at a time.

Usage:
  uv run tools/tracing_eval_harness.py --out output/eval_harness
"""
import argparse
import json
import time
from dataclasses import asdict
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from png_to_ttf_no_potrace import (
    read_bw_grayscale, pad_image, find_contours_with_hierarchy,
    ensure_orientation, skip_tiny_hole_subtrees, signed_area,
)
from segement_extrama import (
    detect_browse_extrema_closed_typed, extract_arc_points_closed,
    tangent_from_extrema_pair_typed, fit_closed_by_extrema_segments, _wrap_pts,
)
from straight_segments import fit_segments_rdp, get_straights_segs, inflections_from_straights_open_window
from bezier_fit import FitConfig, max_error_against_poly
from inflections_dp import detect_real_inflections_multiscale_dp
from contour_metrics_report import contour_metrics

# -----------------------------------------------------------------------
# Fixed test-case manifest.
# contour_index is the index into find_contours_with_hierarchy(bw) for the
# given word PNG at pad_px=24 -- reproducible for a FIXED image, but NOT
# stable across different analyzer runs: re-running mushaf_cli with
# different flags/versions re-segments the word crop at the pixel level,
# which can insert/remove tiny holes and shift every later contour_index
# in that image. If the source word PNGs get regenerated, re-verify every
# entry below with list_contours.py (match by is_hole/area/bbox, not by
# index) before trusting harness results -- a stale index still runs
# without error, it just silently analyzes the wrong contour.
# -----------------------------------------------------------------------
DEFAULT_MANIFEST = [
    {"name": "l11_w07_body", "png": "output/oldmedina2-analysis/page003/wordimgs/l11_w07.png",
     "contour_index": 11, "category": "long_smooth_outer",
     "label": "long multi-tooth word body"},
    {"name": "l11_w07_hole", "png": "output/oldmedina2-analysis/page003/wordimgs/l11_w07.png",
     "contour_index": 12, "category": "bowl_hole",
     "label": "hamza-style loop hole"},
    {"name": "l01_w08_body", "png": "output/oldmedina2-analysis/page003/wordimgs/l01_w08.png",
     "contour_index": 2, "category": "long_smooth_outer",
     "label": "long word body"},
    {"name": "l01_w08_noise_hole", "png": "output/oldmedina2-analysis/page003/wordimgs/l01_w08.png",
     "contour_index": 14, "category": "noise_hole_regression",
     "label": "borderline ~380px2 speck (below default min_hole_area_px filter)"},
    {"name": "l04_w07_body", "png": "output/oldmedina2-analysis/page003/wordimgs/l04_w07.png",
     "contour_index": 0, "category": "long_smooth_outer",
     "label": "word body with clamped control handles (max_handle_ratio~1.0)"},
    {"name": "l04_w07_tiny_hole", "png": "output/oldmedina2-analysis/page003/wordimgs/l04_w07.png",
     "contour_index": 10, "category": "noise_hole_regression",
     "label": "~136px2 hole, near default min_hole_area_px=80 boundary"},
    {"name": "l13_w01_body", "png": "output/oldmedina2-analysis/page003/wordimgs/l13_w01.png",
     "contour_index": 8, "category": "long_smooth_outer",
     "label": "word body, high min-chord flag in metrics report"},
    {"name": "l13_w01_hole", "png": "output/oldmedina2-analysis/page003/wordimgs/l13_w01.png",
     "contour_index": 14, "category": "bowl_hole",
     "label": "larger elongated hole"},
    {"name": "l01_w02_ornament", "png": "output/oldmedina2-analysis/page003/wordimgs/l01_w02.png",
     "contour_index": 0, "category": "ornament_texture",
     "label": "ayah-end marker swirl -- dozens of small holes, some may be "
              "intentional decorative texture rather than scan noise"},
    {"name": "l00_w00_tall_stroke", "png": "output/oldmedina2-analysis/page003/wordimgs/l00_w00.png",
     "contour_index": 5, "category": "narrow_tall_stroke",
     "label": "tall narrow stroke (w=125,h=711), near-straight arc test"},
    {"name": "l00_w00_body", "png": "output/oldmedina2-analysis/page003/wordimgs/l00_w00.png",
     "contour_index": 2, "category": "baseline_control",
     "label": "moderate-complexity body, general baseline case"},
]

DEFAULT_PARAMS = dict(fit_err_px=8.0, min_arc_height=6.0, reparam_iters=5, max_fit_error=9.0, pad_px=24)


def bez3(p0, p1, p2, p3, t):
  u = 1.0 - t
  b0, b1, b2, b3 = u*u*u, 3*u*u*t, 3*u*t*t, t*t*t
  return (p0[0]*b0+p1[0]*b1+p2[0]*b2+p3[0]*b3,
          p0[1]*b0+p1[1]*b1+p2[1]*b2+p3[1]*b3)


def sample_cubic(c, n=40):
  return np.array([bez3(*c, t) for t in np.linspace(0, 1, n)])


def cubic_inflection_roots(c, n_scan=200):
  """
  Exact-ish cubic inflection roots: numerically find t in (0,1) where
  cross(B'(t), B''(t)) changes sign, refined by bisection. No attempt to
  classify/reject them (that judgment call belongs to the harness's future
  consumer); this just reports raw candidates.
  """
  p0, p1, p2, p3 = [np.array(p, dtype=float) for p in c]

  def bprime(t):
    u = 1 - t
    return 3*u*u*(p1-p0) + 6*u*t*(p2-p1) + 3*t*t*(p3-p2)

  def bpp(t):
    u = 1 - t
    return 6*u*(p2-2*p1+p0) + 6*t*(p3-2*p2+p1)

  def f(t):
    d1, d2 = bprime(t), bpp(t)
    return d1[0]*d2[1] - d1[1]*d2[0]

  ts = np.linspace(0.0, 1.0, n_scan)
  vals = [f(t) for t in ts]
  roots = []
  for i in range(n_scan - 1):
    a, b = vals[i], vals[i+1]
    if a == 0.0:
      roots.append(float(ts[i]))
      continue
    if a * b < 0:
      lo, hi = float(ts[i]), float(ts[i+1])
      for _ in range(30):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
          hi = mid
        else:
          lo = mid
      roots.append(0.5 * (lo + hi))
  return [r for r in roots if 0.02 < r < 0.98]


def gather_span_points(pts_ring, i0, i1):
  n = len(pts_ring)
  a, b = i0 % n, i1 % n
  if i1 - i0 >= n:
    return None  # span wrapped more than once; index arithmetic overflow, skip
  if a <= b:
    return pts_ring[a:b+1]
  return pts_ring[a:] + pts_ring[:b+1]


def analyze_extrema_and_inflections(pts_closed, min_arc_height, max_fit_error):
  """
  Lightweight diagnostic pass (independent of the production cubic fit):
  per-arc single-scale vs multiscale inflection comparison, for judging
  whether one scale of straight-segment decomposition is enough to find the
  inflections, or whether they have to be tracked across scales.
  """
  pts = _wrap_pts(pts_closed)
  n = len(pts)
  timings = {}

  t0 = time.perf_counter()
  extrema_typed = detect_browse_extrema_closed_typed(pts, min_arc_height=min_arc_height)
  timings["extrema_detection_s"] = time.perf_counter() - t0

  arcs = []
  t_straight = t_single = t_multi = 0.0

  for k in range(len(extrema_typed)):
    a, a_type = extrema_typed[k]
    b, b_type = extrema_typed[(k + 1) % len(extrema_typed)]
    arc_pts, i0 = extract_arc_points_closed(pts_closed, a, b)
    if len(arc_pts) < 2:
      continue
    td0, td1 = tangent_from_extrema_pair_typed(pts_closed, a, a_type, b, b_type, arc_pts)

    ts = time.perf_counter()
    _, segs1 = fit_segments_rdp(arc_pts, max_error=max_fit_error)
    segs2 = get_straights_segs(arc_pts, segs1)
    t_straight += time.perf_counter() - ts

    ts = time.perf_counter()
    infls_single, _ = inflections_from_straights_open_window(arc_pts, segs2, start_dir=td0, end_dir=td1)
    t_single += time.perf_counter() - ts
    single_points = [list(infl.point) for infl in infls_single]

    ts = time.perf_counter()
    multi_scored = []
    multi_selected_pts = []
    try:
      result = detect_real_inflections_multiscale_dp(arc_pts, segs2)
      for sc in result.scored_candidates:
        multi_scored.append({
            "arc_length_s": round(sc.cand.s, 2),
            "score": round(sc.score, 3),
            "persistence": sc.persistence,
        })
      for idx in result.selected.orig_point_indices:
        if 0 <= idx < len(arc_pts):
          multi_selected_pts.append(list(arc_pts[idx]))
    except Exception as e:
      multi_scored = [{"error": str(e)}]
    t_multi += time.perf_counter() - ts

    arcs.append({
        "extrema_a": [int(a), a_type],
        "extrema_b": [int(b), b_type],
        "arc_n_points": len(arc_pts),
        "n_single_scale_inflections": len(infls_single),
        "single_scale_points": single_points,
        "n_multiscale_candidates": len(multi_scored),
        "n_multiscale_selected": len(multi_selected_pts),
        "multiscale_candidates": multi_scored,
        "multiscale_selected_points": multi_selected_pts,
    })

  timings["straight_decomposition_s"] = round(t_straight, 4)
  timings["single_scale_inflection_s"] = round(t_single, 4)
  timings["multiscale_inflection_s"] = round(t_multi, 4)

  return extrema_typed, arcs, timings


def render_case(png_path, contour_index, cubics, spans, extrema_typed, arcs,
                 pts_closed, out_png, tol, title):
  n_ring = len(_wrap_pts(pts_closed))

  fig, ax = plt.subplots(figsize=(14, 10), dpi=140)
  bw = read_bw_grayscale(png_path)
  bw = pad_image(bw, DEFAULT_PARAMS["pad_px"])
  ax.imshow(bw, cmap="gray", alpha=0.30, origin="upper")
  ax.set_aspect("equal")
  ax.invert_yaxis()

  worst_err = 0.0
  for c, sp in zip(cubics, spans):
    pts_span = gather_span_points(_wrap_pts(pts_closed), sp[0], sp[1])
    if pts_span is None or len(pts_span) < 2:
      err = -1.0
      color = "magenta"
    else:
      err = max_error_against_poly(pts_span, c)
      worst_err = max(worst_err, err)
      if err <= tol:
        color = "cyan"
      elif err <= 1.5 * tol:
        color = "orange"
      else:
        color = "red"
    pts_curve = sample_cubic(c, 40)
    ax.plot(pts_curve[:, 0], pts_curve[:, 1], "-", color=color, linewidth=1.3)
    ax.plot([c[0][0], c[1][0]], [c[0][1], c[1][1]], "-", color="yellow", linewidth=0.4, alpha=0.6)
    ax.plot([c[3][0], c[2][0]], [c[3][1], c[2][1]], "-", color="yellow", linewidth=0.4, alpha=0.6)

    for t_root in cubic_inflection_roots(c):
      p = bez3(*c, t_root)
      ax.plot(p[0], p[1], "D", color="lime", markersize=5, markeredgecolor="black", markeredgewidth=0.5)

  pts_ring = _wrap_pts(pts_closed)
  for idx, etype in extrema_typed:
    p = pts_ring[idx % n_ring]
    ax.plot(p[0], p[1], "s", color="darkorange", markersize=6)

  for arc in arcs:
    for p in arc["multiscale_selected_points"]:
      ax.plot(p[0], p[1], "^", color="magenta", markersize=8, markeredgecolor="black", markeredgewidth=0.5)
    for p in arc.get("single_scale_points", []):
      ax.plot(p[0], p[1], "*", color="yellow", markersize=12, markeredgecolor="black", markeredgewidth=0.5)

  ax.set_title(f"{title}\ncyan=curve ok  orange=err<=1.5x tol  red=err>1.5x tol  "
               f"orange sq=extrema  lime diamond=exact cubic inflection (in final curves)  "
               f"yellow star=single-scale split decision (active)  "
               f"magenta triangle=multiscale-selected split (comparison only)\n"
               f"worst curve err={worst_err:.2f}px (tol={tol}px)")
  fig.tight_layout()
  fig.savefig(out_png)
  plt.close(fig)
  return worst_err


def run_case(case, params, out_dir):
  png_path = Path(case["png"])
  bw = read_bw_grayscale(png_path)
  bw = pad_image(bw, params["pad_px"])
  nodes = find_contours_with_hierarchy(bw)

  n_outer_word = sum(1 for nd in nodes if not nd.is_hole)
  n_holes_word = sum(1 for nd in nodes if nd.is_hole)

  idx = case["contour_index"]
  if idx >= len(nodes):
    return {"name": case["name"], "error": f"contour_index {idx} out of range (only {len(nodes)} contours)"}

  node = nodes[idx]
  want_ccw = not node.is_hole
  pts_np = ensure_orientation(node.pts.astype(np.float64), want_ccw=want_ccw)
  pts_closed = [(float(x), float(y)) for x, y in pts_np.tolist()]

  extrema_typed, arcs, diag_timings = analyze_extrema_and_inflections(
      pts_closed, params["min_arc_height"], params["max_fit_error"])

  t0 = time.perf_counter()
  cfg = FitConfig(
      tol=params["fit_err_px"], min_arc_height=params["min_arc_height"],
      reparam_iters=params["reparam_iters"], straight_seg_tol=params["max_fit_error"])
  cubics, spans, _, _, _ = fit_closed_by_extrema_segments(pts_closed, cfg)
  production_fit_s = time.perf_counter() - t0

  m = contour_metrics(cubics)

  out_png = out_dir / f"{case['name']}.png"
  worst_err = render_case(
      png_path, idx, cubics, spans, extrema_typed, arcs, pts_closed,
      out_png, tol=params["fit_err_px"],
      title=f"{case['name']}  ({case['category']}: {case['label']})")

  n_single_total = sum(a["n_single_scale_inflections"] for a in arcs)
  n_multi_total = sum(a["n_multiscale_selected"] for a in arcs)

  return {
      "name": case["name"],
      "category": case["category"],
      "label": case["label"],
      "png": str(png_path),
      "contour_index": idx,
      "is_hole": node.is_hole,
      "contour_area_px2": round(abs(signed_area(node.pts)), 1),
      "contour_n_points": len(node.pts),
      "word_topology": {"n_outer_contours": n_outer_word, "n_hole_contours": n_holes_word},
      "n_extrema": len(extrema_typed),
      "n_arcs": len(arcs),
      "n_single_scale_inflections_total": n_single_total,
      "n_multiscale_selected_total": n_multi_total,
      "inflection_detector_agrees": n_single_total == n_multi_total,
      "timings_s": {**diag_timings, "production_fit_s": round(production_fit_s, 4)},
      "curve_metrics": m,
      "worst_curve_fit_error_px": round(worst_err, 2),
      "tol_px": params["fit_err_px"],
      "overlay_png": out_png.name,
      "arcs": arcs,
  }


def write_index_html(results, out_dir, params):
  rows = []
  for r in results:
    if "error" in r:
      rows.append(f"<tr><td>{escape(r['name'])}</td><td colspan=8 style='color:red'>{escape(r['error'])}</td></tr>")
      continue
    agree = "yes" if r["inflection_detector_agrees"] else "<b style='color:#c60'>DIFFERS</b>"
    err_color = "#080" if r["worst_curve_fit_error_px"] <= r["tol_px"] else "#c00"
    rows.append(
        f"<tr>"
        f"<td><a href='{escape(r['overlay_png'])}'>{escape(r['name'])}</a></td>"
        f"<td>{escape(r['category'])}</td>"
        f"<td>{escape(r['label'])}</td>"
        f"<td>{r['contour_n_points']}</td>"
        f"<td>{r['n_extrema']}</td>"
        f"<td>{r['curve_metrics'].get('n_curves', '-')}</td>"
        f"<td>{r['n_single_scale_inflections_total']} / {r['n_multiscale_selected_total']}</td>"
        f"<td>{agree}</td>"
        f"<td style='color:{err_color}'>{r['worst_curve_fit_error_px']:.2f}px</td>"
        f"<td>{r['curve_metrics'].get('max_handle_ratio', 0):.2f}</td>"
        f"</tr>"
    )

  html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Tracing eval harness</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 20px; background:#111; color:#eee; }}
table {{ border-collapse: collapse; width: 100%; }}
td, th {{ border: 1px solid #444; padding: 6px 10px; font-size: 13px; text-align: left; }}
th {{ background: #222; position: sticky; top: 0; }}
a {{ color: #6cf; }}
code {{ background:#222; padding:1px 4px; }}
</style></head><body>
<h2>Tracing evaluation harness</h2>
<p>Params: <code>{escape(json.dumps(params))}</code></p>
<p>single/multi column = total inflections selected by the active single-scale detector vs.
the currently-unused multiscale persistent detector (comparison only, not swapped into production).
"DIFFERS" flags contours where the two disagree on inflection count -- worth a closer look
before deciding whether to switch detectors.</p>
<table>
<tr><th>case</th><th>category</th><th>label</th><th>n_pts</th><th>n_extrema</th>
<th>n_curves</th><th>single/multi infl.</th><th>agree?</th><th>worst err</th><th>max handle ratio</th></tr>
{''.join(rows)}
</table>
</body></html>
"""
  (out_dir / "index.html").write_text(html, encoding="utf-8")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", type=str, default="output/eval_harness")
  ap.add_argument("--fit_err_px", type=float, default=DEFAULT_PARAMS["fit_err_px"])
  ap.add_argument("--min_arc_height", type=float, default=DEFAULT_PARAMS["min_arc_height"])
  ap.add_argument("--reparam_iters", type=int, default=DEFAULT_PARAMS["reparam_iters"])
  ap.add_argument("--max_fit_error", type=float, default=DEFAULT_PARAMS["max_fit_error"])
  args = ap.parse_args()

  params = dict(DEFAULT_PARAMS)
  params.update(fit_err_px=args.fit_err_px, min_arc_height=args.min_arc_height,
                 reparam_iters=args.reparam_iters, max_fit_error=args.max_fit_error)

  out_dir = Path(args.out)
  out_dir.mkdir(parents=True, exist_ok=True)

  results = []
  for case in DEFAULT_MANIFEST:
    print(f"Running {case['name']} ...")
    try:
      r = run_case(case, params, out_dir)
    except Exception as e:
      r = {"name": case["name"], "error": f"{type(e).__name__}: {e}"}
    results.append(r)

  (out_dir / "report.json").write_text(
      json.dumps({"params": params, "results": results}, indent=2), encoding="utf-8")
  write_index_html(results, out_dir, params)

  print(f"\nWrote {len(results)} cases to {out_dir}/report.json and {out_dir}/index.html")
  n_disagree = sum(1 for r in results if not r.get("inflection_detector_agrees", True) and "error" not in r)
  print(f"Cases where single-scale and multiscale inflection detectors disagree: {n_disagree}/{len(results)}")


if __name__ == "__main__":
  main()
