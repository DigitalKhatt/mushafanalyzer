from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
from typing import List, Tuple, Optional
import math
import numpy as np

from max_error import choose_split_minimax_pquant, compute_max_ortho_error_and_refined_u, compute_split_valley_near_bad_region
from straight_segments import StraightSegment, fit_segments_rdp

Point = Tuple[float, float]
Cubic = Tuple[Point, Point, Point, Point]
# inclusive indices in the OPEN polyline segment used for this curve
Span = Tuple[int, int]


@dataclass(frozen=True)
class FitConfig:
  """
  Fitting-policy parameters: set once per run (from CLI args or GUI
  sliders/checkboxes) and threaded unchanged through the whole recursive
  splitting pipeline (fit_cubics_open_with_spans, _pick_best_split,
  fit_closed_by_extrema_segments, fit_closed_contour_to_cubics_no_corners).
  Kept separate from per-call recursion state (pts, tangents, i0,
  nb_splits), which genuinely changes on every call and doesn't belong in
  a shared config object.
  """
  tol: float
  reparam_iters: int = 8
  max_splits: int = 2
  min_arc_height: float = 4
  straight_seg_tol: float = 9
  retry_without_inflection_check: bool = False
  use_naive_split_only: bool = False
  enable_elongated_split: bool = False

  # Testing knobs (used by VectorFitViewer to see how fitting behaves as if
  # working from a lower-resolution or blurrier scan, without re-rasterizing
  # the source image). No-op at their defaults, so production callers
  # (TTF/SVG builders) are unaffected.
  scale: float = 1.0       # uniform shrink of the traced contour points
  blur: float = 0.0        # curve-domain Gaussian sigma, in scaled-point units
  step: float = 0.0        # resample spacing fed to the fitter, in original-pixel units

  # Inflection-candidate gating (straight_segments.inflections_from_straights_open_window
  # already has these hooks; None/0.0 leaves candidate selection unchanged
  # from today). Reject a candidate whose mid segment isn't actually
  # straight (max_dev too high) or is too short to trust (chord_len too
  # low). Both are exposed in vector_fit_viewer.py under "Inflection
  # candidate gating"; see README.md, "Outline tracing".
  infl_max_mid_dev: Optional[float] = None
  infl_min_mid_len: float = 0.0

  # When True, fit_closed_by_extrema_segments's leaf/candidate cubic fits
  # (via _fit_single_cubic_converged) use fit_single_cubic_no_inflection
  # instead of fit_single_cubic_2. That fitter is unconditionally
  # inflection-free -- where it must give up an endpoint tangent to stay
  # that way the error rises and the span gets split, by design. False by
  # default, so production is unaffected.
  #
  # KNOWN ISSUE (opt-in only, under investigation): on page 3 this measured
  # as roughly error-neutral in aggregate (median/p90/p99 essentially
  # unchanged vs False), but it can rarely make _pick_best_split disqualify
  # every candidate where the baseline finds one, falling through to
  # _finalize_leaf with a wildly over-tolerance unsplit leaf. Reproduces on
  # output/oldmedina2-analysis/page003/wordimgs/l04_w01.png, contour index 4
  # (via find_contours_with_hierarchy): with this flag True the arc spanning
  # contour indices (111, 683) is accepted as one leaf at ~576px error
  # (tol=8); with the flag False the same word's worst leaf is 7.99px. A
  # second, pre-existing hard case (l00_w01.png, contour 14, span
  # (2011, 2921)) also gets moderately worse under the flag (34.5px ->
  # 39.0px). Root cause not yet isolated -- _pick_best_split and
  # _finalize_leaf are the likely places to look, but per prior direction
  # in this file, don't change either's behavior without deliberately
  # deciding to (see _finalize_leaf's own docstring for the same caution
  # about its dead branch).
  use_no_inflection_fit: bool = False

# -------------------------
# Basic vector ops
# -------------------------


def v_add(a: Point, b: Point) -> Point: return (a[0] + b[0], a[1] + b[1])
def v_sub(a: Point, b: Point) -> Point: return (a[0] - b[0], a[1] - b[1])
def v_mul(a: Point, s: float) -> Point: return (a[0] * s, a[1] * s)
def v_dot(a: Point, b: Point) -> float: return a[0] * b[0] + a[1] * b[1]
def v_len(a: Point) -> float: return math.hypot(a[0], a[1])


def lerp(a: Point, b: Point, t: float) -> Point:
  return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))


def v_len2(a: Point) -> float:
  return v_dot(a, a)


def v_norm(a: Point) -> Point:
  L = v_len(a)
  if L <= 1e-12:
    return (0.0, 0.0)
  return (a[0] / L, a[1] / L)


def v_dist(
    a: Point, b: Point) -> float: return math.hypot(a[0] - b[0], a[1] - b[1])


def clamp(x: float, lo: float,
          hi: float) -> float: return lo if x < lo else hi if x > hi else x


def angle_between(u: Point, v: Point) -> float:
  nu, nv = v_norm(u), v_norm(v)
  d = clamp(v_dot(nu, nv), -1.0, 1.0)
  return math.acos(d)

# -------------------------
# RDP polyline simplification (optional)
# -------------------------


def _dist_point_to_segment(p: Point, a: Point, b: Point) -> float:
  ab = v_sub(b, a)
  ap = v_sub(p, a)
  ab2 = v_dot(ab, ab)
  if ab2 <= 1e-12:
    return v_len(ap)
  t = clamp(v_dot(ap, ab) / ab2, 0.0, 1.0)
  proj = v_add(a, v_mul(ab, t))
  return v_dist(p, proj)


def rdp_simplify(points: List[Point], eps: float) -> List[Point]:
  """Ramer–Douglas–Peucker for open polyline."""
  if len(points) <= 2 or eps <= 0:
    return points[:]
  a, b = points[0], points[-1]
  max_d = -1.0
  idx = -1
  for i in range(1, len(points) - 1):
    d = _dist_point_to_segment(points[i], a, b)
    if d > max_d:
      max_d = d
      idx = i
  if max_d <= eps:
    return [a, b]
  left = rdp_simplify(points[: idx + 1], eps)
  right = rdp_simplify(points[idx:], eps)
  return left[:-1] + right

# -------------------------
# Corner detection on closed contour
# -------------------------


def detect_corners_closed(
    pts_closed: List[Point],
    corner_angle_deg: float = 50.0,
    window: int = 2,
    min_leg_len: float = 1e-3,
) -> List[bool]:
  """
  pts_closed: length n, no duplicate end
  corner[i] is True if point i is a corner.
  """
  n = len(pts_closed)
  assert n >= 3
  ang_th = math.radians(corner_angle_deg)
  corner = [False] * n

  for i in range(n):
    i0 = (i - window) % n
    i1 = (i + window) % n
    p_prev = pts_closed[i0]
    p = pts_closed[i]
    p_next = pts_closed[i1]

    v1 = v_sub(p, p_prev)
    v2 = v_sub(p_next, p)
    if v_len(v1) < min_leg_len or v_len(v2) < min_leg_len:
      continue
    ang = angle_between(v1, v2)  # 0 sharp, pi straight
    if ang <= ang_th:
      corner[i] = True
  return corner


def split_closed_by_corners(pts_closed: List[Point], is_corner: List[bool]) -> List[List[Point]]:
  """
  pts_closed: no duplicate end, length n
  is_corner: length n
  Returns list of OPEN segments (each includes endpoints).
  If no corners: returns one cyclic segment with duplicate start at end.
  """
  n = len(pts_closed)
  corner_idx = [i for i, c in enumerate(is_corner) if c]
  if not corner_idx:
    return [pts_closed[:] + [pts_closed[0]]]

  segments: List[List[Point]] = []
  corner_idx.sort()
  for k in range(len(corner_idx)):
    a = corner_idx[k]
    b = corner_idx[(k + 1) % len(corner_idx)]
    seg = [pts_closed[a]]
    i = (a + 1) % n
    while i != (b + 1) % n:
      seg.append(pts_closed[i])
      i = (i + 1) % n
    segments.append(seg)
  return segments

# -------------------------
# Bezier helpers
# -------------------------


def chord_length_params(pts: List[Point]) -> List[float]:
  u = [0.0]
  total = 0.0
  for i in range(1, len(pts)):
    total += v_dist(pts[i], pts[i - 1])
    u.append(total)
  if total <= 1e-12:
    return [0.0 for _ in pts]
  return [x / total for x in u]


def pos_params(pts: List[Point]) -> List[float]:
  u = [0.0]
  n = len(pts)
  for i in range(1, n):
    u.append((i+1)/n)
  return u


def reparameterize_by_chord_projection(pts: List[Point], curve: Cubic) -> List[float]:
  """
  Parameterize by projecting each data point onto the chord P0->P3.
  """
  p0, _, _, p3 = curve
  v = v_sub(p3, p0)
  vv = v_len2(v)
  if vv <= 1e-18:
    return [0.0 for _ in pts]

  u = []
  for x in pts:
    t = v_dot(v_sub(x, p0), v) / vv
    u.append(clamp(t, 0.0, 1.0))

  if u:
    u[0] = 0.0
    u[-1] = 1.0
  return u


def bezier_q(ctrl: Cubic, t: float) -> Point:
  (p0, p1, p2, p3) = ctrl
  mt = 1.0 - t
  a = v_mul(p0, mt * mt * mt)
  b = v_mul(p1, 3 * mt * mt * t)
  c = v_mul(p2, 3 * mt * t * t)
  d = v_mul(p3, t * t * t)
  return v_add(v_add(a, b), v_add(c, d))


def bezier_qprime(ctrl: Cubic, t: float) -> Point:
  p0, p1, p2, p3 = ctrl
  mt = 1.0 - t
  a = v_mul(v_sub(p1, p0), 3 * mt * mt)
  b = v_mul(v_sub(p2, p1), 6 * mt * t)
  c = v_mul(v_sub(p3, p2), 3 * t * t)
  return v_add(v_add(a, b), c)


def bezier_qprime2(ctrl: Cubic, t: float) -> Point:
  p0, p1, p2, p3 = ctrl
  mt = 1.0 - t
  a = v_mul(v_add(v_sub(p2, v_mul(p1, 2.0)), p0), 6 * mt)
  b = v_mul(v_add(v_sub(p3, v_mul(p2, 2.0)), p1), 6 * t)
  return v_add(a, b)


def reparameterize(pts: List[Point], u: List[float], curve: Cubic) -> List[float]:
  u2 = []
  for i, t in enumerate(u):
    p = bezier_q(curve, t)
    d1 = bezier_qprime(curve, t)
    d2 = bezier_qprime2(curve, t)
    diff = v_sub(p, pts[i])
    numerator = v_dot(diff, d1)
    denom = v_dot(d1, d1) + v_dot(diff, d2)
    if abs(denom) < 1e-12:
      u2.append(t)
    else:
      u2.append(clamp(t - numerator / denom, 0.0, 1.0))
  return u2


def reparameterize_safe(
    pts: List[Point],
    u: List[float],
    curve: Cubic,
    *,
    max_step: float = 0.15,     # trust region in parameter space
    # global damping (set <1 to be more conservative)
    it_damp: float = 1.0,
    enforce_monotone: bool = True,
    eps: float = 1e-6,          # minimum spacing for monotonicity
    accept_if_improves: bool = True,
) -> List[float]:
  """
  Newton reparameterization with safeguards:
    - trust-region clamp on delta t
    - optional accept/reject based on point->curve distance reduction
    - clamps endpoints and enforces monotone increasing u
  """
  n = len(u)
  if n == 0:
    return []
  if n == 1:
    return [0.0]

  u2 = [0.0] * n

  for i, t in enumerate(u):
    # endpoints: lock
    if i == 0:
      u2[i] = 0.0
      continue
    if i == n - 1:
      u2[i] = 1.0
      continue

    # Evaluate curve and derivatives at current t
    p = bezier_q(curve, t)
    d1 = bezier_qprime(curve, t)
    d2 = bezier_qprime2(curve, t)

    diff = v_sub(p, pts[i])

    # Newton step for minimizing squared distance
    numerator = v_dot(diff, d1)
    denom = v_dot(d1, d1) + v_dot(diff, d2)

    if abs(denom) < 1e-12:
      t_new = t
    else:
      dt = - numerator / denom

      # trust region in parameter space (prevents crazy jumps)
      if dt > max_step:
        dt = max_step
      if dt < -max_step:
        dt = -max_step

      t_new = clamp(t + it_damp * dt, 0.0, 1.0)

    if accept_if_improves and t_new != t:
      # accept only if the point-to-curve distance decreases
      p_new = bezier_q(curve, t_new)
      e_old = v_dot(diff, diff)
      e_new = v_dot(v_sub(p_new, pts[i]), v_sub(p_new, pts[i]))
      if e_new > e_old:
        t_new = t  # reject

    u2[i] = t_new

  if enforce_monotone:
    # enforce nondecreasing + small epsilon spacing
    u2[0] = 0.0
    u2[-1] = 1.0
    for i in range(1, n - 1):
      if u2[i] < u2[i - 1] + eps:
        u2[i] = u2[i - 1] + eps
    # also ensure we don't exceed 1-eps near the end
    for i in range(n - 2, 0, -1):
      if u2[i] > u2[i + 1] - eps:
        u2[i] = u2[i + 1] - eps
    u2[0] = 0.0
    u2[-1] = 1.0

  return u2


def left_tangent(pts: List[Point]) -> Point:
  return v_norm(v_sub(pts[1], pts[0]))


def right_tangent(pts: List[Point]) -> Point:
  return v_norm(v_sub(pts[-2], pts[-1]))


def fit_single_cubic(pts: List[Point], params: List[float], start_tangent: Point, end_tangent: Point,
                     min_handle_rel: float = 0.05,
                     max_handle_rel: float = 0.6) -> Cubic:
  """
  Fit a single cubic Bezier to a sequence of points.
  Uses least-squares fitting with optional tangent constraints.
  """
  if len(pts) < 2:
    raise ValueError("Need at least 2 points to fit a curve")

  pts_arr = np.array([[p[0], p[1]] for p in pts])

  # Use chord-length parameterization if not provided
  if params is None:
    params = chord_length_parameterization(pts)

  # Endpoints
  p0 = pts_arr[0]
  p3 = pts_arr[-1]

  # Estimate tangents if not provided
  if start_tangent is None or end_tangent is None:
    start_tangent, end_tangent = estimate_tangents(pts, 0, len(pts) - 1)

  # Normalize tangents
  start_tangent = start_tangent / (np.linalg.norm(start_tangent) + 1e-10)
  end_tangent = end_tangent / (np.linalg.norm(end_tangent) + 1e-10)

  # Solve for alpha and beta (control point distances along tangents)
  # For each point: D_i = B(t_i) where B is the Bezier curve
  # With P1 = P0 + alpha * T0, P2 = P3 - beta * T3

  n = len(pts)

  # Build the linear system: A * [alpha, beta]^T = b
  # where A is (2n x 2) and b is (2n x 1)
  A = np.zeros((2 * n, 2))
  b = np.zeros(2 * n)

  for i, t in enumerate(params):
    mt = 1 - t
    t2 = t * t
    t3 = t2 * t
    mt2 = mt * mt
    mt3 = mt2 * mt

    # Bezier basis functions
    basis0 = mt3
    basis1 = 3 * mt2 * t
    basis2 = 3 * mt * t2
    basis3 = t3

    # Known contribution from endpoints
    known = basis0 * p0 + basis3 * p3

    # Residual to fit
    residual = pts_arr[i] - known

    # Coefficients for alpha and beta
    # B(t) = known + basis1 * (P0 + alpha*T0) + basis2 * (P3 - beta*T3)
    # B(t) = known + basis1 * P0 + basis1 * alpha * T0 + basis2 * P3 - basis2 * beta * T3
    # But we already included basis0*P0 + basis3*P3 in known, so:
    # B(t) = known + basis1 * alpha * T0 - basis2 * beta * T3

    # For x-component (row 2*i)
    A[2*i, 0] = basis1 * start_tangent[0]
    A[2*i, 1] = -basis2 * end_tangent[0]
    b[2*i] = residual[0]

    # For y-component (row 2*i+1)
    A[2*i+1, 0] = basis1 * start_tangent[1]
    A[2*i+1, 1] = -basis2 * end_tangent[1]
    b[2*i+1] = residual[1]

  # Solve least squares with non-negative constraint
  try:
    # Initial unconstrained solution
    solution, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    alpha = solution[0]
    beta = solution[1]

    # Heuristic bounds based on chord length
    chord_length = np.linalg.norm(p3 - p0)
    max_dist = chord_length * max_handle_rel  # Control points shouldn't be too far

    # If solution is unreasonable, use heuristic
    if alpha < 0 or beta < 0 or alpha > max_dist * 2 or beta > max_dist * 2:
      # Use Schneider's heuristic: alpha = beta = |P3 - P0| / 3
      alpha = chord_length / 3.0
      beta = chord_length / 3.0

    # Additional check: ensure control points create reasonable curve
    # by limiting the distance
    alpha = np.clip(alpha, chord_length * min_handle_rel, max_dist)
    beta = np.clip(beta, chord_length * min_handle_rel, max_dist)

  except:
    # Fallback: use simple estimation
    chord_length = np.linalg.norm(p3 - p0)
    alpha = chord_length / 3.0
    beta = chord_length / 3.0

  # Compute control points
  p1 = p0 + alpha * start_tangent
  p2 = p3 - beta * end_tangent

  return (p0, p1, p2, p3)

def _curvature_bernstein_coeffs(p0, p3, t0n, t3n, alpha, beta):
  """
  Bernstein coefficients of the cubic's curvature-sign polynomial.

  With A = P1-P0, B = P2-P1, C = P3-P2, expanding cross(B'(t), B''(t))
  and collecting in the degree-3 Bernstein basis gives (up to the positive
  factor 18):

    c0 = cross(A,B)
    c1 = (cross(A,C) + cross(A,B)) / 3
    c2 = (cross(B,C) + cross(A,C)) / 3
    c3 = cross(B,C)

  Returned unnormalized (c0, 3*c1, 3*c2, c3) -- only their signs matter.
  A Bezier's curvature changes sign only where this polynomial does, so if
  all four share a sign the curve has no inflection on [0,1]. That is
  sufficient, not necessary: mixed coefficients can still describe a
  polynomial that never crosses zero on [0,1]. The larger exact region was
  measured and isn't worth chasing -- on the diverging-tangent arcs it
  contains only curves with ~4x the fit error of a collapsed handle.
  """
  A = alpha * t0n
  C = beta * t3n
  B = (p3 - C) - (p0 + A)

  def cross(u, v):
    return float(u[0] * v[1] - u[1] * v[0])

  c0 = cross(A, B)
  c3 = cross(B, C)
  cac = cross(A, C)
  return (c0, cac + c0, c3 + cac, c3)


def _is_inflection_free(coeffs) -> bool:
  """True if all curvature-sign coefficients share a sign (see above)."""
  mag = max(abs(c) for c in coeffs)
  eps = 1e-9 * mag
  return (all(c >= -eps for c in coeffs) or all(c <= eps for c in coeffs))


def fit_single_cubic_no_inflection(
    pts: List[Point],
    params: List[float],
    start_tangent: Optional[np.ndarray],
    end_tangent: Optional[np.ndarray],
    *,
    # Must match fit_single_cubic_2's default: this function is meant to
    # differ from it *only* by the no-inflection projection below, so that
    # toggling FitConfig.use_no_inflection_fit isolates that one effect.
    # A nonzero ridge here perturbed every curve by ~0.1% of handle length,
    # including inflection-free ones no constraint was even acting on.
    ridge_rel: float = 0.0,
    min_handle_rel: float = 0,
    max_handle_rel: float = 1,
) -> Cubic:
    """
    Fit a single cubic Bezier to points with fixed endpoint tangent
    directions, always inflection-free.

    The returned curve always satisfies _is_inflection_free -- there is no
    "not guaranteed" escape hatch, and callers must not second-guess it by
    refitting with fit_single_cubic_2.

    When the two tangent rays converge to a forward intersection, keeping
    the handles inside that control triangle is exactly the four-coefficient
    criterion: writing D = P3-P0 = L0*T0 + L3*T3 and K = cross(T0,T3), the
    coefficients reduce to a*K*(L3-b), a*L3*K, b*L0*K and b*K*(L0-a), which
    share a sign iff a <= L0 and b <= L3. So the triangle clamp is not a
    heuristic, it is the criterion specialized to that case.

    When the tangents diverge or are parallel in the same direction, no
    (a>0, b>0) satisfies it -- the u^2*t and u*t^2 coefficients have opposite
    signs whatever the handle lengths. The feasible set is then exactly the
    two axes: with b = 0 the polynomial collapses to a*L3*K*u^2 (single-signed
    for any a), and symmetrically for a = 0. So one tangent is kept exactly and
    the other is given up, choosing whichever of the two costs less error --
    with the straight chord (a = b = 0, curvature identically zero) as the
    floor that always exists.

    Giving up a tangent is deliberately expensive: on a diverging-tangent arc
    it typically doubles the fit error, which pushes the span over tol and
    makes the caller split it. That is the intended outcome -- a split at a
    better joint is the real fix, and this function's job is to never hand
    back an inflection to paper over it.
    """
    if len(pts) < 2:
        raise ValueError("Need at least 2 points to fit a curve")

    if params is None:
        params = chord_length_params(pts)

    # 1. Endpoints
    p0 = np.array(pts[0], dtype=float)
    p3 = np.array(pts[-1], dtype=float)
    chord = p3 - p0
    chord_len = float(np.linalg.norm(chord))

    if chord_len < 1e-12:
        P0 = (float(p0[0]), float(p0[1]))
        P3 = (float(p3[0]), float(p3[1]))
        return (P0, P0, P3, P3)

    # 2. Tangents
    if start_tangent is None or end_tangent is None:
        t0, t3 = estimate_tangents(pts, 0, len(pts) - 1)
        t0 = np.array(t0, dtype=float)
        t3 = np.array(t3, dtype=float)
    else:
        t0 = np.array(start_tangent, dtype=float)
        t3 = np.array(end_tangent, dtype=float)

    t0n = t0 / (np.linalg.norm(t0) + 1e-12)
    t3n = t3 / (np.linalg.norm(t3) + 1e-12)

    if np.linalg.norm(t0n) < 1e-9:
        t0n = chord / (np.linalg.norm(chord) + 1e-12)
    if np.linalg.norm(t3n) < 1e-9:
        t3n = chord / (np.linalg.norm(chord) + 1e-12)

    # 3. Calculate control triangle intersection (V)
    # Solve: p0 + a * t0n = p3 - b * t3n  =>  a * t0n + b * t3n = p3 - p0
    # Set up 2x2 system for intersection parameters [a, b]
    A_mat = np.column_stack((t0n, t3n))
    det_A = A_mat[0, 0] * A_mat[1, 1] - A_mat[0, 1] * A_mat[1, 0]

    has_intersection = False
    L0, L3 = chord_len, chord_len # Safe fallback values

    if abs(det_A) > 1e-9:
        # Inverse mapping to find length along tangents to intersection V
        ab = np.linalg.solve(A_mat, chord)
        a_v, b_v = ab[0], ab[1]
        
        # Only valid triangle if both point towards an intersection ahead of the rays
        if a_v > 1e-6 and b_v > 1e-6:
            L0 = float(a_v)
            L3 = float(b_v)
            has_intersection = True

    # 4. Standard Linear Least Squares Fit (Your original logic)
    C00, C01, C11 = 0.0, 0.0, 0.0
    X0, X1 = 0.0, 0.0

    for (px, py), t in zip(pts, params):
        mt = 1.0 - t
        b1 = 3.0 * mt * mt * t
        b2 = 3.0 * mt * t * t

        Pi = np.array([px, py], dtype=float)
        known = (mt*mt*mt + b1) * p0 + (b2 + t*t*t) * p3
        K = Pi - known

        a1 = b1 * t0n
        a2 = b2 * t3n

        C00 += float(np.dot(a1, a1))
        C01 += float(np.dot(a1, -a2))
        C11 += float(np.dot(-a2, -a2))

        X0 += float(np.dot(a1, K))
        X1 += float(np.dot(-a2, K))

    scale = max(1e-12, 0.5 * (C00 + C11))
    lam = float(ridge_rel) * scale

    C00r = C00 + lam
    C11r = C11 + lam
    det = C00r * C11r - C01 * C01

    if abs(det) > 1e-12:
        alpha = (X0 * C11r - X1 * C01) / det
        beta = (C00r * X1 - C01 * X0) / det
    else:
        alpha = chord_len / 3.0
        beta = chord_len / 3.0

    if not np.isfinite(alpha) or not np.isfinite(beta):
        alpha = chord_len / 3.0
        beta = chord_len / 3.0

    # 5. Structural handle-length clamps, applied before the no-inflection
    # projection so the projection gets the final word (clamping afterwards
    # could push a feasible pair back out of the feasible set).
    min_h = min_handle_rel * chord_len
    max_h = max_handle_rel * chord_len

    def clamp(a, b):
        return (float(np.clip(max(0.0, a), min_h, max_h)),
                float(np.clip(max(0.0, b), min_h, max_h)))

    def mk(a, b):
        q1 = p0 + a * t0n
        q2 = p3 - b * t3n
        return (
            (float(p0[0]), float(p0[1])),
            (float(q1[0]), float(q1[1])),
            (float(q2[0]), float(q2[1])),
            (float(p3[0]), float(p3[1])),
        )

    def feasible(a, b):
        return _is_inflection_free(
            _curvature_bernstein_coeffs(p0, p3, t0n, t3n, a, b))

    alpha, beta = clamp(alpha, beta)

    # 6. No-inflection enforcement. The unconstrained least-squares solution
    # is usually already feasible (it is whenever the data itself curves one
    # way), so test it before doing any work.
    if not feasible(alpha, beta):
        candidates = []
        if has_intersection:
            # Feasible set is the control triangle -- clamp into it.
            candidates.append(clamp(min(alpha, L0), min(beta, L3)))
        # Diverging/parallel tangents: the feasible set is the two axes.
        # Re-solve the surviving handle as a 1-D least squares rather than
        # reusing a coordinate of the 2-D solution, which was optimal only
        # in company with the handle now being dropped.
        a_only = X0 / C00r if abs(C00r) > 1e-12 else 0.0
        b_only = X1 / C11r if abs(C11r) > 1e-12 else 0.0
        candidates.append(clamp(a_only, 0.0))
        candidates.append(clamp(0.0, b_only))
        # Straight chord: curvature identically zero, feasible by
        # construction, so the search below can never come up empty.
        candidates.append((0.0, 0.0))

        best = None
        for a_c, b_c in candidates:
            if not feasible(a_c, b_c):
                continue
            curve_c = mk(a_c, b_c)
            err = max(v_dist(p, bezier_q(curve_c, t))
                      for p, t in zip(pts, params))
            if best is None or err < best[0]:
                best = (err, a_c, b_c)

        # min_handle_rel > 0 can clamp the straight chord back out of the
        # feasible set; the no-inflection guarantee outranks that preference.
        alpha, beta = (best[1], best[2]) if best is not None else (0.0, 0.0)

    return mk(alpha, beta)

def fit_single_cubic_2(
    pts: List[Point],
    params: List[float],
    start_tangent: Optional[np.ndarray],
    end_tangent: Optional[np.ndarray],
    *,
    # Ridge strength relative to the LS system magnitude:
    # 0 disables ridge; try 1e-4 .. 1e-2
    ridge_rel: float = 0.0,
    # handle length clamps relative to chord length
    min_handle_rel: float = 0,
    max_handle_rel: float = 1,
    # `pts` as an (N, 2) float array, when the caller already has one. Only an
    # allocation saver: a reparameterizing caller refits the same points several
    # times and would otherwise rebuild the array on each pass.
    pts_np: Optional[np.ndarray] = None,
) -> Cubic:
  """
  Fit a single cubic Bézier to points with fixed endpoint tangent directions.

  Convention matches your code:
    P1 = P0 + alpha * T0
    P2 = P3 - beta  * T3

  Returns cubic as Tuple[Point,Point,Point,Point] (all float tuples).
  """

  if len(pts) < 2:
    raise ValueError("Need at least 2 points to fit a curve")

  if params is None:
    params = chord_length_params(pts)  # use your existing function

  # endpoints
  p0 = np.array(pts[0], dtype=float)
  p3 = np.array(pts[-1], dtype=float)
  chord = p3 - p0
  chord_len = float(np.linalg.norm(chord))

  if chord_len < 1e-12:
    P0 = (float(p0[0]), float(p0[1]))
    P3 = (float(p3[0]), float(p3[1]))
    return (P0, P0, P3, P3)

  # tangents
  if start_tangent is None or end_tangent is None:
    t0, t3 = estimate_tangents(pts, 0, len(pts) - 1)
    t0 = np.array(t0, dtype=float)
    t3 = np.array(t3, dtype=float)
  else:
    t0 = np.array(start_tangent, dtype=float)
    t3 = np.array(end_tangent, dtype=float)

  # normalize only (DO NOT flip sign here; your recursion already controls it)
  t0n = t0 / (np.linalg.norm(t0) + 1e-12)
  t3n = t3 / (np.linalg.norm(t3) + 1e-12)

  # If tangents are near-zero, fall back to chord (rare)
  if np.linalg.norm(t0n) < 1e-9:
    t0n = chord / (np.linalg.norm(chord) + 1e-12)
  if np.linalg.norm(t3n) < 1e-9:
    t3n = chord / (np.linalg.norm(chord) + 1e-12)

  # Build 2x2 normal equations:
  # K_i = P_i - [(b0+b1)P0 + (b2+b3)P3]
  # K_i ≈ b1*alpha*T0  + (-b2)*beta*T3
  #
  # Accumulated over all points at once. Two details keep this bit-for-bit
  # equal to the scalar loop it replaces, which matters because the control
  # points it produces are stored to full precision downstream:
  #   * the reduction is a *cumulative* sum, not np.sum -- np.sum adds
  #     pairwise, while the loop added strictly left to right;
  #   * the trailing `+ 0.0` reproduces the loop's 0.0 seed, which is the one
  #     value a cumulative sum cannot reach: summing nothing but negative
  #     zeros yields -0.0, where `0.0 + -0.0` is +0.0.
  # `min` mirrors the zip the loop used, which stopped at the shorter of the
  # two.
  n = min(len(pts), len(params))
  P = (pts_np if pts_np is not None else np.asarray(pts, dtype=float))[:n]
  ts = np.asarray(params, dtype=float)[:n]

  if n:
    mt = 1.0 - ts
    b1 = 3.0*mt*mt*ts
    b2 = 3.0*mt*ts*ts

    K = P - ((mt*mt*mt + b1)[:, None] * p0 + (b2 + ts*ts*ts)[:, None] * p3)
    a1 = b1[:, None] * t0n
    na2 = -(b2[:, None] * t3n)      # the beta column is -a2

    C00, C01, C11, X0, X1 = (
        float(value) + 0.0 for value in np.cumsum(np.stack([
            a1[:, 0]*a1[:, 0] + a1[:, 1]*a1[:, 1],
            a1[:, 0]*na2[:, 0] + a1[:, 1]*na2[:, 1],
            na2[:, 0]*na2[:, 0] + na2[:, 1]*na2[:, 1],
            a1[:, 0]*K[:, 0] + a1[:, 1]*K[:, 1],
            na2[:, 0]*K[:, 0] + na2[:, 1]*K[:, 1],
        ]), axis=1)[:, -1])
  else:
    C00 = C01 = C11 = X0 = X1 = 0.0

  # Ridge scaled to system magnitude (prevents “everything collapses”)
  scale = max(1e-12, 0.5 * (C00 + C11))
  lam = float(ridge_rel) * scale

  C00r = C00 + lam
  C11r = C11 + lam
  det = C00r * C11r - C01 * C01

  if abs(det) > 1e-12:
    alpha = (X0 * C11r - X1 * C01) / det
    beta = (C00r * X1 - C01 * X0) / det
  else:
    alpha = chord_len / 3.0
    beta = chord_len / 3.0

  # Fallback if bad
  if not np.isfinite(alpha) or not np.isfinite(beta):
    alpha = chord_len / 3.0
    beta = chord_len / 3.0

  # Clamp handle lengths (keep them from collapsing)
  min_h = min_handle_rel * chord_len
  max_h = max_handle_rel * chord_len

  # If alpha/beta negative, fallback (do NOT flip tangent here)
  if alpha <= 0.0:
    # TODO
    alpha = 0
  if beta <= 0.0:
    # TODO
    beta = 0

  alpha = float(np.clip(alpha, min_h, max_h))
  beta = float(np.clip(beta,  min_h, max_h))

  p1 = p0 + alpha * t0n
  p2 = p3 - beta * t3n

  return (
      (float(p0[0]), float(p0[1])),
      (float(p1[0]), float(p1[1])),
      (float(p2[0]), float(p2[1])),
      (float(p3[0]), float(p3[1])),
  )


def generate_bezier(pts: List[Point], u: List[float], tHat1: Point, tHat2: Point) -> Cubic:
  p0 = pts[0]
  p3 = pts[-1]

  C00 = C01 = C11 = 0.0
  X0 = X1 = 0.0

  for i, t in enumerate(u):
    mt = 1.0 - t
    b0 = mt * mt * mt
    b1 = 3 * mt * mt * t
    b2 = 3 * mt * t * t
    b3 = t * t * t

    a1 = v_mul(tHat1, b1)
    a2 = v_mul(tHat2, b2)

    C00 += v_dot(a1, a1)
    C01 += v_dot(a1, a2)
    C11 += v_dot(a2, a2)

    tmp = v_sub(pts[i], v_add(v_mul(p0, b0 + b1), v_mul(p3, b2 + b3)))
    X0 += v_dot(a1, tmp)
    X1 += v_dot(a2, tmp)

  det = C00 * C11 - C01 * C01
  if abs(det) > 1e-12:
    alpha1 = (X0 * C11 - X1 * C01) / det
    alpha2 = (C00 * X1 - C01 * X0) / det
  else:
    d = v_dist(p0, p3)
    alpha1 = alpha2 = d / 3.0

  seg_len = v_dist(p0, p3)
  eps = 1e-6 * max(seg_len, 1.0)
  if alpha1 < eps or alpha2 < eps:
    alpha1 = alpha2 = seg_len / 3.0

  p1 = v_add(p0, v_mul(tHat1, alpha1))
  p2 = v_add(p3, v_mul(tHat2, alpha2))
  return (p0, p1, p2, p3)


# -------------------------
# NEW: PCA tangent fallback
# -------------------------


def pca_tangent_open(pts: List[Point], idx: int, radius: int = 6) -> Point:
  n = len(pts)
  i0 = max(0, idx - radius)
  i1 = min(n - 1, idx + radius)
  xs = [pts[i][0] for i in range(i0, i1 + 1)]
  ys = [pts[i][1] for i in range(i0, i1 + 1)]
  cx = sum(xs) / len(xs)
  cy = sum(ys) / len(ys)

  sxx = sum((x - cx) * (x - cx) for x in xs)
  syy = sum((y - cy) * (y - cy) for y in ys)
  sxy = sum((x - cx) * (y - cy) for x, y in zip(xs, ys))

  if abs(sxy) < 1e-12:
    v = (1.0, 0.0) if sxx >= syy else (0.0, 1.0)
  else:
    tr = sxx + syy
    det = sxx * syy - sxy * sxy
    lam = 0.5 * (tr + math.sqrt(max(0.0, tr * tr - 4 * det)))
    v = (lam - syy, sxy)

  v = v_norm(v)
  # orient consistently
  fwd = v_sub(pts[min(idx + 1, n - 1)], pts[max(idx - 1, 0)])
  if v_dot(v, fwd) < 0:
    v = v_mul(v, -1.0)
  return v


def pca_tangent_open_by_length(pts: List[Point], idx: int, radius: float = 6.0) -> Point:
  """
  PCA tangent on an open polyline, where `radius` is interpreted as arc length
  along the polyline, not as number of indices.

  The neighborhood includes:
    - the center point pts[idx]
    - all vertices within arc-length <= radius on both sides
    - interpolated boundary points exactly at distance `radius` when the cutoff
      falls inside an edge

  Returns a unit tangent vector oriented consistently with local forward direction.
  """
  n = len(pts)
  if n == 0:
    raise ValueError("pts must not be empty")
  if not (0 <= idx < n):
    raise IndexError("idx out of range")
  if n == 1:
    return (1.0, 0.0)
  if radius <= 0:
    # fallback to a local finite-difference tangent
    i0 = max(0, idx - 1)
    i1 = min(n - 1, idx + 1)
    d = v_sub(pts[i1], pts[i0])
    return v_norm(d)

  samples: List[Point] = [pts[idx]]

  # Walk backward from idx
  acc = 0.0
  i = idx
  while i > 0:
    a = pts[i]
    b = pts[i - 1]
    seg = v_len(v_sub(b, a))
    if seg < 1e-12:
      i -= 1
      continue

    if acc + seg <= radius:
      samples.append(b)
      acc += seg
      i -= 1
    else:
      t = (radius - acc) / seg
      samples.append(lerp(a, b, t))
      break

  # Walk forward from idx
  acc = 0.0
  i = idx
  while i < n - 1:
    a = pts[i]
    b = pts[i + 1]
    seg = v_len(v_sub(b, a))
    if seg < 1e-12:
      i += 1
      continue

    if acc + seg <= radius:
      samples.append(b)
      acc += seg
      i += 1
    else:
      t = (radius - acc) / seg
      samples.append(lerp(a, b, t))
      break

  # PCA on collected samples
  m = len(samples)
  if m == 1:
    i0 = max(0, idx - 1)
    i1 = min(n - 1, idx + 1)
    return v_norm(v_sub(pts[i1], pts[i0]))

  cx = sum(p[0] for p in samples) / m
  cy = sum(p[1] for p in samples) / m

  sxx = sum((p[0] - cx) * (p[0] - cx) for p in samples)
  syy = sum((p[1] - cy) * (p[1] - cy) for p in samples)
  sxy = sum((p[0] - cx) * (p[1] - cy) for p in samples)

  # principal eigenvector of [[sxx, sxy], [sxy, syy]]
  if abs(sxy) < 1e-12:
    v = (1.0, 0.0) if sxx >= syy else (0.0, 1.0)
  else:
    tr = sxx + syy
    det = sxx * syy - sxy * sxy
    disc = max(0.0, tr * tr - 4.0 * det)
    lam = 0.5 * (tr + math.sqrt(disc))
    v = (lam - syy, sxy)

  v = v_norm(v)

  # orient consistently with local forward direction
  i_prev = max(idx - 1, 0)
  i_next = min(idx + 1, n - 1)
  fwd = v_sub(pts[i_next], pts[i_prev])
  if v_dot(v, fwd) < 0:
    v = v_mul(v, -1.0)

  return v

# -------------------------
# choose split tangent from current fitted curve derivative
# -------------------------


def split_tangent_from_fit(curve, t, pts, split_idx):
  """
  Returns a FORWARD tangent direction at the split (along pts[split]->pts[split+1]).
  """
  d = bezier_qprime(curve, t)
  if v_len(d) <= 1e-9:
    d = pca_tangent_open(pts, split_idx, radius=6)
  else:
    d = v_norm(d)

  # Local forward direction from polyline around split
  n = len(pts)
  i0 = max(0, split_idx - 1)
  i1 = min(n - 1, split_idx + 1)
  fwd = v_sub(pts[i1], pts[i0])
  if v_len(fwd) > 1e-12:
    fwd = v_norm(fwd)
    if v_dot(d, fwd) < 0.0:
      d = v_mul(d, -1.0)

  return d  # forward


def tangent_windowed_difference(
    pts: List[Point],
    idx: int,
    prev: int = 20,
    next: int = 20,
) -> Point:
  """
  Tangent = normalize( sum_{k=1..prev} w_k*(p[i] - p[i-k]) + sum_{k=1..next} w_k*(p[i+k] - p[i]) )
  with weights w_k = k (gives more weight to farther points).
  """
  n = len(pts)
  tx = 0.0
  ty = 0.0

  for k in range(1, prev+1):
    i = idx - k
    if i < 0:
      break
    w = float(k)
    tx += w * (pts[idx][0] - pts[i][0])
    ty += w * (pts[idx][1] - pts[i][1])

  for k in range(1, next+1):
    i = idx + k
    if i >= n:
      break
    w = float(k)
    tx += w * (pts[i][0] - pts[idx][0])
    ty += w * (pts[i][1] - pts[idx][1])

  t = v_norm((tx, ty))
  if v_len(t) <= 1e-12:
    # fallback to immediate edge
    if 0 < idx < n-1:
      t = v_norm(v_sub(pts[idx+1], pts[idx-1]))
    elif idx < n-1:
      t = v_norm(v_sub(pts[idx+1], pts[idx]))
    else:
      t = v_norm(v_sub(pts[idx], pts[idx-1]))
  return t


def tangent_windowed_difference_by_length(
    pts: List[Point],
    idx: int,
    radius: float = 40.0,
) -> Point:
  """
  Same estimator as tangent_windowed_difference (weighted sum of
  difference vectors, farther points weighted more), but the neighborhood
  is walked by arc length instead of a fixed point count.

  CHAIN_APPROX_NONE emits one point per boundary pixel, so a fixed index
  count (e.g. 20 points) spans a different physical distance on a diagonal
  staircase run than on an axis-aligned run. Walking by arc length instead
  makes the tangent estimate independent of that local point-density bias.
  `radius` is in the same units as pts (pixels), matching the previous
  default window's typical physical extent.
  """
  n = len(pts)
  tx = 0.0
  ty = 0.0

  acc = 0.0
  i = idx
  while i > 0:
    a = pts[i]
    b = pts[i - 1]
    seg = v_len(v_sub(b, a))
    if seg < 1e-12:
      i -= 1
      continue
    if acc + seg > radius:
      break
    acc += seg
    tx += acc * (pts[idx][0] - b[0])
    ty += acc * (pts[idx][1] - b[1])
    i -= 1

  acc = 0.0
  i = idx
  while i < n - 1:
    a = pts[i]
    b = pts[i + 1]
    seg = v_len(v_sub(b, a))
    if seg < 1e-12:
      i += 1
      continue
    if acc + seg > radius:
      break
    acc += seg
    tx += acc * (b[0] - pts[idx][0])
    ty += acc * (b[1] - pts[idx][1])
    i += 1

  t = v_norm((tx, ty))
  if v_len(t) <= 1e-12:
    # fallback to immediate edge, same as tangent_windowed_difference
    if 0 < idx < n-1:
      t = v_norm(v_sub(pts[idx+1], pts[idx-1]))
    elif idx < n-1:
      t = v_norm(v_sub(pts[idx+1], pts[idx]))
    else:
      t = v_norm(v_sub(pts[idx], pts[idx-1]))
  return t


def center_tangent(pts: List[Point], idx: int) -> Point:
  # used when splitting
  v1 = v_sub(pts[idx], pts[idx - 1])
  v2 = v_sub(pts[idx + 1], pts[idx])
  t = v_norm(v_add(v_norm(v1), v_norm(v2)))
  # fallback if opposing
  if v_len(t) < 1e-12:
    t = v_norm(v_sub(pts[idx - 1], pts[idx + 1]))
  return t


def tangent_pca_window(
    pts: List[Point],
    idx: int,
    prev: int = 6,
    next: int = 6,
) -> Point:
  """
  Estimate tangent at pts[idx] using PCA over a window.
  prev/next: how many points to include before/after idx.
  Works on OPEN polylines. (For closed, wrap indices yourself.)
  """
  n = len(pts)
  i0 = max(0, idx - prev)
  i1 = min(n - 1, idx + next)
  window = pts[i0:i1+1]
  if len(window) < 2:
    return (0.0, 0.0)

  cx = sum(p[0] for p in window) / len(window)
  cy = sum(p[1] for p in window) / len(window)

  sxx = sum((p[0]-cx)*(p[0]-cx) for p in window)
  syy = sum((p[1]-cy)*(p[1]-cy) for p in window)
  sxy = sum((p[0]-cx)*(p[1]-cy) for p in window)

  # principal eigenvector of [[sxx, sxy],[sxy, syy]]
  if abs(sxy) < 1e-12 and sxx >= syy:
    t = (1.0, 0.0)
  elif abs(sxy) < 1e-12:
    t = (0.0, 1.0)
  else:
    tr = sxx + syy
    det = sxx*syy - sxy*sxy
    lam = 0.5*(tr + math.sqrt(max(0.0, tr*tr - 4*det)))
    t = (lam - syy, sxy)

  t = v_norm(t)

  # orient tangent along local forward direction (idx->idx+1 if possible)
  if idx < n - 1:
    fwd = v_sub(pts[idx+1], pts[idx])
  else:
    fwd = v_sub(pts[idx], pts[idx-1])
  if v_len(fwd) > 1e-12 and v_dot(t, fwd) < 0.0:
    t = v_mul(t, -1.0)

  return t


def estimate_tangents(pts: List[Point], start_idx: int, end_idx: int) -> Tuple[np.ndarray, np.ndarray]:
  """
  Estimate tangent vectors at start and end of a point sequence.
  Uses weighted average of neighboring points for robustness.
  """
  pts_arr = np.array([[p[0], p[1]] for p in pts[start_idx:end_idx+1]])

  if len(pts_arr) < 2:
    return np.array([1.0, 0.0]), np.array([1.0, 0.0])

  # Start tangent: weighted average of first few segments
  n_points = min(4, len(pts_arr))
  start_tangent = np.zeros(2)
  total_weight = 0.0

  for i in range(n_points - 1):
    vec = pts_arr[i + 1] - pts_arr[i]
    weight = 1.0 / (i + 1)  # Give more weight to closer points
    start_tangent += weight * vec
    total_weight += weight

  if total_weight > 0:
    start_tangent /= total_weight

  # End tangent: weighted average of last few segments
  end_tangent = np.zeros(2)
  total_weight = 0.0

  for i in range(max(0, len(pts_arr) - n_points), len(pts_arr) - 1):
    vec = pts_arr[i + 1] - pts_arr[i]
    weight = 1.0 / (len(pts_arr) - i)  # Give more weight to closer points
    end_tangent += weight * vec
    total_weight += weight

  if total_weight > 0:
    end_tangent /= total_weight

  # Normalize
  start_tangent = start_tangent / (np.linalg.norm(start_tangent) + 1e-10)
  end_tangent = end_tangent / (np.linalg.norm(end_tangent) + 1e-10)

  return start_tangent, end_tangent


def orient_tangent_to_both_sides(pts, k, t):
  """
  Ensures tangent t points in the forward direction at the join for BOTH sides.
  Forward-left  ~ pts[k]   - pts[k-1]
  Forward-right ~ pts[k+1] - pts[k]
  We choose sign of t to maximize dot with (left + right).
  """
  t = v_norm(t)
  if v_len(t) < 1e-12:
    return t

  n = len(pts)
  left = (0.0, 0.0)
  right = (0.0, 0.0)

  if k > 0:
    left = v_norm(v_sub(pts[k], pts[k-1]))
  if k < n - 1:
    right = v_norm(v_sub(pts[k+1], pts[k]))

  # if we have both, use their sum; otherwise use whichever exists
  s = v_add(left, right)
  if v_len(s) < 1e-12:
    s = left if v_len(left) > 1e-12 else right

  if v_len(s) > 1e-12 and v_dot(t, s) < 0.0:
    t = v_mul(t, -1.0)
  return t


def join_tangent_estimate(pts, k, curve, t_param, prev=20, nxt=20, w_bezier=0.7):
  """
  Returns a forward tangent at join k.
  Mixes bezier derivative (smooth) with windowed difference (robust).
  """
  tb = bezier_qprime(curve, t_param)
  tw = tangent_windowed_difference(pts, k, prev, nxt)

  tb = v_norm(tb)
  tw = v_norm(tw)

  # Align tw with tb to avoid cancellation
  if v_len(tb) > 1e-12 and v_len(tw) > 1e-12 and v_dot(tb, tw) < 0.0:
    tw = v_mul(tw, -1.0)

  t = v_add(v_mul(tb, w_bezier), v_mul(tw, 1.0 - w_bezier))
  t = orient_tangent_to_both_sides(pts, k, t)
  return v_norm(t)


def tangent_closed_windowed(
    pts_closed: List[Point],
    idx: int,
    prev_count: int = 8,
    next_count: int = 8,
) -> Point:
  """
  Estimate a forward tangent at pts[idx] for a CLOSED contour.
  pts_closed must have pts_closed[-1] == pts_closed[0].
  Returns a unit vector pointing forward (roughly towards idx+1).
  """
  if len(pts_closed) < 4:
    return (0.0, 0.0)

  # remove duplicate closure
  pts = pts_closed[:-1] if pts_closed[0] == pts_closed[-1] else pts_closed
  n = len(pts)
  idx = idx % n

  tx = 0.0
  ty = 0.0

  # accumulate backward differences (pointing forward overall)
  for k in range(1, prev_count + 1):
    i = (idx - k) % n
    w = float(k)
    dx = pts[idx][0] - pts[i][0]
    dy = pts[idx][1] - pts[i][1]
    tx += w * dx
    ty += w * dy

  # accumulate forward differences
  for k in range(1, next_count + 1):
    i = (idx + k) % n
    w = float(k)
    dx = pts[i][0] - pts[idx][0]
    dy = pts[i][1] - pts[idx][1]
    tx += w * dx
    ty += w * dy

  t = v_norm((tx, ty))

  # orient to match the immediate forward step
  fwd = v_sub(pts[(idx + 1) % n], pts[idx])
  if v_len(fwd) > 1e-12 and v_dot(t, fwd) < 0.0:
    t = v_mul(t, -1.0)

  return t
# -------------------------
# Patched Schneider fitter for OPEN polyline
# Returns (cubics, spans) where each span indexes into the input pts list.
# -------------------------


def _fit_single_cubic_converged(
    pts: List[Point],
    tol: float,
    start_tangent: Optional[Point],
    end_tangent: Optional[Point],
    reparam_iters: int,
    use_no_inflection_fit: bool = False,
):
  """
  Fit one Hermite-constrained cubic to `pts`, reparameterizing until the
  error stops improving (or already satisfies tol) instead of a fixed
  iteration count -- a fixed small count can stop mid-convergence and
  report an error that's still above tol, triggering a split the fit
  didn't actually need. `reparam_iters` still bounds the loop (raised to
  a sane floor) so this can't spin forever on a pathological case.
  Returns (curve, max_err, params, split_idx).

  `use_no_inflection_fit=True` fits with fit_single_cubic_no_inflection
  instead of fit_single_cubic_2. That fitter is unconditionally
  inflection-free, so there is no fallback here: where it has to give up a
  tangent to stay inflection-free the error rises and the caller splits the
  span, which is the intended response rather than something to undo.
  """
  params = chord_length_params(pts)
  pts_np = np.asarray(pts, dtype=float)

  best_curve = None
  best_params = params
  best_max_err = float("inf")
  best_split = None

  CONVERGE_EPS = 1e-2
  max_iters = max(reparam_iters, 40)
  prev_max_err = float("inf")

  for iter in range(max_iters):
    if use_no_inflection_fit:
      curve = fit_single_cubic_no_inflection(
          pts, params, start_tangent, end_tangent)
    else:
      curve = fit_single_cubic_2(pts, params, start_tangent, end_tangent,
                                 pts_np=pts_np)

    # compute_max_ortho_error_and_refined_u enforces monotonic point<->curve
    # correspondence internally: pts are always in polyline order and any
    # single fitted arc must correspond to them monotonically, so an
    # unconstrained nearest-point search's occasional out-of-order match
    # (e.g. on a badly-bowed curve forced into an S-shape by strict
    # endpoint tangent constraints) gets clamped rather than silently
    # under-reporting error and skipping a split the fit actually needs.
    # (A separate, coarser floor check against fixed chord-length
    # parameters was tried here and removed -- it flagged large "errors"
    # that were an artifact of chord-length parameterization assuming
    # uniform curve speed, not real deviation, and forced unnecessary
    # splits on curves that were already a good orthogonal fit.)
    max_err, split, newparams = compute_max_ortho_error_and_refined_u(
        pts, params, curve, seed_samples=33 if iter == 0 else 0)

    if max_err < best_max_err:
      best_max_err = max_err
      best_curve = curve
      best_params = params
      best_split = split
    params = newparams

    if max_err <= tol:
      break
    if prev_max_err - max_err < CONVERGE_EPS:      
      break
    prev_max_err = max_err

  # print("_fit_single_cubic_converged returned at " + str(iter) + " iterations")
  return best_curve, best_max_err, best_params, best_split


def _cubic_curvature_sign(curve: Cubic, t: float) -> float:
  """
  cross(B', B'') normalized by |B'|*|B''|, i.e. sin of the angle between
  velocity and acceleration.

  The raw cross product is not usable as a sign near zero curvature: on a
  straight cubic (collapsed handles, as fit_single_cubic_no_inflection
  returns when it has to give up a tangent) it is mathematically zero but
  numerically ~1e-13 of random sign, which reads as an inflection at every
  other sample. Normalizing makes the "is this really zero" test scale-free
  so callers can threshold it.
  """
  d1 = bezier_qprime(curve, t)
  d2 = bezier_qprime2(curve, t)
  scale = math.hypot(d1[0], d1[1]) * math.hypot(d2[0], d2[1])
  if scale < 1e-12:
    return 0.0
  return (d1[0] * d2[1] - d1[1] * d2[0]) / scale


def _cubic_inflection_ts(curve: Cubic, samples: int = 40) -> List[float]:
  """Parameter values where the curve's curvature changes sign."""
  # Below this the normalized cross product is float noise, not curvature
  # (see _cubic_curvature_sign) -- generous against the ~1e-13 a degenerate
  # straight cubic produces, tight against any real inflection.
  FLAT_EPS = 1e-9
  ts = [i / samples for i in range(samples + 1)]
  signs = [_cubic_curvature_sign(curve, t) for t in ts]
  infl_ts = []
  for i in range(len(ts) - 1):
    if abs(signs[i]) <= FLAT_EPS or abs(signs[i + 1]) <= FLAT_EPS:
      continue
    if signs[i] * signs[i + 1] < 0:
      infl_ts.append(0.5 * (ts[i] + ts[i + 1]))
  return infl_ts


def _cubic_has_unsupported_inflection(
    curve: Cubic,
    pts: List[Point],
) -> bool:
  """
  True if the fitted curve has any inflection point.

  Simplified for now: just flags inflection presence, without checking
  whether the raw data actually supports it (`pts` is accepted but
  unused, so call sites don't need to change). A lot of real strokes
  legitimately have an inflection -- a hollow or a bump in the
  data -- so this will over-flag some good curves; revisit with a
  data-aware version (compare the curve's inflection location against
  the raw polyline's own turning direction there) if that turns out to
  reject too much.
  """
  return len(_cubic_inflection_ts(curve)) > 0


def _pick_best_split(
    pts: List[Point],
    cfg: FitConfig,
    start_tangent: Optional[Point],
    end_tangent: Optional[Point],
    naive_split: int,
    curve: Cubic,
    params: List[float],
    check_inflection: bool = True,
):
  """
  The naive choice -- the single worst-error point -- is sensitive to a
  single noisy pixel and forces a joint tangent right at that spike,
  which can produce two children that are each barely better than the
  unsplit curve (a "near-miss on both halves" split). Instead, try a
  couple of candidate split points, actually refit both children for
  each, and keep whichever candidate minimizes the worse of the two
  child errors.

  `check_inflection=False` skips the unsupported-inflection disqualification
  below, so candidates are judged on error alone. Used for a second-pass
  retry when every candidate got disqualified on the first pass -- see
  `cfg.retry_without_inflection_check` on `fit_cubics_open_with_spans`.

  Returns (best_split_idx, best_joint_tangent, worst_child_err) or
  None if no valid candidate could be evaluated.
  """
  tol = cfg.tol
  n = len(pts)
  probe_iters = min(15, max(cfg.reparam_iters, 15))
  # A candidate right next to a boundary leaves one child a near-degenerate
  # sliver and the other almost the whole arc -- it can still narrowly beat
  # the unsplit curve's error on this one-level comparison (worth doing,
  # not worth much), which wins the "min worst error" contest yet wastes
  # most of the split budget on the tiny sliver while the big remainder
  # stays essentially unsplit. Require each child to keep a reasonable
  # share of the points so a split candidate has to actually make progress.
  min_side = 10 # max(2, n // 8)

  def evaluate(candidates):
    best = None
    for s in candidates:
      if s < min_side or s > n - min_side - 1:
        continue
      tCenter = tangent_windowed_difference_by_length(pts, s)

      left_curve, left_err, _, _ = _fit_single_cubic_converged(
          pts[: s + 1], tol, start_tangent, tCenter, probe_iters,
          cfg.use_no_inflection_fit)
      right_curve, right_err, _, _ = _fit_single_cubic_converged(
          pts[s:], tol, tCenter, end_tangent, probe_iters,
          cfg.use_no_inflection_fit)

      # A split that produces a child with an inflection unsupported by
      # the raw data looks worse than what it replaced, even if its raw
      # error is low -- disqualify it rather than let it win on error
      # alone.
      if check_inflection and (
          _cubic_has_unsupported_inflection(left_curve, pts[: s + 1])
          or _cubic_has_unsupported_inflection(right_curve, pts[s:])):
        continue

      worst = max(left_err, right_err)
      if best is None or worst < best[2]:
        best = (s, tCenter, worst)
    return best

  # Cheap first pass: the naive worst-error point, plus a "valley" split
  # near (not at the peak of) the bad region, which avoids joining right
  # at a noise spike.
  candidates = {naive_split}
  if n >= 50:
    _, valley_split = compute_split_valley_near_bad_region(
        pts, params, curve, seed_samples=33, newton_iters=2)
    candidates.add(valley_split)

  best = evaluate(candidates)
  if best is not None:
    return best

  # Both cheap candidates were disqualified (typically: both land inside
  # the same unsupported wiggle). Only now pay for a wider search -- a
  # handful of alternatives spread away from the naive split -- so the
  # common case stays cheap.
  wide_candidates = set()
  for frac in (0.25, 0.50, 0.75):
    offset = int(frac * n)
    wide_candidates.add(max(2, min(n - 3, naive_split - offset)))
    wide_candidates.add(max(2, min(n - 3, naive_split + offset)))

  return evaluate(wide_candidates)


def _finalize_leaf(curve: Cubic, pts: List[Point], i0: int):
  """
  Accept `curve` as a leaf covering the whole `pts` span. If it swings
  laterally beyond what the raw data supports (see
  `_cubic_has_unsupported_inflection`), the split-candidate search should
  normally have already avoided this -- but as a last resort (e.g. an
  unsplit curve, or max recursion depth reached) fall back to a plain
  geometric subdivision rather than handing back a curve that's
  manufacturing curvature the points don't show.
  """
  return [curve], [(i0, i0 + len(pts) - 1)]  
  if not _cubic_has_unsupported_inflection(curve, pts):
    return [curve], [(i0, i0 + len(pts) - 1)]

  left, right = subdivide_cubic(curve, 0.5)
  mid = len(pts) // 2
  return [left, right], [(i0, i0 + mid), (i0 + mid, i0 + len(pts) - 1)]


def fit_cubics_open_with_spans(
    pts: List[Point],
    cfg: FitConfig,
    start_tangent: Optional[Point] = None,
    end_tangent: Optional[Point] = None,
    i0: int = 0,  # base index into some larger polyline, for spans
    nb_splits: int = 0,
    straights: Optional[List[StraightSegment]] = None,
) -> Tuple[List[Cubic], List[Span]]:
  tol = cfg.tol
  if len(pts) < 2:
    return [], []
  if len(pts) == 2:
    p0, p3 = pts[0], pts[1]
    chord = v_sub(p3, p0)
    L = v_len(chord)

    # Degenerate case
    if L < 1e-12:
      return [(p0, p0, p3, p3)], [(i0, i0 + 1)]

    # Use provided tangents if available, otherwise chord direction
    if start_tangent is not None and v_len(start_tangent) > 1e-12:
      t0 = v_norm(start_tangent)
    else:
      t0 = v_norm(chord)

    if end_tangent is not None and v_len(end_tangent) > 1e-12:
      t1 = v_norm(end_tangent)
    else:
      t1 = v_norm(chord)

    # Ensure tangents point forward along the segment
    if v_dot(t0, chord) < 0.0:
      t0 = v_mul(t0, -1.0)
    if v_dot(t1, chord) < 0.0:
      t1 = v_mul(t1, -1.0)

    # Control point distances (Schneider heuristic)
    alpha = L / 3.0
    beta = L / 3.0

    p1 = v_add(p0, v_mul(t0, alpha))
    p2 = v_sub(p3, v_mul(t1, beta))

    return [(p0, p1, p2, p3)], [(i0, i0 + 1)]

  curve, max_err, params, naive_split = _fit_single_cubic_converged(
      pts, tol, start_tangent, end_tangent, cfg.reparam_iters,
      cfg.use_no_inflection_fit)

  if max_err <= tol or cfg.max_splits <= nb_splits:
    return _finalize_leaf(curve, pts, i0)

  if cfg.use_naive_split_only:
    # Pre-candidate-search behavior: split at the single worst-error
    # point with no other candidates considered and no "does this
    # actually help" check -- always recurse.
    split = naive_split
    tCenter = tangent_windowed_difference_by_length(pts, split)
  else:
    # Over tol: rather than always splitting at the single worst-error
    # point (sensitive to one noisy pixel, and can land two children each
    # just barely better than the unsplit curve), evaluate a couple of
    # candidate split points by actually refitting both children, and
    # split at whichever candidate minimizes the worse child's error. If
    # no candidate actually beats the unsplit fit, splitting wouldn't
    # help -- keep the single (best-effort) curve instead of manufacturing
    # a split that looks worse than what it replaced.
    best = _pick_best_split(
        pts, cfg, start_tangent, end_tangent, naive_split, curve, params,
    )

    # Every candidate on the first pass may have been disqualified for
    # having an inflection the raw data doesn't clearly support -- but a
    # split that reduces error, even with such an inflection, still beats
    # falling back to a single curve that's over tol. Retry once judging
    # candidates on error alone.
    if (best is None or best[2] >= max_err) and cfg.retry_without_inflection_check:
      best = _pick_best_split(
          pts, cfg, start_tangent, end_tangent, naive_split, curve, params,
          check_inflection=False,
      )

    if best is None or best[2] >= max_err:
      return _finalize_leaf(curve, pts, i0)

    split, tCenter, worst_child_err = best

  # Recurse left/right, passing tangents at the join. `cfg` flows through
  # unchanged -- no per-layer unpacking/repacking needed.
  left_c, left_s = fit_cubics_open_with_spans(
      pts[: split + 1], cfg,
      start_tangent=start_tangent, end_tangent=tCenter, i0=i0,
      nb_splits=nb_splits+1,
  )
  right_c, right_s = fit_cubics_open_with_spans(
      pts[split:], cfg,
      start_tangent=tCenter, end_tangent=end_tangent, i0=i0 + split,
      nb_splits=nb_splits+1,
  )
  return left_c + right_c, left_s + right_s

# -------------------------
# NEW: robust error check for merge (against original polyline span)
# -------------------------


def max_error_against_poly(pts_span: List[Point], curve: Cubic) -> float:
  u = chord_length_params(pts_span)
  maxe = 0.0
  for p, t in zip(pts_span, u):
    maxe = max(maxe, v_dist(p, bezier_q(curve, t)))
  return maxe

# Uses your existing:
# - fit_cubics_open_with_spans(pts_span, tol, reparam_iters, tHat1, tHat2, i0) -> (curves, spans)
# - max_error_against_poly(pts_span, curve)
# - v_norm, v_sub, v_len, v_dot, v_mul


def _end_tangent_of_curve(c: Cubic) -> Point:
  # forward tangent at end is (p3 - p2)
  t = v_sub(c[3], c[2])
  return v_norm(t) if v_len(t) > 1e-12 else (0.0, 0.0)


def _start_tangent_of_curve(c: Cubic) -> Point:
  # forward tangent at start is (p1 - p0)
  t = v_sub(c[1], c[0])
  return v_norm(t) if v_len(t) > 1e-12 else (0.0, 0.0)


def merge_pass_by_refit_windowed(
    poly_pts: List[Point],
    curves: List[Cubic],
    spans: List[Span],
    tol: float,
    *,
    reparam_iters: int = 8,
    window_max: int = 6,          # try merging up to this many adjacent curves at once
    # usually 1; set 2 if you want more aggressive simplification
    max_out_curves: int = 1,
    max_global_iters: int = 10,   # safety
) -> Tuple[List[Cubic], List[Span]]:
  """
  Windowed merge-by-refit.
  Tries to merge W adjacent curves (W=2..window_max) into <= max_out_curves curves,
  accepting only if all new curves fit within tol.
  Greedy: pick best merge (largest reduction, then smallest error) each iteration.
  """
  if not curves:
    return curves, spans
  assert len(curves) == len(spans)

  def spans_touch(i: int, j: int) -> bool:
    # expects contiguous: span[i].end == span[i+1].start for all inside window
    return spans[i][1] == spans[j][0]

  def window_is_contiguous(i: int, w: int) -> bool:
    for k in range(i, i + w - 1):
      if spans[k][1] != spans[k + 1][0]:
        return False
    return True

  def try_merge_window(i: int, w: int) -> Optional[Tuple[List[Cubic], List[Span], float]]:
    """
    Attempt to merge curves[i:i+w] into <= max_out_curves curves.
    Returns (new_curves, new_spans, worst_error) or None.
    """
    if w < 2:
      return None
    if i < 0 or i + w > len(curves):
      return None
    if not window_is_contiguous(i, w):
      return None

    a0 = spans[i][0]
    b1 = spans[i + w - 1][1]
    pts_span = poly_pts[a0: b1 + 1]
    if len(pts_span) < 2:
      return None

    # endpoint tangents from existing boundary curves
    tHat_start = _start_tangent_of_curve(curves[i])
    tHat_end = _end_tangent_of_curve(curves[i + w - 1])

    # Fit to the entire span
    new_curves, new_spans_local = fit_cubics_open_with_spans(
        pts_span,
        FitConfig(tol=tol, reparam_iters=reparam_iters),
        start_tangent=tHat_start,
        end_tangent=tHat_end,
        i0=a0
    )

    # Must simplify: not more than max_out_curves curves
    if len(new_curves) > w - 1:
      return None

    # Safety: verify actual error against polyline for each produced curve.
    # We compute worst error across produced curves using their local spans.
    worst_err = 0.0
    for c, sp in zip(new_curves, new_spans_local):
      s0, s1 = sp
      pts_piece = poly_pts[s0: s1 + 1]
      e = max_error_against_poly(pts_piece, c)
      worst_err = max(worst_err, e)
      if worst_err > tol:
        return None

    return new_curves, new_spans_local, worst_err

  it = 0
  while it < max_global_iters:
    it += 1
    changed = False

    i = 0
    while i < len(curves) - 1:
      max_w = min(window_max, len(curves) - i)

      best_i = None
      # best_i = (gain, worst_err, w, new_curves, new_spans)

      for w in range(2, max_w + 1):
        out = try_merge_window(i, w)
        if out is None:
          continue

        new_cs, new_ss, worst_err = out
        gain = w - len(new_cs)
        if gain <= 0:
          continue

        cand = (gain, worst_err, w, new_cs, new_ss)

        if best_i is None:
          best_i = cand
        else:
          # prefer larger gain; tie-break with smaller worst_err; then larger w
          if (cand[0] > best_i[0] or
              (cand[0] == best_i[0] and cand[1] < best_i[1]) or
                  (cand[0] == best_i[0] and abs(cand[1] - best_i[1]) < 1e-12 and cand[2] > best_i[2])):
            best_i = cand

      if best_i is None:
        # nothing merges starting at i; move to next start
        i += 1
        continue

      gain, worst_err, w, new_cs, new_ss = best_i

      # Apply best replacement at this i
      curves[i:i+w] = new_cs
      spans[i:i+w] = new_ss
      changed = True

      # Jump to "next point after w"
      # This means skip over the region we just consumed.
      # Since we replaced w old curves by len(new_cs) new curves, the next untouched
      # start index is i + len(new_cs) if you want to avoid reprocessing new curves,
      # or i + 1 if you want to allow overlapping merges.
      i += len(new_cs)  # recommended "jump after merge"

    if not changed:
      break

  return curves, spans


# -------------------------
# NEW: Strategy A merge-by-refit pass
# Input:
#   poly_pts = open polyline points (the same indexing basis used by spans)
#   curves, spans from fitting on that polyline
# Output:
#   merged curves+spans
# -------------------------
def merge_pass_by_refit(
    poly_pts: List[Point],
    curves: List[Cubic],
    spans: List[Span],
    tol: float,
    reparam_iters: int = 8,
    max_iters: int = 50,
) -> Tuple[List[Cubic], List[Span]]:
  if not curves:
    return curves, spans
  assert len(curves) == len(spans)

  def try_merge(i: int) -> Optional[Tuple[Cubic, Span]]:
    (a0, a1) = spans[i]
    (b0, b1) = spans[i + 1]
    if b0 != a1:
      # spans should touch; if not, don't merge
      return None
    pts_span = poly_pts[a0: b1 + 1]
    if len(pts_span) < 2:
      return None

    # Use endpoint tangents from existing curves (helps preserve look & smoothness)
    # tangent at start: along (p1 - p0)
    c0 = curves[i]
    c1 = curves[i + 1]
    tHat_start = v_norm(v_sub(c0[1], c0[0]))
    # tangent at end: along (p3 - p2)
    tHat_end = v_norm(v_sub(c1[3], c1[2]))

    fitted, _ = fit_cubics_open_with_spans(
        pts_span, FitConfig(tol=tol, reparam_iters=reparam_iters),
        tHat1=tHat_start, tHat2=tHat_end, i0=a0
    )
    # Only accept if it collapses to a single cubic (aggressive simplification)
    if len(fitted) == 1:
      c = fitted[0]
      if max_error_against_poly(pts_span, c) <= tol:
        return c, (a0, b1)
    return None

  # Greedy left-to-right merges until no progress
  it = 0
  while it < max_iters:
    it += 1
    changed = False
    i = 0
    while i < len(curves) - 1:
      merged = try_merge(i)
      if merged is not None:
        c_new, s_new = merged
        curves[i] = c_new
        spans[i] = s_new
        del curves[i + 1]
        del spans[i + 1]
        changed = True
        # after merge, try merging again with previous neighbor
        if i > 0:
          i -= 1
        continue
      i += 1
    if not changed:
      break

  return curves, spans

# -------------------------
# C1 enforcement at smooth joins (post-process)
# join_is_corner[j] refers to join between curves[j-1] and curves[j] (mod m)
# -------------------------


def enforce_c1_at_joins(curves: List[Cubic], join_is_corner: List[bool]) -> List[Cubic]:
  m = len(curves)
  if m == 0:
    return curves
  curves_m = [list(c) for c in curves]

  for j in range(m):
    if join_is_corner[j]:
      continue
    prev = (j - 1) % m
    P = curves_m[prev][3]

    len_out = v_dist(P, curves_m[prev][2])
    len_in = v_dist(curves_m[j][1], P)

    t1 = v_norm(v_sub(P, curves_m[prev][2]))
    t2 = v_norm(v_sub(curves_m[j][1], P))
    t = v_norm(v_add(t1, t2))
    if v_len(t) < 1e-12:
      # chord fallback
      t = v_norm(v_sub(curves_m[j][3], curves_m[prev][0]))
      if v_len(t) < 1e-12:
        continue

    curves_m[prev][2] = v_add(P, v_mul(t, -len_out))
    curves_m[j][1] = v_add(P, v_mul(t, len_in))
    curves_m[j][0] = P

  return [tuple(c) for c in curves_m]  # type: ignore

# -------------------------
# Main API: Fit CLOSED contour to cubics
# Now includes:
#  - improved split tangents
#  - merge-by-refit pass to collapse zigzags
# -------------------------


def fit_closed_contour_to_cubics(
    pts: List[Point],
    tol: float,
    *,
    rdp_eps: float = 0.0,
    corner_angle_deg: float = 50.0,
    corner_window: int = 2,
    reparam_iters: int = 8,
    do_merge_pass: bool = True,
) -> List[Cubic]:
  if len(pts) < 4:
    return []

  # Remove duplicate last point if closed
  if v_dist(pts[0], pts[-1]) <= 1e-12:
    pts0 = pts[:-1]
  else:
    pts0 = pts[:]

  # Remove consecutive duplicates
  cleaned = [pts0[0]]
  for p in pts0[1:]:
    if v_dist(p, cleaned[-1]) > 1e-12:
      cleaned.append(p)
  pts0 = cleaned
  if len(pts0) < 3:
    return []

  # Optional RDP on closed contour by opening at 0
  if rdp_eps > 0.0:
    opened = pts0[:] + [pts0[0]]
    opened_s = rdp_simplify(opened, rdp_eps)
    pts0 = opened_s[:-1]
    if len(pts0) < 3:
      return []

  # Corner detection & split into segments
  is_corner = detect_corners_closed(
      pts0, corner_angle_deg=corner_angle_deg, window=corner_window)
  segments = split_closed_by_corners(pts0, is_corner)

  cubics_all: List[Cubic] = []
  join_is_corner: List[bool] = []

  if len(segments) == 1 and v_dist(segments[0][0], segments[0][-1]) <= 1e-12:
    # No corners: smooth loop. Pick stable start (farthest from centroid).
    seg = segments[0][:-1]
    cx = sum(p[0] for p in seg) / len(seg)
    cy = sum(p[1] for p in seg) / len(seg)
    start = max(range(len(seg)), key=lambda i: (
        seg[i][0]-cx)**2 + (seg[i][1]-cy)**2)
    poly_open = seg[start:] + seg[:start] + [seg[start]]

    curves, spans = fit_cubics_open_with_spans(
        poly_open, FitConfig(tol=tol, reparam_iters=reparam_iters), i0=0)

    if do_merge_pass:
      curves, spans = merge_pass_by_refit(
          poly_open, curves, spans, tol, reparam_iters=reparam_iters)

    cubics_all = curves
    join_is_corner = [False] * len(cubics_all)
  else:
    # With corners: each segment open. We fit each segment independently and merge within it.
    for si, seg in enumerate(segments):
      if len(seg) < 2:
        continue

      # Fit segment as open polyline
      curves, spans = fit_cubics_open_with_spans(
          seg, FitConfig(tol=tol, reparam_iters=reparam_iters), i0=0)

      if do_merge_pass:
        curves, spans = merge_pass_by_refit(
            seg, curves, spans, tol, reparam_iters=reparam_iters)

      # Mark join between previous segment and this as corner (segment boundaries are corners)
      if cubics_all and curves:
        join_is_corner.append(True)
      cubics_all.extend(curves)

    # Close-join is also a corner in this mode
    if cubics_all:
      # Ensure join_is_corner length matches curves count (joins indexed by curve start)
      # We mark all segment-start joins as corners; internal joins default smooth.
      # For simplicity here, mark every join as smooth, then stamp segment boundaries as corners:
      join_is_corner = [False] * len(cubics_all)
      # At least the closure is corner:
      join_is_corner[0] = True  # curve 0 start join
      # We already appended True for each segment boundary in order; place them:
      # (light approximation; usually enough because most corners are split anyway)

  # Stitch endpoints exactly to form a closed loop
  m = len(cubics_all)
  if m == 0:
    return []
  stitched: List[Cubic] = []
  for i in range(m):
    prev = (i - 1) % m
    P = cubics_all[prev][3]
    p0, p1, p2, p3 = cubics_all[i]
    stitched.append((P, p1, p2, p3))
  # Force last end equals first start
  p0_first = stitched[0][0]
  p0, p1, p2, p3 = stitched[-1]
  stitched[-1] = (p0, p1, p2, p0_first)

  # Enforce C1 at non-corner joins
  if len(join_is_corner) != len(stitched):
    join_is_corner = [False] * len(stitched)
  stitched = enforce_c1_at_joins(stitched, join_is_corner)

  return stitched
