import numpy as np
from typing import List, Tuple
import math
from typing import List
import cv2

from smoothing_image import (
    curve_domain_gaussian_smooth,
    fair_outer_contour_close_to_ink,
)

from segement_extrama import fit_closed_by_extrema_segments
from bezier_fit import (
    FitConfig,
    fit_cubics_open_with_spans,
    tangent_closed_windowed,
)


# import the functions from Part A (paste them above this file, or import from a module)
# - read / preprocess helpers (below)
# - fit_closed_contour_to_cubics
# - bez3


Point = Tuple[float, float]
Cubic = Tuple[Point, Point, Point, Point]


def v_add(a, b):
  return (a[0] + b[0], a[1] + b[1])


def v_sub(a, b):
  return (a[0] - b[0], a[1] - b[1])


def v_mul(a, s):
  return (a[0] * s, a[1] * s)


def v_dot(a, b):
  return a[0] * b[0] + a[1] * b[1]


def v_len(a):
  return math.hypot(a[0], a[1])


def v_norm(a):
  l = v_len(a)
  if l == 0:
    return (0.0, 0.0)
  return (a[0] / l, a[1] / l)


def bez3(p0, p1, p2, p3, t):
  u = 1.0 - t
  b0 = u * u * u
  b1 = 3 * u * u * t
  b2 = 3 * u * t * t
  b3 = t * t * t
  return (
      p0[0] * b0 + p1[0] * b1 + p2[0] * b2 + p3[0] * b3,
      p0[1] * b0 + p1[1] * b1 + p2[1] * b2 + p3[1] * b3,
  )


def bez3_d1(p0, p1, p2, p3, t):
  # first derivative
  u = 1.0 - t
  return (
      3 * u * u * (p1[0] - p0[0])
      + 6 * u * t * (p2[0] - p1[0])
      + 3 * t * t * (p3[0] - p2[0]),
      3 * u * u * (p1[1] - p0[1])
      + 6 * u * t * (p2[1] - p1[1])
      + 3 * t * t * (p3[1] - p2[1]),
  )


def bez3_d2(p0, p1, p2, p3, t):
  # second derivative
  u = 1.0 - t
  return (
      6 * u * (p2[0] - 2 * p1[0] + p0[0]) + 6 *
      t * (p3[0] - 2 * p2[0] + p1[0]),
      6 * u * (p2[1] - 2 * p1[1] + p0[1]) + 6 *
      t * (p3[1] - 2 * p2[1] + p1[1]),
  )


def chord_length_params(pts: List[Point]) -> List[float]:
  d = [0.0]
  for i in range(1, len(pts)):
    d.append(d[-1] + v_len(v_sub(pts[i], pts[i - 1])))
  total = d[-1]
  if total == 0:
    return [0.0 for _ in pts]
  return [x / total for x in d]


def left_tangent(pts: List[Point]) -> Point:
  return v_norm(v_sub(pts[1], pts[0]))


def right_tangent(pts: List[Point]) -> Point:
  return v_norm(v_sub(pts[-2], pts[-1]))


def generate_bezier(
    pts: List[Point], u: List[float], tHat1: Point, tHat2: Point
) -> Cubic:
  p0 = pts[0]
  p3 = pts[-1]

  C00 = C01 = C10 = C11 = 0.0
  X0 = X1 = 0.0

  for i in range(len(pts)):
    t = u[i]
    mt = 1.0 - t
    b0 = mt * mt * mt
    b1 = 3 * mt * mt * t
    b2 = 3 * mt * t * t
    b3 = t * t * t

    a1 = v_mul(tHat1, b1)
    a2 = v_mul(tHat2, b2)

    C00 += v_dot(a1, a1)
    C01 += v_dot(a1, a2)
    C10 += v_dot(a2, a1)
    C11 += v_dot(a2, a2)

    tmp = v_sub(
        pts[i],
        (
            p0[0] * (b0 + b1) + p3[0] * (b2 + b3),
            p0[1] * (b0 + b1) + p3[1] * (b2 + b3),
        ),
    )
    X0 += v_dot(a1, tmp)
    X1 += v_dot(a2, tmp)

  det = C00 * C11 - C01 * C10
  segLen = v_len(v_sub(p3, p0))

  if abs(det) > 1e-12:
    alpha1 = (X0 * C11 - X1 * C01) / det
    alpha2 = (C00 * X1 - C10 * X0) / det
  else:
    alpha1 = alpha2 = segLen / 3.0

  eps = 1e-6
  if alpha1 < eps or alpha2 < eps:
    alpha1 = alpha2 = segLen / 3.0

  p1 = v_add(p0, v_mul(tHat1, alpha1))
  p2 = v_add(p3, v_mul(tHat2, alpha2))
  return (p0, p1, p2, p3)


def find_max_error(pts: List[Point], curve: Cubic, u: List[float]) -> Tuple[float, int]:
  maxDist = -1.0
  split = len(pts) // 2
  for i in range(1, len(pts) - 1):
    q = bez3(*curve, u[i])
    dist = v_len(v_sub(q, pts[i]))
    if dist > maxDist:
      maxDist = dist
      split = i
  return maxDist, split


def reparameterize(
    pts: List[Point], curve: Cubic, u: List[float], iters: int = 5
) -> List[float]:
  # Newton-Raphson to find closest t for each point
  p0, p1, p2, p3 = curve
  out = u[:]
  for _ in range(iters):
    for i in range(1, len(pts) - 1):
      t = out[i]
      # clamp
      t = max(0.0, min(1.0, t))

      q = bez3(p0, p1, p2, p3, t)
      q1 = bez3_d1(p0, p1, p2, p3, t)
      q2 = bez3_d2(p0, p1, p2, p3, t)

      r = v_sub(q, pts[i])
      f = v_dot(r, q1)
      df = v_dot(q1, q1) + v_dot(r, q2)

      if df != 0.0:
        t_new = t - f / df
        out[i] = max(0.0, min(1.0, t_new))
  return out


def tangent_at_end(pts, k=6, is_left=True):
  """
  Robust endpoint tangent using k-point average.
  is_left=True -> tangent at pts[0]
  is_left=False -> tangent at pts[-1] (points outward)
  """
  n = len(pts)
  k = max(2, min(k, n - 1))
  if is_left:
    acc = (0.0, 0.0)
    for i in range(1, k + 1):
      acc = v_add(acc, v_sub(pts[i], pts[0]))
    return v_norm(acc)
  else:
    acc = (0.0, 0.0)
    for i in range(1, k + 1):
      acc = v_add(acc, v_sub(pts[-i - 1], pts[-1]))
    return v_norm(acc)


def center_tangent(pts, split, k=4):
  """
  Robust center tangent around split point using symmetric window.
  """
  n = len(pts)
  i0 = max(0, split - k)
  i1 = min(n - 1, split + k)
  acc = (0.0, 0.0)
  # accumulate forward-backward differences
  for i in range(i0 + 1, i1):
    acc = v_add(acc, v_sub(pts[i + 1], pts[i - 1]))
  return v_norm(acc)


def fit_open_segment_C1(pts, max_error, tHat1=None, tHat2=None, reparam_iters=5):
  """
  Schneider FitCurves + reparameterization, with tangent propagation
  to enforce C1 continuity across recursive splits.
  """
  if len(pts) < 2:
    return []
  if len(pts) == 2:
    p0, p3 = pts[0], pts[1]
    v = v_sub(p3, p0)
    return [(p0, v_add(p0, v_mul(v, 1 / 3)), v_add(p0, v_mul(v, 2 / 3)), p3)]

  # Robust endpoint tangents if not supplied
  if tHat1 is None:
    tHat1 = tangent_at_end(pts, k=6, is_left=True)
  if tHat2 is None:
    tHat2 = tangent_at_end(pts, k=6, is_left=False)

  u = chord_length_params(pts)
  curve = generate_bezier(pts, u, tHat1, tHat2)

  # Reparameterize + regenerate (stabilizes tangents & reduces wiggles)
  for _ in range(2):
    u = reparameterize(pts, curve, u, iters=reparam_iters)
    curve = generate_bezier(pts, u, tHat1, tHat2)

  maxDist, split = find_max_error(pts, curve, u)
  if maxDist <= max_error:
    return [curve]

  # Compute center tangent at split and propagate to both sides
  tHatC = center_tangent(pts, split, k=4)
  if v_len(tHatC) == 0.0:
    # fallback if degenerate
    tHatC = (
        v_norm(v_sub(pts[split + 1], pts[split - 1]))
        if 0 < split < len(pts) - 1
        else tHat1
    )

  left_pts = pts[: split + 1]
  right_pts = pts[split:]

  left = fit_open_segment_C1(
      left_pts, max_error, tHat1=tHat1, tHat2=tHatC, reparam_iters=reparam_iters
  )
  right = fit_open_segment_C1(
      right_pts,
      max_error,
      tHat1=v_mul(tHatC, -1.0),
      tHat2=tHat2,
      reparam_iters=reparam_iters,
  )
  return left + right


def fit_cubics_reparam(
    pts: List[Point], max_error: float, reparam_iters: int = 5
) -> List[Cubic]:
  """
  Schneider FitCurves with iterative reparameterization.
  pts is open polyline (>=2 points).
  max_error in same units as pts.
  """
  if len(pts) < 2:
    return []
  if len(pts) == 2:
    p0, p3 = pts[0], pts[1]
    v = v_sub(p3, p0)
    p1 = v_add(p0, v_mul(v, 1 / 3))
    p2 = v_add(p0, v_mul(v, 2 / 3))
    return [(p0, p1, p2, p3)]

  tHat1 = left_tangent(pts)
  tHat2 = right_tangent(pts)

  u = chord_length_params(pts)
  curve = generate_bezier(pts, u, tHat1, tHat2)

  # Reparameterize and regenerate a couple of times (major improvement)
  for _ in range(2):
    u = reparameterize(pts, curve, u, iters=reparam_iters)
    curve = generate_bezier(pts, u, tHat1, tHat2)

  maxDist, split = find_max_error(pts, curve, u)
  if maxDist <= max_error:
    return [curve]

  left = fit_cubics_reparam(
      pts[: split + 1], max_error, reparam_iters=reparam_iters)
  right = fit_cubics_reparam(
      pts[split:], max_error, reparam_iters=reparam_iters)
  return left + right


def angle_at(p_prev: Point, p: Point, p_next: Point) -> float:
  a = v_norm(v_sub(p_prev, p))
  b = v_norm(v_sub(p_next, p))
  # angle between vectors a and b (0..pi)
  d = max(-1.0, min(1.0, v_dot(a, b)))
  return math.acos(d)


def detect_corners(
    poly: List[Point], angle_deg: float = 50.0, min_sep: int = 8
) -> List[int]:
  """
  Detect corners on a closed polyline by angle threshold.
  Smaller angle => sharper corner.
  angle_deg: corners if angle < angle_deg
  min_sep: minimum index separation between corners (avoid clusters)
  Returns sorted list of indices in [0..n-1]
  """
  n = len(poly)
  if n < 10:
    return []

  thr = math.radians(angle_deg)
  candidates = []
  for i in range(n):
    a = angle_at(poly[(i - 1) % n], poly[i], poly[(i + 1) % n])
    if a < thr:
      candidates.append((a, i))

  # sort by sharpest first, then enforce min separation
  candidates.sort(key=lambda x: x[0])
  chosen = []
  for _, i in candidates:
    if all(min((i - j) % n, (j - i) % n) >= min_sep for j in chosen):
      chosen.append(i)

  chosen.sort()
  return chosen


def split_closed_poly_at_indices(
    poly: List[Point], cuts: List[int]
) -> List[List[Point]]:
  """
  Split closed polyline into open segments between cut indices.
  Each returned segment includes its endpoints.
  """
  n = len(poly)
  if not cuts:
    return [poly + [poly[0]]]

  cuts = sorted(set([c % n for c in cuts]))
  segments = []
  for k in range(len(cuts)):
    a = cuts[k]
    b = cuts[(k + 1) % len(cuts)]
    seg = []
    i = a
    seg.append(poly[i])
    while i != b:
      i = (i + 1) % n
      seg.append(poly[i])
    segments.append(seg)
  return segments


def resample_by_arclength(pts, ds, *, closed=False):
  pts = np.asarray(pts, float)
  if len(pts) < 2:
    return pts

  if closed:
    pts2 = np.vstack([pts, pts[0]])
  else:
    pts2 = pts

  d = np.sqrt(np.sum(np.diff(pts2, axis=0)**2, axis=1))
  s = np.concatenate([[0.0], np.cumsum(d)])
  L = s[-1]
  if L <= 0:
    return pts.copy()

  s_new = np.arange(0.0, L + 0.5*ds, ds)
  x_new = np.interp(s_new, s, pts2[:, 0])
  y_new = np.interp(s_new, s, pts2[:, 1])
  # out = np.column_stack([x_new, y_new])

  # if closed:
  #    out = out[:-1]  # avoid duplicating first point
  # return out

  resampled = list(zip(x_new.tolist(), y_new.tolist()))

  if closed:
    resampled = resampled[:-1]  # avoid duplicating first point

  return resampled


def resample_closed_polyline(poly: List[Point], step: float) -> List[Point]:
  if len(poly) < 3:
    return poly[:]

  # avoid duplicate closure point
  if poly[0] == poly[-1]:
    poly = poly[:-1]

  n = len(poly)
  seglens = []
  total = 0.0
  for i in range(n):
    L = v_len(v_sub(poly[(i + 1) % n], poly[i]))
    seglens.append(L)
    total += L
  if total == 0:
    return poly[:]

  m = max(16, int(total / step))
  out = []
  acc = 0.0
  i = 0
  for k in range(m):
    dist_target = (k * total) / m
    while acc + seglens[i] < dist_target:
      acc += seglens[i]
      i = (i + 1) % n
    t = (dist_target - acc) / max(1e-12, seglens[i])
    p0 = poly[i]
    p1 = poly[(i + 1) % n]
    out.append((p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t))
  return out


def fit_closed_contour_to_cubics_no_corners(
    poly: List[Point],
    resample_step: float,
    cfg: FitConfig,
    type: int = 1,
) -> Tuple[List[Cubic], List[Point], List, List, List]:
  """
  Fit a CLOSED contour without corner splitting:
    - resample closed loop
    - cut once to make an open polyline
    - run C1 Schneider fit on the open polyline
  Returns: (cubics, resampled_poly, segs, infls, extrema). segs/infls/extrema
  are only populated for type=2 -- see fit_closed_by_extrema_segments.
  """
  # 1) resample (closed loop)
  if resample_step > 0:
    # r = resample_closed_polyline(poly, step=resample_step)
    r = resample_by_arclength(poly, ds=resample_step)
  else:
    r = poly
  if len(r) < 4:
    return [], r, [], [], []

  # 2) choose a stable cut index (farthest from centroid)
  cx = sum(p[0] for p in r) / len(r)
  cy = sum(p[1] for p in r) / len(r)
  d2 = [(p[0] - cx) ** 2 + (p[1] - cy) ** 2 for p in r]
  k = int(np.argmax(d2))
  k = 0

  # 3) unwrap into an open polyline with explicit closure endpoint
  #    (start == end) so the fitter will close it smoothly via segments
  # open_pts = r[k:] + r[:k] + [r[k]]

  # 4) fit as open polyline
  # cubics = fit_open_segment_C1(open_pts, max_error=fit_err, reparam_iters=reparam_iters)
  # cubics = fit_cubics_reparam(open_pts, max_error=fit_err, reparam_iters=reparam_iters)
  #

  # cubics = fit_cubics_open(open_pts,tol=fit_err)
  segs, infls, extrema = [], [], []
  if type == 1:
    open_pts = r[k:] + r[:k] + [r[k]]
    tan = tangent_closed_windowed(open_pts, 0, prev_count=12, next_count=12)
    cubics, spans = fit_cubics_open_with_spans(
        open_pts,
        cfg,
        start_tangent=tan,
        end_tangent=tan,
    )
  elif type == 2:
    open_pts = r[k:] + r[:k]
    cubics, spans, segs, infls, extrema = fit_closed_by_extrema_segments(open_pts, cfg)

  return cubics, open_pts, segs, infls, extrema


def fit_closed_contour_to_cubics(
    poly: List[Point],
    resample_step: float,
    corner_angle_deg: float,
    corner_min_sep: int,
    fit_err: float,
    reparam_iters: int = 5,
) -> Tuple[List[Cubic], List[int], List[Point]]:
  """
  Returns: (cubics, corner_indices, resampled_poly)
  """
  # 1) resample
  r = resample_closed_polyline(poly, step=resample_step)

  # 2) corner split
  corners = detect_corners(
      r, angle_deg=corner_angle_deg, min_sep=corner_min_sep)
  segs = split_closed_poly_at_indices(r, corners)

  # 3) fit each open segment
  cubics: List[Cubic] = []
  for seg in segs:
    # ensure open with enough points
    if len(seg) < 2:
      continue
    # many segments are short; reduce noise by not overfitting
    # cubics.extend(fit_cubics_reparam(seg, max_error=fit_err, reparam_iters=reparam_iters))
    cubics.extend(
        fit_open_segment_C1(seg, max_error=fit_err,
                            reparam_iters=reparam_iters)
    )

  return cubics, corners, r


def crop_to_ink(bw):
  ys, xs = np.where(bw > 0)
  if len(xs) == 0:
    return bw
  x0, x1 = xs.min(), xs.max()
  y0, y1 = ys.min(), ys.max()
  return bw[y0: y1 + 1, x0: x1 + 1]


def blur_and_rethreshold(bw, sigma=0.8, thresh=127):
  f = bw.astype(np.float32)
  f = cv2.GaussianBlur(f, (0, 0), sigmaX=sigma, sigmaY=sigma)
  _, out = cv2.threshold(f, thresh, 255, cv2.THRESH_BINARY)
  return out.astype(np.uint8)


def find_largest_contour(bw_white_ink):
  # returns Nx2 float points
  contours, hier = cv2.findContours(
      bw_white_ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
  )
  if not contours:
    return None
  areas = [cv2.contourArea(c) for c in contours]
  c = contours[int(np.argmax(areas))].reshape(-1, 2).astype(np.float64)
  return c


def remove_duplicate_closure(pts, eps=1e-9):
  if (
      len(pts) >= 2
      and (abs(pts[0][0] - pts[-1][0]) + abs(pts[0][1] - pts[-1][1])) <= eps
  ):
    return pts[:-1]
  return pts


def resample_closed_polyline(pts, step):
  """
  pts: closed, no duplicate endpoint
  step: target spacing in same units as pts (e.g., pixels)
  """
  n = len(pts)
  if n < 3:
    return pts[:]

  # build cumulative lengths
  lens = [0.0]
  total = 0.0
  for i in range(n):
    a = pts[i]
    b = pts[(i + 1) % n]
    total += ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
    lens.append(total)

  if total <= 1e-12:
    return pts[:]

  m = max(3, int(total / step + 0.5))
  out = []
  for j in range(m):
    s = (j / m) * total
    # locate edge
    # linear search is OK for moderate sizes; can be optimized with bisect
    k = 0
    while k + 1 < len(lens) and lens[k + 1] < s:
      k += 1
    # edge k goes from pts[k % n] to pts[(k+1) % n]
    a = pts[k % n]
    b = pts[(k + 1) % n]
    seg_len = lens[k + 1] - lens[k]
    t = 0.0 if seg_len <= 1e-12 else (s - lens[k]) / seg_len
    out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
  return out


def taubin_smooth_closed_safe(
    pts,
    iterations=10,
    lam=0.10,
    mu=-0.11,
    corner_mask=None,
    max_move=2.0,  # cap per-iteration displacement (pixels)
):
  """
  pts: closed polyline WITHOUT duplicate endpoint
  corner_mask: list[bool] same length as pts; True => frozen point
  max_move: max displacement per point per substep (lam or mu pass)
  """
  pts = [list(p) for p in pts]
  n = len(pts)
  if n < 3:
    return [tuple(p) for p in pts]

  def lap(i, P):
    im = (i - 1) % n
    ip = (i + 1) % n
    return (P[im][0] - 2 * P[i][0] + P[ip][0], P[im][1] - 2 * P[i][1] + P[ip][1])

  def apply_step(P, alpha):
    Q = [p[:] for p in P]
    for i in range(n):
      if corner_mask and corner_mask[i]:
        continue
      dx, dy = lap(i, P)
      dx *= alpha
      dy *= alpha
      # limit motion
      L = (dx * dx + dy * dy) ** 0.5
      if L > max_move and L > 1e-12:
        s = max_move / L
        dx *= s
        dy *= s
      Q[i][0] += dx
      Q[i][1] += dy
    return Q

  for _ in range(iterations):
    P1 = apply_step(pts, lam)
    P2 = apply_step(P1, mu)
    pts = P2

  return [tuple(p) for p in pts]


def smooth_contour_for_bezier_fit(
    pts_closed_with_dup,
    *,
    step=0,  # resample spacing in px
    iterations=6,
    lam=0.30,
    mu=-0.35,
    corner_mask=None,
    max_move=13,
):
  pts = remove_duplicate_closure(pts_closed_with_dup)

  # resample to uniform spacing (this is the big fix)
  if step > 0:
    pts = resample_closed_polyline(pts, step=step)

  # if you have corners, recompute corner_mask AFTER resampling (important)
  # corner_mask = detect_corners_closed(pts, ...)

  pts = taubin_smooth_closed_safe(
      pts,
      iterations=iterations,
      lam=lam,
      mu=mu,
      corner_mask=corner_mask,
      max_move=max_move,
  )

  return pts  # closed polyline WITHOUT duplicate end


def smooth_edge_via_sdf(bin_img, sigma=0.9):
  """
  bin_img: uint8, 0 background, 255 foreground
  returns: smoothed binary image
  """

  # 1) distance inside
  inside = cv2.distanceTransform(bin_img, cv2.DIST_L2, 5)

  # 2) distance outside
  outside = cv2.distanceTransform(255 - bin_img, cv2.DIST_L2, 5)

  # 3) signed distance field
  sdf = inside - outside

  # 4) smooth SDF (small sigma!)
  sdf_s = cv2.GaussianBlur(sdf, (0, 0), sigmaX=sigma, sigmaY=sigma)

  # 5) rethreshold at zero level
  smoothed = np.where(sdf_s >= 0, 255, 0).astype(np.uint8)

  return smoothed


def simplify_contour(pts: List[Point], tolerance: float) -> List[Point]:
  """
  Douglas-Peucker algorithm to reduce point density while maintaining shape.
  Higher tolerance = fewer points (smoother), Lower tolerance = more points (detailed).
  """
  pts_arr = np.array(pts)

  def _rdp(arr: np.ndarray, tol: float) -> np.ndarray:
    if len(arr) < 3:
      return arr

    start, end = arr[0], arr[-1]
    line_vec = end - start
    norm_sq = np.sum(line_vec**2)

    # Calculate perpendicular distance from all points to the start-end line
    if norm_sq == 0:
      dists = np.linalg.norm(arr[1:-1] - start, axis=1)
    else:
      # Distance formula for 2D points to line segment
      dists = np.abs(np.cross(line_vec, start - arr[1:-1])) / np.sqrt(norm_sq)

    max_idx = np.argmax(dists) + 1  # +1 to account for slice offset

    if dists[max_idx - 1] > tol:
      # Point is beyond tolerance; keep it and recurse on both halves
      left = _rdp(arr[: max_idx + 1], tol)
      right = _rdp(arr[max_idx:], tol)
      return np.vstack((left[:-1], right))
    else:
      # All intermediate points are within tolerance; discard them
      return np.array([start, end])

  result = _rdp(pts_arr, tolerance)
  return [tuple(map(float, p)) for p in result]


def read_bw_white_is_ink(path, thresh=127):
  img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  if img is None:
    raise RuntimeError("Cannot read image")
  _, bw = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
  return bw


def read_bw_grayscale(path, thresh=127):
  g0 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  if g0 is None:
    raise FileNotFoundError(f"Could not read image: {path}")
  g0 = g0.astype(np.uint8)
  return g0
