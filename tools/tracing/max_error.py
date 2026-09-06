from typing import List, Tuple, Optional, Callable
import math

import numpy as np
from numpy import quantile

Point = Tuple[float, float]
Cubic = Tuple[Point, Point, Point, Point]

# ---------- basic vector ops ----------


def v_add(a: Point, b: Point) -> Point: return (a[0] + b[0], a[1] + b[1])
def v_sub(a: Point, b: Point) -> Point: return (a[0] - b[0], a[1] - b[1])
def v_mul(a: Point, s: float) -> Point: return (a[0] * s, a[1] * s)
def v_dot(a: Point, b: Point) -> float: return a[0] * b[0] + a[1] * b[1]
def v_len2(a: Point) -> float: return v_dot(a, a)
def v_len(a: Point) -> float: return math.sqrt(v_len2(a))
def v_dist(a: Point, b: Point) -> float: return v_len(v_sub(a, b))


def clamp(x: float, lo: float, hi: float) -> float:
  return lo if x < lo else hi if x > hi else x

# ---------- cubic Bézier eval + derivatives ----------


def bezier_q(c: Cubic, t: float) -> Point:
  p0, p1, p2, p3 = c
  mt = 1.0 - t
  a = mt * mt * mt
  b = 3.0 * mt * mt * t
  d = 3.0 * mt * t * t
  e = t * t * t
  return (
      a * p0[0] + b * p1[0] + d * p2[0] + e * p3[0],
      a * p0[1] + b * p1[1] + d * p2[1] + e * p3[1],
  )


def bezier_qprime(c: Cubic, t: float) -> Point:
  p0, p1, p2, p3 = c
  mt = 1.0 - t
  # 3 * [(1-t)^2 (p1-p0) + 2(1-t)t (p2-p1) + t^2 (p3-p2)]
  term1 = v_mul(v_sub(p1, p0), mt * mt)
  term2 = v_mul(v_sub(p2, p1), 2.0 * mt * t)
  term3 = v_mul(v_sub(p3, p2), t * t)
  return v_mul(v_add(v_add(term1, term2), term3), 3.0)


def bezier_qprime2(c: Cubic, t: float) -> Point:
  p0, p1, p2, p3 = c
  mt = 1.0 - t
  # 6 * [(1-t)(p2 - 2p1 + p0) + t(p3 - 2p2 + p1)]
  a = v_add(v_sub(p2, v_mul(p1, 2.0)), p0)  # p2 - 2p1 + p0
  b = v_add(v_sub(p3, v_mul(p2, 2.0)), p1)  # p3 - 2p2 + p1
  return v_mul(v_add(v_mul(a, mt), v_mul(b, t)), 6.0)

# ---------- closest point on cubic (approx, fast) ----------


def _closest_t_on_cubic_newton(
    curve: Cubic,
    x: Point,
    t0: float,
    *,
    iters: int = 8,
    eps: float = 1e-12,
) -> float:
  """
  Minimize f(t) = ||B(t) - x||^2 using Newton on f'(t)=0:
    g(t) = (B(t)-x)·B'(t)
    g'(t) = B'(t)·B'(t) + (B(t)-x)·B''(t)
    t <- t - g/g'
  """
  t = clamp(t0, 0.0, 1.0)
  for _ in range(iters):
    bt = bezier_q(curve, t)
    d1 = bezier_qprime(curve, t)
    d2 = bezier_qprime2(curve, t)
    r = v_sub(bt, x)

    g = v_dot(r, d1)
    gp = v_dot(d1, d1) + v_dot(r, d2)

    if abs(gp) < eps:
      break

    dt = g / gp
    t_new = t - dt
    t_new = clamp(t_new, 0.0, 1.0)

    # if step is tiny, stop
    if abs(t_new - t) < 1e-10:
      t = t_new
      break

    t = t_new
  return t


def _closest_t_on_cubic_bruteforce_seed(
    curve: Cubic,
    x: Point,
    *,
    samples: int = 33,
) -> float:
  """Coarse seed by sampling; returns t with smallest squared distance."""
  best_t = 0.0
  best_d2 = float("inf")
  for i in range(samples):
    t = i / (samples - 1)
    bt = bezier_q(curve, t)
    d2 = v_len2(v_sub(bt, x))
    if d2 < best_d2:
      best_d2 = d2
      best_t = t
  return best_t


def closest_point_on_cubic(
    curve: Cubic,
    x: Point,
    *,
    seed_t: Optional[float] = None,
    seed_samples: int = 33,
    newton_iters: int = 8,
) -> Tuple[float, Point, float]:
  """
  Returns (t*, B(t*), distance) where t* approximates the orthogonal projection.
  Strategy:
    - seed with provided u[i] if available, else brute-force sampling
    - refine with Newton on squared-distance optimality condition
  """
  if seed_t is None:
    seed_t = _closest_t_on_cubic_bruteforce_seed(
        curve, x, samples=seed_samples)
  else:
    # optional: still improve seed a bit by local sampling around it if you want
    seed_t = clamp(seed_t, 0.0, 1.0)

  t_star = _closest_t_on_cubic_newton(curve, x, seed_t, iters=newton_iters)
  p_star = bezier_q(curve, t_star)
  dist = v_dist(p_star, x)
  return t_star, p_star, dist

# ---------- the same projection, all points at once ----------
#
# The scalar routines above are the readable definition; these are what the
# fitter actually calls. Every point of an arc is projected onto the same
# cubic, so the work is one array operation rather than a few million
# tuple-at-a-time ones -- profiling a corpus build put 75% of it here, almost
# all of it Python call overhead rather than arithmetic.
#
# The expressions below are written to match the scalar ones term by term,
# including the order of additions, so the two agree bit for bit. Change one
# and you must change the other.


def _bezier_q_many(c: Cubic, t: np.ndarray) -> np.ndarray:
  p0, p1, p2, p3 = c
  mt = 1.0 - t
  a = mt * mt * mt
  b = 3.0 * mt * mt * t
  d = 3.0 * mt * t * t
  e = t * t * t
  return np.stack([
      a * p0[0] + b * p1[0] + d * p2[0] + e * p3[0],
      a * p0[1] + b * p1[1] + d * p2[1] + e * p3[1]], axis=-1)


def _bezier_qprime_many(c: Cubic, t: np.ndarray) -> np.ndarray:
  p0, p1, p2, p3 = c
  mt = 1.0 - t
  t1 = np.stack([(p1[0] - p0[0]) * (mt * mt), (p1[1] - p0[1]) * (mt * mt)], -1)
  scale = 2.0 * mt * t
  t2 = np.stack([(p2[0] - p1[0]) * scale, (p2[1] - p1[1]) * scale], -1)
  t3 = np.stack([(p3[0] - p2[0]) * (t * t), (p3[1] - p2[1]) * (t * t)], -1)
  return (t1 + t2 + t3) * 3.0


def _bezier_qprime2_many(c: Cubic, t: np.ndarray) -> np.ndarray:
  p0, p1, p2, p3 = c
  mt = 1.0 - t
  a0, a1 = (p2[0] - p1[0] * 2.0) + p0[0], (p2[1] - p1[1] * 2.0) + p0[1]
  b0, b1 = (p3[0] - p2[0] * 2.0) + p1[0], (p3[1] - p2[1] * 2.0) + p1[1]
  return np.stack([a0 * mt + b0 * t, a1 * mt + b1 * t], axis=-1) * 6.0


def _seed_params(curve: Cubic, pts: np.ndarray, samples: int) -> np.ndarray:
  """Coarse seed per point, the sampled-argmin of the scalar version."""
  grid = np.arange(samples, dtype=float) / (samples - 1)
  on_curve = _bezier_q_many(curve, grid)                     # (samples, 2)
  delta = pts[:, None, :] - on_curve[None, :, :]             # (n, samples, 2)
  d2 = delta[..., 0] * delta[..., 0] + delta[..., 1] * delta[..., 1]
  return grid[np.argmin(d2, axis=1)]                         # argmin ties -> first


def _project_many(curve: Cubic, pts: np.ndarray,
                  seeds: Optional[np.ndarray], seed_samples: int,
                  newton_iters: int, eps: float = 1e-12
                  ) -> Tuple[np.ndarray, np.ndarray]:
  """(t*, distance) per point. Mirrors closest_point_on_cubic exactly.

  Newton runs over every point together; a point that would have hit one of
  the scalar loop's two break conditions is masked out and simply stops
  updating, which is the same thing as leaving the loop early.
  """
  if pts.size == 0:
    return np.zeros(0), np.zeros(0)
  t = (_seed_params(curve, pts, seed_samples) if seeds is None
       else np.clip(np.asarray(seeds, dtype=float), 0.0, 1.0))
  active = np.ones(t.shape, dtype=bool)
  for _ in range(newton_iters):
    if not active.any():
      break
    r = _bezier_q_many(curve, t) - pts
    d1 = _bezier_qprime_many(curve, t)
    d2 = _bezier_qprime2_many(curve, t)
    g = r[:, 0] * d1[:, 0] + r[:, 1] * d1[:, 1]
    gp = ((d1[:, 0] * d1[:, 0] + d1[:, 1] * d1[:, 1])
          + (r[:, 0] * d2[:, 0] + r[:, 1] * d2[:, 1]))
    flat = np.abs(gp) < eps                    # scalar: break, t unchanged
    stepping = active & ~flat
    safe = np.where(flat, 1.0, gp)
    t_new = np.clip(t - g / safe, 0.0, 1.0)
    tiny = np.abs(t_new - t) < 1e-10           # scalar: take the step, then break
    t = np.where(stepping, t_new, t)
    active = stepping & ~tiny
  delta = _bezier_q_many(curve, t) - pts
  return t, np.sqrt(delta[:, 0] * delta[:, 0] + delta[:, 1] * delta[:, 1])


def _as_points(pts: List[Point]) -> np.ndarray:
  return np.asarray(pts, dtype=float).reshape(len(pts), 2)


# ---------- drop-in replacement for your function ----------


def compute_max_ortho_error(
    pts: List[Point],
    u: List[float],
    curve: Cubic,
    *,
    seed_samples: int = 33,
    newton_iters: int = 8,
) -> Tuple[float, int]:
  """
  Error is orthogonal distance to curve (approx projection), rather than
  distance at the fixed parameter u[i]. Uses u[i] as initial guess for the
  projection.
  """
  if not pts or not u:
    return -1.0, len(pts) // 2
  _, errs = _project_many(curve, _as_points(pts), np.asarray(u, dtype=float),
                          seed_samples, newton_iters)
  split = int(np.argmax(errs))          # argmax ties -> first, as the loop did
  return float(errs[split]), split

# ---------- optional: also return refined parameters (useful) ----------


def compute_max_ortho_error_and_refined_u(
    pts: List[Point],
    u: List[float],
    curve: Cubic,
    *,
    seed_samples: int = 33,
    newton_iters: int = 2,
) -> Tuple[float, int, List[float]]:
  """
  Returns max orthogonal error, split index, and refined parameters u* (projections).
  Useful if you want to reparameterize using true closest-point t estimates.

  The unconstrained per-point nearest-point search below has no ordering
  guarantee: on a strongly-bowed curve (e.g. an S-shape forced by strict end
  tangents), the curve can loop close to itself, so the independent global
  search for an early-index point can converge to a *late* curve parameter
  (or vice versa) purely because it's geometrically nearer -- silently
  under-reporting error and skipping a split the fit actually needs. Since
  `pts` are always in polyline order and any single fitted arc must
  correspond to them monotonically, walk the refined parameters forward and
  clamp any regression to the previous point's parameter, recomputing that
  point's error at the clamped (still-genuine, on-curve) position instead of
  the invalid closer-but-wrong-branch one.
  """
  if not pts:
    return -1.0, 0, list(u)
  points = _as_points(pts)
  u_ref, errs = _project_many(
      curve, points, None if seed_samples > 0 else np.asarray(u, dtype=float),
      seed_samples, newton_iters)

  # The running maximum is the loop's prev_t: a parameter that regresses is
  # pulled up to the previous point's, and only those points need their error
  # recomputed at the clamped position.
  monotone = np.maximum.accumulate(np.concatenate(([0.0], u_ref)))[1:]
  clamped = monotone > u_ref
  if clamped.any():
    delta = _bezier_q_many(curve, monotone[clamped]) - points[clamped]
    errs = errs.copy()
    errs[clamped] = np.sqrt(delta[:, 0] * delta[:, 0]
                            + delta[:, 1] * delta[:, 1])
    u_ref = monotone

  split = int(np.argmax(errs))
  return float(errs[split]), split, u_ref.tolist()


def clampi(x: int, lo: int, hi: int) -> int:
  return lo if x < lo else hi if x > hi else x


def median_window(vals: List[float], i: int, w: int) -> float:
  lo = max(0, i - w)
  hi = min(len(vals), i + w + 1)
  win = sorted(vals[lo:hi])
  m = len(win) // 2
  return win[m] if (len(win) % 2) else 0.5 * (win[m-1] + win[m])


def smooth_median(vals: List[float], w: int) -> List[float]:
  return [median_window(vals, i, w) for i in range(len(vals))]


def largest_true_run(mask: List[bool]) -> Optional[Tuple[int, int]]:
  """Return (a,b) inclusive of largest contiguous True run, or None."""
  best = None
  best_len = 0
  i = 0
  n = len(mask)
  while i < n:
    if not mask[i]:
      i += 1
      continue
    a = i
    while i < n and mask[i]:
      i += 1
    b = i - 1
    L = b - a + 1
    if L > best_len:
      best_len = L
      best = (a, b)
  return best

# --- You already have closest_point_on_cubic from earlier ---
# def closest_point_on_cubic(curve: Cubic, x: Point, *, seed_t: float, seed_samples=33, newton_iters=8) -> Tuple[float, Point, float]:
#     ...


def compute_ortho_errors(
    pts: List[Point],
    u: List[float],
    curve: Cubic,
    *,
    seed_samples: int = 33,
    newton_iters: int = 8,
) -> List[float]:
  if not pts:
    return []
  _, errs = _project_many(curve, _as_points(pts), np.asarray(u, dtype=float),
                          seed_samples, newton_iters)
  return errs.tolist()


def compute_split_valley_near_bad_region(
    pts: List[Point],
    u: List[float],
    curve: Cubic,
    *,
    q: float = 0.8,            # robust metric percentile
    smooth_w: int = 4,          # median smoothing radius on error signal
    expand: int = 200,            # search margin around bad cluster
    min_end_margin: int = 20,    # don't split too close to ends
    seed_samples: int = 33,
    newton_iters: int = 2,
) -> Tuple[float, int]:
  """
  Returns (robust_error_metric, split_index)

  robust_error_metric = q-quantile of orthogonal errors (e.g. p90)
  split chosen as a local minimum ("best fit") of smoothed errors
  near the main region of large errors.
  """
  n = len(pts)
  if n < 5:
    return 0.0, n // 2

  err = compute_ortho_errors(pts, u, curve, seed_samples=seed_samples, newton_iters=newton_iters)
  metric = quantile(err, q)

  # Smooth the error to suppress single-point spikes (bumps/hollows)
  err_s = smooth_median(err, smooth_w)
  err_s = err

  # Define "bad" region using the unsmoothed threshold (or smoothed—either works; unsmoothed is stricter)
  thr = metric
  bad = [e >= thr for e in err]

  run = largest_true_run(bad)
  # run = None
  if run is None:
    # Nothing above threshold (degenerate): fall back to robust-min split (global minimum of smoothed)
    i0 = min_end_margin
    i1 = n - 1 - min_end_margin
    split = min(range(i0, i1 + 1), key=lambda i: err_s[i])
    return metric, split

  a, b = run

  # Search for a "valley" (best local fit) near the bad run.
  # Expanding gives you a valley on the boundary of the bad region, not at the spike peak.
  i0 = min(max(min_end_margin, a - expand), n-1)
  i1 = min(n - 1 - min_end_margin, b + expand)
  if i1 < i0:
    i1 = i0

  # Pick the index with minimum smoothed error in [i0..i1]
  split = min(range(i0, i1 + 1), key=lambda i: err_s[i])

  return metric, split


def choose_split_minimax_pquant(
    pts: List[Point],
    u: List[float],
    curve: Cubic,
    *,
    q: float = 0.80,             # "90% error" = p90
    min_seg_pts: int = 50,        # don't allow tiny segments
    candidates: Optional[List[int]] = None,  # if provided, only evaluate these split indices
    window: Optional[Tuple[int, int]] = None,  # optionally restrict splits to [a,b] inclusive
    seed_samples: int = 33,
    newton_iters: int = 8,
) -> Tuple[float, int, float]:
  """
  Optional upgrade:
    score(s) = max( p_q(err[0:s+1]), p_q(err[s:n]) )
  Choose s minimizing score(s). This tends to avoid splitting at a bump peak and
  instead finds a split that improves BOTH sides in a robust way.

  Returns (best_score, best_split_index)

  Notes:
    - Split index s is included in left side by default (0..s), right is (s..n-1).
      If you prefer excluding s from left or right, adjust slices accordingly.
    - This does NOT refit cubics; it's a split chooser for the already computed curve.
  """
  n = len(pts)
  if n < 2 * min_seg_pts + 1:
    return 0.0, n // 2, 0.0

  err = compute_ortho_errors(pts, u, curve, seed_samples=seed_samples, newton_iters=newton_iters)
  metric = quantile(err, q)

  # Split candidates
  if candidates is None:
    a = min_seg_pts - 1
    b = n - min_seg_pts
    if window is not None:
      a = max(a, window[0])
      b = min(b, window[1])
    cand = list(range(a, b + 1))
  else:
    cand = [c for c in candidates if min_seg_pts - 1 <= c <= n - min_seg_pts]
    if window is not None:
      cand = [c for c in cand if window[0] <= c <= window[1]]
    if not cand:
      return 0.0, n // 2, 0.0

  best_s = cand[len(cand) // 2]
  best_score = float("inf")

  # Naive O(K * n log n) if you use full quantile sorting each time,
  # but K is small if you pass "candidates" (recommended).
  for s in cand:
    left = err[0:s+1]
    right = err[s:n]  # include s on the right too to be symmetric; change if desired

    # Ensure both sides have enough points
    if len(left) < min_seg_pts or len(right) < min_seg_pts:
      continue

    qL = quantile(left, q)
    qR = quantile(right, q)
    score = max(qL, qR)

    # Optional tiny tie-breaker: prefer splits that also reduce overall robust error
    # score2 = score + 1e-6 * (qL + qR)

    if score < best_score:
      best_score = score
      best_s = s

  return metric, best_s, best_score
