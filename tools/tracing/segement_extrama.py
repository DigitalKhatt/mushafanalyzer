import dataclasses
import math
from typing import List, Optional, Tuple

from bezier_fit import FitConfig, fit_cubics_open_with_spans, pca_tangent_open_by_length, tangent_windowed_difference, tangent_windowed_difference_by_length
from detect_extrema_by_arcs_closed import detect_extrema_by_arcs_closed
from straight_segments import InflectionFromStraights, StraightSegment, get_straights_segs, min_turning_points_dp, straight_segments_polyline, inflections_from_straights_open_window, fit_segments_rdp
from remove_noise import detect_zigzag_noise_runs
from inflections_dp import detect_real_inflections_multiscale_dp

Point = Tuple[float, float]
Cubic = Tuple[Point, Point, Point, Point]
Span = Tuple[int, int]


def _wrap_pts(pts: List[Point]) -> List[Point]:
  # Remove duplicate closure if present
  if len(pts) >= 2 and pts[0] == pts[-1]:
    return pts[:-1]
  return pts


def v_sub(a: Point, b: Point) -> Point:
  return (a[0] - b[0], a[1] - b[1])


def v_dot(a: Point, b: Point) -> float:
  return a[0] * b[0] + a[1] * b[1]


def v_len(a: Point) -> float:
  return math.hypot(a[0], a[1])


def v_mul(a: Point, s: float) -> Point:
  return (a[0] * s, a[1] * s)


def v_norm(a: Point) -> Point:
  L = v_len(a)
  return (0.0, 0.0) if L <= 1e-12 else (a[0] / L, a[1] / L)


def tangent_closed_windowed(
    pts_closed: List[Point],
    idx: int,
    prev_count: int = 12,
    next_count: int = 12,
) -> Point:
  pts = _wrap_pts(pts_closed)
  n = len(pts)
  idx %= n

  tx = 0.0
  ty = 0.0

  for k in range(1, prev_count + 1):
    i = (idx - k) % n
    w = float(k)
    tx += w * (pts[idx][0] - pts[i][0])
    ty += w * (pts[idx][1] - pts[i][1])

  for k in range(1, next_count + 1):
    i = (idx + k) % n
    w = float(k)
    tx += w * (pts[i][0] - pts[idx][0])
    ty += w * (pts[i][1] - pts[idx][1])

  t = v_norm((tx, ty))
  # orient with immediate forward step
  fwd = v_sub(pts[(idx + 1) % n], pts[idx])
  if v_len(fwd) > 1e-12 and v_dot(t, fwd) < 0.0:
    t = v_mul(t, -1.0)
  return t


def extract_arc_points_closed(
    pts_closed: List[Point], a: int, b: int
) -> Tuple[List[Point], int]:
  """
  Extract points along closed ring from index a to b inclusive (wrapping forward).
  Returns (arc_points, base_index_i0_in_ring)
  base_index_i0 is the ring index of arc_points[0] (used to map spans).
  """
  pts = _wrap_pts(pts_closed)
  n = len(pts)
  a %= n
  b %= n
  arc = [pts[a]]
  i = a
  while i != b:
    i = (i + 1) % n
    arc.append(pts[i])
  return arc, a


def _unwrap_closed(pts):
  # remove duplicate closure point
  if len(pts) >= 2 and pts[0] == pts[-1]:
    return pts[:-1]
  return pts


def _is_window_extremum_1d(
    values: List[float], i: int, p: int, n: int, eps: float, mode: str
) -> bool:
  N = len(values)
  v = values[i]

  if mode == "max":
    for k in range(1, p + 1):
      if v < values[(i - k) % N] - eps:
        return False
    for k in range(1, n + 1):
      if v < values[(i + k) % N] - eps:
        return False
  else:  # "min"
    for k in range(1, p + 1):
      if v > values[(i - k) % N] + eps:
        return False
    for k in range(1, n + 1):
      if v > values[(i + k) % N] + eps:
        return False

  # require at least one strict inequality (avoid picking huge plateaus)
  strict = False
  for k in range(1, max(p, n) + 1):
    if mode == "max":
      if v > values[(i - k) % N] + eps or v > values[(i + k) % N] + eps:
        strict = True
        break
    else:
      if v < values[(i - k) % N] - eps or v < values[(i + k) % N] - eps:
        strict = True
        break
  return strict


def _is_browse_extremum_1d(
    pts: List[Tuple[float, float]], i: int, h: int, v: int, eps: float, mode: str, min_arc_height: float
) -> Tuple[bool, float, float]:
  N = len(pts)
  vh = pts[i][h]
  vv = pts[i][v]

  if mode == "max":
    lp = 1
    lFound = False
    while vh >= pts[(i - lp) % N][h] - eps and lp < N:
      if vh - pts[(i - lp) % N][h] > min_arc_height:
        lFound = True
        break
      lp = lp + 1
    if not lFound:
      return (False, 0, 0)
    rp = 1
    rFound = False
    while vh >= pts[(i + rp) % N][h] - eps and rp < N:
      if vh - pts[(i + rp) % N][h] > min_arc_height:
        rFound = True
        break
      rp = rp + 1
    if not rFound:
      return (False, 0, 0)
    return (True, lp, rp)
  else:  # "min"
    lp = 1
    lFound = False
    while vh <= pts[(i - lp) % N][h] + eps and lp < N:
      if pts[(i - lp) % N][h] - vh > min_arc_height:
        lFound = True
        break
      lp = lp + 1
    if not lFound:
      return (False, 0, 0)
    rp = 1
    rFound = False
    while vh <= pts[(i + rp) % N][h] + eps and rp < N:
      if pts[(i + rp) % N][h] - vh > min_arc_height:
        rFound = True
        break
      rp = rp + 1
    if not rFound:
      return (False, 0, 0)
    return (True, lp, rp)


def find_midpoint(
    pts: List[Point],
    pos: int,
    xory: int,
    left: int,
    right: int
) -> List[Tuple[int, str]]:

  N = len(pts)
  v = pts[pos][xory]
  eps = 1
  yorx = (xory + 1) % 2

  lp = pos
  while abs(v - pts[(lp - 1) % N][xory]) <= eps and (lp - 1) >= pos - left:
    lp = lp - 1

  rp = pos
  while abs(v - pts[(rp + 1) % N][xory]) <= eps and (rp + 1) <= pos + right:
    rp = rp + 1

  # TODO use mean distance not index
  newPos = ((lp + rp) // 2) % N

  # TODO modulo
  diff = newPos - pos

  return newPos, left - diff, right + diff


def detect_browse_extrema_closed_typed(
    pts_closed: List[Point],
    *,
    eps: float = 2,
    min_arc_height: float,
) -> List[Tuple[int, str]]:

  picked: List[Tuple[int, str]] = []

  pts = _unwrap_closed(pts_closed)
  N = len(pts)

  i = 0
  while i < N:
    is_xmax, left, right = _is_browse_extremum_1d(
        pts, i, 0, 1, eps, "max", min_arc_height)
    if is_xmax:
      newi, _, _ = find_midpoint(pts, i, 0, left, right)
      picked.append((newi, "xmax"))
      i = i + right
      continue

    is_xmin, left, right = _is_browse_extremum_1d(
        pts, i, 0, 1, eps, "min", min_arc_height)
    if is_xmin:
      newi, _, _ = find_midpoint(pts, i, 0, left, right)
      picked.append((newi, "xmin"))
      i = i + right
      continue

    is_ymax, left, right = _is_browse_extremum_1d(
        pts, i, 1, 0, eps, "max", min_arc_height)
    if is_ymax:
      newi, _, _ = find_midpoint(pts, i, 1, left, right)
      picked.append((newi, "ymax"))
      i = i + right
      continue

    is_ymin, left, right = _is_browse_extremum_1d(
        pts, i, 1, 0, eps, "min", min_arc_height)
    if is_ymin:
      newi, _, _ = find_midpoint(pts, i, 1, left, right)
      picked.append((newi, "ymin"))
      i = i + right
      continue

    i = i + 1

  return picked


def _angle_deg(u: Point, v: Point) -> float:
  ux, uy = u
  vx, vy = v
  nu = math.hypot(ux, uy)
  nv = math.hypot(vx, vy)
  if nu < 1e-12 or nv < 1e-12:
    return 0.0
  d = (ux * vx + uy * vy) / (nu * nv)
  d = max(-1.0, min(1.0, d))
  return math.degrees(math.acos(d))


def _local_turning_max_deg(pts: List[Point], i: int, radius: int) -> float:
  """
  Max turning angle in a neighborhood around i (closed indexing).
  If it's tiny, the region is nearly straight.
  """
  N = len(pts)
  mx = 0.0
  for k in range(-radius, radius + 1):
    j = (i + k) % N
    jm = (j - 1) % N
    jp = (j + 1) % N
    e0 = (pts[j][0] - pts[jm][0], pts[j][1] - pts[jm][1])
    e1 = (pts[jp][0] - pts[j][0], pts[jp][1] - pts[j][1])
    mx = max(mx, _angle_deg(e0, e1))
  print("mx=" + str(mx))
  return mx


def _local_line_deviation_max(
    pts: List[Point],
    i: int,
    radius: int,
) -> float:
  """
  Returns max perpendicular distance (in point units) of points in the window
  to the best-fit line through that window (PCA).
  Works on CLOSED polyline (wrap indexing).
  """
  N = len(pts)
  # collect window
  win = [pts[(i + k) % N] for k in range(-radius, radius + 1)]
  if len(win) < 3:
    return 0.0

  cx = sum(p[0] for p in win) / len(win)
  cy = sum(p[1] for p in win) / len(win)

  sxx = sum((p[0] - cx) * (p[0] - cx) for p in win)
  syy = sum((p[1] - cy) * (p[1] - cy) for p in win)
  sxy = sum((p[0] - cx) * (p[1] - cy) for p in win)

  # principal direction of covariance (largest eigenvector)
  if abs(sxy) < 1e-12:
    dx, dy = (1.0, 0.0) if sxx >= syy else (0.0, 1.0)
  else:
    tr = sxx + syy
    det = sxx * syy - sxy * sxy
    lam = 0.5 * (tr + math.sqrt(max(0.0, tr * tr - 4 * det)))
    dx, dy = (lam - syy, sxy)

  # normalize
  L = math.hypot(dx, dy)
  if L < 1e-12:
    return 0.0
  dx /= L
  dy /= L

  # perpendicular unit vector
  nx, ny = -dy, dx

  # max perpendicular distance to line through centroid
  max_dev = 0.0
  for x, y in win:
    d = abs((x - cx) * nx + (y - cy) * ny)
    if d > max_dev:
      max_dev = d

  return max_dev


def detect_windowed_extrema_closed_typed(
    pts_closed: List[Point],
    *,
    p: int = 8,
    n: int = 8,
    eps: float = 2,
    min_sep: int = 30,
    # neighborhood (in indices) to test straightness
    straight_radius: int = 20,
    # if max dev in pixels <= this, treat region as straight
    straight_dev_max: float = 10.0,
    # how much larger suppression radius becomes on straight regions
    straight_sep_mult: float = 2.0,
    min_strength: int = 20,
    max_count: int = 5000,
) -> List[Tuple[int, str]]:
  """
  Returns a list of (idx, etype) where etype in {'xmax','xmin','ymax','ymin'}.
  idx refers to the UNIQUE closed ring indexing (duplicate endpoint removed).
  """
  pts = _unwrap_closed(pts_closed)
  N = len(pts)
  if N < p + n + 3:
    return [(0, "ymax")] if N > 0 else []

  xs = [pt[0] for pt in pts]
  ys = [pt[1] for pt in pts]

  # gather candidates with their best type + strength
  cand: List[Tuple[int, str, float]] = []
  for i in range(N):
    is_xmax = _is_window_extremum_1d(xs, i, p, n, eps, "max")
    is_xmin = _is_window_extremum_1d(xs, i, p, n, eps, "min")
    is_ymax = _is_window_extremum_1d(ys, i, p, n, eps, "max")
    is_ymin = _is_window_extremum_1d(ys, i, p, n, eps, "min")

    if not (is_xmax or is_xmin or is_ymax or is_ymin):
      continue

    # strength: deviation from neighborhood average on that axis
    def sx() -> float:
      return abs(xs[i] - 0.5 * (xs[(i - p) % N] + xs[(i + n) % N]))

    def sy() -> float:
      return abs(ys[i] - 0.5 * (ys[(i - p) % N] + ys[(i + n) % N]))

    options = []
    if is_xmax:
      options.append(("xmax", sx()))
    if is_xmin:
      options.append(("xmin", sx()))
    if is_ymax:
      options.append(("ymax", sy()))
    if is_ymin:
      options.append(("ymin", sy()))

    options.sort(key=lambda t: t[1], reverse=True)
    etype, strength = options[0]

    if strength > min_strength:
      cand.append((i, etype, strength))
      if len(cand) > max_count:
        break

  if not cand:
    return [(0, "ymax")]

  # Non-maximum suppression in index space: keep strongest in each neighborhood
  cand.sort(key=lambda t: t[2], reverse=True)  # by strength desc
  blocked = [False] * N
  picked: List[Tuple[int, str]] = []

  for i, etype, _ in cand:
    if blocked[i]:
      continue
    picked.append((i, etype))
    # Decide suppression radius: larger on locally straight regions
    dev = _local_line_deviation_max(pts, i, straight_radius)
    if dev <= straight_dev_max:
      rad = int(max(min_sep, round(min_sep * straight_sep_mult)))
    else:
      rad = min_sep

    for d in range(-rad, rad + 1):
      blocked[(i + d) % N] = True

  picked.sort(key=lambda t: t[0])
  return picked


def _cyc_dist(a: int, b: int, N: int) -> int:
  """cyclic index distance on ring of size N."""
  d = (a - b) % N
  return min(d, N - d)


def _find_straight_runs(mask: List[bool]) -> List[Tuple[int, int, int]]:
  """
  mask length N. Returns list of runs (start, end, length) on a ring.
  Each run is represented in cyclic coordinates with start..end inclusive in circular sense.
  We return them as (start, end, length) where length is number of indices in run.
  """
  N = len(mask)
  if N == 0:
    return []
  # find linear runs
  runs = []
  i = 0
  while i < N:
    if not mask[i]:
      i += 1
      continue
    start = i
    while i < N and mask[i]:
      i += 1
    end = i - 1
    runs.append((start, end))
  if not runs:
    return []

  # merge wrap-around run if first and last are connected
  if mask[0] and mask[-1] and len(runs) >= 2:
    # last run ends at N-1, first run starts at 0
    s0, e0 = runs[0]
    sl, el = runs[-1]
    if s0 == 0 and el == N - 1:
      # merged run wraps: start = sl, end = e0, length = (N-sl) + (e0+1)
      runs = runs[1:-1]
      runs.insert(0, (sl, e0))  # wrap run
  # compute lengths with wrap support
  out = []
  for s, e in runs:
    if s <= e:
      L = e - s + 1
    else:
      L = (N - s) + (e + 1)
    out.append((s, e, L))
  return out


def detect_windowed_extrema_closed_typed_midpoint(
    pts_closed: List[Point],
    *,
    p: int = 8,
    n: int = 8,
    eps: float = 0.5,
    min_sep: int = 30,
    # straight detection
    straight_radius: int = 60,
    straight_dev_max: float = 10.0,  # in pixels (or your units)
    min_straight_run: int = 10,  # only treat as "straight run" if run length >= this
    min_strength: int = 20,
    max_count: int = 5000,
) -> List[Tuple[int, str]]:
  """
  Returns (idx, etype) where etype in {'xmax','xmin','ymax','ymin'}.
  Patch: on long straight runs, keep ONLY the candidate extremum closest to run midpoint.
  """
  pts = _unwrap_closed(pts_closed)
  N = len(pts)
  if N < p + n + 3:
    return [(0, "ymax")] if N > 0 else []

  xs = [pt[0] for pt in pts]
  ys = [pt[1] for pt in pts]

  # --- build candidates with best type + strength ---
  candidates: List[Tuple[int, str, float]] = []
  for i in range(N):
    is_xmax = _is_window_extremum_1d(xs, i, p, n, eps, "max")
    is_xmin = _is_window_extremum_1d(xs, i, p, n, eps, "min")
    is_ymax = _is_window_extremum_1d(ys, i, p, n, eps, "max")
    is_ymin = _is_window_extremum_1d(ys, i, p, n, eps, "min")
    if not (is_xmax or is_xmin or is_ymax or is_ymin):
      continue

    def sx() -> float:
      return abs(xs[i] - 0.5 * (xs[(i - p) % N] + xs[(i + n) % N]))

    def sy() -> float:
      return abs(ys[i] - 0.5 * (ys[(i - p) % N] + ys[(i + n) % N]))

    opts = []
    if is_xmax:
      opts.append(("xmax", sx()))
    if is_xmin:
      opts.append(("xmin", sx()))
    if is_ymax:
      opts.append(("ymax", sy()))
    if is_ymin:
      opts.append(("ymin", sy()))
    opts.sort(key=lambda t: t[1], reverse=True)
    etype, strength = opts[0]

    if strength > min_strength:
      candidates.append((i, etype, strength))
      if len(candidates) > max_count:
        break

  if not candidates:
    return [(0, "ymax")]

  cand_by_i = {i: (etype, strength) for i, etype, strength in candidates}

  # --- straight mask & runs ---
  straight_mask = [
      (_local_line_deviation_max(pts, i, straight_radius) <= straight_dev_max)
      for i in range(N)
  ]
  runs = _find_straight_runs(straight_mask)

  picked: List[Tuple[int, str]] = []
  blocked = [False] * N

  # --- 1) handle long straight runs: pick closest-to-midpoint candidate within run ---
  for s, e, L in runs:
    if L < min_straight_run:
      continue

    # collect candidate indices inside the run
    in_run = []
    if s <= e:
      indices = range(s, e + 1)
    else:
      indices = list(range(s, N)) + list(range(0, e + 1))

    for idx in indices:
      if idx in cand_by_i:
        in_run.append(idx)

    if not in_run:
      # still block the run? no; only block if we actually pick a candidate
      continue

    # midpoint index along the run arc
    mid_offset = L // 2
    mid = (s + mid_offset) % N

    # pick candidate closest to midpoint (tie-break by strength)
    best_idx = None
    best_key = None  # (dist_to_mid, -strength)
    for idx in in_run:
      etype, strength = cand_by_i[idx]
      d = _cyc_dist(idx, mid, N)
      key = (d, -strength)
      if best_key is None or key < best_key:
        best_key = key
        best_idx = idx

    etype, _ = cand_by_i[best_idx]
    picked.append((best_idx, etype))

    # block the whole run so no other extrema are chosen there
    for idx in indices:
      blocked[idx] = True

  # --- 2) remaining candidates: standard NMS by strength with min_sep ---
  remaining = [
      (i, etype, strength) for (i, etype, strength) in candidates if not blocked[i]
  ]
  remaining.sort(key=lambda t: t[2], reverse=True)

  for i, etype, _ in remaining:
    if blocked[i]:
      continue
    picked.append((i, etype))
    for d in range(-min_sep, min_sep + 1):
      blocked[(i + d) % N] = True

  picked.sort(key=lambda t: t[0])
  return picked


def tangent_from_extrema_pair_typed(
    pts_closed: List[Point],
    a_idx: int,
    a_type: str,
    b_idx: int,
    b_type: str,
    arc_pts: List[Point],
) -> Tuple[Point, Point]:
  """
  Returns (t0, t1) unit tangents at endpoints a and b using only extremum types + a->b direction.
  Tangents follow your forward convention.
  """
  pts = _unwrap_closed(pts_closed)
  N = len(pts)
  a = a_idx % N
  b = b_idx % N

  AB = v_norm(v_sub(pts[b], pts[a]))
  if v_len(AB) < 1e-12:
    AB = (1.0, 0.0)
  BA = v_mul(AB, -1.0)

  def axis_tangent(etype: str, hint: Point) -> Point:
    # y-extrema -> horizontal; x-extrema -> vertical; sign from hint
    if etype in ("ymax", "ymin"):
      return (1.0, 0.0) if hint[0] >= 0.0 else (-1.0, 0.0)
    if etype in ("xmax", "xmin"):
      return (0.0, 1.0) if hint[1] >= 0.0 else (0.0, -1.0)
    return v_norm(hint)

  t0 = axis_tangent(a_type, AB)
  t1 = axis_tangent(b_type, BA)

  # Optional: ensure signs match arc direction at ends (uses only arc endpoints)
  if len(arc_pts) >= 2:
    fwd0 = v_sub(arc_pts[1], arc_pts[0])
    if v_len(fwd0) > 1e-12 and v_dot(t0, fwd0) < 0.0:
      t0 = v_mul(t0, -1.0)

    fwd1 = v_sub(arc_pts[-1], arc_pts[-2])
    if v_len(fwd1) > 1e-12 and v_dot(t1, fwd1) < 0.0:
      t1 = v_mul(t1, -1.0)

  return t0, t1


def get_segments_fit(
    pts_closed: List[Point],
    eps_extrema: float = 0,
    min_arc_height: float = 4,
    straight_seg_tol: float = 5,
):
  pts = _wrap_pts(pts_closed)
  n = len(pts)

  all_segements: List["StraightSegment"] = []
  all_infs: List["InflectionFromStraights"] = []

  if n < 4:
    return [], [], []

  extrema_typed = detect_browse_extrema_closed_typed(
      pts, eps=eps_extrema, min_arc_height=min_arc_height)

  for k in range(len(extrema_typed)):
    a, a_type = extrema_typed[k]
    b, b_type = extrema_typed[(k + 1) % len(extrema_typed)]

    arc_pts, i0 = extract_arc_points_closed(pts, a, b)
    if len(arc_pts) < 2:
      continue

    t0, t1 = tangent_from_extrema_pair_typed(
        pts, a, a_type, b, b_type, arc_pts
    )

    # segs = min_turning_points_dp(arc_pts, eps=straight_seg_tol)

    idxs, segs = fit_segments_rdp(arc_pts, max_error=straight_seg_tol)

    segs = get_straights_segs(arc_pts, segs)

    # mask, runs = detect_zigzag_noise_runs(arc_pts, segs)

    # result = detect_real_inflections_multiscale_dp(arc_pts, segs)

    # segs = [((seg[0] + i0) % n, (seg[1] + i0) % n) for seg in segs]

    infls, segs = inflections_from_straights_open_window(
        arc_pts, segs, start_dir=t0, end_dir=t1)

    start = len(all_segements)

    for seg in segs:
      seg.i0 = (seg.i0 + i0) % n
      seg.i1 = (seg.i1 + i0) % n
      seg.mid_index = (seg.mid_index + i0) % n

    for infl in infls:
      infl.prev_idx = infl.prev_idx + start if infl.prev_idx != -1 else -1
      infl.mid_idx = infl.mid_idx + start if infl.mid_idx != -1 else -1
      infl.next_idx = infl.next_idx + start if infl.next_idx != -1 else -1

    all_segements.extend(segs)
    all_infs.extend(infls)

  return all_segements, all_infs, extrema_typed


def get_straights_inflections(
    pts_closed: List[Point],
    eps_extrema: float = 0,
    min_arc_height: float = 4
):
  pts = _wrap_pts(pts_closed)
  n = len(pts)

  all_segements: List["StraightSegment"] = []
  all_infs: List["InflectionFromStraights"] = []

  if n < 4:
    return [], [], []

  extrema_typed = detect_browse_extrema_closed_typed(
      pts, eps=eps_extrema, min_arc_height=min_arc_height)

  for k in range(len(extrema_typed)):
    a, a_type = extrema_typed[k]
    b, b_type = extrema_typed[(k + 1) % len(extrema_typed)]

    arc_pts, i0 = extract_arc_points_closed(pts, a, b)
    if len(arc_pts) < 2:
      continue

    t0, t1 = tangent_from_extrema_pair_typed(
        pts, a, a_type, b, b_type, arc_pts
    )

    segs = straight_segments_polyline(
        arc_pts,
        is_closed=False
    )

    infls, segs = inflections_from_straights_open_window(
        arc_pts, segs, start_dir=t0, end_dir=t1)

    start = len(all_segements)

    for seg in segs:
      seg.i0 = (seg.i0 + i0) % n
      seg.i1 = (seg.i1 + i0) % n
      seg.mid_index = (seg.mid_index + i0) % n

    for infl in infls:
      infl.prev_idx = infl.prev_idx + start if infl.prev_idx != -1 else -1
      infl.mid_idx = infl.mid_idx + start if infl.mid_idx != -1 else -1
      infl.next_idx = infl.next_idx + start if infl.next_idx != -1 else -1

    all_segements.extend(segs)
    all_infs.extend(infls)

  return all_segements, all_infs, extrema_typed


_X_EXTREMA_TYPES = {"xmin", "xmax"}
_Y_EXTREMA_TYPES = {"ymin", "ymax"}


def find_elongated_orthogonal_split(
    arc_pts: List[Point],
    a_type: str,
    b_type: str,
    aspect_thresh: float = 2.5,
    min_arc_len: int = 20,
) -> Optional[int]:
  """
  Detects the "half rectangle" case: an arc between one x-extremum
  (xmin/xmax, whose tangent is forced purely vertical) and one
  y-extremum (ymin/ymax, tangent forced purely horizontal), whose
  bounding box is strongly elongated (high length/width).

  A single cubic under those two perpendicular end-tangent constraints
  cannot trace such a shape without bowing away from the true boundary:
  nearly all of its travel along the long axis gets crammed into one end
  of its parameter range (see the p1 handle barely leaving p0, then a
  long diagonal to p2, that shows up when this fires). This is a
  representational limit of a single cubic under this exact tangent
  combination, not an extrema-detection error -- both endpoints can be
  perfectly correctly located and the arc will still bow.

  Returns a local index into arc_pts to split at (the point of maximum
  perpendicular deviation from the p0->p3 chord, i.e. the geometric
  "elbow" of the half rectangle), or None if the arc doesn't match this
  pattern.
  """
  if len(arc_pts) < min_arc_len:
    return None

  is_a_x = a_type in _X_EXTREMA_TYPES
  is_b_x = b_type in _X_EXTREMA_TYPES
  if is_a_x == is_b_x:
    return None  # same-axis pair: tangents aren't perpendicular, not this case

  xs = [p[0] for p in arc_pts]
  ys = [p[1] for p in arc_pts]
  w = max(xs) - min(xs)
  h = max(ys) - min(ys)
  if min(w, h) < 1e-6:
    return None
  aspect = max(w, h) / min(w, h)
  if aspect < aspect_thresh:
    return None

  p0, p3 = arc_pts[0], arc_pts[-1]
  vx, vy = p3[0] - p0[0], p3[1] - p0[1]
  vv = vx * vx + vy * vy
  if vv < 1e-9:
    return None

  margin = max(3, int(0.05 * len(arc_pts)))
  best_i, best_d = None, -1.0
  for i in range(margin, len(arc_pts) - margin):
    wx, wy = arc_pts[i][0] - p0[0], arc_pts[i][1] - p0[1]
    d = abs(vx * wy - vy * wx) / math.sqrt(vv)
    if d > best_d:
      best_d = d
      best_i = i

  return best_i


def fit_closed_by_extrema_segments(
    pts_closed: List[Point],
    cfg: FitConfig,
    *,
    eps_extrema: float = 0,
) -> Tuple[List[Cubic], List[Span], List[StraightSegment], List[InflectionFromStraights], List]:
  """
  1) detect x/y extrema
  2) split into arcs between consecutive extrema
  3) fit each arc with fixed start/end tangents
  Returns (all_curves, all_spans, all_segments, all_infls, extrema_typed):
  spans refer to indices in pts_unique (no duplicate end); all_segments/
  all_infls/extrema_typed are the same StraightSegment/InflectionFromStraights/
  extrema values used internally to decide the fit -- callers that just want
  to *show* what the fitter saw (e.g. VectorFitViewer) should use these
  instead of recomputing them (see get_segments_fit, which used to duplicate
  this computation for display only).
  """
  pts = _wrap_pts(pts_closed)
  n = len(pts)
  if n < 4:
    return [], [], [], [], []

  """
  extrema_typed = detect_extrema_by_arcs_closed(
      pts,
      arc_min_len=15,
      arc_max_len=4000,
      min_end_gap=5,
      eps=0.0,
      use_straight_gate=False,
      straight_dev_max=1.2,
      step=1
  )
  """

  """
  extrema_typed = detect_windowed_extrema_closed_typed_midpoint(
      pts,
      p=max_prev_pts, n=max_next_pts, eps=eps_extrema,
      min_sep=min_sep, min_strength=min_strength
  )
  """
  extrema_typed = detect_browse_extrema_closed_typed(
      pts, eps=eps_extrema, min_arc_height=cfg.min_arc_height)

  all_curves: List[Cubic] = []
  all_spans: List[Span] = []
  all_segments_dbg: List[StraightSegment] = []
  all_infls_dbg: List[InflectionFromStraights] = []

  for k in range(len(extrema_typed)):
    a, a_type = extrema_typed[k]
    b, b_type = extrema_typed[(k + 1) % len(extrema_typed)]

    arc_pts, i0 = extract_arc_points_closed(pts_closed, a, b)
    if len(arc_pts) < 2:
      continue

    t0, t1 = tangent_from_extrema_pair_typed(
        pts_closed, a, a_type, b, b_type, arc_pts
    )

    idxs1, segs1 = fit_segments_rdp(arc_pts, max_error=cfg.straight_seg_tol)

    segs2 = get_straights_segs(arc_pts, segs1)

    infls, segs = inflections_from_straights_open_window(
        arc_pts, segs2, start_dir=t0, end_dir=t1,
        min_mid_len=cfg.infl_min_mid_len, max_mid_dev=cfg.infl_max_mid_dev,
    )

    # segs/infls below still carry arc-local indices -- fit_cubics_open_with_spans
    # and the (segs[infl.mid_idx].mid_index + i0) lookup further down rely on
    # that, so offset *copies* into the debug lists rather than mutating in place.
    dbg_start = len(all_segments_dbg)
    all_segments_dbg.extend(
        dataclasses.replace(
            seg,
            i0=(seg.i0 + i0) % n,
            i1=(seg.i1 + i0) % n,
            mid_index=(seg.mid_index + i0) % n,
        )
        for seg in segs
    )
    all_infls_dbg.extend(
        dataclasses.replace(
            infl,
            prev_idx=infl.prev_idx + dbg_start if infl.prev_idx != -1 else -1,
            mid_idx=infl.mid_idx + dbg_start if infl.mid_idx != -1 else -1,
            next_idx=infl.next_idx + dbg_start if infl.next_idx != -1 else -1,
        )
        for infl in infls
    )

    if (len(infls) == 0):
      split_idx = find_elongated_orthogonal_split(
          arc_pts, a_type, b_type) if cfg.enable_elongated_split else None

      if split_idx is None:
        curves_k, spans_k = fit_cubics_open_with_spans(
            arc_pts,
            cfg,
            start_tangent=t0,
            end_tangent=t1,
            i0=i0,
            straights=segs,
        )
        all_curves.extend(curves_k)
        all_spans.extend(spans_k)
      else:
        end_index = (split_idx + i0) % n
        end_tang = tangent_windowed_difference_by_length(pts, end_index)

        arc_pts1, span_index1 = extract_arc_points_closed(
            pts_closed, a, end_index)
        curves_k1, spans_k1 = fit_cubics_open_with_spans(
            arc_pts1, cfg,
            start_tangent=t0, end_tangent=end_tang, i0=span_index1,
        )
        all_curves.extend(curves_k1)
        all_spans.extend(spans_k1)

        arc_pts2, span_index2 = extract_arc_points_closed(
            pts_closed, end_index, b)
        curves_k2, spans_k2 = fit_cubics_open_with_spans(
            arc_pts2, cfg,
            start_tangent=end_tang, end_tangent=t1, i0=span_index2,
        )
        all_curves.extend(curves_k2)
        all_spans.extend(spans_k2)
    else:
      start_index = a
      start_tang = t0

      infl_index = 0
      while infl_index < len(infls):
        infl = infls[infl_index]
        end_index = (segs[infl.mid_idx].mid_index + i0) % n        
        end_tang = segs[infl.mid_idx].direction
        arc_pts, span_index = extract_arc_points_closed(
            pts_closed, start_index, end_index)
        curves_k, spans_k = fit_cubics_open_with_spans(
            arc_pts,
            cfg,
            start_tangent=start_tang,
            end_tangent=end_tang,
            i0=span_index,
            straights=segs,
        )
        all_curves.extend(curves_k)
        all_spans.extend(spans_k)
        start_index = end_index
        start_tang = end_tang
        infl_index = infl_index + 1

      arc_pts, span_index = extract_arc_points_closed(
          pts_closed, start_index, b)
      curves_k, spans_k = fit_cubics_open_with_spans(
          arc_pts,
          cfg,
          start_tangent=start_tang,
          end_tangent=t1,
          i0=span_index,
      )
      all_curves.extend(curves_k)
      all_spans.extend(spans_k)

  return all_curves, all_spans, all_segments_dbg, all_infls_dbg, extrema_typed
