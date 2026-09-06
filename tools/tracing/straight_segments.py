from __future__ import annotations
import math
from typing import List, Optional, Tuple
from math import atan2, degrees, hypot, sqrt
from typing import List, Sequence, Tuple

from dataclasses import dataclass
from math import atan2, degrees, hypot, sqrt, pi
from typing import List, Sequence, Tuple, Optional

from merge_straights import best_merge_successive_inflections
from merge_straights_2 import best_merge_successive_inflections_patched


PointI = Tuple[int, int]
Point = Tuple[float, float]


# =========================
# Helpers
# =========================

def _normalize(v: Point, eps: float = 1e-12) -> Point:
  n = hypot(v[0], v[1])
  if n < eps:
    return (0.0, 0.0)
  return (v[0] / n, v[1] / n)


def _pca_line_fit(points: Sequence[Point]) -> Tuple[Point, Point]:
  """
  TLS/PCA line fit: returns (centroid, unit direction of best-fit line).
  """
  m = len(points)
  if m == 0:
    return (0.0, 0.0), (1.0, 0.0)

  cx = sum(p[0] for p in points) / m
  cy = sum(p[1] for p in points) / m

  sxx = syy = sxy = 0.0
  for x, y in points:
    dx = x - cx
    dy = y - cy
    sxx += dx * dx
    syy += dy * dy
    sxy += dx * dy

  if abs(sxy) < 1e-18 and abs(sxx - syy) < 1e-18:
    return (cx, cy), (1.0, 0.0)

  theta = 0.5 * atan2(2.0 * sxy, (sxx - syy))
  d = _normalize((__import__("math").cos(theta),
                 __import__("math").sin(theta)))
  return (cx, cy), d


def _orth_dist_to_line(p: Point, c: Point, d_unit: Point) -> float:
  # dist = |cross(d, p-c)| in 2D
  vx = p[0] - c[0]
  vy = p[1] - c[1]
  return abs(d_unit[0] * vy - d_unit[1] * vx)


def _wrap(i: int, n: int) -> int:
  return i % n


def _unwrap_angle(prev: float, curr: float) -> float:
  """
  Unwrap curr to be continuous relative to prev.
  Keeps jumps within [-pi, pi] by adding/subtracting 2pi.
  """
  d = curr - prev
  if d > pi:
    curr -= 2.0 * pi
  elif d < -pi:
    curr += 2.0 * pi
  return curr


def _polyline_tangent_angles(pts: Sequence[Point]) -> List[float]:
  """
  Tangent angle at each vertex based on forward difference.
  For closed contours, caller should pass unique points (no duplicate endpoint).
  """
  n = len(pts)
  ang = [0.0] * n
  for i in range(n):
    p0 = pts[i]
    p1 = pts[(i + 1) % n]
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    ang[i] = atan2(dy, dx)
  # unwrap around the loop (best-effort; we only need local windows robustly)
  # We'll do local unwrapping in window computations to avoid seam issues.
  return ang


# =========================
# Straight segments on closed contour (robust to zigzags)
# =========================

@dataclass(frozen=False)
class StraightSegment:
  i0: int
  i1: int
  wraps: bool
  p0: Point
  p1: Point
  chord_len: float
  direction: Point
  angle_deg: float
  max_dev: float
  rms_dev: float
  centroid: float
  mid_index: int


def closest_index_to_centroid_on_segment(
    pts_unique: Sequence[Point],
    i0: int,
    i1: int,
    wraps: bool,
    centroid: Point,
) -> int:
  """
  Return the index (in pts_unique) whose point is closest (Euclidean) to `centroid`,
  restricted to the points that belong to the segment [i0..i1] along the contour.

  pts_unique: closed contour without duplicated last point (i.e., pts[:-1] as floats)
  i0,i1,wraps: segment definition in pts_unique indexing (0..n-1)
  centroid: (cx,cy) from _pca_line_fit(pts_seg)

  Returns:
    idx in [0..n-1]
  """
  n = len(pts_unique)
  if n == 0:
    raise ValueError("pts_unique is empty")

  cx, cy = float(centroid[0]), float(centroid[1])

  def dist2(p: Point) -> float:
    dx = p[0] - cx
    dy = p[1] - cy
    return dx * dx + dy * dy

  best_idx = i0 % n
  best_d2 = float("inf")

  def iter_indices():
    if not wraps:
      i = i0 % n
      end = i1 % n
      while True:
        yield i
        if i == end:
          break
        i = (i + 1) % n
    else:
      # [i0..n-1] U [0..i1]
      i = i0 % n
      while True:
        yield i
        if i == n - 1:
          break
        i += 1
      i = 0
      end = i1 % n
      while True:
        yield i
        if i == end:
          break
        i += 1

  for i in iter_indices():
    d2 = dist2(pts_unique[i])
    if d2 < best_d2:
      best_d2 = d2
      best_idx = i

  return best_idx


def closest_index_to_centroid_on_pts_seg(pts_seg: Sequence[Point], centroid: Point) -> int:
  """
  If you already have pts_seg (the points you passed to _pca_line_fit),
  return the local index in pts_seg closest to centroid.
  """
  cx, cy = float(centroid[0]), float(centroid[1])
  best_j = 0
  best_d2 = float("inf")
  for j, (x, y) in enumerate(pts_seg):
    dx = x - cx
    dy = y - cy
    d2 = dx * dx + dy * dy
    if d2 < best_d2:
      best_d2 = d2
      best_j = j
  return best_j


def straight_segments_polyline(
    uniq: List[PointI],
    is_closed: bool,
    *,
    max_dev: float = 0.5,
    min_points: int = 5,
    min_chord_len: float = 5.0,
    max_angle_change_deg: Optional[float] = None,
    step: int = 1,
) -> List["StraightSegment"]:
  """
  Detect straight-ish segments for BOTH:
    - CLOSED contour: pts[0] == pts[-1] (duplicate endpoint)
    - OPEN polyline : pts[0] != pts[-1]

  Output indices always refer to the list `uniq`:
    - closed: uniq = pts[:-1]
    - open  : uniq = pts
  For open polylines: wraps is always False, and no modulo indexing is used.
  """

  n = len(uniq)
  if n < 3:
    return []

  from math import radians
  step = max(1, int(step))
  min_points = max(2, int(min_points))
  max_angle_change = None if max_angle_change_deg is None else radians(
      max_angle_change_deg)

  def get(i: int) -> Point:
    # IMPORTANT: modulo only for closed contours
    return uniq[i % n] if is_closed else uniq[i]

  # Segment index iteration for NMS (closed vs open)
  def iter_indices(seg: "StraightSegment"):
    if not is_closed:
      # open: simple range
      i = seg.i0
      while i <= seg.i1:
        yield i
        i += 1
      return

    # closed: your existing wrap logic
    if not seg.wraps:
      i = seg.i0
      while True:
        yield i
        if i == seg.i1:
          break
        i = (i + 1) % n
    else:
      i = seg.i0
      while True:
        yield i
        if i == n - 1:
          break
        i += 1
      i = 0
      while True:
        yield i
        if i == seg.i1:
          break
        i += 1

  # Helper for "mid_index"
  def mid_index_for_segment(i0: int, i1: int, wraps: bool, centroid: Point) -> int:
    if is_closed:
      return closest_index_to_centroid_on_segment(
          pts_unique=uniq, i0=i0, i1=i1, wraps=wraps, centroid=centroid
      )
    # open: scan [i0..i1]
    cx, cy = float(centroid[0]), float(centroid[1])
    best_i = i0
    best_d2 = float("inf")
    for i in range(i0, i1 + 1):
      dx = uniq[i][0] - cx
      dy = uniq[i][1] - cy
      d2 = dx * dx + dy * dy
      if d2 < best_d2:
        best_d2 = d2
        best_i = i
    return best_i

  candidates: List["StraightSegment"] = []

  i0 = 0
  while i0 < n:
    j = i0 + (min_points - 1)
    if not is_closed and j >= n:
      break  # can't start a segment this late on an open polyline

    p0 = get(i0)
    p_seed = get(j)
    seed_dir = _normalize((p_seed[0] - p0[0], p_seed[1] - p0[1]))
    if seed_dir == (0.0, 0.0):
      i0 += 1
      continue

    best: Optional["StraightSegment"] = None
    best_j: Optional[int] = None

    # growth bound differs open vs closed
    def can_grow(j: int) -> bool:
      if is_closed:
        return (j - i0) < n
      return j < n

    while can_grow(j):
      # build pts_seg (no modulo for open)
      if is_closed:
        pts_seg = [get(t) for t in range(i0, j + 1, step)]
      else:
        pts_seg = [uniq[t] for t in range(i0, j + 1, step)]

      c, d = _pca_line_fit(pts_seg)
      # keep direction consistent with seed (avoid 180° flip)
      if d[0] * seed_dir[0] + d[1] * seed_dir[1] < 0:
        d = (-d[0], -d[1])

      # optional drift limit
      if max_angle_change is not None:
        cross = seed_dir[0] * d[1] - seed_dir[1] * d[0]
        dot = seed_dir[0] * d[0] + seed_dir[1] * d[1]
        if abs(atan2(cross, dot)) > max_angle_change:
          break

      devs = [_orth_dist_to_line(p, c, d) for p in pts_seg]
      dmax = max(devs) if devs else 0.0
      if dmax > max_dev:
        break

      p1 = get(j)
      clen = hypot(p1[0] - p0[0], p1[1] - p0[1])
      if clen >= min_chord_len and (j - i0 + 1) >= min_points:
        ang = degrees(atan2(d[1], d[0]))

        if is_closed:
          i0m = i0 % n
          j0m = j % n
          wraps = (j0m < i0m)
        else:
          i0m = i0
          j0m = j
          wraps = False

        best = StraightSegment(
            i0=i0m, i1=j0m, wraps=wraps,
            p0=p0, p1=p1, chord_len=clen,
            direction=d, angle_deg=ang,
            max_dev=dmax,
            rms_dev=sqrt(sum(x * x for x in devs) / max(1, len(devs))),
            centroid=c,
            mid_index=mid_index_for_segment(i0m, j0m, wraps, c),
        )
        best_j = j

      j += 1

    if best is not None and best_j is not None:
      candidates.append(best)
      # advance: for open polyline, skip past best end; for closed, behave like before
      if is_closed:
        i0 = best_j + 1
      else:
        i0 = best_j + 1
    else:
      i0 += 1

  if not candidates:
    return []

  # Non-max suppression: prefer long and straight (low dev)
  candidates.sort(key=lambda s: (-s.chord_len, s.max_dev))

  taken = [False] * n
  kept: List["StraightSegment"] = []
  for seg in candidates:
    covered = already = 0
    for idx in iter_indices(seg):
      covered += 1
      if taken[idx]:
        already += 1

    if already <= max(2, covered // 5):  # allow <=20% overlap
      kept.append(seg)
      for idx in iter_indices(seg):
        taken[idx] = True

  # sort in contour order for convenience
  if is_closed:
    kept.sort(key=lambda s: (s.i0, s.wraps, s.i1))
  else:
    kept.sort(key=lambda s: (s.i0, s.i1))

  return kept

# =========================
# Inflection segments via "two opposite arcs" with min_radian
# =========================


@dataclass(frozen=False)
class InflectionSegment:
  segment: StraightSegment
  theta_left: float      # signed turning angle on left window (radians)
  theta_right: float     # signed turning angle on right window (radians)
  # slope at inflection (take from straight segment direction)
  slope_rad: float
  score: float           # bigger => stronger/cleaner


def _window_turning_angle(
    ang: Sequence[float],
    start: int,
    end: int,
    n: int,
) -> float:
  """
  Sum of unwrapped delta angles from start..end along forward direction modulo n.
  Interprets ang[i] as tangent angle at vertex i (edge i->i+1).
  Returns signed total turning angle over that walk.
  """
  # walk forward from start to end (exclusive of end), summing delta
  i = start
  prev = ang[_wrap(i, n)]
  total = 0.0
  while i != end:
    j = _wrap(i + 1, n)
    cur = ang[j]
    cur = _unwrap_angle(prev, cur)
    total += (cur - prev)
    prev = cur
    i = j
  return total


@dataclass(frozen=False)
class InflectionFromStraights:
  """
  Inflection 'point' represented by the middle straight segment.
  """
  prev_idx: int
  mid_idx: int
  next_idx: int

  # Representative location: midpoint of mid segment endpoints
  point: Point

  # slope at inflection: mid segment direction
  slope_deg: float

  # signed angles (deg) between mid direction and prev/next directions
  in_angle_deg: float
  out_angle_deg: float

  score: float


def _dot(a: Point, b: Point) -> float:
  return a[0] * b[0] + a[1] * b[1]


def _cross(a: Point, b: Point) -> float:
  return a[0] * b[1] - a[1] * b[0]


def signed_angle_deg(a_unit: Point, b_unit: Point) -> float:
  """
  Signed smallest angle from a -> b in degrees, range [-180, 180].
  Positive => b is to the left of a (CCW), negative => to the right (CW).
  """
  return degrees(atan2(_cross(a_unit, b_unit), _dot(a_unit, b_unit)))


def _midpoint(a: Point, b: Point) -> Point:
  return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)


def _unwrap_forward_steps(i0: int, i1: int, n: int, wraps: bool) -> int:
  """
  Steps along contour order from i0 to i1.
  For a consistent set of segments in contour order, (i1-i0)%n is fine.
  wraps just documents seam crossing; we keep it for clarity.
  """
  return (i1 - i0) % n


def _ensure_contour_order(segs: List[StraightSegment], n: Optional[int]) -> List[StraightSegment]:
  """
  Returns segments sorted by i0 in contour order.
  For closed contours, plain sort by i0 is usually OK if you don't have wrap-around segments
  at the seam mixed in. If you do, we rotate so the biggest gap becomes the seam.
  """
  if not segs:
    return []
  segs_sorted = sorted(segs, key=lambda s: s.i0)
  if n is None:
    return segs_sorted

  # Find the largest gap between consecutive i0's to choose a seam rotation.
  i0s = [s.i0 for s in segs_sorted]
  gaps = []
  for a, b in zip(i0s, i0s[1:] + [i0s[0] + n]):
    gaps.append(b - a)
  # index of largest gap start
  k = max(range(len(gaps)), key=lambda i: gaps[i])
  # Rotate so element after the largest gap becomes first (seam placed in the gap)
  return segs_sorted[k + 1:] + segs_sorted[: k + 1]


def sign(n):
  if n < 0:
    return -1
  elif n > 0:
    return 1
  else:
    return 0


def _segment_bridge_dir(a_end: Point, b_start: Point) -> Point:
  """Direction from a_end to b_start (unit). Returns (0,0) if degenerate."""
  dx = b_start[0] - a_end[0]
  dy = b_start[1] - a_end[1]
  n = hypot(dx, dy)
  if n <= 1e-12:
    return (0.0, 0.0)
  return (dx / n, dy / n)


def simplify_straights(
    pts: List[PointI],
    straights: List["StraightSegment"],
    start_dir: Point,
    end_dir: Point,
):

  chord_len = 0
  segs = []
  for straight in straights:
    chord_len += straight.chord_len

  min_length = chord_len / 20

  for index, straight in enumerate(straights):
    if straight.chord_len > min_length:
      segs.append(straight)
    else:
      segs.append(straight)

  return segs


def angle_between(v1, v2):
  x1, y1 = v1
  x2, y2 = v2
  dot = x1*x2 + y1*y2
  n1 = math.hypot(x1, y1)
  n2 = math.hypot(x2, y2)
  if n1*n2 == 0:
    return 0
  c = max(-1, min(1, dot/(n1*n2)))
  return math.degrees(math.acos(c))


def merge_small_bumps(pts, segs,
                      max_angle=1,
                      min_len=5):
  """
  Merge adjacent nearly-collinear segments.

  segs = [(i0,i1), ...]
  """
  out = []
  i = 0
  while i < len(segs):
    if i == len(segs)-1:
      out.append(segs[i])
      break

    a0 = segs[i].i0
    a1 = segs[i].i1
    b0 = segs[i+1].i0
    b1 = segs[i+1].i1

    v1 = (pts[a1][0]-pts[a0][0], pts[a1][1]-pts[a0][1])
    v2 = (pts[b1][0]-pts[b0][0], pts[b1][1]-pts[b0][1])

    ang = angle_between(v1, v2)
    len2 = math.hypot(*v2)

    if ang < max_angle or len2 < min_len:
      segs[i+1].i0 = a0
      segs[i+1].i1 = b1
    else:
      out.append(segs[i])

    i += 1

  return out


def inflections_from_straights_open_window(
    pts: List[PointI],
    straights: List["StraightSegment"],
    *,
    min_degree: float = 5.0,               # base threshold (deg)
    # stricter threshold if we chain inflections
    min_degree_cons: float = 0.0,
    min_mid_len: float = 0.0,
    min_neighbor_len: float = 0.0,
    max_mid_dev: Optional[float] = None,
    max_neighbor_dev: Optional[float] = None,
    start_dir: Point,
    end_dir: Point,

    # how far to look for a usable neighbor if i±1 fails (open polyline only)
    max_skip: int = 16,
):
  """
  Open-polyline inflection detection using straight segments, with skipping:

  For each mid segment i, we want an "incoming" direction and an "outgoing" direction
  that lie on opposite sides of mid.direction, with magnitude >= threshold.

  Incoming direction candidates (in priority order):
    1) direction of previous straight segment (i-1)
    2) if insufficient, for j=i-2,i-3,... use bridge direction from seg[j].p1 -> seg[i].p0
       (i.e., direction of the gap between segments)

  Outgoing direction candidates (priority):
    1) direction of next straight segment (i+1)
    2) if insufficient, for j=i+2,i+3,... use bridge direction from seg[i].p1 -> seg[j].p0

  For endpoints, start_dir/end_dir can be used as virtual neighbors.

  Notes:
    - This assumes `straights` are already in polyline order.
    - Indices in InflectionFromStraights refer to indices in `straights`.
  """
  if len(straights) == 0:
    return [], straights

  # Convert pts to float points (unique list for open polyline)
  ptsf: List[Point] = [(float(x), float(y)) for (x, y) in pts]

  first = straights[0].p0
  last = straights[-1].p1

  seg_length = hypot(last[0] - first[0], last[1] - first[1])
  # if seg_length < 100:
  #  return [], straights

  """
  res = best_merge_successive_inflections(
      pts, straights,
      eps=10.0,
      max_jump=40,
      successive_inflection_cost=5000.0,
      inflection_cost=0,
  )"""

  """res = best_merge_successive_inflections_patched(
      pts, straights,
      eps=8.0,
      max_jump=40,
      successive_inflection_cost=50000.0,
      inflection_cost=0,
  )

  segs = res.merged_straights"""

  segs = merge_small_bumps(pts, segs=straights)

  # segs = simplify_straights(pts, straights=straights, start_dir=start_dir, end_dir=end_dir)

  m = len(segs)

  is_inflection = [False] * m
  out: List["InflectionFromStraights"] = []

  def passes_filters(prev_idx: int, mid_idx: int, next_idx: int) -> bool:
    prev = segs[prev_idx] if prev_idx >= 0 else None
    mid = segs[mid_idx]
    nxt = segs[next_idx] if next_idx >= 0 else None

    if mid.chord_len < min_mid_len:
      return False
    if prev is not None and prev.chord_len < min_neighbor_len:
      return False
    if nxt is not None and nxt.chord_len < min_neighbor_len:
      return False

    if max_mid_dev is not None and mid.max_dev > max_mid_dev:
      return False
    if max_neighbor_dev is not None:
      if prev is not None and prev.max_dev > max_neighbor_dev:
        return False
      if nxt is not None and nxt.max_dev > max_neighbor_dev:
        return False

    return True

  def threshold_for(mid_idx: int) -> float:
    # If previous segment (mid_idx-1) is already an inflection, tighten threshold
    if mid_idx - 1 >= 0 and is_inflection[mid_idx - 1]:
      return min_degree_cons
    return min_degree

  def best_incoming(mid_idx: int, eps: float = 1) -> Tuple[Optional[int], Optional[Point], Optional[float]]:
    """
    Returns (src_idx, dir, angle_deg) where src_idx is:
      - (mid_idx-1) if using previous segment direction
      - j (some < mid_idx-1) if using bridge from seg[j].p1 -> seg[mid].p0
      - -1 if using start_dir
      - None if not found
    angle_deg is signed_angle_deg(mid_dir, dir).
    """
    mid = segs[mid_idx]
    d_mid = mid.direction

    # 0) boundary virtual
    if mid_idx == 0:
      a = signed_angle_deg(d_mid, start_dir)
      return -1, start_dir, a

    i0 = mid.i0
    p_mid0 = ptsf[i0]
    # 1) immediate previous segment direction
    if mid_idx - 1 >= 0:
      prev_j = mid_idx - 1
      # d_bridge = segs[prev_j].direction
      segj = segs[prev_j]
      pj0 = ptsf[segj.i0]
      d_bridge = _segment_bridge_dir(pj0, p_mid0)
      prev_a = signed_angle_deg(d_mid, d_bridge)
      prev_d = d_bridge

      # return prev_j, prev_d, prev_a
      # if abs(a) >= th and a != 0.0:
      #  return mid_idx - 1, d_prev, a

    # 2) skip backwards: bridge direction from seg[j].p1 -> seg[mid].p0

    for j in range(mid_idx - 2, max(-2, mid_idx - max_skip - 1), -1):
      if j == -1:
        d_bridge = start_dir
      else:
        segj = segs[j]
        pj1 = ptsf[segj.i0]
        d_bridge = _segment_bridge_dir(pj1, p_mid0)
      if d_bridge == (0.0, 0.0):
        break
      a = signed_angle_deg(d_mid, d_bridge)
      if ((sign(prev_a) != sign(a) and abs(prev_a) > eps) or abs(prev_a) - eps >= abs(a)):
        break
      if abs(a) > abs(prev_a):
        prev_a = a
        prev_j = j
        prev_d = d_bridge

    # if abs(prev_a) >= th and prev_a != 0.0:
    #  return prev_j, prev_d, prev_a
    # return None, None, None
    return prev_j, prev_d, prev_a

  def best_outgoing(mid_idx: int, eps: float = 1) -> Tuple[Optional[int], Optional[Point], Optional[float]]:
    """
    Returns (src_idx, dir, angle_deg) where src_idx is:
      - (mid_idx+1) if using next segment direction
      - j (some > mid_idx+1) if using bridge from seg[mid].p1 -> seg[j].p0
      - -1 if using end_dir
      - None if not found
    """
    mid = segs[mid_idx]
    d_mid = mid.direction

    # 0) boundary virtual
    if mid_idx == m - 1:
      a = signed_angle_deg(d_mid, end_dir)
      return -1, end_dir, a

    i1 = mid.i1
    p_mid1 = ptsf[i1]
    # 1) immediate next segment direction
    if mid_idx + 1 < m:
      prev_j = mid_idx + 1
      # d_bridge = segs[prev_j].direction
      segj = segs[prev_j]
      pj1 = ptsf[segj.i1]
      d_bridge = _segment_bridge_dir(p_mid1, pj1)
      prev_a = signed_angle_deg(d_mid, d_bridge)
      prev_d = d_bridge
      # return prev_j, prev_d, prev_a
      # if abs(a) >= th and a != 0.0:
      #  return mid_idx + 1, d_next, a

    # 2) skip forwards: bridge direction from seg[mid].p1 -> seg[j].p0
    for j in range(mid_idx + 2, min(m+1, mid_idx + max_skip + 1)):
      if j == m:
        d_bridge = end_dir
      else:
        segj = segs[j]
        pj1 = ptsf[segj.i1]
        d_bridge = _segment_bridge_dir(p_mid1, pj1)
      if d_bridge == (0.0, 0.0):
        break
      a = signed_angle_deg(d_mid, d_bridge)
      if ((sign(prev_a) != sign(a) and abs(prev_a) > eps) or abs(prev_a) - eps >= abs(a)):
        break
      if abs(a) > abs(prev_a):
        prev_a = a
        prev_j = j if j != m else -1
        prev_d = d_bridge

    # if abs(prev_a) >= th and prev_a != 0.0:
    #  return prev_j, prev_d, prev_a
    # return None, None, None
    return prev_j, prev_d, prev_a

  # Main loop over all segments
  for mid_idx in range(m):

    mid = segs[mid_idx]

    eps = 3
    in_src, d_in, a_in = best_incoming(mid_idx, eps=eps)
    out_src, d_out, a_out = best_outgoing(mid_idx, eps=eps)

    if d_in is None or d_out is None or a_in is None or a_out is None:
      continue

    th = threshold_for(mid_idx)

    angle = abs(signed_angle_deg(d_in, d_out))

    # if in_src == -1 and out_src == -1 and angle == 90:
    #  continue

    if in_src != -1 or out_src != -1 or (angle != 0 and angle != 180):

      if abs(a_in) < th and abs(a_out) < th:
        continue
      if abs(a_in) < 4 or abs(a_out) < 4:
        continue
      if a_in == 0.0 or a_out == 0.0:
        continue
      if a_in * a_out < 0.0:
        continue  # same side => not an inflection region

    # filters that depend on prev/next availability:
    prev_idx = in_src if (
        in_src is not None and in_src >= 0 and in_src < m) else -1
    next_idx = out_src if (
        out_src is not None and out_src >= 0 and out_src < m) else -1
    if not passes_filters(prev_idx, mid_idx, next_idx):
      continue

    pt = _midpoint(mid.p0, mid.p1)

    # score: stronger bends + longer + straighter
    score = 0.5 * (abs(a_in) + abs(a_out))
    score *= (mid.chord_len / (mid.chord_len + 20.0))
    # score *= 1.0 / (1.0 + mid.max_dev)

    to_add = True
    if prev_idx == -1 and next_idx == -1:
      for inf in out:
        if inf.prev_idx == -1 and inf.next_idx == -1:
          if mid.chord_len > segs[inf.mid_idx].chord_len:
            out.remove(inf)
          else:
            to_add = False
          break

    # is_inflection[mid_idx] = True
    if to_add:
      out.append(
          InflectionFromStraights(
              prev_idx=prev_idx,
              mid_idx=mid_idx,
              next_idx=next_idx,
              point=pt,
              slope_deg=mid.angle_deg,
              in_angle_deg=a_in,
              out_angle_deg=a_out,
              score=score,
          )
      )

  # if (len(out) == len(segs)):
  #  out = []
  # keep in contour order (or you can sort by score)
  return out, segs


@dataclass(frozen=True)
class SegmentFit:
  i0: int
  i1: int
  max_err: float
  max_err_i: int


def _dist2_point_to_line(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
  """
  Squared perpendicular distance from P to infinite line through A->B.
  If A==B, returns squared distance to A.
  """
  vx = bx - ax
  vy = by - ay
  wx = px - ax
  wy = py - ay
  vv = vx * vx + vy * vy
  if vv <= 1e-30:
    return wx * wx + wy * wy
  # area^2 / |v|^2
  cross = wx * vy - wy * vx
  return (cross * cross) / vv


def _max_error_to_chord(pts: List[Point], i0: int, i1: int) -> Tuple[float, int]:
  """
  Max perpendicular deviation of points pts[i0..i1] to the chord (infinite line) i0->i1.
  Returns (max_error, argmax_index).
  """
  ax, ay = pts[i0]
  bx, by = pts[i1]
  max_d2 = -1.0
  arg = i0
  for i in range(i0 + 1, i1):
    px, py = pts[i]
    d2 = _dist2_point_to_line(px, py, ax, ay, bx, by)
    if d2 > max_d2:
      max_d2 = d2
      arg = i
  if max_d2 < 0.0:
    return 0.0, i0
  return math.sqrt(max_d2), arg


def fit_segments_rdp(
    pts: List[Point],
    max_error: float,
    *,
    min_points_per_seg: int = 5,
    closed=False
) -> Tuple[List[int], List[Tuple[int, int]]]:
  """
  Fit a list of straight segments to pts using a Ramer–Douglas–Peucker style recursion.

  Returns a list of segments as index pairs (i0, i1), covering the polyline in order.

  Error metric:
    max perpendicular deviation to the infinite line through endpoints (classic RDP).

  Notes:
    - Works best if pts are ordered along the curve.
    - If your curve has duplicate consecutive points, it's fine.
  """

  if closed:
    pts = pts[:] + [pts[0]]

  n = len(pts)
  if n == 0:
    return []
  if n == 1:
    return [(0, 0)]

  start = 0
  end = n - 1

  # Stack-based recursion to avoid deep recursion on large inputs.
  stack: List[Tuple[int, int]] = [(start, end)]
  keep = [False] * n
  keep[start] = True
  keep[end] = True

  while stack:
    i0, i1 = stack.pop()
    if i1 - i0 + 1 < min_points_per_seg:
      continue

    err, imax = _max_error_to_chord(pts, i0, i1)
    if err > max_error and (imax != i0 and imax != i1):
      keep[imax] = True
      # Split at worst point, like cubic fitting.
      stack.append((i0, imax))
      stack.append((imax, i1))

  # Build segments from kept vertices
  idxs = [i for i, k in enumerate(keep) if k]

  if closed:
    idxs.pop()

  idxs.sort()

  segments: List[Tuple[int, int]] = []
  for a, b in zip(idxs[:-1], idxs[1:]):
    segments.append((a, b))

  # Handle degenerate case
  if not segments:
    segments = [(0, n - 1)]

  return idxs, segments


def dist_point_line(p: Point, a: Point, b: Point, eps=1e-12):
  ab = (b[0]-a[0], b[1]-a[1])
  ap = (p[0]-a[0], p[1]-a[1])
  lab = math.hypot(*ab)
  if lab < eps:
    return math.hypot(*ap)
  return abs(ab[0]*ap[1] - ab[1]*ap[0]) / lab


def segment_error_ok(points: List[Point], j: int, i: int, eps: float):
  a, b = points[j], points[i]
  for k in range(j+1, i):
    if dist_point_line(points[k], a, b) > eps:
      return False
  return True


# ---------------- DP solver ----------------

def min_turning_points_dp(points: List[Point], eps: float):
  n = len(points)
  INF = 10**9

  dp = [INF]*n
  prev = [-1]*n

  dp[0] = 0

  for i in range(1, n):
    for j in range(i):
      if segment_error_ok(points, j, i, eps):
        if dp[j] + 1 < dp[i]:
          dp[i] = dp[j] + 1
          prev[i] = j

  # backtrack breakpoints
  idx = n-1
  breaks = []
  while idx > 0:
    breaks.append(idx)
    idx = prev[idx]
  breaks.append(0)

  breaks.reverse()

  segments: List[Tuple[int, int]] = []
  for a, b in zip(breaks[:-1], breaks[1:]):
    segments.append((a, b))

  # Handle degenerate case
  if not segments:
    segments = [(0, n - 1)]

  return segments

  # return dp[n-1]-1, breaks  # turning points, vertex indices


def get_straights_segs(pts: List[Point], segs: List[Tuple[int, int]]):
  """
  Builds a StraightSegment per (i0, i1) RDP breakpoint pair, fit from all
  of pts[i0:i1+1] (a ~100px run RDP already certified straight to within
  straight_seg_tol) rather than just the two endpoints. This makes
  `direction` and `mid_index` robust to a single defective pixel (a pit,
  hollow, or bump from the scan) anywhere in the segment -- the two
  endpoints alone have no such protection.
  """
  straights: List["StraightSegment"] = []

  for seg in segs:
    i0, i1 = seg
    p0 = pts[i0]
    p1 = pts[i1]
    clen = hypot(p1[0] - p0[0], p1[1] - p0[1])

    seg_pts = pts[i0:i1 + 1]
    centroid, pca_dir = _pca_line_fit(seg_pts)

    # PCA's sign is arbitrary; orient it to agree with the endpoint chord
    # so it stays forward-along-contour, matching every other direction in
    # this pipeline (and keeping the old endpoint-chord behavior when the
    # segment is short enough that PCA and chord agree closely).
    chord_dir = _segment_bridge_dir(p0, p1)
    if pca_dir[0] * chord_dir[0] + pca_dir[1] * chord_dir[1] < 0.0:
      pca_dir = (-pca_dir[0], -pca_dir[1])

    devs = [_orth_dist_to_line(p, centroid, pca_dir) for p in seg_pts]
    max_dev = max(devs)
    rms_dev = sqrt(sum(d * d for d in devs) / len(devs))

    # Robust midpoint: among the middle third of the segment (keeps the
    # join near the segment's centre -- an unrestricted minimum-deviation
    # search could drift all the way to an endpoint), pick the point
    # closest to the fitted line. This is what moves the join off a
    # pit/bump instead of the blind index midpoint (i0+i1)//2.
    n_seg = len(seg_pts)
    lo = i0 + n_seg // 3
    hi = i0 + (2 * n_seg) // 3
    if hi <= lo:
      mid_index = (i0 + i1) // 2
    else:
      mid_index = min(
          range(lo, hi + 1),
          key=lambda j: _orth_dist_to_line(pts[j], centroid, pca_dir),
      )

    ang = degrees(atan2(pca_dir[1], pca_dir[0]))
    straight = StraightSegment(
        i0=i0, i1=i1, wraps=False,
        p0=p0, p1=p1, chord_len=clen,
        direction=pca_dir, angle_deg=ang,
        max_dev=max_dev, rms_dev=rms_dev,
        centroid=centroid, mid_index=mid_index,
    )
    straights.append(straight)

  return straights
