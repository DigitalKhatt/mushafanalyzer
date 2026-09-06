from typing import List, Tuple
import math

from straight_segments import StraightSegment, signed_angle_deg

Point = Tuple[float, float]


def _sub(a: Point, b: Point) -> Point: return (a[0]-b[0], a[1]-b[1])
def _dot(a: Point, b: Point) -> float: return a[0]*b[0] + a[1]*b[1]
def _cross(a: Point, b: Point) -> float: return a[0]*b[1] - a[1]*b[0]
def _norm(a: Point) -> float: return math.hypot(a[0], a[1])


def _unit(v: Point, eps=1e-12) -> Point:
  n = _norm(v)
  return (v[0]/n, v[1]/n) if n > eps else (0.0, 0.0)


def _point_line_dist(p: Point, a: Point, b: Point, eps=1e-12) -> float:
  ab = _sub(b, a)
  ap = _sub(p, a)
  lab = _norm(ab)
  if lab < eps:
    return _norm(ap)
  # area / base
  return abs(_cross(ap, ab)) / lab


def detect_zigzag_noise_runs(
    pts: List[Point],
    straights: List["StraightSegment"],
    *,
    win: int = 6,                 # half-window in segments
    theta_min_deg: float = 6.0,   # ignore tiny turns for alternation stats
    c1_thresh: float = 0.35,      # low tangent correlation => unstable direction
    alt_rate_thresh: float = 0.6,  # frequent sign flips => zigzag
    min_run: int = 6,             # minimum consecutive flagged vertices
) -> Tuple[List[bool], List[Tuple[int, int]]]:
  n = len(straights)
  if n < 2:
    return [False]*n, []

  # segment tangents

  L = [s.chord_len for s in straights]
  t = [s.direction for s in straights]
  angles = [s.angle_deg for s in straights]

  # turning angles between consecutive tangents (per vertex i=1..n-2)
  theta = [0.0]*(n-1)
  for i in range(n-1):
    a, b = t[i], t[i+1]
    theta[i] = math.atan2(_cross(a, b), _dot(a, b))  # signed

  theta_min = math.radians(theta_min_deg)

  thetadegree = [0.0]*(n-1)
  for i in range(n-1):
    a, b = t[i], t[i+1]
    thetadegree[i] = signed_angle_deg(t[i], t[i+1])

  # local measures per "inner vertex" index v=1..n-2 mapped to theta index v-1
  flagged_inner = [False]*(n-2)
  for v in range(1, n-1):
    k = v-1  # theta index

    # window bounds in theta space
    lo = max(0, k - win)
    hi = min((n-2)-1, k + win)

    # 1) local lag-1 tangent correlation average over segments
    # corresponds to averaging cos(theta_j)
    cos_vals = [math.cos(theta[j]) for j in range(lo, hi+1)]
    c1 = sum(cos_vals) / len(cos_vals)

    # 2) alternation rate among significant turns
    signs = []
    for j in range(lo, hi+1):
      if abs(theta[j]) >= theta_min:
        signs.append(1 if theta[j] > 0 else -1)
    if len(signs) < 4:
      continue
    flips = sum(1 for j in range(len(signs)-1) if signs[j] != signs[j+1])
    alt_rate = flips / (len(signs)-1)

    if c1 < c1_thresh and alt_rate >= alt_rate_thresh:
      flagged_inner[k] = True

  # expand to vertex mask (size n), mark vertices v=1..n-2
  mask = [False]*n
  for v in range(1, n-1):
    if flagged_inner[v-1]:
      mask[v] = True

  # group into runs, enforce min_run
  runs = []
  i = 0
  while i < n:
    if not mask[i]:
      i += 1
      continue
    a = i
    while i < n and mask[i]:
      i += 1
    b = i-1
    if (b - a + 1) >= min_run:
      runs.append((a, b))
    else:
      # too short -> unflag
      for j in range(a, b+1):
        mask[j] = False

  return mask, runs


def merge_runs_by_chord(
    pts: List[Point],
    runs: List[Tuple[int, int]],
    *,
    merge_tol: float = 1.5,   # max deviation allowed (in your units)
) -> List[Point]:
  if not runs:
    return pts[:]

  out = []
  n = len(pts)
  r_idx = 0
  i = 0

  while i < n:
    if r_idx < len(runs) and i == runs[r_idx][0]:
      a, b = runs[r_idx]
      # include start point once
      if not out or out[-1] != pts[a]:
        out.append(pts[a])

      # test deviation to chord
      A, B = pts[a], pts[b]
      max_dev = 0.0
      for k in range(a+1, b):
        max_dev = max(max_dev, _point_line_dist(pts[k], A, B))

      if max_dev <= merge_tol:
        # collapse entire run to endpoint
        if out[-1] != pts[b]:
          out.append(pts[b])
      else:
        # not flat enough: keep points (or you could do localized RDP here)
        for k in range(a+1, b+1):
          out.append(pts[k])

      i = b + 1
      r_idx += 1
    else:
      out.append(pts[i])
      i += 1

  # remove accidental duplicates
  cleaned = [out[0]]
  for p in out[1:]:
    if p != cleaned[-1]:
      cleaned.append(p)
  return cleaned
