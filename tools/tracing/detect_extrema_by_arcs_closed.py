from typing import List, Tuple, Optional
import math

Point = Tuple[float, float]
ExtremumArc = Tuple[int, str, int, int]  # (ext_idx, etype, arc_start, arc_end)


def _unwrap_closed(pts: List[Point]) -> List[Point]:
  return pts[:-1] if len(pts) >= 2 and pts[0] == pts[-1] else pts


def _cyc_range(N: int, a: int, b: int) -> List[int]:
  """indices from a to b inclusive moving forward on ring"""
  a %= N
  b %= N
  out = [a]
  i = a
  while i != b:
    i = (i + 1) % N
    out.append(i)
  return out


def _cyc_dist_forward(N: int, a: int, b: int) -> int:
  """forward distance in steps from a to b on ring"""
  return (b - a) % N


def _local_peak_strength(
    values: List[float], i: int, left: int, right: int, N: int
) -> float:
  """how much i deviates from the endpoints' average within this arc"""
  return abs(values[i] - 0.5 * (values[left] + values[right]))


def _is_extremum_in_arc(
    values: List[float], i: int, arc_idxs: List[int], eps: float, mode: str, min_dev: int = 5
) -> bool:
  v = values[i]
  if mode == "max":
    for j in arc_idxs:
      if v < values[j] - eps:
        return False
  else:  # "min"
    for j in arc_idxs:
      if v > values[j] + eps:
        return False
  left = arc_idxs[0]
  right = arc_idxs[-1]
  leftDev = abs(values[left] - values[i])
  rightDev = abs(values[right] - values[i])

  return (leftDev > min_dev and rightDev > min_dev)

  # strictness: must beat at least one point avoid picking huge plateaus
  for j in arc_idxs:
    if mode == "max" and v > values[j] + eps:
      return True
    if mode == "min" and v < values[j] - eps:
      return True
  return False


def _arc_line_deviation_max(pts: List[Point], arc: List[int]) -> float:
  """max perpendicular deviation of arc points to PCA best-fit line"""
  if len(arc) < 3:
    return 0.0
  xs = [pts[i][0] for i in arc]
  ys = [pts[i][1] for i in arc]
  cx = sum(xs) / len(xs)
  cy = sum(ys) / len(ys)

  sxx = sum((x - cx) * (x - cx) for x in xs)
  syy = sum((y - cy) * (y - cy) for y in ys)
  sxy = sum((x - cx) * (y - cy) for x, y in zip(xs, ys))

  if abs(sxy) < 1e-12:
    dx, dy = (1.0, 0.0) if sxx >= syy else (0.0, 1.0)
  else:
    tr = sxx + syy
    det = sxx * syy - sxy * sxy
    lam = 0.5 * (tr + math.sqrt(max(0.0, tr * tr - 4.0 * det)))
    dx, dy = (lam - syy, sxy)

  L = math.hypot(dx, dy)
  if L < 1e-12:
    return 0.0
  dx /= L
  dy /= L
  nx, ny = -dy, dx

  max_dev = 0.0
  for i in arc:
    x, y = pts[i]
    d = abs((x - cx) * nx + (y - cy) * ny)
    if d > max_dev:
      max_dev = d
  return max_dev


def detect_extrema_by_arcs_closed(
    pts_closed: List[Point],
    *,
    # arc geometry in index units
    arc_min_len: int = 200,
    arc_max_len: int = 1200,
    min_end_gap: int = 60,  # extremum must be >= this far from arc ends
    step: int = 8,  # how much to advance when searching for an arc (speed)
    # extremum robustness
    eps: float = 0,
    # optional: reject extrema on near-straight arcs
    use_straight_gate: bool = False,
    # max deviation (px) for a whole arc to be considered "straight"
    straight_dev_max: float = 1.0,
    # stop conditions
    max_extrema: int = 5000,
) -> List[ExtremumArc]:
  """
  Sequential arc-based extrema detector on a closed contour (pts[0]==pts[-1] ok).

  Algorithm:
    - start at s
    - try arcs [s..e] of increasing length
    - inside arc, find strongest valid extrema (x/y max/min) that is >= min_end_gap from ends
    - commit: record (ext_idx, type, s, e) and set s = e
    - repeat until we wrap around or hit max_extrema
  """
  pts = _unwrap_closed(pts_closed)
  N = len(pts)
  if N < arc_min_len + 2:
    return []

  xs = [p[0] for p in pts]
  ys = [p[1] for p in pts]

  results: List[ExtremumArc] = []
  start = 0
  visited_steps = 0

  # to detect wraparound completion
  def done(next_start: int) -> bool:
    # if we advanced at least N steps total, we completed a full loop
    return visited_steps >= N

  while len(results) < max_extrema and not done(start):
    best = None
    # best = (score, arc_len, ext_idx, etype, end)

    # search for an arc end that yields a good extremum
    # we try arc lengths from min to max
    L = arc_min_len
    found_any = False

    while L <= arc_max_len and L < N:
      end = (start + L) % N
      arc = _cyc_range(N, start, end)
      if len(arc) < 3:
        L += step
        continue

      # straight gate: if the entire arc is almost a line, skip looking for extrema here
      if use_straight_gate:
        dev = _arc_line_deviation_max(pts, arc)
        if dev <= straight_dev_max:
          L += step
          continue

      # candidate interior indices with end-gap constraint
      interior = arc[min_end_gap: max(
          min_end_gap, len(arc) - min_end_gap)]
      if len(interior) < 3:
        L += step
        continue

      # evaluate extrema types on this arc
      # choose the single best extremum (strongest peakness) among x/y max/min
      for i in interior:
        # check each type; compute strength; keep best
        # x max
        if _is_extremum_in_arc(xs, i, arc, eps, "max"):
          sc = _local_peak_strength(xs, i, arc[0], arc[-1], N)
          cand = (sc, len(arc), i, "xmax", end)
          if best is None or cand[0] > best[0]:
            best = cand
        # x min
        if _is_extremum_in_arc(xs, i, arc, eps, "min"):
          sc = _local_peak_strength(xs, i, arc[0], arc[-1], N)
          cand = (sc, len(arc), i, "xmin", end)
          if best is None or cand[0] > best[0]:
            best = cand
        # y max
        if _is_extremum_in_arc(ys, i, arc, eps, "max"):
          sc = _local_peak_strength(ys, i, arc[0], arc[-1], N)
          cand = (sc, len(arc), i, "ymax", end)
          if best is None or cand[0] > best[0]:
            best = cand
        # y min
        if _is_extremum_in_arc(ys, i, arc, eps, "min"):
          sc = _local_peak_strength(ys, i, arc[0], arc[-1], N)
          cand = (sc, len(arc), i, "ymin", end)
          if best is None or cand[0] > best[0]:
            best = cand

      if best is not None:
        found_any = True
        break  # take the first arc length that yields a valid extremum

      L += step

    if best is None:
      # couldn't find a good arc; advance a bit to avoid infinite loop
      start = (start + arc_min_len) % N
      visited_steps += arc_min_len
      continue

    score, arc_len, ext_idx, etype, end = best
    results.append((ext_idx, etype, start, end))

    # advance to end of arc
    visited_steps += ext_idx - start + 1
    start = ext_idx

  return results
