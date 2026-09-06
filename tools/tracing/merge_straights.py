from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import math

Point = Tuple[float, float]


@dataclass
class StraightSegment:
  i0: int
  i1: int
  p0: Point
  p1: Point
  chord_len: float
  direction: Point
  angle_deg: float
  mid_index: int


# ----------------- geometry helpers -----------------

def v_sub(a: Point, b: Point) -> Point:
  return (a[0] - b[0], a[1] - b[1])


def v_dot(a: Point, b: Point) -> float:
  return a[0]*b[0] + a[1]*b[1]


def v_cross(a: Point, b: Point) -> float:
  return a[0]*b[1] - a[1]*b[0]


def v_norm(a: Point) -> float:
  return math.hypot(a[0], a[1])


def v_unit(a: Point, eps: float = 1e-12) -> Point:
  n = v_norm(a)
  if n <= eps:
    return (0.0, 0.0)
  return (a[0]/n, a[1]/n)


def signnz(x: float, eps: float = 1e-12) -> int:
  if x > eps:
    return 1
  if x < -eps:
    return -1
  return 0


def point_line_dist(p: Point, a: Point, b: Point, eps: float = 1e-12) -> float:
  ab = v_sub(b, a)
  ap = v_sub(p, a)
  lab = v_norm(ab)
  if lab <= eps:
    return v_norm(ap)
  return abs(v_cross(ap, ab)) / lab


def chord_error_max_dev(pts: List[Point], i0: int, i1: int, a: Point, b: Point) -> float:
  """Max orthogonal deviation of original points pts[i0..i1] to chord (a,b)."""
  if i1 <= i0 + 1:
    return 0.0
  md = 0.0
  for k in range(i0+1, i1):
    d = point_line_dist(pts[k], a, b)
    if d > md:
      md = d
  return md


# ----------------- build fitted vertices from straights -----------------

@dataclass
class FittedPolyline:
  V: List[Point]            # vertices length = len(straights)+1
  v_to_orig: List[int]      # vertex -> original point index (best effort)


def build_vertices(straights: List[StraightSegment]) -> FittedPolyline:
  if not straights:
    return FittedPolyline([], [])
  V = [straights[0].p0]
  v_to_orig = [straights[0].i0]
  for s in straights:
    V.append(s.p1)
    v_to_orig.append(s.i1)
  return FittedPolyline(V, v_to_orig)


# ----------------- DP core -----------------

@dataclass
class DPResult:
  kept_vertices: List[int]          # indices in fitted vertices
  merged_straights: List[StraightSegment]
  total_cost: float
  # fitted vertex indices where inflection occurs
  inflection_vertices: List[int]


def best_merge_successive_inflections(
    pts: List[Point],
    straights: List[StraightSegment],
    *,
    # if None, no geometric constraint; else require max-dev <= eps
    eps: Optional[float] = None,
    # limit how far we can merge (speed + avoid over-merging)
    max_jump: Optional[int] = 50,
    # small cost per kept chord (prefers merging)
    keep_segment_cost: float = 0.02,
    # cost per inflection (optional regularizer)
    inflection_cost: float = 0.0,
    # BIG cost if two inflections occur at adjacent original vertices
    successive_inflection_cost: float = 5.0,
    # treat tiny turns as 0-sign (neither left nor right)
    zero_turn_deg: float = 2.0,
) -> DPResult:
  fp = build_vertices(straights)
  V = fp.V
  v_to_orig = fp.v_to_orig
  nV = len(V)
  if nV < 2:
    return DPResult([0], [], 0.0, [])

  zero_turn = math.radians(zero_turn_deg)
  INF = 1e100

  chord_len = 0
  for straight in straights:
    chord_len += straight.chord_len

  min_length = chord_len / 20

  # Precompute feasible chords i->j and their base cost
  # chord cost = keep_segment_cost + (optional) error term to gently prefer better geometry
  feasible: Dict[Tuple[int, int], float] = {}
  for i in range(nV - 1):
    jmax = nV - 1
    if max_jump is not None:
      jmax = min(jmax, i + max_jump)
    for j in range(i + 1, jmax + 1):
      base = keep_segment_cost
      if eps is not None:
        oi, oj = v_to_orig[i], v_to_orig[j]
        err = chord_error_max_dev(pts, oi, oj, V[i], V[j])
        if err > eps:
          continue
        # optional: tiny preference for lower error
        base += 5 * max(err - 3, 0)
      feasible[(i, j)] = base

  def turn_sign(k: int, i: int, j: int) -> int:
    d1 = v_unit(v_sub(V[i], V[k]))
    d2 = v_unit(v_sub(V[j], V[i]))
    ang = math.atan2(v_cross(d1, d2), v_dot(d1, d2))
    if abs(ang) < zero_turn:
      return 0
    return signnz(ang)

  def get_length(i: int, j: int) -> float:
    p0 = V[i]
    p1 = V[j]
    clen = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    return clen

  # State:
  # (k, i, last_s, last_inf_v)
  # where:
  #   k,i are last two kept vertices (k < i)
  #   last_s is last nonzero turn sign already established up to vertex k (0 if none yet)
  #   last_inf_v is fitted vertex index of the last inflection encountered (-1 if none)
  #
  # Transition: choose next kept vertex j>i, compute turn at i, update inflection logic,
  # and add penalty if new inflection is adjacent to previous inflection vertex (i == last_inf_v + 1).
  best: Dict[Tuple[int, int, int, int], float] = {}
  parent: Dict[Tuple[int, int, int, int],
               Tuple[Optional[Tuple[int, int, int, int]], int]] = {}

  # Initialize by choosing first chord 0->i
  start = 0
  for i in range(start + 1, nV):
    if (start, i) not in feasible:
      continue
    st = (start, i, 0, -1)
    c = feasible[(start, i)]
    if c < best.get(st, INF):
      best[st] = c
      parent[st] = (None, start)

  # Expand
  # (We do a straightforward relaxation loop; you can also do for i in increasing order, but this is fine for moderate sizes.)
  for _ in range(nV + 5):
    updated = False
    new_best = dict(best)
    new_parent = dict(parent)

    for (k, i, last_s, last_inf_v), cost0 in best.items():
      if i == nV - 1:
        continue

      jmax = nV - 1
      if max_jump is not None:
        jmax = min(jmax, i + max_jump)

      for j in range(i + 1, jmax + 1):
        if (i, j) not in feasible:
          continue

        s_i = turn_sign(k, i, j)

        # inflection update (ignore zero turns)
        new_last_s = last_s

        new_last_inf_v = last_inf_v
        add_inf_cost = 0.0
        add_succ_cost = 0.0
        length_cost = 0.0

        if s_i != 0:
          is_inflection = (new_last_s != 0 and s_i != new_last_s)
          if is_inflection:
            add_inf_cost += inflection_cost
            ki_length = get_length(k, i)
            ij_length = get_length(i, j)
            if ki_length < min_length:
              length_cost += 5000
            if ij_length < min_length:
              length_cost += 5000

              # penalize successive inflections at adjacent original vertices
            if new_last_inf_v >= 0 and i == new_last_inf_v + 1:
              add_succ_cost += successive_inflection_cost
            new_last_inf_v = i
          new_last_s = s_i

        cost = cost0 + feasible[(i, j)] + add_inf_cost + \
            add_succ_cost + length_cost
        st2 = (i, j, new_last_s, new_last_inf_v)

        if cost < new_best.get(st2, INF):
          new_best[st2] = cost
          new_parent[st2] = ((k, i, last_s, last_inf_v), i)
          updated = True

    best, parent = new_best, new_parent
    if not updated:
      break

  # Pick best ending state (any state with i == last vertex)
  end_state = None
  end_cost = INF
  for st, c in best.items():
    _, i, _, _ = st
    if i == nV - 1 and c < end_cost:
      end_cost = c
      end_state = st
  if end_state is None:
    # fallback: best overall
    end_state = min(best, key=lambda st: best[st])
    end_cost = best[end_state]

  # Backtrack kept vertices
  kept = [nV - 1]
  st = end_state
  while True:
    par = parent.get(st)
    if par is None or par[0] is None:
      kept.append(st[0])  # k
      break
    prev_st, mid = par
    kept.append(mid)
    st = prev_st
  kept = sorted(set(kept))
  if kept[0] != 0:
    kept = [0] + kept
  if kept[-1] != nV - 1:
    kept.append(nV - 1)

  # Recompute inflection vertices on the final kept polyline (for reporting)
  inf_vs: List[int] = []
  last_s = 0
  for t in range(1, len(kept) - 1):
    k, i, j = kept[t - 1], kept[t], kept[t + 1]
    s_i = turn_sign(k, i, j)
    if s_i != 0:
      if last_s != 0 and s_i != last_s:
        inf_vs.append(i)
      last_s = s_i

  # Build merged straights (one per consecutive kept pair)
  merged: List[StraightSegment] = []
  for a, b in zip(kept[:-1], kept[1:]):
    p0, p1 = V[a], V[b]
    d = v_sub(p1, p0)
    L = v_norm(d)
    ang = math.degrees(math.atan2(d[1], d[0])) if L > 1e-12 else 0.0
    merged.append(StraightSegment(
        i0=v_to_orig[a],
        i1=v_to_orig[b],
        p0=p0,
        p1=p1,
        chord_len=L,
        direction=d,
        angle_deg=ang,
        mid_index=(v_to_orig[a] + v_to_orig[b]) // 2
    ))

  return DPResult(
      kept_vertices=kept,
      merged_straights=merged,
      total_cost=end_cost,
      inflection_vertices=inf_vs
  )
