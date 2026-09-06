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


# ----------------- fitted vertices from straights -----------------

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


# ----------------- DP result -----------------

@dataclass
class DPResult:
  kept_vertices: List[int]          # indices in fitted vertices
  merged_straights: List[StraightSegment]
  total_cost: float
  # fitted vertex indices where inflection occurs
  inflection_vertices: List[int]


def best_merge_successive_inflections_patched(
    pts: List[Point],
    straights: List[StraightSegment],
    *,
    # if None: no geometric mergeability constraint
    eps: Optional[float] = None,
    # limit merge span (in fitted vertices)
    max_jump: Optional[int] = 50,
    keep_segment_cost: float = -100.0,      # prefers merging overall
    inflection_cost: float = 0.15,        # mild regularizer (optional)
    # big penalty for "weak successive" inflections
    successive_inflection_cost: float = 8.0,
    zero_turn_deg: float = 2.0,           # tiny turns treated as 0 sign

    # ---- NEW: weak-successive gating ----
    # if two successive inflections are closer than this => suspicious
    succ_sep_min_px: float = 18.0,
    # turning energy between them (sum abs turning) below this => suspicious
    succ_energy_min_deg: float = 5.0,
    # current inflection turn magnitude below this => suspicious
    succ_turn_min_deg: float = 6.0,
    # only penalize if skipping i is feasible under eps
    require_mergeable_to_penalize: bool = False,
) -> DPResult:
  fp = build_vertices(straights)
  V = fp.V
  v_to_orig = fp.v_to_orig
  nV = len(V)
  if nV < 2:
    return DPResult([0], [], 0.0, [])

  zero_turn = math.radians(zero_turn_deg)
  succ_energy_min = math.radians(succ_energy_min_deg)
  succ_turn_min = math.radians(succ_turn_min_deg)

  INF = 1e100

  # Precompute cumulative arc length along fitted vertices (pixel units)
  S = [0.0] * nV
  for i in range(1, nV):
    S[i] = S[i-1] + v_norm(v_sub(V[i], V[i-1]))

  chord_len = S[-1]

  min_length = chord_len / 15

  # Precompute "raw" turning magnitudes at each fitted vertex (using consecutive fitted edges)
  raw_abs_turn = [0.0] * nV  # endpoints remain 0
  for i in range(1, nV-1):
    a = v_unit(v_sub(V[i], V[i-1]))
    b = v_unit(v_sub(V[i+1], V[i]))
    ang = abs(math.atan2(v_cross(a, b), v_dot(a, b)))
    raw_abs_turn[i] = ang

  # Prefix sum of raw turning magnitudes: turning energy between vertices u..v
  pref_turn = [0.0] * (nV + 1)
  for i in range(nV):
    pref_turn[i+1] = pref_turn[i] + raw_abs_turn[i]

  def turning_energy(u: int, v: int) -> float:
    """Sum abs raw turning angles over (u..v) (order-agnostic)."""
    if v < u:
      u, v = v, u
    return pref_turn[v+1] - pref_turn[u]

  # Precompute feasible chords i->j (optionally constrained by eps)
  feasible: Dict[Tuple[int, int], float] = {}
  # feasible[(0, 1)] = 0
  # feasible[(nV - 2, nV - 1)] = 0

  for i in range(0, nV - 1):
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
        base += 0 * max(err - 1, 0)
      feasible[(i, j)] = base

  def turn_angle(k: int, i: int, j: int) -> float:
    d1 = v_unit(v_sub(V[i], V[k]))
    d2 = v_unit(v_sub(V[j], V[i]))
    return math.atan2(v_cross(d1, d2), v_dot(d1, d2))

  def turn_sign(k: int, i: int, j: int) -> int:
    ang = turn_angle(k, i, j)
    if abs(ang) < zero_turn:
      return 0
    return signnz(ang)

  # State: (k, i, last_s, last_inf_v)
  best: Dict[Tuple[int, int, int, int], float] = {}
  parent: Dict[Tuple[int, int, int, int],
               Tuple[Optional[Tuple[int, int, int, int]], int]] = {}

  # init: first chord 0->i
  start = 0
  for i in range(start + 1, nV):
    if (start, i) not in feasible:
      continue
    st = (start, i, 0, -1)
    c = feasible[(start, i)]
    if c < best.get(st, INF):
      best[st] = c
      parent[st] = (None, start)

  # DP relax
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
        ang_i = abs(turn_angle(k, i, j))

        new_last_s = last_s
        new_last_inf_v = last_inf_v

        add_inf_cost = 0.0
        add_succ_cost = 0.0

        if s_i != 0:
          is_inflection = (new_last_s != 0 and s_i != new_last_s)
          if is_inflection:
            add_inf_cost += inflection_cost

            # ---- successive-inflection-events detection: last inflection was at k ----
            if new_last_inf_v >= 0 and new_last_inf_v == k:
              # 1) separation (arc length)
              sepki = abs(S[i] - S[k])  # px
              sepjk = abs(S[j] - S[i])  # px
              # 2) turning energy between the two inflection vertices (raw turns)
              energy = turning_energy(k, i)  # radians
              # 3) optional mergeability: could we skip i by using chord (k -> j)?
              mergeable_skip_i = ((k, j) in feasible) if eps is not None else True

              weak = ((sepki < min_length) or (sepjk < min_length)) or (
                  energy < succ_energy_min) or (ang_i < succ_turn_min)
              if require_mergeable_to_penalize:
                weak = weak and mergeable_skip_i

              if weak:
                add_succ_cost += 0
              add_succ_cost = 5000
            else:
              sepki = abs(S[i] - S[k])  # px
              sepjk = abs(S[j] - S[i])  # px
              if (sepki < min_length) or (sepjk < min_length):
                add_inf_cost = 0

            new_last_inf_v = i
          new_last_s = s_i

        cost = cost0 + feasible[(i, j)] + add_inf_cost + add_succ_cost
        st2 = (i, j, new_last_s, new_last_inf_v)

        if cost < new_best.get(st2, INF):
          new_best[st2] = cost
          new_parent[st2] = ((k, i, last_s, last_inf_v), i)
          updated = True

    best, parent = new_best, new_parent
    if not updated:
      break

  # best ending state
  end_state = None
  end_cost = INF
  for st, c in best.items():
    _, i, _, _ = st
    if i == nV - 1 and c < end_cost:
      end_cost = c
      end_state = st
  if end_state is None:
    end_state = min(best, key=lambda st: best[st])
    end_cost = best[end_state]

  # backtrack kept vertices
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

  # recompute inflections on kept polyline
  inf_vs: List[int] = []
  last_s = 0
  for t in range(1, len(kept) - 1):
    kk, ii, jj = kept[t - 1], kept[t], kept[t + 1]
    si = turn_sign(kk, ii, jj)
    if si != 0:
      if last_s != 0 and si != last_s:
        inf_vs.append(ii)
      last_s = si

  # build merged straights
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
