from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import math

Point = Tuple[float, float]


# ----------------------------
# Input segment type (as you described)
# ----------------------------
@dataclass
class StraightSegment:
  i0: int          # index into original pts (start)
  i1: int          # index into original pts (end)
  p0: Point        # fitted polyline vertex at start
  p1: Point        # fitted polyline vertex at end
  chord_len: float
  direction: Point  # (dx,dy) direction (not necessarily unit)
  angle_deg: float


# ----------------------------
# Small geometry helpers
# ----------------------------
def v_sub(a: Point, b: Point) -> Point:
  return (a[0] - b[0], a[1] - b[1])


def v_dot(a: Point, b: Point) -> float:
  return a[0] * b[0] + a[1] * b[1]


def v_cross(a: Point, b: Point) -> float:
  return a[0] * b[1] - a[1] * b[0]


def v_norm(a: Point) -> float:
  return math.hypot(a[0], a[1])


def v_unit(a: Point, eps: float = 1e-12) -> Point:
  n = v_norm(a)
  if n <= eps:
    return (0.0, 0.0)
  return (a[0] / n, a[1] / n)


def clamp(x: float, lo: float, hi: float) -> float:
  return lo if x < lo else hi if x > hi else x


def signnz(x: float, eps: float = 1e-12) -> int:
  if x > eps:
    return 1
  if x < -eps:
    return -1
  return 0


# ----------------------------
# Build a vertex polyline from fitted straight segments
# ----------------------------
@dataclass
class FittedPolyline:
  V: List[Point]          # vertices, length = len(straights)+1
  s: List[float]          # cumulative arc-length at vertices
  # unit direction per fitted segment, length = len(straights)
  seg_dir: List[Point]
  seg_len: List[float]    # segment length, length = len(straights)
  # mapping to original point indices (best-effort)
  v_to_orig_idx: List[int]  # length = len(V)


def build_fitted_polyline(straights: List[StraightSegment]) -> FittedPolyline:
  if not straights:
    return FittedPolyline(V=[], s=[], seg_dir=[], seg_len=[], v_to_orig_idx=[])

  V: List[Point] = [straights[0].p0]
  v_to_orig_idx: List[int] = [straights[0].i0]

  seg_dir: List[Point] = []
  seg_len: List[float] = []

  for seg in straights:
    V.append(seg.p1)
    v_to_orig_idx.append(seg.i1)

    d = v_unit(seg.direction)
    seg_dir.append(d)

    L = seg.chord_len if seg.chord_len > 0 else v_norm(
        v_sub(seg.p1, seg.p0))
    seg_len.append(L)

  s = [0.0]
  for L in seg_len:
    s.append(s[-1] + L)

  return FittedPolyline(V=V, s=s, seg_dir=seg_dir, seg_len=seg_len, v_to_orig_idx=v_to_orig_idx)


# ----------------------------
# Curvature proxy: signed turning angle at each interior vertex
# theta[v] corresponds to vertex index v, for v=1..M-1 where M=len(V)-1 segments.
# We'll store theta array aligned to vertices length len(V) with zeros at endpoints.
# ----------------------------
@dataclass
class CurvatureSignal:
  # signed turning angle at vertex (radians), length len(V)
  theta: List[float]
  abs_theta: List[float]   # abs turning at vertex, length len(V)


def compute_turning_angles(fp: FittedPolyline, eps: float = 1e-12) -> CurvatureSignal:
  V = fp.V
  m = len(V)
  if m < 3:
    return CurvatureSignal(theta=[0.0] * m, abs_theta=[0.0] * m)

  theta = [0.0] * m
  abs_theta = [0.0] * m

  # Use fitted segment directions if provided
  t = fp.seg_dir  # length m-1

  for v in range(1, m - 1):
    a = t[v - 1]
    b = t[v]
    # robust atan2 for signed turn
    cr = v_cross(a, b)
    dt = v_dot(a, b)
    ang = math.atan2(cr, dt)
    theta[v] = ang
    abs_theta[v] = abs(ang)

  return CurvatureSignal(theta=theta, abs_theta=abs_theta)


# ----------------------------
# Gaussian smoothing on a signal defined at vertices, along arc length (not index)
# We do local kernel per vertex using neighbors within 3*sigma.
# Complexity O(N * neighbors_per_sigma) — good for moderate N.
# ----------------------------
def gaussian_smooth_along_arclength(
    x: List[float],
    s: List[float],
    sigma: float,
) -> List[float]:
  n = len(x)
  if n == 0 or sigma <= 0:
    return x[:]

  out = [0.0] * n
  r = 3.0 * sigma

  # two pointers window
  j0 = 0
  j1 = 0
  for i in range(n):
    si = s[i]
    while j0 < n and s[j0] < si - r:
      j0 += 1
    if j1 < j0:
      j1 = j0
    while j1 < n and s[j1] <= si + r:
      j1 += 1

    wsum = 0.0
    vsum = 0.0
    for j in range(j0, j1):
      ds = s[j] - si
      w = math.exp(-0.5 * (ds * ds) / (sigma * sigma))
      wsum += w
      vsum += w * x[j]
    out[i] = vsum / wsum if wsum > 1e-12 else x[i]

  return out


# ----------------------------
# Candidate inflection detection: zero-crossings of smoothed theta
# We return candidate vertex indices (1..len(V)-2), plus a local "crossing strength".
# ----------------------------
@dataclass
class Candidate:
  v: int                 # vertex index in fitted polyline
  s: float               # arc-length position
  strength: float        # local curvature energy proxy
  base_sign_left: int    # sign just left (in smoothed signal)
  base_sign_right: int   # sign just right


def find_zero_crossing_candidates(
    theta_sm: List[float],
    s: List[float],
    *,
    theta_min_abs: float,
    # strength neighborhood in arc length:
    strength_radius: float,
) -> List[Candidate]:
  n = len(theta_sm)
  if n < 3:
    return []

  # helper to compute local strength around vertex v
  def local_strength(v: int) -> float:
    sv = s[v]
    lo = v
    while lo > 0 and s[lo] >= sv - strength_radius:
      lo -= 1
    hi = v
    while hi < n - 1 and s[hi] <= sv + strength_radius:
      hi += 1
    # sum abs theta (smoothed) in neighborhood
    tot = 0.0
    for k in range(lo, hi + 1):
      tot += abs(theta_sm[k])
    return tot

  cands: List[Candidate] = []
  # consider sign changes across adjacent vertices
  for v in range(1, n - 2):
    a = theta_sm[v]
    b = theta_sm[v + 1]
    sa = signnz(a)
    sb = signnz(b)
    if sa == 0 and abs(a) < theta_min_abs:
      continue
    if sb == 0 and abs(b) < theta_min_abs:
      continue
    if sa == 0:
      # look left for a sign
      u = v
      while u > 0 and signnz(theta_sm[u]) == 0:
        u -= 1
      sa = signnz(theta_sm[u])
    if sb == 0:
      # look right for a sign
      u = v + 1
      while u < n - 1 and signnz(theta_sm[u]) == 0:
        u += 1
      sb = signnz(theta_sm[u]) if u < n else 0

    if sa != 0 and sb != 0 and sa != sb:
      # candidate around v+0.5 -> pick v+1 as representative
      cv = v + 1
      st = local_strength(cv)
      cands.append(Candidate(
          v=cv,
          s=s[cv],
          strength=st,
          base_sign_left=sa,
          base_sign_right=sb
      ))

  # de-duplicate candidates that are too close (can happen with flat near-zero)
  # keep strongest in very small neighborhood
  cands.sort(key=lambda c: c.s)
  merged: List[Candidate] = []
  close_tol = 1e-6
  for c in cands:
    if not merged or abs(c.s - merged[-1].s) > close_tol:
      merged.append(c)
    else:
      if c.strength > merged[-1].strength:
        merged[-1] = c
  return merged


# ----------------------------
# Multi-scale persistence scoring
# A candidate is "persistent" at a given sigma if a sign-change exists within
# +/- search_radius around its arc-length position.
# Score combines:
# - persistence count / fraction
# - strength (curvature energy) near candidate
# - penalty for high-frequency flips (optional)
# ----------------------------
@dataclass
class ScoredCandidate:
  cand: Candidate
  score: float
  persistence: int
  persistence_frac: float
  # debug info
  per_sigmas: Dict[float, bool]


def score_candidates_multiscale(
    fp: FittedPolyline,
    curv: CurvatureSignal,
    *,
    sigmas: List[float],
    base_sigma: float,
    theta_min_abs_deg: float = 4.0,
    strength_radius: float = 20.0,
    search_radius_factor: float = 1.25,  # search_radius = factor * sigma
    min_persistence: int = 2,            # require at least this many scales
    w_persist: float = 2.0,
    w_strength: float = 1.0,
    w_scale_bonus: float = 0.25,         # small bonus for persisting at larger scales
) -> List[ScoredCandidate]:
  V = fp.V
  s = fp.s
  n = len(V)
  if n < 3:
    return []

  theta_min_abs = math.radians(theta_min_abs_deg)

  # base candidates from base_sigma smoothed curvature
  theta_base = gaussian_smooth_along_arclength(curv.theta, s, base_sigma)
  base_cands = find_zero_crossing_candidates(
      theta_base, s,
      theta_min_abs=theta_min_abs,
      strength_radius=strength_radius
  )
  if not base_cands:
    return []

  # Precompute smoothed signals at all scales
  theta_by_sigma: Dict[float, List[float]] = {base_sigma: theta_base}
  for sig in sigmas:
    if sig not in theta_by_sigma:
      theta_by_sigma[sig] = gaussian_smooth_along_arclength(
          curv.theta, s, sig)

  # Helper: check if sign change exists near arc-length position at this sigma
  def persists_at_sigma(c: Candidate, sig: float) -> bool:
    th = theta_by_sigma[sig]
    sr = max(1e-9, search_radius_factor * sig)
    # find indices within [c.s - sr, c.s + sr]
    # (linear scan around v is fine)
    v0 = c.v
    # expand left/right by arc length
    lo = v0
    while lo > 1 and s[lo] >= c.s - sr:
      lo -= 1
    hi = v0
    while hi < n - 2 and s[hi] <= c.s + sr:
      hi += 1

    # look for any adjacent sign change in [lo..hi]
    prev = signnz(th[lo])
    for i in range(lo + 1, hi + 1):
      cur = signnz(th[i])
      if prev == 0:
        prev = cur
        continue
      if cur == 0:
        continue
      if cur != prev:
        return True
      prev = cur
    return False

  scored: List[ScoredCandidate] = []
  for c in base_cands:
    per = {}
    pcount = 0
    # count persistence across requested scales (including base if present)
    for sig in sigmas:
      ok = persists_at_sigma(c, sig)
      per[sig] = ok
      if ok:
        pcount += 1

    if pcount < min_persistence:
      continue

    pfrac = pcount / max(1, len(sigmas))

    # strength already computed on base_sigma (smoothed)
    strength = c.strength

    # optional scale bonus: reward persisting at larger sigmas
    scale_bonus = 0.0
    for sig in sigmas:
      if per.get(sig, False):
        scale_bonus += w_scale_bonus * \
            (sig / (sigmas[0] if sigmas else 1.0))

    score = w_persist * pfrac + w_strength * strength + scale_bonus

    scored.append(ScoredCandidate(
        cand=c,
        score=score,
        persistence=pcount,
        persistence_frac=pfrac,
        per_sigmas=per
    ))

  # sort by arc-length
  scored.sort(key=lambda sc: sc.cand.s)
  return scored


# ----------------------------
# DP selection: pick a subset of scored candidates that are well-separated
# (and optionally limit count).
#
# This is the cleanest for "discard noisy inflections":
# - you score candidates by persistence
# - you enforce minimum spacing (arc length)
#
# Two modes:
#  1) no max_count: weighted interval scheduling (O(M log M))
#  2) with max_count: DP O(M * max_count)
# ----------------------------
@dataclass
class SelectedInflections:
  vertex_indices: List[int]         # in fitted polyline vertex indexing
  arc_positions: List[float]
  # mapped to original pts indices (best-effort)
  orig_point_indices: List[int]
  scored: List[ScoredCandidate]     # selected candidates with scores


def select_candidates_dp(
    scored: List[ScoredCandidate],
    fp: FittedPolyline,
    *,
    min_sep: float = 30.0,           # minimum separation in arc length units
    max_count: Optional[int] = None,  # if set, limit number of inflections
) -> SelectedInflections:
  if not scored:
    return SelectedInflections([], [], [], [])

  # candidates sorted by s
  s_pos = [sc.cand.s for sc in scored]
  m = len(scored)

  # prev index: largest j < i with s[j] <= s[i] - min_sep
  prev = [-1] * m
  j = 0
  for i in range(m):
    while j < i and s_pos[j] <= s_pos[i] - min_sep:
      j += 1
    prev[i] = j - 1

  # ---- Mode 1: no max_count (classic weighted interval scheduling)
  if max_count is None:
    dp = [0.0] * m
    take = [False] * m

    for i in range(m):
      w = scored[i].score
      v_take = w + (dp[prev[i]] if prev[i] >= 0 else 0.0)
      v_skip = dp[i - 1] if i > 0 else 0.0
      if v_take > v_skip:
        dp[i] = v_take
        take[i] = True
      else:
        dp[i] = v_skip
        take[i] = False

    # backtrack
    sel_idx = []
    i = m - 1
    while i >= 0:
      if take[i] and (scored[i].score + (dp[prev[i]] if prev[i] >= 0 else 0.0)) >= (dp[i - 1] if i > 0 else 0.0) - 1e-12:
        sel_idx.append(i)
        i = prev[i]
      else:
        i -= 1
    sel_idx.reverse()

  # ---- Mode 2: limit number of selections (O(M * max_count))
  else:
    K = max_count
    # dp[i][k] = best score using candidates up to i with k picks
    dp = [[0.0] * (K + 1) for _ in range(m)]
    par = [[(-1, False)] * (K + 1) for _ in range(m)]  # (prev_i, took?)

    for i in range(m):
      for k in range(K + 1):
        # skip i
        best = dp[i - 1][k] if i > 0 else (0.0 if k == 0 else -1e100)
        par[i][k] = (i - 1, False)

        # take i
        if k > 0:
          base = dp[prev[i]][k -
                             1] if prev[i] >= 0 else (0.0 if k - 1 == 0 else -1e100)
          cand = base + scored[i].score
          if cand > best:
            best = cand
            par[i][k] = (prev[i], True)

        dp[i][k] = best

    # choose best k <= K
    best_k = max(range(K + 1), key=lambda k: dp[m - 1][k])
    sel_idx = []
    i, k = m - 1, best_k
    while i >= 0 and k >= 0:
      pi, took = par[i][k]
      if took:
        sel_idx.append(i)
        k -= 1
        i = pi
      else:
        i = pi
    sel_idx.reverse()

  chosen = [scored[i] for i in sel_idx]
  v_idx = [sc.cand.v for sc in chosen]
  a_pos = [sc.cand.s for sc in chosen]
  orig_idx = [fp.v_to_orig_idx[v] if 0 <= v < len(
      fp.v_to_orig_idx) else -1 for v in v_idx]

  return SelectedInflections(
      vertex_indices=v_idx,
      arc_positions=a_pos,
      orig_point_indices=orig_idx,
      scored=chosen
  )


# ----------------------------
# Full pipeline function you asked for:
# pts: original points (not directly used except mapping indices)
# straights: fitted polyline segments (your StraightSegment list)
# Returns selected real inflections and also all scored candidates.
# ----------------------------
@dataclass
class InflectionDPResult:
  fitted_polyline: FittedPolyline
  curvature: CurvatureSignal
  scored_candidates: List[ScoredCandidate]
  selected: SelectedInflections


def detect_real_inflections_multiscale_dp(
    pts: List[Point],
    straights: List[StraightSegment],
    *,
    # multi-scale parameters
    # arc-length smoothing (same units as pts)
    base_sigma: float = 20.0,
    # additional scales to test persistence
    sigmas: Optional[List[float]] = None,
    theta_min_abs_deg: float = 6.0,           # ignore tiny turns
    # neighborhood for "strength" sum abs(theta)
    strength_radius: float = 100.0,
    min_persistence: int = 2,                 # must persist across >= this many sigmas
    # persistence neighborhood = factor*sigma
    search_radius_factor: float = 1.25,

    # scoring weights
    w_persist: float = 2.0,
    w_strength: float = 0.5,
    w_scale_bonus: float = 0.15,

    # DP selection constraints
    # minimum arc-length separation between kept inflections
    min_sep: float = 40.0,
    max_count: Optional[int] = None,          # optional cap

) -> InflectionDPResult:
  if sigmas is None:
    # good default progression around base_sigma
    sigmas = [max(1e-6, base_sigma * f) for f in (0.6, 1.0, 1.6, 2.6)]

  fp = build_fitted_polyline(straights)
  curv = compute_turning_angles(fp)

  scored = score_candidates_multiscale(
      fp,
      curv,
      sigmas=sigmas,
      base_sigma=base_sigma,
      theta_min_abs_deg=theta_min_abs_deg,
      strength_radius=strength_radius,
      search_radius_factor=search_radius_factor,
      min_persistence=min_persistence,
      w_persist=w_persist,
      w_strength=w_strength,
      w_scale_bonus=w_scale_bonus,
  )

  selected = select_candidates_dp(
      scored,
      fp,
      min_sep=min_sep,
      max_count=max_count,
  )

  return InflectionDPResult(
      fitted_polyline=fp,
      curvature=curv,
      scored_candidates=scored,
      selected=selected
  )


# ----------------------------
# Example usage:
# result = detect_real_inflections_multiscale_dp(pts, straights, base_sigma=12, min_sep=50)
# result.selected.vertex_indices -> inflection vertex indices in fitted polyline
# result.selected.orig_point_indices -> mapped indices into original pts
# ----------------------------
