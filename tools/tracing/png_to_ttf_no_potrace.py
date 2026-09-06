import json
import math
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict
from html import escape
import cv2
import numpy as np

# Optional: build TTF now (quadratic) using cubic->quadratic conversion
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.ttLib import TTFont
from fontTools.feaLib.builder import addOpenTypeFeatures

from preprocess_image import fit_closed_contour_to_cubics_no_corners
from smoothing_image import curve_domain_gaussian_smooth
from bezier_fit import FitConfig

Point = Tuple[float, float]
Cubic = Tuple[Point, Point, Point, Point]  # (p0, p1, p2, p3)


# -----------------------------
# Preprocess
# -----------------------------

def read_bw_grayscale(path, thresh=127):
  g0 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  if g0 is None:
    raise FileNotFoundError(f"Could not read image: {path}")
  g0 = g0.astype(np.uint8)
  return g0


def read_bw_white_is_ink(png_path: Path, thresh: int = 127) -> np.ndarray:
  img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
  if img is None:
    raise RuntimeError(f"Cannot read {png_path}")
  _, bw = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
  return bw  # 0 background, 255 ink


def downsample_if_needed(bw: np.ndarray, max_h: int = 1400) -> np.ndarray:
  h, w = bw.shape
  if h <= max_h:
    return bw
  scale = max_h / float(h)
  new_w = max(1, int(round(w * scale)))
  bw2 = cv2.resize(bw, (new_w, max_h), interpolation=cv2.INTER_AREA)
  _, bw2 = cv2.threshold(bw2, 127, 255, cv2.THRESH_BINARY)
  return bw2


def blur_and_rethreshold(bw: np.ndarray, sigma: float = 0.8, thresh: int = 127) -> np.ndarray:
  # very effective against pixel-staircase zigzag
  if sigma <= 0:
    return bw
  f = bw.astype(np.float32)
  f = cv2.GaussianBlur(f, (0, 0), sigmaX=sigma, sigmaY=sigma)
  _, out = cv2.threshold(f, thresh, 255, cv2.THRESH_BINARY)
  return out.astype(np.uint8)

# -----------------------------
# Contour extraction with holes (no potrace)
# -----------------------------


@dataclass
class ContourNode:
  pts: np.ndarray        # Nx2 float
  parent: int            # index or -1
  children: List[int]    # indices
  is_hole: bool          # derived from depth parity


def find_contours_with_hierarchy(bw_white_ink: np.ndarray) -> List[ContourNode]:
  # OpenCV expects white=foreground: good.
  contours, hier = cv2.findContours(
      bw_white_ink, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
  )
  if hier is None or len(contours) == 0:
    return []

  hier = hier[0]  # shape (n,4): next, prev, child, parent

  nodes: List[ContourNode] = []
  for i, c in enumerate(contours):
    c = c.reshape(-1, 2).astype(np.float64)
    parent = int(hier[i][3])
    nodes.append(ContourNode(pts=c, parent=parent, children=[], is_hole=False))

  # build children lists
  for i, n in enumerate(nodes):
    if n.parent >= 0:
      nodes[n.parent].children.append(i)

  # hole flag by depth parity in CCOMP tree (outer depth 0 => not hole, depth 1 => hole, etc.)
  def depth(i: int) -> int:
    d = 0
    p = nodes[i].parent
    while p >= 0:
      d += 1
      p = nodes[p].parent
    return d

  for i in range(len(nodes)):
    nodes[i].is_hole = (depth(i) % 2 == 1)

  return nodes


def skip_tiny_hole_subtrees(nodes: List[ContourNode], min_hole_area_px: float) -> set:
  """
  Returns the set of node indices to drop because they are a hole contour
  (or a descendant of one) whose area is below min_hole_area_px.

  Tiny holes are almost always scan/threshold noise (single-pixel diagonal
  staircase notches, ink-texture speckle) rather than intentional design
  counters: at 4800 DPI a physically meaningful gap is much larger than a
  few px^2. Dropping them fills the gap back to ink and avoids wasting
  cubic segments tracing sub-pixel-scale loops.
  """
  if min_hole_area_px <= 0:
    return set()

  skip: set = set()

  def mark_subtree(i: int):
    skip.add(i)
    for c in nodes[i].children:
      mark_subtree(c)

  for i, n in enumerate(nodes):
    if n.is_hole and i not in skip and abs(signed_area(n.pts)) < min_hole_area_px:
      mark_subtree(i)

  return skip


def signed_area(poly: np.ndarray) -> float:
  x = poly[:, 0]
  y = poly[:, 1]
  return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def ensure_orientation(poly: np.ndarray, want_ccw: bool) -> np.ndarray:
  a = signed_area(poly)
  is_ccw = a < 0
  if is_ccw != want_ccw:
    return poly[::-1].copy()
  return poly


# -----------------------------
# Polyline utilities: resample + RDP
# -----------------------------
def resample_closed_polyline(pts: List[Point], step: float) -> List[Point]:
  """
  Uniform resampling along a closed polyline (cyclic), using arc-length.
  pts: closed or open list; we'll treat as closed
  """
  if len(pts) < 3:
    return pts[:]

  # ensure closed without duplicating last == first
  if pts[0] == pts[-1]:
    pts = pts[:-1]

  # cumulative lengths
  n = len(pts)
  seglens = []
  total = 0.0
  for i in range(n):
    x0, y0 = pts[i]
    x1, y1 = pts[(i + 1) % n]
    L = math.hypot(x1 - x0, y1 - y0)
    seglens.append(L)
    total += L

  if total == 0:
    return pts[:]

  m = max(8, int(total / step))
  out = []
  dist_target = 0.0
  i = 0
  acc = 0.0
  for k in range(m):
    dist_target = (k * total) / m
    while acc + seglens[i] < dist_target:
      acc += seglens[i]
      i = (i + 1) % n
    # interpolate on segment i
    t = (dist_target - acc) / max(1e-12, seglens[i])
    x0, y0 = pts[i]
    x1, y1 = pts[(i + 1) % n]
    out.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
  return out


def _perp_dist(p: Point, a: Point, b: Point) -> float:
  (px, py), (ax, ay), (bx, by) = p, a, b
  vx, vy = bx - ax, by - ay
  wx, wy = px - ax, py - ay
  vv = vx * vx + vy * vy
  if vv == 0:
    return math.hypot(wx, wy)
  return abs(vx * wy - vy * wx) / math.sqrt(vv)


def rdp(points: List[Point], eps: float) -> List[Point]:
  if len(points) <= 2:
    return points[:]
  a = points[0]
  b = points[-1]
  max_d = -1.0
  idx = -1
  for i in range(1, len(points) - 1):
    d = _perp_dist(points[i], a, b)
    if d > max_d:
      max_d = d
      idx = i
  if max_d <= eps:
    return [a, b]
  left = rdp(points[:idx + 1], eps)
  right = rdp(points[idx:], eps)
  return left[:-1] + right


# -----------------------------
# Schneider FitCurves (cubic Bézier fitting)
# -----------------------------
def _v_add(a, b): return (a[0]+b[0], a[1]+b[1])
def _v_sub(a, b): return (a[0]-b[0], a[1]-b[1])
def _v_mul(a, s): return (a[0]*s, a[1]*s)
def _v_dot(a, b): return a[0]*b[0] + a[1]*b[1]
def _v_len(a): return math.hypot(a[0], a[1])


def _v_norm(a):
  l = _v_len(a)
  if l == 0:
    return (0.0, 0.0)
  return (a[0]/l, a[1]/l)


def _bezier3(p0, p1, p2, p3, t):
  u = 1.0 - t
  b0 = u*u*u
  b1 = 3*u*u*t
  b2 = 3*u*t*t
  b3 = t*t*t
  return (
      p0[0]*b0 + p1[0]*b1 + p2[0]*b2 + p3[0]*b3,
      p0[1]*b0 + p1[1]*b1 + p2[1]*b2 + p3[1]*b3,
  )


def _chord_length_params(pts: List[Point]) -> List[float]:
  d = [0.0]
  for i in range(1, len(pts)):
    d.append(d[-1] + _v_len(_v_sub(pts[i], pts[i-1])))
  total = d[-1]
  if total == 0:
    return [0.0 for _ in pts]
  return [x/total for x in d]


def _left_tangent(pts: List[Point]) -> Point:
  return _v_norm(_v_sub(pts[1], pts[0]))


def _right_tangent(pts: List[Point]) -> Point:
  return _v_norm(_v_sub(pts[-2], pts[-1]))


def _generate_bezier(pts: List[Point], u: List[float], tHat1: Point, tHat2: Point) -> Cubic:
  p0 = pts[0]
  p3 = pts[-1]

  C00 = C01 = C10 = C11 = 0.0
  X0 = X1 = 0.0

  for i in range(len(pts)):
    t = u[i]
    mt = 1.0 - t
    b0 = mt*mt*mt
    b1 = 3*mt*mt*t
    b2 = 3*mt*t*t
    b3 = t*t*t

    a1 = _v_mul(tHat1, b1)
    a2 = _v_mul(tHat2, b2)

    C00 += _v_dot(a1, a1)
    C01 += _v_dot(a1, a2)
    C10 += _v_dot(a2, a1)
    C11 += _v_dot(a2, a2)

    tmp = _v_sub(pts[i], (
        p0[0]*(b0+b1) + p3[0]*(b2+b3),
        p0[1]*(b0+b1) + p3[1]*(b2+b3),
    ))
    X0 += _v_dot(a1, tmp)
    X1 += _v_dot(a2, tmp)

  det = C00*C11 - C01*C10
  segLen = _v_len(_v_sub(p3, p0))
  if abs(det) > 1e-12:
    alpha1 = (X0*C11 - X1*C01) / det
    alpha2 = (C00*X1 - C10*X0) / det
  else:
    alpha1 = alpha2 = segLen / 3.0

  eps = 1e-6
  if alpha1 < eps or alpha2 < eps:
    alpha1 = alpha2 = segLen / 3.0

  p1 = _v_add(p0, _v_mul(tHat1, alpha1))
  p2 = _v_add(p3, _v_mul(tHat2, alpha2))
  return (p0, p1, p2, p3)


def _find_max_error(pts: List[Point], curve: Cubic, u: List[float]) -> Tuple[float, int]:
  maxDist = -1.0
  split = len(pts)//2
  for i in range(1, len(pts)-1):
    p = _bezier3(curve[0], curve[1], curve[2], curve[3], u[i])
    dist = _v_len(_v_sub(p, pts[i]))
    if dist > maxDist:
      maxDist = dist
      split = i
  return maxDist, split


def fit_cubics_open(pts: List[Point], max_error: float) -> List[Cubic]:
  """
  Fit a polyline (open) with one or more cubic Bezier segments.
  """
  if len(pts) < 2:
    return []
  if len(pts) == 2:
    p0, p3 = pts[0], pts[1]
    v = _v_sub(p3, p0)
    p1 = _v_add(p0, _v_mul(v, 1/3))
    p2 = _v_add(p0, _v_mul(v, 2/3))
    return [(p0, p1, p2, p3)]

  tHat1 = _left_tangent(pts)
  tHat2 = _right_tangent(pts)
  u = _chord_length_params(pts)
  curve = _generate_bezier(pts, u, tHat1, tHat2)

  maxDist, split = _find_max_error(pts, curve, u)
  if maxDist <= max_error:
    return [curve]

  left = fit_cubics_open(pts[:split+1], max_error)
  right = fit_cubics_open(pts[split:], max_error)
  return left + right


def fit_cubics_closed(loop_pts: List[Point], max_error: float) -> List[Cubic]:
  """
  Fit a closed loop by cutting at the farthest point from centroid and fitting as open,
  then returning cubic segments (we will close in output).
  """
  if len(loop_pts) < 4:
    return []
  # ensure no duplicate last point
  if loop_pts[0] == loop_pts[-1]:
    loop_pts = loop_pts[:-1]

  cx = sum(p[0] for p in loop_pts) / len(loop_pts)
  cy = sum(p[1] for p in loop_pts) / len(loop_pts)
  d2 = [(p[0]-cx)**2 + (p[1]-cy)**2 for p in loop_pts]
  k = int(np.argmax(d2))

  cut = loop_pts[k:] + loop_pts[:k] + \
      [loop_pts[k]]  # explicit closure for open fit
  return fit_cubics_open(cut, max_error)


# -----------------------------
# High-level: contour -> cubics (in image coords)
# -----------------------------
@dataclass
class BezierContour:
  cubics: List[Cubic]
  is_hole: bool
  points: List[Point]  # processed (oriented/scaled/blurred/resampled) points actually fit
  # The StraightSegment/InflectionFromStraights/extrema values the fitter
  # itself used to decide where to split, indexed into `points` above --
  # e.g. VectorFitViewer displays these instead of recomputing them.
  segs: list = field(default_factory=list)
  infls: list = field(default_factory=list)
  extrema: list = field(default_factory=list)


def contour_to_beziers(
    pts_np: np.ndarray,
    is_hole: bool,
    cfg: FitConfig,
) -> BezierContour:
  """
  cfg.scale/.blur/.step let a caller (VectorFitViewer) see how fitting
  behaves as if working from a lower-resolution or blurrier scan, without
  re-rasterizing the source image:
    - cfg.scale uniformly shrinks the traced contour points; cfg.tol and
      cfg.min_arc_height are scaled along with them so their meaning stays
      anchored to original-pixel units regardless of cfg.scale.
    - cfg.blur applies curve-domain Gaussian smoothing (sigma in the
      *scaled* point space -- this mirrors how blur used to run on the
      already-resized raster).
    - cfg.step is the resample spacing, in original-pixel units, forwarded
      to fit_closed_contour_to_cubics_no_corners (which already resamples).
  All three default to a no-op (scale=1, blur=0, step=0), so production
  callers (the TTF/SVG builders) are unaffected.
  """
  pts = [(float(x), float(y)) for (x, y) in pts_np.tolist()]

  # Ensure consistent orientation (outer CCW, hole CW)
  want_ccw = (not is_hole)
  pts_np2 = ensure_orientation(
      np.array(pts, dtype=np.float64), want_ccw=want_ccw)
  pts2 = [(float(x), float(y)) for (x, y) in pts_np2.tolist()]

  fit_cfg = cfg
  if cfg.scale != 1.0:
    pts2 = [(x * cfg.scale, y * cfg.scale) for (x, y) in pts2]
    fit_cfg = dataclasses.replace(
        cfg, tol=cfg.tol * cfg.scale, min_arc_height=cfg.min_arc_height * cfg.scale)

  resample_step = cfg.step * cfg.scale

  if cfg.blur > 0:
    pts2 = curve_domain_gaussian_smooth(
        pts2,
        sigma_px=cfg.blur,
        resample_step_px=resample_step if resample_step > 0 else 1.0,
        closed=True,
    )

  cubics, resampled, segs, infls, extrema = fit_closed_contour_to_cubics_no_corners(
      pts2,
      resample_step=resample_step,
      cfg=fit_cfg,
      type=2,
  )

  return BezierContour(cubics=cubics, is_hole=is_hole, points=resampled,
                        segs=segs, infls=infls, extrema=extrema)


# -----------------------------
# Emit: cubics -> TTF (quadratic) using Cu2QuPen
# -----------------------------
def beziers_to_ttglyph(bezier_contours: List[BezierContour], scale_to_upm: float, cu2qu_max_err: float = 1.0):
  base_pen = TTGlyphPen(None)
  qpen = Cu2QuPen(base_pen, max_err=cu2qu_max_err,
                  reverse_direction=False, all_quadratic=True)

  def M(p: Point) -> Point:
    return (p[0] * scale_to_upm, - p[1] * scale_to_upm)

  for bc in bezier_contours:
    if not bc.cubics:
      continue
    # move to first p0
    p0 = bc.cubics[0][0]
    qpen.moveTo(M(p0))
    last_end = p0
    for (a, b, c, d) in bc.cubics:
      # safety: ensure continuity; if not, force lineTo
      if (abs(a[0]-last_end[0]) + abs(a[1]-last_end[1])) > 1e-6:
        qpen.lineTo(M(a))
      qpen.curveTo(M(b), M(c), M(d))
      last_end = d
    qpen.closePath()

  return base_pen.glyph()


def glyph_bbox(glyph):
  coords = glyph.coordinates
  if len(coords) == 0:
    return (0, 0, 0, 0)
  xs = [x for x, y in coords]
  ys = [y for x, y in coords]
  return (min(xs), min(ys), max(xs), max(ys))


def crop_to_ink_with_offset(bw: np.ndarray) -> Tuple[np.ndarray, int, int]:
  ys, xs = np.where(bw > 0)
  if len(xs) == 0:
    return bw.copy(), 0, 0
  x0, x1 = int(xs.min()), int(xs.max())
  y0, y1 = int(ys.min()), int(ys.max())
  return bw[y0:y1+1, x0:x1+1].copy(), x0, y0


def pad_image(bw: np.ndarray, pad_px: int) -> np.ndarray:
  return cv2.copyMakeBorder(bw, pad_px, pad_px, pad_px, pad_px, cv2.BORDER_CONSTANT, value=0)


def shift_glyph(glyph, dx_units, dy_units: float):
  coords = glyph.coordinates
  if len(coords) == 0:
    return glyph
  coords.translate((dx_units, dy_units))
  glyph.coordinates = coords
  return glyph


def generate_test_html_for_page(
    json_words: Path,
    out_ttf: Path,
    pua_start: int = 0xE000,
    html_name: str = "test.html",
    font_family: str = "MushafWords",
    font_px: int = 160,
):
  """
  Creates test.html next to out_ttf.
  Renders each line as a sequence of glyphs (PUA chars) with NO spaces.
  RTL layout.

  Assumes the font cmap was assigned sequentially in the same traversal:
    for line in lines:
      for word in line.words:
        cmap[pua_start + k] = wordId
        k++
  """
  json_words = Path(json_words)
  out_ttf = Path(out_ttf)
  out_dir = out_ttf.parent
  out_dir.mkdir(parents=True, exist_ok=True)

  data = json.loads(json_words.read_text(encoding="utf-8"))
  lines = data.get("lines", [])

  # Build codepoint stream in the same order used in font build
  codepoint = pua_start
  line_runs = []  # list of list of (codepoint, wordId, expectedText)

  for line in lines:
    words = line.get("words", [])
    run = []
    for w in words:
      wid = str(w.get("wordId", ""))
      txt = str(w.get("expectedText", ""))
      run.append((codepoint, wid, txt))
      codepoint += 1
    line_runs.append(run)

  # Relative URL to the font file from the HTML file (same folder)
  font_url = out_ttf.name
  html_path = out_dir / html_name

  def cp_to_char(cp: int) -> str:
    return chr(cp)

  # Build HTML for each line: concatenate chars, no spaces
  # Also add per-word hover tooltip using <span title="">... (still no spaces)
  line_html_blocks = []
  for li, run in enumerate(line_runs):
    # Each word is one glyph char
    word_spans = []
    for (cp, wid, txt) in run:
      ch = cp_to_char(cp)
      title = f"{wid}  {txt}".strip()
      # no whitespace between spans => no visible spaces
      word_spans.append(
          f'<span class="w" title="{escape(title)}">{escape(ch)}</span>'
      )
    line_str = "".join(word_spans)
    line_html_blocks.append(
        f'<div class="line" data-line="{li}">{line_str}</div>'
    )

  html = f"""<!doctype html>
<html lang="ar" dir="rtl">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>TTF Test</title>
  <style>
    @font-face {{
      font-family: "{font_family}";
      src: url("{font_url}") format("truetype");
      font-weight: normal;
      font-style: normal;
    }}

    body {{
      margin: 0;
      padding: 16px;
      background: #0b0b0b;
      color: #eaeaea;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
    }}

    .wrap {{
      direction: rtl;
      unicode-bidi: bidi-override;
    }}

    .line {{
      font-family: "{font_family}";
      font-size: {font_px}px;
      line-height: 1.15;
      white-space: nowrap;
      margin: 0 0 10px 0;
      padding: 6px 8px;
      border-radius: 8px;
      background: rgba(255,255,255,0.06);
      overflow-x: auto;

      direction: rtl;
      unicode-bidi: bidi-override;
      font-feature-settings: "kern" 1;
      font-kerning: normal;
    }}

    .w {{      
      vertical-align: baseline;
    }}

    /* Helpful when inspecting: show a faint caret grid on hover */
    .w:hover {{
      outline: 1px solid rgba(255,255,255,0.25);
      outline-offset: 1px;
      background: rgba(255,255,255,0.05);
    }}

    .bar {{
      position: sticky;
      top: 0;
      background: rgba(20,20,20,0.92);
      backdrop-filter: blur(8px);
      padding: 10px 12px;
      border-radius: 10px;
      margin-bottom: 12px;
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
    }}

    label {{
      font-size: 13px;
      opacity: 0.9;
    }}

    input[type="range"] {{
      width: 280px;
    }}
  </style>
</head>
<body>
  <div class="bar">
    <label>Font size:
      <input id="fs" type="range" min="16" max="300" value="{font_px}" />
      <span id="fsVal">{font_px}px</span>
    </label>
  </div>

  <div class="wrap" id="wrap">
    {''.join(line_html_blocks)}
  </div>

  <script>
    const fs = document.getElementById('fs');
    const fsVal = document.getElementById('fsVal');
    const wrap = document.getElementById('wrap');

    function setSize(px) {{
      fsVal.textContent = px + "px";
      for (const line of wrap.querySelectorAll('.line')) {{
        line.style.fontSize = px + "px";
      }}
    }}

    fs.addEventListener('input', () => setSize(fs.value));
    setSize(fs.value);
  </script>
</body>
</html>
"""

  html_path.write_text(html, encoding="utf-8")
  return html_path


def generate_kern_fea_from_json_words(
    json_words: Path,
    out_fea: Path,
    *,
    units_per_px_by_glyph: Dict[str, float],
    include_languagesystems: bool = True,
    feature_tag: str = "kern",
) -> Path:
  """
  Builds an RTL kerning feature file from json_words.
  Kerning amount for pair (g1,g2) = round(gapToNextPx(g1) * units_per_px_by_glyph[g1])

  Assumes json_words lines contain words in RTL reading order (right to left):
    words[0] is rightmost, words[1] is next to the left, etc.

  If your JSON isn't guaranteed sorted, sort per line by decreasing x before building pairs.
  """
  json_words = Path(json_words)
  out_fea = Path(out_fea)
  data = json.loads(json_words.read_text(encoding="utf-8"))

  rules: Dict[Tuple[str, str], int] = {}

  for line in data.get("lines", []):
    words = list(line.get("words", []))

    # Make robust: sort RTL by decreasing x (rightmost first)
    # If your JSON is already ordered, this doesn't hurt.
    words.sort(key=lambda w: float(w.get("x", 0.0)), reverse=True)

    for i in range(len(words) - 1):
      w1 = words[i]
      w2 = words[i + 1]
      g1 = str(w1.get("wordId", ""))
      g2 = str(w2.get("wordId", ""))

      if not g1 or not g2:
        continue
      # if g1 not in units_per_px_by_glyph:
      #  continue

      gapToNextPx = float(w1.get("gapToNextPx", 0.0))
      bodyLeft = float(w1.get("bodyLeft", 0.0))
      bodyRight = float(w2.get("bodyRight", 0.0))
      gap_px = gapToNextPx + bodyLeft + bodyRight
      k = int(round(gap_px * units_per_px_by_glyph))

      # 0 kern is useless
      if k == 0:
        continue

      # If same pair appears multiple times, keep the most common or average.
      # Here: keep the first; you can also average.
      rules.setdefault((g1, g2), k)

  lines: List[str] = []
  if include_languagesystems:
    lines += [
        "languagesystem DFLT dflt;",
        "languagesystem arab dflt;",
        "",
    ]

  lines += [
      f"feature {feature_tag} {{",
      "  lookup KERN_RTL {",
  ]

  # Emit rules
  for (g1, g2), k in sorted(rules.items()):
    # In fea, `pos A B 50;` means add 50 units of advance between A and B
    # With RightToLeft lookupflag, this applies correctly for RTL runs.
    lines.append(f"    pos {g1} {g2} <{k} 0 {k} 0>;")

  lines += [
      "  } KERN_RTL;",
      f"}} {feature_tag};",
      "",
  ]

  out_fea.write_text("\n".join(lines), encoding="utf-8")
  return out_fea


def compile_fea_into_ttf(ttf_path: Path, fea_path: Path, out_ttf: Path | None = None) -> Path:
  ttf_path = Path(ttf_path)
  fea_path = Path(fea_path)
  out_ttf = Path(out_ttf) if out_ttf else ttf_path

  tt = TTFont(str(ttf_path))
  addOpenTypeFeatures(tt, str(fea_path))
  tt.save(str(out_ttf))
  return out_ttf


def build_ttf_from_json_words_no_potrace(
    json_words: Path,
    out_ttf: Path,
    cfg: FitConfig,
    family_name="MushafWords",
    style_name="Regular",
    upm=2048,
    pua_start=0xE000,
    # tuning knobs in PIXELS (image space)
    pad_px: int = 24,
    max_h_px: int = 1400,
    blur_sigma: float = 0.8,
    cu2qu_max_err_units: float = 3.0,
    min_hole_area_px: float = 500.0,

) -> Dict[str, List[BezierContour]]:
  """
  Reads json_words and builds a TTF.
  - Glyph name = wordId
  - Baseline alignment uses line.baseLine and word.y
  Returns cubics_by_glyph for future CFF build.
  """
  json_words = Path(json_words)
  out_ttf = Path(out_ttf)
  out_ttf.parent.mkdir(parents=True, exist_ok=True)

  data = json.loads(json_words.read_text(encoding="utf-8"))

  # Collect words in reading order
  items = []
  for line in data.get("lines", []):
    baseLine = float(line["baseLine"])
    prev_gap_to_next_px = 0.0
    for w in line.get("words", []):
      gap_before_px = float(prev_gap_to_next_px)
      prev_gap_to_next_px = float(w.get("gapToNextPx", 0.0))
      items.append((line, w, baseLine, gap_before_px))

  if not items:
    raise RuntimeError("No words found in JSON")

  """for line in data.get("lines", []):
    for w in line.get("words", []):
      expectedText = w.get("expectedText", "")
      if "۝" in expectedText:
        h = float(w.get("h", 0.0))
        w = float(w.get("h", 0.0))
        print(expectedText + "=(" + str(w) + "," + str(h) + ")")
  return"""

  glyph_order = [".notdef"]
  glyphs = {".notdef": TTGlyphPen(None).glyph()}
  hmtx = {".notdef": (upm, 0)}
  cmap = {}

  cubics_by_glyph: Dict[str, List[BezierContour]] = {}

  codepoint = pua_start

  ascent = int(0.85 * upm)
  descent = -int(0.25 * upm)
  lineGap = int(0.10 * upm)

  baseLineAdj = 50

  for idx, (line, w, line_base, gap_before_px) in enumerate(items):
    png_path = Path(w["pngPath"])
    gname = str(w["wordId"])

    bw = read_bw_grayscale(png_path)

    # bw, crop_x0, crop_y0 = crop_to_ink_with_offset(bw)

    # baseline inside png coords (y-down)
    baseline_in_png = float(line_base) - float(w["y"]) + baseLineAdj
    # baseline after crop (still y-down)
    baseline_after_crop = baseline_in_png  # - float(crop_y0)

    # pad
    bw = pad_image(bw, pad_px)
    baseline_after_pad = baseline_after_crop + float(pad_px)

    # downsample
    # bw, s_img = downsample_if_needed(bw, max_h_px)
    # baseline_proc_px = baseline_after_pad * s_img  # baseline y in processed image (y-down)

    baseline_proc_px = baseline_after_pad

    # blur
    # bw = blur_and_rethreshold(bw, sigma=blur_sigma)

    # contours (y-down)
    nodes = find_contours_with_hierarchy(bw)
    if not nodes:
      # skip empty
      continue

    skip_idx = skip_tiny_hole_subtrees(nodes, min_hole_area_px)

    # build cubic contours (still in y-down), then flip to y-up
    bezier_contours: List[BezierContour] = []
    for i, n in enumerate(nodes):
      if len(n.pts) < 12:
        continue
      if i in skip_idx:
        continue
      bez = contour_to_beziers(
          pts_np=n.pts,
          is_hole=n.is_hole,
          cfg=cfg,
      )
      if bez.cubics:
        bezier_contours.append(bez)

    if not bezier_contours:
      continue

    cubics_by_glyph[gname] = bezier_contours

    # Scale to UPM: map image height to ~0.9em but cap width to int16 range
    # h_px, w_px = bw.shape[:2]
    # target_h_units = 0.90 * upm
    # scale_h = target_h_units / float(max(1, h_px))
    # max_coord = 32000.0
    # scale_w = max_coord / float(max(1, w_px))
    # scale_to_upm = min(scale_h, scale_w)
    scale_to_upm = 1

    # emit TT glyph (quadratic) from cubics
    glyph = beziers_to_ttglyph(
        bezier_contours, scale_to_upm, cu2qu_max_err=cu2qu_max_err_units)

    # Baseline alignment:
    # baseline y in y-up coords is -baseline_proc_px; after scaling it's -baseline_units.
    # To make baseline at 0, translate by +baseline_units.
    baseline_units = baseline_proc_px * scale_to_upm
    bodyLeft = int(w.get("bodyLeft", 0.0))
    lsb = pad_px + bodyLeft
    glyph = shift_glyph(glyph, dx_units=-lsb, dy_units=baseline_units)

    # store glyph
    if gname not in glyphs:
      glyph_order.append(gname)
    glyphs[gname] = glyph

    # advance width (prefer JSON width if provided)
    # word['w'] is in the ORIGINAL line coordinate system; the PNG is already cropped,
    # so bbox-based width is safest:
    xMin, yMin, xMax, yMax = glyph_bbox(glyph)
    # gap_px = float(w.get("gapToNextPx", 0.0))
    # gap_units_i = int(round(gap_px * scale_to_upm))
    # adv = int(xMax - xMin) + gap_units_i
    gap_before_units_i = int(round(gap_before_px * scale_to_upm))
    # adv = int(xMax - xMin) + gap_before_units_i

    bodyRight = int(w.get("bodyRight", 0.0))
    adv = int(xMax - xMin - bodyLeft - bodyRight)
    hmtx[gname] = (adv, -bodyLeft)

    cmap[codepoint] = gname
    codepoint += 1

  # Build font
  fb = FontBuilder(upm, isTTF=True)
  fb.setupGlyphOrder(glyph_order)
  fb.setupCharacterMap(cmap)
  fb.setupGlyf(glyphs)
  fb.setupHorizontalMetrics(hmtx)
  fb.setupHorizontalHeader(ascent=ascent, descent=descent, lineGap=lineGap)
  fb.setupOS2(
      sTypoAscender=ascent,
      sTypoDescender=descent,
      sTypoLineGap=lineGap,
      usWinAscent=ascent,
      usWinDescent=abs(descent),
  )
  fb.setupNameTable({
      "familyName": family_name,
      "styleName": style_name,
      "uniqueFontIdentifier": f"{family_name} {style_name}",
      "fullName": f"{family_name} {style_name}",
      "psName": f"{family_name.replace(' ', '')}-{style_name.replace(' ', '')}",
      "version": "Version 1.000",
  })
  fb.setupPost()
  fb.setupMaxp()
  fb.save(str(out_ttf))

  fea_path = out_ttf.with_name("kern.fea")

  generate_kern_fea_from_json_words(
      json_words=json_words,
      out_fea=fea_path,
      units_per_px_by_glyph=1,
  )
  compile_fea_into_ttf(out_ttf, fea_path)

  return cubics_by_glyph


# -----------------------------
# CLI example
# -----------------------------
"""if __name__ == "__main__":
  import argparse
  ap = argparse.ArgumentParser()
  ap.add_argument("json_words", type=str)
  ap.add_argument("out_ttf", type=str)
  args = ap.parse_args()

  pua_start = 0xFC40
  # pua_start = 0xFB50

  build_ttf_from_json_words_no_potrace(Path(args.json_words), Path(
      args.out_ttf), fit_err_px=8, min_arc_height=6, reparam_iters=5, straight_seg_tol=9, pua_start=pua_start)
  print("Wrote:", args.out_ttf)

  html_path = generate_test_html_for_page(
      json_words=Path(args.json_words),
      out_ttf=Path(args.out_ttf),
      pua_start=pua_start
  )
  print("Wrote:", html_path)"""


def page_dir_name(page_num: int) -> str:
  return f"page{page_num:03d}"


if __name__ == "__main__":
  import argparse

  ap = argparse.ArgumentParser()
  ap.add_argument("root_dir", type=str, help="Root folder, e.g. output/oldmedina2-analysis")
  ap.add_argument("start_page", type=int, help="Start page number, inclusive")
  ap.add_argument("end_page", type=int, help="End page number, inclusive")
  args = ap.parse_args()

  root_dir = Path(args.root_dir)
  start_page = args.start_page
  end_page = args.end_page

  if end_page < start_page:
    raise ValueError("end_page must be >= start_page")

  pua_start = 0xFC40
  # pua_start = 0xFB50

  cfg = FitConfig(
      tol=8,
      min_arc_height=6,
      reparam_iters=5,
      straight_seg_tol=9,
      retry_without_inflection_check=True,
      use_naive_split_only=False,
  )

  for page_num in range(start_page, end_page + 1):
    page_name = page_dir_name(page_num)
    page_dir = root_dir / page_name
    json_words = page_dir / "words.json"
    out_ttf = page_dir / f"{page_name}.ttf"

    if not json_words.exists():
      print(f"Skipping {page_name}: folder not found -> {json_words}")
      continue

    print(f"Processing {page_name} ...")

    build_ttf_from_json_words_no_potrace(
        json_words,
        out_ttf,
        cfg=cfg,
        pua_start=pua_start,
    )
    print("Wrote:", out_ttf)

    html_path = generate_test_html_for_page(
        json_words=json_words,
        out_ttf=out_ttf,
        pua_start=pua_start,
    )
    print("Wrote:", html_path)
