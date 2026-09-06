"""Letter geometry for the corpus index: frame, outline, mask codec.

Three things live here, all pure and testable without a database.

**The normalized letter frame.** A letter measured on page 3 and the same
letter measured on page 587 are only comparable after the scan scale and the
qalam width are divided out, and only overlay on each other after they are
placed on a common origin. The frame therefore puts the origin at the letter's
entry point on the baseline, points y up and x along the writing direction
(leftwards in the image), and measures in qalam widths -- the page nominal
stroke width from ``paw_width.estimate_page_nominal_width``.

Note the map negates both axes, which is a rotation, not a mirror, so contour
winding survives it and ``fill-rule: nonzero`` behaves the same as it does in
``tools/tracing/svg_from_json_words_no_potrace.py``.

**The outline.** The same no-potrace tracing pipeline the font builder uses,
run on one letter's mask instead of a whole word's, so the stored path is the
letter as it was actually segmented -- entry and exit cut chords included,
which is precisely the join geometry a font needs.

**The mask codec.** Row-major run lengths, zlib'd. 360k letters means 360k
files if these were PNGs on disk; as blobs they cost a few hundred bytes each
and keep a pixel-exact crop available without re-running segmentation.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import List, Optional, Sequence, Tuple
import zlib

import numpy as np

_TRACING = Path(__file__).resolve().parents[1] / "tracing"
if str(_TRACING) not in sys.path:
  sys.path.insert(0, str(_TRACING))

from bezier_fit import FitConfig                      # noqa: E402
from png_to_ttf_no_potrace import (                   # noqa: E402
    contour_to_beziers,
    find_contours_with_hierarchy,
    skip_tiny_hole_subtrees,
)

Point = Tuple[float, float]
Cubic = Tuple[Point, Point, Point, Point]

# Matches the defaults of the production SVG builder's CLI, so a letter traced
# here and the same ink traced as part of its word agree.
DEFAULT_FIT = FitConfig(tol=8.0, min_arc_height=6.0, reparam_iters=5,
                        straight_seg_tol=9.0)
MIN_HOLE_AREA_PX = 500.0
MIN_CONTOUR_POINTS = 12
TRACE_PAD_PX = 4          # keep contours off the crop edge


@dataclass(frozen=True)
class LetterFrame:
  """Word-space -> normalized-letter-space map."""
  entry_x: float
  entry_y: float
  baseline_y: float
  scale: float            # page nominal stroke width, in word-space pixels

  def point(self, x: float, y: float) -> Point:
    return ((self.entry_x - x) / self.scale,
            (self.baseline_y - y) / self.scale)

  def length(self, value: float) -> float:
    return value / self.scale


def trace_mask(mask: np.ndarray, origin: Tuple[int, int],
               cfg: FitConfig = DEFAULT_FIT,
               min_hole_area_px: float = MIN_HOLE_AREA_PX) -> List[List[Cubic]]:
  """Trace one binary letter mask, returning one cubic list per contour.

  ``mask`` is a crop; ``origin`` is that crop's (x, y) in word space, so the
  returned control points are already in word space. Contours are kept apart
  rather than concatenated because a letter with a counter (a loop) needs its
  hole to be a separate subpath for the fill rule to cut it out.
  """
  if mask.size == 0 or not mask.any():
    return []
  padded = np.zeros((mask.shape[0] + 2 * TRACE_PAD_PX,
                     mask.shape[1] + 2 * TRACE_PAD_PX), dtype=np.uint8)
  padded[TRACE_PAD_PX:TRACE_PAD_PX + mask.shape[0],
         TRACE_PAD_PX:TRACE_PAD_PX + mask.shape[1]] = np.where(mask, 255, 0)

  nodes = find_contours_with_hierarchy(padded)
  if not nodes:
    return []
  skip = skip_tiny_hole_subtrees(nodes, min_hole_area_px)

  off_x = float(origin[0]) - TRACE_PAD_PX
  off_y = float(origin[1]) - TRACE_PAD_PX
  contours: List[List[Cubic]] = []
  for i, node in enumerate(nodes):
    if i in skip or len(node.pts) < MIN_CONTOUR_POINTS:
      continue
    fitted = contour_to_beziers(pts_np=node.pts, is_hole=node.is_hole, cfg=cfg)
    if not fitted.cubics:
      continue
    contours.append([tuple((px + off_x, py + off_y) for px, py in cubic)
                     for cubic in fitted.cubics])
  return contours


def contours_to_path(contours: Sequence[Sequence[Cubic]], frame: LetterFrame,
                     precision: int = 3
                     ) -> Tuple[str, Optional[Tuple[float, float, float, float]]]:
  """Emit one SVG path in the normalized frame, plus its bounding box."""
  fmt = f"{{:.{precision}f}}"
  parts: List[str] = []
  xs: List[float] = []
  ys: List[float] = []

  def emit(point: Point) -> str:
    x, y = frame.point(*point)
    xs.append(x)
    ys.append(y)
    return f"{fmt.format(x)},{fmt.format(y)}"

  for cubics in contours:
    if not cubics:
      continue
    parts.append(f"M {emit(cubics[0][0])}")
    for _, control1, control2, end in cubics:
      parts.append(f"C {emit(control1)} {emit(control2)} {emit(end)}")
    parts.append("Z")

  if not xs:
    return "", None
  box = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
  return " ".join(parts), box


def encode_mask_rle(mask: np.ndarray) -> bytes:
  """Row-major run lengths of a binary mask, starting with a zero run."""
  flat = np.ascontiguousarray(mask).reshape(-1)
  if flat.size == 0:
    return b""
  flat = (flat > 0).astype(np.uint8)
  changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
  bounds = np.concatenate(([0], changes, [flat.size]))
  runs = np.diff(bounds).astype(np.uint32)
  if flat[0]:
    runs = np.concatenate(([0], runs)).astype(np.uint32)
  return zlib.compress(runs.tobytes(), 6)


def decode_mask_rle(blob: bytes, width: int, height: int) -> np.ndarray:
  """Inverse of `encode_mask_rle`."""
  out = np.zeros(width * height, dtype=np.uint8)
  if not blob:
    return out.reshape(height, width)
  runs = np.frombuffer(zlib.decompress(blob), dtype=np.uint32)
  position = 0
  for index, run in enumerate(runs):
    if index % 2 and run:
      out[position:position + int(run)] = 1
    position += int(run)
  return out.reshape(height, width)


def advance_width(entry: Optional[Point], exit_: Optional[Point]) -> Optional[float]:
  """Baseline-projected distance from a letter's entry to its exit point.

  This is the font-relevant width: what the pen advances while writing the
  letter, as opposed to how far its ink happens to reach (bbox) or how far the
  stroke travels getting there (spine extent).
  """
  if entry is None or exit_ is None:
    return None
  return abs(float(entry[0]) - float(exit_[0]))


def normalized(value: Optional[float], scale: Optional[float]) -> Optional[float]:
  if value is None or not scale:
    return None
  return float(value) / float(scale)


def relative_path(path: str, root: str) -> str:
  """Store asset paths relative to the analysis root, so the index survives a move."""
  if not path:
    return ""
  try:
    return os.path.relpath(path, root)
  except ValueError:
    return path
