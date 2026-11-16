"""
Variant of png_to_ttf_no_potrace.py that emits a single SVG file cloning a
scanned page as vector art, instead of building a TTF.

Reads a page's words.json (e.g. output/oldmedina2-analysis/page003/words.json),
traces each word's PNG (word["pngPath"]) into cubic Bezier contours with the
same no-potrace pipeline used for font building, and places each word's
contours at its page position: word["x"]/word["y"] are the top-left corner
of the word's PNG crop in page pixel coordinates (verified: PNG shape equals
word["w"] x word["h"] exactly), and pad_image() adds pad_px around that crop
before contour extraction, so a local contour point (lx, ly) maps to page
space as (x - pad_px + lx, y - pad_px + ly). SVG is y-down like the source
images, so no axis flip is needed (unlike the TTF path, which flips y to
match font em-space).

Usage:
  uv run tools/tracing/svg_from_json_words_no_potrace.py \
      output/oldmedina2-analysis 3 3
"""
import json
from html import escape
from pathlib import Path
from typing import List, Optional, Tuple

from png_to_ttf_no_potrace import (
    read_bw_grayscale, pad_image, find_contours_with_hierarchy,
    skip_tiny_hole_subtrees, contour_to_beziers, page_dir_name,
)
from bezier_fit import FitConfig

Point = Tuple[float, float]
Cubic = Tuple[Point, Point, Point, Point]


def cubics_to_path_d(cubics: List[Cubic], offx: float, offy: float) -> str:
  p0 = cubics[0][0]
  parts = [f"M {p0[0]+offx:.2f},{p0[1]+offy:.2f}"]
  for (_, b, c, d) in cubics:
    parts.append(
        f"C {b[0]+offx:.2f},{b[1]+offy:.2f} "
        f"{c[0]+offx:.2f},{c[1]+offy:.2f} "
        f"{d[0]+offx:.2f},{d[1]+offy:.2f}"
    )
  parts.append("Z")
  return " ".join(parts)


def render_svg(
    page_w: int,
    page_h: int,
    word_paths: List[Tuple[str, str, str]],
    fill: str,
    background_href: Optional[str],
) -> str:
  parts = [
      f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {page_w} {page_h}" '
      f'width="{page_w}" height="{page_h}">',
  ]
  if background_href:
    parts.append(
        f'<image href="{escape(background_href)}" x="0" y="0" '
        f'width="{page_w}" height="{page_h}" opacity="0.25"/>'
    )
  for wid, txt, d in word_paths:
    title = escape(f"{wid} {txt}".strip())
    parts.append(
        f'<path id="{escape(wid)}" d="{d}" fill="{fill}" fill-rule="nonzero">'
        f'<title>{title}</title></path>'
    )
  parts.append("</svg>")
  return "\n".join(parts)


def build_svg_from_json_words_no_potrace(
    json_words: Path,
    out_svg: Path,
    cfg: FitConfig,
    pad_px: int = 24,
    min_hole_area_px: float = 500.0,
    fill: str = "#000000",
    background_image: bool = False,
) -> Path:
  """
  background_image: if True, reference the page's original scanned image
  (words.json["sourceImage"]) as a faint backdrop via <image href=...>, for
  visually checking alignment against the scan. Referenced, not embedded,
  since source scans are tens of MB -- keep the SVG light.
  """
  json_words = Path(json_words)
  out_svg = Path(out_svg)
  out_svg.parent.mkdir(parents=True, exist_ok=True)

  data = json.loads(json_words.read_text(encoding="utf-8"))
  page_w = int(data["imageWidth"])
  page_h = int(data["imageHeight"])

  word_paths: List[Tuple[str, str, str]] = []

  for line in data.get("lines", []):
    for w in line.get("words", []):
      png_path = Path(w["pngPath"])
      if not png_path.exists():
        continue

      bw = read_bw_grayscale(png_path)
      bw = pad_image(bw, pad_px)

      nodes = find_contours_with_hierarchy(bw)
      if not nodes:
        continue
      skip_idx = skip_tiny_hole_subtrees(nodes, min_hole_area_px)

      offx = float(w["x"]) - pad_px
      offy = float(w["y"]) - pad_px

      d_parts = []
      for i, n in enumerate(nodes):
        if len(n.pts) < 12 or i in skip_idx:
          continue
        bez = contour_to_beziers(
            pts_np=n.pts,
            is_hole=n.is_hole,
            cfg=cfg,
        )
        if bez.cubics:
          d_parts.append(cubics_to_path_d(bez.cubics, offx, offy))

      if not d_parts:
        continue

      wid = str(w["wordId"])
      txt = str(w.get("expectedText", ""))
      word_paths.append((wid, txt, " ".join(d_parts)))

  background_href = None
  if background_image:
    src = data.get("sourceImage")
    if src:
      background_href = str(src)

  svg = render_svg(page_w, page_h, word_paths, fill=fill, background_href=background_href)
  out_svg.write_text(svg, encoding="utf-8")
  return out_svg


if __name__ == "__main__":
  import argparse

  ap = argparse.ArgumentParser()
  ap.add_argument("root_dir", type=str, help="Root folder, e.g. output/oldmedina2-analysis")
  ap.add_argument("start_page", type=int, help="Start page number, inclusive")
  ap.add_argument("end_page", type=int, help="End page number, inclusive")
  ap.add_argument("--fit_err_px", type=float, default=8.0)
  ap.add_argument("--min_arc_height", type=float, default=6.0)
  ap.add_argument("--reparam_iters", type=int, default=5)
  ap.add_argument("--straight_seg_tol", type=float, default=9.0)
  ap.add_argument("--pad_px", type=int, default=24)
  ap.add_argument("--min_hole_area_px", type=float, default=500.0)
  ap.add_argument("--fill", type=str, default="#000000")
  ap.add_argument("--background_image", action="store_true",
                   help="Reference the original scanned page as a faint backdrop for alignment checks")
  args = ap.parse_args()

  root_dir = Path(args.root_dir)
  if args.end_page < args.start_page:
    raise ValueError("end_page must be >= start_page")

  cfg = FitConfig(
      tol=args.fit_err_px,
      min_arc_height=args.min_arc_height,
      reparam_iters=args.reparam_iters,
      straight_seg_tol=args.straight_seg_tol,
  )

  for page_num in range(args.start_page, args.end_page + 1):
    page_name = page_dir_name(page_num)
    page_dir = root_dir / page_name
    json_words = page_dir / "words.json"
    out_svg = page_dir / f"{page_name}.svg"

    if not json_words.exists():
      print(f"Skipping {page_name}: folder not found -> {json_words}")
      continue

    print(f"Processing {page_name} ...")
    build_svg_from_json_words_no_potrace(
        json_words,
        out_svg,
        cfg=cfg,
        pad_px=args.pad_px,
        min_hole_area_px=args.min_hole_area_px,
        fill=args.fill,
        background_image=args.background_image,
    )
    print("Wrote:", out_svg)
