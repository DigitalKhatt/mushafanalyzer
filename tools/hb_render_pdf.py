#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import uharfbuzz as hb
from fontTools.ttLib import TTFont
from fontTools.pens.basePen import BasePen
from fontTools.pens.qu2cuPen import Qu2CuPen
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4


SQL = """
select l.page,l.line,w.qpc_v1,w.dk_v1
from words w
INNER JOIN qpc_v1_layout l
  ON l.type = 'ayah'
 AND l.range_start <= w.word_number_all
 AND l.range_end   >= w.word_number_all
ORDER BY l.page, l.line, w.word_number_all;
"""


# ----------------------------
# DB layout loading
# ----------------------------

def load_layout(sqlite_path: Path) -> Dict[int, Dict[int, List[str]]]:
    """
    pages[page][line] = list of qpc_v1 "word selector" strings (usually 1 PUA char each)
    in logical reading order from your query.
    """
    con = sqlite3.connect(str(sqlite_path))
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute(SQL)
    rows = cur.fetchall()
    con.close()

    pages: Dict[int, Dict[int, List[str]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        page = int(r["page"])
        line = int(r["line"])
        qpc = str(r["qpc_v1"]) if r["qpc_v1"] is not None else ""
        if qpc:
            pages[page][line].append(qpc)
    return pages


# ----------------------------
# ReportLab glyph outline pen
# ----------------------------

class ReportLabPathPen(BasePen):
    """
    Writes cubic béziers into a ReportLab path (in font units).
    """
    def __init__(self, glyphSet, rl_path):
        super().__init__(glyphSet)
        self.p = rl_path

    def _moveTo(self, p0):
        self.p.moveTo(p0[0], p0[1])

    def _lineTo(self, p1):
        self.p.lineTo(p1[0], p1[1])

    def _curveToOne(self, p1, p2, p3):
        self.p.curveTo(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1])

    def _closePath(self):
        self.p.close()


def draw_glyph(ttf: TTFont, glyph_name: str, c: canvas.Canvas,
               x_pt: float, y_pt: float, scale: float):
    """
    Draw a glyph outline filled at (x_pt, y_pt) in PDF points.
    `scale` converts font units -> points.
    """
    glyphSet = ttf.getGlyphSet()
    g = glyphSet[glyph_name]

    path = c.beginPath()
    rl_pen = ReportLabPathPen(glyphSet, path)

    # Convert TrueType quadratics -> cubics for PDF
    qu2cu = Qu2CuPen(rl_pen, max_err=1.0, reverse_direction=False)
    g.draw(qu2cu)

    c.saveState()
    c.translate(x_pt, y_pt)
    c.scale(scale, scale)
    c.drawPath(path, fill=1, stroke=0)
    c.restoreState()


# ----------------------------
# HarfBuzz shaping
# ----------------------------

def hb_shape(font_bytes: bytes, text: str,
             direction: str = "RTL",
             script: str = "arab",
             lang: str = "ar",
             features: Dict[str, Any] | None = None):
    """
    Returns (infos, positions, upem).
    features: e.g. {"liga": True, "calt": True, "ss01": True}
    """
    face = hb.Face(font_bytes)
    hb_font = hb.Font(face)
    upem = face.upem
    hb_font.scale = (upem, upem)

    buf = hb.Buffer()
    buf.add_str(text)
    buf.direction = direction
    buf.script = script
    buf.language = lang
    buf.guess_segment_properties()

    hb.shape(hb_font, buf, features or {})
    return buf.glyph_infos, buf.glyph_positions, upem


def render_line_hb_pagefont(ttf_path: Path, c: canvas.Canvas,
                            words_qpc: List[str],
                            x_right: float, y: float,
                            font_size_pt: float,
                            hb_features: Dict[str, Any] | None = None):
    """
    Render a line using HarfBuzz on the per-page font and draw glyph outlines.
    words_qpc: sequence of qpc_v1 items; we concatenate them (no spaces).
    """
    font_bytes = ttf_path.read_bytes()
    ttf = TTFont(str(ttf_path))
    glyph_order = ttf.getGlyphOrder()

    # Concatenate qpc word selectors as stored (usually 1 char per word)
    line_text = "".join(words_qpc)

    infos, poss, upem = hb_shape(
        font_bytes,
        line_text,
        direction="RTL",
        script="arab",
        lang="ar",
        features=hb_features
    )

    scale = font_size_pt / float(upem)

    # total advance in font units
    total_adv_units = 0
    for p in poss:
        total_adv_units += p.x_advance
    total_adv_pt = total_adv_units * scale

    # Right align: start at x_right - totalAdvance
    pen_x = x_right - total_adv_pt
    pen_y = y

    # Draw all glyphs
    for info, pos in zip(infos, poss):
        gid = info.codepoint
        if gid < 0 or gid >= len(glyph_order):
            # still advance
            pen_x += pos.x_advance * scale
            pen_y += pos.y_advance * scale
            continue

        gname = glyph_order[gid]

        gx = pen_x + (pos.x_offset * scale)
        gy = pen_y + (pos.y_offset * scale)

        draw_glyph(ttf, gname, c, gx, gy, scale)

        pen_x += pos.x_advance * scale
        pen_y += pos.y_advance * scale


# ----------------------------
# PDF driver (604 pages)
# ----------------------------

def compute_font_size_fit_widest_line(
    ttf_path: Path,
    lines: List[List[str]],
    avail_width_pt: float,
    font_min: float,
    font_max: float,
    hb_features: Dict[str, Any] | None = None,
) -> float:
    """
    Determine a font size so the widest line fits in avail_width_pt.
    We use HarfBuzz to get total advance per line at UPEM scale, then scale to points.
    """
    font_bytes = ttf_path.read_bytes()
    _, _, upem = hb_shape(font_bytes, " ", features=hb_features)  # get upem fast

    max_adv_units = 0
    for words in lines:
        text = "".join(words)
        infos, poss, _ = hb_shape(font_bytes, text, features=hb_features)
        adv = sum(p.x_advance for p in poss)
        if adv > max_adv_units:
            max_adv_units = adv

    if max_adv_units <= 0:
        return font_min

    fs = avail_width_pt * (upem / float(max_adv_units))
    return max(font_min, min(font_max, fs))


def main():
    ap = argparse.ArgumentParser(description="Render per-page TTF mushaf using HarfBuzz shaping to PDF.")
    ap.add_argument("--sqlite", required=True, type=Path, help="Path to quran-data.sqlite")
    ap.add_argument("--fonts_dir", required=True, type=Path, help="Directory containing page_###.ttf fonts")
    ap.add_argument("--font_template", default="page_{page:03d}.ttf",
                    help="Template for page font filenames, uses {page}. Default page_{page:03d}.ttf")
    ap.add_argument("--out", required=True, type=Path, help="Output PDF path")

    ap.add_argument("--page_w", type=float, default=A4[0], help="Page width (pt)")
    ap.add_argument("--page_h", type=float, default=A4[1], help="Page height (pt)")
    ap.add_argument("--margin_l", type=float, default=36.0)
    ap.add_argument("--margin_r", type=float, default=36.0)
    ap.add_argument("--margin_t", type=float, default=36.0)
    ap.add_argument("--margin_b", type=float, default=36.0)
    ap.add_argument("--leading", type=float, default=1.25, help="Line spacing factor")

    ap.add_argument("--font_min", type=float, default=7.0)
    ap.add_argument("--font_max", type=float, default=28.0)

    # HarfBuzz features toggles (optional)
    ap.add_argument("--hb_features", type=str, default="",
                    help="Comma-separated HB features: e.g. 'liga=1,calt=1,ss01=1' (optional)")

    args = ap.parse_args()

    pages = load_layout(args.sqlite)
    c = canvas.Canvas(str(args.out), pagesize=(args.page_w, args.page_h))
    c.setFillGray(0)  # black fill

    # Parse HB feature string
    hb_features = None
    if args.hb_features.strip():
        hb_features = {}
        for part in args.hb_features.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" in part:
                k, v = part.split("=", 1)
                hb_features[k.strip()] = bool(int(v.strip()))
            else:
                hb_features[part] = True

    for page_no in sorted(pages.keys()):
        ttf_path = args.fonts_dir / args.font_template.format(page=page_no)
        if not ttf_path.exists():
            raise FileNotFoundError(f"Missing page font: {ttf_path}")

        # Collect lines in order
        line_nos = sorted(pages[page_no].keys())
        lines = [pages[page_no][ln] for ln in line_nos]

        # Fit font size so widest line fits between margins
        avail_w = max(10.0, args.page_w - args.margin_l - args.margin_r)
        fs = compute_font_size_fit_widest_line(
            ttf_path=ttf_path,
            lines=lines,
            avail_width_pt=avail_w,
            font_min=args.font_min,
            font_max=args.font_max,
            hb_features=hb_features
        )

        # Vertical layout
        line_h = fs * args.leading
        avail_h = max(10.0, args.page_h - args.margin_t - args.margin_b)
        if len(lines) > 0 and line_h * len(lines) > avail_h:
            line_h = avail_h / len(lines)

        x_right = args.page_w - args.margin_r
        y = args.page_h - args.margin_t - fs

        for words in lines:
            render_line_hb_pagefont(
                ttf_path=ttf_path,
                c=c,
                words_qpc=words,
                x_right=x_right,
                y=y,
                font_size_pt=fs,
                hb_features=hb_features
            )
            y -= line_h

        c.showPage()

    c.save()
    print(f"Wrote PDF: {args.out}")


if __name__ == "__main__":
    main()
