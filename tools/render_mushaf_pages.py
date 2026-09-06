#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont as RLTTFont


SQL = """
select l.page,l.line,w.qpc_v1,w.dk_v1
from words w
INNER JOIN qpc_v1_layout l
  ON l.type = 'ayah'
 AND l.range_start <= w.word_number_all
 AND l.range_end   >= w.word_number_all
ORDER BY l.page, l.line, w.word_number_all;
"""


def load_layout(sqlite_path: Path) -> Dict[int, Dict[int, List[str]]]:
    """
    Returns: pages[page][line] = [qpc_char0, qpc_char1, ...] in logical reading order.
    qpc_v1 is expected to be a Unicode string (typically one PUA character per word glyph).
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
        if qpc == "":
            continue
        # Usually qpc_v1 is a single character; if it isn't, we still keep the string as-is.
        pages[page][line].append(qpc)

    return pages


def register_font(font_name: str, ttf_path: Path) -> None:
    # Register once per unique name
    if font_name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(RLTTFont(font_name, str(ttf_path)))


def compute_page_fontsize(
    c: canvas.Canvas,
    font_name: str,
    lines: List[List[str]],
    page_w_pt: float,
    margin_l_pt: float,
    margin_r_pt: float,
    font_size_min: float = 6.0,
    font_size_max: float = 40.0,
) -> float:
    """
    Choose a font size so the widest line fits into available width.
    Uses reportlab stringWidth which relies on font metrics.
    """
    avail = max(10.0, page_w_pt - margin_l_pt - margin_r_pt)

    # Widest line width measured at fontSize=1.0
    max_w_at_1 = 0.0
    for words in lines:
        # RTL: we will render reversed; width is invariant to direction
        s = "".join(reversed(words))
        w = pdfmetrics.stringWidth(s, font_name, 1.0)
        if w > max_w_at_1:
            max_w_at_1 = w

    if max_w_at_1 <= 0:
        return font_size_min

    fs = avail / max_w_at_1  # because width scales linearly with font size
    return max(font_size_min, min(font_size_max, fs))


def main():
    ap = argparse.ArgumentParser(description="Render Mushaf pages from per-page TTF fonts into a PDF.")
    ap.add_argument("--sqlite", required=True, type=Path, help="Path to quran-data.sqlite")
    ap.add_argument("--fonts_dir", required=True, type=Path, help="Directory containing page TTFs")
    ap.add_argument("--font_template", default="page_{page:03d}.ttf",
                    help="Filename template, uses {page}. Default: page_{page:03d}.ttf")
    ap.add_argument("--out", required=True, type=Path, help="Output PDF path")

    # Page geometry (points). Default A4.
    ap.add_argument("--page_w", type=float, default=A4[0], help="Page width in points (default A4)")
    ap.add_argument("--page_h", type=float, default=A4[1], help="Page height in points (default A4)")

    # Margins
    ap.add_argument("--margin_l", type=float, default=36.0, help="Left margin in points (default 36)")
    ap.add_argument("--margin_r", type=float, default=36.0, help="Right margin in points (default 36)")
    ap.add_argument("--margin_t", type=float, default=36.0, help="Top margin in points (default 36)")
    ap.add_argument("--margin_b", type=float, default=36.0, help="Bottom margin in points (default 36)")

    # Line spacing
    ap.add_argument("--leading", type=float, default=1.25, help="Line spacing factor (default 1.25)")

    # Font size clamp
    ap.add_argument("--font_min", type=float, default=7.0, help="Min font size in points")
    ap.add_argument("--font_max", type=float, default=28.0, help="Max font size in points")

    args = ap.parse_args()

    pages = load_layout(args.sqlite)

    c = canvas.Canvas(str(args.out), pagesize=(args.page_w, args.page_h))

    for page_no in sorted(pages.keys()):
        ttf_path = args.fonts_dir / args.font_template.format(page=page_no)
        if not ttf_path.exists():
            raise FileNotFoundError(f"Missing page font: {ttf_path}")

        font_name = f"PAGE_{page_no:03d}"
        register_font(font_name, ttf_path)

        # Collect line word lists in line order
        line_nos = sorted(pages[page_no].keys())
        lines = [pages[page_no][ln] for ln in line_nos]

        # Choose per-page font size so widest line fits between margins
        fs = compute_page_fontsize(
            c, font_name, lines,
            page_w_pt=args.page_w,
            margin_l_pt=args.margin_l,
            margin_r_pt=args.margin_r,
            font_size_min=args.font_min,
            font_size_max=args.font_max,
        )

        c.setFont(font_name, fs)

        # Compute vertical placement
        y_top = args.page_h - args.margin_t
        line_h = fs * args.leading

        # If too many lines, scale leading down slightly (basic safeguard)
        total_h = line_h * len(lines)
        avail_h = max(10.0, args.page_h - args.margin_t - args.margin_b)
        if total_h > avail_h and len(lines) > 0:
            line_h = avail_h / len(lines)

        y = y_top

        for words in lines:
            # Manual RTL: reverse word order, draw as LTR string.

            #reshaped_text = arabic_reshaper.reshape(text)
            s = "".join(words)
            s = s[::-1]

            # Right-align the line inside margins
            w_pt = pdfmetrics.stringWidth(s, font_name, fs)
            x = args.page_w - args.margin_r - w_pt

            c.drawString(x, y, s)
            y -= line_h

        c.showPage()

    c.save()
    print(f"Wrote PDF: {args.out}")


if __name__ == "__main__":
    main()
