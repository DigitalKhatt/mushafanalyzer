#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional

import uharfbuzz as hb

from read_quran_data import load_layout

class HBFont:
    def __init__(self, font_path: Path):
        self.font_bytes = font_path.read_bytes()
        self.face = hb.Face(self.font_bytes)
        self.font = hb.Font(self.face)
        self.upem = self.face.upem
        self.font.scale = (self.upem, self.upem)

    def line_width_units(
        self,
        text: str,
        direction: str = "RTL",
        script: str = "arab",
        language: str = "ar",
        features: Optional[Dict[str, Any]] = None,
    ) -> int:
        buf = hb.Buffer()
        buf.add_str(text)
        buf.direction = direction
        buf.script = script
        buf.language = language
        buf.guess_segment_properties()
        hb.shape(self.font, buf, features or {})
        pos = buf.glyph_positions
        return int(sum(p.x_advance for p in pos))


def parse_hb_features(s: str) -> Dict[str, Any]:
    """
    Parse: "liga=1,calt=1,ss01=1" -> {"liga": True, "calt": True, "ss01": True}
    If value omitted (e.g. "liga,calt"), it's True.
    """
    s = (s or "").strip()
    if not s:
        return {}
    out: Dict[str, Any] = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = bool(int(v.strip()))
        else:
            out[part] = True
    return out


def main():
    ap = argparse.ArgumentParser(description="Export HarfBuzz-computed line widths (font units) for all pages to JSON.")
    ap.add_argument("--sqlite", required=True, type=Path, help="Path to quran-data.sqlite")
    ap.add_argument("--cache", required=True, type=Path, help="Path to pages.json")
    ap.add_argument("--fonts_dir", required=True, type=Path, help="Directory containing page TTF fonts")
    ap.add_argument("--font_template", default="page_{page:03d}.ttf",
                    help="Font filename template using {page}. Default: page_{page:03d}.ttf")
    ap.add_argument("--out", required=True, type=Path, help="Output JSON path")
    ap.add_argument("--hb_features", default="", help="HB features like 'liga=1,calt=1,ss01=1' (optional)")
    ap.add_argument("--direction", default="RTL", choices=["RTL", "LTR"], help="HB direction")
    ap.add_argument("--script", default="arab", help="HB script tag (default arab)")
    ap.add_argument("--language", default="ar", help="HB language tag (default ar)")
    args = ap.parse_args()

    layout = load_layout(args.sqlite,args.cache)
    feats = parse_hb_features(args.hb_features)

     # Group words into pages/lines
    pages: Dict[int, Dict[int, List[str]]] = defaultdict(lambda: defaultdict(list))
    for w in layout:
        pages[w.page][w.line].append(w.qpc)

    # Output structure:
    # {
    #   "meta": {...},
    #   "pages": {
    #      "1": { "1": 12345, "2": 12400, ... },
    #      ...
    #   }
    # }
    out_pages: Dict[str, Dict[str, int]] = {}

    for page_no in sorted(pages.keys()):
        font_path = args.fonts_dir / args.font_template.format(page=page_no)
        if not font_path.exists():
            raise FileNotFoundError(f"Missing font for page {page_no}: {font_path}")

        hbfont = HBFont(font_path)
        page_lines: Dict[str, int] = {}

        for line_no in sorted(pages[page_no].keys()):
            tokens = pages[page_no][line_no]  # list of qpc_v1 strings
            line_text = "".join(tokens)        # no spaces (word advances include spacing)
            w_units = hbfont.line_width_units(
                line_text,
                direction=args.direction,
                script=args.script,
                language=args.language,
                features=feats
            )
            page_lines[str(line_no)] = w_units

        out_pages[str(page_no)] = page_lines

    out_obj = {
        "meta": {
            "sqlite": str(args.sqlite),
            "fonts_dir": str(args.fonts_dir),
            "font_template": args.font_template,
            "hb_features": feats,
            "direction": args.direction,
            "script": args.script,
            "language": args.language,
            "units": "font_units (sum of HarfBuzz x_advance at UPEM scale)",
        },
        "pages": out_pages
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
