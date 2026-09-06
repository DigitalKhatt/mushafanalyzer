#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from fontTools.ttLib import TTFont

from read_quran_data import load_layout
from structures import WordRec


# ----------------------------
# Utilities: parsing + features
# ----------------------------

AR_LETTER = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")

def strip_marks(s: str) -> str:
    out = []
    for ch in s:
        if ("\u064B" <= ch <= "\u065F") or (ch == "\u0670") or ("\u06D6" <= ch <= "\u06ED"):
            continue
        out.append(ch)
    return "".join(out)

def letter_count(s: str) -> int:
    b = strip_marks(s)
    return sum(1 for ch in b if AR_LETTER.match(ch))

def mark_count(s: str) -> int:
    return sum(1 for ch in s if ("\u064B" <= ch <= "\u065F") or (ch == "\u0670") or ("\u06D6" <= ch <= "\u06ED"))

def ends_with_bucket(s: str) -> int:
    b = strip_marks(s)
    if not b:
        return 0
    last = b[-1]
    # coarse buckets (tune later)
    if last in "يى": return 1
    if last in "ن":  return 2
    if last in "م":  return 3
    if last in "هة": return 4
    if last in "رزو": return 5
    return 6

def uhex(u: int) -> str:
    return f"{u:X}".upper()

def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 1e-1) -> np.ndarray:
    n = X.shape[1]
    A = X.T @ X + lam * np.eye(n)
    b = X.T @ y
    return np.linalg.solve(A, b)

def robust_percentiles(x: List[float], ps=(5,10,50,90,95)) -> Dict[str, float]:
    arr = np.array(x, dtype=np.float64)
    return {f"p{p}": float(np.percentile(arr, p)) for p in ps}


# ----------------------------
# Font metric extraction
# ----------------------------

def load_best_cmap(ttf: TTFont) -> Dict[int, str]:
    cmap = ttf.getBestCmap()
    if not cmap:
        raise RuntimeError("No cmap found in font")
    return dict(cmap)  # unicode -> glyphName

def glyph_bbox_from_glyf(ttf: TTFont, gname: str) -> Optional[Tuple[int,int,int,int]]:
    if "glyf" not in ttf:
        # If you end up with CFF, you'll need a different bbox path.
        raise RuntimeError("Expected TTF with 'glyf' outlines.")
    glyf = ttf["glyf"]
    g = glyf[gname]
    if g.isComposite():
        g.recalcBounds(glyf)
    if not hasattr(g, "xMin"):
        return None
    try:
        return int(g.xMin), int(g.yMin), int(g.xMax), int(g.yMax)
    except Exception:
        return None

def extract_page_font_metrics(ttf_path: Path) -> Tuple[int, Dict[str, dict]]:
    ttf = TTFont(str(ttf_path))
    upem = int(ttf["head"].unitsPerEm)
    cmap = ttf.getBestCmap()  # u:int -> glyphName
    hmtx = ttf["hmtx"].metrics

    out: Dict[str, dict] = {}

    for u, gname in cmap.items():
        if gname not in hmtx:
            continue
        adv, lsb = hmtx[gname]
        bbox = glyph_bbox_from_glyf(ttf, gname)
        if bbox is None:
            xMin = xMax = 0
        else:
            xMin, _, xMax, _ = bbox

        ink = int(xMax - xMin)
        rsb = int(adv - (lsb + ink))

        out[chr(u)] = {
            "adv": int(adv),
            "lsb": int(lsb),
            "rsb": int(rsb),
            "inkWidth": int(ink),
            "xMin": int(xMin),
            "xMax": int(xMax),
            "glyphName": gname
        }

    ttf.close()
    return upem, out

# ----------------------------
# Analysis: gaps + widths + rules
# ----------------------------

def gap_features(left_text: str, right_text: str, rsbL: int, lsbR: int) -> np.ndarray:
    # Interpretable starter set. You can extend later with letter class detectors etc.
    L = left_text or ""
    R = right_text or ""
    return np.array([
        1.0,
        float(letter_count(L)),
        float(letter_count(R)),
        float(mark_count(L)),
        float(mark_count(R)),
        float(ends_with_bucket(L)),
        float(abs(rsbL)),
        float(abs(lsbR)),
    ], dtype=np.float64)

def cluster_widths(widths: List[int], step: int = 5, min_count: int = 3) -> List[dict]:
    """
    Turn raw widths into a few discrete levels (candidate alternates).
    Very simple clustering: round widths to 'step' and keep frequent bins.
    """
    if not widths:
        return []
    b = Counter(int(round(w / step) * step) for w in widths)
    # keep bins that appear at least min_count times OR are among top few
    items = sorted(b.items(), key=lambda kv: (-kv[1], kv[0]))
    kept = []
    for w, c in items:
        if c >= min_count:
            kept.append((w, c))
    if len(kept) < 2:
        # keep top 1-3 anyway
        kept = items[:min(3, len(items))]
    total = sum(c for _, c in kept) or 1
    out = [{"width": int(w), "count": int(c), "freq": float(c)/float(total)} for w, c in kept]
    out.sort(key=lambda d: d["width"])
    return out

def metrics_for_qpc_word(qpc: str, metrics: dict):
    """
    qpc: string of 1+ characters representing one word
    metrics: dict[char] -> {adv, lsb, rsb}

    Returns (advance, lsb, rsb).
    """
    if not qpc:
        return 0, 0, 0

    adv = 0
    first_lsb = None
    last_rsb = None

    for ch in qpc:
        m = metrics.get(ch)
        if m is None:
            # missing glyph: treat as zero-width, but keep going
            continue

        adv += int(m["adv"])

        if first_lsb is None:
            first_lsb = int(m["lsb"])
        last_rsb = int(m["rsb"])

    # Fallbacks if everything was missing
    if first_lsb is None:
        first_lsb = 0
    if last_rsb is None:
        last_rsb = 0

    return adv, first_lsb, last_rsb


def analyze(
    sqlite_path: Path,
    cache_path: Path,
    fonts_dir: Path,
    font_template: str,
    out_json: Path,
    width_cluster_step: int = 5,
    ridge_lambda: float = 1e-1
) -> None:
    layout = load_layout(sqlite_path,cache_path)

    # Group words into pages/lines
    pages: Dict[int, Dict[int, List[WordRec]]] = defaultdict(lambda: defaultdict(list))
    for w in layout:
        pages[w.page][w.line].append(w)

    # Aggregate outputs and training samples
    all_gaps: List[float] = []
    all_gap_X: List[np.ndarray] = []
    all_gap_y: List[float] = []  # delta from line median

    # Word width variability by text
    widths_by_text: Dict[str, List[int]] = defaultdict(list)

    # Per page measure estimation
    line_widths_by_page: Dict[int, List[int]] = defaultdict(list)

    # Store some per-line summaries for debugging/inspection
    debug_lines = []

    for page_no in sorted(pages.keys()):
        font_path = fonts_dir / (font_template.format(page=page_no))
        if not font_path.exists():
            raise FileNotFoundError(f"Missing page font for page {page_no}: {font_path}")

        upem, metrics = extract_page_font_metrics(font_path)

        for line_no in sorted(pages[page_no].keys()):
            words = pages[page_no][line_no]
            if not words:
                continue

            # Extract word metrics from font
            adv = []
            lsb = []
            rsb = []
            texts = []
            qpcs = []
            for wr in words:
                a, l, r = metrics_for_qpc_word(wr.qpc, metrics)
                adv.append(a); lsb.append(l); rsb.append(r)
                texts.append(wr.text)
                qpcs.append(wr.qpc)

                # width variability signal: teacher word widths by dk_v1 text
                if wr.text:
                    widths_by_text[wr.text].append(a)

            # Teacher line width (since no explicit spaces): sum of advances
            lineW = int(sum(adv))
            line_widths_by_page[page_no].append(lineW)

            # Implicit boundary gaps: rsb[i] + lsb[i+1]
            gaps = []
            for i in range(len(words)-1):
                g = int(rsb[i] + lsb[i+1])
                gaps.append(g)
                all_gaps.append(float(g))

            # Gap learning samples: predict (gap - line_median_gap)
            if gaps:
                g_med = float(np.median(np.array(gaps, dtype=np.float64)))
                for i, g in enumerate(gaps):
                    X = gap_features(texts[i], texts[i+1], rsb[i], lsb[i+1])
                    all_gap_X.append(X)
                    all_gap_y.append(float(g) - g_med)

            debug_lines.append({
                "page": page_no,
                "line": line_no,
                "upem": upem,
                "lineWidth": lineW,
                "numWords": len(words),
                "gapMedian": float(np.median(gaps)) if gaps else None,
            })

    # Estimate page measures (robust): use p90 of line widths in that page.
    page_measures = {}
    for p, ws in line_widths_by_page.items():
        if not ws:
            continue
        page_measures[p] = int(np.percentile(np.array(ws, dtype=np.float64), 90))

    # Learn global gap model
    if len(all_gap_X) < 2000:
        raise RuntimeError(f"Not enough gap samples: {len(all_gap_X)}. Check layout/TTFs.")
    X = np.vstack(all_gap_X)
    y = np.array(all_gap_y, dtype=np.float64)
    w = ridge_fit(X, y, lam=ridge_lambda)

    gap_stats = robust_percentiles(all_gaps, ps=(1,5,10,25,50,75,90,95,99))
    global_baseline_gap = gap_stats["p50"]
    suggested_min_gap = gap_stats["p10"]
    suggested_max_gap = gap_stats["p90"]

    # Build per-word “alternate width levels” by dk_v1
    word_models = {}
    for text, ws in widths_by_text.items():
        if len(ws) < 5:
            continue
        ws_sorted = sorted(ws)
        wmin = int(np.percentile(np.array(ws_sorted, dtype=np.float64), 5))
        wmed = int(np.percentile(np.array(ws_sorted, dtype=np.float64), 50))
        wmax = int(np.percentile(np.array(ws_sorted, dtype=np.float64), 95))

        levels = cluster_widths(ws, step=width_cluster_step, min_count=3)

        # Convert levels into deltas relative to the most frequent (or median) width
        baseW = int(levels[len(levels)//2]["width"]) if levels else wmed
        for lv in levels:
            lv["deltaFromBase"] = int(lv["width"] - baseW)

        word_models[text] = {
            "count": len(ws),
            "widthMin_p5": wmin,
            "widthMed_p50": wmed,
            "widthMax_p95": wmax,
            "levels": levels,
            "suggestedBaseWidth": baseW,
        }

    out = {
        "inputs": {
            "sqlite": str(sqlite_path),
            "fontsDir": str(fonts_dir),
            "fontTemplate": font_template,
        },
        "gapRuleAnalysis": {
            "model": "gap_delta_from_line_median_ridge",
            "features": [
                "bias",
                "left_letter_count",
                "right_letter_count",
                "left_mark_count",
                "right_mark_count",
                "left_end_bucket",
                "abs_rsb_left",
                "abs_lsb_right",
            ],
            "weights": w.tolist(),
            "globalGapStats": gap_stats,
            "globalBaselineGap_p50": global_baseline_gap,
            "suggestedMinGap_p10": suggested_min_gap,
            "suggestedMaxGap_p90": suggested_max_gap,
            "notes": [
                "No explicit spaces exist; inter-word gaps are reconstructed from glyph metrics:",
                "gap(i,i+1) = rsb(word_i) + lsb(word_{i+1}).",
                "Model predicts a delta relative to the median gap of the line (texture preference).",
            ]
        },
        "pageMeasures": page_measures,
        "wordWidthVariability": {
            "clusterStep": width_cluster_step,
            "words": word_models,
            "notes": [
                "For each dk_v1 word text, we collect teacher advance widths across occurrences.",
                "Width variability is a strong proxy for internal stretching/alternate selection in the teacher mushaf.",
                "Levels provide suggested discrete widths (alternates) to expose in the synthesis phase."
            ]
        },
        "debug": {
            "numGapSamples": len(all_gaps),
            "numGapTrainRows": len(all_gap_X),
            "numWordsWithModels": len(word_models),
            "debugLinesFirst200": debug_lines[:200],
        }
    }

    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out_json}")
    print(f"Gaps: baseline p50={global_baseline_gap:.2f}, suggested [p10={suggested_min_gap:.2f}, p90={suggested_max_gap:.2f}]")
    print(f"Word models: {len(word_models)} words with >=5 occurrences")


def main():
    ap = argparse.ArgumentParser(description="Mushaf justification/stretch analysis from 604 page TTFs + SQLite layout.")
    ap.add_argument("--sqlite", required=True, type=Path, help="Path to quran-data.sqlite")
    ap.add_argument("--cache", required=True, type=Path, help="Path to pages.json")
    ap.add_argument("--fonts_dir", required=True, type=Path, help="Directory containing page_###.ttf fonts")
    ap.add_argument("--font_template", default="page_{page:03d}.ttf",
                    help="Python format template for page font file name. Uses {page}. Default: page_{page:03d}.ttf")
    ap.add_argument("--out", required=True, type=Path, help="Output JSON path")
    ap.add_argument("--width_cluster_step", type=int, default=5, help="Quantization step for discrete width levels (font units)")
    ap.add_argument("--ridge_lambda", type=float, default=1e-1, help="Ridge regression regularization (gap model)")

    args = ap.parse_args()
    analyze(
        sqlite_path=args.sqlite,
        cache_path=args.cache,
        fonts_dir=args.fonts_dir,
        font_template=args.font_template,
        out_json=args.out,
        width_cluster_step=args.width_cluster_step,
        ridge_lambda=args.ridge_lambda
    )

if __name__ == "__main__":
    main()
