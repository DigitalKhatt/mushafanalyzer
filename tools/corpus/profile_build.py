#!/usr/bin/env python3
"""Profile one page of the corpus build, without a database or a pool.

`build_corpus.py --pages N` measures a page end to end, which mixes the work
worth optimizing with process startup, IPC and SQLite. This runs `build_page`
plus `normalize_page` in this process so the profile is the segmentation and
tracing cost and nothing else.

    uv run tools/corpus/profile_build.py \
        --analysis output/oldmedina2-analysis --page 3 --words 20

`--no-trace` splits the two halves apart: the difference between a run with it
and one without is exactly what outline tracing costs.
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
from pathlib import Path
import sys
import time

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
  sys.path.insert(0, str(_HERE))

import build_corpus                                       # noqa: E402
import segment_paw_letters as seg                         # noqa: E402


def main() -> int:
  parser = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--analysis", required=True)
  parser.add_argument("--page", type=int, default=3)
  parser.add_argument("--words", type=int, default=0,
                      help="stop after this many words (0: the whole page)")
  parser.add_argument("--no-trace", action="store_true")
  parser.add_argument("--top", type=int, default=40)
  parser.add_argument("--sort", default="cumulative",
                      choices=("cumulative", "tottime"))
  parser.add_argument("--stats", type=Path,
                      help="also dump raw pstats here, for snakeviz or a diff")
  parser.add_argument("--skeleton-templates", default=seg.DEFAULT_TEMPLATE_PATH)
  arguments = parser.parse_args()

  options = {
      "min_confidence": seg.MIN_CUT_CONFIDENCE,
      "skeleton_shapes": False,
      "logical_ligatures": False,
      # What both VS Code tasks and the shipped defaults use, so the profile is
      # of the code path the real build takes.
      "body_join_alignment": True,
      "cut_policy": "next-start",
      "templates": arguments.skeleton_templates,
      "trace": not arguments.no_trace,
      "limit_words": arguments.words,
      "ayah_words": False,
  }

  analysis = str(Path(arguments.analysis).resolve())
  profiler = cProfile.Profile()
  started = time.perf_counter()
  profiler.enable()
  build = build_corpus.build_page(analysis, arguments.page, options)
  build_corpus.normalize_page(build)
  profiler.disable()
  elapsed = time.perf_counter() - started

  if build.missing:
    print(f"page {arguments.page} is not analyzed under {analysis}",
          file=sys.stderr)
    return 1

  print(f"page {arguments.page}: {len(build.words)} words, "
        f"{len(build.paws)} paws, {len(build.letters)} letters, "
        f"{len(build.marks)} marks, {len(build.errors)} errors "
        f"in {elapsed:.2f}s"
        f"{' (tracing off)' if arguments.no_trace else ''}")
  for message in build.errors[:5]:
    print(f"  ! {message}", file=sys.stderr)

  if arguments.stats:
    arguments.stats.parent.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(str(arguments.stats))
    print(f"stats -> {arguments.stats}")

  pstats.Stats(profiler).sort_stats(arguments.sort).print_stats(arguments.top)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
