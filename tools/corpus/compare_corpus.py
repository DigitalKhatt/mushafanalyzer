#!/usr/bin/env python3
"""Exact row-by-row comparison of two corpus indexes.

The optimization work on the builder is only allowed to make it faster, never
to make it produce something different. "Different" has to mean *bit* different
here, not "close": ``letter_shape.view_x`` and friends are stored as
full-precision REALs, so a change in float accumulation order that moves a value
by one ulp is a real regression in the thing being protected, and ``==`` on
floats would happily hide ``-0.0`` against ``0.0``. Every REAL is therefore
compared through its IEEE-754 bytes.

    uv run tools/corpus/compare_corpus.py ref.sqlite new.sqlite

Exit status is 0 only when every compared table matches exactly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3
import struct
import sys
from typing import Dict, List, Optional, Sequence, Tuple

# Ordered so a difference is reported at the coarsest level that shows it: a
# missing page explains its missing letters, and saying so once beats saying it
# five hundred times.
TABLES: Sequence[Tuple[str, Tuple[str, ...]]] = (
    ("pages", ("page",)),
    ("words", ("word_id",)),
    ("paws", ("paw_id",)),
    ("letters", ("letter_id",)),
    ("letter_shape", ("letter_id",)),
    ("marks", ("mark_id",)),
)

# Per-invocation by construction, so they differ between any two builds and say
# nothing about whether the builder still computes the same thing.
IGNORED_COLUMNS = frozenset({"run_id"})

# Query-planner statistics, not corpus rows.
IGNORED_TABLES = frozenset({"runs", "sqlite_stat1", "sqlite_stat4"})


def canonical(value):
  """A value's identity for exact comparison.

  Floats go through their IEEE-754 bytes so that -0.0 != 0.0 and NaN == NaN:
  either of those slipping through would mean the differ reports "identical"
  about rows that are not.
  """
  if isinstance(value, float):
    return ("f", struct.pack("<d", value))
  if isinstance(value, int):
    return ("i", value)
  if isinstance(value, bytes):
    return ("b", value)
  if value is None:
    return ("n", None)
  return ("s", str(value))


def columns_of(connection: sqlite3.Connection, table: str) -> List[str]:
  return [row[1] for row in connection.execute(f"PRAGMA table_info({table})")]


def describe(value) -> str:
  if isinstance(value, bytes):
    return f"<{len(value)} bytes>"
  if isinstance(value, str) and len(value) > 60:
    return f"{value[:57]!r}..."
  return repr(value)


def compare_table(ref: sqlite3.Connection, new: sqlite3.Connection,
                  table: str, keys: Sequence[str], pages: Optional[Sequence[int]],
                  limit: int) -> Tuple[int, List[str]]:
  """Walk both tables in key order once. Returns (difference count, report)."""
  ref_columns = columns_of(ref, table)
  new_columns = columns_of(new, table)
  if ref_columns != new_columns:
    only_ref = [c for c in ref_columns if c not in new_columns]
    only_new = [c for c in new_columns if c not in ref_columns]
    return 1, [f"{table}: schema differs (ref only: {only_ref}, new only: {only_new})"]

  compared = [c for c in ref_columns if c not in IGNORED_COLUMNS]
  quoted = ",".join(f'"{c}"' for c in compared)
  order = ",".join(f'"{k}"' for k in keys)
  where = ""
  parameters: Tuple = ()
  if pages is not None:
    placeholders = ",".join("?" * len(pages))
    if "page" in ref_columns:
      where = f" WHERE page IN ({placeholders})"
      parameters = tuple(pages)
    elif "letter_id" in ref_columns:
      # letter_shape carries no page of its own; without this it would compare
      # every page in one index against a subset in the other and call the
      # rows that were never built a difference.
      where = (" WHERE letter_id IN (SELECT letter_id FROM letters "
               f"WHERE page IN ({placeholders}))")
      parameters = tuple(pages)
  statement = f"SELECT {quoted} FROM {table}{where} ORDER BY {order}"

  key_index = [compared.index(k) for k in keys]
  differences = 0
  report: List[str] = []

  def key_of(row):
    return tuple(row[i] for i in key_index)

  left = ref.execute(statement, parameters)
  right = new.execute(statement, parameters)
  row_a = left.fetchone()
  row_b = right.fetchone()
  while row_a is not None or row_b is not None:
    if row_b is None or (row_a is not None and key_of(row_a) < key_of(row_b)):
      differences += 1
      if len(report) < limit:
        report.append(f"{table}: {key_of(row_a)} present in ref, missing from new")
      row_a = left.fetchone()
      continue
    if row_a is None or key_of(row_b) < key_of(row_a):
      differences += 1
      if len(report) < limit:
        report.append(f"{table}: {key_of(row_b)} present in new, missing from ref")
      row_b = right.fetchone()
      continue

    for index, column in enumerate(compared):
      if canonical(row_a[index]) != canonical(row_b[index]):
        differences += 1
        if len(report) < limit:
          report.append(
              f"{table}: {key_of(row_a)} column {column}: "
              f"ref {describe(row_a[index])} != new {describe(row_b[index])}")
    row_a = left.fetchone()
    row_b = right.fetchone()
  return differences, report


def open_readonly(path: Path) -> sqlite3.Connection:
  if not path.is_file():
    raise SystemExit(f"no such database: {path}")
  return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("reference", type=Path)
  parser.add_argument("candidate", type=Path)
  parser.add_argument("--pages", default="",
                      help="restrict to these pages: 3, 3-20, 3,7,9 "
                           "(default: every page both indexes hold)")
  parser.add_argument("--limit", type=int, default=25,
                      help="how many differences to print before counting only")
  arguments = parser.parse_args()

  pages: Optional[List[int]] = None
  if arguments.pages:
    pages = []
    for part in arguments.pages.split(","):
      part = part.strip()
      if not part:
        continue
      if "-" in part:
        low, high = part.split("-", 1)
        pages.extend(range(int(low), int(high) + 1))
      else:
        pages.append(int(part))
    pages = sorted(set(pages))

  ref = open_readonly(arguments.reference)
  new = open_readonly(arguments.candidate)

  total = 0
  printed: List[str] = []
  counts: Dict[str, int] = {}
  for table, keys in TABLES:
    if table in IGNORED_TABLES:
      continue
    differences, report = compare_table(ref, new, table, keys, pages,
                                        max(0, arguments.limit - len(printed)))
    counts[table] = differences
    total += differences
    printed.extend(report)

  for line in printed:
    print(line)
  if total > len(printed):
    print(f"... and {total - len(printed)} more")

  summary = ", ".join(f"{table} {count}" for table, count in counts.items())
  if total == 0:
    print(f"identical: {summary}")
    return 0
  print(f"\n{total} differences ({summary})", file=sys.stderr)
  return 1


if __name__ == "__main__":
  raise SystemExit(main())
