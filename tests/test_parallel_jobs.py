#!/usr/bin/env python3
"""Smoke test for concurrent page processing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def fail(message: str) -> None:
    raise SystemExit(f"parallel jobs regression failed: {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--database", required=True, type=Path)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="mushaf-parallel-jobs-") as output:
        output_root = Path(output)
        command = [
            str(args.cli),
            "--input",
            str(args.input),
            "--out",
            str(output_root),
            "--pages",
            "3,18",
            "--jobs",
            "2",
            "--database",
            str(args.database),
            "--analysis-only",
            "--no-debug-reconstructed",
            "--blurK",
            "0",
            "--morphK",
            "0",
            "--recognizer-cache",
            str(output_root / "recognizer-cache"),
            "--recognizer-source",
            "parallel-test",
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode != 0:
            fail(
                "mushaf_cli failed with exit code "
                f"{completed.returncode}: {completed.stderr}"
            )
        progress_pages = sorted(
            int(line.split()[1])
            for line in completed.stdout.splitlines()
            if line.startswith("Page ")
        )
        if progress_pages != [3, 18]:
            fail(f"expected one completion message per page, got {completed.stdout}")

        report_path = output_root / "validation-report.json"
        if not report_path.is_file():
            fail(f"missing validation report {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        pages = report.get("pages", [])
        if [page.get("page") for page in pages] != [3, 18]:
            fail(f"unexpected page order in report: {pages}")
        if any(page.get("status") == "failed" for page in pages):
            fail(f"a parallel page failed: {pages}")
        for page in (3, 18):
            words_path = output_root / f"page{page:03d}" / "words.json"
            if not words_path.is_file():
                fail(f"missing words output {words_path}")

        word_mtimes = {
            page: (output_root / f"page{page:03d}" / "words.json")
            .stat()
            .st_mtime_ns
            for page in (3, 18)
        }
        skipped = subprocess.run(command, text=True, capture_output=True)
        if skipped.returncode != 0:
            fail(
                "default resume run failed with exit code "
                f"{skipped.returncode}: {skipped.stderr}"
            )
        skipped_pages = sorted(
            int(line.split()[1])
            for line in skipped.stdout.splitlines()
            if line.startswith("Page ")
        )
        if skipped_pages != [3, 18] or "skipped" not in skipped.stdout:
            fail(f"expected skipped progress messages, got {skipped.stdout}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("totals", {}).get("skippedPages") != 2:
            fail(f"expected two skipped pages in report, got {report}")
        for page, mtime in word_mtimes.items():
            words_path = output_root / f"page{page:03d}" / "words.json"
            if words_path.stat().st_mtime_ns != mtime:
                fail(f"default resume rewrote {words_path}")

        forced = subprocess.run(
            command + ["--force"], text=True, capture_output=True
        )
        if forced.returncode != 0:
            fail(
                "force run failed with exit code "
                f"{forced.returncode}: {forced.stderr}"
            )
        forced_pages = sorted(
            int(line.split()[1])
            for line in forced.stdout.splitlines()
            if line.startswith("Page ")
        )
        if forced_pages != [3, 18] or "skipped" in forced.stdout:
            fail(f"expected --force to reprocess pages, got {forced.stdout}")


if __name__ == "__main__":
    main()
