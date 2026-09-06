import json
import os
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any, List

from structures import WordRec

# assumes:
# - WordRec dataclass exists with fields: page:int, line:int, qpc:str, text:str
# - normalize_qpc_char() exists

SQL = """
select l.page,l.line,w.qpc_v1,w.dk_v1
from words w
INNER JOIN qpc_v1_layout l
  ON l.type = 'ayah'
 AND l.range_start <= w.word_number_all
 AND l.range_end   >= w.word_number_all
ORDER BY l.page, l.line, w.word_number_all;
"""

def normalize_qpc_char(v: Any) -> str:
    """
    qpc_v1 is a Unicode string representing the glyph selector
    (typically a PUA character).
    """
    if v is None:
        raise ValueError("qpc_v1 is NULL")
    s = str(v)
    if len(s) == 0:
        raise ValueError("qpc_v1 is empty")
    return s

def load_layout_from_cache(cache_path: Path) -> List["WordRec"]:
    out: List["WordRec"] = []
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            out.append(WordRec(**d))
    return out

def save_layout_to_cache(cache_path: Path, rows: List["WordRec"], sqlite_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # write data
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    # write meta (to detect staleness)
    meta = {
        "sqlite_path": str(sqlite_path),
        "sqlite_mtime": sqlite_path.stat().st_mtime,
        "sqlite_size": sqlite_path.stat().st_size,
        "count": len(rows),
    }
    meta_path = cache_path.with_suffix(cache_path.suffix + ".meta.json")
    meta_tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    meta_tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    tmp.replace(cache_path)
    meta_tmp.replace(meta_path)

def cache_is_valid(cache_path: Path, sqlite_path: Path) -> bool:
    if not cache_path.exists():
        return False
    meta_path = cache_path.with_suffix(cache_path.suffix + ".meta.json")
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        st = sqlite_path.stat()
        return (meta.get("sqlite_mtime") == st.st_mtime and
                meta.get("sqlite_size") == st.st_size)
    except Exception:
        return False

def load_layout_from_sqlite(sqlite_path: Path) -> List["WordRec"]:
    con = sqlite3.connect(str(sqlite_path))
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute(SQL)
    rows = cur.fetchall()
    con.close()

    out: List["WordRec"] = []
    for r in rows:
        out.append(WordRec(
            page=int(r["page"]),
            line=int(r["line"]),
            qpc=normalize_qpc_char(r["qpc_v1"]),
            text=str(r["dk_v1"]) if r["dk_v1"] is not None else ""
        ))
    return out

def load_layout(sqlite_path: Path, cache_path: Path) -> List["WordRec"]:
    """
    Uses cache if present and matches sqlite mtime/size; otherwise runs SQL and refreshes cache.
    """
    if cache_is_valid(cache_path, sqlite_path):
        return load_layout_from_cache(cache_path)

    rows = load_layout_from_sqlite(sqlite_path)
    save_layout_to_cache(cache_path, rows, sqlite_path)
    return rows
