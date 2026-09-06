#!/usr/bin/env python3
"""Browse, filter, sort, and review the letter/mark instance SQLite index."""

import argparse
import json
import mimetypes
import pathlib
import sqlite3
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


HTML = r'''<!doctype html><html><head><meta charset="utf-8"><title>Mushaf instances</title>
<style>body{font:15px system-ui;background:#111820;color:#edf2f7;margin:0}header{position:sticky;top:0;background:#1b2733;padding:12px;display:flex;gap:8px;flex-wrap:wrap}input,select,button{background:#263746;color:white;border:1px solid #526679;border-radius:5px;padding:7px}main{padding:18px;display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:15px}.card{background:#1b2733;border:1px solid #3d5061;border-radius:9px;padding:12px}.card img{width:100%;height:190px;object-fit:contain;background:#fff}.arabic{font-size:26px;direction:rtl}.meta{color:#aebdca;font:12px ui-monospace,monospace}.review{display:flex;gap:5px;margin-top:8px}.ok{background:#176c48}.bad{background:#8a3434}</style></head>
<body><header><select id="kind"><option value="letter">Letters</option><option value="mark">Marks</option></select><input id="query" placeholder="letter ا or mark class"><select id="form"><option value="">all forms</option><option>initial</option><option>medial</option><option>final</option><option>isolated</option></select><input id="min" type="number" placeholder="min width"><input id="max" type="number" placeholder="max width"><select id="sort"><option value="width_asc">width ↑</option><option value="width_desc">width ↓</option><option value="confidence">confidence ↓</option></select><select id="status"><option value="">all reviews</option><option>unreviewed</option><option>accepted</option><option>rejected</option><option>uncertain</option></select><button onclick="load()">Apply</button><span id="summary"></span></header><main id="cards"></main>
<script>
const $=x=>document.getElementById(x); const esc=s=>(s??'').toString().replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function review(kind,id,status){let error='';if(status==='rejected')error=prompt('Error type (boundary, owner, class, noise, other)','boundary')||'';await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({kind,id,status,error_type:error})});load()}
async function load(){let p=new URLSearchParams({kind:$('kind').value,q:$('query').value,form:$('form').value,min:$('min').value,max:$('max').value,sort:$('sort').value,status:$('status').value});let d=await(await fetch('/api/instances?'+p)).json();$('summary').textContent=d.total+' matches (showing '+d.records.length+')';$('cards').innerHTML=d.records.map(r=>`<article class="card"><img src="/image?path=${encodeURIComponent(r.image)}"><div class="arabic">${esc(r.character||r.visual_class||r.suggested_class||'?')} · ${esc(r.expected_text)}</div><div class="meta">${esc(r.id)} · p${r.page} ${esc(r.word_id)}<br>${r.width}×${r.height}px · confidence ${(100*(r.confidence||0)).toFixed(1)}% ${esc(r.joining_form||'')}<br>review: ${esc(r.review_status)}</div><div class="review"><button class="ok" onclick="review('${r.kind}','${r.id}','accepted')">Accept</button><button class="bad" onclick="review('${r.kind}','${r.id}','rejected')">Reject</button><button onclick="review('${r.kind}','${r.id}','uncertain')">Uncertain</button></div></article>`).join('')}
$('kind').onchange=()=>{$('form').disabled=$('kind').value==='mark';load()};load();
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    database = None
    allowed_root = None

    def json(self, value, status=200):
        data = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(data)); self.end_headers(); self.wfile.write(data)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path); query = urllib.parse.parse_qs(parsed.query)
        if parsed.path == "/":
            data = HTML.encode(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", len(data)); self.end_headers(); self.wfile.write(data); return
        if parsed.path == "/image":
            path = pathlib.Path(query.get("path", [""])[0]).resolve()
            if not path.is_file() or (self.allowed_root not in path.parents and path.parent != self.allowed_root): self.send_error(404); return
            data = path.read_bytes(); self.send_response(200); self.send_header("Content-Type", mimetypes.guess_type(path)[0] or "image/png"); self.send_header("Content-Length", len(data)); self.end_headers(); self.wfile.write(data); return
        if parsed.path != "/api/instances": self.send_error(404); return
        kind = query.get("kind", ["letter"])[0]; table = "letter_instances" if kind == "letter" else "mark_instances"
        clauses, values = [], []
        q = query.get("q", [""])[0]
        if q:
            clauses.append("character = ?" if kind == "letter" else "COALESCE(visual_class,suggested_class,'') LIKE ?"); values.append(q if kind == "letter" else f"%{q}%")
        if kind == "letter" and query.get("form", [""])[0]: clauses.append("joining_form = ?"); values.append(query["form"][0])
        for key, operator in (("min", ">="), ("max", "<=")):
            if query.get(key, [""])[0]: clauses.append(f"width {operator} ?"); values.append(int(query[key][0]))
        if query.get("status", [""])[0]: clauses.append("review_status = ?"); values.append(query["status"][0])
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        order = {"width_desc":"width DESC", "confidence":"confidence DESC"}.get(query.get("sort", [""])[0], "width ASC")
        db = sqlite3.connect(self.database); db.row_factory = sqlite3.Row
        total = db.execute(f"SELECT count(*) FROM {table}{where}", values).fetchone()[0]
        rows = [dict(row) for row in db.execute(f"SELECT * FROM {table}{where} ORDER BY {order} LIMIT 200", values)]
        db.close()
        for row in rows: row["kind"] = kind; row["image"] = row.get("overlay_path") if kind == "letter" else row.get("context_path")
        self.json({"total": total, "records": rows})

    def do_POST(self):
        if self.path != "/api/review": self.send_error(404); return
        payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        table = "letter_instances" if payload.get("kind") == "letter" else "mark_instances"
        db = sqlite3.connect(self.database); db.execute(f"UPDATE {table} SET review_status=?, error_type=? WHERE id=?", (payload.get("status"), payload.get("error_type", ""), payload.get("id"))); db.commit(); db.close(); self.json({"ok": True})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--database", type=pathlib.Path, required=True); parser.add_argument("--port", type=int, default=8877); args = parser.parse_args()
    Handler.database = args.database.resolve(); Handler.allowed_root = args.database.resolve().parent.parent.resolve()
    print(f"Review interface: http://127.0.0.1:{args.port}/")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__": main()
