#!/usr/bin/env python3
"""Browse the letter/mark corpus: filter, sort by width, compare shapes.

The letter-level counterpart of ``oldmadinaanalyzer``, which returns whole
words. Ask for initial beh sorted by width and get every instance in the
corpus, drawn from its traced outline rather than a raster crop, so the
matches can also be laid on one baseline or superimposed on one another.

A kashida is not a kind of thing, so it is not one of the kinds. A connector
belongs to the letter it leaves and has no row of its own, which made
"kashidas" a letter search wearing a disguise -- and one that could not answer
"does a final qaf take a kashida before it", because a final letter leaves no
connector. It is a measurement instead: pick a kashida metric to sort and
filter by one, and tick k-left or k-right to paint it.

Run it::

    uv run tools/corpus/serve_corpus.py \
        --db output/corpus/corpus.sqlite \
        --analysis output/oldmedina2-analysis --port 8880

Then open http://127.0.0.1:8880/.
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import sqlite3
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
  sys.path.insert(0, str(_HERE))

import numpy as np                                         # noqa: E402

import letter_classes                                      # noqa: E402
from geometry import decode_mask_rle                       # noqa: E402
import urllib.parse

# Ceiling for one page of results, so that "max" means "everything there is"
# without being unbounded: a malformed limit cannot ask for an arbitrary
# number of rows. Larger than any table the corpus holds.
MAX_ROWS = 200000

# Each extra letter in a searched group is another self-join on `letters`.
# Long runs are not a use this browser has, and an unbounded one would let the
# query string decide how many joins the database has to plan.
MAX_GROUP_LETTERS = 6

SORTS = {
    "width_desc": ("adv_w_norm", "DESC"),
    "width_asc": ("adv_w_norm", "ASC"),
    "bbox_desc": ("bbox_w_norm", "DESC"),
    "bbox_asc": ("bbox_w_norm", "ASC"),
    "extent_desc": ("extent_norm", "DESC"),
    "extent_asc": ("extent_norm", "ASC"),
    "height_desc": ("height_norm", "DESC"),
    "height_asc": ("height_norm", "ASC"),
    "confidence_asc": ("confidence", "ASC"),
    "confidence_desc": ("confidence", "DESC"),
    "page_asc": ("page", "ASC"),
}
MARK_SORTS = {
    "width_desc": ("width_norm", "DESC"),
    "width_asc": ("width_norm", "ASC"),
    "height_desc": ("height_norm", "DESC"),
    "height_asc": ("height_norm", "ASC"),
    "area_desc": ("area_norm", "DESC"),
    "area_asc": ("area_norm", "ASC"),
    "page_asc": ("page", "ASC"),
}
WIDTH_COLUMN = {"advance": "adv_w_norm", "bbox": "bbox_w_norm",
                "extent": "extent_norm", "height": "height_norm",
                "kashida": "kashida_norm"}

# Metrics that are not a column of `letters` but a side of it. A letter stores
# the connector it *leaves*, which is the one on its left, so the connector on
# its right is the left one of the letter before it -- the same stroke, read
# from the other end. Nothing new has to be measured or rebuilt for this: the
# right-hand connector is already in the table, one row along.
SIDE_METRICS = {"kashidaLeft", "kashidaRight"}
# What the kashida kind is allowed to measure. Anything else it overrides,
# because a kashida search sorted by bbox would be measuring the letter.
KASHIDA_METRICS = {"kashida"} | SIDE_METRICS
# Sorts that name a metric rather than a column, resolved once the group's
# aliases are known.
METRIC_SORTS = {
    "kleft_desc": ("kashidaLeft", "DESC"),
    "kleft_asc": ("kashidaLeft", "ASC"),
    "kright_desc": ("kashidaRight", "DESC"),
    "kright_asc": ("kashidaRight", "ASC"),
}


def metric_sql(metric: str, aliases: List[str]) -> str:
  """SQL for one named metric, measured over a whole searched run.

  A run's left connector is the one leaving its last letter and its right one
  is the one arriving at its first, so neither is a sum -- unlike ``kashida``,
  which adds up every connector the run owns, the ones inside it included.
  """
  if metric == "kashidaLeft":
    return owned_kashida(aliases[-1])
  if metric == "kashidaRight":
    return owned_kashida("lp")
  if metric == "kashida":
    # Every connector the run touches, both ends included -- otherwise "all"
    # was the left one under another name, and read zero for a final form that
    # has only a right one. COALESCE because a PAW boundary has no stroke to
    # measure on that side and NULL there must not erase the other end.
    inside = " + ".join(f"COALESCE({owned_kashida(a)}, 0)" for a in aliases)
    return f"({inside} + COALESCE({owned_kashida('lp')}, 0))"
  return group_width_sql(WIDTH_COLUMN.get(metric, "adv_w_norm"), aliases)

# A letter carries a kashida when its span runs past its own body, i.e. when
# the next body starts further along the spine. Letters of an uncut group all
# copy the group's span, so the connector would otherwise be counted once per
# letter; it leaves the group, so it is charged to the last letter of it. The
# subquery only runs for the ~1.5% of rows that are in such a group.
def owns_connector(alias: str) -> str:
  return (f"{alias}.kashida_norm > 0 AND ({alias}.group_size = 1 "
          f"OR NOT EXISTS (SELECT 1 FROM letters g "
          f"WHERE g.paw_id = {alias}.paw_id "
          f"AND g.kashida_px = {alias}.kashida_px "
          f"AND g.letter_index > {alias}.letter_index))")


def owned_kashida(alias: str) -> str:
  """`alias`'s connector, or NULL when the connector is not its to report.

  Every member of an uncut group copies the group's span, so reporting it on
  each of them counts one stroke several times -- and offers it on a member
  that holds none of its ink. In p088:l13_w07:paw0 the beh-hah ligature's
  13.01-wide connector was traced onto its second member, which owns nothing
  else, while the first member still advertised the same 13.01 over a body
  7.74 wide. The stroke leaves the group, so it belongs to the member the
  group ends with -- the one the tracer gave the pixels to.
  """
  return f"CASE WHEN {owns_connector(alias)} THEN {alias}.kashida_norm END"


def group_start_sql(alias: str = "l", shape: str = "ls") -> str:
  """Where `alias`'s uncut group begins, as a PAW-relative letter index.

  Members of an uncut group share one entry point -- having no internal cut is
  exactly what that means -- so the group is the letters of the PAW whose
  traced frame starts where this one's does, and the card can name its
  siblings from the index and the size. Asked for only when the letter is in a
  group at all: 11,108 letters of 360k, and nothing to pay on the rest.
  """
  return (f"CASE WHEN {alias}.group_size > 1 THEN "
          f"(SELECT MIN(g.letter_index) FROM letters g "
          f"JOIN letter_shape gs ON gs.letter_id = g.letter_id "
          f"WHERE g.paw_id = {alias}.paw_id "
          f"AND gs.entry_x = {shape}.entry_x) END")


# The `letters` row of an uncut ligature member carries no advance: the
# segmenter refused to cut the shape, so no boundary inside it is measurable
# and build_corpus writes NULL rather than inventing one. That is right about
# the letter and wrong about the search -- 18,732 rows then sort last, fall
# out of every min/max filter, and contribute nothing to the histogram, so
# "class yeh, final, by width" quietly measured half its matches. The group's
# own advance is measurable, is the same for every member, and is already
# stored as the traced frame's entry and exit.
SHAPE_JOIN = " LEFT JOIN letter_shape ls ON ls.letter_id = l.letter_id"


def advance_sql(alias: str, shape: str = "ls") -> str:
  """`alias`'s advance, or its uncut group's when it has none of its own.

  Identical to `adv_w_norm` wherever that is set -- both are entry-to-exit
  over the page nominal stroke -- so an ordinary letter is unaffected and the
  two can be compared in one histogram.
  """
  return (f"COALESCE({alias}.adv_w_norm, CASE WHEN {alias}.group_size > 1 "
          f"THEN ABS({shape}.entry_x - {shape}.exit_x) / {shape}.norm_scale "
          f"END)")


def kashida_clause(metric: str) -> str:
  """Which side the kashida kind requires, following the metric.

  Only a letter that joins forwards leaves a connector, so a final form never
  has one of its own: gating the kind on the letter's own kashida made "qaf
  final" -- and every other final and isolated form -- return nothing at all,
  though 46,653 final letters do have a connector arriving at them. The side
  being measured is the side that has to exist.
  """
  left, right = owns_connector("l"), owns_connector("lp")
  if metric == "kashidaLeft":
    return left
  if metric == "kashidaRight":
    return right
  return f"(({left}) OR ({right}))"


HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mushaf letter corpus</title>
<style>
:root{color-scheme:dark;--bg:#0b1016;--panel:#141d26;--panel2:#1c2935;--line:#344657;
 --text:#edf4fa;--muted:#94a8b9;--accent:#72c7ff;--warn:#ffc267;--bad:#ff6c76;
 --kash:#ffb347;--nbr:#5d7387}
*{box-sizing:border-box}html,body{height:100%}body{margin:0;background:var(--bg);color:var(--text);
 font:13px system-ui,-apple-system,sans-serif;overflow:hidden;
 display:flex;flex-direction:column}
button,input,select{font:inherit;color:inherit;background:var(--panel2);border:1px solid var(--line);
 border-radius:6px;padding:6px 8px}button{cursor:pointer}button:hover{border-color:var(--accent)}
button.on{border-color:var(--accent);background:#24384a}
header{flex:0 0 auto;display:flex;flex-wrap:wrap;align-items:center;gap:7px;
 padding:9px 12px;border-bottom:1px solid var(--line);background:#111a22}
h1{font-size:15px;margin:0 10px 0 0;white-space:nowrap}
.arabic{font-size:19px;direction:rtl}
input.ar{width:104px;font-size:19px;text-align:center;direction:rtl}
.sep{width:1px;height:24px;background:var(--line);margin:0 3px}
.status{color:var(--muted);font-size:12px;margin-left:auto}
.app{flex:1 1 auto;min-height:0;display:grid;
 grid-template-columns:minmax(0,1fr) 300px}
.stage{overflow:auto;padding:14px;background:#080b0f;min-width:0}
.side{overflow:auto;background:var(--panel);border-left:1px solid var(--line);padding:12px}
.side h2{font-size:11px;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);margin:14px 0 6px}
.side h2:first-child{margin-top:0}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:10px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px;cursor:pointer}
.card:hover{border-color:var(--accent)}.card.sel{border-color:var(--accent);background:var(--panel2)}
.card svg{width:100%;height:110px;display:block}
/* Crops keep their own aspect (object-fit:contain), so a mark narrower or wider
   than the slot is letterboxed. The padding is painted in the card colour, not
   white, so the empty space reads as empty; the crop's own black mask
   background is what marks where the bounding box actually ends. */
.card img.crop{width:100%;height:110px;object-fit:contain;background:var(--panel);
 border-radius:4px;display:block}
.card .meta{color:var(--muted);font:11px ui-monospace,monospace;margin-top:5px;line-height:1.45}
.card .val{color:var(--text)}
.flag{font-size:10px;padding:1px 4px;border-radius:3px;vertical-align:1px}
.flag.warn{background:#4a3a15;color:var(--warn)}.flag.bad{background:#4a1f22;color:var(--bad)}
path{fill:currentColor}
.hist{display:flex;align-items:flex-end;gap:2px;height:80px;margin-top:6px}
.hist div{flex:1;background:var(--accent);opacity:.65;min-height:1px;border-radius:2px 2px 0 0}
.kv{display:flex;justify-content:space-between;gap:8px;padding:3px 0;
 border-bottom:1px solid #22303c;font:12px ui-monospace,monospace}
.kv span:first-child{color:var(--muted)}
img.ctx{width:100%;background:#fff;border-radius:6px;margin-top:6px}
a{color:var(--accent)}
.empty{color:var(--muted);padding:30px;text-align:center}
.empty p{max-width:46ch;margin:10px auto 0;line-height:1.55;text-align:left}
.empty code{color:var(--text);background:var(--panel2);padding:1px 5px;border-radius:3px}
.check{color:var(--muted);font-size:12px;display:inline-flex;align-items:center;gap:4px}
/* The pair reads right-to-left like the writing: the letter the connector
   leaves, then the one it runs into. */
.pair{color:var(--text);font-family:'Scheherazade New','Amiri',serif;font-size:15px}
.pair i{color:var(--muted);font-style:normal;font-size:11px;font-family:ui-monospace,monospace}
/* The connector is painted over the letter it leaves, and the left neighbour
   is dimmed: both are context for the measured span, not the subject. */
.kash{fill:var(--kash)}
.nbr{fill:var(--nbr)}
</style>
</head>
<body>
<header>
  <h1>Letter corpus</h1>
  <select id="kind"><option value="letter">letters</option>
    <option value="mark">marks</option></select>
  <input id="char" class="ar" placeholder="بَت" title="One or more letters in
      writing order, matched as a run inside a single PAW: بت finds a beh
      followed by a teh. Type a diacritic after a letter to require it --
      بَ is a beh carrying a fatha.">
  <select id="match" title="whether the typed letters mean any letter written
      with the same shape, or only those exact letters">
    <option value="class" selected>by shape class</option>
    <option value="letter">by letter</option>
  </select>
  <select id="cls"><option value="">any class</option></select>
  <span class="sep"></span>
  <span id="forms" title="where the typed run sits in its PAW: isol is a PAW
      that is only the run, init one that starts with it, fina one that ends
      with it, medi one that has it inside">
    <button data-form="isolated">isol</button><button data-form="initial">init</button>
    <button data-form="medial">medi</button><button data-form="final">fina</button>
  </span>
  <span class="sep"></span>
  <select id="metric" title="which width the min/max, the sort and the histogram mean">
    <option value="advance">advance</option><option value="bbox">bbox</option>
    <option value="extent">spine</option><option value="height">height</option>
    <option value="kashida">kashida (all)</option>
    <option value="kashidaLeft">kashida left</option>
    <option value="kashidaRight">kashida right</option>
  </select>
  <input id="min" type="number" step="0.1" placeholder="min" style="width:66px">
  <input id="max" type="number" step="0.1" placeholder="max" style="width:66px">
  <input id="pages" placeholder="pages 3-20" style="width:92px">
  <input id="conf" type="number" step="0.05" placeholder="min conf" style="width:82px">
  <!-- Clean by default: a failed PAW still contributes one row per letter so
       nothing vanishes from the counts, but those rows carry the whole PAW's
       extent as their width. Left in, they sit at the top of every width sort
       and drag every median with them. -->
  <select id="quality" title="how the segmenter rated these cuts">
    <option value="clean" selected>clean only</option>
    <option value="">any quality</option>
    <option value="low">low confidence</option>
    <option value="failed">failed</option>
    <option value="shared">shared component</option>
    <option value="joined">joined to previous word</option>
  </select>
  <select id="sort">
    <option value="width_desc">width &darr;</option><option value="width_asc">width &uarr;</option>
    <option value="extent_desc">spine &darr;</option><option value="height_desc">height &darr;</option>
    <option value="kleft_desc">kashida left &darr;</option>
    <option value="kleft_asc">kashida left &uarr;</option>
    <option value="kright_desc">kashida right &darr;</option>
    <option value="kright_asc">kashida right &uarr;</option>
    <option value="confidence_asc">confidence &uarr;</option><option value="page_asc">page</option>
  </select>
  <span class="sep"></span>
  <select id="limit" title="how many to draw at once; above a few thousand the
      browser, not the query, is what takes the time">
    <option>200</option><option selected>400</option>
    <option>1000</option><option>2000</option>
    <option>10000</option><option>50000</option>
    <option value="max">max</option>
  </select>
  <label class="check" title="paint the connector this letter leaves, the one
      on its left as written">
    <input type="checkbox" id="kleft"> k&lsaquo;</label>
  <label class="check" title="paint the connector arriving from the right, and
      draw the letter it comes from">
    <input type="checkbox" id="kright"> k&rsaquo;</label>
  <label class="check" title="show the letters on each side of the connector">
    <input type="checkbox" id="pair"> pair</label>
  <label class="check" title="draw the whole word around the match, with the
      matched letters picked out">
    <input type="checkbox" id="word"> word</label>
  <button id="prev" title="previous page">&lsaquo;</button>
  <button id="next" title="next page">&rsaquo;</button>
  <button id="apply">Search</button>
  <span class="status" id="summary">&nbsp;</span>
</header>
<div class="app">
  <div class="stage" id="stage"><div class="empty">Pick a letter and press Search.</div></div>
  <div class="side" id="side"></div>
</div>
<script>
const $=id=>document.getElementById(id);
const esc=s=>(s??'').toString().replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num=v=>v===null||v===undefined?'&mdash;':(+v).toFixed(2);
let state={forms:new Set(),rows:[],shapes:{},bands:{},words:{},
           selected:null,total:0,offset:0,facets:{}};

function params(){
  return new URLSearchParams({kind:$('kind').value,char:$('char').value,
    match:$('match').value,class:$('cls').value,
    forms:[...state.forms].join(','),metric:$('metric').value,min:$('min').value,max:$('max').value,
    pages:$('pages').value,minConfidence:$('conf').value,
    quality:$('quality').value,
    sort:$('sort').value,limit:$('limit').value,
    offset:state.offset});
}

// The stored frame is a 180-degree rotation of image space (origin at the
// letter's entry point on the baseline, y up, x along the writing direction),
// so scale(-1,-1) puts a glyph back the way it was written.
function glyph(shape,pad){
  pad=pad||0.4;
  const x=-(shape.view_x+shape.view_w)-pad, y=-(shape.view_y+shape.view_h)-pad;
  return {viewBox:`${x} ${y} ${shape.view_w+2*pad} ${shape.view_h+2*pad}`};
}

// A letter's stored outline is the body *plus* the connector leaving it --
// letter_mask ORs in the join pixels, because a letter costs the line what it
// is written with. The whole thing is drawn; this only says which slice of it
// is the connector, so that slice can be painted apart from the body.
//
// The frame's origin is the entry point and x runs along the writing, so the
// connector is the last `kashida` of the span -- the high-x end. Returns null
// when the row cannot say how much of its span is connector, and the glyph is
// drawn in one colour as before.
const cssId=s=>s.replace(/[^A-Za-z0-9_-]/g,'_');
// Which connectors the cards are asked to show. Picking a kashida metric
// ticks the matching box once, so measuring one still shows it without the
// two being welded together: either can be turned off.
const showLeft=()=>$('kleft').checked;
const showRight=()=>$('kright').checked;
function kashidaWindow(row,shape){
  if(!showLeft())return null;
  // The owned figure, not the raw column: a group member that does not own
  // the connector would otherwise get a band as wide as the whole group's.
  const k=+row.kashida_left;
  if(!(k>0)||!shape||!shape.path_d)return null;
  const x1=shape.view_x+shape.view_w,x0=Math.max(shape.view_x,x1-k);
  if(!(x1>x0))return null;
  return {x0:x0,x1:x1};
}

// Every letter's outline is stored in its own frame -- origin at that letter's
// entry point, y measured from the baseline. Letters of one PAW therefore share
// a y axis exactly and differ in x by precisely the advance, which is defined
// as entry-to-entry. So a run is placed by a running sum of advances and
// nothing is registered by eye.
const nextId=id=>id.replace(/let(\d+)$/,(m,i)=>'let'+(+i+1));

// The members of the searched run, in writing order, each with the offset that
// puts it beside the one before. The first is the row itself and carries no
// offset. A member whose advance is unknown ends the run: without it the ones
// after it cannot be placed at all, and guessing would draw a false spacing.
function groupDraw(row){
  const out=[];let dx=0;
  for(let i=0;;i++){
    const id=row[`g${i}_id`];
    if(id===undefined)break;
    const s=state.shapes[id];
    if(i&&(!s||!s.path_d))break;
    if(i)out.push({shape:s,dx:dx,id:id});
    const adv=row[`g${i}_adv`];
    if(adv==null)return out;
    dx+=+adv;
  }
  return out;
}

// `pair` adds the letter the run runs into -- the one after its last member,
// not after its first, or a two-letter search would draw its own second letter
// twice.
function neighbourDraw(row){
  if(!$('pair').checked)return null;
  const members=groupDraw(row);
  const lastId=members.length?members[members.length-1].id:row.id;
  const lastDx=members.length?members[members.length-1].dx:0;
  const key=`g${members.length}_adv`;
  const adv=row[key]!==undefined?row[key]:row.adv_w_norm;
  if(adv==null)return null;
  const s=state.shapes[nextId(lastId)];
  if(!s||!s.path_d)return null;
  return {shape:s,dx:lastDx+(+adv)};
}
function unionBox(a,b){
  if(!b)return a;
  const x0=Math.min(a.view_x,b.view_x),y0=Math.min(a.view_y,b.view_y);
  return {view_x:x0,view_y:y0,
          view_w:Math.max(a.view_x+a.view_w,b.view_x+b.view_w)-x0,
          view_h:Math.max(a.view_y+a.view_h,b.view_y+b.view_h)-y0};
}

// An uncut letter has no width and no outline, so a card that only knows how
// to print those reads as empty and the failures cannot be browsed. Fall back
// to the PAW it sits in -- its letters and its span -- named as the PAW so it
// is never mistaken for this letter's own measurement.
function measure(row,label){
  // A fused member reports its group's advance, because it has none of its
  // own -- so the number is named after the ligature rather than after the
  // letter, or the card would claim a Yeh 5.23 wide when 5.23 is the whole
  // of the fa-yeh it is written inside.
  const fused=row.group_size>1&&row.adv_w_norm==null&&label==='w';
  if(fused&&row.width!=null)
    return `<span class="muted">lig ${esc(row.lig||'')}</span> `+
           `<span class="val">${num(row.width)}</span> &middot; h ${num(row.height)}`;
  if(row.width!=null||row.height!=null)
    return `${label} <span class="val">${num(row.width)}</span> &middot; h ${num(row.height)}`;
  if(row.paw_norm==null)return '<span class="muted">not segmented</span>';
  return `<span class="muted">paw ${esc(row.paw_letters||'')} <span class="val">${num(row.paw_norm)}</span></span>`;
}
// The connector runs from this letter into the next one in logical order,
// which is the letter to its left on the line. Joined with a tatweel so the
// pair reads as it is written -- and `char` goes first, because bidi draws the
// first character of an Arabic run rightmost, which is where this letter sits.
// The right/left naming is spelled out in the tooltip and the side panel,
// where there is room for words.
// The connector is painted by redrawing the same path through a clip, rather
// than by splitting the outline: one contour can enclose body and connector
// together, and there is no seam in it to cut at. Overpainting needs no seam.
//
// The clip is the band the server traced through the letter's own mask, which
// excludes body ink leaning into the connector's span. Falls back to the plain
// span -- a rectangle unbounded in y -- when no band could be followed.
// Every member of a searched run carries its own connector, and the kashida
// metric adds all of them up, so all of them are painted -- otherwise the
// number counts three and the picture shows one. Members are keyed by their
// own id, and only the row itself falls back to the plain span when no band
// could be traced, because only the row has a kashida figure of its own.
function connectorClip(id,shape,row){
  // Nothing to carve when the connector arrived as its own outline.
  if(shape&&shape.kashida_path_d)return '';
  const band=state.bands[id];
  if(band&&band.length>1){
    const top=band.map(p=>`${p[0]},${p[1]}`).join(' ');
    const bottom=band.slice().reverse().map(p=>`${p[0]},${p[2]}`).join(' ');
    return `<clipPath id="k_${cssId(id)}"><polygon points="${top} ${bottom}"/></clipPath>`;
  }
  const w=row&&kashidaWindow(row,shape);
  if(!w)return '';
  const tall=Math.abs(shape.view_y)+Math.abs(shape.view_h)+4;
  return `<clipPath id="k_${cssId(id)}"><rect x="${w.x0.toFixed(3)}"
      y="${(-tall).toFixed(3)}" width="${(w.x1-w.x0).toFixed(3)}"
      height="${(2*tall).toFixed(3)}"/></clipPath>`;
}
// The connector is its own outline when the index has one: the segmenter
// decided which pixels are join rather than body, so it is painted rather
// than carved out of the letter. Older rows have no kashida_path_d and fall
// back to clipping the letter through a traced band.
function glyphLayers(id,shape,row){
  const base=`<path d="${shape.path_d}"/>`;
  if(!showLeft())return base;
  if(shape.kashida_path_d)
    return base+`<path class="kash" d="${shape.kashida_path_d}"/>`;
  if(!connectorClip(id,shape,row))return base;
  return base+`<g clip-path="url(#k_${cssId(id)})">`+
         `<path class="kash" d="${shape.path_d}"/></g>`;
}
// Every letter the search asked for, in writing order.
function runText(row){
  const run=[];
  for(let i=0;row[`g${i}_char`]!==undefined;i++)run.push(row[`g${i}_char`]);
  return run.length?run.join(''):(row.char||row.class||'?');
}

// The word the match sits in, each letter offset from the match's own entry
// point. Letters of one word share a baseline, so x is the whole registration.
function wordDraw(row){
  if(!$('word').checked)return null;
  const w=state.words[row.word_id];
  if(!w)return null;
  const mine=w.letters.find(e=>e[0]===row.id);
  if(!mine)return null;
  const own=new Set([row.id]);
  for(let i=1;row[`g${i}_id`]!==undefined;i++)own.add(row[`g${i}_id`]);
  const out=[];
  for(const [id,entry] of w.letters){
    if(own.has(id))continue;
    const s=state.shapes[id];
    if(s&&s.path_d)out.push({shape:s,dx:(mine[1]-entry)/w.scale,id:id});
  }
  return out;
}

// A right-hand connector belongs to the letter before this one, so showing it
// means drawing that letter. It is placed by its own advance, which is by
// definition the distance from its entry point to this letter's.
function rightSided(row){
  return $('kind').value==='letter'&&showRight()&&+row.kashida_right>0;
}
// The letter before this one may be a member of an uncut ligature -- تح in
// p193:l11_w03:paw0 -- and then the connector belongs to the ligature, not to
// one of its letters. Its members share one entry point and therefore one
// frame, so they are all drawn at the same offset and together read as the
// ligature they are.
const withLet=(id,n)=>id.replace(/let\d+$/,'let'+n);
function prevGroupIds(row){
  if(!row.prev_id||row.prev_index==null)return [];
  const size=Math.max(1,+row.prev_group_size||1);
  const last=+row.prev_index;
  const out=[];
  for(let k=size-1;k>=0;k--)if(last-k>=0)out.push(withLet(row.prev_id,last-k));
  return out;
}
function prevDraw(row){
  if(!rightSided(row)||row.prev_adv==null)return null;
  const dx=-(+row.prev_adv);
  const out=[];
  for(const id of prevGroupIds(row)){
    const s=state.shapes[id];
    if(s&&s.path_d)out.push({shape:s,dx:dx,id:id});
  }
  return out.length?out:null;
}

// The row itself may be one member of an uncut ligature, and then it holds
// only part of that ligature's ink. In p088:l13_w07:paw0 the beh-hah body was
// traced onto the first member and the connector is the entire second one, so
// a card for the second drew a bare stroke with nothing to read it against.
// The members share an entry point, so their frames are the same frame and
// the siblings drop in at zero offset -- no advance to look up and none to
// guess. Drawn as context, because they are not what was measured.
function ownGroupIds(row){
  const size=Math.max(1,+row.group_size||1);
  if(size<2||row.group_start==null)return [];
  const run=new Set([row.id]);
  for(let i=1;row[`g${i}_id`]!==undefined;i++)run.add(row[`g${i}_id`]);
  const out=[];
  for(let k=0;k<size;k++){
    const id=withLet(row.id,(+row.group_start)+k);
    if(!run.has(id))out.push(id);
  }
  return out;
}
function ownGroupDraw(row){
  if($('kind').value!=='letter')return null;
  const out=[];
  for(const id of ownGroupIds(row)){
    const s=state.shapes[id];
    if(s&&s.path_d)out.push({shape:s,dx:0,id:id});
  }
  return out.length?out:null;
}

function pairMarkup(row){
  if(!$('pair').checked||!row.char)return '';
  const run=[];
  for(let i=0;row[`g${i}_char`]!==undefined;i++)run.push(row[`g${i}_char`]);
  const chars=run.length?run:[row.char];
  const after=row[`g${chars.length-1}_next`]!==undefined
      ?row[`g${chars.length-1}_next`]:row.next_char;
  if(!after)
    return '<span class="pair" title="no left neighbour">&mdash;</span><br>';
  return `<span class="pair" title="run ${esc(chars.join(''))}, then `+
         `${esc(after)}">${esc(chars.join('\u0640'))}\u0640${esc(after)}`+
         `</span><br>`;
}
// Right and left are the sides as written: the connector arriving at this
// letter and the one leaving it. A dash means there is none -- the PAW starts
// or ends here -- which is different from one measured at zero.
function sides(row){
  if($('kind').value==='mark')return '';
  if(row.kashida_left==null&&row.kashida_right==null)return '';
  const cell=v=>v==null?'<span class="muted">&mdash;</span>'
                       :`<span class="val">${(+v).toFixed(2)}</span>`;
  return `<span class="muted">k &rsaquo; ${cell(row.kashida_right)} `+
         `&middot; &lsaquo; ${cell(row.kashida_left)}</span><br>`;
}
function cardMarkup(row){
  const s=state.shapes[row.id];
  const mine=s&&s.path_d;
  // A letter the segmenter fused into a ligature holds no ink of its own --
  // the final Yeh of a fa-yeh is written inside the Feh's sweep, and 6,095
  // letters are in that position. It still has siblings that do hold the ink,
  // so the card draws the ligature rather than reporting "no outline", which
  // said nothing and made half of all final Yehs look like a broken index.
  const own=ownGroupDraw(row);
  let art;
  if(mine||own){
    const all=mine?[{shape:s,dx:0,id:row.id}].concat(groupDraw(row)):[];
    // The whole word already contains the left neighbour, so it stands in for
    // it rather than being drawn twice.
    const nb=neighbourDraw(row);
    const pv=prevDraw(row);
    const pvIds=new Set((pv||[]).map(m=>m.id));
    // The whole word already carries this letter's ligature siblings, at the
    // same zero offset, so they are only added when it is not being drawn.
    const wd=wordDraw(row);
    const context=(wd||(nb?[nb]:[])).concat(pv||[]).concat((wd?null:own)||[]);
    let box=mine?s:own[0].shape;
    for(const m of all.concat(context))
      box=unionBox(box,{view_x:m.shape.view_x+m.dx,view_y:m.shape.view_y,
                        view_w:m.shape.view_w,view_h:m.shape.view_h});
    art=`<svg viewBox="${glyph(box).viewBox}" preserveAspectRatio="xMidYMid meet">
         <defs>${all.map(m=>connectorClip(m.id,m.shape,m.dx?null:row)).join('')}</defs>
         <g transform="scale(-1,-1)">
         ${context.map(m=>`<g transform="translate(${m.dx.toFixed(3)},0)">
           <path class="nbr" d="${m.shape.path_d}"/>${
             pvIds.has(m.id)&&m.shape.kashida_path_d
               ? `<path class="kash" d="${m.shape.kashida_path_d}"/>` : ''}
           </g>`).join('')}
         ${all.map(m=>`<g transform="translate(${m.dx.toFixed(3)},0)">`+
           glyphLayers(m.id,m.shape,m.dx?null:row)+`</g>`).join('')}
         </g></svg>`;
  }else if($('kind').value==='mark'){
    art=`<img class="crop" src="/image?kind=mark&id=${encodeURIComponent(row.id)}" alt="">`;
  }else{
    art=`<svg viewBox="0 0 10 10"><text x="5" y="6" font-size="3" fill="#7d8fa0"
         text-anchor="middle">no outline</text></svg>`;
  }
  const flag=row.paw_joined?
             '<span class="flag warn" title="this PAW\'s body is drawn into the word before it">joined</span>'
            :row.paw_merged?
             '<span class="flag warn" title="two PAWs in one component, not cut apart">shared</span>'
            :(row.ok===0?'<span class="flag bad">failed</span>'
            :(row.low_confidence?'<span class="flag warn">low conf</span>':''));
  // The sorted-on number goes first and is named, because it is not always a
  // width: in the kashida kind it is the connector, and the outline beside it
  // is the whole letter, which would otherwise read as the measurement.
  // The measurement names itself, because it is no longer implied by a kind.
  const label={kashida:'kashida',kashidaLeft:'kashida \u2039',
               kashidaRight:'kashida \u203a'}[$('metric').value]||'w';
  return `<article class="card ${state.selected===row.id?'sel':''}" data-id="${esc(row.id)}">
    ${art}
    <div class="meta"><span class="arabic">${esc(runText(row))}</span>
      ${esc(row.group_form||row.form||'')} ${flag}<br>
      ${pairMarkup(row)}${measure(row,label)}<br>
      ${sides(row)}
      p${row.page} ${esc(row.word_key||'')}</div></article>`;
}

// "No matches" is ambiguous: it can mean the corpus has none, or that the
// filter combination cannot match by construction. Class plus form is where
// that bites -- beh is stored as senna in initial and medial, and letters that
// do not join on the left have no initial or medial form at all.
function emptyHint(){
  if($('kind').value==='mark')return '';
  const cls=$('cls').value, forms=[...state.forms];
  if(!cls||!forms.length)return '';
  const attested=(state.facets.class_forms||{})[cls]||[];
  const missing=forms.filter(f=>!attested.includes(f));
  if(!missing.length)return '';
  const folded=(state.facets.folds_to||{})[cls];
  const where=attested.length?`only in ${attested.join(', ')}`:'in no form';
  return `<p><code>${esc(cls)}</code> occurs ${esc(where)}.`+
    (folded&&missing.some(f=>f==='initial'||f==='medial')
      ? ` In initial and medial position it shares the senna tooth, so those
          rows are stored under <code>${esc(folded)}</code> &mdash; select that
          class instead, or clear the class and type the letter.`
      : ' Clear the class or pick another form.')+'</p>';
}

function renderGrid(){
  if(!state.rows.length){
    $('stage').innerHTML=`<div class="empty">No matches.${emptyHint()}</div>`;return}
  $('stage').innerHTML=`<div class="grid">${state.rows.map(cardMarkup).join('')}</div>`;
}

function render(){
  renderGrid();
}

// Selecting a card moves one CSS class, so it must not redraw the grid. It
// used to: the click handler called render(), which rebuilt every card's
// markup and handed the browser a fresh subtree to lay out. At ten thousand
// matches that is about twelve megabytes of HTML and the best part of a
// second of blocking layout -- to highlight a border and fill a side panel
// that costs a millisecond. The listener is delegated so it survives the
// innerHTML replacement in renderGrid and does not have to be re-attached
// once per card.
$('stage').onclick=e=>{
  const card=e.target.closest('.card');
  if(!card||card.dataset.id===state.selected)return;
  const previous=$('stage').querySelector('.card.sel');
  if(previous)previous.classList.remove('sel');
  card.classList.add('sel');
  state.selected=card.dataset.id;
  sidebar();
};

function sidebar(){
  const row=state.rows.find(r=>r.id===state.selected);
  const stats=state.stats||{};
  const bars=(stats.histogram||[]).map(v=>{
    const height=stats.peak?Math.round(100*v/stats.peak):0;
    return `<div style="height:${height}%" title="${v}"></div>`}).join('');
  let html=`<h2>Distribution</h2>
    <div class="kv"><span>matches</span><span>${state.total}</span></div>
    <div class="kv"><span>median</span><span>${num(stats.median)}</span></div>
    <div class="kv"><span>MAD</span><span>${num(stats.mad)}</span></div>
    <div class="kv"><span>min / max</span><span>${num(stats.min)} / ${num(stats.max)}</span></div>
    <div class="hist">${bars}</div>`;
  if(row){
    const kind=$('kind').value;
    html+=`<h2>Selected</h2>
      <div class="kv"><span>id</span><span>${esc(row.id)}</span></div>
      <div class="kv"><span>text</span><span class="arabic">${esc(row.word_text||'')}</span></div>
      <div class="kv"><span>advance</span><span>${num(row.adv_w_norm)}</span></div>
      <div class="kv"><span>bbox</span><span>${num(row.bbox_w_norm)}</span></div>
      <div class="kv"><span>spine</span><span>${num(row.extent_norm)}</span></div>
      <div class="kv"><span>kashida (all)</span><span>${num(row.kashida_all)}</span></div>
      <div class="kv"><span>kashida right</span><span>${num(row.kashida_right)}</span></div>
      <div class="kv"><span>kashida left</span><span>${num(row.kashida_left)}</span></div>${
        row.next_char?`
      <div class="kv"><span>right of connector</span>
        <span class="arabic">${esc(row.char||'')}</span></div>
      <div class="kv"><span>left of connector</span>
        <span class="arabic">${esc(row.next_char)}</span></div>`:''}
      <div class="kv"><span>height</span><span>${num(row.height)}</span></div>
      <div class="kv"><span>confidence</span><span>${num(row.confidence)}</span></div>
      <div class="kv"><span>quality</span><span>${
        row.paw_joined?'joined to previous word':
        row.paw_merged?'shared component':
        (row.ok===0?'failed':(row.low_confidence?'low confidence':'clean'))}</span></div>
      <img class="ctx" src="/image?kind=word&id=${encodeURIComponent(row.id)}" alt="word">
      <p><a href="${esc(row.viewer_url||'#')}" target="_blank">open in segmentation viewer</a></p>`;
    void kind;
  }
  $('side').innerHTML=html;
}

// Only the kashida view paints a connector, so only it pays for the bands.
async function loadConnectorBands(){
  state.bands={};
  if($('kind').value!=='letter'||!showLeft())return;
  const seen=new Set();
  for(const row of state.rows){
    seen.add(row.id);
    for(let i=1;row[`g${i}_id`]!==undefined;i++)seen.add(row[`g${i}_id`]);
  }
  // A row whose index already carries the traced connector needs nothing from
  // /api/connectors, which costs about a second per three hundred rows.
  const all=[...seen].filter(id=>{
    const s=state.shapes[id];
    return !(s&&s.kashida_path_d);
  });
  if(!all.length)return;
  for(let i=0;i<all.length;i+=300){
    const ids=all.slice(i,i+300).join(',');
    Object.assign(state.bands,
        await(await fetch('/api/connectors?ids='+encodeURIComponent(ids))).json());
    if(all.length>2000)
      $('summary').textContent=`tracing connectors ${Math.min(i+300,all.length)} / ${all.length}...`;
  }
}

// The left neighbour is not a row of the result -- it is whatever letter this
// one's connector runs into -- so its outline has to be asked for separately.
// Fetched once and kept, so ticking `pair` off and on again costs nothing.
async function loadNeighbourShapes(){
  const want=new Set();
  for(const row of state.rows){
    let last=row.id;
    for(let i=1;row[`g${i}_id`]!==undefined;i++)
      want.add(last=row[`g${i}_id`]);
    if($('pair').checked)want.add(nextId(last));
    if(rightSided(row))for(const id of prevGroupIds(row))want.add(id);
    for(const id of ownGroupIds(row))want.add(id);
  }
  await fetchShapes([...want].filter(id=>!(id in state.shapes)));
  return loadWordShapes();
}

// The word view needs the layout of each match's word, and then an outline for
// every letter in it. Both are cached, so ticking the box off and on is free.
async function loadWordShapes(){
  if(!$('word').checked)return;
  const words=[...new Set(state.rows.map(r=>r.word_id))]
      .filter(id=>id&&!(id in state.words));
  for(let i=0;i<words.length;i+=200){
    const ids=words.slice(i,i+200).join(',');
    Object.assign(state.words,
        await(await fetch('/api/words?ids='+encodeURIComponent(ids))).json());
    if(words.length>500)
      $('summary').textContent=`loading words ${Math.min(i+200,words.length)} / ${words.length}...`;
  }
  const want=new Set();
  for(const row of state.rows){
    const w=state.words[row.word_id];
    if(w)for(const [id] of w.letters)if(!(id in state.shapes))want.add(id);
  }
  return fetchShapes([...want]);
}

async function fetchShapes(want){
  for(let i=0;i<want.length;i+=500){
    const ids=want.slice(i,i+500).join(',');
    Object.assign(state.shapes,
        await(await fetch('/api/shapes?ids='+encodeURIComponent(ids))).json());
  }
  // Remember the misses too, so a later pass does not re-ask for outlines the
  // server has already said it does not have.
  for(const id of want)if(!(id in state.shapes))state.shapes[id]=null;
}

async function load(){
  $('summary').textContent='searching...';
  const p=params();
  const data=await(await fetch('/api/search?'+p)).json();
  if(data.error){
    state.rows=[];state.shapes={};state.bands={};
    $('summary').textContent=data.error;
    $('stage').innerHTML=`<div class="empty">${esc(data.error)}</div>`;
    sidebar();return;
  }
  state.rows=data.rows;state.total=data.total;state.stats=data.stats;state.shapes={};
  if($('kind').value!=='mark'&&data.rows.length){
    // Ids travel in the query string, so they go in batches: at the larger
    // limits a single request line would be about a megabyte and the server
    // would refuse it before the query ever ran.
    const all=data.rows.map(r=>r.id),merged={};
    for(let i=0;i<all.length;i+=500){
      const ids=all.slice(i,i+500).join(',');
      Object.assign(merged,await(await fetch('/api/shapes?ids='+encodeURIComponent(ids))).json());
      if(all.length>2000)$('summary').textContent=`loading outlines ${Math.min(i+500,all.length)} / ${all.length}...`;
    }
    state.shapes=merged;
    await loadNeighbourShapes();
    await loadConnectorBands();
  }
  $('prev').disabled=state.offset<=0;
  $('next').disabled=data.to>=data.total;
  $('summary').textContent = data.rows.length>=data.total
    ? `all ${data.total} matches`
    : `${data.total} matches, showing ${data.from+1}–${data.to}`;
  render();sidebar();
}

function paged(delta){
  // "max" is not a number; a page of everything has no next page anyway, so
  // fall back to how many rows actually came back.
  const step=(+$('limit').value)||state.rows.length||400;
  state.offset=Math.max(0,state.offset+delta*step);
  load();
}

$('forms').onclick=e=>{const f=e.target.dataset.form;if(!f)return;
  state.forms.has(f)?state.forms.delete(f):state.forms.add(f);
  e.target.classList.toggle('on',state.forms.has(f))};
$('apply').onclick=()=>{state.offset=0;load()};
$('prev').onclick=()=>paged(-1);
$('next').onclick=()=>paged(1);
$('limit').onchange=()=>{state.offset=0;load()};
$('pair').onchange=async()=>{await loadNeighbourShapes();render()};
$('word').onchange=async()=>{await loadNeighbourShapes();render()};
// Painting is a view, not a query: no row changes, so nothing is refetched
// beyond the outlines the new side needs.
$('kleft').onchange=async()=>{await loadConnectorBands();render()};
$('kright').onchange=async()=>{await loadNeighbourShapes();render()};
document.addEventListener('keydown',e=>{
  if(e.key==='Enter'&&e.target.tagName==='INPUT'){state.offset=0;load()}});
// The class list means different things per kind -- shape classes for letters,
// mark classes for marks -- so it is refilled whenever the kind changes rather
// than leaving "senna" on offer while marks are being searched.
function fillClasses(){
  const names=(state.facets[$('kind').value==='mark'?'mark_classes':'classes'])||[];
  const keep=$('cls').value;
  $('cls').innerHTML='<option value="">any class</option>'+
    names.map(c=>`<option>${esc(c)}</option>`).join('');
  if(names.includes(keep))$('cls').value=keep;
}
// A mark is measured by its box and nothing else, so the select is set to
// what the server will use rather than left showing something it ignores.
const KASHIDA_METRICS={kashida:'both',kashidaLeft:'left',kashidaRight:'right'};
let letterMetric=$('metric').value;
$('metric').onchange=()=>{
  if($('kind').value!=='letter')return;
  letterMetric=$('metric').value;
  // Measuring a connector usually means wanting to see it, but only as a
  // starting point: neither box is forced back on once it has been cleared.
  const side=KASHIDA_METRICS[$('metric').value];
  if(side&&!$('kleft').checked&&!$('kright').checked){
    if(side!=='right')$('kleft').checked=true;
    if(side!=='left')$('kright').checked=true;
  }
  state.offset=0;load();
};
$('match').onchange=()=>{state.offset=0;load()};
function syncKind(){
  const kind=$('kind').value;
  $('metric').value=kind==='letter'?letterMetric:'bbox';
  $('metric').disabled=kind==='mark';
  for(const id of ['kleft','kright','pair','word'])
    $(id).disabled=kind==='mark';
}
$('kind').onchange=()=>{syncKind();fillClasses();state.offset=0;load()};
syncKind();
fetch('/api/facets').then(r=>r.json()).then(f=>{state.facets=f;fillClasses()});
</script>
</body>
</html>'''


# ---------------------------------------------------------------------------
# queries
# ---------------------------------------------------------------------------

def parse_pages(spec: str) -> Tuple[Optional[int], Optional[int]]:
  spec = (spec or "").strip()
  if not spec:
    return None, None
  if "-" in spec:
    low, _, high = spec.partition("-")
    try:
      return int(low), int(high)
    except ValueError:
      return None, None
  try:
    return int(spec), int(spec)
  except ValueError:
    return None, None


def float_or_none(value: str) -> Optional[float]:
  try:
    return float(value)
  except (TypeError, ValueError):
    return None


# Combining marks, as they are typed, mapped to the classes the recognizer
# writes. Typing a diacritic after a letter asks for that letter carrying that
# mark, so `بَ` is "beh with a fatha". Only the marks that a reader would
# actually type to pick out a letter are here; the waqf signs and the small
# letters are chosen from the mark kind instead, where they are the subject
# rather than a qualifier.
ARABIC_MARK_CLASSES = {
    "\u064b": "fathatan", "\u064c": "dammatan", "\u064d": "kasratan",
    "\u064e": "fatha", "\u064f": "damma", "\u0650": "kasra",
    "\u0651": "shadda", "\u0652": "sukun", "\u0653": "maddah",
    "\u0654": "hamza_above", "\u0655": "hamza_below",
    "\u0670": "small_alef", "\u06e1": "sukun",
}


def parse_letter_sequence(text: str) -> List[Tuple[str, List[str]]]:
  """Typed text -> [(base letter, [mark class, ...]), ...], in writing order.

  Marks bind to the letter they follow, exactly as they are typed and stored.
  A mark with no letter in front of it is a typo rather than a query, so it is
  reported instead of being quietly dropped.
  """
  slots: List[Tuple[str, List[str]]] = []
  for char in text:
    if char.isspace():
      continue
    mark = ARABIC_MARK_CLASSES.get(char)
    if mark is None:
      slots.append((char, []))
    elif slots:
      slots[-1][1].append(mark)
    else:
      raise ValueError(f"mark {char!r} has no letter before it")
  return slots


# The group's form, expressed as where it sits in its PAW. Position and stored
# form agree on every letter in the corpus (48278/48278 initial, 41461 medial,
# 48278 final, 37854 isolated, nothing off the diagonal), which is what makes
# the single-letter meaning extend to a group without redefining anything:
# isolated is a PAW that is only the group, initial one that starts with it.
GROUP_FORM_TESTS = {
    "isolated": "{base} = 0 AND {end} = {last}",
    "initial": "{base} = 0 AND {end} < {last}",
    "final": "{base} > 0 AND {end} = {last}",
    "medial": "{base} > 0 AND {end} < {last}",
}


# A group's measurement has to describe the whole run, and how depends on the
# metric. Advance, spine extent and kashida are contiguous spans along the
# writing, so they add up exactly; a bounding box does not, and neither does a
# height, so those are taken from the members' own boxes.
GROUP_WIDTH_SUMS = {"adv_w_norm", "extent_norm", "kashida_norm"}


def group_width_sql(width: str, aliases: List[str]) -> str:
  """The metric ``width`` measured across a whole group of letters."""
  if len(aliases) == 1:
    # Only a single-letter search substitutes the group's advance. A run that
    # spans a fused ligature would add the same span once per member, and a
    # run whose total is unknown must stay unknown rather than be doubled.
    if width == "adv_w_norm":
      return advance_sql(aliases[0])
    return f"{aliases[0]}.{width}"
  if width in GROUP_WIDTH_SUMS:
    # NULL propagates on purpose: one member the segmenter could not cut makes
    # the group's span unknown rather than shorter, and NULLs already sort last.
    return "(" + " + ".join(f"{a}.{width}" for a in aliases) + ")"
  # Box metrics come from the ink. bbox_x/bbox_y are word-space pixels and a
  # group lives inside one word, so its members already share an origin.
  axis = "bbox_x" if width == "bbox_w_norm" else "bbox_y"
  size = "bbox_w" if width == "bbox_w_norm" else "bbox_h"
  high = ", ".join(f"{a}.{axis} + {a}.{size}" for a in aliases)
  low = ", ".join(f"{a}.{axis}" for a in aliases)
  return f"((max({high}) - min({low})) * 1.0 / pg.nominal_stroke_px)"


def group_form_sql(count: int) -> str:
  """The run's own joining form, by the same rule that gives a letter its own.

  A card that says "medial" about a three-letter run has to mean the run is
  medial, not that its first letter is: `بت` inside `بتغا` is initial as a run
  while its beh is initial and its teh medial, and the second reading is the
  one the search was expressed in.
  """
  last = ("(SELECT MAX(g.letter_index) FROM letters g "
          "WHERE g.paw_id = l.paw_id)")
  end = f"l.letter_index + {count - 1}"
  return ("CASE"
          f" WHEN l.letter_index = 0 AND {end} = {last} THEN 'isolated'"
          f" WHEN l.letter_index = 0 THEN 'initial'"
          f" WHEN {end} = {last} THEN 'final'"
          " ELSE 'medial' END")


def letter_query(query: Dict[str, List[str]], width: str = ""
                 ) -> Tuple[str, List[Any], str, str, List[str]]:
  """WHERE clause, parameters, width expression, joins, and group aliases.

  ``width`` overrides the requested metric, for a kind that measures one thing
  and nothing else.
  """
  first = lambda key, default="": (query.get(key, [default]) or [default])[0]
  metric = width or first("metric", "advance")
  clauses: List[str] = []
  values: List[Any] = []

  # The typed text is a sequence, not a set of alternatives: `بت` is a beh
  # followed by a teh, matched by joining one copy of `letters` per slot on
  # consecutive indices inside one PAW. The row returned is the group's first
  # letter, so every measurement on the card still describes something nameable.
  slots = parse_letter_sequence(first("char").strip())
  if len(slots) > MAX_GROUP_LETTERS:
    raise ValueError(
        f"at most {MAX_GROUP_LETTERS} letters can be searched as a group")
  # The letter before this one in the same PAW, for the connector arriving on
  # its right. Left-joined because the first letter of a PAW has none: a PAW
  # boundary is exactly where the writing lifts, so there is no stroke there to
  # measure, and NULL says that rather than zero pretending it was measured.
  joins = (" LEFT JOIN letters lp ON lp.paw_id = l.paw_id "
           "AND lp.letter_index = l.letter_index - 1")
  aliases = ["l"]
  by_class = first("match") == "class"
  for position, (char, marks) in enumerate(slots):
    alias = "l" if position == 0 else f"lg{position}"
    if position:
      aliases.append(alias)
      joins += (f" JOIN letters {alias} ON {alias}.paw_id = l.paw_id "
                f"AND {alias}.letter_index = l.letter_index + {position}")
    if by_class:
      # A letter presents as one class where it joins on the left and another
      # where it does not -- beh is senna in initial and medial. Both are
      # accepted and the form filter decides which can apply, so "by class"
      # means the shape as written rather than a second spelling of the letter.
      wanted = sorted({letter_classes.shape_class(char, "final"),
                       letter_classes.shape_class(char, "medial")})
      clauses.append(f"{alias}.char_class IN ({','.join('?' * len(wanted))})")
      values.extend(wanted)
    else:
      clauses.append(f"{alias}.char = ?")
      values.append(char)
    for mark in marks:
      # Set membership rather than a correlated EXISTS, so the marks table is
      # visited once for the whole query instead of once per candidate letter.
      clauses.append(f"{alias}.letter_id IN "
                     "(SELECT owning_letter_id FROM marks WHERE class = ?)")
      values.append(mark)

  if first("class"):
    clauses.append("l.char_class = ?")
    values.append(first("class"))

  forms = [f for f in first("forms").split(",") if f]
  if forms and len(slots) > 1:
    # The form describes the group, so it is read off the group's span rather
    # than off its first letter: `بت` is initial in `بتر` and isolated in `بت`,
    # and l.form would call it initial in both.
    last = ("(SELECT MAX(g.letter_index) FROM letters g "
            "WHERE g.paw_id = l.paw_id)")
    tests = [GROUP_FORM_TESTS[f].format(
                 base="l.letter_index",
                 end=f"l.letter_index + {len(slots) - 1}", last=last)
             for f in forms if f in GROUP_FORM_TESTS]
    if tests:
      clauses.append("(" + " OR ".join(f"({t})" for t in tests) + ")")
  elif forms:
    clauses.append(f"l.form IN ({','.join('?' * len(forms))})")
    values.extend(forms)

  width_sql = metric_sql(metric, aliases)
  # `advance_sql` reads the traced frame, so the join has to reach the count
  # and the histogram, not only the rows. Added only for the metric that uses
  # it: on an unfiltered corpus-wide count it costs 147 ms -> 218 ms.
  if metric == "advance" and len(aliases) == 1:
    joins += SHAPE_JOIN

  low, high = parse_pages(first("pages"))
  if low is not None:
    clauses.append("l.page BETWEEN ? AND ?")
    values.extend([low, high])
  minimum = float_or_none(first("min"))
  if minimum is not None:
    clauses.append(f"{width_sql} >= ?")
    values.append(minimum)
  maximum = float_or_none(first("max"))
  if maximum is not None:
    clauses.append(f"{width_sql} <= ?")
    values.append(maximum)
  confidence = float_or_none(first("minConfidence"))
  if confidence is not None:
    clauses.append("l.confidence >= ?")
    values.append(confidence)

  # The segmenter records how well it did rather than dropping what it could
  # not cut, so a plain query mixes clean letters with ones it flagged. Make
  # that a choice instead of a default.
  # A member of an uncut ligature has no ink of its own -- the whole shape was
  # charged to the letter the group fused into -- but it was segmented
  # successfully and its span is known. Reading `ink_px = 0` as failure hid
  # 6,095 such letters, 1,240 of them the final Yeh of a fa-yeh, so "class
  # yeh, final, clean" reported 2,347 of 5,076 and called the rest failures.
  fused = "l.group_size > 1"
  quality = first("quality")
  if quality == "clean":
    clauses.append(
        f"l.ok = 1 AND l.low_confidence = 0 AND (l.ink_px > 0 OR {fused})")
  elif quality == "low":
    clauses.append("l.low_confidence = 1")
  elif quality == "failed":
    clauses.append(f"(l.ok = 0 OR (l.ink_px = 0 AND NOT {fused}))")
  elif quality == "shared":
    # Two PAWs the analyzer found in one component and nothing has cut apart.
    # They carry no measurement at all, so they are invisible in every width
    # sort by construction; this is the way to look at them on purpose.
    clauses.append("pw.merged = 1")
  elif quality == "joined":
    # The other way a PAW loses its ink: its body is drawn into the word
    # before it, so the letters are here and the ink is not. Kept apart from
    # `shared` because it is a different phenomenon -- a touch across a word
    # boundary rather than inside one -- and worth querying on its own.
    clauses.append("pw.connected_to_previous_word = 1")

  return (" AND ".join(clauses) or "1"), values, width_sql, joins, aliases


def mark_query(query: Dict[str, List[str]]) -> Tuple[str, List[Any], str]:
  first = lambda key, default="": (query.get(key, [default]) or [default])[0]
  width = "width_norm"
  clauses: List[str] = []
  values: List[Any] = []
  if first("char"):
    clauses.append("m.class LIKE ?")
    values.append(f"%{first('char')}%")
  if first("class"):
    clauses.append("m.class = ?")
    values.append(first("class"))
  low, high = parse_pages(first("pages"))
  if low is not None:
    clauses.append("m.page BETWEEN ? AND ?")
    values.extend([low, high])
  minimum = float_or_none(first("min"))
  if minimum is not None:
    clauses.append(f"m.{width} >= ?")
    values.append(minimum)
  maximum = float_or_none(first("max"))
  if maximum is not None:
    clauses.append(f"m.{width} <= ?")
    values.append(maximum)
  return (" AND ".join(clauses) or "1"), values, width


def class_forms(connection: sqlite3.Connection) -> Dict[str, List[str]]:
  """The joining forms each shape class is actually attested in."""
  found: Dict[str, List[str]] = {}
  for name, form in connection.execute(
      "SELECT DISTINCT char_class, form FROM letters "
      "WHERE char_class != '' AND form != '' ORDER BY char_class, form"):
    found.setdefault(name, []).append(form)
  return found


def statistics(values: List[float], buckets: int = 24) -> dict:
  numbers = sorted(v for v in values if v is not None)
  if not numbers:
    return {"median": None, "mad": None, "min": None, "max": None,
            "histogram": [], "peak": 0}
  middle = numbers[len(numbers) // 2]
  deviations = sorted(abs(v - middle) for v in numbers)
  low, high = numbers[0], numbers[-1]
  histogram = [0] * buckets
  span = high - low
  for value in numbers:
    index = 0 if span <= 0 else min(buckets - 1, int(buckets * (value - low) / span))
    histogram[index] += 1
  return {"median": middle, "mad": deviations[len(deviations) // 2],
          "min": low, "max": high, "histogram": histogram,
          "peak": max(histogram)}


# ---------------------------------------------------------------------------
# server
# ---------------------------------------------------------------------------

class Corpus:
  """Per-thread connections to the read-only index."""

  def __init__(self, database: Path, analysis: Path, viewer_port: int):
    self.database = database
    self.analysis = analysis.resolve()
    self.viewer_port = viewer_port
    self.local = threading.local()

  def connection(self) -> sqlite3.Connection:
    existing = getattr(self.local, "connection", None)
    if existing is not None:
      return existing
    connection = sqlite3.connect(
        f"file:{self.database}?mode=ro", uri=True, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    self.local.connection = connection
    return connection

  def has_traced_kashida(self) -> bool:
    """Whether this index carries `letter_shape.kashida_path_d`."""
    known = getattr(self, "_traced", None)
    if known is None:
      known = any(row[1] == "kashida_path_d" for row in
                  self.connection().execute("PRAGMA table_info(letter_shape)"))
      self._traced = known
    return known

  def viewer_url(self, page: int, word_key: str) -> str:
    """Deep link into review_paw_segmentation.py.

    The viewer serves any page in its analysis directory, so the link names
    both; it used to carry the word alone and land on whichever page the
    viewer had been started with.
    """
    return (f"http://127.0.0.1:{self.viewer_port}/"
            f"?page={int(page)}&word={urllib.parse.quote(word_key)}")


def search(corpus: Corpus, query: Dict[str, List[str]]) -> dict:
  first = lambda key, default="": (query.get(key, [default]) or [default])[0]
  kind = first("kind", "letter")
  # "max" -- or any non-positive number -- means every match. The cap is the
  # letters table's own size rather than a round number, so asking for
  # everything cannot ask for more than exists.
  requested_limit = first("limit", "400") or "400"
  if requested_limit == "max":
    limit = MAX_ROWS
  else:
    try:
      limit = int(requested_limit)
    except ValueError:
      limit = 400
    limit = MAX_ROWS if limit <= 0 else min(limit, MAX_ROWS)
  offset = max(0, int(first("offset", "0") or 0))
  connection = corpus.connection()

  requested = first("sort", "width_desc")
  if kind == "mark":
    where, values, width = mark_query(query)
    sort_column, direction = MARK_SORTS.get(requested, MARK_SORTS["width_desc"])
    source = ("FROM marks m "
              "LEFT JOIN words w ON w.word_id = m.word_id")
    base = row_base = f"{source} WHERE {where}"
    columns = (f"m.mark_id AS id, m.class, m.page, m.word_id, "
               f"m.{width} AS width, m.height_norm AS height, m.width_norm, "
               f"m.area_norm, m.confidence, "
               f"w.word_key, w.text AS word_text")
    order = f"m.{sort_column}"
  else:
    # Measuring a connector implies having one. A card reading "kashida 0.00"
    # in a list sorted by kashida is not a result, so the metric carries the
    # filter -- which is all the old "kashidas" kind ever did, minus the
    # pretence that a connector is a kind of thing.
    metric = first("metric", "advance")
    where, values, width, joins, aliases = letter_query(query)
    if metric in KASHIDA_METRICS:
      where = f"({where}) AND ({kashida_clause(metric)})"
    if requested in METRIC_SORTS:
      metric, direction = METRIC_SORTS[requested]
      sort_column, sort_sql = None, metric_sql(metric, aliases)
    else:
      sort_column, direction = SORTS.get(requested, SORTS["width_desc"])
      sort_sql = width
    # "width" on a letter is whichever width the filters are already speaking
    # in. Sorting by advance while filtering on the spine extent -- or, in the
    # kashida kind, on the connector -- orders the result by a column the page
    # is not showing.
    if requested.startswith("width_"):
      sort_column = None      # sort on the expression, not on a column name
    # The group joins carry no parameters of their own, so they can be spliced
    # in without disturbing the order the WHERE clause binds in.
    source = ("FROM letters l" + joins +
              " LEFT JOIN words w ON w.word_id = l.word_id "
              "LEFT JOIN paws pw ON pw.paw_id = l.paw_id "
              "LEFT JOIN pages pg ON pg.page = l.page")
    base = f"{source} WHERE {where}"
    # Only the row query needs these: they feed one output column, and putting
    # them in the filter would make the count and the histogram pay for them
    # too -- 165 ms to 216 ms on an unfiltered corpus-wide count.
    row_source = (f"{source}"
                  + ("" if SHAPE_JOIN in source else SHAPE_JOIN)
                  + " LEFT JOIN letter_shape lps ON lps.letter_id = lp.letter_id")
    row_base = f"{row_source} WHERE {where}"
    # A letter the segmenter could not cut has no width of its own -- see
    # build_corpus, which now writes NULL rather than the whole PAW's span.
    # The PAW's span is still worth showing next to it, because otherwise the
    # card carries no measurement at all and the failures become unbrowsable.
    # It is a separate field so it can never be read as the letter's width.
    # prev_char/next_char are the joining neighbours in logical order, so in
    # the image next_char is the letter on the *left* -- the one this letter's
    # kashida runs into. Carried on every letter row; the kashida view is what
    # actually asks "which pair stretches".
    # One column per group member, so the card can draw the whole run: its
    # outline, and the advance that places the next member beside it. The
    # advance is entry-to-entry, so the offsets are a running sum of them and
    # nothing has to be registered by eye.
    group = "".join(
        f", {alias}.letter_id AS g{index}_id, "
        f"{alias}.adv_w_norm AS g{index}_adv, {alias}.char AS g{index}_char, "
        f"{alias}.next_char AS g{index}_next"
        for index, alias in enumerate(aliases))
    columns = (f"l.letter_id AS id, l.char, l.next_char, l.prev_char, "
               f"l.char_class, l.form, l.page, "
               f"l.word_id, {width} AS width, "
               f"{group_width_sql('height_norm', aliases)} AS height, "
               f"l.adv_w_norm, l.bbox_w_norm, l.extent_norm, l.kashida_norm, "
               f"l.confidence, "
               f"l.low_confidence, l.ok, "
               f"w.word_key, w.text AS word_text, "
               f"pw.letters AS paw_letters, pw.merged AS paw_merged, "
               f"pw.connected_to_previous_word AS paw_joined, "
               f"pw.spine_length / pg.nominal_stroke_px AS paw_norm, "
               f"{metric_sql('kashida', aliases)} AS kashida_all, "
               f"{metric_sql('kashidaLeft', aliases)} AS kashida_left, "
               f"{metric_sql('kashidaRight', aliases)} AS kashida_right, "
               f"l.logical_group AS lig, "
               f"l.letter_index AS letter_index, "
               f"l.group_size AS group_size, "
               f"{group_start_sql()} AS group_start, "
               f"lp.letter_id AS prev_id, "
               f"lp.letter_index AS prev_index, "
               f"lp.group_size AS prev_group_size, "
               f"ABS(lps.entry_x - ls.entry_x) / ls.norm_scale AS prev_adv"
               + (f", {group_form_sql(len(aliases))} AS group_form"
                  if len(aliases) > 1 else "") + group)
    order = sort_sql if sort_column is None else f"l.{sort_column}"

  ordering = f"{order} IS NULL, {order} {direction}"
  total = connection.execute(f"SELECT COUNT(*) {base}", values).fetchone()[0]

  rows = connection.execute(
      f"SELECT {columns} {row_base} ORDER BY {ordering} LIMIT ? OFFSET ?",
      values + [limit, offset]).fetchall()
  window = (offset, min(offset + limit, total))

  distribution = connection.execute(
      f"SELECT {'m.' + width if kind == 'mark' else width} {base}",
      values).fetchall()

  records = []
  for row in rows:
    record = dict(row)
    record["viewer_url"] = corpus.viewer_url(record["page"],
                                             record.get("word_key") or "")
    records.append(record)
  return {"total": total, "rows": records, "from": window[0], "to": window[1],
          "stats": statistics([row[0] for row in distribution])}


def shapes(corpus: Corpus, ids: List[str]) -> dict:
  if not ids:
    return {}
  connection = corpus.connection()
  # An index built before the connector was traced has no such column, and the
  # browser already falls back to deriving the band. Ask for it only when it
  # is there, so an older index still serves.
  traced = "kashida_path_d, " if corpus.has_traced_kashida() else ""
  out: Dict[str, dict] = {}
  for start in range(0, len(ids), 500):
    chunk = ids[start:start + 500]
    rows = connection.execute(
        f"SELECT letter_id, path_d, {traced}view_x, view_y, view_w, "
        "view_h, norm_scale "
        f"FROM letter_shape WHERE letter_id IN ({','.join('?' * len(chunk))})",
        chunk).fetchall()
    for row in rows:
      if row["path_d"] and row["view_w"]:
        out[row["letter_id"]] = dict(row)
  return out


# Two ink runs in neighbouring columns continue the same stroke if they still
# overlap once one is grown by this much.
CONNECTOR_TOLERANCE_W = 0.35
# How thick the band may get before it is treated as having merged into the
# body. A connector is a stroke of about the qalam's thickness and thickens as
# it approaches the letter, so this has room above 1; a bowl or an upright
# crossing it is several times taller and gets trimmed back to a stroke.
CONNECTOR_MAX_THICKNESS_W = 2.0
# A band carries one point per pixel column, which is far more than a clip
# outline needs. Reduced by taking each window's outermost top and bottom, so
# the simplified polygon contains the exact one and can never shave ink off the
# stroke it is clipping.
CONNECTOR_BAND_POINTS = 80


def connector_band(row) -> Optional[List[Tuple[float, float, float]]]:
  """Follow the connector's own ink through the letter mask, column by column.

  The kashida is painted by clipping the letter's outline, and clipping it to
  the connector's *span* also catches whatever body ink leans into that span --
  a Lam's upright, a Qaf's bowl. 60% of connectors have at least one column
  where that intruding ink is a separate vertical run, so it can be excluded
  without any new data: walk in from the far end, where the ink is only ever
  connector, and at each column keep the run continuing the stroke.

  Where body and connector genuinely touch there is one run holding both, and
  no amount of looking at this mask separates them. The band is capped at the
  previous column's extent grown by the tolerance so that case leaks a stroke
  width rather than a whole letter. Returns (x, y_top, y_bottom) per column in
  the normalized frame, or None when there is nothing to follow.
  """
  scale = row["norm_scale"]
  if not scale or row["mask_rle"] is None or row["entry_x"] is None:
    return None
  width, height = int(row["mask_w"]), int(row["mask_h"])
  span = row["kashida_px"]
  if not span or span <= 1 or width < 2 or height < 1:
    return None
  mask = decode_mask_rle(row["mask_rle"], width, height)
  # Word-space x decreases along the writing, so the connector -- the end of
  # the span -- is the left edge of the crop, and the far end is column 0.
  columns = min(width, max(2, int(round(float(span)))))
  tolerance = CONNECTOR_TOLERANCE_W * float(scale)
  cap = CONNECTOR_MAX_THICKNESS_W * float(scale)
  band: List[Tuple[float, float, float]] = []
  low = high = None
  for column in range(columns):
    ys = np.flatnonzero(mask[:, column])
    if ys.size == 0:
      if band:
        break
      continue
    edges = np.flatnonzero(np.diff(ys) > 1)
    starts = np.concatenate(([0], edges + 1))
    ends = np.concatenate((edges, [ys.size - 1]))
    runs = [(float(ys[a]), float(ys[b])) for a, b in zip(starts, ends)]
    if low is None:
      # Seed on the run holding the exit point: that is where the connector
      # ends, by definition, so it cannot be body ink.
      target = (float(row["exit_y"]) - float(row["mask_y"])
                if row["exit_y"] is not None else height / 2.0)
      chosen = min(runs, key=lambda r: 0.0 if r[0] <= target <= r[1]
                   else min(abs(r[0] - target), abs(r[1] - target)))
    else:
      reachable = [r for r in runs
                   if r[0] <= high + tolerance and r[1] >= low - tolerance]
      if not reachable:
        break
      centre = 0.5 * (low + high)
      chosen = min(reachable, key=lambda r: abs(0.5 * (r[0] + r[1]) - centre))
      # One run holding both body and connector cannot be split by looking at
      # this mask, so keep a stroke's worth of it around where the connector
      # was: a merge then costs a stroke width, not the whole letter.
      if chosen[1] - chosen[0] > cap:
        chosen = (max(chosen[0], centre - 0.5 * cap),
                  min(chosen[1], centre + 0.5 * cap))
    low, high = chosen
    x_word = float(row["mask_x"]) + column
    top_y = float(row["mask_y"]) + low - 0.5
    bottom_y = float(row["mask_y"]) + high + 0.5
    band.append((
        (float(row["entry_x"]) - x_word) / scale,
        (float(row["baseline_y"]) - top_y) / scale,
        (float(row["baseline_y"]) - bottom_y) / scale))
  return band if len(band) >= 2 else None


def connectors(corpus: Corpus, ids: List[str]) -> dict:
  """Connector clip bands for the kashida view, keyed by letter id."""
  if not ids:
    return {}
  connection = corpus.connection()
  out: Dict[str, list] = {}
  for start in range(0, len(ids), 200):
    chunk = ids[start:start + 200]
    rows = connection.execute(
        "SELECT s.letter_id, s.mask_rle, s.mask_w, s.mask_h, s.mask_x, "
        "s.mask_y, s.entry_x, s.exit_y, s.baseline_y, s.norm_scale, "
        # Only the member that owns the group's connector may be given a band:
        # the others hold none of its ink, so tracing one across their mask
        # paints a stroke that is not theirs.
        f"CASE WHEN {owns_connector('l')} THEN l.kashida_px END AS kashida_px "
        "FROM letter_shape s JOIN letters l "
        f"ON l.letter_id = s.letter_id WHERE s.letter_id IN "
        f"({','.join('?' * len(chunk))})", chunk).fetchall()
    for row in rows:
      band = connector_band(row)
      if band:
        out[row["letter_id"]] = simplify_band(band)
  return out


def simplify_band(band: List[Tuple[float, float, float]]) -> list:
  """Fewer points, always outside the original band."""
  stride = max(1, -(-len(band) // CONNECTOR_BAND_POINTS))
  if stride == 1:
    return [[round(v, 3) for v in point] for point in band]
  out = []
  for start in range(0, len(band), stride):
    window = band[start:start + stride]
    out.append([round(window[0][0], 3),
                round(max(point[1] for point in window), 3),
                round(min(point[2] for point in window), 3)])
  last = band[-1]
  if out[-1][0] != round(last[0], 3):
    out.append([round(last[0], 3), round(last[1], 3), round(last[2], 3)])
  return out


def word_layout(corpus: Corpus, word_ids: List[str]) -> dict:
  """Every letter of each word, with the entry point that places it.

  All letters of one word are measured in that word's pixel space and share a
  baseline, so one x offset per letter is the whole registration: a caller
  picks whichever letter it is drawing around, and shifts the others by the
  difference of their entry points. Returned per word rather than per match so
  that several matches in one word cost one lookup.
  """
  if not word_ids:
    return {}
  connection = corpus.connection()
  out: Dict[str, dict] = {}
  for start in range(0, len(word_ids), 200):
    chunk = word_ids[start:start + 200]
    rows = connection.execute(
        "SELECT l.letter_id, l.word_id, s.entry_x, s.norm_scale "
        "FROM letters l JOIN letter_shape s ON s.letter_id = l.letter_id "
        f"WHERE l.word_id IN ({','.join('?' * len(chunk))}) "
        "AND s.entry_x IS NOT NULL AND s.norm_scale IS NOT NULL "
        "ORDER BY l.paw_id, l.letter_index", chunk).fetchall()
    for row in rows:
      record = out.setdefault(row["word_id"],
                              {"scale": row["norm_scale"], "letters": []})
      record["letters"].append([row["letter_id"], round(row["entry_x"], 2)])
  return out


def image_path(corpus: Corpus, kind: str, identifier: str) -> Optional[Path]:
  """Resolve an instance id to an asset, without ever taking a path from the client."""
  page_part = identifier.split(":", 1)[0]
  if not page_part.startswith("p"):
    return None
  page = int(page_part[1:])
  page_dir = corpus.analysis / f"page{page:03d}"
  parts = identifier.split(":")
  if len(parts) < 2:
    return None
  word_key = parts[1]
  if kind == "paw" and len(parts) > 2 and parts[2].startswith("paw"):
    index = int(parts[2][3:])
    return (page_dir / "subwordimgs" / f"{word_key}_paw{index:02d}.png").resolve()
  return (page_dir / "wordimgs" / f"{word_key}.png").resolve()


def _mark_stencil(corpus: Corpus, row, cv2, np):
  """This mark's own pixels as a (h, w) boolean, in the row's bounding box.

  The mask PNG is drawn on the subword canvas, not the word one, and the two
  origins differ by the PAW's position -- which is not recoverable from the
  index, because ``paws.x`` is a page coordinate while ``marks.x`` is relative
  to the word. What does line up is the ink itself: the mask's own ink bounding
  box has the same size as the row's, so the two are aligned by bounding box
  rather than by canvas origin. Returns None when they disagree, so the caller
  can fall back to the plain rectangle rather than paint a shifted stencil.
  """
  relative = row["mask_path"] if "mask_path" in row.keys() else ""
  if not relative:
    return None
  path = (corpus.analysis / relative).resolve()
  if corpus.analysis not in path.parents or not path.is_file():
    return None
  mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
  if mask is None:
    return None
  ys, xs = np.nonzero(mask)
  if not xs.size:
    return None
  x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
  if (x1 - x0 + 1, y1 - y0 + 1) != (int(row["w"]), int(row["h"])):
    return None
  return mask[y0:y1 + 1, x0:x1 + 1] > 0


def mark_crop(corpus: Corpus, identifier: str, pad: int = 12) -> Optional[bytes]:
  """A PNG of one mark, cut from its word image and masked to its own ink.

  The bounding box alone is not the mark: an over-long fatha can span 11 stroke
  widths and 60% of the word, so its box encloses the shadda and the dot that
  sit under it, and a plain rectangular cut shows all three. The analyzer's
  per-instance mask says which pixels are actually this mark's, so everything
  else in the rectangle is dropped to the background.

  The rectangle itself is unchanged -- box plus padding -- so framing and scale
  stay as they were, and a mark whose mask is missing or misaligned still gets
  the old unmasked crop.
  """
  row = corpus.connection().execute(
      "SELECT x, y, w, h, mask_path FROM marks WHERE mark_id = ?",
      (identifier,)).fetchone()
  if row is None:
    return None
  path = image_path(corpus, "word", identifier)
  if path is None or not path.is_file() or corpus.analysis not in path.parents:
    return None
  import cv2                                   # only needed for this endpoint
  import numpy as np
  image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
  if image is None:
    return None
  height, width = image.shape[:2]
  stencil = _mark_stencil(corpus, row, cv2, np)
  if stencil is not None:
    # Ink is bright on a dark ground, so clearing to zero is clearing to the
    # background the card already letterboxes against.
    box = np.zeros(image.shape[:2], dtype=bool)
    top, left = max(0, int(row["y"])), max(0, int(row["x"]))
    piece = stencil[:height - top, :width - left]
    box[top:top + piece.shape[0], left:left + piece.shape[1]] = piece
    image = np.where(box[..., None] if image.ndim == 3 else box, image, 0)
  x0 = max(0, int(row["x"]) - pad)
  y0 = max(0, int(row["y"]) - pad)
  x1 = min(width, int(row["x"]) + int(row["w"]) + pad)
  y1 = min(height, int(row["y"]) + int(row["h"]) + pad)
  if x1 <= x0 or y1 <= y0:
    return None
  ok, buffer = cv2.imencode(".png", image[y0:y1, x0:x1])
  return buffer.tobytes() if ok else None


class Handler(BaseHTTPRequestHandler):
  corpus: Corpus = None                       # set on the class before serving

  def log_message(self, *_args):              # keep the console readable
    pass

  def send_json(self, payload, status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False, default=float).encode()
    self.send_response(status)
    self.send_header("Content-Type", "application/json; charset=utf-8")
    self.send_header("Content-Length", str(len(body)))
    self.end_headers()
    self.wfile.write(body)

  def send_bytes(self, body: bytes, content_type: str) -> None:
    self.send_response(200)
    self.send_header("Content-Type", content_type)
    self.send_header("Content-Length", str(len(body)))
    self.end_headers()
    self.wfile.write(body)

  def do_GET(self) -> None:
    parsed = urllib.parse.urlparse(self.path)
    query = urllib.parse.parse_qs(parsed.query)
    if parsed.path == "/":
      self.send_bytes(HTML.encode(), "text/html; charset=utf-8")
      return
    if parsed.path == "/api/facets":
      connection = self.corpus.connection()
      self.send_json({
          "classes": [row[0] for row in connection.execute(
              "SELECT DISTINCT char_class FROM letters "
              "WHERE char_class != '' ORDER BY char_class")],
          "mark_classes": [row[0] for row in connection.execute(
              "SELECT class FROM marks WHERE class IS NOT NULL AND class != '' "
              "GROUP BY class ORDER BY COUNT(*) DESC")],
          # Which forms a class actually occurs in, and where it went when it
          # does not occur in one. Selecting "beh" and "medi" is empty by
          # construction -- a medial beh is a senna tooth -- and without this
          # the GUI can only say "no matches", which reads as a bug.
          "class_forms": class_forms(connection),
          "folds_to": letter_classes.folds_to(),
          "pages": [row[0] for row in connection.execute(
              "SELECT page FROM pages ORDER BY page")],
      })
      return
    if parsed.path == "/api/search":
      try:
        self.send_json(search(self.corpus, query))
      except ValueError as error:
        self.send_json({"error": str(error)}, 400)
      return
    if parsed.path == "/api/shapes":
      ids = [i for i in (query.get("ids", [""])[0]).split(",") if i]
      self.send_json(shapes(self.corpus, ids))
      return
    if parsed.path == "/api/connectors":
      ids = [i for i in (query.get("ids", [""])[0]).split(",") if i]
      self.send_json(connectors(self.corpus, ids))
      return
    if parsed.path == "/api/words":
      ids = [i for i in (query.get("ids", [""])[0]).split(",") if i]
      self.send_json(word_layout(self.corpus, ids))
      return
    if parsed.path == "/image":
      kind = query.get("kind", ["word"])[0]
      identifier = query.get("id", [""])[0]
      if kind == "mark":
        try:
          body = mark_crop(self.corpus, identifier)
        except (ValueError, IndexError):
          body = None
        if body is None:
          self.send_error(404)
        else:
          self.send_bytes(body, "image/png")
        return
      try:
        path = image_path(self.corpus, kind, identifier)
      except (ValueError, IndexError):
        path = None
      if (path is None or not path.is_file()
          or self.corpus.analysis not in path.parents):
        self.send_error(404)
        return
      self.send_bytes(path.read_bytes(),
                      mimetypes.guess_type(path)[0] or "image/png")
      return
    self.send_error(404)


def main() -> int:
  parser = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--db", required=True, help="corpus sqlite path")
  parser.add_argument("--analysis", required=True, help="analysis output root")
  parser.add_argument("--port", type=int, default=8880)
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--viewer-port", type=int, default=8879,
                      help="port of review_paw_segmentation.py, for deep links")
  arguments = parser.parse_args()

  database = Path(arguments.db).resolve()
  if not database.is_file():
    parser.error(f"no corpus database at {database}")
  Handler.corpus = Corpus(database, Path(arguments.analysis),
                          arguments.viewer_port)
  server = ThreadingHTTPServer((arguments.host, arguments.port), Handler)
  print(f"corpus  {database}")
  print(f"serving http://{arguments.host}:{arguments.port}/")
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    pass
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
