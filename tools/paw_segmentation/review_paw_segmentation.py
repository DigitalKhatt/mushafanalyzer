#!/usr/bin/env python3
"""Interactive browser diagnostics for PAW letter segmentation.

The viewer reruns one selected word in memory so every diagnostic remains an
independent layer instead of being flattened into the exported overlay PNG.

Example::

    uv run tools/paw_segmentation/review_paw_segmentation.py \
        --analysis output/oldmedina2-analysis --page 3 --port 8879

Then open http://127.0.0.1:8879/. Press ``r`` or use Reload after editing the
template JSON. No analysis masks or segmentation output files are modified.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import sys
import threading
from typing import Sequence
import urllib.parse

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import segment_paw_letters as seg  # noqa: E402


HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PAW segmentation diagnostics</title>
<style>
:root{color-scheme:dark;--bg:#0b1016;--panel:#141d26;--panel2:#1c2935;--line:#344657;
 --text:#edf4fa;--muted:#94a8b9;--accent:#72c7ff;--warn:#ffc267;--bad:#ff6c76;--good:#65dda6}
*{box-sizing:border-box}html,body{height:100%}body{margin:0;background:var(--bg);color:var(--text);
 font:13px system-ui,-apple-system,sans-serif;overflow:hidden}button,input,select{font:inherit;color:inherit;
 background:var(--panel2);border:1px solid var(--line);border-radius:6px;padding:7px 9px}button{cursor:pointer}
button:hover{border-color:var(--accent)}header{height:58px;display:flex;align-items:center;gap:8px;padding:9px 13px;
 border-bottom:1px solid var(--line);background:#111a22}h1{font-size:17px;margin:0 14px 0 0;white-space:nowrap}
#wordSelect{min-width:250px}.grow{flex:1}.status{color:var(--muted);font-size:12px}.app{height:calc(100% - 58px);
 display:grid;grid-template-columns:245px minmax(0,1fr) 335px}.side{overflow:auto;background:var(--panel);
 padding:12px;border-right:1px solid var(--line)}.right{border-left:1px solid var(--line);border-right:0}
.section{margin:0 0 15px}.section h2{font-size:11px;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);
 margin:0 0 7px}.toggle{display:flex;align-items:center;gap:8px;padding:5px 6px;border-radius:5px;cursor:pointer}
.toggle:hover{background:var(--panel2)}.toggle input{accent-color:var(--accent)}.swatch{width:13px;height:4px;border-radius:2px}
.stage{min-width:0;min-height:0;display:grid;grid-template-rows:auto minmax(0,1fr);background:#070a0d}
.stagebar{display:flex;align-items:center;gap:9px;padding:8px 11px;border-bottom:1px solid var(--line);background:#101820}
.stagebar input{width:140px}.viewport{overflow:auto;position:relative;padding:20px;background-color:#080b0f;
 background-image:linear-gradient(45deg,#0d1218 25%,transparent 25%),linear-gradient(-45deg,#0d1218 25%,transparent 25%),
 linear-gradient(45deg,transparent 75%,#0d1218 75%),linear-gradient(-45deg,transparent 75%,#0d1218 75%);
 background-size:24px 24px;background-position:0 0,0 12px,12px -12px,-12px 0}
#canvas{display:block;background:#000;box-shadow:0 0 0 1px #283642,0 8px 35px #000;cursor:crosshair}
.card{border:1px solid var(--line);background:var(--panel2);border-radius:7px;padding:9px;margin-bottom:8px}
.arabic{direction:rtl;font-size:25px}.mono{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;color:#c7d5df}
.muted{color:var(--muted)}.pill{display:inline-block;border-radius:11px;background:#2b3d4c;padding:2px 7px;margin:2px;
 font-size:11px}.good{color:var(--good)}.bad{color:var(--bad)}.warn{color:var(--warn)}table{width:100%;border-collapse:collapse}
th,td{text-align:left;padding:5px;border-bottom:1px solid #30404e;vertical-align:top}th{color:var(--muted);font-size:11px}
.eventKinds{display:grid;grid-template-columns:1fr 1fr;gap:2px}.eventKinds .toggle{font-size:11px;padding:4px}
#eventDetail{white-space:pre-wrap;line-height:1.45}.legend{display:flex;gap:10px;flex-wrap:wrap;color:var(--muted)}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px}.empty{padding:20px;color:var(--muted)}
.traceFail{border-color:#85424a}.traceIncoming{border-color:#49677d}.tracePath{overflow-wrap:anywhere;line-height:1.5}
.tracePathRow{padding:7px 0;border-top:1px solid #30404e}.tracePathRow:first-of-type{border-top:0}.traceButton{width:100%;margin-bottom:8px}
@media(max-width:1050px){.app{grid-template-columns:210px minmax(0,1fr)}.right{display:none}}
</style>
</head>
<body>
<header><h1>PAW diagnostics</h1><select id="pageSelect"></select>
<button id="prev">←</button><button id="next">→</button>
<select id="wordSelect"></select><select id="pawSelect"></select><button id="reload">Reload</button>
<span class="grow"></span><span id="status" class="status">Loading…</span></header>
<div class="app">
 <aside class="side">
  <section class="section"><h2>Ink ownership</h2><div id="rasterToggles"></div></section>
  <section class="section"><h2>Skeleton geometry</h2><div id="geometryToggles"></div></section>
  <section class="section"><h2>Decisions</h2><div id="decisionToggles"></div></section>
  <section class="section"><h2>Event kinds</h2><div id="eventKinds" class="eventKinds"></div></section>
 </aside>
 <main class="stage"><div class="stagebar"><button id="fit">Fit</button><label>Zoom <input id="zoom" type="range" min="10" max="180" value="45"></label>
   <span id="dimensions" class="muted"></span><span class="grow"></span><div class="legend"><span><i class="dot" style="background:#fff"></i>owned event</span><span><i class="dot" style="background:#aaa"></i>context event</span><span><i class="dot" style="background:#ff00ff"></i>branch</span></div></div>
  <div id="viewport" class="viewport"><canvas id="canvas"></canvas></div></main>
 <aside class="side right"><section id="summary" class="section"></section><section id="matches" class="section"></section><section id="alignmentTrace" class="section"></section>
  <section class="section"><h2>Selected event</h2><div id="eventDetail" class="card muted">Click an event marker.</div></section></aside>
</div>
<script>
const layerDefs={
 ink:['Base ink',true,'#fff'],letters:['Letter bodies',true,'#78dc78'],joinsInk:['Kashida pixels',true,'#d3d3d3'],
 unassigned:['Unassigned pixels',true,'#50b4b4'],medial:['Final medial graph',true,'#3c3cc8'],spine:['Main spine',true,'#78e6ff'],
 graphBranches:['Graph branch paths',false,'#ff9b45'],prunedBranches:['Pruned branch paths',true,'#ff00ff'],
 baseline:['Baseline',true,'#00c800'],loops:['Loop evidence',true,'#ffdb54'],events:['Event markers',true,'#ff00ff'],
 matched:['Matched events',true,'#fff'],candidates:['Candidates',false,'#00dcff'],cuts:['Chosen cuts',true,'#ff3030'],
 joinBounds:['Join endpoints',true,'#ff9bff'],labels:['Letter labels',true,'#fff']};
const groups={rasterToggles:['ink','letters','joinsInk','unassigned'],geometryToggles:['medial','spine','graphBranches','prunedBranches','baseline','loops'],decisionToggles:['events','matched','candidates','cuts','joinBounds','labels']};
const state={boot:null,data:null,paw:0,wordIndex:0,layers:{},images:{},zoom:.45,eventKinds:new Set(),eventKindPrefs:{},selectedEvent:null,eventsAtPoint:[],traces:{},traceLoading:false};
const $=id=>document.getElementById(id);const esc=s=>(s??'').toString().replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const eventColors={'x-left':'#28b4ff','x-right':'#ffb428',peak:'#50ff50',valley:'#8b5cf6',continuation:'#bebebe',branch:'#ff00ff','high-branch':'#00ffff','double-high-branch':'#ffc800','zigzag-branch':'#ffff00','terminal-ascent':'#ffdc50',start:'#c8ffc8',end:'#c8c8ff','start-height':'#3cc896','end-height':'#3c96c8'};
// The two spine ends are always present, so they are the one kind of event
// that is there even when nothing happened. Drawn as squares rather than
// diamonds to say so; see endRadius for the overlap.
const endKinds=new Set(['start','end','start-height','end-height']);
function status(t){$('status').textContent=t}function toggleHtml(key){const [label,on,color]=layerDefs[key];if(!(key in state.layers))state.layers[key]=on;return `<label class="toggle"><input type="checkbox" data-layer="${key}" ${state.layers[key]?'checked':''}><i class="swatch" style="background:${color}"></i>${label}</label>`}
function initControls(){for(const [id,keys] of Object.entries(groups))$(id).innerHTML=keys.map(toggleHtml).join('');document.querySelectorAll('[data-layer]').forEach(e=>e.onchange=()=>{state.layers[e.dataset.layer]=e.checked;draw()})}
function setLayer(key,on){state.layers[key]=on;const input=document.querySelector(`[data-layer="${key}"]`);if(input)input.checked=on}
async function getJson(url){const r=await fetch(url);if(!r.ok)throw Error((await r.json()).error||r.statusText);return r.json()}
function image(url){return new Promise((resolve,reject)=>{const im=new Image;im.onload=()=>resolve(im);im.onerror=reject;im.src=url})}
// The page is part of the address, so a link from another tool can name both
// the page and the word. Every request carries it; state.page is the truth.
function pageParam(){return `page=${state.page}`}
async function boot(){initControls();
 const url=new URLSearchParams(location.search);
 state.page=+url.get('page')||0;
 await loadPage(url.get('word')||'')}
async function loadPage(wantedWord){
 status(`Loading page ${state.page||''}…`);
 state.boot=await getJson('/api/bootstrap'+(state.page?`?${pageParam()}`:''));
 state.page=state.boot.page;
 $('pageSelect').innerHTML=(state.boot.pages||[state.page]).map(p=>`<option value="${p}">page ${p}</option>`).join('');
 $('pageSelect').value=state.page;
 $('wordSelect').innerHTML=state.boot.words.map((w,i)=>`<option value="${i}">${esc(w.wordId)} · ${esc(w.text)}</option>`).join('');
 const i=state.boot.words.findIndex(w=>w.wordId===wantedWord);
 await selectWord(i>=0?i:0)}
async function selectPage(page){state.page=+page;try{await loadPage('')}catch(e){status(e.message)}}
async function selectWord(index,force=false){state.wordIndex=Math.max(0,Math.min(state.boot.words.length-1,index));$('wordSelect').value=state.wordIndex;const word=state.boot.words[state.wordIndex];status(`Analyzing ${word.wordId}…`);
 const q=`/api/analyze?${pageParam()}&word=${encodeURIComponent(word.wordId)}${force?'&reload=1':''}`;try{state.data=await getJson(q);state.traces={};const firstWithEvents=state.data.paws.findIndex(p=>(p.skeletonEvents||[]).length);state.paw=firstWithEvents>=0?firstWithEvents:0;state.selectedEvent=null;state.eventsAtPoint=[];populatePaws();await loadImages();renderPanels();fit();status(`p${state.page} · ${word.wordId} · ${state.data.elapsedMs} ms`);history.replaceState(null,'',`?${pageParam()}&word=`+word.wordId)}catch(e){status(e.message);$('summary').innerHTML=`<div class="bad">${esc(e.message)}</div>`}}
function populatePaws(){$('pawSelect').innerHTML=state.data.paws.map((p,i)=>`<option value="${i}">PAW ${i} · ${esc(p.letters.join(''))}</option>`).join('');$('pawSelect').value=state.paw;buildEventKinds()}
async function loadImages(){const id=state.data.wordId,stamp=state.data.cacheToken,p=pageParam();const urls={ink:`/image?${p}&word=${id}&kind=ink&v=${stamp}`,letters:`/image?${p}&word=${id}&kind=letters&v=${stamp}`,joinsInk:`/image?${p}&word=${id}&kind=joins&v=${stamp}`,unassigned:`/image?${p}&word=${id}&kind=unassigned&v=${stamp}`};state.images={};await Promise.all(Object.entries(urls).map(async([k,u])=>state.images[k]=await image(u)))}
function paw(){return state.data.paws[state.paw]}function buildEventKinds(){const counts=new Map;for(const e of paw().skeletonEvents)counts.set(e.kind,(counts.get(e.kind)||0)+1);const kinds=[...counts.keys()];state.eventKinds=new Set(kinds.filter(k=>state.eventKindPrefs[k]!==false));if(!kinds.length){$('eventKinds').innerHTML=`<div class="empty">No skeleton events in PAW ${state.paw} · ${esc(paw().letters.join(''))}. Select another PAW.</div>`;return}$('eventKinds').innerHTML=kinds.map(k=>`<label class="toggle" title="Show or hide ${counts.get(k)} ${k} event marker(s)"><input type="checkbox" data-kind="${k}" ${state.eventKinds.has(k)?'checked':''}><i class="dot" style="background:${eventColors[k]||'#ddd'}"></i><span>${k} <span class="muted">(${counts.get(k)})</span></span></label>`).join('');document.querySelectorAll('[data-kind]').forEach(e=>e.onchange=()=>{state.eventKindPrefs[e.dataset.kind]=e.checked;if(e.checked){state.eventKinds.add(e.dataset.kind);setLayer('events',true)}else state.eventKinds.delete(e.dataset.kind);draw()})}
function renderPanels(){const p=paw(),cls=p.ok?'good':'bad';$('dimensions').textContent=`${state.data.width} × ${state.data.height}px · SW ${p.strokeWidth}`;
 $('summary').innerHTML=`<h2>Word / PAW</h2><div class="card"><div class="arabic">${esc(state.data.text)}</div><div class="mono">${esc(state.data.wordId)} · PAW ${p.pawIndex}</div><p class="${cls}">${p.ok?'segmented':esc(p.reason||'failed')}</p><div>${p.letters.map((l,i)=>`<span class="pill">${i} ${esc(l)} · ${esc(p.forms[i])}</span>`).join('')}</div><p class="mono">confidence ${p.confidence}<br>margin ${p.margin}<br>loops ${p.loops}<br>spine ${p.spineLength}px</p></div>`;
 $('matches').innerHTML='<h2>Template matches</h2>'+((p.diagnostic.groups||[]).map(g=>`<div class="card"><b>${esc(g.template||'untemplated')}</b><div class="mono">letters ${g.letters.join('–')} · s ${g.span.join('…')}</div><div>branches <b>${g.branchCount}</b> ${bounds(g.constraints?.minBranches,g.constraints?.maxBranches)} · loops <b>${g.loopCount}</b> ${bounds(g.constraints?.minLoops,g.constraints?.maxLoops)}</div><div class="muted">owned ${(g.events||[]).join(', ')||'none'}<br>before ${(g.beforeEvents||[]).join(', ')||'none'} · after ${(g.afterEvents||[]).join(', ')||'none'}</div></div>`).join('')||'<div class="empty">No matches</div>');renderTracePanel();renderEventDetail();draw()}
function bounds(a,b){return (a||b!=null)?`<span class="muted">[${a??0}, ${b??'∞'}]</span>`:''}
function traceEvent(e){return e?`event ${e.id} ${esc(e.kind)} · s=${e.s} · strength=${e.strength} · ${esc(e.source)}`:''}
function traceAssignment(p){const before=(p.beforeEvents||[]),owned=(p.events||[]),after=(p.afterEvents||[]);const context=[before.length?`before ${before.join(', ')}`:'',after.length?`after ${after.join(', ')}`:''].filter(Boolean).join(' · ');return `${esc(p.template)} [${owned.join(', ')}]${context?` <span class="muted">(${context})</span>`:''}`}
function tracePath(path){return (path||[]).map(traceAssignment).join(' → ')||'empty path'}
function incomingPathsHtml(stage,failedStage){const paths=(stage?.survivors||[]);if(!paths.length)return '';const total=failedStage?.incomingStates??stage.outgoingStates??paths.length;const target=failedStage?`${esc(failedStage.letter)} at position ${failedStage.position}`:'boundary validation';const rows=paths.map((q,i)=>`<div class="tracePathRow"><div class="muted">path ${i+1} · cursor after event ${q.lastEvent}</div><div class="mono tracePath">${tracePath(q.path)}</div></div>`).join('');return `<div class="card traceIncoming"><b>Incoming paths to ${target}</b><div class="muted">Showing ${paths.length} bounded example${paths.length===1?'':'s'} from ${total} incoming state${total===1?'':'s'}. Events in brackets are owned by that template.</div>${rows}</div>`}
function failureHtml(f){const required=f.required?`<div>needed <b>${esc(f.required.kind)}</b> ≥ ${f.required.minStrength}</div>`:'';const cursor=Number.isInteger(f.afterEvent)?`<div class="muted">search cursor is after event ${f.afterEvent}</div>`:'';const partial=(f.partialEvents||[]).length?`<div class="warn">partial ${esc(f.template||'assignment')} owns [${f.partialEvents.join(', ')}]</div>`:'';const earlier=(f.earlierEligible||[]).map(traceEvent).join('<br>');const weak=(f.laterWeak||[]).map(traceEvent).join('<br>');const blocking=f.blockingEvent?`<div>blocked by ${traceEvent(f.blockingEvent)}</div>`:'';return `<div class="card traceFail"><b>${esc(f.template||'alignment')} · ${esc(f.reason)}</b>${required}${cursor}${partial}${blocking}${earlier?`<div class="muted">eligible at/before cursor:<br>${earlier}</div>`:''}${weak?`<div class="muted">later but too weak:<br>${weak}</div>`:''}<div class="mono tracePath">${tracePath(f.path)}</div></div>`}
function traceHtml(value){const t=value.trace||{},stages=t.stages||[],fail=t.firstFailingPosition;const rows=stages.map(s=>`<tr class="${s.position===fail?'bad':''}"><td>${s.position}</td><td>${esc(s.letter)}<br><span class="muted">${esc(s.form)}</span></td><td>${s.incomingStates} → <b>${s.outgoingStates}</b></td><td>${(s.templates||[]).map(q=>`${esc(q.name)} (${q.survivingOptions}/${q.eventOptions})`).join('<br>')}</td></tr>`).join('');// A boundary failure has no failing *position*: every stage produced survivors
// and the path died afterwards, so firstFailingPosition is absent. Falling back
// to the last stage shows the one stage with nothing to say -- on p105 l01_w06
// the ya stage is clean and the evidence (a third seen tooth 0.329 against a
// 0.5 threshold) sits two stages earlier. Fall back to the last stage that
// actually rejected something instead.
const hasEvidence=s=>(s.failures||[]).length||Object.keys(s.rejectionCounts||{}).length;
let failedIndex=stages.findIndex(s=>s.position===fail),salvaged=false;
if(failedIndex<0&&t.result==='boundary-failed'){
  for(let i=stages.length-1;i>=0;i--){if(hasEvidence(stages[i])){failedIndex=i;salvaged=true;break}}
  if(failedIndex<0)failedIndex=stages.length-1;}
const failedStage=failedIndex>=0?stages[failedIndex]:null;const incomingStage=failedIndex>0?stages[failedIndex-1]:failedStage;const incoming=failedStage?incomingPathsHtml(incomingStage,failedStage):'';const salvagedNote=salvaged?`<div class="card"><b>No stage ran out of paths</b><div class="muted">The alignment completed and was rejected afterwards, during boundary materialization &mdash; see post-topology rejections. The cards below are from position ${failedStage.position} (${esc(failedStage.letter)}), the last stage that rejected anything; they explain why the surviving path owns the events it does.</div></div>`:'';const failures=(failedStage?.failures||[]).slice(0,4).map(failureHtml).join('');const counts=Object.entries(failedStage?.rejectionCounts||{}).map(([k,v])=>`${esc(k)}: ${v}`).join('<br>');const post=Object.entries(t.postTopologyRejections||{}).map(([k,v])=>`${esc(k)}: ${v}`).join('<br>');return `<div class="card"><div><b>${esc(t.finalResult||t.result||'trace')}</b> · ${value.elapsedMs} ms</div><div class="muted">${t.eventCount??0} events · ${t.joinHypotheses??0} join hypotheses</div></div><table><thead><tr><th>Pos</th><th>Letter</th><th>States</th><th>Templates</th></tr></thead><tbody>${rows}</tbody></table>${incoming}${counts?`<div class="card"><b>Rejection summary</b><div class="mono">${counts}</div></div>`:''}${post?`<div class="card"><b>Post-topology rejections</b><div class="mono">${post}</div></div>`:''}${salvagedNote}${failures||'<div class="empty">No bounded rejection example was retained.</div>'}`}
function renderTracePanel(){const p=paw();if(p.ok){$('alignmentTrace').innerHTML='';return}const value=state.traces[state.paw];$('alignmentTrace').innerHTML=`<h2>Alignment failure trace</h2><button id="explainNoMatch" class="traceButton" ${state.traceLoading?'disabled':''}>${state.traceLoading?'Tracing…':value?'Run trace again':'Explain no match'}</button>${value?traceHtml(value):'<div class="muted">Tracing runs only when this button is clicked.</div>'}`;$('explainNoMatch').onclick=explainNoMatch}
async function explainNoMatch(){if(state.traceLoading)return;state.traceLoading=true;renderTracePanel();const word=state.data.wordId,pawIndex=state.paw;status(`Tracing ${word} PAW ${pawIndex}…`);try{const value=await getJson(`/api/trace?${pageParam()}&word=${encodeURIComponent(word)}&paw=${pawIndex}`);if(state.data.wordId===word){state.traces[pawIndex]=value;renderTracePanel()}status(`${word} PAW ${pawIndex} trace · ${value.elapsedMs} ms`)}catch(e){status(e.message)}finally{state.traceLoading=false;renderTracePanel()}}
// Markers that share a point cannot be told apart by clicking -- branch,
// high-branch and zigzag-branch routinely share one anchor, as do an end and
// its -height twin. Rather than move any of them, a click keeps everything
// under the cursor and the panel lists it, so the buried ones are reachable
// here instead of on the canvas.
function renderEventDetail(){const e=state.selectedEvent;if(!e){$('eventDetail').textContent='Click an event marker.';return}
 const here=(state.eventsAtPoint||[]).filter(q=>q!==e);
 const also=here.length?`\n<span class="muted">also here:</span> `+here.map(q=>`<a href="#" data-event="${q.id}" style="color:${eventColors[q.kind]||'#ddd'};text-decoration:none;border-bottom:1px dotted currentColor">${q.id} ${esc(q.kind)}</a>`).join(' <span class="muted">·</span> '):'';
 $('eventDetail').innerHTML=`<b style="color:${eventColors[e.kind]||'#fff'}">event ${e.id} · ${esc(e.kind)}</b>\ns = ${e.s}\nxy = ${e.x}, ${e.y}\nstrength = ${e.strength}\nthickness = ${e.thickness}\nrelief = ${e.relief}\nsource = ${e.source}\nowned by = ${esc((e.matchedBy||[]).join(', ')||'none')}\ncontext for = ${esc((e.contextMatchedBy||[]).join(', ')||'none')}${also}`;
 $('eventDetail').querySelectorAll('[data-event]').forEach(a=>a.onclick=ev=>{ev.preventDefault();const pick=(state.eventsAtPoint||[]).find(q=>q.id===+a.dataset.event);if(pick){state.selectedEvent=pick;renderEventDetail();draw()}})}
// The -40 margin makes the ratio negative in a viewport under 40px tall --
// a collapsed pane, a very short window -- and a negative zoom silently
// breaks click tolerance (18/zoom) as well as the canvas scale, so nothing
// on the image can be selected. Clamp to a positive floor.
function fit(){const v=$('viewport');state.zoom=Math.max(.05,Math.min((v.clientWidth-40)/state.data.width,(v.clientHeight-40)/state.data.height,1.8));$('zoom').value=Math.round(state.zoom*100);draw()}
function line(ctx,pts,color,width=1,dash=[]){if(!pts||pts.length<2)return;ctx.beginPath();ctx.setLineDash(dash);ctx.moveTo(pts[0][0],pts[0][1]);for(const p of pts.slice(1))ctx.lineTo(p[0],p[1]);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke();ctx.setLineDash([])}
function diamond(ctx,x,y,color,r=6){ctx.beginPath();ctx.moveTo(x,y-r);ctx.lineTo(x+r,y);ctx.lineTo(x,y+r);ctx.lineTo(x-r,y);ctx.closePath();ctx.strokeStyle=color;ctx.lineWidth=3;ctx.stroke()}
function square(ctx,x,y,color,r=6){ctx.beginPath();ctx.rect(x-r,y-r,2*r,2*r);ctx.strokeStyle=color;ctx.lineWidth=3;ctx.stroke()}
// Every marker is drawn on its own event, never displaced. `start` and its
// `-height` twin share one point by construction, so they are nested at
// different radii instead: both stay readable and both stay on the spot they
// describe. Coincident markers of the same size still hide each other -- as
// they always have -- and the side panel is where a buried event is read.
const endRadius={start:7,end:7,'start-height':12,'end-height':12};
function draw(){if(!state.data)return;const c=$('canvas'),ctx=c.getContext('2d'),p=paw(),d=p.diagnostic;c.width=state.data.width;c.height=state.data.height;c.style.width=`${c.width*state.zoom}px`;c.style.height=`${c.height*state.zoom}px`;ctx.clearRect(0,0,c.width,c.height);ctx.fillStyle='#000';ctx.fillRect(0,0,c.width,c.height);
 for(const k of ['ink','letters','joinsInk','unassigned'])if(state.layers[k]&&state.images[k])ctx.drawImage(state.images[k],0,0);
 if(state.layers.medial)for(const e of d.medialEdges)line(ctx,e,'#3c3cc8',1.4);if(state.layers.graphBranches)for(const b of d.branchPaths.filter(b=>b.source==='graph'))line(ctx,b.points,'#ff9b45',2);
 if(state.layers.prunedBranches)for(const b of d.branchPaths.filter(b=>b.source==='pruned'))line(ctx,b.points,'#ff00ff',3);if(state.layers.spine)line(ctx,d.spine,'#78e6ff',3);
 if(state.layers.baseline){line(ctx,[[0,state.data.baselineY],[c.width,state.data.baselineY]],'#00c800',2)}
 // A cycle is owned along the spine but drawn at the counter it encloses:
 // both loops of a two-counter Heh attach to the spine within a stroke width
 // of each other, so marking the attachment point showed one blob, not loops.
 if(state.layers.loops)for(const l of p.loopEvidence){if(l.footprintAXY)line(ctx,[l.footprintAXY,l.footprintBXY],'#ffdb5433',9,[3,7]);line(ctx,[l.aXY,l.bXY],'#ffdb54',5,[10,6]);const cs=(l.centres&&l.centres.length)?l.centres:[[l.x,l.y]];ctx.beginPath();ctx.arc(l.x,l.y,4,0,Math.PI*2);ctx.fillStyle='#ffdb54';ctx.fill();for(const [cx,cy] of cs){if(cx!==l.x||cy!==l.y)line(ctx,[[l.x,l.y],[cx,cy]],'#ffdb5477',2,[4,4]);ctx.beginPath();ctx.arc(cx,cy,13,0,Math.PI*2);ctx.strokeStyle='#ffdb54';ctx.lineWidth=3;ctx.stroke()}ctx.fillStyle='#ffdb54';ctx.fillText(`L${l.count}`,cs[0][0]+16,cs[0][1]-12)}
 if(state.layers.candidates)for(const q of p.candidates){ctx.beginPath();ctx.arc(q.x,q.y,7,0,Math.PI*2);ctx.fillStyle='#00dcff';ctx.fill()}
 if(state.layers.cuts)for(const q of p.cuts)line(ctx,q.chord,'#ff3030',7);
 if(state.layers.joinBounds)for(const j of p.joins){if(!j.start||!j.end)continue;line(ctx,[[j.start.x,j.start.y],[j.end.x,j.end.y]],'#d3d3d3',5);for(const [q,col] of [[j.start,'#ff79ff'],[j.end,'#ffa33c']]){ctx.beginPath();ctx.arc(q.x,q.y,9,0,Math.PI*2);ctx.strokeStyle=col;ctx.lineWidth=4;ctx.stroke()}}
 // Ends are drawn as squares: they mark where the PAW stops rather than
 // something the pen did, so they should not read as one more feature.
 if(state.layers.events)for(const e of p.skeletonEvents)if(state.eventKinds.has(e.kind))(endKinds.has(e.kind)?square:diamond)(ctx,e.x,e.y,eventColors[e.kind]||'#ddd',endRadius[e.kind]||7);
 if(state.layers.matched)for(const e of p.skeletonEvents)if(state.eventKinds.has(e.kind)){if(e.contextMatchedBy?.length){ctx.beginPath();ctx.setLineDash([6,5]);ctx.arc(e.x,e.y,17,0,Math.PI*2);ctx.strokeStyle='#aaa';ctx.lineWidth=2;ctx.stroke();ctx.setLineDash([])}if(e.matchedBy?.length){ctx.beginPath();ctx.arc(e.x,e.y,13,0,Math.PI*2);ctx.strokeStyle='#fff';ctx.lineWidth=3;ctx.stroke()}}
 if(state.layers.labels)for(const g of p.letters_geometry||[]){if(!g.bbox)continue;ctx.font='bold 28px system-ui';ctx.fillStyle='#fff';ctx.strokeStyle='#000';ctx.lineWidth=5;const x=(g.bbox[0]+g.bbox[2])/2,y=(g.bbox[1]+g.bbox[3])/2;ctx.strokeText(String(g.index),x,y);ctx.fillText(String(g.index),x,y)}}
// Every visible event within the click tolerance, nearest first. nearestEvent
// stays the head of that list, so hover and the old single-selection path
// behave exactly as before.
function eventsNear(ev){if(!state.layers.events)return [];const rect=$('canvas').getBoundingClientRect(),x=(ev.clientX-rect.left)*state.data.width/rect.width,y=(ev.clientY-rect.top)*state.data.height/rect.height,tol=18/state.zoom;
 return paw().skeletonEvents.filter(e=>state.eventKinds.has(e.kind)).map(e=>[Math.hypot(e.x-x,e.y-y),e]).filter(([d])=>d<tol).sort((a,b)=>a[0]-b[0]).map(([,e])=>e)}
function nearestEvent(ev){return eventsNear(ev)[0]||null}
$('canvas').onclick=e=>{state.eventsAtPoint=eventsNear(e);state.selectedEvent=state.eventsAtPoint[0]||null;renderEventDetail();draw()};$('canvas').onmousemove=e=>{const q=nearestEvent(e);$('canvas').title=q?`${q.id} ${q.kind} · strength=${q.strength} · s=${q.s} · ${q.source}`:''};
$('pageSelect').onchange=e=>selectPage(e.target.value);$('wordSelect').onchange=e=>selectWord(+e.target.value);$('pawSelect').onchange=e=>{state.paw=+e.target.value;state.selectedEvent=null;state.eventsAtPoint=[];buildEventKinds();renderPanels()};$('prev').onclick=()=>selectWord(state.wordIndex-1);$('next').onclick=()=>selectWord(state.wordIndex+1);$('reload').onclick=()=>selectWord(state.wordIndex,true);$('fit').onclick=fit;$('zoom').oninput=e=>{state.zoom=+e.target.value/100;draw()};
window.onkeydown=e=>{if(['INPUT','SELECT'].includes(e.target.tagName))return;if(e.key==='ArrowLeft')selectWord(state.wordIndex-1);if(e.key==='ArrowRight')selectWord(state.wordIndex+1);if(e.key.toLowerCase()==='r')selectWord(state.wordIndex,true)};window.onresize=()=>{};boot().catch(e=>status(e.message));
</script></body></html>'''


def _xy(point: np.ndarray | tuple[float, float]) -> list[float]:
    return [round(float(point[0]) - seg.PAD_PX, 2),
            round(float(point[1]) - seg.PAD_PX, 2)]


class DiagnosticIndex:
    def __init__(self, analysis: Path, page: int, templates: Path,
                 cut_policy: str) -> None:
        self.analysis = analysis.resolve()
        self.page = page
        self.page_dir, self.page_json = seg.load_page(str(self.analysis), page)
        self.page_dir = str(Path(self.page_dir).resolve())
        self.templates = templates.resolve()
        self.cut_policy = cut_policy
        self.words: dict[str, tuple[dict, dict]] = {
            word["wordId"]: (line, word)
            for line, word in seg.iter_words(self.page_json)
        }
        self.cache: dict[str, tuple[tuple[int, int], seg.WordResult, dict]] = {}
        self.lock = threading.RLock()

    def _template_token(self) -> tuple[int, int]:
        stat = self.templates.stat()
        return stat.st_mtime_ns, stat.st_size

    def bootstrap(self, pages: "Sequence[int]" = ()) -> dict:
        records = []
        for word_id, (_, word) in self.words.items():
            records.append({"wordId": word_id,
                            "text": word.get("expectedText", ""),
                            "paws": len(word.get("paws", ()))})
        return {"page": self.page, "words": records,
                "pages": list(pages),
                "templates": str(self.templates)}

    def body_path(self, word_id: str) -> Path:
        path = Path(self.page_dir) / "subwordimgs" / f"{word_id}_body.png"
        if word_id not in self.words or not path.is_file():
            raise ValueError(f"unknown word or missing body image: {word_id}")
        return path

    def analyze(self, word_id: str, force: bool = False) -> tuple[seg.WordResult, dict]:
        token = self._template_token()
        with self.lock:
            cached = self.cache.get(word_id)
            if cached and cached[0] == token and not force:
                return cached[1], cached[2]
            if word_id not in self.words:
                raise ValueError(f"unknown word: {word_id}")
            line, word = self.words[word_id]
            skeleton, elastic = seg.load_template_definitions(str(self.templates))
            baseline = float(line["baseLine"]) - float(word["y"])
            result = seg.segment_word(
                str(self.body_path(word_id)), word.get("expectedText", ""), baseline,
                want_pixels=True, skeleton_templates=skeleton,
                elastic_body_templates=elastic, body_join_alignment=True,
                cut_policy=self.cut_policy, paws_meta=word.get("paws"),
                page_dir=self.page_dir)
            result.word_id = word_id
            payload = self._diagnostic_json(result, elastic, token)
            self.cache[word_id] = token, result, payload
            return result, payload

    def trace_no_match(self, word_id: str, paw_index: int) -> dict:
        """Rerun one word with bounded tracing, only for an explicit GUI call."""
        with self.lock:
            if word_id not in self.words:
                raise ValueError(f"unknown word: {word_id}")
            line, word = self.words[word_id]
            paw_count = len(word.get("paws", ()))
            if paw_index < 0 or paw_index >= paw_count:
                raise ValueError(
                    f"PAW index {paw_index} is outside 0..{paw_count - 1}")
            skeleton, elastic = seg.load_template_definitions(
                str(self.templates))
            baseline = float(line["baseLine"]) - float(word["y"])
            trace: dict = {"targetPaw": paw_index}
            result = seg.segment_word(
                str(self.body_path(word_id)), word.get("expectedText", ""),
                baseline, want_pixels=False, skeleton_templates=skeleton,
                elastic_body_templates=elastic, body_join_alignment=True,
                cut_policy=self.cut_policy, paws_meta=word.get("paws"),
                page_dir=self.page_dir, alignment_trace=trace)
            if paw_index >= len(result.paws):
                raise ValueError(f"PAW {paw_index} was not produced")
            paw_trace = trace.get("paw", {"pawIndex": paw_index})
            paw_trace["segmentationOk"] = result.paws[paw_index].ok
            paw_trace["segmentationReason"] = result.paws[paw_index].reason
            return paw_trace

    def _diagnostic_json(self, result: seg.WordResult,
                         templates: tuple[seg.ElasticBodyTemplate, ...],
                         token: tuple[int, int]) -> dict:
        payload = seg.result_to_json(result)
        image = cv2.imread(str(self.body_path(result.word_id)), cv2.IMREAD_GRAYSCALE)
        payload.update(width=int(image.shape[1]), height=int(image.shape[0]),
                       cacheToken=f"{token[0]}-{token[1]}")
        by_name = {template.name: template for template in templates}
        for pr, record in zip(result.paws, payload["paws"]):
            matched_by: dict[int, list[str]] = {}
            context_matched_by: dict[int, list[str]] = {}
            for match in pr.shape_matches:
                for event_id in match.events:
                    matched_by.setdefault(event_id, []).append(match.template)
                for event_id in match.before_events:
                    context_matched_by.setdefault(event_id, []).append(
                        f"{match.template} (before)")
                for event_id in match.after_events:
                    context_matched_by.setdefault(event_id, []).append(
                        f"{match.template} (after)")
            for event_id, event in enumerate(record["skeletonEvents"]):
                event["id"] = event_id
                event["matchedBy"] = matched_by.get(event_id, [])
                event["contextMatchedBy"] = context_matched_by.get(
                    event_id, [])
            med = pr.medial
            diagnostic = {"medialEdges": [], "spine": [],
                          "branchPaths": [], "groups": []}
            if med is not None:
                diagnostic["medialEdges"] = [
                    [_xy(med.vertices[a]), _xy(med.vertices[b])]
                    for a, b in med.edges]
                diagnostic["spine"] = [_xy(med.vertices[v]) for v in med.spine]
                diagnostic["branchPaths"] = [
                    {"s": round(path.s, 2), "source": path.source,
                     "role": path.role,
                     "length": round(path.length, 2),
                     "strength": round(path.length / max(pr.stroke_width, 1e-6), 3),
                     "points": [_xy(point) for point in path.points]}
                    for path in seg.skeleton_branch_paths(med)]
                for item, exported in zip(pr.loop_evidence, record["loopEvidence"]):
                    points = med.vertices[med.spine]

                    def spine_xy(s_value: float) -> list:
                        return [round(float(np.interp(
                            s_value, med.spine_s, points[:, axis]))
                            - seg.PAD_PX, 2) for axis in (0, 1)]

                    # The foot of the loop is where it attaches to the spine.
                    # a_s/b_s are the wider projected footprint the solver owns
                    # by, and drawing that made the marker overshoot the two
                    # junctions a reader can see.
                    foot = item.attachment or (item.a_s, item.b_s)
                    exported["aXY"] = spine_xy(foot[0])
                    exported["bXY"] = spine_xy(foot[1])
                    exported["footprintAXY"] = spine_xy(item.a_s)
                    exported["footprintBXY"] = spine_xy(item.b_s)
            match_by_group = {
                (match.letter_start, match.letter_end, match.template): match
                for match in pr.shape_matches}
            for group in pr.segment_groups:
                template = by_name.get(group.template or "")
                match = match_by_group.get(
                    (group.letter_start, group.letter_end, group.template))
                constraints = None
                if template is not None:
                    constraints = {
                        "minBranches": template.min_branches,
                        "maxBranches": template.max_branches,
                        "minLoops": template.min_loops,
                        "maxLoops": template.max_loops}
                diagnostic["groups"].append({
                    "letters": [group.letter_start, group.letter_end],
                    "span": [round(group.a_s, 2), round(group.b_s, 2)],
                    "template": group.template,
                    "events": list(match.events) if match else [],
                    "beforeEvents": (list(match.before_events)
                                     if match else []),
                    "afterEvents": (list(match.after_events)
                                    if match else []),
                    "branchCount": seg.branch_count_in_span(
                        pr.skeleton_events, group.a_s, group.b_s),
                    "loopCount": seg.loop_count_in_span(
                        pr.loop_evidence, group.a_s, group.b_s),
                    "constraints": constraints})
            record["diagnostic"] = diagnostic
        return payload

    def image_bytes(self, word_id: str, kind: str) -> bytes:
        if kind == "ink":
            return self.body_path(word_id).read_bytes()
        result, _ = self.analyze(word_id)
        body = cv2.imread(str(self.body_path(word_id)), cv2.IMREAD_GRAYSCALE)
        out = np.zeros((*body.shape, 4), dtype=np.uint8)
        colors = [(90, 120, 255, 190), (120, 220, 90, 190),
                  (255, 160, 120, 190), (90, 210, 250, 190),
                  (240, 120, 220, 190), (235, 230, 110, 190)]
        for pr in result.paws:
            if pr.pixel_letter is None:
                continue
            labels = pr.pixel_letter[
                seg.PAD_PX:seg.PAD_PX + body.shape[0],
                seg.PAD_PX:seg.PAD_PX + body.shape[1]]
            if kind == "letters":
                for index in range(len(pr.letters)):
                    out[labels == index] = colors[index % len(colors)]
            elif kind == "joins":
                out[labels == -3] = (211, 211, 211, 210)
            elif kind == "unassigned":
                out[labels == -2] = (180, 180, 80, 210)
            else:
                raise ValueError(f"unknown image layer: {kind}")
        ok, encoded = cv2.imencode(".png", out)
        if not ok:
            raise ValueError("could not encode diagnostic layer")
        return encoded.tobytes()


class PageLibrary:
    """Every page in one analysis directory, loaded on demand.

    The viewer used to be bound to the single page named on the command line,
    so a deep link from another tool could only ever carry a word id and had to
    hope the server happened to be serving the right page. Pages are loaded
    lazily and a few are kept, because each index also caches segmentation
    results and a whole mushaf of them would not fit.
    """

    def __init__(self, analysis: Path, templates: Path, cut_policy: str,
                 default_page: int, keep: int = 4) -> None:
        self.analysis = analysis
        self.templates = templates
        self.cut_policy = cut_policy
        self.default_page = default_page
        self.keep = keep
        self.loaded: "OrderedDict[int, DiagnosticIndex]" = OrderedDict()
        self.lock = threading.RLock()

    def available(self) -> list[int]:
        """Pages that actually have analyzer output, cheapest check first."""
        pages = []
        for entry in sorted(self.analysis.glob("page[0-9][0-9][0-9]")):
            if (entry / "words.json").is_file():
                try:
                    pages.append(int(entry.name[4:]))
                except ValueError:
                    continue
        return pages

    def get(self, page: int | None) -> DiagnosticIndex:
        page = self.default_page if page is None else int(page)
        with self.lock:
            index = self.loaded.get(page)
            if index is not None:
                self.loaded.move_to_end(page)
                return index
            index = DiagnosticIndex(self.analysis, page, self.templates,
                                    self.cut_policy)
            self.loaded[page] = index
            while len(self.loaded) > self.keep:
                self.loaded.popitem(last=False)
            return index


class Handler(BaseHTTPRequestHandler):
    library: PageLibrary | None = None

    def _write(self, data: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, value: object, status: int = 200) -> None:
        self._write(json.dumps(value, ensure_ascii=False).encode(),
                    "application/json; charset=utf-8", status)

    def do_GET(self) -> None:  # noqa: N802
        assert self.library is not None
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        try:
            requested = query.get("page", [""])[0].strip()
            index = self.library.get(int(requested) if requested else None)
            if parsed.path == "/":
                self._write(HTML.encode(), "text/html; charset=utf-8")
            elif parsed.path == "/api/bootstrap":
                self._json(index.bootstrap(self.library.available()))
            elif parsed.path == "/api/analyze":
                word = query.get("word", [""])[0]
                import time
                started = time.perf_counter()
                _, payload = index.analyze(
                    word, query.get("reload", ["0"])[0] == "1")
                value = dict(payload)
                value["elapsedMs"] = round((time.perf_counter() - started) * 1000)
                self._json(value)
            elif parsed.path == "/api/trace":
                word = query.get("word", [""])[0]
                paw = int(query.get("paw", ["0"])[0])
                import time
                started = time.perf_counter()
                trace = index.trace_no_match(word, paw)
                self._json({
                    "wordId": word, "pawIndex": paw, "trace": trace,
                    "elapsedMs": round(
                        (time.perf_counter() - started) * 1000)})
            elif parsed.path == "/image":
                word = query.get("word", [""])[0]
                kind = query.get("kind", ["ink"])[0]
                self._write(index.image_bytes(word, kind),
                            mimetypes.types_map.get(".png", "image/png"))
            else:
                self.send_error(404)
        except (OSError, ValueError, KeyError) as error:
            self._json({"error": str(error)}, 400)

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--page", type=int, default=3,
                        help="page shown when the URL does not name one; any "
                             "page in --analysis can be reached with ?page=N")
    parser.add_argument("--templates", type=Path,
                        default=Path(seg.DEFAULT_TEMPLATE_PATH))
    parser.add_argument("--cut-policy", choices=("current-end", "midpoint",
                                                   "next-start", "none"),
                        default="next-start")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8879)
    parser.add_argument("--word", help="initial/check word id")
    parser.add_argument("--check", action="store_true",
                        help="analyze one word, print a summary, and exit")
    args = parser.parse_args()
    library = PageLibrary(args.analysis.resolve(), args.templates,
                          args.cut_policy, args.page)
    if args.check:
        index = library.get(args.page)
        word = args.word or next(iter(index.words))
        _, payload = index.analyze(word, True)
        print(json.dumps({"wordId": word, "abstain": payload["abstain"],
                          "paws": len(payload["paws"]),
                          "events": sum(len(p["skeletonEvents"])
                                        for p in payload["paws"])},
                         ensure_ascii=False))
        return 0
    Handler.library = library
    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    initial = f"?page={args.page}"
    if args.word:
        initial += f"&word={urllib.parse.quote(args.word)}"
    print(f"PAW diagnostics: http://{args.bind}:{args.port}/{initial}")
    print(f"{len(library.available())} pages available in {args.analysis}; "
          f"any of them via ?page=N")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
