#!/usr/bin/env python3
"""Local browser GUI for browsing and reviewing word/PAW segmentation output.

Run this from the mushafanalyzer directory, for example:

    python3 tools/review_segmentation.py \
        --root output/oldmedina2-analysis --port 8878

Then open http://127.0.0.1:8878/. Review labels are stored in
``<root>/.segmentation-review.json`` and do not modify words.json or masks.
"""

from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import json
import mimetypes
import os
from pathlib import Path
import re
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.parse


PAGE_RE = re.compile(r"^page\d+$")
REVIEW_STATUSES = {"unreviewed", "correct", "error", "uncertain"}


HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mushaf segmentation review</title>
<style>
:root { color-scheme: dark; --bg:#10161d; --panel:#19232d; --panel2:#202e3a;
  --line:#3a4b5b; --muted:#aab9c7; --accent:#69b7ff; --good:#2c9d70;
  --bad:#c95757; --warn:#c39542; }
* { box-sizing:border-box; }
html, body { height:100%; }
body { margin:0; background:var(--bg); color:#edf3f8; font:14px system-ui,sans-serif; overflow:hidden; }
button, input, select { color:inherit; background:var(--panel2); border:1px solid var(--line);
  border-radius:5px; padding:7px 9px; font:inherit; }
button { cursor:pointer; }
button:hover, .list-row:hover, .group-row:hover { border-color:var(--accent); }
header { position:sticky; top:0; z-index:5; background:#16212b; border-bottom:1px solid var(--line);
  padding:10px 14px; display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
h1 { font-size:18px; margin:0 18px 0 0; white-space:nowrap; }
.nav { display:flex; gap:5px; }
.nav button.active { background:var(--accent); color:#07111b; border-color:var(--accent); font-weight:700; }
.toolbar { margin-left:auto; display:flex; gap:7px; align-items:center; flex-wrap:wrap; }
.toolbar input { min-width:220px; }
#status { color:var(--muted); font-size:12px; }
.app { display:grid; grid-template-columns:230px minmax(270px,330px) minmax(0,1fr);
  height:calc(100vh - 61px); min-height:0; }
.app.global { grid-template-columns:250px minmax(0,1fr); height:calc(100vh - 61px); }
.app > aside, .app > .middle, .app > main { min-height:0; }
aside, .middle { background:var(--panel); border-right:1px solid var(--line); padding:12px; overflow:auto; }
main { padding:16px; min-width:0; min-height:0; overflow:auto; }
.section-title { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.08em;
  margin:5px 5px 8px; }
.list { display:flex; flex-direction:column; gap:5px; }
.list-row, .group-row { border:1px solid transparent; border-radius:6px; padding:8px; cursor:pointer; }
.list-row.selected, .group-row.selected { background:#294257; border-color:#5285ad; }
.list-row .sub, .group-row .sub { display:block; color:var(--muted); font-size:11px; margin-top:3px; }
.count { float:right; color:var(--muted); font-size:12px; }
.browse { display:grid; grid-template-columns:minmax(260px,350px) minmax(0,1fr); gap:15px; min-height:calc(100vh - 95px); }
.word-list { border-right:1px solid var(--line); padding-right:14px; overflow:auto; }
.word-card { background:var(--panel); border:1px solid var(--line); border-radius:7px; padding:10px; margin-bottom:7px;
  cursor:pointer; }
.word-card.selected { border-color:var(--accent); background:#20384b; }
.arabic { direction:rtl; font-size:23px; line-height:1.25; }
.meta { color:var(--muted); font:12px ui-monospace, SFMono-Regular, Menlo, monospace; line-height:1.5; }
.flags { display:flex; gap:4px; flex-wrap:wrap; margin-top:5px; }
.flag, .pill { display:inline-block; border-radius:4px; padding:2px 6px; font-size:11px; background:#30404e; color:#d5e1ea; }
.flag.warn { background:#694f25; color:#ffe5a4; }
.flag.bad { background:#683035; color:#ffd2d2; }
.flag.good { background:#20563f; color:#c5f3dc; }
.detail h2 { margin:0 0 5px; font-size:20px; }
.detail-header { display:flex; gap:15px; align-items:flex-start; flex-wrap:wrap; }
.word-preview { background:white; border-radius:6px; max-width:520px; width:100%; height:170px; object-fit:contain; }
.stats { display:flex; gap:7px; flex-wrap:wrap; margin:10px 0 15px; }
.stat { background:var(--panel2); border:1px solid var(--line); border-radius:5px; padding:7px 9px; }
.stat b { display:block; color:var(--accent); font-size:16px; }
.paw-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:12px; }
.paw-card, .mark-card { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px; }
.paw-card.review-error, .mark-card.review-error { border-color:var(--bad); }
.paw-card.review-correct, .mark-card.review-correct { border-color:var(--good); }
.card-title { display:flex; align-items:center; justify-content:space-between; gap:8px; margin-bottom:7px; }
.card-title b { font-size:15px; }
.image-row { display:grid; grid-template-columns:repeat(3,1fr); gap:6px; }
.image-row figure { margin:0; min-width:0; }
.image-row img, .mark-card img { background:#fff; border-radius:4px; width:100%; height:130px; object-fit:contain; }
figcaption { color:var(--muted); text-align:center; font-size:11px; margin-top:3px; }
.masks { display:flex; gap:5px; flex-wrap:wrap; margin-top:8px; }
.mask-chip { border:1px solid var(--line); border-radius:4px; padding:3px 6px; color:#d9e7f3; font-size:11px; cursor:pointer; }
.mask-chip:hover { border-color:var(--accent); }
.review { display:flex; gap:5px; flex-wrap:wrap; margin-top:9px; }
.review button { padding:5px 7px; font-size:12px; }
.review .correct { background:#205b43; }
.review .error { background:#71343b; }
.review .uncertain { background:#675127; }
.global-toolbar { display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-bottom:14px; }
.global-toolbar input { min-width:240px; }
.global-toolbar .summary { color:var(--muted); margin-left:auto; }
.cards { display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:12px; }
.mark-card img { height:190px; }
.mark-card .mark-name { font-size:17px; margin:4px 0; }
.mark-card .arabic { font-size:20px; }
.empty { color:var(--muted); padding:22px 8px; text-align:center; }
.hint { color:var(--muted); line-height:1.55; max-width:650px; }
.pager { display:flex; gap:7px; align-items:center; justify-content:center; margin-top:15px; }
@media (max-width:900px) {
  body { overflow:auto; }
  .app, .app.global { display:block; height:auto; min-height:0; }
  aside, .middle { max-height:260px; border-right:0; border-bottom:1px solid var(--line); }
  .browse { display:block; }
  .word-list { border-right:0; border-bottom:1px solid var(--line); padding:0 0 12px; margin-bottom:14px; max-height:330px; }
  .toolbar { margin-left:0; width:100%; }
}
</style>
</head>
<body>
<header>
  <h1>Mushaf segmentation</h1>
  <nav class="nav">
    <button id="browseButton" class="active" data-mode="browse">Pages / words</button>
    <button data-mode="marks">Marks by type</button>
    <button data-mode="paws">PAWs</button>
  </nav>
  <div class="toolbar"><span id="status">Loading…</span></div>
</header>
<div id="app" class="app"></div>
<script>
const state={mode:'browse', pages:[], page:null, word:null, markClass:'', markOffset:0,
  pawFilter:'all', pawGroup:'text', markStatus:'', pawStatus:''};
const $=id=>document.getElementById(id);
const esc=value=>(value??'').toString().replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const json=value=>JSON.stringify(value);
const fmt=value=>Number(value||0).toLocaleString();
const statusName=s=>({unreviewed:'unreviewed',correct:'correct',error:'error',uncertain:'uncertain'}[s]||'unreviewed');
const reviewClass=r=>r&&r.status&&r.status!=='unreviewed'?'review-'+r.status:'';
function reviewControls(kind,id,current){
  const labels=[['correct','Correct','correct'],['error','Error','error'],['uncertain','Uncertain','uncertain'],['unreviewed','Reset','']];
  return '<div class="review">'+labels.map(([status,label,klass])=>
    `<button class="${klass}" data-review-kind="${esc(kind)}" data-review-id="${esc(id)}" data-review-status="${status}">${label}</button>`).join('')+
    `<span class="pill">${esc(statusName(current?.status))}</span></div>`;
}
async function getJson(url){const response=await fetch(url);if(!response.ok)throw new Error(await response.text());return response.json()}
function setStatus(text){$('status').textContent=text;}
function pageLabel(p){return `Page ${String(p.number).padStart(3,'0')}`;}
function wordLabel(w){return `${w.wordId} · ${w.expectedText||'?'}`;}
function issueFlags(item){return (item.flags||[]).map(f=>`<span class="flag ${f.level||'warn'}">${esc(f.label)}</span>`).join('');}
function updateNav(){document.querySelectorAll('[data-mode]').forEach(b=>b.classList.toggle('active',b.dataset.mode===state.mode));}
function render(){updateNav(); if(state.mode==='browse')renderBrowse(); else if(state.mode==='marks')renderMarks(); else renderPaws();}

function renderPageSidebar(){
  return `<aside><div class="section-title">Pages <span class="count">${fmt(state.pages.length)}</span></div><div class="list">`+
    state.pages.map(p=>`<div class="list-row ${state.page?.id===p.id?'selected':''}" data-page-id="${esc(p.id)}"><b>${esc(pageLabel(p))}</b><span class="sub">${fmt(p.wordCount)} words · ${fmt(p.pawCount)} PAWs</span></div>`).join('')+
    '</div></aside>';
}
function renderWords(){
  if(!state.page)return '<section class="word-list"><div class="empty">Select a page.</div></section>';
  return `<section class="word-list"><div class="section-title">Words · ${esc(pageLabel(state.page))}</div>`+
    `<div class="list">${state.page.words.map(w=>`<div class="word-card ${state.word?.wordKey===w.wordKey?'selected':''}" data-word-key="${esc(w.wordKey)}">
      <div class="arabic">${esc(w.expectedText||'')}</div><div class="meta">${esc(w.wordId)} · line ${w.lineIndex+1} · word ${w.wordIndex+1}<br>${fmt(w.pawCount)} PAWs · ${fmt(w.markComponentCount)} mark components</div><div class="flags">${issueFlags(w)}</div>
    </div>`).join('')}</div></section>`;
}
function renderWordDetail(){
  if(!state.word)return '<section class="detail"><div class="empty"><p>Select a word to inspect its PAWs and masks.</p><p class="hint">The PAW cards show the complete PAW, its body, its combined marks, and each categorized mark mask. Use the review buttons to record suspected errors.</p></div></section>';
  const w=state.word;
  return `<section class="detail"><div class="detail-header"><div><h2>${esc(w.expectedText||'')} <span class="meta">${esc(w.wordId)}</span></h2><div class="meta">${esc(pageLabel({number:w.pageNumber}))} · line ${w.lineIndex+1} · word ${w.wordIndex+1}</div><div class="flags">${issueFlags(w)}</div></div>${w.imageUrl?`<img class="word-preview" src="${esc(w.imageUrl)}" alt="word">`:''}</div>
    <div class="stats"><div class="stat"><b>${fmt(w.paws.length)}</b>PAWs</div><div class="stat"><b>${fmt(w.componentCount)}</b>components</div><div class="stat"><b>${fmt(w.markComponentCount)}</b>detected marks</div><div class="stat"><b>${fmt(w.expectedMarkComponentCount)}</b>expected mark paths</div></div>
    <div class="meta">Expected marks: ${esc((w.expectedMarks||[]).map(m=>`${m.className}×${m.count}`).join(' · ')||'none')}</div><h3>PAWs</h3><div class="paw-grid">${w.paws.map(renderPawCard).join('')}</div></section>`;
}
function renderPawCard(p){
  const masks=(p.markMasks||[]).map(m=>`<span class="mask-chip" data-open-mark="${esc(m.reviewId)}">${esc(m.className)} · ${fmt(m.componentIndices.length)}</span>`).join('');
  return `<article class="paw-card ${reviewClass(p.review)}" id="paw-${esc(p.pawIndex)}"><div class="card-title"><b>PAW ${String(p.pawIndex).padStart(2,'0')} · ${esc(p.text||'')}</b><span class="pill">${fmt(p.markComponentIndices?.length)} marks</span></div>
    <div class="flags">${issueFlags(p)}</div><div class="image-row"><figure><img src="${esc(p.imageUrl)}" alt="PAW"><figcaption>PAW</figcaption></figure><figure><img src="${esc(p.bodyUrl)}" alt="body"><figcaption>body</figcaption></figure><figure><img src="${esc(p.marksUrl)}" alt="marks"><figcaption>all marks</figcaption></figure></div>
    <div class="meta">${fmt(p.componentIndices?.length)} components · body ${fmt(p.bodyComponentIndices?.length)} · expected mark components ${fmt(p.expectedMarkComponents)} · observed ${fmt(p.observedMarkComponents)}<br>cost ${Number(p.cost||0).toFixed(3)}</div>
    <div class="masks">${masks||'<span class="pill">no categorized marks</span>'}</div>${reviewControls('paw',p.reviewId,p.review)}</article>`;
}
function renderBrowse(){
  $('app').className='app'; $('app').innerHTML=renderPageSidebar()+`<div class="middle">${renderWords()}</div><main>${renderWordDetail()}</main>`;
  document.querySelectorAll('[data-page-id]').forEach(e=>e.onclick=()=>selectPage(e.dataset.pageId));
  document.querySelectorAll('[data-word-key]').forEach(e=>e.onclick=()=>selectWord(e.dataset.wordKey));
  document.querySelectorAll('[data-open-mark]').forEach(e=>e.onclick=()=>focusReview(e.dataset.openMark));
}
async function selectPage(id){
  setStatus('Loading page…'); try{state.page=await getJson('/api/page?id='+encodeURIComponent(id));state.word=null;render();setStatus(`${pageLabel(state.page)} loaded`)}catch(e){setStatus(e.message)}}
async function selectWord(key){
  setStatus('Loading word…'); try{state.word=await getJson('/api/word?key='+encodeURIComponent(key));state.page=state.pages.find(p=>p.id===state.word.pageId)||state.page;render();setStatus(`${state.word.wordId} loaded`)}catch(e){setStatus(e.message)}}
async function focusReview(id){
  const item=state.markIndex?.[id];
  if(item){state.mode='browse';await selectPage(item.pageId);await selectWord(item.wordKey);setTimeout(()=>document.getElementById('paw-'+item.pawIndex)?.scrollIntoView({behavior:'smooth',block:'center'}),80);return;}
  if(state.word){const match=id.match(/:paw(\d+)/);if(match)document.getElementById('paw-'+Number(match[1]))?.scrollIntoView({behavior:'smooth',block:'center'});}
}

function renderGroupSidebar(groups,selected){
  return `<aside><div class="section-title">Detected marks by type</div><div class="list"><div class="group-row ${selected===''?'selected':''}" data-mark-class=""><b>All classes</b><span class="count">${fmt(groups.reduce((n,g)=>n+g.count,0))}</span></div>`+
    groups.map(g=>`<div class="group-row ${selected===g.className?'selected':''}" data-mark-class="${esc(g.className)}"><b>${esc(g.className)}</b><span class="count">${fmt(g.count)}</span><span class="sub">${fmt(g.words)} words · ${fmt(g.paws)} PAWs</span></div>`).join('')+'</div></aside>';
}
async function renderMarks(resetResultScroll=false){
  setStatus('Loading mark groups…');
  try{const boot=await getJson('/api/bootstrap');const data=await getJson('/api/marks?class='+encodeURIComponent(state.markClass)+'&status='+encodeURIComponent(state.markStatus)+'&offset='+state.markOffset);
    state.markIndex=Object.fromEntries(data.records.map(r=>[r.reviewId,r]));
    const previousAside=document.querySelector('.app.global > aside');
    const previousAsideScrollTop=previousAside?.scrollTop||0;
    const previousAsideScrollLeft=previousAside?.scrollLeft||0;
    const previousMain=document.querySelector('.app.global > main');
    const previousScrollTop=previousMain?.scrollTop||0;
    const previousScrollLeft=previousMain?.scrollLeft||0;
    $('app').className='app global';$('app').innerHTML=renderGroupSidebar(boot.markGroups,state.markClass)+`<main><div class="global-toolbar"><input id="markQuery" placeholder="filter page, word, component…"><select id="markStatus"><option value="">all reviews</option><option value="unreviewed">unreviewed</option><option value="correct">correct</option><option value="error">error</option><option value="uncertain">uncertain</option></select><button id="markApply">Apply</button><span class="summary">${fmt(data.total)} matches · showing ${fmt(data.records.length)}</span></div><div class="cards">${data.records.map(renderMarkCard).join('')||'<div class="empty">No marks match this filter.</div>'}</div>${renderPager(data.total,state.markOffset,'mark')}</main>`;
    const nextAside=document.querySelector('.app.global > aside');
    if(nextAside){nextAside.scrollTop=previousAsideScrollTop;nextAside.scrollLeft=previousAsideScrollLeft;}
    const nextMain=document.querySelector('.app.global > main');
    if(nextMain){nextMain.scrollTop=resetResultScroll?0:previousScrollTop;nextMain.scrollLeft=resetResultScroll?0:previousScrollLeft;}
    $('markStatus').value=state.markStatus; $('markApply').onclick=()=>{state.markStatus=$('markStatus').value;state.markOffset=0;loadMarksQuery($('markQuery').value)};
    document.querySelectorAll('[data-mark-class]').forEach(e=>e.onclick=()=>{state.markClass=e.dataset.markClass;state.markOffset=0;renderMarks(true)});
    document.querySelectorAll('[data-open-word]').forEach(e=>e.onclick=()=>openWord(e.dataset.openWord));
    setStatus('Marks loaded');
  }catch(e){setStatus(e.message)}
}
async function loadMarksQuery(q){
  const query='&q='+encodeURIComponent(q||'');const data=await getJson('/api/marks?class='+encodeURIComponent(state.markClass)+'&status='+encodeURIComponent(state.markStatus)+'&offset='+state.markOffset+query);state.markIndex=Object.fromEntries(data.records.map(r=>[r.reviewId,r]));
  const cards=document.querySelector('.cards'); if(cards)cards.innerHTML=data.records.map(renderMarkCard).join('')||'<div class="empty">No marks match this filter.</div>'; const summary=document.querySelector('.summary');if(summary)summary.textContent=`${fmt(data.total)} matches · showing ${fmt(data.records.length)}`; wireCards();
}
function renderMarkCard(r){return `<article class="mark-card ${reviewClass(r.review)}"><img src="${esc(r.imageUrl)}" alt="${esc(r.className)}"><div class="mark-name">${esc(r.className)} <span class="pill">${fmt(r.componentIndices.length)} component${r.componentIndices.length===1?'':'s'}</span></div><div class="arabic">${esc(r.expectedText||'')}</div><div class="meta">${esc(pageLabel({number:r.pageNumber}))} · ${esc(r.wordId)} · PAW ${String(r.pawIndex).padStart(2,'0')}<br>components: ${esc(r.componentIndices.join(', '))}</div>${reviewControls('mark',r.reviewId,r.review)}<button data-open-word="${esc(r.wordKey)}">Open word</button></article>`}
function renderPager(total,offset,kind){const more=offset+250<total;return `<div class="pager">${offset?`<button data-prev="${kind}">Previous</button>`:''}<span class="meta">${offset+1}-${Math.min(offset+250,total)} of ${total}</span>${more?`<button data-next="${kind}">Next</button>`:''}</div>`}

async function renderPaws(){
  setStatus('Loading PAWs…'); try{const data=await getJson('/api/paws?filter='+encodeURIComponent(state.pawFilter)+'&status='+encodeURIComponent(state.pawStatus)+'&q=');
    state.pawRecords=data.records;
    $('app').className='app global';$('app').innerHTML=`<aside><div class="section-title">PAW review filters</div><div class="list">${[['all','All PAWs'],['empty-body','Empty body'],['multi-body','Multiple bodies'],['other-mark','Contains other mark'],['count-mismatch','Count mismatch']].map(([v,l])=>`<div class="group-row ${state.pawFilter===v?'selected':''}" data-paw-filter="${v}"><b>${l}</b></div>`).join('')}</div><div class="section-title" style="margin-top:20px">Group results by</div><select id="pawGroup" style="width:100%"><option value="text">PAW text</option><option value="page">page</option><option value="review">review status</option></select></aside><main><div class="global-toolbar"><input id="pawQuery" placeholder="filter word, text, page…"><select id="pawStatus"><option value="">all reviews</option><option value="unreviewed">unreviewed</option><option value="correct">correct</option><option value="error">error</option><option value="uncertain">uncertain</option></select><button id="pawApply">Apply</button><span class="summary">${fmt(data.total)} PAWs</span></div><div id="pawGroups">${renderPawGroups(data.records)}</div></main>`;
    $('pawStatus').value=state.pawStatus;$('pawGroup').value=state.pawGroup;$('pawApply').onclick=()=>loadPawsQuery($('pawQuery').value);$('pawGroup').onchange=()=>{state.pawGroup=$('pawGroup').value;document.getElementById('pawGroups').innerHTML=renderPawGroups(state.pawRecords)};
    document.querySelectorAll('[data-paw-filter]').forEach(e=>e.onclick=()=>{state.pawFilter=e.dataset.pawFilter;renderPaws()});document.querySelectorAll('[data-open-word]').forEach(e=>e.onclick=()=>openWord(e.dataset.openWord));setStatus('PAWs loaded');
  }catch(e){setStatus(e.message)}
}
function renderPawGroups(records){
  if(!records.length)return '<div class="empty">No PAWs match this filter.</div>';
  const groups={}; records.forEach(r=>{const key=state.pawGroup==='page'?pageLabel({number:r.pageNumber}):state.pawGroup==='review'?statusName(r.review?.status):r.text||'(empty)';(groups[key]??=[]).push(r)});
  return Object.entries(groups).map(([key,items])=>`<section><h3>${esc(key)} <span class="pill">${fmt(items.length)}</span></h3><div class="cards">${items.map(renderPawReviewCard).join('')}</div></section>`).join('');
}
function renderPawReviewCard(r){return `<article class="paw-card ${reviewClass(r.review)}"><div class="card-title"><b>${esc(r.text||'')} · PAW ${String(r.pawIndex).padStart(2,'0')}</b><span class="pill">${esc(pageLabel({number:r.pageNumber}))}</span></div><div class="image-row"><figure><img src="${esc(r.imageUrl)}" alt="PAW"><figcaption>PAW</figcaption></figure><figure><img src="${esc(r.bodyUrl)}" alt="body"><figcaption>body</figcaption></figure><figure><img src="${esc(r.marksUrl)}" alt="marks"><figcaption>marks</figcaption></figure></div><div class="arabic">${esc(r.expectedText||'')}</div><div class="meta">${esc(r.wordId)} · ${fmt(r.componentCount)} components · body ${fmt(r.bodyCount)} · marks ${fmt(r.markCount)}<br>${esc((r.markClasses||[]).join(' · '))}</div><div class="flags">${issueFlags(r)}</div>${reviewControls('paw',r.reviewId,r.review)}<button data-open-word="${esc(r.wordKey)}">Open word</button></article>`}
async function loadPawsQuery(q){const status=$('pawStatus').value;state.pawStatus=status;const data=await getJson('/api/paws?filter='+encodeURIComponent(state.pawFilter)+'&status='+encodeURIComponent(status)+'&q='+encodeURIComponent(q||''));state.pawRecords=data.records;document.querySelector('.summary').textContent=`${fmt(data.total)} PAWs`;document.getElementById('pawGroups').innerHTML=renderPawGroups(data.records);wireCards()}
async function openWord(key){state.mode='browse';await selectWord(key)}
async function setReview(kind,id,status){let note='';if(status==='error')note=prompt('Describe the suspected error (optional):','')??'';try{const r=await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({kind,id,status,note})});if(!r.ok)throw new Error(await r.text());if(state.mode==='browse'&&state.word)await selectWord(state.word.wordKey);else render()}catch(e){setStatus(e.message)}}
function wireCards(){document.querySelectorAll('[data-open-word]').forEach(e=>e.onclick=()=>openWord(e.dataset.openWord));document.querySelectorAll('[data-open-mark]').forEach(e=>e.onclick=()=>focusReview(e.dataset.openMark));}
document.addEventListener('click',e=>{const b=e.target.closest('[data-review-status]');if(b)setReview(b.dataset.reviewKind,b.dataset.reviewId,b.dataset.reviewStatus);const prev=e.target.closest('[data-prev]');if(prev){state.markOffset=Math.max(0,state.markOffset-250);renderMarks()}const next=e.target.closest('[data-next]');if(next){state.markOffset+=250;renderMarks()}});
document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>{state.mode=b.dataset.mode;render()});
async function init(){try{const boot=await getJson('/api/bootstrap');state.pages=boot.pages;setStatus(`${fmt(boot.wordCount)} words · ${fmt(boot.pawCount)} PAWs`);render();if(state.pages.length)selectPage(state.pages[0].id)}catch(e){setStatus(e.message);$('app').innerHTML=`<main><div class="empty">${esc(e.message)}</div></main>`}}
init();
</script>
</body>
</html>'''


def page_number(page_dir: Path, data: dict) -> int:
    value = data.get("page")
    if isinstance(value, int):
        return value
    match = re.search(r"(\d+)$", page_dir.name)
    return int(match.group(1)) if match else 0


def paw_list(word: dict) -> list[dict]:
    paws = word.get("paws")
    if isinstance(paws, list):
        return paws
    pieces = word.get("pieces")
    return pieces if isinstance(pieces, list) else []


class SegmentationIndex:
    def __init__(self, root: Path, review_file: Path | None = None):
        self.root = root.resolve()
        self.review_file = (review_file or (self.root / ".segmentation-review.json")).resolve()
        self.pages: dict[str, dict] = {}
        self.words: dict[str, dict] = {}
        self.paws: list[dict] = []
        self.marks: list[dict] = []
        self.artifacts: dict[str, Path] = {}
        self.artifact_ids: dict[Path, str] = {}
        self.missing_artifacts: list[dict] = []
        self.reviews = self._load_reviews()
        self._load()

    def _load_reviews(self) -> dict:
        if not self.review_file.is_file():
            return {}
        try:
            data = json.loads(self.review_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data.get("items", {}) if isinstance(data, dict) else {}

    def _page_files(self) -> list[Path]:
        if (self.root / "words.json").is_file():
            return [self.root / "words.json"]
        return sorted(self.root.glob("page*/words.json"))

    def _load(self) -> None:
        for words_path in self._page_files():
            try:
                data = json.loads(words_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            page_dir = words_path.parent.resolve()
            page_id = page_dir.name
            number = page_number(page_dir, data)
            page = {"id": page_id, "number": number, "words": [], "path": page_dir}
            for line in data.get("lines", []):
                for word in line.get("words", []):
                    word_id = str(word.get("wordId", ""))
                    word_key = f"{page_id}:{word_id}"
                    ref = {"page": page, "page_dir": page_dir, "word": word,
                           "wordKey": word_key, "wordId": word_id}
                    self.words[word_key] = ref
                    self._artifact_url(word.get("pngPath"), page_dir, word_key)
                    page["words"].append(self._word_summary(ref))
                    for paw_index, paw in enumerate(paw_list(word)):
                        paw_ref = self._paw_record(ref, paw_index, paw)
                        self.paws.append(paw_ref)
                        for mask_index, mask in enumerate(paw.get("markMasks", [])):
                            self.marks.append(
                                self._mark_record(ref, paw_index, mask_index, mask))
            page["wordCount"] = len(page["words"])
            page["pawCount"] = sum(w["pawCount"] for w in page["words"])
            page["markCount"] = sum(w["markMaskCount"] for w in page["words"])
            self.pages[page_id] = page

    def _review(self, kind: str, review_id: str) -> dict:
        value = self.reviews.get(f"{kind}:{review_id}", {})
        return value if isinstance(value, dict) else {}

    def _resolve_artifact(self, raw: object, page_dir: Path) -> Path | None:
        if not raw:
            return None
        value = Path(str(raw))
        candidates: list[Path] = []
        if value.is_absolute():
            candidates.append(value)
        candidates.extend((page_dir / value, self.root / value, Path.cwd() / value))
        for index, part in enumerate(value.parts):
            if PAGE_RE.match(part):
                suffix = Path(*value.parts[index:])
                candidates.append(self.root / suffix)
                if self.root.name == part and len(value.parts) > index + 1:
                    candidates.append(self.root / Path(*value.parts[index + 1:]))
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            try:
                resolved.relative_to(self.root)
            except ValueError:
                continue
            if resolved.is_file():
                return resolved
        return None

    def _artifact_url(self, raw: object, page_dir: Path, owner: str) -> str:
        path = self._resolve_artifact(raw, page_dir)
        if path is None:
            if raw:
                self.missing_artifacts.append({"owner": owner, "path": str(raw)})
            return ""
        artifact_id = self.artifact_ids.get(path)
        if artifact_id is None:
            artifact_id = f"a{len(self.artifacts) + 1}"
            self.artifact_ids[path] = artifact_id
            self.artifacts[artifact_id] = path
        return "/image?id=" + urllib.parse.quote(artifact_id)

    @staticmethod
    def _flags_for_word(word: dict, paw_refs: list[dict] | None = None) -> list[dict]:
        flags = []
        quality = word.get("quality", {})
        if quality.get("spacingSpanValid") is False:
            flags.append({"label": "invalid spacing", "level": "bad"})
        if quality.get("negativeSpacingGap"):
            flags.append({"label": "negative gap", "level": "warn"})
        if paw_refs is not None:
            if any(not paw["bodyComponentIndices"] for paw in paw_refs):
                flags.append({"label": "empty PAW body", "level": "warn"})
            if any(len(paw["bodyComponentIndices"]) > 1 for paw in paw_refs):
                flags.append({"label": "multiple PAW bodies", "level": "bad"})
            if any("other" in paw["markClasses"] for paw in paw_refs):
                flags.append({"label": "other mark class", "level": "warn"})
        return flags

    def _word_summary(self, ref: dict) -> dict:
        word = ref["word"]
        paws = paw_list(word)
        mark_components = sum(len(paw.get("markComponentIndices", [])) for paw in paws)
        mark_masks = sum(len(paw.get("markMasks", [])) for paw in paws)
        return {
            "wordKey": ref["wordKey"], "wordId": ref["wordId"],
            "pageId": ref["page"]["id"], "pageNumber": ref["page"]["number"],
            "lineIndex": word.get("lineIndex", 0), "wordIndex": word.get("wordIndex", 0),
            "expectedText": word.get("expectedText", word.get("text", "")),
            "componentCount": len(word.get("componentIndices", [])),
            "pawCount": len(paws), "markComponentCount": mark_components,
            "markMaskCount": mark_masks,
            "flags": self._flags_for_word(word),
        }

    def _paw_record(self, ref: dict, paw_index: int, paw: dict) -> dict:
        review_id = f"{ref['wordKey']}:paw{paw_index:02d}"
        page_dir = ref["page_dir"]
        body = list(paw.get("bodyComponentIndices", []))
        marks = list(paw.get("markComponentIndices", []))
        masks = paw.get("markMasks", [])
        classes = [str(mask.get("className", "other")) for mask in masks]
        flags = []
        if not body:
            flags.append({"label": "empty body", "level": "warn"})
        if paw.get("bodyConnectedToPreviousWord"):
            flags.append({"label": "body connected to previous word", "level": "warn"})
        if paw.get("bodyMergedWithPrevious"):
            # The body shown here is the previous PAW's: one component holds
            # both, and the letter segmenter is the one that cuts it.
            flags.append({"label": "body shared with previous PAW",
                          "level": "warn"})
        if len(body) > 1:
            flags.append({"label": "multiple bodies", "level": "bad"})
        if "other" in classes:
            flags.append({"label": "other mark", "level": "warn"})
        if paw.get("expectedMarkComponents") is not None and \
                paw.get("observedMarkComponents") != paw.get("expectedMarkComponents"):
            flags.append({"label": "mark count mismatch", "level": "warn"})
        return {
            "reviewId": review_id, "wordKey": ref["wordKey"],
            "wordId": ref["wordId"], "pageId": ref["page"]["id"],
            "pageNumber": ref["page"]["number"], "pawIndex": paw_index,
            "text": paw.get("text", ""),
            "expectedText": ref["word"].get("expectedText", ""),
            "componentIndices": list(paw.get("componentIndices", [])),
            "bodyComponentIndices": body, "markComponentIndices": marks,
            "bodyConnectedToPreviousWord": bool(
                paw.get("bodyConnectedToPreviousWord")
            ),
            "bodyMergedWithPrevious": bool(paw.get("bodyMergedWithPrevious")),
            "expectedMarkComponents": paw.get("expectedMarkComponents", 0),
            "observedMarkComponents": paw.get("observedMarkComponents", len(marks)),
            "componentCount": len(paw.get("componentIndices", [])),
            "bodyCount": len(body), "markCount": len(marks),
            "cost": paw.get("cost", 0), "markClasses": classes,
            "markMasks": [self._mark_record(ref, paw_index, index, mask)
                          for index, mask in enumerate(masks)],
            "flags": flags, "review": self._review("paw", review_id),
            "imageUrl": self._artifact_url(paw.get("pngPath"), page_dir, review_id),
            "bodyUrl": self._artifact_url(paw.get("bodyPath"), page_dir, review_id),
            "marksUrl": self._artifact_url(paw.get("marksPath"), page_dir, review_id),
        }

    def _mark_record(self, ref: dict, paw_index: int, mask_index: int, mask: dict) -> dict:
        review_id = f"{ref['wordKey']}:paw{paw_index:02d}:mark{mask_index:02d}"
        page_dir = ref["page_dir"]
        components = list(mask.get("componentIndices", []))
        return {
            "reviewId": review_id, "wordKey": ref["wordKey"],
            "wordId": ref["wordId"], "pageId": ref["page"]["id"],
            "pageNumber": ref["page"]["number"], "pawIndex": paw_index,
            "className": mask.get("className", "other"),
            "componentIndices": components,
            "expectedText": ref["word"].get("expectedText", ""),
            "expectedWordCount": next(
                (item.get("count", 0) for item in ref["word"].get("expectedMarks", [])
                 if item.get("className") == mask.get("className")), 0),
            "imageUrl": self._artifact_url(mask.get("path"), page_dir, review_id),
            "review": self._review("mark", review_id),
        }

    def page_response(self, page_id: str) -> dict | None:
        page = self.pages.get(page_id)
        if page is None:
            return None
        return {key: value for key, value in page.items() if key != "path"}

    def word_response(self, word_key: str) -> dict | None:
        ref = self.words.get(word_key)
        if ref is None:
            return None
        word = ref["word"]
        paws = [self._paw_record(ref, index, paw)
                for index, paw in enumerate(paw_list(word))]
        detected = Counter(mask.get("className", "other")
                           for paw in paw_list(word)
                           for mask in paw.get("markMasks", []))
        result = {
            "wordKey": word_key, "wordId": ref["wordId"],
            "pageId": ref["page"]["id"], "pageNumber": ref["page"]["number"],
            "lineIndex": word.get("lineIndex", 0), "wordIndex": word.get("wordIndex", 0),
            "expectedText": word.get("expectedText", word.get("text", "")),
            "componentCount": len(word.get("componentIndices", [])),
            "markComponentCount": sum(len(p["markComponentIndices"]) for p in paws),
            "expectedMarkComponentCount": word.get("expectedMarkComponents", 0),
            "expectedMarks": word.get("expectedMarks", []),
            "detectedMarkCounts": [{"className": key, "count": value}
                                   for key, value in sorted(detected.items())],
            "flags": self._flags_for_word(word, paws),
            "imageUrl": self._artifact_url(word.get("pngPath"), ref["page_dir"], word_key),
            "paws": paws,
        }
        return result

    def bootstrap(self) -> dict:
        groups: dict[str, dict] = {}
        for mark in self.marks:
            group = groups.setdefault(mark["className"],
                                      {"className": mark["className"], "count": 0,
                                       "words": set(), "paws": set()})
            group["count"] += 1
            group["words"].add(mark["wordKey"])
            group["paws"].add(f"{mark['wordKey']}:paw{mark['pawIndex']:02d}")
        mark_groups = [{"className": value["className"], "count": value["count"],
                        "words": len(value["words"]), "paws": len(value["paws"])}
                       for value in groups.values()]
        mark_groups.sort(key=lambda value: (-value["count"], value["className"]))
        pages = [{key: value for key, value in page.items() if key != "path"}
                 for page in sorted(self.pages.values(), key=lambda value: value["number"])]
        return {"pages": pages, "markGroups": mark_groups,
                "wordCount": len(self.words), "pawCount": len(self.paws),
                "markCount": len(self.marks)}

    def mark_response(self, query: dict) -> dict:
        class_name = query.get("class", [""])[0]
        status = query.get("status", [""])[0]
        needle = query.get("q", [""])[0].lower()
        records = [mark for mark in self.marks
                   if (not class_name or mark["className"] == class_name)
                   and (not status or mark["review"].get("status", "unreviewed") == status)
                   and (not needle or needle in " ".join((mark["pageId"], mark["wordId"],
                                                            mark["expectedText"],
                                                            mark["className"],
                                                            ",".join(map(str, mark["componentIndices"])))).lower())]
        offset = max(0, int(query.get("offset", ["0"])[0]))
        return {"total": len(records), "offset": offset, "records": records[offset:offset + 250]}

    def paw_response(self, query: dict) -> dict:
        filter_name = query.get("filter", ["all"])[0]
        status = query.get("status", [""])[0]
        needle = query.get("q", [""])[0].lower()
        records = []
        for paw in self.paws:
            flags = {flag["label"] for flag in paw["flags"]}
            if filter_name == "empty-body" and "empty body" not in flags:
                continue
            if filter_name == "multi-body" and "multiple bodies" not in flags:
                continue
            if filter_name == "other-mark" and "other mark" not in flags:
                continue
            if filter_name == "count-mismatch" and "mark count mismatch" not in flags:
                continue
            if status and paw["review"].get("status", "unreviewed") != status:
                continue
            haystack = " ".join((paw["pageId"], paw["wordId"], paw["text"],
                                 paw["expectedText"], ",".join(paw["markClasses"]))).lower()
            if needle and needle not in haystack:
                continue
            records.append(paw)
        return {"total": len(records), "records": records}

    def update_review(self, kind: str, review_id: str, status: str, note: str) -> dict:
        if kind not in {"mark", "paw"}:
            raise ValueError("review kind must be mark or paw")
        if status not in REVIEW_STATUSES:
            raise ValueError("invalid review status")
        if not any(item["reviewId"] == review_id for item in
                   (self.marks if kind == "mark" else self.paws)):
            raise ValueError("unknown review item")
        item = {"status": status, "note": str(note or ""),
                "updatedAt": dt.datetime.now(dt.timezone.utc).isoformat()}
        self.reviews[f"{kind}:{review_id}"] = item
        records = self.marks if kind == "mark" else self.paws
        for record in records:
            if record["reviewId"] == review_id:
                record["review"] = item
                break
        self.review_file.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.review_file.parent,
                                         prefix=".segmentation-review-", delete=False) as handle:
            json.dump({"schemaVersion": 1, "items": self.reviews}, handle,
                      ensure_ascii=False, indent=2)
            handle.write("\n")
            temporary = handle.name
        os.replace(temporary, self.review_file)
        return item

    def validation_summary(self) -> dict:
        return {"pages": len(self.pages), "words": len(self.words),
                "paws": len(self.paws), "marks": len(self.marks),
                "registeredArtifacts": len(self.artifacts),
                "missingArtifacts": self.missing_artifacts[:20]}


class Handler(BaseHTTPRequestHandler):
    index: SegmentationIndex | None = None

    def _write(self, data: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, value: object, status: int = 200) -> None:
        self._write(json.dumps(value, ensure_ascii=False).encode(),
                    "application/json; charset=utf-8", status)

    def do_GET(self) -> None:  # noqa: N802
        assert self.index is not None
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                self._write(HTML.encode(), "text/html; charset=utf-8")
            elif parsed.path == "/image":
                artifact = self.index.artifacts.get(query.get("id", [""])[0])
                if artifact is None:
                    self.send_error(404)
                else:
                    self._write(artifact.read_bytes(),
                                mimetypes.guess_type(artifact.name)[0] or "image/png")
            elif parsed.path == "/api/bootstrap":
                self._json(self.index.bootstrap())
            elif parsed.path == "/api/page":
                value = self.index.page_response(query.get("id", [""])[0])
                self._json(value if value is not None else {"error": "page not found"},
                           200 if value is not None else 404)
            elif parsed.path == "/api/word":
                value = self.index.word_response(query.get("key", [""])[0])
                self._json(value if value is not None else {"error": "word not found"},
                           200 if value is not None else 404)
            elif parsed.path == "/api/marks":
                self._json(self.index.mark_response(query))
            elif parsed.path == "/api/paws":
                self._json(self.index.paw_response(query))
            else:
                self.send_error(404)
        except (OSError, ValueError, KeyError) as error:
            self._json({"error": str(error)}, 500)

    def do_POST(self) -> None:  # noqa: N802
        assert self.index is not None
        if self.path != "/api/review":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            value = self.index.update_review(payload.get("kind", ""),
                                             payload.get("id", ""),
                                             payload.get("status", ""),
                                             payload.get("note", ""))
            self._json({"ok": True, "review": value})
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self._json({"error": str(error)}, 400)

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="output root containing page*/words.json")
    parser.add_argument("--review-file", type=Path,
                        help="review JSON path; defaults to <root>/.segmentation-review.json")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8878)
    parser.add_argument("--check", action="store_true",
                        help="validate the exported JSON/artifact paths and exit")
    args = parser.parse_args()
    index = SegmentationIndex(args.root, args.review_file)
    summary = index.validation_summary()
    if args.check:
        print(json.dumps(summary, ensure_ascii=False))
        if not summary["pages"] or summary["missingArtifacts"]:
            return 1
        return 0
    if not summary["pages"]:
        parser.error(f"no page*/words.json files found below {args.root}")
    Handler.index = index
    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    print(f"Segmentation review GUI: http://{args.bind}:{args.port}/")
    print(f"Loaded {summary['pages']} pages, {summary['words']} words, "
          f"{summary['paws']} PAWs, {summary['marks']} mark masks")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
