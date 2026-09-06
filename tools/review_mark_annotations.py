#!/usr/bin/env python3
"""Local browser UI for reviewing mark-component annotations."""

import argparse
import json
import mimetypes
import os
import pathlib
import tempfile
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


MARK_CLASSES = [
    "dot_above_one",
    "dot_below_one",
    "dot_above_two",
    "dot_below_two",
    "dot_above_three",
    "fatha",
    "kasra",
    "damma",
    "fathatan",
    "kasratan",
    "dammatan",
    "shadda",
    "sukun",
    "maddah",
    "hamza_above",
    "hamza_below",
    "small_alef",
    "small_yeh",
    "small_waw",
    "small_meem",
    "small_noon",
    "small_seen",
    "wasla",
    "waqf_lam",
    "waqf_qaf",
    "waqf_jeem",
    "waqf_sad",
    "waqf_meem",
    "waqf_three_dots",
    "quranic_sign",
    "body_fragment",
    "noise",
    "other",
    "unknown",
]

SEMANTIC_CLASSES = [
    mark_class
    for mark_class in MARK_CLASSES
    if mark_class not in ("body_fragment", "noise", "other", "unknown")
]

PART_ROLES = [
    "whole",
    "body",
    "dot",
    "upper",
    "lower",
    "left",
    "right",
    "stroke",
    "other",
]

# Template roles become subdirectories below a semantic class. "open" is
# used by the kasratan recognizer for the shifted-left lower-bar form; other
# semantic classes normally keep the default "whole" role.
TEMPLATE_ROLES = ["whole", "open"]

ORIENTATION_VARIANTS = ["unknown", "horizontal", "vertical", "diagonal"]
CORRECTION_ACTIONS = ["keep", "reassign", "noise", "merge", "split"]


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mushaf mark review</title>
<style>
:root { color-scheme: dark; font-family: system-ui, sans-serif; }
body { margin: 0; background: #11151a; color: #e9eef5; }
header { display:flex; gap:12px; align-items:center; padding:12px 18px;
  position:sticky; top:0; background:#182029; border-bottom:1px solid #34404c; z-index:2; }
button, select, input { font:inherit; color:inherit; background:#25313d;
  border:1px solid #4a5a69; border-radius:6px; padding:7px 10px; }
button:hover { background:#344556; } button.primary { background:#176b45; }
main { max-width:1200px; margin:auto; padding:20px; display:grid;
  grid-template-columns:minmax(420px, 1.3fr) minmax(330px, .7fr); gap:24px; }
.panel { background:#192129; border:1px solid #34404c; border-radius:10px; padding:18px; }
.images { display:grid; grid-template-columns:1fr 2fr; gap:14px; align-items:center; }
.imagebox { background:#000; min-height:260px; display:flex; align-items:center;
  justify-content:center; position:relative; border-radius:8px; overflow:hidden; }
.imagebox img { image-rendering:auto; max-width:100%; max-height:520px; }
.cross::before,.cross::after { content:""; position:absolute; background:#ff3157; opacity:.7; }
.cross::before { width:28px; height:2px; left:calc(50% - 14px); top:50%; }
.cross::after { height:28px; width:2px; top:calc(50% - 14px); left:50%; }
.arabic { direction:rtl; font-size:34px; line-height:1.7; text-align:center; margin:8px; }
.marks { display:flex; flex-wrap:wrap; gap:6px; }
.badge { background:#293746; border-radius:12px; padding:4px 9px; font-size:13px; }
label { display:block; margin-top:14px; color:#bdc9d4; }
select { width:100%; margin-top:5px; }
.checks { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:14px; }
.checks label { margin:0; background:#222d37; padding:9px; border-radius:6px; }
.checks input { margin-right:7px; }
.class-grid { max-height:230px; overflow:auto; display:grid; grid-template-columns:1fr 1fr; gap:3px; }
.class-grid label { margin:0; font-size:13px; }
.meta { color:#9fb0bf; font-family:ui-monospace,monospace; font-size:13px; white-space:pre-wrap; }
.status { margin-left:auto; color:#a8d9bc; }
@media(max-width:850px) { main { grid-template-columns:1fr; }.images{grid-template-columns:1fr;} }
</style>
</head>
<body>
<header>
  <button id="prev">← Previous</button><button id="next">Next →</button>
  <select id="filter" style="width:auto">
    <option value="needs_review">Needs review</option>
    <option value="proposed">Automatic proposals</option>
    <option value="unreviewed">All unreviewed</option><option value="all">All</option>
    <option value="reviewed">Reviewed</option>
    <option value="auto_accepted">Auto-accepted</option>
  </select>
  <span id="position"></span><span class="status" id="status"></span>
</header>
<main>
  <section class="panel">
    <div class="images">
      <div><h3>Candidate mask</h3><div class="imagebox"><img id="mask"></div></div>
      <div><h3>Line context</h3><div class="imagebox cross"><img id="context"></div></div>
    </div>
    <h3>Whole-word proposal</h3>
    <div class="imagebox" style="min-height:180px"><svg id="wordOverview"
      style="width:100%;max-height:300px"></svg></div>
    <div style="display:flex;gap:8px;margin-top:8px">
      <button class="primary" id="acceptWord">Accept whole word</button>
      <button id="rejectWord">Reject proposal</button>
    </div>
    <div class="arabic" id="text"></div>
    <div class="marks" id="expected"></div>
    <div class="marks" id="reasons"></div>
    <pre class="meta" id="meta"></pre>
  </section>
  <section class="panel">
    <label>Primary visual class<select id="visualClass"></select></label>
    <button id="acceptSuggestion" type="button">Accept classifier suggestion</button>
    <label>Semantic mark<select id="semanticClass"></select></label>
    <label>Shared instance/group ID
      <div style="display:flex;gap:6px">
        <input id="groupId" style="width:100%" placeholder="e.g. p560_l04_w03_waqf_jeem_1">
        <button id="generateGroup" type="button">Generate</button>
      </div>
    </label>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px">
      <label>Part role<select id="partRole"></select></label>
      <label>Template role<select id="templateRole"></select></label>
      <label>Part index<input id="partIndex" type="number" min="1" value="1" style="width:100%"></label>
      <label>Part count<input id="partCount" type="number" min="1" value="1" style="width:100%"></label>
    </div>
    <label>Orientation variant<select id="orientationVariant"></select></label>
    <label>Segmentation correction<select id="correctionAction"></select></label>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
      <label>Corrected line index<input id="correctedLineIndex" type="number"
        min="-1" value="-1" style="width:100%"></label>
      <label>Corrected word ID<input id="correctedWordId" style="width:100%"
        placeholder="e.g. l04_w09"></label>
    </div>
    <label>Merged visual classes</label><div class="class-grid" id="merged"></div>
    <div class="checks">
      <label><input type="checkbox" id="touchesBody">Touches body</label>
      <label><input type="checkbox" id="isNoise">Noise</label>
    </div>
    <label>Owning character index (-1 if unknown)
      <input id="characterIndex" type="number" value="-1" style="width:100%"></label>
    <label>Reviewer note<input id="note" style="width:100%"></label>
    <div style="display:flex;gap:8px;margin-top:18px">
      <button class="primary" id="save">Save & next (S)</button>
      <button id="skip">Skip (K)</button>
      <button id="unreview">Mark unreviewed</button>
    </div>
  </section>
</main>
<script>
const classes = __CLASSES__, semanticClasses = __SEMANTIC_CLASSES__,
  partRoles = __PART_ROLES__, orientationVariants = __ORIENTATION_VARIANTS__,
  correctionActions = __CORRECTION_ACTIONS__, templateRoles = __TEMPLATE_ROLES__;
let state = {index:0, record:null, total:0};
const $ = id => document.getElementById(id);
$("visualClass").innerHTML = '<option value="">— choose —</option>' +
  classes.map(x=>`<option>${x}</option>`).join('');
$("semanticClass").innerHTML = '<option value="">— none / same as visual —</option>' +
  semanticClasses.map(x=>`<option>${x}</option>`).join('');
$("partRole").innerHTML = partRoles.map(x=>`<option>${x}</option>`).join('');
$("templateRole").innerHTML = templateRoles.map(x=>`<option>${x}</option>`).join('');
$("orientationVariant").innerHTML =
  orientationVariants.map(x=>`<option>${x}</option>`).join('');
$("correctionAction").innerHTML =
  correctionActions.map(x=>`<option>${x}</option>`).join('');
$("merged").innerHTML = classes.map((x,i)=>
  `<label><input type="checkbox" data-class="${x}">${x}</label>`).join('');
function filter(){ return $("filter").value; }
async function load(direction=0, explicit=null) {
  const query = new URLSearchParams({filter:filter(), direction, index: explicit ?? state.index});
  const response = await fetch('/api/record?'+query);
  if (!response.ok) { $("status").textContent='No matching records'; return; }
  state = await response.json(); const r=state.record;
  $("position").textContent=`${state.filteredPosition+1}/${state.filteredTotal} · all ${state.index+1}/${state.total}`;
  $("mask").src='/image?path='+encodeURIComponent(r.maskPath)+'&v='+Date.now();
  $("context").src='/image?path='+encodeURIComponent(r.contextPath)+'&v='+Date.now();
  $("text").textContent=r.expectedText || '—';
  $("expected").innerHTML=(r.expectedMarks||[]).map(
    x=>`<span class="badge">${x.className} × ${x.count}</span>`).join('');
  $("reasons").innerHTML=(r.reviewReasons||[]).map(
    x=>`<span class="badge">${x.replaceAll('_',' ')}</span>`).join('');
  $("meta").textContent=`${r.id}\npage ${r.page} · ${r.wordId}\ncomponent ${r.componentIndex}\n`+
    `${r.geometry.width}×${r.geometry.height}, area ${r.geometry.area}\nsplit: ${r.split}\n`+
    `suggested: ${r.suggestedVisualClass||'—'} · confidence ${(100*(r.classifierConfidence||0)).toFixed(1)}%`+
    ` · margin ${(100*(r.classifierMargin||0)).toFixed(1)}%\n`+
    `assignment uniqueness: ${(100*(r.autoAssignmentConfidence||0)).toFixed(1)}%`+
    ` · priority: ${r.reviewPriority||0}`;
  $("visualClass").value=r.visualClass || '';
  $("semanticClass").value=r.semanticClass || '';
  $("groupId").value=r.instanceGroupId || '';
  $("partRole").value=r.partRole || 'whole';
  $("templateRole").value=r.templateRole || 'whole';
  $("partIndex").value=r.partIndex || 1; $("partCount").value=r.partCount || 1;
  $("orientationVariant").value=r.orientationVariant || 'unknown';
  $("correctionAction").value=r.correctionAction || 'keep';
  $("correctedLineIndex").value=r.correctedLineIndex ?? -1;
  $("correctedWordId").value=r.correctedWordId || r.wordId || '';
  document.querySelectorAll('#merged input').forEach(
    x=>x.checked=(r.mergedClasses||[]).includes(x.dataset.class));
  $("touchesBody").checked=r.touchesBody===true;
  $("isNoise").checked=r.isNoise===true;
  $("characterIndex").value=r.owningCharacterIndex ?? -1;
  $("note").value=r.note || '';
  $("status").textContent=r.reviewed?'Reviewed':'Unreviewed';
  renderWord(state.wordRecords||[], r);
}
function renderWord(records, selected) {
  const svg=$("wordOverview"), g=selected.wordGeometry||{};
  svg.replaceChildren();
  if(!selected.wordImagePath || !g.width || !g.height) return;
  svg.setAttribute('viewBox',`0 0 ${g.width} ${g.height}`);
  const ns='http://www.w3.org/2000/svg';
  const image=document.createElementNS(ns,'image');
  image.setAttribute('href','/image?path='+encodeURIComponent(selected.wordImagePath));
  image.setAttribute('width',g.width); image.setAttribute('height',g.height);
  svg.appendChild(image);
  for(const item of records) {
    const r=item.record, box=r.geometry;
    const rect=document.createElementNS(ns,'rect');
    rect.setAttribute('x',box.x-g.x); rect.setAttribute('y',box.y-g.y);
    rect.setAttribute('width',box.width); rect.setAttribute('height',box.height);
    rect.setAttribute('fill','none');
    rect.setAttribute('stroke',r.id===selected.id?'#ff3157':'#36e28a');
    rect.setAttribute('stroke-width',Math.max(3,g.width/350));
    rect.style.cursor='pointer'; rect.onclick=()=>load(0,item.index);
    svg.appendChild(rect);
    const label=document.createElementNS(ns,'text');
    label.setAttribute('x',box.x-g.x); label.setAttribute('y',Math.max(16,box.y-g.y-4));
    label.setAttribute('fill','#ffe36e'); label.setAttribute('font-size',Math.max(16,g.width/55));
    label.textContent=r.suggestedVisualClass||r.visualClass||'?';
    svg.appendChild(label);
  }
}
async function save(reviewed=true, advance=true) {
  const r={...state.record};
  r.visualClass=$("visualClass").value || null;
  r.semanticClass=$("semanticClass").value || null;
  r.instanceGroupId=$("groupId").value.trim();
  r.partRole=$("partRole").value;
  r.templateRole=$("templateRole").value;
  r.partIndex=Number($("partIndex").value); r.partCount=Number($("partCount").value);
  r.orientationVariant=$("orientationVariant").value;
  r.correctionAction=$("correctionAction").value;
  r.correctedLineIndex=Number($("correctedLineIndex").value);
  r.correctedWordId=$("correctedWordId").value.trim();
  r.mergedClasses=[...document.querySelectorAll('#merged input:checked')].map(x=>x.dataset.class);
  r.touchesBody=$("touchesBody").checked; r.isNoise=$("isNoise").checked;
  r.owningCharacterIndex=Number($("characterIndex").value);
  r.note=$("note").value; r.reviewed=reviewed;
  r.reviewStatus=reviewed?'reviewed':'needs_review';
  if(reviewed && !r.visualClass){ $("status").textContent='Choose a visual class'; return; }
  const response=await fetch('/api/record',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({index:state.index,record:r})});
  if(!response.ok){
    const error=await response.json().catch(()=>({error:'Save failed'}));
    $("status").textContent=error.error || 'Save failed'; return;
  }
  $("status").textContent='Saved'; if(advance) await load(1);
}
$("prev").onclick=()=>load(-1); $("next").onclick=()=>load(1);
$("generateGroup").onclick=()=>{
  const semantic=$("semanticClass").value || $("visualClass").value || 'mark';
  $("groupId").value=`${state.record.id}_${semantic}`;
};
$("acceptSuggestion").onclick=()=>{
  if(state.record.suggestedVisualClass)
    $("visualClass").value=state.record.suggestedVisualClass;
};
async function decideWord(action){
  if(!state.record.wordProposalId){ $("status").textContent='No word proposal'; return; }
  const response=await fetch('/api/word-proposal',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({wordProposalId:state.record.wordProposalId,action})});
  if(!response.ok){ const e=await response.json(); $("status").textContent=e.error; return; }
  await load(1);
}
$("acceptWord").onclick=()=>decideWord('accept');
$("rejectWord").onclick=()=>decideWord('reject');
$("correctionAction").onchange=()=>{
  if($("correctionAction").value==='noise'){
    $("visualClass").value='noise'; $("isNoise").checked=true;
  }
};
$("save").onclick=()=>save(true,true); $("skip").onclick=()=>load(1);
$("unreview").onclick=()=>save(false,false); $("filter").onchange=()=>load(0,0);
document.addEventListener('keydown',e=>{
  if(e.target.matches('input,select')) return;
  if(e.key==='ArrowRight'||e.key==='k') load(1);
  if(e.key==='ArrowLeft'||e.key==='j') load(-1);
  if(e.key==='s') save(true,true);
});
load();
</script>
</body></html>"""


class AnnotationStore:
    def __init__(self, path):
        self.path = path.resolve()
        self.records = [
            json.loads(line)
            for line in self.path.read_text().splitlines()
            if line.strip()
        ]
        for record in self.records:
            record.setdefault("semanticClass", None)
            record.setdefault("instanceGroupId", "")
            record.setdefault("partRole", "whole")
            record.setdefault("templateRole", "whole")
            record.setdefault("partIndex", 1)
            record.setdefault("partCount", 1)
            record.setdefault("orientationVariant", "unknown")
            record.setdefault("correctionAction", "keep")
            record.setdefault("correctedLineIndex", record.get("lineIndex", -1))
            record.setdefault("correctedWordId", record.get("wordId", ""))
            record.setdefault("suggestedVisualClass", None)
            record.setdefault("classifierConfidence", 0.0)
            record.setdefault("classifierMargin", 0.0)
            record.setdefault("reviewReasons", ["priority_not_computed"])
            record.setdefault("reviewPriority", 0)
            record.setdefault(
                "reviewStatus",
                "reviewed" if record.get("reviewed") else "needs_review",
            )
        self.allowed_images = {
            pathlib.Path(record[key]).resolve()
            for record in self.records
            for key in ("maskPath", "contextPath")
            if record.get(key)
        }
        self.allowed_images.update(
            pathlib.Path(record["wordImagePath"]).resolve()
            for record in self.records
            if record.get("wordImagePath")
        )

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=self.path.name + ".", dir=self.path.parent
        )
        try:
            with os.fdopen(descriptor, "w") as output:
                for record in self.records:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def make_handler(store):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, value, status=200):
            data = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", len(data))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(parsed.query)
            if parsed.path == "/":
                data = (
                    HTML.replace("__CLASSES__", json.dumps(MARK_CLASSES))
                    .replace("__SEMANTIC_CLASSES__", json.dumps(SEMANTIC_CLASSES))
                    .replace("__PART_ROLES__", json.dumps(PART_ROLES))
                    .replace("__TEMPLATE_ROLES__", json.dumps(TEMPLATE_ROLES))
                    .replace(
                        "__ORIENTATION_VARIANTS__",
                        json.dumps(ORIENTATION_VARIANTS),
                    )
                    .replace(
                        "__CORRECTION_ACTIONS__",
                        json.dumps(CORRECTION_ACTIONS),
                    )
                    .encode()
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", len(data))
                self.end_headers()
                self.wfile.write(data)
                return
            if parsed.path == "/image":
                path = pathlib.Path(query.get("path", [""])[0]).resolve()
                if path not in store.allowed_images or not path.is_file():
                    self.send_error(404)
                    return
                data = path.read_bytes()
                self.send_response(200)
                self.send_header(
                    "Content-Type", mimetypes.guess_type(path)[0] or "image/png"
                )
                self.send_header("Content-Length", len(data))
                self.end_headers()
                self.wfile.write(data)
                return
            if parsed.path == "/api/record":
                mode = query.get("filter", ["unreviewed"])[0]
                selected = [
                    index
                    for index, record in enumerate(store.records)
                    if mode == "all"
                    or (mode == "reviewed" and record.get("reviewed"))
                    or (mode == "unreviewed" and not record.get("reviewed"))
                    or (
                        mode == "needs_review"
                        and record.get("reviewStatus") == "needs_review"
                    )
                    or (
                        mode == "auto_accepted"
                        and record.get("reviewStatus") == "auto_accepted"
                    )
                    or (
                        mode == "proposed"
                        and record.get("reviewStatus") == "proposed"
                    )
                ]
                if mode == "needs_review":
                    selected.sort(
                        key=lambda index: store.records[index].get(
                            "reviewPriority", 0
                        ),
                        reverse=True,
                    )
                if not selected:
                    self.send_json({"error": "no records"}, 404)
                    return
                current = int(query.get("index", [selected[0]])[0])
                direction = int(query.get("direction", [0])[0])
                if current in selected:
                    position = selected.index(current)
                else:
                    position = 0
                position = (position + direction) % len(selected)
                index = selected[position]
                selected_record = store.records[index]
                word_records = [
                    {"index": candidate_index, "record": record}
                    for candidate_index, record in enumerate(store.records)
                    if record.get("page") == selected_record.get("page")
                    and record.get("wordId") == selected_record.get("wordId")
                ]
                self.send_json(
                    {
                        "index": index,
                        "total": len(store.records),
                        "filteredPosition": position,
                        "filteredTotal": len(selected),
                        "record": store.records[index],
                        "wordRecords": word_records,
                    }
                )
                return
            self.send_error(404)

        def do_POST(self):
            if self.path == "/api/word-proposal":
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                proposal_id = payload.get("wordProposalId")
                action = payload.get("action")
                matching = [
                    record for record in store.records
                    if record.get("wordProposalId") == proposal_id
                    and record.get("reviewStatus") == "proposed"
                ]
                if action not in ("accept", "reject") or not matching:
                    self.send_json({"error": "invalid or empty word proposal"}, 400)
                    return
                for record in matching:
                    if action == "accept":
                        record["visualClass"] = record.get(
                            "suggestedVisualClass"
                        )
                        record["reviewed"] = True
                        record["reviewStatus"] = "reviewed"
                        record["note"] = (
                            record.get("note", "")
                            + " accepted constrained word proposal"
                        ).strip()
                    else:
                        record["reviewStatus"] = "needs_review"
                        record["reviewReasons"] = ["automatic_proposal_rejected"]
                        record.pop("wordProposalId", None)
                store.save()
                self.send_json({"updated": len(matching), "action": action})
                return
            if self.path != "/api/record":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            index = int(payload["index"])
            record = payload["record"]
            if record.get("visualClass") not in MARK_CLASSES + [None]:
                self.send_json({"error": "invalid visual class"}, 400)
                return
            if record.get("semanticClass") not in SEMANTIC_CLASSES + [None]:
                self.send_json({"error": "invalid semantic class"}, 400)
                return
            if record.get("templateRole", "whole") not in TEMPLATE_ROLES:
                self.send_json({"error": "invalid template role"}, 400)
                return
            if record.get("orientationVariant") not in ORIENTATION_VARIANTS:
                self.send_json({"error": "invalid orientation variant"}, 400)
                return
            if record.get("correctionAction") not in CORRECTION_ACTIONS:
                self.send_json({"error": "invalid correction action"}, 400)
                return
            if (
                record.get("correctionAction") == "reassign"
                and (
                    int(record.get("correctedLineIndex", -1)) < 0
                    or not record.get("correctedWordId")
                )
            ):
                self.send_json(
                    {"error": "reassignment requires a target line and word"}, 400
                )
                return
            part_count = int(record.get("partCount", 1))
            part_index = int(record.get("partIndex", 1))
            if (
                record.get("partRole", "whole") not in PART_ROLES
                or part_count < 1
                or part_index < 1
                or part_index > part_count
            ):
                self.send_json({"error": "invalid component part metadata"}, 400)
                return
            if part_count > 1 and (
                not record.get("semanticClass")
                or not record.get("instanceGroupId")
            ):
                self.send_json(
                    {
                        "error": "multi-component marks require semantic class and group ID"
                    },
                    400,
                )
                return
            if record["id"] != store.records[index]["id"]:
                self.send_json({"error": "record identity mismatch"}, 409)
                return
            store.records[index] = record
            store.save()
            self.send_json({"saved": True})

        def log_message(self, message, *args):
            print(message % args)

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=pathlib.Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    store = AnnotationStore(args.annotations)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    url = f"http://{args.host}:{args.port}/"
    print(f"Reviewing {len(store.records)} records at {url}")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
