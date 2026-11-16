# PAW letter segmentation — handoff

Updated: 2026-08-09. Read this before changing `segment_paw_letters.py`.

## What this is

Cuts a PAW (Piece of Arabic Word — one connected run of ink) into its
individual letters, using the letter sequence the text already gives us. It
serves the goal in `mushafanalyzer/README.md`: a searchable corpus of the Old
Madinah Mushaf measured letter by letter, supporting queries like "all initial
beh sorted by width".

It replaces `estimate_intervals()` in `tools/build_instance_index.py`, which
splits ink at equal cumulative mass. That fails structurally: ink mass is not
proportional to letter width, column projection cannot separate letters whose
descenders interleave in x, and it never uses the known letter sequence.

Run it:

```sh
uv run tools/paw_segmentation/segment_paw_letters.py \
  --analysis output/oldmedina2-analysis --page 3 --batch \
  --out output/paw-segmentation-page3
```

Single word instead of `--batch`: `--word l13_w07`. VS Code launch
configurations exist for both, plus the prior trainer.

## Pipeline

1. **PAWs come from the analyzer.** `words.json` carries a `paws` array per
   word with `bodyPath`, `text`, `x`, `y`. Each body is pasted back at its
   offset inside the padded word canvas, which reproduces the word body
   pixel-for-pixel, so every coordinate downstream is in word space. The old
   path — connected components plus `align_paws` ink-mass alignment — is kept
   for analysis output without a `paws` array, but it was responsible for 34
   broken-stroke abstentions and 17 merged PAWs on page 3, all of which
   disappeared when the analyzer started emitting this.
2. **Contour → generators.** `contour_generators` resamples and smooths each
   contour. Note this makes the points *coarser* than the raw 1px contour
   (5.4x fewer on the PAW measured); the purpose is denoising, not accuracy —
   raw contours give ~40% more medial vertices, mostly staircase noise. Cost
   is about 1.3% on the radius, which is negligible.
3. **Medial axis.** `scipy.spatial.Voronoi` over the generators; keep vertices
   on ink with radius above `MIN_MEDIAL_RADIUS_W`; prune short leaves; take
   the largest connected component. Radius per vertex is a `cKDTree` query
   against the generators, which is the distance to the nearest boundary
   point, i.e. the maximal inscribed disk.
4. **Spine.** Dijkstra shortest path between the rightmost and leftmost
   substantial vertices. **Both endpoints sit at baseline level, so no upright
   is ever on the spine** — see the traps below, this matters more than it
   sounds.
5. **Cut-cost profile.** Project every medial vertex, then every ink pixel, to
   its graph-nearest spine arclength. At each spine sample, cast a ray along
   the normal and separately record the central run (`core_cut_len`), all ink
   attached within 4 stroke widths along the spine (`owned_cut_len`), and ink
   attached elsewhere (`foreign_cut_len`). Production `cut_len` is the owned
   length plus at most 0.75 stroke widths of foreign ink; this bounded term
   records ambiguity without letting remote letters dominate the result.
6. **Candidates.** The established full-ray candidate pool is retained as a
   fallback, and ownership-aware `min` / `shoulder` / `sample` / `edge` picks
   are added. Local-only candidates pay a small uncertainty penalty, so they
   must materially beat an established candidate rather than win a near tie.
7. **DP** normally picks exactly k-1 cuts for k letters, maximising
   per-candidate score plus segment penalties, subject to two length floors.
   In opt-in logical-ligature mode, a positively matched composite template
   may consume multiple letters without an internal physical cut.

## The three evidence signals, and why there are three

Each answers a question the others cannot. Adding a fourth is not obviously
wrong, but read this first — most of the work below was discovering that a
signal which looks right is measuring something else.

- **Cut cost** (`core_cut_len`, `owned_cut_len`, bounded `cut_len`) — how much
  topologically local ink a cut severs. The unbounded full-ray measurement is
  diagnostic/context evidence only and cannot reject a candidate by itself.
- **Branches** (`branch_positions`, `branch_attachments`) — degree-3 vertices
  plus the junctions `_prune_leaves` removed. Marks real attachments: an
  upright joining the connector, a bowl rim.
- **Spine rises** (`spine_rises`) — where the spine lifts off the baseline.
  Marks teeth.

**A tooth produces no branch.** It is drawn as a parallel excursion of both
contours — the stroke keeps its width through it — so the medial axis runs
through as a single curve and forks nowhere. Only a real attachment forks.
This is why rises exist as a separate signal, and it is worth internalising
before proposing anything branch-based for teeth.

## Constraint tables

- `TOOTH_COUNT` — internal necks per letter (seen/sheen have 2). Script
  knowledge, from the original Fix 1a.
- `BRANCH_EXPECT` — `(min, max)` branches, keyed `"letter|form"` with a
  `"letter|*"` fallback. **Form matters more than letter**: an upright counts
  as a branch only when the spine passes it, so a medial lam's stem forks
  mid-run while an initial lam's hangs off the spine's own start vertex
  (degree 2, invisible). Page 3: initial lam 96.8% zero against medial lam
  84.6% one; initial noon 90.9% zero against medial noon 88.9% one.
- `RISE_EXPECT` — `(min, max)` spine lifts, same keying. Medial single-tooth
  letters and medial ayn take exactly 1; sad and dad take 3. Medial heh has an
  upper bound of 1 but no lower bound because only 13/20 detected a lift;
  other heh forms take exactly 1. Final kaf takes one rise near its segment
  entry, so a preceding yeh cannot steal kaf's body.
- `RISE_CLUSTER_EXPECT` — medial seen/sheen own a local cluster of three tooth
  rises. The rule abstains when no three-rise cluster is detectable anywhere
  in the PAW. When present, the segment must own the complete cluster and end
  after its closing valley. Medial ayn's one-rise rule prevents sheen from
  borrowing ayn's body lift in `يشعر`.
- `RISE_BOUNDARY_FORMS` — a final zay boundary is scored by proximity to the
  nearest rise, irrespective of which side owns it. All three page-3 final
  zays place that boundary within 0.3 stroke widths of a rise. This anchors
  heh|zay without incorrectly requiring every zay to contain a rise.

Entries are present only where page 3 is one-sided with enough samples. Forms
that came out mixed (final noon at 62%, initial beh, initial qaf) are
deliberately absent — guessing there fires on correct output.

## Traps, all of which cost real time

**The spine is not the letter.** `_extract_spine` runs between two
baseline-level endpoints, and climbing a stem and back is strictly longer than
staying low, so uprights are branches off the spine, never on it. The initial
lam of `ٱلَّذِينَ` measures 1.20 stroke widths of spine while standing 8.06
tall. Any size measured along the spine alone is wrong for such letters — use
`segment_extent`, which adds `branch_attachments`.

**But extent alone lets degenerate segments through.** A 0.9-stroke-width meem
cleared the 1.5 floor by containing the *dal's* branch. Hence
`MIN_SPINE_RUN_W` alongside `MIN_SEGMENT_W`; both are checked.

**A lower bound is satisfiable by grabbing a neighbour's evidence.** This is
the single most recurrent failure. `ص|* = (4, None)` made a 3-branch sad reach
past the following lam and take its upright, producing a "sad" 12.8 stroke
widths wide and 7.8 tall. A medial heh took the following teh's tooth as a
second lift, leaving that teh a 1.25-stroke-width sliver. When a letter looks
too big, suspect a lower bound before suspecting the geometry.

**Calibrating from the segmenter's own output is circular.** The `(4, None)`
for sad/dad came from counting branches on segmentations that were themselves
wrong. Measure on *validated* instances, or on a quantity the current
segmentation cannot bias.

**Stale bytecode.** A pre-move copy of these modules exists under
`tools/tracing/__pycache__`. It silently produced a different page-3 result
(96/17 instead of 97/16) for several measurements. Clear caches before any
before/after comparison:

```sh
find tools -name __pycache__ -type d -exec rm -rf {} +
```

**The accepted count is now anti-correlated with correctness.** These
constraints exist because the right cut scores *worse* geometrically than the
wrong one — that is why it needed a constraint. So confidence falls as
correctness rises. `تنذ` is correct at 0.670 and `مهتد` at 0.69, both under
the 0.70 bar. Never tune against the accepted count alone; check cuts and
letter bounding boxes.

**Verify with a regression set, not the target word.** A commit that measured
only its target words plus the page totals silently regressed `يمد`. The
validated words are listed below.

## Current state, page 3

151 multi-letter PAWs: 150 segmented, **120 accepted**, 30 low-confidence,
all 151 with a sufficient candidate pool (the remaining failure is a DP
constraint, not a candidate shortage). Letters per PAW concatenate to each
word's base letters for all 138 words — this invariant has held throughout and
is the cheapest sanity check available.

Validated words, all correct simultaneously (use as the regression set):

| word | PAW | cuts |
|---|---|---|
| `l12_w08` | يمد | 239.2, 717.5 |
| `l13_w05` | لذ | 113.2 |
| `l13_w05` | ين | 131.4 |
| `l00_w01` | ين | 129.6 |
| `l13_w07` | لضللة | 208.7, 1305.4, 1873.1, 2125.5 |
| `l14_w01` | فما | 607.2, 938.4 |
| `l07_w07` | مصلحو | 249.2, 1094.5, 1602.6, 2325.7 |
| `l14_w06` | مهتد | 587.1, 1255.3, 1627.6 |
| `l00_w08` | تنذ | 204.8, 537.5 |
| `l02_w04` | عظيم | 499.0, 1013.1, 1628.1 |
| `l03_w00` | من | 254.9 |
| `l02_w00` | بصر | 190.7, 1239.7 |
| `l13_w06` | شتر | 918.1, 1128.2 |
| `l13_w04` | ليك | 485.6, 835.2 |
| `l12_w03` | مستهز | 227.5, 918.9, 1387.5, 1828.8 |
| `l12_w06` | يستهز | 93.1, 704.7, 1063.7, 1564.6 |
| `l07_w01` | تفسد | 325.7, 788.8, 1613.2 |

Both instances of `ٱلَّذِينَ` cutting their yeh|noon join at 129.6 and 131.4 is
a better check than any single word: the same join on the same word should
measure the same in both places.

## Skeleton shape lattice (experimental)

`segment_paw_letters.py --skeleton-shapes` now augments the local cut score
with an ordered event matcher over complete proposed letter intervals. The
event stream contains stable horizontal reversals, rises, valleys, branch
attachments, tall branch paths and forked tall branches. Short branches keep
their original polylines when pruning removes them; retaining only the old
`(anchor, length)` pair loses the geometry needed for recognition.

The initial-kaf template requires the ordered `x-right`, `x-left` reversals.
On page 3 it changes only the three PAWs where that complete motif is detected:
`l00_w02` 725.7 -> 1283.5, `l09_w02` 508.0 -> 1090.3 and `l09_w07`
629.8 -> 1208.9. Their overlays put the whole upper-left-right-left path in
kaf. No cut in the reviewed regression table above changes. With the flag
absent, all page-3 cuts are identical to the established output.

`--logical-ligatures` enables semi-Markov transitions: a recognized template
may consume several transcription letters while emitting no internal physical
cut. The strict built-in Lam-Alef motif is a fork with two substantial tall
branch paths. It recognizes `l01_w00` and `l07_w00` on page 3; their JSON has
one `segmentGroup` covering letters `[0, 2]`, no cut, shared ink marked `-2`,
and an explicit `shapeMatch`. It intentionally abstains on the other Lam-Alef
variants until reviewed examples support additional motifs.

The discoverable default definitions are in
`templates/oldmadina/skeleton_templates.json`. There are no Python template
built-ins: ordinary skeleton templates and topology-only elastic body templates
are always loaded from that document. Template `letters` entries are character
classes, so `beh-hah-family` uses `["بتثني", "جحخ"]` to cover the same
unpointed body motif in initial `تج`, `بح`, and the other dotted variants.
On `l14_w03` it replaces the false 1.3-stroke medial-Jeem interval with one
logical Teh+Jeem group `[0, 1126.4]`, leaving the only physical cut at 1126.4
before final Reh.

The contextual `beh-hah-before-final-beh` and `beh-family-final-entry`
templates were removed. Their event positions were normalized by a proposed
interval width, which makes the match change when either the body or kashida
stretches. They also still required a cut before they could recognize the
shape, making the model circular.

`--body-join-alignment` is the candidate-free replacement. It aligns complete
body states and elastic join states first. A rising kashida starts where the
dense spine begins a sustained ascent and ends at the local owned-ink/radius
neck before the following body. Foreign crossings are excluded from that neck
measurement, and join length is never scored. Topology-only composite body
templates currently cover Lam-Alef, Kaf-Alef, and the Beh-Hah family; they
require events in order but have no normalized position or body-width target.

Page-3 topology variants are explicit in the JSON. Lam-Alef accepts either a
forked `double-high-branch` or two high branches at distinct spine positions;
Kaf-Alef accepts either the Kaf zigzag or two distinct high stems. Endpoint
events at `s=0` belong to the body and are matchable. Alternative templates
that explain exactly the same body/join alignment are collapsed before margin
calculation, so a second valid description does not create false ambiguity.
These variants add `l06_w04/0`, `l07_w03/1`, `l10_w00/1`, and `l14_w05/0`.
With only those composite variants the strict candidate-free path aligns
31/151 multi-letter PAWs, 20 confidently. The remaining failures are mostly
ordinary flat or descending joins and must be addressed by additional
join-state detectors, not by falsely declaring their letters one fused body.

`l14_w06` adds the first ordered sequence of one-letter elastic body
templates: initial Meem, medial Heh, medial Teh, final Dal, initial Yeh and the
final-Noon allograph following initial Yeh. Each template owns its required
events; the transition between consecutive bodies is then derived from a
nearby entry valley, an owned-ink width shoulder, or the following body's first
event. Template-guided transitions are full join intervals, not point cuts;
their ink is therefore visible in mauve and `--cut-policy` remains meaningful.
For `مهتد` they are `310.24–587.07`, `906.86–1164.60`, and
`1408.02–1627.58`. The Teh body starts at `1164.60`, one measured stroke width
before its rise event at `1260.06`, rather than cutting on the tooth at
`1255.29`; it ends one stroke width after its terminal valley at `1312.56`.
The initial-Yeh curve becomes horizontal at `100.77`, so the visibly straight
Yeh-Noon kashida is `100.77–164.89`. The final-Noon template carries its Yeh
context because applying that allograph to Meem-Noon produced incorrect
page-3 boundaries; those PAWs continue to abstain. With these additions the
body/join path initially aligned 35/151 multi-letter PAWs, but 21 of those
alignments still contained anonymous singleton bodies and were not genuinely
template-backed.

Body ownership is now part of the elastic JSON template rather than a
letter-specific branch in the alignment engine. A template may provide
`entry` and/or `exit` with `event-offset`, `preceding-valley`, `width-entry`,
`width-exit`, or `horizontal-stabilization`. Teh explicitly uses rise `-1.0`
stroke widths for entry and valley `+1.0` for exit; initial Yeh explicitly uses
horizontal stabilization at `0.9`. Omitted boundary objects use the generic
automatic neck fallback, and omitted fields inside an object use typed
defaults. An explicit rule that cannot be observed abstains rather than
silently changing methods. These offsets are local shape ownership in stroke
widths, not ratios of the elastic body or kashida length.

In `بحت` (`l14_w02`) the resulting states are:

- fused `بح` body: `0.0–632.2`
- measured kashida: `632.2–750.2`
- final `ت` body: `750.2–1619.6`

The JSON records the kashida separately in `joins`, including its length.
`--cut-policy current-end|midpoint|next-start|none` derives a requested cut
only after alignment. Thus the default `next-start` cut is `750.2`; `none`
preserves the measured connector without inventing a cut. Join ink is labelled
`-3` in the overlay and fused body ink remains `-2`.

This new path is deliberately strict: it never calls `cut_candidates` and does
not fall back to the candidate DP when the present small event/template set is
insufficient. It therefore abstains on unsupported shapes rather than hiding
missing forced-alignment coverage behind the legacy model.

Anonymous singleton bodies have now also been removed from `_body_partitions`.
A rising stroke cannot identify a boundary by itself because it may be a tooth
inside Seen. Every body in a multi-letter PAW must therefore have a named
elastic template. Initial Seen/Sheen is represented by its ordered
`valley-rise-valley-rise-valley` tooth sequence, final Waw by its high branch
and following valley, and final Teh by its rise/valley tooth. In `سوا` on
`l00_w03`, Seen is `0.00–645.90`, the measured join is `645.90–682.29`, and
Waw starts at `682.29`; the false cut at `122.81` is no longer possible.

The page-3 coverage pass now supplies reusable elastic allographs for the
remaining observed initial, medial, and final forms. Weak terminal templates
can set `orderedMatch: false`: they may validate an interval whose physical
join bounds are already known, but cannot consume a generic valley and drive
the global letter ordering. Ordered matching enumerates a bounded set of
allograph sequences in JSON order, so a later allograph is tried when an
earlier one's explicit boundary rule cannot be observed. Recognized composite
topologies are ranked ahead of all-single explanations; this preserves the
fused Beh-family/Hah-family body in `يخد`, `تجر`, and `بحت` rather than
inventing an internal join.

The final missing PAW, `l14_w00/0` (`بٱ`), exposes why an eventless Alef
template would be unsafe: the Alef is a simple terminal upright with no
interior extremum or substantial branch. The extractor now emits a
`terminal-ascent` only when the final spine rises by at least 1.5 stroke widths
and is steeper than its horizontal displacement. Its position is the first
post-valley spine sample whose smoothed local tangent reaches 45 degrees from
horizontal (`upward component >= abs(horizontal component)`). The
`alef-final-terminal-ascent` template owns that real cue. It places the cut at
the transition into the upright rather than immediately after the Beh valley,
and owns the complete Alef; no empty body or word-specific exception is
involved.

The resulting strict page-3 run aligns all 151/151 multi-letter PAWs, accepts
145, and marks six alignments low-confidence. Every accepted multi-letter PAW
has a named JSON template for every body. Adding the terminal-ascent event
changes only `l14_w00/0`; all previously aligned page-3 groups, joins, cuts,
confidence values, and margins are byte-for-byte unchanged in the structural
comparison.

Sad and Dad templates own their terminal tooth explicitly. Initial `ضا` on
`l05_w08` now consumes two rise/valley pairs after the bowl; Dad is
`0.00–721.63`, the variable join is `721.63–1260.45`, and Alef begins at
`1260.45`. Medial Sad/Dad has a higher-priority allograph for scans where the
bowl produces an additional rise/valley pair, while the previous two-pair
allograph remains the fallback. This moves the Dad exit in `l13_w07` from
`800.69` to `999.65`, after the visible tooth, and leaves its following join
as `999.65–1305.37`. Both allographs use an explicit last-valley plus one
stroke-width exit rule. Other page-3 PAWs are structurally unchanged; strict
coverage remains 151/151 and confident acceptance improves to 146/151 because
`ضا` is no longer ambiguous.

Initial Heh now has one complete `heh-initial-two-loop` template. The former
`heh-initial-double-head` was an artefact of medial-axis encoding: it fired
when both loop arms survived as high graph branches at the same spine
coordinate, while visually identical instances put one arm on the main spine
and exposed only one `high-branch`. Every page-3 initial Heh has the common
ordered sequence `high-branch, valley, peak`, followed by a terminal valley;
`double-high-branch` is therefore supplementary graph evidence, not a separate
allograph. Removing the one-event template makes Heh own both loops. The
terminal valley is expressed as `afterEvents` look-ahead context: it identifies
Heh without belonging to its body, and the following Meem may reuse the same
event as owned topology. It may lie beyond the measured end of Heh's ink. In `l02_w00`, Heh becomes
`0.00–370.24`; in `l10_w02`, it becomes `0.00–355.92`. Strict coverage remains
151/151; confident acceptance is now 148/151 with three low-confidence PAWs.

The two-loop Heh exit uses `width-or-event-exit`, anchored at its last peak.
This matters when the terminal valley is too weak to survive event detection:
searching only after the later matched valley had already passed the real body
neck and put the start of the kashida inside Meem. The peak-anchored search
uses the first owned-width falling edge when one exists. It moves the
`l00_w08` Heh exit from `384.56` to `319.64` (join
`319.64–389.56`) and `l08_w02` from `443.32` to `355.68` (join
`355.68–639.20`), while preserving the observed `l02_w00` neck at `370.24`.
If no width neck exists, the complete matched shape plus one stroke width is
retained; this keeps all of `l12_w08`'s second loop in Heh. This is a local
body/neck decision and does not depend on kashida length. Nearby branch/rise
or branch/valley diamonds remain separate topology and spine cues; same-kind
extrema already have a minimum separation and were not the source of the
boundary error.

Final Meem uses its entry branch as the physical body landmark. Its body starts
one stroke width before that branch, rather than at a later rise or valley
inside the Meem. The preceding kashida remains a measurable interval and no
fraction of its variable length enters the rule. In `l02_w00`, the entry branch
is at `639.23`, the measured stroke width is `102.84`, and the selected dense
sample is `534.79` (the continuous target is `536.39`). The resulting join is
`370.24–534.79`, and the red next-body cut is at `534.79`. This boundary rule is
part of the reusable `meem-final-tail` template, not a Heh-specific exception.

`l06_w05` establishes three entry landmarks for its observed allographs. The
`kaf-medial-zigzag` body requires the rise immediately before its x-right /
x-left skeleton zigzag, but the `rise` event is detected at the upper end of
that ascent (`890.68`), near the branch. Its `rise-onset` entry rule traces
backward along the stroke-width-smoothed spine to the last transition from
flat/falling to rising and starts Kaf at the lower onset (`349.91`). The
`dal-dhal-final-rise-head` body starts half a stroke width before its head
branch: branch `1590.99`, stroke width `106.03`, dense-sample boundary
`1537.47`. The `waw-final-head` body uses the same half-stroke rule: branch
`227.39`, stroke width `82.62`, boundary `185.89`. These rules are scoped to
the named allographs; other Dal and Waw topology variants remain unchanged.

When a complete ordered single-letter template path exists, it is now the
authoritative alignment. The generic rising-join path is not allowed to
recreate the same all-single partition while ignoring explicit entry/exit
rules; it remains available for composite/fused partitions. This removes the
spurious competitor that previously gave `l06_w05/0` margin `0.020`. Its new
margin is `0.940`. Page-3 coverage stays 151/151, with 150 confident PAWs and
one low-confidence PAW.

Medial Heh has exactly two visual templates on page 3. Six older templates
encoded incidental medial-axis variants (`high`, `double-high`, `zigzag`, or
plain branches) as if they were different letter shapes; three of them were
unused and the others split visually identical closed Hehs. They are replaced
by `heh-medial-two-loop` and `heh-medial-open`. The closed form requires one
strong high branch and one owned valley, with the second valley retained as
look-ahead context. It begins 1.5 stroke widths before the high branch. Its
physical end is one stroke width after the last observed plain branch inside
that full two-valley context; the context prevents the following letter from
stealing the second loop without extending Heh through the kashida. The open
form uses branch, rise, valley. On page 3 the split is 19 closed and 4 open
PAWs.

This prevents the following final-Meem template from claiming the second Heh
loop. In `l00_w05/4`, Heh changes from `362.15–581.48` to
`367.25–729.40`, followed by a measurable kashida through `969.13`. In
`l01_w06/1`, Heh changes from `348.16–566.41` to `332.57–737.89`, followed
by kashida through the Meem start at `924.96`. Page-3 coverage and acceptance
remain 151/151 and 150/151.

Contextual exit ownership does not create a multi-letter ligature or consume
two letters with one template. It keeps one `SegmentGroup` per letter and
moves the cut to the next body's first topology event so the transition is
owned by the current letter. The JSON field is `exitOwnsJoinNextLetters`.

For Heh-to-Zay, Heh continues to the first topology event of Zay and Zay begins
there. Thus `l12_w06/0` keeps separate Heh and Zay groups which touch at
`1941.30`; `l12_w03/0` similarly places the boundary at `1892.46`. Entry
ownership is deliberately not modeled; tooth-to-Heh uses the normal template
body and join boundaries. The closed Heh-to-Meem cases above retain their real
kashidas because the exit context does not match. Full page-3 results remain
151/151 aligned, 150/151 confident, with only `l04_w00/0` below the confidence
threshold.

The initial Ain/Ghain body has an enforced event exit: half a stroke width
after its matched high branch. `enforce: true` means an incompatible following
template entry cannot clip that endpoint; the ordered matcher tries its next
topology assignment instead. This matters for `l06_w01/0`, where the old Dhal
match began at Ain's branch (`357.58`). The corrected Ain ends at `405.90`,
and the alternate Dhal topology begins at `570.19`. Across all ten initial
Ain/Ghain PAWs on page 3, the sampled exit is within 2 pixels of
`branch + 0.5 * strokeWidth`; no unrelated body span changes.

Elastic templates support symmetric non-owning context. `beforeEvents` must
occur in order before the first `requiredEvents` event, and `afterEvents` must
occur in order after the last. Only `requiredEvents` belong to the body,
advance the ordered ownership cursor, participate in loop ownership, and
serve as ordinary entry/exit anchors. Context events remain reusable by the
adjacent template. Result JSON records them separately as `beforeEvents` and
`afterEvents`; diagnostics draw owned events with a solid white ring and
context with a dashed grey ring.

Any event after the first logical event may constrain its spacing from the
previous event (the logical order is `beforeEvents`, `requiredEvents`, then
`afterEvents`):

```json
{"kind": "branch", "minStrength": 0.5,
 "gapFromPrevious": {"minStrokeWidths": 0.4,
                     "maxStrokeWidths": 1.5},
 "consecutive": true,
 "adjacent": true}
```

Gap bounds are normalized by the PAW stroke width, not expressed in pixels.
Either bound may be omitted. `consecutive` forbids skipping another event of
the same kind whose strength satisfies this event's `minStrength`; intervening
events of other kinds and weaker same-kind detections are allowed. All sequence
constraints are optional. `adjacent` is stronger: it rejects any emitted
skeleton event at a spine position strictly between the previous and selected
event, regardless of kind or strength. Multiple event labels at the same spine
position do not break adjacency. Existing templates retain fully elastic
spacing unless one of these constraints is present.

Peak and valley detection uses a low `0.02 * strokeWidth` prominence floor
only to decide which extrema are emitted. Their actual normalized prominence is
kept as event `strength`; each template's `minStrength` remains the recognition
threshold. Consequently weak emitted extrema also break `adjacent`, as any
other emitted event does.

The interactive reviewer exposes bounded alignment tracing only for failed
PAWs. Clicking **Explain no match** reruns the selected word with trace mode
enabled and shows incoming/outgoing state counts per letter, template option
counts, the first empty stage, and up to eight concrete rejection examples.
Ordinary GUI analysis and batch segmentation do not allocate trace records.
The trace distinguishes missing/weak/already-owned events, adjacency and gap
failures, loop rejection, and post-topology boundary/partition rejection.

Final Reh is split from Zay at the template level and has an enforced observed
event entry half a stroke width before (to the right of) its last branch. Zay
keeps its previous entry behavior because its body does not always expose the
same branch. The rule covers the rise/valley and terminal-valley Reh variants.
`hah-family-initial-head` keeps its closing valley as `afterEvents`, because in
`خر` that valley is topology context beyond the Reh-owned cut. Enforced entries
are also applied in composite partitions, so the `تج` ligature in `l14_w03`
retains its body exit while its measurable join extends to the Reh entry.
Across all nine final-Reh PAWs on page 3, the sampled entry is within 2.25
pixels of `branch - 0.5 * strokeWidth`.

`heh-medial-two-loop` now requires only `high-branch, valley`; the former
look-ahead valley and contextual Reh/Zay ownership are removed. Its enforced
exit is `last matched valley + 1.0 * strokeWidth`. The ordered matcher retains
a bounded beam of later event realizations: if the earliest following-letter
events fall inside that enforced Heh body, it advances to a later realization;
if only the generic entry heuristic overlaps, the adjacent bodies touch at the
Heh exit while keeping the following required event inside its own body. On
page 3 this gives 20 two-loop and 3 open medial-Heh matches, with all two-loop
exits sampled within 1 dense-spine step of the requested valley offset.

Medial Heh allographs are additionally separated by body-local graph cycles,
not the PAW-wide loop count: `heh-medial-two-loop` requires exactly two cycles
(`minLoops: 2`, `maxLoops: 2`), while `heh-medial-open` permits at most one.
Bridge-free medial-graph components carry their Euler cycle rank and are
projected onto a main-spine interval. Each component first belongs to an
event-owning body with an explicit loop contract, then to the body with the
most matched template events inside that interval; greatest geometric overlap
only breaks remaining ties or handles components without matched events. Thus a
bad boundary cannot evade a loop by assigning most of its interval to an
adjacent body. The localized
evidence is recorded as `loopEvidence` in result JSON. On page 3 all 20 closed Hehs own exactly two
cycles and all 3 open Hehs own zero; this changes `l04_w07` from open to closed
and `l12_w06` from closed to open relative to event-only matching.

Both modes are opt-in while the template set is small:

```sh
uv run tools/paw_segmentation/segment_paw_letters.py \
  --analysis output/oldmedina2-analysis --page 3 --batch \
  --skeleton-shapes --logical-ligatures \
  --skeleton-templates templates/oldmadina/skeleton_templates.json \
      --out output/paw-skeleton-page3
```

Run the candidate-free alignment independently with:

```sh
uv run tools/paw_segmentation/segment_paw_letters.py \
  --analysis output/oldmedina2-analysis --page 3 --batch \
  --body-join-alignment --cut-policy next-start \
  --out output/paw-body-join-page3
```

Every run records `skeletonEvents`, `shapeMatches`, and `segmentGroups` in
JSON and draws event diamonds in the overlay. Template constants are not meant
to remain hand-set. `train_skeleton_templates.py` fits event positions,
tolerances and minimum strengths from an explicit reviewed-cuts file; it never
uses the segmenter's confidence as truth. Load its output with
`--skeleton-templates FILE`.

## Open problems

1. **`لله` (`l01_w04`) ownership fix.** The full-ray contamination described
   here is now separated into owned and foreign ink by graph-derived spine
   coordinates. On the current page-3 analysis its cuts move from 114.8,
   392.3 to 114.8, 267.9 while every regression word keeps its pre-change
   cuts. Do not replace the topological window with a spatially
   capped ray: the latter was measured to create confident errors.
2. **`كا` (`l14_w05`)** — the spine cuts across the kaf's counter instead of
   following its rim, so it never reaches the baseline where the real neck is,
   and the PAW yields no candidate. `_extract_spine` takes the shortest path
   and a chord across a loop is always shorter than the rim. Needs the loop
   handled explicitly; `medial.loops` already records that one is present.
3. **Cuts that split a tooth.** `مهتد` cuts at 97% of the way up the teh's
   tooth and `تنذ` at 91% up the noon's, dividing the tooth between two
   letters. A `flank_fractions` penalty was built and measured — it separates
   these cleanly from every alternative (37% and below) and fixes both — but
   pushing cuts off tooth flanks moved them onto positions other constraints
   relied on, degrading `لضللة`'s third lam to 1.00 stroke widths and costing
   5 accepted. Reverted; the approach is sound but needs the interaction with
   `MIN_SPINE_RUN_W` resolved.
4. **`W_PURITY` now measures local extra ink.** One additional locally owned
   full-width stroke scores approximately `exp(-1)`, close to the old intended
   two-run score of 0.4. Foreign ink is excluded from purity and instead has
   the bounded contribution described above.
5. **`MIN_CUT_CONFIDENCE` deserves its own pass**, for the anti-correlation
   above rather than to make any one word pass.

## Out of scope, deliberately

Analysis-by-synthesis against the digital font, CTC forced alignment, the EM
consistency loop, and any change to `build_instance_index.py` or the C++
analyzer. `train_letter_shape_priors.py` exists and works (`W_SHAPE_PRIOR` was
swept to 0.04 — as a tie-breaker, not an override) but has not been rerun
since the analyzer-supplied PAWs landed, so its priors are stale.
