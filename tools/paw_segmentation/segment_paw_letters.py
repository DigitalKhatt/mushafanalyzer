#!/usr/bin/env python3
"""Text-constrained segmentation of a PAW body into individual letters.

Prototype for the approach described in the session plan:

  A. text          -> expected letters, joining forms, PAW decomposition
  B. body mask     -> PAW connected components (must match A, else abstain)
  C. contours      -> arc-length resampled + curve-domain smoothed polyline
  D. polyline      -> Voronoi medial axis graph with an exact radius function
  E. graph         -> pruned graph + a right-to-left spine
  F. spine         -> cut candidates from the cut-cost profile
  G. skeleton      -> ordered spine/branch events and interval templates
  H. candidates    -> semi-Markov DP over letters and composite ligatures
  I. output        -> overlay PNG + JSON

Detection deliberately runs on the mask and the dense polyline, never on the
fitted cubics: the medial axis is a region operation, and staying off
`FitConfig` keeps this decoupled from the open tracing-tuning questions.

The measurement that actually discriminates a join is not stroke thinness --
in this hand the connectors are barely thinner than the letter strokes -- but
the **local cut cost**: how much ink belonging near this spine position a line
perpendicular to the spine has to sever. At a join that is one run about one
stroke wide; inside a letter the same line also crosses a locally attached
ascender bar, bowl, or returning tail. Ink merely intersected by the infinite
line but attached elsewhere along the spine is recorded as foreign diagnostic
ink and has a bounded influence of at most part of one stroke on candidate
generation and scoring.

A second consequence of measuring it that way: a letter whose whole body is a
flat connector (a medial yeh here) shows up as a long low-cost *plateau*, and
its two boundaries are the plateau's shoulders, not minima inside it. Cut
candidates are therefore generated at shoulders as well as at minima.

Usage:

  tools/paw_segmentation/segment_paw_letters.py \
      --analysis output/oldmedina2-analysis --page 3 --word l02_w04 \
      --out output/paw-segmentation

  tools/paw_segmentation/segment_paw_letters.py \
      --analysis output/oldmedina2-analysis --page 3 --batch \
      --out output/paw-segmentation

Shape templates and logical ligatures are experimental opt-in modes:

  tools/paw_segmentation/segment_paw_letters.py \
      --analysis output/oldmedina2-analysis --page 3 --batch \
      --skeleton-shapes --logical-ligatures \
      --out output/paw-skeleton-segmentation

Candidate-free body/join forced alignment is a separate strict mode:

  tools/paw_segmentation/segment_paw_letters.py \
      --analysis output/oldmedina2-analysis --page 3 --word l14_w02 \
      --body-join-alignment --cut-policy next-start \
      --out output/paw-body-join-segmentation
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import struct
import sys
from dataclasses import dataclass, field, replace
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
from scipy.signal import find_peaks, peak_prominences
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import Voronoi, cKDTree

# tools/tracing/ uses flat sibling imports (no package __init__.py), so its
# modules only resolve when that directory is on sys.path. This file lives
# one directory over, in tools/paw_segmentation/, so add it explicitly
# rather than relying on the entry script's own directory (which Python adds
# automatically, but that's tools/paw_segmentation, not tools/tracing).
_TRACING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "tracing")
# Template banks are per-Mushaf: the letter shapes are what one hand does, and
# a second Mushaf gets its own directory beside this one rather than edits to
# these files.
DEFAULT_TEMPLATE_PATH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "templates", "oldmadina", "skeleton_templates.json"))
if _TRACING_DIR not in sys.path:
  sys.path.insert(0, _TRACING_DIR)

from arabic_joining import describe_word
from png_to_ttf_no_potrace import (
    find_contours_with_hierarchy,
    read_bw_white_is_ink,
    signed_area,
    skip_tiny_hole_subtrees,
)
from smoothing_image import curve_domain_gaussian_smooth

# ---------------------------------------------------------------------------
# Tunables. Every geometric constant is a multiple of the measured stroke
# width, so nothing here is tied to the 4800 DPI scan resolution.
# ---------------------------------------------------------------------------

PAD_PX = 24                      # border so contours never touch the image edge
# Minimum enclosed background area retained as a counter. This must scale with
# the writing: the small medial-Meem counter in page-3 l12_w08 is 326 px^2 but
# still 0.042 stroke-widths squared, while the pinholes in the same word are
# only about 0.0012. An absolute 500 px^2 cutoff erased the real counter before
# medial-axis construction and therefore made its graph incorrectly acyclic.
# Measured against holes identified by hand: the smallest genuine counter in
# the sample is a Waw eye at 0.087 W^2, and the scan-residue pinholes that put
# a false loop in a stroke sit at 0.0175 and below. 0.05 separates them with
# room on both sides. Raising it further is not free -- at 0.10 seventeen PAWs
# per 1521 stop segmenting and at 0.25 sixty-three do, because `waw-fina`,
# `meem-init` and `heh-medi-two-loop` all match on counters in that band.
MIN_HOLE_AREA_W2 = 0.05
# A counter is a rounded pocket the stroke wraps around; scan residue is often
# a sliver -- a long thin gap lying *along* a stroke rather than inside a bowl.
# Measured on holes hand-identified as one or the other: every genuine counter
# scores 0.56 or higher on 4*pi*area/perimeter^2, while the slivers score 0.14
# and 0.17. This does not separate a *round* residue hole from a small counter
# -- p358 l00_w04's Hah residue and p003 l03_w01's Waw eye are 0.089 and 0.087
# W^2 with circularity 0.70 and 0.77, indistinguishable by any hole-local
# measure -- so it is deliberately only the sliver test.
MIN_HOLE_CIRCULARITY = 0.25
SPECKLE_AREA_W2 = 0.10           # component area floor, in stroke widths squared

RESAMPLE_STEP_W = 0.08           # contour resample step
SMOOTH_SIGMA_W = 0.10            # curve-domain smoothing sigma
MIN_MEDIAL_RADIUS_W = 0.08       # drop near-boundary Voronoi vertices
PRUNE_LEN_W = 0.75               # drop leaf branches shorter than this
SUBSTANTIAL_RADIUS_W = 0.25      # spine endpoints must be this thick
# The same rule for the spine's left-hand endpoint, where a lower bar pays.
# A final letter exits by tapering, so the last stretch of a Waw's or Yeh's
# tail is thinner than 0.25 and is dropped -- and with it the upturn that
# makes the tail a valley. On page 530 l11_w04 the on-spine half of that
# upturn is 0.019 stroke widths against a 0.02 emission floor, so ``waw-fina``
# sees no valley and the Waw is not recognized; the other 0.12 sits in the
# off-spine continuation. Only the *left* endpoint moves, so s=0 stays at the
# pen entry and events before the tail keep their arclength -- which is why
# this is affordable where the entry-side band was not. Measured over 26
# pages / 7517 PAWs: 0.20 gains 16 and loses 2; going lower stops gaining
# (0.15 is +19/-5, 0.10 and below +20/-7).
SPINE_EXIT_RADIUS_W = 0.20
# How far *below* the baseline the spine's right-hand endpoint may sit, in
# stroke widths. A pen entry is never deep under the line; ink that is, and is
# also the rightmost in the PAW, is a descender that has swept back underneath.
# Only that case is re-picked -- see _extract_spine. 0 disables the rule.
SPINE_ENTRY_MAX_DROP_W = 2.0

PROFILE_STEP_W = 0.05            # cut-cost profile sampling step
TANGENT_WINDOW_W = 0.50          # half-window for the spine tangent estimate
LOCAL_OWNERSHIP_RANGE_W = 4.00   # keep ink attached this near in spine arclength
FOREIGN_CUT_CAP_W = 0.75         # bounded uncertainty from topologically remote ink
CUTTABLE_MAX_W = 1.80            # a cut this thin (or thinner) is admissible
# Above this the cut is inside letter structure, so `adjacency` treats it as
# the next letter's body. Lowered from 2.20 with the body-entry candidate
# below: a medial meem's head peaks around 1.9 stroke widths, so at 2.20 the
# head was not "structure" at all and adjacency never fired for it, leaving
# the entry candidate scoring below the connector position behind it.
HIGH_COST_W = 1.90
BODY_ENTRY_W = 1.35              # crossing this upward enters a letter body
MIN_REGION_W = 0.50              # ignore cuttable slivers shorter than this
CAND_SAMPLE_W = 0.80             # candidate spacing inside a cuttable region
SHOULDER_INSET_W = 0.12          # keep shoulder candidates just inside the region
ADJACENCY_RANGE_W = 2.50         # distance over which "next to structure" decays
# Tolerance on "the join sits near the baseline". Generous on purpose: for a
# deep-bowl final (noon, yeh, qaf, sad) the join is at the bowl's rim, well
# above the baseline -- measured at 1.5 stroke widths for the yeh-noon of
# `alladhina` on page 3. A tight sigma there scored the true join at 0.002
# and handed the cut to a spurious maximum where the bowl crosses the
# baseline on its way down.
BASELINE_SIGMA_W = 1.20
# Minimum letter extent, in stroke widths. Measured with `segment_extent`
# rather than along the spine alone: a letter written as an upright leaves
# the spine almost immediately, so its spine run is not its size. The
# initial lam of "ٱلَّذِينَ" runs 1.20 along the spine but stands 8.06 tall,
# and every other letter in that word measures 1.1-1.9 ink pixels per unit
# of spine extent against the lam's 5.19. Judged on spine length alone the
# lam looks degenerate and the DP refused to close it, putting the lam|dhal
# cut inside the dhal; judged with its stem it clears any sane floor.
MIN_SEGMENT_W = 1.50
# Floor on the spine run by itself, checked alongside MIN_SEGMENT_W. A
# letter may be short along the spine only when its own body hangs off it,
# the way an upright does; it may not borrow length from a branch that
# merely happens to fall inside its span. Without this the extent test
# admits genuinely degenerate segments: on "يمد" the meem collapsed to 0.9
# stroke widths of spine and still cleared 1.5, because the dal's branch at
# s=769 lay within it. Set below the 1.20 of the initial lam of "ٱلَّذِينَ",
# which is the shortest spine run a real letter is known to produce here.
MIN_SPINE_RUN_W = 1.00
# A connector is a stroke, not a letter: nothing of substance branches off it.
# Measured over 884 accepted joins on pages 3/80/358/600, the strongest branch
# inside a join is 0.000 at the median and 0.515 at the 99th percentile, so a
# join holding a branch above this is not a connector -- it is a body the
# alignment failed to claim. In p358 l00_w04 a composite `beh-hah-core` put the
# whole Hah, six branches and a 4.26 among them, inside a 15.7-stroke "join".
# Applied as a preference, not a veto: an alignment that breaks it is dropped
# only when a clean alternative exists, so a PAW is never failed over it.
MAX_JOIN_BRANCH_W = 1.0

# Cut score weights. Hand-set for the prototype; these are exactly the
# quantities a learned model would later replace.
W_THINNESS = 1.2                 # the cut severs about one stroke
W_NARROWNESS = 2.0               # ... and less than one, where there is a neck
W_PURITY = 0.8                   # ... and only one run of it
W_ADJACENCY = 0.7                # a boundary sits just before the next body
W_BASELINE = 0.5                 # joins tend to lie near the baseline
W_HORIZONTAL = 0.25              # ... and to run along the writing direction
W_LOCAL_ONLY_CANDIDATE = 0.05    # local-only cuts must beat full-ray picks
W_SUM = (W_THINNESS + W_NARROWNESS + W_PURITY + W_ADJACENCY + W_BASELINE
         + W_HORIZONTAL)

# Confidence is a cut's score relative to a *realistic* good join, not to the
# theoretical weight sum -- that maximum assumes narrowness 1.0, i.e. a cut of
# zero width, which never occurs, so normalizing by it made every real cut look
# poor and shifted the whole scale whenever a weight changed. These are the
# feature values of a clean, unambiguous join measured on page 3.
IDEAL_JOIN = dict(thinness=1.0, narrowness=0.35, purity=1.0, adjacency=0.8,
                  baseline=0.9, horizontality=0.95)
SCORE_REFERENCE = (W_THINNESS * IDEAL_JOIN["thinness"]
                   + W_NARROWNESS * IDEAL_JOIN["narrowness"]
                   + W_PURITY * IDEAL_JOIN["purity"]
                   + W_ADJACENCY * IDEAL_JOIN["adjacency"]
                   + W_BASELINE * IDEAL_JOIN["baseline"]
                   + W_HORIZONTAL * IDEAL_JOIN["horizontality"])

# Width prior in the DP. Geometry must decide; the prior only breaks ties --
# notably *within* a plateau, where geometry genuinely cannot say where the
# boundary lies. Hence the dead zone: a letter may be anywhere from half to
# one-and-a-half times the mean extent at no cost, because real letters are
# genuinely that unequal (a wide zah beside a narrow medial yeh). Only
# implausible extremes are penalized. Placeholder for learned per-(letter,
# form) width distributions.
W_WIDTH_PRIOR = 0.25
WIDTH_PRIOR_DEADZONE = 0.50

# A PAW is reported as low-confidence when its weakest cut scores below this
# fraction of the maximum. Empirically on page 3 the cases below ~0.70 are
# dominated by the ligatures this geometry cannot handle -- lam-alef, kaf-alef
# and the stacked `alayha` form -- where the letters are written on top of one
# another and there is no baseline join to find. Abstaining is the correct
# answer there; the cuts are still emitted, but flagged.
MIN_CUT_CONFIDENCE = 0.70

# How many threads scipy may split a nearest-neighbour query across. One PAW's
# query is around 200k ink pixels against a few hundred medial vertices and is
# a quarter of this module's time; scipy answers it about six times faster on
# this machine with every core, and point-for-point identically, because each
# point's nearest neighbour is found independently of the partitioning.
#
# The default of 1 is the only safe one for an unco-ordinated import: the
# corpus builder runs one process per core, and threads underneath that would
# only contend. A caller that knows how many processes it started sets this to
# cores-per-process -- see build_corpus.worker_threads.
QUERY_WORKERS = 1

# A cut also has to *win*, not merely score well. Re-solve with every
# candidate near a chosen cut suppressed; if the resulting alternative is
# nearly as good, the placement was arbitrary. This is what catches the
# yeh-noon of `alladhina`, where two candidates a full bowl apart tied at
# exactly 3.460 and the width prior alone broke the tie.
MARGIN_EXCLUDE_W = 2.0           # suppression radius, in stroke widths
MIN_CUT_MARGIN = 0.05            # required lead over the best alternative

# Expected internal tooth-junction count per base letter (script knowledge,
# not learned): a medial/initial letter made of N teeth has N-1 internal
# necks. Only letters whose tooth count is unambiguous in every joining form
# are listed; a letter absent here is unconstrained (TOOTH_COUNT.get returns
# None). Sad/dad are deliberately excluded -- their internal structure is a
# loop, not a tooth run, and the current features don't characterize it.
#
# This exists because the cut-cost profile alone cannot tell a seen's three
# teeth apart from three separate letters: on page 3, "مستهز" (mustahzi) was
# segmented as seen=1 tooth / teh=3 teeth -- exactly reversed -- because the
# widest, most confident-looking cut fell between the first and second tooth
# of the seen rather than after the third. Nothing in the per-cut score
# distinguishes an inter-tooth valley from an inter-letter one; only knowing
# how many teeth the letter is supposed to contain does.
TOOTH_COUNT: Dict[str, int] = {
    # single-tooth letters: no internal necks expected in any joining form
    "ب": 0, "ت": 0, "ث": 0, "ن": 0, "ي": 0, "ى": 0,
    # triple-tooth letters: exactly two internal necks, between the 3 teeth
    "س": 2, "ش": 2,
}

# An initial seen/sheen starts with its first tooth at the spine endpoint, then
# lifts twice for the remaining two teeth. Its shallow cut-cost valleys can
# contain extra minima inside the long inter-tooth trough ("شتر" has three
# minima for only two junctions), while the two lifts are unambiguous. For
# these forms, count the lifts as the two expected internal tooth transitions.
TOOTH_RISE_FORMS = frozenset({"س|initial", "ش|initial"})

# Tooth valleys are much shallower than genuine letter-boundary necks -- on
# the seen above, the inter-tooth dips measured only 0.83-0.98x stroke width,
# against 0.5-0.7x at real letter cuts -- so they need their own, far more
# sensitive detector than CUT_PROMINENCE_FRAC, run over the whole spine
# independent of the "cuttable region" logic that cut_candidates uses (that
# logic merges shallow dips into one region and reports only its single
# deepest point, which is exactly what lost the internal teeth here).
TOOTH_PROMINENCE_FRAC = 0.15     # minimum depth of a tooth-junction dip
TOOTH_MIN_SEP_W = 0.5            # minimum spacing between distinct teeth

# A tooth also shows as the spine lifting off the baseline. ``spine_peaks`` is
# the only one of the three signals that finds the tooth of a medial noon
# written flat against the connector.
# This is only an event-emission floor. Template ``minStrength`` is the
# recognition threshold. Keep the floor low enough to expose shallow but real
# extrema (for example the final-Waw valley in page 3 l14_w05 is 0.108), while
# still suppressing subpixel fluctuations in the sampled medial spine.
SPINE_EXTREMUM_PROMINENCE_W = 0.02
SPINE_EXTREMUM_MIN_SEP_W = 0.60     # minimum spacing between like extrema
# Strength below which an event is discarded outright rather than left for
# ``minStrength`` to reject -- see ``drop_weak_events`` for why the two are
# not the same thing. Measured over 15 pages / 4288 PAWs: the weakest event a
# template actually used was a 0.132 peak and a 0.100 branch, so these floors
# clear nothing load-bearing, and applying them changed no PAW's outcome while
# removing 304 peaks and 3903 branches from the stream.
#
# There is deliberately no valley entry. A shallow real valley is not
# separable from a wobble by prominence: 51 of the 1222 valleys templates used
# measure under 0.08, down to 0.020, because ``waw-fina`` accepts its valley
# at ``minStrength: 0`` -- page 3 l10_w06's measures 0.048 against 0.035 for
# the noise beside it. A 0.08 valley floor cost 40 PAWs on that sample.
EVENT_STRENGTH_FLOOR_W = {"peak": 0.10, "branch": 0.10}

# Default arclength a body's first required event must lie beyond the previous
# body's last owned event. The cross-body cursor is otherwise an event *index*
# rule, which lets a body claim an event at the coordinate its neighbour already
# occupies -- a senna taking the peak half of the Meem junction whose branch
# half the Meem just used, 0.03 stroke widths away. Templates may still say 0 to
# opt out.
MIN_ENTRY_GAP_STROKE_WIDTHS = 0.5
# Expected number of spine lifts per letter, keyed "letter|form" with a
# "letter|*" fallback, exactly like BRANCH_EXPECT.
#
# Rises count what branches cannot. A tooth is drawn as a parallel
# excursion of both contours -- the stroke keeps its width through it -- so
# the medial axis runs through as one curve and forks nowhere. Only a real
# attachment forks. That is why the sad of "مُصْلِحُونَ" shows a rise at
# s=1055 with no branch beneath it, and why counting branches made sad and
# dad disagree: the sad's own span holds 3 branches and the dad's holds 4,
# so a lower bound set for one forced the other to reach into its
# neighbour. Counted as rises they agree at 3.
#
# Initial forms are absent throughout: an initial tooth sits at the spine's
# own start, where there is no lift to measure.
RISE_EXPECT: Dict[str, Tuple[Optional[int], Optional[int]]] = {
    "ب|medial": (1, 1), "ت|medial": (1, 1), "ث|medial": (1, 1),
    "ن|medial": (1, 1), "ي|medial": (1, 1), "ى|medial": (1, 1),
    # A medial ayn has exactly one detected body lift on all 6 page-3
    # instances. This prevents a preceding sheen from borrowing ayn's lift as
    # a missing third tooth (the ambiguity in both instances of `يشعر`).
    "ع|medial": (1, 1),
    "ص|*": (3, None), "ض|*": (3, None),
    # A medial heh is not guaranteed to lift: only 13 of 20 page-3 instances
    # do. Keep the upper bound, because without it the medial heh of
    # "مُهْتَدِينَ" took the following teh's tooth as a second lift, leaving
    # that teh a 1.25-stroke-width sliver. Other forms retain the established
    # one-lift rule.
    "ه|medial": (None, 1), "ه|*": (1, 1),
    # A final kaf's upright/body is a spine lift. Without this reciprocal
    # ownership rule, a preceding medial yeh can satisfy its own one-rise
    # constraint by taking the kaf's lift while leaving its real tooth in the
    # preceding letter ("ليك" in l13_w04).
    "ك|final": (1, 1),
}


def _rise_bounds(letter: str, form: Optional[str]
                 ) -> Optional[Tuple[Optional[int], Optional[int]]]:
  """RISE_EXPECT for this letter in this form, else the pooled entry."""
  for key in (f"{letter}|{form}", f"{letter}|*"):
    bounds = RISE_EXPECT.get(key)
    if bounds is not None:
      return bounds
  return None
W_RISE_MISSING = 2.0             # weight when such a letter contains no rise
W_RISE_EXTRA = 2.0               # ... or holds a second, its neighbour's
RISE_ENTRY_FORMS = frozenset({"ك|final"})
RISE_ENTRY_MAX_W = 2.0            # its body rise must be near segment entry
W_RISE_LATE_ENTRY = 2.0           # penalty per excess stroke width
# All three medial seen/sheen teeth lift off the incoming connector. Treat
# them as one run only when adjacent lifts are close enough to be teeth rather
# than unrelated neighbouring bodies. If no such run exists anywhere in the
# PAW, the constraint abstains instead of manufacturing teeth from neighbours.
RISE_CLUSTER_EXPECT = {"س|medial": 3, "ش|medial": 3}
RISE_CLUSTER_MAX_GAP_W = 3.0
W_RISE_CLUSTER_MISSING = 10.0    # a detected tooth run belongs to seen
W_RISE_CLUSTER_EARLY_EXIT = 10.0 # its cut follows the third tooth's valley
# A final zay does not always own a spine rise, but on all 3 page-3 instances
# its entry boundary is within 0.3 stroke widths of one. This anchors the
# heh|zay boundary without falsely demanding that the rise lie on zay's side.
RISE_BOUNDARY_FORMS = frozenset({"ز|final"})
RISE_BOUNDARY_MAX_W = 0.0        # choose the candidate nearest that event
W_RISE_BOUNDARY = 3.0
TOOTH_MARGIN_W = 0.3             # excludes a segment's own boundary cuts
                                  # from being counted as one of its teeth

# Weight on |observed - expected| tooth mismatch. Deliberately large relative
# to W_SUM (~5.55): unlike the geometric features, a tooth-count mismatch of
# even 1 should reliably override the cut-cost scores, because it means the
# candidate segment cannot be the claimed letter at all, not merely that it
# scores slightly worse.
W_TOOTH_MISMATCH = 4.0
TOOTH_MISMATCH_SLACK = 1

# Letters whose body cannot be drawn without interior structure -- a bowl, a
# loop, or a tooth run -- and which therefore always leave a branch on the
# medial axis. Where TOOTH_COUNT counts necks *along* the spine, this asks
# whether anything branches *off* it, which is a different and complementary
# question: a segment can have the right neck count and still be a bare
# connector. On "يمد" the DP put both cuts in the meem-to-dal kashida, giving
# a "meem" segment that is a plain ribbon; nothing in the cut cost, the tooth
# count or the width prior objected, but the segment contains no branch at
# all, while the real meem head sits in the neighbouring segment.
#
# Deliberately excludes letters that are legitimately branchless in some
# form -- alef, dal, reh, waw and the single-tooth letters, measured on page
# 3 at 24-82% zero-branch even when correctly segmented.
BRANCHING_LETTERS = frozenset("مهعغسشصضطظفق")

# Two-sided version of the same idea: how many branches a letter's own body
# should account for. An existential test ("does this segment hold *a*
# branch?") is too weak on its own -- the DP satisfies it by sliding every
# boundary until each segment catches whichever branch is nearest, which is
# how "يمد" ended up with a 7.25-stroke-width yeh holding the meem's head.
# Constraining both ends makes each branch answer to one letter.
#
# (min, max) branch count, measured over accepted page-3 segments and kept
# only where the evidence is one-sided: ه 100% >=1 (mean 4.0), و 100%
# exactly 1, ع 100% >=1, م 91.7% >=1; ا 82% zero, ن 71% zero. Letters that
# came out near 50/50 (ل ي ب ك ذ ف) are deliberately absent -- the data does
# not support a rule, and guessing one would fire on correct output.
# Keyed "letter|form", falling back to "letter|*" -- because the count
# depends on the joining form, not just the letter. An upright is a branch
# only when the spine passes it: `_extract_spine` runs between the
# rightmost and leftmost substantial vertices, both at baseline level, so a
# medial lam's stem leaves the spine mid-run and shows a degree-3 junction,
# while an initial lam's stem hangs off the spine's own start vertex, which
# has degree 2 and is not a branch at all. Measured over accepted page-3
# segments the split is stark: initial lam 96.8% zero (30/31), medial lam
# 84.6% one (11/13); initial noon 90.9% zero against medial noon 88.9% one;
# initial feh 100% zero against medial feh 100% two or more. A letter-only
# table cannot express that, and the entry it does pick is wrong for one of
# the two forms.
#
# Entries are kept only where page 3 is one-sided and has the samples to
# say so. Forms that came out mixed (final noon at 62%, initial beh, initial
# qaf) are deliberately absent rather than guessed.
BRANCH_EXPECT: Dict[str, Tuple[Optional[int], Optional[int]]] = {
    # uprights that attach at the spine's start contribute no branch
    "ل|initial": (None, 0), "ن|initial": (None, 0), "ي|initial": (None, 0),
    "ف|initial": (None, 0), "ا|*": (None, 0),
    # the same letters mid-word, where the stem does cross the spine
    "ل|medial": (1, None), "ن|medial": (1, None),
    "ب|medial": (1, None), "ت|medial": (1, None), "ث|medial": (1, None),
    "ف|medial": (2, None),
    # Medial yeh is constrained by RISE_EXPECT instead. Its tooth is a spine
    # lift, not necessarily a graph branch; requiring both made the yeh of
    # l13_w04 steal final kaf's branch and leave kaf as a minimum-width sliver.
    # letters with a body in every form
    "م|*": (1, None), "ع|*": (1, None), "ه|*": (1, None), "ه|medial": (2, None),
    "و|final": (1, 1), "و|*": (1, None),
    "د|final": (1, None), "ذ|final": (1, None),"ن|final": (1, None),
    "س|*": (2, None), "ش|*": (2, None),
    # Sad and dad: the closed bowl contributes several junctions. The count
    # is not stable across instances -- the sad of "مُصْلِحُونَ" holds 3 within
    # its own span while the dad of "ٱلضَّلَٰلَةَ" holds 4 -- because the tooth
    # that finishes the letter forks nowhere: it is a parallel bend of both
    # contours, so the medial axis passes straight through it. Demanding 4
    # therefore made the 3-branch instance reach past the following lam to
    # borrow its upright, giving a "sad" 12.8 stroke widths wide and 7.8
    # tall. The floor is the smallest count actually observed; RISE_EXPECT
    # carries the part of this constraint that branches cannot express.
    "ص|*": (3, None), "ض|*": (3, None),
    "ط|*": (1, None), "ظ|*": (1, None), "ح|*": (1, None), "خ|*": (1, None),
    "ج|*": (1, None), "غ|*": (1, None),
}


def _branch_bounds(letter: str, form: Optional[str]
                   ) -> Optional[Tuple[Optional[int], Optional[int]]]:
  """BRANCH_EXPECT for this letter in this form, else the pooled entry."""
  for key in (f"{letter}|{form}", f"{letter}|*"):
    bounds = BRANCH_EXPECT.get(key)
    if bounds is not None:
      return bounds
  return None

# Slack on the upper bound only. A plain letter's entry stroke or its
# neighbour's overhang can raise one incidental branch; the lower bound
# needs no slack, since "this letter has a body" is not a noisy claim.
BRANCH_EXTRA_SLACK = 1
W_BRANCH_EXTRA = 2.0             # see W_BRANCH_MISSING

# Minimum length, in stroke widths, for a pruned branch to count as evidence
# of a letter body. _prune_leaves drops everything below PRUNE_LEN_W (0.75),
# which on page 3 removes a genuine interior branch from 1 structured letter
# in 15; recording those and admitting the ones above this floor halves the
# false-alarm rate (13.5% -> 6.5%) without letting scan-noise hair count.
BRANCH_MIN_LEN_W = 1

# Weight on a letter whose branch count contradicts its body. Still small
# enough to act as a tie-breaker: it decides between solutions the cut cost
# already rates nearly equally rather than overriding it. On "فما" the
# wrong answer (feh spanning 10.2 stroke widths, swallowing the meem) beat
# the right one on geometry alone, 7.262 to 6.899, with every segment
# inside the width prior's dead zone; the only thing separating them is
# that the meem holds no branch in the first and all three in the second.
# On the "ٱلَّذِينَ" of line 0 the margin was 0.06, which 1.0 could not close.
#
# This was inert until the body-entry candidate existed. Before it, any
# non-zero weight made "يمد" worse, because the correct boundary was not
# among the candidates and the constraint could only be satisfied by
# sliding boundaries onto whichever branch was nearest.
W_BRANCH_MISSING = 2.0

# Learned per-(letter, joining form) shape prior (Fix 1b). TOOTH_COUNT is
# script knowledge available on the first run; this instead estimates typical
# segment shape from the segmenter's own confident output, so it captures
# what the table can't -- e.g. that a final meem's ink extent has a
# characteristic range, which is exactly what disambiguates a shoulder join
# (no local neck, every position along a flat connector scores alike) where
# the neighbouring letter's *identity* is the only signal left. Only three
# features are used, chosen because they are cheap to evaluate at every DP
# transition (a candidate segment, not yet a committed cut): extent along the
# spine, mean medial radius, and tooth-valley count. Features that need
# per-pixel attribution (loop count, height above/below baseline) are
# recorded as diagnostics (see PawResult.letter_shape) but not scored here,
# since attribution only exists after cuts are already chosen.
SHAPE_FEATURES = ("extent", "radius", "teeth")
W_SHAPE_EXTENT = 1.5
W_SHAPE_RADIUS = 1.0
W_SHAPE_TEETH = 1.0
# Swept empirically against page-3 acceptance count (same method as the Fix
# 1a weights): at 1.0 the shape prior dominated the geometric cut score
# outright, yanking cuts to the class median regardless of local evidence --
# accepted PAWs *dropped* from 94 to 68 after one EM round, the opposite of
# the intended effect. Acceptance peaks near 0.04 and falls off on both
# sides (0.08->95, 0.02->98, 1.0->68), consistent with what this term is
# supposed to be: a tie-breaker between geometrically comparable candidates,
# not a vote that can outweigh the cut-cost evidence itself.
W_SHAPE_PRIOR = 0.04
# Deviations are measured in MADs (median absolute deviation), floored per
# feature so a class with only 1-2 samples (MAD 0) doesn't produce an
# infinite/explosive penalty from ordinary noise.
SHAPE_MAD_FLOOR = {"extent": 0.4, "radius": 0.15, "teeth": 0.5}
# Below this sample count a (letter, form) class falls back to the pooled
# (letter, any form) class, then to no shape prior at all. Deliberately small
# -- a single page yields few examples per exact class.
MIN_CLASS_SAMPLES = 4

LETTER_COLORS = [
    (255, 120, 90), (90, 220, 120), (120, 160, 255), (250, 210, 90),
    (220, 120, 240), (110, 230, 235), (255, 165, 120), (170, 200, 110),
]
PRUNED_BRANCH_COLOR = (255, 0, 255)  # BGR magenta: retained but removed leaves
PRUNED_BRANCH_PATH_COLOR = (110, 0, 110)  # all removed geometry, including noise
MATCHED_EVENT_COLOR = (255, 255, 255)  # any event consumed by a template
CONTEXT_EVENT_COLOR = (170, 170, 170)  # reusable before/after template context
SKELETON_EVENT_COLORS = {
    "x-left": (255, 180, 40),
    "x-right": (40, 180, 255),
    "peak": (80, 255, 80),
    "valley": (246, 92, 139),  # BGR violet-blue, distinct from magenta branch
    "continuation": (190, 190, 190),
    "branch": (255, 0, 255),
    "high-branch": (255, 255, 0),
    "double-high-branch": (0, 200, 255),
    "zigzag-branch": (0, 255, 255),
    "terminal-ascent": (80, 220, 255),
    "start": (120, 255, 200),
    "end": (120, 200, 255),
    "start-height": (60, 200, 150),
    "end-height": (60, 150, 200),
}

# Skeleton-template scores are deliberately comparable to one ordinary cut
# feature, not to the complete cut score. They are soft interval evidence.
W_SKELETON_TEMPLATE = 0.75
SKELETON_SMOOTH_W = 0.20
SKELETON_DIR_MIN_RUN_W = 0.30
SKELETON_DIR_EPS = 0.12
SKELETON_HIGH_BRANCH_W = 1.5
SKELETON_ZIGZAG_MIN_RUN_W = 0.35
SKELETON_TERMINAL_ASCENT_MIN_W = 1.5
SKELETON_TERMINAL_ASCENT_MIN_SLOPE = 1.0
SKELETON_TERMINAL_ASCENT_ANGLE_DEG = 45.0
# ``adjacent`` asks that nothing happened between two owned events. These
# kinds are not things that happen: the two ends mark where the PAW stops.
# Counting them would silently retune every existing ``adjacent`` template.
NON_BLOCKING_EVENT_KINDS = frozenset(
    {"start", "end", "start-height", "end-height"})
SKELETON_MATCH_MISSING = 1.0
SKELETON_MATCH_EXTRA = 0.10
LIGATURE_MIN_SCORE = 0.15

# Candidate-free body/join alignment.  Longitudinal distances are deliberately
# absent: a kashida is an elastic state whose length is an output measurement,
# not a prior.  Smoothing is local and expressed only in stroke-width units.
BODY_JOIN_SMOOTH_W = 0.20
JOIN_ASCENT_EPS = 0.02
JOIN_NECK_PROMINENCE_W = 0.50
JOIN_MIN_CONFIDENCE = 0.45
DEFAULT_BODY_EVENT_ENTRY_RANGE_W = 1.50
DEFAULT_TEMPLATE_JOIN_HORIZONTAL = 0.90
# Neutral light gray keeps the measurable kashida distinct from the coloured
# letter bodies without making it look like another letter class.
JOIN_COLOR = (211, 211, 211)
JOIN_START_COLOR = (0, 165, 255)
JOIN_END_COLOR = (255, 200, 0)


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------

@dataclass
class CutMeasure:
  """Local and global ink measurements for one perpendicular cut."""
  core_len: float                # locally owned run containing the spine point
  owned_len: float               # all locally owned ink crossed by the cut
  foreign_len: float             # crossed ink attached elsewhere on the spine
  owned_runs: int
  global_len: float
  global_runs: int
  chord: Tuple[Tuple[float, float], Tuple[float, float]]


@dataclass
class PrunedBranch:
  """A short medial branch removed from the solver graph, with its geometry."""
  anchor: int
  length: float
  # Ordered from the surviving anchor out to the removed leaf.
  points: np.ndarray


@dataclass(frozen=True)
class SkeletonEvent:
  """A stable event on the main spine or an attached skeleton branch."""
  s: float
  kind: str
  strength: float
  xy: Tuple[float, float]
  source: str = "spine"
  # For branch-family events, the (min x, max x) the branch path itself covers.
  # A letter whose arm folds back across the spine -- a medial kaf -- overhangs
  # spine it must own, and the anchor coordinate alone cannot say how far.
  reach: Optional[Tuple[float, float]] = None
  # Half the ink's local width here, in stroke widths, read off the medial
  # axis. Carried on every event rather than emitted as one of its own: a
  # separate landmark would inflate the event list, block ``adjacent``, and
  # eat the ordered-realization budget, while saying nothing the existing
  # landmarks do not already locate. As a property it qualifies them --
  # "the branch in thick ink" is the Meem whose counter has filled in.
  thickness: float = 0.0
  # Prominence recomputed with the off-spine skeleton counted as height, in
  # stroke widths -- how far the ink rises (for a valley) or falls (for a
  # peak) around this landmark, wherever that excursion lives. ``strength``
  # sees only the spine, and the spine cannot enter a closed counter: a Waw's
  # bowl hangs off it as a lasso, so no simple right-to-left path can
  # traverse it. On page 601 l14_w06 that costs the Kaf its valley -- the
  # rise that should bound it is the Waw's loop, 154px above the spine and
  # invisible to it, leaving 0.068 where the ink actually gives 1.656.
  # Carried beside ``strength`` rather than replacing it because raising
  # every valley measured +19/-13 over 7662 PAWs: templates tuned against a
  # weak valley flip, and a newly strong one gets claimed by a neighbour.
  # Equal to ``strength`` on every event that is not a peak or a valley.
  relief: float = 0.0


@dataclass(frozen=True)
class LoopEvidence:
  """Independent medial-graph cycles localized onto the main spine."""
  s: float
  a_s: float
  b_s: float
  count: int
  # Where the cycle attaches to the spine. Ownership is decided along the
  # spine, so this is the coordinate the solver reasons about -- but for a
  # two-counter Heh both cycles attach within a stroke width of each other,
  # which is why it is useless for showing a reader where the loops are.
  xy: Tuple[float, float]
  # One centre per independent cycle, each inside the counter it encloses.
  # There are ``count`` of them: a two-counter Heh whose loops share their
  # junction is one rank-two component, and the mean of that whole component
  # sits on the crossing, inside neither counter.
  centres: Tuple[Tuple[float, float], ...] = ()
  # Where the cycle meets the spine, first attachment to last. ``a_s``/``b_s``
  # are wider: they project *every* vertex of the cycle onto its nearest spine
  # point, so ink that bulges above or below the baseline lands on spine
  # positions outside the attachments. Ownership deliberately uses the wider
  # footprint; this is the span a reader sees as the loop's foot.
  attachment: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class SkeletonBranchPath:
  s: float
  points: np.ndarray
  length: float
  source: str = "graph"
  role: str = "branch"


@dataclass(frozen=True)
class TemplateStep:
  kind: str
  target: float
  tolerance: float = 0.30
  min_strength: float = 0.0
  optional: bool = False
  # "inside" events belong to the interval. "after" events are look-ahead
  # context: they constrain its exit without being consumed by it.
  region: str = "inside"


@dataclass(frozen=True)
class SkeletonTemplate:
  """Ordered skeleton-event template for one letter or a ligature."""
  name: str
  letters: Tuple[str, ...]
  forms: Tuple[str, ...]
  steps: Tuple[TemplateStep, ...]
  weight: float = 1.0
  # Optional normalized positions of logical boundaries inside a composite.
  landmarks: Tuple[float, ...] = ()
  # Sparse cues such as an entry lift are positive evidence when visible but
  # must abstain, rather than penalise the many valid forms where they vanish.
  evidence_only: bool = False
  # Optional transcription context immediately following this interval.
  next_letters: Tuple[str, ...] = ()
  next_forms: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ShapeMatch:
  template: str
  letter_start: int
  letter_end: int
  a_s: float
  b_s: float
  score: float
  # Only these events belong to the body and may anchor its default boundary.
  events: Tuple[int, ...] = ()
  # Non-owning recognition context. Adjacent templates may reuse these events.
  before_events: Tuple[int, ...] = ()
  after_events: Tuple[int, ...] = ()


@dataclass(frozen=True)
class ElasticEventAssignment:
  """Owned events plus reusable context selected for one elastic template."""
  events: Tuple[int, ...]
  before_events: Tuple[int, ...] = ()
  after_events: Tuple[int, ...] = ()


@dataclass(frozen=True)
class EventSequenceConstraint:
  """Optional relation between one template event and its predecessor."""
  min_gap_stroke_widths: Optional[float] = None
  max_gap_stroke_widths: Optional[float] = None
  consecutive: bool = False
  adjacent: bool = False


@dataclass(frozen=True)
class EventSpec:
  """One template event: which landmark, and how strong it may be.

  Both bounds travel together deliberately. ``max_strength`` is the mirror of
  ``min_strength``: some letters are defined by a landmark being *small* -- a
  senna's tooth spur is about one stroke width long and no lower bound tells it
  apart from the Kaf junction in front of it. Splitting the two halves across
  two objects left every consumer reading a spec seeing only one of them.
  """
  kind: str
  min_strength: float = 0.0
  max_strength: Optional[float] = None
  # How thick the ink must be where the landmark sits, in stroke widths. This
  # is what identifies the letters whose usual evidence has degenerated: a
  # final Meem whose counter filled in keeps only spur-length branches, but
  # measures 0.69 at its 10th percentile against 0.77 for the 99th percentile
  # of the joins between letters, so the surviving spur is still recognizable
  # by the ink around it.
  min_thickness: float = 0.0
  max_thickness: Optional[float] = None
  # Bounds on ``SkeletonEvent.relief``. Opt-in: a template that means "the ink
  # rises after this, wherever that rise lives" asks for relief; one that means
  # "the spine itself rises" keeps minStrength.
  min_relief: float = 0.0
  max_relief: Optional[float] = None

  def accepts(self, event: "SkeletonEvent") -> bool:
    return (event.kind == self.kind
            and event.strength >= self.min_strength
            and (self.max_strength is None
                 or event.strength <= self.max_strength)
            and event.thickness >= self.min_thickness
            and (self.max_thickness is None
                 or event.thickness <= self.max_thickness)
            and event.relief >= self.min_relief
            and (self.max_relief is None
                 or event.relief <= self.max_relief))

  def as_json(self) -> dict:
    record = {"kind": self.kind, "minStrength": self.min_strength}
    if self.max_strength is not None:
      record["maxStrength"] = self.max_strength
    if self.min_thickness:
      record["minThickness"] = self.min_thickness
    if self.max_thickness is not None:
      record["maxThickness"] = self.max_thickness
    if self.min_relief:
      record["minRelief"] = self.min_relief
    if self.max_relief is not None:
      record["maxRelief"] = self.max_relief
    return record


def _event_specs(specs: Sequence[object]) -> Tuple[EventSpec, ...]:
  """Accept hand-constructed ``(kind, minStrength)`` pairs alongside specs.

  Templates built in code -- the unit tests, and the compatibility helpers --
  predate the spec object, exactly as they predate sequence constraints.
  """
  return tuple(item if isinstance(item, EventSpec) else EventSpec(*item)
               for item in specs)


@dataclass(frozen=True)
class SegmentGroup:
  """One DP interval, consuming one letter or a composite ligature."""
  letter_start: int
  letter_end: int
  a_s: float
  b_s: float
  template: Optional[str] = None


@dataclass(frozen=True)
class BodyBoundarySpec:
  """How an elastic body owns ink next to a variable-length join.

  ``auto`` preserves the generic geometric fallback for compact templates.
  Shape-specific templates can instead name an event and a local stroke-width
  offset, or request horizontal stabilization.  None of these fields encode a
  fraction of the body or join length.
  """
  method: str = "auto"
  anchor_event: Optional[str] = None
  occurrence: str = "auto"
  offset_stroke_widths: float = 0.0
  max_distance_stroke_widths: float = DEFAULT_BODY_EVENT_ENTRY_RANGE_W
  width_threshold_stroke_widths: float = BODY_ENTRY_W
  inset_stroke_widths: float = SHOULDER_INSET_W
  min_horizontality: float = DEFAULT_TEMPLATE_JOIN_HORIZONTAL
  enforce: bool = False


@dataclass(frozen=True)
class ElasticBodyTemplate:
  """Topology-only body template used by candidate-free forced alignment.

  It intentionally has no normalized longitudinal targets.  Required events
  must occur in order inside the body span. ``before_events`` and
  ``after_events`` are non-owning ordered context outside that required
  sequence. Adjacent bodies may reuse contextual events. Any distance between
  events remains elastic unless that event explicitly supplies a normalized
  gap constraint; no absolute pixel spacing or body-width target is stored.
  """
  name: str
  letters: Tuple[str, ...]
  forms: Tuple[str, ...]
  required_events: Tuple[EventSpec, ...]
  before_events: Tuple[EventSpec, ...] = ()
  after_events: Tuple[EventSpec, ...] = ()
  required_event_constraints: Tuple[EventSequenceConstraint, ...] = ()
  before_event_constraints: Tuple[EventSequenceConstraint, ...] = ()
  after_event_constraints: Tuple[EventSequenceConstraint, ...] = ()
  min_loops: int = 0
  max_loops: Optional[int] = None
  min_branches: int = 0
  max_branches: Optional[int] = None
  # Letter context on either side, gating whether this template applies at all.
  # Both are needed because a shape can be conditioned by what precedes it or
  # by what follows: a medial senna is drawn differently when the next letter
  # is a Reh, whose descender pulls the tooth's exit down.
  previous_letters: Tuple[str, ...] = ()
  previous_forms: Tuple[str, ...] = ()
  next_letters: Tuple[str, ...] = ()
  next_forms: Tuple[str, ...] = ()
  # A contextual exit rule may move the cut to the next body's first event.
  # The two letters remain separate; the transition becomes part of this body.
  exit_owns_join_next_letters: Tuple[str, ...] = ()
  # How far past the preceding body's last owned event this body's first
  # required event must lie, in stroke widths. The cross-body cursor is
  # otherwise an event-index rule, which lets this body claim an event at the
  # position the previous one already occupies -- a senna taking the peak of
  # the lam stem in front of it. Zero keeps that historical behaviour.
  min_entry_gap_stroke_widths: float = MIN_ENTRY_GAP_STROKE_WIDTHS
  entry: BodyBoundarySpec = field(default_factory=BodyBoundarySpec)
  exit: BodyBoundarySpec = field(default_factory=BodyBoundarySpec)
  # Weak terminal shapes may validate an interval selected by physical join
  # evidence without being distinctive enough to drive global event ordering.
  ordered_match: bool = True
  # Selection preference among complete, valid ordered paths. This is not a
  # beam-search weight: larger values are considered first only after the
  # bounded topology search has finished.
  priority: int = 0

  def __post_init__(self) -> None:
    for field_name in ("required_events", "before_events", "after_events"):
      object.__setattr__(
          self, field_name, _event_specs(getattr(self, field_name)))


def _body_boundary_spec_from_json(record: object, role: str
                                  ) -> BodyBoundarySpec:
  """Load one optional boundary rule, filling omitted values with defaults."""
  if record is None:
    return BodyBoundarySpec()
  if not isinstance(record, dict):
    raise ValueError(f"elastic template {role} must be an object")
  method = str(record.get("method", "auto"))
  allowed = ({"auto", "event-offset", "observed-event-offset",
              "rise-onset", "preceding-valley", "width-entry"}
             if role == "entry" else
             {"auto", "event-offset", "horizontal-stabilization",
              "width-exit", "width-or-event-exit",
              "observed-event-offset", "branch-reach"})
  if method not in allowed:
    raise ValueError(
        f"unsupported elastic template {role} method: {method}")
  occurrence = str(record.get("occurrence", "auto"))
  if occurrence not in {"auto", "first", "last"}:
    raise ValueError(
        f"elastic template {role} occurrence must be first, last, or auto")
  return BodyBoundarySpec(
      method=method,
      anchor_event=(str(record["anchorEvent"])
                    if "anchorEvent" in record else None),
      occurrence=occurrence,
      offset_stroke_widths=float(record.get("offsetStrokeWidths", 0.0)),
      max_distance_stroke_widths=float(record.get(
          "maxDistanceStrokeWidths", DEFAULT_BODY_EVENT_ENTRY_RANGE_W)),
      width_threshold_stroke_widths=float(record.get(
          "widthThresholdStrokeWidths", BODY_ENTRY_W)),
      inset_stroke_widths=float(record.get(
          "insetStrokeWidths", SHOULDER_INSET_W)),
      min_horizontality=float(record.get(
          "minHorizontality", DEFAULT_TEMPLATE_JOIN_HORIZONTAL)),
      enforce=bool(record.get("enforce", False)))


def skeleton_templates_from_json(payload: object) -> Tuple[SkeletonTemplate, ...]:
  """Load trainable templates while keeping the runtime representation typed."""
  records = payload.get("templates", []) if isinstance(payload, dict) else payload
  if not isinstance(records, list):
    raise ValueError("skeleton template JSON must contain a templates list")
  templates: List[SkeletonTemplate] = []
  for record in records:
    if not isinstance(record, dict):
      raise ValueError("each skeleton template must be an object")
    letters = tuple(record.get("letters", ()))
    forms = tuple(record.get("forms", ()))
    if not letters or len(forms) != len(letters):
      raise ValueError("template letters/forms must be non-empty and equally sized")
    steps = tuple(TemplateStep(
        kind=str(step["kind"]), target=float(step["target"]),
        tolerance=float(step.get("tolerance", 0.30)),
        min_strength=float(step.get("minStrength", 0.0)),
        optional=bool(step.get("optional", False)),
        region=str(step.get("region", "inside")))
        for step in record.get("steps", ()))
    next_letters = tuple(record.get("nextLetters", ()))
    next_forms = tuple(record.get("nextForms", ()))
    if next_letters and len(next_forms) != len(next_letters):
      raise ValueError("template nextLetters/nextForms must be equally sized")
    templates.append(SkeletonTemplate(
        name=str(record["name"]), letters=letters, forms=forms, steps=steps,
        weight=float(record.get("weight", 1.0)),
        landmarks=tuple(float(v) for v in record.get("landmarks", ())),
        evidence_only=bool(record.get("evidenceOnly", False)),
        next_letters=next_letters, next_forms=next_forms))
  return tuple(templates)


def elastic_body_templates_from_json(
    payload: object) -> Tuple[ElasticBodyTemplate, ...]:
  records = (payload.get("elasticBodyTemplates", [])
             if isinstance(payload, dict) else [])
  if not isinstance(records, list):
    raise ValueError("template JSON elasticBodyTemplates must be a list")
  templates: List[ElasticBodyTemplate] = []
  for record in records:
    if not isinstance(record, dict):
      raise ValueError("each elastic body template must be an object")
    letters = tuple(record.get("letters", ()))
    forms = tuple(record.get("forms", ()))
    if not letters or len(forms) != len(letters):
      raise ValueError(
          "elastic template letters/forms must be non-empty and equally sized")
    def parse_events(key: str
                     ) -> Tuple[Tuple[EventSpec, ...],
                                Tuple[EventSequenceConstraint, ...]]:
      specs: List[EventSpec] = []
      constraints: List[EventSequenceConstraint] = []
      for event in record.get(key, ()):
        if not isinstance(event, dict) or "kind" not in event:
          raise ValueError(f"elastic template {key} entries must be objects")
        unknown_fields = set(event) - {
            "kind", "minStrength", "maxStrength", "minThickness",
            "maxThickness", "minRelief", "maxRelief", "gapFromPrevious",
            "consecutive", "adjacent"}
        if unknown_fields:
          names = ", ".join(sorted(unknown_fields))
          raise ValueError(
              f"elastic template {key} event has unsupported field(s): "
              f"{names}")
        gap = event.get("gapFromPrevious")
        if gap is not None and not isinstance(gap, dict):
          raise ValueError("gapFromPrevious must be an object")
        minimum_gap = (float(gap["minStrokeWidths"])
                       if gap is not None and "minStrokeWidths" in gap
                       else None)
        maximum_gap = (float(gap["maxStrokeWidths"])
                       if gap is not None and "maxStrokeWidths" in gap
                       else None)
        if ((minimum_gap is not None and minimum_gap < 0.0)
            or (maximum_gap is not None and maximum_gap < 0.0)
            or (minimum_gap is not None and maximum_gap is not None
                and minimum_gap > maximum_gap)):
          raise ValueError(
              "event gap bounds must satisfy 0 <= minStrokeWidths "
              "<= maxStrokeWidths")
        minimum_strength = float(event.get("minStrength", 0.0))
        maximum_strength = (float(event["maxStrength"])
                            if "maxStrength" in event else None)
        if maximum_strength is not None and maximum_strength < minimum_strength:
          raise ValueError(
              f"elastic template {key} event maxStrength must not be below "
              "minStrength")
        minimum_thickness = float(event.get("minThickness", 0.0))
        maximum_thickness = (float(event["maxThickness"])
                             if "maxThickness" in event else None)
        if (maximum_thickness is not None
            and maximum_thickness < minimum_thickness):
          raise ValueError(
              f"elastic template {key} event maxThickness must not be below "
              "minThickness")
        minimum_relief = float(event.get("minRelief", 0.0))
        maximum_relief = (float(event["maxRelief"])
                          if "maxRelief" in event else None)
        if maximum_relief is not None and maximum_relief < minimum_relief:
          raise ValueError(
              f"elastic template {key} event maxRelief must not be below "
              "minRelief")
        specs.append(EventSpec(str(event["kind"]), minimum_strength,
                               maximum_strength, minimum_thickness,
                               maximum_thickness, minimum_relief,
                               maximum_relief))
        constraints.append(EventSequenceConstraint(
            min_gap_stroke_widths=minimum_gap,
            max_gap_stroke_widths=maximum_gap,
            consecutive=bool(event.get("consecutive", False)),
            adjacent=bool(event.get("adjacent", False))))
      parsed_constraints = tuple(constraints)
      if all(item == EventSequenceConstraint()
             for item in parsed_constraints):
        parsed_constraints = ()
      return tuple(specs), parsed_constraints

    events, event_constraints = parse_events("requiredEvents")
    before_events, before_event_constraints = parse_events("beforeEvents")
    after_events, after_event_constraints = parse_events("afterEvents")
    if not events:
      raise ValueError("elastic body template requires at least one event")
    def padded_constraints(
        specs: Sequence[Tuple[str, float]],
        constraints: Sequence[EventSequenceConstraint]
        ) -> Tuple[EventSequenceConstraint, ...]:
      return tuple(constraints) + (EventSequenceConstraint(),) * max(
          0, len(specs) - len(constraints))

    all_constraints = (
        padded_constraints(before_events, before_event_constraints)
        + padded_constraints(events, event_constraints)
        + padded_constraints(after_events, after_event_constraints))
    if all_constraints and (all_constraints[0].consecutive
                            or all_constraints[0].adjacent
                            or all_constraints[0].min_gap_stroke_widths
                            is not None
                            or all_constraints[0].max_gap_stroke_widths
                            is not None):
      raise ValueError(
          "the first template event cannot constrain a previous event")
    entry = _body_boundary_spec_from_json(record.get("entry"), "entry")
    exit_spec = _body_boundary_spec_from_json(record.get("exit"), "exit")
    owned_event_kinds = {spec.kind for spec in events}
    for role, boundary in (("entry", entry), ("exit", exit_spec)):
      if (boundary.anchor_event is not None
          and boundary.method != "observed-event-offset"
          and boundary.anchor_event not in owned_event_kinds):
        raise ValueError(
            f"elastic template {record.get('name')} {role} anchorEvent "
            f"{boundary.anchor_event!r} is not an owned required event")
    templates.append(ElasticBodyTemplate(
        name=str(record["name"]), letters=letters, forms=forms,
        required_events=events, before_events=before_events,
        after_events=after_events,
        required_event_constraints=event_constraints,
        before_event_constraints=before_event_constraints,
        after_event_constraints=after_event_constraints,
        min_loops=int(record.get("minLoops", 0)),
        max_loops=(int(record["maxLoops"])
                   if "maxLoops" in record else None),
        min_branches=int(record.get("minBranches", 0)),
        max_branches=(int(record["maxBranches"])
                      if "maxBranches" in record else None),
        previous_letters=tuple(record.get("previousLetters", ())),
        previous_forms=tuple(record.get("previousForms", ())),
        next_letters=tuple(record.get("nextLetters", ())),
        next_forms=tuple(record.get("nextForms", ())),
        exit_owns_join_next_letters=tuple(record.get(
            "exitOwnsJoinNextLetters", ())),
        min_entry_gap_stroke_widths=float(
            record.get("minEntryGapStrokeWidths",
                       MIN_ENTRY_GAP_STROKE_WIDTHS)),
        entry=entry, exit=exit_spec,
        ordered_match=bool(record.get("orderedMatch", True)),
        priority=int(record.get("priority", 0))))
    if (templates[-1].min_loops < 0
        or (templates[-1].max_loops is not None
            and templates[-1].max_loops < templates[-1].min_loops)):
      raise ValueError(
          "elastic template loop bounds must satisfy 0 <= minLoops <= maxLoops")
    if (templates[-1].min_branches < 0
        or (templates[-1].max_branches is not None
            and templates[-1].max_branches < templates[-1].min_branches)):
      raise ValueError(
          "elastic template branch bounds must satisfy "
          "0 <= minBranches <= maxBranches")
    if (templates[-1].previous_letters
        and len(templates[-1].previous_letters)
        != len(templates[-1].previous_forms)):
      raise ValueError(
          "elastic previousLetters/previousForms must be equally sized")
    if (templates[-1].next_letters
        and len(templates[-1].next_letters)
        != len(templates[-1].next_forms)):
      raise ValueError(
          "elastic nextLetters/nextForms must be equally sized")
  return tuple(templates)


def load_template_definitions(
    path: str = DEFAULT_TEMPLATE_PATH
    ) -> Tuple[Tuple[SkeletonTemplate, ...],
               Tuple[ElasticBodyTemplate, ...]]:
  """Load every runtime template from the discoverable JSON document."""
  with open(path, encoding="utf-8") as fh:
    payload = json.load(fh)
  return (skeleton_templates_from_json(payload),
          elastic_body_templates_from_json(payload))


def skeleton_template_to_json(template: SkeletonTemplate) -> dict:
  return {
      "name": template.name, "letters": list(template.letters),
      "forms": list(template.forms), "weight": template.weight,
      "landmarks": list(template.landmarks),
      "evidenceOnly": template.evidence_only,
      "nextLetters": list(template.next_letters),
      "nextForms": list(template.next_forms),
      "steps": [
          {"kind": step.kind, "target": step.target,
           "tolerance": step.tolerance, "minStrength": step.min_strength,
           "optional": step.optional, "region": step.region}
          for step in template.steps],
  }


def elastic_body_template_to_json(template: ElasticBodyTemplate) -> dict:
  def event_records(
      specs: Sequence[EventSpec],
      constraints: Sequence[EventSequenceConstraint]) -> List[dict]:
    padded = tuple(constraints) + (EventSequenceConstraint(),) * max(
        0, len(specs) - len(constraints))
    records: List[dict] = []
    for spec, constraint in zip(specs, padded):
      event = spec.as_json()
      gap = {}
      if constraint.min_gap_stroke_widths is not None:
        gap["minStrokeWidths"] = constraint.min_gap_stroke_widths
      if constraint.max_gap_stroke_widths is not None:
        gap["maxStrokeWidths"] = constraint.max_gap_stroke_widths
      if gap:
        event["gapFromPrevious"] = gap
      if constraint.consecutive:
        event["consecutive"] = True
      if constraint.adjacent:
        event["adjacent"] = True
      records.append(event)
    return records

  record = {
      "name": template.name, "letters": list(template.letters),
      "forms": list(template.forms),
      "requiredEvents": event_records(
          template.required_events, template.required_event_constraints),
      "previousLetters": list(template.previous_letters),
      "previousForms": list(template.previous_forms),
  }
  # Written only when set, unlike the previous-letter pair above, so adding the
  # field does not put an empty array on every template that round-trips.
  if template.next_letters:
    record["nextLetters"] = list(template.next_letters)
    record["nextForms"] = list(template.next_forms)
  if template.exit_owns_join_next_letters:
    record["exitOwnsJoinNextLetters"] = list(
        template.exit_owns_join_next_letters)
  if template.min_entry_gap_stroke_widths != MIN_ENTRY_GAP_STROKE_WIDTHS:
    record["minEntryGapStrokeWidths"] = template.min_entry_gap_stroke_widths
  if template.before_events:
    record["beforeEvents"] = event_records(
        template.before_events, template.before_event_constraints)
  if template.after_events:
    record["afterEvents"] = event_records(
        template.after_events, template.after_event_constraints)
  if template.min_loops:
    record["minLoops"] = template.min_loops
  if template.max_loops is not None:
    record["maxLoops"] = template.max_loops
  if template.min_branches:
    record["minBranches"] = template.min_branches
  if template.max_branches is not None:
    record["maxBranches"] = template.max_branches
  if template.entry.method != "auto":
    record["entry"] = _body_boundary_spec_to_json(template.entry)
  if template.exit.method != "auto":
    record["exit"] = _body_boundary_spec_to_json(template.exit)
  if not template.ordered_match:
    record["orderedMatch"] = False
  if template.priority:
    record["priority"] = template.priority
  return record


def _body_boundary_spec_to_json(spec: BodyBoundarySpec) -> dict:
  """Serialize only parameters used by a non-default boundary method."""
  record = {"method": spec.method}
  if spec.anchor_event is not None:
    record["anchorEvent"] = spec.anchor_event
  if spec.occurrence != "auto":
    record["occurrence"] = spec.occurrence
  if spec.offset_stroke_widths != 0.0:
    record["offsetStrokeWidths"] = spec.offset_stroke_widths
  if (spec.method == "preceding-valley"
      and spec.max_distance_stroke_widths
      != DEFAULT_BODY_EVENT_ENTRY_RANGE_W):
    record["maxDistanceStrokeWidths"] = spec.max_distance_stroke_widths
  if (spec.method in {"width-entry", "width-exit"}
      and spec.width_threshold_stroke_widths != BODY_ENTRY_W):
    record["widthThresholdStrokeWidths"] = \
        spec.width_threshold_stroke_widths
  if (spec.method == "width-entry"
      and spec.inset_stroke_widths != SHOULDER_INSET_W):
    record["insetStrokeWidths"] = spec.inset_stroke_widths
  if (spec.method == "horizontal-stabilization"
      and spec.min_horizontality != DEFAULT_TEMPLATE_JOIN_HORIZONTAL):
    record["minHorizontality"] = spec.min_horizontality
  if spec.enforce:
    record["enforce"] = True
  return record


def _template_applies(template: SkeletonTemplate, letters: Sequence[str],
                      forms: Sequence[str], start: int) -> bool:
  end = start + len(template.letters)
  if (end > len(letters) or end > len(forms)
      or any(actual not in allowed
             for allowed, actual in zip(template.letters,
                                        letters[start:end]))):
    return False
  actual_forms = tuple(forms[start:end])
  if not all(want == "*" or want == actual
             for want, actual in zip(template.forms, actual_forms)):
    return False
  if template.next_letters:
    next_end = end + len(template.next_letters)
    if (next_end > len(letters) or next_end > len(forms)
        or any(actual not in allowed
               for allowed, actual in zip(template.next_letters,
                                          letters[end:next_end]))):
      return False
    return all(want == "*" or want == actual
               for want, actual in zip(template.next_forms,
                                       forms[end:next_end]))
  return True


def match_skeleton_template(events: Sequence[SkeletonEvent], a_s: float,
                            b_s: float, template: SkeletonTemplate
                            ) -> Tuple[float, Tuple[int, ...]]:
  """Soft ordered-event alignment, returning score and matched event ids."""
  span = b_s - a_s
  if span <= 1e-6 or not template.steps:
    return 0.0, ()
  # Look-ahead steps may inspect a bounded region beyond the proposed exit,
  # but an ordinary step is still forbidden from claiming those events.
  after_limit = max(
      [1.0] + [step.target + step.tolerance for step in template.steps
               if step.region == "after"])
  local = [(idx, event, (event.s - a_s) / span)
           for idx, event in enumerate(events)
           if a_s < event.s < a_s + after_limit * span]
  n, m = len(template.steps), len(local)
  inf = float("inf")
  dp = np.full((n + 1, m + 1), inf, dtype=float)
  matches = np.full((n + 1, m + 1), -1, dtype=int)
  back: Dict[Tuple[int, int], Tuple[int, int, Optional[int]]] = {}
  dp[0, 0] = 0.0
  matches[0, 0] = 0

  def relax(i: int, j: int, ni: int, nj: int, cost: float,
            matched_event: Optional[int], add_match: int = 0) -> None:
    value = dp[i, j] + cost
    count = matches[i, j] + add_match
    if (value < dp[ni, nj] - 1e-9
        or (abs(value - dp[ni, nj]) <= 1e-9
            and count > matches[ni, nj])):
      dp[ni, nj] = value
      matches[ni, nj] = count
      back[(ni, nj)] = (i, j, matched_event)

  kinds = {step.kind for step in template.steps}
  for i in range(n + 1):
    for j in range(m + 1):
      if not np.isfinite(dp[i, j]):
        continue
      if j < m:
        extra = SKELETON_MATCH_EXTRA if local[j][1].kind in kinds else 0.0
        relax(i, j, i, j + 1, extra, None)
      if i < n:
        missing = 0.15 if template.steps[i].optional else SKELETON_MATCH_MISSING
        relax(i, j, i + 1, j, missing, None)
      step_region_ok = False
      if i < n and j < m:
        event_s = local[j][1].s
        step_region_ok = (
            (template.steps[i].region == "inside" and event_s < b_s)
            or (template.steps[i].region == "after" and event_s > b_s))
      if (i < n and j < m and step_region_ok
          and template.steps[i].kind == local[j][1].kind):
        step, (_, event, pos) = template.steps[i], local[j]
        position_cost = min(2.0, abs(pos - step.target) /
                            max(step.tolerance, 1e-6))
        strength_cost = (max(0.0, step.min_strength - event.strength) /
                         max(step.min_strength, 1.0))
        relax(i, j, i + 1, j + 1, position_cost + strength_cost,
              local[j][0], 1)

  ids: List[int] = []
  node = (n, m)
  while node != (0, 0):
    prev = back.get(node)
    if prev is None:
      break
    if prev[2] is not None:
      ids.append(prev[2])
    node = (prev[0], prev[1])
  ids.reverse()
  required = max(1, sum(not step.optional for step in template.steps))
  # Optional evidence may strengthen alignment but must not make the score
  # exceed the template's advertised maximum.
  quality = (min(matches[n, m], required) - dp[n, m]) / required
  return (W_SKELETON_TEMPLATE * template.weight * float(quality), tuple(ids))


def best_skeleton_match(ctx: Optional[dict], a_s: float, b_s: float,
                        letter_start: int, consume: int = 1
                        ) -> Tuple[float, Optional[SkeletonTemplate], Tuple[int, ...]]:
  if (ctx is None or not ctx.get("skeleton_shapes")
      or "skeleton_events" not in ctx):
    return 0.0, None, ()
  letters, forms = ctx.get("letters", ()), ctx.get("forms", ())
  templates = ctx.get("skeleton_templates", ())
  applicable = [template for template in templates
                if len(template.letters) == consume
                and _template_applies(template, letters, forms, letter_start)]
  if not applicable:
    return 0.0, None, ()
  scored = [(match_skeleton_template(ctx["skeleton_events"], a_s, b_s,
                                     template), template)
            for template in applicable]
  (score, event_ids), template = max(scored, key=lambda item: item[0][0])
  if template.evidence_only and score <= 0.0:
    return 0.0, None, ()
  return score, template, event_ids


@dataclass
class Candidate:
  s: float                       # arclength along the spine
  xy: Tuple[float, float]
  radius: float                  # medial radius at the spine point
  cut_len: float                 # robust local cost incl. bounded foreign ink
  core_cut_len: float            # locally owned run containing the spine point
  owned_cut_len: float           # all locally owned ink severed
  foreign_cut_len: float         # diagnostic: topologically remote crossed ink
  global_cut_len: float          # diagnostic: unbounded full-ray measurement
  n_runs: int                    # locally owned runs crossed
  global_runs: int               # diagnostic: runs across the whole PAW
  recovered: bool                # absent from the full-ray candidate pool
  chord: Tuple[Tuple[float, float], Tuple[float, float]]
  kind: str                      # "min" | "shoulder" | "sample"
  thinness: float
  narrowness: float
  purity: float
  adjacency: float
  baseline_term: float
  horizontality: float
  score: float


@dataclass(frozen=True)
class JoinHypothesis:
  """An elastic connector inferred directly from the dense spine profile."""
  start_index: int
  end_index: int
  event_index: int
  score: float
  kind: str = "rising-kashida"


@dataclass
class JoinSpan:
  """The region between two bodies; no single cut is intrinsic to it."""
  letter_boundary: int
  kind: str
  start: Optional[Candidate] = None
  end: Optional[Candidate] = None
  score: float = 0.0
  confidence: float = 0.0
  template: Optional[str] = None


@dataclass
class BodyJoinAlignment:
  groups: List[SegmentGroup]
  joins: List[JoinSpan]
  cuts: List[Candidate]
  score: float
  margin: float
  shape_matches: List[ShapeMatch] = field(default_factory=list)
  reason: str = ""


@dataclass
class Medial:
  """Pruned medial-axis graph of one PAW."""
  vertices: np.ndarray           # (M, 2)
  radii: np.ndarray              # (M,)
  edges: List[Tuple[int, int]]
  generators: np.ndarray         # (N, 2) contour points the axis was built from
  spine: List[int]               # vertex indices, right to left
  spine_s: np.ndarray            # cumulative arclength along the spine
  loops: int                     # independent cycles = enclosed counters
  # (vertex index into `vertices`, branch length) for every leaf branch
  # _prune_leaves removed. A branch is evidence that the shape has interior
  # structure there -- a letter body rather than a plain connector -- and
  # pruning is aggressive enough (PRUNE_LEN_W = 0.75 stroke widths) to
  # delete genuine short ones, so the record is kept rather than the branch.
  pruned_branches: List[PrunedBranch] = field(default_factory=list)


@dataclass
class PawResult:
  paw_index: int
  letters: List[str]
  forms: List[str]
  ok: bool
  reason: str = ""
  confidence: float = 0.0        # weakest cut, as a fraction of the maximum
  margin: float = 0.0            # lead over the best materially different fit
  merged: bool = False           # this component holds several touching PAWs
  # This PAW's body is drawn into the word before it, so the analyzer never
  # produced a body image for it and there is nothing here to measure. The row
  # exists so the letters keep their place in the word and can be found.
  connected_to_previous_word: bool = False
  low_confidence: bool = False
  stroke_width: float = 0.0
  spine_length: float = 0.0
  loops: int = 0
  candidates: List[Candidate] = field(default_factory=list)
  cuts: List[Candidate] = field(default_factory=list)
  letter_boxes: List[Optional[Tuple[int, int, int, int]]] = field(default_factory=list)
  letter_ink: List[int] = field(default_factory=list)
  # (expected, observed) internal tooth-junction count per letter; expected
  # is None where TOOTH_COUNT has no entry (unconstrained).
  letter_teeth: List[Tuple[Optional[int], int]] = field(default_factory=list)
  # Fix 1b descriptor per letter -- {"extent", "radius", "teeth"}, all in
  # stroke-width-normalized units -- for training letter_shape_priors.py and
  # for scoring against an already-fitted prior. Populated whenever cuts are
  # chosen, independent of whether a prior was supplied.
  letter_shape: List[Dict[str, float]] = field(default_factory=list)
  skeleton_events: List[SkeletonEvent] = field(default_factory=list)
  loop_evidence: List[LoopEvidence] = field(default_factory=list)
  shape_matches: List[ShapeMatch] = field(default_factory=list)
  segment_groups: List[SegmentGroup] = field(default_factory=list)
  joins: List[JoinSpan] = field(default_factory=list)
  body_join_alignment: bool = False
  cut_policy: str = "next-start"
  medial: Optional[Medial] = None
  pixel_letter: Optional[np.ndarray] = None
  # Per join pixel, the letter boundary its connector spans; -1 elsewhere.
  # Only under body-join alignment, where joins are a state of their own.
  pixel_join: Optional[np.ndarray] = None


@dataclass
class WordResult:
  word_id: str
  text: str
  bases: List[str]
  forms: List[str]
  expected_paws: int
  observed_paws: int
  abstain: bool
  baseline_y: float
  paws: List[PawResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# B. mask -> PAW components
# ---------------------------------------------------------------------------

def estimate_stroke_width(mask: np.ndarray) -> float:
  """Rough stroke width from the distance transform.

  For a strip of width W the distance-transform values over its ink are
  distributed roughly uniformly on [0, W/2], so their median is about W/4 --
  not W/2. Hence the factor 4. This is only the bootstrap estimate used to
  size resampling and pruning; once the medial axis exists it is refined to
  twice the median radius along the spine, which is exact.
  """
  dt = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
  vals = dt[mask > 0]
  if vals.size == 0:
    return 0.0
  return 4.0 * float(np.median(vals))


def paw_components(mask: np.ndarray, stroke_width: float) -> List[np.ndarray]:
  """Full-size binary masks, one per PAW, ordered right to left."""
  min_area = SPECKLE_AREA_W2 * stroke_width * stroke_width
  count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
  keep = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= min_area]
  keep.sort(key=lambda i: -(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]))
  return [((labels == i).astype(np.uint8) * 255) for i in keep]


def align_paws(observed_ink: List[float], expected_letters: List[int]
               ) -> List[Tuple[List[int], List[int]]]:
  """Align observed ink components to text-predicted PAWs, monotonically.

  Pairing the two sequences by position is wrong whenever the scan disagrees
  with the text, which happens on 13% of page-3 words. In `alladhina` the
  alef and lam touch, giving 2 components against 3 predicted PAWs; positional
  pairing then told the 3-letter component it held 1 letter (so it was never
  cut) and told the `ya-nun` component it was `lam-dhal`.

  Both sequences run right to left, so the alignment is monotonic. A block of
  one component against several PAWs is a merge (they touch, and the letters
  of all of them must be cut apart inside it); several components against one
  PAW is a broken stroke, which this cannot repair -- the caller abstains.

  Returns one (component indices, PAW indices) block per group, in order.
  """
  n_obs, n_exp = len(observed_ink), len(expected_letters)
  if n_obs == 0 or n_exp == 0:
    return []
  total_ink = max(sum(observed_ink), 1e-9)
  total_letters = max(sum(expected_letters), 1)
  obs = [v / total_ink for v in observed_ink]
  exp = [v / total_letters for v in expected_letters]

  INF = float("inf")
  dp = [[INF] * (n_exp + 1) for _ in range(n_obs + 1)]
  back: Dict[Tuple[int, int], Tuple[int, int]] = {}
  dp[0][0] = 0.0
  for i in range(n_obs + 1):
    for j in range(n_exp + 1):
      if dp[i][j] == INF:
        continue
      # one component absorbing k PAWs (k == 1 is the ordinary case)
      for k in range(1, n_exp - j + 1):
        if i >= n_obs:
          break
        cost = dp[i][j] + abs(obs[i] - sum(exp[j:j + k]))
        if cost < dp[i + 1][j + k]:
          dp[i + 1][j + k] = cost
          back[(i + 1, j + k)] = (i, j)
      # p components sharing one PAW (a broken stroke)
      for p in range(2, n_obs - i + 1):
        if j >= n_exp:
          break
        cost = dp[i][j] + abs(sum(obs[i:i + p]) - exp[j])
        if cost < dp[i + p][j + 1]:
          dp[i + p][j + 1] = cost
          back[(i + p, j + 1)] = (i, j)

  if dp[n_obs][n_exp] == INF:
    return [([i], [i]) for i in range(min(n_obs, n_exp))]
  blocks, node = [], (n_obs, n_exp)
  while node != (0, 0):
    prev = back[node]
    blocks.append((list(range(prev[0], node[0])), list(range(prev[1], node[1]))))
    node = prev
  blocks.reverse()
  return blocks


# ---------------------------------------------------------------------------
# C. contours -> regularized polyline
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PawContours:
  """One PAW's contour tree, with the holes ruled out as scan noise.

  Both the topology mask and the Voronoi generators need exactly this, and
  deriving it costs a contour extraction plus an area and circularity pass per
  hole. Filling an ignored hole does not move any surviving contour -- it only
  removes that hole's own -- so the generators are the nodes minus ``skip``,
  and the whole thing is computed once per PAW rather than once per consumer.
  """
  nodes: List[object]
  skip: Set[int]


def paw_contours(paw_mask: np.ndarray, stroke_width: float) -> PawContours:
  """Trace one PAW and decide which holes are noise rather than counters."""
  nodes = find_contours_with_hierarchy(paw_mask)
  if not nodes:
    return PawContours([], set())
  min_hole_area = MIN_HOLE_AREA_W2 * max(stroke_width, 1e-6) ** 2
  skip = skip_tiny_hole_subtrees(nodes, min_hole_area)
  skip |= sliver_holes(nodes, skip)
  return PawContours(nodes, skip)


def medial_topology_mask(paw_mask: np.ndarray,
                         contours: PawContours) -> np.ndarray:
  """Fill ignored pinholes for topology without changing measured ink.

  ``contour_generators`` omits holes below the counter threshold. Leaving the
  same pixels as background for the later inside-mask test is inconsistent:
  a discarded six-pixel pinhole can still sever Voronoi ridges and split one
  stroke into two medial components. Return a copy only when such holes exist;
  callers retain ``paw_mask`` for cut profiles, attribution, and rendering.
  """
  nodes, skip = contours.nodes, contours.skip
  if not nodes or not skip:
    return paw_mask
  topology_mask = paw_mask.copy()
  for contour_id in skip:
    points = np.rint(nodes[contour_id].pts).astype(np.int32)
    if len(points) >= 3:
      # The complete subtree was already classified as ignorable. Painting it
      # as foreground is equivalent to omitting it from the contour topology.
      cv2.fillPoly(topology_mask, [points], 255)
  return topology_mask


def sliver_holes(nodes, skip: set) -> set:
  """Hole contours too elongated to be a counter, plus their subtrees.

  Circularity is scale-free, so this needs no stroke-width normalization and
  cannot be fooled by a large sliver -- which is the case an area floor misses.
  """
  extra: set = set()

  def mark(index: int) -> None:
    extra.add(index)
    for child in nodes[index].children:
      mark(child)

  for index, node in enumerate(nodes):
    if not node.is_hole or index in skip or index in extra:
      continue
    points = np.asarray(node.pts, dtype=np.float64)
    if len(points) < 8:
      continue
    area = abs(float(signed_area(node.pts)))
    perimeter = float(cv2.arcLength(
        np.rint(points).astype(np.int32).reshape(-1, 1, 2), True))
    if perimeter <= 0.0 or area <= 0.0:
      continue
    if 4.0 * math.pi * area / (perimeter * perimeter) < MIN_HOLE_CIRCULARITY:
      mark(index)
  return extra


def contour_generators(contours: PawContours,
                       stroke_width: float) -> np.ndarray:
  """Outer + real-hole contours, resampled and smoothed, as one point array."""
  nodes, skip = contours.nodes, contours.skip
  if not nodes:
    return np.empty((0, 2))
  step = max(1.5, RESAMPLE_STEP_W * stroke_width)
  sigma = max(1.0, SMOOTH_SIGMA_W * stroke_width)

  chunks: List[np.ndarray] = []
  for i, node in enumerate(nodes):
    if i in skip or len(node.pts) < 8:
      continue
    smoothed = curve_domain_gaussian_smooth(
        [(float(x), float(y)) for x, y in node.pts],
        sigma_px=sigma, resample_step_px=step, closed=True,
    )
    if len(smoothed) >= 4:
      chunks.append(np.asarray(smoothed, dtype=np.float64))
  if not chunks:
    return np.empty((0, 2))
  return np.vstack(chunks)


# ---------------------------------------------------------------------------
# D/E. medial axis, pruning, spine
# ---------------------------------------------------------------------------

def _largest_component(n: int, edges: List[Tuple[int, int]]) -> np.ndarray:
  if not edges:
    return np.zeros(n, dtype=bool)
  rows = [a for a, _ in edges] + [b for _, b in edges]
  cols = [b for _, b in edges] + [a for a, _ in edges]
  adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
  ncomp, labels = connected_components(adj, directed=False)
  if ncomp == 0:
    return np.zeros(n, dtype=bool)
  counts = np.bincount(labels, minlength=ncomp)
  return labels == int(np.argmax(counts))


def _prune_leaves(keep: np.ndarray, verts: np.ndarray,
                  edges: List[Tuple[int, int]], min_len: float
                  ) -> Tuple[np.ndarray, List[Tuple[int, int]],
                             List[PrunedBranch]]:
  """Iteratively drop leaf branches shorter than `min_len`.

  Also returns, for each dropped branch, the junction it hung off and its
  length. The junction itself survives pruning, so that record says "the
  shape had interior structure here" even though the evidence for it is no
  longer in the graph -- see `Medial.pruned_branches`.
  """
  live_edges = [e for e in edges if keep[e[0]] and keep[e[1]]]
  pruned: List[PrunedBranch] = []
  for _ in range(64):
    adj: Dict[int, List[int]] = {}
    for a, b in live_edges:
      adj.setdefault(a, []).append(b)
      adj.setdefault(b, []).append(a)
    removed = set()
    for leaf in [v for v, nb in adj.items() if len(nb) == 1]:
      if leaf in removed:
        continue
      path = [leaf]
      prev, cur = None, leaf
      length = 0.0
      anchor = None
      while True:
        nbrs = [x for x in adj.get(cur, []) if x != prev]
        if len(nbrs) != 1:
          break
        nxt = nbrs[0]
        length += float(np.linalg.norm(verts[nxt] - verts[cur]))
        if len(adj.get(nxt, [])) != 2:      # reached a junction or another leaf
          anchor = nxt
          break
        path.append(nxt)
        prev, cur = cur, nxt
        if length >= min_len:
          break
      if length < min_len and len(adj.get(path[-1], [])) <= 2:
        removed.update(path)
        if anchor is not None:
          point_ids = [anchor] + list(reversed(path))
          pruned.append(PrunedBranch(
              anchor=int(anchor), length=float(length),
              points=np.asarray([verts[v] for v in point_ids], dtype=float)))
    if not removed:
      break
    for v in removed:
      keep[v] = False
    live_edges = [e for e in live_edges if keep[e[0]] and keep[e[1]]]
  return keep, live_edges, pruned


def build_medial(paw_mask: np.ndarray, generators: np.ndarray,
                 stroke_width: float,
                 baseline_y: Optional[float] = None) -> Optional[Medial]:
  if len(generators) < 8:
    return None
  vor = Voronoi(generators)
  verts = vor.vertices
  h, w = paw_mask.shape

  xi = np.rint(verts[:, 0]).astype(int)
  yi = np.rint(verts[:, 1]).astype(int)
  inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
  inside[inside] &= paw_mask[yi[inside], xi[inside]] > 0

  tree = cKDTree(generators)
  radii = np.zeros(len(verts))
  radii[inside], _ = tree.query(verts[inside])
  keep = inside & (radii >= MIN_MEDIAL_RADIUS_W * stroke_width)

  edges = [(int(a), int(b)) for a, b in vor.ridge_vertices
           if a >= 0 and b >= 0 and keep[a] and keep[b]]
  if not edges:
    return None

  keep, edges, pruned = _prune_leaves(keep, verts, edges,
                                      PRUNE_LEN_W * stroke_width)
  keep &= _largest_component(len(verts), edges)
  edges = [e for e in edges if keep[e[0]] and keep[e[1]]]
  if not edges:
    return None

  index = -np.ones(len(verts), dtype=int)
  order = np.flatnonzero(keep)
  index[order] = np.arange(len(order))
  V, R = verts[order], radii[order]
  E = [(int(index[a]), int(index[b])) for a, b in edges]
  loops = len(E) - len(V) + 1     # cycles in one connected component
  # An anchor is only useful if it survived the later filtering steps.
  branches = [PrunedBranch(int(index[p.anchor]), p.length, p.points)
              for p in pruned if keep[p.anchor]]

  spine, spine_s = _extract_spine(V, R, E, stroke_width, baseline_y)
  if spine is None:
    return None
  return Medial(vertices=V, radii=R, edges=E, generators=generators,
                spine=spine, spine_s=spine_s, loops=max(0, loops),
                pruned_branches=branches)


def counter_centroids(paw_mask: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray]:
  """Centre and area of every enclosed background region -- the counters.

  A cycle in the medial graph exists because the ink encloses a hole, so the
  hole itself is where the loop is. Taking it from the mask rather than from
  the graph avoids the one trap in a cycle basis: which independent cycles a
  spanning tree happens to produce is arbitrary, and for a Heh drawn as a
  figure eight one of them runs around *both* counters, putting its centre on
  the crossing instead of inside either counter.

  Scanned ink also encloses specks of background, so areas are returned with
  the centres and the caller keeps only as many as the graph found cycles.
  """
  contours, hierarchy = cv2.findContours(
      (paw_mask > 0).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
  if hierarchy is None:
    return np.empty((0, 2), dtype=float), np.empty(0, dtype=float)
  centres: List[Tuple[float, float]] = []
  areas: List[float] = []
  for contour, record in zip(contours, hierarchy[0]):
    if record[3] < 0:                       # no parent: outer boundary, not a hole
      continue
    moments = cv2.moments(contour)
    if moments["m00"] <= 0.0:
      continue
    centres.append((moments["m10"] / moments["m00"],
                    moments["m01"] / moments["m00"]))
    areas.append(float(moments["m00"]))
  return (np.asarray(centres, dtype=float).reshape(-1, 2),
          np.asarray(areas, dtype=float))


def medial_loop_evidence(medial: Medial,
                         paw_mask: Optional[np.ndarray] = None
                         ) -> List[LoopEvidence]:
  """Localize every independent graph cycle onto the main spine.

  Removing graph bridges leaves maximal cyclic components. Their Euler ranks
  sum to ``medial.loops`` even when two counters share a junction (a rank-two
  component). Each component is represented at the nearest spine point to its
  geometric centre, allowing a proposed letter body to own its local cycles
  without borrowing loops from another letter in the PAW.
  """
  vertex_count = len(medial.vertices)
  if vertex_count == 0 or not medial.edges or medial.loops <= 0:
    return []
  adjacency: List[List[Tuple[int, int]]] = [
      [] for _ in range(vertex_count)]
  for edge_id, (a, b) in enumerate(medial.edges):
    adjacency[a].append((b, edge_id))
    adjacency[b].append((a, edge_id))

  entered = [-1] * vertex_count
  low = [0] * vertex_count
  bridges = set()
  clock = 0
  parent_vertex = [-1] * vertex_count
  parent_edge = [-1] * vertex_count
  next_neighbour = [0] * vertex_count
  for root in range(vertex_count):
    if entered[root] >= 0 or not adjacency[root]:
      continue
    entered[root] = low[root] = clock
    clock += 1
    stack = [root]
    while stack:
      vertex = stack[-1]
      if next_neighbour[vertex] < len(adjacency[vertex]):
        neighbour, edge_id = adjacency[vertex][next_neighbour[vertex]]
        next_neighbour[vertex] += 1
        if edge_id == parent_edge[vertex]:
          continue
        if entered[neighbour] >= 0:
          low[vertex] = min(low[vertex], entered[neighbour])
          continue
        parent_vertex[neighbour] = vertex
        parent_edge[neighbour] = edge_id
        entered[neighbour] = low[neighbour] = clock
        clock += 1
        stack.append(neighbour)
        continue
      stack.pop()
      parent = parent_vertex[vertex]
      if parent < 0:
        continue
      low[parent] = min(low[parent], low[vertex])
      if low[vertex] > entered[parent]:
        bridges.add(parent_edge[vertex])

  cyclic_adjacency: List[List[Tuple[int, int]]] = [
      [] for _ in range(vertex_count)]
  for edge_id, (a, b) in enumerate(medial.edges):
    if edge_id in bridges:
      continue
    cyclic_adjacency[a].append((b, edge_id))
    cyclic_adjacency[b].append((a, edge_id))

  spine_vertices = np.asarray(medial.spine, dtype=int)
  spine_points = medial.vertices[spine_vertices]
  spine_tree = cKDTree(spine_points)
  counters, counter_areas = (
      (np.empty((0, 2), dtype=float), np.empty(0, dtype=float))
      if paw_mask is None else counter_centroids(paw_mask))
  visited = set()
  evidence: List[LoopEvidence] = []
  component_trees: List[cKDTree] = []
  spine_s_of = {int(vertex): float(medial.spine_s[i])
                for i, vertex in enumerate(medial.spine)}
  for seed in range(vertex_count):
    if seed in visited or not cyclic_adjacency[seed]:
      continue
    stack = [seed]
    component_vertices = set()
    component_edges = set()
    while stack:
      vertex = stack.pop()
      if vertex in component_vertices:
        continue
      component_vertices.add(vertex)
      visited.add(vertex)
      for neighbour, edge_id in cyclic_adjacency[vertex]:
        component_edges.add(edge_id)
        if neighbour not in component_vertices:
          stack.append(neighbour)
    rank = len(component_edges) - len(component_vertices) + 1
    if rank <= 0:
      continue
    centre = np.mean(
        medial.vertices[list(component_vertices)], axis=0)
    component_points = medial.vertices[list(component_vertices)]
    _, component_spine_indices = spine_tree.query(component_points)
    component_s = medial.spine_s[
        np.asarray(component_spine_indices, dtype=int)]
    _, local_spine_index = spine_tree.query(centre)
    local_spine_index = int(local_spine_index)
    xy = spine_points[local_spine_index]
    component_trees.append(cKDTree(component_points))
    attached = sorted(spine_s_of[vertex] for vertex in component_vertices
                      if vertex in spine_s_of)
    evidence.append(LoopEvidence(
        s=float(medial.spine_s[local_spine_index]),
        a_s=float(np.min(component_s)), b_s=float(np.max(component_s)),
        count=int(rank),
        xy=(float(xy[0]), float(xy[1])),
        attachment=((attached[0], attached[-1]) if attached else None)))

  # A counter belongs to whichever cyclic component encloses it, and that is
  # the component holding the medial vertex nearest to it. Each component then
  # keeps its ``count`` largest, which discards enclosed scanning noise
  # without needing a size threshold: the graph already counted the real ones.
  if len(counters) and component_trees:
    owners: List[List[int]] = [[] for _ in evidence]
    distances = np.stack(
        [tree.query(counters)[0] for tree in component_trees], axis=1)
    for index, owner in enumerate(np.argmin(distances, axis=1)):
      owners[int(owner)].append(index)
    evidence = [
        replace(item, centres=tuple(
            (float(counters[i][0]), float(counters[i][1]))
            for i in sorted(sorted(found,
                                   key=lambda i: -counter_areas[i]
                                   )[:item.count])))
        for item, found in zip(evidence, owners)]
  evidence.sort(key=lambda item: item.s)
  return evidence


def _spine_extrema(profile: dict, stroke_width: float,
                   valleys: bool) -> Tuple[np.ndarray, np.ndarray]:
  """Return positions and stroke-normalized strengths of spine extrema.

  Image y grows downward. A peak (rise followed by descent) is therefore a
  local minimum of y, while a valley (descent followed by rise) is a local
  maximum. Both use the same prominence and separation so they are exact
  opposites rather than unrelated width-profile signals. Strength is the
  vertical prominence reported by ``find_peaks``, normalized by stroke width.
  The detector threshold is deliberately only a low emission floor; templates
  decide whether an emitted extremum is strong enough through ``minStrength``.
  """
  yp, ds = profile["y"], profile["ds"]
  if yp.size < 3:
    return np.empty(0), np.empty(0), np.empty(0)
  signal = yp if valleys else -yp
  idx, properties = find_peaks(
      signal, prominence=SPINE_EXTREMUM_PROMINENCE_W * stroke_width,
      distance=max(1, int(round(
          SPINE_EXTREMUM_MIN_SEP_W * stroke_width / ds))))
  strengths = (np.asarray(properties["prominences"], dtype=float)
               / max(stroke_width, 1e-6))
  reliefs = _extremum_relief(profile, idx, valleys, stroke_width, strengths)
  return profile["grid"][idx], strengths, reliefs


def _extremum_relief(profile: dict, idx: np.ndarray, valleys: bool,
                     stroke_width: float, strengths: np.ndarray
                     ) -> np.ndarray:
  """``strengths`` recomputed against the off-spine height envelope.

  Positions come from the spine alone, so this only ever re-measures landmarks
  that already exist -- adding ink beside the spine cannot invent or move one.
  The envelope is neutralized at the extremum's own sample, because a branch
  anchored exactly there is part of the landmark and cannot bound it; without
  that, folding the branch in would flatten the extremum out of its own
  signal. The result is clamped at ``strengths`` since extra ink can only ever
  raise a prominence.
  """
  envelope = profile.get("y_up" if valleys else "y_down")
  if envelope is None or idx.size == 0:
    return strengths.copy()
  envelope = np.asarray(envelope, dtype=float).copy()
  envelope[idx] = np.asarray(profile["y"], dtype=float)[idx]
  prominences = peak_prominences(
      envelope if valleys else -envelope, idx)[0]
  return np.maximum(strengths,
                    np.asarray(prominences, dtype=float)
                    / max(stroke_width, 1e-6))


def _branch_height_envelopes(medial: Medial, grid: np.ndarray,
                             yp: np.ndarray, stroke_width: float
                             ) -> Tuple[np.ndarray, np.ndarray]:
  """Spine y with each off-spine branch's extreme y folded in at its anchor.

  Returns (upward, downward): the first pulls y toward the top of the branch
  hanging off each anchor, the second toward its bottom. One grid sample wide
  by construction -- a branch owns a single spine coordinate, and prominence
  only needs a bound, not a shape. Endpoint continuations are excluded: they
  are the PAW running off its own spine, not ink rising beside it, so folding
  them in would let a tail's own descent bound the extremum before it.
  """
  up, down = np.asarray(yp, dtype=float).copy(), np.asarray(yp, float).copy()
  for path in skeleton_branch_paths(medial):
    if path.source != "graph" or path.role == "continuation":
      continue
    index = int(np.clip(np.searchsorted(grid, path.s), 0, len(grid) - 1))
    up[index] = min(up[index], float(path.points[:, 1].min()))
    down[index] = max(down[index], float(path.points[:, 1].max()))
  return up, down


def _spine_radius_profile(profile: dict, medial: Medial, stroke_width: float
                          ) -> Optional[np.ndarray]:
  """Medial-axis radius along the spine, in stroke widths, on the spine grid.

  The radius is half the local ink width, so this is the stroke's thickness
  profile. ``medial.radii`` is per vertex and ``medial.spine_s`` gives each
  spine vertex its arclength, so the two resample onto the event grid the
  same way the spine's own x and y do, smoothed identically.
  """
  spine = np.asarray(medial.spine, dtype=int)
  spine_s = np.asarray(medial.spine_s, dtype=float)
  if spine.size < 3 or spine_s.size != spine.size:
    return None
  radii = np.asarray(medial.radii, dtype=float)[spine] / max(stroke_width, 1e-6)
  order = np.argsort(spine_s)
  values = np.interp(profile["grid"], spine_s[order], radii[order])
  sigma = max(0.0, SKELETON_SMOOTH_W * stroke_width / max(profile["ds"], 1e-6))
  return gaussian_filter1d(values, sigma) if sigma >= 0.5 else values


def spine_peaks(profile: dict, stroke_width: float) -> np.ndarray:
  """Highest spine points: an ascent followed by a descent.

  A third source of "a letter body is here", independent of both the
  branch graph and the cut-cost profile, because on the letters that need
  it neither of those works. On "تُنذِرْهُمْ" the teh's tooth and the noon's
  measure 0.21 and 0.23 stroke widths as medial branches while a spur
  between them measures 0.38, so no length threshold separates them; the
  branch tip direction does not either, since the teh's points downward
  like the spurs. The spine's own height does: it lifts by 0.33 stroke
  widths over the noon's tooth and not at all over the spurs.

  This was formerly called ``spine_rises`` even though it represented only
  the extremum, not the rising interval.
  """
  return _spine_extrema(profile, stroke_width, valleys=False)[0]


def spine_valleys(profile: dict, stroke_width: float) -> np.ndarray:
  """Lowest spine points: a descent followed by an ascent."""
  return _spine_extrema(profile, stroke_width, valleys=True)[0]


def _polyline_length(points: np.ndarray) -> float:
  if len(points) < 2:
    return 0.0
  return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def skeleton_branch_paths(medial: Medial) -> List[SkeletonBranchPath]:
  """Maximal off-spine paths, including geometry discarded by pruning.

  Paths are oriented from their attachment on the main spine toward a leaf.
  A branching subtree yields one path per leaf. This is intentionally a
  geometric diagnostic rather than exact graph isomorphism: Voronoi skeletons
  acquire small topology changes under tiny contour perturbations.
  """
  V = medial.vertices
  adj: Dict[int, List[int]] = {}
  for a, b in medial.edges:
    adj.setdefault(a, []).append(b)
    adj.setdefault(b, []).append(a)
  on_spine = set(medial.spine)
  spine_endpoints = {int(medial.spine[0]), int(medial.spine[-1])}
  spine_s = {int(v): float(medial.spine_s[i])
             for i, v in enumerate(medial.spine)}
  paths: List[SkeletonBranchPath] = []
  used_edges = set()

  def walk(anchor: int, prev: int, cur: int, vertices: List[int]) -> None:
    # Depth-first, but with an explicit stack: an unpruned off-spine chain runs
    # one vertex per medial edge, so a single stray hair on a large PAW is
    # thousands of frames deep and overflows the interpreter stack. Pushing the
    # successors reversed keeps the visit order identical to the recursion,
    # which matters because `used_edges` is mutated as we go.
    stack = [(prev, cur, vertices)]
    while stack:
      prev, cur, vertices = stack.pop()
      edge = tuple(sorted((prev, cur)))
      if edge in used_edges:
        continue
      used_edges.add(edge)
      vertices = vertices + [cur]
      # An off-spine path can terminate at another spine vertex instead of at a
      # leaf.  It is then the alternate side of a medial-graph cycle.  Record
      # the complete attachment-to-attachment path at *both* endpoints.  If we
      # treated it like a leaf branch, traversal from the first endpoint would
      # consume almost the whole cycle and the second endpoint would own only
      # its final closing edge, making branch strength depend on spine order.
      other_anchors = [
          v for v in adj.get(cur, [])
          if v in on_spine and v != anchor
          and tuple(sorted((cur, v))) not in used_edges]
      emitted_connector = False
      for other_anchor in other_anchors:
        used_edges.add(tuple(sorted((cur, other_anchor))))
        pts = V[np.asarray(vertices + [other_anchor], dtype=int)]
        length = _polyline_length(pts)
        paths.append(SkeletonBranchPath(
            s=spine_s[anchor], points=pts, length=length, source="graph"))
        paths.append(SkeletonBranchPath(
            s=spine_s[other_anchor], points=pts[::-1].copy(),
            length=length, source="graph"))
        emitted_connector = True
      nxt = [v for v in adj.get(cur, [])
             if v != prev and v not in on_spine
             and tuple(sorted((cur, v))) not in used_edges]
      if not nxt and not emitted_connector:
        pts = V[np.asarray(vertices, dtype=int)]
        # At either main-spine endpoint, one spine edge plus one off-spine edge
        # is a continuation, not a fork. Preserve its geometry as endpoint body
        # evidence (initial Lam/Teh/Yeh or a final tail), but do not expose it
        # as a branch or let it affect branch-count contracts. Paths
        # reconnecting to the spine are cycles, emitted above as graph paths.
        role = ("continuation"
                if anchor in spine_endpoints and len(adj.get(anchor, [])) == 2
                else "branch")
        paths.append(SkeletonBranchPath(
            s=spine_s[anchor], points=pts,
            length=_polyline_length(pts), source="graph", role=role))
        continue
      stack.extend((cur, v, vertices) for v in reversed(nxt))

  for anchor in medial.spine:
    for nb in adj.get(anchor, []):
      if nb not in on_spine:
        walk(int(anchor), int(anchor), int(nb), [int(anchor)])

  for branch in medial.pruned_branches:
    if branch.anchor in spine_s and len(branch.points) >= 2:
      paths.append(SkeletonBranchPath(
          s=spine_s[branch.anchor], points=branch.points,
          length=branch.length, source="pruned"))
  return paths


def _direction_runs(points: np.ndarray, stroke_width: float
                    ) -> List[Tuple[int, float, int]]:
  """Stable horizontal-direction runs as (sign, length, end-index)."""
  if len(points) < 2:
    return []
  delta = np.diff(points, axis=0)
  lengths = np.linalg.norm(delta, axis=1)
  direction = np.divide(delta[:, 0], lengths,
                        out=np.zeros_like(lengths), where=lengths > 1e-9)
  signs = np.where(direction > SKELETON_DIR_EPS, 1,
                   np.where(direction < -SKELETON_DIR_EPS, -1, 0))
  raw: List[Tuple[int, float, int]] = []
  sign, length = 0, 0.0
  for i, (sgn, seg_len) in enumerate(zip(signs, lengths)):
    sgn = int(sgn)
    if sgn == 0:
      length += float(seg_len)
      continue
    if sign and sgn != sign:
      raw.append((sign, length, i))
      length = 0.0
    sign = sgn
    length += float(seg_len)
  if sign:
    raw.append((sign, length, len(points) - 1))

  floor = SKELETON_DIR_MIN_RUN_W * stroke_width
  stable = [run for run in raw if run[1] >= floor]
  merged: List[Tuple[int, float, int]] = []
  for run in stable:
    if merged and merged[-1][0] == run[0]:
      merged[-1] = (run[0], merged[-1][1] + run[1], run[2])
    else:
      merged.append(run)
  return merged


def skeleton_events(profile: dict, medial: Medial, baseline_y: float,
                    stroke_width: float, peaks: np.ndarray,
                    valleys: np.ndarray,
                    peak_strengths: Optional[np.ndarray] = None,
                    valley_strengths: Optional[np.ndarray] = None,
                    peak_reliefs: Optional[np.ndarray] = None,
                    valley_reliefs: Optional[np.ndarray] = None
                    ) -> List[SkeletonEvent]:
  """Extract soft, stroke-width-normalized events from spine and branches."""
  grid = profile["grid"]
  if grid.size == 0:
    return []
  sigma = max(0.0, SKELETON_SMOOTH_W * stroke_width / max(profile["ds"], 1e-6))
  x = gaussian_filter1d(profile["x"], sigma) if sigma >= 0.5 else profile["x"]
  y = gaussian_filter1d(profile["y"], sigma) if sigma >= 0.5 else profile["y"]
  spine_points = np.column_stack([x, y])
  events: List[SkeletonEvent] = []
  high_paths: Dict[float, List[Tuple[float, Tuple[float, float], str]]] = {}

  runs = _direction_runs(spine_points, stroke_width)
  for prev, cur in zip(runs, runs[1:]):
    idx = int(np.clip(prev[2], 0, len(grid) - 1))
    events.append(SkeletonEvent(
        s=float(grid[idx]), kind="x-right" if cur[0] > 0 else "x-left",
        strength=min(prev[1], cur[1]) / max(stroke_width, 1e-6),
        xy=(float(x[idx]), float(y[idx])), source="spine"))

  def point_at(s: float) -> Tuple[float, float]:
    return (float(np.interp(s, grid, x)), float(np.interp(s, grid, y)))

  if peak_strengths is None:
    peak_strengths = np.ones(len(peaks), dtype=float)
  if valley_strengths is None:
    valley_strengths = np.ones(len(valleys), dtype=float)
  if peak_reliefs is None:
    peak_reliefs = peak_strengths
  if valley_reliefs is None:
    valley_reliefs = valley_strengths
  for kind, positions, strengths, reliefs in (
      ("peak", peaks, peak_strengths, peak_reliefs),
      ("valley", valleys, valley_strengths, valley_reliefs)):
    for s, strength, relief in zip(positions, strengths, reliefs):
      events.append(SkeletonEvent(
          float(s), kind, float(strength), point_at(float(s)),
          relief=float(relief)))

  # A final Alef can be a simple terminal upright.  Its lower turning point is
  # already a valley owned by the preceding letter, while the upright itself
  # has no interior extremum or substantial branch.  Represent the start of a
  # sustained terminal climb explicitly instead of admitting an eventless
  # template.  Requiring both stroke-width-normalized height and steepness
  # keeps long horizontal tails from being mistaken for ascenders.
  if len(valleys):
    valley_s = float(valleys[-1])
    valley_index = int(np.argmin(np.abs(grid - valley_s)))
    if valley_index + 1 < len(grid):
      vertical_gain = float(y[valley_index] - y[-1])
      terminal_height = float(baseline_y - y[-1])
      horizontal_run = abs(float(x[valley_index] - x[-1]))
      if (vertical_gain >= SKELETON_TERMINAL_ASCENT_MIN_W * stroke_width
          and terminal_height
          >= SKELETON_TERMINAL_ASCENT_MIN_W * stroke_width
          and vertical_gain >= (SKELETON_TERMINAL_ASCENT_MIN_SLOPE
                                * horizontal_run)):
        # The event is the end of the preceding connector / entry into the
        # terminal upright. Locate it from the local tangent, not merely at
        # the sample after the valley. Image y decreases during an ascent; at
        # 45 degrees its upward component equals its horizontal component.
        if "tangents" in profile:
          tangents = np.asarray(profile["tangents"], dtype=float)
          dx, dy = tangents[:, 0], tangents[:, 1]
        else:
          dx, dy = np.gradient(x, grid), np.gradient(y, grid)
        angle_threshold = math.tan(math.radians(
            SKELETON_TERMINAL_ASCENT_ANGLE_DEG))
        locally_steep = ((dy < 0.0)
                         & (-dy >= angle_threshold * np.abs(dx)))
        steep_indices = np.flatnonzero(locally_steep[valley_index + 1:])
        if steep_indices.size:
          ascent_index = valley_index + 1 + int(steep_indices[0])
          events.append(SkeletonEvent(
              float(grid[ascent_index]), "terminal-ascent",
              vertical_gain / max(stroke_width, 1e-6),
              (float(x[ascent_index]), float(y[ascent_index])), "spine"))

  for path in skeleton_branch_paths(medial):
    if len(path.points) < 2:
      continue
    anchor_xy = tuple(float(v) for v in path.points[0])
    length_w = path.length / max(stroke_width, 1e-6)
    reach = (float(np.min(path.points[:, 0])), float(np.max(path.points[:, 0])))
    if path.role == "continuation":
      events.append(SkeletonEvent(path.s, "continuation", length_w,
                                  anchor_xy, path.source, reach))
      continue
    events.append(SkeletonEvent(path.s, "branch", length_w, anchor_xy,
                                path.source, reach))
    top_height_w = (max(0.0, baseline_y - float(np.min(path.points[:, 1]))) /
                    max(stroke_width, 1e-6))
    vertical_span_w = (float(np.ptp(path.points[:, 1])) /
                       max(stroke_width, 1e-6))
    if (length_w >= SKELETON_HIGH_BRANCH_W
        and vertical_span_w >= SKELETON_HIGH_BRANCH_W):
      events.append(SkeletonEvent(path.s, "high-branch", top_height_w,
                                  anchor_xy, path.source, reach))
      high_paths.setdefault(round(path.s, 6), []).append(
          (top_height_w, anchor_xy, path.source))
    # Initial kaf's high arm has three stable horizontal runs when traversed
    # from its top leaf back to the baseline: left, right, left. Reversing the
    # stored anchor-to-leaf path negates and reverses the signs, so accept the
    # equivalent +,-,+ representation as well.
    direction_runs = _direction_runs(path.points, stroke_width)
    signs = [run[0] for run in direction_runs]
    min_run_w = (min((run[1] for run in direction_runs), default=0.0) /
                 max(stroke_width, 1e-6))
    zigzag = any(signs[i:i + 3] in ([-1, 1, -1], [1, -1, 1])
                 for i in range(max(0, len(signs) - 2)))
    if (zigzag and min_run_w >= SKELETON_ZIGZAG_MIN_RUN_W
        and vertical_span_w >= SKELETON_HIGH_BRANCH_W):
      events.append(SkeletonEvent(
          path.s, "zigzag-branch", min(top_height_w, min_run_w), anchor_xy,
          path.source, reach))
  for s_key, members in high_paths.items():
    if len(members) >= 2:
      events.append(SkeletonEvent(
          float(s_key), "double-high-branch",
          min(member[0] for member in members), members[0][1], "graph"))

  # The spine's two ends. Every other kind is evidence that something
  # happened, so a template asking for one may find nothing; these are simply
  # where the PAW starts and stops, so they are always present. That is what
  # lets a template measure a gap from the start: ``continuation`` cannot,
  # because it exists only where an end carries an off-spine path (see the
  # role assigned in ``skeleton_branch_paths``), and a bare end -- 11% of
  # Beh-initial PAWs, 32% of Kaf-initial -- emits no event at all today.
  #
  # Two properties are worth testing against, so each end emits two events at
  # the same arclength. ``start``/``end`` carry the terminal path's length in
  # stroke widths, agreeing with the ``continuation`` there wherever one
  # exists and reading 0.0 for a bare end. ``start-height`` / ``end-height``
  # carry the end's rise above the baseline, which is what separates an
  # initial Lam's tall head from an initial Beh's low one.
  #
  # Neither can disturb an ``adjacent`` or ``consecutive`` constraint: those
  # ask whether some event lies strictly between two owned ones, and these are
  # placed at or outside the extremes of everything already emitted. The
  # extremes are taken from the events rather than from the grid because a
  # branch path's arclength comes from the spine graph and can land just past
  # ``grid[-1]``.
  s_start = min([float(grid[0])] + [event.s for event in events])
  s_finish = max([float(grid[-1])] + [event.s for event in events])
  middle = 0.5 * (s_start + s_finish)
  for kind, s_end, at_start in (("start", s_start, True),
                                ("end", s_finish, False)):
    at = point_at(s_end)
    # A continuation only ever sits at one end or the other, so the nearer end
    # owns it; no distance tolerance has to be guessed.
    events.append(SkeletonEvent(
        s_end, kind,
        max((event.strength for event in events
             if event.kind == "continuation"
             and ((event.s <= middle) is at_start)), default=0.0),
        at, "spine"))
    events.append(SkeletonEvent(
        s_end, f"{kind}-height",
        max(0.0, baseline_y - at[1]) / max(stroke_width, 1e-6), at, "spine"))

  # Ink thickness where each landmark sits. Every event gets it, including the
  # branch family, whose anchor is on the spine even when the path leaves it --
  # the question a template asks is how thick the stroke is at the junction.
  # Against the join regions between letters (radius 0.53 median, 0.77 at the
  # 99th percentile), the letter-forms this separates best are Ayn medial
  # (0.86 at its own 10th percentile), Heh medial (0.73) and Meem final
  # (0.69); the thinnest initials -- Feh, Noon, Teh, Yeh, Lam -- sit at
  # 0.48-0.53.
  radius = _spine_radius_profile(profile, medial, stroke_width)
  if radius is not None:
    events = [replace(event, thickness=float(np.interp(event.s, grid, radius)))
              for event in events]

  # Only the extrema carry a distinct relief; for every other kind it is the
  # strength, so a template may bound relief uniformly without having to know
  # which landmarks the envelope applies to.
  events = [event if event.kind in ("peak", "valley")
            else replace(event, relief=event.strength) for event in events]

  events.sort(key=lambda event: (event.s, event.kind))
  return events


def drop_weak_events(events: Sequence[SkeletonEvent]
                     ) -> List[SkeletonEvent]:
  """Discard events too faint to be a stroke feature.

  Templates already gate *ownership* through ``minStrength``, but several
  consumers read only an event's kind: ``branch_count_in_span``, the automatic
  ``preceding-valley`` entry, the ``observed-event-offset`` anchors, and the
  ``adjacent`` sequence constraint. A 0.03-stroke-width prominence -- two
  pixels of wobble along a flat kashida -- must therefore be removed at the
  source to be excluded from all of them at once.

  Only the kinds in ``EVENT_STRENGTH_FLOOR_W`` are filtered, because the other
  kinds measure different quantities on different scales. ``high-branch`` and
  ``zigzag-branch`` cannot be orphaned by this: both require a branch far
  longer than any floor here.
  """
  return [event for event in events
          if event.strength >= EVENT_STRENGTH_FLOOR_W.get(event.kind, 0.0)]


def used_pruned_branches(medial: Medial, stroke_width: float,
                         min_branch_w: float) -> List[Tuple[int, float]]:
  """Pruned branches promoted to positions used by the DP context.

  Keep this filter shared with ``branch_positions`` so the overlay never
  labels a pruned branch that the solver ignored. Multiple records anchored
  at one spine vertex collapse to the same position, just as they do in the
  branch-position array.
  """
  spine_vertices = {int(v) for v in medial.spine}
  threshold = min_branch_w * stroke_width
  selected: List[Tuple[int, float]] = []
  seen = set()
  for branch in medial.pruned_branches:
    vertex, length = int(branch.anchor), branch.length
    if vertex not in spine_vertices or length < threshold or vertex in seen:
      continue
    selected.append((vertex, float(length)))
    seen.add(vertex)
  return selected


def branch_positions(medial: Medial, stroke_width: float,
                     min_branch_w: float = 0.0) -> np.ndarray:
  """Sorted spine arclengths where the shape has interior structure.

  Two sources, because neither alone is complete: vertices that still have
  three or more incident edges, and the junctions where `_prune_leaves`
  removed a branch. Reading only the first would miss every letter whose
  interior branch is shorter than PRUNE_LEN_W -- on "يمد" that is meem's
  entire head, whose branch at s=293.6 is pruned, leaving the segmenter no
  evidence that a letter body sits there at all.

  `min_branch_w` ignores pruned branches shorter than that many stroke
  widths, so scan-noise hair can be excluded without lowering PRUNE_LEN_W
  (which would also let that hair back into the spine).
  """
  deg: Dict[int, int] = {}
  for a, b in medial.edges:
    deg[a] = deg.get(a, 0) + 1
    deg[b] = deg.get(b, 0) + 1
  anchors = {v for v, _ in used_pruned_branches(
      medial, stroke_width, min_branch_w)}
  out = [float(medial.spine_s[i]) for i, v in enumerate(medial.spine)
         if deg.get(v, 0) >= 3 or v in anchors]
  return np.array(sorted(out))


def branch_attachments(medial: Medial) -> Tuple[np.ndarray, np.ndarray]:
  """Off-spine medial length hanging from each spine position.

  Returns (s, cumulative length), so the length attached to any spine range
  is one subtraction. Everything the spine does not itself traverse is
  counted here: the surviving off-spine subtrees, walked outward from each
  spine vertex, plus the branches `_prune_leaves` removed.

  This exists because spine arclength is not a letter's size. A letter
  written as an upright leaves the spine almost at once -- `_extract_spine`
  runs between the rightmost and leftmost substantial vertices, both at
  baseline level, and climbing a stem and back is strictly longer than
  staying low, so no upright is ever on the path. The initial lam of
  "ٱلَّذِينَ" measures 1.20 stroke widths of spine while standing 8.06 tall.
  """
  V = medial.vertices
  adj: Dict[int, List[Tuple[int, float]]] = {}
  for a, b in medial.edges:
    w = float(np.linalg.norm(V[a] - V[b]))
    adj.setdefault(a, []).append((b, w))
    adj.setdefault(b, []).append((a, w))

  on_spine = set(medial.spine)
  extra: Dict[int, float] = {}
  claimed: set = set()
  for v in medial.spine:
    total = 0.0
    stack = [(nb, w) for nb, w in adj.get(v, []) if nb not in on_spine]
    seen: set = set()
    while stack:
      u, w = stack.pop()
      if u in claimed or u in seen:
        continue
      seen.add(u)
      total += w
      for nb, w2 in adj.get(u, []):
        if nb not in on_spine and nb not in seen and nb not in claimed:
          stack.append((nb, w2))
    claimed |= seen
    if total > 0.0:
      extra[v] = extra.get(v, 0.0) + total
  # Pruned branches are gone from the graph but their junction survived, so
  # a letter is not penalised for structure the pruning step deleted.
  for branch in medial.pruned_branches:
    if branch.anchor in on_spine:
      extra[branch.anchor] = extra.get(branch.anchor, 0.0) + branch.length

  s_vals = np.asarray(medial.spine_s, dtype=float)
  lens = np.array([extra.get(v, 0.0) for v in medial.spine], dtype=float)
  return s_vals, np.concatenate([[0.0], np.cumsum(lens)])


def attached_between(s_vals: np.ndarray, cum: np.ndarray,
                     a_s: float, b_s: float) -> float:
  """Off-spine length attached within [a_s, b_s), via the prefix sums."""
  if s_vals.size == 0:
    return 0.0
  i0 = int(np.searchsorted(s_vals, a_s, side="left"))
  i1 = int(np.searchsorted(s_vals, b_s, side="left"))
  return float(cum[min(i1, len(cum) - 1)] - cum[min(i0, len(cum) - 1)])


def segment_extent(a_s: float, b_s: float, stroke_width: float,
                   ctx: Optional[dict]) -> float:
  """A letter's size in stroke widths: spine run plus what hangs off it."""
  run = b_s - a_s
  if ctx is not None and "branch_s" in ctx:
    run += attached_between(ctx["branch_s"], ctx["branch_cum"], a_s, b_s)
  return run / max(stroke_width, 1e-6)


def medial_graph(medial: Medial):
  n = len(medial.vertices)
  rows = [a for a, _ in medial.edges] + [b for _, b in medial.edges]
  cols = [b for _, b in medial.edges] + [a for a, _ in medial.edges]
  wts = [float(np.linalg.norm(medial.vertices[a] - medial.vertices[b]))
         for a, b in medial.edges] * 2
  return coo_matrix((wts, (rows, cols)), shape=(n, n)).tocsr()


def vertex_spine_coordinates(medial: Medial) -> np.ndarray:
  """Project every medial vertex to its graph-nearest spine arclength.

  Euclidean projection is ambiguous where bowls, stems, or neighbouring
  letters pass close to one another. Graph distance follows the ink topology,
  so an upright inherits the spine coordinate of the junction from which it
  actually hangs rather than that of whichever stroke happens to be closest
  in the image.
  """
  dist = dijkstra(medial_graph(medial), indices=medial.spine, min_only=False)
  return medial.spine_s[np.argmin(dist, axis=0)]


def pixel_spine_coordinates(paw_mask: np.ndarray, medial: Medial,
                            vertex_s: Optional[np.ndarray] = None,
                            ink: Optional[Tuple[np.ndarray, np.ndarray]] = None
                            ) -> np.ndarray:
  """Return the topological spine coordinate of every ink pixel.

  Background pixels are NaN. The map depends only on the medial graph and not
  on any proposed cuts, so using it to localise a cut measurement is not
  circular.

  ``ink`` is ``np.nonzero(paw_mask > 0)`` when the caller already has it. A
  PAW mask is the size of the whole padded word, so locating its ink is a
  multi-megapixel scan, and three separate places used to do it per PAW.
  """
  if vertex_s is None:
    vertex_s = vertex_spine_coordinates(medial)
  out = np.full(paw_mask.shape, np.nan, dtype=np.float32)
  ys, xs = np.nonzero(paw_mask > 0) if ink is None else ink
  if xs.size:
    _, idx = cKDTree(medial.vertices).query(np.column_stack([xs, ys]),
                                           workers=QUERY_WORKERS)
    out[ys, xs] = vertex_s[idx]
  return out


def _extract_spine(V: np.ndarray, R: np.ndarray, E: List[Tuple[int, int]],
                   stroke_width: float, baseline_y: Optional[float] = None):
  """Shortest path from the rightmost to the leftmost substantial vertex.

  Writing runs right to left, so this path threads the joins. Where the graph
  contains a bowl the path takes one side arbitrarily; the loop count is
  recorded so a caller can tell that a counter was present.

  The pen enters a PAW on the line, so the right endpoint is the rightmost
  vertex *at baseline level* rather than the rightmost outright. A final Ayn's
  descender sweeps left, down, and back to the right underneath the word: on
  page 18 l09_w05 its bowl reaches further right than the Yeh it follows and
  sits 4.94 stroke widths below the line, so a plain ``argmax(x)`` starts the
  spine inside the bowl and threads the whole PAW backwards. The left endpoint
  keeps the full range of heights, because a final letter legitimately ends in
  a descender, and uses the lower ``SPINE_EXIT_RADIUS_W`` thickness bar,
  because a final letter also legitimately ends by tapering. Falls back to the
  unrestricted pick when the band holds nothing, so a PAW written entirely off
  the line still gets a spine.
  """
  n = len(V)
  rows = [a for a, _ in E] + [b for _, b in E]
  cols = [b for _, b in E] + [a for a, _ in E]
  wts = [float(np.linalg.norm(V[a] - V[b])) for a, b in E] * 2
  graph = coo_matrix((wts, (rows, cols)), shape=(n, n)).tocsr()

  substantial = np.flatnonzero(R >= SUBSTANTIAL_RADIUS_W * stroke_width)
  if substantial.size < 2:
    substantial = np.arange(n)
  start = int(substantial[np.argmax(V[substantial, 0])])
  # Re-pick only when the rightmost vertex is far *below* the line, which no
  # pen entry is: that is a descender that has swept back underneath. Applying
  # the band unconditionally instead costs 372 PAWs against 15 gained, because
  # every template is tuned to the spine as it stands and re-seating the entry
  # on a PAW that was already right moves every event with it.
  if (baseline_y is not None and SPINE_ENTRY_MAX_DROP_W > 0.0
      and V[start, 1] - baseline_y > SPINE_ENTRY_MAX_DROP_W * stroke_width):
    on_line = substantial[V[substantial, 1] - baseline_y
                          <= SPINE_ENTRY_MAX_DROP_W * stroke_width]
    if on_line.size:
      start = int(on_line[np.argmax(V[on_line, 0])])
  exiting = np.flatnonzero(R >= SPINE_EXIT_RADIUS_W * stroke_width)
  if exiting.size < 2:
    exiting = np.arange(n)
  goal = int(exiting[np.argmin(V[exiting, 0])])
  if start == goal:
    return None, None

  dist, pred = dijkstra(graph, indices=start, return_predecessors=True)
  if not np.isfinite(dist[goal]):
    return None, None
  path = [goal]
  while path[-1] != start:
    nxt = int(pred[path[-1]])
    if nxt < 0:
      return None, None
    path.append(nxt)
  path.reverse()                                  # start (right) -> goal (left)

  seg = np.linalg.norm(np.diff(V[path], axis=0), axis=1)
  return path, np.concatenate([[0.0], np.cumsum(seg)])


# ---------------------------------------------------------------------------
# F. cut-cost profile and candidates
# ---------------------------------------------------------------------------

def _flag_runs(flags: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Start and exclusive-end indices of true runs in a boolean array."""
  edges = np.diff(flags.astype(np.int8))
  starts = np.flatnonzero(edges == 1) + 1
  ends = np.flatnonzero(edges == -1) + 1
  if flags[0]:
    starts = np.concatenate([[0], starts])
  if flags[-1]:
    ends = np.concatenate([ends, [len(flags)]])
  return starts, ends


def _ray_window(t: np.ndarray, point: np.ndarray, normal: np.ndarray,
                box: Tuple[int, int, int, int]) -> Tuple[int, int]:
  """Index range of `t` whose samples can possibly land inside `box`.

  `box` is (x0, x1, y0, y1), inclusive in pixels. A sample lands on pixel
  ``rint(point + normal * t)``, so it can only be inside the box while the
  unrounded coordinate is within half a pixel of it. The range is widened by
  an index at each end so that no rounding case sits on the boundary.
  """
  if t.size == 0:
    return 0, 0
  origin = float(t[0])
  low, high = 0.0, float(t.size)
  for axis, (near, far) in ((0, (box[0], box[1])), (1, (box[2], box[3]))):
    direction = float(normal[axis])
    start = float(point[axis])
    if abs(direction) < 1e-12:
      # The ray never moves along this axis, so it is either always within the
      # box's band or never.
      if not (near - 0.5 <= start <= far + 0.5):
        return 0, 0
      continue
    a = (near - 0.5 - start) / direction
    b = (far + 0.5 - start) / direction
    if a > b:
      a, b = b, a
    low = max(low, math.floor(a - origin) - 1.0)
    high = min(high, math.ceil(b - origin) + 2.0)
  first = int(max(0.0, low))
  last = int(min(float(t.size), high))
  return (first, last) if first < last else (0, 0)


def _cut_cost_along(paw_mask: np.ndarray, pixel_s: np.ndarray,
                    sample_s: float, ownership_reach: float,
                    point: np.ndarray, normal: np.ndarray,
                    ink_box: Optional[Tuple[int, int, int, int]] = None
                    ) -> CutMeasure:
  """Measure a perpendicular cut without charging topologically remote ink.

  The full ray is retained for diagnostics. Its production measurement keeps
  only ink pixels whose graph-derived spine coordinate is near ``sample_s``.
  ``core_len`` is the locally owned run containing ``point``; additional local
  runs describe genuine nearby letter structure. A remote upright or bowl may
  intersect the same infinite line, but is counted only in ``foreign_len``.
  """
  h, w = paw_mask.shape
  reach = float(math.hypot(h, w))
  t = np.arange(-reach, reach, 1.0)
  here = (float(point[0]), float(point[1]))
  # The sample positions are fixed by `reach`, the diagonal of the padded
  # *word* -- which is the canvas every PAW mask is drawn on, whatever the
  # PAW's own size -- so most of this ray is guaranteed background. Where the
  # caller knows the mask's ink box, narrow the arrays to the stretch that can
  # reach it. `t` is sliced, never rebuilt, so every retained sample keeps its
  # exact position; the samples dropped have no ink and a NaN spine
  # coordinate, so they belong to no run and add to no length or chord.
  centre = min(t.size - 1, int(math.floor(reach))
               + (0 if reach - math.floor(reach) <= 0.5 else 1))
  if ink_box is not None:
    first, last = _ray_window(t, point, normal, ink_box)
    if first >= last:
      return CutMeasure(0.0, 0.0, 0.0, 0, 0.0, 0, (here, here))
    t = t[first:last]
    # May now fall outside the window, in which case no run contains it --
    # which is the right answer: the ray's midpoint is then off the ink.
    centre -= first
  xs = np.rint(point[0] + normal[0] * t).astype(int)
  ys = np.rint(point[1] + normal[1] * t).astype(int)
  ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
  ink = np.zeros(t.shape, dtype=bool)
  ink[ok] = paw_mask[ys[ok], xs[ok]] > 0
  ray_s = np.full(t.shape, np.nan, dtype=float)
  ray_s[ok] = pixel_s[ys[ok], xs[ok]]
  owned = (ink & np.isfinite(ray_s)
           & (np.abs(ray_s - sample_s) <= ownership_reach))
  if not ink.any():
    return CutMeasure(0.0, 0.0, 0.0, 0, 0.0, 0, (here, here))

  global_starts, _ = _flag_runs(ink)
  starts, ends = _flag_runs(owned)

  chord_a = chord_b = here
  core_len = 0.0
  for a, b in zip(starts, ends):
    if a <= centre < b:
      core_len = float(b - a)
      chord_a = (float(point[0] + normal[0] * t[a]),
                 float(point[1] + normal[1] * t[a]))
      chord_b = (float(point[0] + normal[0] * t[b - 1]),
                 float(point[1] + normal[1] * t[b - 1]))
      break
  global_len = float(ink.sum())
  owned_len = float(owned.sum())
  return CutMeasure(
      core_len=core_len, owned_len=owned_len,
      foreign_len=max(0.0, global_len - owned_len),
      owned_runs=int(len(starts)), global_len=global_len,
      global_runs=int(len(global_starts)), chord=(chord_a, chord_b))


def cut_profile(paw_mask: np.ndarray, medial: Medial, stroke_width: float,
                pixel_s: Optional[np.ndarray] = None):
  """Sample local cut cost, ownership diagnostics, and tangent along the spine."""
  V, R = medial.vertices, medial.radii
  spine, s = medial.spine, medial.spine_s
  ds = max(1.0, PROFILE_STEP_W * stroke_width)
  grid = np.arange(0.0, s[-1], ds)
  if grid.size < 8:
    return None

  xp = np.interp(grid, s, V[spine, 0])
  yp = np.interp(grid, s, V[spine, 1])
  rp = np.interp(grid, s, R[spine])
  half = max(1, int(round(TANGENT_WINDOW_W * stroke_width / ds)))

  if pixel_s is None:
    pixel_s = pixel_spine_coordinates(paw_mask, medial)
  ownership_reach = LOCAL_OWNERSHIP_RANGE_W * stroke_width
  # Once per profile rather than once per sample: see _cut_cost_along.
  mask8 = (paw_mask if paw_mask.dtype == np.uint8
           else (paw_mask > 0).astype(np.uint8))
  box_x, box_y, box_w, box_h = cv2.boundingRect(mask8)
  ink_box = ((box_x, box_x + box_w - 1, box_y, box_y + box_h - 1)
             if box_w > 0 and box_h > 0 else None)

  core_len = np.zeros(grid.size)      # central locally owned run
  owned_len = np.zeros(grid.size)     # all locally owned crossed ink
  foreign_len = np.zeros(grid.size)
  global_cut_len = np.zeros(grid.size)
  n_runs = np.zeros(grid.size, dtype=int)
  global_runs = np.zeros(grid.size, dtype=int)
  tangents = np.zeros((grid.size, 2))
  chords: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
  for i in range(grid.size):
    lo, hi = max(0, i - half), min(grid.size - 1, i + half)
    tan = np.array([xp[hi] - xp[lo], yp[hi] - yp[lo]])
    tn = float(np.linalg.norm(tan))
    tan = tan / tn if tn > 1e-9 else np.array([1.0, 0.0])
    tangents[i] = tan
    measure = _cut_cost_along(
        paw_mask, pixel_s, float(grid[i]), ownership_reach,
        np.array([xp[i], yp[i]]), np.array([-tan[1], tan[0]]), ink_box)
    core_len[i] = measure.core_len
    owned_len[i] = measure.owned_len
    foreign_len[i] = measure.foreign_len
    global_cut_len[i] = measure.global_len
    n_runs[i] = measure.owned_runs
    global_runs[i] = measure.global_runs
    chords.append(measure.chord)
  # Compatibility heuristic, not a geometric cut thickness: the current
  # hand-tuned DP weights were calibrated with this bounded ambiguity term.
  # Removing it requires recalibrating the non-foreign shape/segment model;
  # using the uncapped foreign length is never valid.
  cut_len = owned_len + np.minimum(
      foreign_len, FOREIGN_CUT_CAP_W * stroke_width)
  y_up, y_down = _branch_height_envelopes(medial, grid, yp, stroke_width)
  return dict(grid=grid, total_s=float(s[-1]), ds=ds,
              x=xp, y=yp, y_up=y_up, y_down=y_down, r=rp, cut_len=cut_len,
              core_len=core_len, owned_len=owned_len,
              foreign_len=foreign_len,
              global_cut_len=global_cut_len, n_runs=n_runs,
              global_runs=global_runs, tangents=tangents, chords=chords)


def _regions(flags: np.ndarray) -> List[Tuple[int, int]]:
  """Contiguous [start, end) index ranges where `flags` is True."""
  if not flags.any():
    return []
  d = np.diff(flags.astype(np.int8))
  starts = list(np.flatnonzero(d == 1) + 1)
  ends = list(np.flatnonzero(d == -1) + 1)
  if flags[0]:
    starts = [0] + starts
  if flags[-1]:
    ends = ends + [len(flags)]
  return list(zip(starts, ends))


def _candidate_picks(cost: np.ndarray, cuttable: np.ndarray,
                     stroke_width: float, ds: float) -> Dict[int, str]:
  """Candidate index -> kind for one cut-cost profile.

  Both the unbounded full-ray and topology-owned profiles use the same
  selection policy. Keeping that policy here makes their later merge explicit
  and prevents source-order bookkeeping from leaking into candidate finding.
  """
  min_region = max(2, int(round(MIN_REGION_W * stroke_width / ds)))
  inset = max(1, int(round(SHOULDER_INSET_W * stroke_width / ds)))
  sample_step = max(1, int(round(CAND_SAMPLE_W * stroke_width / ds)))
  picks: Dict[int, str] = {}

  for a, b in _regions(cuttable):
    if b - a < min_region:
      continue
    # shoulders: a letter boundary is where the neighbouring body ends
    for i in (min(a + inset, b - 1), max(b - 1 - inset, a)):
      picks.setdefault(int(i), "shoulder")
    # the thinnest point in the region
    picks.setdefault(int(a + np.argmin(cost[a:b])), "min")
    # and a coarse sweep, so the width prior can choose inside a plateau
    for i in range(a + sample_step, b - 1, sample_step):
      picks.setdefault(int(i), "sample")

  # Body entries: the foot of a rising width edge. Region picks all come from
  # necks, so a boundary visible only where the stroke widens into the next
  # letter body needs this independent crossing candidate. The yeh|meem join
  # in `يمد` is the reference case: coarse neck samples straddle the foot.
  entering = cost > BODY_ENTRY_W * stroke_width
  for i in np.flatnonzero(entering[1:] & ~entering[:-1]):
    if cuttable[i]:
      picks.setdefault(int(i), "edge")
  return picks


def candidate_from_profile_index(
    profile: dict, baseline_y: float, stroke_width: float, index: int,
    kind: str, recovered: bool = False,
    structural_indices: Optional[np.ndarray] = None) -> Candidate:
  """Materialize an already selected spine anchor as a ``Candidate`` record.

  Body/join forced alignment calls this only *after* choosing its boundaries;
  it supplies the existing chord/JSON/overlay representation without routing
  the decision back through ``cut_candidates``.
  """
  i = int(np.clip(index, 0, len(profile["grid"]) - 1))
  sw = max(stroke_width, 1e-6)
  ratio = float(profile["cut_len"][i]) / sw
  thinness = math.exp(-max(ratio - 1.0, 0.0) ** 2)
  narrowness = float(np.clip(1.0 - ratio, 0.0, 1.0))
  owned_extra = max(
      0.0, float(profile["owned_len"][i] - profile["core_len"][i]))
  purity = math.exp(-owned_extra / sw)
  if structural_indices is None:
    structural_indices = np.flatnonzero(
        profile["global_cut_len"] > HIGH_COST_W * stroke_width)
  ahead = structural_indices[structural_indices > i]
  if ahead.size:
    gap = float(ahead[0] - i) * profile["ds"]
    adjacency = float(np.clip(
        1.0 - gap / max(ADJACENCY_RANGE_W * stroke_width, 1e-6), 0.0, 1.0))
  else:
    adjacency = 0.0
  dy = abs(float(profile["y"][i]) - baseline_y)
  baseline_term = math.exp(
      -(dy / max(BASELINE_SIGMA_W * stroke_width, 1e-6)) ** 2)
  horizontality = abs(float(profile["tangents"][i][0]))
  score = (W_THINNESS * thinness + W_NARROWNESS * narrowness
           + W_PURITY * purity + W_ADJACENCY * adjacency
           + W_BASELINE * baseline_term + W_HORIZONTAL * horizontality
           - (W_LOCAL_ONLY_CANDIDATE if recovered else 0.0))
  return Candidate(
      s=float(profile["grid"][i]),
      xy=(float(profile["x"][i]), float(profile["y"][i])),
      radius=float(profile["r"][i]),
      cut_len=float(profile["cut_len"][i]),
      core_cut_len=float(profile["core_len"][i]),
      owned_cut_len=float(profile["owned_len"][i]),
      foreign_cut_len=float(profile["foreign_len"][i]),
      global_cut_len=float(profile["global_cut_len"][i]),
      n_runs=int(profile["n_runs"][i]),
      global_runs=int(profile["global_runs"][i]), recovered=recovered,
      chord=profile["chords"][i], kind=kind, thinness=thinness,
      narrowness=narrowness, purity=purity, adjacency=adjacency,
      baseline_term=baseline_term, horizontality=horizontality,
      score=float(score))


def cut_candidates(profile: dict, baseline_y: float,
                   stroke_width: float) -> List[Candidate]:
  grid, ds = profile["grid"], profile["ds"]
  cut_len = profile["cut_len"]
  core_len = profile["core_len"]

  # Candidate eligibility uses the bounded scoring profile above. An ascender
  # or bowl genuinely attached here can make the location uncuttable, without
  # allowing an unbounded remote crossing to erase the boundary. Run count is
  # deliberately not a hard gate.
  owned_cuttable = ((core_len > 0.0)
                    & (cut_len <= CUTTABLE_MAX_W * stroke_width))
  full_ray_cuttable = ((profile["global_runs"] == 1)
                       & (profile["global_cut_len"]
                          <= CUTTABLE_MAX_W * stroke_width))
  # Adjacency is contextual evidence about where the whole PAW widens, not a
  # measurement of the ink severed by this candidate. Keep it distinct from
  # the topology-owned cut thickness.
  structural = profile["global_cut_len"] > HIGH_COST_W * stroke_width
  struct_idx = np.flatnonzero(structural)

  # Retain the established full-ray locations for candidate recall, then add
  # locations exposed by topology ownership. When both sources choose the same
  # index, the full-ray kind retains precedence.
  full_ray_picks = _candidate_picks(
      profile["global_cut_len"], full_ray_cuttable, stroke_width, ds)
  owned_picks = _candidate_picks(
      cut_len, owned_cuttable, stroke_width, ds)
  picks = dict(full_ray_picks)
  for i, kind in owned_picks.items():
    picks.setdefault(i, kind)

  out: List[Candidate] = []
  for i in sorted(picks):
    recovered = i not in full_ray_picks
    out.append(candidate_from_profile_index(
        profile, baseline_y, stroke_width, i, picks[i], recovered, struct_idx))
  return out


def _elastic_template_applies(template: ElasticBodyTemplate,
                              letters: Sequence[str], forms: Sequence[str],
                              start: int) -> bool:
  """Character-class/form match without any geometric position targets."""
  end = start + len(template.letters)
  if not bool(
      end <= len(letters) and end <= len(forms)
      and all(actual in allowed for allowed, actual in zip(
          template.letters, letters[start:end]))
      and all(want == "*" or want == actual for want, actual in zip(
          template.forms, forms[start:end]))):
    return False
  if template.previous_letters:
    previous_start = start - len(template.previous_letters)
    if previous_start < 0:
      return False
    if any(actual not in allowed for allowed, actual in zip(
        template.previous_letters, letters[previous_start:start])):
      return False
    if not all(want == "*" or want == actual for want, actual in zip(
        template.previous_forms, forms[previous_start:start])):
      return False
  if template.next_letters:
    # ``end`` is one past this template's own letters, so the window starts
    # where the following letter does. A template asking for context the PAW
    # does not have does not apply -- the same rule as looking backwards.
    next_end = end + len(template.next_letters)
    if next_end > len(letters) or next_end > len(forms):
      return False
    if any(actual not in allowed for allowed, actual in zip(
        template.next_letters, letters[end:next_end])):
      return False
    if not all(want == "*" or want == actual for want, actual in zip(
        template.next_forms, forms[end:next_end])):
      return False
  return True


# Which interval of a cyclic component decides which body owns it.
# ``footprint`` is ``a_s``/``b_s``: every vertex of the cycle mapped to its
# nearest spine point, so ink bulging off the baseline widens it past the
# junctions -- on page 3 l11_w07's Heh by 0.74 stroke widths, which lets a
# neighbouring body claim the loop on overlap it does not really have.
# ``attachment`` is where the cycle actually meets the spine, and is the
# definition used: a loop belongs to the letter it is anchored to, not to a
# neighbour its overhang happens to project onto. Measured over 15 pages the
# two agree on all 6832 bodies -- no boundary and no owned event fell in the
# difference -- so this is a definition made honest, not a fix.
LOOP_OWNERSHIP_SPAN = "attachment"


def loop_ownership_span(item: LoopEvidence) -> Tuple[float, float]:
  """The interval of one cyclic component that decides which body owns it."""
  if LOOP_OWNERSHIP_SPAN == "attachment" and item.attachment is not None:
    return item.attachment
  return (item.a_s, item.b_s)


def loops_contained_by_events(
    loop_evidence: Sequence[LoopEvidence],
    events: Sequence[SkeletonEvent],
    event_ids: Sequence[int]) -> int:
  """Count the cyclic components a realization's own events enclose.

  ``minLoops`` says the letter *has* a loop, and the evidence for that is a
  cycle attached to the spine between the letter's first and last matched
  event, endpoints included: a cycle meets the spine exactly at branch points,
  so a Heh's own loops attach on its own branches. Overlap with the proposed
  body span cannot serve here -- a span stretches until it holds most of a
  neighbour's loop, which is how page 9 l01_w04's Hah satisfied
  ``hah-init-loop`` with the following Tah's counter -- whereas the matched
  events are fixed by the match itself.

  A component the spine never enters has no attachment to contain. Its upper
  counter typically sits beside the spine rather than on it, which is the usual
  shape of a two-counter Heh, so it is held by its single projected anchor
  instead. Its footprint would not do: that is every vertex projected onto the
  nearest spine point, and it reaches past the letter's own branches by design.
  """
  if not event_ids:
    return 0
  first = events[event_ids[0]].s
  last = events[event_ids[-1]].s
  spans = [item.attachment or (item.s, item.s) for item in loop_evidence]
  return sum(item.count for item, (a_s, b_s) in zip(loop_evidence, spans)
             if first - 1e-6 <= a_s and b_s <= last + 1e-6)


def loop_count_in_span(loop_evidence: Sequence[LoopEvidence],
                       a_s: float, b_s: float) -> int:
  """Return independent medial cycles whose owned interval overlaps."""
  total = 0
  for item in loop_evidence:
    item_a, item_b = loop_ownership_span(item)
    if min(b_s, item_b) >= max(a_s, item_a) - 1e-6:
      total += item.count
  return total


def template_loop_constraints_match(
    template: ElasticBodyTemplate, loop_evidence: Sequence[LoopEvidence],
    a_s: float, b_s: float) -> bool:
  count = loop_count_in_span(loop_evidence, a_s, b_s)
  return (count >= template.min_loops
          and (template.max_loops is None or count <= template.max_loops))


def loop_partition_constraints_match(
    templates: Sequence[ElasticBodyTemplate],
    body_spans: Sequence[Tuple[float, float]],
    loop_evidence: Sequence[LoopEvidence],
    events: Optional[Sequence[SkeletonEvent]] = None,
    matched_event_ids: Optional[Sequence[Sequence[int]]] = None) -> bool:
  """Assign each cyclic component to its event-owning body and validate.

  A proposed boundary may pass through a loop, leaving slightly more of the
  projected interval in an adjacent body.  Greatest overlap alone would then
  reward the bad boundary by moving the loop out of a zero-loop template.
  Matched topology is stronger evidence.  Among bodies with events inside the
  component, first prefer templates that explicitly declare a loop contract;
  otherwise prefer the body with the most matched events.  Geometric overlap
  only breaks the remaining ties or handles components without matched events.

  That assignment answers ``maxLoops`` -- whether a letter's territory contains
  a loop it should not.  ``minLoops`` asks the opposite question and is settled
  by ``loops_contained_by_events`` instead, which no proposed boundary can
  influence.  Without a matched-event list there is nothing to contain the loop
  with, so ownership answers both questions as before.
  """
  counts = [0] * len(body_spans)
  for item in loop_evidence:
    item_a, item_b = loop_ownership_span(item)
    overlaps = [max(0.0, min(b_s, item_b) - max(a_s, item_a))
                for a_s, b_s in body_spans]
    candidates = list(range(len(body_spans)))
    if events is not None and matched_event_ids is not None:
      event_hits = [
          sum(item_a - 1e-6 <= events[event_id].s <= item_b + 1e-6
              for event_id in event_ids)
          for event_ids in matched_event_ids]
      constrained = [
          i for i, template in enumerate(templates)
          if (template.min_loops > 0 or template.max_loops is not None)
          and event_hits[i] > 0]
      hit_candidates = constrained or [
          i for i, hit_count in enumerate(event_hits) if hit_count > 0]
      if hit_candidates:
        most_hits = max(event_hits[i] for i in hit_candidates)
        candidates = [
            i for i in hit_candidates if event_hits[i] == most_hits]
    if max((overlaps[i] for i in candidates), default=0.0) > 1e-6:
      owner = max(candidates, key=lambda i: overlaps[i])
    else:
      distances = [
          max(a_s - item.s, 0.0, item.s - b_s)
          for a_s, b_s in body_spans]
      owner = min(candidates, key=lambda i: distances[i])
    counts[owner] += item.count
  held = counts if events is None or matched_event_ids is None else [
      loops_contained_by_events(loop_evidence, events, event_ids)
      for event_ids in matched_event_ids]
  return all(
      contained >= template.min_loops
      and (template.max_loops is None or count <= template.max_loops)
          for template, count, contained in zip(templates, counts, held))


def branch_count_in_span(
    events: Sequence[SkeletonEvent], a_s: float, b_s: float) -> int:
  """Count current ordinary branch events in a body's closed span.

  ``high-branch`` and ``zigzag-branch`` are aliases describing the same path,
  so only events whose actual kind is ``branch`` count. No source or strength
  filter is imposed: these are exactly the branch events already exposed to
  templates. Endpoints are included because an event on the body's entry or
  terminal coordinate is visibly part of that body and must obey its bounds.
  """
  return sum(event.kind == "branch"
             and event.s >= a_s - 1e-6 and event.s <= b_s + 1e-6
             for event in events)


def template_branch_constraints_match(
    template: ElasticBodyTemplate, events: Sequence[SkeletonEvent],
    a_s: float, b_s: float) -> bool:
  count = branch_count_in_span(events, a_s, b_s)
  return (count >= template.min_branches
          and (template.max_branches is None
               or count <= template.max_branches))


def branch_partition_constraints_match(
    templates: Sequence[ElasticBodyTemplate],
    body_spans: Sequence[Tuple[float, float]],
    events: Sequence[SkeletonEvent]) -> bool:
  """Validate each body's branch-count contract on the final partition."""
  return all(
      template_branch_constraints_match(template, events, a_s, b_s)
      for template, (a_s, b_s) in zip(templates, body_spans))


def _event_constraints(
    specs: Sequence[EventSpec],
    constraints: Sequence[EventSequenceConstraint]
    ) -> Tuple[EventSequenceConstraint, ...]:
  """Pad hand-constructed templates which predate sequence constraints."""
  return tuple(constraints) + (EventSequenceConstraint(),) * max(
      0, len(specs) - len(constraints))


def _event_sequence_matches(
    events: Sequence[SkeletonEvent], template: ElasticBodyTemplate,
    assignment: ElasticEventAssignment, stroke_width: float) -> bool:
  """Validate ordered event gaps and no-skip requirements."""
  specs = (template.before_events + template.required_events
           + template.after_events)
  constraints = (
      _event_constraints(template.before_events,
                         template.before_event_constraints)
      + _event_constraints(template.required_events,
                           template.required_event_constraints)
      + _event_constraints(template.after_events,
                           template.after_event_constraints))
  event_ids = (assignment.before_events + assignment.events
               + assignment.after_events)
  if len(event_ids) != len(specs):
    return False
  sw = max(float(stroke_width), 1e-6)
  for position in range(1, len(event_ids)):
    previous = events[event_ids[position - 1]]
    current = events[event_ids[position]]
    constraint = constraints[position]
    gap = (current.s - previous.s) / sw
    if (constraint.min_gap_stroke_widths is not None
        and gap < constraint.min_gap_stroke_widths - 1e-6):
      return False
    if (constraint.max_gap_stroke_widths is not None
        and gap > constraint.max_gap_stroke_widths + 1e-6):
      return False
    if constraint.consecutive:
      if any(specs[position].accepts(event)
             and previous.s < event.s < current.s
             for event in events):
        return False
    if constraint.adjacent and any(
        previous.s < event.s < current.s for event in events):
      return False
  return True


def _match_elastic_body_assignment(
    events: Sequence[SkeletonEvent], a_s: float, b_s: float,
    template: ElasticBodyTemplate,
    loop_evidence: Optional[Sequence[LoopEvidence]] = None,
    stroke_width: float = 1.0
    ) -> Tuple[float, Optional[ElasticEventAssignment]]:
  """Match body-owned topology plus reusable before/after context.

  Event coordinates and interval width are deliberately absent from the
  score.  This permits both the body and its neighbouring kashida to stretch
  without changing the template match. Context may lie outside the body span,
  but never becomes body topology or a default boundary anchor.
  """
  if b_s <= a_s:
    return 0.0, None
  if not template_branch_constraints_match(template, events, a_s, b_s):
    return 0.0, None
  if (loop_evidence is not None
      and not template_loop_constraints_match(
          template, loop_evidence, a_s, b_s)):
    return 0.0, None

  assignments = _template_event_options(
      events, template, -1, limit=MAX_ORDERED_EVENT_REALIZATIONS,
      stroke_width=stroke_width, owned_a_s=a_s, owned_b_s=b_s)
  if not assignments:
    return 0.0, None
  assignment = assignments[0]
  return _assignment_score(events, template, assignment), assignment


def match_elastic_body_template(
    events: Sequence[SkeletonEvent], a_s: float, b_s: float,
    template: ElasticBodyTemplate,
    loop_evidence: Optional[Sequence[LoopEvidence]] = None,
    stroke_width: float = 1.0
    ) -> Tuple[float, Tuple[int, ...]]:
  """Compatibility API returning only body-owned required event IDs."""
  score, assignment = _match_elastic_body_assignment(
      events, a_s, b_s, template, loop_evidence, stroke_width)
  return score, (() if assignment is None else assignment.events)


def rising_join_hypotheses(profile: dict,
                           events: Sequence[SkeletonEvent],
                           stroke_width: float) -> List[JoinHypothesis]:
  """Infer elastic rising connectors directly from the dense spine.

  A connector starts where a sustained ascent begins and ends at the local
  neck before the following body continues that ascent.  Only local owned
  ink and medial radius enter the neck measurement, so a nearby foreign
  letter crossing the same perpendicular ray cannot falsify it.  No term
  rewards or penalises connector length.
  """
  grid = np.asarray(profile["grid"], dtype=float)
  if grid.size < 5:
    return []
  ds = max(float(profile["ds"]), 1e-6)
  sw = max(float(stroke_width), 1e-6)
  sigma = BODY_JOIN_SMOOTH_W * sw / ds
  y = gaussian_filter1d(np.asarray(profile["y"], dtype=float), sigma)
  owned = gaussian_filter1d(
      np.asarray(profile["owned_len"], dtype=float) / sw, sigma)
  radius = gaussian_filter1d(
      np.asarray(profile["r"], dtype=float) / sw, sigma)
  dy = np.gradient(y, grid)
  neck_feature = owned + radius

  hypotheses: List[JoinHypothesis] = []
  for event_id, event in enumerate(events):
    if event.kind != "peak":
      continue
    rise_i = int(np.clip(np.searchsorted(grid, event.s), 1, len(grid) - 2))
    crossings = np.flatnonzero(
        (dy[:rise_i - 1] >= 0.0) & (dy[1:rise_i] < 0.0)) + 1
    if crossings.size == 0:
      continue
    start_i = int(crossings[-1])
    if rise_i - start_i < 2:
      continue
    # The body entry is the most economical local cross-section before the
    # rise.  This is a boundary estimate inside an already selected join,
    # not a member of the old candidate pool.
    end_i = start_i + int(np.argmin(neck_feature[start_i:rise_i + 1]))
    if end_i <= start_i or end_i >= rise_i:
      continue

    ascent = dy[start_i:rise_i + 1]
    monotonic = float(np.mean(ascent < -JOIN_ASCENT_EPS))
    simple = float(np.mean(
        np.asarray(profile["n_runs"])[start_i:end_i + 1] == 1))
    shoulder = min(float(neck_feature[start_i]),
                   float(neck_feature[rise_i]))
    prominence = max(0.0, shoulder - float(neck_feature[end_i]))
    neck = float(np.clip(prominence / JOIN_NECK_PROMINENCE_W, 0.0, 1.0))
    slope = float(np.clip(np.max(-ascent) / 0.30, 0.0, 1.0))
    structure = sum(
        1 for other in events
        if grid[start_i] < other.s < grid[end_i]
        and other.kind in {"branch", "high-branch", "double-high-branch",
                           "zigzag-branch"})
    score = float(np.clip(
        0.35 * monotonic + 0.25 * simple + 0.25 * neck + 0.15 * slope
        - 0.20 * structure, 0.0, 1.0))
    if score >= JOIN_MIN_CONFIDENCE:
      hypotheses.append(JoinHypothesis(
          start_index=start_i, end_index=end_i, event_index=event_id,
          score=score))

  # Closely duplicated rise detections imply the same connector.  Retain the
  # stronger one without imposing any distance model on distinct joins.
  hypotheses.sort(key=lambda item: (item.end_index, -item.score))
  deduped: List[JoinHypothesis] = []
  for hypothesis in hypotheses:
    if (deduped
        and abs(hypothesis.end_index - deduped[-1].end_index) <= 1):
      if hypothesis.score > deduped[-1].score:
        deduped[-1] = hypothesis
    else:
      deduped.append(hypothesis)
  return deduped


def _body_partitions(letters: Sequence[str], forms: Sequence[str],
                     templates: Sequence[ElasticBodyTemplate]
                     ) -> List[List[Tuple[int, int,
                                          Optional[ElasticBodyTemplate]]]]:
  """Enumerate complete template-backed body partitions.

  A rising stroke by itself is not enough to identify a letter boundary: it
  may be a tooth inside Seen or another structured body.  Consequently the
  strict body/join path has no anonymous singleton-body fallback.
  """
  out: List[List[Tuple[int, int, Optional[ElasticBodyTemplate]]]] = []

  def visit(start: int,
            groups: List[Tuple[int, int, Optional[ElasticBodyTemplate]]]
            ) -> None:
    if start == len(letters):
      out.append(list(groups))
      return
    for template in templates:
      consume = len(template.letters)
      if _elastic_template_applies(template, letters, forms, start):
        visit(start + consume,
              groups + [(start, start + consume, template)])

  visit(0, [])
  return out


MAX_ORDERED_EVENT_REALIZATIONS = 16
MAX_ORDERED_ALIGNMENT_STATES = 256
MAX_ALIGNMENT_TRACE_EXAMPLES = 8


def _trace_event(events: Sequence[SkeletonEvent], event_id: int) -> dict:
  event = events[event_id]
  return {
      "id": event_id, "kind": event.kind, "s": round(event.s, 2),
      "strength": round(event.strength, 3), "source": event.source}


def _event_sequence_rejection(
    events: Sequence[SkeletonEvent], template: ElasticBodyTemplate,
    assignment: ElasticEventAssignment, stroke_width: float
    ) -> Optional[dict]:
  """Describe the first relational constraint rejected by an assignment."""
  specs = (template.before_events + template.required_events
           + template.after_events)
  constraints = (
      _event_constraints(template.before_events,
                         template.before_event_constraints)
      + _event_constraints(template.required_events,
                           template.required_event_constraints)
      + _event_constraints(template.after_events,
                           template.after_event_constraints))
  event_ids = (assignment.before_events + assignment.events
               + assignment.after_events)
  if len(event_ids) != len(specs):
    return {"reason": "incomplete-assignment"}
  sw = max(float(stroke_width), 1e-6)
  for position in range(1, len(event_ids)):
    previous_id, current_id = event_ids[position - 1], event_ids[position]
    previous, current = events[previous_id], events[current_id]
    constraint = constraints[position]
    gap = (current.s - previous.s) / sw
    if (constraint.min_gap_stroke_widths is not None
        and gap < constraint.min_gap_stroke_widths - 1e-6):
      return {
          "reason": "gap-too-small", "previous": _trace_event(events, previous_id),
          "current": _trace_event(events, current_id),
          "gapStrokeWidths": round(gap, 3),
          "minimum": constraint.min_gap_stroke_widths}
    if (constraint.max_gap_stroke_widths is not None
        and gap > constraint.max_gap_stroke_widths + 1e-6):
      return {
          "reason": "gap-too-large", "previous": _trace_event(events, previous_id),
          "current": _trace_event(events, current_id),
          "gapStrokeWidths": round(gap, 3),
          "maximum": constraint.max_gap_stroke_widths}
    if constraint.consecutive:
      skipped = next((event_id for event_id, event in enumerate(events)
                      if specs[position].accepts(event)
                      and previous.s < event.s < current.s), None)
      if skipped is not None:
        return {
            "reason": "non-consecutive", "blockingEvent": _trace_event(
                events, skipped), "previous": _trace_event(events, previous_id),
            "current": _trace_event(events, current_id)}
    if constraint.adjacent:
      blocking = next((event_id for event_id, event in enumerate(events)
                       if previous.s < event.s < current.s
                       and event.kind not in NON_BLOCKING_EVENT_KINDS), None)
      if blocking is not None:
        return {
            "reason": "not-adjacent", "blockingEvent": _trace_event(
                events, blocking), "previous": _trace_event(events, previous_id),
            "current": _trace_event(events, current_id)}
  return None


def _entry_gap_floor(
    events: Sequence[SkeletonEvent], template: ElasticBodyTemplate,
    after_event: int, stroke_width: float) -> float:
  """Arclength this body's first required event must lie strictly beyond.

  Ordering between bodies is otherwise carried by the event index alone, so
  the next body may own an event at the arclength the previous body's last
  owned event already occupies. Templates that declare a minimum entry gap
  turn that into a distance rule; the default reproduces the index-only
  cursor exactly.
  """
  gap = template.min_entry_gap_stroke_widths
  if after_event < 0 or gap <= 0.0:
    return -float("inf")
  return events[after_event].s + gap * max(float(stroke_width), 1e-6)


def _diagnose_template_event_failure(
    events: Sequence[SkeletonEvent], template: ElasticBodyTemplate,
    after_event: int, stroke_width: float) -> dict:
  """Build one bounded, human-readable near miss for GUI trace mode."""
  cursor = after_event + 1
  last_s = _entry_gap_floor(events, template, after_event, stroke_width)
  owned: List[int] = []
  for spec_i, spec in enumerate(template.required_events):
    found = next((event_id for event_id in range(cursor, len(events))
                  if spec.accepts(events[event_id])
                  and events[event_id].s > last_s + 1e-6), None)
    if found is None:
      earlier = [event_id for event_id in range(0, cursor)
                 if spec.accepts(events[event_id])]
      weak = [event_id for event_id in range(cursor, len(events))
              if events[event_id].kind == spec.kind
              and events[event_id].strength < spec.min_strength]
      strong = [event_id for event_id in range(cursor, len(events))
                if spec.max_strength is not None
                and events[event_id].kind == spec.kind
                and events[event_id].strength > spec.max_strength]
      # An eligible event too close behind the previous body reads as a
      # missing one unless the entry gap is named as the cause.
      crowded = [event_id for event_id in range(cursor, len(events))
                 if spec.accepts(events[event_id])
                 and events[event_id].s <= last_s + 1e-6]
      reason = "required-event-owned-or-before-cursor" if earlier else (
          "required-event-within-entry-gap" if crowded else
          "required-event-below-strength" if weak else
          "required-event-above-strength" if strong else
          "required-event-missing")
      return {
          "reason": reason, "requiredIndex": spec_i,
          "required": spec.as_json(),
          "afterEvent": after_event, "partialEvents": owned,
          "earlierEligible": [
              _trace_event(events, event_id) for event_id in earlier[-3:]],
          "laterWeak": [
              _trace_event(events, event_id) for event_id in weak[:3]],
          "laterStrong": [
              _trace_event(events, event_id) for event_id in strong[:3]]}
    owned.append(found)
    cursor, last_s = found + 1, events[found].s

  # The ordinary matcher may still fail in context or in a sequence relation.
  before_reversed: List[int] = []
  upper = owned[0]
  for reverse_i in range(len(template.before_events) - 1, -1, -1):
    spec = template.before_events[reverse_i]
    found = next((event_id for event_id in range(upper - 1,
                                                  max(0, after_event) - 1, -1)
                  if spec.accepts(events[event_id])
                  and events[event_id].s < events[upper].s - 1e-6), None)
    if found is None:
      return {"reason": "before-context-missing", "contextIndex": reverse_i,
              "required": spec.as_json(),
              "afterEvent": after_event, "partialEvents": owned}
    before_reversed.append(found)
    upper = found
  after: List[int] = []
  cursor, last_s = owned[-1] + 1, events[owned[-1]].s
  for context_i, spec in enumerate(template.after_events):
    found = next((event_id for event_id in range(cursor, len(events))
                  if spec.accepts(events[event_id])
                  and events[event_id].s > last_s + 1e-6), None)
    if found is None:
      return {"reason": "after-context-missing", "contextIndex": context_i,
              "required": spec.as_json(),
              "afterEvent": after_event, "partialEvents": owned}
    after.append(found)
    cursor, last_s = found + 1, events[found].s
  assignment = ElasticEventAssignment(
      tuple(owned), tuple(reversed(before_reversed)), tuple(after))
  relational = _event_sequence_rejection(
      events, template, assignment, stroke_width)
  return relational or {
      "reason": "no-realization-within-search-limit", "afterEvent": after_event,
      "partialEvents": owned}


def _template_event_options(
    events: Sequence[SkeletonEvent], template: ElasticBodyTemplate,
    after_event: int,
    limit: int = MAX_ORDERED_EVENT_REALIZATIONS,
    stroke_width: float = 1.0,
    owned_a_s: Optional[float] = None,
    owned_b_s: Optional[float] = None
    ) -> List[ElasticEventAssignment]:
  """Return bounded owned realizations with reusable context.

  The earliest realization is not necessarily geometrically compatible with
  both body-boundary rules: it can put the following body's first event inside
  an enforced current-body exit. Retain later realizations so boundary
  validation can advance. Only required events must follow ``after_event``;
  before-context may reuse that preceding body's final owned event, and
  after-context does not advance the cursor seen by the following body.
  """
  options: List[ElasticEventAssignment] = []
  owned: List[int] = []
  def add_contextual_assignments() -> None:
    lower = max(0, after_event)
    before_reversed: List[int] = []
    after: List[int] = []

    def visit_after(spec_i: int, cursor: int, last_s: float) -> None:
      if len(options) >= limit:
        return
      if spec_i == len(template.after_events):
        assignment = ElasticEventAssignment(
            tuple(owned), tuple(reversed(before_reversed)), tuple(after))
        if _event_sequence_matches(
            events, template, assignment, stroke_width):
          options.append(assignment)
        return
      spec = template.after_events[spec_i]
      for event_id in range(cursor, len(events)):
        event = events[event_id]
        if not spec.accepts(event) or event.s <= last_s + 1e-6:
          continue
        after.append(event_id)
        visit_after(spec_i + 1, event_id + 1, event.s)
        after.pop()
        if len(options) >= limit:
          return

    def visit_before(reverse_spec_i: int, upper: int) -> None:
      if len(options) >= limit:
        return
      if reverse_spec_i < 0:
        visit_after(0, owned[-1] + 1, events[owned[-1]].s)
        return
      spec = template.before_events[reverse_spec_i]
      # Search nearest-first, retaining farther realizations when a gap
      # constraint rejects the closest contextual event.
      for event_id in range(upper - 1, lower - 1, -1):
        event = events[event_id]
        if not spec.accepts(event) or event.s >= events[upper].s - 1e-6:
          continue
        before_reversed.append(event_id)
        visit_before(reverse_spec_i - 1, event_id)
        before_reversed.pop()
        if len(options) >= limit:
          return

    visit_before(len(template.before_events) - 1, owned[0])

  def visit(spec_i: int, cursor: int, last_s: float) -> None:
    if len(options) >= limit:
      return
    if spec_i == len(template.required_events):
      add_contextual_assignments()
      return
    spec = template.required_events[spec_i]
    for event_id in range(cursor, len(events)):
      event = events[event_id]
      if (not spec.accepts(event)
          or event.s <= last_s + 1e-6
          or (owned_a_s is not None and event.s < owned_a_s)
          or (owned_b_s is not None and event.s > owned_b_s)):
        continue
      owned.append(event_id)
      visit(spec_i + 1, event_id + 1, event.s)
      owned.pop()
      if len(options) >= limit:
        return

  visit(0, after_event + 1,
        _entry_gap_floor(events, template, after_event, stroke_width))
  return options


def _earliest_template_events(
    events: Sequence[SkeletonEvent], template: ElasticBodyTemplate,
    after_event: int) -> Optional[Tuple[int, ...]]:
  """Compatibility helper returning the first ordered realization."""
  options = _template_event_options(events, template, after_event, limit=1)
  return options[0].events if options else None


def _assignment_score(
    events: Sequence[SkeletonEvent], template: ElasticBodyTemplate,
    assignment: ElasticEventAssignment) -> float:
  pairs = list(zip(assignment.before_events, template.before_events))
  pairs.extend(zip(assignment.events, template.required_events))
  pairs.extend(zip(assignment.after_events, template.after_events))
  strengths = [
      min(1.0, events[event_id].strength / max(spec.min_strength, 1e-6))
      for event_id, spec in pairs]
  return float(np.mean(strengths)) if strengths else 1.0


def _ordered_single_body_match_options(
    events: Sequence[SkeletonEvent], letters: Sequence[str],
    forms: Sequence[str], templates: Sequence[ElasticBodyTemplate],
    loop_evidence: Optional[Sequence[LoopEvidence]] = None,
    stroke_width: float = 1.0,
    trace: Optional[dict] = None
    ) -> List[List[Tuple[ElasticBodyTemplate,
                         ElasticEventAssignment, float]]]:
  """Enumerate bounded topology assignments for every letter in order.

  Only one-letter templates marked for ordered matching participate. Earlier
  realizations are preferred because they leave event space for following
  bodies, but later realizations are retained when an enforced adjacent-body
  boundary rejects the earliest one. Multiple template allographs are retained
  because their boundary rules may differ.
  """
  by_position = [
      [template for template in templates
       if len(template.letters) == 1
       and template.ordered_match
       and _elastic_template_applies(template, letters, forms, position)]
      for position in range(len(letters))]
  if trace is not None:
    trace["stages"] = []
  if trace is None and any(not choices for choices in by_position):
    return []

  states: List[Tuple[int, List[Tuple[
      ElasticBodyTemplate, ElasticEventAssignment, float]]]] = [(-1, [])]
  for position, choices in enumerate(by_position):
    stage = None
    template_stats: Optional[Dict[str, dict]] = None
    if trace is not None:
      stage = {
          "position": position, "letter": letters[position],
          "form": forms[position], "incomingStates": len(states),
          "outgoingStates": 0, "templates": [], "failures": [],
          "survivors": [], "rejectionCounts": {}}
      if not choices:
        stage["failures"].append({"reason": "no-applicable-template"})
        stage["rejectionCounts"]["no-applicable-template"] = 1
      trace["stages"].append(stage)
      template_stats = {
          template.name: {"name": template.name, "attempts": 0,
                          "eventOptions": 0, "loopRejected": 0,
                          "survivingOptions": 0}
          for template in choices}
    following: List[Tuple[int, List[Tuple[
        ElasticBodyTemplate, ElasticEventAssignment, float]]]] = []
    for after_event, path in states:
      for template in choices:
        if stage is not None:
          assert template_stats is not None
          template_stats[template.name]["attempts"] += 1
        assignments = _template_event_options(
            events, template, after_event, stroke_width=stroke_width)
        if stage is not None:
          assert template_stats is not None
          template_stats[template.name]["eventOptions"] += len(assignments)
          if not assignments:
            failure = _diagnose_template_event_failure(
                events, template, after_event, stroke_width)
            reason = failure["reason"]
            stage["rejectionCounts"][reason] = (
                stage["rejectionCounts"].get(reason, 0) + 1)
            failure.update({
                "template": template.name,
                "path": [{"template": item[0].name,
                          "events": list(item[1].events)} for item in path]})
            if len(stage["failures"]) < MAX_ALIGNMENT_TRACE_EXAMPLES:
              stage["failures"].append(failure)
        for assignment in assignments:
          # Reject an event realization that visibly enters more cyclic
          # components than this template permits.  Doing this before the
          # bounded beam prevents numerous invalid zero-loop allographs from
          # crowding a valid closed-loop realization out of the state set.
          if loop_evidence is not None and template.max_loops is not None:
            touched_loops = sum(
                item.count for item in loop_evidence
                if any(loop_ownership_span(item)[0] - 1e-6
                       <= events[event_id].s
                       <= loop_ownership_span(item)[1] + 1e-6
                       for event_id in assignment.events))
            if touched_loops > template.max_loops:
              if stage is not None:
                assert template_stats is not None
                template_stats[template.name]["loopRejected"] += 1
                stage["rejectionCounts"]["max-loops-exceeded"] = (
                    stage["rejectionCounts"].get(
                        "max-loops-exceeded", 0) + 1)
                if len(stage["failures"]) < MAX_ALIGNMENT_TRACE_EXAMPLES:
                  stage["failures"].append({
                      "reason": "max-loops-exceeded",
                      "template": template.name,
                      "events": list(assignment.events),
                      "observedLoops": touched_loops,
                      "maxLoops": template.max_loops,
                      "path": [{"template": item[0].name,
                                "events": list(item[1].events)}
                               for item in path]})
              continue
          # The mirror of the rule above. Without it the beam keeps only the
          # realization ending earliest -- page 3 l11_w07's Heh stops one
          # branch short of its own second loop -- and the realization that
          # does contain the loops never reaches the partition check.
          if loop_evidence is not None and template.min_loops > 0:
            held = loops_contained_by_events(
                loop_evidence, events, assignment.events)
            if held < template.min_loops:
              if stage is not None:
                assert template_stats is not None
                template_stats[template.name]["loopRejected"] += 1
                stage["rejectionCounts"]["min-loops-uncontained"] = (
                    stage["rejectionCounts"].get(
                        "min-loops-uncontained", 0) + 1)
                if len(stage["failures"]) < MAX_ALIGNMENT_TRACE_EXAMPLES:
                  stage["failures"].append({
                      "reason": "min-loops-uncontained",
                      "template": template.name,
                      "events": list(assignment.events),
                      "containedLoops": held,
                      "minLoops": template.min_loops,
                      "path": [{"template": item[0].name,
                                "events": list(item[1].events)}
                               for item in path]})
              continue
          if stage is not None:
            assert template_stats is not None
            template_stats[template.name]["survivingOptions"] += 1
          following.append((assignment.events[-1], path + [(
              template, assignment,
              _assignment_score(events, template, assignment))]))
    # Prevent allograph combinations from growing exponentially on long PAWs.
    # Keep the paths which finish earliest, independent of how many event
    # realizations an earlier JSON template produced. This prevents noisy
    # alternatives for one body from crowding the ordinary earliest path out
    # of the bounded beam.
    following.sort(key=lambda state: (
        state[0], sum(sum(item[1].events) for item in state[1])))
    if stage is not None:
      assert template_stats is not None
      stage["outgoingStates"] = min(
          len(following), MAX_ORDERED_ALIGNMENT_STATES)
      stage["generatedStates"] = len(following)
      stage["beamDiscarded"] = max(
          0, len(following) - MAX_ORDERED_ALIGNMENT_STATES)
      stage["templates"] = list(template_stats.values())
      stage["survivors"] = [
          {"lastEvent": after_event,
           "path": [{"template": item[0].name,
                     "events": list(item[1].events),
                     "beforeEvents": list(item[1].before_events),
                     "afterEvents": list(item[1].after_events)}
                    for item in path]}
          for after_event, path in following[:MAX_ALIGNMENT_TRACE_EXAMPLES]]
    states = following[:MAX_ORDERED_ALIGNMENT_STATES]
    if not states:
      if trace is not None:
        trace["result"] = "topology-failed"
        trace["firstFailingPosition"] = position
      return []
  if trace is not None:
    trace["result"] = "topology-matched"
    trace["completePaths"] = len(states)
  return [path for _, path in states]


def _ordered_single_body_matches(
    events: Sequence[SkeletonEvent], letters: Sequence[str],
    forms: Sequence[str], templates: Sequence[ElasticBodyTemplate]
    ) -> Optional[List[Tuple[ElasticBodyTemplate, Tuple[int, ...], float]]]:
  """Return the established earliest assignment (compatibility helper)."""
  options = _ordered_single_body_match_options(
      events, letters, forms, templates)
  if not options:
    return None
  return [(template, assignment.events, score)
          for template, assignment, score in options[0]]


def _ordered_fixed_template_match_options(
    events: Sequence[SkeletonEvent],
    templates: Sequence[ElasticBodyTemplate],
    loop_evidence: Optional[Sequence[LoopEvidence]] = None,
    stroke_width: float = 1.0,
    ) -> List[List[Tuple[ElasticBodyTemplate,
                         ElasticEventAssignment, float]]]:
  """Bounded ordered assignments for one already selected body partition.

  Unlike ``_ordered_single_body_match_options``, the caller has already
  chosen the templates and their letter spans, so a template may consume more
  than one transcription letter. The event cursor still advances once per
  physical body, which is exactly what a composite template represents.
  """
  states: List[Tuple[int, List[Tuple[
      ElasticBodyTemplate, ElasticEventAssignment, float]]]] = [(-1, [])]
  for template in templates:
    following: List[Tuple[int, List[Tuple[
        ElasticBodyTemplate, ElasticEventAssignment, float]]]] = []
    for after_event, path in states:
      for assignment in _template_event_options(
          events, template, after_event, stroke_width=stroke_width):
        if loop_evidence is not None and template.max_loops is not None:
          touched_loops = sum(
              item.count for item in loop_evidence
              if any(loop_ownership_span(item)[0] - 1e-6
                     <= events[event_id].s
                     <= loop_ownership_span(item)[1] + 1e-6
                     for event_id in assignment.events))
          if touched_loops > template.max_loops:
            continue
        if (loop_evidence is not None and template.min_loops > 0
            and loops_contained_by_events(
                loop_evidence, events, assignment.events)
            < template.min_loops):
          continue
        following.append((assignment.events[-1], path + [(
            template, assignment,
            _assignment_score(events, template, assignment))]))
    following.sort(key=lambda state: (
        state[0], sum(sum(item[1].events) for item in state[1])))
    states = following[:MAX_ORDERED_ALIGNMENT_STATES]
    if not states:
      return []
  return [path for _, path in states]


def _complete_ordered_path_selection_key(
    path: Sequence[Tuple[ElasticBodyTemplate,
                         ElasticEventAssignment, float]],
    template_order: Dict[int, int]) -> Tuple[object, ...]:
  """Rank a complete path without changing bounded-search exploration.

  The topology beam prefers early-ending partial assignments because they
  leave event space for later letters. Once a path is complete, however, that
  feasibility heuristic must not decide which allograph wins. Prefer an
  explicit author priority, then the template which explains more and stronger
  required topology. JSON order is the deterministic final semantic tie-break.
  """
  priority = sum(template.priority for template, _, _ in path)
  required_count = sum(
      len(template.required_events) for template, _, _ in path)
  required_strength = sum(
      spec.min_strength
      for template, _, _ in path
      for spec in template.required_events)
  file_order = tuple(
      template_order.get(id(template), len(template_order))
      for template, _, _ in path)
  event_order = tuple(
      assignment.events for _, assignment, _ in path)
  return (-priority, -required_count, -required_strength,
          file_order, event_order)


def _boundary_anchor_event(
    events: Sequence[SkeletonEvent], event_ids: Sequence[int],
    spec: BodyBoundarySpec, role: str) -> Optional[int]:
  """Resolve a boundary anchor among the events owned by one template."""
  matching = [
      event_id for event_id in event_ids
      if spec.anchor_event is None
      or events[event_id].kind == spec.anchor_event]
  if not matching:
    return None
  occurrence = spec.occurrence
  if occurrence == "auto":
    occurrence = "first" if role == "entry" else "last"
  return matching[0] if occurrence == "first" else matching[-1]


def _template_body_boundary_index(
    profile: dict, events: Sequence[SkeletonEvent], current_last_event: int,
    next_event_ids: Sequence[int], spec: BodyBoundarySpec,
    stroke_width: float) -> Optional[Tuple[int, str, float]]:
  """Evaluate the following template's entry rule in the event gap."""
  grid = profile["grid"]
  sw = max(stroke_width, 1e-6)
  a_s = events[current_last_event].s
  next_first_event = next_event_ids[0]
  b_s = events[next_first_event].s
  lo = int(np.searchsorted(grid, a_s, side="right"))
  hi = int(np.searchsorted(grid, b_s, side="left"))
  lo = int(np.clip(lo, 0, len(grid) - 1))
  hi = int(np.clip(hi, lo, len(grid) - 1))

  def event_entry() -> Optional[Tuple[int, str, float]]:
    event_id = _boundary_anchor_event(
        events, next_event_ids, spec, "entry")
    if event_id is None:
      return None
    entry_s = float(np.clip(
        events[event_id].s + spec.offset_stroke_widths * sw, a_s, b_s))
    # The next body must include its anchor event.  Choose the dense sample on
    # the connector side when the exact event falls between profile samples.
    entry_index = int(np.searchsorted(grid, entry_s, side="right") - 1)
    entry_index = int(np.clip(entry_index, lo, hi))
    return (entry_index, "template-event-entry", 0.80)

  def observed_event_entry() -> Optional[Tuple[int, str, float]]:
    """Offset from an observed event in the following template context."""
    if spec.anchor_event is None:
      return None
    context_end_s = events[next_event_ids[-1]].s
    matching = [event for event in events
                if a_s < event.s <= context_end_s
                and event.kind == spec.anchor_event]
    if not matching:
      return None
    occurrence = spec.occurrence
    if occurrence == "auto":
      occurrence = "first"
    anchor = matching[0] if occurrence == "first" else matching[-1]
    lower_s = float(grid[0]) if spec.enforce else a_s
    entry_s = float(np.clip(
        anchor.s + spec.offset_stroke_widths * sw, lower_s, context_end_s))
    entry_index = int(np.argmin(np.abs(grid - entry_s)))
    return (entry_index, "template-observed-event-entry", 0.90)

  def valley_entry() -> Optional[Tuple[int, str, float]]:
    nearby_valleys = [
        event for event in events
        if event.kind == "valley" and a_s < event.s < b_s
        and b_s - event.s <= spec.max_distance_stroke_widths * sw]
    if not nearby_valleys:
      return None
    valley = nearby_valleys[-1]
    return (int(np.argmin(np.abs(grid - valley.s))),
            "template-valley-entry", 1.0)

  def rise_onset_entry() -> Optional[Tuple[int, str, float]]:
    """Start the following body where its smoothed ascent actually begins."""
    event_id = _boundary_anchor_event(
        events, next_event_ids, spec, "entry")
    if event_id is None:
      return None
    anchor_index = int(np.clip(np.searchsorted(
        grid, events[event_id].s, side="left"), lo, hi))
    if anchor_index <= lo:
      return None
    ds = max(float(profile["ds"]), 1e-6)
    sigma = BODY_JOIN_SMOOTH_W * sw / ds
    smooth_y = gaussian_filter1d(
        np.asarray(profile["y"], dtype=float), sigma)
    dy = np.gradient(smooth_y, np.asarray(grid, dtype=float))
    crossings = np.flatnonzero(
        (dy[lo:anchor_index] >= 0.0)
        & (dy[lo + 1:anchor_index + 1] < 0.0)) + lo + 1
    if not crossings.size:
      return None
    return (int(crossings[-1]), "template-rise-onset", 0.90)

  def width_entry() -> Optional[Tuple[int, str, float]]:
    entering = (np.asarray(profile["owned_len"])
                > spec.width_threshold_stroke_widths * sw)
    crossings = np.flatnonzero(entering[1:] & ~entering[:-1]) + 1
    crossings = crossings[(crossings >= lo) & (crossings <= hi)]
    if not crossings.size:
      return None
    inset = max(1, int(round(
        spec.inset_stroke_widths * sw / profile["ds"])))
    return (max(lo, int(crossings[-1]) - inset),
            "template-width-entry", 0.90)

  evaluators = {
      "event-offset": event_entry,
      "observed-event-offset": observed_event_entry,
      "rise-onset": rise_onset_entry,
      "preceding-valley": valley_entry,
      "width-entry": width_entry,
  }
  if spec.method != "auto":
    result = evaluators[spec.method]()
    return result

  # Compact templates default to the generic observed-neck cascade.  The
  # final event offset is a local one-stroke fallback, never a body/join ratio.
  for evaluator in (valley_entry, width_entry):
    result = evaluator()
    if result is not None:
      return result
  fallback = BodyBoundarySpec(
      method="event-offset", occurrence="first",
      offset_stroke_widths=-1.0)
  event_id = _boundary_anchor_event(
      events, next_event_ids, fallback, "entry")
  body_start_s = max(a_s, events[event_id].s - sw)
  return (int(np.argmin(np.abs(grid - body_start_s))),
          "template-event-entry", 0.80)


def _template_body_exit_index(
    profile: dict, events: Sequence[SkeletonEvent],
    current_event_ids: Sequence[int], spec: BodyBoundarySpec,
    join_end_index: int, stroke_width: float) -> Optional[int]:
  """Evaluate the current template's exit rule before its elastic join."""
  grid = profile["grid"]
  sw = max(stroke_width, 1e-6)
  current_last_event = current_event_ids[-1]
  event_index = int(np.searchsorted(
      grid, events[current_last_event].s, side="left"))
  event_index = int(np.clip(event_index, 0, len(grid) - 1))
  event = events[current_last_event]

  def event_exit() -> Optional[int]:
    event_id = _boundary_anchor_event(
        events, current_event_ids, spec, "exit")
    if event_id is None:
      return None
    body_end_s = float(np.clip(
        events[event_id].s + spec.offset_stroke_widths * sw,
        float(grid[0]), float(grid[join_end_index])))
    return int(np.argmin(np.abs(grid - body_end_s)))

  def horizontal_exit() -> Optional[int]:
    anchor = _boundary_anchor_event(
        events, current_event_ids, spec, "exit")
    anchor_index = event_index if anchor is None else int(np.searchsorted(
        grid, events[anchor].s, side="left"))
    horizontal = np.flatnonzero(
        np.abs(np.asarray(profile["tangents"])[:, 0])
        >= spec.min_horizontality)
    horizontal = horizontal[
        (horizontal > anchor_index) & (horizontal < join_end_index)]
    return int(horizontal[0]) if horizontal.size else None

  def width_exit() -> Optional[int]:
    anchor = _boundary_anchor_event(
        events, current_event_ids, spec, "exit")
    anchor_index = event_index if anchor is None else int(np.searchsorted(
        grid, events[anchor].s, side="left"))
    entering = (np.asarray(profile["owned_len"])
                > spec.width_threshold_stroke_widths * sw)
    falling = np.flatnonzero(~entering[1:] & entering[:-1]) + 1
    falling = falling[(falling > anchor_index) & (falling < join_end_index)]
    return int(falling[0]) if falling.size else None

  def width_or_event_exit() -> Optional[int]:
    """Prefer an observed body neck; otherwise keep the full matched shape."""
    result = width_exit()
    if result is not None:
      return result
    fallback_s = float(np.clip(
        event.s + spec.offset_stroke_widths * sw,
        float(grid[0]), float(grid[join_end_index])))
    return int(np.argmin(np.abs(grid - fallback_s)))

  def observed_event_exit() -> Optional[int]:
    """Offset from an observed event inside the full template context.

    Unlike ``event-offset``, the anchor need not be a required template event.
    This lets a stable topological context delimit a variable number of graph
    branches while the boundary uses the last actually observed branch.
    """
    if spec.anchor_event is None:
      return None
    first_s = events[current_event_ids[0]].s
    last_s = events[current_event_ids[-1]].s
    matching = [event for event in events
                if first_s <= event.s <= last_s
                and event.kind == spec.anchor_event]
    if not matching:
      return None
    occurrence = spec.occurrence
    if occurrence == "auto":
      occurrence = "last"
    anchor = matching[0] if occurrence == "first" else matching[-1]
    body_end_s = float(np.clip(
        anchor.s + spec.offset_stroke_widths * sw,
        float(grid[0]), float(grid[join_end_index])))
    return int(np.argmin(np.abs(grid - body_end_s)))

  def branch_reach_exit() -> Optional[int]:
    """End the body where its anchor branch stops overhanging the spine.

    A medial Kaf's arm folds back across many stroke widths of spine. Its ink
    is attributed to the letter either way -- branch pixels carry their
    anchor's coordinate -- but the body *span* is not, so without this the
    letter measures two stroke widths and its connector absorbs the rest,
    which is what reaches the corpus as ``spine_extent`` and ``kashida``.

    An arm that still overhangs where the next body starts ends at the join:
    see the clamp below. Only an unusable anchor returns ``None``, because a
    missing exit index discards the whole alignment.
    """
    event_id = _boundary_anchor_event(
        events, current_event_ids, spec, "exit")
    if event_id is None or events[event_id].reach is None:
      return None
    xs = np.asarray(profile["x"], dtype=float)
    start = int(np.clip(np.searchsorted(
        grid, events[event_id].s, side="left"), 0, len(grid) - 1))
    limit = int(np.clip(join_end_index, 0, len(grid) - 1))
    if limit <= start:
      return None
    ahead = np.arange(start + 1, limit + 1)
    reach_min, reach_max = events[event_id].reach
    # Follow whichever way the spine is actually travelling here rather than
    # assuming right-to-left, so a reversed spine cannot invert the test.
    passed = (xs[ahead] <= reach_min if xs[limit] <= xs[start]
              else xs[ahead] >= reach_max)
    if not passed.any():
      # The arm is still overhanging the spine where the next body begins.
      # That is a shape, not a contradiction -- page 105 l01_w03's Hah
      # overhangs 0.2 stroke widths past the Qaf's entry -- so the body takes
      # everything up to the join rather than the alignment being discarded.
      # ``offset_stroke_widths`` is deliberately not applied: it shifts the
      # point the rule found, and here the rule found none. A non-negative
      # offset would clip back to this same index anyway.
      return limit
    index = int(ahead[int(np.argmax(passed))])
    offset = int(round(spec.offset_stroke_widths * sw / profile["ds"]))
    return int(np.clip(index + offset, start, limit))

  evaluators = {
      "event-offset": event_exit,
      "horizontal-stabilization": horizontal_exit,
      "width-exit": width_exit,
      "width-or-event-exit": width_or_event_exit,
      "observed-event-offset": observed_event_exit,
      "branch-reach": branch_reach_exit,
  }
  if spec.method != "auto":
    result = evaluators[spec.method]()
    return result

  # Default exit behavior is retained for templates that do not need to say
  # anything special.  Explicit template rules bypass this cascade.
  if event.kind == "valley":
    default_spec = BodyBoundarySpec(
        method="event-offset", anchor_event="valley",
        occurrence="last", offset_stroke_widths=1.0)
    return _template_body_exit_index(
        profile, events, current_event_ids, default_spec,
        join_end_index, stroke_width)
  for evaluator in (horizontal_exit, width_exit):
    result = evaluator()
    if result is not None:
      return result
  # Keep pixels attached exactly at the last body event with that body; the
  # following dense sample is the earliest observable connector position.
  return min(join_end_index, event_index + 1)


def _materialize_template_guided_path(
    profile: dict, events: Sequence[SkeletonEvent], letters: Sequence[str],
    baseline_y: float, stroke_width: float,
    group_specs: Sequence[Tuple[int, int, ElasticBodyTemplate]],
    matches: Sequence[Tuple[ElasticBodyTemplate,
                            ElasticEventAssignment, float]],
    ) -> Tuple[Optional[Tuple[float, float, List[SegmentGroup],
                              List[JoinSpan], List[ShapeMatch]]], str]:
  """Turn one event-ordered physical-body path into bodies and joins."""
  boundaries: List[Tuple[Candidate, Candidate, str, float]] = []
  for boundary_i, (current, following) in enumerate(
      zip(matches, matches[1:])):
    boundary = _template_body_boundary_index(
        profile, events, current[1].events[-1], following[1].events,
        following[0].entry, stroke_width)
    if boundary is None:
      return None, "entry-boundary-unavailable"
    # ``enforce`` on an entry is the mirror of ``enforce`` on an exit: the
    # offset is a requirement, not a preference. Without this the offset is
    # quietly clipped to the preceding body's last owned event -- page 9
    # l05_w06's senna asks to start one stroke width before its peak, is
    # clipped to 0.03, and reports a 1q body as though the rule had been
    # honoured. There is no analogue of the exit's "may move into the join":
    # the clamp floor already *is* the preceding body's last event, so an
    # offset that cannot be met is one that would overlap it.
    entry_spec = following[0].entry
    if (entry_spec.enforce
        and entry_spec.method in {"event-offset", "observed-event-offset"}):
      anchor = _boundary_anchor_event(
          events, following[1].events, entry_spec, "entry")
      if anchor is not None and (
          events[anchor].s + entry_spec.offset_stroke_widths * stroke_width
          < events[current[1].events[-1]].s - 1e-6):
        return None, "enforced-entry-overlaps-preceding-body"
    end_index, kind, confidence = boundary
    following_position = group_specs[boundary_i + 1][0]
    exit_owns_join = bool(
        current[0].exit_owns_join_next_letters
        and any(letters[following_position] in allowed
                for allowed in current[0].exit_owns_join_next_letters))
    hard_event_exit = (current[0].exit.enforce
                       and current[0].exit.method in {
                           "event-offset", "observed-event-offset"}
                       and not exit_owns_join)
    exit_limit_index = (len(profile["grid"]) - 1
                        if hard_event_exit else end_index)
    start_index = _template_body_exit_index(
        profile, events, current[1].events, current[0].exit,
        exit_limit_index, stroke_width)
    if start_index is None:
      return None, "exit-boundary-unavailable"
    if hard_event_exit and start_index > end_index:
      following_first_index = int(np.searchsorted(
          profile["grid"], events[following[1].events[0]].s, side="left"))
      if start_index >= following_first_index:
        return None, "enforced-exit-overlaps-following-body"
      end_index = start_index
      kind, confidence = "template-enforced-exit", 1.0
    if exit_owns_join:
      start_index = int(np.argmin(np.abs(
          profile["grid"] - events[following[1].events[0]].s)))
      end_index = start_index
      kind, confidence = "template-exit-owns-join", 1.0
    start = candidate_from_profile_index(
        profile, baseline_y, stroke_width, start_index,
        "template-body-exit")
    end = candidate_from_profile_index(
        profile, baseline_y, stroke_width, end_index, kind)
    boundaries.append((start, end, kind, confidence))
  if any(right[0].s <= left[1].s
         for left, right in zip(boundaries, boundaries[1:])):
    return None, "join-intervals-overlap"

  total_s = float(profile.get("total_s", profile["grid"][-1]))
  body_spans = [
      ((0.0 if i == 0 else boundaries[i - 1][1].s),
       (total_s if i == len(group_specs) - 1 else boundaries[i][0].s))
      for i in range(len(group_specs))]
  if any(b_s <= a_s + 1e-6 for a_s, b_s in body_spans):
    return None, "empty-body-span"
  path_templates = [match[0] for match in matches]
  if not branch_partition_constraints_match(
      path_templates, body_spans, events):
    return None, "branch-partition-constraint"
  loop_evidence = profile.get("loop_evidence")
  if (loop_evidence is not None
      and not loop_partition_constraints_match(
          path_templates, body_spans, loop_evidence, events,
          [match[1].events for match in matches])):
    return None, "loop-partition-constraint"

  groups = [
      SegmentGroup(letter_start, letter_end, a_s, b_s, template.name)
      for (letter_start, letter_end, template), (a_s, b_s)
      in zip(group_specs, body_spans)]
  joins = [
      JoinSpan(group_specs[i + 1][0], kind, start, end,
               score=confidence, confidence=confidence)
      for i, (start, end, kind, confidence) in enumerate(boundaries)]
  # A composite owns one physical body but still spans multiple transcription
  # letters. Preserve those logical boundaries as fused, cut-free joins.
  for group_i, (letter_start, letter_end, template) in enumerate(group_specs):
    if letter_end - letter_start <= 1:
      continue
    score = matches[group_i][2]
    for boundary in range(letter_start + 1, letter_end):
      joins.append(JoinSpan(
          boundary, "fused", score=score, confidence=score,
          template=template.name))
  joins.sort(key=lambda join: join.letter_boundary)
  shape_matches = [
      ShapeMatch(template.name, letter_start, letter_end, a_s, b_s, score,
                 assignment.events, assignment.before_events,
                 assignment.after_events)
      for (letter_start, letter_end, _), (a_s, b_s),
          (template, assignment, score)
      in zip(group_specs, body_spans, matches)]
  evidence = [score for _, _, score in matches]
  evidence.extend(confidence for _, _, _, confidence in boundaries)
  return ((float(np.mean(evidence)), float(min(evidence)),
           groups, joins, shape_matches), "")


def template_guided_single_alignment(
    profile: dict, events: Sequence[SkeletonEvent], letters: Sequence[str],
    forms: Sequence[str], baseline_y: float, stroke_width: float,
    templates: Sequence[ElasticBodyTemplate],
    trace: Optional[dict] = None
    ) -> Optional[Tuple[float, float, List[SegmentGroup],
                        List[JoinSpan], List[ShapeMatch]]]:
  options = _ordered_single_body_match_options(
      events, letters, forms, templates, profile.get("loop_evidence"),
      stroke_width, trace)
  template_order = {id(template): i
                    for i, template in enumerate(templates)}
  options.sort(key=lambda path: _complete_ordered_path_selection_key(
      path, template_order))
  if trace is not None:
    trace["evaluationOrder"] = [
        [item[0].name for item in path]
        for path in options[:MAX_ALIGNMENT_TRACE_EXAMPLES]]
  rejection_counts: Optional[Dict[str, int]] = {} if trace is not None else None
  for matches in options:
    # All options apply at the same transcription positions, but can select
    # different allograph templates at those positions.
    group_specs = [(i, i + 1, match[0])
                   for i, match in enumerate(matches)]
    alignment, reason = _materialize_template_guided_path(
        profile, events, letters, baseline_y, stroke_width,
        group_specs, matches)
    if alignment is not None:
      if trace is not None:
        trace["result"] = "matched"
        trace["postTopologyRejections"] = rejection_counts or {}
      return alignment
    if rejection_counts is not None:
      rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
  if trace is not None and options:
    trace["result"] = "boundary-failed"
    trace["postTopologyRejections"] = rejection_counts or {}
  return None


def template_guided_composite_alignments(
    profile: dict, events: Sequence[SkeletonEvent], letters: Sequence[str],
    forms: Sequence[str], baseline_y: float, stroke_width: float,
    templates: Sequence[ElasticBodyTemplate],
    ) -> List[Tuple[float, float, List[SegmentGroup],
                    List[JoinSpan], List[ShapeMatch]]]:
  """Align every partition containing a multi-letter physical body."""
  template_order = {id(template): i
                    for i, template in enumerate(templates)}
  alignments = []
  for raw_partition in _body_partitions(letters, forms, templates):
    if not any(end - start > 1 for start, end, _ in raw_partition):
      continue
    group_specs = [(start, end, template)
                   for start, end, template in raw_partition
                   if template is not None]
    if (len(group_specs) != len(raw_partition)
        or any(not spec[2].ordered_match for spec in group_specs)):
      continue
    options = _ordered_fixed_template_match_options(
        events, [spec[2] for spec in group_specs],
        profile.get("loop_evidence"), stroke_width)
    options.sort(key=lambda path: _complete_ordered_path_selection_key(
        path, template_order))
    for matches in options:
      alignment, _ = _materialize_template_guided_path(
          profile, events, letters, baseline_y, stroke_width,
          group_specs, matches)
      if alignment is not None:
        alignments.append(alignment)
        break
  return alignments


BRANCH_EVENT_KINDS = frozenset(
    {"branch", "high-branch", "double-high-branch", "zigzag-branch"})


def _join_holds_strong_branch(joins: Sequence[JoinSpan],
                              events: Sequence[SkeletonEvent]) -> bool:
  """Whether any measured join in this alignment contains a real branch.

  A fused join has no start/end anchors and no ink of its own, so it is not
  examined; only spans the alignment actually calls connector are.
  """
  for join in joins:
    if join.start is None or join.end is None:
      continue
    low, high = join.start.s, join.end.s
    for event in events:
      if (event.kind in BRANCH_EVENT_KINDS
          and low - 1e-6 <= event.s <= high + 1e-6
          and event.strength > MAX_JOIN_BRANCH_W):
        return True
  return False


def force_align_bodies_and_joins(
    profile: dict, events: Sequence[SkeletonEvent], letters: Sequence[str],
    forms: Sequence[str], baseline_y: float, stroke_width: float,
    cut_policy: str = "next-start",
    elastic_templates: Optional[Sequence[ElasticBodyTemplate]] = None,
    trace: Optional[dict] = None
    ) -> BodyJoinAlignment:
  """Jointly align body states and elastic joins, then materialize cuts.

  This solver never calls ``cut_candidates``.  It chooses complete join
  intervals first; a cut is merely a requested view of each chosen interval.
  """
  if cut_policy not in {"current-end", "midpoint", "next-start", "none"}:
    raise ValueError(f"unknown cut policy: {cut_policy}")
  if elastic_templates is None:
    _, elastic_templates = load_template_definitions()
  total_s = float(profile.get("total_s", profile["grid"][-1]))
  if trace is not None:
    trace.update({
        "letters": list(letters), "forms": list(forms),
        "eventCount": len(events), "strokeWidth": round(stroke_width, 2)})
  alignments: List[Tuple[float, float, List[SegmentGroup],
                         List[JoinSpan], List[ShapeMatch]]] = []

  guided = template_guided_single_alignment(
      profile, events, letters, forms, baseline_y, stroke_width,
      elastic_templates, trace)
  if guided is not None:
    alignments.append(guided)
  composite_guided = template_guided_composite_alignments(
      profile, events, letters, forms, baseline_y, stroke_width,
      elastic_templates)
  alignments.extend(composite_guided)

  # Rising joins are now only a fallback when no complete template-guided path
  # exists. In particular, a matched composite partition neither evaluates nor
  # depends on this older hypothesis generator.
  fallback_partitions = [
      partition
      for partition in _body_partitions(letters, forms, elastic_templates)
      if all(letter_end - letter_start == 1
             for letter_start, letter_end, _ in partition)]
  hypotheses = ([] if guided is not None or composite_guided else
                rising_join_hypotheses(profile, events, stroke_width))
  if trace is not None:
    trace["joinHypotheses"] = len(hypotheses)
  for partition in fallback_partitions:
    # A complete all-single template path has already evaluated every
    # template-specific entry and exit rule. Reintroducing it through generic
    # rising-join hypotheses would ignore those rules and can win merely
    # because its generic score is larger. The remaining path is only a
    # fallback when ordered all-single matching failed.
    if guided is not None or composite_guided:
      continue
    needed = len(partition) - 1
    if needed > len(hypotheses):
      continue
    for chosen in combinations(hypotheses, needed):
      if any(left.end_index >= right.start_index
             for left, right in zip(chosen, chosen[1:])):
        continue
      group_bounds: List[Tuple[float, float]] = []
      for group_i in range(len(partition)):
        a_s = (0.0 if group_i == 0
               else float(profile["grid"][chosen[group_i - 1].end_index]))
        b_s = (total_s if group_i == len(partition) - 1
               else float(profile["grid"][chosen[group_i].start_index]))
        group_bounds.append((a_s, b_s))

      # In the all-single fallback an enforced following-body entry remains
      # authoritative. First obtain topology under the generic bounds, then
      # move only the join end and following body start to the template entry.
      boundary_end_indices = [item.end_index for item in chosen]
      if any(spec[2] is not None and spec[2].entry.enforce
             for spec in partition[1:]):
        preliminary_ids: List[Tuple[int, ...]] = []
        valid = True
        for spec, (a_s, b_s) in zip(partition, group_bounds):
          template = spec[2]
          if template is None:
            valid = False
            break
          body_score, event_ids = match_elastic_body_template(
              events, a_s, b_s, template, stroke_width=stroke_width)
          if body_score <= 0.0:
            valid = False
            break
          preliminary_ids.append(event_ids)
        if not valid:
          continue
        for boundary_i in range(len(partition) - 1):
          following_template = partition[boundary_i + 1][2]
          if (following_template is None
              or not following_template.entry.enforce):
            continue
          boundary = _template_body_boundary_index(
              profile, events, preliminary_ids[boundary_i][-1],
              preliminary_ids[boundary_i + 1], following_template.entry,
              stroke_width)
          if boundary is None:
            valid = False
            break
          entry_index = boundary[0]
          if entry_index <= chosen[boundary_i].start_index:
            valid = False
            break
          entry_s = float(profile["grid"][entry_index])
          next_a, next_b = group_bounds[boundary_i + 1]
          if entry_s >= next_b:
            valid = False
            break
          group_bounds[boundary_i + 1] = (entry_s, next_b)
          boundary_end_indices[boundary_i] = entry_index
        if not valid:
          continue
      if any((b_s - a_s) < MIN_SPINE_RUN_W * stroke_width
             for a_s, b_s in group_bounds):
        continue
      groups: List[SegmentGroup] = []
      matches: List[ShapeMatch] = []
      body_scores: List[float] = []
      valid = True
      for spec, (a_s, b_s) in zip(partition, group_bounds):
        letter_start, letter_end, template = spec
        groups.append(SegmentGroup(
            letter_start, letter_end, a_s, b_s,
            template.name if template else None))
        if template is not None:
          body_score, assignment = _match_elastic_body_assignment(
              events, a_s, b_s, template, stroke_width=stroke_width)
          if body_score <= 0.0 or assignment is None:
            valid = False
            break
          body_scores.append(body_score)
          matches.append(ShapeMatch(
              template.name, letter_start, letter_end, a_s, b_s,
              body_score, assignment.events, assignment.before_events,
              assignment.after_events))
      if not valid:
        continue
      loop_evidence = profile.get("loop_evidence")
      partition_templates = [spec[2] for spec in partition]
      if (loop_evidence is not None
          and (any(template is None for template in partition_templates)
               or not loop_partition_constraints_match(
                   [template for template in partition_templates
                    if template is not None],
                   group_bounds, loop_evidence, events,
                   [match.events for match in matches]))):
        continue

      joins: List[JoinSpan] = []
      for boundary_i, hypothesis in enumerate(chosen):
        boundary = partition[boundary_i][1]
        start = candidate_from_profile_index(
            profile, baseline_y, stroke_width, hypothesis.start_index,
            "join-start")
        end = candidate_from_profile_index(
            profile, baseline_y, stroke_width,
            boundary_end_indices[boundary_i],
            "join-end")
        joins.append(JoinSpan(
            letter_boundary=boundary, kind=hypothesis.kind,
            start=start, end=end, score=hypothesis.score,
            confidence=hypothesis.score))
      # A fused template is a real logical transition but has no physical
      # connector and therefore no start/end cut anchors.
      matches_by_start = {match.letter_start: match for match in matches}
      for group in groups:
        match = matches_by_start.get(group.letter_start)
        if group.letter_end - group.letter_start > 1:
          joins.append(JoinSpan(
              letter_boundary=group.letter_start + 1, kind="fused",
              score=match.score if match else 0.0,
              confidence=match.score if match else 0.0,
              template=group.template))
      joins.sort(key=lambda join: join.letter_boundary)
      evidence = [item.score for item in chosen] + body_scores
      score = float(np.mean(evidence)) if evidence else 1.0
      confidence = float(min(evidence)) if evidence else 1.0
      alignments.append((score, confidence, groups, joins, matches))

  if not alignments:
    if trace is not None:
      trace["finalResult"] = "no-match"
      trace["compositeGuidedAlignments"] = len(composite_guided)
      trace["genericPartitions"] = len(_body_partitions(
          letters, forms, elastic_templates))
    return BodyJoinAlignment(
        [], [], [], 0.0, 0.0,
        reason=(
            "no body/join alignment satisfies the topology and body constraints"))
  # A connector that contains a branch is a body the alignment did not claim.
  # Dropped only when something clean survives, so this never turns a
  # segmented PAW into a failed one.
  clean = [alignment for alignment in alignments
           if not _join_holds_strong_branch(alignment[3], events)]
  if clean and len(clean) != len(alignments):
    if trace is not None:
      trace["branchInJoinRejected"] = len(alignments) - len(clean)
    alignments = clean
  # Alternative topology templates can explain the same physical alignment.
  # They are evidence variants, not a materially different segmentation, so
  # collapse them before computing the winning margin.
  distinct: Dict[Tuple[object, ...],
                 Tuple[float, float, List[SegmentGroup],
                       List[JoinSpan], List[ShapeMatch]]] = {}
  for alignment in alignments:
    signature: Tuple[object, ...] = (
        tuple((group.letter_start, group.letter_end,
               round(group.a_s, 6), round(group.b_s, 6))
              for group in alignment[2]),
        tuple((join.letter_boundary,
               round(join.start.s, 6) if join.start else None,
               round(join.end.s, 6) if join.end else None)
              for join in alignment[3] if join.kind != "fused"))
    previous = distinct.get(signature)
    if previous is None or alignment[:2] > previous[:2]:
      distinct[signature] = alignment
  alignments = list(distinct.values())
  # A recognized composite is direct evidence that several transcription
  # letters share one physical body. Once that topology matches, do not let a
  # weaker all-single explanation invent an internal join merely because its
  # generic boundary scores average slightly higher.
  alignments.sort(key=lambda item: (
      sum(join.kind == "fused" for join in item[3]), item[0], item[1]),
      reverse=True)
  score, confidence, groups, joins, shape_matches = alignments[0]
  fused_count = sum(join.kind == "fused" for join in joins)
  second = max(
      (item[0] for item in alignments[1:]
       if sum(join.kind == "fused" for join in item[3]) == fused_count),
      default=0.0)
  cuts: List[Candidate] = []
  for join in joins:
    if join.kind == "fused" or cut_policy == "none":
      continue
    assert join.start is not None and join.end is not None
    if cut_policy == "current-end":
      cuts.append(join.start)
    elif cut_policy == "next-start":
      cuts.append(join.end)
    else:
      midpoint = 0.5 * (join.start.s + join.end.s)
      index = int(np.argmin(np.abs(profile["grid"] - midpoint)))
      cuts.append(candidate_from_profile_index(
          profile, baseline_y, stroke_width, index, "join-midpoint"))
  margin = float(score - second)
  if trace is not None:
    trace["finalResult"] = "matched"
    trace["compositeGuidedAlignments"] = len(composite_guided)
  return BodyJoinAlignment(
      groups, joins, cuts, score, margin, shape_matches=shape_matches)


def tooth_necks(profile: dict, stroke_width: float) -> np.ndarray:
  """Sorted spine positions of narrow tooth-junction cross-sections.

  Independent of `cut_candidates`'s "cuttable region" logic on purpose: that
  logic merges a run of shallow dips (a seen's three teeth) into one region
  and reports only its single deepest point, which is exactly what loses the
  internal teeth. This scans the whole cut-cost profile directly with a much
  lower prominence bar.
  """
  # Tooth counting is an independent, already calibrated signal. Keep its
  # unbounded full-ray profile here; ownership localisation is for candidate cut
  # cost and should not silently change the script constraints at the same
  # time.
  cut_len, ds = profile.get("global_cut_len", profile["cut_len"]), profile["ds"]
  if cut_len.size < 3:
    return np.empty(0)
  prom = TOOTH_PROMINENCE_FRAC * stroke_width
  dist = max(1, int(round(TOOTH_MIN_SEP_W * stroke_width / ds)))
  idx, _ = find_peaks(-cut_len, prominence=prom, distance=dist)
  return profile["grid"][idx]


def count_teeth_between(valleys: np.ndarray, a_s: float, b_s: float,
                        margin: float, exclude_s: Optional[np.ndarray] = None,
                        exclude_tol: float = 0.0) -> int:
  """Tooth valleys strictly inside (a_s, b_s), excluding a margin at each end.

  The margin excludes the segment's own boundary cuts, which are themselves
  valleys (deeper ones) and must not count as one of the letter's own teeth.

  `exclude_s`, when given, drops any valley within `exclude_tol` of one of
  those positions first -- for letters whose tooth count must be exactly
  zero, a valley that coincides with an independently strong cut-candidate
  boundary is far more likely to be a misattributed letter boundary than a
  genuine tooth (see `tooth_prior`).
  """
  lo, hi = a_s + margin, b_s - margin
  if hi <= lo or valleys.size == 0:
    return 0
  i0 = int(np.searchsorted(valleys, lo, side="left"))
  i1 = int(np.searchsorted(valleys, hi, side="right"))
  sel = valleys[i0:i1]
  if exclude_s is not None and exclude_s.size and sel.size:
    near = (np.abs(sel[:, None] - exclude_s[None, :]) <= exclude_tol).any(axis=1)
    sel = sel[~near]
  return int(sel.size)


def observed_teeth(valleys: np.ndarray, a_s: float, b_s: float, margin: float,
                   letter: str, boundary_s: Optional[np.ndarray] = None,
                   boundary_tol: float = 0.0,
                   form: Optional[str] = None,
                   rises: Optional[np.ndarray] = None) -> int:
  """Teeth inside a segment, counted the one way the solver counts them.

  A letter with zero expected teeth can have no legitimate internal valley,
  so a valley there that coincides with an independently strong
  `cut_candidates` "min" pick is far more likely to be a neighbouring
  letter's real boundary than this letter's own structure, and is dropped.
  On "مَن" the true meem|noon neck was exactly such a point: counting it as
  one of the noon's teeth forced the cut two spurious dips further down the
  noon's own bowl. Multi-tooth letters (seen/sheen) never take that path,
  since a valley they contain may well be one of their genuine teeth.

  Every caller -- the tooth penalty, the learned shape descriptor, and the
  reported per-letter count -- must agree, or the JSON and the training
  features describe a segmentation the solver never considered.

  Initial seen/sheen are the one form-specific exception to valley counting:
  their first tooth is at the spine endpoint and their remaining two teeth
  appear as two reliable rises, while a broad trough between them may contain
  more than one shallow minimum.
  """
  lo, hi = a_s + margin, b_s - margin
  if (f"{letter}|{form}" in TOOTH_RISE_FORMS and rises is not None):
    if hi <= lo or rises.size == 0:
      return 0
    return int(((rises > lo) & (rises < hi)).sum())
  if TOOTH_COUNT.get(letter) == 0:
    return count_teeth_between(valleys, a_s, b_s, margin,
                               exclude_s=boundary_s, exclude_tol=boundary_tol)
  return count_teeth_between(valleys, a_s, b_s, margin)


def max_rise_cluster(rises: np.ndarray, max_gap: float) -> int:
  """Largest run of consecutive rises separated by at most ``max_gap``."""
  if rises.size == 0:
    return 0
  best = run = 1
  for gap in np.diff(rises):
    run = run + 1 if gap <= max_gap else 1
    best = max(best, run)
  return best


def rise_clusters(rises: np.ndarray, max_gap: float) -> List[np.ndarray]:
  """Consecutive rise runs, split where the gap exceeds ``max_gap``."""
  if rises.size == 0:
    return []
  start = 0
  out: List[np.ndarray] = []
  for end in range(1, len(rises) + 1):
    if end == len(rises) or rises[end] - rises[end - 1] > max_gap:
      out.append(rises[start:end])
      start = end
  return out


def profile_prefix_radius(profile: dict) -> np.ndarray:
  """Cumulative sum of the medial radius along the spine, for O(1) mean-radius
  queries over any [a_s, b_s) range via prefix-sum subtraction."""
  return np.concatenate([[0.0], np.cumsum(profile["r"])])


def mean_radius_between(profile: dict, prefix_r: np.ndarray,
                        a_s: float, b_s: float) -> float:
  n = len(profile["r"])
  ds = profile["ds"]
  i0 = int(np.clip(round(a_s / ds), 0, n))
  i1 = int(np.clip(round(b_s / ds), 0, n))
  if i1 <= i0:
    i1 = min(i0 + 1, n)
    if i1 <= i0:
      return float(profile["r"][-1]) if n > 0 else 0.0
  return float((prefix_r[i1] - prefix_r[i0]) / (i1 - i0))


# ---------------------------------------------------------------------------
# H. semi-Markov DP: physical cuts plus optional composite ligature arcs
# ---------------------------------------------------------------------------

def choose_cuts(candidates: List[Candidate], n_letters: int, total_len: float,
                stroke_width: float, ctx: Optional[dict] = None
                ) -> Tuple[List[Candidate], List[SegmentGroup], str]:
  chosen, _, reason, groups = _solve_cuts(
      candidates, n_letters, total_len, stroke_width, ctx)
  return chosen, groups, reason


def cut_margin(candidates: List[Candidate], chosen: List[Candidate],
               n_letters: int, total_len: float, stroke_width: float,
               ctx: Optional[dict] = None,
               groups: Optional[Sequence[SegmentGroup]] = None) -> float:
  """How much the chosen solution beats the best materially different one.

  For each chosen cut in turn, re-solve with all candidates within
  MARGIN_EXCLUDE_W of it removed; the margin is the smallest resulting drop.
  Near zero means the geometry did not actually decide where the cut goes.
  """
  _, best_value, _, _ = _solve_cuts(candidates, n_letters, total_len,
                                    stroke_width, ctx)
  if best_value is None:
    return 0.0
  worst = float("inf")
  if groups and any(group.letter_end - group.letter_start > 1
                    for group in groups):
    physical_ctx = dict(ctx or {})
    physical_ctx["logical_ligatures"] = False
    _, alt_value, _, _ = _solve_cuts(
        candidates, n_letters, total_len, stroke_width, physical_ctx)
    drop = best_value if alt_value is None else best_value - alt_value
    worst = min(worst, drop)
  radius = MARGIN_EXCLUDE_W * stroke_width
  for cut in chosen:
    pool = [c for c in candidates if abs(c.s - cut.s) > radius]
    _, alt_value, _, _ = _solve_cuts(
        pool, n_letters, total_len, stroke_width, ctx)
    drop = best_value if alt_value is None else best_value - alt_value
    worst = min(worst, drop)
  return float(max(0.0, worst if np.isfinite(worst) else 0.0))


def _class_stats(priors: dict, letter: str, form: Optional[str]) -> Optional[dict]:
  """(letter, form) stats, falling back to the pooled (letter, any form)."""
  for key in (f"{letter}|{form}", f"{letter}|*"):
    cls = priors.get(key)
    if cls is not None and cls.get("n", 0) >= MIN_CLASS_SAMPLES:
      return cls
  return None


def _solve_cuts(candidates: List[Candidate], n_letters: int, total_len: float,
                stroke_width: float, ctx: Optional[dict] = None):
  """DP core. Returns (cuts, objective value, failure reason).

  `ctx`, when given, carries everything the text-derived penalty terms need:
  `letters`/`forms` (one entry per letter, right to left, same order as the
  DP's segments), `valleys`/`tooth_margin` (script-knowledge tooth check,
  Fix 1a), and `profile`/`prefix_r`/`priors` (learned shape check, Fix 1b).
  Any piece may be absent -- each term degrades to 0 on its own.
  """
  n_cuts = n_letters - 1
  if n_cuts <= 0:
    return [], 0.0, "", [SegmentGroup(0, n_letters, 0.0, total_len)]
  logical_ligatures = bool(ctx and ctx.get("logical_ligatures"))
  if len(candidates) < n_cuts and not logical_ligatures:
    return ([], None,
            f"only {len(candidates)} cut candidates for {n_cuts} cuts", [])

  cand = sorted(candidates, key=lambda c: c.s)
  m = len(cand)
  target = total_len / n_letters

  def too_short(a_s: float, b_s: float) -> bool:
    # Two floors, because either alone admits a bad segment. Total extent
    # counts the off-spine body, so an upright is not judged degenerate for
    # leaving the spine early; the spine run on its own stops a segment
    # from clearing that bar on a branch it merely contains.
    if (b_s - a_s) < MIN_SPINE_RUN_W * stroke_width:
      return True
    return segment_extent(a_s, b_s, stroke_width, ctx) < MIN_SEGMENT_W

  def width_prior(length: float, consume: int = 1) -> float:
    expected = target * consume
    deviation = abs(length - expected) / max(expected, 1e-6)
    return -max(0.0, deviation - WIDTH_PRIOR_DEADZONE) ** 2

  def tooth_prior(a_s: float, b_s: float, letter: str,
                  form: Optional[str]) -> float:
    if ctx is None or "tooth_necks" not in ctx:
      return 0.0
    expected = TOOTH_COUNT.get(letter)
    if expected is None:
      return 0.0
    observed = observed_teeth(ctx["tooth_necks"], a_s, b_s,
                              ctx["tooth_margin"],
                              letter, ctx.get("boundary_s"),
                              ctx.get("boundary_tol", 0.0), form,
                              ctx.get("rises"))
    # Valley detection can register one incidental shallow dip in a hook,
    # entry stroke, or broad trough. Keep the established one-count tolerance
    # for valley-based forms. Initial seen/sheen use two unambiguous rises
    # instead, so their count is exact: allowing one missing rise is precisely
    # what permits a cut before the third tooth.
    rise_counted = f"{letter}|{form}" in TOOTH_RISE_FORMS
    slack = 0 if rise_counted else TOOTH_MISMATCH_SLACK
    mismatch = max(0, abs(observed - expected) - slack)
    return -W_TOOTH_MISMATCH * mismatch

  def shape_prior(a_s: float, b_s: float, letter: str, form: Optional[str]) -> float:
    if ctx is None or ctx.get("priors") is None or "profile" not in ctx:
      return 0.0
    cls = _class_stats(ctx["priors"], letter, form)
    if cls is None:
      return 0.0
    extent = (b_s - a_s) / stroke_width
    radius = mean_radius_between(ctx["profile"], ctx["prefix_r"], a_s, b_s) / stroke_width
    teeth = (observed_teeth(ctx["tooth_necks"], a_s, b_s,
                            ctx["tooth_margin"],
                            letter, ctx.get("boundary_s"),
                            ctx.get("boundary_tol", 0.0), form,
                            ctx.get("rises"))
             if "tooth_necks" in ctx else 0.0)
    values = {"extent": extent, "radius": radius, "teeth": float(teeth)}
    weights = {"extent": W_SHAPE_EXTENT, "radius": W_SHAPE_RADIUS, "teeth": W_SHAPE_TEETH}
    penalty = 0.0
    for feat in SHAPE_FEATURES:
      stat = cls.get(feat)
      if stat is None:
        continue
      mad = max(stat["mad"], SHAPE_MAD_FLOOR[feat])
      penalty += weights[feat] * abs(values[feat] - stat["median"]) / mad
    return -W_SHAPE_PRIOR * penalty

  def branch_prior(a_s: float, b_s: float, letter: str,
                   form: Optional[str]) -> float:
    """Score a letter's segment against the branch count its body implies.

    Two-sided on purpose: the lower bound catches a letter reduced to a bare
    connector, the upper bound stops a letter from swallowing a neighbour's
    body to satisfy that lower bound.
    """
    if ctx is None or "branches" not in ctx:
      return 0.0
    bounds = _branch_bounds(letter, form)
    if bounds is None:
      return 0.0
    lo_n, hi_n = bounds
    br = ctx["branches"]
    observed = int(((br > a_s) & (br < b_s)).sum()) if br.size else 0
    penalty = 0.0
    if lo_n is not None and observed < lo_n:
      penalty += W_BRANCH_MISSING * (lo_n - observed)
    if hi_n is not None:
      excess = observed - hi_n - BRANCH_EXTRA_SLACK
      if excess > 0:
        penalty += W_BRANCH_EXTRA * excess
    return -penalty

  def rise_prior(a_s: float, b_s: float, letter: str,
                 form: Optional[str]) -> float:
    """A constrained letter must contain its own tooth or body lift.

    Guards the case the branch bounds cannot: on "تُنذِرْهُمْ" the noon's
    lower bound was already met, by the branch at s=377, while its actual
    tooth at s=216 sat inside the teh -- a letter holding its neighbour's
    tooth violates nothing when only counts are checked. Inert where the
    PAW shows no rise at all, so it cannot fire on a flat connector. Selected
    forms also require the first lift near segment entry to enforce outgoing-
    connector ownership.
    """
    if ctx is None or "rises" not in ctx:
      return 0.0
    key = f"{letter}|{form}"
    bounds = _rise_bounds(letter, form)
    cluster_expected = RISE_CLUSTER_EXPECT.get(key)
    boundary_constrained = key in RISE_BOUNDARY_FORMS
    if bounds is None and cluster_expected is None and not boundary_constrained:
      return 0.0
    rises = ctx["rises"]
    if rises.size == 0:
      return 0.0
    local_rises = rises[(rises > a_s) & (rises < b_s)]
    penalty = 0.0
    if bounds is not None:
      lo_n, hi_n = bounds
      n = int(local_rises.size)
      if lo_n is not None and n < lo_n:
        penalty += W_RISE_MISSING * (lo_n - n)
      # Symmetric upper bound: one of these letters is one tooth, so a second
      # rise inside it is a neighbour's tooth this segment has taken over.
      # Every correctly segmented medial tooth letter on page 3 holds exactly
      # one, 13 of 13, so the bound costs nothing where the segmentation is
      # already right. The same bound on branch counts would be inert: it
      # never fires, because the tooth it needs to see is shorter than the
      # spurs beside it and BRANCH_MIN_LEN_W has already dropped it.
      if hi_n is not None and n > hi_n:
        penalty += W_RISE_EXTRA * (n - hi_n)
    if cluster_expected is not None:
      max_gap = RISE_CLUSTER_MAX_GAP_W * stroke_width
      # Do not force a tooth run the detector cannot see anywhere in this PAW.
      # Where one exists, the claimed segment must own exactly that run rather
      # than omit tooth one or absorb a neighbouring body's lift.
      if max_rise_cluster(rises, max_gap) >= cluster_expected:
        observed = max_rise_cluster(local_rises, max_gap)
        if observed < cluster_expected:
          penalty += W_RISE_CLUSTER_MISSING * (cluster_expected - observed)
        elif observed > cluster_expected:
          penalty += W_RISE_EXTRA * (observed - cluster_expected)
        # Owning rise three is not enough if the cut still crosses its tooth.
        # Close seen only after the first valley following the complete local
        # run. Bound the search by the next rise so a later letter's valley
        # cannot be mistaken for the tooth exit.
        complete = [run for run in rise_clusters(rises, max_gap)
                    if (run.size == cluster_expected and run[0] > a_s
                        and run[-1] < b_s)]
        if complete and "tooth_necks" in ctx:
          run = complete[0]
          following = rises[rises > run[-1]]
          limit = float(following[0]) if following.size else float("inf")
          closing = ctx["tooth_necks"][(ctx["tooth_necks"] > run[-1]) &
                                        (ctx["tooth_necks"] < limit)]
          if closing.size and b_s < closing[0]:
            penalty += W_RISE_CLUSTER_EARLY_EXIT
    # Each letter carries its outgoing connector, so the following final kaf
    # should begin shortly before its own body lift. Merely requiring the lift
    # somewhere in kaf lets the boundary collapse immediately after yeh's
    # tooth, assigning the entire intervening connector to kaf.
    if key in RISE_ENTRY_FORMS and local_rises.size:
      gap_w = (float(local_rises[0]) - a_s) / max(stroke_width, 1e-6)
      penalty += W_RISE_LATE_ENTRY * max(0.0, gap_w - RISE_ENTRY_MAX_W)
    if boundary_constrained:
      distance_w = float(np.min(np.abs(rises - a_s))) / max(stroke_width, 1e-6)
      penalty += W_RISE_BOUNDARY * max(0.0,
                                       distance_w - RISE_BOUNDARY_MAX_W)
    return -penalty

  def segment_penalty(a_s: float, b_s: float, letter_idx: int) -> float:
    # The DP closes letters[letter_idx] (0-indexed) when it places the cut at
    # b_s -- letters are ordered right to left, one per segment, same order
    # as the DP steps -- so the letter identity of a segment is always known
    # from which DP step is closing it, no separate alignment needed.
    if ctx is None or "letters" not in ctx:
      return 0.0
    letter = ctx["letters"][letter_idx]
    form = ctx["forms"][letter_idx] if ctx.get("forms") else None
    skeleton_score, _, _ = best_skeleton_match(
        ctx, a_s, b_s, letter_idx, consume=1)
    return (tooth_prior(a_s, b_s, letter, form)
            + shape_prior(a_s, b_s, letter, form)
            + branch_prior(a_s, b_s, letter, form)
            + rise_prior(a_s, b_s, letter, form)
            + skeleton_score)

  def interval_value(a_s: float, b_s: float, letter_start: int,
                     consume: int
                     ) -> Optional[Tuple[float, Optional[str]]]:
    if consume == 1:
      if too_short(a_s, b_s):
        return None
      skeleton_score, template, _ = best_skeleton_match(
          ctx, a_s, b_s, letter_start, consume=1)
      # segment_penalty already includes skeleton_score; the second lookup is
      # only to retain the winning template name in the reconstructed lattice.
      template_name = (template.name
                       if template is not None and skeleton_score > 0 else None)
      return (W_WIDTH_PRIOR * width_prior(b_s - a_s)
              + segment_penalty(a_s, b_s, letter_start), template_name)
    if not logical_ligatures:
      return None
    # A composite is admitted only by positive shape evidence. Unlike an
    # ordinary letter, it is not required to contain one minimum-size run per
    # consumed letter: overlapping ligature components are precisely why this
    # transition exists.
    if (b_s - a_s) < MIN_SPINE_RUN_W * stroke_width:
      return None
    shape_score, template, _ = best_skeleton_match(
        ctx, a_s, b_s, letter_start, consume)
    if template is None or shape_score < LIGATURE_MIN_SCORE:
      return None
    return (W_WIDTH_PRIOR * width_prior(b_s - a_s, consume)
            + shape_score, template.name)

  def consumptions(letter_start: int) -> List[int]:
    out = [1]
    if logical_ligatures:
      for template in ctx.get("skeleton_templates", ()):
        consume = len(template.letters)
        if (consume > 1
            and _template_applies(template, ctx.get("letters", ()),
                                  ctx.get("forms", ()), letter_start)):
          out.append(consume)
    return sorted(set(out))

  NEG = -1e18
  # dp[l][i]: best total after closing l transcription letters at physical
  # candidate i. A transition may consume one letter or a recognized
  # multi-letter ligature, so l is no longer synonymous with cut count.
  dp = [[NEG] * m for _ in range(n_letters)]
  back: List[List[Optional[Tuple[int, int, Optional[str]]]]] = [
      [None] * m for _ in range(n_letters)]

  for consume in consumptions(0):
    if consume >= n_letters:
      continue
    for i, c in enumerate(cand):
      interval = interval_value(0.0, c.s, 0, consume)
      if interval is None:
        continue
      val = c.score + interval[0]
      if val > dp[consume][i]:
        dp[consume][i] = val
        back[consume][i] = (0, -1, interval[1])

  for closed in range(1, n_letters):
    for p in range(m):
      if dp[closed][p] <= NEG / 2:
        continue
      for consume in consumptions(closed):
        new_closed = closed + consume
        if new_closed >= n_letters:
          continue
        for i in range(p + 1, m):
          interval = interval_value(
              cand[p].s, cand[i].s, closed, consume)
          if interval is None:
            continue
          val = dp[closed][p] + cand[i].score + interval[0]
          if val > dp[new_closed][i]:
            dp[new_closed][i] = val
            back[new_closed][i] = (closed, p, interval[1])

  best = NEG
  best_tail: Optional[Tuple[int, int, int, Optional[str]]] = None
  # A whole-PAW ligature has no preceding physical cut.
  for consume in consumptions(0):
    if consume != n_letters:
      continue
    interval = interval_value(0.0, total_len, 0, consume)
    if interval is not None and interval[0] > best:
      best = interval[0]
      best_tail = (0, -1, consume, interval[1])

  for closed in range(1, n_letters):
    consume = n_letters - closed
    if consume not in consumptions(closed):
      continue
    for p in range(m):
      if dp[closed][p] <= NEG / 2:
        continue
      interval = interval_value(cand[p].s, total_len, closed, consume)
      if interval is None:
        continue
      val = dp[closed][p] + interval[0]
      if val > best:
        best = val
        best_tail = (closed, p, consume, interval[1])

  if best_tail is None:
    return ([], None,
            "no candidate set satisfies the minimum-segment constraint", [])

  closed, i, consume, template_name = best_tail
  a_s = 0.0 if i < 0 else cand[i].s
  groups = [SegmentGroup(closed, closed + consume, a_s, total_len,
                         template_name)]
  while closed > 0 and i >= 0:
    prev = back[closed][i]
    if prev is None:
      return [], None, "incomplete DP backtrace", []
    prev_closed, prev_i, template_name = prev
    a_s = 0.0 if prev_i < 0 else cand[prev_i].s
    groups.append(SegmentGroup(prev_closed, closed, a_s, cand[i].s,
                               template_name))
    closed, i = prev_closed, prev_i
  groups.reverse()

  endpoint_map = {round(c.s, 9): c for c in cand}
  chosen = [endpoint_map[round(group.b_s, 9)] for group in groups[:-1]]
  return chosen, float(best), "", groups


# ---------------------------------------------------------------------------
# Letter attribution: every ink pixel goes to the letter owning the nearest
# medial vertex, so branches off the spine (the tah bar, the meem tail) follow
# the spine interval they hang from.
# ---------------------------------------------------------------------------

def attribute_pixels(paw_mask: np.ndarray, medial: Medial,
                     cuts: Sequence[Candidate], n_letters: int,
                     vertex_s: Optional[np.ndarray] = None,
                     pixel_s: Optional[np.ndarray] = None,
                     groups: Optional[Sequence[SegmentGroup]] = None
                     ) -> np.ndarray:
  bounds = [c.s for c in cuts]
  ys, xs = np.nonzero(paw_mask > 0)
  out = -np.ones(paw_mask.shape, dtype=np.int16)
  def assign(group_index: np.ndarray) -> np.ndarray:
    if not groups:
      return np.clip(group_index, 0, n_letters - 1)
    owners = np.array([
        group.letter_start if group.letter_end - group.letter_start == 1 else -2
        for group in groups], dtype=np.int16)
    return owners[np.clip(group_index, 0, len(owners) - 1)]
  if pixel_s is not None:
    out[ys, xs] = assign(np.searchsorted(
        bounds, pixel_s[ys, xs], side="right"))
  else:
    if vertex_s is None:
      vertex_s = vertex_spine_coordinates(medial)
    vertex_letter = assign(np.searchsorted(bounds, vertex_s, side="right"))
    _, idx = cKDTree(medial.vertices).query(np.column_stack([xs, ys]),
                                           workers=QUERY_WORKERS)
    out[ys, xs] = vertex_letter[idx]
  return out


def attribute_body_join_pixels(
    paw_mask: np.ndarray, medial: Medial, groups: Sequence[SegmentGroup],
    joins: Sequence[JoinSpan], n_letters: int,
    vertex_s: Optional[np.ndarray] = None,
    pixel_s: Optional[np.ndarray] = None,
    return_join_owner: bool = False,
    ink: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> np.ndarray:
  """Label body ink and elastic connector ink as separate states.

  ``-2`` denotes a fused multi-letter body and ``-3`` a measured join.
  Unlike cut-based attribution, no arbitrary point partitions the connector
  between its neighbours.

  ``return_join_owner`` additionally yields, per join pixel, the letter
  boundary its connector spans -- boundary ``k`` lying between letters ``k-1``
  and ``k``. The join label alone says a pixel is connector ink but not which
  connector, and a consumer that wants to charge a kashida to the letter it
  leaves needs to know that. Everything else is ``-1``.

  ``ink`` is ``np.nonzero(paw_mask > 0)`` when the caller already has it; see
  `pixel_spine_coordinates`.
  """
  ys, xs = np.nonzero(paw_mask > 0) if ink is None else ink
  out = -np.ones(paw_mask.shape, dtype=np.int16)
  join_owner = -np.ones(paw_mask.shape, dtype=np.int16)
  if xs.size == 0:
    return (out, join_owner) if return_join_owner else out
  if pixel_s is not None:
    coordinates = np.asarray(pixel_s[ys, xs], dtype=float)
  else:
    if vertex_s is None:
      vertex_s = vertex_spine_coordinates(medial)
    _, nearest = cKDTree(medial.vertices).query(np.column_stack([xs, ys]),
                                               workers=QUERY_WORKERS)
    coordinates = np.asarray(vertex_s[nearest], dtype=float)

  labels = -np.ones(xs.size, dtype=np.int16)
  max_s = max(
      [1.0] + [group.b_s for group in groups]
      + [join.end.s for join in joins if join.end is not None])
  # pixel_spine_coordinates is intentionally float32 to keep page-sized maps
  # small.  At a spine endpoint its rounded value can exceed the float64 body
  # endpoint by a few ulps, leaving an entire attached branch unassigned.  The
  # tolerance below covers only that representation error; it is orders of
  # magnitude below one image pixel or one profile sample.
  coordinate_tolerance = max(
      1e-6, 4.0 * float(np.finfo(np.float32).eps) * max_s)
  for group in groups:
    owner = (group.letter_start
             if group.letter_end - group.letter_start == 1 else -2)
    inside = ((coordinates >= group.a_s - coordinate_tolerance)
              & (coordinates <= group.b_s + coordinate_tolerance))
    labels[inside] = int(np.clip(owner, -2, n_letters - 1))
  boundaries = -np.ones(xs.size, dtype=np.int16)
  for join in joins:
    if join.start is None or join.end is None:
      continue
    inside = ((coordinates >= join.start.s - coordinate_tolerance)
              & (coordinates <= join.end.s + coordinate_tolerance))
    labels[inside] = -3
    boundaries[inside] = int(join.letter_boundary)
  out[ys, xs] = labels
  if not return_join_owner:
    return out
  join_owner[ys, xs] = boundaries
  return out, join_owner


# ---------------------------------------------------------------------------
# Per-word pipeline
# ---------------------------------------------------------------------------

def png_size(path: str) -> Optional[Tuple[int, int]]:
  """(height, width) from a PNG's IHDR chunk, without decoding the image.

  Returns None for anything that is not a PNG this can read, so a caller can
  fall back to decoding rather than guess.
  """
  try:
    with open(path, "rb") as handle:
      header = handle.read(24)
  except OSError:
    return None
  if (len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n"
      or header[12:16] != b"IHDR"):
    return None
  width, height = struct.unpack(">II", header[16:24])
  return (int(height), int(width))


def paw_masks_from_analysis(paws_meta: Sequence[dict], page_dir: str,
                            shape: Tuple[int, int]
                            ) -> List[Tuple[int, np.ndarray, List[str],
                                            List[str]]]:
  """PAW masks straight from the analyzer, in padded word coordinates.

  The analyzer now emits a body image per PAW plus the text that produced
  it, so the two guesses this module used to make -- splitting the word into
  connected components, then aligning those components to text-derived PAW
  ranges by ink mass -- are both unnecessary. Each body is pasted back at
  its (x, y) offset within the word, which reproduces the word body exactly
  (verified pixel-identical on page 3), so every coordinate downstream --
  the overlay, letter_boxes, the baseline -- is unchanged.

  Joining forms are derived from the PAW's own text. A PAW boundary falls
  exactly where a letter does not join forwards, so its first letter has no
  preceding neighbour either way and the forms match the word-level ones.
  """
  out: List[Tuple[int, np.ndarray, List[str], List[str]]] = []
  for index, spec in enumerate(paws_meta):
    src = spec.get("bodyPath") or ""
    # Resolve against page_dir rather than trusting the recorded relative
    # path, which is written relative to the analyzer's working directory.
    path = src if os.path.isabs(src) and os.path.isfile(src) else os.path.join(
        page_dir, "subwordimgs", os.path.basename(src))
    if not src or not os.path.isfile(path):
      continue
    bw = read_bw_white_is_ink(path)
    canvas = np.zeros(shape, dtype=bw.dtype)
    x0, y0 = int(spec.get("x", 0)) + PAD_PX, int(spec.get("y", 0)) + PAD_PX
    h, w = bw.shape[:2]
    if y0 + h > shape[0] or x0 + w > shape[1]:
      continue
    canvas[y0:y0 + h, x0:x0 + w] = bw
    p_bases, p_forms, _ = describe_word(spec.get("text", ""))
    # The analyzer's own index, not the position in this list: a PAW whose
    # body image is missing is dropped here, and everything downstream --
    # `paws.paw_index` in the corpus, the mark's owning PAW -- addresses PAWs
    # by the analyzer's numbering.
    out.append((index, canvas, p_bases, p_forms))
  return out


@dataclass
class PawEntry:
  """One unit of work for the segmenter: a mask and the letters written in it.

  Normally one PAW. Where the analyzer reports that a single component holds
  two PAWs that only touch in the scan, one entry carries both of them, and
  `members` says how its letters divide between them.
  """
  mask: np.ndarray
  letters: List[str]
  forms: List[str]
  broken: bool = False
  merged: bool = False
  # (analyzer PAW index, letter count), in writing order.
  members: List[Tuple[int, int]] = field(default_factory=list)

  @property
  def paw_index(self) -> int:
    return self.members[0][0] if self.members else 0


def segment_word(body_path: str, text: str, baseline_y: float,
                 want_pixels: bool = True,
                 min_confidence: float = MIN_CUT_CONFIDENCE,
                 shape_priors: Optional[dict] = None,
                 skeleton_shapes: bool = False,
                 logical_ligatures: bool = False,
                 skeleton_templates: Optional[Sequence[SkeletonTemplate]] = None,
                 elastic_body_templates: Optional[
                     Sequence[ElasticBodyTemplate]] = None,
                 body_join_alignment: bool = False,
                 cut_policy: str = "next-start",
                 paws_meta: Optional[Sequence[dict]] = None,
                 page_dir: str = "",
                 alignment_trace: Optional[dict] = None) -> WordResult:
  if skeleton_templates is None or elastic_body_templates is None:
    loaded_skeleton, loaded_elastic = load_template_definitions()
    if skeleton_templates is None:
      skeleton_templates = loaded_skeleton
    if elastic_body_templates is None:
      elastic_body_templates = loaded_elastic
  bases, forms, paw_ranges = describe_word(text)
  baseline = baseline_y + PAD_PX

  work: List[PawEntry] = []

  if paws_meta:
    # The analyzer has already split this word into PAW bodies, so the word
    # body image contributes nothing here but its dimensions -- every pixel
    # used below comes from the per-PAW images pasted onto that canvas.
    # Decoding a 3.6 M-pixel PNG to read its shape costs ~0.6 s a page.
    size = png_size(body_path)
    if size is None:
      size = read_bw_white_is_ink(body_path).shape[:2]
    shape = (size[0] + 2 * PAD_PX, size[1] + 2 * PAD_PX)
    for index, pm, p_bases, p_forms in paw_masks_from_analysis(
        paws_meta, page_dir, shape):
      # The analyzer decided which ink belongs to which PAW, so a PAW is never
      # handed another PAW's letters -- except where it could not decide,
      # because the two PAWs touch and are one component. It says so, and the
      # two are collected into one entry holding the ink they share and the
      # letters of both. Nothing here separates them; the entry exists so that
      # neither is measured on the other's ink, and so that whoever writes the
      # cut has the pair in one place.
      if (work and paws_meta[index].get("bodyMergedWithPrevious")
          and work[-1].members
          and work[-1].members[-1][0] == index - 1):
        previous = work[-1]
        previous.mask = np.maximum(previous.mask, pm)
        previous.letters = previous.letters + p_bases
        previous.forms = previous.forms + p_forms
        previous.merged = True
        previous.members.append((index, len(p_bases)))
        continue
      work.append(PawEntry(mask=pm, letters=p_bases, forms=p_forms,
                           members=[(index, len(p_bases))]))
  else:
    bw = read_bw_white_is_ink(body_path)
    mask = cv2.copyMakeBorder(bw, PAD_PX, PAD_PX, PAD_PX, PAD_PX,
                              cv2.BORDER_CONSTANT, value=0)
    width0 = estimate_stroke_width(mask)
    paws: List[np.ndarray] = []
    if width0 > 0:
      paws = paw_components(mask, width0)
    # Align components to text-predicted PAWs rather than pairing by
    # position, so a merged pair of PAWs is still told how many letters it
    # holds.
    ink = [float((m > 0).sum()) for m in paws]
    letters_per_paw = [hi - lo for lo, hi in paw_ranges]
    assignment: List[Tuple[List[str], List[str], bool, bool]] = [
        ([], [], True, False) for _ in paws]
    for comp_idx, paw_idx in align_paws(ink, letters_per_paw):
      letters, letter_forms = [], []
      for j in paw_idx:
        lo, hi = paw_ranges[j]
        letters.extend(bases[lo:hi])
        letter_forms.extend(forms[lo:hi])
      # Several components sharing one PAW means a broken stroke: the letters
      # cannot be located from one fragment, so those components abstain.
      broken = len(comp_idx) > 1
      if broken:
        for i in comp_idx:
          assignment[i] = ([], [], True, False)
      else:
        assignment[comp_idx[0]] = (letters, letter_forms, False, len(paw_idx) > 1)
    # A PAW here is a component, not a text PAW, so a merged one cannot be cut
    # into numbered PAWs the way the analyzer's can: there is nothing to number
    # them by. It is reported merged, as before.
    work = [PawEntry(mask=component, letters=letters, forms=letter_forms,
                     broken=broken, merged=merged,
                     members=[(index, len(letters))])
            for index, (component, (letters, letter_forms, broken, merged))
            in enumerate(zip(paws, assignment))]

  # A PAW whose body the analyzer gave to the word before it has no body
  # image, so it never became a unit of work above. It is still one of this
  # word's PAWs: dropping it would slide every later letter's index along the
  # word by its letter count. Report it for what it is instead.
  absent: List[PawResult] = []
  if paws_meta:
    present = {index for entry in work for index, _ in entry.members}
    for index, spec in enumerate(paws_meta):
      if index in present or not spec.get("bodyConnectedToPreviousWord"):
        continue
      if index >= len(paw_ranges):
        continue
      lo, hi = paw_ranges[index]
      absent.append(PawResult(
          paw_index=index, letters=list(bases[lo:hi]), forms=list(forms[lo:hi]),
          ok=False, low_confidence=True, connected_to_previous_word=True,
          reason="body is joined to the previous word",
          body_join_alignment=body_join_alignment, cut_policy=cut_policy))

  # PAWs, not units of work: two that arrived as one component are still two
  # PAWs, and counting the blob once would say the word is a PAW short when
  # the analyzer has just told us it is not.
  observed_paws = sum(len(entry.members) for entry in work) + len(absent)
  result = WordResult(word_id=os.path.basename(body_path), text=text, bases=bases,
                      forms=forms, expected_paws=len(paw_ranges),
                      observed_paws=observed_paws,
                      abstain=observed_paws != len(paw_ranges),
                      baseline_y=baseline)

  for entry in work:
    # Two PAWs the analyzer could not separate, because their ink touches and
    # connected-component labeling has no way to tell them apart. Cutting the
    # contact is a separate piece of work; until it exists, both PAWs are
    # reported as what they are -- one row each, letters intact, no
    # measurement -- rather than each being measured on ink that is half its
    # neighbour's.
    if len(entry.members) > 1:
      offset = 0
      for member, (paw_index, count) in enumerate(entry.members):
        shared = PawResult(
            paw_index=paw_index,
            letters=entry.letters[offset:offset + count],
            forms=entry.forms[offset:offset + count],
            ok=False, merged=True, low_confidence=True,
            reason="two PAWs share one component; not cut apart",
            body_join_alignment=body_join_alignment, cut_policy=cut_policy)
        # The shape exists even though the cut does not, so the first member
        # carries the joint blob under the fused-body label -- the same
        # convention an uncut ligature uses. It gives the corpus an outline to
        # draw and ink to count for the pair as a whole; what it deliberately
        # does not give is a spine, because there is none, and every
        # measurement downstream is gated on that.
        if want_pixels and member == 0:
          shared.pixel_letter = np.where(entry.mask, -2, -1).astype(np.int16)
        result.paws.append(shared)
        offset += count
      continue

    paw_mask = entry.mask
    k = entry.paw_index
    letters, letter_forms = entry.letters, entry.forms
    broken, merged = entry.broken, entry.merged
    pr = PawResult(paw_index=k, letters=letters, forms=letter_forms, ok=False,
                   merged=merged, body_join_alignment=body_join_alignment,
                   cut_policy=cut_policy)
    result.paws.append(pr)

    paw_trace: Optional[dict] = None
    if (alignment_trace is not None
        and int(alignment_trace.get("targetPaw", -1)) == k):
      paw_trace = {"pawIndex": k, "letters": list(letters),
                   "forms": list(letter_forms)}
      alignment_trace["paw"] = paw_trace

    bootstrap = estimate_stroke_width(paw_mask)
    if bootstrap <= 0 or not letters:
      pr.reason = ("broken stroke: several components share one PAW"
                   if broken else "no ink or no expected letters")
      continue

    contours = paw_contours(paw_mask, bootstrap)
    topology_mask = medial_topology_mask(paw_mask, contours)
    generators = contour_generators(contours, bootstrap)
    medial = build_medial(topology_mask, generators, bootstrap, baseline)
    if medial is None:
      pr.reason = "medial axis could not be built"
      continue
    pr.medial = medial
    pr.loops = medial.loops
    pr.loop_evidence = medial_loop_evidence(medial, paw_mask)
    pr.spine_length = float(medial.spine_s[-1])

    # exact refinement: twice the median medial radius along the spine
    stroke_width = 2.0 * float(np.median(medial.radii[medial.spine]))
    pr.stroke_width = stroke_width
    vertex_s: Optional[np.ndarray] = None
    pixel_s: Optional[np.ndarray] = None
    # Located once and handed to everything below: the attribution maps, the
    # spine map and the per-letter boxes all want exactly these pixels, and
    # the mask they are in is the size of the whole padded word.
    paw_ink = np.nonzero(paw_mask > 0)
    single_letter = len(letters) == 1
    if single_letter:
      pr.ok = True
      pr.confidence = 1.0
      pr.margin = 1.0
      pr.segment_groups = [SegmentGroup(
          0, 1, 0.0, pr.spine_length, None)]
    else:
      vertex_s = vertex_spine_coordinates(medial)
      pixel_s = pixel_spine_coordinates(paw_mask, medial, vertex_s, paw_ink)
      profile = cut_profile(paw_mask, medial, stroke_width, pixel_s)
      if profile is None:
        pr.reason = "spine too short to profile"
        continue
      profile["loop_evidence"] = pr.loop_evidence
      necks = tooth_necks(profile, stroke_width)
      peaks, peak_strengths, peak_reliefs = _spine_extrema(
          profile, stroke_width, valleys=False)
      valleys, valley_strengths, valley_reliefs = _spine_extrema(
          profile, stroke_width, valleys=True)
      pr.skeleton_events = drop_weak_events(skeleton_events(
          profile, medial, baseline, stroke_width, peaks, valleys,
          peak_strengths, valley_strengths, peak_reliefs, valley_reliefs))
      prefix_r = profile_prefix_radius(profile)
      if body_join_alignment:
        alignment = force_align_bodies_and_joins(
            profile, pr.skeleton_events, letters, letter_forms, baseline,
            stroke_width, cut_policy, elastic_body_templates, paw_trace)
        if alignment.reason:
          pr.reason = alignment.reason
          continue
        pr.cuts = alignment.cuts
        pr.joins = alignment.joins
        pr.segment_groups = alignment.groups
        # Keep the exact topology realization selected by the alignment DP.
        # Re-matching inside the materialized body span can lose an event when
        # an event-relative endpoint is rounded to the preceding profile sample.
        pr.shape_matches = list(alignment.shape_matches)
        pr.ok = True
        pr.margin = alignment.margin
        confidence_terms = [join.confidence for join in pr.joins]
        pr.confidence = min(confidence_terms) if confidence_terms else 1.0
        boundary_s = np.array(
            [anchor.s for join in pr.joins
             for anchor in (join.start, join.end) if anchor is not None],
            dtype=float)
      else:
        pr.candidates = cut_candidates(profile, baseline, stroke_width)
        boundary_s = np.array(
            [c.s for c in pr.candidates if c.kind == "min"])
      ctx = {
          "letters": letters, "forms": letter_forms,
          "tooth_necks": necks,
          "tooth_margin": TOOTH_MARGIN_W * stroke_width,
          "profile": profile, "prefix_r": prefix_r, "priors": shape_priors,
          "boundary_s": boundary_s, "boundary_tol": TOOTH_MIN_SEP_W * stroke_width,
          "branches": branch_positions(medial, stroke_width, BRANCH_MIN_LEN_W),
          "skeleton_events": pr.skeleton_events,
          "skeleton_shapes": skeleton_shapes,
          "logical_ligatures": logical_ligatures,
          "skeleton_templates": tuple(skeleton_templates),
      }
      ctx["branch_s"], ctx["branch_cum"] = branch_attachments(medial)
      ctx["rises"] = peaks
      if not body_join_alignment:
        cuts, groups, reason = choose_cuts(
            pr.candidates, len(letters), pr.spine_length, stroke_width, ctx)
        if reason:
          pr.reason = reason
          continue
        pr.cuts = cuts
        pr.segment_groups = groups
        pr.ok = True
        confidence_terms = [c.score / SCORE_REFERENCE for c in cuts]
        pr.margin = cut_margin(pr.candidates, cuts, len(letters),
                               pr.spine_length, stroke_width, ctx, groups)

        for group in groups:
          consume = group.letter_end - group.letter_start
          match_score, template, event_ids = best_skeleton_match(
              ctx, group.a_s, group.b_s, group.letter_start, consume)
          if template is not None:
            pr.shape_matches.append(ShapeMatch(
                template.name, group.letter_start, group.letter_end,
                group.a_s, group.b_s,
                match_score, event_ids))
            if consume > 1:
              confidence_terms.append(match_score / max(
                  W_SKELETON_TEMPLATE * template.weight, 1e-6))
        pr.confidence = min(confidence_terms) if confidence_terms else 0.0
      letter_intervals: List[Tuple[float, float]] = [
          (0.0, pr.spine_length) for _ in letters]
      for group in pr.segment_groups:
        for li in range(group.letter_start, group.letter_end):
          letter_intervals[li] = (group.a_s, group.b_s)
      pr.letter_teeth = [
          (TOOTH_COUNT.get(letters[li]),
           observed_teeth(valleys, letter_intervals[li][0],
                          letter_intervals[li][1],
                          ctx["tooth_margin"], letters[li], boundary_s,
                          ctx["boundary_tol"], letter_forms[li],
                          ctx["rises"]))
          for li in range(len(letters))]
      pr.letter_shape = [
          {"extent": ((letter_intervals[li][1] - letter_intervals[li][0]) /
                      stroke_width),
           "radius": mean_radius_between(
               profile, prefix_r, letter_intervals[li][0],
               letter_intervals[li][1]) / stroke_width,
           "teeth": float(pr.letter_teeth[li][1])}
          for li in range(len(letters))]
      # A merged component holds PAWs that only *touch* in the scan. At least
      # one of its cuts therefore falls at an accidental contact, which looks
      # nothing like a cursive join, so the geometry cannot be trusted there
      # however well it scores.
      pr.low_confidence = (pr.confidence < min_confidence
                           or pr.margin < MIN_CUT_MARGIN
                           or pr.merged)

    pixel_join = None
    if body_join_alignment and single_letter:
      # The one-letter PAW above owns a single body group spanning [0,
      # spine_length] and has no joins, and every ink pixel's spine coordinate
      # lies in that interval by construction -- so the attribution is "letter
      # 0 wherever there is ink", already known before any of it is computed.
      # Deriving it anyway costs a multi-source Dijkstra plus a nearest-vertex
      # query over every ink pixel, and single-letter PAWs are half of them.
      pixel_letter = np.full(paw_mask.shape, -1, dtype=np.int16)
      pixel_letter[paw_ink] = 0
      pixel_join = np.full(paw_mask.shape, -1, dtype=np.int16)
    elif body_join_alignment:
      pixel_letter, pixel_join = attribute_body_join_pixels(
          paw_mask, medial, pr.segment_groups, pr.joins, len(letters),
          vertex_s, pixel_s, return_join_owner=True, ink=paw_ink)
    else:
      pixel_letter = attribute_pixels(
          paw_mask, medial, pr.cuts, len(letters), vertex_s, pixel_s,
          pr.segment_groups or None)
    if want_pixels:
      pr.pixel_letter = pixel_letter
      pr.pixel_join = pixel_join
    # A pass per letter over just this PAW's ink, rather than over the whole
    # padded word once per letter -- which cost more than deriving the labels
    # did.
    label_ys, label_xs = paw_ink
    labels_here = pixel_letter[label_ys, label_xs]
    for li in range(len(letters)):
      mine = labels_here == li
      xs, ys = label_xs[mine], label_ys[mine]
      if xs.size == 0:
        pr.letter_boxes.append(None)
        pr.letter_ink.append(0)
      else:
        pr.letter_boxes.append((int(xs.min()) - PAD_PX, int(ys.min()) - PAD_PX,
                                int(xs.max()) - PAD_PX, int(ys.max()) - PAD_PX))
        pr.letter_ink.append(int(xs.size))

    # A component holding two PAWs is a fact about the scan, not about the
    # hand, so a PAW cut out of one cannot be trusted the way an ordinary one
    # can. Set on the way out because a one-letter PAW never reaches the cut
    # machinery that otherwise says this.
    pr.low_confidence = pr.low_confidence or pr.merged

  # Slot the absent PAWs back where the text puts them, so a reader of
  # result.paws walks the word in writing order and the letter indices line up.
  if absent:
    result.paws.extend(absent)
    result.paws.sort(key=lambda paw: paw.paw_index)

  # Say which PAWs the analyzer handed to the word before them. Usually the
  # PAW is here with an empty body image rather than missing outright, so the
  # flag is stamped by index and covers both: what matters downstream is that
  # the row says why it has no ink, instead of reading as an ordinary failure.
  if paws_meta:
    joined = {index for index, spec in enumerate(paws_meta)
              if spec.get("bodyConnectedToPreviousWord")}
    for paw in result.paws:
      if paw.paw_index in joined:
        paw.connected_to_previous_word = True
        paw.low_confidence = True
        paw.reason = "body is joined to the previous word"
  return result


# ---------------------------------------------------------------------------
# H. output
# ---------------------------------------------------------------------------

def render_overlay(body_path: str, result: WordResult, scale: int = 3) -> np.ndarray:
  bw = read_bw_white_is_ink(body_path)
  mask = cv2.copyMakeBorder(bw, PAD_PX, PAD_PX, PAD_PX, PAD_PX,
                            cv2.BORDER_CONSTANT, value=0)
  canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
  canvas[mask > 0] = (70, 70, 70)

  for pr in result.paws:
    if pr.pixel_letter is not None:
      canvas[pr.pixel_letter == -2] = (80, 180, 180)
      canvas[pr.pixel_letter == -3] = JOIN_COLOR
      for li in range(len(pr.letters)):
        canvas[pr.pixel_letter == li] = LETTER_COLORS[li % len(LETTER_COLORS)]

  cv2.line(canvas, (0, int(result.baseline_y)),
           (canvas.shape[1], int(result.baseline_y)), (0, 200, 0), 2)

  for pr in result.paws:
    if pr.medial is None:
      continue
    med = pr.medial
    matched_event_ids = {
        event_id for match in pr.shape_matches for event_id in match.events}
    context_event_ids = {
        event_id for match in pr.shape_matches
        for event_id in match.before_events + match.after_events}
    matched_branch_event_ids = {
        event_id for event_id in matched_event_ids
        if (0 <= event_id < len(pr.skeleton_events)
            and pr.skeleton_events[event_id].kind.endswith("branch"))}
    matched_pruned_points = [
        np.asarray(pr.skeleton_events[event_id].xy, dtype=float)
        for event_id in matched_branch_event_ids
        if pr.skeleton_events[event_id].source == "pruned"]
    for a, b in med.edges:
      cv2.line(canvas, tuple(np.rint(med.vertices[a]).astype(int)),
               tuple(np.rint(med.vertices[b]).astype(int)), (200, 60, 60), 1)
    # Pruning removes short arms from ``med.edges``, but they remain topology
    # evidence and must stay inspectable in diagnostics. Draw every recorded
    # path; make a path prominent when the winning template consumed its event.
    for branch in med.pruned_branches:
      points = np.rint(branch.points).astype(int).reshape(-1, 1, 2)
      anchor_xy = med.vertices[branch.anchor]
      matched = any(
          np.linalg.norm(anchor_xy - event_xy) <= 1.0
          for event_xy in matched_pruned_points)
      cv2.polylines(
          canvas, [points], False,
          PRUNED_BRANCH_COLOR if matched else PRUNED_BRANCH_PATH_COLOR,
          5 if matched else 2)
    for branch_index, (vertex, branch_length) in enumerate(
        used_pruned_branches(med, pr.stroke_width, BRANCH_MIN_LEN_W)):
      point = tuple(np.rint(med.vertices[vertex]).astype(int))
      cv2.drawMarker(canvas, point, PRUNED_BRANCH_COLOR,
                     markerType=cv2.MARKER_CROSS, markerSize=28, thickness=4)
      #cv2.putText(canvas, f"P{branch_index}:{branch_length:.0f}",
      #            (point[0] + 14, point[1] - 14),
      #            cv2.FONT_HERSHEY_SIMPLEX, 0.75, PRUNED_BRANCH_COLOR, 2,
      #            cv2.LINE_AA)
    sp = np.rint(med.vertices[med.spine]).astype(int)
    cv2.polylines(canvas, [sp.reshape(-1, 1, 2)], False, (255, 230, 120), 3)
    for c in pr.candidates:
      cv2.circle(canvas, tuple(np.rint(c.xy).astype(int)), 8, (0, 220, 255), -1)
    cut_color = (0, 140, 255) if pr.low_confidence else (0, 0, 255)
    for c in pr.cuts:
      a, b = c.chord
      cv2.line(canvas, tuple(np.rint(a).astype(int)),
               tuple(np.rint(b).astype(int)), cut_color, 7)
    for join in pr.joins:
      if join.start is not None:
        cv2.drawMarker(
            canvas, tuple(np.rint(join.start.xy).astype(int)),
            JOIN_START_COLOR, markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=22, thickness=4)
      if join.end is not None:
        cv2.drawMarker(
            canvas, tuple(np.rint(join.end.xy).astype(int)),
            JOIN_END_COLOR, markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=22, thickness=4)
    for event_id, event in enumerate(pr.skeleton_events):
      color = SKELETON_EVENT_COLORS.get(event.kind, (220, 220, 220))
      point = tuple(np.rint(event.xy).astype(int))
      cv2.drawMarker(canvas, point, color, markerType=cv2.MARKER_DIAMOND,
                     markerSize=18, thickness=3)
      if event_id in context_event_ids:
        for angle in range(0, 360, 45):
          cv2.ellipse(canvas, point, (17, 17), 0, angle, angle + 24,
                      CONTEXT_EVENT_COLOR, 2, cv2.LINE_AA)
      if event_id in matched_event_ids:
        cv2.circle(canvas, point, 13, MATCHED_EVENT_COLOR, 3)

  out = cv2.resize(canvas, (canvas.shape[1] // scale, canvas.shape[0] // scale),
                   interpolation=cv2.INTER_AREA)
  for pr in result.paws:
    for li, box in enumerate(pr.letter_boxes):
      if box is None:
        continue
      cx = ((box[0] + box[2]) // 2 + PAD_PX) // scale
      cy = ((box[1] + box[3]) // 2 + PAD_PX) // scale
      cv2.putText(out, str(li), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                  (255, 255, 255), 2, cv2.LINE_AA)
  return out


def result_to_json(result: WordResult) -> dict:
  def logical_group(pr: PawResult, letter_index: int) -> Optional[List[int]]:
    for group in pr.segment_groups:
      if (group.letter_end - group.letter_start > 1
          and group.letter_start <= letter_index < group.letter_end):
        return [group.letter_start, group.letter_end]
    return None

  def cut_json(c: Candidate) -> dict:
    return {
        "s": round(c.s, 2), "x": round(c.xy[0] - PAD_PX, 2),
        "y": round(c.xy[1] - PAD_PX, 2), "kind": c.kind,
        "radius": round(c.radius, 2), "cutLen": round(c.cut_len, 2),
        "coreCutLen": round(c.core_cut_len, 2),
        "ownedCutLen": round(c.owned_cut_len, 2),
        "ownedExtraLen": round(c.owned_cut_len - c.core_cut_len, 2),
        "foreignCutLen": round(c.foreign_cut_len, 2),
        "globalCutLen": round(c.global_cut_len, 2),
        "runs": c.n_runs, "globalRuns": c.global_runs,
        "recoveredFromForeignInk": c.recovered,
        "chord": [[round(c.chord[0][0] - PAD_PX, 2), round(c.chord[0][1] - PAD_PX, 2)],
                  [round(c.chord[1][0] - PAD_PX, 2), round(c.chord[1][1] - PAD_PX, 2)]],
        "score": round(c.score, 3), "confidence": round(c.score / SCORE_REFERENCE, 3),
        "features": {"thinness": round(c.thinness, 3),
                     "narrowness": round(c.narrowness, 3),
                     "purity": round(c.purity, 3),
                     "adjacency": round(c.adjacency, 3),
                     "baseline": round(c.baseline_term, 3),
                     "horizontality": round(c.horizontality, 3)},
    }

  return {
      "wordId": result.word_id, "text": result.text, "bases": result.bases,
      "forms": result.forms, "expectedPaws": result.expected_paws,
      "observedPaws": result.observed_paws, "abstain": result.abstain,
      "baselineY": round(result.baseline_y - PAD_PX, 2),
      "paws": [{
          "pawIndex": pr.paw_index, "letters": pr.letters, "forms": pr.forms,
          "ok": pr.ok, "reason": pr.reason,
          "confidence": round(pr.confidence, 3),
          "margin": round(pr.margin, 3),
          "merged": pr.merged,
          "lowConfidence": pr.low_confidence,
          "strokeWidth": round(pr.stroke_width, 2),
          "spineLength": round(pr.spine_length, 2), "loops": pr.loops,
          "candidateCount": len(pr.candidates),
          "alignmentMode": ("body-join" if pr.body_join_alignment
                            else "candidate-dp"),
          "cutPolicy": pr.cut_policy,
          "skeletonEvents": [
              {"s": round(event.s, 2), "kind": event.kind,
               "strength": round(event.strength, 3),
               "x": round(event.xy[0] - PAD_PX, 2),
               "y": round(event.xy[1] - PAD_PX, 2),
               "source": event.source,
               "thickness": round(event.thickness, 3),
               "relief": round(event.relief, 3)}
              for event in pr.skeleton_events],
          "loopEvidence": [
              {"s": round(item.s, 2),
               "span": [round(item.a_s, 2), round(item.b_s, 2)],
               "count": item.count,
               "x": round(item.xy[0] - PAD_PX, 2),
               "y": round(item.xy[1] - PAD_PX, 2),
               "centres": [[round(centre[0] - PAD_PX, 2),
                            round(centre[1] - PAD_PX, 2)]
                           for centre in item.centres],
               "attachment": (None if item.attachment is None else
                              [round(item.attachment[0], 2),
                               round(item.attachment[1], 2)])}
              for item in pr.loop_evidence],
          "shapeMatches": [
              {"template": match.template,
               "letters": [match.letter_start, match.letter_end],
               "span": [round(match.a_s, 2), round(match.b_s, 2)],
               "score": round(match.score, 3),
               "events": list(match.events),
               "beforeEvents": list(match.before_events),
               "afterEvents": list(match.after_events)}
              for match in pr.shape_matches],
          "segmentGroups": [
              {"letters": [group.letter_start, group.letter_end],
               "span": [round(group.a_s, 2), round(group.b_s, 2)],
               "template": group.template}
              for group in pr.segment_groups],
          "joins": [
              {"letterBoundary": join.letter_boundary,
               "kind": join.kind,
               "span": ([round(join.start.s, 2), round(join.end.s, 2)]
                        if join.start is not None and join.end is not None
                        else None),
               "length": (round(join.end.s - join.start.s, 2)
                          if join.start is not None and join.end is not None
                          else None),
               "score": round(join.score, 3),
               "confidence": round(join.confidence, 3),
               "template": join.template,
               "start": (cut_json(join.start)
                         if join.start is not None else None),
               "end": (cut_json(join.end)
                       if join.end is not None else None)}
              for join in pr.joins],
          "candidates": [cut_json(c) for c in pr.candidates],
          "cuts": [cut_json(c) for c in pr.cuts],
          "letters_geometry": [
              {"index": i, "letter": pr.letters[i] if i < len(pr.letters) else None,
               "form": pr.forms[i] if i < len(pr.forms) else None,
               "logicalGroup": logical_group(pr, i),
               "bbox": pr.letter_boxes[i] if i < len(pr.letter_boxes) else None,
               "inkPixels": pr.letter_ink[i] if i < len(pr.letter_ink) else 0,
               "expectedTeeth": pr.letter_teeth[i][0] if i < len(pr.letter_teeth) else None,
               "observedTeeth": pr.letter_teeth[i][1] if i < len(pr.letter_teeth) else None,
               "shape": (
                   {k: round(v, 3) for k, v in pr.letter_shape[i].items()}
                   if i < len(pr.letter_shape) else None)}
              for i in range(len(pr.letters))],
      } for pr in result.paws],
  }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_page(analysis_root: str, page: int):
  page_dir = os.path.join(analysis_root, f"page{page:03d}")
  with open(os.path.join(page_dir, "words.json"), encoding="utf-8") as fh:
    return page_dir, json.load(fh)


def iter_words(page_json: dict):
  for line in page_json["lines"]:
    for word in line["words"]:
      yield line, word


def run_one(page_dir: str, line: dict, word: dict, out_dir: str,
            write_overlay: bool = True,
            min_confidence: float = MIN_CUT_CONFIDENCE,
            shape_priors: Optional[dict] = None,
            skeleton_shapes: bool = False,
            logical_ligatures: bool = False,
            skeleton_templates: Optional[Sequence[SkeletonTemplate]] = None,
            elastic_body_templates: Optional[
                Sequence[ElasticBodyTemplate]] = None,
            body_join_alignment: bool = False,
            cut_policy: str = "next-start",
            want_pixels: Optional[bool] = None):
  """Segment one analyzed word.

  ``want_pixels`` keeps the per-pixel letter attribution on the result. It
  defaults to ``write_overlay`` because the overlay is what needs it here, but
  a caller that measures letters rather than drawing them -- the corpus index
  builder -- needs the labels without paying for the PNG.
  """
  body_path = os.path.join(page_dir, "subwordimgs", f"{word['wordId']}_body.png")
  if not os.path.isfile(body_path):
    return None
  baseline_y = float(line["baseLine"]) - float(word["y"])
  result = segment_word(body_path, word.get("expectedText", ""), baseline_y,
                        want_pixels=(write_overlay if want_pixels is None
                                     else want_pixels),
                        min_confidence=min_confidence,
                        shape_priors=shape_priors,
                        skeleton_shapes=skeleton_shapes,
                        logical_ligatures=logical_ligatures,
                        skeleton_templates=skeleton_templates,
                        elastic_body_templates=elastic_body_templates,
                        body_join_alignment=body_join_alignment,
                        cut_policy=cut_policy,
                        paws_meta=word.get("paws"), page_dir=page_dir)
  result.word_id = word["wordId"]
  if write_overlay:
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f"{word['wordId']}_segmentation.png"),
                render_overlay(body_path, result))
  return result


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--analysis", required=True, help="analysis output root")
  ap.add_argument("--page", type=int, required=True)
  ap.add_argument("--word", help="word id, e.g. l02_w04")
  ap.add_argument("--batch", action="store_true", help="run over the whole page")
  ap.add_argument("--out", default="output/paw-segmentation")
  ap.add_argument("--min-confidence", type=float, default=MIN_CUT_CONFIDENCE,
                  help="flag a PAW low-confidence when its weakest cut is below this")
  ap.add_argument("--priors", help="path to a letter_shape_priors.json from "
                  "train_letter_shape_priors.py; uses the learned Fix 1b "
                  "shape prior at inference time when given")
  ap.add_argument("--skeleton-shapes", action="store_true",
                  help="score complete skeleton intervals with shape templates")
  ap.add_argument("--logical-ligatures", action="store_true",
                  help="allow a recognized template to consume multiple letters "
                  "without forcing an internal physical cut")
  ap.add_argument(
      "--skeleton-templates", default=DEFAULT_TEMPLATE_PATH,
      help="JSON template document (default: templates/oldmadina/"
           "skeleton_templates.json)")
  ap.add_argument(
      "--body-join-alignment", action="store_true",
      help="align body and elastic-join intervals directly, without cut candidates")
  ap.add_argument(
      "--cut-policy",
      choices=("current-end", "midpoint", "next-start", "none"),
      default="next-start",
      help="derive cuts from an aligned join (default: start of next body)")
  args = ap.parse_args()

  page_dir, page_json = load_page(args.analysis, args.page)
  os.makedirs(args.out, exist_ok=True)
  shape_priors = None
  if args.priors:
    with open(args.priors, encoding="utf-8") as fh:
      shape_priors = json.load(fh)
  skeleton_templates, elastic_body_templates = load_template_definitions(
      args.skeleton_templates)

  if not args.batch:
    if not args.word:
      ap.error("--word is required unless --batch is given")
    for line, word in iter_words(page_json):
      if word["wordId"] != args.word:
        continue
      result = run_one(page_dir, line, word, args.out,
                       min_confidence=args.min_confidence,
                       shape_priors=shape_priors,
                       skeleton_shapes=args.skeleton_shapes,
                       logical_ligatures=args.logical_ligatures,
                       skeleton_templates=skeleton_templates,
                       elastic_body_templates=elastic_body_templates,
                       body_join_alignment=args.body_join_alignment,
                       cut_policy=args.cut_policy)
      if result is None:
        print(f"no body image for {args.word}")
        return 1
      payload = result_to_json(result)
      with open(os.path.join(args.out, f"{args.word}_segmentation.json"), "w",
                encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
      print(json.dumps(payload, ensure_ascii=False, indent=2))
      return 0
    print(f"word {args.word} not found on page {args.page}")
    return 1

  summary, rows = [], []
  for line, word in iter_words(page_json):
    result = run_one(page_dir, line, word, os.path.join(args.out, "overlays"),
                     min_confidence=args.min_confidence,
                     shape_priors=shape_priors,
                     skeleton_shapes=args.skeleton_shapes,
                     logical_ligatures=args.logical_ligatures,
                     skeleton_templates=skeleton_templates,
                     elastic_body_templates=elastic_body_templates,
                     body_join_alignment=args.body_join_alignment,
                     cut_policy=args.cut_policy)
    if result is None:
      continue
    summary.append(result_to_json(result))
    for pr in result.paws:
      k = len(pr.letters)
      rows.append({
          "wordId": result.word_id, "text": result.text, "paw": pr.paw_index,
          "letters": "".join(pr.letters), "nLetters": k,
          "countMatch": not result.abstain,
          "candidates": len(pr.candidates),
          "joins": sum(join.kind != "fused" for join in pr.joins),
          "needed": (max(0, len(pr.segment_groups) - 1)
                     if pr.segment_groups else max(0, k - 1)),
          "sufficient": (pr.ok if args.body_join_alignment
                         else len(pr.candidates) >= max(0, k - 1)),
          "ok": pr.ok, "reason": pr.reason, "loops": pr.loops,
          "confidence": round(pr.confidence, 3),
          "margin": round(pr.margin, 3),
          "merged": pr.merged,
          "lowConfidence": pr.low_confidence,
          "accepted": pr.ok and not pr.low_confidence and not result.abstain,
          "strokeWidth": round(pr.stroke_width, 1),
      })

  with open(os.path.join(args.out, "page_summary.json"), "w", encoding="utf-8") as fh:
    json.dump(summary, fh, ensure_ascii=False, indent=2)
  if rows:
    with open(os.path.join(args.out, "page_summary.csv"), "w", newline="",
              encoding="utf-8") as fh:
      writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
      writer.writeheader()
      writer.writerows(rows)

  multi = [r for r in rows if r["nLetters"] >= 2]
  matched = [r for r in multi if r["countMatch"]]
  print(f"words: {len(summary)}  PAWs: {len(rows)}  multi-letter PAWs: {len(multi)}")
  if multi:
    evidence_label = ("body/join alignment found" if args.body_join_alignment
                      else "candidate pool sufficient (>= k-1)")
    print(f"{evidence_label}: "
          f"{sum(r['sufficient'] for r in multi)}/{len(multi)}")
    print(f"segmented ok: {sum(r['ok'] for r in multi)}/{len(multi)}")
    print(f"low confidence (< {args.min_confidence}): "
          f"{sum(r['lowConfidence'] for r in multi)}/{len(multi)}")
    print(f"accepted (count matches, segmented, confident): "
          f"{sum(r['accepted'] for r in multi)}/{len(multi)}")
  if matched:
    print(f"among count-matching words: ok "
          f"{sum(r['ok'] for r in matched)}/{len(matched)}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
