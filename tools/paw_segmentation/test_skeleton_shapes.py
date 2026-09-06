import unittest
import os
import sys
import json
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from segment_paw_letters import (
    BodyBoundarySpec,
    DEFAULT_TEMPLATE_PATH,
    ElasticBodyTemplate,
    LoopEvidence,
    Medial,
    SegmentGroup,
    SkeletonEvent,
    SkeletonTemplate,
    TemplateStep,
    match_skeleton_template,
    match_elastic_body_template,
    medial_loop_evidence,
    loop_count_in_span,
    loop_partition_constraints_match,
    branch_count_in_span,
    contour_generators,
    medial_topology_mask,
    paw_contours,
    template_branch_constraints_match,
    skeleton_branch_paths,
    skeleton_events,
    _spine_extrema,
    spine_peaks,
    spine_valleys,
    force_align_bodies_and_joins,
    attribute_body_join_pixels,
    _ordered_single_body_matches,
    _ordered_single_body_match_options,
    _complete_ordered_path_selection_key,
    _body_partitions,
    _template_body_boundary_index,
    _template_body_exit_index,
    elastic_body_template_to_json,
    elastic_body_templates_from_json,
    load_template_definitions,
    skeleton_template_to_json,
    skeleton_templates_from_json,
    _template_applies,
    _elastic_template_applies,
)


class SkeletonTemplateTest(unittest.TestCase):

  def test_weak_spine_extrema_are_emitted_with_measured_strength(self):
    # A 0.10-stroke valley is useful to a permissive template and must not be
    # discarded by the event generator's former 0.12 recognition threshold.
    profile = {
        "grid": np.arange(7.0),
        "y": np.array([0.0, 2.0, 5.0, 10.0, 7.0, 4.0, 3.0]),
        "ds": 1.0,
    }
    positions, strengths, reliefs = _spine_extrema(
        profile, stroke_width=100.0, valleys=True)

    np.testing.assert_array_equal(positions, [3.0])
    np.testing.assert_allclose(strengths, [0.07])
    # No off-spine envelope in this profile, so relief is just the strength.
    np.testing.assert_allclose(reliefs, [0.07])

  def test_spine_valley_is_geometric_opposite_of_peak(self):
    profile = {
        "grid": np.arange(9.0),
        "x": np.arange(9.0),
        "y": np.array([5.0, 4.0, 2.0, 4.0, 5.0, 7.0, 5.0, 4.0, 5.0]),
        "ds": 1.0,
    }

    np.testing.assert_array_equal(spine_peaks(profile, 1.0), [2.0, 7.0])
    np.testing.assert_array_equal(spine_valleys(profile, 1.0), [5.0])

    vertices = np.column_stack([profile["grid"], profile["y"]])
    medial = Medial(
        vertices=vertices, radii=np.ones(len(vertices)),
        edges=[(i, i + 1) for i in range(len(vertices) - 1)],
        generators=np.empty((0, 2)), spine=list(range(len(vertices))),
        spine_s=profile["grid"], loops=0)
    events = skeleton_events(
        profile, medial, baseline_y=5.0, stroke_width=1.0,
        peaks=np.array([2.0, 7.0]), valleys=np.array([5.0]),
        peak_strengths=np.array([3.0, 1.0]),
        valley_strengths=np.array([3.0]))
    extrema = [(event.kind, event.s, event.strength) for event in events
               if event.kind in {"peak", "valley"}]
    self.assertEqual(extrema, [
        ("peak", 2.0, 3.0), ("valley", 5.0, 3.0),
        ("peak", 7.0, 1.0)])

  def test_contour_generators_keep_small_counter_but_drop_pinhole(self):
    stroke_width = 80.0

    def generators_for(hole_radius: int) -> np.ndarray:
      mask = np.zeros((120, 260), dtype=np.uint8)
      cv2.rectangle(mask, (20, 20), (240, 100), 255, -1)
      cv2.circle(mask, (130, 60), hole_radius, 0, -1)
      return contour_generators(paw_contours(mask, stroke_width), stroke_width)

    # A 14-pixel radius counter occupies about 0.096 stroke-widths squared --
    # the size of the smallest genuine counter measured in the Mushaf, a Waw
    # eye at 0.087 -- so its inner contour must contribute generators near the
    # hole. A two-pixel scan pinhole is only about 0.002 and must not.
    counter = generators_for(14)
    pinhole = generators_for(2)
    centre = np.array([130.0, 60.0])
    self.assertLess(np.min(np.linalg.norm(counter - centre, axis=1)), 20.0)
    self.assertGreater(np.min(np.linalg.norm(pinhole - centre, axis=1)), 30.0)

  def test_medial_topology_mask_fills_only_ignored_pinhole(self):
    mask = np.zeros((140, 300), dtype=np.uint8)
    cv2.rectangle(mask, (20, 20), (280, 120), 255, -1)
    cv2.circle(mask, (90, 70), 2, 0, -1)
    cv2.circle(mask, (220, 70), 14, 0, -1)

    topology = medial_topology_mask(mask, paw_contours(mask, stroke_width=80.0))

    self.assertEqual(mask[70, 90], 0)       # measured ink is untouched
    self.assertEqual(topology[70, 90], 255)  # scan pinhole is filled
    self.assertEqual(topology[70, 220], 0)   # real counter remains a hole

  def setUp(self):
    self.skeleton_templates, self.elastic_templates = load_template_definitions()
    self.template = SkeletonTemplate(
        name="zigzag", letters=("ك",), forms=("initial",),
        steps=(TemplateStep("x-right", 0.4, 0.25, 0.5),
               TemplateStep("x-left", 0.75, 0.25, 0.5)))

  def event(self, s, kind, strength=1.0):
    return SkeletonEvent(s=s, kind=kind, strength=strength, xy=(s, 0.0))

  def test_ordered_match_scores_positive(self):
    score, event_ids = match_skeleton_template(
        [self.event(40.0, "x-right"), self.event(75.0, "x-left")],
        0.0, 100.0, self.template)
    self.assertGreater(score, 0.0)
    self.assertEqual(event_ids, (0, 1))

  def test_reversed_events_do_not_match(self):
    score, _ = match_skeleton_template(
        [self.event(40.0, "x-left"), self.event(75.0, "x-right")],
        0.0, 100.0, self.template)
    self.assertLess(score, 0.0)

  def test_json_round_trip(self):
    payload = {"templates": [skeleton_template_to_json(self.template)]}
    self.assertEqual(skeleton_templates_from_json(payload), (self.template,))

  def test_discoverable_json_loads_all_template_types(self):
    with open(DEFAULT_TEMPLATE_PATH, encoding="utf-8") as fh:
      payload = json.load(fh)
    self.assertEqual(skeleton_templates_from_json(payload),
                     self.skeleton_templates)
    self.assertEqual(elastic_body_templates_from_json(payload),
                     self.elastic_templates)
    self.assertTrue({"lam-alef-separated-stems", "kaf-alef-zigzag",
                     "kaf-alef-separated-stems", "meem-initial-head",
                     "heh-medial-two-loop", "teh-medial-tooth",
                     "dal-final-head", "seen-initial-three-tooth",
                     "waw-final-head", "teh-final-tooth", "yeh-initial-entry",
                     "noon-final-after-yeh-bowl"}.issubset(
                        {template.name for template in self.elastic_templates}))

  def test_elastic_boundary_rules_round_trip_and_omissions_use_defaults(self):
    teh = next(template for template in self.elastic_templates
               if template.name == "teh-medial-tooth")
    self.assertEqual(teh.entry, BodyBoundarySpec(
        method="event-offset", anchor_event="peak",
        offset_stroke_widths=-1.0))
    self.assertEqual(teh.exit, BodyBoundarySpec(
        method="event-offset", anchor_event="valley",
        offset_stroke_widths=1.0))
    payload = {"elasticBodyTemplates": [
        elastic_body_template_to_json(teh)]}
    self.assertEqual(elastic_body_templates_from_json(payload), (teh,))

    defaulted = elastic_body_templates_from_json({
        "elasticBodyTemplates": [{
            "name": "plain", "letters": ["ب"], "forms": ["initial"],
            "requiredEvents": [{"kind": "branch"}]
        }]})[0]
    self.assertEqual(defaulted.entry, BodyBoundarySpec())
    self.assertEqual(defaulted.exit, BodyBoundarySpec())
    serialized = elastic_body_template_to_json(defaulted)
    self.assertNotIn("entry", serialized)
    self.assertNotIn("exit", serialized)

  def test_elastic_branch_bounds_round_trip_and_validate_body_events(self):
    bounded = ElasticBodyTemplate(
        name="bounded-branches", letters=("م",), forms=("final",),
        required_events=(("branch", 0.0),),
        min_branches=2, max_branches=2)
    payload = {"elasticBodyTemplates": [
        elastic_body_template_to_json(bounded)]}

    self.assertEqual(payload["elasticBodyTemplates"][0]["minBranches"], 2)
    self.assertEqual(payload["elasticBodyTemplates"][0]["maxBranches"], 2)
    self.assertEqual(elastic_body_templates_from_json(payload), (bounded,))

    events = [
        self.event(0.0, "branch", 0.01),
        self.event(10.0, "branch", 0.01),
        self.event(10.0, "high-branch", 10.0),
        self.event(20.0, "branch", 10.0),
        self.event(30.0, "branch", 10.0),
    ]
    # Event strength and graph/pruned source do not change whether a branch
    # exists. Closed body bounds include their entry and terminal events.
    self.assertEqual(branch_count_in_span(events, 5.0, 25.0), 2)
    self.assertTrue(template_branch_constraints_match(
        bounded, events, 5.0, 25.0))
    self.assertFalse(template_branch_constraints_match(
        bounded, events, 0.0, 30.0))

    with self.assertRaisesRegex(ValueError, "branch bounds"):
      elastic_body_templates_from_json({"elasticBodyTemplates": [{
          "name": "bad", "letters": ["م"], "forms": ["final"],
          "requiredEvents": [{"kind": "branch"}],
          "minBranches": 3, "maxBranches": 2,
      }]})

  def test_interval_only_template_round_trips_and_cannot_drive_ordering(self):
    interval_only = ElasticBodyTemplate(
        name="final-tail-validator", letters=("ر",), forms=("final",),
        required_events=(("valley", 0.5),), ordered_match=False)
    payload = {"elasticBodyTemplates": [
        elastic_body_template_to_json(interval_only)]}
    self.assertEqual(
        elastic_body_templates_from_json(payload), (interval_only,))
    self.assertFalse(payload["elasticBodyTemplates"][0]["orderedMatch"])
    self.assertIsNone(_ordered_single_body_matches(
        [self.event(10.0, "valley")], ("ر",), ("final",),
        (interval_only,)))

  def test_complete_paths_prefer_specific_template_after_bounded_search(self):
    specific = ElasticBodyTemplate(
        name="meem-medi", letters=("م",), forms=("medial",),
        required_events=(("branch", 2.0), ("branch", 2.0)))
    fallback = ElasticBodyTemplate(
        name="meem-medi-small", letters=("م",), forms=("medial",),
        required_events=(("branch", 0.8),))
    templates = (specific, fallback)
    events = [self.event(10.0, "branch", 3.0),
              self.event(20.0, "branch", 3.0)]

    paths = _ordered_single_body_match_options(
        events, ("م",), ("medial",), templates)
    # The beam still explores the path which finishes at event 0 first.
    self.assertEqual(paths[0][0][0].name, "meem-medi-small")
    paths.sort(key=lambda path: _complete_ordered_path_selection_key(
        path, {id(template): i for i, template in enumerate(templates)}))

    self.assertEqual(paths[0][0][0].name, "meem-medi")
    self.assertEqual(paths[0][0][1].events, (0, 1))

  def test_elastic_template_priority_round_trips(self):
    preferred = ElasticBodyTemplate(
        name="preferred", letters=("م",), forms=("medial",),
        required_events=(("branch", 0.8),), priority=7)
    payload = {"elasticBodyTemplates": [
        elastic_body_template_to_json(preferred)]}

    self.assertEqual(payload["elasticBodyTemplates"][0]["priority"], 7)
    self.assertEqual(
        elastic_body_templates_from_json(payload), (preferred,))

  def test_before_and_after_events_are_reusable_non_owning_context(self):
    first = ElasticBodyTemplate(
        name="first", letters=("ب",), forms=("initial",),
        required_events=(("branch", 0.5),),
        after_events=(("valley", 0.5),))
    second = ElasticBodyTemplate(
        name="second", letters=("ت",), forms=("final",),
        required_events=(("valley", 0.5),),
        before_events=(("branch", 0.5),))
    payload = {"elasticBodyTemplates": [
        elastic_body_template_to_json(first),
        elastic_body_template_to_json(second)]}
    self.assertEqual(
        elastic_body_templates_from_json(payload), (first, second))
    self.assertEqual(payload["elasticBodyTemplates"][0]["afterEvents"], [
        {"kind": "valley", "minStrength": 0.5}])
    self.assertEqual(payload["elasticBodyTemplates"][1]["beforeEvents"], [
        {"kind": "branch", "minStrength": 0.5}])

    events = [self.event(10.0, "branch"), self.event(20.0, "valley")]
    options = _ordered_single_body_match_options(
        events, ("ب", "ت"), ("initial", "final"), (first, second))
    self.assertTrue(options)
    first_assignment = options[0][0][1]
    second_assignment = options[0][1][1]
    self.assertEqual(first_assignment.events, (0,))
    self.assertEqual(first_assignment.after_events, (1,))
    self.assertEqual(second_assignment.events, (1,))
    self.assertEqual(second_assignment.before_events, (0,))

    matches = _ordered_single_body_matches(
        events, ("ب", "ت"), ("initial", "final"), (first, second))
    self.assertEqual([match[1] for match in matches], [(0,), (1,)])

    with self.assertRaisesRegex(ValueError, "owned required event"):
      elastic_body_templates_from_json({"elasticBodyTemplates": [{
          "name": "context-anchor", "letters": ["ب"],
          "forms": ["initial"],
          "beforeEvents": [{"kind": "branch"}],
          "requiredEvents": [{"kind": "valley"}],
          "entry": {"method": "event-offset", "anchorEvent": "branch"},
      }]})

  def test_event_gap_constraints_round_trip_and_select_valid_spacing(self):
    payload = {"elasticBodyTemplates": [{
        "name": "spaced", "letters": ["ه"], "forms": ["medial"],
        "requiredEvents": [
            {"kind": "branch", "minStrength": 0.5},
            {"kind": "valley", "minStrength": 0.5,
             "gapFromPrevious": {
                 "minStrokeWidths": 1.0, "maxStrokeWidths": 1.5}},
        ],
    }]}
    template = elastic_body_templates_from_json(payload)[0]
    self.assertEqual(elastic_body_template_to_json(template),
                     payload["elasticBodyTemplates"][0] | {
                         "previousLetters": [], "previousForms": []})

    events = [self.event(0.0, "branch"),
              self.event(5.0, "valley"),
              self.event(12.0, "valley")]
    options = _ordered_single_body_match_options(
        events, ("ه",), ("medial",), (template,), stroke_width=10.0)
    self.assertTrue(options)
    self.assertEqual(options[0][0][1].events, (0, 2))
    score, event_ids = match_elastic_body_template(
        events, -1.0, 13.0, template, stroke_width=10.0)
    self.assertGreater(score, 0.0)
    self.assertEqual(event_ids, (0, 2))

    contextual = elastic_body_templates_from_json({
        "elasticBodyTemplates": [{
            "name": "context-spaced", "letters": ["ه"],
            "forms": ["medial"],
            "beforeEvents": [{"kind": "branch", "minStrength": 0.5}],
            "requiredEvents": [{
                "kind": "valley", "minStrength": 0.5,
                "gapFromPrevious": {
                    "minStrokeWidths": 1.0, "maxStrokeWidths": 1.5}}],
        }]})[0]
    options = _ordered_single_body_match_options(
        events, ("ه",), ("medial",), (contextual,), stroke_width=10.0)
    self.assertEqual(options[0][0][1].before_events, (0,))
    self.assertEqual(options[0][0][1].events, (2,))

  def test_consecutive_forbids_skipping_next_eligible_same_kind_event(self):
    payload = {"elasticBodyTemplates": [{
        "name": "adjacent-branches", "letters": ["ه"],
        "forms": ["medial"],
        "requiredEvents": [
            {"kind": "branch", "minStrength": 0.5},
            {"kind": "branch", "minStrength": 0.5,
             "consecutive": True},
        ],
    }]}
    template = elastic_body_templates_from_json(payload)[0]
    events = [self.event(0.0, "branch"), self.event(5.0, "branch"),
              self.event(12.0, "branch")]
    assignments = [path[0][1].events for path in
                   _ordered_single_body_match_options(
                       events, ("ه",), ("medial",), (template,))]
    self.assertIn((0, 1), assignments)
    self.assertIn((1, 2), assignments)
    self.assertNotIn((0, 2), assignments)

    # A same-kind event below this specification's minimum strength is not
    # eligible and therefore does not block the later event.
    events[1] = self.event(5.0, "branch", 0.4)
    assignments = [path[0][1].events for path in
                   _ordered_single_body_match_options(
                       events, ("ه",), ("medial",), (template,))]
    self.assertIn((0, 2), assignments)

  def test_adjacent_forbids_every_intermediate_event_kind(self):
    payload = {"elasticBodyTemplates": [{
        "name": "branch-to-tail", "letters": ["م"], "forms": ["final"],
        "requiredEvents": [
            {"kind": "branch", "minStrength": 0.5},
            {"kind": "continuation", "minStrength": 4.0,
             "adjacent": True},
        ],
    }]}
    template = elastic_body_templates_from_json(payload)[0]
    serialized = elastic_body_template_to_json(template)
    self.assertTrue(serialized["requiredEvents"][1]["adjacent"])

    direct = [self.event(0.0, "branch"),
              self.event(10.0, "continuation", 4.5)]
    self.assertTrue(_ordered_single_body_match_options(
        direct, ("م",), ("final",), (template,)))

    interrupted = [self.event(0.0, "branch"),
                   self.event(5.0, "valley", 0.01),
                   self.event(10.0, "continuation", 4.5)]
    self.assertFalse(_ordered_single_body_match_options(
        interrupted, ("م",), ("final",), (template,)))

  def test_event_gap_schema_rejects_invalid_bounds_and_first_constraint(self):
    base = {
        "name": "bad", "letters": ["ه"], "forms": ["medial"],
        "requiredEvents": [{"kind": "branch"}, {"kind": "valley"}],
    }
    invalid_gap = json.loads(json.dumps(base))
    invalid_gap["requiredEvents"][1]["gapFromPrevious"] = {
        "minStrokeWidths": 2.0, "maxStrokeWidths": 1.0}
    with self.assertRaisesRegex(ValueError, "event gap bounds"):
      elastic_body_templates_from_json(
          {"elasticBodyTemplates": [invalid_gap]})

    invalid_first = json.loads(json.dumps(base))
    invalid_first["requiredEvents"][0]["consecutive"] = True
    with self.assertRaisesRegex(ValueError, "first template event"):
      elastic_body_templates_from_json(
          {"elasticBodyTemplates": [invalid_first]})

    invalid_first = json.loads(json.dumps(base))
    invalid_first["requiredEvents"][0]["adjacent"] = True
    with self.assertRaisesRegex(ValueError, "first template event"):
      elastic_body_templates_from_json(
          {"elasticBodyTemplates": [invalid_first]})

    misspelled = json.loads(json.dumps(base))
    misspelled["requiredEvents"][1]["adjacency"] = True
    with self.assertRaisesRegex(ValueError, "unsupported field.*adjacency"):
      elastic_body_templates_from_json(
          {"elasticBodyTemplates": [misspelled]})

  def test_terminal_ascent_is_a_real_event_not_an_empty_body(self):
    grid = np.arange(101.0)
    x = np.where(grid <= 20.0, grid, 20.0 + 0.2 * (grid - 20.0))
    y = np.where(grid <= 20.0, 100.0, 100.0 - 1.25 * (grid - 20.0))
    vertices = np.column_stack([x, y])
    medial = Medial(
        vertices=vertices, radii=np.ones_like(grid),
        edges=[(i, i + 1) for i in range(len(grid) - 1)],
        generators=np.empty((0, 2)), spine=list(range(len(grid))),
        spine_s=grid, loops=0)
    events = skeleton_events(
        {"grid": grid, "ds": 1.0, "x": x, "y": y}, medial,
        baseline_y=100.0, stroke_width=10.0,
        peaks=np.empty(0), valleys=np.array([20.0]))
    ascent = [event for event in events
              if event.kind == "terminal-ascent"]
    self.assertEqual(len(ascent), 1)
    self.assertEqual(ascent[0].s, 21.0)
    self.assertGreater(ascent[0].strength, 1.5)

  def test_terminal_ascent_anchor_waits_for_45_degree_tangent(self):
    grid = np.arange(101.0)
    x = np.where(grid <= 20.0, grid, 20.0 + 0.2 * (grid - 20.0))
    y = np.where(grid <= 20.0, 100.0, 100.0 - 1.25 * (grid - 20.0))
    tangents = np.tile([0.95, -0.31], (len(grid), 1))
    tangents[40:] = [0.5, -0.866]
    vertices = np.column_stack([x, y])
    medial = Medial(
        vertices=vertices, radii=np.ones_like(grid),
        edges=[(i, i + 1) for i in range(len(grid) - 1)],
        generators=np.empty((0, 2)), spine=list(range(len(grid))),
        spine_s=grid, loops=0)

    events = skeleton_events(
        {"grid": grid, "ds": 1.0, "x": x, "y": y,
         "tangents": tangents},
        medial, baseline_y=100.0, stroke_width=10.0,
        peaks=np.empty(0), valleys=np.array([20.0]))
    ascent = [event for event in events
              if event.kind == "terminal-ascent"]

    self.assertEqual(len(ascent), 1)
    self.assertEqual(ascent[0].s, 40.0)

  def test_sad_dad_templates_own_the_terminal_tooth(self):
    initial = next(template for template in self.elastic_templates
                   if template.name == "dad-initial-bowl")
    initial_events = [
        self.event(0.0, "high-branch", 2.0),
        self.event(10.0, "valley"),
        self.event(20.0, "peak"), self.event(30.0, "valley"),
        self.event(40.0, "peak"), self.event(50.0, "valley")]
    score, event_ids = match_elastic_body_template(
        initial_events, 0.0, 60.0, initial)
    self.assertGreater(score, 0.0)
    self.assertEqual(event_ids[-2:], (4, 5))
    incomplete, _ = match_elastic_body_template(
        initial_events[:-2], 0.0, 60.0, initial)
    self.assertEqual(incomplete, 0.0)
    self.assertEqual(initial.exit, BodyBoundarySpec(
        method="event-offset", anchor_event="valley", occurrence="last",
        offset_stroke_widths=1.0))

    medial_events = [
        self.event(0.0, "high-branch", 2.0),
        self.event(10.0, "peak"), self.event(20.0, "valley"),
        self.event(30.0, "peak"), self.event(40.0, "valley"),
        self.event(50.0, "peak"), self.event(60.0, "valley")]
    matches = _ordered_single_body_matches(
        medial_events, ("ص",), ("medial",), self.elastic_templates)
    self.assertIsNotNone(matches)
    self.assertEqual(matches[0][0].name,
                     "sad-dad-medial-bowl-extra-cycle")
    self.assertEqual(matches[0][1][-2:], (5, 6))

  def test_initial_heh_uses_one_complete_two_loop_template(self):
    names = {template.name for template in self.elastic_templates}
    self.assertIn("heh-initial-two-loop", names)
    self.assertNotIn("heh-initial-double-head", names)
    self.assertNotIn("heh-initial-high-head", names)
    template = next(template for template in self.elastic_templates
                    if template.name == "heh-initial-two-loop")
    self.assertEqual(template.required_events, (
        ("high-branch", 1.0), ("valley", 0.5), ("peak", 0.5)))
    self.assertEqual(template.after_events, (("valley", 0.5),))
    self.assertEqual(template.exit, BodyBoundarySpec(
        method="width-or-event-exit", anchor_event="peak",
        occurrence="last", offset_stroke_widths=1.0))
    # A double-high branch is additional graph evidence, not a distinct
    # allograph.  Both medial-axis encodings share the complete loop sequence.
    ordinary = [
        self.event(0.0, "high-branch", 1.5),
        self.event(10.0, "valley"), self.event(20.0, "peak"),
        self.event(30.0, "valley")]
    doubled = [
        self.event(0.0, "double-high-branch", 2.5),
        self.event(0.0, "high-branch", 2.5),
        self.event(10.0, "valley"), self.event(20.0, "peak"),
        self.event(30.0, "valley")]
    ordinary_score, _ = match_elastic_body_template(
        ordinary, 0.0, 25.0, template)
    doubled_score, doubled_ids = match_elastic_body_template(
        doubled, 0.0, 25.0, template)
    self.assertGreater(ordinary_score, 0.0)
    self.assertGreater(doubled_score, 0.0)
    # The closing valley is reusable after-context, not Heh-owned topology.
    self.assertEqual(doubled_ids, (1, 2, 3))

  def test_medial_heh_has_only_two_visual_templates(self):
    templates = [template for template in self.elastic_templates
                 if template.letters == ("ه",)
                 and template.forms == ("medial",)]
    self.assertEqual([template.name for template in templates], [
        "heh-medial-two-loop", "heh-medial-open"])

    closed, opened = templates
    self.assertEqual(closed.required_events, (
        ("high-branch", 1.0), ("valley", 0.5)))
    self.assertEqual(closed.after_events, ())
    self.assertEqual(closed.exit_owns_join_next_letters, ())
    self.assertEqual((closed.min_loops, closed.max_loops), (2, 2))
    self.assertEqual((opened.min_loops, opened.max_loops), (0, 0))
    self.assertEqual(closed.entry, BodyBoundarySpec(
        method="event-offset", anchor_event="high-branch",
        offset_stroke_widths=-1.5))
    self.assertEqual(closed.exit, BodyBoundarySpec(
        method="observed-event-offset", anchor_event="valley",
        occurrence="last", offset_stroke_widths=1.3, enforce=True))
    self.assertEqual(opened.required_events, (
        ("branch", 1.0), ("branch", 0.5)))
    self.assertEqual(opened.exit_owns_join_next_letters, ())

    closed_events = [self.event(10.0, "high-branch", 2.0),
                     self.event(20.0, "valley")]
    open_events = [self.event(10.0, "branch", 2.0),
                   self.event(20.0, "branch")]
    two_loops = [LoopEvidence(15.0, 10.0, 20.0, 2, (15.0, 0.0))]
    one_loop = [LoopEvidence(15.0, 10.0, 20.0, 1, (15.0, 0.0))]
    self.assertGreater(match_elastic_body_template(
        closed_events, 0.0, 30.0, closed, two_loops)[0], 0.0)
    self.assertEqual(match_elastic_body_template(
        closed_events, 0.0, 30.0, closed, one_loop)[0], 0.0)
    self.assertGreater(match_elastic_body_template(
        open_events, 0.0, 40.0, opened, [])[0], 0.0)
    self.assertEqual(match_elastic_body_template(
        open_events, 0.0, 40.0, opened, one_loop)[0], 0.0)
    self.assertEqual(match_elastic_body_template(
        open_events, 0.0, 40.0, opened, two_loops)[0], 0.0)

  def test_initial_ayn_ends_half_stroke_after_its_branch(self):
    template = next(template for template in self.elastic_templates
                    if template.name == "ayn-family-initial-head")
    self.assertEqual(template.exit, BodyBoundarySpec(
        method="event-offset", anchor_event="high-branch",
        offset_stroke_widths=0.5, enforce=True))

  def test_final_reh_family_starts_half_stroke_before_its_branch(self):
    templates = [template for template in self.elastic_templates
                 if template.name in {
                     "reh-final-tail", "reh-final-valley-tail",
                     "reh-final-terminal-valley"}]
    self.assertEqual(len(templates), 3)
    expected = BodyBoundarySpec(
        method="observed-event-offset", anchor_event="branch",
        occurrence="last", offset_stroke_widths=-0.5, enforce=True)
    self.assertTrue(all(template.entry == expected for template in templates))
    hah = next(template for template in self.elastic_templates
               if template.name == "hah-family-initial-head")
    self.assertEqual(hah.required_events, (
        ("peak", 0.5), ("branch", 0.5)))
    self.assertEqual(hah.after_events, (("valley", 0.5),))

  def test_character_class_template_covers_dotted_variants(self):
    family = next(template for template in self.skeleton_templates
                  if template.name == "beh-hah-family")
    for letters in (("ت", "ج"), ("ب", "ح"), ("ي", "خ")):
      self.assertTrue(_template_applies(
          family, letters, ("initial", "medial"), 0))
    self.assertFalse(_template_applies(
        family, ("ك", "ج"), ("initial", "medial"), 0))

  def test_elastic_body_template_has_no_position_or_width_target(self):
    family = next(template for template in self.elastic_templates
                  if template.name == "beh-hah-core")
    narrow, narrow_ids = match_elastic_body_template(
        [self.event(20.0, "high-branch", 3.0)], 0.0, 50.0, family)
    stretched, stretched_ids = match_elastic_body_template(
        [self.event(160.0, "high-branch", 3.0)], 0.0, 200.0, family)
    self.assertEqual(narrow, stretched)
    self.assertEqual(narrow_ids, stretched_ids)

  def test_separated_stems_include_endpoint_but_require_distinct_positions(self):
    template = next(template for template in self.elastic_templates
                    if template.name == "lam-alef-separated-stems")
    score, _ = match_elastic_body_template(
        [self.event(0.0, "high-branch", 2.0),
         self.event(50.0, "high-branch", 2.0)], 0.0, 100.0, template)
    duplicate_score, _ = match_elastic_body_template(
        [self.event(0.0, "high-branch", 2.0),
         self.event(0.0, "high-branch", 2.0)], 0.0, 100.0, template)
    self.assertGreater(score, 0.0)
    self.assertEqual(duplicate_score, 0.0)

  def test_forced_alignment_derives_cut_from_elastic_join(self):
    grid = np.arange(101.0)
    y = np.zeros_like(grid)
    y[30:71] = -np.arange(41.0)
    y[71:] = -40.0 + np.arange(30.0)
    radius = 0.5 + 0.004 * np.abs(grid - 60.0)
    owned = np.ones_like(grid)
    tangents = np.column_stack([np.ones_like(grid), np.zeros_like(grid)])
    chords = [((float(i), -1.0), (float(i), 1.0)) for i in grid]
    profile = {
        "grid": grid, "total_s": 101.0, "ds": 1.0,
        "x": grid, "y": y, "r": radius,
        "cut_len": owned, "core_len": owned, "owned_len": owned,
        "foreign_len": np.zeros_like(grid), "global_cut_len": owned,
        "n_runs": np.ones_like(grid, dtype=int),
        "global_runs": np.ones_like(grid, dtype=int),
        "tangents": tangents, "chords": chords,
    }
    events = [self.event(10.0, "high-branch", 3.0),
              self.event(70.0, "peak"),
              self.event(90.0, "valley")]
    aligned = force_align_bodies_and_joins(
        profile, events, ("ب", "ح", "ت"),
        ("initial", "medial", "final"), 0.0, 1.0, "next-start",
        self.elastic_templates)
    self.assertFalse(aligned.reason)
    self.assertEqual([(group.letter_start, group.letter_end)
                      for group in aligned.groups], [(0, 2), (2, 3)])
    self.assertEqual(aligned.groups[-1].b_s, 101.0)
    self.assertEqual(
        [match.template for match in aligned.shape_matches],
        [group.template for group in aligned.groups])
    self.assertTrue(all(match.events for match in aligned.shape_matches))
    physical = [join for join in aligned.joins if join.kind != "fused"]
    self.assertEqual(len(physical), 1)
    self.assertGreater(physical[0].end.s, physical[0].start.s)
    self.assertEqual(aligned.cuts[0].s, physical[0].end.s)

    family = next(template for template in self.elastic_templates
                  if template.name == "beh-hah-core")
    duplicate = ElasticBodyTemplate(
        "beh-hah-alternate", family.letters, family.forms,
        family.required_events)
    with_duplicate = force_align_bodies_and_joins(
        profile, events, ("ب", "ح", "ت"),
        ("initial", "medial", "final"), 0.0, 1.0, "next-start",
        self.elastic_templates + (duplicate,))
    self.assertGreater(with_duplicate.margin, 0.0)

    current_end = force_align_bodies_and_joins(
        profile, events, ("ب", "ح", "ت"),
        ("initial", "medial", "final"), 0.0, 1.0, "current-end",
        self.elastic_templates)
    self.assertEqual(current_end.cuts[0].s, physical[0].start.s)
    no_cut = force_align_bodies_and_joins(
        profile, events, ("ب", "ح", "ت"),
        ("initial", "medial", "final"), 0.0, 1.0, "none",
        self.elastic_templates)
    self.assertEqual(no_cut.cuts, [])

  def test_two_loop_heh_ends_1_3_strokes_after_required_valley(self):
    grid = np.arange(101.0)
    profile = {
        "grid": grid, "ds": 1.0, "owned_len": np.ones_like(grid)}
    events = [
        self.event(40.0, "high-branch", 2.0),
        self.event(50.0, "valley"), self.event(60.0, "branch", 2.0),
        self.event(70.0, "valley"),
    ]
    template = next(template for template in self.elastic_templates
                    if template.name == "heh-medial-two-loop")
    index = _template_body_exit_index(
        profile, events, (0, 1), template.exit, 100, 10.0)

    self.assertEqual(index, 63)

  def test_enforced_heh_exit_advances_following_topology_match(self):
    grid = np.arange(101.0)
    owned = np.ones_like(grid)
    tangents = np.column_stack([np.ones_like(grid), np.zeros_like(grid)])
    profile = {
        "grid": grid, "total_s": 100.0, "ds": 1.0,
        "x": grid, "y": np.zeros_like(grid), "r": owned,
        "cut_len": owned, "core_len": owned, "owned_len": owned,
        "foreign_len": np.zeros_like(grid), "global_cut_len": owned,
        "n_runs": np.ones_like(grid, dtype=int),
        "global_runs": np.ones_like(grid, dtype=int),
        "tangents": tangents,
        "chords": [((float(i), -1.0), (float(i), 1.0)) for i in grid],
        "loop_evidence": [
            LoopEvidence(25.0, 20.0, 30.0, 2, (25.0, 0.0))],
    }
    events = [
        self.event(20.0, "high-branch", 2.0),
        self.event(30.0, "valley"),
        self.event(35.0, "branch"),
        self.event(60.0, "branch"),
    ]
    aligned = force_align_bodies_and_joins(
        profile, events, ("ه", "م"), ("medial", "final"),
        0.0, 10.0, "next-start", self.elastic_templates)

    self.assertFalse(aligned.reason)
    self.assertEqual([group.template for group in aligned.groups], [
        "heh-medial-two-loop", "meem-final-tail"])
    self.assertEqual(aligned.groups[0].b_s, 43.0)
    self.assertEqual(aligned.groups[1].a_s, 50.0)
    self.assertEqual(aligned.joins[0].kind, "template-event-entry")

  def test_medial_loops_are_localized_to_body_spans(self):
    vertices = np.array([
        [0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0],
        [40.0, 0.0], [50.0, 0.0], [60.0, 0.0],
        [15.0, 10.0], [45.0, 10.0]])
    edges = [(i, i + 1) for i in range(6)]
    edges += [(1, 7), (7, 2), (4, 8), (8, 5)]
    medial = Medial(
        vertices=vertices, radii=np.ones(len(vertices)), edges=edges,
        generators=np.empty((0, 2)), spine=list(range(7)),
        spine_s=np.arange(0.0, 70.0, 10.0), loops=2)

    evidence = medial_loop_evidence(medial)

    self.assertEqual(sum(item.count for item in evidence), medial.loops)
    self.assertEqual(loop_count_in_span(evidence, 0.0, 30.0), 1)
    self.assertEqual(loop_count_in_span(evidence, 30.0, 60.0), 1)

    plain = ElasticBodyTemplate(
        "plain", ("ب",), ("initial",), (("branch", 0.0),))
    closed = next(template for template in self.elastic_templates
                  if template.name == "heh-medial-two-loop")
    opened = next(template for template in self.elastic_templates
                  if template.name == "heh-medial-open")
    self.assertTrue(loop_partition_constraints_match(
        (plain, plain), ((0.0, 30.0), (30.0, 60.0)), evidence))
    two_loop_interval = [
        LoopEvidence(15.0, 10.0, 20.0, 2, (15.0, 0.0))]
    self.assertTrue(loop_partition_constraints_match(
        (plain, closed, plain),
        ((0.0, 5.0), (8.0, 22.0), (25.0, 60.0)),
        two_loop_interval))
    self.assertFalse(loop_partition_constraints_match(
        (plain, opened, plain),
        ((0.0, 5.0), (12.0, 19.0), (25.0, 60.0)),
        two_loop_interval))

  def test_loop_side_has_full_strength_at_both_spine_attachments(self):
    vertices = np.array([
        [0.0, 0.0], [10.0, 0.0], [20.0, 0.0],
        [0.0, 10.0], [10.0, 10.0], [20.0, 10.0]])
    medial = Medial(
        vertices=vertices, radii=np.ones(len(vertices)),
        edges=[(0, 1), (1, 2), (0, 3), (3, 4), (4, 5), (5, 2)],
        generators=np.empty((0, 2)), spine=[0, 1, 2],
        spine_s=np.array([0.0, 10.0, 20.0]), loops=1)

    paths = skeleton_branch_paths(medial)
    loop_paths = sorted(
        (path for path in paths if path.s in {0.0, 20.0}),
        key=lambda path: path.s)

    self.assertEqual([path.s for path in loop_paths], [0.0, 20.0])
    self.assertEqual([path.length for path in loop_paths], [40.0, 40.0])
    np.testing.assert_array_equal(loop_paths[0].points,
                                  loop_paths[1].points[::-1])

  def test_degree_two_spine_endpoint_is_classified_as_continuation(self):
    vertices = np.array([
        [0.0, 0.0], [10.0, 0.0], [20.0, 0.0],
        [30.0, 1.0], [40.0, 2.0]])
    medial = Medial(
        vertices=vertices, radii=np.ones(len(vertices)),
        edges=[(0, 1), (1, 2), (2, 3), (3, 4)],
        generators=np.empty((0, 2)), spine=[0, 1, 2],
        spine_s=np.array([0.0, 10.0, 20.0]), loops=0)

    paths = skeleton_branch_paths(medial)
    self.assertEqual(len(paths), 1)
    self.assertEqual(paths[0].role, "continuation")
    events = skeleton_events(
        {"grid": np.array([0.0, 10.0, 20.0]), "ds": 10.0,
         "x": vertices[:3, 0], "y": vertices[:3, 1]},
        medial, baseline_y=0.0, stroke_width=1.0,
        peaks=np.empty(0), valleys=np.empty(0))
    # The two spine ends always emit their own markers; what this test is
    # about is that the off-spine path became a continuation and not a branch.
    ends = {"start", "end", "start-height", "end-height"}
    kinds = [event.kind for event in events if event.kind not in ends]
    self.assertEqual(kinds, ["continuation"])

    # Adding another off-spine edge makes the endpoint a genuine fork.
    forked = Medial(
        vertices=np.vstack([vertices, [30.0, -5.0]]),
        radii=np.ones(len(vertices) + 1),
        edges=[(0, 1), (1, 2), (2, 3), (3, 4), (2, 5)],
        generators=np.empty((0, 2)), spine=[0, 1, 2],
        spine_s=np.array([0.0, 10.0, 20.0]), loops=0)
    fork_paths = skeleton_branch_paths(forked)
    self.assertEqual(len(fork_paths), 2)
    self.assertTrue(all(path.role == "branch" for path in fork_paths))

  def test_matched_events_own_loop_before_geometric_overlap(self):
    plain = ElasticBodyTemplate(
        "plain", ("ب",), ("medial",), (("peak", 0.0),))
    opened = ElasticBodyTemplate(
        "open", ("ه",), ("medial",), (("branch", 0.0),), max_loops=0)
    events = [
        self.event(12.0, "peak"), self.event(20.0, "valley"),
        self.event(27.0, "branch"), self.event(32.0, "branch"),
        self.event(40.0, "branch")]
    spans = ((0.0, 25.0), (25.0, 35.0), (35.0, 60.0))
    evidence = [LoopEvidence(25.0, 10.0, 40.0, 2, (25.0, 0.0))]

    # Geometric overlap alone gives the loop to the first body and recreates
    # the bad l00_w04 hypothesis.  Both that unconstrained body and the Heh
    # match two events in the loop, but Heh explicitly requires zero loops;
    # its loop contract takes precedence and rejects the hypothesis.
    self.assertTrue(loop_partition_constraints_match(
        (plain, opened, plain), spans, evidence))
    self.assertFalse(loop_partition_constraints_match(
        (plain, opened, plain), spans, evidence, events,
        ((0, 1), (2, 3), (4,))))

  def test_strict_body_partitions_reject_anonymous_singletons(self):
    self.assertEqual(_body_partitions(
        ("ل", "ذ"), ("initial", "final"), ()), [])
    covered = _body_partitions(
        ("س", "و"), ("initial", "final"), self.elastic_templates)
    self.assertTrue(covered)
    self.assertTrue(all(template is not None
                        for partition in covered
                        for _, _, template in partition))

  def test_float32_endpoint_pixels_remain_in_final_body(self):
    mask = np.full((1, 2), 255, dtype=np.uint8)
    endpoint = np.nextafter(np.float32(100.0), np.float32(np.inf))
    pixel_s = np.array([[0.0, endpoint]], dtype=np.float32)
    labels = attribute_body_join_pixels(
        mask, None, [SegmentGroup(0, 2, 0.0, 100.0, "fused")], [], 2,
        pixel_s=pixel_s)
    self.assertEqual(labels.tolist(), [[-2, -2]])

  def test_l14_w06_single_body_events_are_owned_in_letter_order(self):
    events = [
        self.event(0.0, "branch", 2.7),
        self.event(167.0, "peak"),
        self.event(214.0, "valley"),
        self.event(689.0, "branch", 5.8),
        self.event(763.0, "branch", 5.7),
        self.event(764.0, "peak"),
        self.event(811.0, "valley"),
        self.event(1260.0, "peak"),
        self.event(1312.0, "valley"),
        self.event(1687.0, "high-branch", 4.0),
    ]
    matches = _ordered_single_body_matches(
        events, ("م", "ه", "ت", "د"),
        ("initial", "medial", "medial", "final"),
        self.elastic_templates)
    self.assertIsNotNone(matches)
    self.assertEqual([match[0].name for match in matches], [
        "meem-initial-head", "heh-medial-open",
        "teh-medial-tooth", "dal-final-head"])
    self.assertEqual([match[1] for match in matches], [
        (0, 1, 2), (3, 4), (5, 6), (9,)])

  def test_final_noon_body_variant_requires_initial_yeh_context(self):
    template = next(template for template in self.elastic_templates
                    if template.name == "noon-final-after-yeh-bowl")
    self.assertTrue(_elastic_template_applies(
        template, ("ي", "ن"), ("initial", "final"), 1))
    self.assertFalse(_elastic_template_applies(
        template, ("م", "ن"), ("initial", "final"), 1))

  def test_final_meem_starts_one_stroke_before_its_entry_branch(self):
    template = next(template for template in self.elastic_templates
                    if template.name == "meem-final-tail")
    self.assertEqual(template.entry, BodyBoundarySpec(
        method="event-offset", anchor_event="branch", occurrence="first",
        offset_stroke_widths=-1.0))
    events = [
        self.event(0.0, "valley"), self.event(100.0, "branch", 1.0),
        self.event(120.0, "peak"), self.event(150.0, "valley"),
        self.event(180.0, "high-branch")]
    score, event_ids = match_elastic_body_template(
        events, 1.0, 200.0, template)
    self.assertGreater(score, 0.0)
    self.assertEqual(event_ids, (1,))
    grid = np.arange(201.0)
    boundary = _template_body_boundary_index(
        {"grid": grid, "ds": 1.0, "owned_len": np.ones_like(grid)},
        events, 0, event_ids, template.entry, 10.0)
    self.assertIsNotNone(boundary)
    self.assertEqual(boundary[:2], (90, "template-event-entry"))

  def test_kaf_dal_and_waw_use_physical_entry_landmarks(self):
    kaf = next(template for template in self.elastic_templates
               if template.name == "kaf-medial-zigzag")
    self.assertEqual(kaf.required_events, (
        ("peak", 0.5), ("x-right", 0.5), ("x-left", 0.5)))
    self.assertEqual(kaf.entry, BodyBoundarySpec(
        method="rise-onset", anchor_event="peak"))

    half_stroke = BodyBoundarySpec(
        method="event-offset", anchor_event="high-branch",
        offset_stroke_widths=-0.5)
    for name in ("dal-dhal-final-rise-head", "waw-final-head"):
      template = next(template for template in self.elastic_templates
                      if template.name == name)
      self.assertEqual(template.entry, half_stroke)

  def test_rise_onset_entry_traces_from_event_to_start_of_ascent(self):
    grid = np.arange(101.0)
    y = np.r_[np.zeros(31), -np.arange(1.0, 51.0),
              np.full(20, -50.0)]
    profile = {
        "grid": grid, "ds": 1.0, "y": y,
        "owned_len": np.ones_like(grid)}
    events = [self.event(10.0, "branch"),
              self.event(80.0, "peak"),
              self.event(90.0, "x-right"),
              self.event(100.0, "x-left")]
    boundary = _template_body_boundary_index(
        profile, events, 0, (1, 2, 3), BodyBoundarySpec(
            method="rise-onset", anchor_event="peak"), 10.0)
    self.assertIsNotNone(boundary)
    self.assertEqual(boundary[1], "template-rise-onset")
    # Gaussian smoothing deliberately begins the stable ascent slightly before
    # the sharp synthetic corner, but must keep it near that lower corner.
    self.assertLessEqual(abs(boundary[0] - 30), 10)

  def test_observed_event_exit_uses_last_branch_inside_shape_context(self):
    grid = np.arange(101.0)
    profile = {"grid": grid, "ds": 1.0,
               "owned_len": np.ones_like(grid)}
    events = [self.event(10.0, "high-branch", 2.0),
              self.event(20.0, "valley"),
              self.event(30.0, "branch"),
              self.event(40.0, "branch"),
              self.event(80.0, "valley")]
    index = _template_body_exit_index(
        profile, events, (0, 1, 4), BodyBoundarySpec(
            method="observed-event-offset", anchor_event="branch",
            occurrence="last", offset_stroke_widths=1.0), 100, 10.0)
    self.assertEqual(index, 50)

  def test_template_body_boundary_prefers_nearby_valley_then_width_entry(self):
    grid = np.arange(201.0)
    profile = {"grid": grid, "ds": 1.0,
               "owned_len": np.r_[np.ones(80) * 10.0,
                                   np.ones(121) * 20.0]}
    events = [self.event(0.0, "branch", 2.0),
              self.event(90.0, "valley"),
              self.event(100.0, "branch", 2.0)]
    index, kind, _ = _template_body_boundary_index(
        profile, events, 0, (2,), BodyBoundarySpec(), 10.0)
    self.assertEqual((index, kind), (90, "template-valley-entry"))

    index, kind, _ = _template_body_boundary_index(
        profile, [events[0], events[2]], 0, (1,),
        BodyBoundarySpec(), 10.0)
    self.assertEqual((index, kind), (79, "template-width-entry"))

    flat_profile = {"grid": grid, "ds": 1.0,
                    "owned_len": np.ones_like(grid) * 10.0}
    index, kind, _ = _template_body_boundary_index(
        flat_profile, [events[0], events[2]], 0, (1,),
        BodyBoundarySpec(), 10.0)
    self.assertEqual((index, kind), (90, "template-event-entry"))

    explicit_events = [self.event(0.0, "branch"),
                       self.event(100.0, "peak"),
                       self.event(120.0, "valley")]
    index, kind, _ = _template_body_boundary_index(
        flat_profile, explicit_events, 0, (1, 2), BodyBoundarySpec(
            method="event-offset", anchor_event="peak",
            offset_stroke_widths=-1.0), 10.0)
    self.assertEqual((index, kind), (90, "template-event-entry"))

  def test_template_body_exit_keeps_stroke_after_valley_and_curved_entry(self):
    grid = np.arange(201.0)
    tangents = np.column_stack([
        np.r_[np.ones(80) * 0.5, np.ones(121) * 0.95],
        np.zeros(201)])
    profile = {"grid": grid, "ds": 1.0,
               "owned_len": np.ones_like(grid) * 10.0,
               "tangents": tangents}
    valley_events = [self.event(50.0, "valley"),
                     self.event(150.0, "branch")]
    self.assertEqual(_template_body_exit_index(
        profile, valley_events, (0,), BodyBoundarySpec(
            method="event-offset", anchor_event="valley",
            offset_stroke_widths=1.0), 120, 10.0), 60)

    branch_events = [self.event(0.0, "branch"),
                     self.event(150.0, "branch")]
    self.assertEqual(_template_body_exit_index(
        profile, branch_events, (0,), BodyBoundarySpec(
            method="horizontal-stabilization", min_horizontality=0.9),
        120, 10.0), 80)

    no_horizontal = dict(profile)
    no_horizontal["tangents"] = np.column_stack([
        np.ones(201) * 0.5, np.zeros(201)])
    self.assertIsNone(_template_body_exit_index(
        no_horizontal, branch_events, (0,), BodyBoundarySpec(
            method="horizontal-stabilization", min_horizontality=0.9),
        120, 10.0))

    loop_events = [self.event(50.0, "valley")]
    neck_profile = dict(profile)
    neck_profile["owned_len"] = np.r_[
        np.ones(80) * 30.0, np.ones(121) * 10.0]
    loop_exit = BodyBoundarySpec(
        method="width-or-event-exit", anchor_event="valley",
        occurrence="last")
    self.assertEqual(_template_body_exit_index(
        neck_profile, loop_events, (0,), loop_exit, 120, 10.0), 80)
    self.assertEqual(_template_body_exit_index(
        profile, loop_events, (0,), loop_exit, 120, 10.0), 50)

    missing_terminal_valley = [self.event(50.0, "peak"),
                               self.event(100.0, "valley")]
    rise_anchored = BodyBoundarySpec(
        method="width-or-event-exit", anchor_event="peak",
        occurrence="last", offset_stroke_widths=1.0)
    self.assertEqual(_template_body_exit_index(
        neck_profile, missing_terminal_valley, (0, 1), rise_anchored,
        120, 10.0), 80)
    self.assertEqual(_template_body_exit_index(
        profile, missing_terminal_valley, (0, 1), rise_anchored,
        120, 10.0), 110)


class AlignmentTraceTest(unittest.TestCase):

  def test_composite_partition_is_template_guided_without_rising_join(self):
    grid = np.arange(101.0)
    owned = np.r_[np.full(60, 10.0), np.full(41, 20.0)]
    ones = np.ones_like(grid)
    profile = {
        "grid": grid, "total_s": 100.0, "ds": 1.0,
        "x": grid, "y": np.zeros_like(grid), "r": ones,
        "cut_len": owned, "core_len": owned, "owned_len": owned,
        "foreign_len": np.zeros_like(grid), "global_cut_len": owned,
        "n_runs": np.ones_like(grid, dtype=int),
        "global_runs": np.ones_like(grid, dtype=int),
        "tangents": np.column_stack([ones, np.zeros_like(grid)]),
        "chords": [((float(i), -1.0), (float(i), 1.0)) for i in grid],
        "loop_evidence": [],
    }
    events = [
        SkeletonEvent(20.0, "high-branch", 4.0, (20.0, 0.0)),
        SkeletonEvent(70.0, "branch", 2.0, (70.0, 0.0)),
    ]
    core = ElasticBodyTemplate(
        "beh-hah-core", ("بتثني", "جحخ"), ("initial", "medial"),
        (("high-branch", 2.5),))
    noon = ElasticBodyTemplate(
        "noon-final", ("ن",), ("final",), (("branch", 0.5),))
    trace = {}

    aligned = force_align_bodies_and_joins(
        profile, events, ("ن", "ح", "ن"),
        ("initial", "medial", "final"), 0.0, 10.0, "next-start",
        (core, noon), trace)

    self.assertFalse(aligned.reason)
    self.assertEqual(trace["joinHypotheses"], 0)
    self.assertEqual(trace["compositeGuidedAlignments"], 1)
    self.assertEqual(
        [(group.letter_start, group.letter_end, group.template)
         for group in aligned.groups],
        [(0, 2, "beh-hah-core"), (2, 3, "noon-final")])
    self.assertEqual(
        [(join.letter_boundary, join.kind) for join in aligned.joins],
        [(1, "fused"), (2, "template-width-entry")])
    physical = aligned.joins[1]
    self.assertLess(physical.start.s, physical.end.s)
    self.assertEqual(aligned.cuts[0].s, physical.end.s)

  def test_reports_first_empty_event_stage_without_template_file(self):
    first = ElasticBodyTemplate(
        "first", ("ب",), ("initial",), (("branch", 0.5),))
    second = ElasticBodyTemplate(
        "second", ("ت",), ("final",), (("branch", 0.5),))
    events = [SkeletonEvent(
        s=10.0, kind="branch", strength=1.0, xy=(10.0, 0.0))]
    trace = {}

    options = _ordered_single_body_match_options(
        events, ("ب", "ت"), ("initial", "final"),
        (first, second), trace=trace)

    self.assertFalse(options)
    self.assertEqual(trace["result"], "topology-failed")
    self.assertEqual(trace["firstFailingPosition"], 1)
    self.assertEqual(
        [(stage["incomingStates"], stage["outgoingStates"])
         for stage in trace["stages"]], [(1, 1), (1, 0)])
    failure = trace["stages"][1]["failures"][0]
    self.assertEqual(failure["reason"],
                     "required-event-owned-or-before-cursor")
    self.assertEqual(failure["earlierEligible"][0]["id"], 0)
    self.assertEqual(
        trace["stages"][1]["rejectionCounts"],
        {"required-event-owned-or-before-cursor": 1})


if __name__ == "__main__":
  unittest.main()
