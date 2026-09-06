-- Letter/mark corpus index. Derived and disposable: every table here is
-- rebuilt from output/<analysis>/pageNNN/words.json plus a live run of
-- tools/paw_segmentation/segment_paw_letters.py. Human review state lives in
-- a separate database (see reviews_schema.sql) so a rebuild never destroys it.
--
-- Measurement conventions, in one place:
--   * "_px" columns are word-space pixels (the word body crop, origin at the
--     word's top-left in page space).
--   * "_norm" columns are the same measurement divided by the page's nominal
--     stroke width, i.e. expressed in qalam widths, which is what makes rows
--     from different pages comparable at all.
--   * The letter frame used by letter_shape.path_d has its origin at the
--     letter's entry point on the baseline, y up, x positive along the
--     writing direction (leftwards in the image), scaled to qalam widths.

PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS runs (
  run_id      TEXT PRIMARY KEY,
  tool        TEXT NOT NULL,
  argv        TEXT NOT NULL,
  git_commit  TEXT,
  created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pages (
  page                INTEGER PRIMARY KEY,
  source_image        TEXT,
  image_w             INTEGER,
  image_h             INTEGER,
  nominal_stroke_px   REAL,     -- paw_width.estimate_page_nominal_width
  nominal_stroke_mad  REAL,
  nominal_method      TEXT,
  word_count          INTEGER,
  paw_count           INTEGER,
  letter_count        INTEGER,
  -- Base letters the page's text calls for. Equal to letter_count unless a
  -- PAW was dropped rather than merely segmented badly, which is the one
  -- failure mode that would make a query silently under-report.
  letters_expected    INTEGER,
  mark_count          INTEGER,
  -- 'instance' when the analyzer wrote one mask per semantic mark, the
  -- current contract; 'class-group' for older output where one entry could
  -- hold several marks of a class. Recorded because one analysis directory
  -- can hold pages written by both.
  mark_format         TEXT,
  marks_unindexed     INTEGER,
  run_id              TEXT REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS words (
  word_id          TEXT PRIMARY KEY,   -- p003:l00_w04
  page             INTEGER NOT NULL,
  line_index       INTEGER,
  word_index       INTEGER,
  word_key         TEXT,               -- l00_w04, the id used by the analyzer
  text             TEXT,
  bases            TEXT,
  x INTEGER, y INTEGER, w INTEGER, h INTEGER,
  baseline_y_page  REAL,
  expected_paws    INTEGER,
  observed_paws    INTEGER,
  abstain          INTEGER
);

CREATE TABLE IF NOT EXISTS paws (
  paw_id          TEXT PRIMARY KEY,    -- p003:l00_w04:paw0
  word_id         TEXT NOT NULL REFERENCES words(word_id),
  page            INTEGER NOT NULL,
  paw_index       INTEGER NOT NULL,
  text            TEXT,
  letters         TEXT,
  x INTEGER, y INTEGER, w INTEGER, h INTEGER,
  stroke_width    REAL,
  spine_length    REAL,
  loops           INTEGER,
  alignment_mode  TEXT,
  cut_policy      TEXT,
  ok              INTEGER,
  reason          TEXT,
  merged          INTEGER,
  -- The body is drawn into the word before this one, so this PAW has no ink
  -- of its own here and no measurement. Distinct from `merged`, which is two
  -- PAWs sharing one component inside a single word.
  connected_to_previous_word INTEGER,
  low_confidence  INTEGER,
  confidence      REAL,
  margin          REAL
);

CREATE TABLE IF NOT EXISTS letters (
  letter_id           TEXT PRIMARY KEY,  -- p003:l00_w04:paw0:let2
  paw_id              TEXT NOT NULL REFERENCES paws(paw_id),
  word_id             TEXT NOT NULL REFERENCES words(word_id),
  page                INTEGER NOT NULL,
  letter_index        INTEGER NOT NULL,  -- within the PAW, right to left
  word_letter_index   INTEGER,           -- within the word
  char                TEXT,
  char_class          TEXT,              -- shape class; senna in initial/medial
  form                TEXT,              -- isolated | initial | medial | final
  prev_char           TEXT,
  next_char           TEXT,
  logical_group       TEXT,              -- ligature group name, when any
  template            TEXT,              -- matched skeleton template, when any
  group_size          INTEGER,           -- letters sharing one uncut body
  bbox_x INTEGER, bbox_y INTEGER, bbox_w INTEGER, bbox_h INTEGER,
  page_x INTEGER, page_y INTEGER,
  ink_px              INTEGER,
  -- All three widths include the letter's left kashida: a letter is measured
  -- together with the connector that leaves it, so the advance runs from this
  -- letter's entry to the next letter's entry, as a font advance does.
  advance_width_px    REAL,   -- entry point to exit point, along the baseline
  bbox_width_px       REAL,   -- ink bounding box
  spine_extent_px     REAL,   -- medial-axis arclength span
  kashida_px          REAL,   -- how much of the span is connector, not body
  height_px           REAL,
  above_baseline_px   REAL,
  below_baseline_px   REAL,
  radius_px           REAL,   -- mean inscribed-disk radius over the span
  adv_w_norm          REAL,
  bbox_w_norm         REAL,
  extent_norm         REAL,
  kashida_norm        REAL,
  height_norm         REAL,
  above_norm          REAL,
  below_norm          REAL,
  radius_norm         REAL,
  teeth_expected      INTEGER,
  teeth_observed      INTEGER,
  confidence          REAL,
  low_confidence      INTEGER,
  ok                  INTEGER,
  run_id              TEXT REFERENCES runs(run_id)
);

-- Geometry is split off so the query table stays small and cache-friendly:
-- a filter/sort scan never touches the path and mask blobs.
CREATE TABLE IF NOT EXISTS letter_shape (
  letter_id      TEXT PRIMARY KEY REFERENCES letters(letter_id),
  path_d         TEXT,     -- cubic outline in the normalized letter frame
  -- The connector leaving this letter, traced from the segmenter's own join
  -- attribution and expressed in the same frame as path_d. Empty when the
  -- letter carries no connector.
  kashida_path_d TEXT,
  contour_count  INTEGER,
  view_x REAL, view_y REAL, view_w REAL, view_h REAL,  -- bbox in that frame
  mask_rle       BLOB,     -- zlib'd row-major run lengths of the letter mask
  mask_w INTEGER, mask_h INTEGER, mask_x INTEGER, mask_y INTEGER,
  entry_x REAL, entry_y REAL, exit_x REAL, exit_y REAL,  -- word space
  baseline_y     REAL,
  norm_scale     REAL       -- page nominal stroke width used for the frame
);

CREATE TABLE IF NOT EXISTS marks (
  mark_id            TEXT PRIMARY KEY,  -- p003:l00_w04:paw0:mk:fatha:20
  page               INTEGER NOT NULL,
  word_id            TEXT NOT NULL REFERENCES words(word_id),
  paw_id             TEXT REFERENCES paws(paw_id),
  class              TEXT,
  component_indices  TEXT,
  component_count    INTEGER,
  grouping           TEXT,    -- component | class-parts | ambiguous
  x INTEGER, y INTEGER, w INTEGER, h INTEGER,
  page_x INTEGER, page_y INTEGER,
  area               INTEGER,
  mask_path          TEXT,
  width_norm         REAL,
  height_norm        REAL,
  area_norm          REAL,
  owning_letter_id   TEXT REFERENCES letters(letter_id),
  confidence         REAL,
  run_id             TEXT REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS letter_query ON letters(char, form, adv_w_norm);
CREATE INDEX IF NOT EXISTS letter_class ON letters(char_class, form, adv_w_norm);
CREATE INDEX IF NOT EXISTS letter_page  ON letters(page, word_id);
CREATE INDEX IF NOT EXISTS letter_paw   ON letters(paw_id);
-- Showing a match inside its whole word gathers that word's letters.
CREATE INDEX IF NOT EXISTS letter_word  ON letters(word_id);
CREATE INDEX IF NOT EXISTS paw_word     ON paws(word_id);
CREATE INDEX IF NOT EXISTS word_page    ON words(page);
CREATE INDEX IF NOT EXISTS mark_query   ON marks(class, width_norm);
-- Typing a diacritic after a letter filters on the marks that letter owns.
CREATE INDEX IF NOT EXISTS mark_owner   ON marks(owning_letter_id);
CREATE INDEX IF NOT EXISTS mark_page    ON marks(page, word_id);
