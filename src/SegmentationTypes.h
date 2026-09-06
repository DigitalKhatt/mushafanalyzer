#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "MarkTaxonomy.h"

struct ExpectedMarkInstance {
  MarkClass markClass = MarkClass::Other;
  int baseIndex = -1;
  // Most semantic marks occupy one physical component. Some glyphs, such as
  // U+06E9 place-of-sajdah, remain one textual mark in one PAW but are drawn
  // as multiple disconnected components in the scan.
  int physicalComponentCount = 0;
};

// Text-derived expectations for one connected Arabic joining group inside a
// word. The word-level DP uses these expectations after the line DP has
// already selected the word's component range.
struct WordPawExpectation {
  std::string text;
  int expectedBaseCount = 0;
  int expectedBodyComponents = 1;
  int expectedMarkPaths = 0;
  int expectedSmallLetterPaths = 0;
  MarkInventory expectedMarks{};
  std::vector<ExpectedMarkInstance> expectedMarkInstances;
};

// Preliminary evidence copied onto the local connected components of one
// word. Keeping the source index lets the generated PAW manifest refer back
// to PageModel::comps while the masks continue to use local labels.
struct WordComponentEvidence {
  int sourceComponentIndex = -1;
  double bodyProbability = 0.0;
  double dotProbability = 0.0;
  double ordinaryMarkProbability = 0.0;
  double smallLetterProbability = 0.0;
  double unknownProbability = 1.0;
  bool compactFloatingMark = false;
  std::array<double, kMarkClassCount> markClassProbabilities{};
  // Soft evidence from the reviewed component-template recognizer. The mask
  // distinguishes a class that has a template bank from a zero match, so a
  // negative competitor can still penalize an assignment.
  std::array<double, kMarkClassCount> templateMarkProbabilities{};
  std::array<std::uint8_t, kMarkClassCount> templateMarkClasses{};
  int detectedThreeDotsAboveComponentCount = 0;
  // The U+06E9 place-of-sajdah glyph is one semantic mark but has two
  // disconnected physical components in this scan set.
  int detectedSajdaComponentCount = 0;
  double attachedAlefHamzaAboveProbability = 0.0;
  // Reviewed evidence that this component is a letter body, from the body
  // template bank. It answers the question the mark banks cannot: the
  // preliminary classifier's rule is binary, and a small round letter whose
  // ink stops a hair above the line satisfies its definition of a mark.
  double bodyTemplateProbability = 0.0;
};

struct PawMarkMask {
  MarkClass markClass = MarkClass::Other;
  std::vector<int> componentIndices;
  std::string path;
};

struct WordPaw {
  int pawIndex = -1;
  std::string text;
  cv::Rect bbox;
  std::vector<int> componentLabels;
  std::vector<int> componentIndices;
  std::vector<int> bodyComponentLabels;
  std::vector<int> bodyComponentIndices;
  std::vector<int> markComponentLabels;
  std::vector<int> markComponentIndices;
  std::vector<MarkClass> markComponentClasses;
  // Parallel to markComponentLabels/markComponentIndices. Components that
  // belong to one multi-path semantic mark share an instance index; a
  // negative value means that the component has no expected instance and
  // must be exported separately.
  std::vector<int> markComponentInstanceIndices;
  int expectedMarkComponents = 0;
  int observedMarkComponents = 0;
  // The first expected Alef PAW may have no separable body when the Alef is
  // physically connected toward the preceding word. This flag preserves the
  // reason for the empty body when no separable Alef body remains.
  bool bodyConnectedToPreviousWord = false;
  // One physical component carries this PAW and the preceding one: they only
  // touch in the scan, so connected-component labeling cannot tell them
  // apart. The shared component is listed in both PAWs' body components, and
  // cutting it is left to the letter segmenter, which has the medial axis and
  // the cut cost to find the contact with.
  bool bodyMergedWithPrevious = false;
  double cost = 0.0;
  std::string pngPath;
  std::string bodyPath;
  std::string marksPath;
  std::vector<PawMarkMask> markMasks;
};

// A connected component from the binarized image
struct CC {
  int label;  // label id
  cv::Rect bb;
  int area = 0;
  cv::Point2f cent{0, 0};
  int line = -1;
  // Decorative scan components such as the thin sajda bar are retained in
  // the page component inventory for diagnostics but excluded from line/word
  // ownership and PAW segmentation.
  bool ignored = false;
  // Preliminary, word-independent classification probabilities. These are
  // deliberately soft: word segmentation may use the evidence without
  // turning an uncertain shape into a hard label.
  double bodyProbability = 0.0;
  double dotProbability = 0.0;
  double ordinaryMarkProbability = 0.0;
  double smallLetterProbability = 0.0;
  double unknownProbability = 1.0;
  bool compactFloatingMark = false;
  std::array<double, kMarkClassCount> markClassProbabilities{};
  std::array<double, kMarkClassCount> templateMarkProbabilities{};
  std::array<std::uint8_t, kMarkClassCount> templateMarkClasses{};
  // A reviewed/template-matched semantic three-dots-above instance may be
  // rendered as one connected blob instead of the nominal two blobs. Zero
  // means that no sufficiently reliable scan-form detection is available.
  int detectedThreeDotsAboveComponentCount = 0;
  int detectedSajdaComponentCount = 0;
  // A textual alef and hamza-above may touch in the scan and therefore form
  // one physical connected component. This confidence is produced before
  // word DP; ownership is still decided jointly with textual expectations.
  double attachedAlefHamzaAboveProbability = 0.0;
  // See WordComponentEvidence::bodyTemplateProbability.
  double bodyTemplateProbability = 0.0;
};

enum class ComponentTypeEnum { Uncertain = 0,
                               Body,
                               Dot,
                               OtherMark };

struct ComponentType {
  int componentLabel;
  ComponentTypeEnum componentType;
  cv::Rect bbox;
};

// Main segmentation result
struct MarkSegmentationResult {
  std::vector<ComponentType> componentTypes;
  std::vector<WordPaw> paws;
  cv::Rect bodyRect;  // in word image coordinates
  int totalInkPixels = 0;
  int uncertainInkPixels = 0;
};

// Horizontal ink support measured only inside a corridor around the baseline.
// Unlike a component/body bounding box, this deliberately ignores long tails
// and marks outside the writing band, so it is the geometry used for
// justification and visible inter-word spacing.
struct HorizontalInkSpan {
  int left = 0;   // word-image coordinates, inclusive
  int right = 0;  // word-image coordinates, inclusive
  int width = 0;
  int inkPixels = 0;
  bool valid = false;
};

struct WordBox {
  cv::Rect rect;  // in original image coordinates
  int lineIndex = -1;
  int wordIndex = -1;
  int bboxGapToNextPx = 0;     // full-component boxes; may be negative
  int spacingGapToNextPx = 0;  // baseline-corridor ink spans; canonical gap
  int expectedBodyComponents = 0;
  int expectedMarkComponents = 0;
  int recognizedComponentCountAdjustment = 0;
  int recognizedAttachedAlefHamzaAboveCount = 0;
  int expectedSmallLetterComponents = 0;
  double observedSmallLetterComponents = 0.0;
  MarkInventory expectedMarks{};
  std::string expectedText;  // optional: token from mushaf text if aligned
  std::string wordId;        // e.g. "p001_l03_w07"
  std::string pngPath;       // extracted bitmap
  std::string svgPath;       // vectorized output
  std::vector<int> compIds;
  std::vector<WordPawExpectation> pawExpectations;
  MarkSegmentationResult bodySegmentation;
  HorizontalInkSpan spacingSpan;
  double foreignLineInkFraction = 0.0;
  double heightToLinePitch = 0.0;
};

struct LineBox {
  cv::Rect rect;
  double baseLine = 0.0;
  double baseLineConfidence = 0.0;
  bool baseLineSearchBoundaryHit = false;
  std::vector<WordBox> words;
};

struct PageModel {
  cv::Mat labels;     // CV_32S
  cv::Mat stats;      // CV_32S
  cv::Mat centroids;  // CV_64F
  std::vector<LineBox> lines;
  std::vector<CC> comps;
  std::vector<std::string> assignmentCorrections;
};
