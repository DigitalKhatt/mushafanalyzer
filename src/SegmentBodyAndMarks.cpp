#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "PageProcessor.h"
#include "ComponentGeometry.h"

static std::vector<std::vector<cv::Point>> contoursFromLabel(const cv::Mat& labels, int label) {
  cv::Mat mask = (labels == label);
  mask.convertTo(mask, CV_8U, 255);
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
  return contours;
}

static double computePerimeter(const std::vector<std::vector<cv::Point>>& contours) {
  double p = 0.0;
  for (const auto& c : contours) p += cv::arcLength(c, true);
  return p;
}

static double computeSolidityFromContour(const std::vector<cv::Point>& contour, double area) {
  if (contour.size() < 3) return 0.0;
  std::vector<cv::Point> hull;
  cv::convexHull(contour, hull);
  double hullArea = std::fabs(cv::contourArea(hull));
  if (hullArea <= 1e-9) return 0.0;
  return area / hullArea;
}

static double computeEllipseAspect(const std::vector<cv::Point>& contour) {
  if (contour.size() < 5) return 0.0;
  cv::RotatedRect e = cv::fitEllipse(contour);

  double a = std::max(e.size.width, e.size.height) * 0.5;  // semi-major
  double b = std::min(e.size.width, e.size.height) * 0.5;  // semi-minor
  if (a <= 1e-6 || b <= 1e-6) return 0.0;

  return a / b;
}

static bool bboxIntersectsBaseline(const cv::Rect& bbox, int baselineY, int margin) {
  int bandTop = baselineY - margin;
  int bandBottom = baselineY + margin;

  return !(bbox.y + bbox.height < bandTop || bbox.y > bandBottom);
}

static bool markClassIsBelow(MarkClass markClass) {
  switch (markClass) {
    case MarkClass::DotBelowOne:
    case MarkClass::DotBelowTwo:
    case MarkClass::Kasra:
    case MarkClass::Kasratan:
    case MarkClass::HamzaBelow:
      return true;
    default:
      return false;
  }
}

static bool markClassIsDot(MarkClass markClass) {
  switch (markClass) {
    case MarkClass::DotAboveOne:
    case MarkClass::DotBelowOne:
    case MarkClass::DotAboveTwo:
    case MarkClass::DotBelowTwo:
    case MarkClass::DotAboveThree:
      return true;
    default:
      return false;
  }
}

static int markClassComponentCapacity(const ExpectedMarkInstance& expected) {
  if (expected.physicalComponentCount > 0)
    return expected.physicalComponentCount;

  const MarkClass markClass = expected.markClass;
  switch (markClass) {
    case MarkClass::DotAboveTwo:
    case MarkClass::DotBelowTwo:
      // In the Old Madina scan the two dots are usually joined into one
      // connected component, and a second component means a second logical
      // two-dot mark, not a second slot for this one. When the scan draws the
      // pair vertically the halves can come out separate, but a second slot
      // is the wrong place to fix that: measured on 51 pages it joined 7 real
      // pairs and stole 8 unrelated single dots, because a shape test cannot
      // tell the other half of this pair from any other round dot and this
      // DP's state carries no way to require adjacency to the first half.
      // Those pairs have to be joined before the assignment runs.
      return 1;
    case MarkClass::DotAboveThree:
      // The font draws this mark as two contours, a joined pair and a single
      // dot, and joinSplitDotClusters welds them into one component before any
      // segmentation runs -- so by the time the assignment DP sees it, it is
      // one. Every extra slot is an opening, not a safety margin: with three
      // it reached 1,062px across a PAW for another letter's dot pair, and
      // with two, 19 of the 36 marks that used the second slot filled it from
      // 199 to 2,064px away, almost always with a small alef or a fatha,
      // against 17 that were genuine splits the join had missed. A
      // missing-plus-unassigned score prefers two slots, but only because
      // swallowing an intruder removes an unassigned component without ever
      // recording the mislabel; counting the thefts directly settles it.
      return 1;
    case MarkClass::Fathatan:
    case MarkClass::Kasratan:
    case MarkClass::Dammatan:
      // Tanwin is encoded as two physical vowel components even though it is
      // one combining-mark inventory entry.
      return 2;
    case MarkClass::WaqfQaf:
    case MarkClass::WaqfJeem:
      // These waqf symbols consist of a main glyph and a detached companion
      // component in the Old Madina scan.
      return 2;
    case MarkClass::WaqfThreeDots:
      return 3;
    default:
      return 1;
  }
}

// Marks on one base are stacked from the body outwards.  In the Old Madina
// scans the base dots and hamza-above are closest to the body, shadda is next,
// and the short vowels/tanwins are furthest away.  The image coordinate
// system grows downwards, so this means that the corresponding component
// centers are ordered bottom-to-top (decreasing y) as the stack rank
// increases. Below the baseline the component order is reversed: kasra is
// farther from the body than a below-dot or hamza-below, so it is processed
// first and gets the lower rank. Maddah and small alef are intentionally not put in this
// geometry-only order: their scan shapes overlap, so their template evidence
// must decide which component occupies which expected slot.
static int markClassStackRank(MarkClass markClass) {
  switch (markClass) {
    case MarkClass::DotAboveOne:
    case MarkClass::DotAboveTwo:
    case MarkClass::DotAboveThree:
    case MarkClass::HamzaAbove:
      return 0;
    case MarkClass::Shadda:
      return 1;
    case MarkClass::Fatha:
    case MarkClass::Damma:
    case MarkClass::Fathatan:
    case MarkClass::Dammatan:
    case MarkClass::Sukun:
      return 2;
    case MarkClass::DotBelowOne:
    case MarkClass::DotBelowTwo:
    case MarkClass::HamzaBelow:
      return 1;
    case MarkClass::Kasra:
    case MarkClass::Kasratan:
      return 0;
    case MarkClass::WaqfQaf:
    case MarkClass::WaqfJeem:
    case MarkClass::WaqfSad:
    case MarkClass::WaqfLam:
    case MarkClass::WaqfMeem:
    case MarkClass::WaqfThreeDots:
      // Waqf qaf is drawn above the vowel on the same base; its two physical
      // paths therefore occupy the outermost/top stack level. Waqf jeem sits
      // the same way and needs the rank more: its second path is a dot in the
      // hollow of the curl, small enough to pass for nothing in particular,
      // and with no rank here the stacking order never engages at all. Of the
      // 585 PAWs holding both a damma and a waqf jeem, 203 -- better than one
      // in three -- gave the damma slot to that dot, against 3.3% for waqf
      // qaf, which has had this rank all along, and 0.43% for a damma with no
      // waqf mark beside it.
      return 3;
    default:
      return -1;
  }
}

static bool textContainsAlefWithHamzaAboveBase(const std::string& text) {
  // The input normally uses alef + U+0654, but accept precomposed alef with
  // hamza above as well.  A tall, narrow selected body in these PAWs is the
  // alef carrying the attached hamza; the separate upper component is then
  // the vowel that must be assigned above it.
  return text.find("\xD8\xA7") != std::string::npos ||
         text.find("\xD8\xA3") != std::string::npos;
}

static bool textStartsWithAlefBase(const std::string& text) {
  // Keep this test on the first UTF-8 code point so a later Alef in a longer
  // PAW does not trigger the connected-first-Alef rule.
  static constexpr const char* alefBases[] = {
      "\xD8\xA7",  // ا
      "\xD8\xA2",  // آ
      "\xD8\xA3",  // أ
      "\xD8\xA5",  // إ
      "\xD9\xB1",  // ٱ
      "\xD9\xB2",  // ٲ
      "\xD9\xB3",  // ٳ
      "\xD9\xB5",  // ٵ
  };
  for (const char* alef : alefBases) {
    const std::string prefix(alef);
    if (text.size() >= prefix.size() &&
        text.compare(0, prefix.size(), prefix) == 0)
      return true;
  }
  return false;
}

// A PAW that is not written on the writing line: the Arabic-Indic digits of
// an ayah number, U+0660 to U+0669, and the two ornaments they sit in --
// U+06DD end-of-ayah and U+06DE rub el hizb. `MakeExpectations` groups the
// same set as `otherCharacters`. Nothing here meets the baseline, so every
// test that asks "is this ink on the line" says nothing about them.
static bool textIsOffTheLine(const std::string& text) {
  if (text.empty()) return false;
  for (std::size_t i = 0; i + 1 < text.size(); i += 2) {
    const auto lead = static_cast<unsigned char>(text[i]);
    const auto trail = static_cast<unsigned char>(text[i + 1]);
    const bool digit = lead == 0xD9 && trail >= 0xA0 && trail <= 0xA9;
    const bool ornament = lead == 0xDB && (trail == 0x9D || trail == 0x9E);
    if (!digit && !ornament) return false;
  }
  return true;
}

// Whether a mark class is a miniature of a letter the PAW's text writes in
// full. The "small X" classes are exactly that -- a small_waw is a waw drawn
// small -- and the hamza above and below are the same glyph as the letter
// hamza. So a template match on one of these classes says nothing against a
// component that is the PAW's body: it is the shape that letter has. Without
// this the letter hamza is charged for matching hamza_above, which it does at
// 0.78 to 0.85 in 257 of the 330 template-named bodies in a 76-page sample.
static bool templateMarkClassIsThisPawsLetter(MarkClass markClass,
                                              const std::string& text) {
  const auto writes = [&text](const char* letter) {
    return text.find(letter) != std::string::npos;
  };
  switch (markClass) {
    case MarkClass::HamzaAbove:
    case MarkClass::HamzaBelow:
      return writes("\xD8\xA1");  // ء
    case MarkClass::SmallAlef:
      return writes("\xD8\xA7");  // ا
    case MarkClass::SmallWaw:
      return writes("\xD9\x88");  // و
    case MarkClass::SmallYeh:
      return writes("\xD9\x8A");  // ي
    case MarkClass::SmallNoon:
      return writes("\xD9\x86");  // ن
    case MarkClass::SmallMeem:
      return writes("\xD9\x85");  // م
    case MarkClass::SmallSeen:
      return writes("\xD8\xB3");  // س
    default:
      return false;
  }
}

static bool markClassIsSmallLetter(MarkClass markClass) {
  return isSmallLetterClass(markClass);
}

static bool markClassIsWaqf(MarkClass markClass) {
  switch (markClass) {
    case MarkClass::WaqfLam:
    case MarkClass::WaqfQaf:
    case MarkClass::WaqfJeem:
    case MarkClass::WaqfSad:
    case MarkClass::WaqfMeem:
    case MarkClass::WaqfThreeDots:
      return true;
    default:
      return false;
  }
}

MarkSegmentationResult PageProcessor::segmentBodyAndMarks(
    const cv::Mat& wordMask, int baselineY, double dotCompactness,
    double dotSolidity,
    const std::vector<WordPawExpectation>& pawExpectations,
    const std::vector<WordComponentEvidence>& componentEvidence,
    double markTemplatePenalty,
    bool firstPawConnectedToPreviousWord,
    const WordConnectedComponents* connectedComponents,
    double linePitch) {
  MarkSegmentationResult out;

  // Keep the original standalone API behavior for callers that do not have
  // precomputed components, while allowing exportWords() to share one pass
  // among evidence construction, this DP, and mask export.
  WordConnectedComponents computedComponents;
  if (connectedComponents == nullptr) {
    cv::connectedComponentsWithStats(
        wordMask, computedComponents.labels, computedComponents.stats,
        computedComponents.centroids, 8, CV_32S);
    connectedComponents = &computedComponents;
  }
  const cv::Mat& labels = connectedComponents->labels;
  const cv::Mat& stats = connectedComponents->stats;
  const int n = stats.rows;
  if (n <= 1) return out;

  struct ComponentFeature {
    int label = 0;
    int area = 0;
    cv::Rect bbox;
    double areaNorm = 0.0;
    double aspect = 0.0;
    bool dotLike = false;
    bool touchesBaseline = false;
    bool hasBaselineInk = false;
    double bodyProbability = 0.5;
    double markProbability = 0.5;
    double bodyCost = 0.6931471805599453;
    double markCost = 0.6931471805599453;
    bool compactFloatingMark = false;
    // The component's ink confined to the writing band. A bounding box is
    // the wrong ruler for asking whether a blob spans two joining groups: a
    // single ascender or tail decides it. This box ignores both, the way
    // HorizontalInkSpan already does for spacing.
    cv::Rect baselineBandBox;
    // Strongest confidence with which the reviewed mark-template bank named
    // this component, over the classes it has a bank for. Kept apart from
    // compactFloatingMark, which the geometric rule also sets: one is a guess
    // about shape, the other is reviewed data about identity, and the body
    // slot is entitled to charge them differently.
    double templateMarkProbability = 0.0;
    MarkClass templateMarkClass = MarkClass::Other;
    bool bodyTemplateMatch = false;
    WordComponentEvidence evidence;
  };

  constexpr double scale = 100.0;
  constexpr int bandHeight = 30;
  constexpr double minimumProbability = 0.02;
  // Confidence at which a reviewed mark template is taken to have identified
  // the component. Used both to call it a floating mark and to charge it in
  // the body slot, so the two cannot drift apart.
  constexpr double templateMarkConfidence = 0.72;
  // The same bar for the body bank. It is the level the mark banks are known
  // to work at -- their own within-class agreement runs 0.63 to 0.77 -- and
  // the shapes a body template has to beat top out below it.
  constexpr double bodyTemplateConfidence = 0.72;
  std::vector<ComponentFeature> features(n);

  auto geometryBodyProbability = [](const ComponentFeature& feature) {
    if (feature.areaNorm > 10.0) return 0.99;
    return feature.touchesBaseline ? 0.90 : 0.10;
  };

  for (int label = 1; label < n; ++label) {
    ComponentFeature& feature = features[label];
    feature.label = label;
    feature.area = stats.at<int>(label, cv::CC_STAT_AREA);
    if (feature.area <= 0) continue;
    feature.bbox = cv::Rect(
        stats.at<int>(label, cv::CC_STAT_LEFT),
        stats.at<int>(label, cv::CC_STAT_TOP),
        stats.at<int>(label, cv::CC_STAT_WIDTH),
        stats.at<int>(label, cv::CC_STAT_HEIGHT));
    feature.areaNorm = feature.area / (scale * scale);

    auto contours = contoursFromLabel(labels, label);
    double compactness = 0.0;
    double solidity = 1.0;
    if (!contours.empty()) {
      int largestIndex = 0;
      double largestArea = 0.0;
      for (int index = 0; index < static_cast<int>(contours.size());
           ++index) {
        const double contourArea =
            std::fabs(cv::contourArea(contours[index]));
        if (contourArea > largestArea) {
          largestArea = contourArea;
          largestIndex = index;
        }
      }
      const auto& largestContour = contours[largestIndex];
      const double perimeter = computePerimeter(contours);
      if (perimeter > 1e-9)
        compactness =
            4.0 * CV_PI * feature.area / (perimeter * perimeter);
      solidity = computeSolidityFromContour(
          largestContour, static_cast<double>(feature.area));
      feature.aspect = computeEllipseAspect(largestContour);
    }
    if (feature.aspect <= 0.0) {
      const double major =
          std::max(feature.bbox.width, feature.bbox.height);
      const double minor =
          std::max(1, std::min(feature.bbox.width, feature.bbox.height));
      feature.aspect = major / minor;
    }
    feature.dotLike = compactness >= dotCompactness &&
                      solidity >= dotSolidity && feature.areaNorm <= 1.5 &&
                      feature.aspect <= 1.5;

    const bool bboxNearBaseline =
        bboxIntersectsBaseline(feature.bbox, baselineY, bandHeight);
    const bool hasBaselineInk = componentHasInkNearBaseline(
        labels, label, baselineY, 5, feature.bbox);
    const bool startsNearBaseline =
        componentStartsNearBaseline(feature.bbox, baselineY);
    const double bboxAspect =
        std::max(feature.bbox.width, feature.bbox.height) /
        double(std::max(1, std::min(feature.bbox.width,
                                    feature.bbox.height)));
    const bool tallBodyStrokeAboveBaseline =
        !bboxNearBaseline &&
        feature.bbox.y + feature.bbox.height >= baselineY - 60 &&
        feature.bbox.height >= 400 && bboxAspect >= 3.0 &&
        feature.areaNorm >= 3.0;
    const bool compactBaselineMark =
        bboxNearBaseline && startsNearBaseline && feature.areaNorm <= 3.0 &&
        bboxAspect <= 2.0;
    const bool compactFloatingMark =
        compactBaselineMark ||
        (bboxNearBaseline && !hasBaselineInk && feature.areaNorm <= 3.0 &&
         feature.aspect <= 2.0 && solidity >= 0.85);
    feature.touchesBaseline =
        (bboxNearBaseline || tallBodyStrokeAboveBaseline) &&
        !compactFloatingMark;
    feature.hasBaselineInk = hasBaselineInk;
    {
      // A hand writing this script keeps the bodies in a band about a qalam
      // and a half either side of the line; what leaves it is an ascender, a
      // tail or a mark. Measuring the blob here answers "does it span two
      // entry points on the line" without an upright deciding the question.
      const int band = static_cast<int>(std::lround(1.5 * scale));
      const int top = std::max(0, baselineY - band);
      const int bottom = std::min(labels.rows - 1, baselineY + band);
      int left = std::numeric_limits<int>::max();
      int right = std::numeric_limits<int>::min();
      int first = std::numeric_limits<int>::max();
      int last = std::numeric_limits<int>::min();
      const cv::Rect box = feature.bbox & cv::Rect(0, 0, labels.cols, labels.rows);
      for (int y = top; y <= bottom; ++y) {
        const int* row = labels.ptr<int>(y);
        for (int x = box.x; x < box.x + box.width; ++x)
          if (row[x] == label) {
            left = std::min(left, x);
            right = std::max(right, x);
            first = std::min(first, y);
            last = std::max(last, y);
          }
      }
      if (left <= right)
        feature.baselineBandBox =
            cv::Rect(left, first, right - left + 1, last - first + 1);
    }

    if (label < static_cast<int>(componentEvidence.size()))
      feature.evidence = componentEvidence[label];
    const bool hasProbability =
        feature.evidence.bodyProbability > 0.0 ||
        feature.evidence.dotProbability > 0.0 ||
        feature.evidence.ordinaryMarkProbability > 0.0 ||
        feature.evidence.smallLetterProbability > 0.0;
    feature.bodyProbability = hasProbability
                                  ? feature.evidence.bodyProbability
                                  : geometryBodyProbability(feature);
    feature.markProbability =
        hasProbability
            ? std::max({feature.evidence.dotProbability,
                        feature.evidence.ordinaryMarkProbability,
                        feature.evidence.smallLetterProbability,
                        feature.evidence.unknownProbability})
            : 1.0 - feature.bodyProbability;
    double strongestTemplateMarkProbability = 0.0;
    std::size_t strongestTemplateMark =
        static_cast<std::size_t>(MarkClass::Other);
    for (std::size_t mark = 0; mark < kMarkClassCount; ++mark)
      if (feature.evidence.templateMarkClasses[mark] &&
          feature.evidence.templateMarkProbabilities[mark] >
              strongestTemplateMarkProbability) {
        strongestTemplateMarkProbability =
            feature.evidence.templateMarkProbabilities[mark];
        strongestTemplateMark = mark;
      }
    // The preliminary classifier intentionally keeps its baseline-start
    // decision. A strong reviewed mark template is nevertheless enough for
    // the final PAW DP to avoid treating that compact component as a body.
    // This is especially important for connected two-dot marks whose bbox
    // straddles the baseline but whose top does not start exactly on it.
    feature.templateMarkProbability = strongestTemplateMarkProbability;
    feature.templateMarkClass = static_cast<MarkClass>(strongestTemplateMark);
    // Carried as evidence only. Whether it counts is a per-PAW question: an
    // ayah marker's digits are never letters however round they are, and the
    // digit five is the same loop as a heh. See the body slot, which applies
    // this only where the PAW is written on a line.
    feature.bodyTemplateMatch =
        feature.evidence.bodyTemplateProbability >= bodyTemplateConfidence;
    feature.compactFloatingMark =
        compactFloatingMark || feature.evidence.compactFloatingMark ||
        strongestTemplateMarkProbability >= templateMarkConfidence;
    feature.bodyCost = -std::log(
        std::clamp(feature.bodyProbability, minimumProbability, 1.0));
    feature.markCost = -std::log(
        std::clamp(feature.markProbability, minimumProbability, 1.0));
  }

  auto appendComponentType = [&](int label, ComponentTypeEnum type) {
    if (label <= 0 || label >= n || features[label].area <= 0) return;
    out.componentTypes.push_back({label, type, features[label].bbox});
  };

  auto finalizeTotals = [&]() {
    cv::Rect bodyRect;
    for (const ComponentType& component : out.componentTypes) {
      const int area = stats.at<int>(component.componentLabel,
                                     cv::CC_STAT_AREA);
      out.totalInkPixels += area;
      if (component.componentType == ComponentTypeEnum::Uncertain)
        out.uncertainInkPixels += area;
      if (component.componentType == ComponentTypeEnum::Body)
        bodyRect |= component.bbox;
    }
    out.bodyRect = cv::Rect(bodyRect.x, bodyRect.y, bodyRect.width,
                            bodyRect.height);
  };

  auto appendHeuristicClassification = [&]() {
    for (int label = 1; label < n; ++label) {
      const ComponentFeature& feature = features[label];
      if (feature.area <= 0) continue;
      if (feature.areaNorm > 10.0 || feature.touchesBaseline)
        appendComponentType(label, ComponentTypeEnum::Body);
      else if (feature.dotLike)
        appendComponentType(label, ComponentTypeEnum::Dot);
      else
        appendComponentType(label, ComponentTypeEnum::OtherMark);
    }
  };

  if (pawExpectations.empty()) {
    appendHeuristicClassification();
    finalizeTotals();
    return out;
  }

  std::vector<int> order;
  for (int label = 1; label < n; ++label)
    if (features[label].area > 0) order.push_back(label);

  // The line segmenter orders components by the rightmost substantial ink in
  // a corridor around the baseline. Keep that same ordering for the
  // body/mark DP. A complete bounding-box edge is unstable for connected
  // Arabic bodies: an upper overhang of Ghayn, Kaf, or another large body can
  // contain the narrow Alef in its box and make the two bodies appear in the
  // wrong RTL order. Floating marks that do not reach the corridor retain
  // their complete-box edge, so they still sort naturally next to their base.
  constexpr int baselineCorridorAbove = 250;
  constexpr int baselineCorridorBelow = 100;
  std::vector<int> baselineRightEdge(n, -1);
  const int corridorTop = std::max(0, baselineY - baselineCorridorAbove);
  const int corridorBottom =
      std::min(labels.rows - 1, baselineY + baselineCorridorBelow);
  for (int label : order) {
    const cv::Rect& bbox = features[label].bbox;
    const int top = std::max(bbox.y, corridorTop);
    const int bottom =
        std::min(bbox.y + bbox.height - 1, corridorBottom);
    if (bottom < top) continue;
    for (int x = bbox.x + bbox.width - 1; x >= bbox.x; --x) {
      int ink = 0;
      for (int y = top; y <= bottom; ++y)
        if (labels.at<int>(y, x) == label) ++ink;
      if (ink < 2) continue;
      baselineRightEdge[label] = x;
      break;
    }
  }

  std::sort(order.begin(), order.end(), [&](int left, int right) {
    const cv::Rect& a = features[left].bbox;
    const cv::Rect& b = features[right].bbox;
    const int rightA = baselineRightEdge[left] >= 0
                           ? baselineRightEdge[left]
                           : a.x + a.width - 1;
    const int rightB = baselineRightEdge[right] >= 0
                           ? baselineRightEdge[right]
                           : b.x + b.width - 1;
    if (rightA != rightB) return rightA > rightB;
    if (a.x != b.x) return a.x > b.x;
    return a.y < b.y;
  });

  const int componentCount = static_cast<int>(order.size());
  const int pawCount = static_cast<int>(pawExpectations.size());
  constexpr double infinity = std::numeric_limits<double>::infinity();

  // The text calls for this many separate blobs. Seeing fewer is the one
  // situation in which two PAWs sharing a body is a repair rather than an
  // invention: something the hand drew apart arrived joined, and with the
  // partition into contiguous ranges being one-to-one, every PAW after the
  // join is otherwise satisfied by its neighbour's ink. So gate the whole
  // mechanism on the deficit rather than on a per-PAW judgement.
  int expectedComponentTotal = 0;
  int expectedBodyTotal = 0;
  for (const WordPawExpectation& expectation : pawExpectations) {
    expectedComponentTotal +=
        expectation.expectedBodyComponents + expectation.expectedMarkPaths;
    // Only PAWs written on the line are counted: an ayah marker and its
    // digits float, so they never contribute a component at the baseline and
    // counting them makes every word carrying one look body-short. That is
    // how page 11 l10_w01 ۞أَفَتَطْمَعُونَ came to hand the Alef of أَ to the
    // marker and merge أَ into فَتَطْمَعُو.
    if (!textIsOffTheLine(expectation.text))
      expectedBodyTotal += expectation.expectedBodyComponents;
  }
  // And the shortfall has to be in the bodies. A word can be a component
  // short because two marks touched, or because a three-dots was drawn as one
  // blob, and none of that is a reason to hang one PAW's body on another's.
  // What counts as something that could carry a body is ink at the baseline
  // and nothing else -- the same discrimination the body-slot cost above
  // rests on. The preliminary body/mark probabilities are deliberately not
  // consulted: they lean mark on a small round shape, so a floating stroke
  // 400 px above the line would be counted and a genuinely merged word would
  // not look short.
  int bodyLikeCount = 0;
  for (const int label : order)
    if (features[label].hasBaselineInk && !features[label].compactFloatingMark)
      ++bodyLikeCount;
  const bool componentsAreShort = m_p.shareBodyBetweenTouchingPaws &&
                                  componentCount < expectedComponentTotal &&
                                  bodyLikeCount < expectedBodyTotal;

  struct PawChoice {
    double cost = infinity;
    int bodyOrder = -1;
    int expectedMarkComponents = 0;
    int observedMarkComponents = 0;
    int bodyArea = 0;
    bool bodyConnectedToPreviousWord = false;
    int handoffBodyPosition = -1;
    // This PAW has no body of its own and takes the previous PAW's: one
    // component holds them both.
    bool sharesBodyWithPrevious = false;
  };

  bool hasMarkClassEvidence = false;
  for (int label : order)
    for (double probability : features[label].evidence.markClassProbabilities)
      if (probability > 0.0) hasMarkClassEvidence = true;

  auto horizontalOverlapRatio = [](const cv::Rect& left,
                                   const cv::Rect& right) {
    const int overlap = std::max(
        0, std::min(left.x + left.width, right.x + right.width) -
               std::max(left.x, right.x));
    const int denominator = std::max(
        1, std::min(left.width, right.width));
    return overlap / double(denominator);
  };

  // A scanned connected body can straddle the boundary between two text
  // PAWs. In that situation the range for one PAW may contain only a very
  // clear mark, while the neighboring range contains the shared body blob.
  // Keep the geometry test local to the final DP: it does not change the
  // preliminary classifier or the line-level component ownership.
  auto hasAdjacentBodyOverlap = [&](int start, int end) {
    for (const int neighborPosition : {start - 1, end}) {
      if (neighborPosition < 0 || neighborPosition >= componentCount)
        continue;
      const ComponentFeature& neighbor =
          features[order[neighborPosition]];
      if (neighbor.bodyProbability < 0.5 ||
          neighbor.bodyProbability < neighbor.markProbability ||
          neighbor.compactFloatingMark)
        continue;
      for (int position = start; position < end; ++position) {
        const ComponentFeature& component = features[order[position]];
        if (horizontalOverlapRatio(component.bbox, neighbor.bbox) >= 0.15)
          return true;
      }
    }
    return false;
  };

  auto isWideConnectedAlefBody = [&](int bodyPosition, int end,
                                     int pawIndex) {
    if (pawIndex != 0 || bodyPosition < 0 || bodyPosition >= end)
      return false;
    const WordPawExpectation& expectation = pawExpectations[pawIndex];
    if (expectation.expectedBodyComponents != 1 ||
        !textStartsWithAlefBase(expectation.text))
      return false;

    const ComponentFeature& body = features[order[bodyPosition]];
    if (body.bodyProbability < 0.75 || body.bbox.height <= 0)
      return false;

    // A standalone Alef is tall and narrow. A first-PAW body wider than its
    // height is strong evidence that connected baseline ink at the preceding
    // word boundary was absorbed into the same physical component.
    if (body.bbox.width <= body.bbox.height)
      return false;

    // The component must reach into the still-unassigned suffix. This avoids
    // rejecting a merely wide body that is fully contained in the first PAW.
    for (int position = end; position < componentCount; ++position) {
      const ComponentFeature& following = features[order[position]];
      const int horizontalGap =
          std::max(0, std::max(body.bbox.x, following.bbox.x) -
                          std::min(body.bbox.x + body.bbox.width,
                                   following.bbox.x + following.bbox.width));
      if (horizontalOverlapRatio(body.bbox, following.bbox) >= 0.10 ||
          horizontalGap <= 40)
        return true;
    }
    return false;
  };

  auto isPreviousWordBoundaryAlefBody = [&](int bodyPosition, int end,
                                            int pawIndex) {
    if (!firstPawConnectedToPreviousWord || pawIndex != 0 ||
        bodyPosition < 0 || bodyPosition >= end)
      return false;
    const WordPawExpectation& expectation = pawExpectations[pawIndex];
    if (expectation.expectedBodyComponents != 1 ||
        !textStartsWithAlefBase(expectation.text))
      return false;
    const ComponentFeature& body = features[order[bodyPosition]];
    return body.bodyProbability >= 0.75 && body.bbox.height > 0 &&
           body.bbox.width > body.bbox.height;
  };

  auto hasWideConnectedAlefBodyAhead = [&](int end, int pawIndex) {
    if (pawIndex != 0 || end >= componentCount) return false;
    const WordPawExpectation& expectation = pawExpectations[pawIndex];
    if (expectation.expectedBodyComponents != 1 ||
        !textStartsWithAlefBase(expectation.text))
      return false;
    for (int position = end; position < componentCount; ++position) {
      const ComponentFeature& candidate = features[order[position]];
      if (candidate.bodyProbability < 0.75 || candidate.bbox.height <= 0 ||
          candidate.bbox.width <= candidate.bbox.height)
        continue;
      for (int following = position + 1; following < componentCount;
           ++following) {
        const ComponentFeature& suffix = features[order[following]];
        const int horizontalGap =
            std::max(0, std::max(candidate.bbox.x, suffix.bbox.x) -
                            std::min(candidate.bbox.x + candidate.bbox.width,
                                     suffix.bbox.x + suffix.bbox.width));
        if (horizontalOverlapRatio(candidate.bbox, suffix.bbox) >= 0.10 ||
            horizontalGap <= 40)
          return true;
      }
    }
    return false;
  };

  // `sharedBodyLabel` is the body the previous PAW chose, or -1 where there
  // is none to lend or the word is not short of components. It enables one
  // extra alternative: this PAW keeps its own marks but has no body of its
  // own, because the previous PAW's body physically contains it too.
  auto segmentChoice = [&](int start, int end, int pawIndex,
                           int sharedBodyLabel, int sharedRangeStart,
                           int sharedBodyPosition) {
    PawChoice best;
    const WordPawExpectation& expectation = pawExpectations[pawIndex];
    int detectedThreeDots = 0;
    int detectedAttachedAlefHamzaAbove = 0;
    for (int position = start; position < end; ++position) {
      const ComponentFeature& feature = features[order[position]];
      if (feature.evidence.detectedThreeDotsAboveComponentCount == 1)
        ++detectedThreeDots;
      if (feature.evidence.attachedAlefHamzaAboveProbability >= 0.82)
        ++detectedAttachedAlefHamzaAbove;
    }
    const int expectedThreeDots =
        expectation.expectedMarks[static_cast<std::size_t>(
            MarkClass::DotAboveThree)];
    const int threeDotsReduction =
        std::min(expectedThreeDots, detectedThreeDots);
    const int expectedHamzaAbove =
        expectation.expectedMarks[static_cast<std::size_t>(
            MarkClass::HamzaAbove)];
    const bool expectsQuranicSign =
        expectation.expectedMarks[static_cast<std::size_t>(
            MarkClass::QuranicSign)] > 0;
    const bool isDigitText = textIsOffTheLine(expectation.text);
    const int attachedAlefHamzaReduction =
        std::min(expectedHamzaAbove, detectedAttachedAlefHamzaAbove);
    const int expectedMarkComponents = std::max(
        0, expectation.expectedMarkPaths - threeDotsReduction -
               attachedAlefHamzaReduction);

    auto markInventoryCost = [&](int bodyPosition) {
      if (!hasMarkClassEvidence) return 0.0;
      double inventoryDifference = 0.0;
      for (std::size_t mark = 0; mark < kMarkClassCount; ++mark) {
        if (static_cast<MarkClass>(mark) == MarkClass::Other) continue;
        double observed = 0.0;
        for (int position = start; position < end; ++position) {
          if (position == bodyPosition) continue;
          observed += features[order[position]]
                          .evidence.markClassProbabilities[mark];
        }
        inventoryDifference += std::abs(
            observed - expectation.expectedMarks[mark]);
      }
      return 2.0 * inventoryDifference;
    };

    auto markCountCost = [&](int observedMarkComponents) {
      const int missing = std::max(
          0, expectedMarkComponents - observedMarkComponents);
      const int extra = std::max(
          0, observedMarkComponents - expectedMarkComponents);
      return 1.50 * missing + 1.25 * extra;
    };

    for (int bodyPosition = start; bodyPosition < end; ++bodyPosition) {
      const int bodyLabel = order[bodyPosition];
      double cost = features[bodyLabel].bodyCost;
      // Whether the baseline and template costs below say anything about this
      // PAW at all. See the note on ayah markers before the baseline tests.
      const bool writtenOnTheLine = !expectsQuranicSign && !isDigitText;
      // Reviewed evidence that this component is a letter body outranks every
      // rule that called it a mark: those rules are proxies -- compactness,
      // solidity, ink near the line -- for a question the bank answers
      // directly, and where they disagree with reviewed data it is the
      // proxies that are wrong. The preliminary classifier's verdict goes
      // with them, since it is binary and would otherwise keep charging
      // -log(0.05) = 3.0 to a component just identified as a letter.
      const bool bodyTemplateMatch =
          writtenOnTheLine && features[bodyLabel].bodyTemplateMatch;
      if (bodyTemplateMatch)
        cost = -std::log(std::clamp(
            std::max(features[bodyLabel].bodyProbability,
                     features[bodyLabel].evidence.bodyTemplateProbability),
            minimumProbability, 1.0));
      if (!bodyTemplateMatch && features[bodyLabel].compactFloatingMark)
        cost += 1.25;
      // A component the reviewed template bank has named is a mark, and using
      // it as a body should cost more than any structural explanation of why a
      // body is missing -- more than two PAWs sharing one (1.50), more than a
      // handoff (1.75), more than admitting the PAW has no body (2.25). It did
      // not: the flag above is all a template match earned, and 1.25 is
      // cheaper than all three. In page 518 l08_w04 زَوْجِۭ the Zain and Waw
      // are fused into one component, and the Waw was given the kasra of جِ as
      // its body for 1.02 -- a stroke the bank matches to kasratan at 0.875 --
      // because that undercut every honest account of the missing body. The
      // off-baseline penalties below do not catch it: a kasra is drawn across
      // the line, not floating above it like page 382's fatha, so it has ink
      // in the band. Still a cost and not a veto, so a genuine oddity can
      // outvote it when nothing else is available.
      // Ayah markers and their digits are exempt for the same reason the
      // baseline costs below exempt them: nothing in them is written on a
      // line, the bank has no template for a digit, and applying the penalty
      // only reshuffles which small component each digit takes -- 19 words in
      // the sample, one of which left ۝ with no body at all.
      if (writtenOnTheLine && !bodyTemplateMatch &&
          features[bodyLabel].templateMarkProbability >=
              templateMarkConfidence &&
          !templateMarkClassIsThisPawsLetter(
              features[bodyLabel].templateMarkClass, expectation.text))
        cost += 2.0;
      if (expectsQuranicSign &&
          features[bodyLabel].evidence.detectedSajdaComponentCount > 0) {
        // U+06E9 is a two-component mark. Its components are deliberately
        // large and body-like, so protect the actual Arabic body from the
        // same component-count ambiguity that originally selected the inner
        // sajda shape as the Alef.
        constexpr double sajdaBodyPenalty = 7.0;
        cost += sajdaBodyPenalty;
      }
      // A PAW body is written on the line. A component that neither meets the
      // baseline nor carries body-scale ink is a mark, and letting one into
      // the body slot is how a single merged pair of PAWs corrupts a whole
      // word: in page 382 l09_w00 the Reh and Waw touch, leaving two body
      // components for three PAWs, and the DP satisfied `سِيَ` with the
      // word's long fatha -- 573 px above the baseline and a third of the
      // ink of a real body. This is the same test
      // `appendHeuristicClassification` already uses to call a component a
      // body, applied here as a cost rather than a veto so that a genuine
      // oddity can still outvote it when nothing else is available.
      // `touchesBaseline` is a bounding-box test and a long mark can graze the
      // band from above: page 382 l09_w00's fatha is 1075x573 with its lowest
      // ink 20 px clear of the line, so its box overlaps and it reads as a
      // body. Ink at the baseline is the sharper question, and it is the one
      // that matches how the hand works.
      // Ayah markers and their digits are not written on a line at all -- the
      // digits sit inside the circle -- so the baseline tests say nothing
      // about them and applying these costs only reshuffles which small
      // component each digit takes. Measured: without this guard the rule
      // changed 17 words on eight pages, and 12 of them were ayah numbers.
      // Both baseline costs below ask whether a component is where a body is
      // written, as a stand-in for whether it is one. A body template has
      // already said that it is, so the stand-in has nothing left to add --
      // and the case it would misjudge is exactly the one the bank exists
      // for, a small letter whose ink stops short of the line.
      // An upright says the same thing the body bank does, for a shape no
      // bank can hold: an Alef has no counter to recognize it by. Both costs
      // are guarded by area alone, and area does not notice an upright -- the
      // Alef of إِ in page 433 l12_w04 is 120x692 and only 3.87 qalam squared,
      // so stopping 77 px above a sloping line costs it 9.0 and it loses its
      // own body slot to the hamza beneath it.
      // Tall alone is not enough: the waqf signs are large letter-shaped
      // marks, and waqf_jeem reaches 9.18 qalam tall at its ninetieth
      // percentile, waqf_qaf 7.48. Slenderness is what no mark has. Stated
      // as a height it also needs the page's own qalam, which is not here --
      // the nominal 100 px used for areaNorm is up to a tenth out, and that
      // was enough to lose the Alef of ٱلْجَحِيمِ on page 467 l14_w07 by 26 px
      // and hand its PAW's body slot away. Aspect needs no such calibration:
      // an Alef runs above five, and the most slender mark in the corpus,
      // small_meem, only reaches 2.34.
      const bool uprightStroke =
          features[bodyLabel].bbox.height >=
              3.5 * features[bodyLabel].bbox.width &&
          features[bodyLabel].bbox.height >= 4.0 * scale;
      const bool positionSaysNothing =
          !writtenOnTheLine || bodyTemplateMatch || uprightStroke;
      if (!positionSaysNothing && !features[bodyLabel].hasBaselineInk &&
          !features[bodyLabel].touchesBaseline &&
          features[bodyLabel].areaNorm <= 10.0) {
        constexpr double floatingBodyPenalty = 6.0;
        cost += floatingBodyPenalty;
      }
      if (!positionSaysNothing && !features[bodyLabel].hasBaselineInk &&
          features[bodyLabel].areaNorm <= 10.0) {
        // Weaker, because a body whose ink stops short of the line is rare
        // but real -- a final Reh sits below it, an isolated Alef can end
        // above it on a sloping line.
        constexpr double offBaselineInkPenalty = 3.0;
        cost += offBaselineInkPenalty;
      }
      // A compact baseline-start rule can still leave a small mark with a
      // body probability equal to a real glyph. Use component size only as
      // a soft final-DP prior so a taller/larger Alef wins this ambiguity while
      // strong preliminary body/mark probabilities remain dominant.
      cost -= 0.35 * std::log1p(features[bodyLabel].areaNorm);
      for (int position = start; position < end; ++position)
        if (position != bodyPosition)
          cost += features[order[position]].markCost;

      const int observedMarkComponents = end - start - 1;
      cost += markCountCost(observedMarkComponents);
      cost += 0.75 *
              std::abs(expectation.expectedBodyComponents - 1);
      double observedSmallLetters = 0.0;
      for (int position = start; position < end; ++position)
        if (position != bodyPosition)
          observedSmallLetters +=
              features[order[position]].evidence.smallLetterProbability;
      cost += 0.50 * std::abs(
                         observedSmallLetters - expectation.expectedSmallLetterPaths);
      cost += markInventoryCost(bodyPosition);

      if (isWideConnectedAlefBody(bodyPosition, end, pawIndex) ||
          isPreviousWordBoundaryAlefBody(bodyPosition, end, pawIndex)) {
        // Do not let a single connected baseline blob satisfy the first Alef
        // body slot when it visibly reaches the preceding word boundary. The
        // missing-body alternative below records that the Alef has no
        // separable body in this word.
        cost += 4.0;
      }

      // The preliminary classifier deliberately keeps its baseline-start
      // rule. It can therefore give a compact mark and the real body equal
      // body probabilities. In that tie, the larger connected component is
      // the safer body candidate; keep the size preference out of the
      // preliminary probabilities so it only resolves this final-DP tie.
      const bool lowerCost = cost < best.cost - 1e-9;
      const bool equalCost = std::abs(cost - best.cost) <= 1e-9;
      if (lowerCost ||
          (equalCost && features[bodyLabel].area > best.bodyArea))
        best = {cost, bodyPosition, expectedMarkComponents,
                observedMarkComponents, features[bodyLabel].area, false, -1};
    }

    // Do not force a strongly mark-like component into the body slot merely
    // because the neighboring body is connected to this PAW. A missing body
    // is an explicit, lower-cost alternative when a high-confidence body
    // blob overlaps the segment boundary. This is preferable to exporting a
    // fatha/kasra/dot as a PAW body; the connected blob remains available in
    // the adjacent PAW and the missing-body condition is visible through the
    // empty bodyComponentIndices array.
    bool allComponentsPreferMark = end > start;
    for (int position = start; position < end; ++position) {
      const ComponentFeature& component = features[order[position]];
      allComponentsPreferMark =
          allComponentsPreferMark &&
          component.markProbability > component.bodyProbability;
    }
    if (expectation.expectedBodyComponents > 0 &&
        end - start == 1 && allComponentsPreferMark &&
        hasAdjacentBodyOverlap(start, end)) {
      constexpr double missingBodyPenalty = 2.25;
      double cost = missingBodyPenalty;
      for (int position = start; position < end; ++position)
        cost += features[order[position]].markCost;
      const int observedMarkComponents = end - start;
      cost += markCountCost(observedMarkComponents);
      cost += 0.50 * std::abs(
                         std::accumulate(
                             order.begin() + start, order.begin() + end, 0.0,
                             [&](double total, int label) {
                               return total + features[label].evidence.smallLetterProbability;
                             }) -
                         expectation.expectedSmallLetterPaths);
      cost += markInventoryCost(-1);
      if (cost < best.cost - 1e-9)
        best = {cost, -1, expectedMarkComponents, observedMarkComponents, 0,
                false, -1};
    }

    // In some scans the body component of the next subword is the first
    // component in the Alef PAW's RTL range. It is a wide connected baseline
    // blob reaching the preceding-word boundary, while the Alef's wasla (or
    // another mark) follows it in component order. Allow the DP to hand that
    // one physical blob to the next PAW; a contiguous range alone cannot do
    // this without also moving the Alef mark.
    if (firstPawConnectedToPreviousWord && pawCount > 1 && pawIndex == 0 &&
        end > start &&
        textStartsWithAlefBase(expectation.text)) {
      int handoffBodyPosition = -1;
      for (int position = start; position < end; ++position) {
        if (isPreviousWordBoundaryAlefBody(position, end, pawIndex)) {
          handoffBodyPosition = position;
          break;
        }
      }
      if (handoffBodyPosition >= 0) {
        constexpr double handoffMissingBodyPenalty = 1.75;
        double cost = handoffMissingBodyPenalty;
        for (int position = start; position < end; ++position) {
          if (position != handoffBodyPosition)
            cost += features[order[position]].markCost;
        }
        const int observedMarkComponents = end - start - 1;
        cost += markCountCost(observedMarkComponents);
        cost += 0.50 * std::abs(
                           std::accumulate(
                               order.begin() + start, order.begin() + end, 0.0,
                               [&](double total, int label) {
                                 if (label == order[handoffBodyPosition]) return total;
                                 return total + features[label].evidence.smallLetterProbability;
                               }) -
                           expectation.expectedSmallLetterPaths);
        cost += markInventoryCost(handoffBodyPosition);
        if (cost < best.cost - 1e-9) {
          best = {cost, -1, expectedMarkComponents, observedMarkComponents,
                  0, true, handoffBodyPosition};
        }
      }
    }

    // The whole first PAW can be in the previous word, not just its body. A
    // wasla sits *on* its Alef, so when that Alef is drawn into the word
    // before it the wasla goes with it, and both are labelled there: page 456
    // l10_w07 ٱلْمِهَادُ is a six-component word whose text asks for eight, and
    // the two it is missing are the Alef and its wasla, both inside فَبِئْسَ.
    // The handoff below models only the body crossing over and keeps a mark
    // for this PAW, so the word still reads as two components short and the
    // partition slides: لْمِهَا was given the Dal as one of its marks and دُ
    // was left with the damma and no body. Saying the PAW is simply not here
    // costs one penalty and leaves the rest of the word alone.
    if (firstPawConnectedToPreviousWord && pawIndex == 0 && start == end &&
        textStartsWithAlefBase(expectation.text)) {
      constexpr double absentFirstPawPenalty = 1.75;
      if (absentFirstPawPenalty < best.cost - 1e-9)
        best = {absentFirstPawPenalty, -1, expectedMarkComponents, 0, 0, true,
                -1};
    }

    // A standalone Alef at the beginning of a word can be physically merged
    // toward the preceding word. In that case keeping the first component as
    // a fake Alef body is worse than explicitly reporting a missing separable
    // Alef body. The connected component cannot be split by connected-
    // component labeling, so preserve the boundary condition in the flag.
    if (expectation.expectedBodyComponents > 0 &&
        firstPawConnectedToPreviousWord && pawIndex == 0 &&
        textStartsWithAlefBase(expectation.text) && end > start &&
        allComponentsPreferMark &&
        hasWideConnectedAlefBodyAhead(end, pawIndex)) {
      constexpr double connectedAlefMissingBodyPenalty = 1.75;
      double cost = connectedAlefMissingBodyPenalty;
      for (int position = start; position < end; ++position)
        cost += features[order[position]].markCost;
      const int observedMarkComponents = end - start;
      cost += markCountCost(observedMarkComponents);
      cost += 0.50 * std::abs(
                         std::accumulate(
                             order.begin() + start, order.begin() + end, 0.0,
                             [&](double total, int label) {
                               return total + features[label].evidence.smallLetterProbability;
                             }) -
                         expectation.expectedSmallLetterPaths);
      cost += markInventoryCost(-1);
      if (cost < best.cost - 1e-9)
        best = {cost, -1, expectedMarkComponents, observedMarkComponents, 0,
                true, -1};
    }

    // Two PAWs in one component. Connected-component labeling cannot separate
    // ink that touches, so when the scan is a component short the DP's
    // one-range-per-PAW partition has to satisfy some PAW with a neighbour's
    // ink -- and because the ranges are contiguous and ordered, that error
    // then shifts every assignment after it. Saying "this PAW's body is the
    // previous PAW's, and one of them will have to be cut" keeps the rest of
    // the word intact and puts the failure where it belongs.
    if (sharedBodyLabel >= 0 && expectation.expectedBodyComponents > 0) {
      const ComponentFeature& shared = features[sharedBodyLabel];
      // A blob holding two PAWs spans two entry points on the line, so it is
      // wider than it is tall and it meets the baseline. Both of those are
      // asked of its ink inside the writing band, not of its bounding box: an
      // ascender answers neither question and distorts both. On page 225
      // l02_w09 طَرَدتُّهُمْۚ the Ta, Reh and Dal arrive as one component and the
      // Taa's upright makes it 978 x 1043 -- taller than wide by 65 px -- so
      // the box says no and the Dal alone was charged with the whole of طرد.
      const cv::Rect& band =
          shared.baselineBandBox.empty() ? shared.bbox : shared.baselineBandBox;
      bool plausible = band.width > band.height && shared.hasBaselineInk;
      // A mark is written over the letter it belongs to. This PAW's own marks
      // sitting inside the blob's horizontal span is direct evidence that the
      // blob carries this PAW's body as well.
      for (int position = start; position < end && plausible; ++position)
        if (horizontalOverlapRatio(features[order[position]].bbox,
                                   shared.bbox) < 0.50)
          plausible = false;
      // Being over the blob is not enough on its own: a PAW's marks are over
      // its *own* body too, so that test is satisfied by a blob that is only
      // this one PAW. What says the blob carries two is ink reaching well
      // past the marks of one of them -- to the right of this PAW's marks,
      // where the previous PAW is written, or to the left of the previous
      // PAW's, where this one is. Without it, page 3 l05_w01 reads the Noon
      // of يشعرون -- 476 x 452 px, its own two marks over it -- as holding
      // the Waw as well, and the Waw is then cut out of the Noon's arc.
      // Measured on the same band, and for the same reason: half of 1043 is
      // 521 px of overhang demanded of a word whose bodies are a third of
      // that tall.
      const int reach = band.height / 2;
      bool reachesPast = false;
      int marksLeft = std::numeric_limits<int>::max();
      int marksRight = std::numeric_limits<int>::min();
      for (int position = start; position < end; ++position) {
        marksLeft = std::min(marksLeft, features[order[position]].bbox.x);
        marksRight = std::max(marksRight,
                              features[order[position]].bbox.x +
                                  features[order[position]].bbox.width);
      }
      if (marksLeft <= marksRight &&
          shared.bbox.x + shared.bbox.width - marksRight >= reach)
        reachesPast = true;
      marksLeft = std::numeric_limits<int>::max();
      marksRight = std::numeric_limits<int>::min();
      for (int position = sharedRangeStart; position < start; ++position) {
        if (position == sharedBodyPosition) continue;
        marksLeft = std::min(marksLeft, features[order[position]].bbox.x);
        marksRight = std::max(marksRight,
                              features[order[position]].bbox.x +
                                  features[order[position]].bbox.width);
      }
      if (marksLeft <= marksRight && marksLeft - shared.bbox.x >= reach)
        reachesPast = true;
      plausible = plausible && reachesPast;
      // And this PAW must hold nothing of its own that could be a body. Page
      // 363 l07_w05 كَثِيرࣰا ends in a free-standing Alef, 287 x 686 px, with
      // the كثير blob sweeping left underneath it -- enough to satisfy both
      // tests above, but a PAW that has its own body has no reason to borrow
      // one. Same body/mark preference the missing-body alternative uses.
      for (int position = start; position < end && plausible; ++position) {
        const ComponentFeature& component = features[order[position]];
        if (component.markProbability <= component.bodyProbability)
          plausible = false;
      }
      if (plausible) {
        // Below the 2.25 of a missing body: where both readings are open,
        // a body that is present but joined to its neighbour is a better
        // account of the ink than a body that is absent.
        constexpr double sharedBodyPenalty = 1.50;
        double cost = sharedBodyPenalty;
        for (int position = start; position < end; ++position)
          cost += features[order[position]].markCost;
        const int observedMarkComponents = end - start;
        cost += markCountCost(observedMarkComponents);
        cost += 0.50 * std::abs(
                           std::accumulate(
                               order.begin() + start, order.begin() + end, 0.0,
                               [&](double total, int label) {
                                 return total + features[label].evidence.smallLetterProbability;
                               }) -
                           expectation.expectedSmallLetterPaths);
        cost += markInventoryCost(-1);
        if (cost < best.cost - 1e-9)
          best = {cost, -1, expectedMarkComponents, observedMarkComponents, 0,
                  false, -1, true};
      }
    }
    return best;
  };

  auto boundaryCost = [&](int previousPosition, int currentPosition) {
    if (previousPosition < 0 || currentPosition >= componentCount) return 0.0;
    const cv::Rect& previous =
        features[order[previousPosition]].bbox;
    const cv::Rect& current = features[order[currentPosition]].bbox;
    const int overlap = std::max(
        0, current.x + current.width - previous.x + 1);
    const int gap = std::max(
        0, previous.x - (current.x + current.width) - 1);
    const int denominator = std::max(
        1, std::min(previous.width, current.width));
    return 0.90 * overlap / double(denominator) -
           1.10 * std::log(1.0 + gap / 150.0);
  };

  struct BackPointer {
    int start = -1;
    PawChoice choice;
  };
  std::vector<std::vector<double>> dp(
      pawCount + 1, std::vector<double>(componentCount + 1, infinity));
  std::vector<std::vector<BackPointer>> back(
      pawCount + 1, std::vector<BackPointer>(componentCount + 1));
  dp[0][0] = 0.0;

  // `start == end` -- a PAW holding no component of its own -- is reachable
  // only through the shared-body alternative, which is the only one that
  // scores a body-less, mark-less range at all. The bounds are opened for it
  // rather than the one-component-per-PAW floor being dropped: a word that
  // cannot hold a merge keeps the old, smaller search exactly, so the DP pays
  // for this only where it might find something.
  for (int paw = 1; paw <= pawCount; ++paw) {
    for (int end = componentsAreShort ? 0 : paw; end <= componentCount; ++end) {
      const int lastStart = componentsAreShort ? end : end - 1;
      for (int start = componentsAreShort ? 0 : paw - 1; start <= lastStart;
           ++start) {
        if (!std::isfinite(dp[paw - 1][start])) continue;
        // Row paw-1 is complete before this row begins, so its back pointer is
        // final and the previous PAW's body is known here. A PAW that is
        // itself sharing has no body of its own to lend, which stops a whole
        // run of PAWs from being hung on one blob.
        int sharedBodyLabel = -1;
        int sharedRangeStart = start;
        int sharedBodyPosition = -1;
        if (componentsAreShort && paw >= 2) {
          const PawChoice& previous = back[paw - 1][start].choice;
          if (previous.bodyOrder >= 0 && !previous.sharesBodyWithPrevious) {
            sharedBodyLabel = order[previous.bodyOrder];
            sharedRangeStart = back[paw - 1][start].start;
            sharedBodyPosition = previous.bodyOrder;
          }
        }
        const PawChoice choice =
            segmentChoice(start, end, paw - 1, sharedBodyLabel,
                          sharedRangeStart, sharedBodyPosition);
        if (!std::isfinite(choice.cost)) continue;
        // The boundary cost asks whether two neighbouring ranges are
        // spatially separate. Between a PAW and the one it shares a body with
        // the question is void: the boundary falls inside a single component,
        // and this PAW's own marks are written over the shared blob, so the
        // overlap it measures is the expected geometry rather than evidence
        // against the split. Charging it here would penalize the merge twice.
        const double candidate =
            dp[paw - 1][start] + choice.cost +
            (choice.sharesBodyWithPrevious ? 0.0
                                           : boundaryCost(start - 1, start));
        if (candidate < dp[paw][end]) {
          dp[paw][end] = candidate;
          back[paw][end] = {start, choice};
        }
      }
    }
  }

  if (!std::isfinite(dp[pawCount][componentCount])) {
    appendHeuristicClassification();
    finalizeTotals();
    return out;
  }

  struct ChosenPaw {
    int start = 0;
    int end = 0;
    PawChoice choice;
  };
  std::vector<ChosenPaw> chosen(pawCount);
  int end = componentCount;
  for (int paw = pawCount; paw >= 1; --paw) {
    const BackPointer& pointer = back[paw][end];
    if (pointer.start < 0) {
      out = MarkSegmentationResult();
      appendHeuristicClassification();
      finalizeTotals();
      return out;
    }
    chosen[paw - 1] = {pointer.start, end, pointer.choice};
    end = pointer.start;
  }

  const int handoffBodyLabel =
      chosen.empty() || chosen[0].choice.handoffBodyPosition < 0
          ? -1
          : order[chosen[0].choice.handoffBodyPosition];

  // The component order is normally a reliable RTL order, but a connected
  // body can extend past a detached first-PAW body.  Its bounding box then
  // contains the first-PAW body and ends farther to the right, so the normal
  // right-edge sort puts the second-PAW body before the first-PAW body.  This
  // is the pattern behind both a wide connected body followed by a Waw (page
  // 105) and a connected word body followed by a detached Alef (page 49).
  // Detect the pattern after the DP has selected the candidate ranges and
  // exchange only the two body components.  Marks are restored separately
  // when the body exchange leaves a mark deficit in the first PAW.
  int invertedFirstPawIndex = -1;
  int invertedSecondPawIndex = -1;
  int invertedFirstBodyLabel = -1;
  int invertedSecondBodyLabel = -1;
  if (pawCount >= 2) {
    auto choiceBodyLabel = [&](const ChosenPaw& paw) {
      return paw.choice.bodyOrder >= 0 ? order[paw.choice.bodyOrder] : -1;
    };
    double bestInversionScore = std::numeric_limits<double>::infinity();

    auto isRightEdgeBodyInversion = [&](int outerLabel, int innerLabel) {
      if (outerLabel < 0 || innerLabel < 0 || outerLabel == innerLabel)
        return false;
      const ComponentFeature& outer = features[outerLabel];
      const ComponentFeature& inner = features[innerLabel];
      if (outer.bodyProbability < 0.75 || inner.bodyProbability < 0.75)
        return false;
      if (outer.bbox.width < 1.5 * inner.bbox.width ||
          inner.bbox.height < 0.20 * outer.bbox.height)
        return false;
      const int outerRight = outer.bbox.x + outer.bbox.width;
      const int innerRight = inner.bbox.x + inner.bbox.width;
      // The right-edge relationship is the main signal. The DP sorts
      // components by descending right edge, but a connected body can
      // overhang the detached body on either side.  Use the absolute edge
      // distance instead of requiring containment or a particular ordering.
      const int rightEdgeDifference = std::abs(outerRight - innerRight);
      if (rightEdgeDifference > std::max(200, outer.bbox.width / 4))
        return false;

      // A detached mark can also satisfy the baseline-start body rule.  The
      // recovered first-letter body in this boundary pattern ends close to
      // the baseline; reject a candidate that extends substantially below
      // it, which is the geometry of a kasra-like false body.
      const int innerBottomFromBaseline =
          inner.bbox.y + inner.bbox.height - baselineY;
      if (innerBottomFromBaseline > 0.30 * outer.bbox.height)
        return false;

      // Keep the same-line test so a wide waqf/digit pair with nearby right
      // edges is not mistaken for a connected body crossing. This is an
      // overlap test, not a containment requirement.
      const int outerBottom = outer.bbox.y + outer.bbox.height;
      const int innerBottom = inner.bbox.y + inner.bbox.height;
      const int verticalOverlap =
          std::max(0, std::min(outerBottom, innerBottom) -
                          std::max(outer.bbox.y, inner.bbox.y));
      if (verticalOverlap < 0.55 * inner.bbox.height ||
          inner.bbox.x + inner.bbox.width / 2.0 <
              outer.bbox.x + 0.65 * outer.bbox.width)
        return false;
      return true;
    };

    for (int firstPaw = 0; firstPaw + 1 < pawCount; ++firstPaw) {
      if (pawExpectations[firstPaw].expectedBodyComponents != 1 ||
          pawExpectations[firstPaw + 1].expectedBodyComponents != 1)
        continue;
      const ChosenPaw& firstChosen = chosen[firstPaw];
      const ChosenPaw& secondChosen = chosen[firstPaw + 1];
      const std::array<int, 2> selectedBodyLabels = {
          choiceBodyLabel(firstChosen),
          firstPaw == 0 ? handoffBodyLabel : -1};
      for (const int outerLabel : selectedBodyLabels) {
        if (outerLabel < 0) continue;
        for (int position = secondChosen.start;
             position < secondChosen.end; ++position) {
          const int innerLabel = order[position];
          if (!isRightEdgeBodyInversion(outerLabel, innerLabel)) continue;
          const ComponentFeature& outer = features[outerLabel];
          const ComponentFeature& inner = features[innerLabel];
          const double edgeScore =
              std::abs(outer.bbox.x + outer.bbox.width - inner.bbox.x -
                       inner.bbox.width) /
              double(std::max(1, outer.bbox.width));
          if (edgeScore < bestInversionScore) {
            bestInversionScore = edgeScore;
            invertedFirstPawIndex = firstPaw;
            invertedSecondPawIndex = firstPaw + 1;
            invertedFirstBodyLabel = innerLabel;
            invertedSecondBodyLabel = outerLabel;
          }
        }
      }
    }
  }

  // A long initial Kaf can be sorted into the preceding PAW's component
  // range.  The range then contains two body-looking components: one is the
  // true first-PAW body and the wide component belongs to the following PAW.
  // The following PAW may have a detached contour that the DP selected as
  // its body, so looking only at the selected body in that range misses this
  // case.  Compare the two right edges directly; containment is not required
  // because the strokes can overhang in either vertical direction.
  int nestedFirstPawIndex = -1;
  int nestedSecondPawIndex = -1;
  int nestedFirstBodyLabel = -1;
  int nestedSecondBodyLabel = -1;
  if (invertedFirstBodyLabel < 0 && pawCount >= 2) {
    auto choiceBodyLabel = [&](const ChosenPaw& paw) {
      return paw.choice.bodyOrder >= 0 ? order[paw.choice.bodyOrder] : -1;
    };

    auto isNestedLongBodyPair = [&](int longBodyLabel,
                                    int firstBodyLabel) {
      if (longBodyLabel < 0 || firstBodyLabel < 0 ||
          longBodyLabel == firstBodyLabel)
        return false;
      const ComponentFeature& longBody = features[longBodyLabel];
      const ComponentFeature& firstBody = features[firstBodyLabel];
      if (longBody.bodyProbability < 0.75 ||
          firstBody.bodyProbability < 0.75 ||
          longBody.bodyProbability < longBody.markProbability ||
          firstBody.bodyProbability < firstBody.markProbability ||
          longBody.compactFloatingMark || firstBody.compactFloatingMark)
        return false;
      if (longBody.bbox.width < 1.5 * firstBody.bbox.width ||
          firstBody.bbox.height < 0.20 * longBody.bbox.height)
        return false;

      const int longRight = longBody.bbox.x + longBody.bbox.width;
      const int firstRight = firstBody.bbox.x + firstBody.bbox.width;
      const int firstOverhang = std::abs(firstRight - longRight);
      if (firstOverhang > std::max(200, longBody.bbox.width / 4))
        return false;

      const int firstBottomFromBaseline =
          firstBody.bbox.y + firstBody.bbox.height - baselineY;
      if (firstBottomFromBaseline > 0.28 * longBody.bbox.height)
        return false;

      const int longBottom = longBody.bbox.y + longBody.bbox.height;
      const int firstBottom = firstBody.bbox.y + firstBody.bbox.height;
      const int verticalOverlap =
          std::max(0, std::min(longBottom, firstBottom) -
                          std::max(longBody.bbox.y, firstBody.bbox.y));
      if (verticalOverlap < 0.55 * firstBody.bbox.height ||
          horizontalOverlapRatio(longBody.bbox, firstBody.bbox) < 0.20 ||
          firstBody.bbox.x + firstBody.bbox.width / 2.0 <
              longBody.bbox.x + 0.65 * longBody.bbox.width)
        return false;
      return true;
    };

    double bestNestedScore = std::numeric_limits<double>::infinity();
    for (int firstPaw = 0; firstPaw + 1 < pawCount; ++firstPaw) {
      if (pawExpectations[firstPaw].expectedBodyComponents != 1 ||
          pawExpectations[firstPaw + 1].expectedBodyComponents != 1)
        continue;
      const int longBodyLabel = choiceBodyLabel(chosen[firstPaw]);
      if (longBodyLabel < 0) continue;

      for (int position = chosen[firstPaw].start;
           position < chosen[firstPaw].end; ++position) {
        const int firstBodyLabel = order[position];
        if (!isNestedLongBodyPair(longBodyLabel, firstBodyLabel)) continue;

        const ComponentFeature& longBody = features[longBodyLabel];
        const ComponentFeature& firstBody = features[firstBodyLabel];
        const double edgeScore =
            std::abs(firstBody.bbox.x + firstBody.bbox.width -
                     longBody.bbox.x - longBody.bbox.width) /
            double(std::max(1, longBody.bbox.width));
        if (edgeScore < bestNestedScore) {
          bestNestedScore = edgeScore;
          nestedFirstPawIndex = firstPaw;
          nestedSecondPawIndex = firstPaw + 1;
          nestedFirstBodyLabel = firstBodyLabel;
          nestedSecondBodyLabel = longBodyLabel;
        }
      }
    }
  }

  const bool hasBodyInversion = invertedFirstBodyLabel >= 0 &&
                                invertedSecondBodyLabel >= 0;
  const bool hasNestedBodyInversion = nestedFirstBodyLabel >= 0 &&
                                      nestedSecondBodyLabel >= 0;
  const bool hasAnyBodyInversion = hasBodyInversion ||
                                   hasNestedBodyInversion;
  const int correctedFirstPawIndex =
      hasBodyInversion ? invertedFirstPawIndex : nestedFirstPawIndex;
  const int correctedSecondPawIndex =
      hasBodyInversion ? invertedSecondPawIndex : nestedSecondPawIndex;
  const int correctedFirstBodyLabel =
      hasBodyInversion ? invertedFirstBodyLabel : nestedFirstBodyLabel;
  const int correctedSecondBodyLabel =
      hasBodyInversion ? invertedSecondBodyLabel : nestedSecondBodyLabel;
  auto labelInChosenRange = [&](int pawIndex, int label) {
    if (pawIndex < 0 || pawIndex >= pawCount || label < 0) return false;
    for (int position = chosen[pawIndex].start;
         position < chosen[pawIndex].end; ++position)
      if (order[position] == label) return true;
    return false;
  };
  std::vector<int> invertedFirstMarkLabels;
  if (hasAnyBodyInversion && correctedFirstPawIndex >= 0 &&
      correctedSecondPawIndex >= 0) {
    const int missingFirstMarks = std::max(
        0, chosen[correctedFirstPawIndex].choice.expectedMarkComponents -
               chosen[correctedFirstPawIndex].choice.observedMarkComponents +
               (labelInChosenRange(correctedFirstPawIndex,
                                   correctedFirstBodyLabel) &&
                        correctedFirstBodyLabel !=
                            (chosen[correctedFirstPawIndex].choice.bodyOrder >= 0
                                 ? order[chosen[correctedFirstPawIndex]
                                             .choice.bodyOrder]
                                 : -1)
                    ? 1
                    : 0));
    if (missingFirstMarks > 0) {
      const ComponentFeature& firstBody =
          features[correctedFirstBodyLabel];
      std::vector<std::pair<double, int>> markCandidates;
      for (int position = chosen[correctedSecondPawIndex].start;
           position < chosen[correctedSecondPawIndex].end; ++position) {
        const int label = order[position];
        if (label == correctedFirstBodyLabel ||
            label == correctedSecondBodyLabel)
          continue;
        const ComponentFeature& mark = features[label];
        if (mark.bodyProbability >= 0.50 &&
            mark.bodyProbability >= mark.markProbability)
          continue;

        const double horizontalOverlap =
            horizontalOverlapRatio(mark.bbox, firstBody.bbox);
        const int horizontalGap =
            std::max(0, std::max(mark.bbox.x, firstBody.bbox.x) -
                            std::min(mark.bbox.x + mark.bbox.width,
                                     firstBody.bbox.x + firstBody.bbox.width));
        const int verticalGap =
            std::max(0, std::max(mark.bbox.y, firstBody.bbox.y) -
                            std::min(mark.bbox.y + mark.bbox.height,
                                     firstBody.bbox.y + firstBody.bbox.height));
        if (horizontalOverlap < 0.10 && horizontalGap > 160) continue;

        // Prefer a mark above or below the recovered body with horizontal
        // support. This keeps a nearby PAW1 mark from being moved merely
        // because it is in the same component range.
        const double score = 4.0 * horizontalOverlap -
                             horizontalGap / 200.0 - verticalGap / 500.0;
        markCandidates.emplace_back(score, label);
      }
      std::sort(markCandidates.begin(), markCandidates.end(),
                [](const auto& left, const auto& right) {
                  return left.first > right.first;
                });
      for (int index = 0;
           index < missingFirstMarks && index < static_cast<int>(markCandidates.size());
           ++index)
        invertedFirstMarkLabels.push_back(markCandidates[index].second);
    }
  }

  // The body the PAW before this one ended up with, for the shared-body
  // choice. Read from the loop rather than from `chosen`, so it is the label
  // actually exported after the inversion and handoff corrections.
  int previousPawBodyLabel = -1;
  for (int pawIndex = 0; pawIndex < pawCount; ++pawIndex) {
    const ChosenPaw& chosenPaw = chosen[pawIndex];
    WordPaw paw;
    paw.pawIndex = pawIndex;
    paw.text = pawExpectations[pawIndex].text;
    paw.expectedMarkComponents = chosenPaw.choice.expectedMarkComponents;
    paw.observedMarkComponents = chosenPaw.choice.observedMarkComponents;
    const bool pawHasBodyCorrection =
        hasAnyBodyInversion &&
        (pawIndex == correctedFirstPawIndex ||
         pawIndex == correctedSecondPawIndex);
    paw.bodyConnectedToPreviousWord =
        pawHasBodyCorrection ? false
                             : chosenPaw.choice.bodyConnectedToPreviousWord;
    paw.cost = chosenPaw.choice.cost;

    const bool receivesHandoff = !hasAnyBodyInversion && pawIndex == 1 &&
                                 handoffBodyLabel >= 0;
    const int inversionBodyLabel =
        hasAnyBodyInversion
            ? (pawIndex == correctedFirstPawIndex
                   ? correctedFirstBodyLabel
                   : (pawIndex == correctedSecondPawIndex
                          ? correctedSecondBodyLabel
                          : -1))
            : -1;
    // The blob this PAW shares with the one before it. The corrections above
    // move bodies between PAWs on their own terms, so a PAW they touch is not
    // the PAW the DP was describing and the share is dropped there.
    const int sharedBodyLabel =
        (chosenPaw.choice.sharesBodyWithPrevious && inversionBodyLabel < 0 &&
         !receivesHandoff && !hasAnyBodyInversion)
            ? previousPawBodyLabel
            : -1;
    paw.bodyMergedWithPrevious = sharedBodyLabel >= 0;

    const int bodyLabel =
        inversionBodyLabel >= 0
            ? inversionBodyLabel
        : receivesHandoff
            ? handoffBodyLabel
        : sharedBodyLabel >= 0
            ? sharedBodyLabel
            : (chosenPaw.choice.bodyOrder >= 0
                   ? order[chosenPaw.choice.bodyOrder]
                   : -1);

    const int externalBodyLabel =
        inversionBodyLabel >= 0
            ? inversionBodyLabel
            : (receivesHandoff ? handoffBodyLabel : sharedBodyLabel);
    if (externalBodyLabel >= 0) {
      const ComponentFeature& feature = features[externalBodyLabel];
      paw.componentLabels.push_back(externalBodyLabel);
      paw.bbox |= feature.bbox;
      const int sourceIndex = feature.evidence.sourceComponentIndex;
      if (sourceIndex >= 0) paw.componentIndices.push_back(sourceIndex);
      paw.bodyComponentLabels.push_back(externalBodyLabel);
      if (sourceIndex >= 0)
        paw.bodyComponentIndices.push_back(sourceIndex);
      // A shared component was already typed, and counted, by the PAW it is
      // shared from. Typing it again would double its ink in the word totals.
      if (sharedBodyLabel < 0)
        appendComponentType(externalBodyLabel, ComponentTypeEnum::Body);
      if (receivesHandoff && chosenPaw.choice.bodyOrder >= 0)
        ++paw.observedMarkComponents;
    }
    if (hasAnyBodyInversion && pawIndex == correctedFirstPawIndex) {
      for (const int label : invertedFirstMarkLabels) {
        const ComponentFeature& feature = features[label];
        paw.componentLabels.push_back(label);
        paw.bbox |= feature.bbox;
        const int sourceIndex = feature.evidence.sourceComponentIndex;
        if (sourceIndex >= 0) paw.componentIndices.push_back(sourceIndex);
        paw.markComponentLabels.push_back(label);
        if (sourceIndex >= 0)
          paw.markComponentIndices.push_back(sourceIndex);
        appendComponentType(label, feature.dotLike
                                       ? ComponentTypeEnum::Dot
                                       : ComponentTypeEnum::OtherMark);
      }
    }

    for (int position = chosenPaw.start;
         position < chosenPaw.end; ++position) {
      const int label = order[position];
      if (hasAnyBodyInversion &&
          ((pawIndex == correctedFirstPawIndex &&
            label == correctedSecondBodyLabel) ||
           (pawIndex == correctedSecondPawIndex &&
            (label == correctedFirstBodyLabel ||
             std::find(invertedFirstMarkLabels.begin(),
                       invertedFirstMarkLabels.end(), label) !=
                 invertedFirstMarkLabels.end()))))
        continue;
      if (label == externalBodyLabel) continue;
      const ComponentFeature& feature = features[label];
      paw.componentLabels.push_back(label);
      paw.bbox |= feature.bbox;
      const int sourceIndex = feature.evidence.sourceComponentIndex;
      if (sourceIndex >= 0) paw.componentIndices.push_back(sourceIndex);
      if (label == bodyLabel) {
        paw.bodyComponentLabels.push_back(label);
        if (sourceIndex >= 0)
          paw.bodyComponentIndices.push_back(sourceIndex);
        appendComponentType(label, ComponentTypeEnum::Body);
      } else {
        paw.markComponentLabels.push_back(label);
        if (sourceIndex >= 0)
          paw.markComponentIndices.push_back(sourceIndex);
        appendComponentType(label, feature.dotLike
                                       ? ComponentTypeEnum::Dot
                                       : ComponentTypeEnum::OtherMark);
      }
    }

    std::vector<ExpectedMarkInstance> expectedMarks =
        pawExpectations[pawIndex].expectedMarkInstances;
    if (expectedMarks.empty()) {
      for (std::size_t mark = 0; mark < kMarkClassCount; ++mark)
        for (int count = 0;
             count < pawExpectations[pawIndex].expectedMarks[mark]; ++count)
          expectedMarks.push_back(
              {static_cast<MarkClass>(mark), -1});
    }

    // Hamza-above can be physically attached to an Alef body.  In that form
    // the remaining detached component is often the fatha/damma, but the
    // text expectation still contains both logical marks.  The component
    // recognizer marks the reliable cases explicitly; for the same tall,
    // narrow Alef body, a one-component mark deficit supplies a conservative
    // final-DP fallback.  Remove the attached Hamza only from the mark
    // assignment; the body/mark component ownership remains unchanged.
    if (!expectedMarks.empty() && paw.bodyComponentLabels.size() == 1) {
      const ComponentFeature& bodyFeature =
          features[paw.bodyComponentLabels.front()];
      const bool recognizedAttachedHamza =
          bodyFeature.evidence.attachedAlefHamzaAboveProbability >= 0.82;
      const bool tallAlefBody =
          bodyFeature.bodyProbability >= 0.50 &&
          bodyFeature.bbox.height >= 500 &&
          bodyFeature.bbox.height >= 2.50 * bodyFeature.bbox.width &&
          textContainsAlefWithHamzaAboveBase(paw.text);
      const bool markDeficit =
          paw.markComponentLabels.size() < expectedMarks.size();
      if (markDeficit && (recognizedAttachedHamza || tallAlefBody)) {
        const auto hamza = std::find_if(
            expectedMarks.begin(), expectedMarks.end(),
            [](const ExpectedMarkInstance& expected) {
              return expected.markClass == MarkClass::HamzaAbove;
            });
        if (hamza != expectedMarks.end()) {
          expectedMarks.erase(hamza);
          if (!recognizedAttachedHamza && paw.expectedMarkComponents > 0)
            --paw.expectedMarkComponents;
        }
      }
    }

    const bool hasExpectedHamza = std::any_of(
        expectedMarks.begin(), expectedMarks.end(),
        [](const ExpectedMarkInstance& expected) {
          return expected.markClass == MarkClass::HamzaAbove ||
                 expected.markClass == MarkClass::HamzaBelow;
        });
    const bool hasExpectedQuranicSign = std::any_of(
        expectedMarks.begin(), expectedMarks.end(),
        [](const ExpectedMarkInstance& expected) {
          return expected.markClass == MarkClass::QuranicSign;
        });
    const bool hasExpectedSmallAlef = std::any_of(
        expectedMarks.begin(), expectedMarks.end(),
        [](const ExpectedMarkInstance& expected) {
          return expected.markClass == MarkClass::SmallAlef;
        });
    const bool hasExpectedMaddah = std::any_of(
        expectedMarks.begin(), expectedMarks.end(),
        [](const ExpectedMarkInstance& expected) {
          return expected.markClass == MarkClass::Maddah;
        });
    const bool hasExpectedSukun = std::any_of(
        expectedMarks.begin(), expectedMarks.end(),
        [](const ExpectedMarkInstance& expected) {
          return expected.markClass == MarkClass::Sukun;
        });
    const bool hasExpectedWaqf = std::any_of(
        expectedMarks.begin(), expectedMarks.end(),
        [](const ExpectedMarkInstance& expected) {
          return markClassIsWaqf(expected.markClass);
        });
    const bool hasExpectedDotAboveOne = std::any_of(
        expectedMarks.begin(), expectedMarks.end(),
        [](const ExpectedMarkInstance& expected) {
          return expected.markClass == MarkClass::DotAboveOne;
        });
    const bool hasExpectedDotAboveTwo = std::any_of(
        expectedMarks.begin(), expectedMarks.end(),
        [](const ExpectedMarkInstance& expected) {
          return expected.markClass == MarkClass::DotAboveTwo;
        });
    const bool hasExpectedDotBelowOne = std::any_of(
        expectedMarks.begin(), expectedMarks.end(),
        [](const ExpectedMarkInstance& expected) {
          return expected.markClass == MarkClass::DotBelowOne;
        });
    const bool hasExpectedDotBelowTwo = std::any_of(
        expectedMarks.begin(), expectedMarks.end(),
        [](const ExpectedMarkInstance& expected) {
          return expected.markClass == MarkClass::DotBelowTwo;
        });
    const bool hasExpectedDotAboveThree = std::any_of(
        expectedMarks.begin(), expectedMarks.end(),
        [](const ExpectedMarkInstance& expected) {
          return expected.markClass == MarkClass::DotAboveThree;
        });
    const bool useMarkTemplates = hasExpectedHamza || hasExpectedSmallAlef;
    const bool useAboveDotTemplates = hasExpectedDotAboveOne ||
                                      hasExpectedDotAboveTwo ||
                                      hasExpectedDotAboveThree;
    const bool useBelowDotTemplates =
        hasExpectedDotBelowOne || hasExpectedDotBelowTwo;

    auto markClassCost = [&](int label, const ExpectedMarkInstance& expected,
                             int slotOrdinal) {
      const ComponentFeature& feature = features[label];
      const MarkClass markClass = expected.markClass;
      const int centerY = feature.bbox.y + feature.bbox.height / 2;
      const double pawHeight = std::max(1, paw.bbox.height);
      // The qaf and jeem waqf symbols occupy two components: the glyph and
      // the small round dot inside its hook. Only the first slot is the
      // composite glyph; the second is a dot, and every test in this cost
      // that says "a waqf is big and is not a dot" is wrong about it.
      // Measured over the corpus, the companion is indistinguishable from
      // the ones the DP already pairs correctly -- median
      // ordinaryMarkProbability 0.048, dotProbability 0.95, area 3.3k in
      // both groups -- yet the shared slot cost charges it -log(0.048) =
      // 3.04 plus a 1.59 size penalty, which is enough for a neighbouring
      // ordinary mark to buy the slot and push the dot into that mark's
      // class instead. This happened to 306 of 2,362 two-component waqf
      // marks.
      // The companion is a dot and nothing else: over the 1,727 the DP
      // already pairs correctly its area tops out at 6,819 (areaNorm 0.68),
      // median 3,291. A ceiling of 1.0 keeps all of them with half again in
      // headroom, and keeps the exemption below from turning the slot into
      // a cheap parking space for a neighbouring damma, dammatan, sukun or
      // superscript yeh, which measure areaNorm 1.39 to 3.83.
      constexpr double companionMaximumAreaNorm = 1.0;
      const bool waqfCompanionSlot =
          slotOrdinal > 0 &&
          (markClass == MarkClass::WaqfQaf ||
           markClass == MarkClass::WaqfJeem) &&
          feature.areaNorm <= companionMaximumAreaNorm;
      // Nothing larger belongs in the companion slot at all. Without this the
      // slot stays open at its unexempted cost and can still be bought.
      if (slotOrdinal > 0 && (markClass == MarkClass::WaqfQaf ||
                              markClass == MarkClass::WaqfJeem) &&
          !waqfCompanionSlot)
        return infinity;
      const double clearance = 0.015 * pawHeight;
      const bool observedBelow = centerY > baselineY + clearance;
      if (markClass != MarkClass::Other &&
          observedBelow != markClassIsBelow(markClass))
        return infinity;

      double typeProbability = 0.0;
      if (hasMarkClassEvidence) {
        typeProbability = feature.evidence.markClassProbabilities[static_cast<std::size_t>(markClass)];
      } else if (markClassIsDot(markClass)) {
        typeProbability = std::max(
            feature.evidence.dotProbability,
            feature.dotLike
                ? 0.85
                : 0.55 * feature.evidence.ordinaryMarkProbability);
        if (markClass == MarkClass::DotBelowTwo ||
            markClass == MarkClass::DotAboveTwo ||
            markClass == MarkClass::DotAboveThree) {
          // A connected pair of dots is larger and wider than a single
          // dot. It is not necessarily small enough for dotLike, but this
          // final-DP shape prior distinguishes it from a diagonal fatha
          // without changing the preliminary classifier.
          const double connectedDotScore =
              markClass == MarkClass::DotBelowTwo ||
                      markClass == MarkClass::DotAboveTwo
                  ? std::clamp(
                        1.0 - std::abs(feature.aspect - 2.10) / 1.10,
                        0.0, 1.0)
                  : std::clamp(
                        (std::max(feature.bbox.width, feature.bbox.height) /
                             double(std::max(
                                 1, std::min(feature.bbox.width,
                                             feature.bbox.height))) -
                         1.25) /
                            0.30,
                        0.0, 1.0);
          typeProbability = std::max(
              typeProbability, 0.85 * connectedDotScore);
        }
        if (markClass == MarkClass::DotAboveOne) {
          // A single round dot has a compact ellipse, unlike the connected
          // pair used by dot_above_two. This prevents the singleton from
          // filling one of the two-dot mark's physical slots when the pair
          // is represented by one connected component.
          const double singleDotScore =
              std::clamp((1.65 - feature.aspect) / 0.45, 0.0, 1.0);
          typeProbability = std::max(
              typeProbability, 0.85 * singleDotScore);
        }
      } else if (markClassIsSmallLetter(markClass)) {
        typeProbability = std::max(
            feature.evidence.smallLetterProbability,
            0.35 * feature.evidence.ordinaryMarkProbability);
      } else if (markClass == MarkClass::Other) {
        typeProbability = std::max(feature.evidence.ordinaryMarkProbability,
                                   feature.evidence.unknownProbability);
      } else if (markClass == MarkClass::QuranicSign &&
                 hasExpectedQuranicSign &&
                 feature.evidence.detectedSajdaComponentCount > 0) {
        // The recognizer has matched the two physical parts of U+06E9. Use
        // that evidence for both slots of the one semantic mark instance.
        typeProbability = std::max(typeProbability, 0.95);
      } else {
        typeProbability = feature.evidence.ordinaryMarkProbability;
        if (waqfCompanionSlot)
          typeProbability =
              std::max(typeProbability, feature.evidence.dotProbability);
      }

      double cost = -std::log(std::clamp(typeProbability, 0.02, 1.0));

      // A pronoun suffix's superscript waw or yeh is not stacked above its
      // base at all: it is written on the line, at the end of the subword.
      // Measured over the corpus its centre sits 0.02 line pitches from the
      // baseline -- p25 -0.05, p75 -0.01 -- while the damma on that same base
      // sits at -0.38, p25 -0.40, p75 -0.35. The two do not overlap, and
      // nothing else in this cost separates them: both are above-the-line
      // classes, so the above/below test above says nothing about either, and
      // the template bank is consulted only for PAWs expecting a hamza or a
      // small alef. Left to smallLetterProbability and ordinaryMarkProbability
      // the pair came out swapped in 968 of 1,036 PAWs.
      if (linePitch > 0.0) {
        constexpr double baselinePenalty = 2.0;
        const double offset = std::abs(centerY - baselineY) / linePitch;
        if (markClass == MarkClass::SmallWaw ||
            markClass == MarkClass::SmallYeh) {
          // Free on the line, charged away from it. A small yeh, which the
          // above/below test already assigns correctly, has a median offset of
          // 0.03 and a first quartile of 0.01.
          constexpr double onTheLine = 0.08;
          constexpr double falloff = 0.22;
          cost += baselinePenalty *
                  std::clamp((offset - onTheLine) / falloff, 0.0, 1.0);
        } else if (markClass == MarkClass::Fatha ||
                   markClass == MarkClass::Damma ||
                   markClass == MarkClass::Fathatan ||
                   markClass == MarkClass::Dammatan ||
                   markClass == MarkClass::Sukun ||
                   markClass == MarkClass::Shadda ||
                   markClass == MarkClass::Wasla) {
          // The converse, for the short vowels only. They are the classes that
          // never come near the line: of the components carrying them, 0.7% of
          // fathas, 1.3% of fathatans, 1.6% of shaddas, 2.6% of sukuns and no
          // waslas at all sit within 0.20 pitches of the baseline. The dots
          // above (40%), small alef (82%), dot-above-three (25%), maddah (14%)
          // and hamza above (11%) all legitimately do, so none of them is
          // charged here.
          constexpr double clearOfTheLine = 0.20;
          constexpr double falloff = 0.10;
          cost += baselinePenalty *
                  std::clamp((clearOfTheLine - offset) / falloff, 0.0, 1.0);
        }
      }
      const std::size_t markIndex = static_cast<std::size_t>(markClass);
      const bool useTemplateForClass =
          useMarkTemplates ||
          (useAboveDotTemplates &&
           (markClass == MarkClass::DotAboveOne ||
            markClass == MarkClass::DotAboveTwo ||
            // Only meaningful once joinSplitDotClusters has run: while the
            // cluster arrived as a pair plus a loose dot, the bank -- whose
            // templates are whole clusters -- scored the real members 0.28 and
            // below and would have evicted them from their own class. Joined,
            // they score 0.81 and 0.82, against 0.24 for a fatha and 0.34 for
            // another letter's two-dot pair. Without this the class was the
            // only expected dot class that consulted no template at all, so
            // its spare slots took any elongated mark for the 0.16 its
            // bounding-box prior charges.
            markClass == MarkClass::DotAboveThree)) ||
          (useBelowDotTemplates &&
           (markClass == MarkClass::DotBelowOne ||
            markClass == MarkClass::DotBelowTwo)) ||
          (hasExpectedSukun && markClass == MarkClass::Sukun);
      if (useTemplateForClass && markIndex < kMarkClassCount &&
          feature.evidence.templateMarkClasses[markIndex]) {
        // Template scores are IoU values in [0, 1]. A template bank is used
        // here as a competing class model for text-licensed hamza/small-alef
        // PAWs and for the explicitly expected above-dot classes. A low
        // competitor score is useful negative evidence, while a strong score
        // for the expected class contributes little extra cost. Restricting
        // generic fatha/damma use to text-licensed PAWs prevents unrelated
        // template assignments from moving.
        const double templateScore = std::clamp(
            feature.evidence.templateMarkProbabilities[markIndex], 0.02,
            1.0);
        cost += markTemplatePenalty * -std::log(templateScore);
      }
      // A three-dot mark can have three physical components, so its capacity
      // intentionally allows all three slots.  A vertical small alef can
      // nevertheless be consumed by those slots when the generic connected
      // dot prior is stronger than the small-letter fallback.  In a PAW that
      // textually expects a small alef, use its positive template score as a
      // soft negative competitor for every other semantic mark class.
      if (hasExpectedSmallAlef && markClass != MarkClass::SmallAlef &&
          markClass != MarkClass::Other) {
        const std::size_t smallAlefIndex =
            static_cast<std::size_t>(MarkClass::SmallAlef);
        if (feature.evidence.templateMarkClasses[smallAlefIndex]) {
          const double smallAlefScore = std::clamp(
              feature.evidence.templateMarkProbabilities[smallAlefIndex],
              0.0, 1.0);
          cost += markTemplatePenalty * smallAlefScore;
        }

        // A diagonal fatha can also look like an elongated connected-dot
        // component to the generic DotAboveThree geometry prior.  Keep a
        // strong fatha/damma template match out of a dot slot when this PAW
        // already expects a small alef; the textual dot slots remain
        // available for genuine round or paired dot components.
        if (markClassIsDot(markClass)) {
          const std::size_t fathaIndex =
              static_cast<std::size_t>(MarkClass::Fatha);
          const std::size_t dammaIndex =
              static_cast<std::size_t>(MarkClass::Damma);
          double vowelScore = 0.0;
          if (feature.evidence.templateMarkClasses[fathaIndex])
            vowelScore = std::max(
                vowelScore,
                feature.evidence.templateMarkProbabilities[fathaIndex]);
          if (feature.evidence.templateMarkClasses[dammaIndex])
            vowelScore = std::max(
                vowelScore,
                feature.evidence.templateMarkProbabilities[dammaIndex]);
          constexpr double strongVowelThreshold = 0.70;
          if (vowelScore > strongVowelThreshold)
            cost += markTemplatePenalty *
                    (vowelScore - strongVowelThreshold) /
                    (1.0 - strongVowelThreshold);
        }
      }
      // Maddah and small alef can both be narrow floating components, but a
      // reviewed maddah template is a useful negative competitor when the DP
      // is considering the small-alef slot. This prevents the strongest
      // small-alef template match from stealing a genuine maddah while still
      // leaving the textual slot available for the other component.
      if (hasExpectedSmallAlef && hasExpectedMaddah &&
          markClass == MarkClass::SmallAlef) {
        const std::size_t maddahIndex =
            static_cast<std::size_t>(MarkClass::Maddah);
        if (feature.evidence.templateMarkClasses[maddahIndex]) {
          const double maddahScore = std::clamp(
              feature.evidence.templateMarkProbabilities[maddahIndex], 0.0,
              1.0);
          cost += markTemplatePenalty * maddahScore;
        }
      }
      const int centerX = feature.bbox.x + feature.bbox.width / 2;
      if (expected.baseIndex >= 0 &&
          pawExpectations[pawIndex].expectedBaseCount > 1) {
        const double observedPosition =
            (paw.bbox.x + paw.bbox.width - centerX) /
            double(std::max(1, paw.bbox.width));
        const double expectedPosition =
            expected.baseIndex /
            double(pawExpectations[pawIndex].expectedBaseCount - 1);
        cost += 2.5 * std::abs(observedPosition - expectedPosition);
      }
      cost += 0.35 * std::abs(centerY - baselineY) / pawHeight;
      const double belowDistance = (centerY - baselineY) / pawHeight;
      if (markClass == MarkClass::HamzaBelow)
        cost += 0.40 * std::max(0.0, belowDistance - 0.16);
      if (markClass == MarkClass::Kasra ||
          markClass == MarkClass::Kasratan)
        cost += 0.40 * std::max(0.0, 0.16 - belowDistance);
      // Prefer a reviewed sukun shape when both sukun and waqf are expected
      // in the same PAW. This is the primary distinction between the two
      // slots when a sukun template bank is available.
      const std::size_t sukunIndex =
          static_cast<std::size_t>(MarkClass::Sukun);
      const bool hasSukunTemplate =
          feature.evidence.templateMarkClasses[sukunIndex];
      if (hasExpectedSukun && markClass != MarkClass::Sukun &&
          markClass != MarkClass::Other && hasSukunTemplate) {
        // A compact sukun can look like a connected dot mark. When the text
        // licenses a sukun, its reviewed shape is useful negative evidence
        // against every competing semantic slot, especially dot_above_two.
        const double sukunScore = std::clamp(
            feature.evidence.templateMarkProbabilities[sukunIndex], 0.02,
            1.0);
        cost += markTemplatePenalty * sukunScore;
      }
      if (hasExpectedSukun && hasExpectedWaqf && hasSukunTemplate) {
        const double sukunScore = std::clamp(
            feature.evidence.templateMarkProbabilities[sukunIndex], 0.02,
            1.0);
        if (markClass == MarkClass::Sukun)
          cost += markTemplatePenalty * -std::log(sukunScore);
        else if (markClassIsWaqf(markClass) && !waqfCompanionSlot)
          cost += markTemplatePenalty * sukunScore;
      }

      // Waqf symbols in this scan are larger composite glyphs, while sukun
      // is a compact loop. Keep this as a fallback for deployments that do
      // not yet have a reviewed sukun template bank.
      constexpr double waqfMinimumAreaNorm = 3.5;
      constexpr double waqfShapePenalty = 1.75;
      if (!hasSukunTemplate || !hasExpectedSukun || !hasExpectedWaqf) {
        if (markClassIsWaqf(markClass) && !waqfCompanionSlot) {
          cost += waqfShapePenalty *
                  std::max(0.0, waqfMinimumAreaNorm - feature.areaNorm) /
                  waqfMinimumAreaNorm;
        } else if (markClass == MarkClass::Sukun) {
          cost += waqfShapePenalty *
                  std::max(0.0, feature.areaNorm - waqfMinimumAreaNorm) /
                  waqfMinimumAreaNorm;
        }
      }
      return cost;
    };

    std::vector<ExpectedMarkInstance> expectedMarkSlots;
    std::vector<int> expectedMarkSlotInstanceIndices;
    // Position of the slot within its mark instance. Slot 0 is the mark
    // proper; later slots are its additional physical components, which for
    // some classes are shaped nothing like the first.
    std::vector<int> expectedMarkSlotOrdinals;
    for (std::size_t instanceIndex = 0;
         instanceIndex < expectedMarks.size(); ++instanceIndex) {
      const ExpectedMarkInstance& expected = expectedMarks[instanceIndex];
      for (int slot = 0;
           slot < markClassComponentCapacity(expected); ++slot) {
        expectedMarkSlots.push_back(expected);
        expectedMarkSlotInstanceIndices.push_back(
            static_cast<int>(instanceIndex));
        expectedMarkSlotOrdinals.push_back(slot);
      }
    }

    const int markCount = static_cast<int>(paw.markComponentLabels.size());
    const int expectedMarkCount = static_cast<int>(expectedMarkSlots.size());
    std::vector<int> assignedExpected(markCount, -1);

    // The slots of one instance are no longer interchangeable: slot 0 is the
    // mark proper and later slots are its extra components. Require the set
    // slots of an instance to form a prefix, so a lone waqf glyph cannot slip
    // into the companion slot and skip the shape tests that belong to it.
    std::vector<int> expectedMarkSlotPrerequisite(expectedMarkCount, -1);
    for (int slot = 1; slot < expectedMarkCount; ++slot) {
      if (expectedMarkSlotOrdinals[slot] > 0 &&
          expectedMarkSlotInstanceIndices[slot] ==
              expectedMarkSlotInstanceIndices[slot - 1])
        expectedMarkSlotPrerequisite[slot] = slot - 1;
    }

    auto stackRankForExpected = [&](int expectedIndex) {
      return markClassStackRank(expectedMarkSlots[expectedIndex].markClass);
    };

    // A stack relation is useful only when both marks are textually attached
    // to the same base.  It intentionally allows a missing counterpart: the
    // constraint is applied only to observed components that are assigned by
    // the DP.
    bool enforceStackingOrder = false;
    for (int left = 0; left < expectedMarkCount && !enforceStackingOrder;
         ++left) {
      const ExpectedMarkInstance& leftExpected = expectedMarkSlots[left];
      const int leftRank = stackRankForExpected(left);
      if (leftExpected.baseIndex < 0 || leftRank < 0) continue;
      for (int right = left + 1; right < expectedMarkCount; ++right) {
        const ExpectedMarkInstance& rightExpected =
            expectedMarkSlots[right];
        const int rightRank = stackRankForExpected(right);
        if (rightExpected.baseIndex == leftExpected.baseIndex &&
            rightRank >= 0 && rightRank != leftRank) {
          enforceStackingOrder = true;
          break;
        }
      }
    }

    auto stackingAssignmentAllowed = [&](int expectedIndex,
                                         const auto& isAssigned) {
      const ExpectedMarkInstance& expected =
          expectedMarkSlots[expectedIndex];
      const int currentRank = stackRankForExpected(expectedIndex);
      if (expected.baseIndex < 0 || currentRank < 0) return true;
      for (int previousIndex = 0; previousIndex < expectedMarkCount;
           ++previousIndex) {
        if (!isAssigned(previousIndex)) continue;
        const ExpectedMarkInstance& previous =
            expectedMarkSlots[previousIndex];
        if (previous.baseIndex != expected.baseIndex) continue;
        const int previousRank = stackRankForExpected(previousIndex);
        if (previousRank > currentRank) return false;
      }
      return true;
    };

    std::vector<int> assignmentComponentOrder(markCount);
    std::iota(assignmentComponentOrder.begin(),
              assignmentComponentOrder.end(), 0);
    if (enforceStackingOrder) {
      std::stable_sort(
          assignmentComponentOrder.begin(), assignmentComponentOrder.end(),
          [&](int left, int right) {
            const cv::Rect& leftBox =
                features[paw.markComponentLabels[left]].bbox;
            const cv::Rect& rightBox =
                features[paw.markComponentLabels[right]].bbox;
            const int leftCenterY = leftBox.y + leftBox.height / 2;
            const int rightCenterY = rightBox.y + rightBox.height / 2;
            if (leftCenterY != rightCenterY)
              return leftCenterY > rightCenterY;
            return left < right;
          });
    }

    // A PAW can hold more mark components than the text has slots for: a hizb
    // or ayah marker sitting inside the word box, a speck, or a mark the scan
    // draws as two pieces where the inventory expects one. Those components
    // have to go somewhere, and until now the presence of even one of them
    // disabled this DP entirely and handed the PAW to the greedy fallback
    // below, which commits each component to its cheapest free slot in
    // bottom-to-top order and never reconsiders. Measured over the corpus that
    // happens in 6,004 of 159,772 PAWs, and in 212 of them it strands a
    // component whose reviewed template beats, by at least 0.15, the one that
    // took its class.
    //
    // Instead, let the DP choose which components to leave out. The count is
    // forced -- exactly markCount - requiredAssigned of them -- so every
    // candidate solution skips the same number and a skip needs no cost of its
    // own to stay comparable. When there are at least as many slots as
    // components this is a no-op and the search is exactly what it was.
    const int requiredAssigned = std::min(markCount, expectedMarkCount);
    const int allowedSkips = markCount - requiredAssigned;
    if (markCount > 0 && expectedMarkCount > 0 && expectedMarkCount <= 16) {
      const int stateCount = 1 << expectedMarkCount;
      std::vector<std::vector<double>> assignmentDp(
          markCount + 1,
          std::vector<double>(stateCount, infinity));
      std::vector<std::vector<int>> previousMask(
          markCount + 1, std::vector<int>(stateCount, -1));
      std::vector<std::vector<int>> chosenExpected(
          markCount + 1, std::vector<int>(stateCount, -1));
      assignmentDp[0][0] = 0.0;
      for (int component = 0; component < markCount; ++component) {
        for (int mask = 0; mask < stateCount; ++mask) {
          if (!std::isfinite(assignmentDp[component][mask])) continue;
          for (int expectedIndex = 0;
               expectedIndex < expectedMarkCount; ++expectedIndex) {
            if (mask & (1 << expectedIndex)) continue;
            const int componentIndex =
                assignmentComponentOrder[component];
            const int label = paw.markComponentLabels[componentIndex];
            const ExpectedMarkInstance& expected =
                expectedMarkSlots[expectedIndex];
            if (enforceStackingOrder &&
                !stackingAssignmentAllowed(
                    expectedIndex, [&](int index) {
                      return (mask & (1 << index)) != 0;
                    }))
              continue;
            const double classCost = markClassCost(
                label, expected, expectedMarkSlotOrdinals[expectedIndex]);
            if (!std::isfinite(classCost)) continue;
            const int nextMask = mask | (1 << expectedIndex);
            const double candidate =
                assignmentDp[component][mask] + classCost;
            if (candidate < assignmentDp[component + 1][nextMask]) {
              assignmentDp[component + 1][nextMask] = candidate;
              previousMask[component + 1][nextMask] = mask;
              chosenExpected[component + 1][nextMask] = expectedIndex;
            }
          }
          if (allowedSkips > 0 &&
              component + 1 -
                      __builtin_popcount(static_cast<unsigned int>(mask)) <=
                  allowedSkips &&
              assignmentDp[component][mask] <
                  assignmentDp[component + 1][mask]) {
            assignmentDp[component + 1][mask] = assignmentDp[component][mask];
            previousMask[component + 1][mask] = mask;
            chosenExpected[component + 1][mask] = -1;
          }
        }
      }

      double bestAssignment = infinity;
      int bestMask = -1;
      for (int mask = 0; mask < stateCount; ++mask) {
        if (__builtin_popcount(static_cast<unsigned int>(mask)) !=
            requiredAssigned)
          continue;
        bool prefixSlotsUsed = true;
        for (int slot = 0; slot < expectedMarkCount && prefixSlotsUsed;
             ++slot) {
          if (expectedMarkSlotPrerequisite[slot] >= 0 &&
              (mask & (1 << slot)) &&
              !(mask & (1 << expectedMarkSlotPrerequisite[slot])))
            prefixSlotsUsed = false;
        }
        if (!prefixSlotsUsed) continue;
        if (assignmentDp[markCount][mask] < bestAssignment) {
          bestAssignment = assignmentDp[markCount][mask];
          bestMask = mask;
        }
      }
      if (bestMask >= 0) {
        for (int component = markCount; component > 0; --component) {
          const int componentIndex =
              assignmentComponentOrder[component - 1];
          assignedExpected[componentIndex] =
              chosenExpected[component][bestMask];
          bestMask = previousMask[component][bestMask];
        }
      }
    }

    if (markCount > 0 && expectedMarkCount > 0 &&
        std::all_of(assignedExpected.begin(), assignedExpected.end(),
                    [](int index) { return index < 0; })) {
      std::vector<bool> used(expectedMarkCount, false);
      for (int component = 0; component < markCount; ++component) {
        const int componentIndex = assignmentComponentOrder[component];
        double bestCost = infinity;
        int bestExpected = -1;
        for (int expectedIndex = 0;
             expectedIndex < expectedMarkCount; ++expectedIndex) {
          if (used[expectedIndex]) continue;
          if (enforceStackingOrder &&
              !stackingAssignmentAllowed(
                  expectedIndex, [&](int index) { return used[index]; }))
            continue;
          const double classCost = markClassCost(
              paw.markComponentLabels[componentIndex],
              expectedMarkSlots[expectedIndex],
              expectedMarkSlotOrdinals[expectedIndex]);
          if (classCost < bestCost) {
            bestCost = classCost;
            bestExpected = expectedIndex;
          }
        }
        if (bestExpected >= 0) {
          assignedExpected[componentIndex] = bestExpected;
          used[bestExpected] = true;
        }
      }
    }

    constexpr double strongHamzaBelowTemplate = 0.75;
    const std::size_t hamzaBelowIndex =
        static_cast<std::size_t>(MarkClass::HamzaBelow);
    for (std::size_t componentIndex = 0;
         componentIndex < assignedExpected.size(); ++componentIndex) {
      const int expectedIndex = assignedExpected[componentIndex];
      MarkClass assignedClass =
          expectedIndex >= 0 ? expectedMarkSlots[expectedIndex].markClass
                             : MarkClass::Other;
      // A yeh's below-dot slot is sometimes occupied by a visually distinct
      // hamza-below form that is not present in the aligned combining text.
      // Let a strong reviewed template correct only that final visual class;
      // the preliminary classifier and the component ownership DP remain
      // unchanged.
      if (assignedClass == MarkClass::DotBelowTwo &&
          features[paw.markComponentLabels[componentIndex]]
              .evidence.templateMarkClasses[hamzaBelowIndex] &&
          features[paw.markComponentLabels[componentIndex]]
                  .evidence.templateMarkProbabilities[hamzaBelowIndex] >=
              strongHamzaBelowTemplate)
        assignedClass = MarkClass::HamzaBelow;
      paw.markComponentClasses.push_back(assignedClass);
      paw.markComponentInstanceIndices.push_back(
          expectedIndex >= 0 &&
                  expectedIndex <
                      static_cast<int>(expectedMarkSlotInstanceIndices.size())
              ? expectedMarkSlotInstanceIndices[expectedIndex]
              : -1);
    }
    // The selected DP range is normally identical to the final ownership,
    // but nested-body recovery can move a mark across the range boundary.
    // Report the count from the components actually assigned to this PAW.
    paw.observedMarkComponents =
        static_cast<int>(paw.markComponentLabels.size());
    previousPawBodyLabel = bodyLabel;
    out.paws.push_back(std::move(paw));
  }

  finalizeTotals();
  return out;
}
