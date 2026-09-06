#include "PageProcessor.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

#include "BaselineEstimator.h"
#include "BaselineSegmenter.h"
#include "ComponentAssignmentCorrector.h"
#include "DiagnosticPages.h"
#include "JsonIo.h"
#include "QualityReport.h"
#include "SpacingGeometry.h"

namespace page_json {
struct SpacingSpanJson {
  int left, right, width, inkPixels;
  bool valid;
};
struct QualityJson {
  double uncertainInkFraction;
  bool spacingSpanValid, negativeSpacingGap;
  double foreignLineInkFraction, heightToLinePitch;
  bool crossLineContamination;
};
struct MarkCountJson {
  std::string className;
  int count;
};
struct MarkMaskJson {
  std::string className;
  std::vector<int> componentIndices;
  std::string path;
};
struct PawJson {
  int pawIndex;
  std::string text;
  int x, y, w, h;
  std::vector<int> componentIndices;
  std::vector<int> bodyComponentIndices;
  std::vector<int> markComponentIndices;
  int expectedMarkComponents, observedMarkComponents;
  bool bodyConnectedToPreviousWord;
  bool bodyMergedWithPrevious;
  double cost;
  std::string pngPath, bodyPath, marksPath;
  std::vector<MarkMaskJson> markMasks;
};
struct WordJson {
  std::string wordId;
  int lineIndex, wordIndex, x, y, w, h;
  int gapToNextPx, bboxGapToNextPx, spacingGapToNextPx;
  int expectedBodyComponents, expectedMarkComponents;
  int expectedComponentCount, recognizedComponentCountAdjustment;
  int recognizedAttachedAlefHamzaAboveCount;
  int effectiveExpectedComponentCount, observedComponentCount;
  std::vector<int> componentIndices;
  int expectedSmallLetterComponents;
  double observedSmallLetterComponents;
  std::vector<MarkCountJson> expectedMarks;
  std::vector<PawJson> paws;
  std::string expectedText, pngPath, svgPath;
  int bodyLeft, bodyRight;
  SpacingSpanJson spacingSpan;
  QualityJson quality;
};
struct LineJson {
  int lineIndex;
  double baseLine, baseLineConfidence;
  bool baseLineSearchBoundaryHit, baseLineReliable;
  int x, y, w, h;
  std::vector<WordJson> words;
};
struct PreprocessingJson {
  int cropX, cropY, cropWidth, cropHeight;
  double deskewDegrees;
};
struct ComponentJson {
  int componentIndex, label, lineIndex;
  bool ignored;
  int x, y, w, h, area;
  double bodyProbability, dotProbability, ordinaryMarkProbability;
  double smallLetterProbability, unknownProbability;
  double attachedAlefHamzaAboveProbability;
  double bodyTemplateProbability;
  std::string maskPath, contextPath;
  std::vector<std::pair<std::string, double>> markProbabilities;
  std::vector<std::pair<std::string, double>> templateMarkProbabilities;
};
struct WordsJson {
  int schemaVersion, page;
  std::string sourceImage;
  int imageWidth, imageHeight;
  PreprocessingJson preprocessing;
  std::vector<std::string> assignmentCorrections;
  std::vector<ComponentJson> components;
  std::vector<LineJson> lines;
};
}  // namespace page_json
using namespace page_json;

static cv::Mat toGray(const cv::Mat& src) {
  cv::Mat g;
  if (src.channels() == 3)
    cv::cvtColor(src, g, cv::COLOR_BGR2GRAY);
  else
    g = src.clone();
  return g;
}

PageProcessor::PageProcessor(Params p) : m_p(p) {}

bool PageProcessor::loadImage(const std::filesystem::path& path) {
  m_sourcePath = path;
  m_src = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);

  return !m_src.empty();
}

// labels: CV_32S output from connectedComponentsWithStats
// stats:  CV_32S output from connectedComponentsWithStats
// compIds: list of component label IDs (>=1; 0 is background)
// pad: extra pixels around union bbox
static cv::Mat buildWordMaskFromLabels(const PageModel& page, const WordBox& word, int pad = 0) {
  if (word.compIds.empty()) return cv::Mat();

  auto u = word.rect;

  const auto& labels = page.labels;
  const auto& stats = page.stats;

  // pad + clamp
  u.x = std::max(0, u.x - pad);
  u.y = std::max(0, u.y - pad);
  u.width = std::min(labels.cols - u.x, u.width + 2 * pad);
  u.height = std::min(labels.rows - u.y, u.height + 2 * pad);

  cv::Mat out(u.height, u.width, CV_8UC1, cv::Scalar(0));

  // Fast membership test: mark IDs in a boolean table
  int maxId = stats.rows - 1;
  std::vector<uint8_t> isWord(maxId + 1, 0);
  for (int compId : word.compIds) {
    auto id = page.comps[compId].label;
    if (id >= 0 && id <= maxId) isWord[id] = 1;
  }

  // Fill mask by scanning only the ROI
  for (int yy = 0; yy < u.height; ++yy) {
    const int* L = labels.ptr<int>(u.y + yy);
    uchar* O = out.ptr<uchar>(yy);
    for (int xx = 0; xx < u.width; ++xx) {
      int id = L[u.x + xx];
      if (isWord[id]) {
        O[xx] = 255;
      }
    }
  }

  return out;  // ink=255, background=0
}

// Render only the requested crop instead of first creating a full word-sized
// mask for every component. The output is intentionally the same binary
// 0/255 mask produced by `(labels == label)` and OR operations in the old
// exporter.
static cv::Mat renderSelectedLabels(const cv::Mat& labels, int componentCount,
                                    const std::vector<int>& selectedLabels,
                                    const cv::Rect& crop) {
  CV_Assert(labels.type() == CV_32S);
  CV_Assert(crop.x >= 0 && crop.y >= 0 && crop.x + crop.width <= labels.cols &&
            crop.y + crop.height <= labels.rows);

  cv::Mat output = cv::Mat::zeros(crop.size(), CV_8U);
  if (selectedLabels.empty() || crop.empty()) return output;

  std::vector<uint8_t> selected(componentCount, 0);
  for (const int label : selectedLabels) {
    if (label > 0 && label < componentCount) selected[label] = 1;
  }

  for (int y = 0; y < crop.height; ++y) {
    const int* source = labels.ptr<int>(crop.y + y) + crop.x;
    uchar* destination = output.ptr<uchar>(y);
    for (int x = 0; x < crop.width; ++x) {
      const int label = source[x];
      if (label > 0 && label < componentCount && selected[label])
        destination[x] = 255;
    }
  }
  return output;
}

static std::vector<WordComponentEvidence> buildWordComponentEvidence(
    const PageModel& page, const WordBox& word, const cv::Mat& localStats) {
  std::vector<WordComponentEvidence> result(localStats.rows);
  for (int localLabel = 1; localLabel < localStats.rows; ++localLabel) {
    const cv::Rect localBox(
        localStats.at<int>(localLabel, cv::CC_STAT_LEFT),
        localStats.at<int>(localLabel, cv::CC_STAT_TOP),
        localStats.at<int>(localLabel, cv::CC_STAT_WIDTH),
        localStats.at<int>(localLabel, cv::CC_STAT_HEIGHT));
    const int localArea =
        localStats.at<int>(localLabel, cv::CC_STAT_AREA);
    int sourceIndex = -1;
    for (const int candidateIndex : word.compIds) {
      const CC& candidate = page.comps.at(candidateIndex);
      const cv::Rect expectedLocalBox(
          candidate.bb.x - word.rect.x, candidate.bb.y - word.rect.y,
          candidate.bb.width, candidate.bb.height);
      if (expectedLocalBox == localBox && candidate.area == localArea) {
        sourceIndex = candidateIndex;
        break;
      }
    }
    if (sourceIndex < 0) continue;

    const CC& source = page.comps.at(sourceIndex);
    WordComponentEvidence& evidence = result[localLabel];
    evidence.sourceComponentIndex = sourceIndex;
    evidence.bodyProbability = source.bodyProbability;
    evidence.dotProbability = source.dotProbability;
    evidence.ordinaryMarkProbability = source.ordinaryMarkProbability;
    evidence.smallLetterProbability = source.smallLetterProbability;
    evidence.unknownProbability = source.unknownProbability;
    evidence.compactFloatingMark = source.compactFloatingMark;
    evidence.markClassProbabilities = source.markClassProbabilities;
    evidence.templateMarkProbabilities = source.templateMarkProbabilities;
    evidence.templateMarkClasses = source.templateMarkClasses;
    evidence.detectedThreeDotsAboveComponentCount =
        source.detectedThreeDotsAboveComponentCount;
    evidence.detectedSajdaComponentCount =
        source.detectedSajdaComponentCount;
    evidence.attachedAlefHamzaAboveProbability =
        source.attachedAlefHamzaAboveProbability;
    evidence.bodyTemplateProbability = source.bodyTemplateProbability;
  }
  return result;
}

bool PageProcessor::processWithText(const std::filesystem::path& outDir, const PageLayout& pageLayout) {
  using Clock = std::chrono::steady_clock;
  const auto started = Clock::now();
  auto previous = started;
  auto timing = [&](std::string_view stage) {
    if (!m_p.reportTimings) return;
    const auto now = Clock::now();
    std::cerr << "Timing page " << m_p.pageNumber << ' ' << stage << ": "
              << std::chrono::duration<double>(now - previous).count()
              << " s\n";
    previous = now;
  };
  preprocess();
  timing("preprocess");
  if (m_p.doDeskew) deskewIfNeeded();
  removeOuterFrameComponents();

  std::vector<BaselineEstimate> baselineEstimates;
  BaselineEstParams bp;
  bp.minBodyArea = 1200;
  bp.minBodyHeight = 50;
  constexpr int kExpectedLineCount = 15;
  baselineEstimates = estimateLinesByComponentGaps(m_bin, kExpectedLineCount, bp, m_p.minCompArea);
  timing("baselines");
  if (m_p.deblendConnectedMarks)
    deblendConnectedUpperMarks(baselineEstimates);
  timing("mark-deblending");
  if (m_p.regroupSplitDots) joinSplitDotClusters(baselineEstimates);
  timing("dot-regrouping");

  BaselineSegmenter::Params sp;
  sp.minCompArea = m_p.minCompArea;
  sp.markModelPath = m_p.markModelPath;
  sp.markInventoryPenalty = m_p.markInventoryPenalty;
  sp.smallYehClassWeight = m_p.smallYehClassWeight;
  sp.orderByBaselineCorridor = m_p.orderByBaselineCorridor;
  sp.otsuThreshold = m_otsuThreshold;
  sp.threeDotsAboveDetections = m_p.threeDotsAboveDetections;
  sp.threeDotsAboveDetectionSource = m_p.threeDotsAboveDetectionSource;
  sp.waqfDetections = m_p.waqfDetections;
  sp.waqfDetectionSource = m_p.waqfDetectionSource;
  sp.waqfTemplateRoot = m_p.waqfTemplateRoot;
  sp.threeDotsAboveTemplateRoot = m_p.threeDotsAboveTemplateRoot;
  sp.attachedAlefHamzaAboveTemplateRoot =
      m_p.attachedAlefHamzaAboveTemplateRoot;
  sp.markTemplateRoot = m_p.markTemplateRoot;
  sp.recognizerCacheRoot = m_p.recognizerCacheRoot;
  sp.recognizerSource = m_p.recognizerSource;
  sp.reportTimings = m_p.reportTimings;
  sp.resetRecognizerCache = m_p.resetRecognizerCache;
  sp.pageNumber = m_p.pageNumber;

  BaselineSegmenter seg(sp);
  if (!seg.run(m_bin, m_src, baselineEstimates, pageLayout, m_page)) return false;
  correctComponentAssignments(m_page, m_p.pageNumber, m_p.segmentationExceptions, m_p.waqfDetections, m_p.waqfDetectionSource);
  timing("word-segmentation");

  for (size_t i = 0; i < m_page.lines.size() && i < baselineEstimates.size(); ++i) {
    m_page.lines[i].baseLineConfidence = baselineEstimates[i].confidence;
    m_page.lines[i].baseLineSearchBoundaryHit = baselineEstimates[i].searchBoundaryHit;
  }

  exportWords(outDir);
  timing("export-and-reports");
  if (m_p.reportTimings)
    std::cerr << "Timing page " << m_p.pageNumber << " total: " << std::chrono::duration<double>(Clock::now() - started).count() << " s\n";
  return true;
}

cv::Rect cropByProjection(const cv::Mat& bin, int minInk = 250) {
  int top = 0, bottom = bin.rows - 1;
  int left = 0, right = bin.cols - 1;
  int hit = 0;
  int maxHit = 10;
  int margin = 220;

  return cv::Rect(left + margin, top + margin, right - 2 * margin - left + 1, bottom - 2 * margin - top + 1);

  // rows
  for (; top < bin.rows; ++top)
    if (cv::countNonZero(bin.row(top)) < minInk && hit > maxHit)
      break;
    else
      hit++;
  hit = 0;
  for (; bottom >= 0; --bottom)
    if (cv::countNonZero(bin.row(bottom)) < minInk && hit > maxHit)
      break;
    else
      hit++;

  // cols
  hit = 0;
  for (; left < bin.cols; ++left)
    if (cv::countNonZero(bin.col(left)) < minInk && hit > maxHit)
      break;
    else
      hit++;
  hit = 0;
  for (; right >= 0; --right)
    if (cv::countNonZero(bin.col(right)) < minInk && hit > maxHit)
      break;
    else
      hit++;

  return cv::Rect(left, top, right - left + 1, bottom - top + 1);
}

void PageProcessor::preprocess() {
  m_gray = toGray(m_src);

  if (m_p.inputIsPrebinarized) {
    cv::threshold(m_gray, m_bin, 0, 255, cv::THRESH_BINARY);
    m_cropRect = cv::Rect(0, 0, m_bin.cols, m_bin.rows);
    m_src = m_gray.clone();
    return;
  }

  if (m_p.blurK > 1) {
    cv::GaussianBlur(m_gray, m_gray, cv::Size(m_p.blurK, m_p.blurK), 0);
  }

  // cv::GaussianBlur(m_gray, m_gray, cv::Size(0,0), 4.0);

  // Adaptive threshold is robust to uneven scan lighting.
  // cv::adaptiveThreshold(m_gray, m_bin, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 41, 1);
  // cv::threshold(m_gray, m_bin, 115, 255, cv::THRESH_BINARY_INV);
  m_otsuThreshold = cv::threshold(
      m_gray, m_bin, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
  //  Now text is white(255) on black(0). Ensure that:
  //  THRESH_BINARY_INV makes dark text -> white.

  // Morph open to remove speckles
  if (m_p.morphK > 0) {
    cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(m_p.morphK, m_p.morphK));
    cv::morphologyEx(m_bin, m_bin, cv::MORPH_OPEN, k);

    // cv::Mat kclose = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(m_p.morphK, m_p.morphK));
    // cv::morphologyEx(m_bin, m_bin, cv::MORPH_CLOSE, kclose);
  }

  cv::Rect textBB = cropByProjection(m_bin);
  m_cropRect = cv::Rect(textBB.x, textBB.y, textBB.width, textBB.height);

  m_bin = m_bin(textBB).clone();
  m_src = m_src(textBB).clone();  // apply to original too
  m_gray = m_gray(textBB).clone();

  repairUnderInkedHoles();
}

int PageProcessor::repairUnderInkedHoles() {
  if (!m_p.repairUnderInkedHoles || m_bin.empty() || m_gray.empty() ||
      m_bin.size() != m_gray.size())
    return 0;

  // Page-wide ink and paper references. Otsu already assumes the page has one
  // of each, so anything finer would be claiming more than the threshold it
  // is repairing. Sampled on a grid because the scan is ~500 megapixels, and
  // the ink level is read from pixels well inside a stroke -- the antialiased
  // rim would drag it towards the paper and flatten the very contrast this
  // test depends on.
  constexpr int kStep = 11;
  constexpr int kCoreRadius = 2;
  std::vector<unsigned char> inkValues;
  std::vector<unsigned char> paperValues;
  const auto isInkCore = [&](int y, int x) {
    if (x < kCoreRadius || y < kCoreRadius ||
        x + kCoreRadius >= m_bin.cols || y + kCoreRadius >= m_bin.rows)
      return false;
    for (int dy = -kCoreRadius; dy <= kCoreRadius; ++dy) {
      const unsigned char* row = m_bin.ptr<unsigned char>(y + dy);
      for (int dx = -kCoreRadius; dx <= kCoreRadius; ++dx)
        if (!row[x + dx]) return false;
    }
    return true;
  };
  for (int y = 0; y < m_bin.rows; y += kStep) {
    const unsigned char* grayRow = m_gray.ptr<unsigned char>(y);
    const unsigned char* binRow = m_bin.ptr<unsigned char>(y);
    for (int x = 0; x < m_bin.cols; x += kStep) {
      if (binRow[x]) {
        if (isInkCore(y, x)) inkValues.push_back(grayRow[x]);
      } else {
        paperValues.push_back(grayRow[x]);
      }
    }
  }
  const auto median = [](std::vector<unsigned char>& values) {
    std::nth_element(values.begin(), values.begin() + values.size() / 2,
                     values.end());
    return static_cast<double>(values[values.size() / 2]);
  };
  if (inkValues.size() < 500 || paperValues.size() < 500) return 0;
  const double inkLevel = median(inkValues);
  const double paperLevel = median(paperValues);
  // Too little separation to judge anything: leave the mask alone rather than
  // guess with a scale that is mostly noise.
  if (paperLevel - inkLevel < 20.0) return 0;

  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(m_bin, contours, hierarchy, cv::RETR_CCOMP,
                   cv::CHAIN_APPROX_SIMPLE);

  int filled = 0;
  std::vector<unsigned char> holeValues;
  for (size_t index = 0; index < contours.size(); ++index) {
    if (hierarchy[index][3] < 0) continue;   // no parent: an outer boundary
    const cv::Rect box = cv::boundingRect(contours[index]);
    if (box.width <= 0 || box.height <= 0) continue;

    cv::Mat holeMask = cv::Mat::zeros(box.size(), CV_8U);
    cv::drawContours(holeMask, contours, static_cast<int>(index),
                     cv::Scalar(255), cv::FILLED, cv::LINE_8, cv::noArray(), 0,
                     -box.tl());
    // The contour is filled as a polygon, so an ink island sitting inside the
    // hole comes with it. Keep only the pixels that are actually background,
    // so filling the hole cannot swallow the island.
    holeMask.setTo(0, m_bin(box) != 0);

    holeValues.clear();
    const cv::Mat grayBox = m_gray(box);
    for (int y = 0; y < box.height; ++y) {
      const unsigned char* maskRow = holeMask.ptr<unsigned char>(y);
      const unsigned char* grayRow = grayBox.ptr<unsigned char>(y);
      for (int x = 0; x < box.width; ++x)
        if (maskRow[x]) holeValues.push_back(grayRow[x]);
    }
    if (holeValues.empty()) continue;

    // The brightest part of the hole, not its median: a small aperture ringed
    // by heavy ink is darkened by bleed and by the scanner's own blur, and its
    // median reads as ink even where paper plainly shows through the middle.
    const size_t rank = (holeValues.size() * 9) / 10;
    std::nth_element(holeValues.begin(), holeValues.begin() + rank,
                     holeValues.end());
    const double peak = holeValues[rank];
    if ((peak - inkLevel) / (paperLevel - inkLevel) >=
        m_p.holeRepairPaperness)
      continue;

    m_bin(box).setTo(255, holeMask);
    ++filled;
  }
  return filled;
}

int PageProcessor::removeOuterFrameComponents() {
  cv::Mat labels, stats, centroids;
  const int componentCount = cv::connectedComponentsWithStats(
      m_bin, labels, stats, centroids, 8, CV_32S);
  const int horizontalMargin = std::max(20, m_bin.cols / 50);
  const int verticalMargin = std::max(20, m_bin.rows / 50);
  // The scan can contain a broken page-edge rule. Its connected-component
  // fragments may cover only one or two text lines, so the full-page frame
  // threshold below is not sufficient. Because preprocessing deliberately
  // leaves a page margin, a very thin component that actually touches the
  // cropped image boundary is a border artifact rather than a glyph.
  const int edgeSliverMaxWidth = std::max(32, m_bin.cols / 500);
  const int edgeSliverMinHeight = std::max(500, m_bin.rows / 48);
  int removedCount = 0;

  for (int label = 1; label < componentCount; ++label) {
    const int x = stats.at<int>(label, cv::CC_STAT_LEFT);
    const int y = stats.at<int>(label, cv::CC_STAT_TOP);
    const int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
    const int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
    const bool nearVerticalEdge =
        x <= horizontalMargin || x + width >= m_bin.cols - horizontalMargin;
    const bool nearHorizontalEdge =
        y <= verticalMargin || y + height >= m_bin.rows - verticalMargin;
    const bool outerVerticalFrame =
        nearVerticalEdge && height >= m_bin.rows / 8;
    const bool outerHorizontalFrame =
        nearHorizontalEdge && width >= m_bin.cols / 2;
    const bool touchesVerticalBoundary =
        x == 0 || x + width >= m_bin.cols;
    const bool thinVerticalEdgeSliver =
        touchesVerticalBoundary && width <= edgeSliverMaxWidth &&
        height >= edgeSliverMinHeight;
    if (!outerVerticalFrame && !outerHorizontalFrame &&
        !thinVerticalEdgeSliver)
      continue;

    m_bin.setTo(0, labels == label);
    ++removedCount;
  }
  return removedCount;
}

// A two- or three-dot mark is normally one connected component, and the rest
// of the pipeline counts on that: markClassComponentCapacity gives the two-dot
// classes a single slot, and the line and word DPs compare component counts
// against the expected inventory. When the scan draws a cluster with a gap the
// halves survive as separate components, and every one of those assumptions
// then works against the truth -- the two-dot mark can only claim one half,
// the three-dot mark's spare slot goes hunting across the PAW, and the counts
// the line DP balances are off by one.
//
// Join them here, in the page mask, so that everything downstream sees the one
// component it expects. This is the inverse of deblendConnectedUpperMarks and
// runs in the same place, before any segmentation.
//
// The separation measured over the corpus is wide. Members of one cluster have
// a median bounding-box gap of 0 -- their boxes overlap and only the ink fails
// to meet -- and 99.5% are within 20px. Round dots belonging to different
// marks sit 142px apart at the first percentile, and of the 5,813 such pairs
// only 47 come within 25px, most of which are themselves split clusters that
// this pass is meant to join. Single dots are tightly sized too: area 6,929 to
// 11,818 and aspect ratio at most 1.22 between the 1st and 99th percentiles,
// so the candidate bounds below sit well clear of a fatha or a shadda.
int PageProcessor::joinSplitDotClusters(
    const std::vector<BaselineEstimate>& baselineEstimates) {
  constexpr int kMinDotArea = 5000;
  constexpr int kMaxDotArea = 15000;
  constexpr int kMinDotSide = 85;
  constexpr int kMaxDotSide = 165;
  constexpr double kMaxDotAspect = 1.45;
  // Members of one cluster essentially touch: over the corpus their
  // bounding-box gap has median 0, 99th percentile 6 and maximum 9. Ten keeps
  // every one of them. Twenty, which the wider distribution of round dots
  // belonging to different marks also tolerated, additionally welded a feh's
  // sukun to the feh's own dot at 14 to 15px on three pages. This is not a
  // substitute for the pair aspect floor -- a sukun at aspect 1.44 sat 2px
  // from its dot -- the two bound different failures.
  constexpr int kMaxJoinGap = 10;

  cv::Mat labels, stats, centroids;
  const int componentCount = cv::connectedComponentsWithStats(
      m_bin, labels, stats, centroids, 8, CV_32S);

  // The pair half of a three-dot cluster is drawn a little tighter than a
  // standalone two-dot mark, whose own long side runs 204 to 258 and area
  // 15,298 to 22,284 between the 5th and 95th percentiles. Bounds set to that
  // population alone missed real pairs at 13,960 to 14,949 and at aspect 1.45,
  // so they are widened to cover the lower tail. Nothing above the line is
  // separable from a pair by shape -- a sukun runs 213 to 278 long and a fatha
  // 267 to 450 -- so the adjacency test carries the weight, and widening these
  // bounds moves the wrongly joined count over the corpus only from 18 to 19
  // against 3,157 correct joins.
  constexpr int kMinPairArea = 12000;
  constexpr int kMaxPairArea = 26000;
  // A sukun is the one thing that reaches into these bounds, and in this scan
  // it is an open crescent, so neither an enclosed hole nor solidity tells it
  // apart -- measured, a real pair scores 0.79 and 0.84 solidity against the
  // sukun's 0.80. Only aspect separates them, and only above 1.5: the sukun
  // runs 1.27 to 1.61 and a real pair 1.54 to 1.94. Two of the corpus's
  // three-dot clusters carry a pair at 1.45 and 1.48 and are left unjoined
  // rather than admit every sukun at the distribution's own median.
  constexpr double kMinPairAspect = 1.50;
  constexpr double kMaxPairAspect = 2.20;
  constexpr int kMinPairLongSide = 170;
  constexpr int kMaxPairLongSide = 290;

  struct DotCandidate {
    cv::Rect box;
    cv::Point center;
    bool joinedPair = false;
    bool aboveBaseline = false;
  };
  std::vector<DotCandidate> dots;
  for (int label = 1; label < componentCount; ++label) {
    const int area = stats.at<int>(label, cv::CC_STAT_AREA);
    const cv::Rect box(stats.at<int>(label, cv::CC_STAT_LEFT),
                       stats.at<int>(label, cv::CC_STAT_TOP),
                       stats.at<int>(label, cv::CC_STAT_WIDTH),
                       stats.at<int>(label, cv::CC_STAT_HEIGHT));
    const int longSide = std::max(box.width, box.height);
    const int shortSide = std::max(1, std::min(box.width, box.height));
    const double aspect = double(longSide) / shortSide;
    if (longSide < kMinDotSide) continue;
    bool joinedPair = false;
    if (area >= kMinDotArea && area <= kMaxDotArea &&
        longSide <= kMaxDotSide && aspect <= kMaxDotAspect) {
      joinedPair = false;
    } else if (area >= kMinPairArea && area <= kMaxPairArea &&
               aspect >= kMinPairAspect && aspect <= kMaxPairAspect &&
               longSide >= kMinPairLongSide && longSide <= kMaxPairLongSide) {
      joinedPair = true;
    } else {
      continue;
    }
    const cv::Point center(box.x + box.width / 2, box.y + box.height / 2);
    bool aboveBaseline = false;
    int nearest = std::numeric_limits<int>::max();
    for (const BaselineEstimate& baseline : baselineEstimates) {
      const int distance = std::abs(baseline.y - center.y);
      if (distance < nearest) {
        nearest = distance;
        aboveBaseline = center.y < baseline.y;
      }
    }
    dots.push_back({box, center, joinedPair, aboveBaseline});
  }

  int joinCount = 0;
  for (std::size_t i = 0; i < dots.size(); ++i) {
    for (std::size_t j = i + 1; j < dots.size(); ++j) {
      // Two round dots are the split halves of a two-dot mark, or two of the
      // three dots of a tha/sheen. A round dot beside an already joined pair
      // is the third dot of such a cluster -- but only above the baseline. A
      // three-dot mark exists nowhere else, and below the line the same
      // geometry is a beh's dot sitting against its own kasra: measured over
      // the corpus, single-beside-pair within 20px is the same mark 3,155
      // times above the line and twice below, and two different marks 1,110
      // times below and 18 times above. Two joined pairs are never one mark.
      if (dots[i].joinedPair && dots[j].joinedPair) continue;
      if ((dots[i].joinedPair || dots[j].joinedPair) &&
          !(dots[i].aboveBaseline && dots[j].aboveBaseline))
        continue;
      const cv::Rect& a = dots[i].box;
      const cv::Rect& b = dots[j].box;
      const int dx = std::max(0, std::max(a.x - (b.x + b.width),
                                          b.x - (a.x + a.width)));
      const int dy = std::max(0, std::max(a.y - (b.y + b.height),
                                          b.y - (a.y + a.height)));
      if (dx > kMaxJoinGap || dy > kMaxJoinGap) continue;
      if (dx * dx + dy * dy > kMaxJoinGap * kMaxJoinGap) continue;
      // A bridge one seventh of a dot wide is enough to make the pair one
      // component while leaving its shape, and so its fitted-ellipse aspect,
      // essentially that of a naturally joined pair.
      const int thickness = std::max(
          3, std::min(a.width, std::min(a.height,
                                        std::min(b.width, b.height))) / 7);
      cv::line(m_bin, dots[i].center, dots[j].center, cv::Scalar(255),
               thickness, cv::LINE_8);
      ++joinCount;
      if (m_p.reportTimings)
        std::cerr << "Joined split dot cluster at x=" << a.x << " y=" << a.y
                  << " to x=" << b.x << " y=" << b.y << '\n';
    }
  }
  return joinCount;
}

int PageProcessor::deblendConnectedUpperMarks(
    const std::vector<BaselineEstimate>& baselineEstimates) {
  if (m_otsuThreshold <= 0.0 || baselineEstimates.empty()) return 0;

  cv::Mat labels, stats, centroids;
  const int componentCount = cv::connectedComponentsWithStats(
      m_bin, labels, stats, centroids, 8, CV_32S);
  int splitCount = 0;

  for (int label = 1; label < componentCount; ++label) {
    const cv::Rect box(stats.at<int>(label, cv::CC_STAT_LEFT),
                       stats.at<int>(label, cv::CC_STAT_TOP),
                       stats.at<int>(label, cv::CC_STAT_WIDTH),
                       stats.at<int>(label, cv::CC_STAT_HEIGHT));
    const int area = stats.at<int>(label, cv::CC_STAT_AREA);
    if (area < 4 * m_p.minCompArea || area > 160000 || box.width < 180 ||
        box.width > 650 || box.height < 120 || box.height > 550)
      continue;

    const BaselineEstimate* nearestBaseline = nullptr;
    int nearestDistance = std::numeric_limits<int>::max();
    for (const BaselineEstimate& baseline : baselineEstimates) {
      const int distance = std::abs(baseline.y - (box.y + box.height));
      if (distance < nearestDistance) {
        nearestDistance = distance;
        nearestBaseline = &baseline;
      }
    }
    if (nearestBaseline == nullptr || box.y + box.height >= nearestBaseline->y - 250 ||
        nearestBaseline->y - (box.y + box.height) > 1400)
      continue;

    const cv::Mat originalMask = labels(box) == label;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(originalMask.clone(), contours, hierarchy,
                     cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
    std::vector<cv::Point2d> holeCenters;
    std::vector<double> holeAreas;
    for (std::size_t contour = 0; contour < hierarchy.size(); ++contour) {
      const double holeArea = std::abs(cv::contourArea(contours[contour]));
      if (hierarchy[contour][3] < 0 ||
          holeArea < std::max(200.0, area / 200.0))
        continue;
      const cv::Moments moments = cv::moments(contours[contour]);
      if (std::abs(moments.m00) < 1e-6) continue;
      holeCenters.emplace_back(moments.m10 / moments.m00,
                               moments.m01 / moments.m00);
      holeAreas.push_back(holeArea);
    }
    if (holeCenters.size() != 2 ||
        std::min(holeAreas[0], holeAreas[1]) <
            0.30 * std::max(holeAreas[0], holeAreas[1]) ||
        std::abs(holeCenters[0].x - holeCenters[1].x) < 0.22 * box.width ||
        std::abs(holeCenters[0].y - holeCenters[1].y) > 0.25 * box.height)
      continue;

    cv::Mat seedLabels;
    int firstSeed = -1;
    int secondSeed = -1;
    cv::Point2d firstSeedCenter;
    cv::Point2d secondSeedCenter;
    for (int threshold = static_cast<int>(std::floor(m_otsuThreshold)) - 1;
         threshold >= std::max(20, static_cast<int>(m_otsuThreshold) - 70);
         threshold -= 2) {
      cv::Mat strictInk;
      cv::threshold(m_gray(box), strictInk, threshold, 255,
                    cv::THRESH_BINARY_INV);
      cv::bitwise_and(strictInk, originalMask, strictInk);
      cv::Mat seedStats, seedCentroids;
      const int seedCount = cv::connectedComponentsWithStats(
          strictInk, seedLabels, seedStats, seedCentroids, 8, CV_32S);
      std::vector<std::pair<int, int>> seeds;
      for (int seed = 1; seed < seedCount; ++seed) {
        const int seedArea = seedStats.at<int>(seed, cv::CC_STAT_AREA);
        if (seedArea >= std::max(m_p.minCompArea, area / 8))
          seeds.emplace_back(seedArea, seed);
      }
      if (seeds.size() != 2) continue;
      std::sort(seeds.begin(), seeds.end(), std::greater<>());
      firstSeed = seeds[0].second;
      secondSeed = seeds[1].second;
      firstSeedCenter =
          {seedCentroids.at<double>(firstSeed, 0),
           seedCentroids.at<double>(firstSeed, 1)};
      secondSeedCenter =
          {seedCentroids.at<double>(secondSeed, 0),
           seedCentroids.at<double>(secondSeed, 1)};
      break;
    }
    if (firstSeed < 0) continue;
    const double horizontalSeparation =
        std::abs(firstSeedCenter.x - secondSeedCenter.x);
    const double verticalSeparation =
        std::abs(firstSeedCenter.y - secondSeedCenter.y);
    if (horizontalSeparation < 1.25 * verticalSeparation) continue;

    cv::Mat firstDistance, secondDistance;
    cv::distanceTransform(seedLabels != firstSeed, firstDistance,
                          cv::DIST_L2, cv::DIST_MASK_PRECISE);
    cv::distanceTransform(seedLabels != secondSeed, secondDistance,
                          cv::DIST_L2, cv::DIST_MASK_PRECISE);
    cv::Mat firstPartition = (firstDistance <= secondDistance) & originalMask;
    cv::Mat secondPartition = (secondDistance < firstDistance) & originalMask;
    cv::Mat firstDilated, secondDilated;
    const cv::Mat neighbourhood = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(firstPartition, firstDilated, neighbourhood);
    cv::dilate(secondPartition, secondDilated, neighbourhood);
    cv::Mat seam = originalMask & firstDilated & secondDilated;
    if (cv::countNonZero(seam) > area / 12) continue;

    cv::Mat candidate = originalMask.clone();
    candidate.setTo(0, seam);
    cv::Mat verifyLabels, verifyStats, verifyCentroids;
    const int verifiedCount = cv::connectedComponentsWithStats(
        candidate, verifyLabels, verifyStats, verifyCentroids, 8, CV_32S);
    std::vector<int> verifiedAreas;
    for (int part = 1; part < verifiedCount; ++part)
      if (const int partArea =
              verifyStats.at<int>(part, cv::CC_STAT_AREA);
          partArea >= m_p.minCompArea)
        verifiedAreas.push_back(partArea);
    if (verifiedAreas.size() != 2 ||
        *std::min_element(verifiedAreas.begin(), verifiedAreas.end()) <
            area / 5)
      continue;

    m_bin(box).setTo(0, seam);
    if (m_p.reportTimings)
      std::cerr << "Deblended connected upper mark at x=" << box.x
                << " y=" << box.y << " w=" << box.width
                << " h=" << box.height << '\n';
    ++splitCount;
  }
  return splitCount;
}

void PageProcessor::deskewIfNeeded() {
  // Simple Hough-based skew estimation on edges:
  cv::Mat edges;
  cv::Canny(m_bin, edges, 50, 150, 3);

  std::vector<cv::Vec2f> lines;
  cv::HoughLines(edges, lines, 1, CV_PI / 180, 200);

  double angleSum = 0.0;
  int count = 0;
  for (auto& l : lines) {
    float theta = l[1];
    // Convert to degrees around horizontal
    double deg = theta * 180.0 / CV_PI;
    // Keep near horizontal lines only
    if (deg > 80 && deg < 100) continue;  // near vertical
    // Normalize to [-45,45]
    double a = deg;
    while (a > 90) a -= 180;
    while (a < -90) a += 180;
    if (std::abs(a) < 15) {
      angleSum += a;
      count++;
    }
  }
  if (count < 5) return;
  double skew = angleSum / count;
  if (std::abs(skew) < 0.3) return;
  m_deskewDegrees = skew;

  cv::Point2f center(m_bin.cols / 2.0f, m_bin.rows / 2.0f);
  cv::Mat M = cv::getRotationMatrix2D(center, skew, 1.0);
  cv::warpAffine(m_bin, m_bin, M, m_bin.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT, 0);
  cv::warpAffine(m_src, m_src, M, m_src.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
  m_gray = toGray(m_src);
}

void exportMarkSegmentationResult(
    WordBox& word, const cv::Mat& wordMask,
    const WordConnectedComponents& connectedComponents,
    const std::filesystem::path& subWordimgsDir) {
  CV_Assert(wordMask.type() == CV_8U);
  CV_Assert(connectedComponents.labels.type() == CV_32S);
  CV_Assert(connectedComponents.labels.size() == wordMask.size());

  const auto png_body = subWordimgsDir / (word.wordId + "_body.png");

  const cv::Mat& labels = connectedComponents.labels;
  const int componentCount = connectedComponents.stats.rows;
  const cv::Rect wordBounds(0, 0, wordMask.cols, wordMask.rows);
  std::vector<int> bodyLabels;

  for (auto const& comp : word.bodySegmentation.componentTypes) {
    if (comp.componentType != ComponentTypeEnum::Body) continue;
    bodyLabels.push_back(comp.componentLabel);
  }
  const cv::Mat bodyMask =
      renderSelectedLabels(labels, componentCount, bodyLabels, wordBounds);

  for (WordPaw& paw : word.bodySegmentation.paws) {
    const cv::Rect crop = paw.bbox & wordBounds;
    const cv::Rect outputCrop = crop.empty() ? wordBounds : crop;
    const cv::Mat pawMask = renderSelectedLabels(
        labels, componentCount, paw.componentLabels, outputCrop);
    const cv::Mat pawBodyMask = renderSelectedLabels(
        labels, componentCount, paw.bodyComponentLabels, outputCrop);
    const cv::Mat pawMarksMask = renderSelectedLabels(
        labels, componentCount, paw.markComponentLabels, outputCrop);

    struct MarkInstanceMask {
      MarkClass markClass = MarkClass::Other;
      int instanceIndex = -1;
      std::vector<int> componentLabels;
      std::vector<int> componentIndices;
    };
    std::vector<MarkInstanceMask> markInstanceMasks;
    for (std::size_t index = 0; index < paw.markComponentLabels.size();
         ++index) {
      const MarkClass markClass =
          index < paw.markComponentClasses.size()
              ? paw.markComponentClasses[index]
              : MarkClass::Other;
      const int expectedInstance =
          index < paw.markComponentInstanceIndices.size()
              ? paw.markComponentInstanceIndices[index]
              : -1;
      // A component without a textual instance is kept separate. This also
      // prevents two unrelated `other` paths from being merged.
      const int instanceKey = expectedInstance >= 0
                                  ? expectedInstance
                                  : -1 - static_cast<int>(index);
      auto instanceIt = std::find_if(
          markInstanceMasks.begin(), markInstanceMasks.end(),
          [&](const MarkInstanceMask& item) {
            return item.instanceIndex == instanceKey;
          });
      if (instanceIt == markInstanceMasks.end()) {
        markInstanceMasks.push_back({markClass, instanceKey, {}, {}});
        instanceIt = std::prev(markInstanceMasks.end());
      }
      instanceIt->componentLabels.push_back(paw.markComponentLabels[index]);
      if (index < paw.markComponentIndices.size() &&
          paw.markComponentIndices[index] >= 0)
        instanceIt->componentIndices.push_back(
            paw.markComponentIndices[index]);
    }

    std::ostringstream name;
    name << word.wordId << "_paw" << std::setw(2) << std::setfill('0')
         << paw.pawIndex;
    const auto pngPaw = subWordimgsDir / (name.str() + ".png");
    const auto pngPawBody =
        subWordimgsDir / (name.str() + "_body.png");
    const auto pngPawMarks =
        subWordimgsDir / (name.str() + "_marks.png");
    cv::imwrite(pngPaw.string(), pawMask);
    cv::imwrite(pngPawBody.string(), pawBodyMask);
    cv::imwrite(pngPawMarks.string(), pawMarksMask);
    paw.pngPath = pngPaw.string();
    paw.bodyPath = pngPawBody.string();
    paw.marksPath = pngPawMarks.string();

    paw.markMasks.clear();
    for (std::size_t markIndex = 0; markIndex < markInstanceMasks.size();
         ++markIndex) {
      MarkInstanceMask& mark = markInstanceMasks[markIndex];
      const cv::Mat markMask = renderSelectedLabels(
          labels, componentCount, mark.componentLabels, outputCrop);
      if (cv::countNonZero(markMask) == 0) continue;
      std::ostringstream markName;
      markName << name.str() << "_" << markClassName(mark.markClass) << "_"
               << std::setw(2) << std::setfill('0') << markIndex << ".png";
      const auto pngMark = subWordimgsDir / markName.str();
      cv::imwrite(pngMark.string(), markMask);
      paw.markMasks.push_back(
          {mark.markClass, mark.componentIndices, pngMark.string()});
    }
  }

  if (cv::countNonZero(bodyMask) != 0) {
    cv::imwrite(png_body.string(), bodyMask);
  }
}

void PageProcessor::exportWords(const std::filesystem::path& outDir) {
  namespace fs = std::filesystem;
  fs::create_directories(outDir);
  const fs::path wordimgsDir = outDir / "wordimgs";
  const fs::path subWordimgsDir = outDir / "subwordimgs";
  fs::remove_all(wordimgsDir);
  fs::remove_all(subWordimgsDir);
  fs::create_directories(wordimgsDir);
  fs::create_directories(subWordimgsDir);
  std::vector<std::string> componentMaskPaths(m_page.comps.size());
  std::vector<std::string> componentContextPaths(m_page.comps.size());
  if (m_p.exportComponentDataset) {
    const fs::path masksDir = outDir / "component-dataset" / "masks";
    const fs::path contextsDir = outDir / "component-dataset" / "contexts";
    fs::create_directories(masksDir);
    fs::create_directories(contextsDir);
    for (size_t index = 0; index < m_page.comps.size(); ++index) {
      const CC& component = m_page.comps[index];
      cv::Mat componentMask =
          m_page.labels(component.bb) == component.label;
      cv::Mat normalizedMask(96, 96, CV_8UC1, cv::Scalar(0));
      const double maskScale =
          std::min(88.0 / component.bb.width, 88.0 / component.bb.height);
      cv::Mat resizedMask;
      const cv::Size resizedSize(
          std::max(1, static_cast<int>(std::lround(componentMask.cols *
                                                   maskScale))),
          std::max(1, static_cast<int>(std::lround(componentMask.rows *
                                                   maskScale))));
      cv::resize(componentMask, resizedMask, resizedSize, 0.0, 0.0,
                 cv::INTER_NEAREST);
      const int maskX = (normalizedMask.cols - resizedMask.cols) / 2;
      const int maskY = (normalizedMask.rows - resizedMask.rows) / 2;
      resizedMask.copyTo(
          normalizedMask(cv::Rect(maskX, maskY, resizedMask.cols,
                                  resizedMask.rows)));

      double pitch = 1.0;
      if (component.line > 0)
        pitch = m_page.lines[component.line].baseLine -
                m_page.lines[component.line - 1].baseLine;
      else if (component.line + 1 < static_cast<int>(m_page.lines.size()))
        pitch = m_page.lines[component.line + 1].baseLine -
                m_page.lines[component.line].baseLine;
      const int contextWidth = std::max(1, static_cast<int>(1.2 * pitch));
      const int contextHeight = std::max(1, static_cast<int>(0.9 * pitch));
      cv::Rect contextRect(
          static_cast<int>(component.cent.x) - contextWidth / 2,
          static_cast<int>(component.cent.y) - contextHeight / 2,
          contextWidth, contextHeight);
      contextRect &= cv::Rect(0, 0, m_bin.cols, m_bin.rows);
      cv::Mat normalizedContext;
      cv::resize(m_bin(contextRect), normalizedContext, cv::Size(192, 128),
                 0.0, 0.0, cv::INTER_AREA);

      std::ostringstream name;
      name << 'c' << std::setw(5) << std::setfill('0') << index << ".png";
      const fs::path maskPath = masksDir / name.str();
      const fs::path contextPath = contextsDir / name.str();
      cv::imwrite(maskPath.string(), normalizedMask);
      cv::imwrite(contextPath.string(), normalizedContext);
      componentMaskPaths[index] = maskPath.string();
      componentContextPaths[index] = contextPath.string();
    }
  }

  auto& lines = m_page.lines;
  std::vector<double> linePitches;
  for (int li = 1; li < lines.size(); ++li) {
    linePitches.push_back(lines[li].baseLine - lines[li - 1].baseLine);
  }
  std::sort(linePitches.begin(), linePitches.end());
  const double medianLinePitch =
      linePitches.empty() ? 1.0 : linePitches[linePitches.size() / 2];

  // First measure every word. The canonical gap is computed only after both
  // neighboring baseline-corridor spans are available.
  for (int li = 0; li < lines.size(); ++li) {
    auto& line = lines[li];
    for (int wi = 0; wi < line.words.size(); ++wi) {
      WordBox& w = line.words[wi];

      cv::Mat wordMask = buildWordMaskFromLabels(m_page, w, 0);
      int baselineY = line.baseLine - w.rect.y;

      WordConnectedComponents wordComponents;
      cv::connectedComponentsWithStats(
          wordMask, wordComponents.labels, wordComponents.stats,
          wordComponents.centroids, 8, CV_32S);
      const std::vector<WordComponentEvidence> componentEvidence =
          buildWordComponentEvidence(m_page, w, wordComponents.stats);

      // In RTL order, the preceding word is the word immediately to the
      // right. A wide, high-confidence body component ending at that word's
      // boundary is the characteristic scan form where the current word's
      // initial Alef has fused toward the preceding word. Keep this as a
      // boundary hint for the PAW DP; ordinary narrow Alefs do not qualify.
      bool firstPawConnectedToPreviousWord = false;
      if (wi > 0 && !w.pawExpectations.empty()) {
        const std::string& firstPawText = w.pawExpectations.front().text;
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
        bool startsWithAlef = false;
        for (const char* alef : alefBases) {
          const std::string prefix(alef);
          if (firstPawText.size() >= prefix.size() &&
              firstPawText.compare(0, prefix.size(), prefix) == 0) {
            startsWithAlef = true;
            break;
          }
        }
        const WordBox& previousWord = line.words[wi - 1];
        // An alef joined to the word before it arrives as a letter's worth of
        // ink. Without a size floor the hamza-below of إِ passes for one: it is
        // 220x209 to 260x222, a hair wider than tall, and the preliminary
        // classifier scores it 0.95 -- the same body/mark inversion that put a
        // kasra where page 321's ta-marbuta belonged. The word then loses two
        // letters at once, the first PAW keeping no body and the hamza taking
        // the next PAW's, which is how إِذَا came to be read with a Dal. Of the
        // 37 PAWs carrying this flag over 151 pages, 30 were triggered this
        // way. The ceiling is the one the rest of the pipeline already uses to
        // mean "too big to be a compact mark", and the separation is not
        // close: a hamza-below is around 26,000 px against a bound of 114,900,
        // while the genuine attachments run 681x502, 2224x818, 4761x145.
        const double linePitch =
            li > 0 ? line.baseLine - m_page.lines[li - 1].baseLine
                   : li + 1 < static_cast<int>(m_page.lines.size())
                         ? m_page.lines[li + 1].baseLine - line.baseLine
                         : 0.0;
        // The bound has to clear a mark without clearing a letter, and 0.030
        // of the pitch squared -- 11.4 qalam squared -- does not: it excludes
        // a hamza below at 3.17, but also the Noon of إِنَّ on page 566 l02_w05
        // at 7.4, and that Noon is the evidence. With it refused the word slid
        // by one, the Alef's PAW taking the Noon and the Noon's taking the
        // shadda. The gap the corpus actually offers is 3.63 at a hamza
        // below's ninetieth percentile against 7.79 at an isolated Noon's
        // tenth; this sits in the middle of it.
        const double attachedBodyMinimumArea = 0.0145 * linePitch * linePitch;
        if (startsWithAlef && previousWord.rect.x > w.rect.x) {
          const int currentWordRight = w.rect.x + w.rect.width;
          for (const int componentIndex : w.compIds) {
            const CC& component = m_page.comps.at(componentIndex);
            if (component.bodyProbability < 0.75 ||
                component.bb.width <= component.bb.height ||
                component.area <= attachedBodyMinimumArea)
              continue;
            // And reviewed evidence that a component is a mark disqualifies it
            // from being evidence about a body. This catches what size alone
            // does not: page 363's trigger is a dot_below_two at 0.81.
            double strongestTemplateMark = 0.0;
            for (std::size_t mark = 0; mark < kMarkClassCount; ++mark)
              if (component.templateMarkClasses[mark])
                strongestTemplateMark =
                    std::max(strongestTemplateMark,
                             component.templateMarkProbabilities[mark]);
            if (strongestTemplateMark >= 0.72) continue;
            const int componentRight = component.bb.x + component.bb.width;
            const int boundaryGap = previousWord.rect.x - componentRight;
            const int rightInset = currentWordRight - componentRight;
            if (boundaryGap >= -100 && boundaryGap <= 120 &&
                rightInset >= -20 && rightInset <= 300) {
              firstPawConnectedToPreviousWord = true;
              break;
            }
          }
        }
      }
      w.bodySegmentation = segmentBodyAndMarks(
          wordMask, baselineY, 0.7, 0.9, w.pawExpectations,
          componentEvidence, m_p.markTemplatePenalty,
          firstPawConnectedToPreviousWord, &wordComponents, medianLinePitch);
      w.heightToLinePitch = w.rect.height / std::max(1.0, medianLinePitch);

      const int localComponentCount = wordComponents.stats.rows;
      std::vector<uint8_t> isBody(localComponentCount, 0);
      for (const ComponentType& component :
           w.bodySegmentation.componentTypes) {
        if (component.componentType == ComponentTypeEnum::Body &&
            component.componentLabel > 0 &&
            component.componentLabel < localComponentCount) {
          isBody[component.componentLabel] = 1;
        }
      }

      int totalInk = 0;
      int foreignInk = 0;
      for (int y = 0; y < wordMask.rows; ++y) {
        const double pageY = w.rect.y + y;
        int nearestLine = li;
        double nearestDistance = std::abs(pageY - lines[li].baseLine);
        for (int candidate = 0; candidate < lines.size(); ++candidate) {
          const double distance =
              std::abs(pageY - lines[candidate].baseLine);
          if (distance < nearestDistance) {
            nearestDistance = distance;
            nearestLine = candidate;
          }
        }
        const int* labelRow = wordComponents.labels.ptr<int>(y);
        for (int x = 0; x < wordMask.cols; ++x) {
          if (!isBody[labelRow[x]]) continue;
          ++totalInk;
          if (nearestLine != li) ++foreignInk;
        }
      }
      w.foreignLineInkFraction =
          totalInk > 0 ? double(foreignInk) / totalInk : 0.0;
      SpacingGeometryParams spacingParams;
      spacingParams.corridorAbove = m_p.spacingCorridorAbove;
      spacingParams.corridorBelow = m_p.spacingCorridorBelow;
      w.spacingSpan =
          measureBaselineCorridorSpan(wordMask, w.bodySegmentation, baselineY, spacingParams);

      if (!m_p.analysisOnly) {
        exportMarkSegmentationResult(w, wordMask, wordComponents,
                                     subWordimgsDir);
        const fs::path png = wordimgsDir / (w.wordId + ".png");
        cv::imwrite(png.string(), wordMask);
        w.pngPath = png.string();
      }
    }
    computeLineSpacingGeometry(line);
  }

  std::vector<LineJson> linesJson;
  for (int li = 0; li < lines.size(); ++li) {
    std::vector<WordJson> wordsJson;
    auto& line = lines[li];
    for (int wi = 0; wi < line.words.size(); ++wi) {
      WordBox& w = line.words[wi];
      const int bodyLeft = w.bodySegmentation.bodyRect.x;
      const int bodyRight =
          w.rect.width -
          (w.bodySegmentation.bodyRect.x +
           w.bodySegmentation.bodyRect.width - 1) -
          1;
      const double uncertainFraction =
          w.bodySegmentation.totalInkPixels > 0
              ? double(w.bodySegmentation.uncertainInkPixels) /
                    w.bodySegmentation.totalInkPixels
              : 0.0;
      std::vector<MarkCountJson> expectedMarks;
      for (std::size_t mark = 0; mark < kMarkClassCount; ++mark) {
        if (w.expectedMarks[mark] == 0) continue;
        expectedMarks.push_back(
            {std::string(markClassName(static_cast<MarkClass>(mark))),
             w.expectedMarks[mark]});
      }

      std::vector<PawJson> pawsJson;
      pawsJson.reserve(w.bodySegmentation.paws.size());
      for (const WordPaw& paw : w.bodySegmentation.paws) {
        pawsJson.push_back(
            {paw.pawIndex,
             paw.text,
             paw.bbox.x,
             paw.bbox.y,
             paw.bbox.width,
             paw.bbox.height,
             paw.componentIndices,
             paw.bodyComponentIndices,
             paw.markComponentIndices,
             paw.expectedMarkComponents,
             paw.observedMarkComponents,
             paw.bodyConnectedToPreviousWord,
             paw.bodyMergedWithPrevious,
             paw.cost,
             paw.pngPath,
             paw.bodyPath,
             paw.marksPath,
             [&paw] {
               std::vector<MarkMaskJson> markMasks;
               markMasks.reserve(paw.markMasks.size());
               for (const PawMarkMask& markMask : paw.markMasks)
                 markMasks.push_back(
                     {std::string(markClassName(markMask.markClass)),
                      markMask.componentIndices, markMask.path});
               return markMasks;
             }()});
      }

      const int expectedComponentCount = w.expectedBodyComponents + w.expectedMarkComponents;
      const int effectiveExpectedComponentCount =
          expectedComponentCount - w.recognizedComponentCountAdjustment;
      wordsJson.push_back(
          {w.wordId,
           w.lineIndex,
           w.wordIndex,
           w.rect.x,
           w.rect.y,
           w.rect.width,
           w.rect.height,
           w.spacingGapToNextPx,
           w.bboxGapToNextPx,
           w.spacingGapToNextPx,
           w.expectedBodyComponents,
           w.expectedMarkComponents,
           expectedComponentCount,
           w.recognizedComponentCountAdjustment,
           w.recognizedAttachedAlefHamzaAboveCount,
           effectiveExpectedComponentCount,
           static_cast<int>(w.compIds.size()),
           w.compIds,
           w.expectedSmallLetterComponents,
           w.observedSmallLetterComponents,
           std::move(expectedMarks),
           std::move(pawsJson),
           w.expectedText,
           w.pngPath,
           w.svgPath,
           bodyLeft,
           bodyRight,
           {w.spacingSpan.left, w.spacingSpan.right, w.spacingSpan.width,
            w.spacingSpan.inkPixels, w.spacingSpan.valid},
           {uncertainFraction, w.spacingSpan.valid,
            wi + 1 < line.words.size() && w.spacingGapToNextPx < 0,
            w.foreignLineInkFraction, w.heightToLinePitch,
            w.foreignLineInkFraction > 0.05 ||
                w.heightToLinePitch > 1.35}});
    }
    linesJson.push_back(
        {li,
         line.baseLine,
         line.baseLineConfidence,
         line.baseLineSearchBoundaryHit,
         !line.baseLineSearchBoundaryHit && line.baseLineConfidence >= 0.20,
         line.rect.x,
         line.rect.y,
         line.rect.width,
         line.rect.height,
         std::move(wordsJson)});
  }

  std::vector<ComponentJson> componentsJson;
  componentsJson.reserve(m_page.comps.size());
  for (size_t index = 0; index < m_page.comps.size(); ++index) {
    const CC& component = m_page.comps[index];
    std::vector<std::pair<std::string, double>> markProbabilities;
    for (std::size_t mark = 0; mark < kMarkClassCount; ++mark) {
      if (component.markClassProbabilities[mark] <= 0.0) continue;
      markProbabilities.emplace_back(
          markClassName(static_cast<MarkClass>(mark)),
          component.markClassProbabilities[mark]);
    }
    std::vector<std::pair<std::string, double>> templateMarkProbabilities;
    for (std::size_t mark = 0; mark < kMarkClassCount; ++mark) {
      if (!component.templateMarkClasses[mark]) continue;
      templateMarkProbabilities.emplace_back(
          markClassName(static_cast<MarkClass>(mark)),
          component.templateMarkProbabilities[mark]);
    }
    componentsJson.push_back(
        {static_cast<int>(index),
         component.label,
         component.line,
         component.ignored,
         component.bb.x,
         component.bb.y,
         component.bb.width,
         component.bb.height,
         component.area,
         component.bodyProbability,
         component.dotProbability,
         component.ordinaryMarkProbability,
         component.smallLetterProbability,
         component.unknownProbability,
         component.attachedAlefHamzaAboveProbability,
         component.bodyTemplateProbability,
         componentMaskPaths[index],
         componentContextPaths[index],
         std::move(markProbabilities),
         std::move(templateMarkProbabilities)});
  }

  writeJsonFile(
      outDir / "words.json",
      WordsJson{8,
                m_p.pageNumber,
                m_sourcePath.string(),
                m_bin.cols,
                m_bin.rows,
                {m_cropRect.x, m_cropRect.y, m_cropRect.width,
                 m_cropRect.height, m_deskewDegrees},
                m_page.assignmentCorrections,
                std::move(componentsJson),
                std::move(linesJson)});
  if (!m_p.analysisOnly) {
    writeQualityReport(lines, outDir, m_p.pageNumber);
    if (m_p.debugImages || m_p.debugReconstructed)
      writeDiagnosticPages(m_bin, m_src, m_page, outDir, m_p.debugImages,
                           m_p.debugReconstructed);
  }
}
