#include "BaselineSegmenter.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>

#include <glaze/glaze.hpp>

#include "PreliminaryComponentClassifier.h"
#include "ComponentRecognizer.h"
#include "MarkClassifier.h"

struct ThreeDotsAboveDetection {
  std::string source;
  int page = -1;
  std::vector<int> componentParts;
  int connectedComponentCount = 0;
  double score = 0.0;
  bool reviewed = false;
  bool autoAcceptedCount = false;
};

struct PreDpWaqfDetection {
  std::string source;
  int page = -1;
  std::vector<int> componentParts;
  std::string className;
  double score = 0.0;
  int targetLineIndex = -1;
  bool autoAcceptedCorrection = false;
};

namespace {
std::string wordId(int line, int word) {
  std::ostringstream stream;
  stream << 'l' << std::setw(2) << std::setfill('0') << line << "_w"
         << std::setw(2) << word;
  return stream.str();
}
}  // namespace

void BaselineSegmenter::loadThreeDotsAboveDetections(
    std::vector<CC>& components) const {
  if (m_p.threeDotsAboveDetections.empty()) return;
  std::ifstream input(m_p.threeDotsAboveDetections);
  if (!input)
    throw std::runtime_error(
        "Could not open three-dots-above detections " +
        m_p.threeDotsAboveDetections.string());

  std::string json;
  while (std::getline(input, json)) {
    if (json.empty()) continue;
    ThreeDotsAboveDetection detection;
    if (const auto error =
            glz::read<glz::opts{.error_on_unknown_keys = false}>(detection,
                                                                 json))
      throw std::runtime_error(
          "Could not parse three-dots-above detection: " +
          glz::format_error(error, json));
    if (detection.page != m_p.pageNumber ||
        detection.source != m_p.threeDotsAboveDetectionSource ||
        (!detection.reviewed && !detection.autoAcceptedCount) ||
        (detection.connectedComponentCount != 1 &&
         detection.connectedComponentCount != 2) ||
        detection.componentParts.size() !=
            static_cast<std::size_t>(detection.connectedComponentCount))
      continue;
    for (const int componentIndex : detection.componentParts) {
      if (componentIndex < 0 ||
          componentIndex >= static_cast<int>(components.size()))
        throw std::runtime_error(
            "Invalid component index in three-dots-above detection");
      components[componentIndex].detectedThreeDotsAboveComponentCount =
          detection.connectedComponentCount;
    }
  }
}

void BaselineSegmenter::applyDetectedWaqfLineAssignments(
    std::vector<CC>& components, const std::vector<LineModel>& lines,
    PageModel& page) const {
  if (m_p.waqfDetections.empty()) return;
  std::ifstream input(m_p.waqfDetections);
  if (!input)
    throw std::runtime_error("Could not open waqf detections " +
                             m_p.waqfDetections.string());

  std::string json;
  while (std::getline(input, json)) {
    if (json.empty()) continue;
    PreDpWaqfDetection detection;
    if (const auto error =
            glz::read<glz::opts{.error_on_unknown_keys = false}>(detection,
                                                                 json))
      throw std::runtime_error("Could not parse waqf detection: " +
                               glz::format_error(error, json));
    if (detection.page != m_p.pageNumber ||
        detection.source != m_p.waqfDetectionSource ||
        !detection.autoAcceptedCorrection || detection.componentParts.empty() ||
        detection.targetLineIndex < 0 ||
        detection.targetLineIndex >= static_cast<int>(lines.size()))
      continue;

    bool valid = true;
    for (const int componentIndex : detection.componentParts) {
      if (componentIndex < 0 ||
          componentIndex >= static_cast<int>(components.size())) {
        valid = false;
        break;
      }
      const CC& component = components[componentIndex];
      if (std::abs(component.line - detection.targetLineIndex) > 1 ||
          component.bb.y + component.bb.height >
              lines[detection.targetLineIndex].yBase) {
        valid = false;
        break;
      }
    }
    if (!valid) continue;

    for (const int componentIndex : detection.componentParts) {
      CC& component = components[componentIndex];
      if (component.line == detection.targetLineIndex) continue;
      const int previousLine = component.line;
      component.line = detection.targetLineIndex;
      page.assignmentCorrections.push_back(
          "pre-dp-detected-" + detection.className + " component=" +
          std::to_string(componentIndex) + " line=" +
          std::to_string(previousLine) + "->" +
          std::to_string(detection.targetLineIndex) + " score=" +
          std::to_string(detection.score));
    }
  }
}

static int clampi(int v, int lo, int hi) { return std::max(lo, std::min(hi, v)); }

void BaselineSegmenter::extractComponents(const cv::Mat& bin, int minArea, PageModel& page) {
  CV_Assert(bin.type() == CV_8UC1);

  int n = cv::connectedComponentsWithStats(bin, page.labels, page.stats, page.centroids, 8, CV_32S);

  auto& out = page.comps;
  out.reserve(std::max(0, n - 1));

  for (int i = 1; i < n; ++i) {
    int area = page.stats.at<int>(i, cv::CC_STAT_AREA);
    if (area < minArea) continue;
    int x = page.stats.at<int>(i, cv::CC_STAT_LEFT);
    int y = page.stats.at<int>(i, cv::CC_STAT_TOP);
    int w = page.stats.at<int>(i, cv::CC_STAT_WIDTH);
    int h = page.stats.at<int>(i, cv::CC_STAT_HEIGHT);

    CC c;
    c.label = i;
    c.bb = cv::Rect(x, y, w, h);
    c.area = area;
    c.cent = cv::Point2f((float)(x + 0.5 * w), (float)(y + 0.5 * h));
    out.push_back(c);
  }
}
std::vector<LineModel> BaselineSegmenter::buildLineModels(const std::vector<BaselineEstimate>& lineEstimates, int H) {
  std::vector<LineModel> lines;
  lines.reserve(lineEstimates.size());

  constexpr double kAscenderFraction = 0.70;
  constexpr double kDescenderFraction = 1.0 - kAscenderFraction;
  for (size_t index = 0; index < lineEstimates.size(); ++index) {
    const BaselineEstimate& estimate = lineEstimates[index];
    LineModel lm;
    lm.yBase = estimate.y;
    int top = 0;
    int bottom = H - 1;
    if (index > 0) {
      const int pitchAbove = estimate.y - lineEstimates[index - 1].y;
      top = estimate.y - static_cast<int>(std::lround(kAscenderFraction * pitchAbove));
    }
    if (index + 1 < lineEstimates.size()) {
      const int pitchBelow = lineEstimates[index + 1].y - estimate.y;
      bottom = estimate.y + static_cast<int>(std::lround(kDescenderFraction * pitchBelow));
    }
    lm.yTop = clampi(top, 0, H - 1);
    lm.yBot = clampi(bottom, lm.yTop, H - 1);
    lines.push_back(lm);
  }

  return lines;
}

void BaselineSegmenter::ignoreSajdaBars(PageModel& page) const {
  // In this scan set the sajda sign is rendered as one detached, nearly
  // horizontal bar. It is much wider and thinner than an Arabic body or a
  // vowel mark. Normalize the thresholds by the local line pitch so this
  // remains stable if the page is rendered at another resolution.
  constexpr double kMinimumAspectRatio = 24.0;
  constexpr double kMinimumWidthInPitches = 1.25;
  constexpr double kMaximumHeightInPitches = 0.08;

  for (std::size_t index = 0; index < page.comps.size(); ++index) {
    CC& component = page.comps[index];
    if (component.line < 0 ||
        component.line >= static_cast<int>(page.lines.size()) ||
        component.bb.height <= 0)
      continue;
    const int line = component.line;
    double pitch = 0.0;
    if (line > 0)
      pitch = page.lines[line].baseLine - page.lines[line - 1].baseLine;
    if (line + 1 < static_cast<int>(page.lines.size())) {
      const double nextPitch =
          page.lines[line + 1].baseLine - page.lines[line].baseLine;
      pitch = pitch > 0.0 ? std::min(pitch, nextPitch) : nextPitch;
    }
    if (pitch <= 0.0) continue;

    const double aspectRatio =
        component.bb.width / static_cast<double>(component.bb.height);
    if (aspectRatio < kMinimumAspectRatio ||
        component.bb.width < kMinimumWidthInPitches * pitch ||
        component.bb.height > kMaximumHeightInPitches * pitch)
      continue;

    component.ignored = true;
    const int previousLine = component.line;
    component.line = -1;
    page.assignmentCorrections.push_back(
        "ignored-sajda-bar component=" + std::to_string(index) +
        " line=" + std::to_string(previousLine));
  }
}

int BaselineSegmenter::chooseLineByOverlap(const CC& c, const std::vector<LineModel>& lines) {
  // Choose the line whose Voronoi region [yTop,yBot] overlaps bbox the most.
  int best = -1;
  int bestOverlap = -1;

  int y0 = c.bb.y;
  int y1 = c.bb.y + c.bb.height;

  for (int i = 0; i < lines.size(); ++i) {
    int a = std::max(y0, lines[i].yTop);
    int b = std::min(y1, lines[i].yBot);
    int ov = std::max(0, b - a);
    if (ov > bestOverlap) {
      bestOverlap = ov;
      best = i;
    }
  }

  // If the component does not overlap any line region, assign it to the
  // nearest detected baseline by centroid.
  if (bestOverlap <= 0) {
    int best2 = 0;
    double bestD = 1e18;
    for (int i = 0; i < lines.size(); ++i) {
      double d = std::abs(c.cent.y - lines[i].yBase);
      if (d < bestD) {
        bestD = d;
        best2 = i;
      }
    }
    return best2;
  }
  return best;
}

std::vector<int> BaselineSegmenter::sortIndicesRTL(
    const std::vector<CC>& comps, int lineIndex, const cv::Mat& labels,
    int baselineY, const Params& params) {
  std::vector<int> idx;
  idx.reserve(comps.size());
  for (int i = 0; i < comps.size(); ++i)
    if (comps[i].line == lineIndex) idx.push_back(i);

  std::vector<int> orderingRight(comps.size(), -1);
  if (params.orderByBaselineCorridor) {
    const int corridorTop =
        std::max(0, baselineY - params.boundaryCorridorAbove);
    const int corridorBottom =
        std::min(labels.rows - 1, baselineY + params.boundaryCorridorBelow);
    for (const int componentIndex : idx) {
      const CC& component = comps[componentIndex];
      const int top = std::max(component.bb.y, corridorTop);
      const int bottom =
          std::min(component.bb.y + component.bb.height - 1, corridorBottom);
      if (bottom < top) continue;
      for (int x = component.bb.x + component.bb.width - 1;
           x >= component.bb.x; --x) {
        int ink = 0;
        for (int y = top; y <= bottom; ++y)
          if (labels.at<int>(y, x) == component.label) ++ink;
        if (ink < 2) continue;
        orderingRight[componentIndex] = x;
        break;
      }
    }
  }

  // Use the rightmost substantial ink close to the baseline for ordering body
  // components. Floating marks that do not reach the corridor retain their
  // complete-bounding-box right edge.
  std::sort(idx.begin(), idx.end(), [&](int a, int b) {
    const int rightA = orderingRight[a] >= 0
                           ? orderingRight[a]
                           : comps[a].bb.x + comps[a].bb.width - 1;
    const int rightB = orderingRight[b] >= 0
                           ? orderingRight[b]
                           : comps[b].bb.x + comps[b].bb.width - 1;

    if (rightA == rightB) return comps[a].cent.y < comps[b].cent.y;
    return rightA > rightB;
  });
  return idx;
}

cv::Rect BaselineSegmenter::unionRange(const std::vector<CC>& comps, const std::vector<int>& idx, int k, int i) {
  if (k >= i) return cv::Rect();
  cv::Rect u = comps[idx[k]].bb;
  for (int t = k + 1; t < i; ++t) u |= comps[idx[t]].bb;
  return cv::Rect(u.x, u.y, u.width, u.height);
}

bool BaselineSegmenter::run(const cv::Mat& binInk255,
                            const cv::Mat& srcBgrOrGray,
                            const std::vector<BaselineEstimate>& lineEstimates,
                            const PageLayout& pageLayout, PageModel& page) {
  auto& outLines = page.lines;
  outLines.clear();
  if (binInk255.empty() || binInk255.type() != CV_8UC1) {
    std::cerr << "BaselineSegmenter: binary image is empty or not CV_8UC1\n";
    return false;
  }
  if (lineEstimates.empty()) {
    std::cerr << "BaselineSegmenter: no line estimates\n";
    return false;
  }

  const int H = binInk255.rows;

  // 1) Extract connected components
  extractComponents(binInk255, m_p.minCompArea, page);
  std::vector<CC>& comps = page.comps;
  if (comps.empty()) {
    std::cerr << "BaselineSegmenter: no components above minimum area "
              << m_p.minCompArea << "; ink pixels "
              << cv::countNonZero(binInk255) << '\n';
    return false;
  }
  loadThreeDotsAboveDetections(comps);

  // 2) Build line models (Voronoi bands between baselines)
  std::vector<LineModel> lines = buildLineModels(lineEstimates, H);

  // 3) Assign each component to a line (by overlap with bands)
  for (CC& c : comps) c.line = chooseLineByOverlap(c, lines);

  // Prepare output LineBox list
  outLines.resize(lines.size());
  for (int li = 0; li < lines.size(); ++li) {
    outLines[li].rect = cv::Rect(0, lines[li].yTop, binInk255.cols, lines[li].yBot - lines[li].yTop + 1);
    outLines[li].words.clear();
    outLines[li].baseLine = lines[li].yBase;
  }
  classifyComponentsPreliminary(page);
  if (!m_p.markModelPath.empty())
    classifyMarksWithOnnx(page, m_p.markModelPath);
  ignoreSajdaBars(page);

  std::vector<MarkInventory> expectedMarksByLine(outLines.size());
  for (std::size_t line = 0;
       line < outLines.size() && line < pageLayout.lines.size(); ++line) {
    for (const WordInfo& word : makeExpectations(pageLayout.lines[line]))
      for (std::size_t mark = 0; mark < kMarkClassCount; ++mark)
        expectedMarksByLine[line][mark] += word.expectedMarks[mark];
  }
  ComponentRecognizerParams recognizerParams;
  recognizerParams.waqfTemplateRoot = m_p.waqfTemplateRoot;
  recognizerParams.threeDotsAboveTemplateRoot =
      m_p.threeDotsAboveTemplateRoot;
  recognizerParams.attachedAlefHamzaAboveTemplateRoot =
      m_p.attachedAlefHamzaAboveTemplateRoot;
  recognizerParams.markTemplateRoot = m_p.markTemplateRoot;
  recognizerParams.cacheRoot = m_p.recognizerCacheRoot;
  recognizerParams.sourceName = m_p.recognizerSource;
  recognizerParams.pageNumber = m_p.pageNumber;
  recognizerParams.reportTimings = m_p.reportTimings;
  recognizerParams.resetCache = m_p.resetRecognizerCache;
  recognizerParams.minimumAttachedAlefHamzaAboveProbability =
      m_p.minimumAttachedAlefHamzaAboveProbability;
  ComponentRecognizer recognizer(std::move(recognizerParams));
  recognizer.recognize(page, expectedMarksByLine, srcBgrOrGray,
                       m_p.otsuThreshold);

  // Reviewed manifests remain optional overrides of automatic recognition.
  applyDetectedWaqfLineAssignments(comps, lines, page);

  // Word segmentation per line using DP + word expectations
  int L = outLines.size();
  for (int li = 0; li < L; ++li) {
    // Build word infos from text
    LineLayout lineLayout = (li < pageLayout.lines.size() ? pageLayout.lines[li] : LineLayout());
    std::vector<WordInfo> exp = makeExpectations(lineLayout);

    // Indices of components for this line in RTL order
    std::vector<int> idx = sortIndicesRTL(comps, li, page.labels,
                                          lines[li].yBase, m_p);
    if (idx.empty() || exp.empty()) continue;

    // DP segmentation -> ranges per token
    std::vector<std::pair<int, int>> ranges;
    if (!dpSegmentLine_DiscretePrevLeft(comps, idx, exp, m_p, page.labels,
                                        lines[li].yBase, ranges)) {
      // Fallback: single word bbox containing everything
      WordBox wb;
      wb.lineIndex = li;
      wb.wordIndex = 0;
      wb.wordId = wordId(li, 0);
      wb.expectedText = lineLayout.words.front().text;
      for (auto it = std::next(lineLayout.words.begin());
           it != lineLayout.words.end(); ++it)
        wb.expectedText += " " + it->text;
      cv::Rect u = unionRange(comps, idx, 0, idx.size());
      wb.rect = u;
      for (int id = 0; id < idx.size(); id++) {
        wb.compIds.push_back(idx[id]);
      }
      for (const WordInfo& word : exp)
        for (const SubWordInfo& subWord : word.subWords) {
          WordPawExpectation expectation;
          expectation.text = subWord.utf8Text;
          expectation.expectedBaseCount = subWord.baseCount;
          expectation.expectedBodyComponents = subWord.expBodyConn;
          expectation.expectedMarkPaths = subWord.expMarkPaths;
          expectation.expectedSmallLetterPaths = subWord.expSmallLetterPaths;
          expectation.expectedMarks = subWord.expectedMarks;
          expectation.expectedMarkInstances = subWord.expectedMarkInstances;
          wb.pawExpectations.push_back(std::move(expectation));
        }
      outLines[li].words.push_back(wb);
      continue;
    }

    // Convert ranges to WordBox
    outLines[li].words.reserve(ranges.size());
    for (int w = 0; w < ranges.size(); ++w) {
      int k = ranges[w].first;
      int i = ranges[w].second;

      cv::Rect r = unionRange(comps, idx, k, i);

      WordBox wb;
      wb.lineIndex = li;
      wb.wordIndex = w;
      wb.expectedText = exp[w].text;
      wb.expectedBodyComponents = exp[w].expBodyConn;
      wb.expectedMarkComponents = exp[w].expMarkPaths;
      wb.expectedSmallLetterComponents = exp[w].expSmallLetterPaths;
      wb.expectedMarks = exp[w].expectedMarks;
      for (const SubWordInfo& subWord : exp[w].subWords) {
        WordPawExpectation expectation;
        expectation.text = subWord.utf8Text;
        expectation.expectedBaseCount = subWord.baseCount;
        expectation.expectedBodyComponents = subWord.expBodyConn;
        expectation.expectedMarkPaths = subWord.expMarkPaths;
        expectation.expectedSmallLetterPaths = subWord.expSmallLetterPaths;
        expectation.expectedMarks = subWord.expectedMarks;
        expectation.expectedMarkInstances = subWord.expectedMarkInstances;
        wb.pawExpectations.push_back(std::move(expectation));
      }
      int recognizedOneComponentThreeDots = 0;
      int recognizedAttachedAlefHamzaAbove = 0;
      for (int id = k; id < i; ++id)
        if (comps[idx[id]].detectedThreeDotsAboveComponentCount == 1)
          ++recognizedOneComponentThreeDots;
      for (int id = k; id < i; ++id)
        if (comps[idx[id]].attachedAlefHamzaAboveProbability >=
            recognizerParams.minimumAttachedAlefHamzaAboveProbability)
          ++recognizedAttachedAlefHamzaAbove;
      const int threeDotsAdjustment = std::min(
          exp[w].expectedMarks[static_cast<std::size_t>(
              MarkClass::DotAboveThree)],
          recognizedOneComponentThreeDots);
      wb.recognizedAttachedAlefHamzaAboveCount = std::min(
          exp[w].expectedMarks[static_cast<std::size_t>(
              MarkClass::HamzaAbove)],
          recognizedAttachedAlefHamzaAbove);
      wb.recognizedComponentCountAdjustment =
          threeDotsAdjustment + wb.recognizedAttachedAlefHamzaAboveCount;
      for (int id = k; id < i; ++id)
        wb.observedSmallLetterComponents +=
            comps[idx[id]].smallLetterProbability;
      wb.rect = r;
      wb.wordId = wordId(li, w);
      for (int id = k; id < i; id++) {
        wb.compIds.push_back(idx[id]);
      }
      outLines[li].words.push_back(wb);
    }
  }

  return true;
}

struct State {
  int left;   // prev word bbox left in pixels
  int width;  // prev word bbox width in pixels
  int corridorLeft;
  int corridorWidth;
  bool operator==(const State& o) const noexcept {
    return left == o.left && width == o.width &&
           corridorLeft == o.corridorLeft &&
           corridorWidth == o.corridorWidth;
  }
};

struct StateHash {
  size_t operator()(const State& s) const noexcept {
    size_t result = 0;
    for (const int value :
         {s.left, s.width, s.corridorLeft, s.corridorWidth})
      result ^= std::hash<int>{}(value) + 0x9e3779b9 + (result << 6) +
                (result >> 2);
    return result;
  }
};

struct BackPtr {
  int prevK = -1;
  State prevState{0, 0, 0, 0};
};

static inline double boundaryCostNormRTL(const State& prev, int curRight,
                                         int curWidth, int curCorridorRight,
                                         int curCorridorWidth,
                                         bool useCorridor, double wOverlap,
                                         double wGapBonus,
                                         double gapScalePx) {
  const bool corridorValid =
      useCorridor && prev.corridorWidth > 0 && curCorridorWidth > 0;
  const int previousLeft =
      corridorValid ? prev.corridorLeft : prev.left;
  const int previousWidth =
      corridorValid ? prev.corridorWidth : prev.width;
  const int currentRight = corridorValid ? curCorridorRight : curRight;
  const int currentWidth =
      corridorValid ? curCorridorWidth : curWidth;
  const int ov = std::max(0, currentRight - previousLeft + 1);
  const int g = std::max(0, previousLeft - currentRight - 1);
  // Normalized overlap in [0..]
  const int denom = std::max(1, std::min(previousWidth, currentWidth));
  const double ovNorm = double(ov) / double(denom);

  // Gap reward with diminishing returns (dimensionless)
  const double gapReward = -std::log(1.0 + double(g) / std::max(1.0, gapScalePx));

  return wOverlap * ovNorm + wGapBonus * gapReward;
}

bool BaselineSegmenter::dpSegmentLine_DiscretePrevLeft(const std::vector<CC>& comps, const std::vector<int>& idx,
                                                       const std::vector<WordInfo>& exp, const Params& p,
                                                       const cv::Mat& labels, int baselineY,
                                                       std::vector<std::pair<int, int>>& outRanges) {
  outRanges.clear();
  const int N = (int)idx.size();
  const int M = (int)exp.size();
  if (N == 0 || M == 0) return false;

  // Prefix comps count (each CC counts 1)
  std::vector<int> prefComps(N + 1, 0);
  for (int i = 0; i < N; ++i) prefComps[i + 1] = prefComps[i] + 1;
  struct CorridorInterval {
    int left = 0;
    int right = -1;
    bool valid() const { return right >= left; }
  };
  std::vector<CorridorInterval> componentCorridors(N);
  const int corridorTop = std::max(0, baselineY - p.boundaryCorridorAbove);
  const int corridorBottom =
      std::min(labels.rows - 1, baselineY + p.boundaryCorridorBelow);
  for (int order = 0; order < N; ++order) {
    const CC& component = comps[idx[order]];
    const int top = std::max(component.bb.y, corridorTop);
    const int bottom = std::min(component.bb.y + component.bb.height - 1,
                                corridorBottom);
    if (bottom < top) continue;
    int left = labels.cols;
    int right = -1;
    for (int x = component.bb.x;
         x < component.bb.x + component.bb.width; ++x) {
      int ink = 0;
      for (int y = top; y <= bottom; ++y)
        if (labels.at<int>(y, x) == component.label) ++ink;
      if (ink < 2) continue;
      left = std::min(left, x);
      right = std::max(right, x);
    }
    if (right >= left) componentCorridors[order] = {left, right};
  }
  std::vector<std::array<double, kMarkClassCount>> prefMarkMass(N + 1);
  const bool hasMarkEvidence = !p.markModelPath.empty();
  if (p.markInventoryPenalty > 0.0 && hasMarkEvidence) {
    for (int i = 0; i < N; ++i) {
      prefMarkMass[i + 1] = prefMarkMass[i];
      for (std::size_t mark = 0; mark < kMarkClassCount; ++mark)
        prefMarkMass[i + 1][mark] +=
            comps[idx[i]].markClassProbabilities[mark];
    }
  }

  using CostMap = std::unordered_map<State, double, StateHash>;
  using BackMap = std::unordered_map<State, BackPtr, StateHash>;

  std::vector<std::vector<CostMap>> dp(M + 1, std::vector<CostMap>(N + 1));
  std::vector<std::vector<BackMap>> back(M + 1, std::vector<BackMap>(N + 1));

  // Base: 0 words, 0 comps. Dummy state (unused until j==1).
  dp[0][0][State{0, 0, 0, 0}] = 0.0;

  const double INF = 1e30;

  auto segCost = [&](int k, int i, int wordJ) -> double {
    const int nbComps = prefComps[i] - prefComps[k];

    const WordInfo& E = exp[wordJ];

    const int expectedThreeDots = E.expectedMarks[static_cast<std::size_t>(MarkClass::DotAboveThree)];
    int detectedOneComponentThreeDots = 0;
    for (int component = k; component < i; ++component)
      if (comps[idx[component]]
              .detectedThreeDotsAboveComponentCount == 1)
        ++detectedOneComponentThreeDots;
    const int threeDotsReduction = std::min(
        expectedThreeDots, detectedOneComponentThreeDots);
    int detectedAttachedAlefHamzaAbove = 0;
    for (int component = k; component < i; ++component)
      if (comps[idx[component]].attachedAlefHamzaAboveProbability >=
          p.minimumAttachedAlefHamzaAboveProbability)
        ++detectedAttachedAlefHamzaAbove;
    const int expectedHamzaAbove =
        E.expectedMarks[static_cast<std::size_t>(MarkClass::HamzaAbove)];
    const int attachedAlefHamzaReduction =
        std::min(expectedHamzaAbove, detectedAttachedAlefHamzaAbove);
    const int expectedCount =
        E.expBodyConn + E.expMarkPaths - threeDotsReduction -
        attachedAlefHamzaReduction;
    const int missing = std::max(0, expectedCount - nbComps);
    const int extra = std::max(0, nbComps - expectedCount);
    const double componentCountCost =
        p.missingComponentPenalty * double(missing) +
        p.extraComponentPenalty * double(extra);

    double observedSmallLetters = 0.0;
    for (int component = k; component < i; ++component)
      observedSmallLetters +=
          comps[idx[component]].smallLetterProbability;
    const double smallLetterDifference =
        std::abs(observedSmallLetters - E.expSmallLetterPaths);

    double observedBodyMass = 0.0;
    for (int component = k; component < i; ++component)
      observedBodyMass += comps[idx[component]].bodyProbability;
    const double bodyEvidenceDifference =
        std::abs(observedBodyMass - double(E.expBodyConn));

    double markInventoryDifference = 0.0;
    if (p.markInventoryPenalty > 0.0 && hasMarkEvidence) {
      for (std::size_t mark = 0; mark < kMarkClassCount; ++mark) {
        const auto markClass = static_cast<MarkClass>(mark);
        if (markClass == MarkClass::Other) continue;
        const double observedMarkMass =
            prefMarkMass[i][mark] - prefMarkMass[k][mark];
        const double classWeight =
            markClass == MarkClass::SmallYeh ? p.smallYehClassWeight : 1.0;
        markInventoryDifference +=
            classWeight *
            std::abs(observedMarkMass - E.expectedMarks[mark]);
      }
    }

    return componentCountCost +
           p.smallLetterPenalty * smallLetterDifference +
           p.bodyEvidencePenalty * bodyEvidenceDifference +
           p.markInventoryPenalty * markInventoryDifference;
  };

  for (int j = 1; j <= M; ++j) {
    for (int i = 1; i <= N; ++i) {
      const int kMin = std::max(0, i - p.maxTokenSpan);

      // Incrementally build bbox for segment [k,i) while k decreases
      bool hasBox = false;
      cv::Rect boxKI;
      CorridorInterval corridorKI;

      for (int k = i - 1; k >= kMin; --k) {
        // IMPORTANT: always update bbox, even if dp[j-1][k] is empty
        const cv::Rect& bb = comps[idx[k]].bb;
        const cv::Rect r(bb.x, bb.y, bb.width, bb.height);
        if (!hasBox) {
          boxKI = r;
          hasBox = true;
        } else {
          boxKI |= r;
        }
        const CorridorInterval& componentCorridor = componentCorridors[k];
        if (componentCorridor.valid()) {
          if (!corridorKI.valid())
            corridorKI = componentCorridor;
          else {
            corridorKI.left =
                std::min(corridorKI.left, componentCorridor.left);
            corridorKI.right =
                std::max(corridorKI.right, componentCorridor.right);
          }
        }

        auto& prevMap = dp[j - 1][k];
        if (prevMap.empty()) continue;

        const double costSeg = segCost(k, i, j - 1);

        // Current segment state (for dp[j][i]): store its left + width
        const int corridorWidth =
            corridorKI.valid() ? corridorKI.right - corridorKI.left + 1 : 0;
        const State curState{boxKI.x, boxKI.width, corridorKI.left,
                             corridorWidth};
        const int curRight = boxKI.x + boxKI.width - 1;
        const int curWidth = boxKI.width;

        // Transition from all prev states
        for (const auto& it : prevMap) {
          const State& prevState = it.first;
          const double prevCost = it.second;

          double costB = 0.0;
          if (j > 1) {
            const bool beginsWithInitialKaf =
                !exp[j - 1].baseText.empty() &&
                exp[j - 1].baseText.front() == U'ك';
            costB = boundaryCostNormRTL(
                prevState, curRight, curWidth, corridorKI.right,
                corridorWidth, beginsWithInitialKaf, p.wOverlapX,
                p.wGapBonus, p.gapScalePx);
          }

          const double cand = prevCost + costSeg + costB;

          auto& curMap = dp[j][i];
          auto found = curMap.find(curState);
          if (found == curMap.end() || cand < found->second) {
            curMap[curState] = cand;
            back[j][i][curState] = BackPtr{k, prevState};
          }
        }
      }
    }
  }

  // Choose best ending state at (M,N)
  if (dp[M][N].empty()) return false;

  State bestState{0, 0, 0, 0};
  double bestCost = INF;
  for (const auto& it : dp[M][N]) {
    if (it.second < bestCost) {
      bestCost = it.second;
      bestState = it.first;
    }
  }
  if (bestCost >= INF / 2) return false;

  // Backtrack
  outRanges.resize(M);
  int i = N;
  State st = bestState;

  for (int j = M; j >= 1; --j) {
    auto itB = back[j][i].find(st);
    if (itB == back[j][i].end()) return false;

    const BackPtr bp = itB->second;
    const int k = bp.prevK;

    outRanges[j - 1] = {k, i};  // [k,i)
    i = k;
    st = bp.prevState;
  }

  return (i == 0);
}
