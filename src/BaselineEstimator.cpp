#include "BaselineEstimator.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>

static int clampi(int v, int lo, int hi) { return std::max(lo, std::min(hi, v)); }

std::vector<BaselineEstimate> estimateLinesByComponentGaps(const cv::Mat& binInk255, int expectedLineCount, const BaselineEstParams& p, int minimumComponentArea) {
  CV_Assert(!binInk255.empty() && binInk255.type() == CV_8UC1);
  if (expectedLineCount <= 0) return {};

  const int height = binInk255.rows;
  const int width = binInk255.cols;
  const int lineCount = expectedLineCount;
  const double nominalPitch = height / double(lineCount);

  cv::Mat labels, stats, centroids;
  const int componentCount = cv::connectedComponentsWithStats(binInk255, labels, stats, centroids, 8, CV_32S);
  std::vector<int> topEdgesAtRow(height, 0);
  std::vector<int> bottomEdgesAtRow(height, 0);
  std::vector<int> crossingDifference(height + 1, 0);
  for (int label = 1; label < componentCount; ++label) {
    const int area = stats.at<int>(label, cv::CC_STAT_AREA);
    if (area < minimumComponentArea) continue;
    const int top = stats.at<int>(label, cv::CC_STAT_TOP);
    const int bottom = top + stats.at<int>(label, cv::CC_STAT_HEIGHT) - 1;
    ++topEdgesAtRow[top];
    ++bottomEdgesAtRow[bottom];
    ++crossingDifference[top];
    if (bottom + 1 < static_cast<int>(crossingDifference.size()))
      --crossingDifference[bottom + 1];
  }
  std::vector<int> crossingComponents(height, 0);
  int active = 0;
  for (int y = 0; y < height; ++y) {
    active += crossingDifference[y];
    crossingComponents[y] = active;
  }
  std::vector<int> rowInk(height, 0);
  for (int y = 0; y < height; ++y)
    rowInk[y] = cv::countNonZero(binInk255.row(y));

  struct GapCandidate {
    int row = 0, topGapLength = 0, bottomGapLength = 0;
    int crossing = 0, ink = 0;
  };
  std::vector<GapCandidate> candidates;
  auto emptyRunLengths = [&](const std::vector<int>& edgeEvents) {
    std::vector<int> lengths(height, 0);
    int previousEdge = -1;
    for (int y = 0; y < height; ++y) {
      if (edgeEvents[y] == 0) continue;
      const int first = previousEdge + 1;
      const int last = y - 1;
      const int length = last - first + 1;
      if (previousEdge >= 0 && length >= 12)
        std::fill(lengths.begin() + first, lengths.begin() + last + 1, length);
      previousEdge = y;
    }
    return lengths;
  };
  const std::vector<int> topGapLengths = emptyRunLengths(topEdgesAtRow);
  const std::vector<int> bottomGapLengths = emptyRunLengths(bottomEdgesAtRow);
  auto addGapCandidates = [&](const std::vector<int>& edgeEvents) {
    int previousEdge = -1;
    for (int y = 0; y < height; ++y) {
      if (edgeEvents[y] == 0) continue;
      const int first = previousEdge + 1;
      const int last = y - 1;
      const int length = last - first + 1;
      if (previousEdge >= 0 && length >= 12) {
        int bestRow = first;
        auto rowCost = [&](int row) { return 600.0 * crossingComponents[row] + rowInk[row] + 0.002 * std::abs(2 * row - first - last); };
        double bestCost = rowCost(bestRow);
        for (int row = first + 1; row <= last; ++row) {
          const double cost = rowCost(row);
          if (cost < bestCost) {
            bestCost = cost;
            bestRow = row;
          }
        }
        candidates.push_back({bestRow, topGapLengths[bestRow], bottomGapLengths[bestRow], crossingComponents[bestRow], rowInk[bestRow]});
      }
      previousEdge = y;
    }
  };
  addGapCandidates(topEdgesAtRow);
  addGapCandidates(bottomEdgesAtRow);
  std::sort(candidates.begin(), candidates.end(), [](const GapCandidate& left, const GapCandidate& right) { return left.row < right.row; });
  candidates.erase(std::unique(candidates.begin(), candidates.end(), [](const GapCandidate& left, const GapCandidate& right) { return left.row == right.row; }), candidates.end());

  const int separatorCount = lineCount - 1;
  std::vector<std::vector<double>> dp(separatorCount, std::vector<double>(candidates.size(), -std::numeric_limits<double>::infinity()));
  std::vector<std::vector<int>> previous(separatorCount, std::vector<int>(candidates.size(), -1));
  auto unary = [&](const GapCandidate& candidate) {
    // Preserve the proven top-edge signal and use the bottom-up view as
    // corroboration. A bottom-only candidate can still win, but it must have
    // substantially cleaner crossing/ink evidence than a top-edge gap.
    return 1.4 * std::log1p(candidate.topGapLength) + 0.35 * std::log1p(candidate.bottomGapLength) - 1.8 * candidate.crossing - 0.0015 * candidate.ink;
  };
  if (separatorCount > 0)
    for (size_t candidate = 0; candidate < candidates.size(); ++candidate) {
      const double firstBandPenalty = 2.0 * std::abs(candidates[candidate].row - nominalPitch) / nominalPitch;
      dp[0][candidate] = unary(candidates[candidate]) - firstBandPenalty;
    }
  for (int separator = 1; separator < separatorCount; ++separator) {
    for (size_t current = 0; current < candidates.size(); ++current) {
      for (size_t prior = 0; prior < current; ++prior) {
        if (!std::isfinite(dp[separator - 1][prior])) continue;
        const double spacing =
            candidates[current].row - candidates[prior].row;
        if (spacing < 0.50 * nominalPitch || spacing > 1.50 * nominalPitch)
          continue;
        const double transition = -2.0 * std::abs(spacing - nominalPitch) / nominalPitch;
        const double score = dp[separator - 1][prior] + unary(candidates[current]) + transition;
        if (score > dp[separator][current]) {
          dp[separator][current] = score;
          previous[separator][current] = static_cast<int>(prior);
        }
      }
    }
  }

  std::vector<int> boundaries(lineCount + 1);
  boundaries.front() = 0;
  boundaries.back() = height;
  int selected = -1;
  if (separatorCount > 0 && !candidates.empty()) {
    double bestFinalScore = -std::numeric_limits<double>::infinity();
    for (size_t candidate = 0; candidate < candidates.size(); ++candidate) {
      if (!std::isfinite(dp.back()[candidate])) continue;
      const double finalBandPenalty = 2.0 * std::abs((height - candidates[candidate].row) - nominalPitch) / nominalPitch;
      const double score = dp.back()[candidate] - finalBandPenalty;
      if (score > bestFinalScore) {
        bestFinalScore = score;
        selected = static_cast<int>(candidate);
      }
    }
  }
  for (int separator = separatorCount - 1; separator >= 0; --separator) {
    if (selected >= 0) {
      boundaries[separator + 1] = candidates[selected].row;
      selected = previous[separator][selected];
    } else {
      boundaries[separator + 1] = static_cast<int>(std::lround((separator + 1) * nominalPitch));
    }
  }

  // Baselines are estimated only after the component-gap DP has fixed the
  // line bands. This body mask is not involved in deciding line membership.
  const int minimumArea = std::max(p.minBodyArea, static_cast<int>(0.008 * nominalPitch * nominalPitch));
  const int minimumHeight = std::max(p.minBodyHeight, static_cast<int>(0.16 * nominalPitch));
  std::vector<uint8_t> body(componentCount, 0);
  for (int label = 1; label < componentCount; ++label) {
    const int area = stats.at<int>(label, cv::CC_STAT_AREA);
    const int componentHeight = stats.at<int>(label, cv::CC_STAT_HEIGHT);
    if (area >= minimumArea && componentHeight >= minimumHeight)
      body[label] = 1;
  }

  constexpr int kHorizontalBins = 96;
  const int binWidth = std::max(1, (width + kHorizontalBins - 1) / kHorizontalBins);
  const int verticalRadius = std::max(2, static_cast<int>(0.012 * nominalPitch));

  // Prefix summaries make every candidate evaluation O(horizontal bins)
  // instead of rescanning a high-resolution corridor.
  std::vector<std::array<int, kHorizontalBins>> binPrefix(height + 1);
  std::vector<int> pixelPrefix(height + 1, 0);
  std::vector<int> bodyComponentDifference(height + 1, 0);
  for (int label = 1; label < componentCount; ++label) {
    if (!body[label]) continue;
    const int top = stats.at<int>(label, cv::CC_STAT_TOP);
    const int bottom = top + stats.at<int>(label, cv::CC_STAT_HEIGHT);
    ++bodyComponentDifference[top];
    if (bottom < static_cast<int>(bodyComponentDifference.size()))
      --bodyComponentDifference[bottom];
  }
  std::vector<int> intersectingComponents(height, 0);
  int activeComponents = 0;
  for (int y = 0; y < height; ++y) {
    activeComponents += bodyComponentDifference[y];
    intersectingComponents[y] = activeComponents;
    std::array<uint8_t, kHorizontalBins> occupied{};
    int bodyPixels = 0;
    const int* row = labels.ptr<int>(y);
    for (int x = 0; x < width; ++x) {
      const int label = row[x];
      if (label == 0 || !body[label]) continue;
      occupied[std::min(kHorizontalBins - 1, x / binWidth)] = 1;
      ++bodyPixels;
    }
    binPrefix[y + 1] = binPrefix[y];
    for (int bin = 0; bin < kHorizontalBins; ++bin)
      binPrefix[y + 1][bin] += occupied[bin];
    pixelPrefix[y + 1] = pixelPrefix[y] + bodyPixels;
  }

  std::vector<BaselineEstimate> result;
  result.reserve(lineCount);
  for (int line = 0; line < lineCount; ++line) {
    const int top = boundaries[line];
    const int bottom = boundaries[line + 1] - 1;
    // The Arabic baseline normally lies in the lower part of its line band.
    // Keep the search broad enough for irregular short-surah pages.
    const int searchTop = std::max(top, top + static_cast<int>(0.38 * (bottom - top)));
    const int searchBottom =
        std::min(bottom, top + static_cast<int>(0.86 * (bottom - top)));
    std::vector<double> scores(std::max(0, searchBottom - searchTop + 1), 0.0);
    double maximum = 0.0;
    double sum = 0.0;
    for (int y = searchTop; y <= searchBottom; ++y) {
      const int corridorTop = std::max(top, y - verticalRadius);
      const int corridorBottom = std::min(bottom, y + verticalRadius);
      int coverage = 0;
      for (int bin = 0; bin < kHorizontalBins; ++bin)
        coverage += binPrefix[corridorBottom + 1][bin] > binPrefix[corridorTop][bin];
      const int components = intersectingComponents[y];
      const int bodyPixels = pixelPrefix[corridorBottom + 1] - pixelPrefix[corridorTop];
      // Coverage dominates: a dense ornament in one location must not beat
      // body strokes distributed across the complete text line.
      const double score = coverage + 0.35 * components + 0.00002 * bodyPixels;
      scores[y - searchTop] = score;
      maximum = std::max(maximum, score);
      sum += score;
    }

    const auto bestIt = std::max_element(scores.begin(), scores.end());
    const int bandReference = top + static_cast<int>(0.65 * (bottom - top));
    int bestY = scores.empty() ? clampi(bandReference, top, bottom) : searchTop + static_cast<int>(bestIt - scores.begin());
    const double mean = scores.empty() ? 0.0 : sum / scores.size();
    const double prominence = maximum > 0.0 ? std::clamp((maximum - mean) / maximum, 0.0, 1.0) : 0.0;
    const bool boundaryHit = scores.empty() || bestY == searchTop || bestY == searchBottom;
    result.push_back({bestY, prominence, maximum, boundaryHit, top, bottom});
  }
  return result;
}
