#include "SpacingGeometry.h"

#include <algorithm>
#include <vector>

HorizontalInkSpan measureBaselineCorridorSpan(
    const cv::Mat& wordMask,
    const MarkSegmentationResult& segmentation,
    int baselineY,
    const SpacingGeometryParams& params) {
  HorizontalInkSpan result;
  if (wordMask.empty() || wordMask.type() != CV_8UC1) return result;

  cv::Mat labels, stats, centroids;
  cv::connectedComponentsWithStats(wordMask, labels, stats, centroids, 8, CV_32S);
  cv::Mat bodyMask = cv::Mat::zeros(wordMask.size(), CV_8UC1);
  for (const auto& component : segmentation.componentTypes) {
    if (component.componentType == ComponentTypeEnum::Body) {
      bodyMask.setTo(255, labels == component.componentLabel);
    }
  }

  const int top = std::clamp(baselineY - params.corridorAbove, 0, bodyMask.rows - 1);
  const int bottom = std::clamp(baselineY + params.corridorBelow, 0, bodyMask.rows - 1);
  if (bottom < top) return result;

  std::vector<uint8_t> occupied(bodyMask.cols, 0);
  for (int x = 0; x < bodyMask.cols; ++x) {
    int ink = 0;
    for (int y = top; y <= bottom; ++y) {
      if (bodyMask.at<uint8_t>(y, x) != 0) ++ink;
    }
    if (ink >= params.minInkPerColumn) {
      occupied[x] = 1;
      result.inkPixels += ink;
    }
  }

  // Fill very small holes in the one-dimensional support. They are usually
  // counters or scan breaks, not meaningful boundaries of the spacing span.
  int lastOccupied = -1;
  for (int x = 0; x < static_cast<int>(occupied.size()); ++x) {
    if (!occupied[x]) continue;
    if (lastOccupied >= 0 && x - lastOccupied - 1 <= params.closeGapColumns) {
      std::fill(occupied.begin() + lastOccupied + 1, occupied.begin() + x, 1);
    }
    lastOccupied = x;
  }

  auto left = std::find(occupied.begin(), occupied.end(), 1);
  auto right = std::find(occupied.rbegin(), occupied.rend(), 1);
  if (left == occupied.end() || right == occupied.rend()) return result;

  result.left = static_cast<int>(std::distance(occupied.begin(), left));
  result.right = static_cast<int>(occupied.size() - 1 - std::distance(occupied.rbegin(), right));
  result.width = result.right - result.left + 1;
  result.valid = result.width > 0;
  return result;
}

void computeLineSpacingGeometry(LineBox& line) {
  for (size_t index = 0; index < line.words.size(); ++index) {
    WordBox& current = line.words[index];
    current.bboxGapToNextPx = 0;
    current.spacingGapToNextPx = 0;
    if (index + 1 >= line.words.size()) continue;

    const WordBox& next = line.words[index + 1];
    current.bboxGapToNextPx =
        current.rect.x - (next.rect.x + next.rect.width - 1) - 1;
    if (current.spacingSpan.valid && next.spacingSpan.valid) {
      const int currentLeft = current.rect.x + current.spacingSpan.left;
      const int nextRight = next.rect.x + next.spacingSpan.right;
      current.spacingGapToNextPx = currentLeft - nextRight - 1;
    }
  }
}
