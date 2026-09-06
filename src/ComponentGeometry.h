#pragma once

#include <algorithm>

#include <opencv2/opencv.hpp>

// A component bbox can overlap a baseline band even when the component's
// actual ink stops just above the baseline. This is common for large sukun
// marks in the Old Madinah scan. Use a narrow pixel-support test when a
// baseline decision must distinguish body ink from a nearby floating mark.
inline bool componentHasInkNearBaseline(const cv::Mat& labels, int label,
                                        int baselineY, int halfBand,
                                        const cv::Rect& componentBox) {
  if (labels.empty() || labels.type() != CV_32S || label <= 0) return false;

  const cv::Rect imageBox(0, 0, labels.cols, labels.rows);
  const cv::Rect searchBox = componentBox.empty() ? imageBox
                                                   : (componentBox & imageBox);
  if (searchBox.empty()) return false;

  const int top = std::max(0, baselineY - halfBand);
  const int bottom = std::min(labels.rows - 1, baselineY + halfBand);
  for (int y = top; y <= bottom; ++y) {
    const int* row = labels.ptr<int>(y);
    for (int x = searchBox.x; x < searchBox.x + searchBox.width; ++x)
      if (row[x] == label) return true;
  }
  return false;
}

inline bool componentHasInkNearBaseline(const cv::Mat& labels, int label,
                                        int baselineY, int halfBand = 5) {
  return componentHasInkNearBaseline(labels, label, baselineY, halfBand,
                                     cv::Rect());
}

inline bool componentStartsNearBaseline(const cv::Rect& componentBox,
                                        int baselineY, int tolerance = 5) {
  return componentBox.y >= baselineY - tolerance &&
         componentBox.y <= baselineY + tolerance;
}
