#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

struct BaselineEstParams {
  int minBodyArea = 800;
  int minBodyHeight = 40;
};

struct BaselineEstimate {
  int y = 0;
  double confidence = 0.0;
  double peakScore = 0.0;
  bool searchBoundaryHit = false;
  int bandTop = -1;
  int bandBottom = -1;
};

// Select the requested number of line bands from top- and bottom-edge component
// gaps, then estimate a baseline from body ink independently inside each band.
std::vector<BaselineEstimate> estimateLinesByComponentGaps(const cv::Mat& binInk255, int expectedLineCount, const BaselineEstParams& p, int minimumComponentArea);
