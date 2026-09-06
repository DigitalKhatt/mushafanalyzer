#pragma once

#include <opencv2/opencv.hpp>

#include "SegmentationTypes.h"

struct SpacingGeometryParams {
  int corridorAbove = 250;
  int corridorBelow = 100;
  int minInkPerColumn = 2;
  int closeGapColumns = 8;
};

HorizontalInkSpan measureBaselineCorridorSpan(
    const cv::Mat& wordMask,
    const MarkSegmentationResult& segmentation,
    int baselineY,
    const SpacingGeometryParams& params);

void computeLineSpacingGeometry(LineBox& line);
