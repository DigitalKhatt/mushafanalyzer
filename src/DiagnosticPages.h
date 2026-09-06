#pragma once
#include <filesystem>

#include <opencv2/opencv.hpp>

#include "SegmentationTypes.h"

void writeDiagnosticPages(const cv::Mat& binaryInk, const cv::Mat& source,
                          const PageModel& page,
                          const std::filesystem::path& outDir,
                          bool writeOtherDebugImages,
                          bool writeReconstructed);
