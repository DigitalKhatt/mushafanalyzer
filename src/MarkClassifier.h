#pragma once

#include <filesystem>

#include "SegmentationTypes.h"

// Runs an optional ONNX classifier before word segmentation. The DP consumes
// the resulting class probability mass as soft textual-inventory evidence.
void classifyMarksWithOnnx(PageModel& page,
                           const std::filesystem::path& modelPath);
