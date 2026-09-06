#pragma once

#include <filesystem>
#include <string>

#include "SegmentationTypes.h"

// Repairs assignments that cannot be represented by the initial RTL
// connected-component DP: inter-line waqf marks, trailing small yeh marks
// crossing a word boundary, and compact isolated scan noise.
void correctComponentAssignments(
    PageModel& page, int pageNumber,
    const std::filesystem::path& segmentationExceptions = {},
    const std::filesystem::path& waqfDetections = {},
    const std::string& waqfDetectionSource = {});
