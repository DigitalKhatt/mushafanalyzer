#pragma once
#include <filesystem>

#include <vector>

#include "SegmentationTypes.h"

void writeQualityReport(const std::vector<LineBox>& lines,
                        const std::filesystem::path& outDir, int pageNumber,
                        int contactSheetWordCount = 30);
