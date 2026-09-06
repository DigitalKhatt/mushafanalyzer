#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "LayoutDb.h"
#include "PageProcessor.h"

struct PageValidation {
  int page = -1;
  std::string status;
  std::string sourceImage;
  std::string outputDirectory;
  std::string error;
  int expectedLineCount = 0;
  int detectedLineCount = 0;
  int expectedWordCount = 0;
  int detectedWordCount = 0;
  int invalidSpacingSpans = 0;
  int negativeSpacingGaps = 0;
  int bodyInkContacts = 0;
  int lowConfidenceBaselines = 0;
  int baselineBoundaryHits = 0;
  int crossLineContaminatedWords = 0;
  int totalInkPixels = 0;
  long long assignedInkPixels = 0;
  long long ignoredInkPixels = 0;
  long long unassignedInkPixels = 0;
  long long duplicateInkPixels = 0;
  int measuredGapCount = 0;
  double gapMeanPx = 0.0;
  double gapMedianPx = 0.0;
  double gapMinPx = 0.0;
  double gapMaxPx = 0.0;
  double anomalyScore = 0.0;
};

PageValidation summarizePageValidation(
    int pageNumber, const std::filesystem::path& imagePath,
    const std::filesystem::path& pageOutput, const PageLayout& expected,
    const PageProcessor& processor);

void writeBatchValidationReport(
    const std::filesystem::path& outputRoot,
    const std::vector<PageValidation>& pages);

void writePageValidationContactSheet(
    const std::filesystem::path& outputRoot,
    const std::vector<PageValidation>& pages, int maximumPages = 20);
