#include "BatchValidation.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>

#include "JsonIo.h"
#include <filesystem>

namespace batch_validation_detail {
struct Totals {
  int requestedPages = 0;
  int processedPages = 0;
  int skippedPages = 0;
  int failedPages = 0;
  int detectedWords = 0;
  int invalidSpacingSpans = 0;
  int negativeSpacingGaps = 0;
  int bodyInkContacts = 0;
  int lowConfidenceBaselines = 0;
  int baselineBoundaryHits = 0;
  int crossLineContaminatedWords = 0;
  long long totalInkPixels = 0;
  long long ignoredInkPixels = 0;
  long long unassignedInkPixels = 0;
  long long duplicateInkPixels = 0;
};

struct BatchReport {
  int schemaVersion = 3;
  Totals totals;
  std::vector<PageValidation> pages;
};

int expectedWords(const PageLayout& page) {
  return std::accumulate(
      page.lines.begin(), page.lines.end(), 0,
      [](int total, const LineLayout& line) {
        return total + static_cast<int>(line.words.size());
      });
}

double median(std::vector<double> values) {
  if (values.empty()) return 0.0;
  const size_t middle = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + middle, values.end());
  double result = values[middle];
  if (values.size() % 2 == 0) {
    std::nth_element(values.begin(), values.begin() + middle - 1,
                     values.end());
    result = (result + values[middle - 1]) * 0.5;
  }
  return result;
}

bool isVerseMarker(const std::string& text) {
  return text.starts_with("\xDB\x9D");  // U+06DD in UTF-8
}

cv::Mat bodyMask(const PageModel& model, const WordBox& word) {
  cv::Mat wordMask(word.rect.size(), CV_8UC1, cv::Scalar(0));
  std::vector<uint8_t> selected(model.stats.rows, 0);
  for (int componentIndex : word.compIds) {
    const int label = model.comps.at(componentIndex).label;
    if (label > 0 && label < model.stats.rows) selected[label] = 1;
  }
  for (int y = 0; y < word.rect.height; ++y) {
    const int* labels = model.labels.ptr<int>(word.rect.y + y);
    uint8_t* output = wordMask.ptr<uint8_t>(y);
    for (int x = 0; x < word.rect.width; ++x)
      if (selected[labels[word.rect.x + x]]) output[x] = 255;
  }

  cv::Mat labels, stats, centroids;
  cv::connectedComponentsWithStats(wordMask, labels, stats, centroids, 8,
                                   CV_32S);
  cv::Mat body = cv::Mat::zeros(wordMask.size(), CV_8UC1);
  for (const ComponentType& component :
       word.bodySegmentation.componentTypes)
    if (component.componentType == ComponentTypeEnum::Body)
      body.setTo(255, labels == component.componentLabel);
  return body;
}

bool bodyInkTouches(const PageModel& model, const WordBox& first,
                    const WordBox& second) {
  constexpr int contactRadius = 2;
  cv::Mat firstBody = bodyMask(model, first);
  cv::Mat secondBody = bodyMask(model, second);
  cv::copyMakeBorder(firstBody, firstBody, contactRadius, contactRadius,
                     contactRadius, contactRadius, cv::BORDER_CONSTANT, 0);
  cv::copyMakeBorder(secondBody, secondBody, contactRadius, contactRadius,
                     contactRadius, contactRadius, cv::BORDER_CONSTANT, 0);
  const cv::Mat kernel = cv::getStructuringElement(
      cv::MORPH_ELLIPSE,
      {contactRadius * 2 + 1, contactRadius * 2 + 1});
  cv::dilate(firstBody, firstBody, kernel);

  const cv::Rect firstPageRect(first.rect.x - contactRadius,
                               first.rect.y - contactRadius, firstBody.cols,
                               firstBody.rows);
  const cv::Rect secondPageRect(second.rect.x - contactRadius,
                                second.rect.y - contactRadius, secondBody.cols,
                                secondBody.rows);
  const cv::Rect overlap = firstPageRect & secondPageRect;
  if (overlap.empty()) return false;
  const cv::Rect firstRoi(overlap.x - firstPageRect.x,
                          overlap.y - firstPageRect.y, overlap.width,
                          overlap.height);
  const cv::Rect secondRoi(overlap.x - secondPageRect.x,
                           overlap.y - secondPageRect.y, overlap.width,
                           overlap.height);
  cv::Mat contact;
  cv::bitwise_and(firstBody(firstRoi), secondBody(secondRoi), contact);
  return cv::countNonZero(contact) > 0;
}
}  // namespace batch_validation_detail

using namespace batch_validation_detail;

PageValidation summarizePageValidation(
    int pageNumber, const std::filesystem::path& imagePath,
    const std::filesystem::path& pageOutput, const PageLayout& expected,
    const PageProcessor& processor) {
  PageValidation result;
  result.page = pageNumber;
  result.status = "ok";
  result.sourceImage = imagePath.string();
  result.outputDirectory = pageOutput.string();
  result.expectedLineCount = static_cast<int>(expected.lines.size());
  result.detectedLineCount = static_cast<int>(processor.lines().size());
  result.expectedWordCount = expectedWords(expected);
  const PageModel& model = processor.pageModel();

  std::vector<double> gaps;
  for (const LineBox& line : processor.lines()) {
    if (line.baseLineConfidence < 0.20) ++result.lowConfidenceBaselines;
    if (line.baseLineSearchBoundaryHit) ++result.baselineBoundaryHits;
    result.detectedWordCount += static_cast<int>(line.words.size());
    for (size_t index = 0; index < line.words.size(); ++index) {
      const WordBox& word = line.words[index];
      if (!word.spacingSpan.valid) ++result.invalidSpacingSpans;
      if (!isVerseMarker(word.expectedText) &&
          (word.foreignLineInkFraction > 0.05 ||
           word.heightToLinePitch > 1.35))
        ++result.crossLineContaminatedWords;
      if (index + 1 < line.words.size() && word.spacingSpan.valid &&
          line.words[index + 1].spacingSpan.valid) {
        gaps.push_back(word.spacingGapToNextPx);
        if (word.spacingGapToNextPx < 0) {
          ++result.negativeSpacingGaps;
          if (bodyInkTouches(model, word, line.words[index + 1]))
            ++result.bodyInkContacts;
        }
      }
    }
  }

  result.measuredGapCount = static_cast<int>(gaps.size());
  if (!gaps.empty()) {
    result.gapMeanPx =
        std::accumulate(gaps.begin(), gaps.end(), 0.0) / gaps.size();
    result.gapMedianPx = median(gaps);
    const auto [minimum, maximum] =
        std::minmax_element(gaps.begin(), gaps.end());
    result.gapMinPx = *minimum;
    result.gapMaxPx = *maximum;
  }

  result.totalInkPixels = cv::countNonZero(processor.binary());
  std::vector<int> owners(model.stats.rows, 0);
  for (const LineBox& line : model.lines)
    for (const WordBox& word : line.words)
      for (int componentIndex : word.compIds) {
        const int label = model.comps.at(componentIndex).label;
        if (label > 0 && label < model.stats.rows) ++owners[label];
      }
  std::vector<int> componentByLabel(model.stats.rows, -1);
  for (std::size_t index = 0; index < model.comps.size(); ++index) {
    const int label = model.comps[index].label;
    if (label > 0 && label < model.stats.rows)
      componentByLabel[label] = static_cast<int>(index);
  }
  for (int label = 1; label < model.stats.rows; ++label) {
    const int area = model.stats.at<int>(label, cv::CC_STAT_AREA);
    const int componentIndex = componentByLabel[label];
    if (componentIndex >= 0 && model.comps[componentIndex].ignored)
      result.ignoredInkPixels += area;
    else if (owners[label] > 0)
      result.assignedInkPixels += area;
    if (owners[label] > 1) result.duplicateInkPixels += area;
  }
  result.unassignedInkPixels =
      std::max<long long>(0, result.totalInkPixels -
                                 result.assignedInkPixels -
                                 result.ignoredInkPixels);

  constexpr double unassignedToleranceFraction = 0.0001;  // 0.01%
  const double unassignedFraction =
      double(result.unassignedInkPixels) /
      std::max(1, result.totalInkPixels);
  result.anomalyScore =
      20.0 * std::abs(result.detectedLineCount - result.expectedLineCount) +
      10.0 * std::abs(result.detectedWordCount - result.expectedWordCount) +
      5.0 * result.invalidSpacingSpans +
      3.0 * result.bodyInkContacts +
      4.0 * result.baselineBoundaryHits +
      2.0 * result.lowConfidenceBaselines +
      2.0 * result.crossLineContaminatedWords +
      100.0 *
          std::max(0.0,
                   unassignedFraction - unassignedToleranceFraction) +
      100.0 * result.duplicateInkPixels /
          std::max(1, result.totalInkPixels);
  if (result.anomalyScore > 0.0) result.status = "review";
  return result;
}

void writeBatchValidationReport(
    const std::filesystem::path& outputRoot,
    const std::vector<PageValidation>& pages) {
  BatchReport report;
  report.pages = pages;
  report.totals.requestedPages = static_cast<int>(pages.size());
  for (const PageValidation& page : pages) {
    if (page.status == "failed") {
      ++report.totals.failedPages;
      continue;
    }
    if (page.status == "skipped") {
      ++report.totals.skippedPages;
      continue;
    }
    ++report.totals.processedPages;
    report.totals.detectedWords += page.detectedWordCount;
    report.totals.invalidSpacingSpans += page.invalidSpacingSpans;
    report.totals.negativeSpacingGaps += page.negativeSpacingGaps;
    report.totals.bodyInkContacts += page.bodyInkContacts;
    report.totals.lowConfidenceBaselines += page.lowConfidenceBaselines;
    report.totals.baselineBoundaryHits += page.baselineBoundaryHits;
    report.totals.crossLineContaminatedWords +=
        page.crossLineContaminatedWords;
    report.totals.totalInkPixels += page.totalInkPixels;
    report.totals.ignoredInkPixels += page.ignoredInkPixels;
    report.totals.unassignedInkPixels += page.unassignedInkPixels;
    report.totals.duplicateInkPixels += page.duplicateInkPixels;
  }
  writeJsonFile(outputRoot / "validation-report.json", report);
}

void writePageValidationContactSheet(
    const std::filesystem::path& outputRoot,
    const std::vector<PageValidation>& pages, int maximumPages) {
  std::vector<const PageValidation*> ranked;
  for (const PageValidation& page : pages)
    if (page.status == "ok" || page.status == "review")
      ranked.push_back(&page);
  std::stable_sort(ranked.begin(), ranked.end(), [](const auto* left, const auto* right) {
    return left->anomalyScore > right->anomalyScore;
  });
  if (ranked.empty()) return;

  constexpr int columns = 3;
  constexpr int tileWidth = 900;
  constexpr int tileHeight = 1300;
  constexpr int headerHeight = 80;
  const int count =
      std::min(maximumPages, static_cast<int>(ranked.size()));
  const int rows = (count + columns - 1) / columns;
  cv::Mat sheet(rows * tileHeight, columns * tileWidth, CV_8UC3,
                cv::Scalar(245, 245, 245));
  for (int index = 0; index < count; ++index) {
    const PageValidation& page = *ranked[index];
    const int tileX = (index % columns) * tileWidth;
    const int tileY = (index / columns) * tileHeight;
    cv::rectangle(sheet, {tileX, tileY, tileWidth, tileHeight},
                  cv::Scalar(180, 180, 180), 1);
    const std::string label =
        "page " + std::to_string(page.page) + "  score=" +
        cv::format("%.2f", page.anomalyScore) + "  words=" +
        std::to_string(page.detectedWordCount) + "/" +
        std::to_string(page.expectedWordCount);
    cv::putText(sheet, label, {tileX + 15, tileY + 48},
                cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(20, 20, 20), 2,
                cv::LINE_AA);
    const auto imagePath = std::filesystem::path(page.outputDirectory) / "debug_geometry.png";
    if (!std::filesystem::exists(imagePath)) continue;
    cv::Mat image = cv::imread(imagePath.string(), cv::IMREAD_COLOR);
    if (image.empty()) continue;
    const double scale =
        std::min((tileWidth - 20.0) / image.cols,
                 (tileHeight - headerHeight - 20.0) / image.rows);
    cv::Mat resized;
    cv::resize(image, resized, {}, scale, scale, cv::INTER_AREA);
    const int x = tileX + (tileWidth - resized.cols) / 2;
    const int y = tileY + headerHeight +
                  (tileHeight - headerHeight - resized.rows) / 2;
    resized.copyTo(sheet({x, y, resized.cols, resized.rows}));
  }
  cv::imwrite((outputRoot / "validation-contact-sheet.png").string(), sheet);
}
