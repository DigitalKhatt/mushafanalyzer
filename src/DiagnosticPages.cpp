#include "DiagnosticPages.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

#include "JsonIo.h"

namespace diagnostic_detail {

constexpr int kHeaderHeight = 180;
constexpr int kMaxDiagnosticWidth = 5000;

struct ReconstructionReport {
  int schemaVersion;
  double diagnosticScale;
  int totalInkPixels;
  long long assignedInkPixels, ignoredInkPixels, unassignedInkPixels;
  double unassignedInkFraction;
  int assignedComponentCount, duplicateComponentCount;
  long long duplicateInkPixels;
};

struct ComponentInsertionEvent {
  int row;
  int insertedComponents;
};

struct ComponentInsertionGap {
  int firstEmptyRow, lastEmptyRow, length;
};

struct ComponentInsertionReport {
  int schemaVersion, imageHeight, retainedComponentCount;
  std::vector<ComponentInsertionEvent> insertionEvents;
  std::vector<ComponentInsertionGap> emptyInsertionRuns;
};

int scaled(double value, double scale) {
  return static_cast<int>(std::lround(value * scale));
}

cv::Scalar componentColor(ComponentTypeEnum type) {
  switch (type) {
    case ComponentTypeEnum::Body:
      return {20, 20, 20};
    case ComponentTypeEnum::Dot:
      return {40, 40, 230};
    case ComponentTypeEnum::OtherMark:
      return {220, 90, 20};
    case ComponentTypeEnum::Uncertain:
      return {0, 150, 255};
  }
  return {0, 0, 0};
}

void drawLegend(cv::Mat& image) {
  cv::putText(image, "Segmentation reconstruction", cv::Point(24, 38),
              cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(20, 20, 20), 2,
              cv::LINE_AA);
  struct LegendItem {
    const char* label;
    cv::Scalar color;
  };
  const std::vector<LegendItem> items = {
      {"body", componentColor(ComponentTypeEnum::Body)},
      {"dot", componentColor(ComponentTypeEnum::Dot)},
      {"other mark", componentColor(ComponentTypeEnum::OtherMark)},
      {"uncertain", componentColor(ComponentTypeEnum::Uncertain)},
      {"word box", {0, 170, 0}},
      {"baseline", {180, 0, 180}},
      {"measured gap", {200, 180, 0}},
  };
  int x = 24;
  int y = 88;
  for (const LegendItem& item : items) {
    cv::rectangle(image, cv::Rect(x, y - 20, 32, 24), item.color, -1);
    cv::putText(image, item.label, cv::Point(x + 42, y),
                cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(20, 20, 20), 2,
                cv::LINE_AA);
    x += 185;
    if (x + 185 >= image.cols) {
      x = 24;
      y += 45;
    }
  }
}

void drawGeometry(cv::Mat& image, const PageModel& page, double scale) {
  const int thickness = std::max(1, scaled(7, scale));
  const int tick = std::max(5, scaled(55, scale));
  for (const LineBox& line : page.lines) {
    const int baseline = kHeaderHeight + scaled(line.baseLine, scale);
    cv::line(image, cv::Point(0, baseline),
             cv::Point(image.cols - 1, baseline), cv::Scalar(180, 0, 180),
             thickness);
    for (size_t index = 0; index < line.words.size(); ++index) {
      const WordBox& word = line.words[index];
      cv::Rect box(scaled(word.rect.x, scale),
                   kHeaderHeight + scaled(word.rect.y, scale),
                   std::max(1, scaled(word.rect.width, scale)),
                   std::max(1, scaled(word.rect.height, scale)));
      box &= cv::Rect(0, kHeaderHeight, image.cols,
                      image.rows - kHeaderHeight);
      if (box.area() > 0)
        cv::rectangle(image, box, cv::Scalar(0, 170, 0), thickness);

      cv::putText(image, word.wordId,
                  cv::Point(box.x, std::max(kHeaderHeight + 16, box.y - 5)),
                  cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 110, 0),
                  std::max(1, thickness), cv::LINE_AA);

      if (index + 1 >= line.words.size() || !word.spacingSpan.valid ||
          !line.words[index + 1].spacingSpan.valid)
        continue;
      const WordBox& next = line.words[index + 1];
      const int currentLeft =
          scaled(word.rect.x + word.spacingSpan.left, scale);
      const int nextRight =
          scaled(next.rect.x + next.spacingSpan.right, scale);
      const int gapY = baseline + std::max(8, scaled(130, scale));
      cv::line(image, cv::Point(nextRight, gapY),
               cv::Point(currentLeft, gapY), cv::Scalar(200, 180, 0),
               thickness);
      cv::line(image, cv::Point(nextRight, gapY - tick),
               cv::Point(nextRight, gapY + tick), cv::Scalar(200, 180, 0),
               thickness);
      cv::line(image, cv::Point(currentLeft, gapY - tick),
               cv::Point(currentLeft, gapY + tick),
               cv::Scalar(200, 180, 0), thickness);
    }
  }
}

void writeComponentInsertionDiagnostic(
    const PageModel& page, int imageHeight,
    const std::filesystem::path& outDir) {
  std::vector<int> insertedAtRow(imageHeight, 0);
  for (const CC& component : page.comps) {
    if (component.bb.y >= 0 && component.bb.y < imageHeight)
      ++insertedAtRow[component.bb.y];
  }

  std::vector<ComponentInsertionEvent> events;
  for (int row = 0; row < imageHeight; ++row)
    if (insertedAtRow[row] != 0)
      events.push_back({row, insertedAtRow[row]});

  std::vector<ComponentInsertionGap> gaps;
  for (size_t index = 1; index < events.size(); ++index) {
    const int first = events[index - 1].row + 1;
    const int last = events[index].row - 1;
    if (first <= last) gaps.push_back({first, last, last - first + 1});
  }
  std::sort(gaps.begin(), gaps.end(), [](const auto& left, const auto& right) {
    if (left.length != right.length) return left.length > right.length;
    return left.firstEmptyRow < right.firstEmptyRow;
  });

  writeJsonFile(outDir / "component-insertion-gaps.json",
                ComponentInsertionReport{1, imageHeight,
                                         static_cast<int>(page.comps.size()),
                                         events, gaps});

  constexpr int plotWidth = 1400;
  constexpr int plotHeight = 2400;
  constexpr int leftMargin = 190;
  cv::Mat plot(plotHeight, plotWidth, CV_8UC3, cv::Scalar(255, 255, 255));
  const double scaleY = plotHeight / double(std::max(1, imageHeight));
  const int maximumInsertion =
      *std::max_element(insertedAtRow.begin(), insertedAtRow.end());

  // Show the 30 longest zero-insertion runs. These are candidates only: long
  // runs can also occur inside a line because components start on relatively
  // few rows.
  for (size_t index = 0; index < std::min<size_t>(30, gaps.size()); ++index) {
    const ComponentInsertionGap& gap = gaps[index];
    const int top = static_cast<int>(std::lround(gap.firstEmptyRow * scaleY));
    const int bottom = static_cast<int>(std::lround((gap.lastEmptyRow + 1) * scaleY));
    cv::rectangle(plot, cv::Rect(leftMargin, top, plotWidth - leftMargin, std::max(1, bottom - top)),
                  cv::Scalar(225, 250, 225), -1);
    cv::putText(plot, std::to_string(gap.length),
                cv::Point(8, std::clamp((top + bottom) / 2, 15, plotHeight - 4)),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 100, 0), 1,
                cv::LINE_AA);
  }
  for (const ComponentInsertionEvent& event : events) {
    const int y = std::clamp(static_cast<int>(std::lround(event.row * scaleY)),
                             0, plotHeight - 1);
    const int length = maximumInsertion > 0
                           ? event.insertedComponents *
                                 (plotWidth - leftMargin - 20) /
                                 maximumInsertion
                           : 0;
    cv::line(plot, cv::Point(leftMargin, y),
             cv::Point(leftMargin + length, y), cv::Scalar(40, 40, 220), 2);
  }
  for (const LineBox& line : page.lines) {
    const int y = std::clamp(
        static_cast<int>(std::lround(line.baseLine * scaleY)), 0,
        plotHeight - 1);
    cv::line(plot, cv::Point(leftMargin, y), cv::Point(plotWidth - 1, y),
             cv::Scalar(180, 0, 180), 1);
  }
  cv::putText(plot,
              "red=new components at row; green=30 longest empty runs; "
              "magenta=baseline",
              cv::Point(15, 28), cv::FONT_HERSHEY_SIMPLEX, 0.6,
              cv::Scalar(20, 20, 20), 1, cv::LINE_AA);
  cv::imwrite((outDir / "debug_component_insertions.png").string(), plot);
}

}  // namespace diagnostic_detail
using namespace diagnostic_detail;

void writeDiagnosticPages(const cv::Mat& binaryInk, const cv::Mat& source,
                          const PageModel& page,
                          const std::filesystem::path& outDir,
                          bool writeOtherDebugImages,
                          bool writeReconstructed) {
  if (binaryInk.empty() || page.labels.empty()) return;
  if (writeOtherDebugImages)
    writeComponentInsertionDiagnostic(page, binaryInk.rows, outDir);
  const double scale =
      std::min(1.0, kMaxDiagnosticWidth / double(binaryInk.cols));
  const cv::Size diagnosticSize(std::max(1, scaled(binaryInk.cols, scale)),
                                std::max(1, scaled(binaryInk.rows, scale)));

  cv::Mat reconstructed(diagnosticSize.height + kHeaderHeight,
                        diagnosticSize.width, CV_8UC3,
                        cv::Scalar(255, 255, 255));
  cv::Mat reconstructedMask(diagnosticSize, CV_8UC1, cv::Scalar(0));

  std::vector<int> labelOwners(page.stats.rows, 0);
  for (const LineBox& line : page.lines) {
    for (const WordBox& word : line.words) {
      if (word.rect.empty() || word.compIds.empty()) continue;
      cv::Mat localLabels, localStats, localCentroids;
      cv::Mat wordMask(word.rect.height, word.rect.width, CV_8UC1,
                       cv::Scalar(0));
      std::vector<uint8_t> pageLabels(page.stats.rows, 0);
      for (int componentIndex : word.compIds) {
        const int label = page.comps[componentIndex].label;
        if (label > 0 && label < page.stats.rows) {
          pageLabels[label] = 1;
          ++labelOwners[label];
        }
      }
      for (int y = 0; y < wordMask.rows; ++y) {
        const int* labels = page.labels.ptr<int>(word.rect.y + y);
        uchar* output = wordMask.ptr<uchar>(y);
        for (int x = 0; x < wordMask.cols; ++x) {
          if (pageLabels[labels[word.rect.x + x]]) output[x] = 255;
        }
      }
      const int componentCount = cv::connectedComponentsWithStats(
          wordMask, localLabels, localStats, localCentroids, 8, CV_32S);
      std::vector<ComponentTypeEnum> types(
          componentCount, ComponentTypeEnum::Uncertain);
      for (const ComponentType& component :
           word.bodySegmentation.componentTypes) {
        if (component.componentLabel > 0 &&
            component.componentLabel < componentCount)
          types[component.componentLabel] = component.componentType;
      }

      cv::Mat localColor(wordMask.size(), CV_8UC3,
                         cv::Scalar(255, 255, 255));
      for (int y = 0; y < localLabels.rows; ++y) {
        const int* labels = localLabels.ptr<int>(y);
        cv::Vec3b* colors = localColor.ptr<cv::Vec3b>(y);
        for (int x = 0; x < localLabels.cols; ++x) {
          const int label = labels[x];
          if (label == 0) continue;
          const cv::Scalar color = componentColor(types[label]);
          colors[x] = cv::Vec3b(color[0], color[1], color[2]);
        }
      }

      cv::Mat resizedColor, resizedMask;
      cv::resize(localColor, resizedColor, cv::Size(), scale, scale,
                 cv::INTER_NEAREST);
      cv::resize(wordMask, resizedMask, cv::Size(), scale, scale,
                 cv::INTER_NEAREST);
      cv::Rect destination(
          scaled(word.rect.x, scale),
          kHeaderHeight + scaled(word.rect.y, scale), resizedColor.cols,
          resizedColor.rows);
      destination &=
          cv::Rect(0, kHeaderHeight, reconstructed.cols,
                   reconstructed.rows - kHeaderHeight);
      if (destination.area() <= 0) continue;
      resizedColor(cv::Rect(0, 0, destination.width, destination.height))
          .copyTo(reconstructed(destination),
                  resizedMask(cv::Rect(0, 0, destination.width,
                                       destination.height)));
      resizedMask(cv::Rect(0, 0, destination.width, destination.height))
          .copyTo(reconstructedMask(
                      cv::Rect(destination.x, destination.y - kHeaderHeight,
                               destination.width, destination.height)),
                  resizedMask(cv::Rect(0, 0, destination.width,
                                       destination.height)));
    }
  }

  drawLegend(reconstructed);
  drawGeometry(reconstructed, page, scale);
  if (writeReconstructed)
    cv::imwrite((outDir / "debug_reconstructed.png").string(),
                reconstructed);

  if (writeOtherDebugImages) {
    cv::Mat sourceGray;
    if (source.channels() == 1)
      sourceGray = source;
    else
      cv::cvtColor(source, sourceGray, cv::COLOR_BGR2GRAY);
    cv::Mat resizedSource;
    cv::resize(sourceGray, resizedSource, diagnosticSize, 0, 0,
               cv::INTER_AREA);
    cv::Mat geometry(diagnosticSize.height + kHeaderHeight,
                     diagnosticSize.width, CV_8UC3,
                     cv::Scalar(255, 255, 255));
    cv::cvtColor(resizedSource,
                 geometry(cv::Rect(0, kHeaderHeight, diagnosticSize.width,
                                   diagnosticSize.height)),
                 cv::COLOR_GRAY2BGR);
    drawLegend(geometry);
    drawGeometry(geometry, page, scale);
    cv::imwrite((outDir / "debug_geometry.png").string(), geometry);

    cv::Mat resizedBinary;
    cv::resize(binaryInk, resizedBinary, diagnosticSize, 0, 0,
               cv::INTER_NEAREST);
    cv::Mat difference(diagnosticSize.height + kHeaderHeight,
                       diagnosticSize.width, CV_8UC3,
                       cv::Scalar(255, 255, 255));
    for (int y = 0; y < diagnosticSize.height; ++y) {
      const uchar* scan = resizedBinary.ptr<uchar>(y);
      const uchar* assigned = reconstructedMask.ptr<uchar>(y);
      cv::Vec3b* output = difference.ptr<cv::Vec3b>(y + kHeaderHeight);
      for (int x = 0; x < diagnosticSize.width; ++x) {
        if (scan[x] && assigned[x])
          output[x] = cv::Vec3b(150, 150, 150);
        else if (scan[x])
          output[x] = cv::Vec3b(40, 40, 230);
        else if (assigned[x])
          output[x] = cv::Vec3b(220, 90, 20);
      }
    }
    cv::putText(difference,
                "Difference: gray=assigned, red=unassigned scan ink, "
                "blue=reconstruction-only",
                cv::Point(24, 55), cv::FONT_HERSHEY_SIMPLEX, 0.75,
                cv::Scalar(20, 20, 20), 2, cv::LINE_AA);
    cv::imwrite((outDir / "debug_reconstruction_difference.png").string(),
                difference);
  }

  const int totalInkPixels = cv::countNonZero(binaryInk);
  long long assignedInkPixels = 0;
  long long ignoredInkPixels = 0;
  long long duplicateInkPixels = 0;
  int assignedComponentCount = 0;
  int duplicateComponentCount = 0;
  for (int label = 1; label < page.stats.rows; ++label) {
    const int area = page.stats.at<int>(label, cv::CC_STAT_AREA);
    if (labelOwners[label] > 0) {
      assignedInkPixels += area;
      ++assignedComponentCount;
    }
    if (labelOwners[label] > 1) {
      duplicateInkPixels += area;
      ++duplicateComponentCount;
    }
  }
  for (const CC& component : page.comps)
    if (component.ignored)
      ignoredInkPixels += component.area;
  const long long unassignedInkPixels =
      std::max<long long>(0, totalInkPixels - assignedInkPixels -
                                 ignoredInkPixels);
  writeJsonFile(
      outDir / "reconstruction-report.json",
      ReconstructionReport{
          1, scale, totalInkPixels, assignedInkPixels, ignoredInkPixels,
          unassignedInkPixels,
          totalInkPixels > 0
              ? double(unassignedInkPixels) / totalInkPixels
              : 0.0,
          assignedComponentCount, duplicateComponentCount,
          duplicateInkPixels});
}
