#include "MarkClassifier.h"

#include <glaze/glaze.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "MarkTaxonomy.h"

struct ModelMetadata {
  std::vector<std::string> classes;
};

namespace {
ModelMetadata loadMetadata(const std::filesystem::path& modelPath) {
  std::filesystem::path metadataPath = modelPath;
  metadataPath.replace_extension(".json");
  std::ifstream input(metadataPath);
  if (!input)
    throw std::runtime_error("Could not open mark model metadata " +
                             metadataPath.string());
  const std::string json((std::istreambuf_iterator<char>(input)),
                         std::istreambuf_iterator<char>());
  ModelMetadata metadata;
  if (const auto error = glz::read_json(metadata, json))
    throw std::runtime_error("Could not parse mark model metadata: " +
                             glz::format_error(error, json));
  return metadata;
}

double componentPitch(const PageModel& page, const CC& component) {
  if (component.line > 0)
    return page.lines[component.line].baseLine -
           page.lines[component.line - 1].baseLine;
  if (component.line + 1 < static_cast<int>(page.lines.size()))
    return page.lines[component.line + 1].baseLine -
           page.lines[component.line].baseLine;
  return 1.0;
}

cv::Mat normalizedMask(const PageModel& page, const CC& component) {
  const cv::Mat mask = page.labels(component.bb) == component.label;
  cv::Mat output(96, 96, CV_8UC1, cv::Scalar(0));
  const double scale =
      std::min(88.0 / component.bb.width, 88.0 / component.bb.height);
  cv::Mat resized;
  cv::resize(mask, resized, cv::Size(), scale, scale, cv::INTER_NEAREST);
  resized.copyTo(output(cv::Rect((96 - resized.cols) / 2,
                                 (96 - resized.rows) / 2, resized.cols,
                                 resized.rows)));
  return output;
}

cv::Mat normalizedContext(const PageModel& page, const CC& component) {
  const double pitch = componentPitch(page, component);
  const int width = std::max(1, static_cast<int>(1.2 * pitch));
  const int height = std::max(1, static_cast<int>(0.9 * pitch));
  cv::Rect rect(static_cast<int>(component.cent.x) - width / 2,
                static_cast<int>(component.cent.y) - height / 2, width,
                height);
  rect &= cv::Rect(0, 0, page.labels.cols, page.labels.rows);
  cv::Mat ink = page.labels(rect) != 0;
  cv::Mat output;
  cv::resize(ink, output, cv::Size(192, 128), 0.0, 0.0, cv::INTER_AREA);
  return output;
}

MarkClass findClass(std::string_view name) {
  for (std::size_t index = 0; index < kMarkClassCount; ++index)
    if (markClassName(static_cast<MarkClass>(index)) == name)
      return static_cast<MarkClass>(index);
  return MarkClass::Other;
}
}  // namespace

void classifyMarksWithOnnx(PageModel& page,
                           const std::filesystem::path& modelPath) {
  if (modelPath.empty()) return;
  cv::dnn::Net model = cv::dnn::readNetFromONNX(modelPath.string());
  const ModelMetadata metadata = loadMetadata(modelPath);

  for (CC& component : page.comps) {
    if (component.line < 0) continue;
    const cv::Mat candidateBlob =
        cv::dnn::blobFromImage(normalizedMask(page, component), 1.0 / 255.0);
    const cv::Mat contextBlob =
        cv::dnn::blobFromImage(normalizedContext(page, component),
                               1.0 / 255.0);
    const float width = component.bb.width;
    const float height = component.bb.height;
    const float area = component.area;
    cv::Mat geometry =
        (cv::Mat_<float>(1, 5) << std::log1p(width), std::log1p(height),
         std::log1p(area), std::log1p(width / std::max(1.0F, height)),
         std::log1p(area / std::max(1.0F, width * height)));
    model.setInput(candidateBlob, "candidate");
    model.setInput(contextBlob, "context");
    model.setInput(geometry, "geometry");
    cv::Mat logits = model.forward("logits").reshape(1, 1);
    if (logits.cols != static_cast<int>(metadata.classes.size()))
      throw std::runtime_error("Mark model class count differs from metadata");
    double maximum = -std::numeric_limits<double>::infinity();
    for (int column = 0; column < logits.cols; ++column)
      maximum = std::max(maximum, static_cast<double>(logits.at<float>(column)));
    double total = 0.0;
    std::vector<double> probabilities(logits.cols);
    for (int column = 0; column < logits.cols; ++column) {
      probabilities[column] =
          std::exp(static_cast<double>(logits.at<float>(column)) - maximum);
      total += probabilities[column];
    }
    for (int column = 0; column < logits.cols; ++column) {
      const MarkClass markClass = findClass(metadata.classes[column]);
      component.markClassProbabilities[static_cast<std::size_t>(markClass)] +=
          probabilities[column] / total;
    }
    component.smallLetterProbability = 0.0;
    for (std::size_t mark = static_cast<std::size_t>(MarkClass::SmallAlef);
         mark <= static_cast<std::size_t>(MarkClass::SmallSeen); ++mark)
      component.smallLetterProbability +=
          component.markClassProbabilities[mark];
  }
}
