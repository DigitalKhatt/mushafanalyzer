#include "ComponentRecognizer.h"

#include <glaze/glaze.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace component_recognizer_json {
inline constexpr int kCacheSchemaVersion = 8;

struct Recognition {
  std::vector<int> componentParts;
  int threeDotsAboveComponentCount = 0;
  double attachedAlefHamzaAboveProbability = 0.0;
  // A high-confidence grouped mark may also correct the line assignment
  // before the word DP. This includes waqf groups and cross-line kasratan
  // pairs.
  std::string lineAssignmentClass;
  double score = 0.0;
  int targetLine = -1;
  std::array<double, kMarkClassCount> templateMarkProbabilities{};
  std::array<std::uint8_t, kMarkClassCount> templateMarkClasses{};
  int sajdaComponentCount = 0;
  double bodyTemplateProbability = 0.0;
};

struct Cache {
  int schemaVersion = kCacheSchemaVersion;
  std::string fingerprint;
  std::vector<Recognition> recognitions;
};
}  // namespace component_recognizer_json

namespace {
using component_recognizer_json::Cache;
using component_recognizer_json::Recognition;

struct TemplateImage {
  std::string className;
  std::string role;
  std::filesystem::path path;
  cv::Mat image;
};

void hashBytes(std::uint64_t& hash, const void* data, std::size_t size) {
  const auto* bytes = static_cast<const unsigned char*>(data);
  for (std::size_t index = 0; index < size; ++index) {
    hash ^= bytes[index];
    hash *= 1099511628211ULL;
  }
}

template <typename T>
void hashValue(std::uint64_t& hash, const T& value) {
  hashBytes(hash, &value, sizeof(value));
}

void hashString(std::uint64_t& hash, std::string_view value) {
  hashBytes(hash, value.data(), value.size());
}

std::string hashText(std::uint64_t hash) {
  std::ostringstream stream;
  stream << std::hex << std::setw(16) << std::setfill('0') << hash;
  return stream.str();
}

std::string safeName(std::string value) {
  for (char& character : value)
    if (!std::isalnum(static_cast<unsigned char>(character)) &&
        character != '-' && character != '_')
      character = '_';
  return value.empty() ? "default" : value;
}

// Whether the component encloses background -- the counter of a heh, ta
// marbuta, waw, meem, fa. Marks do not have one, and it is the only thing
// that separates them from a heh once the shape is scaled to a fixed box: a
// filled dot differs from a ring by exactly the hole, which is a quarter of
// the area, so it still scores 0.73 to 0.76 against a heh template. Topology
// answers what the overlap cannot.
bool componentEnclosesBackground(const PageModel& page, const CC& component) {
  cv::Mat mask = page.labels(component.bb) == component.label;
  cv::Mat padded;
  cv::copyMakeBorder(mask, padded, 1, 1, 1, 1, cv::BORDER_CONSTANT,
                     cv::Scalar(0));
  cv::Mat background;
  cv::bitwise_not(padded, background);
  cv::Mat regions;
  const int count = cv::connectedComponents(background, regions, 8, CV_32S);
  if (count <= 2) return false;  // ink plus the outside only
  const int outside = regions.at<int>(0, 0);
  std::vector<int> areas(static_cast<std::size_t>(count), 0);
  for (int y = 0; y < regions.rows; ++y)
    for (int x = 0; x < regions.cols; ++x) {
      const int region = regions.at<int>(y, x);
      if (region > 0 && region != outside)
        ++areas[static_cast<std::size_t>(region)];
    }
  // The counter has to be a real one: a ragged scan edge leaves pinholes a
  // couple of pixels across, and those are not what a heh is made of.
  const int minimumHoleArea =
      std::max(4, static_cast<int>(0.02 * component.area));
  return *std::max_element(areas.begin(), areas.end()) >= minimumHoleArea;
}

cv::Mat normalizedMask(const PageModel& page, const CC& component) {
  cv::Mat mask = page.labels(component.bb) == component.label;
  cv::Mat normalized(96, 96, CV_8UC1, cv::Scalar(0));
  const double scale =
      std::min(88.0 / component.bb.width, 88.0 / component.bb.height);
  cv::Mat resized;
  const cv::Size resizedSize(
      std::max(1, static_cast<int>(std::lround(mask.cols * scale))),
      std::max(1, static_cast<int>(std::lround(mask.rows * scale))));
  cv::resize(mask, resized, resizedSize, 0.0, 0.0, cv::INTER_NEAREST);
  const int x = (normalized.cols - resized.cols) / 2;
  const int y = (normalized.rows - resized.rows) / 2;
  resized.copyTo(normalized(cv::Rect(x, y, resized.cols, resized.rows)));
  return normalized;
}

double binaryIou(const cv::Mat& left, const cv::Mat& right) {
  if (left.empty() || right.empty()) return 0.0;
  cv::Mat resizedRight;
  if (right.size() == left.size())
    resizedRight = right;
  else
    cv::resize(right, resizedRight, left.size(), 0.0, 0.0,
               cv::INTER_NEAREST);
  cv::Mat intersection;
  cv::Mat unionMask;
  cv::bitwise_and(left, resizedRight, intersection);
  cv::bitwise_or(left, resizedRight, unionMask);
  const int unionPixels = cv::countNonZero(unionMask);
  return unionPixels > 0
             ? cv::countNonZero(intersection) / double(unionPixels)
             : 0.0;
}

std::vector<TemplateImage> loadTemplates(
    const std::filesystem::path& root, bool directThreeDots) {
  std::vector<TemplateImage> result;
  if (root.empty() || !std::filesystem::is_directory(root)) return result;
  for (const auto& entry :
       std::filesystem::recursive_directory_iterator(root)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".png")
      continue;
    TemplateImage item;
    item.path = entry.path();
    if (directThreeDots) {
      item.className = "dot_above_three";
      item.role = "whole";
    } else {
      const auto relative = std::filesystem::relative(entry.path(), root);
      auto part = relative.begin();
      if (part == relative.end()) continue;
      item.className = part->string();
      if (++part == relative.end()) continue;
      if (part->extension() == ".png")
        item.role = item.className == "waqf_qaf" ||
                            item.className == "waqf_jeem"
                        ? "body"
                        : "whole";
      else
        item.role = part->string();
    }
    item.image = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
    if (!item.image.empty()) result.push_back(std::move(item));
  }
  std::sort(result.begin(), result.end(), [](const auto& left, const auto& right) {
    return left.path < right.path;
  });
  return result;
}

std::vector<TemplateImage> loadMarkTemplates(
    const ComponentRecognizerParams& params) {
  std::vector<TemplateImage> result = loadTemplates(params.markTemplateRoot,
                                                    false);
  // Keep old command lines usable while making markTemplateRoot the single
  // source used by the recognizer. This fallback can be removed once all
  // external configurations have migrated.
  if (!params.waqfTemplateRoot.empty() &&
      params.waqfTemplateRoot != params.markTemplateRoot) {
    std::vector<TemplateImage> legacy =
        loadTemplates(params.waqfTemplateRoot, false);
    result.insert(result.end(), legacy.begin(), legacy.end());
    std::sort(result.begin(), result.end(),
              [](const auto& left, const auto& right) {
                return left.path < right.path;
              });
  }
  return result;
}

std::vector<TemplateImage> loadDirectTemplates(
    const std::filesystem::path& root, std::string_view className) {
  std::vector<TemplateImage> result;
  if (root.empty() || !std::filesystem::is_directory(root)) return result;
  for (const auto& entry : std::filesystem::directory_iterator(root)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".png")
      continue;
    TemplateImage item;
    item.className = className;
    item.role = "whole";
    item.path = entry.path();
    item.image = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
    if (!item.image.empty()) result.push_back(std::move(item));
  }
  std::sort(result.begin(), result.end(),
            [](const auto& left, const auto& right) {
              return left.path < right.path;
            });
  return result;
}

std::vector<TemplateImage> loadAllMarkTemplates(
    const ComponentRecognizerParams& params) {
  std::vector<TemplateImage> result = loadMarkTemplates(params);
  // The canonical bank contains these classes under their semantic directory
  // names. Keep the old specialized roots as compatibility fallbacks only.
  if (!params.threeDotsAboveTemplateRoot.empty() &&
      params.threeDotsAboveTemplateRoot != params.markTemplateRoot) {
    std::vector<TemplateImage> legacy =
        loadTemplates(params.threeDotsAboveTemplateRoot, true);
    result.insert(result.end(), legacy.begin(), legacy.end());
  }
  if (!params.attachedAlefHamzaAboveTemplateRoot.empty() &&
      params.attachedAlefHamzaAboveTemplateRoot != params.markTemplateRoot) {
    std::vector<TemplateImage> legacy = loadDirectTemplates(
        params.attachedAlefHamzaAboveTemplateRoot,
        "alef_hamza_above_attached");
    result.insert(result.end(), legacy.begin(), legacy.end());
  }
  std::sort(result.begin(), result.end(),
            [](const auto& left, const auto& right) {
              return left.path < right.path;
            });
  return result;
}

double bestScore(const cv::Mat& mask,
                 const std::vector<TemplateImage>& templates,
                 std::string_view className, std::string_view role) {
  double result = 0.0;
  for (const TemplateImage& item : templates)
    if (item.className == className && item.role == role)
      result = std::max(result, binaryIou(mask, item.image));
  return result;
}

bool hasTemplate(const std::vector<TemplateImage>& templates,
                 std::string_view className, std::string_view role) {
  return std::any_of(templates.begin(), templates.end(),
                     [&](const TemplateImage& item) {
                       return item.className == className &&
                              item.role == role;
                     });
}

std::optional<MarkClass> markTemplateClass(std::string_view name) {
  for (std::size_t index = 0; index < kMarkClassCount; ++index) {
    const MarkClass markClass = static_cast<MarkClass>(index);
    if (markClass != MarkClass::Other &&
        markClassName(markClass) == name)
      return markClass;
  }
  return std::nullopt;
}

bool markClassIsBelow(MarkClass markClass) {
  switch (markClass) {
    case MarkClass::DotBelowOne:
    case MarkClass::DotBelowTwo:
    case MarkClass::Kasra:
    case MarkClass::Kasratan:
    case MarkClass::HamzaBelow:
    // U+06ED is the small *low* meem: it is written below by definition, and
    // listing it as an above class meant its own bank could never be tried on
    // it, however good the match.
    case MarkClass::SmallMeem:
      return true;
    default:
      return false;
  }
}

std::vector<MarkClass> availableMarkTemplateClasses(
    const std::vector<TemplateImage>& templates) {
  std::vector<MarkClass> result;
  for (const TemplateImage& templateImage : templates) {
    if ((templateImage.role != "whole" &&
         !(templateImage.className == "kasratan" &&
           templateImage.role == "open")) ||
        templateImage.className.rfind("waqf_", 0) == 0)
      continue;
    const std::optional<MarkClass> markClass =
        markTemplateClass(templateImage.className);
    if (!markClass ||
        std::find(result.begin(), result.end(), *markClass) != result.end())
      continue;
    result.push_back(*markClass);
  }
  return result;
}

std::optional<double> sajdaPairScore(const PageModel& page,
                                     const CC& first, const CC& second,
                                     int targetLine) {
  if (targetLine < 0 || targetLine >= static_cast<int>(page.lines.size()) ||
      first.line < 0 || second.line < 0 ||
      std::abs(first.line - second.line) > 1 ||
      std::abs(first.line - targetLine) > 1 ||
      std::abs(second.line - targetLine) > 1 || first.ignored ||
      second.ignored)
    return std::nullopt;

  const CC* outer = &first;
  const CC* inner = &second;
  if (inner->bb.width > outer->bb.width ||
      inner->bb.height > outer->bb.height) {
    std::swap(outer, inner);
  }
  if (outer->bb.width <= inner->bb.width ||
      outer->bb.height <= inner->bb.height)
    return std::nullopt;

  // In the Old Madina scan U+06E9 is a distinctive nested pair: a thin
  // outlined arch surrounding a dense stepped component. The recognizer uses
  // the relation between the two component boxes and their ink density, not
  // their page-specific component labels.
  const double widthRatio = inner->bb.width / double(outer->bb.width);
  const double heightRatio = inner->bb.height / double(outer->bb.height);
  const double xOffset =
      (inner->bb.x - outer->bb.x) / double(outer->bb.width);
  const double yOffset =
      (inner->bb.y - outer->bb.y) / double(outer->bb.height);
  const double outerFill =
      outer->area / double(std::max(1, outer->bb.width * outer->bb.height));
  const double innerFill =
      inner->area / double(std::max(1, inner->bb.width * inner->bb.height));

  if (widthRatio < 0.52 || widthRatio > 0.78 || heightRatio < 0.56 ||
      heightRatio > 0.84 || xOffset < 0.06 || xOffset > 0.30 ||
      yOffset < 0.08 || yOffset > 0.34 || outerFill < 0.08 ||
      outerFill > 0.30 || innerFill < 0.30 || innerFill > 0.72 ||
      inner->bb.x < outer->bb.x - 0.05 * outer->bb.width ||
      inner->bb.x + inner->bb.width >
          outer->bb.x + outer->bb.width + 0.05 * outer->bb.width ||
      inner->bb.y < outer->bb.y - 0.05 * outer->bb.height ||
      inner->bb.y + inner->bb.height >
          outer->bb.y + outer->bb.height + 0.05 * outer->bb.height)
    return std::nullopt;

  const double baseline = page.lines[targetLine].baseLine;
  const double pitch =
      targetLine > 0
          ? page.lines[targetLine].baseLine -
                page.lines[targetLine - 1].baseLine
          : targetLine + 1 < static_cast<int>(page.lines.size())
                ? page.lines[targetLine + 1].baseLine - baseline
                : 1.0;
  if (pitch <= 0.0 || outer->bb.y > baseline - 0.35 * outer->bb.height ||
      outer->bb.y + outer->bb.height < baseline - 0.25 * pitch ||
      inner->bb.y > baseline - 0.15 * inner->bb.height ||
      inner->bb.y + inner->bb.height < baseline - 0.25 * pitch)
    return std::nullopt;

  const auto closeness = [](double value, double target, double tolerance) {
    return std::clamp(1.0 - std::abs(value - target) / tolerance, 0.0, 1.0);
  };
  const double score =
      0.24 * closeness(widthRatio, 0.64, 0.12) +
      0.20 * closeness(heightRatio, 0.70, 0.14) +
      0.18 * closeness(xOffset, 0.18, 0.12) +
      0.16 * closeness(yOffset, 0.20, 0.14) +
      0.12 * closeness(outerFill, 0.17, 0.10) +
      0.10 * closeness(innerFill, 0.48, 0.20);
  return score >= 0.55 ? std::optional<double>(score) : std::nullopt;
}

std::optional<MarkClass> waqfMarkClass(std::string_view name) {
  if (name == "waqf_lam") return MarkClass::WaqfLam;
  if (name == "waqf_qaf") return MarkClass::WaqfQaf;
  if (name == "waqf_jeem") return MarkClass::WaqfJeem;
  if (name == "waqf_sad") return MarkClass::WaqfSad;
  if (name == "waqf_meem") return MarkClass::WaqfMeem;
  if (name == "waqf_three_dots") return MarkClass::WaqfThreeDots;
  return std::nullopt;
}

// The hamza broadens the top of an attached Alef, so the fused pair is a
// wider component than a bare Alef stem. Over the 7,225 words whose expected
// component count this reduction lowered, the ones that turned out to hold
// every component separately -- so the reduction was wrong -- run at an
// aspect of 4.8 at the tenth percentile and 5.4 at the median, while the ones
// that genuinely had a fused pair run 3.7 at the median. Rejecting at 4.2
// clears 4,412 of those 4,427 false positives and costs 486 of the 2,384
// genuine detections. It is worth that trade: the false positives are 76% of
// every over-count word in the corpus. A plain Alef stroke thins somewhere
// along its length at a lower threshold, breaks into two seeds, and passes
// the separation test that is meant to find a hamza sitting above an Alef.
constexpr double kMaximumAttachedAlefHamzaAspect = 4.2;

double attachedAlefHamzaAboveProbability(
    const PageModel& page, const CC& component, const cv::Mat& grayscale,
    double otsuThreshold) {
  if (grayscale.empty() || otsuThreshold <= 0.0 || component.line < 0 ||
      component.line >= static_cast<int>(page.lines.size()) ||
      component.bodyProbability < 0.50 || component.bb.width < 100 ||
      component.bb.width > 700 || component.bb.height < 500 ||
      component.bb.height > 1700 ||
      component.bb.height < 1.15 * component.bb.width ||
      component.bb.height >=
          kMaximumAttachedAlefHamzaAspect * component.bb.width)
    return 0.0;

  const int baseline =
      static_cast<int>(std::lround(page.lines[component.line].baseLine));
  const int bottom = component.bb.y + component.bb.height;
  if (component.bb.y > baseline - 400 || bottom < baseline - 300 ||
      bottom > baseline + 500)
    return 0.0;

  const cv::Mat originalMask =
      page.labels(component.bb) == component.label;
  const int corridorTop =
      std::max(0, baseline - 100 - component.bb.y);
  const int corridorBottom =
      std::min(component.bb.height - 1, baseline + 100 - component.bb.y);
  if (corridorBottom < corridorTop) return 0.0;
  int corridorLeft = component.bb.width;
  int corridorRight = -1;
  for (int x = 0; x < component.bb.width; ++x)
    if (cv::countNonZero(originalMask(
            cv::Rect(x, corridorTop, 1, corridorBottom - corridorTop + 1))) >=
        2) {
      corridorLeft = std::min(corridorLeft, x);
      corridorRight = std::max(corridorRight, x);
    }
  const int corridorWidth = corridorRight - corridorLeft + 1;
  if (corridorWidth <= 0 || corridorWidth > 180 ||
      corridorWidth > 0.65 * component.bb.width)
    return 0.0;

  const int minimumSeedArea = std::max(1500, component.area / 14);
  for (int threshold = static_cast<int>(std::floor(otsuThreshold)) - 1;
       threshold >= std::max(20, static_cast<int>(otsuThreshold) - 70);
       threshold -= 2) {
    cv::Mat strictInk;
    cv::threshold(grayscale(component.bb), strictInk, threshold, 255,
                  cv::THRESH_BINARY_INV);
    cv::bitwise_and(strictInk, originalMask, strictInk);
    cv::Mat labels, stats, centroids;
    const int count = cv::connectedComponentsWithStats(
        strictInk, labels, stats, centroids, 8, CV_32S);
    std::vector<int> seeds;
    for (int seed = 1; seed < count; ++seed)
      if (stats.at<int>(seed, cv::CC_STAT_AREA) >= minimumSeedArea)
        seeds.push_back(seed);
    if (seeds.size() != 2) continue;

    const int first = seeds[0];
    const int second = seeds[1];
    const int upper = centroids.at<double>(first, 1) <
                              centroids.at<double>(second, 1)
                          ? first
                          : second;
    const int lower = upper == first ? second : first;
    const double upperX = centroids.at<double>(upper, 0);
    const double upperY = centroids.at<double>(upper, 1);
    const double lowerX = centroids.at<double>(lower, 0);
    const double lowerY = centroids.at<double>(lower, 1);
    const int upperArea = stats.at<int>(upper, cv::CC_STAT_AREA);
    const int lowerArea = stats.at<int>(lower, cv::CC_STAT_AREA);
    const int lowerWidth = stats.at<int>(lower, cv::CC_STAT_WIDTH);
    const int lowerHeight = stats.at<int>(lower, cv::CC_STAT_HEIGHT);
    const int lowerBottom = component.bb.y +
                            stats.at<int>(lower, cv::CC_STAT_TOP) +
                            lowerHeight;
    const double areaRatio = upperArea / double(std::max(1, lowerArea));
    const double verticalSeparation = lowerY - upperY;
    if (areaRatio < 0.08 || areaRatio > 0.80 ||
        verticalSeparation < 0.16 * component.bb.height ||
        std::abs(upperX - lowerX) > 0.85 * component.bb.width ||
        lowerHeight < 1.30 * lowerWidth || lowerBottom < baseline - 300)
      continue;

    const double thresholdRetention =
        1.0 - (otsuThreshold - threshold) / 70.0;
    const double separationScore = std::min(
        1.0, verticalSeparation / (0.35 * component.bb.height));
    return std::clamp(0.72 + 0.16 * thresholdRetention +
                          0.12 * separationScore,
                      0.0, 1.0);
  }
  return 0.0;
}

std::vector<std::string> availableWaqfClasses(
    const std::vector<TemplateImage>& templates) {
  std::vector<std::string> result;
  for (const TemplateImage& item : templates)
    if (waqfMarkClass(item.className) &&
        std::find(result.begin(), result.end(), item.className) == result.end())
      result.push_back(item.className);
  return result;
}

std::vector<int> expectedTargetLines(
    const CC& component, MarkClass markClass,
    const std::vector<MarkInventory>& expectedMarksByLine,
    const PageModel& page) {
  std::vector<int> result;
  for (int line = std::max(0, component.line - 1);
       line <= std::min(static_cast<int>(page.lines.size()) - 1,
                        component.line + 1);
       ++line) {
    if (expectedMarksByLine[line][static_cast<std::size_t>(markClass)] == 0)
      continue;
    if (component.bb.y + component.bb.height > page.lines[line].baseLine)
      continue;
    result.push_back(line);
  }
  return result;
}

int chooseCompanion(const PageModel& page, int bodyIndex,
                    std::string_view className) {
  const CC& body = page.comps[bodyIndex];
  const double centerX = body.cent.x;
  const double centerY = body.cent.y;
  double bestDistance = std::numeric_limits<double>::infinity();
  int best = -1;
  for (int index = 0; index < static_cast<int>(page.comps.size()); ++index) {
    if (index == bodyIndex) continue;
    const CC& candidate = page.comps[index];
    if (std::abs(candidate.line - body.line) > 1 ||
        candidate.bodyProbability >= 0.50 ||
        candidate.area > 0.30 * body.area ||
        std::abs(candidate.cent.x - centerX) > 0.70 * body.bb.width)
      continue;
    if (className == "waqf_qaf" &&
        candidate.cent.y > centerY + 0.10 * body.bb.height)
      continue;
    if (className == "waqf_jeem" &&
        std::abs(candidate.cent.y - centerY) > 0.65 * body.bb.height)
      continue;
    const double distance = std::hypot(
        (candidate.cent.x - centerX) / std::max(1, body.bb.width),
        (candidate.cent.y - centerY) / std::max(1, body.bb.height));
    if (distance < bestDistance) {
      bestDistance = distance;
      best = index;
    }
  }
  return best;
}

std::uint64_t recognitionFingerprint(
    const PageModel& page, const ComponentRecognizerParams& params,
    const std::vector<TemplateImage>& markTemplates,
    const std::vector<MarkInventory>& expectedMarksByLine,
    double otsuThreshold) {
  std::uint64_t hash = 1469598103934665603ULL;
  hashValue(hash, params.pageNumber);
  hashString(hash, params.sourceName);
  hashValue(hash, params.minimumWaqfIou);
  hashValue(hash, params.minimumWaqfMargin);
  hashValue(hash, params.minimumThreeDotsIou);
  hashValue(hash, params.minimumKasratanIou);
  hashValue(hash, params.minimumAttachedAlefHamzaAboveProbability);
  hashValue(hash, params.minimumAttachedAlefHamzaAboveIou);
  hashValue(hash, otsuThreshold);
  for (const CC& component : page.comps) {
    hashValue(hash, component.label);
    hashValue(hash, component.bb.x);
    hashValue(hash, component.bb.y);
    hashValue(hash, component.bb.width);
    hashValue(hash, component.bb.height);
    hashValue(hash, component.area);
    hashValue(hash, component.line);
  }
  for (std::size_t line = 0; line < page.lines.size(); ++line) {
    hashValue(hash, page.lines[line].baseLine);
    for (const int count : expectedMarksByLine[line]) hashValue(hash, count);
  }
  auto hashTemplates = [&](const std::vector<TemplateImage>& templates) {
    for (const TemplateImage& item : templates) {
      hashString(hash, item.path.string());
      const auto size = std::filesystem::file_size(item.path);
      const auto modified =
          std::filesystem::last_write_time(item.path).time_since_epoch().count();
      hashValue(hash, size);
      hashValue(hash, modified);
    }
  };
  hashTemplates(markTemplates);
  return hash;
}

std::filesystem::path cachePath(const ComponentRecognizerParams& params) {
  std::ostringstream name;
  name << safeName(params.sourceName) << "-page" << std::setw(3)
       << std::setfill('0') << params.pageNumber << ".json";
  return params.cacheRoot / name.str();
}

void applyRecognitions(PageModel& page,
                       const std::vector<Recognition>& recognitions,
                       std::string_view origin) {
  for (const Recognition& recognition : recognitions) {
    if (recognition.threeDotsAboveComponentCount == 1) {
      for (const int componentIndex : recognition.componentParts)
        if (componentIndex >= 0 &&
            componentIndex < static_cast<int>(page.comps.size()))
          page.comps[componentIndex]
              .detectedThreeDotsAboveComponentCount = 1;
    }
    if (recognition.sajdaComponentCount > 0) {
      for (const int componentIndex : recognition.componentParts)
        if (componentIndex >= 0 &&
            componentIndex < static_cast<int>(page.comps.size()))
          page.comps[componentIndex].detectedSajdaComponentCount =
              std::max(page.comps[componentIndex].detectedSajdaComponentCount,
                       recognition.sajdaComponentCount);
    }
    if (recognition.attachedAlefHamzaAboveProbability > 0.0) {
      for (const int componentIndex : recognition.componentParts)
        if (componentIndex >= 0 &&
            componentIndex < static_cast<int>(page.comps.size()))
          page.comps[componentIndex].attachedAlefHamzaAboveProbability =
              recognition.attachedAlefHamzaAboveProbability;
    }
    if (recognition.bodyTemplateProbability > 0.0) {
      for (const int componentIndex : recognition.componentParts)
        if (componentIndex >= 0 &&
            componentIndex < static_cast<int>(page.comps.size()))
          page.comps[componentIndex].bodyTemplateProbability = std::max(
              page.comps[componentIndex].bodyTemplateProbability,
              recognition.bodyTemplateProbability);
    }
    for (const int componentIndex : recognition.componentParts) {
      if (componentIndex < 0 ||
          componentIndex >= static_cast<int>(page.comps.size()))
        continue;
      CC& component = page.comps[componentIndex];
      for (std::size_t mark = 0; mark < kMarkClassCount; ++mark) {
        if (!recognition.templateMarkClasses[mark]) continue;
        component.templateMarkClasses[mark] = 1;
        component.templateMarkProbabilities[mark] =
            recognition.templateMarkProbabilities[mark];
      }
    }
    if (recognition.targetLine < 0 ||
        recognition.lineAssignmentClass.empty())
      continue;
    for (const int componentIndex : recognition.componentParts) {
      if (componentIndex < 0 ||
          componentIndex >= static_cast<int>(page.comps.size()))
        continue;
      CC& component = page.comps[componentIndex];
      if (component.line == recognition.targetLine) continue;
      const int previousLine = component.line;
      component.line = recognition.targetLine;
      page.assignmentCorrections.push_back(
          std::string(origin) + "-" + recognition.lineAssignmentClass +
          " component=" +
          std::to_string(componentIndex) + " line=" +
          std::to_string(previousLine) + "->" +
          std::to_string(recognition.targetLine) + " score=" +
          std::to_string(recognition.score));
    }
  }
}
}  // namespace

ComponentRecognizer::ComponentRecognizer(ComponentRecognizerParams params)
    : m_params(std::move(params)) {}

void ComponentRecognizer::recognize(
    PageModel& page,
    const std::vector<MarkInventory>& expectedMarksByLine,
    const cv::Mat& grayscale, double otsuThreshold) {
  using Clock = std::chrono::steady_clock;
  const auto recognizerStarted = Clock::now();
  const std::vector<TemplateImage> markTemplates =
      loadAllMarkTemplates(m_params);
  const std::string fingerprint = hashText(recognitionFingerprint(
      page, m_params, markTemplates, expectedMarksByLine, otsuThreshold));
  const std::filesystem::path path = cachePath(m_params);
  if (!m_params.resetCache && !m_params.cacheRoot.empty() &&
      std::filesystem::is_regular_file(path)) {
    std::ifstream input(path);
    const std::string json((std::istreambuf_iterator<char>(input)),
                           std::istreambuf_iterator<char>());
    Cache cache;
    if (!glz::read_json(cache, json) &&
        cache.schemaVersion ==
            component_recognizer_json::kCacheSchemaVersion &&
        cache.fingerprint == fingerprint) {
      applyRecognitions(page, cache.recognitions, "cached-recognition");
      if (m_params.reportTimings)
        std::cerr << "Timing page " << m_params.pageNumber
                  << " component-recognizer cache-hit: "
                  << std::chrono::duration<double>(Clock::now() -
                                                   recognizerStarted)
                         .count()
                  << " s\n";
      return;
    }
  }

  const auto recognitionComputationStarted = Clock::now();

  std::vector<cv::Mat> masks;
  masks.reserve(page.comps.size());
  for (const CC& component : page.comps)
    masks.push_back(normalizedMask(page, component));

  std::vector<Recognition> recognitions;
  const std::size_t quranicSignIndex =
      static_cast<std::size_t>(MarkClass::QuranicSign);

  // U+06E9 is one semantic combining mark but is rendered by this scan set as
  // an outer arch plus an inner stepped component. Recognize that pair before
  // PAW DP so both components receive one quranic-sign instance and neither
  // can be selected as the Arabic body merely because it is large.
  std::vector<std::uint8_t> usedSajdaComponent(page.comps.size(), 0);
  for (int targetLine = 0;
       targetLine < static_cast<int>(expectedMarksByLine.size());
       ++targetLine) {
    const int maximumPairs = expectedMarksByLine[targetLine][quranicSignIndex];
    if (maximumPairs <= 0 ||
        targetLine >= static_cast<int>(page.lines.size()))
      continue;

    struct SajdaCandidate {
      int outer = -1;
      int inner = -1;
      double score = 0.0;
    };
    std::vector<SajdaCandidate> candidates;
    for (int first = 0; first < static_cast<int>(page.comps.size()); ++first) {
      for (int second = first + 1;
           second < static_cast<int>(page.comps.size()); ++second) {
        if (usedSajdaComponent[first] || usedSajdaComponent[second]) continue;
        int outer = first;
        int inner = second;
        if (page.comps[inner].bb.width > page.comps[outer].bb.width ||
            page.comps[inner].bb.height > page.comps[outer].bb.height)
          std::swap(outer, inner);
        const std::optional<double> score = sajdaPairScore(
            page, page.comps[first], page.comps[second], targetLine);
        if (!score) continue;
        candidates.push_back({outer, inner, *score});
      }
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const SajdaCandidate& left, const SajdaCandidate& right) {
                return left.score > right.score;
              });
    int selectedPairs = 0;
    for (const SajdaCandidate& candidate : candidates) {
      if (selectedPairs >= maximumPairs ||
          usedSajdaComponent[candidate.outer] ||
          usedSajdaComponent[candidate.inner])
        continue;
      usedSajdaComponent[candidate.outer] = 1;
      usedSajdaComponent[candidate.inner] = 1;
      ++selectedPairs;

      Recognition recognition;
      recognition.componentParts = {candidate.outer, candidate.inner};
      recognition.lineAssignmentClass = "sajda";
      recognition.score = candidate.score;
      recognition.targetLine = targetLine;
      recognition.sajdaComponentCount = 2;
      recognition.templateMarkClasses[quranicSignIndex] = 1;
      recognition.templateMarkProbabilities[quranicSignIndex] =
          candidate.score;
      recognitions.push_back(std::move(recognition));
    }
  }

  // The mark banks answer "is this a known mark?". A body bank answers the
  // opposite question for the components the preliminary classifier already
  // called marks -- and that classifier's rule is binary, so a small round
  // letter whose ink stops a hair above the line meets its definition of one.
  // On page 321 l08_w09 the ta-marbuta of بِٱلصَّلَوٰةِ ends 19 px above a 1736 px
  // pitch and is scored 0.05, after which the kasra beneath it -- which does
  // cross the line -- becomes the PAW's body. The shapes do separate: that
  // ta-marbuta scores at most 0.657 against any mark bank, under the 0.72
  // those banks work at, so a bank of its own can claim it without contest.
  if (hasTemplate(markTemplates, "body", "whole")) {
    for (int index = 0; index < static_cast<int>(page.comps.size()); ++index) {
      const CC& component = page.comps[index];
      if (component.line < 0 ||
          component.line >= static_cast<int>(page.lines.size()) ||
          component.bodyProbability >= 0.50 ||
          !componentEnclosesBackground(page, component))
        continue;
      // Shape only means something where a body could be written. The small
      // high rounded zero of وا۟ is a ring like a heh and survives both the
      // overlap and the counter -- what it does not survive is being 645 px
      // clear of the line on a 1700 px pitch, where no body is. The band is
      // the one the preliminary classifier uses, and the heh this bank exists
      // for misses the line by 19 px.
      const double pitch =
          component.line > 0
              ? page.lines[component.line].baseLine -
                    page.lines[component.line - 1].baseLine
              : component.line + 1 < static_cast<int>(page.lines.size())
                    ? page.lines[component.line + 1].baseLine -
                          page.lines[component.line].baseLine
                    : 1.0;
      const double baseline = page.lines[component.line].baseLine;
      const double band = 0.045 * pitch;
      if (component.bb.y > baseline + band ||
          component.bb.y + component.bb.height < baseline - band)
        continue;
      const double score =
          bestScore(masks[index], markTemplates, "body", "whole");
      if (score <= 0.0) continue;
      Recognition recognition;
      recognition.componentParts = {index};
      recognition.bodyTemplateProbability = score;
      recognitions.push_back(std::move(recognition));
    }
  }

  const std::vector<MarkClass> markTemplateClasses =
      availableMarkTemplateClasses(markTemplates);
  if (!markTemplateClasses.empty()) {
    for (int index = 0; index < static_cast<int>(page.comps.size()); ++index) {
      const CC& component = page.comps[index];
      if (component.line < 0 ||
          component.line >= static_cast<int>(page.lines.size()))
        continue;
      const double pitch =
          component.line > 0
              ? page.lines[component.line].baseLine -
                    page.lines[component.line - 1].baseLine
              : component.line + 1 < static_cast<int>(page.lines.size())
                    ? page.lines[component.line + 1].baseLine -
                          page.lines[component.line].baseLine
                    : 1.0;
      const double baseline = page.lines[component.line].baseLine;
      const double bboxAspect =
          std::max(component.bb.width, component.bb.height) /
          double(std::max(1, std::min(component.bb.width,
                                      component.bb.height)));
      const bool compactBaselineTemplateCandidate =
          component.area <= 0.030 * pitch * pitch &&
          component.bb.height <= 0.25 * pitch && bboxAspect <= 2.0 &&
          component.bb.y <= baseline + 0.065 * pitch &&
          component.bb.y + component.bb.height >= baseline - 0.065 * pitch;
      // A mark written below the line is not compact: the small low meem of
      // an iqlab is a head with a descender, 183x459 at an aspect of 2.51 on
      // page 518, so the compact test above rejects it on both height and
      // aspect and its bank is never consulted. Below the line there is room
      // for a tail, and the area bound still keeps real bodies out -- the
      // jeem of that same word is 137,449 px against a 90,203 ceiling.
      const double centerY = component.bb.y + component.bb.height / 2.0;
      const bool descendingBelowTemplateCandidate =
          component.area <= 0.030 * pitch * pitch &&
          component.bb.height <= 0.35 * pitch && bboxAspect <= 3.0 &&
          centerY > baseline &&
          component.bb.y <= baseline + 0.065 * pitch;
      if (component.bodyProbability >= 0.50 &&
          !compactBaselineTemplateCandidate &&
          !descendingBelowTemplateCandidate)
        continue;
      const bool observedBelow = centerY > baseline + 0.015 * pitch;
      Recognition recognition;
      recognition.componentParts = {index};
      bool hasTemplateClass = false;
      for (const MarkClass markClass : markTemplateClasses) {
        if (markClassIsBelow(markClass) != observedBelow) continue;
        const std::size_t mark = static_cast<std::size_t>(markClass);
        recognition.templateMarkClasses[mark] = 1;
        recognition.templateMarkProbabilities[mark] = bestScore(
            masks[index], markTemplates, markClassName(markClass), "whole");
        if (markClass == MarkClass::Kasratan)
          recognition.templateMarkProbabilities[mark] = std::max(
              recognition.templateMarkProbabilities[mark],
              bestScore(masks[index], markTemplates, "kasratan", "open"));
        // The pair over a Teh is drawn two ways in these scans, side by side
        // and stacked, and the stacked form is not a rotation of the other --
        // its lobes meet at a waist. Kept as its own role rather than mixed
        // into the class: the score is the same either way, since both are a
        // best-of, but a small alef has nearly the same bounding box as a
        // stacked pair and scores a median 0.50 against these templates, so
        // when that has to be told apart it must be told apart on this path
        // alone. Of 37,643 two-dot components, 1,396 are stacked, and the
        // horizontal bank scored them a median 0.24 against 0.87 for its own
        // form -- below what it gives a fatha.
        if (markClass == MarkClass::DotAboveTwo ||
            markClass == MarkClass::DotBelowTwo)
          recognition.templateMarkProbabilities[mark] = std::max(
              recognition.templateMarkProbabilities[mark],
              bestScore(masks[index], markTemplates, markClassName(markClass),
                        "vertical"));
        hasTemplateClass = true;
      }
      if (hasTemplateClass) recognitions.push_back(std::move(recognition));
    }
  }

  // A kasratan is one semantic mark represented by two detached bars. A
  // bar that lies just across a line's Voronoi boundary is assigned to the
  // following line by the initial geometric split, so the word DP would
  // never have a chance to join it with its mate. Recognize the pair before
  // word segmentation and move the lower bar to the line whose text expects
  // kasratan. The template test is required for both bars; geometry alone is
  // too permissive around dense line boundaries.
  if (hasTemplate(markTemplates, "kasratan", "whole") ||
      hasTemplate(markTemplates, "kasratan", "open")) {
    struct KasratanPair {
      int upper = -1;
      int lower = -1;
      int targetLine = -1;
      double score = 0.0;
      std::string templateRole;
    };

    const std::size_t kasratanIndex =
        static_cast<std::size_t>(MarkClass::Kasratan);
    std::vector<double> alignedScores(page.comps.size(), 0.0);
    std::vector<double> openScores(page.comps.size(), 0.0);
    for (int index = 0; index < static_cast<int>(page.comps.size()); ++index) {
      if (page.comps[index].bodyProbability >= 0.50) continue;
      alignedScores[index] = bestScore(
          masks[index], markTemplates, "kasratan", "whole");
      openScores[index] = bestScore(
          masks[index], markTemplates, "kasratan", "open");
    }

    for (int upperLine = 0;
         upperLine + 1 < static_cast<int>(page.lines.size()); ++upperLine) {
      if (upperLine >= static_cast<int>(expectedMarksByLine.size()) ||
          expectedMarksByLine[upperLine][kasratanIndex] == 0)
        continue;
      const int lowerLine = upperLine + 1;
      const double pitch =
          page.lines[lowerLine].baseLine - page.lines[upperLine].baseLine;
      if (pitch <= 0.0) continue;

      std::vector<KasratanPair> candidates;
      for (int upper = 0; upper < static_cast<int>(page.comps.size());
           ++upper) {
        const CC& upperComponent = page.comps[upper];
        if (upperComponent.line != upperLine ||
            std::max(alignedScores[upper], openScores[upper]) <
                m_params.minimumKasratanIou ||
            upperComponent.bb.y + upperComponent.bb.height / 2.0 <=
                page.lines[upperLine].baseLine + 0.015 * pitch)
          continue;
        for (int lower = 0; lower < static_cast<int>(page.comps.size());
             ++lower) {
          const CC& lowerComponent = page.comps[lower];
          if (lowerComponent.line != lowerLine ||
              std::max(alignedScores[lower], openScores[lower]) <
                  m_params.minimumKasratanIou ||
              lowerComponent.bb.y >= page.lines[lowerLine].baseLine)
            continue;

          const int overlapLeft = std::max(upperComponent.bb.x,
                                           lowerComponent.bb.x);
          const int overlapRight = std::min(
              upperComponent.bb.x + upperComponent.bb.width,
              lowerComponent.bb.x + lowerComponent.bb.width);
          const int overlap = std::max(0, overlapRight - overlapLeft);
          const double minimumWidth = std::min(upperComponent.bb.width,
                                               lowerComponent.bb.width);
          const double centerXUpper =
              upperComponent.bb.x + upperComponent.bb.width / 2.0;
          const double centerXLower =
              lowerComponent.bb.x + lowerComponent.bb.width / 2.0;
          const double verticalGap =
              lowerComponent.bb.y -
              (upperComponent.bb.y + upperComponent.bb.height);
          const double minimumHeight = std::min(upperComponent.bb.height,
                                                lowerComponent.bb.height);
          if (overlap < 0.20 * minimumWidth ||
              verticalGap < -0.25 * minimumHeight ||
              verticalGap > 0.50 * minimumHeight ||
              // The two bars of a tanwin are close together. A larger gap
              // usually means two unrelated marks from neighboring lines
              // that merely have a similar diagonal shape.
              verticalGap > 0.12 * pitch)
            continue;

          const bool verticallyAligned =
              std::abs(centerXUpper - centerXLower) <= 0.25 * minimumWidth;
          const bool shiftedLeftOpen =
              centerXLower < centerXUpper - 0.25 * minimumWidth;
          double score = 0.0;
          std::string templateRole;
          if (verticallyAligned &&
              std::min(alignedScores[upper], alignedScores[lower]) >=
                  m_params.minimumKasratanIou) {
            score = std::min(alignedScores[upper], alignedScores[lower]);
            templateRole = "whole";
          } else if (shiftedLeftOpen &&
                     std::min(openScores[upper], openScores[lower]) >=
                         m_params.minimumKasratanIou) {
            score = std::min(openScores[upper], openScores[lower]);
            templateRole = "open";
          } else {
            continue;
          }
          candidates.push_back(
              {upper, lower, upperLine, score, std::move(templateRole)});
        }
      }

      std::sort(candidates.begin(), candidates.end(),
                [](const KasratanPair& left, const KasratanPair& right) {
                  return left.score > right.score;
                });
      const int maximumPairs =
          expectedMarksByLine[upperLine][kasratanIndex];
      int selectedPairs = 0;
      std::vector<std::uint8_t> used(page.comps.size(), 0);
      for (const KasratanPair& pair : candidates) {
        if (selectedPairs >= maximumPairs || used[pair.upper] ||
            used[pair.lower])
          continue;
        used[pair.upper] = 1;
        used[pair.lower] = 1;
        ++selectedPairs;

        Recognition recognition;
        recognition.componentParts = {pair.upper, pair.lower};
        recognition.lineAssignmentClass =
            pair.templateRole == "open" ? "kasratan-open" : "kasratan";
        recognition.score = pair.score;
        recognition.targetLine = pair.targetLine;
        recognition.templateMarkClasses[kasratanIndex] = 1;
        recognition.templateMarkProbabilities[kasratanIndex] = pair.score;
        recognitions.push_back(std::move(recognition));
      }
    }
  }

  for (int index = 0; index < static_cast<int>(page.comps.size()); ++index) {
    const CC& component = page.comps[index];
    if (component.line < 0 ||
        component.line >= static_cast<int>(expectedMarksByLine.size()) ||
        expectedMarksByLine[component.line][static_cast<std::size_t>(
            MarkClass::HamzaAbove)] == 0)
      continue;
    double probability = attachedAlefHamzaAboveProbability(
        page, component, grayscale, otsuThreshold);
    const bool hasAttachedAlefTemplate =
        hasTemplate(markTemplates, "alef_hamza_above_attached", "whole");
    double templateIou = 0.0;
    bool usedTemplateFallback = false;

    // The two-seed test is the primary recognizer: it verifies that the
    // component contains the expected separated hamza and Alef regions at a
    // useful threshold. Some attached forms are nevertheless connected at
    // every threshold considered above. For those cases, use the canonical
    // whole-component template only after applying a strict Alef-like shape
    // and baseline guard. This keeps ordinary tall marks from becoming an
    // attached Alef merely because they resemble one at one threshold.
    if (probability < m_params.minimumAttachedAlefHamzaAboveProbability &&
        hasAttachedAlefTemplate) {
      const int baseline = static_cast<int>(
          std::lround(page.lines[component.line].baseLine));
      const int bottom = component.bb.y + component.bb.height;
      const bool tallAlefTemplateCandidate =
          component.bodyProbability >= 0.50 && component.bb.width >= 100 &&
          component.bb.width <= 700 && component.bb.height >= 500 &&
          component.bb.height <= 1700 &&
          component.bb.height >= 2.50 * component.bb.width &&
          component.bb.height <
              kMaximumAttachedAlefHamzaAspect * component.bb.width &&
          component.bb.y <= baseline - 300 && bottom >= baseline - 300 &&
          bottom <= baseline + 500;
      if (tallAlefTemplateCandidate) {
        templateIou = bestScore(
            masks[index], markTemplates, "alef_hamza_above_attached",
            "whole");
        if (templateIou >= m_params.minimumAttachedAlefHamzaAboveIou) {
          // This is a gated template recognition, not a raw probability from
          // the two-seed classifier. Store the configured acceptance floor so
          // downstream PAW DP treats the attached hamza as already present.
          probability = m_params.minimumAttachedAlefHamzaAboveProbability;
          usedTemplateFallback = true;
        }
      }
    }
    if (probability < m_params.minimumAttachedAlefHamzaAboveProbability)
      continue;
    double score = probability;
    if (hasAttachedAlefTemplate) {
      if (templateIou == 0.0)
        templateIou = bestScore(
            masks[index], markTemplates, "alef_hamza_above_attached",
            "whole");
      if (templateIou < m_params.minimumAttachedAlefHamzaAboveIou) continue;
      score = usedTemplateFallback
                  ? templateIou
                  : 0.55 * probability + 0.45 * templateIou;
    }
    recognitions.push_back(
        Recognition{{index}, 0, probability, {}, score, -1});
  }
  if (hasTemplate(markTemplates, "dot_above_three", "whole")) {
    for (int index = 0; index < static_cast<int>(page.comps.size()); ++index) {
      const CC& component = page.comps[index];
      if (component.line < 0 ||
          component.line >= static_cast<int>(page.lines.size()) ||
          component.bodyProbability >= 0.50 ||
          component.bb.y + component.bb.height >
              page.lines[component.line].baseLine)
        continue;
      if (expectedMarksByLine[component.line][static_cast<std::size_t>(
              MarkClass::DotAboveThree)] == 0)
        continue;
      const double score = bestScore(masks[index], markTemplates,
                                     "dot_above_three", "whole");
      if (score >= m_params.minimumThreeDotsIou)
        recognitions.push_back(
            Recognition{{index}, 1, 0.0, {}, score, -1});
    }
  }

  const std::vector<std::string> classes =
      availableWaqfClasses(markTemplates);
  if (!classes.empty()) {
    for (int index = 0; index < static_cast<int>(page.comps.size()); ++index) {
      const CC& component = page.comps[index];
      if (component.line < 0 ||
          component.line >= static_cast<int>(page.lines.size()) ||
          component.bodyProbability >= 0.50)
        continue;
      std::vector<std::pair<double, std::string>> scores;
      for (const std::string& className : classes) {
        const auto markClass = waqfMarkClass(className);
        if (!markClass ||
            expectedTargetLines(component, *markClass, expectedMarksByLine,
                                page)
                .empty())
          continue;
        const std::string role =
            className == "waqf_qaf" || className == "waqf_jeem" ? "body"
                                                                : "whole";
        scores.emplace_back(
            bestScore(masks[index], markTemplates, className, role),
            className);
      }
      if (scores.empty()) continue;
      std::sort(scores.begin(), scores.end(), std::greater<>());
      const double score = scores.front().first;
      const double runnerUp = scores.size() > 1 ? scores[1].first : 0.0;
      if (score < m_params.minimumWaqfIou ||
          score - runnerUp < m_params.minimumWaqfMargin)
        continue;
      const std::string& className = scores.front().second;
      const MarkClass markClass = *waqfMarkClass(className);
      const std::vector<int> targets = expectedTargetLines(
          component, markClass, expectedMarksByLine, page);
      int targetLine = -1;
      if (targets.size() == 1)
        targetLine = targets.front();
      else if (std::find(targets.begin(), targets.end(), component.line) !=
               targets.end())
        targetLine = component.line;
      else
        continue;

      std::vector<int> parts{index};
      if (className == "waqf_qaf" || className == "waqf_jeem") {
        const int companion = chooseCompanion(page, index, className);
        if (companion < 0) continue;
        parts.push_back(companion);
      }
      if (std::any_of(parts.begin(), parts.end(), [&](int part) {
            return page.comps[part].bb.y + page.comps[part].bb.height >
                   page.lines[targetLine].baseLine;
          }))
        continue;
      recognitions.push_back(
          Recognition{std::move(parts), 0, 0.0, className, score, targetLine});
    }
  }

  applyRecognitions(page, recognitions, "in-memory-recognition");
  const auto recognitionComputed = Clock::now();
  if (!m_params.cacheRoot.empty()) {
    std::filesystem::create_directories(m_params.cacheRoot);
    Cache cache{component_recognizer_json::kCacheSchemaVersion, fingerprint,
                recognitions};
    std::string json;
    if (const auto error =
            glz::write<glz::opts{.prettify = true}>(cache, json))
      throw std::runtime_error("Could not serialize recognizer cache: " +
                               glz::format_error(error, json));
    std::ofstream output(path);
    if (!output) throw std::runtime_error("Could not write " + path.string());
    output << json;
  }
  if (m_params.reportTimings) {
    std::cerr << "Timing page " << m_params.pageNumber
              << " component-recognizer compute: "
              << std::chrono::duration<double>(recognitionComputed -
                                               recognitionComputationStarted)
                     .count()
              << " s\n";
    std::cerr << "Timing page " << m_params.pageNumber
              << " component-recognizer total: "
              << std::chrono::duration<double>(Clock::now() -
                                               recognizerStarted)
                     .count()
              << " s\n";
  }
}
