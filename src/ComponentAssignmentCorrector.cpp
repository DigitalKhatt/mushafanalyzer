#include "ComponentAssignmentCorrector.h"

#include <glaze/glaze.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace component_assignment_json {
struct ComponentSelector {
  std::string component_type;
  int ordinal = -1;
  std::string order;
};

struct ReassignmentPreconditions {
  int from_observed_minus_expected = 0;
  int to_expected_minus_observed = 0;
};

struct ReassignmentPostconditions {
  int from_component_difference = 0;
  int to_component_difference = 0;
};

struct ComponentReassignment {
  int page = -1;
  int line = -1;
  int from_word = -1;
  int to_word = -1;
  ComponentSelector selector;
  ReassignmentPreconditions preconditions;
  ReassignmentPostconditions postconditions;
  std::string reason;
};

struct SegmentationExceptions {
  int version = 0;
  std::vector<ComponentReassignment> component_reassignments;
};

struct WaqfSamplePath {
  std::string role;
  std::string maskPath;
  std::string contextPath;
};

struct WaqfDetection {
  std::string source;
  int page = -1;
  int componentIndex = -1;
  std::vector<int> componentParts;
  std::vector<std::string> componentPartRoles;
  std::vector<std::string> partMaskPaths;
  std::vector<std::string> partContextPaths;
  std::string className;
  double score = 0.0;
  double margin = 0.0;
  int currentLineIndex = -1;
  std::string currentWordId;
  int currentComponentExcess = 0;
  int targetLineIndex = -1;
  std::string targetWordId;
  std::string targetExpectedText;
  int targetComponentDeficit = 0;
  std::string maskPath;
  std::string contextPath;
  bool reviewed = false;
  bool autoAcceptedCorrection = false;
  std::vector<WaqfSamplePath> samplePaths;
};
}  // namespace component_assignment_json

namespace {
using namespace component_assignment_json;

int expectedComponentCount(const WordBox& word) {
  return word.expectedBodyComponents + word.expectedMarkComponents -
         word.recognizedComponentCountAdjustment;
}

int componentDifference(const WordBox& word) {
  return static_cast<int>(word.compIds.size()) - expectedComponentCount(word);
}

bool isNonBody(const CC& component) {
  return component.bodyProbability <
         std::max({component.dotProbability,
                   component.ordinaryMarkProbability,
                   component.smallLetterProbability,
                   component.unknownProbability});
}

void sortComponentsRtl(const PageModel& page, std::vector<int>& components) {
  std::sort(components.begin(), components.end(), [&](int left, int right) {
    const CC& leftComponent = page.comps.at(left);
    const CC& rightComponent = page.comps.at(right);
    const int leftEdge = leftComponent.bb.x + leftComponent.bb.width;
    const int rightEdge = rightComponent.bb.x + rightComponent.bb.width;
    if (leftEdge != rightEdge) return leftEdge > rightEdge;
    return leftComponent.cent.y < rightComponent.cent.y;
  });
}

void recomputeObservedSmallLetters(const PageModel& page, WordBox& word) {
  word.observedSmallLetterComponents = 0.0;
  for (int componentIndex : word.compIds)
    word.observedSmallLetterComponents +=
        page.comps.at(componentIndex).smallLetterProbability;
}

SegmentationExceptions loadExceptions(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error("Could not open segmentation exceptions " +
                             path.string());
  const std::string json((std::istreambuf_iterator<char>(input)),
                         std::istreambuf_iterator<char>());
  SegmentationExceptions result;
  if (const auto error = glz::read_json(result, json))
    throw std::runtime_error(
        "Could not parse segmentation exceptions: " +
        glz::format_error(error, json));
  if (result.version != 1)
    throw std::runtime_error("Unsupported segmentation exception version " +
                             std::to_string(result.version));
  return result;
}

bool containsCodePoint(std::string_view text, char32_t wanted) {
  for (size_t index = 0; index < text.size();) {
    const unsigned char lead = static_cast<unsigned char>(text[index++]);
    char32_t value = lead;
    int continuationCount = 0;
    if ((lead & 0xE0) == 0xC0) {
      value = lead & 0x1F;
      continuationCount = 1;
    } else if ((lead & 0xF0) == 0xE0) {
      value = lead & 0x0F;
      continuationCount = 2;
    } else if ((lead & 0xF8) == 0xF0) {
      value = lead & 0x07;
      continuationCount = 3;
    }
    for (int part = 0; part < continuationCount && index < text.size();
         ++part)
      value =
          (value << 6) |
          (static_cast<unsigned char>(text[index++]) & 0x3F);
    if (value == wanted) return true;
  }
  return false;
}

// An ayah marker and its number: written inside a circle, not on a line.
bool isAyahMarkerText(const std::string& text) {
  if (text.empty()) return false;
  for (std::size_t i = 0; i + 1 < text.size(); i += 2) {
    const auto lead = static_cast<unsigned char>(text[i]);
    const auto trail = static_cast<unsigned char>(text[i + 1]);
    const bool digit = lead == 0xD9 && trail >= 0xA0 && trail <= 0xA9;
    const bool ornament = lead == 0xDB && (trail == 0x9D || trail == 0x9E);
    if (!digit && !ornament) return false;
  }
  return true;
}

bool containsWaqf(const std::string& text) {
  for (char32_t value = 0x06D6; value <= 0x06DC; ++value)
    if (containsCodePoint(text, value)) return true;
  return false;
}

int horizontalDistance(const cv::Rect& first, const cv::Rect& second) {
  if (first.x > second.x + second.width)
    return first.x - (second.x + second.width);
  if (second.x > first.x + first.width)
    return second.x - (first.x + first.width);
  return 0;
}

double medianLinePitch(const PageModel& page) {
  std::vector<double> pitches;
  for (size_t line = 1; line < page.lines.size(); ++line)
    pitches.push_back(page.lines[line].baseLine -
                      page.lines[line - 1].baseLine);
  if (pitches.empty()) return 1.0;
  std::sort(pitches.begin(), pitches.end());
  return pitches[pitches.size() / 2];
}

void recomputeWordRect(const PageModel& page, WordBox& word) {
  cv::Rect rect;
  for (int componentIndex : word.compIds)
    rect |= page.comps.at(componentIndex).bb;
  word.rect = rect;
}

void moveComponent(PageModel& page, WordBox& source, WordBox& destination,
                   int componentIndex, std::string reason) {
  source.compIds.erase(
      std::remove(source.compIds.begin(), source.compIds.end(),
                  componentIndex),
      source.compIds.end());
  if (std::find(destination.compIds.begin(), destination.compIds.end(),
                componentIndex) == destination.compIds.end())
    destination.compIds.push_back(componentIndex);
  page.assignmentCorrections.push_back(
      std::move(reason) + ": " + source.wordId + " -> " +
      destination.wordId);
}

// The frame of an ayah marker is ornament, and a scan breaks ornament: on page
// 403 line 1 the right-hand tip where its two arcs meet came away as a 98x217
// sliver, which the line DP then gave to يَشْعُرُونَ as the body of its Noon.
// The damage is not only the wrong body. The sliver counts as a third
// body-like component in a word that has three PAWs, so the word no longer
// looks short of bodies, and the Waw fused into يشعر can no longer be seen to
// share it -- the same masking a stolen letter causes on page 411.
//
// A fragment of the frame lies inside the marker's own box while the marker
// holds more components than its text asks for. Both together are narrow
// enough to be safe: a marker is a self-contained ornament with a gap around
// it, and one that has not shed anything is left alone.
void reclaimAyahMarkerFragments(PageModel& page) {
  for (LineBox& line : page.lines) {
    for (WordBox& marker : line.words) {
      if (!isAyahMarkerText(marker.expectedText)) continue;
      if (componentDifference(marker) <= 0) continue;
      for (WordBox& word : line.words) {
        if (&word == &marker) continue;
        std::vector<int> fragments;
        for (const int index : word.compIds) {
          const cv::Rect& box = page.comps.at(index).bb;
          const int inside = (box & marker.rect).area();
          if (inside * 4 >= box.area() * 3) fragments.push_back(index);
        }
        for (const int index : fragments) {
          if (word.compIds.size() <= 1) break;
          moveComponent(page, word, marker, index, "ayah-marker fragment");
        }
      }
    }
  }
}


void reassignWaqfAcrossLines(PageModel& page, double linePitch) {
  struct Owner {
    int line = -1;
    int word = -1;
  };
  std::vector<Owner> owners(page.comps.size());
  for (size_t line = 0; line < page.lines.size(); ++line)
    for (size_t word = 0; word < page.lines[line].words.size(); ++word)
      for (int component : page.lines[line].words[word].compIds)
        owners.at(component) = {static_cast<int>(line),
                                static_cast<int>(word)};

  for (size_t componentIndex = 0; componentIndex < page.comps.size();
       ++componentIndex) {
    const Owner owner = owners[componentIndex];
    if (owner.line < 0) continue;
    const CC& component = page.comps[componentIndex];
    WordBox& source = page.lines[owner.line].words[owner.word];
    const int centroidY = static_cast<int>(std::lround(
        page.centroids.at<double>(component.label, 1)));
    int geometricLine = owner.line;
    for (size_t line = 0; line < page.lines.size(); ++line)
      if (page.lines[line].rect.contains(
              cv::Point(component.bb.x, centroidY))) {
        geometricLine = static_cast<int>(line);
        break;
      }
    const double sourceBaseline = page.lines[owner.line].baseLine;
    if (component.bb.y <= sourceBaseline + 100 &&
        component.bb.y + component.bb.height >= sourceBaseline - 250)
      continue;

    bool bottomEdgeCandidate = false;
    if (geometricLine == owner.line &&
        owner.line + 1 < static_cast<int>(page.lines.size()) &&
        isNonBody(component) && componentDifference(source) == 1) {
      const LineBox& nextLine = page.lines[owner.line + 1];
      const int bottomOverlap =
          component.bb.y + component.bb.height - nextLine.rect.y;
      bottomEdgeCandidate =
          bottomOverlap > 0 &&
          component.bb.y > sourceBaseline + 0.10 * linePitch;
      if (bottomEdgeCandidate) geometricLine = owner.line + 1;
    }
    if (geometricLine == owner.line ||
        std::abs(geometricLine - owner.line) != 1)
      continue;

    int bestWord = -1;
    int bestDistance = std::numeric_limits<int>::max();
    for (size_t word = 0; word < page.lines[geometricLine].words.size();
         ++word) {
      const WordBox& candidate = page.lines[geometricLine].words[word];
      if (!containsWaqf(candidate.expectedText)) continue;
      if (bottomEdgeCandidate && componentDifference(candidate) != -1)
        continue;
      const int distance = horizontalDistance(component.bb, candidate.rect);
      if (distance < bestDistance) {
        bestDistance = distance;
        bestWord = static_cast<int>(word);
      }
    }
    if (bestWord < 0 || bestDistance > 0.6 * linePitch) continue;
    WordBox& destination = page.lines[geometricLine].words[bestWord];
    const int destinationOverlap =
        std::max(0, component.bb.y + component.bb.height -
                        page.lines[geometricLine].rect.y);
    moveComponent(page, source, destination,
                  static_cast<int>(componentIndex),
                  bottomEdgeCandidate
                      ? "waqf-bottom-edge overlap=" +
                            std::to_string(destinationOverlap)
                      : "waqf-line");
  }
}

void reassignUnambiguousCrossLineComponents(PageModel& page, double linePitch) {
  struct Owner {
    int line = -1;
    int word = -1;
  };
  std::vector<Owner> owners(page.comps.size());
  for (size_t line = 0; line < page.lines.size(); ++line)
    for (size_t word = 0; word < page.lines[line].words.size(); ++word)
      for (int component : page.lines[line].words[word].compIds)
        owners.at(component) = {static_cast<int>(line), static_cast<int>(word)};

  auto lineAtY = [&](int y) {
    for (size_t line = 0; line < page.lines.size(); ++line)
      if (y >= page.lines[line].rect.y && y < page.lines[line].rect.y + page.lines[line].rect.height)
        return static_cast<int>(line);
    return -1;
  };

  for (size_t componentIndex = 0; componentIndex < page.comps.size(); ++componentIndex) {
    const Owner owner = owners[componentIndex];
    if (owner.line < 0) continue;
    const CC& component = page.comps[componentIndex];
    if (!isNonBody(component)) continue;
    const int topLine = lineAtY(component.bb.y);
    const int bottomLine = lineAtY(component.bb.y + component.bb.height - 1);
    if (topLine < 0 || bottomLine < 0 || topLine == bottomLine || std::abs(topLine - bottomLine) != 1) continue;
    if (owner.line != topLine && owner.line != bottomLine) continue;

    WordBox& source = page.lines[owner.line].words[owner.word];
    if (componentDifference(source) != 1) continue;
    const int destinationLine = owner.line == topLine ? bottomLine : topLine;
    std::vector<int> compatibleWords;
    for (size_t word = 0; word < page.lines[destinationLine].words.size(); ++word) {
      const WordBox& candidate = page.lines[destinationLine].words[word];
      if (componentDifference(candidate) != -1) continue;
      if (horizontalDistance(component.bb, candidate.rect) <= 0.15 * linePitch)
        compatibleWords.push_back(static_cast<int>(word));
    }
    if (compatibleWords.size() != 1) continue;
    WordBox& destination = page.lines[destinationLine].words[compatibleWords.front()];
    moveComponent(page, source, destination, static_cast<int>(componentIndex), "bidirectional-edge-count top=" + std::to_string(topLine) + " bottom=" + std::to_string(bottomLine));
    owners[componentIndex] = {destinationLine, compatibleWords.front()};
  }
}

void applyDetectedWaqfGroups(PageModel& page, int pageNumber, const std::filesystem::path& path, const std::string& sourceName) {
  if (path.empty()) return;
  std::ifstream input(path);
  if (!input) throw std::runtime_error("Could not open waqf detections " + path.string());
  std::string json;
  while (std::getline(input, json)) {
    if (json.empty()) continue;
    WaqfDetection detection;
    if (const auto error = glz::read_json(detection, json))
      throw std::runtime_error("Could not parse waqf detection: " + glz::format_error(error, json));
    if (detection.page != pageNumber || detection.source != sourceName || !detection.autoAcceptedCorrection) continue;
    if (detection.targetLineIndex < 0 || detection.targetLineIndex >= static_cast<int>(page.lines.size())) continue;
    WordBox* destination = nullptr;
    for (WordBox& word : page.lines[detection.targetLineIndex].words)
      if (word.wordId == detection.targetWordId) destination = &word;
    if (!destination || componentDifference(*destination) != -static_cast<int>(detection.componentParts.size())) continue;

    std::vector<WordBox*> sources;
    bool valid = true;
    for (int componentIndex : detection.componentParts) {
      if (componentIndex < 0 || componentIndex >= static_cast<int>(page.comps.size())) {
        valid = false;
        break;
      }
      const CC& component = page.comps[componentIndex];
      if (component.bb.y + component.bb.height > page.lines[detection.targetLineIndex].baseLine) {
        valid = false;
        break;
      }
      WordBox* source = nullptr;
      for (LineBox& line : page.lines)
        for (WordBox& word : line.words)
          if (std::find(word.compIds.begin(), word.compIds.end(), componentIndex) != word.compIds.end()) source = &word;
      if (!source || source == destination || componentDifference(*source) <= 0) {
        valid = false;
        break;
      }
      sources.push_back(source);
    }
    if (!valid) continue;
    for (size_t part = 0; part < detection.componentParts.size(); ++part)
      moveComponent(page, *sources[part], *destination, detection.componentParts[part], "detected-" + detection.className + " score=" + std::to_string(detection.score));
  }
}

void removeIsolatedNoise(PageModel& page, double linePitch) {
  const double maximumSize = 0.10 * linePitch;
  const double minimumIsolation = 0.45 * linePitch;
  for (LineBox& line : page.lines) {
    for (WordBox& word : line.words) {
      if (containsWaqf(word.expectedText)) continue;
      std::vector<int> noise;
      for (int componentIndex : word.compIds) {
        const CC& component = page.comps.at(componentIndex);
        if (component.bb.width > maximumSize ||
            component.bb.height > maximumSize)
          continue;
        const double aspect =
            component.bb.width / double(std::max(1, component.bb.height));
        if (aspect < 0.75 || aspect > 1.33 || component.area < 2000 ||
            component.area > 8000)
          continue;
        int nearest = std::numeric_limits<int>::max();
        for (int otherIndex : word.compIds) {
          if (otherIndex == componentIndex) continue;
          nearest =
              std::min(nearest,
                       horizontalDistance(component.bb,
                                          page.comps.at(otherIndex).bb));
        }
        if (nearest > minimumIsolation) noise.push_back(componentIndex);
      }
      for (int componentIndex : noise) {
        word.compIds.erase(
            std::remove(word.compIds.begin(), word.compIds.end(),
                        componentIndex),
            word.compIds.end());
        page.assignmentCorrections.push_back(
            "isolated-noise: removed from " + word.wordId);
      }
    }
  }
}

void applyExplicitReassignments(PageModel& page, int pageNumber,
                                const std::filesystem::path& path) {
  if (path.empty()) return;
  const SegmentationExceptions exceptions = loadExceptions(path);
  for (const ComponentReassignment& correction :
       exceptions.component_reassignments) {
    if (correction.page != pageNumber) continue;
    const std::string location =
        "page " + std::to_string(pageNumber) + " line " +
        std::to_string(correction.line) + " word " +
        std::to_string(correction.from_word);
    auto skip = [&](std::string_view reason) {
      const std::string message = "explicit-reassignment skipped at " +
                                  location + ": " + std::string(reason);
      page.assignmentCorrections.push_back(message);
      std::cerr << message << '\n';
    };

    if (correction.line < 0 ||
        correction.line >= static_cast<int>(page.lines.size())) {
      skip("line does not exist");
      continue;
    }
    LineBox& line = page.lines[correction.line];
    if (correction.from_word < 0 || correction.to_word < 0 ||
        correction.from_word >= static_cast<int>(line.words.size()) ||
        correction.to_word >= static_cast<int>(line.words.size())) {
      skip("word does not exist");
      continue;
    }
    WordBox& source = line.words[correction.from_word];
    WordBox& destination = line.words[correction.to_word];
    if (componentDifference(source) !=
            correction.preconditions.from_observed_minus_expected ||
        -componentDifference(destination) !=
            correction.preconditions.to_expected_minus_observed) {
      skip("component-count precondition failed");
      continue;
    }
    // "body" as well as "non_body": the line DP can put a whole letter in the
    // wrong word, not only a mark. It happens where a word's own count gives
    // it no reason to prefer either reading -- moving a body across a
    // boundary hands the deficit from one word to the other, so both readings
    // cost the same -- and the boundary term then decides on the two facing
    // edges alone. See page 411 line 13.
    const bool wantsBody = correction.selector.component_type == "body";
    if ((correction.selector.component_type != "non_body" && !wantsBody) ||
        correction.selector.order != "right_edge_descending" ||
        correction.selector.ordinal < 0) {
      skip("unsupported selector");
      continue;
    }

    std::vector<int> candidates;
    for (int componentIndex : source.compIds)
      if (isNonBody(page.comps.at(componentIndex)) != wantsBody)
        candidates.push_back(componentIndex);
    sortComponentsRtl(page, candidates);
    if (correction.selector.ordinal >=
        static_cast<int>(candidates.size())) {
      skip("selected component does not exist");
      continue;
    }
    const int componentIndex = candidates[correction.selector.ordinal];
    moveComponent(page, source, destination, componentIndex,
                  "explicit-reassignment (" + correction.reason + ")");
    sortComponentsRtl(page, source.compIds);
    sortComponentsRtl(page, destination.compIds);
    recomputeWordRect(page, source);
    recomputeWordRect(page, destination);
    recomputeObservedSmallLetters(page, source);
    recomputeObservedSmallLetters(page, destination);

    if (componentDifference(source) !=
            correction.postconditions.from_component_difference ||
        componentDifference(destination) !=
            correction.postconditions.to_component_difference) {
      throw std::runtime_error("Segmentation exception postcondition failed at " +
                               location);
    }
    std::cerr << "Applied explicit component " << componentIndex << " at "
              << location << " -> word " << correction.to_word << '\n';
  }
}
}  // namespace

void correctComponentAssignments(
    PageModel& page, int pageNumber,
    const std::filesystem::path& segmentationExceptions,
    const std::filesystem::path& waqfDetections,
    const std::string& waqfDetectionSource) {
  if (page.lines.empty()) return;
  const double linePitch = medianLinePitch(page);
  applyDetectedWaqfGroups(page, pageNumber, waqfDetections, waqfDetectionSource);
  reassignWaqfAcrossLines(page, linePitch);
  reassignUnambiguousCrossLineComponents(page, linePitch);
  removeIsolatedNoise(page, linePitch);
  reclaimAyahMarkerFragments(page);
  applyExplicitReassignments(page, pageNumber, segmentationExceptions);
  for (LineBox& line : page.lines)
    for (WordBox& word : line.words) {
      sortComponentsRtl(page, word.compIds);
      recomputeWordRect(page, word);
      recomputeObservedSmallLetters(page, word);
    }
}
