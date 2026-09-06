#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "SegmentationTypes.h"

struct ComponentRecognizerParams {
  // Canonical layout: root/<mark-class>/*.png, for example
  // root/hamza_above/template01.png and root/fatha/template02.png.
  // Waqf, connected three-dot, and attached alef+hamza classes belong in this
  // same root. Direct waqf_qaf and waqf_jeem files are body templates;
  // nested role directories may be used when a class has more than one role.
  std::filesystem::path markTemplateRoot;
  // Deprecated compatibility roots. New configurations should put all these
  // templates under markTemplateRoot instead.
  std::filesystem::path waqfTemplateRoot;
  std::filesystem::path threeDotsAboveTemplateRoot;
  std::filesystem::path attachedAlefHamzaAboveTemplateRoot;
  std::filesystem::path cacheRoot;
  std::string sourceName;
  int pageNumber = -1;
  bool reportTimings = false;
  bool resetCache = false;
  double minimumWaqfIou = 0.65;
  double minimumWaqfMargin = 0.08;
  double minimumThreeDotsIou = 0.72;
  double minimumKasratanIou = 0.55;
  double minimumAttachedAlefHamzaAboveProbability = 0.82;
  double minimumAttachedAlefHamzaAboveIou = 0.58;
};

class ComponentRecognizer {
 public:
  explicit ComponentRecognizer(ComponentRecognizerParams params);

  // Classifies extracted components in memory. High-confidence grouped marks
  // may change component.line before word DP; grouped three-dot and sajda
  // detections are stored on components for candidate-specific DP decisions.
  void recognize(PageModel& page,
                 const std::vector<MarkInventory>& expectedMarksByLine,
                 const cv::Mat& grayscale, double otsuThreshold);

 private:
  ComponentRecognizerParams m_params;
};
