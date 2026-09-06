#pragma once
#include <filesystem>
#include <string>
#include <utility>
#include <vector>
#include <opencv2/opencv.hpp>

#include "BaselineEstimator.h"
#include "LayoutDb.h"
#include "MarkTaxonomy.h"
#include "PageProcessor.h"
#include "SegmentationTypes.h"

struct SubWordInfo {
  std::u32string text;
  std::string utf8Text;
  std::u32string baseText;
  int baseCount = 0;

  int expBodyConn = 0;   // number of connected body blobs
  int expMarkPaths = 0;  // sum of per-glyph mark paths including dots
  int expSmallLetterPaths = 0;
  MarkInventory expectedMarks{};
  std::vector<ExpectedMarkInstance> expectedMarkInstances;
};

struct WordInfo {
  std::string text;
  std::u32string baseText;

  std::vector<SubWordInfo> subWords;

  int expBodyConn = 0;   // number of connected body blobs
  int expMarkPaths = 0;  // sum of per-glyph mark paths including dots
  int expSmallLetterPaths = 0;
  MarkInventory expectedMarks{};
};

struct LineModel {
  int yBase = 0;
  int yTop = 0;  // Voronoi region top (midpoint between baselines)
  int yBot = 0;  // Voronoi region bottom
};

// Full pipeline using baselines + text
class BaselineSegmenter {
 public:
  struct Params {
    Params() {}
    int minCompArea = 2500;

    double wOverlapX = 10.0;  // overlap penalty in pixels (RTL)

    // Safety:
    int maxTokenSpan = 50;      // max #components per token group in DP (guard)
    double wGapBonus = 1.5;     // reward for gaps (log)
    double gapScalePx = 150.0;  // g0 in log(1 + gap/g0) at 4096 DPI (tune)

    // Connected paths in the layout can merge in the scan, so one missing
    // component is less suspicious than one additional component. Keep these
    // penalties independent of the expected word component count; otherwise
    // the DP prefers assigning an unavoidable mismatch to a larger word.
    double missingComponentPenalty = 0.75;
    double extraComponentPenalty = 1.25;
    // Enable only after the preliminary probabilities have been calibrated
    // against reviewed component examples. Keeping the feature observable but
    // weightless prevents an unvalidated classifier from moving boundaries.
    double smallLetterPenalty = 0.0;
    // Compare the preliminary body evidence with the expected number of body
    // paths. This keeps the line DP from assigning a body-looking component
    // to a neighboring word merely to satisfy component counts.
    double bodyEvidencePenalty = 1.0;
    // Compare the expected textual mark inventory with the available learned
    // or prototype class probabilities in each candidate word range.
    double markInventoryPenalty = 6.0;
    // Boundary-crossing final small yeh is the first targeted use case. Keep
    // it as a multiplier of the common inventory penalty so it can be tuned
    // independently after page-disjoint validation.
    double smallYehClassWeight = 2.0;
    std::string markModelPath;
    std::filesystem::path threeDotsAboveDetections;
    std::string threeDotsAboveDetectionSource;
    std::filesystem::path waqfDetections;
    std::string waqfDetectionSource;
    std::filesystem::path waqfTemplateRoot;
    std::filesystem::path threeDotsAboveTemplateRoot;
    std::filesystem::path attachedAlefHamzaAboveTemplateRoot;
    std::filesystem::path markTemplateRoot;
    std::filesystem::path recognizerCacheRoot;
    std::string recognizerSource;
    bool reportTimings = false;
    bool resetRecognizerCache = false;
    int boundaryCorridorAbove = 250;
    int boundaryCorridorBelow = 100;
    bool orderByBaselineCorridor = true;
    double otsuThreshold = -1.0;
    double minimumAttachedAlefHamzaAboveProbability = 0.82;
    int pageNumber = -1;
  };

  explicit BaselineSegmenter(Params p = Params()) : m_p(p) {}

  // Main entry: fills LineBox.words with WordBox (bbox + gap + expectedText).
  // Estimates and expected text lines are both ordered top-to-bottom.
  bool run(const cv::Mat& binInk255, const cv::Mat& srcBgrOrGray, const std::vector<BaselineEstimate>& lineEstimates, const PageLayout& pageLayout, PageModel& page);

 private:
  Params m_p;

  void extractComponents(const cv::Mat& bin, int minArea, PageModel& page);
  void ignoreSajdaBars(PageModel& page) const;
  void loadThreeDotsAboveDetections(std::vector<CC>& components) const;
  void applyDetectedWaqfLineAssignments(
      std::vector<CC>& components, const std::vector<LineModel>& lines,
      PageModel& page) const;
  static std::vector<LineModel> buildLineModels(const std::vector<BaselineEstimate>& lineEstimates, int imageH);

  static std::vector<WordInfo> makeExpectations(const LineLayout& lineLayout);

  static int chooseLineByOverlap(const CC& c, const std::vector<LineModel>& lines);

  static std::vector<int> sortIndicesRTL(const std::vector<CC>& comps,
                                         int lineIndex,
                                         const cv::Mat& labels, int baselineY,
                                         const Params& params);

  bool dpSegmentLine_DiscretePrevLeft(const std::vector<CC>& comps, const std::vector<int>& idx,
                                      const std::vector<WordInfo>& exp, const Params& p,
                                      const cv::Mat& labels, int baselineY,
                                      std::vector<std::pair<int, int>>& outRanges);

  static cv::Rect unionRange(const std::vector<CC>& comps,
                             const std::vector<int>& idx, int k, int i);
};
