#pragma once
#include <filesystem>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#if __has_include(<opencv2/geometry/2d.hpp>)
#include <opencv2/geometry/2d.hpp>
#endif

#include "BaselineEstimator.h"
#include "LayoutDb.h"
#include "SegmentationTypes.h"

// Connected-component labels for one word, in the word's local coordinates.
// The same result is shared by component evidence, PAW segmentation, and
// mask export so those stages do not repeat the identical labeling pass.
struct WordConnectedComponents {
  cv::Mat labels;     // CV_32S
  cv::Mat stats;      // CV_32S
  cv::Mat centroids;  // CV_64F
};

class PageProcessor {
 public:
  struct Params {
    Params() {}
    // preprocessing
    int blurK = 0;
    int morphK = 15;
    bool doDeskew = false;
    bool inputIsPrebinarized = false;
    bool deblendConnectedMarks = true;
    bool regroupSplitDots = true;
    // A single global Otsu cut leaves the lighter interior of a stroke below
    // the threshold, so the mask ends up peppered with enclosed background
    // that is really ink. Those holes each add a cycle to the medial graph
    // and spurious events to the letter segmenter, so they are filled unless
    // the scan says the hole is paper.
    bool repairUnderInkedHoles = true;
    // Where a hole's brightness sits between the page's ink level (0) and its
    // paper level (1). Measured over nine pages: enclosed background that is
    // certainly a dropout reaches 0.70, and a counter round enough that the
    // pen must have gone around it starts at 0.79.
    double holeRepairPaperness = 0.72;

    // line segmentation
    int minCompArea = 2500;
    int spacingCorridorAbove = 250;
    int spacingCorridorBelow = 100;
    int pageNumber = -1;

    // export
    int padPx = 6;
    bool exportComponentDataset = false;
    bool analysisOnly = false;
    bool debugImages = false;
    bool debugReconstructed = false;
    bool reportTimings = false;
    std::filesystem::path segmentationExceptions;
    std::filesystem::path waqfDetections;
    std::string waqfDetectionSource;
    std::filesystem::path threeDotsAboveDetections;
    std::string threeDotsAboveDetectionSource;
    std::filesystem::path waqfTemplateRoot;
    std::filesystem::path threeDotsAboveTemplateRoot;
    std::filesystem::path attachedAlefHamzaAboveTemplateRoot;
    std::filesystem::path markTemplateRoot;
    std::filesystem::path recognizerCacheRoot;
    std::string recognizerSource;
    bool resetRecognizerCache = false;
    std::string markModelPath;
    double markInventoryPenalty = 6.0;
    double markTemplatePenalty = 1.5;
    double smallYehClassWeight = 2.0;
    bool orderByBaselineCorridor = true;
    // Let the word DP give one component to two adjacent PAWs where the scan
    // has joined them. Off, a merged pair still costs the word a body and the
    // assignment shifts along; the flag exists so that behaviour can be
    // reproduced, not because it is better.
    bool shareBodyBetweenTouchingPaws = true;
  };

  explicit PageProcessor(Params p = Params());

  bool loadImage(const std::filesystem::path& path);
  const cv::Mat& original() const { return m_src; }
  const cv::Mat& binary() const { return m_bin; }
  const PageModel& pageModel() const { return m_page; }
  const std::vector<LineBox>& lines() const { return m_page.lines; }
  bool processWithText(const std::filesystem::path& outDir, const PageLayout& pageLayout);
  MarkSegmentationResult segmentBodyAndMarks(
      const cv::Mat& wordMask,  // CV_8U, 0/255
      int baselineY,            // baseline in wordMask coordinates
      double dotCompactness = 0.7, double dotSolidity = 0.9,
      const std::vector<WordPawExpectation>& pawExpectations = {},
      const std::vector<WordComponentEvidence>& componentEvidence = {},
      double markTemplatePenalty = 1.5,
      bool firstPawConnectedToPreviousWord = false,
      const WordConnectedComponents* connectedComponents = nullptr,
      double linePitch = 0.0);

 private:
  Params m_p;
  cv::Mat m_src;  // BGR or gray
  cv::Mat m_gray;
  cv::Mat m_bin;  // 0/255, text=255
  std::filesystem::path m_sourcePath;
  cv::Rect m_cropRect;
  double m_deskewDegrees = 0.0;
  double m_otsuThreshold = -1.0;

  PageModel m_page;

  void preprocess();
  int removeOuterFrameComponents();
  int repairUnderInkedHoles();
  void deskewIfNeeded();
  int joinSplitDotClusters(
      const std::vector<BaselineEstimate>& baselineEstimates);
  int deblendConnectedUpperMarks(
      const std::vector<BaselineEstimate>& baselineEstimates);

  void exportWords(const std::filesystem::path& outDir);
};
