#include "PreliminaryComponentClassifier.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/geometry/2d.hpp>

#include "ComponentGeometry.h"

namespace {
double clamp01(double value) {
  return std::clamp(value, 0.0, 1.0);
}

double linePitch(const PageModel& page, int lineIndex) {
  std::vector<double> values;
  if (lineIndex > 0)
    values.push_back(page.lines[lineIndex].baseLine -
                     page.lines[lineIndex - 1].baseLine);
  if (lineIndex + 1 < static_cast<int>(page.lines.size()))
    values.push_back(page.lines[lineIndex + 1].baseLine -
                     page.lines[lineIndex].baseLine);
  if (values.empty()) return 1.0;
  return std::max(1.0, values.front());
}

double componentSolidity(const PageModel& page, const CC& component) {
  const cv::Mat labels = page.labels(component.bb);
  cv::Mat mask = labels == component.label;
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);
  if (contours.empty()) return 1.0;
  const auto largest = std::max_element(
      contours.begin(), contours.end(),
      [](const auto& left, const auto& right) {
        return std::abs(cv::contourArea(left)) <
               std::abs(cv::contourArea(right));
      });
  std::vector<cv::Point> hull;
  cv::convexHull(*largest, hull);
  const double hullArea = std::abs(cv::contourArea(hull));
  return hullArea > 0.0 ? clamp01(component.area / hullArea) : 1.0;
}
}  // namespace

void classifyComponentsPreliminary(PageModel& page) {
  for (CC& component : page.comps) {
    if (component.line < 0 ||
        component.line >= static_cast<int>(page.lines.size()))
      continue;

    const double pitch = linePitch(page, component.line);
    const double areaRatio = component.area / (pitch * pitch);
    const double baseline = page.lines[component.line].baseLine;
    const double band = 0.045 * pitch;
    const bool bboxIntersectsBaseline =
        component.bb.y <= baseline + band &&
        component.bb.y + component.bb.height >= baseline - band;
    const bool hasBaselineInk = componentHasInkNearBaseline(
        page.labels, component.label, static_cast<int>(std::lround(baseline)),
        5, component.bb);
    const bool startsNearBaseline = componentStartsNearBaseline(
        component.bb, static_cast<int>(std::lround(baseline)));
    const double solidity = componentSolidity(page, component);
    const double aspect =
        std::max(component.bb.width, component.bb.height) /
        double(std::max(1, std::min(component.bb.width,
                                    component.bb.height)));

    // A compact mark can have a bbox that reaches the broad baseline band
    // without contributing any ink to the baseline itself. Treat that shape
    // as a mark candidate; keep the elongated/large fallback for body glyphs
    // whose scan does not happen to cross the narrow core band.
    const bool compactBaselineMark =
        bboxIntersectsBaseline && startsNearBaseline && areaRatio < 0.030 &&
        aspect <= 2.0;
    const bool compactFloatingMark =
        compactBaselineMark ||
        (bboxIntersectsBaseline && !hasBaselineInk && areaRatio < 0.030 &&
         aspect <= 2.0 && solidity >= 0.85);
    // Meeting the baseline band is not enough on its own. The band is 0.045
    // of the line pitch, so every mark written on or across the line falls in
    // it -- a kasra, a hamza below, the superscript waw of a pronoun -- and
    // each was scored 0.95 for it. That is the single most expensive mistake
    // this classifier makes, because the word DP then pays -log(0.05) = 3.0
    // to call the component a mark and would rather hand it a body slot: it
    // is how page 321's kasra took the ta-marbuta's place in بِٱلصَّلَوٰةِ, how
    // the hamza below إِ was read as the Dal of إِذَا, and how the ۥ of هُۥ
    // displaced the Heh in six words.
    //
    // Size settles it, once the compact shapes are considered on their own.
    // In qalam squared a kasra is 2.33 at the median and a hamza below 3.17,
    // against 4.42 for the smallest tenth of the letter hamza and 4.77 for a
    // Heh; the six pronoun waws measured 2.16 to 3.04. The bound sits in that
    // gap. It is confined to compact components on purpose -- an Alef is only
    // 4.23 qalam squared and would fall foul of any size rule, but at an
    // aspect near five it is an upright, and no mark is.
    constexpr double markMaximumAreaRatio = 0.0100;
    const bool tooSmallToBeABody = aspect <= 2.0 &&
                                   areaRatio < markMaximumAreaRatio;
    // And the same rule read the other way. Meeting the band is how a body is
    // normally recognized, so a letter whose ink stops short of it is called a
    // mark: the Alef of ٱلْإِنسَٰنُ on page 585 l05_w11 is 88x480 and ends 140 px
    // above a band half that wide, is scored 0.05, and its PAW then takes the
    // لإ blob beside it while لْإِ is left with no body at all. An upright is a
    // letter wherever its ink stops -- no mark class in the corpus is both
    // this slender and this tall, the nearest being small_meem at an aspect
    // of 2.34.
    const bool uprightStroke =
        aspect >= 3.0 && component.bb.height >= 0.25 * pitch;
    const double body =
        (!compactFloatingMark && !tooSmallToBeABody &&
         (bboxIntersectsBaseline || areaRatio >= 0.030 || uprightStroke))
            ? 0.95
            : 0.05;
    const double dotSize = clamp01((0.006 - areaRatio) / 0.004);
    const double dotShape = clamp01((solidity - 0.78) / 0.18) *
                            clamp01((1.8 - aspect) / 0.6);
    const double dot = (1.0 - body) * dotSize * dotShape;

    // Small Quranic letters are normally larger and more concave than dots,
    // and float away from the baseline. This is only candidate evidence; a
    // learned shape model can replace this term later.
    const double glyphSize =
        clamp01((areaRatio - 0.004) / 0.005) *
        clamp01((0.030 - areaRatio) / 0.015);
    const double concavity = clamp01((0.82 - solidity) / 0.30);
    const double smallLetter =
        (1.0 - body) * (1.0 - dot) * glyphSize * concavity;
    const double ordinaryMark =
        (1.0 - body) * (1.0 - dot) * (1.0 - 0.75 * smallLetter);

    component.bodyProbability = body;
    component.dotProbability = dot;
    component.smallLetterProbability = smallLetter;
    component.ordinaryMarkProbability = ordinaryMark;
    component.compactFloatingMark = compactFloatingMark;
    component.unknownProbability =
        clamp01(1.0 - std::max({body, dot, ordinaryMark, smallLetter}));
  }
}
