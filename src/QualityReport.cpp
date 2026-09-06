#include "QualityReport.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string_view>

#include "JsonIo.h"

namespace quality_detail {

struct RankedWord {
  const WordBox* word = nullptr;
  double score = 0.0;
  double gapRobustZ = 0.0;
  double widthPerLetterRobustZ = 0.0;
  std::vector<std::string> reasons;
};
struct RankingJson {
  int rank;
  std::string wordId, expectedText;
  double score;
  int lineIndex, wordIndex;
  std::string kind, wordImage, bodyImage;
  double foreignLineInkFraction, heightToLinePitch, gapRobustZ;
  double widthPerLetterRobustZ;
  std::vector<std::string> reasons;
};
struct ReviewWordJson {
  std::string wordId, status;
  std::vector<std::string> errors;
  std::string notes;
};
struct QualityReportJson {
  int schemaVersion, page, wordCount;
  double gapMedianPx, gapMadPx, widthPerLetterMedianPx;
  double widthPerLetterMadPx;
  std::vector<RankingJson> ranking;
};
struct ReviewTemplateJson {
  int schemaVersion, page;
  std::vector<std::string> allowedStatuses, suggestedErrors;
  std::vector<ReviewWordJson> words;
};

double median(std::vector<double> values) {
  if (values.empty()) return 0.0;
  const size_t middle = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + middle, values.end());
  double result = values[middle];
  if (values.size() % 2 == 0) {
    std::nth_element(values.begin(), values.begin() + middle - 1,
                     values.end());
    result = 0.5 * (result + values[middle - 1]);
  }
  return result;
}

double robustZ(double value, double center, double mad) {
  return mad > 1e-9 ? 0.67448975 * (value - center) / mad : 0.0;
}

int baseLetterCount(const std::string& text) {
  int count = 0;
  for (size_t i = 0; i < text.size();) {
    const unsigned char lead = static_cast<unsigned char>(text[i++]);
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
    for (int j = 0; j < continuationCount && i < text.size(); ++j)
      value = (value << 6) |
              (static_cast<unsigned char>(text[i++]) & 0x3F);
    const bool combining =
        (value >= 0x064B && value <= 0x065F) || value == 0x0670 ||
        (value >= 0x06D6 && value <= 0x06ED);
    if (!combining && value >= 0x0600 && value <= 0x06FF) ++count;
  }
  return std::max(1, count);
}

bool isVerseMarker(const std::string& text) {
  return text.starts_with("\xDB\x9D");  // U+06DD in UTF-8
}

void addReason(RankedWord& ranked, const std::string& reason, double penalty) {
  ranked.reasons.push_back(reason);
  ranked.score += penalty;
}

cv::Mat makeContactSheet(const std::vector<RankedWord>& ranked, int count) {
  constexpr int kColumns = 3;
  constexpr int kTileWidth = 720;
  constexpr int kTileHeight = 300;
  constexpr int kHeaderHeight = 48;
  const int actualCount = std::min(count, static_cast<int>(ranked.size()));
  const int rows = std::max(1, (actualCount + kColumns - 1) / kColumns);
  cv::Mat sheet(rows * kTileHeight, kColumns * kTileWidth, CV_8UC3,
                cv::Scalar(242, 242, 242));

  for (int index = 0; index < actualCount; ++index) {
    const int tileX = (index % kColumns) * kTileWidth;
    const int tileY = (index / kColumns) * kTileHeight;
    const RankedWord& item = ranked[index];
    cv::rectangle(sheet, cv::Rect(tileX, tileY, kTileWidth, kTileHeight),
                  cv::Scalar(180, 180, 180), 1);

    const std::string label =
        "#" + std::to_string(index + 1) + "  " +
        item.word->wordId + "  score=" +
        cv::format("%.2f", item.score);
    cv::putText(sheet, label, cv::Point(tileX + 12, tileY + 31),
                cv::FONT_HERSHEY_SIMPLEX, 0.72, cv::Scalar(20, 20, 20), 2,
                cv::LINE_AA);

    cv::Mat word =
        cv::imread(item.word->pngPath, cv::IMREAD_GRAYSCALE);
    if (word.empty()) continue;
    cv::Mat inverted;
    cv::bitwise_not(word, inverted);
    const double scale =
        std::min((kTileWidth - 24.0) / inverted.cols,
                 (kTileHeight - kHeaderHeight - 20.0) / inverted.rows);
    cv::Mat resized;
    cv::resize(inverted, resized, cv::Size(), scale, scale, cv::INTER_AREA);
    cv::Mat colored;
    cv::cvtColor(resized, colored, cv::COLOR_GRAY2BGR);
    const int imageX = tileX + (kTileWidth - colored.cols) / 2;
    const int imageY =
        tileY + kHeaderHeight +
        (kTileHeight - kHeaderHeight - colored.rows) / 2;
    colored.copyTo(sheet(cv::Rect(imageX, imageY, colored.cols, colored.rows)));
  }
  return sheet;
}

}  // namespace quality_detail
using namespace quality_detail;

void writeQualityReport(const std::vector<LineBox>& lines,
                        const std::filesystem::path& outDir, int pageNumber,
                        int contactSheetWordCount) {
  std::vector<double> gaps;
  std::vector<double> widthsPerLetter;
  for (const LineBox& line : lines) {
    for (size_t index = 0; index < line.words.size(); ++index) {
      const WordBox& word = line.words[index];
      if (isVerseMarker(word.expectedText)) continue;
      widthsPerLetter.push_back(
          word.spacingSpan.width / double(baseLetterCount(word.expectedText)));
      if (index + 1 < line.words.size() && word.spacingSpan.valid &&
          line.words[index + 1].spacingSpan.valid &&
          !isVerseMarker(line.words[index + 1].expectedText)) {
        gaps.push_back(word.spacingGapToNextPx);
      }
    }
  }

  const double gapMedian = median(gaps);
  std::vector<double> gapDeviations;
  for (double gap : gaps) gapDeviations.push_back(std::abs(gap - gapMedian));
  const double gapMad = median(gapDeviations);
  const double widthMedian = median(widthsPerLetter);
  std::vector<double> widthDeviations;
  for (double width : widthsPerLetter)
    widthDeviations.push_back(std::abs(width - widthMedian));
  const double widthMad = median(widthDeviations);

  std::vector<RankedWord> ranked;
  for (const LineBox& line : lines) {
    for (size_t index = 0; index < line.words.size(); ++index) {
      const WordBox& word = line.words[index];
      RankedWord item;
      item.word = &word;
      if (line.baseLineSearchBoundaryHit)
        addReason(item, "baseline_search_boundary", 5.0);
      if (line.baseLineConfidence < 0.20)
        addReason(item, "low_baseline_confidence",
                  (0.20 - line.baseLineConfidence) * 10.0);
      if (word.foreignLineInkFraction > 0.01)
        if (!isVerseMarker(word.expectedText))
          addReason(item, "foreign_line_body_ink",
                    word.foreignLineInkFraction * 20.0);
      if (word.heightToLinePitch > 1.10 && !isVerseMarker(word.expectedText))
        addReason(item, "unusually_tall_word",
                  (word.heightToLinePitch - 1.10) * 4.0);
      const double uncertainFraction =
          word.bodySegmentation.totalInkPixels > 0
              ? double(word.bodySegmentation.uncertainInkPixels) /
                    word.bodySegmentation.totalInkPixels
              : 0.0;
      if (uncertainFraction > 0.05)
        addReason(item, "uncertain_body_classification",
                  uncertainFraction * 2.0);
      if (!word.spacingSpan.valid)
        addReason(item, "invalid_spacing_span", 4.0);
      if (index + 1 < line.words.size() &&
          !isVerseMarker(word.expectedText) &&
          !isVerseMarker(line.words[index + 1].expectedText)) {
        item.gapRobustZ =
            robustZ(word.spacingGapToNextPx, gapMedian, gapMad);
        item.score +=
            0.08 * std::min(3.0, std::max(0.0, std::abs(item.gapRobustZ) - 1.5));
        if (std::abs(item.gapRobustZ) > 3.5)
          addReason(item, "spacing_gap_outlier",
                    std::min(3.0, (std::abs(item.gapRobustZ) - 3.5) * 0.5));
      }
      if (!isVerseMarker(word.expectedText)) {
        const double widthPerLetter =
            word.spacingSpan.width / double(baseLetterCount(word.expectedText));
        item.widthPerLetterRobustZ =
            robustZ(widthPerLetter, widthMedian, widthMad);
        item.score += 0.05 * std::min(
                                 3.0, std::max(0.0, std::abs(
                                                        item.widthPerLetterRobustZ) -
                                                        1.5));
        if (std::abs(item.widthPerLetterRobustZ) > 4.5)
          addReason(
              item, "width_per_letter_outlier",
              std::min(2.0,
                       (std::abs(item.widthPerLetterRobustZ) - 4.5) * 0.3));
      }
      ranked.push_back(item);
    }
  }
  std::stable_sort(ranked.begin(), ranked.end(),
                   [](const RankedWord& left, const RankedWord& right) {
                     return left.score > right.score;
                   });

  std::vector<RankingJson> entries;
  std::vector<ReviewWordJson> reviewEntries;
  for (size_t index = 0; index < ranked.size(); ++index) {
    const RankedWord& item = ranked[index];
    entries.push_back(
        {static_cast<int>(index + 1),
         item.word->wordId,
         item.word->expectedText,
         item.score,
         item.word->lineIndex,
         item.word->wordIndex,
         isVerseMarker(item.word->expectedText) ? "verse_marker" : "word",
         item.word->pngPath,
         (outDir / "subwordimgs" /
          (item.word->wordId + "_body.png"))
             .string(),
         item.word->foreignLineInkFraction,
         item.word->heightToLinePitch,
         item.gapRobustZ,
         item.widthPerLetterRobustZ,
         item.reasons});
    reviewEntries.push_back(
        {item.word->wordId, "unreviewed", {}, ""});
  }

  writeJsonFile(
      outDir / "quality-report.json",
      QualityReportJson{1, pageNumber, static_cast<int>(ranked.size()),
                        gapMedian, gapMad, widthMedian, widthMad,
                        std::move(entries)});
  writeJsonFile(
      outDir / "review-template.json",
      ReviewTemplateJson{
          1,
          pageNumber,
          {"unreviewed", "ok", "bad", "uncertain"},
          {"wrong_line", "wrong_word_boundary", "wrong_body_marks",
           "missing_ink", "foreign_ink", "wrong_spacing_span"},
          std::move(reviewEntries)});

  const cv::Mat sheet = makeContactSheet(ranked, contactSheetWordCount);
  cv::imwrite((outDir / "quality-contact-sheet.png").string(), sheet);
}
