#include <sqlite3.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "BatchValidation.h"
#include "LayoutDb.h"
#include "PageProcessor.h"

namespace fs = std::filesystem;

namespace {
struct Options {
  fs::path input;
  fs::path output = "output/wordseg";
  fs::path database = "input/quran-data.sqlite";
  int page = 3;
  std::vector<int> pages;
  bool batch = false;
  int jobs = 1;
  bool force = false;
  int blurK = 3;
  int morphK = 15;
  int minCompArea = 2500;
  bool deskew = false;
  bool prebinarized = false;
  bool deblendConnectedMarks = true;
  bool regroupSplitDots = true;
  bool shareBodyBetweenTouchingPaws = true;
  bool repairUnderInkedHoles = true;
  double holeRepairPaperness = 0.72;
  bool exportComponentDataset = false;
  bool analysisOnly = false;
  bool debugImages = false;
  bool debugReconstructed = false;
  bool reportTimings = false;
  fs::path segmentationExceptions;
  fs::path waqfDetections;
  std::string waqfDetectionSource;
  fs::path threeDotsAboveDetections;
  std::string threeDotsAboveDetectionSource;
  fs::path markModel;
  fs::path waqfTemplateRoot;
  fs::path threeDotsAboveTemplateRoot;
  fs::path attachedAlefHamzaAboveTemplateRoot;
  fs::path markTemplateRoot;
  fs::path recognizerCache = "output/component-recognizer-cache";
  std::string recognizerSource;
  bool resetRecognizerCache = false;
  double markInventoryPenalty = 6.0;
  double markTemplatePenalty = 1.5;
  double smallYehClassWeight = 2.0;
  bool orderByBaselineCorridor = true;
};

void usage(std::ostream& stream) {
  stream << "Usage: mushaf_cli --input PATH [options]\n"
            "  -i, --input PATH       Page image or numbered-image directory\n"
            "  -o, --out DIR          Output root (default: output/wordseg)\n"
            "  -p, --page N           One-based page number (default: 3)\n"
            "  --pages LIST           Batch pages, e.g. 3,10,50-55 or all\n"
            "  --jobs N               Maximum parallel page jobs (default: 1)\n"
            "  --force                Reprocess pages that already have words.json\n"
            "  --database PATH        quran-data.sqlite path\n"
            "  --blurK N              Gaussian blur kernel size\n"
            "  --morphK N             Morphology-open kernel size\n"
            "  --minCompArea N        Minimum retained component area\n"
            "  --deskew               Enable deskew\n"
            "  --prebinarized         Input is cropped white-ink binary page\n"
            "  --no-mark-deblend      Disable local splitting of connected upper marks\n"
            "  --no-mark-regroup      Disable local joining of split dot clusters\n"
            "  --no-shared-paw-body   Do not let two touching PAWs share one body\n"
            "  --no-hole-repair       Keep enclosed background the scan says is ink\n"
            "  --hole-repair-paperness N  Brightness a hole needs to count as paper (0..1)\n"
            "  --export-component-dataset  Write candidate masks and context crops\n"
            "  --analysis-only        Write JSON reports but skip review images\n"
            "  --debug-images        Write all debug images and diagnostic reports\n"
            "  --no-debug-reconstructed  With --debug-images, omit debug_reconstructed.png\n"
            "  --timings              Print pipeline stage durations\n"
            "  --segmentation-exceptions PATH  Apply reviewed component moves\n"
            "  --waqf-detections PATH  Apply high-confidence detected waqf groups\n"
            "  --waqf-source NAME      Detection scan set, for example old or new\n"
            "  --three-dots-above-detections PATH  Detected scan forms for ث/ش dots\n"
            "  --three-dots-above-source NAME  Detection scan set name\n"
            "  --mark-model PATH      ONNX mark model used by word segmentation\n"
            "  --mark-inventory-penalty N  Weight of expected mark-class evidence\n"
            "  --mark-template-root PATH  Unified reviewed mark template directory\n"
            "  --mark-template-penalty N  Weight of template class evidence\n"
            "  --small-yeh-class-weight N  Extra weight for final small yeh\n"
            "  --bbox-right-order     Use complete bounding-box right edges (A/B control)\n"
            "  --waqf-template-root PATH  Deprecated waqf-template compatibility root\n"
            "  --three-dots-above-template-root PATH  Deprecated three-dot compatibility root\n"
            "  --attached-alef-hamza-template-root PATH  Deprecated attached alef+hamza compatibility root\n"
            "  --recognizer-cache PATH  Per-page component recognition cache\n"
            "  --recognizer-source NAME  Scan-set name used in cache keys\n"
            "  --reset-recognizer-cache  Reclassify and rewrite page caches\n"
            "  -h, --help             Show this help\n"
            "  --version              Show version\n";
}

int parseInt(std::string_view value, std::string_view option) {
  int result = 0;
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), result);
  if (error != std::errc{} || end != value.data() + value.size())
    throw std::runtime_error("Invalid integer for " + std::string(option));
  return result;
}

double parseDouble(std::string_view value, std::string_view option) {
  double result = 0.0;
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), result);
  if (error != std::errc{} || end != value.data() + value.size() ||
      !std::isfinite(result) || result < 0.0)
    throw std::runtime_error("Invalid non-negative number for " +
                             std::string(option));
  return result;
}

std::vector<int> parsePages(std::string_view value) {
  std::vector<int> pages;
  if (value == "all") {
    for (int page = 1; page <= 604; ++page) pages.push_back(page);
    return pages;
  }
  size_t start = 0;
  while (start < value.size()) {
    const size_t comma = value.find(',', start);
    const std::string_view item =
        value.substr(start, comma == std::string_view::npos
                                ? value.size() - start
                                : comma - start);
    const size_t dash = item.find('-');
    if (dash == std::string_view::npos) {
      pages.push_back(parseInt(item, "--pages"));
    } else {
      const int first = parseInt(item.substr(0, dash), "--pages");
      const int last = parseInt(item.substr(dash + 1), "--pages");
      if (last < first) throw std::runtime_error("Descending --pages range");
      for (int page = first; page <= last; ++page) pages.push_back(page);
    }
    if (comma == std::string_view::npos) break;
    start = comma + 1;
  }
  std::sort(pages.begin(), pages.end());
  pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
  return pages;
}

Options parseOptions(int argc, char** argv) {
  Options options;
  auto value = [&](int& index, std::string_view option) -> std::string {
    if (++index >= argc)
      throw std::runtime_error("Missing value for " + std::string(option));
    return argv[index];
  };
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      usage(std::cout);
      std::exit(0);
    } else if (arg == "--version") {
      std::cout << "mushaf_cli 0.3\n";
      std::exit(0);
    } else if (arg == "-i" || arg == "--input") {
      options.input = value(i, arg);
    } else if (arg == "-o" || arg == "--out") {
      options.output = value(i, arg);
    } else if (arg == "-p" || arg == "--page") {
      options.page = parseInt(value(i, arg), arg);
    } else if (arg == "--pages") {
      options.pages = parsePages(value(i, arg));
      options.batch = true;
    } else if (arg == "--jobs") {
      options.jobs = parseInt(value(i, arg), arg);
      if (options.jobs < 1)
        throw std::runtime_error("--jobs must be at least 1");
    } else if (arg == "--force") {
      options.force = true;
    } else if (arg == "--database") {
      options.database = value(i, arg);
    } else if (arg == "--blurK") {
      options.blurK = parseInt(value(i, arg), arg);
    } else if (arg == "--morphK") {
      options.morphK = parseInt(value(i, arg), arg);
    } else if (arg == "--minCompArea") {
      options.minCompArea = parseInt(value(i, arg), arg);
    } else if (arg == "--deskew") {
      options.deskew = true;
    } else if (arg == "--prebinarized") {
      options.prebinarized = true;
    } else if (arg == "--no-mark-deblend") {
      options.deblendConnectedMarks = false;
    } else if (arg == "--no-mark-regroup") {
      options.regroupSplitDots = false;
    } else if (arg == "--no-shared-paw-body") {
      options.shareBodyBetweenTouchingPaws = false;
    } else if (arg == "--no-hole-repair") {
      options.repairUnderInkedHoles = false;
    } else if (arg == "--hole-repair-paperness") {
      options.holeRepairPaperness = parseDouble(value(i, arg), arg);
      if (options.holeRepairPaperness > 1.0)
        throw std::runtime_error("--hole-repair-paperness must be <= 1");
    } else if (arg == "--export-component-dataset") {
      options.exportComponentDataset = true;
    } else if (arg == "--analysis-only") {
      options.analysisOnly = true;
    } else if (arg == "--debug-images") {
      options.debugImages = true;
      options.debugReconstructed = true;
    } else if (arg == "--no-debug-reconstructed") {
      options.debugReconstructed = false;
    } else if (arg == "--timings") {
      options.reportTimings = true;
    } else if (arg == "--segmentation-exceptions") {
      options.segmentationExceptions = value(i, arg);
    } else if (arg == "--waqf-detections") {
      options.waqfDetections = value(i, arg);
    } else if (arg == "--waqf-source") {
      options.waqfDetectionSource = value(i, arg);
    } else if (arg == "--three-dots-above-detections") {
      options.threeDotsAboveDetections = value(i, arg);
    } else if (arg == "--three-dots-above-source") {
      options.threeDotsAboveDetectionSource = value(i, arg);
    } else if (arg == "--mark-model") {
      options.markModel = value(i, arg);
    } else if (arg == "--mark-inventory-penalty") {
      options.markInventoryPenalty = parseDouble(value(i, arg), arg);
    } else if (arg == "--mark-template-root") {
      options.markTemplateRoot = value(i, arg);
    } else if (arg == "--mark-template-penalty") {
      options.markTemplatePenalty = parseDouble(value(i, arg), arg);
    } else if (arg == "--small-yeh-class-weight") {
      options.smallYehClassWeight = parseDouble(value(i, arg), arg);
    } else if (arg == "--bbox-right-order") {
      options.orderByBaselineCorridor = false;
    } else if (arg == "--waqf-template-root") {
      options.waqfTemplateRoot = value(i, arg);
    } else if (arg == "--three-dots-above-template-root") {
      options.threeDotsAboveTemplateRoot = value(i, arg);
    } else if (arg == "--attached-alef-hamza-template-root") {
      options.attachedAlefHamzaAboveTemplateRoot = value(i, arg);
    } else if (arg == "--recognizer-cache") {
      options.recognizerCache = value(i, arg);
    } else if (arg == "--recognizer-source") {
      options.recognizerSource = value(i, arg);
    } else if (arg == "--reset-recognizer-cache") {
      options.resetRecognizerCache = true;
    } else {
      throw std::runtime_error("Unknown option: " + std::string(arg));
    }
  }
  if (options.input.empty()) throw std::runtime_error("Missing --input");
  return options;
}

std::string pageKey(int page) {
  std::ostringstream stream;
  stream << std::setw(3) << std::setfill('0') << page;
  return stream.str();
}

fs::path resolvePageImage(const fs::path& input, int page) {
  if (fs::is_regular_file(input)) return fs::absolute(input);
  if (!fs::is_directory(input)) return {};
  for (const std::string& stem : {std::to_string(page), pageKey(page)})
    for (const char* extension : {".png", ".jpg", ".jpeg"}) {
      const fs::path candidate = input / (stem + extension);
      if (fs::exists(candidate)) return candidate;
    }
  return {};
}

bool isDebugArtifact(const fs::path& path) {
  const std::string filename = path.filename().string();
  return (filename.starts_with("debug_") && path.extension() == ".png") ||
         filename == "component-insertion-gaps.json" ||
         filename == "reconstruction-report.json";
}

void removeStaleDebugArtifacts(const fs::path& output) {
  if (!fs::is_directory(output)) return;
  for (const fs::directory_entry& entry : fs::directory_iterator(output)) {
    if (!entry.is_regular_file()) continue;
    if (isDebugArtifact(entry.path()))
      fs::remove(entry.path());
  }
}

void removeStaleBatchDebugArtifacts(const fs::path& output) {
  std::error_code error;
  fs::remove(output / "validation-contact-sheet.png", error);
}

cv::Mat drawOverlayBGR(const cv::Mat& source,
                       const std::vector<LineBox>& lines) {
  cv::Mat out;
  if (source.channels() == 1)
    cv::cvtColor(source, out, cv::COLOR_GRAY2BGR);
  else
    out = source.clone();
  for (const LineBox& line : lines) {
    cv::rectangle(out, line.rect, cv::Scalar(255, 150, 0), 2);
    const int baseline = std::clamp(static_cast<int>(std::lround(line.baseLine)),
                                    0, out.rows - 1);
    cv::line(out, {0, baseline}, {out.cols - 1, baseline},
             cv::Scalar(0, 200, 0), 2);
    for (const WordBox& word : line.words) {
      cv::rectangle(out, word.rect, cv::Scalar(0, 0, 255), 2);
      cv::putText(out,
                  std::to_string(word.wordIndex) + " g=" +
                      std::to_string(word.spacingGapToNextPx),
                  {word.rect.x, std::max(0, word.rect.y - 5)},
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
    }
  }
  return out;
}

PageValidation processPage(
    const Options& options, int pageNumber,
    const std::map<int, PageLayout>& pages) {
  const fs::path pageOutput =
      options.output / ("page" + pageKey(pageNumber));
  const fs::path imagePath = resolvePageImage(options.input, pageNumber);
  const fs::path wordsPath = pageOutput / "words.json";
  std::error_code fileError;
  if (!options.force && fs::is_regular_file(wordsPath, fileError) &&
      fs::file_size(wordsPath, fileError) > 0 && !fileError) {
    return {pageNumber, "skipped", imagePath.string(), pageOutput.string(),
            "existing words.json"};
  }
  try {
    if (imagePath.empty())
      throw std::runtime_error("Could not find page image");
    if (!pages.contains(pageNumber))
      throw std::runtime_error("No database layout for page");
    fs::create_directories(pageOutput);
    removeStaleDebugArtifacts(pageOutput);
    PageProcessor::Params params;
    params.blurK = options.blurK;
    params.morphK = options.morphK;
    params.minCompArea = options.minCompArea;
    params.doDeskew = options.deskew;
    params.inputIsPrebinarized = options.prebinarized;
    params.deblendConnectedMarks = options.deblendConnectedMarks;
    params.regroupSplitDots = options.regroupSplitDots;
    params.shareBodyBetweenTouchingPaws = options.shareBodyBetweenTouchingPaws;
    params.repairUnderInkedHoles = options.repairUnderInkedHoles;
    params.holeRepairPaperness = options.holeRepairPaperness;
    params.pageNumber = pageNumber;
    params.exportComponentDataset = options.exportComponentDataset;
    params.analysisOnly = options.analysisOnly;
    params.debugImages = options.debugImages;
    params.debugReconstructed = options.debugReconstructed;
    params.reportTimings = options.reportTimings;
    params.segmentationExceptions = options.segmentationExceptions;
    params.waqfDetections = options.waqfDetections;
    params.waqfDetectionSource = options.waqfDetectionSource;
    params.threeDotsAboveDetections = options.threeDotsAboveDetections;
    params.threeDotsAboveDetectionSource =
        options.threeDotsAboveDetectionSource;
    params.markModelPath = options.markModel.string();
    params.markInventoryPenalty = options.markInventoryPenalty;
    params.markTemplatePenalty = options.markTemplatePenalty;
    params.smallYehClassWeight = options.smallYehClassWeight;
    params.orderByBaselineCorridor = options.orderByBaselineCorridor;
    params.waqfTemplateRoot = options.waqfTemplateRoot;
    params.threeDotsAboveTemplateRoot = options.threeDotsAboveTemplateRoot;
    params.attachedAlefHamzaAboveTemplateRoot =
        options.attachedAlefHamzaAboveTemplateRoot;
    params.markTemplateRoot = options.markTemplateRoot;
    params.recognizerCacheRoot = options.recognizerCache;
    params.recognizerSource =
        options.recognizerSource.empty()
            ? options.input.filename().string()
            : options.recognizerSource;
    params.resetRecognizerCache = options.resetRecognizerCache;
    PageProcessor processor(params);
    if (!processor.loadImage(imagePath))
      throw std::runtime_error("Failed to load page image");
    if (!processor.processWithText(pageOutput, pages.at(pageNumber)))
      throw std::runtime_error("Page processing failed");
    if (!options.analysisOnly && options.debugImages) {
      cv::imwrite((pageOutput / "debug_bin.png").string(),
                  processor.binary());
      cv::imwrite((pageOutput / "debug_overlay.png").string(),
                  drawOverlayBGR(processor.original(), processor.lines()));
    }

    return summarizePageValidation(pageNumber, imagePath, pageOutput,
                                   pages.at(pageNumber), processor);
  } catch (const std::exception& error) {
    return {pageNumber, "failed", imagePath.string(), pageOutput.string(),
            error.what()};
  }
}

void printPageValidation(const PageValidation& summary) {
  if (summary.status == "skipped") {
    std::cout << "Page " << summary.page
              << " skipped (existing words.json)\n";
    return;
  }
  if (summary.status == "failed") {
    std::cerr << "Page " << summary.page << " failed: " << summary.error
              << '\n';
    return;
  }
  std::cout << "Page " << summary.page
            << " Lines: " << summary.detectedLineCount
            << " Words: " << summary.detectedWordCount
            << " Invalid spacing spans: " << summary.invalidSpacingSpans
            << " Negative spacing gaps: " << summary.negativeSpacingGaps
            << " Score: " << summary.anomalyScore << '\n';
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parseOptions(argc, argv);
    const std::vector<int> requestedPages =
        options.batch ? options.pages : std::vector<int>{options.page};
    if (requestedPages.empty())
      throw std::runtime_error("--pages did not select any pages");
    for (int page : requestedPages)
      if (page < 1 || page > 604)
        throw std::runtime_error("Page numbers must be in [1, 604]");
    if (options.batch && fs::is_regular_file(options.input) &&
        requestedPages.size() > 1)
      throw std::runtime_error(
          "A single input image cannot be used for multiple pages");

    sqlite3* rawDatabase = nullptr;
    if (sqlite3_open_v2(options.database.string().c_str(), &rawDatabase,
                        SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
      const std::string error =
          rawDatabase ? sqlite3_errmsg(rawDatabase) : "unknown error";
      if (rawDatabase) sqlite3_close(rawDatabase);
      throw std::runtime_error("Could not open database: " + error);
    }
    std::unique_ptr<sqlite3, decltype(&sqlite3_close)> database(rawDatabase,
                                                                sqlite3_close);
    std::map<int, PageLayout> pages;
    loadLayout(database.get(), pages);

    fs::create_directories(options.output);
    if (options.batch && !options.debugImages)
      removeStaleBatchDebugArtifacts(options.output);
    std::vector<PageValidation> validation(requestedPages.size());
    std::mutex progressMutex;
    auto reportPageFinished = [&](const PageValidation& summary) {
      std::lock_guard<std::mutex> lock(progressMutex);
      printPageValidation(summary);
    };
    if (options.jobs == 1 || requestedPages.size() == 1) {
      for (std::size_t index = 0; index < requestedPages.size(); ++index) {
        PageValidation result =
            processPage(options, requestedPages[index], pages);
        reportPageFinished(result);
        validation[index] = std::move(result);
      }
    } else {
      std::atomic<std::size_t> nextPage{0};
      const std::size_t workerCount = std::min<std::size_t>(
          static_cast<std::size_t>(options.jobs), requestedPages.size());
      std::vector<std::thread> workers;
      workers.reserve(workerCount);
      for (std::size_t worker = 0; worker < workerCount; ++worker) {
        workers.emplace_back([&] {
          while (true) {
            const std::size_t index =
                nextPage.fetch_add(1, std::memory_order_relaxed);
            if (index >= requestedPages.size()) return;
            PageValidation result =
                processPage(options, requestedPages[index], pages);
            reportPageFinished(result);
            validation[index] = std::move(result);
          }
        });
      }
      for (std::thread& worker : workers) worker.join();
    }

    int failures = 0;
    for (const PageValidation& summary : validation)
      if (summary.status == "failed") ++failures;
    if (options.batch) {
      writeBatchValidationReport(options.output, validation);
      if (!options.analysisOnly && options.debugImages)
        writePageValidationContactSheet(options.output, validation);
      std::cout << "Validation report: "
                << (options.output / "validation-report.json") << '\n';
    }
    return failures == 0 ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "mushaf_cli: " << error.what() << '\n';
    return 2;
  }
}
