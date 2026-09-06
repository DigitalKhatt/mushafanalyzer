cmake_minimum_required(VERSION 3.16)
project(MushafAnalyzer LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  glaze
  GIT_REPOSITORY https://github.com/stephenberry/glaze.git
  GIT_TAG v7.5.0
  GIT_SHALLOW TRUE
  FIND_PACKAGE_ARGS 7.5.0
)
FetchContent_MakeAvailable(glaze)
find_package(OpenCV REQUIRED)
find_package(SQLite3 REQUIRED)
find_package(Threads REQUIRED)

set(LIB_FILES   
  src/PageProcessor.h src/PageProcessor.cpp
  src/SegmentationTypes.h
  src/ComponentGeometry.h
  src/BaselineSegmenter.h src/BaselineSegmenter.cpp
  src/BaselineEstimator.h src/BaselineEstimator.cpp
  src/SpacingGeometry.h src/SpacingGeometry.cpp
  src/QualityReport.h src/QualityReport.cpp
  src/DiagnosticPages.h src/DiagnosticPages.cpp
  src/JsonIo.h
  src/BatchValidation.h src/BatchValidation.cpp
  src/ComponentAssignmentCorrector.h src/ComponentAssignmentCorrector.cpp
  src/ComponentRecognizer.h src/ComponentRecognizer.cpp
  src/PreliminaryComponentClassifier.h src/PreliminaryComponentClassifier.cpp
  src/MarkTaxonomy.h src/MarkTaxonomy.cpp
  src/MarkClassifier.h src/MarkClassifier.cpp
  src/LayoutDb.h src/LayoutDb.cpp
  src/MakeExpectations.cpp
  src/SegmentBodyAndMarks.cpp
)

add_executable(mushaf_cli ${LIB_FILES}  src/main_cli.cpp)

target_include_directories(mushaf_cli PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(mushaf_cli PRIVATE glaze::glaze SQLite::SQLite3 Threads::Threads ${OpenCV_LIBS})
target_compile_options(
  mushaf_cli PRIVATE
  -Wall
  -Wextra
  -Wpedantic
  -Werror=unused-function
  -Werror=unused-parameter
  -Werror=unused-variable
)

if(BUILD_TESTING)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  set(MUSHAF_TEST_INPUT "${CMAKE_CURRENT_SOURCE_DIR}/input/oldmedina2")
  set(MUSHAF_TEST_DATABASE "${CMAKE_CURRENT_SOURCE_DIR}/input/quran-data.sqlite")
  if(EXISTS "${MUSHAF_TEST_INPUT}" AND EXISTS "${MUSHAF_TEST_DATABASE}")
    add_test(
      NAME word_paw_segmentation_page003
      COMMAND ${Python3_EXECUTABLE}
              "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_word_paw_segmentation.py"
              --cli $<TARGET_FILE:mushaf_cli>
              --input "${MUSHAF_TEST_INPUT}"
              --database "${MUSHAF_TEST_DATABASE}")
    set_tests_properties(word_paw_segmentation_page003 PROPERTIES
                         TIMEOUT 90)

    add_test(
      NAME parallel_page_jobs
      COMMAND ${Python3_EXECUTABLE}
              "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_parallel_jobs.py"
              --cli $<TARGET_FILE:mushaf_cli>
              --input "${MUSHAF_TEST_INPUT}"
              --database "${MUSHAF_TEST_DATABASE}")
    set_tests_properties(parallel_page_jobs PROPERTIES TIMEOUT 60)

    add_test(
      NAME sajda_bar_page251
      COMMAND ${Python3_EXECUTABLE}
              "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_sajda_bar.py"
              --cli $<TARGET_FILE:mushaf_cli>
              --input "${MUSHAF_TEST_INPUT}"
              --database "${MUSHAF_TEST_DATABASE}")
    set_tests_properties(sajda_bar_page251 PROPERTIES TIMEOUT 60)

    add_test(
      NAME sajda_glyph_page365
      COMMAND ${Python3_EXECUTABLE}
              "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_sajda_glyph.py"
              --cli $<TARGET_FILE:mushaf_cli>
              --input "${MUSHAF_TEST_INPUT}"
              --database "${MUSHAF_TEST_DATABASE}")
    set_tests_properties(sajda_glyph_page365 PROPERTIES TIMEOUT 90)

    add_test(
      NAME page192_word_boundary
      COMMAND ${Python3_EXECUTABLE}
              "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_page192_word_boundary.py"
              --cli $<TARGET_FILE:mushaf_cli>
              --input "${MUSHAF_TEST_INPUT}"
              --database "${MUSHAF_TEST_DATABASE}")
    set_tests_properties(page192_word_boundary PROPERTIES TIMEOUT 60)

    add_test(
      NAME page565_connected_alef
      COMMAND ${Python3_EXECUTABLE}
              "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_page565_connected_alef.py"
              --cli $<TARGET_FILE:mushaf_cli>
              --input "${MUSHAF_TEST_INPUT}"
              --database "${MUSHAF_TEST_DATABASE}")
    set_tests_properties(page565_connected_alef PROPERTIES TIMEOUT 60)

    add_test(
      NAME page030_connected_alef
      COMMAND ${Python3_EXECUTABLE}
              "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_page030_connected_alef.py"
              --cli $<TARGET_FILE:mushaf_cli>
              --input "${MUSHAF_TEST_INPUT}"
              --database "${MUSHAF_TEST_DATABASE}")
    set_tests_properties(page030_connected_alef PROPERTIES TIMEOUT 60)

    add_test(
      NAME paw_body_inversion
      COMMAND ${Python3_EXECUTABLE}
              "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_paw_body_inversion.py"
              --cli $<TARGET_FILE:mushaf_cli>
              --input "${MUSHAF_TEST_INPUT}"
              --database "${MUSHAF_TEST_DATABASE}")
    set_tests_properties(paw_body_inversion PROPERTIES TIMEOUT 60)

    add_test(
      NAME debug_output_opt_in
      COMMAND ${Python3_EXECUTABLE}
              "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_debug_output.py"
              --cli $<TARGET_FILE:mushaf_cli>
              --input "${MUSHAF_TEST_INPUT}"
              --database "${MUSHAF_TEST_DATABASE}")
    set_tests_properties(debug_output_opt_in PROPERTIES TIMEOUT 90)

    set(MUSHAF_TEST_OUTPUT
        "${CMAKE_CURRENT_SOURCE_DIR}/output/oldmedina2-analysis")
    if(EXISTS "${MUSHAF_TEST_OUTPUT}")
      add_test(
        NAME segmentation_viewer_data
        COMMAND ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_review_segmentation.py"
                --root "${MUSHAF_TEST_OUTPUT}")
      set_tests_properties(segmentation_viewer_data PROPERTIES TIMEOUT 150)
    endif()
  endif()
endif()
