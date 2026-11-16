# Contributing

Thank you for helping improve Mushaf Analyzer. Bug reports, documentation,
segmentation fixes, tests, and improvements to the analysis tools are welcome.

## Set up a development checkout

Install these system requirements first:

- [Git](https://git-scm.com/)
- [uv](https://docs.astral.sh/uv/)

The C++ analyzer also requires CMake 3.24 or newer, a C++20 compiler, OpenCV,
and SQLite3.

Create the Python environment and install its locked dependencies:

```bash
uv sync --locked
```

Build the analyzer with:

```bash
uv run cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
uv run cmake --build build
```

Run Python tools through `uv` so they use the locked environment:

```bash
uv run tools/review_segmentation.py --help
```

The graphical outline viewer uses Qt and is not installed by default:

```bash
uv sync --locked --group viewer
uv run tools/tracing/vector_fit_viewer.py
```

The optional mark-classifier training environment is larger and is not
installed by default:

```bash
uv sync --locked --group training
```

## Data and tests

`input/` is intentionally not versioned. The analyzer expects:

- `input/quran-data.sqlite`, prepared from Quranic text and the
  [KFGQPC V1 1405H layout](https://qul.tarteel.ai/resources/mushaf-layout/15)
  published by [QUL](https://qul.tarteel.ai/)
- high-resolution page scans under `input/oldmedina2/`

The high-resolution scans used for the published measurements cannot be
redistributed. The lower-resolution reference scans mentioned in the README
are not suitable for measurement.

After building, run the tests with:

```bash
ctest --test-dir build --output-on-failure
```

CMake adds the end-to-end segmentation tests only when both the database and
scans are present, so check the number of discovered tests before interpreting
the result. A public clone always includes the data-independent Bézier
projection regression test.

## Proposing a change

Open an issue before undertaking a large change. Keep pull requests focused,
explain how the result was verified, and do not commit scans, generated output,
corpus databases, virtual environments, or build directories. C++ changes
should follow `.clang-format`; Python changes should follow the surrounding
style.

See the [README](README.md) for the pipeline, file formats, and research
methodology.
