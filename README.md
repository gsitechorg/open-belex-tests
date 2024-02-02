# Belex Tests (Open Source)

Unit and integration tests for the Bit-Engine Language of Expressions (Belex) of
GSI's APU.

Version of 02-Feb-2024

# Initialization

At the moment, only conda environments are supported. The following shows how
to set up yours:

```bash
# let $WORKSPACE be the parent working directory of open-belex-libs
cd "$WORKSPACE"

# Clone the open-belex repositories (unless you know what you are doing,
# please choose the same branch for all repositories):
# 1. "master" -> clone latest release code
# 2. "develop" -> clone latest development code
DEFAULT_BRANCH="master"
BELEX_BRANCH="$DEFAULT_BRANCH"
BELEX_LIBS_BRANCH="$DEFAULT_BRANCH"
BELEX_TESTS_BRANCH="$DEFAULT_BRANCH"
BARYON_BRANCH="$DEFAULT_BRANCH"

# Baryon may be built in either "Debug" or "Release" mode. For development,
# you should use "Debug".
BARYON_BUILD_TYPE="Debug"

git clone --branch "$BELEX_BRANCH" \
    https://github.com/gsitechorg/open-belex.git
git clone --branch "$BELEX_LIBS_BRANCH" \
    https://github.com/gsitechorg/open-belex-libs.git
git clone --branch "$BELEX_TESTS_BRANCH" \
    https://github.com/gsitechorg/open-belex-tests.git
git clone --branch "$BARYON_BRANCH" \
    https://github.com/gsitechorg/fermion.git

cd open-belex-tests

# Create the conda environment
mamba env create --force -f environment.yml
conda activate open-belex-test

# Tell pip to use the cloned versions of open-belex, open-belex-libs,
# and open-belex-tests
pip install -e ../open-belex -e ../open-belex-libs -e .

# Install the desired version of Baryon
pushd ../fermion
mamba env create --force -f environment.yml
conda activate baryon
./setup.sh --build-type "$BARYON_BUILD_TYPE" \
           --prefix ~/mambaforge/envs/open-belex-test \
           --install
conda deactivate
popd
```

# Testing

The tests utilize a combination of pytest, hypothesis, and a custom test harness
composed with them. Many of the Python tests are designed such that they can be
represented as self-contained native applications (snippets). The snippets may
be run on all three supported targets (python, baryon, and hw).

## Python

To run the Python tests, execute the following:

```bash
conda activate open-belex-test  # if you have not activated it
pytest
```

The test suite utilizes `pytest` and supports [all the things you may do with
it](https://docs.pytest.org/en/7.1.x/how-to/usage.html), including running a
specific test or tests:

```bash
conda activate open-belex-test  # if you have not activated it
pytest tests/test_belex_library.py::test_write_to_marked \
    tests/test_belex_exercises.py::test_exercise_6
```

If you want the tests to stop as soon as a failure is encountered, pass the `-x`
flag:

```bash
conda activate open-belex-test  # if you have not activated it
pytest -x
```

Should you prefer to run them in parallel (highly recommended), make sure you've
installed `pytest-xdist` as described in the `Initialization` section, and
execute the following:

```bash
conda activate open-belex-test  # if you have not activated it
pytest -n auto  # auto implies all cores, you may specify a quantity in its place
```

`pytest-xdist` also supports breaking on the first error with `-x`.

## Baryon

Baryon is our C simulator for the APU and related libraries (e.g. sys-apu and
libs-gvml). The Baryon suite may be run without access to an actual APU, and
runs on Linux and Mac OS X.

```bash
conda activate open-belex-test  # if you have not activated it
belex-test -t baryon"
```

To make the interface familiar, you may run specific tests using `pytest`
syntax:

```bash
conda activate open-belex-test  # if you have not activated it
belex-test -t baryon \
    tests/test_belex_library.py::test_write_to_marked \
    tests/test_belex_exercises.py::test_exercise_6
```

## Additional Options

To learn more about the capabilities of `belex-test`, run `belex-test --help`:

```
$ belex-test --help
Executes BELEX tests on hardware or baryon with a pytest-like interface.

Options:
  -h|--help                          Prints this help text.
  -O|--optimization-level 0|1|2|3|4  Specifies the level of optimizations and
                                     transformative features to enable for
                                     codegen. (Default: 2)
  -f|--enable-optimization <NAME>    [Optional] Specifies an optimization over
                                     the generated code.
  -d|--belex-dir <PATH>              [Optional] Path to the folder containing the
                                     BELEX files. If neither a --belex-dir nor
                                     BELEX_FILE is provided, an attempt will be
                                     made to guess the --belex-dir path.
  -o|--output-dir <PATH>             [Optional] Where to place the generated
                                     examples. If no --output-dir is provided an
                                     attempt will be made to guess it.
  -C|--build-dir <PATH>              [Optional] Path to the meson build directory.
  -t|--build-type baryon             [Optional] Target platform of the build
                                     ("baryon"; default: "baryon")
  --config <PATH>                    Specifies path to the BELEX config YAML file.
  --log-level DEFAULT|VERBOSE|DEBUG  [Optional] Specifies the verbosity of output
                                     from the compiler. (Default: DEFAULT)
  --[no-]generate                    Whether to generate examples or reuse those
                                     which have already been generated.
  --[no-]build                       Whether to rebuild the belex tests or use
                                     the already existing executables.
  --source-only                      Whether to only generate the source code
                                     (no building or execution).
  --run-only                         Just run the test(s), do not generate
                                     sources or build them.
  --shuffle                          Shuffle the test for execution

Usage: belex-test \
  [--optimization-level OPTIMIZATION_LEVEL] \
  [--enable-optimization OPTIMIZATION_NAME]... \
  [--belex-dir /path/to/tests] \
  [--output-dir /path/to/libs-gvml/gensrc] \
  [--[no-]generate] \
  [--[no-]build] \
  [--source-only] \
  [--run-only] \
  [--shuffle] \
  [BELEX_FILE[::FRAGMENT_OR_TEST_NAME]]...

Examples:

  # Default --belex-dir and --output-dir (runs all high-level tests)
  belex-test -O2

  # Default --output-dir (runs all high-level tests)
  belex-test --belex-dir tests

  # Default --belex-dir (runs all tests)
  belex-test -O3 --output-dir gensrc

  # Explicit --belex-dir and --output-dir (runs all tests)
  belex-test --belex-dir tests --output-dir gensrc

  # Runs just test_belex_xor_u16 with default options
  belex-test tests/test_belex_xor_u16.py::xor_u16

  # Explicit output dir with two optimizations, a specific fragment, and a test
  # file to run.
  belex-test --output-dir gensrc \
    -f eliminate-read-after-write \
    -f delete-dead-writes \
    -f replace-zero-xor \
    tests/test_belex_basic_tests.py::and_u16d \
    tests/test_belex_add_u16.py
```
