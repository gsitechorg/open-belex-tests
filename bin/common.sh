# -------------------------------------------------------------------------------
# By Dylon Edwards
#
# Copyright 2019 - 2023 GSI Technology, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the “Software”), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

export EXIT_SUCCESS=0

# Keep common failure exit codes odd
export EXIT_NOT_IN_C_SIM=1
export EXIT_BUILD_FAILED=3
export EXIT_TEST_FAILED=5
export EXIT_AWK_FAILED=7
export EXIT_NO_MANIFEST=9
export EXIT_CLEAN_FAILED=11
export EXIT_SOURCE_FAILED=13

# special value for xargs
export EXIT_XARGS_FAILURE=255

# You may override this before executing the script
export MESON_BUILD_DIR="${MESON_BUILD_DIR:-build}"

# Holds the location of the manifest
export MANIFEST

# Holds the project-root dir
export BELEX_TEST_DIR

# Whether to shuffle the tests
export SHUFFLE_TESTS

export BUILD_TYPE="baryon"  # "baryon"

# Holds the names of the BELEX tests to run
declare -a BELEX_TESTS

declare -a MESON_OPTS

export PATH="${CONDA_PREFIX}/bin:${PATH}"
export C_INCLUDE_PATH="${CONDA_PREFIX}/include:${C_INCLUDE_PATH}"
export CXX_INCLUDE_PATH="${CONDA_PREFIX}/include:${CXX_INCLUDE_PATH}"
export LIBRARY_PATH="${CONDA_PREFIX}/lib:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
export PKG_CONFIG_PATH="${CONDA_PREFIX}/lib/pkgconfig:${CONDA_PREFIX}/share/pkgconfig:${PKG_CONFIG_PATH}"

function find-belex-test-dir() {
    local CURRENT_DIR="$1"
    local PROJECT_ROOT

    CURRENT_DIR="$(realpath "$CURRENT_DIR")"
    while [[ "$CURRENT_DIR" != "/" ]]; do
        PROJECT_ROOT="${CURRENT_DIR}/.project-root"
        if [[ -e "$PROJECT_ROOT" && "$(cat "$PROJECT_ROOT")" == "open-belex-tests" ]]; then
            export BELEX_TEST_DIR="$CURRENT_DIR"
            return $EXIT_SUCCESS
        fi
        CURRENT_DIR="${CURRENT_DIR}/.."
        CURRENT_DIR="$(realpath "$CURRENT_DIR")"
    done

    echo "Directory is not part of belex-tests: $1" 1>&2
    return $EXIT_NOT_IN_C_SIM
}

export -f find-belex-test-dir

function prepare-environment() {
    return $EXIT_SUCCESS
}

function build-belex-tests() {
    local RETURN_CODE

    pushd "$BELEX_TEST_DIR"

    local BELEX_TEST_BUILD_DIR

    if [ -e conda_prefix ]; then
        rm conda_prefix
    fi
    ln -s "$CONDA_PREFIX" conda_prefix

    if [ ! -d "$MESON_BUILD_DIR" ]; then
        if [[ "$BUILD_TYPE" == "baryon" ]]; then
            meson setup "${MESON_OPTS[@]}" "$MESON_BUILD_DIR"
            RETURN_CODE=$?
        fi
    fi

    for BELEX_TEST in "${BELEX_TESTS[@]}"; do
        BELEX_TEST_BUILD_DIR="${MESON_BUILD_DIR}/gensrc/${BELEX_TEST}"
        if [ -d "${BELEX_TEST_BUILD_DIR}" ]; then
            rm -rf "${BELEX_TEST_BUILD_DIR}"
            RETURN_CODE=$?

            if (( RETURN_CODE != EXIT_SUCCESS )); then
                echo "Failed to delete build dir: ${BELEX_TEST_BUILD_DIR}" 1>&2
                echo "`rm -rf \"${BELEX_TEST_BUILD_DIR}\"` failed with status: ${RETURN_CODE}" 1>&2
                RETURN_CODE=$EXIT_CLEAN_FAILED
                break
            fi
        fi
    done

    if (( RETURN_CODE == EXIT_SUCCESS )); then
        meson compile -C "$MESON_BUILD_DIR" --verbose
        RETURN_CODE=$?
    fi

    popd

    if (( RETURN_CODE != EXIT_SUCCESS )); then
        echo "meson build failed with exit code: $RETURN_CODE"
        return $EXIT_BUILD_FAILED
    fi

    return $EXIT_SUCCESS
}

function has-command() {
    command -v "$1" &>/dev/null
    return $?
}

function parse-manifest() {
    local RETURN_CODE

    if [ ! -e "$MANIFEST" ]; then
        echo "Manifest does not exist: $MANIFEST" 1>&2
        return $EXIT_NO_MANIFEST
    fi

    export BELEX_TESTS=($(awk "BEGIN {FS=\"'\"} /^subdir/ {print \$2}" "$MANIFEST"; exit $?))
    RETURN_CODE=$?

    if (( RETURN_CODE != EXIT_SUCCESS )); then
        echo "awk failed with exit code: $RETURN_CODE" 1>&2
        return $EXIT_AWK_FAILED
    fi

    if [ -z "$SHUFFLE_TESTS" ]; then
        readarray -t BELEX_TESTS < <(sort < <(printf '%s\n' "${BELEX_TESTS[@]}"))
    else
        readarray -t BELEX_TESTS < <(shuf -e "${BELEX_TESTS[@]}")
    fi

    return $EXIT_SUCCESS
}

function run-belex-tests() {
    local RETURN_CODE

    pushd "$BELEX_TEST_DIR"

    RETURN_CODE=$EXIT_SUCCESS
    for BELEX_TEST in "${BELEX_TESTS[@]}"; do
        pushd "${MESON_BUILD_DIR}/gensrc/${BELEX_TEST}"
        if ! ./"${BELEX_TEST}"; then
            RETURN_CODE=$EXIT_TEST_FAILED
        fi
        popd
    done

    popd

    return $RETURN_CODE
}
