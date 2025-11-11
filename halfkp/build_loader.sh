#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
SRC_FILE="${REPO_ROOT}/nnue-pytorch-master/training_data_loader.cpp"
OUT_DIR="${SCRIPT_DIR}"
INCLUDE_FLAGS=(-I"${REPO_ROOT}/nnue-pytorch-master")
COMMON_FLAGS=(-std=c++17 -O3 -DNDEBUG)

usage() {
    cat <<EOF
Usage: $(basename "$0") [auto|mac|linux|windows|all]

Targets:
  auto     Build the library for the current host platform.
  mac      Build ${OUT_DIR}/training_data_loader.dylib using clang++.
  linux    Build ${OUT_DIR}/training_data_loader.so using g++/clang++.
  windows  Build ${OUT_DIR}/training_data_loader.dll using a MinGW cross-compiler.
  all      Attempt every target sequentially (skips ones without the required compiler).

Environment overrides:
  CXX           Override the host compiler for mac/linux builds.
  MINGW_CXX     Override the MinGW compiler for the windows build (default: x86_64-w64-mingw32-g++).
EOF
}

ensure_source() {
    if [[ ! -f "${SRC_FILE}" ]]; then
        echo "error: C++ source missing: ${SRC_FILE}" >&2
        exit 1
    fi
}

have_cmd() {
    command -v "$1" >/dev/null 2>&1
}

build_mac() {
    local compiler=${CXX:-clang++}
    if ! have_cmd "${compiler}"; then
        echo "warning: compiler '${compiler}' not found; skipping macOS build" >&2
        return 1
    fi
    echo "Building macOS dylib with ${compiler}" >&2
    "${compiler}" "${SRC_FILE}" "${INCLUDE_FLAGS[@]}" "${COMMON_FLAGS[@]}" \
        -fPIC -shared -pthread -o "${OUT_DIR}/training_data_loader.dylib"
}

build_linux() {
    local compiler=${CXX:-g++}
    if ! have_cmd "${compiler}"; then
        compiler=clang++
    fi
    if ! have_cmd "${compiler}"; then
        echo "warning: no g++/clang++ found; skipping Linux build" >&2
        return 1
    fi
    echo "Building Linux shared object with ${compiler}" >&2
    "${compiler}" "${SRC_FILE}" "${INCLUDE_FLAGS[@]}" "${COMMON_FLAGS[@]}" \
        -fPIC -shared -pthread -o "${OUT_DIR}/training_data_loader.so"
}

build_windows() {
    local compiler=${MINGW_CXX:-x86_64-w64-mingw32-g++}
    if ! have_cmd "${compiler}"; then
        echo "warning: MinGW compiler '${compiler}' not found; skipping Windows build" >&2
        return 1
    fi
    echo "Building Windows DLL with ${compiler}" >&2
    "${compiler}" "${SRC_FILE}" "${INCLUDE_FLAGS[@]}" "${COMMON_FLAGS[@]}" \
        -shared -static -static-libgcc -static-libstdc++ -lws2_32 -o "${OUT_DIR}/training_data_loader.dll"
}

run_target() {
    case "$1" in
        mac) build_mac ;;
        linux) build_linux ;;
        windows) build_windows ;;
        *) echo "error: unknown target '$1'" >&2; usage; exit 1 ;;
    esac
}

main() {
    ensure_source
    local target=${1:-auto}

    case "${target}" in
        -h|--help|help)
            usage
            exit 0
            ;;
        auto)
            case "$(uname -s)" in
                Darwin)
                    build_mac || exit 1
                    ;;
                Linux)
                    build_linux || exit 1
                    ;;
                MINGW*|MSYS*|CYGWIN*)
                    build_windows || exit 1
                    ;;
                *)
                    echo "error: unsupported host platform '$(uname -s)'" >&2
                    exit 1
                    ;;
            esac
            ;;
        all)
            local status=0
                build_mac || status=1
                build_linux || status=1
                build_windows || status=1
            if [[ ${status} -ne 0 ]]; then
                echo "Finished with some targets skipped or failed; see messages above." >&2
            fi
                exit ${status}
            ;;
        mac|linux|windows)
            run_target "${target}"
            ;;
        *)
            echo "error: unrecognised target '${target}'" >&2
            usage
            exit 1
            ;;
    esac
}

main "$@"
