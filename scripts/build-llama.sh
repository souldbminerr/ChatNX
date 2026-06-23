set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA="$ROOT/vendor/llama.cpp"
BUILD="$LLAMA/build-switch"
COMPAT="$ROOT/vendor/switch-compat"
SYSROOT="$ROOT/vendor/sysroot"

: "${DEVKITPRO:=/opt/devkitpro}"
export DEVKITPRO
if [ ! -f "$DEVKITPRO/cmake/Switch.cmake" ]; then
    echo "error: devkitPro not found at DEVKITPRO=$DEVKITPRO" >&2
    echo "set DEVKITPRO to your devkitpro install and re-run" >&2
    exit 1
fi

if [ ! -f "$LLAMA/CMakeLists.txt" ]; then
    echo "fetching llama.cpp submodule ..."
    git -C "$ROOT" submodule update --init --depth 1 vendor/llama.cpp
fi

# Switch patch: ggml-cpu.c and ggml-cpu.cpp both compile to ggml-cpu.o under
# devkitPro's short object names and collide, dropping ggml_backend_cpu_reg.
# Rename the .cpp so it gets its own object. Idempotent.
if [ -f "$LLAMA/ggml/src/ggml-cpu/ggml-cpu.cpp" ]; then
    echo "applying Switch patch to llama.cpp ..."
    mv "$LLAMA/ggml/src/ggml-cpu/ggml-cpu.cpp" \
       "$LLAMA/ggml/src/ggml-cpu/ggml-cpu-backend.cpp"
    sed -i 's#ggml-cpu/ggml-cpu\.cpp#ggml-cpu/ggml-cpu-backend.cpp#' \
       "$LLAMA/ggml/src/ggml-cpu/CMakeLists.txt"
fi

cmake -S "$LLAMA" -B "$BUILD" -G "Unix Makefiles" \
    -DCMAKE_TOOLCHAIN_FILE="$DEVKITPRO/cmake/Switch.cmake" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DGGML_BACKEND_DL=OFF \
    -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_COMMON=OFF \
    -DCMAKE_DEPENDS_USE_COMPILER=OFF \
    -DCMAKE_C_FLAGS="-I$COMPAT" \
    -DCMAKE_CXX_FLAGS="-I$COMPAT"

cmake --build "$BUILD" --target ggml llama -j"$(nproc 2>/dev/null || echo 4)"

mkdir -p "$SYSROOT/include" "$SYSROOT/lib"
cp "$LLAMA/include/llama.h"       "$SYSROOT/include/"
cp "$LLAMA"/ggml/include/*.h      "$SYSROOT/include/"
cp "$BUILD/src/libllama.a"             "$SYSROOT/lib/"
cp "$BUILD/ggml/src/libggml.a"         "$SYSROOT/lib/"
cp "$BUILD/ggml/src/libggml-cpu.a"     "$SYSROOT/lib/"
cp "$BUILD/ggml/src/libggml-base.a"    "$SYSROOT/lib/"

echo "Staged llama/ggml libs into $SYSROOT"
