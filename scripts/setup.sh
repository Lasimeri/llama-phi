#!/usr/bin/env bash
# setup.sh [WORK_DIR]: fetch the three pinned upstreams, apply this
# repository's patches, and build everything the co-processor setup runs.
#
#   llama.cpp            ggml-org/llama.cpp @ 4b77dee, unmodified:
#                        build-native-cuda (the host, design A) and,
#                        with WITH_AVX512=1, build-avx512f-cuda (the
#                        preload path)
#   Intel-Phi-AVX-512-C  @ db092f6 + patches/intel-phi-avx-512-c.patch:
#                        card worker, ggml backend libggml_phi.so,
#                        preload libphi512-c.so, phi-vpu driver
#   Intel-Phi-7120A      @ 22712a2 + patches/intel-phi-7120a-phitop.patch:
#                        phitop with the worker's DMA channel on its pcie line
#
# Needs: git, cmake, ninja, gcc/g++, GNU as and ld, tcc, cargo; the CUDA
# toolkit for the CUDA builds (CUDA=0 builds llama.cpp without it).
# JOBS (default 14) caps the parallel compile.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
work=${1:-$here/work}
jobs=${JOBS:-14}
cuda=${CUDA:-1}
mkdir -p "$work"
cd "$work"

fetch() { # fetch URL DIR COMMIT
    if [ ! -d "$2/.git" ]; then git clone -q "$1" "$2"; fi
    git -C "$2" fetch -q origin
    git -C "$2" checkout -q "$3"
}

echo "== llama.cpp"
fetch https://github.com/ggml-org/llama.cpp llama.cpp 4b77dee0524c64b8471869232e2958bd9589607d
common=(-G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CPU_REPACK=OFF)
if [ "$cuda" = 1 ]; then common+=(-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86); fi
cmake -S llama.cpp -B llama.cpp/build-native-cuda "${common[@]}" -DGGML_NATIVE=ON > /dev/null
ninja -C llama.cpp/build-native-cuda -j "$jobs" llama-bench llama-completion llama-server test-backend-ops
if [ "${WITH_AVX512:-0}" = 1 ]; then
    f='-mavx512f -mno-avx512vl -mno-avx512bw -mno-avx512dq -mno-avx512cd'
    cmake -S llama.cpp -B llama.cpp/build-avx512f-cuda "${common[@]}" -DGGML_NATIVE=OFF \
        -DGGML_AVX512=OFF -DCMAKE_C_FLAGS="$f" -DCMAKE_CXX_FLAGS="$f" > /dev/null
    ninja -C llama.cpp/build-avx512f-cuda -j "$jobs"
fi

echo "== Intel-Phi-AVX-512-C"
fetch https://github.com/Lasimeri/Intel-Phi-AVX-512-C Intel-Phi-AVX-512-C db092f61c6d03ad73cb9845e3abd66a712fabdf1
w=Intel-Phi-AVX-512-C
if git -C "$w" apply --check "$here/patches/intel-phi-avx-512-c.patch" 2> /dev/null; then
    git -C "$w" apply "$here/patches/intel-phi-avx-512-c.patch"
else
    git -C "$w" apply --reverse --check "$here/patches/intel-phi-avx-512-c.patch"   # already applied
fi
L=$work/llama.cpp
tcc -I"$L/ggml/include" -I"$L/ggml/src" -I"$L/ggml/src/ggml-cpu" -run "$w/tools/ggml-layout-check.c" \
    "$w/host/asm/ggml-phi/ggml_layout.inc"
( cd "$w" && sh card/vpu/build.sh > /dev/null && sh host/asm/ggml-phi/build.sh \
    && /usr/bin/make -C host/c -j "$jobs" > /dev/null && tcc -run tools/vpu-layout-check.c | tail -1 )
( cd "$w/host" && cargo build --release -q -p phi-vpu )

echo "== Intel-Phi-7120A phitop"
fetch https://github.com/Lasimeri/Intel-Phi-7120A Intel-Phi-7120A 22712a25aba2105d4d03014a1889cafb60f750ef
s=Intel-Phi-7120A
if git -C "$s" apply --check "$here/patches/intel-phi-7120a-phitop.patch" 2> /dev/null; then
    git -C "$s" apply "$here/patches/intel-phi-7120a-phitop.patch"
else
    git -C "$s" apply --reverse --check "$here/patches/intel-phi-7120a-phitop.patch"
fi
bash "$s/host/asm/phitop/build.sh" > /dev/null

echo "== built"
for f in "$w/host/asm/out/phi-vpu-worker" "$w/host/asm/out/libggml_phi.so" "$w/host/c/out/libphi512-c.so" \
    "$s/host/asm/out/phitop" llama.cpp/build-native-cuda/bin/llama-bench; do
    printf '%10s  %s  %s\n' "$(stat -c %s "$f")" "$(md5sum < "$f" | cut -c1-32)" "$f"
done
