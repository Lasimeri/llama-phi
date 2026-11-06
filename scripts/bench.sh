#!/usr/bin/env bash
# bench.sh WORK_DIR MODEL.gguf [a|host] [llama-bench args...]
#
#   a     (default) design A: llama.cpp native on the host's threads, the
#         card's share of every matrix multiply through libggml_phi.so
#   host  the same llama-bench without the backend: the control
#
# Defaults: -p 16 -n 16 -r 2 -t 14 -ngl 0 (extra arguments replace them).
# CARD_BYTES (default 10500000000) is the card's budget for resident rows;
# HOST_THREADS (default 14) the host's threads for its rows and its ops.
# The GPU is hidden (CUDA_VISIBLE_DEVICES empty) so the matrix work is the
# host's and the card's alone.
set -euo pipefail
work=${1:?usage: bench.sh WORK_DIR MODEL.gguf [a|host] [args...]}
model=${2:?usage: bench.sh WORK_DIR MODEL.gguf [a|host] [args...]}
mode=${3:-a}
shift $(( $# < 3 ? $# : 3 ))
args=("$@")
[ ${#args[@]} -gt 0 ] || args=(-p 16 -n 16 -r 2 -t "${HOST_THREADS:-14}" -ngl 0)
bin=$work/llama.cpp/build-native-cuda/bin/llama-bench
case "$mode" in
    a)
        exec env CUDA_VISIBLE_DEVICES= \
            GGML_BACKEND_PATH="$work/Intel-Phi-AVX-512-C/host/asm/out/libggml_phi.so" \
            PHI_GGML_CARDS=${PHI_GGML_CARDS:-0} PHI_GGML_THREADS=120 \
            PHI_GGML_CARD_BYTES=${CARD_BYTES:-10500000000} PHI_GGML_OFFLOAD=1 \
            PHI_GGML_HOST_THREADS=${HOST_THREADS:-14} \
            "$bin" -m "$model" -o md "${args[@]}" ;;
    host)
        exec env CUDA_VISIBLE_DEVICES= "$bin" -m "$model" -o md "${args[@]}" ;;
    *)
        echo "bench.sh: mode is a or host" >&2; exit 2 ;;
esac
