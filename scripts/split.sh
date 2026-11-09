#!/usr/bin/env bash
# split.sh WORK_DIR MODEL prefill|decode: the two servers of the split
# layout (docs/q4_0.md), each in the foreground:
#
#   prefill  the GPU: the model and the context in VRAM, port 8091
#   decode   the host and the card: weights mapped from the file, every
#            row the card's budget holds uploaded to it and dropped from
#            the host (PHI_GGML_OFFLOAD, PHI_GGML_ALL_ROWS), port 8092
#
# Both take the same model, context and cache types, and share a slot
# directory: a prompt is prefilled on the GPU, saved as a slot file and
# restored on the decode server (scripts/handoff.sh). The card's worker
# must be running (scripts/worker.sh, with PHI_VPU_HUGEPAGES sized for
# SPLIT_CARD_BYTES: 6437 pages for the default).
#
#   SPLIT_CTX         context of both servers (131072)
#   SPLIT_SLOTS       the slot directory (/dev/shm/llama-phi-pd)
#   SPLIT_CARD_BYTES  the card's budget for resident rows (13100000000)
#   SPLIT_THREADS     host threads of the decode server (8)
#   SPLIT_FIT_TARGET  MiB of VRAM the prefill server leaves free (2048)
#   SPLIT_DRY=1       print the command and its environment, start nothing
set -euo pipefail
work=${1:?usage: split.sh WORK_DIR MODEL prefill|decode}
model=${2:?usage: split.sh WORK_DIR MODEL prefill|decode}
mode=${3:?usage: split.sh WORK_DIR MODEL prefill|decode}
bin=$work/llama.cpp/build-native-cuda/bin/llama-server
shim=$work/runs/serve/libnopopulate.so
slots=${SPLIT_SLOTS:-/dev/shm/llama-phi-pd}
threads=${SPLIT_THREADS:-8}
[ -x "$bin" ] || { echo "split.sh: $bin is not built (scripts/setup.sh)" >&2; exit 1; }
[ -f "$model" ] || { echo "split.sh: no model at $model" >&2; exit 1; }

common=(-m "$model" -c "${SPLIT_CTX:-131072}" -ctk q8_0 -ctv q8_0 -fa on -np 1
        -b 256 -ub 128 --no-warmup --slot-save-path "$slots" --host 127.0.0.1)
env=()
# The shim keeps llama.cpp from reading the whole file at load
# (tools/nopopulate.md); without it the servers still work.
[ -r "$shim" ] && env+=("LD_PRELOAD=$shim")

case $mode in
    prefill)
        args=("${common[@]}" -t 2 -tb 2 -ngl 99 --fit on
              --fit-target "${SPLIT_FIT_TARGET:-2048}" --port 8091)
        ;;
    decode)
        env+=("CUDA_VISIBLE_DEVICES="
              "GGML_BACKEND_PATH=$work/Intel-Phi-AVX-512-C/host/asm/out/libggml_phi.so"
              "PHI_GGML_CARDS=${PHI_GGML_CARDS:-0}" "PHI_GGML_THREADS=120"
              "PHI_GGML_CARD_BYTES=${SPLIT_CARD_BYTES:-13100000000}"
              "PHI_GGML_OFFLOAD=1" "PHI_GGML_ALL_ROWS=1"
              "PHI_GGML_HOST_THREADS=$threads")
        args=("${common[@]}" -t "$threads" -ngl 0 -ot no_such_tensor=CPU --port 8092)
        ;;
    *)
        echo "usage: split.sh WORK_DIR MODEL prefill|decode" >&2; exit 2
        ;;
esac

if [ -n "${SPLIT_DRY:-}" ]; then
    printf '%s\n' "${env[@]}"
    printf '%q ' "$bin" "${args[@]}"; printf '\n'
    exit 0
fi
mkdir -p "$slots"
exec env "${env[@]}" "$bin" "${args[@]}"
