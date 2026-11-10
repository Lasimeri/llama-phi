#!/usr/bin/env bash
# one.sh WORK_DIR MODEL.gguf start|stop|status: one llama-server for a
# model whose matrices almost fit the card (docs/one-server.md), through
# scripts/serve.sh with this placement:
#
#   GPU   the context and attention of every layer; the weights of the
#         last layers and of the output head (ONE_GPU_LAYERS); every prompt
#         of more than 31 tokens, its weights fed by the tridma router
#   card  every row of every other matrix (PHI_GGML_ALL_ROWS under the
#         offload): the host computes no matrix rows
#   host  the small multiplies the card declines, the router, the glue
#
# ONE_GPU_LAYERS is a regex alternation of layer numbers; it must leave
# the card no more than CARD_BYTES of matrices, which the backend's line
# at load confirms ("each keeps 100.0% of every dense matrix's rows").
# The default is the last four layers of a 64-layer model. The card's
# worker must be running with a pool of PHI_VPU_HUGEPAGES pages
# (scripts/worker.sh), and work/runs/serve/libtridma.so built
# (tools/tridma_router.md).
#
#   ONE_GPU_LAYERS  layers whose weights go to the GPU (60|61|62|63)
#   ONE_DRY=1       print what start would hand to serve.sh, start nothing
#   CARD_BYTES, PHI_VPU_HUGEPAGES, HOST_THREADS, TG_ONLY, SERVE_CTX,
#   SERVE_ALIAS, SERVE_ARGS, SERVE_PORT: as in serve.sh, with the defaults
#   of this layout below
#   CARD_THREADS    the slices a multiply is cut into on the card (120:
#                   two a core; 180 with a worker started with 181 threads)
# Read by the backend from the environment, passed through untouched:
#   PHI_GGML_AHEAD=1         a multiply whose input is ready is begun
#                            before its node (the measured layout sets it)
#   OMP_WAIT_POLICY=PASSIVE  the host's idle threads sleep: about six
#                            cores freed, generation about 5 % slower
set -euo pipefail
work=${1:?usage: one.sh WORK_DIR MODEL.gguf start|stop|status}
model=${2:?usage: one.sh WORK_DIR MODEL.gguf start|stop|status}
cmd=${3:-status}
here=$(cd "$(dirname "$0")" && pwd)
layers=${ONE_GPU_LAYERS:-60|61|62|63}
router=$work/runs/serve/libtridma.so

export PHI_GGML_ALL_ROWS=1
export TG_ONLY=${TG_ONLY:-31}
export PHI_VPU_HUGEPAGES=${PHI_VPU_HUGEPAGES:-6437}
export CARD_BYTES=${CARD_BYTES:-13100000000}
export HOST_THREADS=${HOST_THREADS:-8}
export SERVE_CTX=${SERVE_CTX:-131072}
export SERVE_ALIAS=${SERVE_ALIAS:-one-server}
export SERVE_ARGS=${SERVE_ARGS:--ctk q8_0 -ctv q8_0 -fa on}
# First match wins, and a pattern may hold no comma: the output head and
# the chosen layers to the GPU, everything else to the host side.
export SERVE_PLACE=${SERVE_PLACE:-"-ngl 99 -ot ^output\\.weight\$=CUDA0,^blk\\.($layers)\\.=CUDA0,.*=CPU"}

if [ -n "${ONE_DRY:-}" ]; then
    for v in PHI_GGML_ALL_ROWS TG_ONLY PHI_VPU_HUGEPAGES CARD_BYTES HOST_THREADS \
             SERVE_CTX SERVE_ALIAS SERVE_ARGS SERVE_PLACE; do
        printf '%s=%s\n' "$v" "${!v}"
    done
    printf 'LD_PRELOAD=%s\n' "$router"
    exit 0
fi
case $cmd in
    start)
        [ -r "$router" ] || { echo "one.sh: $router is not built" >&2; exit 1; }
        LD_PRELOAD="$router${LD_PRELOAD:+:$LD_PRELOAD}" exec bash "$here/serve.sh" "$work" "$model" start
        ;;
    stop|status)
        exec bash "$here/serve.sh" "$work" "$model" "$cmd"
        ;;
    *)
        echo "usage: one.sh WORK_DIR MODEL.gguf start|stop|status" >&2; exit 2
        ;;
esac
