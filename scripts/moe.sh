#!/usr/bin/env bash
# moe.sh WORK_DIR MODEL.gguf start|stop|status|route|rank: one llama-server
# for a mixture-of-experts model that does not fit host memory beside the
# card (docs/moe.md), through scripts/serve.sh:
#
#   card  half of every dense matrix's rows and whole experts, the most
#         used first, up to CARD_BYTES; offloaded (the host drops them)
#   host  every other expert, the other half of the dense rows, the small
#         matrices and the small operations between the multiplies
#   GPU   the context of every layer and every prompt of more than 31
#         tokens; no weights of its own: a prompt's come from host RAM and,
#         for what the card holds, from the card through the tridma router
#
#   route   the server with a row share on the card and the routing of every
#           generated token logged (PHI_GGML_IDS=1). Generate text of the
#           kinds the model will be used for, then:
#   rank    each layer's experts in order of use, from the last server's
#           log, written to MOE_PLACEMENT
#   start   the layout, with MOE_PLACEMENT
#
#   MOE_PLACEMENT   the ranking file (WORK_DIR/runs/serve/placement.txt)
#   MOE_DRY=1       print what start would set, start nothing
#   CARD_BYTES, PHI_VPU_HUGEPAGES, HOST_THREADS, SERVE_CTX, SERVE_ARGS,
#   SERVE_PORT, SERVE_ALIAS: as in serve.sh, with this layout's defaults
#
# Needs the card's worker started with a pool of PHI_VPU_HUGEPAGES pages
# (scripts/worker.sh) and, in WORK_DIR/runs/serve, libtridma.so
# (tools/tridma_router.md) and libunpin.so (tools/unpin.md); start goes on
# without either, slower, and says so.
set -euo pipefail
work=${1:?usage: moe.sh WORK_DIR MODEL.gguf start|stop|status|route|rank}
model=${2:?usage: moe.sh WORK_DIR MODEL.gguf start|stop|status|route|rank}
cmd=${3:-status}
here=$(cd "$(dirname "$0")" && pwd)
place=${MOE_PLACEMENT:-$work/runs/serve/placement.txt}

export CARD_BYTES=${CARD_BYTES:-13100000000}
export PHI_VPU_HUGEPAGES=${PHI_VPU_HUGEPAGES:-6437}
export HOST_THREADS=${HOST_THREADS:-14}
export SERVE_CTX=${SERVE_CTX:-262144}
export SERVE_ALIAS=${SERVE_ALIAS:-moe}

case $cmd in
    route)
        PHI_GGML_IDS=1 exec bash "$here/serve.sh" "$work" "$model" start
        ;;
    rank)
        log=$(readlink -f "$work/runs/serve/server.log")
        ids=$work/runs/serve/ids.log
        grep 'ggml-phi: ids layer' "$log" > "$ids" || { echo "moe.sh: no routing lines in $log (start with route first)" >&2; exit 1; }
        tcc -run "$work/Intel-Phi-AVX-512-C/tools/expert-placement.c" stats "$ids" | sed -n '/^generation/,$p'
        tcc -run "$work/Intel-Phi-AVX-512-C/tools/expert-placement.c" rank "$ids" 256 > "$place"
        echo "moe.sh: $(wc -l < "$place") layers ranked from $(wc -l < "$ids") routing lines, in $place"
        ;;
    start)
        [ -s "$place" ] || { echo "moe.sh: no ranking at $place (route, generate, rank)" >&2; exit 1; }
        pre=
        for lib in libtridma.so libunpin.so; do
            if [ -r "$work/runs/serve/$lib" ]; then pre=${pre:+$pre:}$work/runs/serve/$lib
            else echo "moe.sh: $work/runs/serve/$lib is not built: going on without it" >&2; fi
        done
        export PHI_GGML_EXPERTS=$place PHI_GGML_AHEAD=1 PHI_GGML_HOSTOPS=1
        if [ -n "${MOE_DRY:-}" ]; then
            for v in PHI_GGML_EXPERTS PHI_GGML_AHEAD PHI_GGML_HOSTOPS CARD_BYTES PHI_VPU_HUGEPAGES HOST_THREADS SERVE_CTX SERVE_ALIAS; do
                printf '%s=%s\n' "$v" "${!v}"
            done
            printf 'LD_PRELOAD=%s\n' "$pre"
            exit 0
        fi
        LD_PRELOAD="$pre${LD_PRELOAD:+:$LD_PRELOAD}" exec bash "$here/serve.sh" "$work" "$model" start
        ;;
    stop|status)
        exec bash "$here/serve.sh" "$work" "$model" "$cmd"
        ;;
    *)
        echo "usage: moe.sh WORK_DIR MODEL.gguf start|stop|status|route|rank" >&2; exit 2
        ;;
esac
