#!/usr/bin/env bash
# cards35.sh WORK_DIR MODEL.gguf start|stop|status: the Q8_0 mixture model of vram35.sh on a
# machine with several GPUs and several Xeon Phi cards (docs/multi-card.md), through
# scripts/serve.sh:
#   VRAM  blocks CARDS_BLOCKS (4) to VRAM_BLOCKS (40) and the output matrices over the GPUs
#         (llama.cpp's own layer split), with the context (CARDS_CTX tokens, Q8_0 keys and
#         values) and every prompt
#   Phi   CARDS (0,1,2,3): together at most CARDS_BYTES (4.0 GB) of the first CARDS_BLOCKS
#         blocks, an equal part on each card. CARDS_SPLIT=rows (the default, tensor
#         parallel): every card keeps an equal part of the rows of every tensor, the host
#         none. CARDS_SPLIT=experts: whole experts by use and a share of the dense rows
#         (needs a ranking: scripts/moe.sh route, rank)
#   host  what is left of the first blocks and the small operations, as locked memory
#         (libhostram.so). The DMA router (libtridma.so) feeds a prompt on the GPUs each
#         card's rows from that card, the cards side by side
# CARDS_THREADS: slices a card cuts a request into (the worker's threads over the number of
# cards by default: 114, 57 and 28 measured best for 1, 2 and 4 cards with 114-thread workers).
# Each card must be up with a worker and a pool of CARDS_PAGES (2200) huge pages:
#   PHI_VPU_HUGEPAGES=2200 PHI_VPU_ARGS="-e 0" Intel-Phi-AVX-512-C/scripts/phi-vpu.sh -c N start 110
# Needs, in WORK_DIR/runs/serve: libtridma.so, libunpin.so, libhostram.so (tools/*.md).
# This form with arguments was not run: the same commands ran with the paths written in.
set -euo pipefail
work=${1:?usage: cards35.sh WORK_DIR MODEL.gguf start|stop|status}
model=${2:?usage: cards35.sh WORK_DIR MODEL.gguf start|stop|status}
here=$(cd "$(dirname "$0")" && pwd)
cards=${CARDS:-0,1,2,3}
list=${cards//,/ }
n=$(echo $list | wc -w)
first=${CARDS_BLOCKS:-4}
blocks=${VRAM_BLOCKS:-40}
pages=${CARDS_PAGES:-2200}
runs=$work/runs/serve
vpu=$work/Intel-Phi-AVX-512-C/scripts/phi-vpu.sh
if [ "${CARDS_SPLIT:-rows}" = rows ]; then how=PHI_GGML_ALL_ROWS=1; else how=PHI_GGML_EXPERTS=$runs/placement.txt; fi
export PHI_GGML_CARDS=$cards SERVE_PORT=${CARDS_PORT:-8095}
case ${3:-status} in
    start)
        # the byte ranges of what goes to VRAM, neighbours merged: read once, never kept
        pre=(); for l in $(seq "$first" "$blocks"); do pre+=("blk.$l."); done
        tcc -run "$here/../tools/ggufranges.c" "$model" "${pre[@]}" output | sort -n |
            awk 'NR == 1 { o = $1; e = $1 + $2; next } $1 <= e + 4096 { e = $1 + $2; next }
                 { print o, e - o; o = $1; e = $1 + $2 } END { print o, e - o }' > "$runs/transient.txt"
        go=$runs/hostram.go; mv -f "$go" "$go.old" 2>/dev/null || true
        # the -ot is required twice over: it keeps the first blocks off the GPUs, and this llama.cpp
        # opens GGML_BACKEND_PATH only from the handlers of -ot, --device and --list-devices
        env LD_PRELOAD="$runs/libtridma.so:$runs/libunpin.so:$runs/libhostram.so" HOSTRAM_GO="$go" HOSTRAM_TRANSIENT="$runs/transient.txt" \
            CARD_BYTES=$((${CARDS_BYTES:-4000000000} / n)) PHI_VPU_HUGEPAGES="$pages" CARD_THREADS=${CARDS_THREADS:-$((110 / n))} \
            HOST_THREADS=${CARDS_HOST_THREADS:-32} SERVE_CTX=${CARDS_CTX:-204800} SERVE_ARGS="-ctk q8_0 -ctv q8_0 -fa on" \
            SERVE_PLACE="-ngl 99 -ot ^blk\\.($(seq -s '|' 0 $((first - 1))))\\.=CPU" SERVE_ALIAS=${CARDS_ALIAS:-moe-q8_0} \
            "$how" PHI_GGML_AHEAD=1 PHI_GGML_HOSTOPS=1 \
            bash "$here/serve.sh" "$work" "$model" start
        touch "$go"
        log=$(readlink -f "$runs/server.log")
        for _ in $(seq 1 120); do grep -q 'locked anonymous memory' "$log" && break; sleep 1; done
        grep 'hostram:' "$log" | tail -n 1 ;;
    stop)
        bash "$here/serve.sh" "$work" "$model" stop
        # the stop frees card 0's weights; a card that still holds some gets a fresh worker
        for c in $list; do
            [ "$c" = 0 ] && continue
            used=$(timeout 30 phi -c "$c" run sh -c 'awk "/HugePages_Total/ {t=\$2} /HugePages_Free/ {f=\$2} END {print t - f}" /proc/meminfo' < /dev/null)
            if [ "${used:-0}" -gt 64 ]; then
                echo "cards35: card $c still holds $used huge pages: restarting its worker"
                bash "$vpu" -c "$c" stop || true
                PHI_VPU_HUGEPAGES="$pages" PHI_VPU_ARGS="-e 0" bash "$vpu" -c "$c" start 110 | tail -n 1
            fi
        done ;;
    status) exec bash "$here/serve.sh" "$work" "$model" status ;;
    *) echo "usage: cards35.sh WORK_DIR MODEL.gguf start|stop|status" >&2; exit 2 ;;
esac
