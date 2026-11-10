#!/usr/bin/env bash
# vram35.sh WORK_DIR MODEL.gguf start|stop|status: a mixture-of-experts model larger than
# host memory, held in three memories and read from the disk once (docs/moe.md,
# tools/hostram.md), through scripts/moe.sh:
#   VRAM   the last VRAM_LAYERS blocks and the output matrix, with the context
#          (VRAM_CTX tokens, Q8_0 keys and values) and every prompt
#   card   VRAM_CARD_BYTES of the other blocks: half of each dense matrix and whole
#          experts, the most used first, offloaded
#   host   everything else, as locked anonymous memory: libhostram.so gives llama.cpp an
#          anonymous region in place of a mapping of the file and locks the host's share
#
#   VRAM_LAYERS (18)   blocks whose weights live in VRAM
#   VRAM_BLOCKS (40)   the model's highest block number
#   VRAM_CTX (204800), VRAM_CARD_BYTES (a quarter of the file), VRAM_ALIAS, VRAM_ARGS
#   (more llama-server arguments)
# Needs, in WORK_DIR/runs/serve: libhostram.so (tools/hostram.md), libtridma.so,
# libunpin.so and a ranking (scripts/moe.sh route, rank). The server runs in a systemd
# scope of its own, so that a memory shortage ends it and nothing else.
set -euo pipefail
work=${1:?usage: vram35.sh WORK_DIR MODEL.gguf start|stop|status}
model=${2:?usage: vram35.sh WORK_DIR MODEL.gguf start|stop|status}
n=${VRAM_LAYERS:-18}
blocks=${VRAM_BLOCKS:-40}
here=$(cd "$(dirname "$0")" && pwd)
runs=$work/runs/serve
go=$runs/hostram.go
case ${3:-status} in
    start)
        [ -r "$runs/libhostram.so" ] || { echo "vram35: $runs/libhostram.so is not built (tools/hostram.md)" >&2; exit 1; }
        first=$((blocks - n))
        layers=$(seq -s '|' "$first" "$blocks")
        prefixes=()
        for l in $(seq "$first" "$blocks"); do prefixes+=("blk.$l."); done
        # the byte ranges of what goes to VRAM, neighbours merged: read once, never kept
        tcc -run "$here/../tools/ggufranges.c" "$model" "${prefixes[@]}" output.weight | sort -n |
            awk 'NR == 1 { o = $1; e = $1 + $2; next } $1 <= e + 4096 { e = $1 + $2; next }
                 { print o, e - o; o = $1; e = $1 + $2 } END { print o, e - o }' > "$runs/transient.txt"
        mv -f "$go" "$go.old" 2>/dev/null || true
        systemd-run --user --scope --quiet --unit="phi-llama-$(date +%H%M%S)" env \
            LD_PRELOAD="$runs/libhostram.so" HOSTRAM_GO="$go" HOSTRAM_TRANSIENT="$runs/transient.txt" \
            CARD_BYTES="${VRAM_CARD_BYTES:-$(( $(stat -c %s "$model") / 4 ))}" PHI_VPU_HUGEPAGES="${PHI_VPU_HUGEPAGES:-5000}" \
            SERVE_CTX="${VRAM_CTX:-204800}" SERVE_ARGS="-ctk q8_0 -ctv q8_0 -fa on ${VRAM_ARGS:-}" \
            SERVE_PLACE="-ngl 99 -ot ^output\\.weight\$=CUDA0,^blk\\.($layers)\\.=CUDA0,.*=CPU" \
            SERVE_ALIAS="${VRAM_ALIAS:-$(basename "$model" .gguf)}" \
            bash "$here/moe.sh" "$work" "$model" start
        touch "$go"                              # the server listens: the host's share is read and locked
        log=$(readlink -f "$runs/server.log")
        for _ in $(seq 1 180); do grep -q 'locked anonymous memory' "$log" && break; sleep 1; done
        grep 'hostram:' "$log" | tail -n 1
        echo "vram35: VRAM $(nvidia-smi --query-gpu=memory.used --format=csv,noheader), $(grep -E 'VmLck|VmSwap' "/proc/$(cat "$runs/server.pid")/status" | tr -s '\t ' ' ' | tr '\n' ' ')" ;;
    stop|status)
        exec bash "$here/moe.sh" "$work" "$model" "$3" ;;
    *) echo "usage: vram35.sh WORK_DIR MODEL.gguf start|stop|status" >&2; exit 2 ;;
esac
