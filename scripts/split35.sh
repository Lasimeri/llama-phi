#!/usr/bin/env bash
# split35.sh prefill|decode|stop|status: the two servers behind the split
# proxy on 127.0.0.1:8096 (its source is not in this repository) for a mixture-of-experts
# model:
#   prefill  127.0.0.1:8091  CUDA: as many layers as fit in VRAM, the rest
#            from host RAM; computes a prompt and saves slot 0
#   decode   127.0.0.1:8092  no GPU: the host's threads and the Xeon Phi;
#            the card holds SPLIT_CARD_BYTES of the weights (half of every
#            dense matrix, then whole experts, the most used first:
#            docs/moe.md), offloaded; restores slot 0 and generates
# SPLIT_WORK (the work directory), SPLIT_MODEL (the GGUF file), SPLIT_CTX (131072),
# SPLIT_CARD_BYTES (a quarter of the
# file), SPLIT_PLACEMENT, SPLIT_ALIAS. Both servers must agree on the
# model, the context and the cache types, or a slot does not restore.
#
# Status: not verified. The prefill branch started once (it puts as many layers as fit
# into VRAM); the decode branch and the hand-over of a slot between the two have never
# been run. scripts/vram35.sh is the layout that was measured.
set -euo pipefail
work=${SPLIT_WORK:?set SPLIT_WORK to the work directory}
model=${SPLIT_MODEL:?set SPLIT_MODEL to the GGUF file}
alias=${SPLIT_ALIAS:-$(basename "$model" .gguf)}
ctx=${SPLIT_CTX:-131072}
slot=/dev/shm/llama-phi-pd
runs=$work/runs/pd
bin=$work/llama.cpp/build-native-cuda/bin/llama-server
shim=$work/runs/serve/libnopopulate.so
phi=$work/Intel-Phi-AVX-512-C
place=${SPLIT_PLACEMENT:-$work/runs/serve/placement.txt}
bytes=${SPLIT_CARD_BYTES:-$(( $(stat -c %s "$model") / 4 ))}
mkdir -p "$slot" "$runs"
common=(-m "$model" --alias "$alias" -c "$ctx" -ctk q8_0 -ctv q8_0 -fa on -np 1 --no-warmup --jinja
        --slot-save-path "$slot" --host 127.0.0.1)
pid_of() { [ -f "$runs/$1-35.pid" ] && p=$(cat "$runs/$1-35.pid") && [ "$(cat "/proc/$p/comm" 2>/dev/null)" = llama-server ] && echo "$p"; }
up() {   # up NAME PORT LOG PID: wait for the listener
    for _ in $(seq 1 900); do
        grep -q 'listening on' "$3" && { echo "split35: $1 up on 127.0.0.1:$2 (pid $4), log $3"; return 0; }
        kill -0 "$4" 2>/dev/null || { echo "split35: $1 exited, see $3" >&2; return 1; }
        sleep 1
    done
    echo "split35: $1 not up after 900 s, see $3" >&2; return 1
}
case ${1-} in
    prefill)
        if p=$(pid_of prefill); then echo "split35: prefill already running (pid $p)"; exit 0; fi
        log=$runs/prefill-35-$(date +%Y%m%d-%H%M%S).log
        LD_PRELOAD=$shim nohup "$bin" "${common[@]}" -b 2048 -ub 2048 -t 8 -tb 8 \
            -ngl auto --fit on --fit-target 3072 --port 8091 > "$log" 2>&1 &
        echo $! > "$runs/prefill-35.pid"; ln -sfn "$log" "$runs/prefill-35.log"
        up prefill 8091 "$log" $! ;;
    decode)
        if p=$(pid_of decode); then echo "split35: decode already running (pid $p)"; exit 0; fi
        bash "$phi/scripts/card-users.sh" 0 || { echo "split35: the card is in use; one program at a time" >&2; exit 1; }
        [ -s "$place" ] || { echo "split35: no ranking at $place (scripts/moe.sh route, rank)" >&2; exit 1; }
        log=$runs/decode-35-$(date +%Y%m%d-%H%M%S).log
        env CUDA_VISIBLE_DEVICES= LD_PRELOAD=$shim GGML_BACKEND_PATH="$phi/host/asm/out/libggml_phi.so" \
            PHI_GGML_CARDS=0 PHI_GGML_THREADS=120 PHI_GGML_CARD_BYTES=$bytes PHI_GGML_OFFLOAD=1 \
            PHI_GGML_HOST_THREADS=14 PHI_GGML_EXPERTS=$place PHI_GGML_AHEAD=1 PHI_GGML_TIME=1 \
            nohup "$bin" "${common[@]}" -b 256 -ub 128 -t 14 -tb 14 -ngl 0 --port 8092 \
            --temp 0.5 --top-k 75 --top-p 0.95 --min-p 0.05 --repeat-penalty 1.2 > "$log" 2>&1 &
        echo $! > "$runs/decode-35.pid"; ln -sfn "$log" "$runs/decode-35.log"
        up decode 8092 "$log" $! ;;
    stop)
        for s in prefill decode; do
            if p=$(pid_of $s); then
                kill "$p"; for _ in $(seq 1 10); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
                kill -0 "$p" 2>/dev/null && kill -9 "$p"
                echo "split35: $s stopped (pid $p)"
            fi
        done
        if bash "$phi/scripts/card-users.sh" 0; then "$work/runs/cuda/vpu_dma" -F; fi ;;
    status)
        for s in prefill:8091 decode:8092 proxy:8096; do
            printf 'split35: %s %s\n' "$s" "$(curl -s -m 3 "http://127.0.0.1:${s#*:}/health" || echo 'not answering')"
        done ;;
    *) echo "usage: split35.sh prefill|decode|stop|status" >&2; exit 2 ;;
esac
