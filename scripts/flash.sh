#!/usr/bin/env bash
# flash.sh WORK_DIR MODEL.gguf start|stop|status: a mixture-of-experts model several times
# the VRAM (measured: 188.2 GB at Q8_0, 48 blocks of 2.76 GB of which 2.67 GB are experts,
# and a 54.4 GB per-layer embedding table) on four GPUs, system memory and Xeon Phi cards
# (docs/multi-card.md), through scripts/serve.sh:
#   system memory  the experts of every block not named in FLASH_VRAM_BLOCKS, the embedding
#                  table and the first FLASH_PHI_BLOCKS blocks: allocated memory
#                  (--load-mode none, --lazy-mode off), no mapping of the file
#   GPUs           the dense weights of every other block and the output matrix, the context
#                  (204,800 tokens, Q8_0 keys and values), the experts of FLASH_VRAM_BLOCKS
#                  fixed in VRAM, and a cache of FLASH_CACHE_MIB of the experts in system
#                  memory: an expert a token routes to is uploaded on a miss and multiplied
#                  on the GPU, never on the host
#   Phi cards      FLASH_CARDS (0,1,2,3): blocks 0..FLASH_PHI_BLOCKS-1 (1), every tensor
#                  split by rows, an equal part on each card (at most 4.0 GB in all), through
#                  libggml_phi.so; the host keeps no row to multiply (PHI_GGML_ALL_ROWS=1,
#                  PHI_GGML_JUDGE=0, PHI_GGML_MIN_BYTES=1, PHI_GGML_PP_SHARE=1.0)
# FLASH_CARDS=none: no card, every block on the GPUs' side.
# Each card must be up with a worker of 110 threads and a pool of 2200 huge pages
# (Intel-Phi-AVX-512-C/scripts/phi-vpu.sh -c N start 110).
# The host cores that look busy while it generates wait, they do not multiply (86 % of their
# cycles are OpenMP spinning, perf 2026-11-10): OMP_WAIT_POLICY=passive GOMP_SPINCOUNT=0 in the
# environment of start drops 9.4 busy cores to 1.7 and costs 9 % (21.2-22.8 against 23.4-24.7 tok/s).
# This form with arguments was not run: the same commands ran with the paths written in.
set -euo pipefail
R=${1:?usage: flash.sh WORK_DIR MODEL.gguf start|stop|status}
M=${2:?usage: flash.sh WORK_DIR MODEL.gguf start|stop|status}
here=$(cd "$(dirname "$0")" && pwd)
cards=${FLASH_CARDS:-0,1,2,3}
runs=$R/runs/serve
vpu=$R/Intel-Phi-AVX-512-C/scripts/phi-vpu.sh
export SERVE_PORT=${FLASH_PORT:-8095} SERVE_ALIAS=${FLASH_ALIAS:-flash-q8_0} SERVE_CTX=204800
# FLASH_VRAM_BLOCKS: the blocks whose experts are fixed in VRAM, on the GPU that owns the block
# (-ts 1,1,1,1: GPU 0 blocks 1-12, GPU 1 13-24, GPU 2 25-36, GPU 3 37-47 and the output layer);
# a prompt multiplies them in place. Every other block keeps its experts in system memory: a
# prompt hands those multiplies to GPU 0 alone, which copies the experts in for each batch.
# "all" keeps every expert in system memory (--cpu-moe), cache FLASH_CACHE_MIB 40300.
vram=${FLASH_VRAM_BLOCKS:-10 11 12 22 23 24 34 35 36 45 46 47}
if [ "$vram" = all ]; then moe="--cpu-moe"; cache=${FLASH_CACHE_MIB:-40300}; hostexp=
else
    moe=; cache=${FLASH_CACHE_MIB:-21200}; hostexp=
    for b in $(seq 1 47); do case " $vram " in *" $b "*) ;; *) hostexp=${hostexp:+$hostexp|}$b ;; esac; done
fi
# FLASH_UBATCH: tokens a prompt batch holds. A batch pays one pull of the system-memory experts
# whatever its size (1,515 tokens 7.65 s, 2,048 tokens about 7 s, measured 2026-11-10), so a
# larger batch raises the prompt rate; its buffers take VRAM.
ub=${FLASH_UBATCH:-4096}
export SERVE_ARGS="-ctk q8_0 -ctv q8_0 -fa on --load-mode none --lazy-mode off -ts 1,1,1,1 $moe --moe-cache-mib $cache -b $ub -ub $ub"
export HOST_THREADS=${FLASH_HOST_THREADS:-32} GGML_CUDA_NO_PINNED=1
if [ "$cards" = none ]; then
    list=
    export GGML_BACKEND_PATH=/nonexistent PHI_GGML_CARDS=0 SERVE_PLACE="-ngl 99"
else
    list=${cards//,/ }; n=$(echo $list | wc -w); phi=${FLASH_PHI_BLOCKS:-1}
    # 48 blocks and the output layer: the first phi blocks are not the GPUs'. The -ot names the same
    # blocks and is REQUIRED: this llama.cpp opens GGML_BACKEND_PATH only from the handlers of -ot,
    # --device and --list-devices; without one of them libggml_phi.so is never loaded.
    export SERVE_PLACE="-ngl $((49 - phi)) -ot ^blk\\.($(seq -s "|" 0 $((phi - 1))))\\.=CPU${hostexp:+,^blk\\.($hostexp)\\.ffn_(up|gate|down)_exps\\.=CPU}"
    export PHI_GGML_CARDS=$cards CARD_BYTES=$((4000000000 / n)) PHI_VPU_HUGEPAGES=2200 CARD_THREADS=${FLASH_THREADS:-110}
    # PP_SHARE 1.0 and PP_ADAPT 0: a batch of 2 to 31 tokens is the cards' as well (the default gives the host a share of it)
    export OFFLOAD=0 PHI_GGML_ALL_ROWS=1 PHI_GGML_JUDGE=0 PHI_GGML_MIN_BYTES=${FLASH_MIN_BYTES:-1} PHI_GGML_AHEAD=1 PHI_GGML_HOSTOPS=1 PHI_GGML_PP_SHARE=1.0 PHI_GGML_PP_ADAPT=0
    export LD_PRELOAD="$runs/libunpin.so"
fi
case ${3:-status} in
    start) cd "$R"; bash "$here/serve.sh" "$R" "$M" start ;;
    stop)
        bash "$here/serve.sh" "$R" "$M" stop
        for c in $list; do
            [ "$c" = 0 ] && continue
            used=$(timeout 30 phi -c "$c" run sh -c 'awk "/HugePages_Total/ {t=\$2} /HugePages_Free/ {f=\$2} END {print t - f}" /proc/meminfo' < /dev/null)
            if [ "${used:-0}" -gt 64 ]; then
                bash "$vpu" -c "$c" stop || true
                PHI_VPU_HUGEPAGES=2200 PHI_VPU_ARGS="-e 0" bash "$vpu" -c "$c" start 110 | tail -n 1
            fi
        done ;;
    status) exec bash "$here/serve.sh" "$R" "$M" status ;;
    *) echo "usage: flash.sh WORK_DIR MODEL.gguf start|stop|status" >&2; exit 2 ;;
esac
