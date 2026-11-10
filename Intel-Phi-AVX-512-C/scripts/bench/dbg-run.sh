#!/usr/bin/env bash
# dbg-run.sh NAME [--native]: llama-debug (every tensor's values and sum;
# the prompt "The cat sat on", 4 tokens) of the AVX-512 build under the
# preload on the cards, or with --native of the AVX2 twin with no preload
# (the reference). NAME.txt in $PHI_BENCH_DIR; compare two with
# dbg-cmp.sh. See dbg-run.md.
. "$(dirname "$0")/common.sh"
N=${1:?usage: dbg-run.sh NAME [--native]}
args=(-m "$model" -p "The cat sat on" -t 1 -ngl 0 -c 256 --no-warmup --verbose)
cd "$bdir" || exit 1
ulimit -c 0
if [ "${2:-}" = --native ]; then
    echo "== $(date '+%F %T') llama-debug of the AVX2 twin, natively; log $bdir/$N.txt"
    LD_LIBRARY_PATH=$llama/build-phi512-cpu/bin "$llama/build-phi512-cpu/bin/llama-debug" "${args[@]}" > "$bdir/$N.txt" 2>&1
    echo "== $(date '+%F %T') exited with status $?"
    exit 0
fi
L=$(bench_lib "$N") || exit 2
bench_workers
export PHI512_CARDS=$cards PHI512_CARD_ATOMICS=1 LD_LIBRARY_PATH=$llama/build-avx512f/bin
echo "== $(date '+%F %T') llama-debug under $L; log $bdir/$N.txt"
LD_PRELOAD=$L stdbuf -oL -eL "$llama/build-avx512f/bin/llama-debug" "${args[@]}" 2>&1 | tee "$bdir/$N.txt" \
    | grep --line-buffered -E 'common_debug_cb_eval|sum = |phi512: (progress|region left|card|registers|  blob)'
echo "== $(date '+%F %T') llama-debug exited with status ${PIPESTATUS[0]}"
[ -t 0 ] && read -r -p "Enter closes this window " _
exit 0
