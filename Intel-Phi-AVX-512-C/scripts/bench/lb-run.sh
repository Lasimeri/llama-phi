#!/usr/bin/env bash
# lb-run.sh NAME [P N]: llama-bench of llama.phi's AVX-512 build (the BF16
# Flash-Next) under the preload on the cards: fresh workers, prompt P (4)
# and generation N (2) tokens, one repetition, no warmup, one host thread,
# -ngl 0; the log on this terminal and in $PHI_BENCH_DIR/NAME.log. Meant
# for a visible terminal. See lb-run.md.
. "$(dirname "$0")/common.sh"
N=${1:?usage: lb-run.sh NAME [P N]}; P=${2:-4}; G=${3:-2}
L=$(bench_lib "$N") || exit 2
bench_workers
cd "$bdir" || exit 1
ulimit -c 0
export PHI512_CARDS=$cards PHI512_CARD_ATOMICS=1 LD_LIBRARY_PATH=$llama/build-avx512f/bin
echo "== $(date '+%F %T') llama-bench -p $P -n $G under $L; log $bdir/$N.log"
LD_PRELOAD=$L stdbuf -oL -eL "$llama/build-avx512f/bin/llama-bench" -m "$model" -p "$P" -n "$G" -r 1 -t 1 -ngl 0 \
    --no-warmup --progress -v 2>&1 | tee "$bdir/$N.log"
echo "== $(date '+%F %T') llama-bench exited with status ${PIPESTATUS[0]}"
[ -t 0 ] && read -r -p "Enter closes this window " _
exit 0
