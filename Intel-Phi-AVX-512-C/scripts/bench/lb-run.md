# lb-run.sh: llama-bench under the preload

`lb-run.sh NAME [P N]`: llama-bench of `build-avx512f` on the BF16
Flash-Next with `-p P -n N -r 1 -t 1 -ngl 0 --no-warmup --progress -v`
(defaults P 4, N 2; `NAME 0 2` measures generation alone), the library
from `LIB` copied as `lib-NAME.so`, fresh workers on `PHI512_CARDS`
([`common.md`](common.md)). The output goes to the terminal and to
`$PHI_BENCH_DIR/NAME.log`; llama-bench's table at the end has the tokens
per second. With a terminal it waits for Enter before closing.

Baseline (the assembly preload at 2c8ceb3, card 0, 2026-10-14): pp4
0.0078 tok/s (511 s), tg2 0.0037 tok/s (547 s, 273 s a token). The
backend path (libggml_phi.so, not this preload) on the same model, one
engine with GPUs, host and cards: 13.14 and 13.36 tok/s decode, and 13.23
without the cards (2026-10-14, `/mnt/raid5/phi/bench/setups-2026-10-14`).
