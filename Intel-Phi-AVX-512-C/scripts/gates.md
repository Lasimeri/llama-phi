# gates.sh: the preload's regression gates on a card

Every translator and runtime bug the wrapper met on llama.cpp
(2026-10-13 and 14) has a check in one of these programs. `gates.sh` runs
them through a preload library on one card, each with a fresh worker, and
fails when any count misses. The gates are not part of `make check`: they
restart a card worker.

```
scripts/gates.sh                                  # host/c/out/libphi512-c.so on card 0, all six
scripts/gates.sh --lib host/c/out/libphi512-c-asm.so --only "region fforms"
scripts/gates.sh --card 2 --force                 # restart card 2's worker under whatever maps it
```

| gate | program | built with | passes when |
| --- | --- | --- | --- |
| region | [`../tools/avx512-region-test.c`](../tools/avx512-region-test.md) | `-O2 -mprefer-vector-width=512`, AVX512F only | `PASS` and 8 `ok` lines |
| fforms | [`../tools/avx512-f-forms-test.c`](../tools/avx512-f-forms-test.md) | `-O1`, AVX512F only | `PASS` and 42 `ok` lines |
| narrow | [`../tools/avx512-narrow-test.c`](../tools/avx512-narrow-test.md) | `-O1 -mavx512f -mavx512vl -mavx512dq -mavx512bw -mf16c` | `PASS` and 94 `ok` lines |
| libm | [`../tools/avx512-libm-test.c`](../tools/avx512-libm-test.md) | `-O2 -mavx512f`, with `PHI512_TRAMP_ANY=1` | `0 of 4096 differ` |
| router | [`../tools/ggml-router-test.c`](../tools/ggml-router-test.md) | linked against `build-avx512f`, 8 rounds | `PASS` and 8 rounds with soft_max and argsort ok |
| op | [`../tools/ggml-op-test.c`](../tools/ggml-op-test.md) | the same source linked against `build-avx512f` (under the preload) and `build-phi512-cpu` (natively) | the two outputs identical |

The expected counts are the ones the assembly preload reached at
2c8ceb3 (2026-10-14, card 0); a change to a test changes its row here
and in the script.

## What it does

1. Refuses when another process maps the card's window
   (`/dev/shm/phi-hostmem`, or `phi-hostmem-N` for card N): restarting
   the worker would take the card from it (production's `llama-server`
   holds all four). `--force` goes ahead anyway.
2. Copies the library to `$PHI_GATES_DIR/bin/lib.so` (default
   `~/.cache/phi512-c-gates`): `LD_PRELOAD` splits on spaces, and the
   checkout's name has them.
3. Builds the six programs (gcc, the flags above; the ggml tests need
   `LLAMA_CPP_DIR`, default `/mnt/raid5/phi/llama.phi`, with both builds)
   and runs the AVX2 twin's op-test natively for the reference.
4. For each gate: `phi-vpu.sh stop`, then
   `PHI_VPU_ARGS="-S 1 -C $((cores - 2)) -T 2 -R 128 -s 1000" phi-vpu.sh start $cores`
   (`cores` from `phi-vpu.sh cores`: `-C 55` and 57 on a 3120A, 59 and 61 on a 7120A),
   then the program with `PHI512_CARD_ATOMICS=1 PHI512_CARDS=N
   LD_PRELOAD=lib.so` under a 900 s timeout.
5. Restores the worker the card had before (`phi-vpu.sh config`: its huge
   pages and arguments), or leaves it stopped when there was none.
6. Prints one row per gate (counts, exit status, verdict, seconds) and a
   summary line; exit 1 when a gate failed. Every program's output, the
   build logs and the worker starts are in
   `$PHI_GATES_DIR/logs-YYYYMMDD-HHMMSS/`.
