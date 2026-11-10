# common.sh: what the bench scripts share

Sourced, not run. Sets `root` (the checkout), `llama`
(`LLAMA_CPP_DIR`, default `/mnt/raid5/phi/llama.phi`), `model`
(`PHI_MODEL`, default the BF16 Flash-Next's first shard), `bdir`
(`PHI_BENCH_DIR`, refused when it has a space), `cards`
(`PHI512_CARDS`, default `0,1,2,3`), `HP` (huge pages per card, 2060)
and `R` (the resident pool's pages, 2048: the worker's `RES_MAX` caps it
there).

- `bench_lib NAME` copies the library under test (`LIB`, default
  `host/c/out/libphi512-c.so`) to `$bdir/lib-NAME.so` and prints that
  path: `LD_PRELOAD` splits on spaces, and the copy records which binary
  a run used.
- `bench_workers` stops when [`../card-users.sh`](../card-users.md) finds
  a process holding one of the cards (production's `llama-server`), unless
  `FORCE=1`; otherwise restarts each card's worker in slot mode:
  `PHI_VPU_HUGEPAGES=$HP PHI_VPU_ARGS="-S 1 -C $((n - 2)) -T 2 -R $R -e 0 -s 1000"
  phi-vpu.sh -c N start $n` with `n` the card's cores (`phi-vpu.sh cores`: 57 on a
  3120A, 61 on a 7120A).
