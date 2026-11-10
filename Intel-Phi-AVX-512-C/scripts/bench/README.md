# scripts/bench: the preload on llama.cpp

The runs that measure the wrapper on its target: llama.phi's AVX-512 build
(`build-avx512f`) with the BF16 Qwen3.8 Flash-Next, under a preload
library, on the rack's cards. Results land in `$PHI_BENCH_DIR` (default
`/mnt/raid5/phi/bench/c-YYYY-MM-DD`), each with the exact library it ran
(`lib-NAME.so`).

| script | what |
| --- | --- |
| [`lb-run.sh`](lb-run.md) | llama-bench, prompt and generation tokens per second |
| [`dbg-run.sh`](dbg-run.md) | llama-debug, every tensor's sum, under the preload or (`--native`) the AVX2 twin |
| [`dbg-cmp.sh`](dbg-cmp.md) | two llama-debug logs compared: the first op that drifts |
| [`common.sh`](common.md) | the paths, the library copy, the card guard, the worker restart |

A visible run on the desktop (the person watches llama.cpp's own log):

```
setsid konsole --separate -p tabtitle=lb1 -e env RACK_NOMUX=1 ~/.local/bin/rack -t \
    "bash '/mnt/raid5/phi/Intel Phi AVX-512 C/scripts/bench/lb-run.sh' lb1 4 2"
```

A host crash under the preload leaves nothing in the log: read the
kernel journal (`journalctl -k | grep segfault`).
