# dbg-run.sh: every tensor of a forward pass

`dbg-run.sh NAME` runs llama-debug of `build-avx512f` under the preload
on the cards (fresh workers, [`common.md`](common.md)) with the prompt
"The cat sat on" (4 tokens), `-t 1 -ngl 0 -c 256 --no-warmup
--verbose`: every tensor's name, op, a few values and its sum, into
`$PHI_BENCH_DIR/NAME.txt`; the terminal shows the tensor lines and the
preload's progress. `dbg-run.sh NAME --native` runs the AVX2 twin
(`build-phi512-cpu`) with no preload and no cards: the reference.
[`dbg-cmp.sh`](dbg-cmp.md) then finds the first op whose sum drifts.

The assembly preload's first complete pass took 1336 s (2026-10-14,
dbg-wrap2); its router top-10 matched the twin in layers 0 to 22 and
differed in 8 of 48, the final logits off by up to 0.35.
