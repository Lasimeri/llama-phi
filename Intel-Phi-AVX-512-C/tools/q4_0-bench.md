# q4_0-bench.S, q4_0-bench.sh

Two card programs that time the Q4_0 one-row kernels outside the
worker, and the script that builds and runs them. They exist because
the service's own numbers could not say why a kernel with a third
fewer instructions was no faster (`card/vpu/rows.md`, "Q4_0 against one
row": the answer was in the worker's trace, but these ruled out the
instructions and the memory first).

```
tools/q4_0-bench.sh OUT_DIR [CARD]
```

The card must have no other program on it (a parked worker is fine; a
serving one is not). The script generates the kernels
(`tools/q4_0-kernels.c`), builds, puts three binaries in the card's
`/tmp`, runs them through `phi`, and removes them.

## q4_0-bench.S

One thread, 200000 calls of each case between two `rdtsc`, every
operand in the first-level cache (a kernel's stride argument is 0, so
its prefetches hit the lines it is working on). It prints
`name cycles` lines; the name's underscores stand for spaces. Static,
no C library: `_start`, `write` and `exit` by system call.

Measured on card 0 (a 7120A), 2026-11-09, the script run twice
(`OUT_DIR/q4_0-bench.log`; a run takes 3.5 s in all):

| case | cycles a call |
| --- | --- |
| `phi_q4_0_1`, one superblock (114 vector operations) | 306, 306 |
| `phi_q4_0_row1p`, one superblock (72 and six prefetches) | 204, 204 |
| an empty call with its five arguments | 11, 11 |
| 16 unpack loads `{uint8}` (four low halves, then four high) | 40, 40 |
| 8 low halves only | 24, 24 |
| 8 aligned `vmovaps {uint8}` | 24, 24 |
| 8 `vbroadcastss {float16}` | 24, 24 |
| 8 `vmulps {1to16}` and 8 `vrndfxpntps` | 40, 40 |
| 8 `vmulps {1to16}` | 24, 24 |
| 16 multiplies and FMAs with memory operands | 40, 40 |
| 8 `vsubps {4to16}` | 24, 24 |
| 8 register FMAs | 24, 24 |
| 8 prefetches of lines already in the cache | 24, 24 |

The row kernel measured 196 and 201 before it had its jump argument
(the compare and the add that steer the prefetch pointer are the
difference), and `phi_q4_0_1` 324 in one earlier run.

Eight vector operations of any kind are about 13 cycles past the call:
a lone thread issues one about every other cycle, and nothing here is
slower than that.

## The streams

`q4_0-stream.S` (its own document) is run alone for the prefetch
distances and, built with an 8 MiB buffer, as one to four streams on
each of 40 cores, pinned with the card's `taskset` to hardware threads
in the pool's order (0, 3, 1, 2). Each stream is a process with a
prepared row of its own, which the worker's threads of a core do not
have (`q4_0-stream.md` has the numbers and what they do and do not say
about the worker).
