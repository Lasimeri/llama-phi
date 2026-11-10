# q4_0-stream.S

`phi_q4_0_row1p` streaming a weight buffer on one thread: rows of 20
superblocks (2880 bytes, a row of 5120 Q4_0 weights) against one
prepared activation row, the buffer written first so its pages are its
own. It prints, per case, tenths of a cycle per superblock. The cases
are the row kernel at eight prefetch settings (`q4_0-variants.S`, made
by `tools/q4_0-bench.sh` from the generator's output by rewriting the
two displacements), over consecutive rows and over every other row.

```
as --64 [--defsym BIG=8388608] -I card/vpu -I OUT_DIR -o q4_0-stream.o tools/q4_0-stream.S
```

`BIG` is the buffer, 64 MiB unless given. Built and run by
`tools/q4_0-bench.sh` (`tools/q4_0-bench.md`).

Measured on card 0 (a 7120A), 2026-11-09, `tools/q4_0-bench.sh` run
twice (`OUT_DIR/q4_0-stream.log`), a buffer of 4 KiB pages, cycles a
superblock:

| first-level prefetch, second-level prefetch | consecutive rows | every other row |
| --- | --- | --- |
| none, none | 248.6, 249.9 | 311.9, 314.4 |
| +576, +2304 (as built) | 202.2, 203.6 | 226.7, 228.7 |
| +288, +1152 | 201.7, 202.7 | |
| +1152, +4608 | 221.8, 203.9 | |
| +1152, +8640 | 202.0, 204.5 | 289.2, 292.6 |
| +2304, +11520 | 203.8, 224.6 | |
| none, +2304 | 245.7, 247.1 | |
| +576, none | 195.6, 200.3 | |

The first-level prefetch is what hides the stream, at any of the
distances tried; the second-level one adds nothing to it. One case in
each pass is about 20 cycles above its other pass (221.8, 224.6), a
different case each time: run-to-run noise of that size exists here.

Every other row costs the kernel as built 24.5 to 25.1 cycles a
superblock more than consecutive rows. The kernel is told the jump to
the next row it will take and turns its prefetch pointer four
superblocks before a row's end, which is the +576 distance; a variant
that prefetches further ahead (+1152) has already run past the row by
then and pays 87 to 88 cycles. The service gives a thread consecutive
rows wherever it can.

## Several streams on a core

Built with an 8 MiB buffer, one to four processes on each of 40 cores
(`OUT_DIR/q4_0-many.log`; the median of the streams, both runs):

| streams a core | cycles a superblock, each | superblocks per 1000 cycles, a core |
| --- | --- | --- |
| 1 | 201.3, 200.5 | 4.97, 4.99 |
| 2 | 560.5, 560.5 | 3.57, 3.57 |
| 3 | 639.2, 638.4 | 4.69, 4.70 |
| 4 | 721.0, 719.6 | 5.55, 5.56 |

Two streams on a core do less together than one alone. Each process
has its own prepared row of 23040 bytes, so two put 46080 bytes of
activations through a 32 KiB first-level cache and every superblock's
eighteen lines of activations come from the second level. This is the
reason the worker keeps one prepared row a core, written by both of
its threads (`card/vpu/rows.md`, "Q4_0 against one row"), and it is
not the worker's case: there the two threads read the same 23040
bytes. In the first run one of the 40 single streams took 549.6 (the
others 199.6 to 206.6); the cause is not established, and the second
run's slowest was 207.9.

## With the activations free

An earlier build of this program did not write its prepared row. An
unwritten row is one zero page shared by every process, 4 KiB of cache
where a real row is 23040 bytes, so those runs measured the stream
with the activations costing nothing (the same day, two passes):

| | consecutive rows | every other row |
| --- | --- | --- |
| none, none | 237.3, 237.2 | 320.6, 321.6 |
| +576, +2304, before the jump argument | 186.4, 186.4 | 263.4, 263.7 |
| none, +2304 | 230.6, 230.7 | |

A real row therefore costs one stream about 16 cycles a superblock
(186.4 against 202.2 to 203.6). As many such streams at once, one a
core: 8 streams 187.4 each, 28 streams 184.2, 57 streams 184.3. On 40
cores: one a core 185.6 each (5.39 superblocks per 1000 cycles a
core), two 200.7 (9.96), three 275.3 (10.90), four 368.4 (10.86).
The worker's two threads of a core, sharing one real row, lie between
this table and the one above; neither run is that case.
