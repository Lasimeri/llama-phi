# rows.S: the row loops of the matrix service

What a thread of the pool does with its slice of a request: which rows
are its own, how the request's columns are grouped so that a weight row
is read once per group, what it asks the cache for before it starts, its
share of a fused pull of the activations, and the loop that multiplies
its rows with the kernels of `kernels.S`. The design is the C worker's
(`vpu_matmul.md`: "Grouping a mixture's columns by expert", "One token's
multiply, and whose lines the pool reads", "Two threads per core");
this document says what the assembly keeps and what it must reproduce
exactly.

- `slice_rows`: rows `i0..i1` of slice `s` of `n`: with `ncores` slices
  (the card's cores, `worker.S`: 57 on a 3120A, 61 on a 7120A) every
  core has one; with twice that (two threads a core) the slices `s` and `s + ncores`
  share a core and take alternate rows of the same chunk, so the
  activation block they read is one and the same in their shared L1.
- `groups_plain`, `groups_mixture`, `groups_own`: the column groups
  (`struct group` in `matmul.inc`: a matrix, up to eight activation rows
  as pointers, their destination columns, the count 1, 4 or 8). An
  ordinary multiply's are its consecutive columns; a mixture's are the
  columns that chose each expert (a counting sort), so an expert's rows
  are read once per group rather than once per column. A request of a
  few columns (`OWN_GROUPS_MAX`) has each thread build the groups it
  reads in its own region (`groups_own`, an insertion sort up to 32
  columns, the counting sort above), rather than the dispatcher one
  table for all, whose lines had been read by every core the request
  before. An id of -1 is a column without an expert on this card (the
  host's whole-expert placement): no group.
- `warm_slice`: a small slice's whole working set asked for before it
  runs (`vprefetch1`: the card's cores are in order and every cold line
  costs its whole miss); the table and the activation rows are the same
  lines for every core, so each starts on a different line of them.
- `copy_share`, `fused_setup`, `copy64_nt`: a small request's
  activations copied inside its compute dispatch by the first cores'
  threads, since 2026-10-08 only when the card's DMA channel does not
  carry them (`cdma.md`: the channel down, or the request past its
  landing area) (at least `PULL_LINES` lines each: a row copied a line or two
  per core and then read by all 57 was seven times slower than one
  written by one core), counted so the others can wait for the last
  copy. The copy's stores are the no-read, non-globally-ordered kind
  (`vmovnrngoaps`, ordered by the locked add that counts the copy): a
  plain store leaves the line dirty in the copying core's L2, from where
  fifty-six readers then take it one at a time, and the request is slow
  or fast by whether the lines happened to be written back before the
  readers came (below, "The slow mode").
- `vpu_phase`: each slice's clock after the copy wait, after the
  activation warm-up and after its rows, for the `phases` line the
  matrix service prints at `-v -v` (`matmul.md`).
- `rows_range_q`: rows `i0..i1` of one quantized matrix by every group:
  chunks of `ROW_CHUNK` rows; the groups outermost so their activation
  rows are read from L2 for every weight row of the chunk; the
  superblock outermost within a group so its activation block and the
  accumulators stay in L1 while the weights stream; a kernel call per
  row and superblock with the seventh argument (the distance to the row
  this thread does next, which the kernels' prefetch follows) pushed
  around the call; then the sums. Two threads on a core divide an
  eight-column group by columns (each walks every row, taking four of the
  eight) rather than by rows, so they read the same weight bytes.
- `run_pieces`, `rows_slice_q`, `rows_slice_multi`: a slice's rows of one
  matrix, or of each matrix of a request of several, cut as one.

## Q4_0 against one row: the prepared row (2026-11-09)

A Q4_0 group of one column does not go superblock by superblock. Its
activation row is rewritten once by every thread (`q4_0_prepare`: per
block `x0`, `y = x1 - 16 x0` and the four floats `p`, `kernels.md`,
"Q4_0, one row, prepared") and `phi_q4_0_row1p` then takes this thread's
weight rows a whole row a call (or a segment of each, below), the rows outermost: at one activation
row the whole prepared row stays in the first-level cache, so nothing
is gained by keeping a superblock's activations hot across the rows,
and a row is one sequential stream the prefetch can lead.

It is taken when all of these hold, and `phi_q4_0_1` with the
superblocks outermost runs otherwise:

| condition | why |
| --- | --- |
| `q40_prep_ok`: the request is one matrix against one activation row, no mixture (`matmul.S` sets it around the dispatch) | the buffer is shared, below |
| the row is at most `PREP_SB_MAX` superblocks (128: 32768 weights) | the core's buffer, 147456 bytes |
| the range still has `PREP_ROWS_MIN` rows (24) | the preparation is about 96 vector operations a superblock; 1024 x 5120, nine rows a slice, measured 5 % slower with it |

Two things decided the design, both found by measuring, not by
reading the code:

- **The dispatcher takes a slice of every request itself.** With a
  buffer for pool threads only, the dispatcher ran `phi_q4_0_1` and its
  slice set the request's time whatever the pool did (the worker's `-t`
  trace: `dispatcher's slice 1182.7` of a 1480.5 us dispatch): three
  kernels of 114, 77 and 77 operations measured alike. With the
  dispatcher on the prepared path one slice a core went from 1.81 to
  1.15 ms (17408 x 5120).
- **The buffer is the core's, not the thread's** (`prep_bufs`, indexed
  by `TLS_CORE`; the dispatcher has the last core to itself). Two
  threads of a core with 23040 bytes each put 46080 bytes through one
  32 KiB cache: 1.48 ms at two slices a core where one took 1.15. With
  one buffer a core: 0.67 ms. Both threads fill it with the same bytes
  and each reads it only after its own fill, so a load sees the row
  whichever of the two stores it follows; no lock. That argument needs
  every thread of the dispatch to prepare the same row, which is what
  `q40_prep_ok` asserts. The same effect outside the worker
  (`tools/q4_0-stream.md`): the kernel as one stream a core costs 200.5
  to 201.3 cycles a superblock, as two streams a core with a row each
  560.5 apiece, less work a core than one stream.

- **A longer row goes in segments, and half-size ones.** A prepared row
  of more than 5120 weights does not fit the cache (17408 weights are
  78336 bytes), so its columns are walked in segments, the segments
  outermost and the thread's rows inside; the kernel adds to the row's
  accumulator, and its prefetch is told the jump from the end of one
  row's run to the start of the next row's (`phi_q4_0_row1p`'s sixth
  argument). With segments of 20 superblocks (23040 bytes, `PREP_SEG`)
  5120 x 17408 gained nothing (1.054, 1.092, 1.118 ms against 1.054,
  1.122, 1.034), and the trace said why: the dispatcher, alone on its
  core, took 627 us for its slice and the longest pool slice 891. The
  two threads of a core are on different segments most of the time,
  two times 23040 bytes in one cache. With segments of 10
  (`PREP_SEG_LONG`, 11520 bytes: two fit) it is 0.889, 0.866 ms
  against 1.043, 1.026, 1.037 (the two workers deployed in turn, 120
  slices), and a row of one segment is unchanged (17408 x 5120: 0.652,
  0.668 against 0.662, 0.641, 0.636).

`rows_range_q` remembers in its frame the row it prepared (`RQ_PREP_X`),
so a call prepares once however many chunks it walks.

## The sums: the vector unit (2026-11-09)

A result is the sixteen partial sums of its accumulator vector added
together, plus, for a Q8_0 row whose length is not a multiple of 256, the
row's last blocks, and for a float row its last `k % 16` elements. All
three are done on the vector unit; the worker has no x87 instruction.

| step | how |
| --- | --- |
| a Q8_0 row's last blocks (`q8_0_tail`) | per 34-byte block: the scale's half up-converted into one lane and stored to the scratch, the 32 signed bytes as two vectors (`{sint8}` unpack pairs), `w0 * x0 + w1 * x1` against the activations (float32, or `{float16}` at a 32-byte stride), and that times the scale, broadcast from the scratch, into the row's accumulator |
| a float row's last `k % 16` elements (`tail_dot`, `matmul.S`) | one masked expand-load of the weights (float32, `{float16}`, or `{uint16}` shifted left 16 for bfloat16) and one of the activations: the low `k % 16` lanes take that many consecutive elements, the others stay zero, nothing past the row is read; their product added to the sixteen partial sums |
| the sixteen partial sums (`SUM16`, `matmul.inc`) | each float a `{1to16}` broadcast operand of an add, four chains of four (lanes `i`, `i + 4`, `i + 8`, `i + 12`), then two levels: six adds deep |
| the store | the vector to a 64-byte aligned slot (the kernels' scratch, or the float path's own `out`), lane 0 moved to the result as four integer bytes |

Both tails go into the accumulator before it is summed, so a result is
one reduction whatever the row's length.

What this replaced, and what it cost: until 2026-11-09 the sums and the
tails were in the x87 unit, read off the C worker's compiled code
(`fldz`, sixteen `fadds` in lane order, the tail in `fimuls`, one
`fstps`), so that a result was the same float to the bit as under the C
worker: in x87 a chain of additions stays in the 80-bit register and
rounds once. The vector reduction adds in float32 and in another order,
so a result now differs in its last bits from every run before it; the
check's tolerance (`matmul.md` of `phi-vpu`) is unaffected.

Measured on card 0, Q4_0, float16 activations, 114 threads, the card's
compute time, the two worker binaries deployed in turn twice, best of
15 each (`phi-vpu matmul-check --only q4_0 --act 1 --m M --k K
--repeat 15`):

| shape, activation rows | x87 sums | vector sums |
| --- | --- | --- |
| 17408 x 5120, 1 | 1.063 ms | 1.069 ms |
| 17408 x 5120, 8 | 2.615 ms | 2.529 ms |
| 17408 x 5120, 64 | 22.320 ms | 21.597 ms |
| 61440 x 256, 1 | 0.277 ms | 0.280 ms |
| 61440 x 256, 8 | 1.270 ms | 1.179 ms |
| 61440 x 256, 64 | 11.485 ms | 10.123 ms |

At one activation row the two are equal within the runs' spread (the
medians cross); from eight rows the vector sums are 3 to 12 percent
faster, more where a row is short and its reductions are a larger share.

## Registers and frames

The pool calls a slice function as `fn(arg, slice, nslices)` in `rdi`,
`esi`, `edx` (System V), and the kernels are called as C would call
them (`kernels.md`). `rows_range_q` keeps its frame (the accumulators,
the scratch table, the row pointers, its variables) 64-byte aligned
below `rbp`, which is not the standard frame pointer here but the
aligned base, with the pre-alignment stack pointer saved in the frame.
Its row loop carries the superblock pointer and the accumulator pointer
in registers the kernels keep (`r14`, `r15`, the count in `r13`) and
adds the row stride to them; the first version recomputed both with
three multiplies and a dozen frame accesses per kernel call, which on
this in-order core was about 100 ns on top of a 280 ns kernel (the
2048-row mixture shape: 1.03 ms against the C's 0.90, now 0.86).
`run_pieces` takes its last two arguments on the stack as the C did.
Everything is written for the card's scalar core: no SSE, no `cmov`.

## The slow mode (2026-09-30)

With the C worker's plain stores in the fused copy, the one-row mixture
shape (128 x 2048, eight experts) computed in 95 us at best but 190 us at
the median, where the C worker's median was 105 to 165 us; the per-slice
phases showed every thread's row loop three to four times slower in the
slow requests, with no dependence on which experts were drawn, and the
mode absent with `-m 0` (the activations arriving by DMA into memory).
The activation row is written by the eight copying threads and read by
every other core right after: a line dirty in a copier's L2 is served
from there, one reader at a time, and whether the request is slow is
whether those lines were still dirty when the readers came. The no-read
stores leave nothing in the copier's L2; the median fell to 93 us over
two restarts (270 of 300 requests under 100 us), the best unchanged.
The C worker has the same exposure with a different luck of timing.

## Gates

Step 1c (2026-09-30, card 1, the C worker on card 0): `phi-vpu
matmul-check` passes every case for f32, f16, q4_K, q5_K, q6_K, q8_0 and
iq4_xs: the plain multiplies at 1, 4, 8 and 13 columns, the mixtures
(including two with every third column at -1 through both groupings),
up to four matrices in one request, the fused feed-forward with both
intermediates; the full run's text is compared with the C worker's in
`matmul.md`. Defects met: `groups_own`'s many-column path returned a
stale register as the group count (the walk ran off the table on the
first 64-column mixture); the down projection of a feed-forward request
took its matrix from a register two calls had clobbered (a general
protection fault in the one-row Q6_K kernel on a garbage superblock
address).
