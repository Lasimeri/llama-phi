# matmul.S and matmul.inc: the matrix service

The card as a matrix-multiply engine for the ggml backend
(`host/asm/ggml-phi`), in assembly: the request kinds `VPU_K_UPLOAD`
(keep a tensor by id), `VPU_K_MATMUL` (d = a . b^T), `VPU_K_MATMUL_ID` (a
mixture of experts, one matrix per column), `VPU_K_MATMUL_MORE` (several
matrices by the same activations in one request), `VPU_K_FFN` (a
feed-forward block's share in one request), `VPU_K_DOWNLOAD` (a resident tensor's
bytes back to the window) and `VPU_K_FREE`, with the
descriptors of `vpu_matmul.h` (their offsets in `proto.inc`). It replaces
`vpu_matmul.c`; the row loops are `rows.S`, the kernels `kernels.S`. The
behaviour is the C worker's (`vpu_matmul.md` documents the design and
its measurements: the request kinds, how a request's data crosses, the
grouping, the fused pull, two threads per core); this document says
what is where in the assembly and what its gates were.

- **Layouts** (`matmul.inc`): a mapping (pointer, length, huge flag: the
  unmap must match the length), the cache of uploaded tensors by id
  (4096 entries), a growable buffer, a column group, a job (one matrix by
  the request's columns), the fused pull, a request of several matrices,
  a feed-forward job, a pooled copy, and the constants of the loops.
- **Copies** (`dma_pull`, `copy_pool`, `pull_data`, `push_data`): since
  2026-10-08 a request's ids and activations come in one copy by the
  card's own DMA channel (`dma_pull`: `cdma_submit` and `cdma_wait` of
  `cdma.S`, the data then read from the channel's landing area, `r_acts`)
  whenever the channel is up and the data with `SLACK` fits the landing
  area (`CD_LAND_MAX`, about 2 MiB); otherwise, and for every result,
  they cross through the worker's mapping of the
  window in whole 64-byte vectors (`phi_copy64`) up to `MAP_POOL_MAX`
  (one thread up to `PULL_ONE_MAX` or `PUSH_ONE_MAX`, the pool above),
  else through the block device (`vpu_pull`, `vpu_push` in `worker.S`).
  A copy the engine fails closes the channel (said once in the log) and
  the request, like every later one, takes the other path.
  `ctrl_read` takes a descriptor out of the uncached control area in
  64-byte loads. The worker's `-m 0` (`map_small`) sends everything the
  engine does not carry through the block device.
- **Memory**: `big_alloc` from 2 MiB huge pages when the card has them
  (else 4 KiB pages), every buffer with `SLACK` past its data for the
  unaligned load pairs; `grow` for the streaming buffers kept between
  requests; the cache's `cache_find` and `cache_drop`.
- **The constants** the quantized kernels share (`g_consts`, `C_*` in
  `kernels.md`) are data here, in `.float` and `.long`, where the C
  computed them at start; **the kernel table** `g_fmt` names each
  format's block bytes and its six kernels. Q4_0 (2026-11-09) is row 8
  of the table (144 bytes, `phi_q4_0_*`) and the 8.0 at offset 44 of the
  constants, which was padding.
- **The float weight types** (`rows_slice`, `rows_slice_id`): a row
  against four activation rows (`phi_dot4_*`) or one (`phi_dot_*`), the
  row's last `k % 16` elements added to the sixteen partial sums by one
  masked expand-load and multiply (`tail_dot`), the sums added on the
  vector unit (`rows.md`, "The sums"). A request of several
  matrices (`VPU_K_MATMUL_MORE`) with a float base takes pieces of the
  base's own type only (`r_fmore`; a quantized base takes quantized
  pieces of any type, as before) and runs them by `rows_slice_multi_f`:
  each piece in turn as the single-matrix float paths take it, every
  thread its slice of each piece's rows, no groups. Added 2026-10-09:
  until then the request wanted a quantized base, and a bfloat16 gate
  and up pair from the backend was rejected (`matmul-check` now covers
  it for every type).
- **The SwiGLU** (`swiglu_range`): a range of h in whole vectors with
  its ends under lane masks (`phi_swiglu*`, `kernels.md`), so every lane
  is stored by exactly one thread; also the conformance diagnostic
  (`MM_SWIGLU`) that `matmul-check` runs first.
- **The feed-forward request** (`ffn_run`, `ffn_slice`): gate and up
  rows for every column, their SwiGLU into the card's own intermediate,
  then the down projection over it, as two dispatches.
- **The requests** (`vpu_matmul_run`): the C signature (control area,
  kind, threads, verbosity, and pointers for the compute, pull and push
  times and the slices run), every check the C made in the same order,
  the ids checked once before any thread runs (-1 allowed: a column
  without an expert), the activations by the engine (`dma_pull`, waited
  for before the ids check) or else, for a small request, pulled inside
  its compute dispatch (`fused_setup`), the results pushed, the stage
  marks for the worker's trace. The request's variables are statics: only the
  dispatcher runs this. At `-v -v` a line at the start of each request
  names it (kind, type, shape) and one at the end gives its times.

- **The probe** (`MM_PROBE`, the diagnostics behind `matmul-check
  --probe`): what each kernel instruction produces (`phi_probe`), then
  the rates the C measured in the same order and units, as 8-byte times
  after the probe's 512 bytes: this thread's issue and streaming rates
  (`phi_bench`), the Q4_K and Q5_K kernels on one L1-resident superblock
  (`probe_kernels`, an aligned frame for the table and accumulators), a
  dispatch of nothing, the pool's read bandwidth and issue rate
  (`bench_pool`) and the eight-row kernel across it (`kernel_slice`,
  `bench_kernel_pool`), and the link both ways at a token's and a
  prompt's sizes (`probe_rounds` over `copy_pool`, `vpu_pull`,
  `vpu_push`; the C's `memcpy` measured as `rep movsq`, which is what it
  compiled to). The driver's text over the assembly worker differs from
  the C worker's only in the numbers.
- **The stage marks** (`MARK 1..5`): the time-stamp counter at the
  request's stages, for the worker's `-t` trace (`worker.md`); at
  `-v -v` a fused request also prints a `phases` line (`log_phases`):
  the longest wait past the copy's completion, the longest activation
  warm-up, the longest rows phase and its thread (`vpu_phase`,
  `rows.md`), which is what found the slow mode of the fused copy.
- **`q40_prep_ok`** (2026-11-09): set to 1 around the dispatch of a
  request of one matrix against one activation row with no mixture
  (label 45, `MM_N` 1 and `r_mixture` 0), 0 otherwise. It tells
  `rows_range_q` that every thread of the dispatch prepares the same
  activation row, which is the condition under which the threads of a
  core may share one prepared buffer (`rows.md`, "Q4_0 against one row").
- **`map_threads`** (2026-11-09): a request's threads are clamped to
  the pool plus the dispatcher and to four a core, where the clamp was
  two a core. The pool the worker is started with decides (its thread
  `t` is on core `t` mod the cores, `worker.S` `create_thread`: 121
  threads are two a core on a 7120A, 181 three, 241 four). Measured on
  card 0, Q4_0, float16 activations, each pool with as many slices,
  best of 15, the rounds in order (`matmul-check --only q4_0 --act 1
  --threads N --repeat 15`):

  | shape | two a core (121) | three (181) | four (241) |
  | --- | --- | --- | --- |
  | 17408 x 5120, one row | 0.640, 0.689 ms | 0.719, 0.700 ms | 0.783 ms |
  | 5120 x 5120, one row | 0.254, 0.246 ms | 0.261, 0.258 ms | 0.329 ms |
  | 17408 x 5120, eight rows | 2.720, 2.410 ms | 2.406, 2.297 ms | 2.125 ms |
  | 5120 x 5120, eight rows | 0.671, 0.685 ms | 0.746, 0.735 ms | 0.702 ms |

  A third thread costs 2 to 12 % at one row and gains 5 to 12 % on the
  large shape at eight; two a core stays the start scripts' default.
  The row kernel alone, as streams pinned to 40 cores with the
  activations costing nothing (an unwritten row, `tools/q4_0-stream.md`),
  gives a core 5.39 superblocks per 1000 cycles at one stream, 9.96 at
  two, 10.90 at three and 10.86 at four: two threads already issue
  nearly every cycle. With a prepared row of its own for every stream
  (`tools/q4_0-bench.sh`, two runs) it is 4.97 to 4.99, 3.57, 4.69 to
  4.70 and 5.55 to 5.56: the activations of two streams no longer fit
  the core's first-level cache. The worker's threads of a core share
  one row and lie between the two.

## Gates

Step 1c (2026-09-30, card 1 the assembly worker, card 0 the C worker,
the same `phi-vpu matmul-check` driver): every format's section passes
(`rows.md` lists the cases); the full run's output is diffed against
the C worker's after the timing fields are stripped. Defects met on the
way are in `rows.md`; the fused feed-forward's down projection is the one
that was here (a matrix pointer read through a clobbered register).

## Results by the engine (2026-11-06)

`push_data` sends every result through the card's DMA channel first,
whatever its size: pieces of at most 983,040 bytes (15,360 lines, under
the descriptor's length field and the landing area) are copied into the
landing area in card memory (`phi_copy64`), then one `cdma_push` to the
window and `cdma_wait`; `stat_dma_push` counts them. A refusal or a
timeout sends the rest the way it went before (the mapping or the block
device), and a failed copy closes the channel, said once in the log.
Lengths round up to whole lines: the window's result area is whole 4 KiB
blocks and the card's buffers keep `SLACK` past their data. Gated by
`phi-vpu matmul-check` on the 7120A (every format ok, mixtures, up to
four matrices, the fused feed-forward) with channel 7 up throughout.

## Weights straight from host RAM (`engine_pull`, 2026-11-10)

A matrix that arrives in the window (an upload, `K_UPLOAD`, or a multiply
whose matrix is named by `a_off`, the streamed weights of the ggml
backend) was read through the block device: `vpu_pull`, a `pread` that
the card's kernel hands to the host daemon's DMA channels, so every
matrix waited on host CPU time. `pull_weights` now tries the card's own
channel first (`engine_pull`), and `vpu_pull` only when that does not
carry it:

| step | how |
| --- | --- |
| the destination | the buffer the matrix is multiplied from (`g_a`, or the cache entry's mapping), which `big_alloc` puts on huge pages when the card has them (`MAP_HUGE`); on 4 KiB pages the block device serves as before |
| a piece | ends at a 2 MiB boundary of the destination (a huge page is one run of frames) or at a descriptor's 16383 lines (`ENG_PIECE`) |
| its frame | the page is read once (a private huge page gets its frame at the first fault; a read, so no dirty line of a core lies under the engine's write), then one entry of `/proc/self/pagemap` (`eng_pm_fd`, opened at the first pull) |
| the copy | `cdma_submit_phys(window offset, card physical address, bytes)` (`cdma.S`): the engine writes the window's bytes into the buffer's own frames; no core of the card copies, nothing of the host's takes part but its memory |
| completion | the pieces are posted back to back (the ring holds 63) and the last is waited for (`cdma_wait`) |

Any refusal (the channel down, an address or an offset that is not a
whole line, no frame, a copy refused or not completed) answers -1 after
the pieces in flight have landed, and the caller pulls through the block
device. The first matrix of each kind is said once in the worker's log.
The engine's writes are coherent with every core's cache (measured on a
7120A: a buffer every thread had in its L2, refilled by the engine, read
back with no stale word).

Measured on card 0 (a 7120A):

- `phi-vpu matmul-check --threads 114 --m 17408 --k 5120 --repeat 5`:
  every one of its 15 format cases ok against the host reference, the
  worker's log saying the matrices came by the engine and no line for
  the block device. Every case uploads its matrices, so the check is a
  check of the pulled bytes.
- The streamed layout of the ggml backend (`PHI_GGML_STREAM=1`, a 27 B
  Q4_0 model, 8.1 % of every matrix's rows pulled at every multiply,
  12,352 requests a 64-token request, sampled from the host): the
  card's pull time 27.780 s through the block device, 15.818 s by the
  engine; its compute 1.847 and 1.852 s. Generation 1.559, 1.567, 1.536
  tok/s before, 2.155, 2.203, 2.127 after. The host alone, the card left
  out, gives 2.267 and 2.259: with nothing resident the same rows cross
  the link at every token, and no transfer method removes that.
- An upload of 10.49 GB of a 35 B Q8_0 model's rows: the server up 44 s
  after its start, the uploads included.

Not converted: the other `vpu_pull` sites (the fused feed-forward's
descriptors, the mixture paths' small pulls), which move kilobytes.
