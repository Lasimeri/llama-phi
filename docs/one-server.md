# One server: context on the GPU, matrices on the card

Measured 2026-11-09 on the hardware of [results.md](results.md) with an
RTX 3090 Ti beside the card, on the 27 B Q4_0 model of
[q4_0.md](q4_0.md). One llama-server instead of that document's two: no
slot file, no second copy of the model in VRAM, no cold decode server.
Every number below is one request; where runs were repeated every run
is given. Generation went from 2.42 to 2.54 tok/s in this layout to
3.22 to 3.28 by three changes, each measured on its own below: a
multiply issued ahead of its node, a kernel for one activation row,
and long rows walked in half-size segments.

## Placement

```
╔═══════════════════════════════ desktop: Ryzen 7 5800X, 32 GB ════════════════════════════════╗
║  ┌────────────────────────────────────────────────────────────────────────────────────────┐  ║
║  │ one llama-server :8095 (scripts/one.sh): -ngl 99, weights by -ot, libtridma.so         │  ║
║  └───────────────┬────────────────────────────────┬────────────────────────────┬──────────┘  ║
║                  │ context, attention             │ small multiplies           │ MUL_MAT     ║
║                  ▼                                ▼                            ▼             ║
║  ╔════════════════════════════════╗  ┌──────────────────────────┐  ╔════════════════════════╗║
║  ║ RTX 3090 Ti, 10.5 GB in use    ║  │ 8 threads, no matrix rows│  ║ Xeon Phi 7120A         ║║
║  ║ 128K q8_0 context              ║  │ ssm_alpha, ssm_beta      │  ║ 12.99 GB, 100 % of the ║║
║  ║ layers 60-63 + output head     ║  │ router: card rows back   │  ║ rows of layers 0-59    ║║
║  ║ prompts over 31 tokens:        ║  │ to the GPU at 3.32 GB/s  │  ║ 120 threads, no x87    ║║
║  ║ 315 to 319 tok/s (2nd prompt)  ║  │ next multiply sent ahead │  ║ decode 3.22-3.28 tok/s ║║
║  ╚════════════════════════════════╝  └──────────────────────────┘  ╚════════════════════════╝║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝
```

`-ngl 99` gives every layer's context and attention to the GPU; `-ot`
then says where each weight tensor lives, first match first:

```
-ot ^output\.weight$=CUDA0,^blk\.(60|61|62|63)\.=CUDA0,.*=CPU
```

The matrices are 14,747,996,160 bytes and the card's budget is 13.1 GB.
The output head (1,042,944,000 bytes) and the last four layers
(856,565,760 bytes) go to VRAM, which leaves the card 12,848,486,400
bytes: every row of every remaining matrix fits, and the backend says so
at load:

```
ggml-phi: 12.8 GB of dense weights and 0.0 GB of experts offered to the cards: each keeps 100.0% of every dense matrix's rows and 0.0% of the experts' (0 expert tensors a step more), 12.99 GB
```

With all of a matrix's rows on the card the host has no share of a
multiply to compute. Prompts of more than 31 tokens are declined by the
card's backend and run on the GPU, their weights reaching it through the
tridma router ([router.md](router.md)).

```
PHI_VPU_HUGEPAGES=6437 scripts/worker.sh work
PHI_GGML_AHEAD=1 scripts/one.sh work MODEL.gguf start
scripts/request.sh out docs/design.md 64
scripts/one.sh work MODEL.gguf stop
```

`ONE_DRY=1 scripts/one.sh ...` prints the placement and the settings
without starting anything. `PHI_GGML_AHEAD=1` is the switch of the
section "A multiply issued ahead" below; `scripts/one.sh` passes it
through and does not set it.

## Measured

The prompt is `docs/design.md` with the chat template, 1,410 tokens, 64
tokens generated, prompt cache off (`scripts/request.sh`). The rates are
the server's own; its generation rate is 63 tokens over the time it
prints for 64.

| Layout | Rows on the card | Prefill, first prompt | Prefill, second | Generation | VRAM |
|---|---|---|---|---|---|
| two servers ([q4_0.md](q4_0.md)) | 13.09 GB, 88.5 % | 364.65 tok/s | | 2.605 tok/s | 21,839 MiB |
| one server, host keeps 11.5 % of the rows | 13.09 GB, 88.5 % | 170.56 to 172.39 tok/s | 329.68 to 330.69 tok/s | 2.410 to 2.518 tok/s | 8,757 MiB |
| one server, this placement | 12.99 GB, 100 % | 159.64 tok/s | 313.28 tok/s | 2.425, 2.451, 2.478, 2.541 tok/s | 10,475 MiB |
| the same, multiplies issued ahead | 12.99 GB, 100 % | | | 2.540, 2.619, 2.618, 2.623 tok/s | |
| the same, and the one-row kernel | 12.99 GB, 100 % | | 314 to 316 tok/s | 2.949, 2.990, 2.997, 2.995 tok/s | |
| the same, and long rows in half segments | 12.99 GB, 100 % | | 315 to 319 tok/s | 3.221, 3.228, 3.282, 3.277 tok/s | 10,525 MiB |

- The last row is the worker and the backend that
  `patches/intel-phi-avx-512-c.patch` builds. Its 64 tokens took
  19.558, 19.518, 19.198 and 19.228 s; the first request after a
  later restart of the same server gave 3.192 tok/s (19.734 s).
- The second prompt's prefill is about twice the first's: the router
  learns the order in which the matrices are asked for from the first
  prompt and reads ahead from the second on. A prompt streams 11.34 GB
  of the card's rows back at 3.32 GB/s, the card link's ceiling. The
  first prompt after that restart ran at 101.50 tok/s where the first
  prompt of the earlier layout ran at 159.64; one run each, and the
  cause of the difference is not established.
- The sets of runs were taken one after another, not interleaved.
  Neighbouring rows of the first four touch or overlap (2.541 against
  2.540); the last two steps are clear of the spread.
- The two-server request took 62.8 s wall (its decode server's first
  token alone was 34.24 s); a one-server request takes 23.6 to 34.2 s
  wall, 23.6 to 24.0 s in the last layout from the second prompt on.
- The card, per request of the last layout: 0 errors, busy 20.49 to
  20.60 s, of that 15.48 to 15.52 s computing (`tools/overlap.c`). In
  the layout of the third row it was busy 25.4 s and computed 20.2 s
  (23,763 requests: 22,142 matrix multiplies, kind 4, and 1,621
  downloads for the router, kind 32).
- At temperature zero the layouts of rows three and four give the same
  24 tokens and the same top-5 log-probabilities byte for byte, twice
  each. The one-row kernel adds a row's products in another order: the
  24 tokens are still the same 24, and the largest difference in a
  chosen token's log-probability is 0.01061.

## Where a token's time goes

About 305 to 310 ms a token now; the card computes for about 238 ms of
it (15.48 to 15.52 s over the 65 passes of a request). Before the
one-row kernel it was about 400 ms and 310 ms (20.15 to 20.23 s). The
rest is the host's and the GPU's steps between the card's multiplies
and the round trips themselves, about 340 a token.

One activation row (one token) is where the card's kernels are slowest.
With the kernels of [q4_0.md](q4_0.md) Q4_0 ran 126 to 159 GFLOP/s at
one row and 450 to 530 at eight, because decoding a weight cost nine
vector operations and a single row gave them one product to pay for.
The one-row kernel below reaches 264.5 GFLOP/s on 17408 x 5120
(0.674 ms).

## What reaches the card, in order

`PHI_GGML_GRAPH=400` prints the sub-graphs the backend is handed. A
layer of the model's linear-attention kind (45 of the 60 on the card):

| Step | Weight | Input | Sub-graph |
|---|---|---|---|
| 1 | `attn_qkv` | `attn_norm` | its own |
| 2 | `ssm_alpha` | the same `attn_norm` | its own (too small for a card) |
| 3 | `ssm_beta` | the same `attn_norm` | its own (too small for a card) |
| 4 | `attn_gate` | the same `attn_norm` | its own |
| 5 | `ssm_out` | the attention's result | its own |
| 6, 7 | `ffn_gate`, `ffn_up` | the same `attn_post_norm` | one, two requests |
| 8 | `ffn_down` | the SwiGLU's result | its own |

A layer of the attention kind (15 of the 60) has `attn_q`, `attn_v` and
`attn_k` on the same `attn_norm`, each its own sub-graph, then
`attn_output`, then the same feed-forward. Every request is one matrix:
the card saw no multi-matrix request (kind 8) in generation.

So in every layer several multiplies read an input that is ready before
the first of them is issued, and the graph still issues each only after
the previous one's result has gone to the GPU and come back.

## A multiply issued ahead of its node

`PHI_GGML_AHEAD=1` (with the offload; `host/asm/ggml-phi/glue.md` in
the patch). The card and the host no longer alternate strictly:

| Step | What the backend does |
|---|---|
| a multiply is handed over | it records, under the previous multiply's weight address, this one's weight, activation address and sizes (a table of 2,048 slots) |
| a multiply's result is back | if the multiply that followed this weight last time reads the same activation address, the activations are copied aside (up to 1 MiB) and that multiply is begun on the card at once |
| its node arrives | the weight, the activation address, the sizes and the activations, byte for byte against the copy, are compared; equal: nothing is sent, the result is awaited; not equal: the outstanding request is drained and the multiply runs as before |

A guess can only cost time, never a result: what is taken was computed
from the same bytes the node holds. A multiply whose input does not
exist yet (`ssm_out`, `attn_output`, `ffn_down`) is never begun ahead.

Measured, 64 tokens, four runs each, the sets one after the other:

| | Generation | Mean |
|---|---|---|
| in graph order | 25.979, 25.703, 25.423, 24.791 s | 25.474 s |
| issued ahead | 24.802, 24.055, 24.061, 24.019 s | 24.234 s |

4.9 % less time on the means; the slowest run ahead (24.802 s) is
11 ms behind the fastest in graph order. The card's compute time is the
same in both (20.04 to 20.09 s): what is saved is waiting. The output
is identical byte for byte (above), and the backend's counter line at
16,384 multiplies taken ahead reported none dropped.

Since 2026-11-10 a multiply that begins a pass is never begun ahead: the
one issued after a pass's last multiply held the window's lock until the
next token, and a prompt arriving first waited for ever for the router's
turn ([moe.md](moe.md), where a 35 B model's buffer layout exposed it).

## The host's threads

Per-thread CPU time over a 25.4 s generation (`scripts/threads.sh`): the
main thread 27.3 s (it also ran the prompt), seven more threads 25.0 to
25.1 s each, the router's two 3.9 and 3.2 s (the prompt). Eight threads
are busy for the whole generation.

- They were the same with the host computing 11.5 % of every matrix's
  rows (26.3 s each over 26.1 s) and with the host computing none, so
  they are not doing matrix rows: they wait.
- `--poll 0` (llama.cpp's own thread pool told not to poll) changed
  neither them nor the speed (2.425 tok/s).
- They are the OpenMP runtime's workers spinning between parallel
  regions: the server maps libgomp (llama.cpp is built with
  `GGML_OPENMP=ON`). With `OMP_WAIT_POLICY=PASSIVE` in the server's
  environment the process uses 1.20 to 1.22 cores of wall where it
  used 7.10, and the output stays identical byte for byte.
- That costs generation time: 25.291, 25.631, 25.735, 25.461 s against
  24.019 to 24.802 s spinning in the same layout, 5.3 % on the means
  (25.530 against 24.234 s). `scripts/one.sh` leaves the variable
  alone; six host cores or five percent is the operator's choice.

## The card's threads

Sampled on the card over 12 s inside a generation (`/proc/stat` twice):
of its 244 logical CPUs 121 are at 100 %, 122 at 0 to 3 % and one in
between, 49.7 % in all. That is the layout, not idleness: the matrix
pool has two of each core's four hardware threads (120 threads on 60
cores) and the dispatcher a core of its own. The 100 % includes the
pool's spin between requests; of that generation's 25.150 s the card
computed for 20.295 s.

More threads a core were measured and are not used. The worker's pool
takes up to four a core by the count it is started with
(`WORKER_THREADS=121`, 181 or 241 for two, three or four on 60 cores
plus the dispatcher; the request clamp was two a core and is four in
the patch), and `CARD_THREADS` gives `scripts/serve.sh` the slices to
ask for. Card compute time, Q4_0, one worker, as many slices as pool
threads, best of 15, the rounds in order:

| Shape, activation rows | Two a core | Three | Four |
|---|---|---|---|
| 17408 x 5120, 1 | 0.640, 0.689 ms | 0.719, 0.700 ms | 0.783 ms |
| 5120 x 5120, 1 | 0.254, 0.246 ms | 0.261, 0.258 ms | 0.329 ms |
| 17408 x 5120, 8 | 2.720, 2.410 ms | 2.406, 2.297 ms | 2.125 ms |
| 5120 x 5120, 8 | 0.671, 0.685 ms | 0.746, 0.735 ms | 0.702 ms |

End to end, 64 tokens: two a core 21.361, 21.067, 21.024, 21.033 s;
three a core (`WORKER_THREADS=181 CARD_THREADS=180`) 21.974, 21.396,
21.378 s, 1.5 to 3 % slower. A third thread helps only the large shape
at eight rows, which generation does not issue.

## The one-row kernel

All of it is in the patch: `tools/q4_0-kernels.c` generates the kernel
(`phi_q4_0_row1p`), `card/vpu/rows.S` holds the preparation and the
loops, and `card/vpu/kernels.md` and `card/vpu/rows.md` document them.

A Q4_0 byte `b` holds two weights, `lo = b mod 16` and
`hi = floor(b / 16)`, each meaning `(q - 8) d`. For one activation row,
with `x0` the activations of the low nibbles and `x1` of the high:

```
sum (lo - 8) x0 + (hi - 8) x1  =  sum b x0 + hi (x1 - 16 x0)  -  8 sum (x0 + x1)
```

The right side needs neither the low nibble nor the `- 8` per weight.
Each thread rewrites the activation row once a request into `x0`,
`y = x1 - 16 x0` and the block sums `p` (1152 bytes for 256 weights),
and the kernel then does, per block of 32 weights: the sixteen bytes
as floats (an unpack-load pair with `{uint8}` up-conversion), `hi` as
`b / 16` rounded down, `b x0 + hi y - p`, and that times the block's
scale, broadcast from its float16 in the load. Nine vector operations a
block, 72 for 256 weights, where the kernel of [q4_0.md](q4_0.md) takes
114; the accumulators stay in registers for a whole row.

```
╔══════ Xeon Phi 7120A, worker 0a063493: one token's multiply (Q4_0, one activation row) ══════╗
║  ┌────────────────────────────────────────────────────────────────────────────────────────┐  ║
║  │ dispatcher (core 60): sets q40_prep_ok, wakes the pool, takes a slice itself           │  ║
║  └────────────────────┬─────────────────────────────────────────────┬─────────────────────┘  ║
║                       ▼ 120 slices, consecutive rows                ▼                        ║
║  ┌──────────────────────────────────────────┐  ┌──────────────────────────────────────────┐  ║
║  │ q4_0_prepare, once a request a thread    │  │ phi_q4_0_row1p, 72 vector ops / 256 w    │  ║
║  │ x0, y = x1 - 16 x0, p: one buffer a core │  │ in cache 204 cycles (old kernel 306)     │  ║
║  │ shared by its two threads (same bytes)   │  │ rows <= 5120 w: whole row a call         │  ║
║  │ 1152 B a superblock, up to 32768 weights │  │ longer: segments of 2560 w, rows inside  │  ║
║  └──────────────────────────────────────────┘  └──────────────────────────────────────────┘  ║
║  17408 x 5120: 1.06 -> 0.65-0.67 ms    5120 x 5120: 0.38 -> 0.22-0.25 ms                     ║
║  5120 x 17408: 1.04 -> 0.87-0.89 ms    card compute a request: 20.1 -> 15.5 s                ║
║  generation, 64 tokens: 24.0-24.8 s -> 19.2-19.6 s = 3.22-3.28 tok/s                         ║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝
```

The first build of it, a third fewer operations, was 10 % faster and
slower on two shapes. Three explanations were tested and fell (the
first-level cache shared by a core's threads as first tested, the order
of traversal, a slow instruction: each is about two cycles on a lone
thread). The worker's own trace (`-t`) gave the two causes:

- **The dispatcher takes a slice of every request itself.** It had no
  prepared buffer, ran the old kernel, and its slice was 1,183 us of a
  1,480 us dispatch whatever the pool did. On the prepared path too,
  one slice a core went from 1.81 to 1.15 ms (17408 x 5120).
- **A prepared row a thread is too much for a core.** A row of 5120
  weights is 23,040 bytes prepared; two threads of a core put 46,080
  bytes through one 32 KiB first-level cache: 1.48 ms at two slices a
  core where one took 1.15. With one buffer a core, which both threads
  fill with the same bytes and each reads after its own fill, 0.67 ms.
  The worker allows it only when every thread of a request prepares
  the same row: one matrix, one activation row, no mixture.

The same effect outside the worker (`tools/q4_0-bench.sh` in the patch,
run twice, cycles per 256 weights; each stream is a process with a
prepared row of its own):

| | One stream a core | Two | Three | Four |
|---|---|---|---|---|
| each stream, the median of 40 cores | 201.3, 200.5 | 560.5, 560.5 | 639.2, 638.4 | 721.0, 719.6 |
| blocks of 256 weights per 1000 cycles, a core | 4.97, 4.99 | 3.57, 3.57 | 4.69, 4.70 | 5.55, 5.56 |

Two streams with a row each do less work together than one alone. The
same script times the kernel on one thread: 204 cycles for 256 weights
with everything in the first-level cache against 306 for the old one,
202.2 to 203.6 streaming 64 MiB of weights with its prefetch and 248.6
to 249.9 without.

**Long rows.** A prepared row of 17408 weights is 78,336 bytes, so the
down projection (5120 x 17408) is walked in segments of its columns,
the thread's rows inside each segment, the kernel adding to each row's
accumulator and its prefetch told the jump to the next row's segment.
At segments of 5120 weights (23,040 bytes) it gained nothing: 1.054,
1.092, 1.118 ms against 1.054, 1.122, 1.034. The trace again: the
dispatcher, alone on its core, took 627 us for its slice, the longest
pool slice 891; the two threads of a core are on different segments
most of the time, twice 23,040 bytes in one cache. At 2560 weights
(11,520 bytes, two fit) it is 0.889, 0.866 ms against 1.043, 1.026,
1.037.

Card compute time in the service, one activation row, best of 15, the
workers deployed in turn (`phi-vpu matmul-check --only q4_0 --act 1
--repeat 15`):

| Shape | Kernel of [q4_0.md](q4_0.md) | One-row kernel | And half segments |
|---|---|---|---|
| 17408 x 5120, 114 slices | 1.058, 1.062 ms | 0.891, 0.681 ms | |
| 5120 x 5120, 114 slices | 0.372, 0.383 ms | 0.238, 0.229 ms | |
| 17408 x 5120, 120 slices | | 0.662, 0.641, 0.636 ms | 0.652, 0.668 ms |
| 5120 x 17408, 120 slices | | 1.043, 1.026, 1.037 ms (old kernel: too long a row) | 0.889, 0.866 ms |
| 1024 x 5120, 114 slices | 0.082, 0.081 ms | 0.089, 0.082 ms (not taken) | |

Not on the new path, and unchanged: a slice of fewer than 24 rows (the
preparation is about 96 vector operations for 256 weights of the
activation row against the kernel's 72 for 256 weights of a weight
row; 1024 x 5120 at nine rows a slice measured 5 % slower with it), a
request of
two or more activation rows, a mixture of experts, and a request of
several matrices.

## Every weight in host RAM, nothing on the card

Measured 2026-11-10, the same model and prompt. The card's backend can
leave every weight in host RAM (`PHI_GGML_STREAM=1`): the card's rows sit
in the shared window and the card pulls them at every multiply. The
window is 2 GiB, which leaves 1.25 GiB for rows: 8.1 % of every matrix
(1.03 GB). The GPU takes prompts with the weights copied from host RAM
(`-ngl 99 -ot ".*=CPU"`), and holds the context.

| Layout | Generation | Prefill, second prompt |
|---|---|---|
| rows resident on the card (above) | 3.221 to 3.282 tok/s | 315 to 319 tok/s |
| host RAM, the card left out | 2.267, 2.259 tok/s | 569.2 tok/s |
| host RAM, the card pulling 8.1 % through its block device | 1.559, 1.567, 1.536 tok/s | 538.9 tok/s |
| the same, pulled by the card's own DMA engine | 2.155, 2.203, 2.127 tok/s | 557.8, 556.1 tok/s |

- Sampled from the host over one 64-token request with the block
  device (`tools/overlap.c`): the card busy 29.911 s, of that 27.780 s
  pulling weights, 1.847 s computing. The pull is a `pread` the card's
  kernel hands to the host daemon, with fourteen host threads competing
  for the cores.
- The patch's worker pulls a matrix with the card's own DMA channel
  straight into the frames of the buffer it is multiplied from
  (`card/vpu/matmul.md`, "Weights straight from host RAM"): 15.818 s of
  pulling, 1.852 s computing, and 39 % more tokens a second.
- It is still 3 to 6 % behind leaving the card out. With nothing
  resident the same rows cross the link at every token; no transfer
  method removes that, and the host reads its own memory at about ten
  times the link's rate. A card pays here only for what it holds.
- The GPU fed from host RAM prefills faster than fed from the card
  (569 against 315 to 319 tok/s): the card link is the slower source.

## The card worker since [q4_0.md](q4_0.md)

Both are in `patches/intel-phi-avx-512-c.patch`, with their documents
(`card/vpu/rows.md`, "The sums").

- The sixteen partial sums of a result are added on the vector unit: each
  a broadcast operand of an add, four chains of four, then two levels
  (six adds deep where the scalar chain was sixteen).
- The worker executes no x87 instruction. A Q8_0 row's last blocks and a
  float row's last `k % 16` elements, which were scalar loops, are added
  into the row's accumulator on the vector unit before it is summed.
  The float tail had no test (the check's float rows were all 512 long);
  it has one now, at 521.

Until this change the sums were kept in the x87 unit on purpose: there
a chain of additions stays in the 80-bit register and rounds once, which
made a result equal to the earlier C worker's to the bit. That equality
is given up; the check against the host reference is unaffected.

The two worker builds deployed in turn, twice, card compute time, Q4_0,
float16 activations, 114 threads, best of 15 each:

| Shape, activation rows | x87 sums | Vector sums |
|---|---|---|
| 17408 x 5120, 1 | 1.063 ms | 1.069 ms |
| 17408 x 5120, 8 | 2.615 ms | 2.529 ms |
| 17408 x 5120, 64 | 22.320 ms | 21.597 ms |
| 61440 x 256, 1 | 0.277 ms | 0.280 ms |
| 61440 x 256, 8 | 1.270 ms | 1.179 ms |
| 61440 x 256, 64 | 11.485 ms | 10.123 ms |

Equal at one row (the medians cross), 3 to 12 % faster from eight.
Every weight type passes `phi-vpu matmul-check` on the result, with
float32 and float16 activations.

## The fused feed-forward, measured and not used

The worker can take a layer's gate, up, SwiGLU and down as one request
(`K_FFN`), which would save two round trips a layer. On the card, gate
and up 4352 x 5120, down 5120 x 4352, one activation row, best of 9:

| | Best | Median |
|---|---|---|
| fused, float16 intermediate | 1.179 ms | 1.800 ms |
| fused, float32 intermediate | 1.240 ms | 1.934 ms |
| the three multiplies it replaces, compute summed | 1.181 ms | |

Its best ties the three multiplies and its median is half again as
long: the request stalls between its phases some of the time. It is
left off (the backend also keeps it off under the offload, and its gate
on the host does not name Q4_0).

## Limits

- The sets of runs are consecutive, not interleaved; the host drifts
  over tens of minutes, so a difference inside a set's spread is not a
  result.
- The four GPU layers and the head are a deviation from "every matrix on
  the card", taken because the card's 16 GB do not hold the model's
  14.75 GB of matrices inside the pool used here.
- A multiply whose input is not ready is still issued in graph order,
  after the host and the GPU have finished the step before it.
- The one-row kernel's result differs in its last bits from the old
  kernel's; equality of output was checked on 24 tokens of one prompt
  at temperature zero, not more.
- The patch's worker is md5 `0a063493e9ef4ad87b7af0d200570db3`
  (272,904 bytes) and its backend `9f9c73d6fe2edac85b1cf3571e44bc5b`
  (137,704 bytes) when built from a clean checkout; those are the two
  binaries every number of the last table row was taken on.
