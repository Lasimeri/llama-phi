# One server: context on the GPU, matrices on the card

Measured 2026-11-09 on the hardware of [results.md](results.md) with an
RTX 3090 Ti beside the card, on the 27 B Q4_0 model of
[q4_0.md](q4_0.md). One llama-server instead of that document's two: no
slot file, no second copy of the model in VRAM, no cold decode server.
Every number below is one request; where runs were repeated the spread
is given.

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
║  ║ 313.28 tok/s (2nd prompt)      ║  │                          │  ║ generation 2.478 tok/s ║║
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
scripts/one.sh work MODEL.gguf start
scripts/request.sh out docs/design.md 64
scripts/one.sh work MODEL.gguf stop
```

`ONE_DRY=1 scripts/one.sh ...` prints the placement and the settings
without starting anything.

## Measured

The prompt is `docs/design.md` with the chat template, 1,410 tokens, 64
tokens generated, prompt cache off (`scripts/request.sh`).

| Layout | Rows on the card | Prefill, first prompt | Prefill, second | Generation | VRAM |
|---|---|---|---|---|---|
| two servers ([q4_0.md](q4_0.md)) | 13.09 GB, 88.5 % | 364.65 tok/s | | 2.605 tok/s | 21,839 MiB |
| one server, host keeps 11.5 % of the rows | 13.09 GB, 88.5 % | 170.56 to 172.39 tok/s | 329.68 to 330.69 tok/s | 2.410 to 2.518 tok/s | 8,757 MiB |
| one server, this placement | 12.99 GB, 100 % | 159.64 tok/s | 313.28 tok/s | 2.478 tok/s | 10,475 MiB |

- The second prompt's prefill is about twice the first's: the router
  learns the order in which the matrices are asked for from the first
  prompt and reads ahead from the second on. A prompt streams 11.34 GB
  of the card's rows back at 3.32 GB/s, the card link's ceiling.
- Generation in the one-server layout was measured three times across
  three builds of the card worker (2.423, 2.410, 2.518 tok/s); that 4 %
  is the run-to-run spread, and the differences between the rows of the
  table above are inside it.
- The two-server request took 62.8 s wall (its decode server's first
  token alone was 34.24 s); the one-server request takes 29.9 to 34.2 s.
- The card: 23,763 requests in one request of the last layout, 0
  errors; 22,142 matrix multiplies (kind 4) and 1,621 downloads for the
  router (kind 32); busy 25.4 s, of that 20.2 s computing
  (`tools/overlap.c`).

## Where a token's time goes

About 400 ms a token; the card computes for about 310 ms of it
(20.15 to 20.23 s over the 65 passes of a request). The other 90 ms are
the host's and the GPU's steps between the card's multiplies and the
round trips themselves, about 340 a token.

The card's kernels are slowest exactly here: at one activation row (one
token) Q4_0 runs 126 to 159 GFLOP/s, at eight rows 450 to 530, because
decoding a weight costs nine vector operations and a single row gives
them one product to pay for.

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
the first of them is issued, and each is still issued only after the
previous one's result has gone to the GPU and come back. Issuing such a
multiply ahead, while the host and the GPU work on the previous result,
is the next step; it is not built. A multiply whose input does not
exist yet (`ssm_out`, `attn_output`, `ffn_down`) cannot run ahead.

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
- What keeps them busy is not established.

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

- One run per configuration except where a range is given.
- The four GPU layers and the head are a deviation from "every matrix on
  the card", taken because the card's 16 GB do not hold the model's
  14.75 GB of matrices inside the pool used here.
- The order of a token's multiplies is the graph's: the card and the
  host alternate, and nothing here changes that yet.
