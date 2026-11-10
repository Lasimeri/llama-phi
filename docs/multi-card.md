# Several cards, several GPUs

Measured on 2026-11-10 on a second machine: an EPYC 7742 with 251 GB, four
RTX 3080 20 GB (PCIe gen 4, 8 lanes each) and four Xeon Phi 3120-series cards
(57 cores, workers of 110 to 114 threads, a pool of 2,200 huge pages each).
Every number is llama-server's own timing, one run each unless a list is given;
prompts are cold (`cache_prompt` false), generation is 220 greedy tokens.

## What had to change

- **The backend's hand-off to the DMA router** named the first card for every
  card's rows. With four cards the first prompt stopped at
  `K_DOWNLOAD ... failed: status -4`. Each card's slice now goes with its own
  card (`Intel-Phi-AVX-512-C` `backend.S` `back_split`).
- **The router** knew one window. It now keeps a table of cards, with a fetch
  thread and a fill thread for each ([tools/tridma_router.md](../tools/tridma_router.md)).
- **`scripts/serve.sh`** reserves and releases the huge-page pool of every card
  in `PHI_GGML_CARDS`.

## The Q8_0 mixture model of 35.21 GiB (`scripts/cards35.sh`)

Blocks 4 to 40 and the output matrices in VRAM over the four GPUs, the first
four blocks on the cards (at most 4.0 GB in all) and the host, context 204,800.

| cards hold | threads a card | generation, tok/s | prompts, tok/s |
|---|---|---|---|
| one card, half of each dense matrix and every expert (3.53 GB) | 114 | 56.2 to 57.8 | 878, 906 |
| four cards, whole experts by use (0.91 GB each) | 114 | 59.3, 59.4 | 1,736 to 1,811 |
| four cards, whole experts by use | 28 | 60.7, 61.0 | |
| four cards, every tensor split by rows (0.94 GB each) | 28 | 62.0, 62.2 | 1,505, 1,548 |
| four cards, every tensor split by rows | 114 | 58.5, 58.4 | |
| no card: every block in VRAM | | 120.5, 120.3 | 4,770, 4,811 |

Splitting every tensor by rows is tensor parallelism across the cards: each
card answers every multiply with a quarter of the rows (9,695 requests on each
card over two generations, against 10,011 to 10,211 with whole experts).

### Why four cards are not four times one

| cards (every tensor split by rows) | threads a card | tok/s | ms a token |
|---|---|---|---|
| 1 | 114 | 51.3, 51.6 | 19.4 |
| 2 | 57 | 57.1, 57.2 | 17.5 |
| 4 | 28 | 62.0, 62.2 | 16.1 |

The three times fit 15.0 ms + 4.4 ms / cards. What divides by the number of
cards is their arithmetic. What does not: the 36 blocks on the GPUs (8.3 ms for
40 blocks with no card in the path) and about 22 card round trips a token, which
all cards pay at the same moment. The division of the 15.0 ms between those two
is inferred from the fit and the reference, not timed. The best thread count a
card fell with the slice it holds: 114, 57 and 28 for one, two and four cards.

## A model several times the VRAM (`scripts/flash.sh`)

188.2 GB at Q8_0: 48 blocks of 2.76 GB, 2.67 GB of each the experts (512 a
layer), and a 54.4 GB per-layer embedding table. The four GPUs hold 80 GB.

llama.cpp at the pinned commit keeps experts in system memory with `--cpu-moe`
(or a tensor override) and caches them on the GPUs with `--moe-cache-mib`: an
expert a token routes to is uploaded on a miss and multiplied on the GPU. A
block's experts can also be fixed in the VRAM of the GPU that owns the block.
The first block goes to the four cards, split by rows.

| layout | prompts at 1,515 / 6,581 / 21,826 tokens, tok/s | generation, tok/s |
|---|---|---|
| 17 blocks whole in VRAM, the host multiplying the other 31, no card | 150 at 1,442 tokens | 13.9, 13.9 |
| every expert in system memory, cache 40,000 MiB, no card | | 25.1 to 27.9 |
| the same with the first block on the cards, cache 40,300 MiB, batch 2,048 | 147 / 194 / 222 | 23.5 to 25.1 |
| 19 blocks' experts fixed in VRAM, cache 2,000 MiB, batch 2,048 | 198 / 270 / not run | 7.5, 7.7 |
| 19 blocks' experts fixed, cache 2,000 MiB, batch 4,096 | 194 / 417 / 475 | not measured |
| every expert in system memory, cache 50,000 MiB, batch 4,096 | 147 / 309 / 351 | 24.2 to 30.8 |
| 12 blocks' experts fixed, cache 21,200 MiB, batch 4,096 (the script's default) | 174 / 371 / 419 | 22.7 to 25.1 |
| CUDA unified memory, every block assigned to the GPUs | | 6.4 to 6.9 |

What the measurements say:

- **A prompt's cost is per batch, not per token.** For each large batch
  llama.cpp hands the multiplies of system-memory experts to the first GPU
  alone, which pulls those experts in: during a 6,581-token prompt GPU 0 was
  busy in 30 of 34 one-second samples (`nvidia-smi dmon`), the other three in 3
  to 6. A 1,515-token batch took 7.65 s, a 2,048-token batch about 7 s, a
  4,096-token batch 7.6 to 7.8 s. Doubling the batch nearly doubles the rate;
  its buffers cost 2,094 MiB on GPU 0 and 1,362 MiB on each other GPU.
- **Fixed blocks serve prompts, the cache serves generation.** A fixed block is
  multiplied where it lives; a cached expert is still copied to the first GPU
  for each batch (351 against 475 tok/s for the same VRAM). A generated token
  needs eight experts a layer, which a large cache holds and fixed blocks of
  other layers do not (7.6 against 24 tok/s).
- **Short prompts barely move:** a batch of 63 to 437 tokens costs 3.5 to 4.4 s.
- **VRAM headroom:** 20 fixed blocks with an 8,800 MiB cache left one GPU 50 MiB
  and the first large batch ran out of memory; the scratch pool grew 48 to 160
  MiB in the layouts that ran.

### The host while it generates

With the first block's multiplies on the cards and the experts multiplied on
the GPUs, a 3 s profile of the server during generation (`perf record -p`):
86.4 % of the host's cycles in OpenMP's wait loop, 8.7 % in the CUDA driver,
2.1 % in libc, 0.87 % in ggml's CPU code (embedding rows, the block's fused
small operators, one f32 dot kernel at 0.08 %). `OMP_WAIT_POLICY=passive` with
`GOMP_SPINCOUNT=0` drops 9.4 busy cores to 1.7 and costs 9 % (21.2 to 22.8
against 23.4 to 24.7 tok/s).

Two settings keep weight multiplies off the host that the defaults leave on it:
`PHI_GGML_MIN_BYTES=1` (the default 4 MB keeps small matrices on the host) and
`PHI_GGML_PP_SHARE=1.0` with `PHI_GGML_PP_ADAPT=0` (the default gives the host
a share of the rows of a 2 to 31 token batch: 60 such multiplies in three
requests before the change, none after). One one-row f32 matrix stays.

## Traps

- **`GGML_BACKEND_PATH` is read only from the handlers of `-ot`, `--device` and
  `--list-devices`** at this commit. A llama-server command line without one of
  them never opens `libggml_phi.so`: no line from the backend, the cards idle,
  the block computed by the host. Every launcher here passes `-ot`.
- **A tensor override does not move a layer.** With `-ot` sending 31 blocks to
  the host, the 17 left for VRAM all belonged to the last GPU by llama.cpp's
  layer split (29.6 GiB asked of a 20 GB card). Blocks meant for VRAM have to be
  spread over the ranges the GPUs own.
- **A card's part is rounded up to 64-row units** from the last card down; the
  first card takes the remainder (a 640-row matrix: 64, 192, 192, 192). The
  slowest card carries 192 rows in an even split too, so it costs balance of
  memory, not time.

## Not verified

- No equality check of the large model's output against a run without cards:
  the texts agree in their opening and then diverge.
- `scripts/cards35.sh` and `scripts/flash.sh` take the work directory and the
  model as arguments; the measurements ran the same commands with the paths
  written into the scripts.
