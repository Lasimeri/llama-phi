# A mixture-of-experts model: the active weights on the card

Measured 2026-11-10 on the hardware of [results.md](results.md) with an
RTX 3090 Ti beside the card, on Qwen3.8-35B-A3B at Q6_K (29.2 GB, 40
layers, 256 experts a layer of which a token uses 8). The host has
32 GB, so the model cannot sit in host RAM beside a desktop; the split
below puts what a token is most likely to use on the card and nothing
anywhere twice. Every number is one request unless several are given.

## Placement

```
╔═══════════ desktop: Ryzen 7 5800X, 32 GB; one llama-server :8095 (scripts/moe.sh) ═══════════╗
║  ┌────────────────────────────────────────────────────────────────────────────────────────┐  ║
║  │ 35 B Q6_K mixture of experts, 29.2 GB: -ngl 99, every weight tensor on the host side   │  ║
║  │ libggml_phi.so + libtridma.so + libunpin.so; one token = 393 splits (612 before)       │  ║
║  └──────────────┬─────────────────────────────┬─────────────────────────────┬─────────────┘  ║
║                 │ prompts, context            │ in place                    │ card DMA       ║
║                 ▼                             ▼                             ▼                ║
║  ┌────────────────────────────┐  ┌────────────────────────┐  ┌────────────────────────────┐  ║
║  │ RTX 3090 Ti, 9.4 GB used   │  │ host RAM, 14 threads   │  │ Xeon Phi 7120A, 12.88 GB   │  ║
║  │ 262K context, no weights   │  │ the other experts,     │  │ half of each dense matrix, │  ║
║  │ a prompt: host rows by     │  │ half the dense rows,   │  │ 119 of 256 experts a layer │  ║
║  │ CUDA, card rows by the     │  │ small matrices, norms, │  │ whole, offloaded to it     │  ║
║  │ router at 3.0 GB/s         │  │ adds, routing, SwiGLU  │  │ no weight held twice       │  ║
║  └────────────────────────────┘  └────────────────────────┘  └────────────────────────────┘  ║
║  prefill 345 tok/s from the second prompt (83 with the card rows read from disk)             ║
║  generation 10.8 to 11.0 tok/s over 220 tokens, 12.1 after a prompt (7.5 at the start)       ║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝
```

| Where | What | How |
|---|---|---|
| card, 12.88 GB | half of every dense matrix's rows; 119 whole experts of each layer's 256, the most used first | `PHI_GGML_EXPERTS=FILE` under the offload: the host drops those pages |
| host RAM | every other expert, the other half of the dense rows, the matrices too small to pay for a round trip, the small operations between the multiplies | `PHI_GGML_HOSTOPS=1`, `libunpin.so` |
| GPU | every layer's context (`-ngl 99`), every prompt of more than 31 tokens; no weights | `-ot ".*=CPU"`, `PHI_GGML_TG_ONLY=31`; the card's weights reach it through `libtridma.so` |

```
PHI_VPU_HUGEPAGES=6437 scripts/worker.sh work
scripts/moe.sh work MODEL.gguf route      # a row share on the card, routing logged
scripts/request.sh out TEXT 220           # generate what the model will be used for
scripts/moe.sh work MODEL.gguf rank       # each layer's experts by use
scripts/moe.sh work MODEL.gguf stop
scripts/moe.sh work MODEL.gguf start      # the layout
```

The ranking used here came from 660 generated tokens (C, prose, shell;
26,519 routing lines). Routing is far from uniform: in generation the
most used eighth of a layer's experts took 51.1 % of its selections, the
most used quarter 73.8 %, three eighths 86.3 %, half 93.2 % (uniform
routing over as many draws: 17.2, 32.1, 45.9, 58.7 %).

## What each step gave

Generation over 220 tokens, three prompts (C, prose, a third); prefill
of a 1,410-token prompt, the second time it is sent.

| Step | Generation, tok/s | Prefill, tok/s |
|---|---|---|
| Q8_0 (37.8 GB), 25.1 % of every expert's rows on the card | 5.354, 5.279 (64 tokens) | 69.1, 74.2 |
| Q6_K, 40.6 % of every expert's rows on the card | 8.216, 10.136, 8.231 | |
| Q6_K, every dense row and 109 whole experts a layer on the card | 7.529, 7.842, 7.498 | 83.4 |
| and a multiply begun ahead of its node (`PHI_GGML_AHEAD=1`) | 8.230, 8.280, 8.729 | |
| and the card's experts to the GPU through the router | 8.709, 8.655, 8.757 | 358.0 |
| half of each dense matrix back to the host (119 experts a layer) | 10.60, 10.37, 10.51 | 344.2 |
| and the small operations on the host, the norms unpinned | 10.93, 10.77, 11.04 | 343.1 |
| the same, started by `scripts/moe.sh`, 64 tokens after the prompt | 12.17, 12.10 | 345.3, 345.6 |

- The first prompt after a start runs at 86 to 94 tok/s while the router
  learns the order in which the weights are asked for.
- Paging moved these numbers by more than some steps did (below): a
  difference of a few percent between two rows is not a result.

## The run-ahead

In graph order the card waits while a multiply's result goes to the host
and the GPU and the next input comes back. With every dense row on the
card, [one-server.md](one-server.md)'s run-ahead applies to this model's
dense multiplies: the card was busy 60.3 % of the time in graph order
and 71.0 % with it (sampled from the host, `tools/overlap.c`), and the
output at temperature 0 is the same byte for byte (24 tokens with their
top five log-probabilities). With half of the dense rows on the host no
two multiplies of the card share an input, and it finds nothing to
issue ahead.

It had a defect that this layout exposed and the patch fixes: a multiply
begun ahead after the last one of a pass held the window's lock until
the next token, and a prompt arriving first, needing that lock for the
router, waited for ever. A multiply that begins a pass is never begun
ahead now.

## The card's experts to the GPU

Under the offload the host drops the pages of what the card holds. The
backend handed a dense matrix's range to the router
([router.md](router.md)), so a prompt on the GPU read it back from the
card, and dropped a mixture's experts outright: every prompt read them
back from the disk. Each expert the card holds is now a range of the
router (13,191 ranges; `tools/tridma_router.md`).

| | Prefill | Disk read in a 220-token generation |
|---|---|---|
| experts read back from the disk | 83.4 tok/s | 1,836 MiB |
| from the card, the router's check on (`TRIDMA_VERIFY=4`) | 225.4 tok/s | 267 MiB |
| from the card | 358.0 tok/s | 437 MiB |

The check compared 12,008 delivered chunks with the file: none
differed. The first generated token's log-probabilities are the same to
the last digit with and without the router. A prompt's burst is 11.25 GB
at 3.05 GB/s.

## Balance

A multiply takes as long as the slower of the card and the host. The
backend's timeline (`PHI_GGML_TIME=1`) says which:

| Card holds | Wait for the card | Host's rows | Outside the backend |
|---|---|---|---|
| every dense row, 109 experts a layer | 58.9 % | 11.6 % | 29.0 % |
| half of each dense matrix, 119 experts a layer | 27.8 % | 26.6 % | 44.9 % |

With every dense row the card was the bottleneck and the host's cores
waited; half of each dense matrix on the host is about 20 % more tokens a
second. Small matrices then stay on the host whole (the card's half is
under the backend's 4 MB floor), which also saves their round trips.

## The small operations

The time outside the backend was the GPU being handed every norm, add,
routing soft max and SwiGLU between two multiplies: 612 splits for one
token. With `PHI_GGML_HOSTOPS=1` the backend takes those for a
generation step, and the multiplies the host keeps whole, and runs each
run of them on the host; `libunpin.so` ([unpin.md](../tools/unpin.md))
declines llama.cpp's pin of each layer's norm to the GPU.

| Backend's host operations | `libunpin.so` | Splits a token |
|---|---|---|
| off | no | 612 |
| element-wise only | no | 702 |
| with the SwiGLU | yes | 543 |
| and the multiplies the host keeps whole | yes | 393 |

A layer is then four sub-graphs of the backend, the GPU between them for
the convolution, the attention's core and its state. The time outside
the backend fell from 44.9 to 16.6 %; what replaced it is the host's own
work on the small matrices (24.4 %), which is why the gain in tokens a
second is small (10.5 to 10.9).

Checked with float32 activations (`PHI_GGML_ACT=0`), temperature 0, the
switch on against off: 24 of 24 tokens the same on this model (the
chosen token's log-probability at most 0.0638 apart) and on a dense
27 B model (0.0156). Not bit-equal: a norm on the host rounds
differently from the GPU's.

## Measured and not used

- **Whole experts beside every dense row.** The record of 2026-09-29
  had whole experts 20 % ahead of a row share on two smaller cards; here,
  with the card holding 13 GB, they were behind it (7.5 to 7.8 against
  8.2 to 10.1) until the dense rows were shared.
- **A fixed share of 70 % of each dense matrix** (`PHI_GGML_FRACTION`):
  7.94, 8.99, 9.64 tok/s and heavy paging.
- **The host's pool polling** (`PHI_GGML_HOST_POLL=100`): 10.87, 10.61,
  10.75. **A one-thread CPU backend for the runs with no multiply**:
  10.74, 10.87, 10.80. Neither is in the patch.
- **A one-row kernel for Q6_K** like [one-server.md](one-server.md)'s
  for Q4_0: the existing kernel is 167 vector operations for 256 weights
  and already computes the nibbles as floats; a prepared row would save
  about a tenth by count. Not built.
- **`llama-completion` to log the routing**: it does not load the card
  backend here (`llama-bench` and `llama-server` do), and ground through
  the model on the host alone.

## Host memory

The host's share is 15.2 GiB of the file; the page cache held 12.7 to
13.2 GiB of it, and a 220-token generation read 267 to 1,836 MiB from
the disk depending on the layout. On this host the rest of RAM was not
free: the swap device is zram, which held 16.8 GB in 9.06 GB of RAM,
most of it the pages of a RAM file system. With less in memory the host's
share fits and nothing is read at a token.

Card rows and host rows do not agree to rounding: the card is sent the
activations as float16 (`PHI_GGML_ACT=1`). Layouts in which the card
computes every dense row give the same first-token log-probabilities to
the last digit; layouts that share a matrix with the host differ from
them by up to 0.3 in the tail (the chosen token's by 0.002).

## Three memories: VRAM, the card, locked host RAM

The Q8_0 file is 35.21 GiB. With a quarter of it on the card the host's
share is 26.4 GiB, and this host had 16.1 to 16.5 GiB of memory
available (a compressed swap held 16.6 GiB of other data in 8.9 GiB of
RAM). Three layouts of the Q8_0 file, the card holding a quarter in
each, in the order they were tried:

| Layout | Generation, tok/s | Prefill, tok/s | Read from the disk |
|---|---|---|---|
| the file mapped, no weights in VRAM (`scripts/moe.sh`, 9.44 GB on the card: half of each dense matrix, 62 experts a layer) | 7.88, 7.45 (220 tokens); 5.89, 6.50, 6.57 (64 after a prompt) | 59.1, 77.8, 74.1 | at every prompt: 13.8 of the file's 35.2 GiB were in the page cache |
| the same with the host's share as locked memory (`libhostram.so`, its first build) | none: ended by `systemd-oomd` 33 s into the load | | |
| 18 blocks and the output matrix in VRAM (16.35 GiB), 9.33 GB on the card (half of each dense matrix and 120 experts a layer of the 22 other blocks), 10.32 GiB locked on the host (`scripts/vram35.sh`) | 17.71, 17.82 (64 after a prompt); 14.61 (220 tokens) | 204.1, then 471.7 | 0 bytes of the file after the load |

```
scripts/vram35.sh work MODEL.gguf start
```

- The context is 204,800 tokens with Q8_0 keys and values; VRAM in use
  was 22,100 to 22,189 MiB of 24,564, the desktop's 2,290 included.
- The host's share is anonymous memory, not a mapping of the file
  ([hostram.md](../tools/hostram.md)): `VmLck` 10,819,504 kB, nothing of
  the server in the swap. The tensors that go to VRAM are read from the
  file once for their upload and never held by the host.
- Across two prompt requests and a 220-token generation the library
  read 0 bytes from the file, and every process on the machine together
  read 31 MiB from both drives. A prompt's burst from the card is
  8.585 GB at 3.07 GB/s.
- The second layout's end is the library's first build as much as the
  host's size: it left pages unlocked, the kernel moved weights into the
  swap, and the swap passed 90 %. The build that locks on demand has not
  been run on that layout, so whether 26.4 GiB can be held there is not
  measured.
- 18 blocks was a first estimate. 2.4 GB of VRAM stayed free; more
  blocks in VRAM were not tried.
- One run each.

## The documented benchmark, run again

`scripts/bench.sh work MODEL a` on the Q8_0 file, as in
[results.md](results.md):

| | tg16 | pp16 |
|---|---|---|
| documented 2026-11-06 | 8.41, 8.83 | 16.08, 15.63 |
| the binaries of that day, rebuilt bit for bit, 2026-11-10 | 7.33, 6.67 | 12.86, 15.32 |
| the current binaries | 6.55, 6.94, 4.92 | 13.60, 14.69, 15.25 |

Neither set reaches the documented numbers on this night: the host held
14 to 15 of the file's 35.2 GiB in memory and read the rest at every
token. The two sets overlap, and one binary alone ranges from 4.92 to
6.94: a regression in the changes since is neither shown nor excluded.

## Limits

- One run per configuration except where several numbers are given; the
  sets were taken one after another.
- The ranking was made from 660 tokens of three kinds of text and
  measured on text of the same kinds.
- One server hung during this work before the run-ahead's fix; none in
  the twenty requests with prompts over nine server starts since.
- The context is 262,144 tokens of which the measurements used about
  1,500.
