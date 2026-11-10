# llama-phi

An Intel Xeon Phi 7120A (Knights Corner), an NVIDIA GPU and the host CPU
running one model together through
[llama.cpp](https://github.com/ggml-org/llama.cpp). The card's 512-bit
vector units multiply the weights it holds and exchange activations and
results with the host by the card's own DMA engine; the GPU takes the
prompts, the context and the layers that fit in VRAM; the host holds and
computes the rest.

llama.cpp is built from unpatched source, and it is not stock llama.cpp
when it runs: a ggml backend for the card and preload libraries replace
its memory mapping and part of its scheduling
([What is changed in llama.cpp](#what-is-changed-in-llamacpp)).

On a Ryzen 7 5800X with 32 GB, an RTX 3090 Ti and one 7120A:

| Layout | Script | Model | Generation, tok/s | Prompts, tok/s |
|---|---|---|---|---|
| VRAM + card + locked host RAM | `scripts/vram35.sh` | Qwen3.8-35B-A3B Q8_0, 35.21 GiB | 14.6 to 17.8 | 472 |
| GPU context, card holds the most used experts, host the rest | `scripts/moe.sh` | Qwen3.8-35B-A3B Q6_K, 29.2 GB | 10.8 to 11.0 | 345 |
| GPU context, card holds every dense row | `scripts/one.sh` | a 27 B Q4_0 model | 3.22 to 3.28 | 315 to 319 |
| host and card, no GPU | `scripts/bench.sh` | Qwen3.8-35B-A3B Q8_0 | 8.62 (host alone 4.32) | 15.86 (host alone 4.74) |

The rows are different models, lengths and tools (the last is
llama-bench at 16 tokens, the others llama-server), so they are not a
ladder; each is documented with its conditions below.

On an EPYC 7742 with 251 GB, four RTX 3080 20 GB and four 3120-series
cards ([docs/multi-card.md](docs/multi-card.md)):

| Layout | Script | Model | Generation, tok/s | Prompts, tok/s |
|---|---|---|---|---|
| VRAM over four GPUs, the first four blocks split by rows across four cards | `scripts/cards35.sh` | Qwen3.8-35B-A3B Q8_0, 35.21 GiB | 62.0, 62.2 (no card: 120.4) | 1,505 to 1,548 |
| experts in system memory with a GPU cache, 12 blocks' experts fixed in VRAM, the first block on four cards | `scripts/flash.sh` | a 188.2 GB Q8_0 mixture model | 22.7 to 25.1 | 174 to 419, rising with the prompt's length |

The cards cost speed on this machine, where the GPUs alone hold the
smaller model: what four cards buy over one, and why it is not four
times, is measured in that document.

This repository holds the integration: patches to two existing public
repositories, the scripts that fetch, patch and build everything, and
the measurements. llama.cpp's source is not patched: it is built from a
pinned commit with build flags only. The program that runs is not stock
llama.cpp, though: a fourth ggml backend and up to three preload
libraries replace parts of what it does at run time
([below](#what-is-changed-in-llamacpp)).

| Upstream | Commit | What this repository adds |
|---|---|---|
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | `4b77dee` | no source patch: build flags (`scripts/setup.sh`), and at run time a backend and preload libraries that change its behaviour |
| [Lasimeri/Intel-Phi-AVX-512-C](https://github.com/Lasimeri/Intel-Phi-AVX-512-C) | `db092f6` | `patches/intel-phi-avx-512-c.patch`: card worker, ggml backend, preload |
| [Lasimeri/Intel-Phi-7120A](https://github.com/Lasimeri/Intel-Phi-7120A) | `22712a2` | `patches/intel-phi-7120a-phitop.patch`: phitop shows the worker's DMA; the card no longer swaps into host memory (`PHI_SWAP_HOSTMEM=1` opts in) |

## Source in this repository

The source the measured servers were built from is here, beside the
integration:

| Directory | What it is | How to check |
|---|---|---|
| `llama.cpp/` | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) at commit `4b77dee0524c64b8471869232e2958bd9589607d`, every file, none changed | `git rev-parse HEAD:llama.cpp` gives `b02cf014b4690a51d8530f06edecb69cb91dfce9`, the tree of that commit |
| `Intel-Phi-AVX-512-C/` | [Lasimeri/Intel-Phi-AVX-512-C](https://github.com/Lasimeri/Intel-Phi-AVX-512-C) at `e56d8a5`: `db092f6` with `patches/intel-phi-avx-512-c.patch` applied (card worker, ggml backend) | `git rev-parse HEAD:Intel-Phi-AVX-512-C` gives `f604c826379fd9d23e18b4fce26e703e3b2be200`, the tree of that commit |
| `tools/`, `cuda/`, `scripts/` | this repository's own: the preload libraries, the CUDA programs, the launchers | |

Both directories were placed as git trees, not re-added file by file, so
the identifiers above are equal to the origins' by construction. They
are snapshots: `scripts/setup.sh` still fetches the pinned upstreams and
applies the patches, and does not build from these copies. The workflow
files under `llama.cpp/.github/` are upstream's and do not run here
(GitHub runs only a repository's own top-level workflows).

## What is changed in llama.cpp

No file of llama.cpp is edited: the copy in `llama.cpp/` is upstream's
tree, and the checkout the servers were built from is clean at
`4b77dee`. What it does when it runs is changed from outside, by a
backend it loads and by libraries placed in front of it with
`LD_PRELOAD`:

| Loaded | Replaces | Effect on llama.cpp | Used by |
|---|---|---|---|
| `libggml_phi.so` (`GGML_BACKEND_PATH`) | nothing: registers as a ggml backend | takes the weight multiplies, and with `PHI_GGML_HOSTOPS=1` the small operations between them, so ggml's scheduler splits every graph differently | every layout |
| `libtridma.so` ([tools/tridma_router.md](tools/tridma_router.md)) | `mmap`, `munmap`, `madvise`, `posix_madvise` | the model's mapping is not prefetched, and the ranges a card holds become memory fed from the card instead of from the file | `one.sh`, `moe.sh`, `vram35.sh` |
| `libunpin.so` ([tools/unpin.md](tools/unpin.md)) | `ggml_backend_sched_set_tensor_backend` | llama.cpp's pin of each layer's norm to the layer's device is declined for a generation step | `moe.sh`, `vram35.sh` |
| `libhostram.so` ([tools/hostram.md](tools/hostram.md)) | `mmap`, `munmap` | the model is not a mapping of its file: llama.cpp is handed an anonymous region that is filled from the file and locked | `vram35.sh` |
| `libnopopulate.so` ([tools/nopopulate.md](tools/nopopulate.md)) | `mmap`, `munmap`, `madvise`, `posix_madvise` | the model's mapping is not prefetched | `split.sh` |

Results obtained this way are results of this combination, not of
upstream llama.cpp: its loader, its page handling and the placement its
scheduler computes all differ from a stock build's. `scripts/serve.sh`
alone (the backend, no preload) is the closest to stock.

## Host and card, no GPU

The first layout, and the only one without a preload library.
llama-bench, Qwen3.8-35B-A3B Q8_0 (35.20 GiB, 35.51 B parameters), host
Ryzen 7 5800X with 31.3 GiB RAM, card Xeon Phi 7120A on PCIe Gen2 x8,
`-t 14 -ngl 0`, the GPU hidden. The card holds 10.49 GB of the model
(half of every dense matrix's rows, 25.1 % of every expert's); the host
holds the rest, part of it paged from NVMe because the model is larger
than RAM and the card's budget together.

| Run | Card | pp16 tok/s | tg16 tok/s |
|---|---|---|---|
| A-1 | yes | 16.08 ± 1.17 | 8.41 ± 0.13 |
| host-1 | no | 4.90 ± 0.27 | 4.40 ± 0.08 |
| A-2 | yes | 15.63 ± 0.85 | 8.83 ± 0.07 |
| host-2 | no | 4.58 ± 0.51 | 4.23 ± 0.10 |
| **means** | | **15.86 vs 4.74 (3.35x)** | **8.62 vs 4.32 (2.00x)** |

The four runs alternated, each `-r 2`; the ranges do not overlap. Longer
runs with the card: `-p 64 -n 128 -r 2` gave pp64 39.32 ± 7.13 and
tg128 11.95 ± 1.39; `-n 256 -r 1` gave tg256 9.15 (no host-only control
was run at these lengths). Details, the per-request costs and how the
numbers were taken: [docs/results.md](docs/results.md).

### How that layout works

```
╔══════════════════════════════════════════════════════════════════════════════════════════════╗
║ HOST  Ryzen 7 5800X                                                                          ║
║ ┌──────────────────────────────────────────────────────────────────────────────────────────┐ ║
║ │ llama-bench (build-native-cuda, native AVX2, no preload)   -t 14, CUDA hidden            │ ║
║ │ non-matmul ops + host rows of every multiply: ggml CPU backend, 14 threads               │ ║
║ └──────────────────────────────────────────────────────────────────────────────────────────┘ ║
║ ┌──────────────────────────────────────────────────────────────────────────────────────────┐ ║
║ │ libggml_phi.so (GGML_BACKEND_PATH): splits each multiply's rows host | card, concurrent  │ ║
║ │ card: 50 % of dense rows, 25.1 % of expert rows = 10.49 GB resident; judge on            │ ║
║ └──────────────────────────────────────────────────────────────────────────────────────────┘ ║
║ ┌──────────────────────────────────────────────────────────────────────────────────────────┐ ║
║ │ shared window 4 GiB: activations OFF_B, results OFF_D, control 0..16 KiB                 │ ║
║ │ control lines read by phitop: OFF_STATS 320 (VPU), OFF_CDMA_STATS 192 (card DMA, new)    │ ║
║ │ phictl: vfio, host DMA channels 0/1 = weight uploads 347-544 MB/s                        │ ║
║ └──────────────────────────────────────────────────────────────────────────────────────────┘ ║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝
            │ PCIe Gen2 x8 (5.0 GT/s): card DMA ch 7 pulls activations, pushes results          
            ▼ steady decode: card dma in 10.1 MB/s, out 21.8 MB/s, 1,561.7 requests/s           
╔══════════════════════════════════════════════════════════════════════════════════════════════╗
║ CARD  Xeon Phi 7120A, 61 cores, 16 GB GDDR5, 5600 huge pages (11.2 GB)                       ║
║ ┌──────────────────────────────────────────────────────────────────────────────────────────┐ ║
║ │ phi-vpu-worker -O -S 1 -C 59 -T 2 (DMA push + counters)                                  │ ║
║ │ matrix pool 120 threads: cores 0-59, hw threads 0+3; dispatcher CPU 0                    │ ║
║ │ slot 118 threads: cores 0-58, hw threads 1+2 (idle in this design)                       │ ║
║ │ cdma.S ch 7: cdma_submit pull, cdma_push result, cdma_stats; phi_q8_0 MVEX F32 kernels   │ ║
║ │ request: pull 0.008 ms, compute 0.144 ms, push 0.039 ms (push was 0.48-0.59 ms)          │ ║
║ └──────────────────────────────────────────────────────────────────────────────────────────┘ ║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝
```

- The host runs llama.cpp built for the host CPU. Scalar and small ops
  stay on the host; whole weight multiplies are offered to the card.
- `libggml_phi.so` registers the card as a ggml accelerator. For each
  matrix it keeps a share of the rows resident in the card's memory
  (uploaded once), computes the host's rows on the host at the same time,
  and gathers the results. A timing judge moves a tensor back to the host
  when the card is slower for it.
- On the card, Q8_0 weights are up-converted from int8 to float32 inside
  the load (MVEX `{sint8}`), and every product is a 16-lane float32 FMA.
- Activations reach the card, and results come back, through the card's
  own DMA channel. phitop shows that traffic beside the host daemon's.

The design, every change in the patches, and what was tried and dropped:
[docs/design.md](docs/design.md).

## With a GPU

An NVIDIA GPU can take prompt prefill and the KV cache while the weights stay in
host RAM and on the card: `scripts/serve.sh` (one llama-server, OpenAI API on
127.0.0.1:8095). It also has CUDA kernels that drive the card worker by DMA
and a two-server prefill/generation overlap. Layout, the scheduler setting it
needs, and measurements: [docs/cuda.md](docs/cuda.md). The DMA router that serves
a model with its weights in host RAM and on the card while the GPU does every
prompt (no disk reads during inference): [docs/router.md](docs/router.md).

The card takes Q4_0 weights as well (six kernels in the patch, their
weights equal to ggml's bit for bit). In a split layout, one GPU server
for prompts and one host-and-card server for generation with a slot file
between them (`scripts/split.sh`), a 27 B Q4_0 model generates 2.605
tok/s with 13.09 GB of its rows on the card, against 0.2076 tok/s when
the card could take only its one Q6_K matrix: [docs/q4_0.md](docs/q4_0.md).

The same model on one server (`scripts/one.sh`): the context and
attention on the GPU, every row of 60 layers' matrices on the card, the
last four layers and the output head in VRAM, so the host computes no
matrix rows; 3.22 to 3.28 tok/s, prompts 315 to 319 tok/s from the
second on, and half the VRAM of the split. Three changes took it there
from 2.42 to 2.54: the backend begins a multiply whose input is ready
before its node arrives, so the card works while the host and the GPU
handle the previous result; a kernel for one activation row that needs
72 vector operations for 256 weights where the first took 114; and long
rows walked in segments that fit a core's first-level cache twice. The
card worker no longer executes an x87 instruction. What was measured,
what each change gave, and what was tried and dropped:
[docs/one-server.md](docs/one-server.md).

A mixture-of-experts model larger than host memory allows
(`scripts/moe.sh`, Qwen3.8-35B-A3B at Q6_K, 29.2 GB on a 32 GB host): the
card holds half of every dense matrix and the most used whole experts
of each layer (12.88 GB, held nowhere else), the host every other
expert, the GPU the 262K context and every prompt, with no weights of
its own. 10.8 to 11.0 tok/s, where the same split by rows of the Q8_0
file gave 5.3, and prompts at 345 tok/s where they ran at 83 while the
card's experts came back from the disk: they reach the GPU from the card
through the router now. The backend also keeps the small operations
between two multiplies on the host, which cut one token's graph from 612
splits to 393, and the card worker pulls a matrix out of host RAM with
its own DMA engine, straight into the buffer it is multiplied from.
Every step, the timeline that chose them, and what did not work:
[docs/moe.md](docs/moe.md).

The Q8_0 file of that model (35.21 GiB) in three memories
(`scripts/vram35.sh`): 18 blocks and the output matrix in VRAM beside a
204,800-token context, a quarter of the file on the card, and the
remaining 10.32 GiB on the host as locked anonymous memory, not as a
mapping of the file. 17.7 to 17.8 tok/s after a prompt and 14.6 over
220 tokens, prompts at 472 tok/s, and no byte of the file read after
the load. `tools/hostram.c` is what makes the host's share real memory
without the model ever being in host RAM whole: llama.cpp gets an
anonymous region in place of the file mapping, the card's and the GPU's
parts pass through it once, and the rest is read and locked
([tools/hostram.md](tools/hostram.md)).

## Use

Requirements: a Xeon Phi x100 card booted with the
[Intel-Phi-7120A](https://github.com/Lasimeri/Intel-Phi-7120A) stack
(`phictl`, the `phi` CLI); git, cmake, ninja, gcc, GNU binutils, tcc,
cargo; the CUDA toolkit only for the CUDA builds (`CUDA=0` otherwise).

```
scripts/setup.sh work            # fetch the pinned upstreams, patch, build
scripts/worker.sh work           # deploy and start the card worker
scripts/bench.sh work MODEL.gguf a      # host + card
scripts/bench.sh work MODEL.gguf host   # host only, the control

scripts/serve.sh work MODEL.gguf start  # one server: prompts and context on the GPU
scripts/moe.sh work MODEL.gguf route    # a mixture model: log its routing, generate, then
scripts/moe.sh work MODEL.gguf rank     #   rank each layer's experts by use
scripts/moe.sh work MODEL.gguf start    #   the most used experts on the card
scripts/vram35.sh work MODEL.gguf start # the same with layers in VRAM and the host's
                                        #   share as locked memory
```

The server layouts answer the OpenAI API on 127.0.0.1:8095. `moe.sh` and
`vram35.sh` load preload libraries from `work/runs/serve`; each one's
document under `tools/` gives its build line.

`setup.sh` prints the size and md5 of each built binary. Built from these
patches, the card worker is deterministic (same bytes on every build).
To watch the card, run the patched phitop
(`work/Intel-Phi-7120A/host/asm/out/phitop`, or install it with the
stack's own `make`) during a run of a few hundred tokens: its pcie line
starts with `card dma in, out`, and the vpu line shows the requests.

## License

MIT, as the upstream repositories: [LICENSE](LICENSE). llama.cpp is
MIT-licensed by its authors; the copy in `llama.cpp/` is theirs,
unchanged, under its own [LICENSE](llama.cpp/LICENSE), and the code it
bundles from others keeps the notices it carries there.
