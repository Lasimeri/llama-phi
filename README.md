# llama-phi

An Intel Xeon Phi 7120A (Knights Corner) as a matrix co-processor for
unmodified [llama.cpp](https://github.com/ggml-org/llama.cpp): the host
runs llama.cpp natively on its own threads, the card's 512-bit vector
units take a share of every weight matrix multiply through a ggml
backend, and the two exchange activations and results by the card's own
DMA engine. One 7120A doubles decode on Qwen3.8-35B-A3B Q8_0 on a
Ryzen 7 5800X.

This repository holds the integration: patches to two existing public
repositories, the scripts that fetch, patch and build everything, and
the measurements. llama.cpp itself is used unmodified at a pinned commit.

| Upstream | Commit | What this repository adds |
|---|---|---|
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | `4b77dee` | nothing: build flags only (`scripts/setup.sh`) |
| [Lasimeri/Intel-Phi-AVX-512-C](https://github.com/Lasimeri/Intel-Phi-AVX-512-C) | `db092f6` | `patches/intel-phi-avx-512-c.patch`: card worker, ggml backend, preload |
| [Lasimeri/Intel-Phi-7120A](https://github.com/Lasimeri/Intel-Phi-7120A) | `22712a2` | `patches/intel-phi-7120a-phitop.patch`: phitop shows the worker's DMA; the card no longer swaps into host memory (`PHI_SWAP_HOSTMEM=1` opts in) |

## Measured

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

## How it works

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
```

`setup.sh` prints the size and md5 of each built binary. Built from these
patches, the card worker is deterministic (same bytes on every build).
To watch the card, run the patched phitop
(`work/Intel-Phi-7120A/host/asm/out/phitop`, or install it with the
stack's own `make`) during a run of a few hundred tokens: its pcie line
starts with `card dma in, out`, and the vpu line shows the requests.

## License

MIT, as the upstream repositories: [LICENSE](LICENSE). llama.cpp is
MIT-licensed by its authors and is fetched, not copied.
