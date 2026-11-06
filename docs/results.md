# Results

Hardware: Ryzen 7 5800X (8 cores, 16 threads, AVX2, no AVX-512),
31.3 GiB RAM; Xeon Phi 7120A (61 cores, 4 threads each, 16 GB GDDR5) on
PCIe Gen2 x8 (5.0 GT/s x8, read from sysfs); the card bound to vfio-pci
with a translating IOMMU. Software: llama.cpp 4b77dee, the patched
repositories of this README, GCC 16.2.1.

## Design A, Qwen3.8-35B-A3B Q8_0

Command (`scripts/bench.sh`): llama-bench `-p 16 -n 16 -r 2 -t 14 -ngl 0`,
`PHI_GGML_CARD_BYTES=10500000000 PHI_GGML_OFFLOAD=1
PHI_GGML_HOST_THREADS=14`, the timing judge on, the GPU hidden. The host
only control is the same llama-bench without `GGML_BACKEND_PATH`. Runs
alternated, one after the other.

| Run | pp16 | tg16 |
|---|---|---|
| A-1 (card) | 16.08 ± 1.17 | 8.41 ± 0.13 |
| host-1 | 4.90 ± 0.27 | 4.40 ± 0.08 |
| A-2 (card) | 15.63 ± 0.85 | 8.83 ± 0.07 |
| host-2 | 4.58 ± 0.51 | 4.23 ± 0.10 |

Means: prompt 15.86 vs 4.74 tok/s (3.35x), decode 8.62 vs 4.32 tok/s
(2.00x). Each A run took 51-63 s with the model load. The worker's
request counter rose by 6,848 during A-2 (51 s).

Longer, card only: `-p 64 -n 128 -r 2` pp64 39.32 ± 7.13, tg128
11.95 ± 1.39 (76.5 s); `-p 0 -n 256 -r 1` tg256 9.15. The first pass of a
run uploads the card's rows (the host daemon's DMA at 347-544 MB/s),
which longer runs amortize. No host-only control was run at these
lengths, so their gain over the host is not measured.

The split the backend chose: 2.1 GB of dense weights at 50.0 % on the
card (the cap of an equal split with the host), 34.2 GB of experts at
25.1 %, 10.49 GB in all. The model (37.8 GB) is larger than the free RAM
and the card's budget together; the rest pages from NVMe.

## Where a request's time goes (phitop, steady decode)

| | pull | compute | push | total | requests/s | VPU busy |
|---|---|---|---|---|---|---|
| results through the cores (before) | 0.000-0.003 ms | 0.007-0.217 ms | 0.48-0.59 ms | 0.69-2.0 ms | | |
| results by DMA | 0.008 ms | 0.144 ms | 0.039 ms | 0.197 ms | 1,561.7 | 30.7 % |

Card DMA during that sample: 10.1 MB/s in, 21.8 MB/s out.

## The card's kernels, alone (`phi-vpu matmul-check`, 57 threads)

Q8_0, one column: 0.397 ms, 56.1 GB/s of weights; 8 columns 2.224 ms
(150.9 GFLOP/s); 64 columns 18.122 ms (148.1 GFLOP/s). Every format
(f32, f16, bf16, q4_K, q5_K, q6_K, q8_0, iq4_xs) checks ok against the
host reference.

## The instruction-level path, for comparison

llama.cpp `-mavx512f` under the preload, one host thread, the matrix
multiplies through the backend: Qwen3.8-2B Q8_0 pp4 0.05-0.06, tg8 0.01
tok/s; Qwen3.8-35B-A3B Q8_0 tg2 0.01 tok/s. The card's matrix compute was
0.26 s of a 4-token prompt pass of about 80 s; the rest was non-matmul
ops run as card regions.

## Streamed weights (`PHI_GGML_STREAM=1`), Qwen3.8-2B Q8_0

One 6144 x 2048 Q8_0 matrix (13,369,344 bytes) pulled per multiply: 13.591
ms on the first pass (0.98 GB/s, cold pages), 5.989 ms after (2.23 GB/s),
compute 0.73-1.31 ms. Through the block device, without overlap.
