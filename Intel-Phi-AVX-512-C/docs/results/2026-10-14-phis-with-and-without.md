# 2026-10-14: llama.cpp on the rack with the Phis and without them

The person asked for "the nearly 13 tok/s I got when the Phi's were in
my desktop". That number came from the `libggml_phi.so` backend on the
35B-A3B Q4_K_M (10.80 tok/s plain and 13.23 with phi-pld drafting,
[`2026-09-27-phi-pld.md`](2026-09-27-phi-pld.md)), never from the
intercept preload. The backend was brought back up on the rack with the
BF16 Flash-Next and measured interleaved, with the four cards and
without them.

## Setup

| item | value |
| --- | --- |
| host | the GPU rack: EPYC 7742 (no AVX-512), 251 GB, kernel 7.2.9-1-cachyos, four 20 GB GPUs, four Xeon Phi 3120A (PCIe Gen2 x8 each) |
| model | `/mnt/raid0/Qwen3.8-Flash-Next-GGUF/BF16/Qwen3.8-Flash-Next-BF16-00001-of-00008.gguf` (329.71 GiB, 48 blocks, 512 experts with 10 used); larger than RAM, so decodes read experts back from the page cache and the NVMe |
| server | llama.phi `build/bin/llama-server` (the binary of 2026-10-08 20:36, the one of the 2026-10-09 13.90 measurement), `scripts/phi-serve.sh` with `PHI_ENGINE=one`: one server on port 8001 for prompts and generation |
| placement | phi-serve's plan: GPU 0 blocks 0 and 1 whole, GPU 1 blocks 2 to 4, GPU 2 blocks 5 to 7, GPU 3 the non-expert weights of blocks 8 to 47, the output and block 8's experts; the experts of blocks 9 to 47 and the per-layer embeddings (102.40 GB) in host memory (`-ot`) |
| with the cards | all four: `PHI_GGML_CARDS=0,1,2,3` (the server maps the four windows, `/dev/shm/phi-hostmem` and `-1` to `-3`); `GGML_BACKEND_PATH` = `libggml_phi.so` (built from this tree at 3ee445c; sha256 7aa188b0e8cc3c54..., byte-identical across the rebuilds of the day), `PHI_GGML_EXPERTS=/mnt/raid5/phi/bench/place-live-gen-40.txt` (4 whole experts a layer on each card, the most used first), `PHI_GGML_TG_ONLY` (the cards take generation steps), `-t 12` |
| without | `PHI_CARDS=0`: no backend library, no window mapped (checked in `/proc/PID/environ` and `/proc/PID/maps`), `-t 32` |
| card workers | `PHI_VPU_HUGEPAGES=2400 PHI_VPU_ARGS="-e 0" scripts/phi-vpu.sh -c N start` (114 threads); the worker binary of 2c8ceb3; `phi-vpu -c N matmul-check` passed on all four cards just before (every format, bf16 included) |

## Method

`/mnt/raid5/phi/bench/bench-setups-2026-10-14.sh one-cards-1
one-nocards-1 one-cards-2` (the script of 2026-10-09,
`bench-setups2.sh`, with its own output directory and round names). Per
round: the engine stopped and started clean; warm-up decodes of 128
tokens until one takes fewer than 2000 major page faults (at most 16);
a prefill of this repository's README (8008 tokens, one token
generated); three decodes of 128 tokens, greedy, `ignore_eos`, of three
short prompts (`timings.predicted_per_second`); then a decode started
3 s into a second prefill (`card/vpu/vpu_matmul.md`, 7683 tokens). The
cards' request counters are read from each window's stats line before
and after the three decodes. Data:
`/mnt/raid5/phi/bench/setups-2026-10-14/` (`summary.txt`, one log per
round, every response as JSON).

## Results

| round (in order) | decode runs, tok/s | decode | prefill | beside a prefill: decode, prefill | warm-up |
| --- | --- | --- | --- | --- | --- |
| cards, `-t 12` (10:51) | 13.45, 12.69, 13.27 | **13.14** | 119.7 | 12.59, 126.2 | 3 decodes |
| no cards, `-t 32` (10:56) | 13.52, 12.70, 13.48 | **13.23** | 98.9 | 12.90, 107.9 | 16 (never under 2000 faults) |
| cards, `-t 12` (11:06) | 13.24, 12.73, 14.10 | **13.36** | 117.4 | 13.69, 122.6 | 6 decodes |

The cards over the three decodes of a round (both card rounds alike:
greedy decoding of the same prompts gives the same tokens and the same
requests; the raw counters differ, card 0 170,457 against 170,277 ns a
request):

| card | requests (384 tokens) | errors | per request |
| --- | --- | --- | --- |
| 0 | 8680 | 0 | 0.170 ms |
| 1 | 9022 | 0 | 0.176 ms |
| 2 | 9124 | 0 | 0.165 ms |
| 3 | 9386 | 0 | 0.169 ms |

About 23 requests a token a card, 4 ms of card time a token a card
(arithmetic from the table), against a 76 ms token at 13.2 tok/s.

## What it shows

- **Decode is the same with the cards and without them**, within the
  noise: runs 12.69 to 14.10 with, 12.70 to 13.52 without. The rack's
  ~13 tok/s is the GPUs' and the host's work. Confounds: the thread
  counts differ (phi-serve's defaults, 12 beside the cards, 32 without),
  the no-card round never settled its page cache (40,000 to 100,000
  faults a decode against about 1,000 with the cards), one round without
  against two with.
- **Prefill** was 117 to 120 with the cards against 98.9 without, but the
  no-card round's page faults (661,927 during its prefill) are the likelier
  cause than the cards: the cards take generation steps only
  (`PHI_GGML_TG_ONLY`).
- The 2026-10-09 harness A/B (13.90 with the cards, 12.31 without) ran on
  the harness's own traffic and is not repeated by this series.
- Without the GPUs the record has 5.38 tok/s on the host alone (32
  threads) and 5.69 with the cards (2026-10-09,
  `/mnt/raid5/phi/bench/decode-nogpu-2026-10-09/summary.txt`). llama.phi's
  `build-avx512f`, the preload's target, has no GPU backend (its
  `CMakeCache.txt`: `GGML_CUDA:BOOL=OFF`).
- For comparison, the intercept preload on the same model: llama-bench
  through the assembly library (2c8ceb3) on card 0, one host thread, pp4
  0.0078 tok/s (511 s) and tg2 0.0037 tok/s (273 s a token)
  (`/mnt/raid5/phi/bench/restart-2026-10-13/lb9.log`, `lb12.log`; the
  preload's regions run on card 0 only, cards 1 to 3 opened and idle; llama-bench
  prints them rounded, 0.01 and 0.00: the rates are the tokens over the
  measured seconds).

## A note on the run

The first start of the series wrote into the 2026-10-09 directory
(`setups2-2026-10-09`: a substitution in the copied script had not
applied) and replaced that day's `one-cards.log` and
`serve-one-cards.out` before it was stopped, 30 s in. That directory's
`summary.txt` (its one line per round) is intact; the 2026-10-09 record
does not quote the lost files.
