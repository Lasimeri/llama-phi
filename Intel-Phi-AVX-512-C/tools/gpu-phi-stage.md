# gpu-phi-stage.c: GPUs to Phi cards through system memory

The host-staged path of
[`../docs/research/phi-vram-dma.md`](../docs/research/phi-vram-dma.md):
system memory is the buffer between each GPU and its neighbouring Phi
card, and each transfer stays on its owner's normal path. The GPU's copy
engine (OpenCL `clEnqueueReadBuffer`) writes 2 MiB chunks of a device
buffer into the card's host window, in the 16 staging slots of the
`RESIDENT` area (window offset 96 MiB, `vpu_fexec.h`); the card's DMA
engine pulls each chunk into a pool page through the worker's slot
protocol (`K_RESIDENT`: `fn_id` the chunk's tag, `entry` the staging slot,
class 0, every 64 KiB unit, the fill range the whole chunk). No device's
memory is mapped for another.

Two coordinated workers per pair share a ring of 16 slot states: the
**feeder** waits for a free slot and fills it from the GPU (a blocking
read, so the slot is complete when it returns); the **puller** waits for
the next filled slot, posts `K_RESIDENT`, and frees the slot on the card's
reply. The legs run at once; every pair runs at once. Pass 1 is timed per
leg (the GPU's from OpenCL profiling events, the card's around each
request) and end to end. Pass 2 sends every chunk again, pulls it back
from the card's pool page with `K_PUSH` into a staging slot poisoned with
`0xa5`, and compares every 32-bit word with the pattern the GPU wrote,
`(i * 2654435761) ^ 0x5a5a5a5a` for word `i`.

```
gcc -O2 -Wall -Wextra -pthread -I$OPENCL_HEADERS -Icard/vpu -o gpu-phi-stage tools/gpu-phi-stage.c -lOpenCL
for c in 0 1 2 3; do scripts/phi-vpu.sh -c $c stop
  PHI_VPU_ARGS="-S 1 -C 55 -T 2 -R 128 -s 1000" scripts/phi-vpu.sh -c $c start 57; done
./gpu-phi-stage 256 0:c7 1:02 2:44 3:82     # MiB per pair, then CARD:GPU_BUS pairs
```

Each card pairs with the GPU in its EPYC quadrant (cards in
`~/.config/phi/cards` order: 0 is c6, 1 is 01, 2 is 43, 3 is 81). The
workers must run with slots (`-S 1`); nothing else may hold the cards
(`scripts/card-users.sh`). The rack has no `opencl-headers`: Khronos'
OpenCL-Headers were cloned to `/mnt/raid5/phi/nvidia-p2p`.
`CL_DEVICE_PCI_BUS_ID_NV` (0x4008, `cl_nv_device_attribute_query`; the
value from hwloc's `include/hwloc/opencl.h` and NVIDIA's `cl_ext.h`)
picks each GPU by its PCI bus.

## Measured (2026-10-15, 256 MiB a pair)

One pair, card 0 and GPU c7: end to end 2.89 GB/s with a single thread
doing both legs in turn, **3.30 GB/s** with the two workers (81.3 ms; the
card's leg alone 76.6 ms).

All four pairs at once:

| pair | end to end | GPU to host | host to card | words wrong |
| --- | --- | --- | --- | --- |
| card 0, GPU c7 | 3.24 GB/s | 12.91 GB/s | 3.48 GB/s | 0 of 67,108,864 |
| card 1, GPU 02 | 3.27 GB/s | 12.60 GB/s | 3.47 GB/s | 0 of 67,108,864 |
| card 2, GPU 44 | 3.25 GB/s | 12.64 GB/s | 3.49 GB/s | 0 of 67,108,864 |
| card 3, GPU 82 | 3.24 GB/s | 12.40 GB/s | 3.46 GB/s | 0 of 67,108,864 |
| together | **12.94 GB/s** (1024 MiB in 83.0 ms) | | | PASS |

The cards' links are the limit (the design's 4 x 3.2 = 12.8 GB/s); the
GPUs' copies take a quarter of the time and hide behind them.
