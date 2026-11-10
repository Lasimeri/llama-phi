# Xeon Phi cards reading GPU memory by DMA (design, 2026-10-15)

The person's direction (2026-10-15): keep the active mixture-of-experts
weights of GLM-5.3-Flash in the four RTX 3080s' memory, as much as fits,
everything else in system memory, and have the Phi cards stream the
weights straight from GPU memory with their DMA engines. This document
designs that path on documented interfaces only: each memory region is
shared by the driver that owns it, through dma-buf, and mapped for the
card by the card's own daemon. Nothing here works around a driver's or the
kernel's protections; where a needed interface is not available yet the
design says so and keeps a fallback.

## What bounds it (arithmetic from measured or documented numbers)

| quantity | value | source |
| --- | --- | --- |
| a Phi card's link | 3.2 GB/s (PCIe Gen2 x8), 12.8 GB/s for four | measured, `card/vpu/cdma.md` (64 KiB in 20.3 us) |
| GLM-5.3-Flash expert bytes a token (BF16) | 8 used x 42 MoE layers x 50.3 MB = 16.9 GB | GGUF metadata (`gguf-kv`): 288 experts, ff 2048, n_embd 4096 |
| ceiling if the cards serve every expert | 16.9 GB / 12.8 GB/s = 1.3 s a token, 0.76 tok/s | arithmetic |
| one 3080 reading its own memory | about 760 GB/s | NVIDIA specification, not measured here |
| measured today, BF16 streamed from disk | 0.18 to 0.20 tok/s | `bench/glm53flash-2026-10-14` |

The cards' links, not the GPUs, set the ceiling of this design.

## Facts of the machine (2026-10-15)

- GPUs: 4 x GeForce RTX 3080 20 GB at 02:00, 44:00, 82:00, c7:00; each
  sits in the same EPYC quadrant as one card (01, 43, 81, c6). Driver:
  NVIDIA open kernel module 615.71.09.
- BAR1 is 256 MiB a GPU and cannot grow: `resource1_resize` reads `0x1c0`
  (64, 128 and 256 MB supported). Vulkan reports a 246 MiB device-local
  heap that the host can map: the part of the GPU's memory a peer device
  can address at one time.
- Vulkan 1.4.351 with `VK_EXT_external_memory_dma_buf` and
  `VK_KHR_external_memory_fd`.
- Kernel 7.2.9: `CONFIG_PCI_P2PDMA=y`, `CONFIG_IOMMUFD=m`,
  `CONFIG_VFIO_DEVICE_CDEV=y`; `/dev/iommu` exists (root:root 0660);
  the cards' VFIO devices are also available as `/dev/vfio/devices/vfioN`.
  The IOMMU is on (93 groups); the cards are bound to vfio-pci and driven
  by the stack's `phictl` through the legacy VFIO container.
- A card reaches IOVA `x` at card address `0x80_0000_0000 + x`: SMPT
  entries 0 to 3 map the first 64 GiB of IOVA space one to one
  (Intel-Phi-3120A `docs/research/memory-map.md`, `phictl/boot.md`).

## The documented interfaces and their state

| link | interface | state on this machine |
| --- | --- | --- |
| GPU memory to a shareable object | Vulkan `VK_EXT_external_memory_dma_buf` on memory from the 246 MiB heap; or CUDA `cuMemGetHandleForAddressRange(CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD)` | Vulkan exposes the extension; NVIDIA open-gpu-kernel-modules issue #1130 reports that a Vulkan-exported dma-buf of host-visible memory could not be imported by a third-party device (driver 595.71.05). CUDA's dma-buf export is reported unsupported on GeForce (`CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED` = 0, NVIDIA forum, 2026-09-30). To be probed on 615.71.09. |
| the object into the card's IOMMU view | iommufd `IOMMU_IOAS_MAP_FILE` with the dma-buf fd | kernel documentation: "supported dma-buf files may also be accepted ... currently supported dma-bufs include single-range VFIO PCI dma-bufs ... other dma-bufs may be rejected" (docs.kernel.org/userspace-api/iommufd.html). A 2026-02 series generalises the importer through mapping types (LWN 1059366). To be probed. |
| the card's DMA reading it | the worker's DMA channel 7 (`card/vpu/cdma.S`) reading card address `0x80_0000_0000 + IOVA` | works today for host memory; a peer read through the EPYC root complex is not measured |
| lifecycle | dma-buf `move_notify` revocation, per the kernel's P2P DMA documentation (docs.kernel.org/driver-api/pci/p2pdma.html) | provided by the exporter and iommufd |

## Design

```
┌─ GPU c7 (RTX 3080, 20 GB) ─────────────┐    ┌─ host ─────────────────────────────────┐
│ hot experts in VRAM                    │    │ vram-dma helper (C): Vulkan device,    │
│   │ GPU copy, hundreds of GB/s         │    │ 246 MiB heap exported as a dma-buf     │
│   ▼                                    │    │   │ fd over phictl's control socket    │
│ staging ring, 246 MiB heap             │    │   ▼                                    │
└──────────────────────▲─────────────────┘    │ phictl --card 0: iommufd IOAS,         │
                       │ peer read (PCIe)     │ IOMMU_IOAS_MAP_FILE at IOVA 48 GiB     │
                       │                      └───────────────────┬────────────────────┘
┌──────────────────────┴─ card c6 (Xeon Phi 3120A) ───────────────┴────────────────────┐
│ worker K_VPULL: DMA channel 7 reads 0x80_0000_0000 + 48 GiB into pool pages,         │
│ then the BF16 expert matmul on 114 threads; the reply frees the ring slots           │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

1. **Placement.** llama.phi places as many expert tensors as fit in
   each GPU's memory (up to the person's 75% limit), the rest in system
   memory. A hot-expert set per layer (ranked from a live routing log, as
   `PHI_GGML_EXPERTS` does for the cards) decides which experts live in
   VRAM when not all fit.
2. **Staging.** Each GPU's 246 MiB host-visible heap is carved into a
   ring of slots (for example 48 slots of 5 MiB, or whole experts of
   16.8 MB per matrix). For a token, the backend asks the GPU to copy the
   selected experts' rows from their VRAM tensors into free slots (a
   device-to-device copy at memory speed); completion is a Vulkan
   timeline semaphore, or the CUDA stream's event when the copy is issued
   by llama.cpp's own CUDA context.
3. **Sharing.** A small helper process owns the Vulkan allocation of the
   heap and exports it once as a dma-buf. It hands the fd to `phictl`
   over phictl's control socket (`SCM_RIGHTS`): a consented, documented
   transfer between two processes of the same user.
4. **Mapping.** `phictl` drives its card through the VFIO device cdev
   and iommufd (`/dev/vfio/devices/vfioN`, `/dev/iommu`) instead of the
   legacy container, and maps the dma-buf into the card's IOAS with
   `IOMMU_IOAS_MAP_FILE` at an IOVA below 64 GiB that its allocator never
   uses (48 GiB). The card then reads the ring at card address
   `0x80_0000_0000 + 48 GiB`. If the exporter revokes the buffer,
   iommufd's `move_notify` handling removes the mapping.
5. **Pulling and computing.** A new worker request kind (`K_VPULL`)
   names ring slots; the dispatcher's DMA channel 7 copies them into pool
   pages, and the matmul kernels (BF16 already supported, `MM_BF16`) run
   on them. The reply frees the slots; the backend reuses a slot only
   after the reply.
6. **Integration.** The ggml-phi backend takes `MUL_MAT_ID` for the
   VRAM-resident experts: it reads the `ids`, queues the copies of the
   selected experts into ring slots, posts `K_VPULL` plus the activation
   rows to the card next to that GPU, and gathers the outputs. Experts in
   system memory keep today's path.

## What changes where (from the code, 2026-10-15)

- **Card worker.** No request carries a mixture's weights today:
  `K_MATMUL_ID` and `K_MATMUL_MORE` refuse a weight that is not
  resident (`card/vpu/matmul.S`, the `a_id` 0 checks), and the one
  window-weight path (plain `K_MATMUL` with `MM_A_OFF`) reads through
  the block device (`vpu_pull`, `card/vpu/worker.S`), not DMA. `K_VPULL`
  is new: a list of slot addresses, pulled with `cdma_submit_phys` into
  pool pages in pieces of at most 16383 lines (`card/vpu/cdma.md`), then
  the existing BF16 kernels (`rows_slice_multi_f`).
- **Backend.** It accepts only host-memory weights: `weight_readable`
  uses `ggml_backend_buft_is_host` and `supports_buft` is host-only
  (`host/asm/ggml-phi/glue.S`); no `offload_op` is installed; the `ids`
  are read from host memory. For VRAM experts it needs: the ids copied
  to the host (`ggml_backend_tensor_get`), the ring's slots chosen and
  filled by the GPU, `K_VPULL` posted, and `plan`/`take` taught that these
  experts have no host rows.
- **Window.** Each card's window is 4 GiB on the rack, but the worker
  maps only its first 768 MiB (`card/vpu/defs.inc` `WIN_BYTES`); the
  fallback ring fits above that.
- **Measured so far.** The card's DMA from host memory is measured only up
  to 64 KiB a copy (3.2 GB/s); the large-copy rate is part of P3. The
  2026-10-09 record found the host CPU computing Flash-Next's experts at
  15.3 tok/s and that streaming misses over the links did not beat it
  (`docs/results/2026-10-09-bf16-cards-whole-experts.md`).

## Probes before building (each a go or no-go)

| probe | what it shows | needs |
| --- | --- | --- |
| P1 export | Vulkan on 615.71.09 exports a dma-buf of the 246 MiB heap (`vkGetMemoryFdKHR` with `VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT`), and the CPU sees the pattern written through it | no privileges |
| P2 import | iommufd accepts that dma-buf (`IOMMU_IOAS_ALLOC`, then `IOMMU_IOAS_MAP_FILE`) | access to `/dev/iommu`: a udev rule giving group `phi` read and write on it, installed by the person |
| P3 peer read | a card, attached to an IOAS through the VFIO cdev, reads the mapped ring by DMA: bytes equal, GB/s measured | phictl on iommufd: a phictl change and restart, which reboots the cards (asked first) |

If P1 or P2 fails on this driver and kernel, the direct path waits for
NVIDIA's dma-buf peer support on GeForce or for iommufd's generic
dma-buf importer; the fallback until then is the same ring in host
memory (the GPU's copy engine writes the selected experts into the card's
shared window, the card pulls them as today): the same ceiling, twice the
host memory traffic of the stream (about 26 GB/s at full rate), no new
interfaces.
