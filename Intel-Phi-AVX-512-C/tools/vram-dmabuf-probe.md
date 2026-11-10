# vram-dmabuf-probe.c: probes P1 and P2 of the Phi-reads-VRAM design

The first two go or no-go probes of
[`../docs/research/phi-vram-dma.md`](../docs/research/phi-vram-dma.md),
on documented interfaces only.

- **P1 (export):** on the GPU at the given PCI bus, allocate memory from
  the device-local, host-visible heap (the GPU's 256 MiB BAR window,
  246 MiB usable) as a buffer exportable as a dma-buf
  (`VK_EXT_external_memory_dma_buf`, `VK_KHR_external_memory_fd`), write
  a pattern through the Vulkan mapping, export the dma-buf with
  `vkGetMemoryFdKHR`, and check the fd is a dma-buf of the right size
  whose own `mmap` shows the pattern.
- **P2 (import):** when `/dev/iommu` can be opened, create an iommufd IOAS
  (`IOMMU_IOAS_ALLOC`), map the dma-buf into it at IOVA 48 GiB
  (`IOMMU_IOAS_MAP_FILE`, read only), then unmap it and destroy the IOAS.
  No device is attached: the probe only asks whether iommufd accepts this
  exporter's dma-buf. `/dev/iommu` is root-only on the rack until the
  phi group is given access (`80-phi-iommufd.rules`, the same form as the
  stack's `80-phi-vfio.rules`).

```
gcc -O1 -Wall -Wextra -I$VULKAN_HEADERS/include -o vram-dmabuf-probe tools/vram-dmabuf-probe.c -lvulkan
./vram-dmabuf-probe c7 64        # GPU on bus c7 (next to card 0), 64 MiB
```

The rack has the Vulkan loader but not the headers: Khronos'
Vulkan-Headers at tag v1.4.351 (the driver's version) were cloned to
`/mnt/raid5/phi/nvidia-p2p/Vulkan-Headers`.

## Measured (2026-10-15, driver 615.71.09, kernel 7.2.9)

GPU c7: the BAR heap is memory type 5, heap 2, 246 MiB; dma-buf buffers
are exportable and importable, not dedicated-only. 64 MiB exported: the
fd is `/dmabuf:` of 67,108,864 bytes, and its `mmap` shows the pattern
written through Vulkan (0 of 16,777,216 words differ). **P1 passes.**
P2 has not run yet: `/dev/iommu` returns "Permission denied" to the phi
group.
