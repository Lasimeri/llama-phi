# hostram.c

`libhostram.so`, loaded into llama-server with `LD_PRELOAD`. It keeps a
GGUF model in the host's own memory instead of in a mapping of its file,
without changing llama.cpp and without the model ever being in host
memory whole.

## Why

llama.cpp has two ways to load: a mapping of the file, whose pages are
page cache (the kernel evicts them under pressure and reads them back
from the disk, and a monitor shows them as cache, not as the program's
memory), or a copy of the whole file in allocated memory before anything
else happens. A model that is split between a GPU, a Xeon Phi and the
host fits neither: mapped, the host's share is not held; copied, the
whole file must fit the host first.

## What it does

- **`mmap` of a `.gguf`** (from its first byte, at least
  `HOSTRAM_MIN_MB`) returns an anonymous region registered with
  `userfaultfd`. llama.cpp goes on as if the file were mapped. A page is
  filled from the file the first time it is touched, a 2 MiB chunk at a
  time, with direct I/O, so nothing passes through the page cache.
- **Whatever maps over a part of the region or unmaps one takes those
  pages away.** That is how a card's share leaves the host: the DMA
  router (`tridma_router.md`) puts its own card-backed range over the
  rows a card holds.
- **Tensors another device loads** (`HOSTRAM_TRANSIENT`, the byte ranges
  `ggufranges.md` prints for the layers that go to VRAM) are filled when
  that device reads them and given back behind it, at most
  `HOSTRAM_TRANSIENT_MB` at a time, never locked.
- **Every other chunk is locked (`mlock`) the moment it is filled.** An
  unlocked page of a model is one the kernel may push into the swap, and
  weights do not compress: the first build left them unlocked and a
  compressed swap filled with them (below).
- **When the file named by `HOSTRAM_GO` exists** (the launcher makes it
  once the server listens), threads read and lock every page that is
  still the host's. From then on a read from the file is counted and
  printed (`... GiB read from the file after the load`); such chunks are
  not locked and at most `HOSTRAM_SPILL_MB` of them stay.
- The load stops instead of exhausting the machine: below
  `HOSTRAM_FLOOR_MB` of available memory, or above `HOSTRAM_SWAP_PCT` of
  the swap in use, it waits `HOSTRAM_PATIENCE_S` for the kernel to make
  room and then gives up, saying how much is not loaded.

| Variable | Default | Meaning |
|---|---|---|
| `HOSTRAM` | 1 | 0: the file is mapped as usual |
| `HOSTRAM_GO` | unset: at once | a file whose existence starts the load of the host's share |
| `HOSTRAM_TRANSIENT` | unset | a file of `offset bytes` lines: another device's tensors |
| `HOSTRAM_TRANSIENT_MB` | 512 | how much of those may be resident at a time |
| `HOSTRAM_THREADS` | 4 | threads of the load |
| `HOSTRAM_FLOOR_MB` | 1024 | available memory under which the load waits |
| `HOSTRAM_SWAP_PCT` | 75 | swap use at which the load waits |
| `HOSTRAM_PATIENCE_S` | 6 | how long it waits before giving up |
| `HOSTRAM_SPILL_MB` | 1024 | chunks read after the load that may stay |
| `HOSTRAM_MIN_MB` | 1024 | smaller files are mapped as usual |

Build and order:

```
gcc -O2 -Wall -Wextra -shared -fPIC -o libhostram.so tools/hostram.c -ldl -lpthread
LD_PRELOAD=libtridma.so:libunpin.so:libhostram.so
```

After `libtridma.so`, so that the router's own `mmap` calls pass through
this library and it learns which ranges the card took. Build to a new
name and rename over the old file: a running server has the old one
mapped.

## Measured

A 1.93 GiB GGUF file through the region against the same file through
`pread`, three 96 MiB windows and 256 MiB at 1 GiB, before and after the
load, with a 64 MiB range taken over and 256 MiB declared another
device's: equal each time. 1.87 GiB locked (`VmLck` 1,962,468 kB,
`RssAnon` 1,981,668 kB), of which 1.59 GiB read in 0.4 s; 0 bytes read
after the load.

A 35.21 GiB Q8_0 mixture-of-experts model ([moe.md](../docs/moe.md)):
16.35 GiB declared the GPU's, 9.33 GB taken by the card's ranges,
`VmLck` 10,819,504 kB (10.32 GiB) for the host with nothing in the swap;
the server listened 33 s after its start and the load of the remainder
took 0.6 s (16.2 GiB had been filled on demand while the card and the
GPU were loaded). Over three requests after it: 0 bytes read from the
file by the library, 31 MiB read from both drives by every process on
the machine.

The first build did not lock on demand. On that model with nothing in
VRAM (the host's share 26.4 GiB, 16.1 GiB of memory available) the
kernel moved the unlocked weights into a compressed swap while the load
locked others, and `systemd-oomd` ended the server 33 s into the load at
33.06 of 33.56 GB of memory and 30.2 of 33.56 GB of swap in use; the
server's group had peaked at 14.5 GiB of memory and 11.6 GiB of swap.
The present build has not been run on that layout.

With 2.0 GiB of memory available the same 1.93 GiB test stopped at the
floor: 1.16 of the host's 1.62 GiB locked, 0.46 GiB not loaded, bytes
equal.

## Limits

- User-mode faults only (`UFFD_USER_MODE_ONLY`: what
  `vm.unprivileged_userfaultfd = 0` allows). A system call that reads a
  page of the region nobody has touched yet fails. llama.cpp, the CUDA
  backend's copies and the card backend's uploads all read in user mode.
- One model a process.
- Locked memory is not reclaimable: the host's share must fit beside
  everything else, and the library says when it does not.
