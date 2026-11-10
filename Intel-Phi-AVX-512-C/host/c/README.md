# host/c: the intercept wrapper in C

`libphi512-c.so` is the preload that runs llama.cpp's AVX-512 build on a
host without AVX-512 (the rack's EPYC 7742), with the Xeon Phi cards
doing the work AVX-512 asked for. It replaces the assembly preload of
[`../asm/phi512`](../asm/phi512/phi512.md) one module at a time; that
library stays the reference until this one passes every gate on its own.

## Direction (2026-10-14)

The assembly preload sent every function holding an AVX-512 instruction
to a card. Measured on llama-bench with the BF16 Flash-Next, card 0:
generation at 0.0037 tok/s (273 s a token), about 6,900 card round trips
a token for graph building alone, each at least 38 to 48 us. That floor
by itself (6,900 x 48 us = 0.33 s, arithmetic) keeps it under 3 tok/s
whatever the language.

The person chose, 2026-10-14: **heavy ops on the cards, the rest
native**. `mul_mat` and `mul_mat_id` run on the four cards against
weights kept resident there, spread over each card's threads; graph
building and the small ops run on the EPYC through the AVX2 twin
(`build-phi512-cpu`, the same source built for AVX2). The target is the
plan's stage 4: at least 12 tok/s generation on the BF16 Flash-Next. For
scale, the same machine with the `libggml_phi.so` backend measured 13.14
and 13.36 tok/s generation with GPUs, host and cards, and 13.23 without
the cards (2026-10-14, `/mnt/raid5/phi/bench/setups-2026-10-14`): that
speed is the GPUs' and the host's. `build-avx512f` has no GPU backend;
without GPUs the record has 5.38 tok/s (host, 32 threads) and 5.69 (host
and cards, 2026-10-09).

## Layout

| path | what |
| --- | --- |
| `Makefile` | the build and the checks (below) |
| `phi512/*.c`, `phi512/*.h` | the C modules, each with a sibling `.md` |
| [`phi512/asm.h`](phi512/asm.md) | the assembly symbols the C modules use, while the library is mixed |
| [`phi512/main.c`](phi512/main.md) | the constructor and destructor (the first port) |
| `phi512.map` | the version script: the library exports nothing |
| `imports.want` | the dynamic symbols the library may import (libc) |
| `.clang-format`, `.clang-tidy` | layout and lint, enforced by `make check` |
| `out/` | objects and libraries (not tracked) |

## Commands (on the rack)

```
make -C host/c                 # out/libphi512-c.so: C_MODULES in C, the rest assembly
make -C host/c asm-only        # out/libphi512-c-asm.so: the same link, all assembly (A/B)
make -C host/c check           # build, no AVX-512 from C, imports, clang-format, clang-tidy
make -C host/c compdb          # compile_commands.json at the checkout's root (clangd)
make check                     # the whole repository, host/c included
scripts/gates.sh               # the card gates through out/libphi512-c.so on card 0
scripts/bench/lb-run.sh NAME   # llama-bench under the preload (scripts/bench/README.md)
```

`make` cannot take paths with spaces, so the Makefile uses relative
paths; `LD_PRELOAD` cannot either, so the gate and bench scripts copy the
library to a path without one (`~/.cache/phi512-c-gates`,
`/mnt/raid5/phi/bench/c-DATE`). The checkout also has the alias
`~/.cache/intel-phi-avx512-c`.

## The hybrid build

The library links the C objects of `C_MODULES` with the assembly
objects of every other module of `host/asm/phi512` and `host/asm/common`,
assembled from that tree. A port moves a name into `C_MODULES` (the
Makefile's default) and its symbols out of `asm.h`; every commit
therefore has a library that runs end to end, and `make asm-only` gives
the all-assembly library from the same link for an A/B.

Calls across the boundary follow the assembly's convention (System V's
callee-saved registers kept everywhere, `host/asm/README.md`): C may call
assembly routines whose arguments sit in rdi, rsi, rdx, rcx, r8, r9; an
assembly caller of a C function needs the registers its contract names,
and the C side is built with `-mincoming-stack-boundary=3` for callers
that do not keep the stack 16-byte aligned. Details in
[`phi512/asm.md`](phi512/asm.md).

With the direction above, most of the assembly runtime (`region.S`,
`pool.S`, the per-region coherence) is replaced rather than ported; the
translator (`decode.S`, `fnscan.S`, `mvex.S`, `xlate.S`, `blob.S`) is
ported with its offline checks. Planned check for each translator port:
translate every function of `build-avx512f`'s `libggml-cpu`,
`libggml-base` and `libm` with the assembly and the C module and compare
the blobs byte for byte (no card needed), in `make check`.

## Conventions

- C17 with GNU extensions, gcc on the rack; `-Wall -Wextra -Werror` and
  the warnings in the Makefile; clang-format (`.clang-format`: LLVM base,
  4 spaces, 110 columns, Linux braces) and clang-tidy (bugprone,
  clang-analyzer, performance, portability) clean.
- Never AVX-512 in the preload's own code (`make check`'s `noavx512`).
- libc is allowed (the assembly imported nothing); every new import is a
  line added to `imports.want`.
- The library exports nothing (`phi512.map`): a preloaded library's
  global names would interpose the program's.
- A sibling `.md` per source file; no em or en dashes; every hardware
  claim with its source (CONTRIBUTING.md).

## The regression suite (bugs the wrapper met on llama.cpp)

Every item below was a real failure of the assembly preload on
llama.cpp (2026-10-13 and 14). A C translator must pass a check for each;
the gates ([`../../scripts/gates.md`](../../scripts/gates.md)) hold them.

- `vpextrq`/`vpextrd` to memory store the element, not its lane
  position (packed to the slot's start).
- `vmovss`/`vmovsd` 0F 11 register form is MVR: the destination is
  ModRM.rm, the element ModRM.reg, the upper part vvvv; the moves must
  not clobber the element's source when it is the destination.
- `vcomiss`/`vucomiss` flags are not built in a register the memory
  operand's address uses (an rcx index).
- `vextractps` (sincosf) needs a recipe; `vmovmskps`, `vpmaskmovd`,
  `shlx`/`shrx`/`sarx` too.
- `kmov` with r8 to r15 raises #UD on the card (through eax).
- `endbr64`/`endbr32` are dropped (#UD on Knights Corner).
- A card fault at an address with no host mapping is reported (rip, blob
  offset, source instruction, registers), never made resident.
- The scratch slot must not overlap XFER_HI; in-card call returns must
  not land in the exit sequence; lazily bound PLT stubs; tinyBLAS chunk
  atomics (`lock xadd`); `memcpy@plt` in the graph thread (inline
  copies); the translation arena's size.
- Callees reached through PLT stubs and thunks (a `jmp` at the entry);
  functions sized from `.eh_frame_hdr` when there is no symbol, never a
  `.plt` FDE.
- A host crash under the preload leaves nothing in the bench log: the
  kernel journal has it.
