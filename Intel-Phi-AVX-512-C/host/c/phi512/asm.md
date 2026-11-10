# asm.h: the assembly modules seen from C

While the preload is part C and part assembly, `asm.h` declares every
assembly symbol a C module reads, writes or calls, with the file that
defines it. The C modules are compiled against it; the link
(`-z defs`) fails on a name that no object defines.

## Why calls across the boundary are safe

The assembly modules keep the System V convention's callee-saved
registers (rbx, rbp, r12 to r15) in every routine, exported or internal,
and may clobber the others (`host/asm/README.md`, "Conventions";
`host/asm/common/text.md`). A C caller assumes exactly that. A routine
whose arguments are not in rdi, rsi, rdx, rcx, r8, r9 in that order
(`say` takes its string in rsi; `parse_u64` returns a flag in rdx) is
not declared here: C does that work itself.

The other direction (assembly calling a C function) is where the
conventions can differ: an assembly caller may pass an argument in a
register C does not read, or call with a stack that is not 16-byte
aligned. The C modules are compiled with
`-mincoming-stack-boundary=3` (gcc then realigns where it needs to,
host/c/Makefile), and a C function an assembly module calls is declared
with the registers its `.md` contract names. None exists yet:
`main.c` is only called by the dynamic linker.

## The data words

Every option word and counter is a `.quad` in `.bss` of the defining
module (`nm` of `host/asm/out/p512_*.o`, 2026-10-14):

| symbol | defined in | meaning |
| --- | --- | --- |
| `verbose`, `heapcheck`, `ymm_off`, `stat_calls`, `stat_call_ns` | `sigill.S` | options, the XSAVE offset, the call counters |
| `handblobs` | `xlfn.S` | hand-written blobs |
| `card_atomics` | `blob.S` | atomics on the card |
| `cardwb_off`, `promote_off`, `cont_off`, `tramp_any`, `dt_off`, `hb_t0`, `hb_last`, `stat_regions`, `stat_cardwb`, `stat_cardwb_full`, `stat_wb_lines` | `region.S` | region options, the progress clock, region counters |
| `ncards`, `stat_resident`, `stat_resident_ns`, `stat_push_ns` | `card.S` | the cards open, residency counters |
| `nseeds` | `seeds.S` (`.data`) | the hand-seeded functions |
| `incall_off` | `main.c` (was `main.S`) | blobs leave the card at every call |

When a module moves to C, its symbols are defined in its `.c` file and
leave this header for that module's own header.
