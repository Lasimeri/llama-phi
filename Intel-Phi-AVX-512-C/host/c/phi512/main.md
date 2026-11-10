# main.c: libphi512-c.so's entry

The first module of the preload in C: the port of
[`../../asm/phi512/main.S`](../../asm/phi512/main.S)
([`main.md`](../../asm/phi512/main.md) there). The rest of the library is
still the assembly modules, linked into the same shared object
([`../README.md`](../README.md), "The hybrid build").

## The constructor

`phi512_init` runs from `.init_array`; the library is linked with
`-z initfirst`, so it runs before any other library's constructor (a
constructor compiled for AVX-512 would otherwise fault before the
handler exists). In order:

1. `PHI512_DISABLE` set: nothing is installed.
2. CPUID leaf 7, EBX bit 16 (AVX512F): a host with AVX-512 needs nothing.
3. The option words of the assembly modules from the environment the
   dynamic linker passes to the constructor (`envp`, not `getenv`: see
   below), with the assembly's own parsing rules: a flag is set
   when the variable is set to anything; a number is an unsigned decimal,
   and anything else (empty, a sign, a letter, past 64 bits) is the
   default.
4. The progress line's clock (`hb_t0`, `hb_last`) from `now_ns`.
5. CPUID 0xD subleaf 2, EBX: where the YMM halves lie in an XSAVE area,
   for the handler's register file (`ymm_off`).
6. `elf_init`, then `cards_init`. No card: one line, no handler, and the
   program ends at its first AVX-512 instruction as it would have.
7. `alt_install` for the main thread, then SIGILL and SIGTRAP to
   `phi512_sigill` with `SA_SIGINFO | SA_ONSTACK` and an empty mask.

Differences from `main.S`, deliberate:

- The environment comes from the constructor's third argument. With
  `-z initfirst` the constructor runs before libc's own init: glibc's
  `_dl_init` (elf/dl-init.c) calls the initfirst object's `call_init`
  before any other object's, libc's included, so `__environ` is not set
  yet and `getenv` finds nothing (measured 2026-10-14: `PHI512_DISABLE=1`
  was ignored). `call_init` passes `(argc, argv, env)` to every
  `DT_INIT_ARRAY` entry. `main.S` never met this: `common/env.S` reads
  `/proc/self/environ` itself.
- `sigaction(2)` through libc: glibc supplies `SA_RESTORER` and its own
  `__restore_rt`, which is what `phi512_sig_restorer` does (one
  `rt_sigreturn`).
- The init lines no longer carry a stray prefix: `main.S` printed them
  through `say` (common/text.S), which writes `ggml-phi: ` first, so the
  assembly printed `ggml-phi: phi512: init: ...`. The C lines start with
  `phi512: `.

| variable | word | default |
| --- | --- | --- |
| `PHI512_VERBOSE` | `verbose` | unset |
| `PHI512_HANDBLOBS` | `handblobs` | unset |
| `PHI512_HEAPCHECK` | `heapcheck` | unset |
| `PHI512_INCALL=0` | `incall_off` = 1 | 1 (calls stay on the card) |
| `PHI512_CARDWB=0` | `cardwb_off` = 1 | 1 |
| `PHI512_PROMOTE=0` | `promote_off` = 1 | 1 |
| `PHI512_CONT=0` | `cont_off` = 1 | 1 |
| `PHI512_TRAMP_ANY=N` | `tramp_any` | 0 |
| `PHI512_DIRTY=0` | `dt_off` = 1 | 1 |
| `PHI512_CARD_ATOMICS=N` | `card_atomics` | 0 |

`PHI512_CARDS`, `PHI512_CARD`, `PHI512_SLOT` and `PHI512_THREADS` are
read by `card.S` itself.

## The destructor

When cards were open: the regions and how their write-backs came (always,
when regions ran), and with `PHI512_VERBOSE` the calls, chunks and
fetch times, the milliseconds with three decimals rounded to the
microsecond (common/text.S `w_ms`). Each line is one `write(2)`.

## libc in the preload

The assembly library imports nothing; this one imports libc. The
constructor runs before libc's own init (above), so it calls only what
needs none: the string functions (their IFUNCs are resolved at
relocation, before any init), `vsnprintf` into a buffer, `write`,
`sigemptyset` and `sigaction`; no `getenv`, no `malloc`, no stdio
streams. `imports.want` lists every import. A future C module that runs
inside the SIGILL handler may call libc more freely: by then libc is
initialized, and the fault is raised by the program's own AVX-512 code,
never inside libc, so libc's internal locks are not held by the
faulting thread. Each such module still documents what it calls.

## Test

`make -C host/c check` builds it, checks the library holds no AVX-512
instruction from C and imports only what `imports.want` lists;
`scripts/gates.sh` runs the card gates through it.
