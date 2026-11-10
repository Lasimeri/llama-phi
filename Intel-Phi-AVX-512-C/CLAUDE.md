# Intel-Phi-AVX-512-C: notes for an agent working here

Read `CONTRIBUTING.md` first; it is the authority. The non-obvious rules:

- What this is: the Xeon Phi cards as an AVX-512 co-processor, with the
  intercept preload written in C (`host/c`, `libphi512-c.so`;
  `host/c/README.md` is its guide). The assembly preload
  (`host/asm/phi512`, `libphi512-asm.so`) stays as the reference and as
  the modules not yet ported, linked into the C library (the hybrid
  build). Its own repository, Intel-Phi-AVX-512-ASM, is frozen at
  3ee445c; development, the card worker (`card/vpu`) and its deploys
  continue here. Production inference is the ggml backend
  (`host/asm/ggml-phi`, `libggml_phi.so`, through llama.phi's
  `scripts/phi-serve.sh`), not the preload.
- The direction (the person, 2026-10-14): heavy ops (`mul_mat`,
  `mul_mat_id`) on the four cards against resident weights, everything
  else native on the host through the AVX2 twin build; target at least
  12 tok/s generation on the BF16 Qwen3.8 Flash-Next. The target program
  is llama.phi's `build-avx512f` with that model: no stand-in programs,
  smaller models or simulations.
- Everything runs on the GPU rack: builds (`-j128`), tests, card work.
  The working copy is `/mnt/raid5/phi/Intel Phi AVX-512 C` (alias
  `~/.cache/intel-phi-avx512-c`); a desktop clone only relays pushes.
  `make` and `LD_PRELOAD` cannot take the spaces in that path: the
  Makefiles use relative paths, the scripts copy libraries elsewhere.
- C (gcc) for the preload, assembly for the card worker, Rust only for
  what is not ported yet, shell for scripts; never Python, JavaScript or
  Perl, not even for a one-line edit.
- Every code file gets a sibling `.md` with the same stem, written in the
  same change. No em or en dashes anywhere. Every hardware claim cites a
  source: Intel document and section, source file and function, or a
  measurement with its command.
- `make -j128 check LLAMA_CPP_DIR=/mnt/raid5/phi/llama.phi` before every
  commit; a failed check stops the commit. `scripts/gates.sh` (card
  gates) is separate: it restarts a card worker and refuses while
  production holds the card (`scripts/card-users.sh`).
- The cards: one program at a time per card. Production's
  `llama-server` maps all four windows; stop it before gates or bench
  runs of the preload, and say so.
- Traps: `pkill -f` from a tool shell can kill the shell itself (kill by
  pid); a host crash under the preload is only in the kernel journal; the
  host drifts a quarter over tens of minutes (interleave A/B runs).
