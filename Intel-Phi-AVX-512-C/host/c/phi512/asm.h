/* asm.h: the assembly modules' symbols the C modules use, while the
 * preload is part C and part assembly (host/c/README.md, "The hybrid
 * build"). Each name is defined in the .S file named beside it under
 * host/asm/phi512 or host/asm/common. A module ported to C defines the
 * symbols it took over and they move out of this header. See asm.md. */
#ifndef PHI512_ASM_H
#define PHI512_ASM_H

#include <signal.h>
#include <stdint.h>

/* Every symbol here is in this shared object: hidden, so C reaches it
 * directly and the library exports nothing. */
#pragma GCC visibility push(hidden)

/* The handler and its option words (sigill.S). Every word is a .quad. */
extern uint64_t verbose;   /* PHI512_VERBOSE set: a line per step */
extern uint64_t heapcheck; /* PHI512_HEAPCHECK set: the heap walked per region */
extern uint64_t ymm_off;   /* CPUID 0xD.2 EBX: the YMM halves in an XSAVE area */
extern uint64_t stat_calls, stat_call_ns;
void phi512_sigill(int sig, siginfo_t *info, void *uc);
void alt_install(void); /* the calling thread's alternate signal stack */

/* Translation (xlfn.S, blob.S). */
extern uint64_t handblobs;    /* PHI512_HANDBLOBS set: the hand-written blobs */
extern uint64_t card_atomics; /* PHI512_CARD_ATOMICS: atomics run on the card */

/* Regions (region.S): option words (1 = the feature off) and counters. */
extern uint64_t cardwb_off, promote_off, cont_off, tramp_any, dt_off;
extern uint64_t hb_t0, hb_last; /* the progress line's clock, ns */
extern uint64_t stat_regions, stat_cardwb, stat_cardwb_full, stat_wb_lines;

/* The cards (card.S). */
extern uint64_t ncards;
extern uint64_t stat_resident, stat_resident_ns, stat_push_ns;
uint64_t cards_init(void); /* nonzero when at least one card is open */

/* The executable and the seeds (elf.S, seeds.S). */
extern uint64_t nseeds;
void elf_init(void);

/* common/mem.S */
uint64_t now_ns(void); /* CLOCK_MONOTONIC in ns */

#pragma GCC visibility pop

#endif
