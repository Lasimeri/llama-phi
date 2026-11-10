/* main.c: libphi512-c.so's entry, the port of host/asm/phi512/main.S.
 * The constructor reads the environment, finds the cards, maps the
 * executable's symbols and installs the SIGILL and SIGTRAP handler
 * (sigill.S); the destructor reports what ran. See main.md. */
#define _GNU_SOURCE
#include <cpuid.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>

#include <string.h>
#include <unistd.h>

#include "asm.h"

/* 1: blobs leave the card at every call (PHI512_INCALL=0); blob.S reads it. */
uint64_t incall_off;

/* One line on standard error in one write(2), so lines from threads do not
 * interleave. */
__attribute__((format(printf, 1, 2))) static void say(const char *fmt, ...)
{
    char b[512];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(b, sizeof b - 1, fmt, ap);
    va_end(ap);
    if (n < 0)
        return;
    if (n > (int)sizeof b - 2)
        n = (int)sizeof b - 2;
    b[n++] = '\n';
    for (const char *p = b; n > 0;) {
        ssize_t w = write(2, p, (size_t)n);
        if (w <= 0)
            return;
        p += w;
        n -= (int)w;
    }
}

/* The environment as the dynamic linker passed it to the constructor. The
 * library is linked -z initfirst, so the constructor runs before libc's
 * own init (glibc elf/dl-init.c, _dl_init: the initfirst object's
 * call_init comes first) and getenv() still sees no environment; glibc
 * calls every DT_INIT_ARRAY entry with (argc, argv, env). */
static char **environment;

static const char *env_get(const char *name)
{
    size_t n = strlen(name);
    for (char **e = environment; e != NULL && *e != NULL; e++)
        if (strncmp(*e, name, n) == 0 && (*e)[n] == '=')
            return *e + n + 1;
    return NULL;
}

/* common/env.S env_set: set to anything, empty included. */
static int env_set(const char *name) { return env_get(name) != NULL; }

/* common/env.S env_u64: the value as an unsigned decimal; the default when
 * unset, empty, not all digits, or past 64 bits (Rust's parse::<u64>). */
static uint64_t env_u64(const char *name, uint64_t dflt)
{
    const char *v = env_get(name);
    if (v == NULL || *v == '\0')
        return dflt;
    uint64_t x = 0;
    for (; *v != '\0'; v++) {
        if (*v < '0' || *v > '9')
            return dflt;
        uint64_t d = (uint64_t)(*v - '0');
        if (x > (UINT64_MAX - d) / 10)
            return dflt;
        x = x * 10 + d;
    }
    return x;
}

/* common/text.S w_ms: ns as milliseconds with three decimals, rounded to
 * the microsecond. */
static const char *ms(char *b, size_t n, uint64_t ns)
{
    uint64_t us = (ns + 500) / 1000;
    snprintf(b, n, "%llu.%03llu", (unsigned long long)(us / 1000), (unsigned long long)(us % 1000));
    return b;
}

__attribute__((constructor)) static void phi512_init(int argc, char **argv, char **envp)
{
    unsigned a, b, c, d;
    (void)argc;
    (void)argv;
    environment = envp;
    if (env_set("PHI512_DISABLE"))
        return;
    /* A host with AVX-512 needs nothing: CPUID leaf 7, EBX bit 16. */
    if (__get_cpuid_count(7, 0, &a, &b, &c, &d) && (b & bit_AVX512F))
        return;
    verbose = env_set("PHI512_VERBOSE");
    handblobs = env_set("PHI512_HANDBLOBS");
    heapcheck = env_set("PHI512_HEAPCHECK");
    incall_off = env_u64("PHI512_INCALL", 1) == 0;    /* blob.S .Lincall */
    cardwb_off = env_u64("PHI512_CARDWB", 1) == 0;    /* region.S wu_writeback */
    promote_off = env_u64("PHI512_PROMOTE", 1) == 0;  /* region.S promote_check */
    cont_off = env_u64("PHI512_CONT", 1) == 0;        /* region.S continuation */
    tramp_any = env_u64("PHI512_TRAMP_ANY", 0);       /* region.S trampoline */
    dt_off = env_u64("PHI512_DIRTY", 1) == 0;         /* region.S dt_* */
    card_atomics = env_u64("PHI512_CARD_ATOMICS", 0); /* blob.S */
    if (verbose)
        say("phi512: init: environment read");
    hb_t0 = hb_last = now_ns(); /* the progress line's clock (region.S heartbeat) */
    /* Where the YMM halves lie in an XSAVE area: CPUID 0xD, subleaf 2, EBX. */
    __cpuid_count(0xd, 2, a, b, c, d);
    ymm_off = b;
    if (verbose)
        say("phi512: init: cpuid done");
    elf_init();
    if (verbose)
        say("phi512: init: executable mapped");
    if (cards_init() == 0) {
        say("phi512: no card (no window, no worker): AVX-512 instructions will end the program");
        return;
    }
    if (verbose)
        say("phi512: init: cards opened");
    alt_install(); /* the main thread's; the others get theirs at their first fault */
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = phi512_sigill;
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGILL, &sa, NULL);
    sigaction(SIGTRAP, &sa, NULL); /* the single step after an instruction the host ran (region.S) */
    if (verbose)
        say("phi512: %llu card(s), the handler installed; seeds: %llu", (unsigned long long)ncards,
            (unsigned long long)nseeds);
}

__attribute__((destructor)) static void phi512_fini(void)
{
    char m1[32], m2[32], m3[32];
    if (ncards == 0)
        return;
    if (stat_regions != 0) /* always, when regions ran */
        say("phi512: %llu region(s); write-backs from the card's own diff %llu, lists that overflowed "
            "(units pulled) %llu, lines written back %llu",
            (unsigned long long)stat_regions, (unsigned long long)stat_cardwb,
            (unsigned long long)stat_cardwb_full, (unsigned long long)stat_wb_lines);
    if (verbose)
        say("phi512: the cards ran %llu call(s) in %s ms; %llu chunk(s) made resident in %s ms; "
            "outputs fetched in %s ms",
            (unsigned long long)stat_calls, ms(m1, sizeof m1, stat_call_ns),
            (unsigned long long)stat_resident, ms(m2, sizeof m2, stat_resident_ns),
            ms(m3, sizeof m3, stat_push_ns));
}
