// hostram.c: libhostram.so, an LD_PRELOAD library that keeps a GGUF model in the host's
// own memory instead of in a mapping of its file (tools/hostram.md).
//
//   mmap of a .gguf   returns an anonymous region under userfaultfd. A page is filled from
//                     the file (O_DIRECT, no page cache) the first time it is touched, a
//                     2 MiB chunk at a time.
//   a card's ranges   whatever maps over a part of the region (tridma_back, MAP_FIXED) or
//                     unmaps one takes those pages away: they are no longer this library's.
//   HOSTRAM_GO=FILE   once FILE exists, every page still this library's is read and locked
//                     (mlock): the host's share of the model is then anonymous, resident and
//                     never read from the disk again. A read after that is counted and said.
//
// Load order: after libtridma.so in LD_PRELOAD, so that the router's own mmap calls pass
// through this one. User-mode faults only (vm.unprivileged_userfaultfd = 0 allows no more).
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/userfaultfd.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define PAGE 4096ul
#define CHUNK (2ul << 20)
#define GIB 1073741824.0

static void *(*real_mmap)(void *, size_t, int, int, int, off_t);
static int (*real_munmap)(void *, size_t);

static uintptr_t ra, rb;              // the region, whole pages
static int fd_d = -1, fd_b = -1;      // the file: direct, buffered
static int uffd = -1;
static uint64_t nchunks;
static uint8_t *taken;                // a bit a page: no longer this library's
static _Atomic uint8_t *state;        // a byte a chunk: 0 empty, 1 being filled, 2 filled
static uint8_t *locked;               // a byte a chunk: mlocked
static pthread_mutex_t taken_mu = PTHREAD_MUTEX_INITIALIZER;
static atomic_ullong b_taken, b_demand, b_late, b_pre, b_locked, next_chunk;
static atomic_int late, dead, stop_pre, stop_errno;
static int lock_said;
static uint32_t *spill;               // chunks filled on demand after the load, oldest first
static uint64_t spill_cap, spill_n, spill_at;
static uint8_t *trans;                // a byte a chunk: another device's (read for its load, then let go)
static uint32_t *tring;               // such chunks now resident, oldest first
static uint64_t tring_cap, tring_n, tring_at, trans_chunks;
static atomic_ullong b_trans;

static uint64_t env_u64(const char *name, uint64_t dflt) {
    const char *v = getenv(name);
    if (!v || !*v) return dflt;
    char *end;
    const unsigned long long x = strtoull(v, &end, 10);
    return *end ? dflt : x;
}

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double) t.tv_sec + (double) t.tv_nsec * 1e-9;
}

static int is_gguf(int fd, char *path, size_t cap) {
    char link[64];
    snprintf(link, sizeof link, "/proc/self/fd/%d", fd);
    const ssize_t n = readlink(link, path, cap - 1);
    if (n < 5) return 0;
    path[n] = 0;
    return !strcmp(path + n - 5, ".gguf");
}

static int page_taken(uint64_t p) { return (taken[p >> 3] >> (p & 7)) & 1; }

// [a, b) of the address space leaves this library's hands
static void take(uintptr_t a, uintptr_t b) {
    if (a < ra) a = ra;
    if (b > rb) b = rb;
    if (a >= b) return;
    uint64_t n = 0;
    pthread_mutex_lock(&taken_mu);
    for (uint64_t p = (a - ra) / PAGE; p < (b - ra + PAGE - 1) / PAGE; p++)
        if (!page_taken(p)) { taken[p >> 3] |= (uint8_t) (1u << (p & 7)); n++; }
    pthread_mutex_unlock(&taken_mu);
    atomic_fetch_add(&b_taken, n * PAGE);
}

// len bytes of the file at off into buf (both whole pages); past the end reads as zero
static int read_file(uint8_t *buf, uint64_t off, uint64_t len) {
    uint64_t got = 0;
    while (got < len) {
        ssize_t r = fd_d >= 0 ? pread(fd_d, buf + got, len - got, (off_t) (off + got)) : -1;
        if (r < 0) {
            r = pread(fd_b, buf + got, len - got, (off_t) (off + got));
            if (r > 0) posix_fadvise(fd_b, (off_t) (off + got), r, POSIX_FADV_DONTNEED);
        }
        if (r < 0) return -1;
        if (r == 0) break;
        got += (uint64_t) r;
        if (got & (PAGE - 1)) break;          // a short read: the end of the file
    }
    if (got < len) memset(buf + got, 0, len - got);
    return 0;
}

// src into [dst, dst + len): at once, or page by page around what is already there
static uint64_t copy_run(uintptr_t dst, const uint8_t *src, uint64_t len) {
    struct uffdio_copy c = {.dst = dst, .src = (uintptr_t) src, .len = len, .mode = 0};
    if (ioctl(uffd, UFFDIO_COPY, &c) == 0) return len;
    uint64_t done = 0;
    for (uint64_t o = 0; o < len; o += PAGE) {
        struct uffdio_copy p = {.dst = dst + o, .src = (uintptr_t) src + o, .len = PAGE, .mode = 0};
        if (ioctl(uffd, UFFDIO_COPY, &p) == 0) done += PAGE;
        else if (errno == EEXIST) {
            struct uffdio_range w = {.start = dst + o, .len = PAGE};
            ioctl(uffd, UFFDIO_WAKE, &w);
        }
    }
    return done;
}

static uint64_t chunk_len(uint64_t ci) {
    const uint64_t off = ci * CHUNK, all = rb - ra;
    return all - off < CHUNK ? all - off : CHUNK;
}

// chunk ci from the file into the pages still this library's; the bytes put there, or -1
static int64_t fill_chunk(uint64_t ci, uint8_t *buf) {
    const uint64_t off = ci * CHUNK, len = chunk_len(ci), p0 = off / PAGE, np = len / PAGE;
    uint64_t first = 0;
    while (first < np && page_taken(p0 + first)) first++;
    if (first == np) return 0;
    if (read_file(buf, off, len)) return -1;
    uint64_t done = 0;
    for (uint64_t i = first; i < np;) {
        if (page_taken(p0 + i)) { i++; continue; }
        uint64_t j = i;
        while (j < np && !page_taken(p0 + j)) j++;
        done += copy_run(ra + off + i * PAGE, buf + i * PAGE, (j - i) * PAGE);
        i = j;
    }
    return (int64_t) done;
}

// the pages of chunk ci still this library's, locked; 0, or -1 with errno
static int lock_chunk(uint64_t ci) {
    if (locked[ci]) return 0;
    const uint64_t off = ci * CHUNK, np = chunk_len(ci) / PAGE, p0 = off / PAGE;
    for (uint64_t i = 0; i < np;) {
        if (page_taken(p0 + i)) { i++; continue; }
        uint64_t j = i;
        while (j < np && !page_taken(p0 + j)) j++;
        if (mlock((void *) (ra + off + i * PAGE), (j - i) * PAGE)) return -1;
        atomic_fetch_add(&b_locked, (j - i) * PAGE);
        i = j;
    }
    locked[ci] = 1;
    return 0;
}

// MemAvailable in kB, and the share of the swap in use in percent
static uint64_t mem_available_kb(unsigned *swap_pct) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return ~0ull;
    char line[128];
    uint64_t kb = ~0ull, st = 0, sf = 0, v;
    while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "MemAvailable: %lu kB", &v) == 1) kb = v;
        else if (sscanf(line, "SwapTotal: %lu kB", &v) == 1) st = v;
        else if (sscanf(line, "SwapFree: %lu kB", &v) == 1) sf = v;
    }
    fclose(f);
    if (swap_pct) *swap_pct = st ? (unsigned) ((st - sf) * 100 / st) : 0;
    return kb;
}

// a chunk filled after the load: the oldest of them gives its pages back past the budget
static void spilled(uint64_t ci) {
    if (!spill_cap) return;
    if (spill_n == spill_cap) {
        const uint64_t old = spill[spill_at];
        if (!locked[old] && atomic_load(&state[old]) == 2) {
            madvise((void *) (ra + old * CHUNK), chunk_len(old), MADV_DONTNEED);
            atomic_store(&state[old], 0);
        }
        spill[spill_at] = (uint32_t) ci;
        spill_at = (spill_at + 1) % spill_cap;
    } else {
        spill[(spill_at + spill_n++) % spill_cap] = (uint32_t) ci;
    }
}

// a chunk another device reads once: kept until the budget is full, then given back
static void transient(uint64_t ci) {
    if (tring_n == tring_cap) {
        const uint64_t old = tring[tring_at];
        madvise((void *) (ra + old * CHUNK), chunk_len(old), MADV_DONTNEED);
        atomic_store(&state[old], 0);
        tring[tring_at] = (uint32_t) ci;
        tring_at = (tring_at + 1) % tring_cap;
    } else {
        tring[(tring_at + tring_n++) % tring_cap] = (uint32_t) ci;
    }
}

static void *fault_thread(void *arg) {
    (void) arg;
    uint8_t *buf = aligned_alloc(PAGE, CHUNK);
    uint64_t said = 0;
    for (;;) {
        struct uffd_msg m;
        const ssize_t n = read(uffd, &m, sizeof m);
        if (n != (ssize_t) sizeof m) { if (n < 0 && errno == EINTR) continue; break; }
        if (m.event != UFFD_EVENT_PAGEFAULT) continue;
        const uintptr_t a = (uintptr_t) m.arg.pagefault.address & ~(PAGE - 1);
        if (a < ra || a >= rb) continue;
        const uint64_t ci = (a - ra) / CHUNK;
        uint8_t empty = 0;
        int64_t got = 0;
        if (atomic_compare_exchange_strong(&state[ci], &empty, 1)) {
            got = fill_chunk(ci, buf);
            if (trans[ci]) {
                atomic_store(&state[ci], 2);
                transient(ci);
                if (got > 0) atomic_fetch_add(&b_trans, (uint64_t) got);
                continue;
            }
            // locked at once before the load: an unlocked page of the model is one the kernel
            // may push into the swap, and weights do not compress
            if (!atomic_load(&late) && got >= 0 && lock_chunk(ci) && !lock_said++) fprintf(stderr, "hostram: mlock: %s\n", strerror(errno));
            atomic_store(&state[ci], 2);
            if (atomic_load(&late)) spilled(ci);
        } else if (empty == 2) {              // filled once and this page is gone: the page alone
            if (read_file(buf, a - ra, PAGE) == 0) got = (int64_t) copy_run(a, buf, PAGE);
        }                                     // 1: the thread filling it wakes this fault
        if (got < 0) { fprintf(stderr, "hostram: reading the model at %#lx: %s\n", (unsigned long) (a - ra), strerror(errno)); continue; }
        if (atomic_load(&late)) {
            const uint64_t t = atomic_fetch_add(&b_late, (uint64_t) got) + (uint64_t) got;
            if (t >> 28 != said) {
                said = t >> 28;
                fprintf(stderr, "hostram: %.2f GiB read from the file after the load\n", (double) t / GIB);
            }
        } else {
            atomic_fetch_add(&b_demand, (uint64_t) got);
        }
    }
    return NULL;
}

static void *pre_thread(void *arg) {
    (void) arg;
    uint8_t *buf = aligned_alloc(PAGE, CHUNK);
    const uint64_t floor_kb = env_u64("HOSTRAM_FLOOR_MB", 1024) * 1024;
    const unsigned swap_max = (unsigned) env_u64("HOSTRAM_SWAP_PCT", 75);
    const double patience = (double) env_u64("HOSTRAM_PATIENCE_S", 6);
    for (uint64_t k = 0;; k++) {
        if (atomic_load(&dead) || atomic_load(&stop_pre)) break;
        const uint64_t ci = atomic_fetch_add(&next_chunk, 1);
        if (ci >= nchunks) break;
        if (trans[ci]) continue;
        unsigned sw = 0;
        if ((k & 7) == 0 && (mem_available_kb(&sw) < floor_kb || sw >= swap_max)) {   // let the kernel make room, for a while
            const double t0 = now();
            while ((mem_available_kb(&sw) < floor_kb || sw >= swap_max) && now() - t0 < patience && !atomic_load(&stop_pre)) usleep(200000);
            if (mem_available_kb(&sw) < floor_kb || sw >= swap_max) { atomic_store(&stop_errno, ENOMEM); atomic_store(&stop_pre, 1); break; }
        }
        uint8_t empty = 0;
        if (atomic_compare_exchange_strong(&state[ci], &empty, 1)) {
            const int64_t got = fill_chunk(ci, buf);
            atomic_store(&state[ci], 2);
            if (got < 0) { atomic_store(&stop_errno, errno); atomic_store(&stop_pre, 1); break; }
            atomic_fetch_add(&b_pre, (uint64_t) got);
        } else {
            while (atomic_load(&state[ci]) == 1) usleep(1000);
        }
        if (lock_chunk(ci)) { atomic_store(&stop_errno, errno); atomic_store(&stop_pre, 1); break; }
    }
    free(buf);
    return NULL;
}

static void *go_thread(void *arg) {
    (void) arg;
    const char *go = getenv("HOSTRAM_GO");
    struct stat st;
    while (go && *go && stat(go, &st) && !atomic_load(&dead)) usleep(100000);
    if (atomic_load(&dead)) return NULL;
    const double t0 = now();
    for (uint64_t ci = 0; ci < nchunks; ci++)      // what another device read for its load goes back
        if (trans[ci] && atomic_load(&state[ci]) == 2) {
            madvise((void *) (ra + ci * CHUNK), chunk_len(ci), MADV_DONTNEED);
            atomic_store(&state[ci], 0);
        }
    if (trans_chunks)
        fprintf(stderr, "hostram: %.2f GiB of the model are another device's (HOSTRAM_TRANSIENT): %.2f GiB read for its load, given back\n",
                (double) (trans_chunks * CHUNK) / GIB, (double) atomic_load(&b_trans) / GIB);
    const unsigned nt = (unsigned) env_u64("HOSTRAM_THREADS", 4);
    pthread_t th[16];
    const unsigned n = nt < 1 ? 1 : nt > 16 ? 16 : nt;
    for (unsigned i = 0; i < n; i++) pthread_create(&th[i], NULL, pre_thread, NULL);
    for (unsigned i = 0; i < n; i++) pthread_join(th[i], NULL);
    atomic_store(&late, 1);
    // counted now, page by page: a chunk locked earlier may since have been taken away
    uint64_t mine_p = 0, locked_p = 0;
    for (uint64_t ci = 0; ci < nchunks; ci++) {
        if (trans[ci]) continue;
        const uint64_t p0 = ci * CHUNK / PAGE, np = chunk_len(ci) / PAGE;
        for (uint64_t i = 0; i < np; i++)
            if (!page_taken(p0 + i)) { mine_p++; locked_p += locked[ci]; }
    }
    const double mine = (double) (mine_p * PAGE), lk = (double) (locked_p * PAGE);
    fprintf(stderr, "hostram: %.2f GiB of the %.2f GiB the host holds are in locked anonymous memory "
            "(%.2f GiB read now, %.2f GiB on demand before); %.2f GiB of the model are elsewhere (a card's ranges, "
            "another device's tensors); %.2f GiB not loaded%s%s; %.1f s\n",
            lk / GIB, mine / GIB, (double) atomic_load(&b_pre) / GIB, (double) atomic_load(&b_demand) / GIB,
            ((double) (rb - ra) - mine) / GIB, (mine - lk) / GIB,
            atomic_load(&stop_pre) ? ": stopped, " : "", atomic_load(&stop_pre) ? strerror(atomic_load(&stop_errno)) : "",
            now() - t0);
    return NULL;
}

static void *region(size_t len, const char *path) {
    uffd = (int) syscall(SYS_userfaultfd, O_CLOEXEC | UFFD_USER_MODE_ONLY);
    if (uffd < 0) { fprintf(stderr, "hostram: userfaultfd: %s\n", strerror(errno)); return MAP_FAILED; }
    struct uffdio_api api = {.api = UFFD_API, .features = 0};
    if (ioctl(uffd, UFFDIO_API, &api)) { fprintf(stderr, "hostram: UFFDIO_API: %s\n", strerror(errno)); return MAP_FAILED; }
    const size_t all = (len + PAGE - 1) & ~(PAGE - 1);
    void *r = real_mmap(NULL, all, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (r == MAP_FAILED) return r;
    madvise(r, all, MADV_NOHUGEPAGE);
    struct uffdio_register reg = {.range = {.start = (uintptr_t) r, .len = all}, .mode = UFFDIO_REGISTER_MODE_MISSING};
    if (ioctl(uffd, UFFDIO_REGISTER, &reg)) {
        fprintf(stderr, "hostram: UFFDIO_REGISTER: %s\n", strerror(errno));
        real_munmap(r, all);
        return MAP_FAILED;
    }
    fd_d = open(path, O_RDONLY | O_CLOEXEC | O_DIRECT);
    fd_b = open(path, O_RDONLY | O_CLOEXEC);
    nchunks = (all + CHUNK - 1) / CHUNK;
    taken = calloc((all / PAGE + 7) / 8, 1);
    state = calloc(nchunks, 1);
    locked = calloc(nchunks, 1);
    trans = calloc(nchunks, 1);
    tring_cap = env_u64("HOSTRAM_TRANSIENT_MB", 512) / 2;
    if (!tring_cap) tring_cap = 1;
    tring = calloc(tring_cap, sizeof *tring);
    const char *tf = getenv("HOSTRAM_TRANSIENT");
    FILE *tfile = tf && *tf ? fopen(tf, "r") : NULL;
    if (tfile && trans) {
        unsigned long long o, l;
        while (fscanf(tfile, "%llu %llu", &o, &l) == 2)
            for (uint64_t ci = (o + CHUNK - 1) / CHUNK; (ci + 1) * CHUNK <= o + l && ci < nchunks; ci++)
                if (!trans[ci]) { trans[ci] = 1; trans_chunks++; }
    }
    if (tfile) fclose(tfile);
    spill_cap = env_u64("HOSTRAM_SPILL_MB", 1024) / 2;
    spill = calloc(spill_cap ? spill_cap : 1, sizeof *spill);
    if (fd_b < 0 || !taken || !state || !locked || !spill || !trans || !tring) {
        fprintf(stderr, "hostram: setting up for %s: %s\n", path, strerror(errno));
        real_munmap(r, all);
        return MAP_FAILED;
    }
    ra = (uintptr_t) r;
    rb = ra + all;
    pthread_t t;
    pthread_create(&t, NULL, fault_thread, NULL);
    pthread_detach(t);
    pthread_create(&t, NULL, go_thread, NULL);
    pthread_detach(t);
    fprintf(stderr, "hostram: %s (%.2f GiB) is an anonymous region, not a mapping of the file: pages come from the "
            "file once (%s), the host's share is locked when %s exists\n", path, (double) len / GIB,
            fd_d >= 0 ? "direct I/O" : "buffered, dropped from the page cache", getenv("HOSTRAM_GO") ? getenv("HOSTRAM_GO") : "(HOSTRAM_GO unset: at once)");
    return r;
}

void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t off) {
    if (!real_mmap) real_mmap = (void *(*)(void *, size_t, int, int, int, off_t)) dlsym(RTLD_NEXT, "mmap");
    if (!real_munmap) real_munmap = (int (*)(void *, size_t)) dlsym(RTLD_NEXT, "munmap");
    char path[4096];
    if (fd >= 0 && !ra && off == 0 && len >= env_u64("HOSTRAM_MIN_MB", 1024) << 20 && env_u64("HOSTRAM", 1) && is_gguf(fd, path, sizeof path)) {
        void *r = region(len, path);
        if (r != MAP_FAILED) return r;
        fprintf(stderr, "hostram: falling back to the file mapping\n");
    }
    void *r = real_mmap(addr, len, prot, flags, fd, off);
    if (r != MAP_FAILED && ra && (flags & MAP_FIXED)) take((uintptr_t) r, (uintptr_t) r + len);
    return r;
}
void *mmap64(void *addr, size_t len, int prot, int flags, int fd, off_t off) __attribute__((alias("mmap")));

int munmap(void *addr, size_t len) {
    if (!real_munmap) real_munmap = (int (*)(void *, size_t)) dlsym(RTLD_NEXT, "munmap");
    if (ra) {
        if ((uintptr_t) addr <= ra && (uintptr_t) addr + len >= rb) atomic_store(&dead, 1);
        take((uintptr_t) addr, (uintptr_t) addr + len);
    }
    return real_munmap(addr, len);
}

__attribute__((destructor)) static void bye(void) {
    if (!ra) return;
    fprintf(stderr, "hostram: at exit: %.2f GiB locked, %.2f GiB read from the file after the load\n",
            (double) atomic_load(&b_locked) / GIB, (double) atomic_load(&b_late) / GIB);
}
