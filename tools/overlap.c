/* overlap.c PID STOPFILE [WINDOW]: who works when, during a generation.
 *
 * Samples every 500 us, from the host only (the card is not asked):
 *   the card: a request outstanding when the window's request seq (64)
 *   differs from its reply seq (256); the request's kind (72);
 *   the host: how many threads of PID are runnable (state R).
 * Until STOPFILE exists. Prints the kinds seen, the worker's counters
 * over the interval (struct vpu_stats at 320), the process's CPU time,
 * and the table card busy/idle against host threads runnable. */
#define _GNU_SOURCE
#include <dirent.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define MAXT 256
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static double cpu_seconds(const char *pid)
{
    char p[64], b[1024]; snprintf(p, sizeof p, "/proc/%s/stat", pid);
    int fd = open(p, O_RDONLY); if (fd < 0) return -1;
    ssize_t n = read(fd, b, sizeof b - 1); close(fd); if (n <= 0) return -1; b[n] = 0;
    char *s = strrchr(b, ')'); if (!s) return -1;
    unsigned long ut = 0, st = 0; int f = 0;      /* fields after ")": state is the 1st, utime the 12th, stime the 13th */
    for (char *t = strtok(s + 2, " "); t; t = strtok(NULL, " ")) {
        f++;
        if (f == 12) ut = strtoul(t, NULL, 10);
        if (f == 13) { st = strtoul(t, NULL, 10); break; }
    }
    return (double)(ut + st) / (double)sysconf(_SC_CLK_TCK);
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: overlap PID STOPFILE [WINDOW]\n"); return 2; }
    const char *pid = argv[1], *stop = argv[2], *win = argc > 3 ? argv[3] : "/dev/shm/phi-hostmem";
    int wf = open(win, O_RDONLY); if (wf < 0) { perror(win); return 1; }
    const volatile uint8_t *w = mmap(NULL, 4096, PROT_READ, MAP_SHARED, wf, 0);
    if (w == MAP_FAILED) { perror("mmap"); return 1; }
    const volatile uint64_t *req_seq = (const void *)(w + 64), *rep_seq = (const void *)(w + 256);
    const volatile uint32_t *req_kind = (const void *)(w + 72);
    const volatile uint64_t *st = (const void *)(w + 320);  /* requests, errors, busy, compute, pull, push ns */

    int tfd[MAXT], nt = 0; char p[320];
    snprintf(p, sizeof p, "/proc/%s/task", pid);
    DIR *d = opendir(p); if (!d) { perror(p); return 1; }
    for (struct dirent *e; (e = readdir(d)) && nt < MAXT;) {
        if (e->d_name[0] == '.') continue;
        snprintf(p, sizeof p, "/proc/%s/task/%s/stat", pid, e->d_name);
        if ((tfd[nt] = open(p, O_RDONLY)) >= 0) nt++;
    }
    closedir(d);

    uint64_t s0[6]; for (int i = 0; i < 6; i++) s0[i] = st[i];
    double t0 = now(), c0 = cpu_seconds(pid);
    static uint64_t kinds[64], busy_r[MAXT + 1], idle_r[MAXT + 1];
    uint64_t last = *req_seq, nreq = 0, missed = 0, samples = 0;
    struct timespec nap = { 0, 500000 };
    for (;;) {
        uint64_t rs = *req_seq, ps = *rep_seq; uint32_t k = *req_kind;
        if (rs != last) { nreq++; if (rs > last + 1) missed += rs - last - 1; kinds[k < 64 ? k : 63]++; last = rs; }
        int r = 0; char b[512];
        for (int i = 0; i < nt; i++) {
            ssize_t n = pread(tfd[i], b, sizeof b - 1, 0); if (n <= 0) continue; b[n] = 0;
            char *s = strrchr(b, ')'); if (s && s[2] == 'R') r++;
        }
        (rs != ps ? busy_r : idle_r)[r]++;
        if ((++samples & 255) == 0 && access(stop, F_OK) == 0) break;
        nanosleep(&nap, NULL);
    }
    double t1 = now(), c1 = cpu_seconds(pid);
    uint64_t s1[6]; for (int i = 0; i < 6; i++) s1[i] = st[i];

    printf("wall %.3f s, %llu samples, %d threads watched\n", t1 - t0, (unsigned long long)samples, nt);
    printf("card: requests %llu, errors %llu, busy %.3f s (%.1f %% of wall), compute %.3f s, pull %.3f s, push %.3f s\n",
           (unsigned long long)(s1[0] - s0[0]), (unsigned long long)(s1[1] - s0[1]),
           (s1[2] - s0[2]) * 1e-9, 100.0 * (s1[2] - s0[2]) * 1e-9 / (t1 - t0),
           (s1[3] - s0[3]) * 1e-9, (s1[4] - s0[4]) * 1e-9, (s1[5] - s0[5]) * 1e-9);
    printf("request kinds sampled (%llu seen, %llu passed between samples):", (unsigned long long)nreq, (unsigned long long)missed);
    for (int k = 0; k < 64; k++) if (kinds[k]) printf(" kind %d x %llu", k, (unsigned long long)kinds[k]);
    printf("\nhost process CPU %.2f s = %.2f cores of wall\n", c1 - c0, (c1 - c0) / (t1 - t0));
    uint64_t bt = 0, it = 0, b2 = 0, i2 = 0;
    for (int r = 0; r <= MAXT; r++) { bt += busy_r[r]; it += idle_r[r]; if (r >= 2) { b2 += busy_r[r]; i2 += idle_r[r]; } }
    printf("card busy   : %5.1f %% of samples; of those, host threads runnable >= 2 in %5.1f %%\n",
           100.0 * bt / samples, bt ? 100.0 * b2 / bt : 0.0);
    printf("card waiting: %5.1f %% of samples; of those, host threads runnable >= 2 in %5.1f %%\n",
           100.0 * it / samples, it ? 100.0 * i2 / it : 0.0);
    printf("runnable host threads, card busy:");
    for (int r = 0; r <= 16; r++) if (busy_r[r]) printf(" %d:%.1f%%", r, 100.0 * busy_r[r] / bt);
    printf("\nrunnable host threads, card waiting:");
    for (int r = 0; r <= 16; r++) if (idle_r[r]) printf(" %d:%.1f%%", r, 100.0 * idle_r[r] / it);
    printf("\n");
    return 0;
}
