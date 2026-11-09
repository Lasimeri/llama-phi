/* ggufcensus.c: byte census of a GGUF file's tensors by element type.
 * usage: tcc -run ggufcensus.c FILE.gguf
 * Prints, per type: tensor count and bytes, for (a) every tensor and
 * (b) the 2-D matrices a mul_mat reads (token_embd and the unused
 * blk.64 MTP block excluded). Reads the header only. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

static FILE *f;
static void rd(void *p, size_t n) { if (fread(p, 1, n, f) != n) { fprintf(stderr, "short read\n"); exit(1); } }
static uint32_t u32(void) { uint32_t v; rd(&v, 4); return v; }
static uint64_t u64(void) { uint64_t v; rd(&v, 8); return v; }
static void skip(uint64_t n) { if (fseeko(f, (off_t)n, SEEK_CUR)) { perror("seek"); exit(1); } }
static void skip_str(void) { skip(u64()); }

/* scalar sizes of GGUF value types 0..12 (8 = string, 9 = array: 0 here) */
static const int vsz[13] = { 1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8 };
static void skip_val(uint32_t t)
{
    if (t == 8) { skip_str(); return; }
    if (t == 9) {
        uint32_t et = u32(); uint64_t n = u64();
        if (et == 8) { while (n--) skip_str(); }
        else if (et < 13 && vsz[et]) skip(n * (uint64_t)vsz[et]);
        else { fprintf(stderr, "nested array\n"); exit(1); }
        return;
    }
    if (t < 13 && vsz[t]) { skip((uint64_t)vsz[t]); return; }
    fprintf(stderr, "bad value type %u\n", t); exit(1);
}

/* bytes per block and elements per block, by ggml type id */
static const struct { const char *name; int bytes, elems; } ty[40] = {
    [0] = { "F32", 4, 1 }, [1] = { "F16", 2, 1 }, [2] = { "Q4_0", 18, 32 },
    [3] = { "Q4_1", 20, 32 }, [6] = { "Q5_0", 22, 32 }, [7] = { "Q5_1", 24, 32 },
    [8] = { "Q8_0", 34, 32 }, [9] = { "Q8_1", 36, 32 }, [10] = { "Q2_K", 84, 256 },
    [11] = { "Q3_K", 110, 256 }, [12] = { "Q4_K", 144, 256 }, [13] = { "Q5_K", 176, 256 },
    [14] = { "Q6_K", 210, 256 }, [15] = { "Q8_K", 292, 256 }, [23] = { "IQ4_XS", 136, 256 },
    [30] = { "BF16", 2, 1 },
};

int main(int argc, char **argv)
{
    static uint64_t all_n[40], all_b[40], mm_n[40], mm_b[40];
    uint64_t embd = 0, mtp = 0, big_n = 0, big_b = 0;
    if (argc != 2 || !(f = fopen(argv[1], "rb"))) { fprintf(stderr, "usage: ggufcensus FILE.gguf\n"); return 2; }
    char magic[4]; rd(magic, 4);
    if (memcmp(magic, "GGUF", 4)) { fprintf(stderr, "not GGUF\n"); return 1; }
    uint32_t ver = u32(); uint64_t nt = u64(), nkv = u64();
    for (uint64_t i = 0; i < nkv; i++) { skip_str(); skip_val(u32()); }
    for (uint64_t i = 0; i < nt; i++) {
        char name[256]; uint64_t len = u64();
        if (len >= sizeof name) { fprintf(stderr, "name too long\n"); return 1; }
        rd(name, len); name[len] = 0;
        uint32_t nd = u32(); uint64_t el = 1, d[8] = { 0 };
        if (nd > 8) { fprintf(stderr, "dims\n"); return 1; }
        for (uint32_t k = 0; k < nd; k++) { d[k] = u64(); el *= d[k]; }
        uint32_t t = u32(); (void)u64();
        if (t >= 40 || !ty[t].bytes) { fprintf(stderr, "unknown type %u in %s\n", t, name); return 1; }
        uint64_t b = el / (uint64_t)ty[t].elems * (uint64_t)ty[t].bytes;
        all_n[t]++; all_b[t] += b;
        if (!strncmp(name, "token_embd", 10)) { embd += b; continue; }
        if (!strncmp(name, "blk.64.", 7)) { mtp += b; continue; }
        if (nd == 2) { mm_n[t]++; mm_b[t] += b; if (b >= 4000000) { big_n++; big_b += b; } }
    }
    printf("GGUF v%u, %llu tensors\n", ver, (unsigned long long)nt);
    printf("%-7s %8s %16s | %8s %16s\n", "type", "all", "bytes", "matrices", "bytes");
    uint64_t ta = 0, tm = 0;
    for (int t = 0; t < 40; t++) if (all_n[t]) {
        printf("%-7s %8llu %16llu | %8llu %16llu\n", ty[t].name, (unsigned long long)all_n[t],
               (unsigned long long)all_b[t], (unsigned long long)mm_n[t], (unsigned long long)mm_b[t]);
        ta += all_b[t]; tm += mm_b[t];
    }
    printf("%-7s %8s %16llu | %8s %16llu\n", "total", "", (unsigned long long)ta, "", (unsigned long long)tm);
    printf("token_embd %llu B, blk.64 (unused MTP block) %llu B\n", (unsigned long long)embd, (unsigned long long)mtp);
    printf("matrices of 4,000,000 B and over (PHI_GGML_MIN_BYTES): %llu, %llu B\n",
           (unsigned long long)big_n, (unsigned long long)big_b);
    return 0;
}
