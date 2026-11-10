// ggufranges MODEL.gguf PREFIX...: the byte ranges of a GGUF file ("offset bytes", one a
// line) that hold the tensors whose names begin with one of the prefixes. For
// libhostram.so's HOSTRAM_TRANSIENT: the tensors another device takes at the load
// (tools/hostram.md). tcc -run tools/ggufranges.c MODEL blk.38. blk.39. output.weight
#define _FILE_OFFSET_BITS 64
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static FILE *f;
static uint32_t u32(void) { uint32_t v = 0; if (fread(&v, 4, 1, f) != 1) { fprintf(stderr, "ggufranges: short file\n"); exit(1); } return v; }
static uint64_t u64(void) { uint64_t v = 0; if (fread(&v, 8, 1, f) != 1) { fprintf(stderr, "ggufranges: short file\n"); exit(1); } return v; }
static void skip(uint64_t n) { fseeko(f, (off_t) n, SEEK_CUR); }
static void value(uint32_t t) {
    static const int size[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    if (t == 8) skip(u64());
    else if (t == 9) {
        const uint32_t et = u32();
        const uint64_t n = u64();
        if (et == 8) for (uint64_t i = 0; i < n; i++) skip(u64());
        else if (et < 13 && size[et]) skip(n * (uint64_t) size[et]);
        else { fprintf(stderr, "ggufranges: array of type %u\n", et); exit(1); }
    } else if (t < 13 && size[t]) skip((uint64_t) size[t]);
    else { fprintf(stderr, "ggufranges: value of type %u\n", t); exit(1); }
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: ggufranges MODEL.gguf PREFIX...\n"); return 2; }
    f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    if (u32() != 0x46554747u) { fprintf(stderr, "ggufranges: not a GGUF file\n"); return 1; }
    const uint32_t version = u32();
    const uint64_t nt = u64(), nkv = u64();
    uint64_t align = 32;
    for (uint64_t i = 0; i < nkv; i++) {
        char key[256];
        const uint64_t kl = u64();
        const uint64_t take = kl < sizeof key - 1 ? kl : sizeof key - 1;
        if (fread(key, 1, take, f) != take) return 1;
        key[take] = 0;
        skip(kl - take);
        const uint32_t t = u32();
        if (!strcmp(key, "general.alignment") && t == 4) align = u32();
        else value(t);
    }
    char (*name)[128] = calloc(nt, sizeof *name);
    uint64_t *off = calloc(nt, sizeof *off);
    for (uint64_t i = 0; i < nt; i++) {
        const uint64_t nl = u64();
        const uint64_t take = nl < 127 ? nl : 127;
        if (fread(name[i], 1, take, f) != take) return 1;
        skip(nl - take);
        const uint32_t nd = u32();
        skip(8ull * nd);
        (void) u32();
        off[i] = u64();
    }
    uint64_t data = (uint64_t) ftello(f);
    data = (data + align - 1) / align * align;
    fseeko(f, 0, SEEK_END);
    const uint64_t dlen = (uint64_t) ftello(f) - data;
    uint64_t n = 0, bytes = 0;
    for (uint64_t i = 0; i < nt; i++) {
        int hit = 0;
        for (int a = 2; a < argc && !hit; a++) hit = !strncmp(name[i], argv[a], strlen(argv[a]));
        if (!hit) continue;
        uint64_t len = dlen - off[i];
        for (uint64_t j = 0; j < nt; j++)
            if (off[j] > off[i] && off[j] - off[i] < len) len = off[j] - off[i];
        printf("%llu %llu\n", (unsigned long long) (data + off[i]), (unsigned long long) len);
        n++; bytes += len;
    }
    fprintf(stderr, "ggufranges: GGUF v%u, %llu tensors (first %s, last %s); %llu match, %.2f GiB\n", version,
            (unsigned long long) nt, name[0], name[nt - 1], (unsigned long long) n, (double) bytes / 1073741824.0);
    return 0;
}
