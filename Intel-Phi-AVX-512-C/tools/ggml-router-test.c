/* ggml-router-test: ggml's own CPU ops of the MoE router (soft_max, then argsort
 * descending, then the top-10 view), from the AVX-512 build of llama.phi,
 * on 4 rows of 512 logits drawn like the BF16 Flash-Next's (a normal
 * spread); each op's output checked against plain scalar C on the same
 * input: soft_max element by element, the argsort as a permutation that
 * orders ggml's own probabilities. Under the preload the ops run on the
 * card. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "ggml.h"
#include "ggml-cpu.h"
static uint64_t rs = 88172645463325252ull;
static double urand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (rs >> 11) * (1.0 / 9007199254740992.0); }
static double nrand(void) { double u = urand() + 1e-12, v = urand(); return sqrt(-2 * log(u)) * cos(6.283185307179586 * v); }
int main(int argc, char **argv) {
    const int n = 512, rows = 4, k = 10, rounds = argc > 1 ? atoi(argv[1]) : 8;
    int bad = 0;
    for (int r = 0; r < rounds; r++) {
        struct ggml_init_params ip = { 64 << 20, NULL, false };
        struct ggml_context *ctx = ggml_init(ip);
        struct ggml_tensor *lg = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, rows);
        float *L = (float *) lg->data;
        for (int i = 0; i < n * rows; i++) L[i] = (float) (2.5 * nrand() - 3.0);
        struct ggml_tensor *pr = ggml_soft_max(ctx, lg);
        struct ggml_tensor *as = ggml_argsort(ctx, pr, GGML_SORT_ORDER_DESC);
        struct ggml_cgraph *g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, as);
        ggml_graph_compute_with_ctx(ctx, g, 1);
        const float *P = (const float *) pr->data;
        const int32_t *A = (const int32_t *) as->data;
        int sb = 0, ab = 0;
        for (int y = 0; y < rows; y++) {
            double m = -1e30, s = 0;
            for (int x = 0; x < n; x++) if (L[y * n + x] > m) m = L[y * n + x];
            for (int x = 0; x < n; x++) s += exp(L[y * n + x] - m);
            for (int x = 0; x < n; x++) { double w = exp(L[y * n + x] - m) / s, d = fabs(w - P[y * n + x]); if (d > 1e-6 + 1e-4 * w) { if (sb < 3) printf("round %d row %d: soft_max[%d] %.9g, scalar %.9g\n", r, y, x, P[y * n + x], w); sb++; } }
            char seen[512] = {0};
            for (int x = 0; x < n; x++) { int a = A[y * n + x]; if (a < 0 || a >= n || seen[a]) { ab++; break; } seen[a] = 1; }
            for (int x = 1; x < n; x++) { int a = A[y * n + x - 1], b = A[y * n + x]; if (a >= 0 && a < n && b >= 0 && b < n && P[y * n + a] < P[y * n + b]) { if (ab < 3) printf("round %d row %d: argsort[%d] = %d (%.9g) before [%d] = %d (%.9g)\n", r, y, x - 1, a, P[y * n + a], x, b, P[y * n + b]); ab++; } }
        }
        printf("round %d: soft_max %s (%d), argsort %s (%d)\n", r, sb ? "WRONG" : "ok", sb, ab ? "WRONG" : "ok", ab);
        bad += sb + ab;
        ggml_free(ctx);
    }
    printf("%s\n", bad ? "FAIL" : "PASS: soft_max and argsort match plain C");
    return bad != 0;
}
