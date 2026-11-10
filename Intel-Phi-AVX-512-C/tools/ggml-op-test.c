/* ggml-op-test: ggml's own CPU ops (the build linked: the AVX2 twin natively,
 * the AVX-512 build under the preload on the card) on random inputs,
 * each element against a double-precision reference: the largest error
 * in units of float epsilon relative to max(|ref|, floor), per op. The
 * card's numbers next to the twin's show an op whose card form is less
 * precise (the final logits drifted 1.6% from the twin, 2026-10-14). */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ggml.h"
#include "ggml-cpu.h"
static uint64_t rs = 0x9E3779B97F4A7C15ull;
static double ur(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (rs >> 11) * (1.0 / 9007199254740992.0); }
static uint16_t bf16(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16); }
static float fbf(uint16_t h) { uint32_t u = (uint32_t) h << 16; float f; memcpy(&f, &u, 4); return f; }
enum { N = 4096 };
static double ref1(int op, double x) {
    switch (op) {
    case 0: return exp(x); case 1: return 1 / (1 + exp(-x)); case 2: return x / (1 + exp(-x));
    case 3: return tanh(x); case 4: return log1p(exp(x)); case 5: return sqrt(x); case 6: return log(x);
    case 7: return sin(x); case 8: return cos(x);
    }
    return 0;
}
static const char *n1[] = { "exp", "sigmoid", "silu", "tanh", "softplus", "sqrt", "log", "sin", "cos" };
int main(void) {
    struct ggml_init_params ip = { 256 << 20, NULL, false };
    struct ggml_context *ctx = ggml_init(ip);
    struct ggml_tensor *x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
    float *X = (float *) x->data;
    for (int i = 0; i < N; i++) X[i] = (float) (20 * ur() - 10);
    struct ggml_tensor *xa = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
    float *XA = (float *) xa->data;
    for (int i = 0; i < N; i++) XA[i] = fabsf(X[i]) + 1e-3f;
    struct ggml_tensor *o[9];
    o[0] = ggml_exp(ctx, x); o[1] = ggml_sigmoid(ctx, x); o[2] = ggml_silu(ctx, x); o[3] = ggml_tanh(ctx, x);
    o[4] = ggml_softplus(ctx, x); o[5] = ggml_sqrt(ctx, xa); o[6] = ggml_log(ctx, xa); o[7] = ggml_sin(ctx, x); o[8] = ggml_cos(ctx, x);
    /* rms_norm and soft_max over rows of 2560 / 512; a BF16 mul_mat 2560 x 512 by 2560 x 4 */
    struct ggml_tensor *r = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, 4);
    float *R = (float *) r->data;
    for (int i = 0; i < 2560 * 4; i++) R[i] = (float) (4 * ur() - 2);
    struct ggml_tensor *rn = ggml_rms_norm(ctx, r, 1e-6f);
    struct ggml_tensor *sm_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 4);
    float *S = (float *) sm_in->data;
    for (int i = 0; i < 512 * 4; i++) S[i] = (float) (8 * ur() - 6);
    struct ggml_tensor *sm = ggml_soft_max(ctx, sm_in);
    struct ggml_tensor *w = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, 2560, 512);
    uint16_t *W = (uint16_t *) w->data;
    for (int i = 0; i < 2560 * 512; i++) W[i] = bf16((float) (0.1 * (2 * ur() - 1)));
    struct ggml_tensor *mm = ggml_mul_mat(ctx, w, r);
    struct ggml_cgraph *g = ggml_new_graph(ctx);
    for (int k = 0; k < 9; k++) ggml_build_forward_expand(g, o[k]);
    ggml_build_forward_expand(g, rn); ggml_build_forward_expand(g, sm); ggml_build_forward_expand(g, mm);
    ggml_graph_compute_with_ctx(ctx, g, 1);
    const double eps = 1.1920928955078125e-07;
    for (int k = 0; k < 9; k++) {
        const float *Y = (const float *) o[k]->data; double worst = 0; int wi = 0;
        for (int i = 0; i < N; i++) { double in = (k == 5 || k == 6) ? XA[i] : X[i]; double rf = ref1(k, in); double e = fabs(Y[i] - rf) / fmax(fabs(rf), 1e-6) / eps; if (e > worst) { worst = e; wi = i; } }
        printf("%-9s max %9.1f eps (at x %.6g: got %.9g, ref %.9g)\n", n1[k], worst, (k == 5 || k == 6) ? XA[wi] : X[wi], ((const float *) o[k]->data)[wi], ref1(k, (k == 5 || k == 6) ? XA[wi] : X[wi]));
    }
    { const float *Y = (const float *) rn->data; double worst = 0;
      for (int row = 0; row < 4; row++) { double s = 0; for (int i = 0; i < 2560; i++) s += (double) R[row * 2560 + i] * R[row * 2560 + i]; double sc = 1 / sqrt(s / 2560 + 1e-6);
        for (int i = 0; i < 2560; i++) { double rf = R[row * 2560 + i] * sc, e = fabs(Y[row * 2560 + i] - rf) / fmax(fabs(rf), 1e-6) / eps; if (e > worst) worst = e; } }
      printf("%-9s max %9.1f eps\n", "rms_norm", worst); }
    { const float *Y = (const float *) sm->data; double worst = 0;
      for (int row = 0; row < 4; row++) { double m = -1e30, s = 0; for (int i = 0; i < 512; i++) if (S[row * 512 + i] > m) m = S[row * 512 + i]; for (int i = 0; i < 512; i++) s += exp(S[row * 512 + i] - m);
        for (int i = 0; i < 512; i++) { double rf = exp(S[row * 512 + i] - m) / s, e = fabs(Y[row * 512 + i] - rf) / fmax(fabs(rf), 1e-12) / eps; if (e > worst) worst = e; } }
      printf("%-9s max %9.1f eps\n", "soft_max", worst); }
    { const float *Y = (const float *) mm->data; double worst = 0, wa = 0;
      for (int c = 0; c < 4; c++) for (int j = 0; j < 512; j++) { double s = 0, sa = 0; for (int i = 0; i < 2560; i++) { double p = (double) fbf(W[j * 2560 + i]) * R[c * 2560 + i]; s += p; sa += fabs(p); }
        double e = fabs(Y[c * 512 + j] - s) / sa / eps; if (e > worst) worst = e; }
      printf("%-9s max %9.1f eps of the sum of |products| (BF16 2560x512 by 2560x4)\n", "mul_mat", worst); }
    ggml_free(ctx);
    return 0;
}
