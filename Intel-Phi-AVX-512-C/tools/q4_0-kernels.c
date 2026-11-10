/* q4_0-kernels.c: prints the six Q4_0 kernels of card/vpu/kernels.S.
 *
 *   tcc -run tools/q4_0-kernels.c > q4_0.S
 *
 * phi_q4_0_{1,4,8} and their float16 twins phi_q4_0_{1,4,8}h, each one
 * superblock (eight 18-byte Q4_0 blocks, 256 weights) of a weight row
 * against 1, 4 or 8 activation rows, in the calling convention and the
 * line format of kernels.S (the MVEX macro that assembles, then the
 * instruction in Intel syntax as a comment). See q4_0-kernels.md. */
#include <stdio.h>

static const char *base[8] = { "rsi", "r10", "r11", "rax", "rbx", "r13", "r14", "r15" };

/* one weight vector `w` (a zmm number) times its sixteen activations of
 * every row, at vector index v of the superblock */
static void products(int rows, int half, int w, int v)
{
    int off = v * (half ? 32 : 64);
    for (int q = 0; q < rows; q++) {
        int acc = rows == 1 ? v & 3 : q;
        printf("\tVFMADD231PS zmm%d, zmm%d, %s, %d%s\t# vfmadd231ps zmm%d, zmm%d, [%s+%d]%s\n",
               acc, w, base[q], off, half ? ", conv=CV_F16" : "",
               acc, w, base[q], off, half ? "{float16}" : "");
    }
}

static void kernel(int rows, int half)
{
    char name[32];
    snprintf(name, sizeof name, "phi_q4_0_%d%s", rows, half ? "h" : "");
    printf("# %s: eight Q4_0 blocks, against %d activation row(s)%s\n", name, rows,
           half ? ", float16" : "");
    printf("    .globl %s\n    .type %s, @function\n%s:\n", name, name, name);

    /* the superblock this thread processes next: L1 two calls ahead, L2 four */
    printf("    mov 8(%%rsp), %%r11\n    lea (%%rdi,%%r11,2), %%rax\n");
    for (int d = 0; d < 256; d += 64)
        printf("\tVPREFETCH0 rax, %d\t# vprefetch0 [rax+%d]\n", d, d);
    printf("    lea (%%rax,%%r11,2), %%rax\n");
    for (int d = 0; d < 256; d += 64)
        printf("\tVPREFETCH1 rax, %d\t# vprefetch1 [rax+%d]\n", d, d);
    if (rows == 8)
        printf("    push %%rbx\n    push %%r13\n    push %%r14\n    push %%r15\n");

    /* the eight block scales (float16 at 18 i) into lanes 0..7, to the scratch */
    printf("\tVPXORD zmm16, zmm16, zmm16\t# vpxord zmm16, zmm16, zmm16\n");
    for (int i = 0; i < 8; i++) {
        printf("    mov $%d, %%eax\n\tKMOV_KR k1, eax\t# kmov k1, eax\n", 1 << i);
        printf("\tVLOADUNPACKLPS zmm16, rdi, %d, conv=CV_F16, k=k1\t# vloadunpacklps zmm16 {k1}, [rdi+%d]{float16}\n",
               18 * i, 18 * i);
        printf("\tVLOADUNPACKHPS zmm16, rdi, %d, conv=CV_F16, k=k1\t# vloadunpackhps zmm16 {k1}, [rdi+%d]{float16}\n",
               18 * i + 64, 18 * i + 64);
    }
    printf("\tVMOVAPS_ST rcx, 0, zmm16\t# vmovaps [rcx+0], zmm16\n");

    /* the other activation rows, the accumulators */
    for (int q = 1; q < rows; q++)
        printf("    mov %d(%%rdx), %%%s\n", 8 * q, base[q]);
    if (rows == 1) {
        printf("\tVMOVAPS zmm0, r8, 0\t# vmovaps zmm0, [r8+0]\n");
        for (int a = 1; a < 4; a++)
            printf("\tVPXORD zmm%d, zmm%d, zmm%d\t# vpxord zmm%d, zmm%d, zmm%d\n", a, a, a, a, a, a);
    } else {
        for (int q = 0; q < rows; q++)
            printf("\tVMOVAPS zmm%d, r8, %d\t# vmovaps zmm%d, [r8+%d]\n", q, 64 * q, q, 64 * q);
    }

    /* two batches of four blocks: zmm8..11 the bytes and then the low
     * nibbles, zmm12..15 the high nibbles; every stage for the whole batch */
    for (int g = 0; g < 2; g++) {
        for (int j = 0; j < 4; j++) {
            int o = 18 * (4 * g + j) + 2;
            printf("\tVLOADUNPACKLPS zmm%d, rdi, %d, conv=CV_U8\t# vloadunpacklps zmm%d, [rdi+%d]{uint8}\n",
                   8 + j, o, 8 + j, o);
            printf("\tVLOADUNPACKHPS zmm%d, rdi, %d, conv=CV_U8\t# vloadunpackhps zmm%d, [rdi+%d]{uint8}\n",
                   8 + j, o + 64, 8 + j, o + 64);
        }
        for (int j = 0; j < 4; j++)     /* b * 2^-4 */
            printf("\tVMULPS zmm%d, zmm%d, r9, 24, conv=CV_1TO16\t# vmulps zmm%d, zmm%d, [r9+24]{1to16}\n",
                   12 + j, 8 + j, 12 + j, 8 + j);
        for (int j = 0; j < 4; j++)     /* its floor: the high nibble */
            printf("\tVRNDFXPNTPS zmm%d, zmm%d, 1\t# vrndfxpntps zmm%d, zmm%d, 1\n",
                   12 + j, 12 + j, 12 + j, 12 + j);
        for (int j = 0; j < 4; j++)     /* b - 16 hi: the low nibble */
            printf("\tVFNMADD231PS zmm%d, zmm%d, r9, 0, conv=CV_1TO16\t# vfnmadd231ps zmm%d, zmm%d, [r9+0]{1to16}\n",
                   8 + j, 12 + j, 8 + j, 12 + j);
        for (int j = 0; j < 8; j++)     /* q - 8 */
            printf("\tVSUBPS zmm%d, zmm%d, r9, 44, conv=CV_1TO16\t# vsubps zmm%d, zmm%d, [r9+44]{1to16}\n",
                   8 + j, 8 + j, 8 + j, 8 + j);
        for (int j = 0; j < 8; j++) {   /* (q - 8) * d, the block's scale */
            int d = 4 * (4 * g + (j & 3));
            printf("\tVMULPS zmm%d, zmm%d, rcx, %d, conv=CV_1TO16\t# vmulps zmm%d, zmm%d, [rcx+%d]{1to16}\n",
                   8 + j, 8 + j, d, 8 + j, 8 + j, d);
        }
        for (int j = 0; j < 4; j++) {   /* block i: weights 32 i.. low, 32 i + 16.. high */
            int i = 4 * g + j;
            products(rows, half, 8 + j, 2 * i);
            products(rows, half, 12 + j, 2 * i + 1);
        }
    }

    if (rows == 1) {
        printf("\tVADDPS zmm0, zmm0, zmm1\t# vaddps zmm0, zmm0, zmm1\n");
        printf("\tVADDPS zmm2, zmm2, zmm3\t# vaddps zmm2, zmm2, zmm3\n");
        printf("\tVADDPS zmm0, zmm0, zmm2\t# vaddps zmm0, zmm0, zmm2\n");
    }
    for (int q = 0; q < rows; q++)
        printf("\tVMOVAPS_ST r8, %d, zmm%d\t# vmovaps [r8+%d], zmm%d\n", 64 * q, q, 64 * q, q);
    if (rows == 8)
        printf("    pop %%r15\n    pop %%r14\n    pop %%r13\n    pop %%rbx\n");
    printf("    ret\n    .size %s, .-%s\n", name, name);
}

/* phi_q4_0_row1p(row rdi, prepared rsi, ns rdx, acc r8, consts r9): a
 * whole Q4_0 weight row of ns superblocks against one prepared
 * activation row (card/vpu/rows.S, q4_0_prepare): per superblock 1152
 * bytes, for block i the vectors x0 = x[32 i..] at 128 i and
 * y = x[32 i + 16..] - 16 x0 at 128 i + 64, and four floats p at
 * 1024 + 16 i with 4 sum(p) = 8 sum(x[32 i..32 i + 31]). With b the
 * block's bytes and hi = floor(b / 16),
 *
 *   sum (lo - 8) x0 + (hi - 8) x1  =  sum b x0 + hi y - p{4to16}
 *
 * so neither the low nibble nor the q - 8 is computed, and the scale is
 * one vbroadcastss of its float16: nine vector operations a block where
 * phi_q4_0_1 has thirteen and a serial chain of sixteen masked loads.
 * The row is one sequential stream: the lines 576 bytes ahead to the
 * first-level cache, the lines 2304 bytes ahead to the second. The four
 * accumulators stay in registers across the row; their sum is added to
 * the sixteen lanes at acc (so a row may be given in segments).
 *
 * rcx is the jump: the distance from the end of these ns superblocks to
 * the start of the ones the caller gives next (0 when they follow on,
 * as a thread's whole rows do; a row's distance less 144 ns for a
 * segment of each row). The first-level prefetch runs four superblocks
 * ahead through r10 and takes the jump when four are left, so the last
 * four prefetches of a call fetch the next call's first four. Keeps
 * every general register but rax, rdx, rsi, rdi, r10. */
static void kernel_prepared(void)
{
    const char *name = "phi_q4_0_row1p";
    printf("# %s: a Q4_0 weight row, or a segment of one, against 1 prepared activation row\n", name);
    printf("    .globl %s\n    .type %s, @function\n%s:\n", name, name, name);
    printf("\tVMOVAPS zmm0, r8, 0\t# vmovaps zmm0, [r8+0]\n");
    for (int a = 1; a < 4; a++)
        printf("\tVPXORD zmm%d, zmm%d, zmm%d\t# vpxord zmm%d, zmm%d, zmm%d\n", a, a, a, a, a, a);
    printf("    lea 576(%%rdi), %%r10\n    cmp $4, %%rdx\n    ja 1f\n    add %%rcx, %%r10\n");
    printf("1:\n");
    for (int d = 0; d < 192; d += 64)
        printf("\tVPREFETCH0 r10, %d\t# vprefetch0 [r10+%d]\n", d, d);
    for (int d = 0; d < 192; d += 64)
        printf("\tVPREFETCH1 rdi, %d\t# vprefetch1 [rdi+%d]\n", 2304 + d, 2304 + d);

    /* two batches of four blocks, every stage for the whole batch: zmm8..11
     * the bytes and then the block's sum, zmm12..15 the high nibbles,
     * zmm16..19 the scales */
    for (int g = 0; g < 2; g++) {
        for (int j = 0; j < 4; j++) {
            int o = 18 * (4 * g + j) + 2;
            printf("\tVLOADUNPACKLPS zmm%d, rdi, %d, conv=CV_U8\t# vloadunpacklps zmm%d, [rdi+%d]{uint8}\n",
                   8 + j, o, 8 + j, o);
        }
        for (int j = 0; j < 4; j++) {
            int o = 18 * (4 * g + j) + 2 + 64;
            printf("\tVLOADUNPACKHPS zmm%d, rdi, %d, conv=CV_U8\t# vloadunpackhps zmm%d, [rdi+%d]{uint8}\n",
                   8 + j, o, 8 + j, o);
        }
        for (int j = 0; j < 4; j++) {   /* the block's scale, to every lane */
            int o = 18 * (4 * g + j);
            printf("\tVBROADCASTSS zmm%d, rdi, %d, conv=CV_F16\t# vbroadcastss zmm%d, [rdi+%d]{float16}\n",
                   16 + j, o, 16 + j, o);
        }
        for (int j = 0; j < 4; j++)     /* b * 2^-4 */
            printf("\tVMULPS zmm%d, zmm%d, r9, 24, conv=CV_1TO16\t# vmulps zmm%d, zmm%d, [r9+24]{1to16}\n",
                   12 + j, 8 + j, 12 + j, 8 + j);
        for (int j = 0; j < 4; j++) {   /* b x0 */
            int o = 128 * (4 * g + j);
            printf("\tVMULPS zmm%d, zmm%d, rsi, %d\t# vmulps zmm%d, zmm%d, [rsi+%d]\n",
                   8 + j, 8 + j, o, 8 + j, 8 + j, o);
        }
        for (int j = 0; j < 4; j++)     /* the floor: the high nibble */
            printf("\tVRNDFXPNTPS zmm%d, zmm%d, 1\t# vrndfxpntps zmm%d, zmm%d, 1\n",
                   12 + j, 12 + j, 12 + j, 12 + j);
        for (int j = 0; j < 4; j++) {   /* + hi y */
            int o = 128 * (4 * g + j) + 64;
            printf("\tVFMADD231PS zmm%d, zmm%d, rsi, %d\t# vfmadd231ps zmm%d, zmm%d, [rsi+%d]\n",
                   8 + j, 12 + j, o, 8 + j, 12 + j, o);
        }
        for (int j = 0; j < 4; j++) {   /* - p */
            int o = 1024 + 16 * (4 * g + j);
            printf("\tVSUBPS zmm%d, zmm%d, rsi, %d, conv=CV_4TO16\t# vsubps zmm%d, zmm%d, [rsi+%d]{4to16}\n",
                   8 + j, 8 + j, o, 8 + j, 8 + j, o);
        }
        for (int j = 0; j < 4; j++)     /* times the scale, into four accumulators */
            printf("\tVFMADD231PS zmm%d, zmm%d, zmm%d\t# vfmadd231ps zmm%d, zmm%d, zmm%d\n",
                   j, 8 + j, 16 + j, j, 8 + j, 16 + j);
    }
    printf("    add $144, %%rdi\n    add $144, %%r10\n    add $1152, %%rsi\n    dec %%rdx\n");
    printf("    cmp $4, %%rdx\n    jne 2f\n    add %%rcx, %%r10\n2:\n    test %%rdx, %%rdx\n    jnz 1b\n");
    printf("\tVADDPS zmm0, zmm0, zmm1\t# vaddps zmm0, zmm0, zmm1\n");
    printf("\tVADDPS zmm2, zmm2, zmm3\t# vaddps zmm2, zmm2, zmm3\n");
    printf("\tVADDPS zmm0, zmm0, zmm2\t# vaddps zmm0, zmm0, zmm2\n");
    printf("\tVMOVAPS_ST r8, 0, zmm0\t# vmovaps [r8+0], zmm0\n");
    printf("    ret\n    .size %s, .-%s\n", name, name);
}

int main(void)
{
    printf("# The Q4_0 kernels (2026-11-09): the output of tools/q4_0-kernels.c.\n"
           "# A block is 18 bytes, the float16 scale d and sixteen bytes whose low\n"
           "# nibbles are weights 0..15 and whose high nibbles are weights 16..31, each\n"
           "# (q - 8) * d (ggml's dequantize_row_q4_0); eight blocks a call.\n");
    for (int half = 0; half < 2; half++)
        for (int rows = 1; rows <= 8; rows = rows == 1 ? 4 : 8 + (rows == 8)) {
            if (half || rows > 1)
                printf("\n");           /* a blank line between kernels, none after the last */
            kernel(rows, half);
        }
    printf("\n");
    kernel_prepared();
    return 0;
}
