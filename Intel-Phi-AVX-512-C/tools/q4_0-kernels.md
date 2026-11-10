# q4_0-kernels.c

Prints the seven Q4_0 kernels of `card/vpu/kernels.S`: `phi_q4_0_1`,
`phi_q4_0_4`, `phi_q4_0_8` and their float16-activation twins
`phi_q4_0_1h`, `phi_q4_0_4h`, `phi_q4_0_8h`, and `phi_q4_0_row1p` (below,
"The prepared row"), in that file's line format
(the `mvex.inc` macro that assembles, a tab, the instruction in Intel
syntax as a comment). Its output was appended to `kernels.S` on
2026-11-09; the program stays as the record of how those 1371 lines were
made and as the way to make them again.

```
tcc -run tools/q4_0-kernels.c > q4_0.S
```

## What a kernel does

One call takes one superblock of a Q4_0 weight row, eight 18-byte blocks
(144 bytes, 256 weights), against 1, 4 or 8 activation rows, in the
calling convention every quantized kernel shares (`card/vpu/kernels.md`,
"Calling convention"). The format (ggml-common.h `block_q4_0`,
ggml-quants.c `dequantize_row_q4_0`): a float16 scale `d` at byte 0 of a
block, then sixteen bytes; the low nibbles are weights 0..15 of the
block, the high nibbles weights 16..31, each `(q - 8) * d`.

| step | instructions | where |
| --- | --- | --- |
| prefetch the superblock this thread takes next | four `vprefetch0` two calls ahead, four `vprefetch1` four ahead | 144 bytes at any alignment touch at most four lines |
| the eight scales | a masked `vloadunpacklps`/`hps` `{float16}` pair per block at `18 i`, lane `i` of zmm16 | the single unmasked lane receives the element at the address (`kernels.md`, the unpack facts) |
| store them | `vmovaps [rcx+0], zmm16` | the scratch's first 32 bytes; both nibble vectors of block `i` use `[rcx + 4 i]`, so no expanding permute is needed |
| the bytes | a `vloadunpacklps`/`hps` `{uint8}` pair at `18 i + 2`, four blocks a batch into zmm8..11 | float lanes, 0..255 |
| the high nibble | `vmulps` by 2^-4 (`[r9+24]`), `vrndfxpntps` round down | zmm12..15 |
| the low nibble | `vfnmadd231ps`: `b - 16 hi` (`[r9+0]`) | zmm8..11 |
| `q - 8` | `vsubps` by 8.0 (`[r9+44]`, the constant added to `g_consts` with these kernels) | all eight vectors |
| `(q - 8) * d` | `vmulps` by the block's scale, broadcast from the scratch | all eight vectors |
| the products | `vfmadd231ps acc, w, [row + off]`, one per activation row; `{float16}` and a 32-byte stride in the twins | vector `2 i` is the low nibbles of block `i`, `2 i + 1` the high ones |

Every value before the scale multiply is an exact small integer in a
float lane, and the subtraction comes before the multiply, so a weight
is rounded once, exactly where `dequantize_row_q4_0` rounds it: the
card's weights are the reference's bit for bit. (The K formats of
`kernels.md` fold their constant into a minuend, `w * sc - mn`, and
round twice.)

The registers are the family's: zmm0..7 accumulate (the one-row kernel
rotates over four and adds them at the end), zmm8..15 are the batch,
zmm16 the scales; the activation rows' bases are `rsi`, `r10`, `r11`,
`rax`, `rbx`, `r13`, `r14`, `r15`, the last four pushed and popped by the
eight-row kernels. A stage is written for the whole batch of four blocks
before the next stage, so dependent instructions sit a batch apart.

## The prepared row

`phi_q4_0_row1p(row rdi, prepared rsi, ns rdx, jump rcx, acc r8, consts r9)`
takes `ns` superblocks of a weight row (the whole row, or a segment of
it) against one activation row that
`card/vpu/rows.S` (`q4_0_prepare`) has rewritten, and is what a
one-row request runs, for rows of up to 32768 weights
(`card/vpu/rows.md`, "Q4_0 against one row"). The identity, the
prepared layout and the measurements are in `card/vpu/kernels.md`,
"Q4_0, one row, prepared". What the generator emits (`kernel_prepared`):

| step | instructions |
| --- | --- |
| the accumulators | `vmovaps zmm0, [r8]`, zmm1..3 cleared, once a row |
| the stream | three `vprefetch0` through `r10`, which starts at `rdi + 576`, advances with the row and takes the jump (`rcx`: the distance from the end of these superblocks to the ones the caller gives next, 0 when they follow on) when four superblocks are left; three `vprefetch1` at `rdi + 2304`; each superblock |
| per batch of four blocks | four `vloadunpacklps` then four `vloadunpackhps` `{uint8}` (zmm8..11); four `vbroadcastss {float16}` of the scales (zmm16..19); `vmulps` by 2^-4 into zmm12..15; `vmulps zmm8.., [rsi + 128 i]`; `vrndfxpntps` round down; `vfmadd231ps zmm8.., zmm12.., [rsi + 128 i + 64]`; `vsubps zmm8.., [rsi + 1024 + 16 i]{4to16}`; `vfmadd231ps` into zmm0..3 by the scale |
| the loop | `rdi` + 144, `r10` + 144, `rsi` + 1152, `rdx` - 1 |
| the end | the four accumulators added (zmm0 began as `[r8]`, so a row given in segments accumulates), stored to `[r8]` |

The two halves of an unpack pair are a batch apart (the high half
merges into the register the low half wrote), and so is every other
dependent pair. It keeps every general register but `rax`, `rdx`,
`rsi`, `rdi` and `r10`.

## Checks

- Assembled alone with `mvex.inc`: 11136 bytes of text, six symbols.
- `card/vpu/build.sh`: the worker with them is 272424 bytes, audit `0
  illegal, 0 suspect`.
- On card 0 (2026-11-09), `phi-vpu matmul-check -c 0 --only q4_0
  --threads 114 --repeat 3`, with `--act 0` and `--act 1`: every plain,
  mixture, several-matrix and fused feed-forward case ok. The card's
  compute time at 4096 x 5120: float32 activations 139.3 GFLOP/s at one
  row, 246.4 at eight, 251.3 at sixty-four; float16 126.4, 452.9, 490.3.
- The row kernel (2026-11-09): the worker with it is 272904 bytes
  (`card/vpu/build.sh`, audit `0 illegal, 0 suspect, 8349 KNC vector`);
  `matmul-check --only q4_0` passes with `--act 0` and `--act 1` at 57,
  114, 121, 181 and 241 threads; its cycle counts and the service's
  times are in `card/vpu/kernels.md` and `tools/q4_0-bench.md`.
