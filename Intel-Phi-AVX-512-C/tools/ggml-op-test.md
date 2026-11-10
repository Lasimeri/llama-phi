# ggml-op-test.c: ggml's CPU ops against a double reference

Random inputs through ggml's own ops of the build it is linked against,
each element compared with a double-precision reference: the largest
error per op in units of float epsilon relative to max(|ref|, a floor).
Ops: exp, sigmoid, silu, tanh, softplus, sqrt, log, sin, cos, rms_norm,
soft_max, and a BF16 mul_mat (2560 x 512 by 2560 x 4, the error relative
to the sum of the absolute products).

The same source is linked twice: against `build-phi512-cpu` (the AVX2
twin, run natively) and `build-avx512f` (run under the preload on a
card). The gate passes when the two outputs are identical
([`../scripts/gates.md`](../scripts/gates.md)): the card's form of every
op is then exactly as precise as the host's.

```
gcc -O1 -I$LLAMA_CPP_DIR/ggml/include -o op-avx2 tools/ggml-op-test.c -L$B2 -lggml -lggml-base -lggml-cpu -lm -Wl,-rpath,$B2
gcc -O1 -I$LLAMA_CPP_DIR/ggml/include -o op-512 tools/ggml-op-test.c -L$B5 -lggml -lggml-base -lggml-cpu -lm -Wl,-rpath,$B5
```

The twin's output at llama.phi 9aa38a9b5 (2026-10-14), which the
assembly preload's card run matched line for line:

```
exp       max       0.5 eps
sigmoid   max       0.9 eps
silu      max       1.7 eps
tanh      max       0.5 eps
softplus  max    9888.9 eps
sqrt      max       0.5 eps
log       max       0.6 eps
sin, cos  max       0.5 eps
rms_norm  max       1.0 eps
soft_max  max       3.7 eps
mul_mat   max    1203.0 eps of the sum of |products|
```

(the elementwise lines also print where the worst element was).
softplus's 9888.9 is ggml's own formula far from zero, the same on both.
