# ggml-router-test.c: the MoE router's ops through ggml

ggml's own CPU ops of the BF16 Flash-Next's expert router, from the
llama.phi build it is linked against: `soft_max` over 4 rows of 512
logits drawn like the model's (a normal spread, mean -3, deviation 2.5),
then `argsort` descending, then the top-10 view. Each round checks
`soft_max` element by element against plain C in double (relative
1e-4 plus 1e-6), and the argsort as a permutation that orders ggml's own
probabilities. Linked against `build-avx512f` and run under the preload,
the ops run on a card; that is the gate ([`../scripts/gates.md`](../scripts/gates.md)).

```
gcc -O1 -I$LLAMA_CPP_DIR/ggml/include -o router tools/ggml-router-test.c \
    -L$B -lggml -lggml-base -lggml-cpu -lm -Wl,-rpath,$B     # B = .../build-avx512f/bin
PHI512_CARD_ATOMICS=1 PHI512_CARDS=0 LD_PRELOAD=lib.so ./router 8
```

Output: a line per round (`round R: soft_max ok (0), argsort ok (0)`),
up to three lines per wrong element, then `PASS: soft_max and argsort
match plain C` or `FAIL`. Written 2026-10-14 when the router's top-10
differed from the AVX2 twin's in 8 of 48 layers of a llama-debug run:
both ops passed 8 of 8 on the card, so the drift is elsewhere.
