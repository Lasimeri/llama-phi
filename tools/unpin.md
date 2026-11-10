# unpin.c

`libunpin.so`, loaded into llama-server with `LD_PRELOAD` beside the card
backend. It interposes one ggml function,
`ggml_backend_sched_set_tensor_backend`, and declines one kind of call.

## Why

llama.cpp pins the RMS norm of every layer to the layer's device (the
graph callback in `llama-context.cpp`, for the tensor named `norm`, when
a batch has fewer than 32 tokens or every layer is offloaded). With every
layer's context on a GPU and the weights on the host and a Xeon Phi
(`-ngl 99 -ot ".*=CPU"`), that pin sends a vector of a few kilobytes to
the GPU and back twice a layer for every generated token, and it cuts
the chain of host-side operations in two each time.

## What it does

A call for a tensor whose operation is `RMS_NORM` or `NORM` and whose
shape is a generation step's (at most 31 columns, nothing in the third
and fourth dimensions) returns without pinning. ggml's scheduler then
gives the norm to the backend of the node before it in the graph, which
under `PHI_GGML_HOSTOPS=1` is the card backend's host side
(`host/asm/ggml-phi/glue.md` in the patched Intel-Phi-AVX-512-C). Every
other call, a prompt's norms among them, goes through to ggml unchanged.
No file of llama.cpp is edited, but its behaviour is: `libllama.so`
reaches the function through the dynamic linker, which hands it this
library's.

| Variable | Default | Meaning |
|---|---|---|
| `UNPIN` | 1 | 0: every call goes through |
| `UNPIN_VERBOSE` | 0 | 1: at exit, the pins declined and the calls passed on |

Build: `gcc -O2 -Wall -Wextra -shared -fPIC -I<llama.cpp>/ggml/include -o
libunpin.so unpin.c -ldl`.

## Measured

A 35 B Q6_K mixture-of-experts model, one generated token's graph as
llama-server logs it at `-lv 4` (`sched_graph_realloc: ... splits = N`):

| Backend's host operations | This library | Splits a token |
|---|---|---|
| off | no | 612 |
| element-wise only | no | 702 |
| with the SwiGLU | yes | 543 |
| and the multiplies the host keeps whole | yes | 393 |
