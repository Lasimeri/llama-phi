# glue.S

The ggml side of the card backend: the glue ggml's C interface
requires and nothing more, what `csrc/ggml-phi.c` was, slot for slot.
It registers one device of type ACCEL named `Phi` that shares the
CPU's host buffers, accepts the `MUL_MAT`, `MUL_MAT_ID` and (opt in)
SwiGLU nodes whose weight type and shape the cards take, and runs each
in three steps: `phi_ggml_begin` (backend.S: the cards start on their
rows), the host's own rows on a private ggml CPU backend built here
from ggml's own kernels, `phi_ggml_end` (the cards' rows gathered).
Every ggml function it calls is an undefined symbol of the library,
bound when the program loads it (`imports.list`, `build.md`); every
ggml layout number comes from `ggml_layout.inc`.

## Registration

`ggml_backend_init` (what `ggml_backend_load` looks up) returns
`ggml_backend_phi_reg`: the `ggml_backend_reg` with the API version
and the registration interface table, built once by `tables_init`
(each interface pointer at the slot `ggml_layout.inc` names, the rest
null). The device count is decided once, by `phi_ggml_open`
(`phi_reg_get_device_count`): 1 when a card answers, else 0, and
llama.cpp runs on the CPU without the backend. The device's properties
(`phi_dev_get_props`): the name, the description, memory 0/0, type
ACCEL, capabilities `buffer_from_host_ptr` only. `phi_dev_init_backend`
hands out a backend from a pool of eight (`BE_SLOT`, `BE_USED`).

## What is accepted (`phi_dev_supports_op`)

- `MUL_MAT` with a weight source that is a leaf named `*weight*`, of a
  type the cards take (`card_type`: F32, F16, BF16, Q4_K, Q5_K, Q6_K,
  Q8_0, IQ4_XS, and Q4_0 since 2026-11-09), whose shape `phi_ggml_supports` accepts, not flagged
  `GGML_HINT_SRC0_IS_HADAMARD` (`T_OP_PARAMS+4`); the weight is noted
  for the shares (`phi_ggml_note_weight`, with its layer from the name
  `blk.N.`).
- `MUL_MAT_ID` likewise, over the experts.
- Either multiply only when `phi_ggml_take` (`backend.md`) says the cards
  may take it: not when a rule `prepare` applies would keep it with the
  host anyway (the batch class: `PHI_GGML_PP_ONLY`, `PHI_GGML_TG_ONLY`,
  float weights at a batch; and, once the weight has been multiplied, its
  split: the host keeping every row, the judge having taken it off at
  that class). Declined, the multiply stays inside the CPU's sub-graph and
  is not offered for the shares. ggml's scheduler asks again whenever
  llama.cpp builds a graph (each new shape: in a server, each request),
  not at a graph it reuses.
- `GLU` (SwiGLU, split) when `PHI_GGML_FFN` is on and the node is part
  of a fusable block (`ffn_enabled`; off under `PHI_GGML_OFFLOAD` with
  the message the C gave; off while `PHI_GGML_STREAM` is on, since
  `K_FFN` names resident slices).
- views, reshapes, permutes and transposes are ggml's own and pass.

The scheduler asks about every node of a graph before it computes any,
so at the first multiply the offered weights are the model.

## Computing a graph (`phi_graph_compute`)

The nodes are scanned once into `role[]`: each `MUL_MAT`, `MUL_MAT_ID`,
and each block of four (`find_quad`: gate, up, GLU, down over the same
input, when the fused path is on) gets its role, the rest are skipped.
Then in order:

- `run_mul_mat`: `phi_ggml_begin(a, type, m, k, nb_a, keep, b, n,
  nb_b)`; for each host range `phi_ggml_host_range(i)` the host's rows
  through `host_rows` (a ggml context of two tensors aliasing the
  weight's rows and the activations, one `ggml_mul_mat`, computed on
  the private CPU backend, the result rows at their place in the
  destination); `phi_ggml_end(d, nb_d)`.
- `run_mul_mat_id`: `phi_ggml_begin_id` with the mixture's columns;
  the host's rows through `host_rows_id`, over the backend's
  substituted ids (`phi_ggml_host_ids`) when there are any;
  `phi_ggml_end_id`.
- `run_id_pair`: a layer's gate and up over the same ids as one
  request (`phi_ggml_begin_id_pair`; -2 means the two go one by one);
  `host_rows_id_pair`; `phi_ggml_end_id_pair`. `id_pair` pairs two
  quantized weights of any types, or two of the same float type (the
  card runs a float request's pieces one by one with its float
  kernels); until 2026-10-09 the test was "a card type of 2 or more",
  which let a bfloat16 pair through to a request the card rejected.
- `run_ffn`: `phi_ggml_ffn_begin(&args)`; -2 declines and the four
  nodes run as they are; else the host's runs of the intermediate
  through `host_ffn` (gate and up rows, the SwiGLU and down's columns,
  in a scratch buffer), the result zeroed when the host has no run,
  then `phi_ggml_ffn_end`.
- `PHI_GGML_GRAPH=N` dumps the first N sub-graphs (`dump_graph`), one
  line per node with its op, name, sources and data pointers.

Every one of these prints the lines the C printed under
`PHI_GGML_VERBOSE` (`host rows A..B of TYPE MxK, n N: compute T ms`,
the feed-forward's host part), with `w_elapsed_ms` giving C's `%.3f`
of the elapsed milliseconds.

## A multiply issued ahead of its node (`PHI_GGML_AHEAD`, 2026-11-09)

ggml hands this backend one sub-graph at a time, and a model's graph
puts other backends' nodes between two multiplies that read the same
activations (a layer's `attn_qkv` and `attn_gate` both read `attn_norm`;
`attn_q`, `attn_v` and `attn_k` likewise): the second is begun only
after the first one's result has been copied to the GPU, worked on
there and the scheduler has come back, though its input was there all
along. With `PHI_GGML_AHEAD=1` the cards do not wait for that.

| step | where | what |
| --- | --- | --- |
| learn | `run_mul_mat`, on entry | under the weight of the multiply before this one (`ah_prev`), `ah_succ` keeps this one's weight, activations' address, rows, columns, row stride, card type and `keep`: 2048 entries of 64 bytes by the weight's address (`ah_slot`: eight places tried) |
| issue ahead | `run_mul_mat`, after `phi_ggml_end` | when the entry under this weight names activations at the address this multiply read: the activations copied to `ah_copy`, `phi_ggml_begin` for that multiply at once, its arguments kept (`ah_a`, `ah_x`, `ah_n`, `ah_nb_b`, `ah_m`, `ah_nr`), `ah_valid` set; the return to ggml leaves its request with the cards |
| take | `run_mul_mat`, on entry, `ah_valid` set | the node's weight, its activations' address, their count and stride, its rows, and the activations' bytes against `ah_copy` (`repe cmpsb`): all equal, the node skips `phi_ggml_begin` and goes to its host ranges and `phi_ggml_end` |
| drop | `ahead_drain` | any difference: `phi_ggml_end` into `ah_drain`, and the node runs as it always did; also at the top of `run_ffn`, `run_id_pair` and `run_mul_mat_id`, which ring the cards themselves |

What makes it safe:

- A wrong guess costs a multiply, never a result: the node's request is
  the one issued ahead only when every argument and every byte of its
  input is what that request was begun with.
- `phi_ggml_begin` takes the backend's lock and drops it before it
  returns (`backend.S`, `begin_common`); between it and `phi_ggml_end`
  the multiply lives in the backend's statics, as it does inside
  `run_mul_mat`, and nothing else enters the backend while a graph is
  computed.
- The window's doorbell lock stays taken from the request to its reply
  (`../common/window.md`). Its only other user is a DMA router's fault
  handler, which is asked for a card's rows when a batch is computed on
  a GPU from the host's mapping; this backend declines those batches
  whole, so no fault of that kind falls inside a pass that has a
  multiply outstanding.
- A multiply is begun ahead only when its activations are at the
  address the finished one read, so none is left outstanding at the end
  of a pass: the next pass's first multiply reads another tensor.
- It needs `PHI_GGML_OFFLOAD` or `PHI_GGML_JUDGE=0` (`ahead_enabled`
  says so and stays off without either): where tensors are judged,
  `phi_ggml_end` times a multiply from its `phi_ggml_begin` and judges
  the tensor by it, and a request left running across ggml's other work
  would be counted against its tensor. With the judge off nothing is
  timed for a verdict, and the run-ahead works with the weights kept in
  host RAM as well (since 2026-11-10; `backend.md`, "Every row on the
  cards with the weights still in host RAM").

The copy is 1 MiB and the drop buffer 8 MiB (`AH_COPY_MAX`,
`AH_DRAIN_MAX`); a multiply whose activations or results would not fit
is not begun ahead. `ah_report` says the two counts at 256 multiplies
taken and then every 16384 (`ahead: N multiplies were with the cards
before their node arrived, M begun ahead were not what came`).

Measured on card 0 with a 27 B Q4_0 model, one server, the GPU doing
every node but the multiplies, the cards holding every row of 60
layers, 64 tokens a run, four runs each, the plain runs first:

| | plain | `PHI_GGML_AHEAD=1` |
| --- | --- | --- |
| 64 tokens | 25.979, 25.703, 25.423, 24.791 s | 24.802, 24.055, 24.061, 24.019 s |
| tokens a second | 2.425 to 2.541 | 2.540 to 2.623 |
| the card's compute time | 20.09 to 20.18 s | 20.04 to 20.09 s |
| the output at temperature 0, 24 tokens with the top five log-probabilities of each | the reference (two runs equal) | the same bytes (two runs) |

The mean falls from 25.474 to 24.234 s (5.1 %) with the card's compute
time unchanged: the time between its multiplies is what shrank, about
19 ms a token. 16384 multiplies had been taken and none dropped when
the count was last said. The two sets were not interleaved, and their
ranges touch (24.791 and 24.802 s).

**Never across a pass (2026-11-10).** A request begun ahead holds the
window's doorbell lock until its node takes it or the next multiply
drains it. The multiply that follows a pass's last one (the output
head) is the first of the next pass, and it was begun ahead whenever
the two happened to read activations at the same address: an address
ggml's allocator may give to two unrelated tensors. Nothing then came
for the request until the next token; with a DMA router in the process
(`back_or_drop`), a prompt arriving first needed the lock to fetch its
weights and waited for ever (seen 2026-11-10 on a 35 B model once the
host's small operations had changed the buffer layout: the lock byte 1,
the request and reply sequence numbers equal, one thread spinning).
`ah_first` is the weight of the first multiply the backend ever ran,
the one a pass begins with; a successor with that weight is never
begun.

## The timeline (`PHI_GGML_TIME`, 2026-11-10)

`PHI_GGML_TIME=1` has the backend account for a generation's time as it
sees it. `TM slot` (a macro: the slot pushed, `tm_mark` called, every
register kept) adds the time since the last mark to a slot; the marks
sit right after a call or before one, where no flag is live:

| slot | what it is |
| --- | --- |
| outside | between two sub-graphs: ggml's scheduler, the other backends' sub-graphs (a GPU's), the copies between backends; `tm_enter`, at `phi_graph_compute`'s entry |
| issue | `phi_ggml_begin`, `_begin_id`, `_begin_id_pair`: the activations to the window, the descriptor, the doorbell (and, at a tensor's first multiply, its upload) |
| host rows | the host's ranges on the private CPU backend |
| end | `phi_ggml_end` and its kin: the wait for the cards after the host's rows are done, their rows gathered |
| rest | the scan of the graph, a drain, the bookkeeping; closed by `tm_leave` at the exit |
| issued ahead | the activations kept and the request begun for a multiply issued ahead of its node |
| host ops | a run of small operations and of multiplies the host keeps whole (`PHI_GGML_HOSTOPS`, below) |
| long gaps | an outside gap of more than 20 ms (between requests, a prompt on a GPU), kept apart so it does not drown the rest |

`tm_leave` prints one line every 8192 sub-graphs, the sums since the
start in microseconds (`time: N sub-graphs; us outside X, issue X, host
rows X, end X, rest X, issued ahead X, host ops X; outside gaps over 20 ms X`); the
difference of two lines is an interval. Off, a mark is one compare and
a jump.

Measured on a 35 B Q6_K mixture-of-experts model, card 0 (a 7120A)
holding every dense row and 109 whole experts of each layer's 256
(`PHI_GGML_EXPERTS`, 12.92 GB), a GPU holding the context, the
run-ahead on; 24,576 sub-graphs of generation, 0.641 ms each:

| slot | share |
| --- | --- |
| end | 52.6 % |
| outside | 28.0 % |
| host rows | 18.8 % |
| issue | 0.33 % |
| rest, issued ahead | 0.18 %, 0.04 % |

With the marks in and the switch on, the temperature-0 output of that
server is the plain one's byte for byte (24 tokens with their top-5
log-probabilities). What the table says: the copies of the activations
are not where the time goes; in that layout the host waits for the card
half the time, and a quarter is spent outside the backend.

The run-ahead on that model and layout, 220 tokens, three prompts: 7.529,
7.842, 7.498 tok/s in graph order, 8.230, 8.280, 8.729 with
`PHI_GGML_AHEAD=1`, the output identical; the card busy 60.3 % of the
time before and 71.0 % after (sampled from the host); 16,384 multiplies
taken ahead, 81 begun ahead that were not what came.

## Small operations on the host (`PHI_GGML_HOSTOPS`, 2026-11-10)

ggml's scheduler gives a node that has no weight to the backend of the
node before it in the graph, when that backend accepts it
(`ggml-backend.cpp`, the passes that expand an assignment down and up;
every backend but the last, the CPU, expands alike). This backend
accepted multiplies only, so with every layer's context on a GPU
(`-ngl 99 -ot ".*=CPU"`) each norm, residual add, routing soft max and
SwiGLU of a generated token went to the GPU and came back: the graph of
one token of a 35 B mixture-of-experts model was cut into 612 splits
(llama-server at `-lv 4`: `sched_graph_realloc: ... nodes = 5015,
splits = 612`).

`PHI_GGML_HOSTOPS=1` has `phi_dev_supports_op` also accept
(`supports_hostop`):

| what | condition |
| --- | --- |
| `ADD SUB MUL DIV SQR SQRT SCALE NORM RMS_NORM L2_NORM SOFT_MAX ARGSORT TOP_K SUM_ROWS CONCAT REPEAT GET_ROWS CONT CPY UNARY CLAMP ROPE` (`host_op`) | the result is a generation step's: no dimension past the first over 31 (`HOSTOP_DIM`: a prompt's tensors carry their tokens there and stay with the GPU), at most 1,048,576 elements |
| a `GLU` that `supports_glu` declines (not a fused block's) | the same |
| a `MUL_MAT` that `supports_mul_mat` declines (the host keeps every row of it) | the same, and its matrix is a weight (it has a buffer when the scheduler asks) |

The last line matters as much as the first: declined, such a multiply
went to ggml's own CPU backend and cut the chain there.

`phi_graph_compute` runs these as a run (`host_run`): from the node to
the next multiply the cards take, mixture multiply or node with a role,
one view of the graph (the graph's structure copied, `size` 0, the node
count, the nodes from that place, no leafs and no gradients: what
`ggml_graph_view` makes, which ggml does not export) computed on the
private CPU backend. `tools/ggml-layout-check.c` gives the structure's
size and fields (`CG_BYTES` 96) and the operations' numbers.

Two norms a layer stay with the GPU whatever this backend accepts:
llama.cpp pins them to the layer's device. A preload library declines
that pin for a generation step's norm (`tools/unpin.c` in
Lasimeri/llama-phi).

Measured on a 35 B Q6_K model, card 0 (a 7120A) holding half of every
dense matrix's rows and 119 whole experts of each layer's 256, an RTX
3090 Ti holding the context, one generated token's graph:

| | splits |
| --- | --- |
| the switch off | 612 |
| the element-wise operations only | 702 |
| with the SwiGLU, and the norms unpinned | 543 |
| and the multiplies the host keeps whole | 393 |

A linear-attention layer is then four sub-graphs of this backend with a
GPU sub-graph between each pair (the convolution, the attention's core
and its state): the mixed projection; the convolution's results, alpha,
the gate, beta; `z`; and one of 49 nodes from the attention's output
through the residual, the norm, the routing, the experts' gate and up,
the SwiGLU, down, their weighted sum, the shared expert and its gate to
the next layer's norm.

Generation, 220 tokens, three prompts: 10.60, 10.37, 10.51 tok/s with
the switch off, 10.93, 10.77, 11.04 with it and the norms unpinned. The
timeline of that run, uploads left out: the host's rows 29.5 %, the wait
for the card 28.8 %, the host's small operations and whole multiplies
24.4 %, outside the backend 16.6 % (44.9 % before, which counted the
whole multiplies then run by ggml's CPU backend). Two variants measured
the same and were not kept: the segments that hold no multiply on a
second CPU backend of one thread (10.74, 10.87, 10.80), and the private
pool polling instead of sleeping (10.87, 10.61, 10.75).

What was checked of the results, with float32 activations
(`PHI_GGML_ACT=0`) so that the card's and the host's rows differ only by
their kernels, temperature 0, 24 tokens with their top five
log-probabilities, the switch on against off: a dense 27 B Q4_0 model,
24 of 24 tokens the same, the chosen token's log-probability at most
0.0156 apart; the 35 B model, 24 of 24 the same, at most 0.0638 apart.
Two builds of one card kernel that differ in the order of a row's
additions were 0.0106 apart on that 27 B model (2026-11-09). The
results are not bit-equal: a norm or an add on the host rounds
differently from the GPU's. Why the mixture model shows more is not
established (its routing picks eight experts of 256 a layer, and a
score's last bits can change the eighth).

## The host's rows

`host_backend` creates the private CPU backend once: `ggml_backend_dev_by_type(CPU)`,
`ggml_backend_dev_init`, `ggml_backend_set_n_threads` through the
registration's `get_proc_address` with `PHI_GGML_HOST_THREADS`
(default 12), and a threadpool of its own unless
`PHI_GGML_HOST_POOL=0`, polling `PHI_GGML_HOST_POLL` (default 0: the
pool's threads sleep between graphs; the busy wait is only asked for).
The messages (`the host's rows run on ggml's CPU backend with N
threads`, the threadpool's, the `-t` advice) are the C's.

A ggml call is made with the stack 16-byte aligned; `ggml_init` takes
its 24-byte parameter struct in memory (built on the stack); the
threadpool parameters come back through a hidden pointer.
