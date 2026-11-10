# backend.S

Everything that talks to the cards: the port of
`host/crates/phi-ggml/src/lib.rs` (its design and measurements are
`lib.md` of that crate in the history up to the commit that deleted
it, and `docs/results/` from 2026-09-23 on). One state under one lock
(`lock.S`, what Rust's `CTX` mutex was); every collection a fixed
table (`defs.md`); every line printed is `lib.rs`'s, word for word.

## Opening (`phi_ggml_open`)

Called once by the glue when ggml asks for the device count. Reads the
variables (every one with its Rust default):

| variable | default | what |
| --- | --- | --- |
| `PHI_GGML_CARDS` | every card whose window exists | a comma list of card indices, sorted and deduplicated |
| `PHI_GGML_THREADS` | 114 | the card threads a request asks for |
| `PHI_GGML_CARD_BYTES` | 4400000000 | each card's budget for resident rows |
| `PHI_GGML_FRACTION` | unset: sized at the first multiply | a fixed share of every matrix's rows, capped at `share_cap` |
| `PHI_GGML_PP_SHARE`, `PHI_GGML_PP_ADAPT` | 0.75, 1 | the batch share and whether it follows what the two sides measure (off under the offload) |
| `PHI_GGML_MIN_BYTES` | 4000000 | the weights a multiply must take off the host before a card is worth its round trip |
| `PHI_GGML_ACT` | 1 | activations as float16 |
| `PHI_GGML_OFFLOAD` | 0 | the cards' rows are theirs alone and their pages are dropped |
| `PHI_GGML_ALL_ROWS` | 0 | with the offload, or with `PHI_GGML_JUDGE=0` (since 2026-11-10: nothing is judged, so the host needs no rows of its own), the cards may keep every row |
| `PHI_GGML_EXPERTS` | unset | a placement file: whole experts on the cards. Under the offload the cards' experts are theirs alone (their pages dropped); without it (since 2026-10-06) the host keeps its copy of every expert, so a batch handed to a GPU by llama.cpp's op offload reads it while a generation step skips the cards' experts through the substituted ids, and the judge is forced off (a tensor judged back to the host would be computed in full there) |
| `PHI_GGML_JUDGE` | 1 | tensors judged by their timings |
| `PHI_GGML_TG_ONLY` | 0 | `K`: the inverse: a multiply of more than K tokens (a mixture's tokens) stays with the host, so the prompts stay with the GPUs and the host (llama.phi's prefill engine) and the cards take the generation steps alone (2026-10-06, the user's split for the rack). K is the sequences that generate at once (llama-server's slots): a step of two slots is a two-token multiply, which `K=1` kept off the cards (seen the same night in the two-process server) |
| `PHI_GGML_PP_ONLY` | 0 | a multiply of that many tokens or fewer (a mixture's tokens, not its columns) stays with the host: the cards take the prompts, token generation stays with the GPUs and the host. Ignored under the offload (the host has no copy to generate with). Measured 2026-10-06 on the four-card rack, Flash-Next Q6_K_XL, twice interleaved: prompts 87.9 and 88.1 against 85.5 and 85.8 tok/s, generation 8.02 and 7.80 against 7.68 and 7.59 (`PHI_GGML_PP_ONLY=1`) |
| `PHI_GGML_SPIN_US` | unset: spin throughout | how long a wait spins before it naps |
| `PHI_GGML_FFN_H16` | 0 | the fused block's intermediate as float16 on the card |
| `PHI_GGML_VERBOSE`, `PHI_GGML_IDS` | unset | the ledger; the routed ids of every pair (a calibration instrument) |

Each card: the window file's size checked against `WINDOW_LEN`, the
window mapped, the worker seen polling (`wait_ready`, 2 s), the record
filled (`open_card`); a card that fails is said and skipped. Then every
card is cleared (`K_FREE`, 30 s), because a card's uploads outlive the
process that made them. The messages and their order are the Rust's,
down to `no window for card N at PATH: No such file or directory (os
error 2)` (`w_os_error` gives the common errnos their text).

## The shares (`settle_fraction`, at the first multiply)

The weights the glue offered (`phi_ggml_note_weight`, until settled)
are sized to fill the cards: `size_shares` gives the dense matrices
the largest share within the cap at which every card's uploads fit
the budget (a bisection of 40 steps over `card_rows`, which rounds
exactly as `plan` uploads), the experts the largest share in what is
left, then one 64-row step more per expert tensor in address order
while the budget holds. Under a placement the mixtures are not shared
by rows: `size_whole` finds how many whole experts of every mixture
tensor fit beside the dense rows, never more than an equal split
between the cards. One line says the result (`X GB of dense weights
and Y GB of experts offered to the cards: each keeps ...`).

`card_cost` counts an upload as the worker allocates it: the bytes
rounded to 4096 plus 64 of slack, in whole 2 MiB pages.

## Planning a tensor (`prepare`, `plan`, `plan_whole`)

At a tensor's first multiply its split is decided: declined outright
when every card's share together cannot reach `PHI_GGML_MIN_BYTES`
(`declined`); otherwise each card with budget takes its share of the
rows from the top down, the host's remainder a multiple of 64
(`card_rows`), each slice copied into the window and uploaded
(`K_UPLOAD`, 120 s), the id kept in the split (`SP_ID[ci]`). A mixture
under a placement (`whole_k > 0`, a layer's tensor) is placed whole
instead: the layer's experts in the placement's order, card c taking
the ranks c, c + ncards, ..., each expert's matrix whole at its index
in the card's slice. Offloaded, the pages of the cards' rows are given
to `MADV_PAGEOUT` when the tensor is a mapping of a file
(`drop_pages`, `file_backed` over `/proc/self/maps`); a model read
into memory is said once.

A tensor of another shape at an address already planned (a second
model in the same process) is planned again, with the message.

## A multiply (`prepare`, `issue`, `end`)

`prepare` decides without starting: nothing for no columns; the host
for a shape past the window (`past the window's limits`), a fused
block's tensor, a whole-expert tensor asked for plainly, float weights
at a batch, a tensor the judgement took off the cards, or a card part
under `PHI_GGML_MIN_BYTES` (`too small to pay for a card`); else the
cards' work (each card the first `pp_share` of its slice at a batch,
all of it at one token or offloaded) and the host's ranges.

`issue` starts it: the activations into the first card's window as
float16 (`put_rows`, `to_f16`), or as float32 to every card when a
value is past 65504 (`activations up to X do not fit a half`), or to
the host alone when float32 does not fit the window; the other cards
get a copy (`copy_window`); a mixture's ids first (with whole experts,
each card's own experts by their local index and -1 elsewhere, and the
host's ids substituted so ggml's kernel reads no page the cards own,
`phi_ggml_host_ids`); the descriptor at `OFF_MATMUL` (and a second
matrix at `OFF_MORE` for a pair, `K_MATMUL_MORE`); the doorbell. The
host's ranges are what `phi_ggml_host_range` and
`phi_ggml_host_range_of` answer.

`end` waits for each card's reply (60 s; a timeout or a card error is
said with the rows and columns), gathers its runs into the destination
(`gather`: a mixture's column j + t * n_used to j * nb_d + t * nb_d2;
a column at -1 left alone), and judges: the host did 1 - share of the
rows in `t_host` and then waited, so alone it would have taken
`t_host / (1 - share)`; a call that was longer with the cards counts
against the tensor (two counts, or one plainly worse call, and the
host keeps it at that batch class), unless offloaded or the judge is
off. A plain batch multiply feeds `pp_feed`: the relative gap between
the host's time and the slowest card's, averaged over eight, moves the
batch share by half of it, at most 24 times, never under 0.2. The
verbose line (`multiply N: host part T ms, waited W ms more; card C
rows R: ...; the cards' rows read by the host so far X MB`) joins the
cards' parts, which were kept in a second buffer (`lines_buf`) while
other lines may have been said.

`phi_ggml_free_all` clears every card and forgets every split.

## Numbers and text

Floats go through `fp.S` in the order the Rust computed them
(`as f64` conversions, `as_secs_f64`, `round`, `clamp`), so the same
inputs print the same digits; the time stamps are `now_ns`. The line
buffer is 64 KiB for the `PHI_GGML_IDS` lines. The routines with more
arguments than registers (`put_rows`, `plan`, `prepare`, `gather`, the
ends) take them from named static blocks (`pr_*`, `pa_*`, `q_*`,
`ga_*`, `en_*`), which the one lock makes as private as a frame.

## Q4_0 (2026-11-09)

`row_bytes` gives a Q4_0 row 144 bytes per 256 weights (eight 18-byte
blocks, the number Q4_K has), `type_name` says `q4_0`, and `shape_ok`
takes it through the 256-block rules (k a multiple of 256, the
activation stride a multiple of 64, no alignment rule for the row: the
card reads it with unpack pairs). The fused feed-forward's gate
(`ffn.S`, `quantized`) still names Q4_K to IQ4_XS only, so a Q4_0
feed-forward is three plain multiplies; under `PHI_GGML_OFFLOAD` the
fused path is off for every type. Measured on a 27 B Q4_0 model (card 0,
`PHI_GGML_CARD_BYTES=13100000000 PHI_GGML_OFFLOAD=1
PHI_GGML_ALL_ROWS=1`, 120 card threads, 8 host threads): 14.7 GB of
dense weights offered, 88.5% of every matrix's rows on the card (13.09
GB), 2.605 tokens a second of generation against 0.2076 when the card
took only the one Q6_K matrix (0.52 GB).

## Limits, said when reached

8192 planned tensors (`splits`), 8192 offers, 512 whole-expert
placements, 512 experts in a mixture placed whole (a larger one is
shared by rows), 256 layers in a placement file, 64 entries of
`PHI_GGML_CARDS`, 65536 columns of ids, 8192 file-backed ranges. None
is near what a model here has.

## The gate (`phi_ggml_take`, 2026-10-07)

`supports_op` asks it before offering a multiply: 0 when what `prepare`
applies would keep that multiply with the host whatever happens, so
ggml's scheduler leaves it in the CPU backend's own sub-graph. The rules
are `prepare`'s, in its order: the batch class (`pp_only`, `tg_only`,
over a mixture's tokens), float weights at a batch (8 tokens and over,
unless offloaded), and, when the weight has a split (it has been
multiplied once), the split's record: whole experts placed (taken), the
host keeping every row (`SP_NCARDS` 0: the share's card part under
`min_bytes`), the judge having taken it off at this class
(`SP_AVOID`). A weight never multiplied is taken (the first graph is as
before); the scheduler asks again at the next graph llama.cpp builds.

Why: Qwen3.8 Flash Next generating on the rack (`PHI_GGML_TG_ONLY=1`,
`-t 12`, GPUs hidden) made 905 multiplies a token, 134 of them on the
cards; every one of the 905 was a sub-graph of one node handed to this
backend, the host's rows computed through `host_rows` on the private CPU
backend. The backend loaded with no tensor allowed on a card
(`PHI_GGML_MIN_BYTES` huge) ran 6.48 and 6.16 tokens a second against
the CPU alone's 7.76 and 7.92, interleaved: the hand-offs, not the
cards, cost about 30 ms a token (`docs/results/2026-10-07-rack-generation.md`).

## The card rows handed to a DMA router (`back_or_drop`, 2026-11-06)

Under `PHI_GGML_OFFLOAD`, after a dense matrix's card rows are uploaded (the
range from row `SP_R0` to the last, card byte 0 of the slice, id `SP_ID`),
`back_or_drop` looks up `tridma_back` once with `dlsym(RTLD_DEFAULT, ...)`
(`imports.list` has `dlsym`; ggml loads this library `RTLD_LOCAL`, so the router
is an `LD_PRELOAD` library and this side passes it the window base and the thread
count). When the process has it, the range goes to the router, which backs those
pages with the card (`VPU_K_DOWNLOAD` on a fault) instead of the disk; else the
pages go to `MADV_PAGEOUT` as before
(`drop_pages`). `phi_ggml_open` also clears the window's doorbell lock
(`OFF_LOCK`, see `../common/window.md`) when it maps a card.

A mixture's whole experts go the same way since 2026-11-10 (`back_whole`,
called for every expert a card holds where `plan_whole` dropped its pages):
the range is the expert's matrix in the mapped file, the id the holding
card's slice (`SP_ID`), the card byte the expert's place in that slice
(`WH_LOCAL` times the expert's bytes). The router speaks to the first card
only: an expert another card holds, and every expert when the process has no
router, is dropped as before. Until then the experts were always dropped, so a
prompt computed on a GPU read every card-held expert back from the disk.
Measured on a 35 B Q6_K model, a 7120A holding every dense row and 109 whole
experts of each layer's 256 (12.92 GB), a 1,410-token prompt on an RTX 3090 Ti:
83.4 tok/s with the experts read from the disk, 358.0 with them from the card
(13,191 ranges card-backed; with `TRIDMA_VERIFY=4`, 225.4 tok/s and 12,008 chunks equal to
the file, 0 differing); the first generated token's log-probabilities are the
same to the last digit either way; the disk read during a 220-token generation fell
from 1,836 to 437 MiB.

## Every row on the cards with the weights still in host RAM (2026-11-10)

`PHI_GGML_ALL_ROWS=1` asked for the offload because the judgement of a
tensor needs rows the host computes. With `PHI_GGML_JUDGE=0` nothing is
judged, and the cards may keep every row while the host keeps its copy
(`PHI_GGML_OFFLOAD=0`): a prompt computed on a GPU then reads host
memory only. `PHI_GGML_AHEAD` is allowed in the same case (`glue.md`).

Measured on a 9 B Q8_0 model whose file was wholly in host memory, card
0 (a 7120A), an RTX 3090 Ti holding the context and taking the prompts,
14 host threads, the host's small operations on (`PHI_GGML_HOSTOPS=1`),
220 generated tokens:

| the card holds | generation |
| --- | --- |
| nothing (the host alone, without this backend) | 4.09, 4.02 tok/s |
| 50.0 % of every dense matrix's rows (4.35 GB), judged | 6.55, 6.45, 6.37 tok/s |
| every row (8.14 GB), judge off, run-ahead on | 5.73 tok/s (one run) |
| nothing resident: 15.6 % of every matrix streamed (`PHI_GGML_STREAM`) | 4.14, 4.20, 4.19 tok/s |

With half the rows the timeline (`PHI_GGML_TIME`) has the host's rows at
55.4 % of the time, the host's small operations at 30.8 % and the wait
for the card at 1.1 %: the host is the slower side. With every row the
card is, and the result is lower than the shared one; the 24 tokens at
temperature 0 are the same in both. No share between the two was
measured. Streamed, no multiply went to the card after the first
tokens (the time spent issuing stays at 0.49 s over the next 57,344
sub-graphs), and the result is the host's.

## Streamed weights (`PHI_GGML_STREAM`, 2026-11-06)

`PHI_GGML_STREAM=1` keeps every weight in host RAM: `upload_area` copies
the slice it was handed (staged at `OFF_A`, the layout an upload would
send) to this card's next place in the window above the transfer areas
(`WINDOW_LEN`, 768 MiB, up to the mapped file's length `CD_WIN_LEN`; the
place advances in `CD_STREAM_AT`) and sends no `K_UPLOAD`. The id it
returns is that window offset with bit 63 set; the callers keep it in
`SP_ID` as before. `issue` turns such an id into `a_id` 0 and `a_off`,
the card's own path for a matrix in the window
(`card/vpu/vpu_matmul.h`: "with a cached (a_id) or in the window
(a_off)"; `matmul.S` grows `g_a` and pulls `m * nb_a` bytes through
`vpu_pull`, the block device whose DMA is the stack's host channels,
3.58 GB/s measured on the desktop link, then computes; since 2026-11-10
the worker pulls it with its own DMA engine straight into the buffer's
frames when the buffer is on huge pages, `card/vpu/matmul.md`,
"Weights straight from host RAM"). A slice that
does not fit what is left of the window stays with the host
(`stream: the window is full; ...`). Pairs (`phi_ggml_begin_id_pair`
answers -2) and the fused feed-forward (`ffn_enabled` in `glue.S`) are
off while streaming, because `K_MATMUL_MORE` and `K_FFN` name resident
slices. `phi_ggml_free_all` rewinds `CD_STREAM_AT` with the splits. A
`K_FREE` of a streamed id finds nothing on the card and is answered OK.
The card pulls a whole matrix and then computes (no overlap yet).
