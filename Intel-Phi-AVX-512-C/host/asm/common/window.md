# window.S

A card's window as the host sees it, and the doorbell protocol over its
control words: what `phi_vpu::window` and the backend's `ring`,
`doorbell` and `wait` did (`card/vpu/proto.inc` has the offsets; the
protocol is `card/vpu/vpu_proto.md`).

The window is the file the stack's daemon pinned for the card:
`/dev/shm/phi-hostmem` for card 0, `/dev/shm/phi-hostmem-N` after
(`hostmem_path(index)`, a static buffer); the card maps the same bytes
uncached.

| routine | arguments | result |
| --- | --- | --- |
| `hostmem_path` | rdi index | rax the path |
| `hostmem_exists` | rdi index | eax 1 when the file is there (`stat` succeeds) |
| `hostmem_len` | rdi index | rax the file's size, or 0 with the negated errno in rdx |
| `window_open` | rdi index, rsi len | rax the mapping (read-write, shared), or 0 with the negated errno in rdx and ecx 0 when `open` failed, 1 when `mmap` did |
| `wait_ready` | rdi base, rsi timeout ns | the readiness word cleared, then waited for (a live worker re-asserts it within a millisecond; a dead one's stale word would otherwise satisfy a plain read); eax 1 when the worker is polling, 0 on timeout; looks every millisecond |
| `doorbell` | rdi base, esi kernel, edx threads | takes the doorbell lock at `OFF_LOCK` (2026-11-06), then a request for the kernel whose descriptor is already in the control area: the request's fields, a fence, then the sequence number (the one word the card polls), a fence; rax the number |
| `wait_reply` | rdi base, rsi seq, rdx timeout ns, rcx out | waits for the reply to the request; the 48-byte reply copied to rcx; eax the reply's status (0 ok, negative a card error), or 1 on timeout; frees the doorbell lock on every return |

`wait_reply` spins with `pause` for `spin_ns` nanoseconds of the wait
(the global, set by the backend from `PHI_GGML_SPIN_US`; unset it is
the largest value, a spin throughout, as it always had), then naps
`NAP_NS` (50 us, plus the kernel's timer slack) between looks. The card
writes its reply fields and then the echoed sequence number last, so
seeing the number means the rest of the reply is there.

## The doorbell lock (2026-11-06)

The card has one request slot, and a second host-side client in the same window
(the DMA router's fault handler, `tools/tridma_router.c` in llama-phi, issuing
`VPU_K_DOWNLOAD` from its own thread) must not ring it between another client's
request and its reply. `doorbell` therefore takes a test-and-set lock at
`OFF_LOCK` (15360, a control-area line the card never touches; `lock.S`
`lock_take`) before it reads the sequence number, and `wait_reply` frees it at its
single exit, which every return (the reply, a card error, the timeout) goes
through. Every request this library makes is a `doorbell` followed by a
`wait_reply` on the same window (`ring` / `wait_card` in `backend.S`, and the
feed-forward request in `ffn.S`), so a lock taken is always freed; between the
two the backend computes the host's rows, which never touch the router's
card-backed pages. `phi_ggml_open` clears the word when it maps the window, since
a client killed between a request and its reply leaves it taken.
