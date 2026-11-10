# card-users.sh: who holds a card

A program that drives a card maps its host-memory window
(`/dev/shm/phi-hostmem` for card 0, `/dev/shm/phi-hostmem-N` for card N):
production's `llama-server` through `libggml_phi.so`, a run under the
preload, a `phi-vpu` driver. Restarting the card's worker under it takes
the card away mid-request. `card-users.sh 0 1 2 3` lists those processes
(card, pid, command line) by reading every `/proc/PID/maps`, and exits 1
when it found any, 0 when the cards are free.

The stack's own `phictl` (the card's daemon, one per card) and `phitop`
(its monitor, which reads the worker's stats line) map every window for
as long as the card is up and send the worker no requests; they are
skipped by their `/proc/PID/comm`. Found 2026-10-14: without the
exception every card always had users and the gates never ran.

[`gates.sh`](gates.md) and the scripts under [`bench/`](bench/README.md)
call it before they restart a worker and stop unless `--force` (gates) or
`FORCE=1` (bench) is given.
