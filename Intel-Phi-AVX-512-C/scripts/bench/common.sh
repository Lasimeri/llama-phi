# common.sh: sourced by the bench scripts (lb-run.sh, dbg-run.sh): the
# paths, the library copy, the card guard and the worker restart. See
# common.md.
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
llama=${LLAMA_CPP_DIR:-/mnt/raid5/phi/llama.phi}
model=${PHI_MODEL:-/mnt/raid0/Qwen3.8-Flash-Next-GGUF/BF16/Qwen3.8-Flash-Next-BF16-00001-of-00008.gguf}
bdir=${PHI_BENCH_DIR:-/mnt/raid5/phi/bench/c-$(date +%F)}
cards=${PHI512_CARDS:-0,1,2,3}
HP=${HP:-2060}       # huge pages per card
R=${R:-2048}         # resident pool pages per card (the card caps it at 2048, RES_MAX)
mkdir -p "$bdir" || exit 1
case "$bdir" in *" "*) echo "bench: PHI_BENCH_DIR has a space (LD_PRELOAD splits on spaces): $bdir" >&2; exit 2 ;; esac

# bench_lib NAME: the library under test copied to $bdir/lib-NAME.so (LIB,
# default the C build), so every run keeps the exact binary it ran.
bench_lib() {
    local src=${LIB:-$root/host/c/out/libphi512-c.so}
    [ -f "$src" ] || { echo "bench: no library at $src" >&2; exit 2; }
    [ "$src" = "$bdir/lib-$1.so" ] || cp "$src" "$bdir/lib-$1.so" || exit 2
    echo "$bdir/lib-$1.so"
}

# bench_workers: stop unless the cards are free (FORCE=1 overrides), then
# a fresh worker on each card in slot mode.
bench_workers() {
    local users
    users=$("$root/scripts/card-users.sh" ${cards//,/ })
    if [ -n "$users" ] && [ "${FORCE:-0}" != 1 ]; then
        echo "$users"; echo "bench: the cards are in use; stop those first or FORCE=1"; exit 2
    fi
    echo "== $(date '+%F %T') fresh workers on cards $cards ($HP huge pages, pool -R $R)"
    for c in ${cards//,/ }; do
        "$root/scripts/phi-vpu.sh" -c "$c" stop > /dev/null 2>&1
        # two threads on every core but the last two: -C 55 on a 3120A, 59 on a 7120A
        n=$("$root/scripts/phi-vpu.sh" -c "$c" cores)
        PHI_VPU_HUGEPAGES=$HP PHI_VPU_ARGS="-S 1 -C $((n - 2)) -T 2 -R $R -e 0 -s 1000" "$root/scripts/phi-vpu.sh" -c "$c" start "$n" 2>&1 | grep -iv "^worker started" | head -2
    done
}
