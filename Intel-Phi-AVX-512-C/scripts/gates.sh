#!/usr/bin/env bash
# gates.sh: the preload's regression gates on one card, each with a fresh
# worker in slot mode; a table of expected and measured counts, exit 1 on
# any miss. The card's previous worker configuration is restored at the
# end. See gates.md.
#
#   scripts/gates.sh [--card N] [--lib PATH] [--only "region fforms ..."] [--force]
#
# Gates: region (tools/avx512-region-test.c), fforms (avx512-f-forms-test.c),
# narrow (avx512-narrow-test.c), libm (avx512-libm-test.c), router
# (ggml-router-test.c), op (ggml-op-test.c against the AVX2 twin's output).
set -uo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
card=0
lib=$root/host/c/out/libphi512-c.so
only="region fforms narrow libm router op"
force=0
while [ $# -gt 0 ]; do
    case $1 in
        --card) card=$2; shift 2 ;;
        --lib) lib=$2; shift 2 ;;
        --only) only=$2; shift 2 ;;
        --force) force=1; shift ;;
        -h|--help) sed -n 2,13p "$0"; exit 0 ;;
        *) echo "gates.sh: unknown argument $1" >&2; exit 2 ;;
    esac
done
llama=${LLAMA_CPP_DIR:-/mnt/raid5/phi/llama.phi}
b512=$llama/build-avx512f/bin
b2=$llama/build-phi512-cpu/bin
out=${PHI_GATES_DIR:-$HOME/.cache/phi512-c-gates}
case "$out" in *" "*) echo "gates.sh: PHI_GATES_DIR has a space (LD_PRELOAD splits on spaces): $out" >&2; exit 2 ;; esac
[ -f "$lib" ] || { echo "gates.sh: no library at $lib (make -C host/c)" >&2; exit 2; }
vpu=$root/scripts/phi-vpu.sh

# A program that maps the card's window (production's llama-server, a run of
# the preload) would lose the card when its worker restarts (card-users.md).
users=$("$root/scripts/card-users.sh" "$card")
if [ -n "$users" ] && [ $force = 0 ]; then
    echo "$users" >&2
    echo "gates.sh: stop them first, or --force to restart the worker under them" >&2
    exit 2
fi

stamp=$(date +%Y%m%d-%H%M%S)
logs=$out/logs-$stamp
mkdir -p "$out/bin" "$logs" || exit 2
cp "$lib" "$out/bin/lib.so" || exit 2
L=$out/bin/lib.so
f_only="-mavx512f -mno-avx512vl -mno-avx512bw -mno-avx512dq -mno-avx512cd"
build() {   # NAME COMMAND...: build into $out/bin, the log in $logs
    local n=$1; shift
    "$@" > "$logs/build-$n.txt" 2>&1 || { echo "gates.sh: building $n failed:"; tail -5 "$logs/build-$n.txt"; return 1; }
}
build region gcc -O2 -mprefer-vector-width=512 $f_only -o "$out/bin/region" "$root/tools/avx512-region-test.c" || exit 1
build fforms gcc -O1 $f_only -o "$out/bin/fforms" "$root/tools/avx512-f-forms-test.c" || exit 1
build narrow gcc -O1 -mavx512f -mavx512vl -mavx512dq -mavx512bw -mf16c -o "$out/bin/narrow" "$root/tools/avx512-narrow-test.c" -lm || exit 1
build libm gcc -O2 -mavx512f -o "$out/bin/libm" "$root/tools/avx512-libm-test.c" -lm || exit 1
ggml() {    # NAME SOURCE BIN: a ggml test linked against one llama.phi build
    build "$1" gcc -O1 -I"$llama/ggml/include" -o "$out/bin/$1" "$root/tools/$2" -L"$3" -lggml -lggml-base -lggml-cpu -lm -Wl,-rpath,"$3"
}
case " $only " in *" router "*) ggml router ggml-router-test.c "$b512" || exit 1 ;; esac
case " $only " in *" op "*) ggml op-512 ggml-op-test.c "$b512" && ggml op-avx2 ggml-op-test.c "$b2" || exit 1
    "$out/bin/op-avx2" > "$logs/op-avx2.txt" 2>&1 || { echo "gates.sh: the AVX2 twin's op-test failed"; exit 1; } ;; esac

prev_hp=$("$vpu" -c "$card" config 2>/dev/null | awk '/^hugepages/ {print $2}')
prev_w=$("$vpu" -c "$card" config 2>/dev/null | awk '/^worker/ {$1 = ""; sub(/^ +-v */, ""); print}')
# One slot with two threads on every core but the last two (-C 55 on a
# 3120A, 59 on a 7120A), the wide pool and the dispatcher on those.
cores=$("$vpu" -c "$card" cores)
fresh() {
    "$vpu" -c "$card" stop > /dev/null 2>&1
    PHI_VPU_ARGS="-S 1 -C $((cores - 2)) -T 2 -R 128 -s 1000" "$vpu" -c "$card" start "$cores" > "$logs/worker-$1.txt" 2>&1
}
# NAME EXPECTED_OK EXTRA_ENV BIN [ARGS]: a fresh worker, the program under
# the preload, its ok lines counted and its verdict line read.
declare -a rows
fails=0
gate() {
    local n=$1 want=$2 extra=$3 bin=$4; shift 4
    case " $only " in *" $n "*) ;; *) return ;; esac
    fresh "$n"
    local t0 t1 rc got verdict
    t0=$(date +%s.%N)
    env $extra PHI512_CARD_ATOMICS=1 PHI512_CARDS=$card LD_PRELOAD=$L timeout 900 "$out/bin/$bin" "$@" > "$logs/$n.txt" 2>&1
    rc=$?
    t1=$(date +%s.%N)
    case $n in
        op) if diff -q "$logs/op-avx2.txt" <(grep -v '^phi512' "$logs/$n.txt") > /dev/null; then got=same; else got=differs; fi
            [ $got = same ] && [ $rc = 0 ] && verdict=ok || verdict=FAIL ;;
        libm) got=$(grep -o -E '[0-9]+ of 4096 differ' "$logs/$n.txt" | head -1)
            [ "$got" = "0 of 4096 differ" ] && [ $rc = 0 ] && verdict=ok || verdict=FAIL ;;
        router) got=$(grep -c -E '^round [0-9]+: soft_max ok .*argsort ok' "$logs/$n.txt")/$want
            grep -q '^PASS' "$logs/$n.txt" && [ "$got" = "$want/$want" ] && [ $rc = 0 ] && verdict=ok || verdict=FAIL ;;
        *) got=$(grep -c '^ok ' "$logs/$n.txt")/$want
            grep -q '^PASS' "$logs/$n.txt" && [ "$got" = "$want/$want" ] && [ $rc = 0 ] && verdict=ok || verdict=FAIL ;;
    esac
    [ $verdict = ok ] || fails=$((fails + 1))
    rows+=("$(printf '%-7s %-18s %-8s %-6s %7.1f' "$n" "$got" "rc $rc" "$verdict" "$(echo "$t0 $t1" | awk '{print $2 - $1}')")")
}
gate region 8 "" region
gate fforms 42 "" fforms
gate narrow 94 "" narrow
gate libm 0 "PHI512_TRAMP_ANY=1" libm
gate router 8 "" router 8
gate op 0 "" op-512

"$vpu" -c "$card" stop > /dev/null 2>&1
if [ -n "$prev_w" ]; then
    set -- $prev_w
    t=${!#}
    a=${prev_w% *}
    [ "$a" = "$prev_w" ] && a=""
    PHI_VPU_HUGEPAGES=${prev_hp:-768} PHI_VPU_ARGS="$a" "$vpu" -c "$card" start "$t" > "$logs/worker-restore.txt" 2>&1
    restored="restored: hugepages ${prev_hp:-768}, worker $prev_w"
else
    restored="no previous worker; left stopped"
fi
echo "gate    got                rc       verdict seconds"
printf '%s\n' "${rows[@]}"
echo "gates: $(( ${#rows[@]} - fails )) of ${#rows[@]} passed; library $lib ($(wc -c < "$L") bytes), card $card, logs $logs; $restored"
[ $fails = 0 ]
