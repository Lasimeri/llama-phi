#!/usr/bin/env bash
# q4_0-bench.sh OUT_DIR [CARD]: builds the two Q4_0 one-row benchmarks
# (q4_0-bench.S: the kernels and their parts on first-level-cache data;
# q4_0-stream.S: the row kernel streaming a weight buffer at several
# prefetch distances) and runs them on a card through `phi`. Needs GNU
# as and ld, tcc, and a card with no other program on it.
#
#   OUT_DIR/q4_0-bench.log    cycles per call
#   OUT_DIR/q4_0-stream.log   tenths of a cycle per superblock
#   OUT_DIR/q4_0-many.log     the row kernel as 1 to 4 streams a core on 40 cores
set -euo pipefail
out=${1:?usage: q4_0-bench.sh OUT_DIR [CARD]}
card=${2:-0}
here=$(cd "$(dirname "$0")" && pwd)
vpu=$here/../card/vpu
mkdir -p "$out"
tcc -run "$here/q4_0-kernels.c" > "$out/q4_0.S"

# the row kernel alone, then one copy per prefetch pair (0: none)
awk '/^# phi_q4_0_row1p:/ { p = 1 } p { print } /size phi_q4_0_row1p,/ { p = 0 }' "$out/q4_0.S" > "$out/row.S"
variant() {     # name, first-level distance, second-level distance
    sed -e "s/phi_q4_0_row1p/$1/g" "$out/row.S" | awk -v a="$2" -v b="$3" '
        /lea 576\(%rdi\), %r10/ { printf "    lea %d(%%rdi), %%r10\n", (a == 0 ? 576 : a); next }
        /VPREFETCH0 r10, / { if (a == 0) next }
        /VPREFETCH1 rdi, / { if (b == 0) next; split($0, f, ","); printf "\tVPREFETCH1 rdi, %d\n", f[2] - 2304 + b; next }
        { print }'
}
{ variant kv_none 0 0; variant kv_576_2304 576 2304; variant kv_288_1152 288 1152
  variant kv_1152_4608 1152 4608; variant kv_1152_8640 1152 8640; variant kv_2304_11520 2304 11520
  variant kv_0_2304 0 2304; variant kv_576_0 576 0; } > "$out/q4_0-variants.S"

build() {       # source, binary, extra assembler arguments
    as --64 "${@:3}" -I "$vpu" -I "$out" -o "$out/$2.o" "$here/$1"
    ld -static -nostdlib -e _start -z noexecstack -o "$out/$2" "$out/$2.o"
}
build q4_0-bench.S q4_0-bench
build q4_0-stream.S q4_0-stream
build q4_0-stream.S q4_0-stream8 --defsym BIG=8388608

for b in q4_0-bench q4_0-stream q4_0-stream8; do
    phi -c "$card" put "$out/$b" "/tmp/$b" < /dev/null > /dev/null
done
phi -c "$card" run sh -c 'chmod +x /tmp/q4_0-bench; /tmp/q4_0-bench' < /dev/null > "$out/q4_0-bench.log"
phi -c "$card" run sh -c 'chmod +x /tmp/q4_0-stream; /tmp/q4_0-stream' < /dev/null > "$out/q4_0-stream.log"
# 1 to 4 streams on each of 40 cores (hardware threads in the pool's order 0, 3, 1, 2)
phi -c "$card" run sh -c 'chmod +x /tmp/q4_0-stream8
for per in 1 2 3 4; do i=0; while [ $i -lt 40 ]; do n=0; for h in 1 4 2 3; do n=$((n + 1)); if [ $n -le $per ]; then taskset -c $((4 * i + h)) /tmp/q4_0-stream8 > /tmp/o.$i.$h & fi; done; i=$((i + 1)); done; wait
echo "$per a core:"; cat /tmp/o.* | grep pf_576_2304_step1; rm -f /tmp/o.*; done' < /dev/null > "$out/q4_0-many.log"
phi -c "$card" run sh -c 'rm -f /tmp/q4_0-bench /tmp/q4_0-stream /tmp/q4_0-stream8' < /dev/null
tail -n 3 "$out/q4_0-bench.log"
