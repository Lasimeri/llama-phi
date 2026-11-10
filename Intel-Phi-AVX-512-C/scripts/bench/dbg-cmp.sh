#!/usr/bin/env bash
# dbg-cmp.sh A B [TOL]: two llama-debug logs (dbg-run.sh) compared tensor
# by tensor in graph order: the first tensors whose sums differ by more
# than TOL relative (default 0.01) or are not finite, with both values,
# then the count compared. See dbg-cmp.md.
A=${1:?usage: dbg-cmp.sh A B [TOL]}; B=${2:?usage: dbg-cmp.sh A B [TOL]}; T=${3:-0.01}
ex() { awk '/^common_debug_cb_eval:/ { n++; name=$2; op=$0; sub(/.*= \([a-z0-9]+\) */, "", op); sub(/\(.*/, "", op) } /^    sum = / { print n, name, op, $3 }' "$1"; }
paste -d' ' <(ex "$A") <(ex "$B") | awk -v T="$T" '
{ if (NF < 8) exit; i=$1; na=$2; op=$3; a=$4; nb=$6; b=$8; c++
  bad=0
  if (na != nb) { print "graph differs at", i, na, nb; exit }
  if (a == b) bad=0; else if (a ~ /nan|inf/ || b ~ /nan|inf/) bad=1
  else { d=a-b; if (d<0) d=-d; m=(a<0?-a:a); if (m<1) m=1; if (d/m > T) bad=1 }
  if (bad) { printf "tensor %d %s (%s): reference %s, card %s\n", i, na, op, a, b; if (++k >= 8) exit } }
END { print c, "tensors compared" }'
