#!/usr/bin/env bash
# threads.sh PID: one line per thread of PID: its id, its CPU time in
# clock ticks (user plus system, 100 a second on Linux), its name. Two
# snapshots around a request, joined on the id, give the CPU seconds each
# thread spent in it:
#
#   scripts/threads.sh $pid | sort > before
#   scripts/request.sh out DOC 64
#   scripts/threads.sh $pid | sort > after
#   join before after | awk '{ printf "%.2f s  %s\n", ($4 - $2) / 100, $1 }' | sort -rn
set -euo pipefail
pid=${1:?usage: threads.sh PID}
for t in /proc/"$pid"/task/*; do
    stat=$(cat "$t/stat" 2>/dev/null) || continue
    # the fields after the parenthesised name: state is the first, utime
    # the twelfth, stime the thirteenth
    rest=${stat##*) }
    read -r -a f <<< "$rest"
    echo "${t##*/} $(( f[11] + f[12] )) $(cat "$t/comm" 2>/dev/null || echo ?)"
done
