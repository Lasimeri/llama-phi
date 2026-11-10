#!/usr/bin/env bash
# card-users.sh CARD...: the processes that map a card's host-memory
# window (/dev/shm/phi-hostmem for card 0, phi-hostmem-N for card N), one
# line each: card, pid, command. Exit 1 when there is at least one. Used by
# gates.sh and scripts/bench before they restart a worker. See card-users.md.
found=1
for c in "$@"; do
    w=/dev/shm/phi-hostmem; [ "$c" != 0 ] && w=$w-$c
    for p in /proc/[0-9]*; do
        pid=${p#/proc/}
        [ "$pid" = $$ ] && continue
        # The stack's card daemon and its monitor map every window for good and
        # send the worker no requests.
        case $(cat "$p/comm" 2>/dev/null) in phictl|phitop) continue ;; esac
        if grep -qs " $w\$" "$p/maps"; then
            echo "card $c: pid $pid $(tr '\0' ' ' < "$p/cmdline" 2>/dev/null | cut -c1-120)"
            found=0
        fi
    done
done
[ $found = 1 ]
