#!/usr/bin/env bash
# worker.sh WORK_DIR [CARD]: (re)deploy the patched card worker and start it
# in the layout the measurements used: the matrix pool's 120 threads on
# hardware threads 0 and 3 of cores 0-59, one slot of 118 threads on
# hardware threads 1 and 2 of cores 0-58 (-O; idle unless the preload is
# used), the dispatcher on CPU 0, 5600 huge pages (11.2 GB of the card).
# Needs a booted card and its stack (phictl / phi) from Intel-Phi-7120A.
set -euo pipefail
work=${1:?usage: worker.sh WORK_DIR [CARD]}
card=${2:-0}
w=$work/Intel-Phi-AVX-512-C
bash "$w/scripts/phi-vpu.sh" -c "$card" stop || true
bash "$w/scripts/phi-vpu.sh" -c "$card" deploy
PHI_VPU_HUGEPAGES=${PHI_VPU_HUGEPAGES:-5600} \
PHI_VPU_ARGS="${PHI_VPU_ARGS:--e 0 -O -S 1 -C 59 -T 2 -R 128 -s 1000}" \
    bash "$w/scripts/phi-vpu.sh" -c "$card" start "${WORKER_THREADS:-121}"
bash "$w/scripts/phi-vpu.sh" -c "$card" status | head -2
