// unpin.c: libunpin.so, loaded into llama-server with LD_PRELOAD beside the
// card backend when PHI_GGML_HOSTOPS is on.
//
// llama.cpp pins the RMS norm of every layer to the layer's device
// (llama-context.cpp, the graph callback: ggml_backend_sched_set_tensor_backend
// on the tensor named "norm"), so that a norm does not land on the previous
// layer's device. With every layer's context on a GPU and the weights on the
// host and a Xeon Phi, that pin sends a vector of a few kilobytes to the GPU and
// back twice a layer for every generated token. This library declines the pin
// for a norm of a generation step (no more than 31 columns): the scheduler then
// gives the norm to the backend of the node before it, which under
// PHI_GGML_HOSTOPS is the card backend's host side. Every other call goes
// through unchanged. UNPIN=0 turns it off; UNPIN_VERBOSE=1 counts at exit.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

#include "ggml.h"
#include "ggml-backend.h"

static void (*real_set)(ggml_backend_sched_t, struct ggml_tensor *, ggml_backend_t);
static int on = -1, verbose;
static unsigned long declined, passed;

static void report(void) {
    if (verbose) fprintf(stderr, "unpin: %lu norm pins declined, %lu calls passed on\n", declined, passed);
}

void ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor *node, ggml_backend_t backend) {
    if (on < 0) {
        const char *e = getenv("UNPIN");
        on = !(e && e[0] == '0');
        e = getenv("UNPIN_VERBOSE");
        verbose = e && e[0] == '1';
        real_set = (void (*)(ggml_backend_sched_t, struct ggml_tensor *, ggml_backend_t)) dlsym(RTLD_NEXT, "ggml_backend_sched_set_tensor_backend");
        if (!real_set) { fprintf(stderr, "unpin: no ggml_backend_sched_set_tensor_backend behind this library\n"); abort(); }
        atexit(report);
    }
    if (on && node && (node->op == GGML_OP_RMS_NORM || node->op == GGML_OP_NORM) &&
        node->ne[1] <= 31 && node->ne[2] == 1 && node->ne[3] == 1) {
        declined++;
        return;
    }
    passed++;
    real_set(sched, node, backend);
}
