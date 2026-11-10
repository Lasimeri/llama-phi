/* gpu-phi-stage [MIB [CARD:GPU_BUS_HEX ...]]: the host-staged path from
 * GPUs to Phi cards (docs/research/phi-vram-dma.md, the fallback): system
 * memory is the buffer between them. For each pair, a GPU's copy engine
 * writes 2 MiB chunks of a device buffer into the card's host window (the
 * 16 RESIDENT staging slots), and the card's DMA engine pulls each chunk
 * into a pool page (K_RESIDENT). Two coordinated workers per pair share a
 * ring of slot states: the feeder fills free slots from the GPU (OpenCL),
 * the puller hands filled slots to the card and frees them on its reply,
 * so the two legs run at once. Pass 1 is timed per leg and end to end;
 * pass 2 pulls every chunk back (K_PUSH) and compares every 32-bit word
 * with the pattern the GPU wrote. Every transfer stays on its owner's
 * normal path (OpenCL for the GPU, the worker's slot protocol for the
 * card); no device's memory is mapped for another. Default: 256 MiB on
 * card 0 with the GPU on bus c7. See gpu-phi-stage.md. */
#define _GNU_SOURCE
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "vpu_fexec.h"
#include "vpu_proto.h"

/* cl_nv_device_attribute_query: the device's PCI bus (hwloc include/hwloc/opencl.h,
 * NVIDIA's cl_ext.h: 0x4008). */
#define CL_DEVICE_PCI_BUS_ID_NV 0x4008

#define CHUNK (2u << 20)
#define NSTAGE 16 /* VPU_RESIDENT_BYTES / CHUNK */
#define TAG_BASE 0x600000000000ull /* chunk tags inside the worker's reserved range */
#define POOL_USE 120               /* distinct tags, under the worker's -R 128 pool */
#define WIN_MAP (128u << 20)       /* the slots, the slot data and the RESIDENT area */
#define TIMEOUT_NS 5000000000ull
#define MAXPAIRS 4

enum { S_FREE, S_FILLED };

struct pair {
    int card;
    unsigned bus;
    uint8_t *win;
    volatile struct vpu_slot_req *rq;
    volatile struct vpu_slot_reply *rp;
    volatile uint16_t *bell;
    uint64_t seq;
    cl_context ctx;
    cl_command_queue q;
    cl_mem dbuf;
    int state[NSTAGE]; /* S_FREE or S_FILLED, read and written atomically */
    size_t nchunks;
    uint64_t gpu_ns, card_ns, t_end;
    size_t bad, checked;
    int err;
};

static const char *fill_src = "__kernel void fill(__global uint *p, uint base)\n"
                              "{ uint i = (uint)get_global_id(0); p[i] = ((base + i) * 2654435761u) ^ 0x5a5a5a5au; }\n";

static uint32_t pat(uint32_t i) { return (i * 2654435761u) ^ 0x5a5a5a5au; }

static uint64_t now_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

static uint8_t *stage_at(struct pair *p, unsigned s) { return p->win + VPU_OFF_RESIDENT + (size_t)s * CHUNK; }

/* The request cleared, its fields set by the caller, then posted: the
 * sequence number and kind, a fence, the doorbell (kind << 11 | seq). */
static void clear_req(struct pair *p) { memset((void *)&p->rq->kind, 0, sizeof(struct vpu_slot_req) - 8); }

static int post_wait(struct pair *p, uint32_t kind)
{
    p->rq->seq = ++p->seq;
    p->rq->kind = kind;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    *p->bell = (uint16_t)((kind << 11) | (p->seq & 0x7ff));
    uint64_t t0 = now_ns();
    while (p->rp->seq != p->seq) {
        __builtin_ia32_pause();
        if (now_ns() - t0 > TIMEOUT_NS)
            return 1;
    }
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    return p->rp->status;
}

static int card_resident(struct pair *p, uint64_t tag, unsigned stage)
{
    clear_req(p);
    p->rq->fn_id = tag;
    p->rq->entry = stage;
    p->rq->threads = 0;         /* the immutable class: a pool page filled by DMA */
    p->rq->flags = 0xffffffffu; /* every 64 KiB unit of the chunk */
    p->rq->fill_off = 0;
    p->rq->fill_len = CHUNK;
    return post_wait(p, VPU_K_RESIDENT);
}

static int card_push(struct pair *p, uint64_t tag, unsigned stage)
{
    clear_req(p);
    p->rq->fn_id = tag;
    p->rq->entry = stage;
    p->rq->fill_len = CHUNK;
    return post_wait(p, VPU_K_PUSH);
}

static uint64_t tag_of(size_t i) { return TAG_BASE + (uint64_t)(i % POOL_USE) * CHUNK; }

/* The feeder: chunk i into staging slot i % NSTAGE once the puller has
 * freed it; a blocking read, so the slot is filled when it returns. */
static void *feeder(void *arg)
{
    struct pair *p = arg;
    for (size_t i = 0; i < p->nchunks && !p->err; i++) {
        unsigned s = (unsigned)(i % NSTAGE);
        while (__atomic_load_n(&p->state[s], __ATOMIC_ACQUIRE) != S_FREE)
            if (__atomic_load_n(&p->err, __ATOMIC_RELAXED))
                return NULL;
        cl_event ev;
        if (clEnqueueReadBuffer(p->q, p->dbuf, CL_TRUE, i * CHUNK, CHUNK, stage_at(p, s), 0, NULL, &ev) !=
            CL_SUCCESS) {
            __atomic_store_n(&p->err, 1, __ATOMIC_RELAXED);
            return NULL;
        }
        cl_ulong a = 0, b = 0;
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof a, &a, NULL);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof b, &b, NULL);
        p->gpu_ns += b - a;
        clReleaseEvent(ev);
        __atomic_store_n(&p->state[s], S_FILLED, __ATOMIC_RELEASE);
    }
    return NULL;
}

/* The puller: filled slots to the card in order, each freed on the
 * card's reply. */
static void *puller(void *arg)
{
    struct pair *p = arg;
    for (size_t i = 0; i < p->nchunks && !p->err; i++) {
        unsigned s = (unsigned)(i % NSTAGE);
        while (__atomic_load_n(&p->state[s], __ATOMIC_ACQUIRE) != S_FILLED)
            if (__atomic_load_n(&p->err, __ATOMIC_RELAXED))
                return NULL;
        uint64_t c0 = now_ns();
        int st = card_resident(p, tag_of(i), s);
        p->card_ns += now_ns() - c0;
        if (st != 0) {
            fprintf(stderr, "card %d: K_RESIDENT of chunk %zu: status %d\n", p->card, i, st);
            __atomic_store_n(&p->err, 2, __ATOMIC_RELAXED);
            return NULL;
        }
        __atomic_store_n(&p->state[s], S_FREE, __ATOMIC_RELEASE);
    }
    p->t_end = now_ns();
    return NULL;
}

/* Pass 2 for one pair: each chunk to the card again, pushed back into a
 * poisoned staging slot, every word compared. */
static void *verifier(void *arg)
{
    struct pair *p = arg;
    for (size_t i = 0; i < p->nchunks; i++) {
        if (clEnqueueReadBuffer(p->q, p->dbuf, CL_TRUE, i * CHUNK, CHUNK, stage_at(p, 0), 0, NULL, NULL) !=
                CL_SUCCESS ||
            card_resident(p, tag_of(i), 0) != 0) {
            p->err = 3;
            return NULL;
        }
        memset(stage_at(p, 1), 0xa5, CHUNK);
        if (card_push(p, tag_of(i), 1) != 0) {
            p->err = 4;
            return NULL;
        }
        const uint32_t *w = (const uint32_t *)stage_at(p, 1);
        uint32_t first = (uint32_t)(i * (CHUNK / 4));
        for (uint32_t j = 0; j < CHUNK / 4; j++) {
            if (w[j] != pat(first + j)) {
                if (p->bad < 4)
                    printf("card %d chunk %zu word %u: %08x, expected %08x\n", p->card, i, j, w[j], pat(first + j));
                p->bad++;
            }
        }
        p->checked += CHUNK / 4;
    }
    return NULL;
}

static int setup(struct pair *p, size_t total)
{
    cl_platform_id plats[8];
    cl_uint np = 0;
    cl_device_id dev = NULL;
    cl_int e = clGetPlatformIDs(8, plats, &np);
    for (cl_uint i = 0; e == CL_SUCCESS && i < np && !dev; i++) {
        cl_device_id devs[16];
        cl_uint nd = 0;
        if (clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_GPU, 16, devs, &nd) != CL_SUCCESS)
            continue;
        for (cl_uint d = 0; d < nd; d++) {
            cl_uint b = 0;
            if (clGetDeviceInfo(devs[d], CL_DEVICE_PCI_BUS_ID_NV, sizeof b, &b, NULL) == CL_SUCCESS && b == p->bus)
                dev = devs[d];
        }
    }
    if (!dev) {
        fprintf(stderr, "gpu-phi-stage: no OpenCL GPU on bus %02x\n", p->bus);
        return 1;
    }
    p->ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &e);
    if (e != CL_SUCCESS)
        return 1;
    cl_queue_properties qp[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    p->q = clCreateCommandQueueWithProperties(p->ctx, dev, qp, &e);
    if (e != CL_SUCCESS)
        return 1;
    p->dbuf = clCreateBuffer(p->ctx, CL_MEM_READ_WRITE, total, NULL, &e);
    if (e != CL_SUCCESS)
        return 1;
    cl_program prog = clCreateProgramWithSource(p->ctx, 1, &fill_src, NULL, &e);
    if (e != CL_SUCCESS || clBuildProgram(prog, 1, &dev, "", NULL, NULL) != CL_SUCCESS)
        return 1;
    cl_kernel k = clCreateKernel(prog, "fill", &e);
    if (e != CL_SUCCESS)
        return 1;
    cl_uint base = 0;
    size_t words = total / 4;
    clSetKernelArg(k, 0, sizeof p->dbuf, &p->dbuf);
    clSetKernelArg(k, 1, sizeof base, &base);
    if (clEnqueueNDRangeKernel(p->q, k, 1, NULL, &words, NULL, 0, NULL, NULL) != CL_SUCCESS ||
        clFinish(p->q) != CL_SUCCESS)
        return 1;

    char path[64];
    if (p->card == 0)
        snprintf(path, sizeof path, "/dev/shm/phi-hostmem");
    else
        snprintf(path, sizeof path, "/dev/shm/phi-hostmem-%d", p->card);
    int wfd = open(path, O_RDWR);
    if (wfd < 0) {
        fprintf(stderr, "gpu-phi-stage: %s: %s\n", path, strerror(errno));
        return 1;
    }
    p->win = mmap(NULL, WIN_MAP, PROT_READ | PROT_WRITE, MAP_SHARED, wfd, 0);
    close(wfd);
    if (p->win == MAP_FAILED) {
        fprintf(stderr, "gpu-phi-stage: mmap %s: %s\n", path, strerror(errno));
        return 1;
    }
    if (*(volatile uint64_t *)(p->win + VPU_OFF_READY) != VPU_MAGIC) {
        fprintf(stderr, "gpu-phi-stage: card %d: no worker polling\n", p->card);
        return 1;
    }
    p->rq = (volatile struct vpu_slot_req *)(p->win + VPU_OFF_SLOTS); /* slot 0 */
    p->rp = (volatile struct vpu_slot_reply *)((volatile uint8_t *)p->rq + VPU_SLOT_REPLY);
    p->bell = (volatile uint16_t *)(p->win + VPU_OFF_SLOTSEQ);
    *p->bell = 0;
    p->rp->seq = 0;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    usleep(1000);
    clear_req(p);
    int st = post_wait(p, VPU_K_FLUSH);
    if (st != 0) {
        fprintf(stderr, "gpu-phi-stage: card %d: K_FLUSH status %d (the worker needs slots: -S 1)\n", p->card, st);
        return 1;
    }
    p->nchunks = total / CHUNK;
    return 0;
}

int main(int argc, char **argv)
{
    unsigned mib = argc > 1 ? (unsigned)atoi(argv[1]) : 256;
    if (mib < 2 || mib % 2) {
        fprintf(stderr, "usage: gpu-phi-stage [MIB, even [CARD:GPU_BUS_HEX ...]]\n");
        return 2;
    }
    size_t total = (size_t)mib << 20;
    struct pair pairs[MAXPAIRS];
    int np = 0;
    memset(pairs, 0, sizeof pairs);
    for (int a = 2; a < argc && np < MAXPAIRS; a++) {
        if (sscanf(argv[a], "%d:%x", &pairs[np].card, &pairs[np].bus) != 2) {
            fprintf(stderr, "gpu-phi-stage: pair '%s' is not CARD:GPU_BUS_HEX\n", argv[a]);
            return 2;
        }
        np++;
    }
    if (np == 0) {
        pairs[0].card = 0;
        pairs[0].bus = 0xc7;
        np = 1;
    }
    for (int i = 0; i < np; i++) {
        if (setup(&pairs[i], total) != 0)
            return 1;
        printf("card %d with the GPU on bus %02x: %u MiB in %zu chunks of 2 MiB through %d staging slots\n",
               pairs[i].card, pairs[i].bus, mib, pairs[i].nchunks, NSTAGE);
    }

    /* Pass 1: two coordinated workers per pair, every pair at once. */
    pthread_t th[2 * MAXPAIRS];
    uint64_t t0 = now_ns();
    for (int i = 0; i < np; i++) {
        pthread_create(&th[2 * i], NULL, feeder, &pairs[i]);
        pthread_create(&th[2 * i + 1], NULL, puller, &pairs[i]);
    }
    for (int i = 0; i < 2 * np; i++)
        pthread_join(th[i], NULL);
    uint64_t wall = now_ns() - t0;
    int err = 0;
    for (int i = 0; i < np; i++) {
        struct pair *p = &pairs[i];
        err |= p->err;
        uint64_t pw = p->t_end - t0;
        printf("pass 1, card %d: %u MiB end to end in %.1f ms = %.2f GB/s; GPU to host %.2f GB/s (%.1f ms of copies); "
               "host to card %.2f GB/s (%.1f ms of requests)\n",
               p->card, mib, pw / 1e6, total / (double)pw, total / (double)p->gpu_ns, p->gpu_ns / 1e6,
               total / (double)p->card_ns, p->card_ns / 1e6);
    }
    if (np > 1)
        printf("pass 1, all %d pairs: %.2f GB/s together (%zu MiB in %.1f ms)\n", np,
               (double)np * total / (double)wall, (size_t)np * mib, wall / 1e6);
    if (err) {
        fprintf(stderr, "gpu-phi-stage: pass 1 failed\n");
        return 1;
    }

    /* Pass 2: every chunk of every pair back from its card, compared. */
    for (int i = 0; i < np; i++)
        pthread_create(&th[i], NULL, verifier, &pairs[i]);
    size_t bad = 0, checked = 0;
    for (int i = 0; i < np; i++) {
        pthread_join(th[i], NULL);
        err |= pairs[i].err;
        bad += pairs[i].bad;
        checked += pairs[i].checked;
        printf("pass 2, card %d: %zu of %zu words differ after GPU -> host -> card -> host%s\n", pairs[i].card,
               pairs[i].bad, pairs[i].checked, pairs[i].err ? " (a request failed)" : "");
        clear_req(&pairs[i]);
        post_wait(&pairs[i], VPU_K_FLUSH);
    }
    printf("%s\n", bad == 0 && !err && checked == (size_t)np * (total / 4) ? "PASS" : "FAIL");
    return bad != 0 || err;
}
