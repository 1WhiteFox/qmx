/*
 * Copyright 2026 1WhiteFox
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/qmx_runtime.h"
#include "../include/qmx_kernels.h"
#include "../include/qmx_plugin.h"
#ifndef QMX_NO_MALLOC
#include <stdlib.h>
#endif
#include <string.h>
#if defined(_MSC_VER) && defined(QMX_ENABLE_THREADS) && !defined(QMX_NO_THREADS)
#include <intrin.h>
#endif

#define QMX_PREFILL_MAX_BATCH 32
#define QMX_GEMM_ROW_STRIP    4u
#if defined(QMX_NO_MALLOC)
    #define QMX_GQA_MAX_GROUP 1u
    #define QMX_FLASH_BLOCK   32u
#else
    #define QMX_GQA_MAX_GROUP 8u
    #define QMX_FLASH_BLOCK   64u
#endif

static int qmx_stream_read_bytes(QmxModel *model, uint64_t rel_off, void *dst, size_t len) {
    if (!model->reader.read) return -1;
    if (model->reader.read(model->reader.ctx, rel_off, dst, len) != 0) return -1;
    model->disk_reads += 1u;
    model->disk_bytes_read += (uint64_t)len;
    return 0;
}

static size_t qmx_scale_tile_bytes(uint8_t qtype) {
    if (qtype == QMX_QUANT_Q4_ROW32 || qtype == QMX_QUANT_Q8_ROW32) {
        return (size_t)QMX_TILE_ROWS * sizeof(uint16_t);
    }
    return sizeof(float);
}

static inline uint32_t qmx_atomic_fetch_add_u32(volatile uint32_t *ptr, uint32_t add) {
#if defined(QMX_ENABLE_THREADS) && !defined(QMX_NO_THREADS)
    #if defined(_MSC_VER)
        return (uint32_t)_InterlockedExchangeAdd((volatile long *)ptr, (long)add);
    #elif defined(__GNUC__) || defined(__clang__)
        return __atomic_fetch_add(ptr, add, __ATOMIC_RELAXED);
    #else
        uint32_t old = *ptr;
        *ptr = old + add;
        return old;
    #endif
#else
    uint32_t old = *ptr;
    *ptr = old + add;
    return old;
#endif
}

static uint32_t qmx_pick_ws_chunk(uint32_t n_tr, int n_threads) {
    if (n_threads <= 1) return n_tr ? n_tr : 1u;
    uint32_t chunk = n_tr / ((uint32_t)n_threads * 4u);
    chunk &= ~1u;
    if (chunk < 2u) chunk = 2u;
    if (chunk > 16u) chunk = 16u;
    return chunk;
}

static void run_matmul_tr_range(const QmxBackendOps *ops,
                                float * const *dst_batch,
                                int8_t * const *xq_batch,
                                int16_t * const *xq16_batch,
                                int8_t * const *xq_eo_batch,
                                int32_t * const *xsums_batch,
                                float * const *xscales_batch,
                                int batch_size,
                                const QmxTensor *w,
                                uint32_t n_tc,
                                size_t tile_bytes,
                                size_t sc_tile_bytes,
                                int accumulate,
                                uint32_t tr_start,
                                uint32_t tr_end) {
    if (tr_start >= tr_end) return;

    if (batch_size >= 2 && ops->matmul_tiled_batch != NULL &&
        xq_eo_batch != NULL && xsums_batch != NULL) {
        uint32_t r_base = tr_start * QMX_TILE_ROWS;
        uint32_t max_rows = (tr_end - tr_start) * QMX_TILE_ROWS;
        uint32_t sub_rows = w->rows - r_base;
        if (sub_rows > max_rows) sub_rows = max_rows;

        uint32_t tile_off = tr_start * n_tc;
        QmxTensor sub_w;
        memset(&sub_w, 0, sizeof(sub_w));
        sub_w.data       = (const uint8_t *)w->data + (size_t)tile_off * tile_bytes;
        sub_w.scales     = (const float *)((const uint8_t *)w->scales + (size_t)tile_off * sc_tile_bytes);
        sub_w.rows       = sub_rows;
        sub_w.cols       = w->cols;
        sub_w.quant_type = w->quant_type;
        sub_w.tile_rows  = QMX_TILE_ROWS;

        float *sub_dst[QMX_PREFILL_MAX_BATCH];
        int bsz = (batch_size > QMX_PREFILL_MAX_BATCH) ? QMX_PREFILL_MAX_BATCH : batch_size;
        for (int b = 0; b < bsz; b++) sub_dst[b] = dst_batch[b] + r_base;

        ops->matmul_tiled_batch(sub_dst, xq_batch, xq_eo_batch, xsums_batch, xscales_batch,
                                bsz, &sub_w, accumulate);
        return;
    }

    uint32_t step = (batch_size > 1) ? QMX_GEMM_ROW_STRIP : (tr_end - tr_start);
    if (step == 0u) step = 1u;

    for (uint32_t s_tr = tr_start; s_tr < tr_end; s_tr += step) {
        uint32_t strip_tr = tr_end - s_tr;
        if (strip_tr > step) strip_tr = step;

        uint32_t r_base = s_tr * QMX_TILE_ROWS;
        uint32_t max_rows = strip_tr * QMX_TILE_ROWS;
        uint32_t sub_rows = w->rows - r_base;
        if (sub_rows > max_rows) sub_rows = max_rows;

        uint32_t tile_off = s_tr * n_tc;
        QmxTensor sub_w;
        memset(&sub_w, 0, sizeof(sub_w));
        sub_w.data       = (const uint8_t *)w->data + (size_t)tile_off * tile_bytes;
        sub_w.scales     = (const float *)((const uint8_t *)w->scales + (size_t)tile_off * sc_tile_bytes);
        sub_w.rows       = sub_rows;
        sub_w.cols       = w->cols;
        sub_w.quant_type = w->quant_type;
        sub_w.tile_rows  = QMX_TILE_ROWS;

        int use_eo = (ops->matmul_tiled_eo != NULL && xq_eo_batch != NULL && xsums_batch != NULL);
        for (int b = 0; b < batch_size; b++) {
            if (use_eo && xq_eo_batch[b] != NULL && xsums_batch[b] != NULL) {
                ops->matmul_tiled_eo(dst_batch[b] + r_base,
                                     xq_batch[b],
                                     xq_eo_batch[b],
                                     xsums_batch[b],
                                     xscales_batch[b],
                                     &sub_w,
                                     accumulate);
            } else {
                ops->matmul_tiled(dst_batch[b] + r_base,
                                  xq_batch[b],
                                  xq16_batch[b],
                                  xscales_batch[b],
                                  &sub_w,
                                  accumulate);
            }
        }
    }
}

typedef struct {
    const QmxBackendOps *ops;
    float * const       *dst_batch;
    int8_t * const      *xq_batch;
    int16_t * const     *xq16_batch;
    int8_t * const      *xq_eo_batch;
    int32_t * const     *xsums_batch;
    float * const       *xscales_batch;
    int                  batch_size;
    const QmxTensor     *w;
    int                  accumulate;
    uint32_t             n_tr;
    uint32_t             n_tc;
    size_t               tile_bytes;
    size_t               sc_tile_bytes;
    uint32_t             chunk_tr;
    uint8_t              _pad[64];
    volatile uint32_t    next_tr;
} QmxMatmulJob;

static void exec_matmul_slice_job(void *raw_ctx, int tid, int n_threads) {
    QmxMatmulJob *job = (QmxMatmulJob *)raw_ctx;
    (void)tid;
    uint32_t n_tr = job->n_tr;
    if (n_threads <= 1) {
        run_matmul_tr_range(job->ops, job->dst_batch, job->xq_batch, job->xq16_batch,
                            job->xq_eo_batch, job->xsums_batch, job->xscales_batch,
                            job->batch_size, job->w, job->n_tc,
                            job->tile_bytes, job->sc_tile_bytes, job->accumulate,
                            0u, n_tr);
        return;
    }

    uint32_t chunk = job->chunk_tr ? job->chunk_tr : 2u;
    for (;;) {
        uint32_t tr_start = qmx_atomic_fetch_add_u32(&job->next_tr, chunk);
        if (tr_start >= n_tr) break;
        uint32_t tr_end = tr_start + chunk;
        if (tr_end > n_tr) tr_end = n_tr;
        run_matmul_tr_range(job->ops, job->dst_batch, job->xq_batch, job->xq16_batch,
                            job->xq_eo_batch, job->xsums_batch, job->xscales_batch,
                            job->batch_size, job->w, job->n_tc,
                            job->tile_bytes, job->sc_tile_bytes, job->accumulate,
                            tr_start, tr_end);
    }
}

typedef struct {
    const QmxBackendOps *ops;
    float               *dst[3][QMX_PREFILL_MAX_BATCH];
    int8_t * const      *xq_batch;
    int16_t * const     *xq16_batch;
    int8_t * const      *xq_eo_batch;
    int32_t * const     *xsums_batch;
    float * const       *xscales_batch;
    int                  batch_size;
    const QmxTensor     *w[3];
    uint32_t             n_tr[3];
    uint32_t             n_tc[3];
    size_t               tile_bytes[3];
    size_t               sc_tile_bytes[3];
    uint32_t             total_tr;
    uint32_t             chunk_tr;
    uint8_t              _pad[64];
    volatile uint32_t    next_tr;
} QmxQkvJob;

static void exec_qkv_slice_job(void *raw_ctx, int tid, int n_threads) {
    QmxQkvJob *qj = (QmxQkvJob *)raw_ctx;
    (void)tid;
    uint32_t total = qj->total_tr;
    uint32_t chunk = (n_threads <= 1) ? total : (qj->chunk_tr ? qj->chunk_tr : 2u);
    uint32_t b0 = qj->n_tr[0];
    uint32_t b1 = b0 + qj->n_tr[1];

    for (;;) {
        uint32_t g_start = (n_threads <= 1) ? 0u : qmx_atomic_fetch_add_u32(&qj->next_tr, chunk);
        if (g_start >= total) break;
        uint32_t g_end = g_start + chunk;
        if (g_end > total) g_end = total;

        uint32_t cur = g_start;
        while (cur < g_end) {
            int m;
            uint32_t base;
            if (cur < b0)      { m = 0; base = 0u; }
            else if (cur < b1) { m = 1; base = b0; }
            else               { m = 2; base = b1; }
            uint32_t l_start = cur - base;
            uint32_t l_end = g_end - base;
            if (l_end > qj->n_tr[m]) l_end = qj->n_tr[m];
            run_matmul_tr_range(qj->ops, qj->dst[m], qj->xq_batch, qj->xq16_batch,
                                qj->xq_eo_batch, qj->xsums_batch, qj->xscales_batch,
                                qj->batch_size, qj->w[m], qj->n_tc[m],
                                qj->tile_bytes[m], qj->sc_tile_bytes[m],
                                0, l_start, l_end);
            cur = base + l_end;
        }
        if (n_threads <= 1) break;
    }
}

typedef struct {
    const QmxBackendOps *ops;
    const QmxTensor     *w_gate;
    const QmxTensor     *w_up;
    int8_t              *in_xq[QMX_PREFILL_MAX_BATCH];
    int16_t             *in_xq16[QMX_PREFILL_MAX_BATCH];
    int8_t              *in_xq_eo[QMX_PREFILL_MAX_BATCH];
    int32_t             *in_xsums[QMX_PREFILL_MAX_BATCH];
    float               *in_xs[QMX_PREFILL_MAX_BATCH];
    int8_t * const      *out_xq;
    int16_t * const     *out_xq16;
    int8_t * const      *out_xq_eo;
    int32_t * const     *out_xsums;
    float * const       *out_xs;
    int                  batch_size;
    int                  need_xq16;
    uint32_t             n_tr;
    uint32_t             n_tc;
    size_t               g_tile_bytes;
    size_t               g_sc_bytes;
    size_t               u_tile_bytes;
    size_t               u_sc_bytes;
    uint32_t             chunk_tr;
    uint8_t              _pad[64];
    volatile uint32_t    next_tr;
} QmxFfnFusedJob;

static void ffn_fused_strip_b1(const QmxFfnFusedJob *fj, uint32_t tr_s, uint32_t tr_e) {
    const QmxBackendOps *ops = fj->ops;
    uint32_t r_base = tr_s * QMX_TILE_ROWS;
    uint32_t sub_rows = fj->w_gate->rows - r_base;
    uint32_t max_rows = (tr_e - tr_s) * QMX_TILE_ROWS;
    if (sub_rows > max_rows) sub_rows = max_rows;

    QMX_ALIGNED(64) float g32[32];
    QMX_ALIGNED(64) float u32[32];
    float *gp[1] = { g32 };
    float *up[1] = { u32 };

    uint32_t tile_off = tr_s * fj->n_tc;
    QmxTensor sg, su;
    memset(&sg, 0, sizeof(sg));
    sg.data       = (const uint8_t *)fj->w_gate->data + (size_t)tile_off * fj->g_tile_bytes;
    sg.scales     = (const float *)((const uint8_t *)fj->w_gate->scales + (size_t)tile_off * fj->g_sc_bytes);
    sg.rows       = sub_rows;
    sg.cols       = fj->w_gate->cols;
    sg.quant_type = fj->w_gate->quant_type;
    sg.tile_rows  = QMX_TILE_ROWS;

    memset(&su, 0, sizeof(su));
    su.data       = (const uint8_t *)fj->w_up->data + (size_t)tile_off * fj->u_tile_bytes;
    su.scales     = (const float *)((const uint8_t *)fj->w_up->scales + (size_t)tile_off * fj->u_sc_bytes);
    su.rows       = sub_rows;
    su.cols       = fj->w_up->cols;
    su.quant_type = fj->w_up->quant_type;
    su.tile_rows  = QMX_TILE_ROWS;

    run_matmul_tr_range(ops, gp, fj->in_xq, fj->in_xq16, fj->in_xq_eo, fj->in_xsums, fj->in_xs,
                        1, &sg, fj->n_tc, fj->g_tile_bytes, fj->g_sc_bytes, 0, 0u, tr_e - tr_s);
    run_matmul_tr_range(ops, up, fj->in_xq, fj->in_xq16, fj->in_xq_eo, fj->in_xsums, fj->in_xs,
                        1, &su, fj->n_tc, fj->u_tile_bytes, fj->u_sc_bytes, 0, 0u, tr_e - tr_s);

    uint32_t g_idx = r_base / QMX_TILE_COLS;
    int8_t  *oxq   = fj->out_xq[0] + r_base;
    int16_t *oxq16 = (fj->need_xq16 && fj->out_xq16[0]) ? (fj->out_xq16[0] + r_base) : NULL;
    float   *oxs   = fj->out_xs[0] + g_idx;
    if (ops->swiglu_quant) {
        ops->swiglu_quant(g32, u32, oxq, oxq16, oxs, sub_rows);
    } else {
        qmx_swiglu_quant_scalar(g32, u32, oxq, oxq16, oxs, sub_rows);
    }
    if (ops->prepare_act_eo && fj->out_xq_eo[0] && fj->out_xsums[0]) {
        ops->prepare_act_eo(oxq, fj->out_xq_eo[0] + r_base, fj->out_xsums[0] + g_idx, sub_rows);
    }
}

#ifndef QMX_NO_MALLOC
static void ffn_fused_strip_batch(const QmxFfnFusedJob *fj, uint32_t tr_s, uint32_t tr_e) {
    const QmxBackendOps *ops = fj->ops;
    uint32_t r_base = tr_s * QMX_TILE_ROWS;
    uint32_t sub_rows = fj->w_gate->rows - r_base;
    uint32_t max_rows = (tr_e - tr_s) * QMX_TILE_ROWS;
    if (sub_rows > max_rows) sub_rows = max_rows;
    uint32_t g_idx = r_base / QMX_TILE_COLS;

    uint32_t tile_off = tr_s * fj->n_tc;
    QmxTensor sg, su;
    memset(&sg, 0, sizeof(sg));
    sg.data       = (const uint8_t *)fj->w_gate->data + (size_t)tile_off * fj->g_tile_bytes;
    sg.scales     = (const float *)((const uint8_t *)fj->w_gate->scales + (size_t)tile_off * fj->g_sc_bytes);
    sg.rows       = sub_rows;
    sg.cols       = fj->w_gate->cols;
    sg.quant_type = fj->w_gate->quant_type;
    sg.tile_rows  = QMX_TILE_ROWS;

    memset(&su, 0, sizeof(su));
    su.data       = (const uint8_t *)fj->w_up->data + (size_t)tile_off * fj->u_tile_bytes;
    su.scales     = (const float *)((const uint8_t *)fj->w_up->scales + (size_t)tile_off * fj->u_sc_bytes);
    su.rows       = sub_rows;
    su.cols       = fj->w_up->cols;
    su.quant_type = fj->w_up->quant_type;
    su.tile_rows  = QMX_TILE_ROWS;

    QMX_ALIGNED(64) float g_buf[16][32];
    QMX_ALIGNED(64) float u_buf[16][32];
    float *gp[16];
    float *up[16];
    for (int k = 0; k < 16; k++) { gp[k] = g_buf[k]; up[k] = u_buf[k]; }

    for (int b0 = 0; b0 < fj->batch_size; b0 += 16) {
        int bs = fj->batch_size - b0;
        if (bs > 16) bs = 16;

        run_matmul_tr_range(ops, gp, fj->in_xq + b0, fj->in_xq16 + b0,
                            fj->in_xq_eo + b0, fj->in_xsums + b0, fj->in_xs + b0,
                            bs, &sg, fj->n_tc, fj->g_tile_bytes, fj->g_sc_bytes, 0, 0u, tr_e - tr_s);
        run_matmul_tr_range(ops, up, fj->in_xq + b0, fj->in_xq16 + b0,
                            fj->in_xq_eo + b0, fj->in_xsums + b0, fj->in_xs + b0,
                            bs, &su, fj->n_tc, fj->u_tile_bytes, fj->u_sc_bytes, 0, 0u, tr_e - tr_s);

        for (int k = 0; k < bs; k++) {
            int b = b0 + k;
            int8_t  *oxq   = fj->out_xq[b] + r_base;
            int16_t *oxq16 = (fj->need_xq16 && fj->out_xq16[b]) ? (fj->out_xq16[b] + r_base) : NULL;
            float   *oxs   = fj->out_xs[b] + g_idx;
            if (ops->swiglu_quant) {
                ops->swiglu_quant(g_buf[k], u_buf[k], oxq, oxq16, oxs, sub_rows);
            } else {
                qmx_swiglu_quant_scalar(g_buf[k], u_buf[k], oxq, oxq16, oxs, sub_rows);
            }
            if (ops->prepare_act_eo && fj->out_xq_eo[b] && fj->out_xsums[b]) {
                ops->prepare_act_eo(oxq, fj->out_xq_eo[b] + r_base, fj->out_xsums[b] + g_idx, sub_rows);
            }
        }
    }
}
#endif

static void exec_ffn_fused_job(void *raw_ctx, int tid, int n_threads) {
    QmxFfnFusedJob *fj = (QmxFfnFusedJob *)raw_ctx;
    (void)tid;
    uint32_t n_tr = fj->n_tr;
    uint32_t chunk = (n_threads <= 1) ? n_tr : (fj->chunk_tr ? fj->chunk_tr : 2u);

    for (;;) {
        uint32_t tr_start = (n_threads <= 1) ? 0u : qmx_atomic_fetch_add_u32(&fj->next_tr, chunk);
        if (tr_start >= n_tr) break;
        uint32_t tr_end = tr_start + chunk;
        if (tr_end > n_tr) tr_end = n_tr;

        for (uint32_t s_tr = tr_start; s_tr < tr_end; s_tr += 2u) {
            uint32_t e_tr = s_tr + 2u;
            if (e_tr > tr_end) e_tr = tr_end;
#ifndef QMX_NO_MALLOC
            if (fj->batch_size > 1) ffn_fused_strip_batch(fj, s_tr, e_tr);
            else
#endif
            ffn_fused_strip_b1(fj, s_tr, e_tr);
        }
        if (n_threads <= 1) break;
    }
}


void qmx_set_num_threads(QmxModel *model, int num_threads) {
    if (!model) return;
    if (num_threads < 1) num_threads = 1;
    if (num_threads > QMX_MAX_THREADS) num_threads = QMX_MAX_THREADS;

    if (num_threads == 1) {
        if (model->pool) {
            qmx_pool_destroy(model->pool);
            model->pool = NULL;
        }
        model->num_threads = 1;
        return;
    }

    if (!model->pool || qmx_pool_size(model->pool) != num_threads) {
        if (model->pool) {
            qmx_pool_destroy(model->pool);
            model->pool = NULL;
        }
        model->pool = qmx_pool_create(num_threads);
    }
    model->num_threads = model->pool ? qmx_pool_size(model->pool) : 1;
}

static void dispatch_matmul_batch_parallel(QmxModel *model,
                                           float * const *dst_batch,
                                           int8_t * const *xq_batch,
                                           int16_t * const *xq16_batch,
                                           int8_t * const *xq_eo_batch,
                                           int32_t * const *xsums_batch,
                                           float * const *xscales_batch,
                                           int batch_size,
                                           const QmxTensor *w,
                                           int accumulate) {
    const QmxBackendOps *ops = model->ops ? model->ops : &qmx_ops_scalar;
    uint32_t pad_r = (uint32_t)qmx_align_up(w->rows, QMX_TILE_ROWS);
    uint32_t n_tr  = pad_r / QMX_TILE_ROWS;
    uint32_t pad_c = (uint32_t)qmx_align_up(w->cols, QMX_TILE_COLS);
    uint32_t n_tc  = pad_c / QMX_TILE_COLS;

    QmxMatmulJob job;
    job.ops           = ops;
    job.dst_batch     = dst_batch;
    job.xq_batch      = xq_batch;
    job.xq16_batch    = xq16_batch;
    job.xq_eo_batch   = xq_eo_batch;
    job.xsums_batch   = xsums_batch;
    job.xscales_batch = xscales_batch;
    job.batch_size    = batch_size;
    job.w             = w;
    job.accumulate    = accumulate;
    job.n_tr          = n_tr;
    job.n_tc          = n_tc;
    job.tile_bytes    = qmx_tile_data_bytes(w->quant_type);
    job.sc_tile_bytes = qmx_scale_tile_bytes(w->quant_type);
    job.chunk_tr      = 2u;
    job.next_tr       = 0u;

    int n_th = model->num_threads;
    size_t work_elems = (size_t)w->rows * (size_t)w->cols * (size_t)batch_size;
    if (model->pool && n_th > 1 && n_tr >= 8u && work_elems >= 65536u) {
        int pool_n = qmx_pool_size(model->pool);
        if (n_th > pool_n) n_th = pool_n;
        if ((uint32_t)n_th > n_tr / 2u) n_th = (int)(n_tr / 2u);
        if (n_th > 1) {
            job.chunk_tr = qmx_pick_ws_chunk(n_tr, n_th);
            qmx_pool_run(model->pool, n_th, exec_matmul_slice_job, &job);
            return;
        }
    }

    if (batch_size >= 2 && ops->matmul_tiled_batch != NULL && xq_eo_batch != NULL && xsums_batch != NULL) {
        ops->matmul_tiled_batch(dst_batch, xq_batch, xq_eo_batch, xsums_batch, xscales_batch,
                                batch_size, w, accumulate);
        return;
    }

    if (batch_size > 1 && n_tr > QMX_GEMM_ROW_STRIP) {
        exec_matmul_slice_job(&job, 0, 1);
        return;
    }

    int use_eo = (ops->matmul_tiled_eo != NULL && xq_eo_batch != NULL && xsums_batch != NULL);
    for (int b = 0; b < batch_size; b++) {
        if (use_eo && xq_eo_batch[b] != NULL && xsums_batch[b] != NULL) {
            ops->matmul_tiled_eo(dst_batch[b], xq_batch[b], xq_eo_batch[b], xsums_batch[b], xscales_batch[b], w, accumulate);
        } else {
            ops->matmul_tiled(dst_batch[b], xq_batch[b], xq16_batch[b], xscales_batch[b], w, accumulate);
        }
    }
}

static void quantize_head_q8(const float * QMX_RESTRICT vec, int8_t * QMX_RESTRICT q_out,
                             float * QMX_RESTRICT scale_out, uint32_t head_size) {
    float max_abs = 0.0f;
    for (uint32_t i = 0; i < head_size; i++) {
        float av = (vec[i] < 0.0f) ? -vec[i] : vec[i];
        if (av > max_abs) max_abs = av;
    }
    float sc = max_abs / 127.0f;
    float inv = (max_abs == 0.0f) ? 0.0f : (127.0f / max_abs);
    *scale_out = sc;
    for (uint32_t i = 0; i < head_size; i++) {
        int32_t iv = qmx_round_nearest_i32(vec[i] * inv);
        if (iv > 127) iv = 127;
        if (iv < -127) iv = -127;
        q_out[i] = (int8_t)iv;
    }
}

typedef struct {
    QmxModel    *model;
    QmxMatmulJob mm_job;
    int          cur_read_ok;
    uint32_t     cur_sub_rows;
    uint64_t     next_d_pos;
    uint8_t     *next_buf;
    size_t       next_d_bytes;
    int          next_read_rc;
} QmxStreamDbufJob;

static void exec_stream_dbuf_job(void *raw_ctx, int tid, int n_threads) {
    QmxStreamDbufJob *dj = (QmxStreamDbufJob *)raw_ctx;
    int io_tid = n_threads - 1;
    if (tid == io_tid) {
        dj->next_read_rc = dj->model->reader.read(dj->model->reader.ctx,
                                                  dj->next_d_pos,
                                                  dj->next_buf,
                                                  dj->next_d_bytes);
        return;
    }
    if (dj->cur_read_ok) {
        exec_matmul_slice_job(&dj->mm_job, tid, io_tid);
    } else if (tid == 0 && !dj->mm_job.accumulate) {
        for (int b = 0; b < dj->mm_job.batch_size; b++) {
            memset(dj->mm_job.dst_batch[b], 0, (size_t)dj->cur_sub_rows * sizeof(float));
        }
    }
}

static void qmx_model_matmul_batch(QmxModel *model,
                                   float * const *dst_batch,
                                   int8_t * const *xq_batch,
                                   int16_t * const *xq16_batch,
                                   int8_t * const *xq_eo_batch,
                                   int32_t * const *xsums_batch,
                                   float * const *xscales_batch,
                                   int batch_size,
                                   const QmxTensor *w,
                                   int32_t layer_idx,
                                   int accumulate) {
    if (w->data != NULL && w->scales != NULL) {
        dispatch_matmul_batch_parallel(model, dst_batch, xq_batch, xq16_batch, xq_eo_batch, xsums_batch, xscales_batch,
                                       batch_size, w, accumulate);
        return;
    }

    QmxRunState *s = &model->state;
    const QmxBackendOps *ops = model->ops ? model->ops : &qmx_ops_scalar;
    model->last_stream_layer = layer_idx;
    uint32_t orig_rows = w->rows;
    uint32_t orig_cols = w->cols;
    uint32_t pad_r = (uint32_t)qmx_align_up(orig_rows, QMX_TILE_ROWS);
    uint32_t pad_c = (uint32_t)qmx_align_up(orig_cols, QMX_TILE_COLS);
    uint32_t n_tr = pad_r / QMX_TILE_ROWS;
    uint32_t n_tc = pad_c / QMX_TILE_COLS;
    uint32_t total_tiles = n_tr * n_tc;
    size_t tile_bytes = qmx_tile_data_bytes(w->quant_type);
    size_t sc_tile_bytes = qmx_scale_tile_bytes(w->quant_type);
    uint32_t data_tile_cap = (tile_bytes > 0u) ? (uint32_t)(((size_t)s->stream_tile_cap * 512u) / tile_bytes) : s->stream_tile_cap;
    uint32_t cap = (w->scales != NULL) ? data_tile_cap : s->stream_tile_cap;
    uint64_t pay_base = model->payload_offset_64;

    const float *base_scales = w->scales;
    if (base_scales == NULL && total_tiles <= s->stream_tile_cap) {
        size_t all_s_bytes = (size_t)total_tiles * sc_tile_bytes;
        if (qmx_stream_read_bytes(model, pay_base + w->scales_offset, s->stream_scales_buf, all_s_bytes) == 0) {
            base_scales = s->stream_scales_buf;
            cap = data_tile_cap;
        }
    }

    if (n_tc <= cap) {
        uint32_t rows_per_chunk = cap / n_tc;
        if (rows_per_chunk == 0u) rows_per_chunk = 1u;

        if (base_scales != NULL && s->stream_data_buf2 != NULL &&
            model->pool != NULL && model->num_threads >= 2 &&
            model->reader.read != NULL && n_tr > rows_per_chunk) {
            int active_th = model->num_threads;
            int pool_n = qmx_pool_size(model->pool);
            if (active_th > pool_n) active_th = pool_n;
            if (active_th >= 2) {
                uint8_t *bufs[2] = { s->stream_data_buf, s->stream_data_buf2 };
                int slot = 0;
                uint32_t first_tr = (n_tr > rows_per_chunk) ? rows_per_chunk : n_tr;
                size_t first_bytes = (size_t)(first_tr * n_tc) * tile_bytes;
                int cur_ok = (qmx_stream_read_bytes(model, pay_base + w->data_offset, bufs[0], first_bytes) == 0);

                for (uint32_t tr_start = 0; tr_start < n_tr; tr_start += rows_per_chunk) {
                    uint32_t chunk_tr = n_tr - tr_start;
                    if (chunk_tr > rows_per_chunk) chunk_tr = rows_per_chunk;
                    uint32_t tile_idx = tr_start * n_tc;
                    uint32_t r_base = tr_start * QMX_TILE_ROWS;
                    uint32_t sub_rows = orig_rows - r_base;
                    if (sub_rows > chunk_tr * QMX_TILE_ROWS) sub_rows = chunk_tr * QMX_TILE_ROWS;

                    QmxTensor sub_w;
                    memset(&sub_w, 0, sizeof(sub_w));
                    sub_w.data       = bufs[slot];
                    sub_w.scales     = (const float *)((const uint8_t *)base_scales + (size_t)tile_idx * sc_tile_bytes);
                    sub_w.rows       = sub_rows;
                    sub_w.cols       = orig_cols;
                    sub_w.quant_type = w->quant_type;
                    sub_w.tile_rows  = QMX_TILE_ROWS;

                    float *chunk_dst[QMX_PREFILL_MAX_BATCH];
                    for (int b = 0; b < batch_size; b++) chunk_dst[b] = dst_batch[b] + r_base;

                    uint32_t next_tr_start = tr_start + rows_per_chunk;
                    if (next_tr_start < n_tr) {
                        uint32_t next_tr = n_tr - next_tr_start;
                        if (next_tr > rows_per_chunk) next_tr = rows_per_chunk;
                        uint32_t next_tile_idx = next_tr_start * n_tc;
                        size_t next_d_bytes = (size_t)(next_tr * n_tc) * tile_bytes;

                        QmxStreamDbufJob dj;
                        dj.model                = model;
                        dj.mm_job.ops           = ops;
                        dj.mm_job.dst_batch     = chunk_dst;
                        dj.mm_job.xq_batch      = xq_batch;
                        dj.mm_job.xq16_batch    = xq16_batch;
                        dj.mm_job.xq_eo_batch   = xq_eo_batch;
                        dj.mm_job.xsums_batch   = xsums_batch;
                        dj.mm_job.xscales_batch = xscales_batch;
                        dj.mm_job.batch_size    = batch_size;
                        dj.mm_job.w             = &sub_w;
                        dj.mm_job.accumulate    = accumulate;
                        dj.mm_job.n_tr          = chunk_tr;
                        dj.mm_job.n_tc          = n_tc;
                        dj.mm_job.tile_bytes    = tile_bytes;
                        dj.mm_job.sc_tile_bytes = sc_tile_bytes;
                        dj.mm_job.chunk_tr      = qmx_pick_ws_chunk(chunk_tr, active_th - 1);
                        dj.mm_job.next_tr       = 0u;
                        dj.cur_read_ok          = cur_ok;
                        dj.cur_sub_rows         = sub_rows;
                        dj.next_d_pos           = pay_base + w->data_offset + (uint64_t)next_tile_idx * (uint64_t)tile_bytes;
                        dj.next_buf             = bufs[1 - slot];
                        dj.next_d_bytes         = next_d_bytes;
                        dj.next_read_rc         = -1;

                        qmx_pool_run(model->pool, active_th, exec_stream_dbuf_job, &dj);
                        cur_ok = (dj.next_read_rc == 0);
                        if (cur_ok) {
                            model->disk_reads += 1u;
                            model->disk_bytes_read += (uint64_t)next_d_bytes;
                        }
                        slot = 1 - slot;
                    } else if (cur_ok) {
                        dispatch_matmul_batch_parallel(model, chunk_dst, xq_batch, xq16_batch, xq_eo_batch, xsums_batch, xscales_batch,
                                                       batch_size, &sub_w, accumulate);
                    } else if (!accumulate) {
                        for (int b = 0; b < batch_size; b++) {
                            memset(chunk_dst[b], 0, (size_t)sub_rows * sizeof(float));
                        }
                    }
                }
                return;
            }
        }

        for (uint32_t tr_start = 0; tr_start < n_tr; tr_start += rows_per_chunk) {
            uint32_t chunk_tr = n_tr - tr_start;
            if (chunk_tr > rows_per_chunk) chunk_tr = rows_per_chunk;

            uint32_t tile_idx = tr_start * n_tc;
            uint32_t chunk_tiles = chunk_tr * n_tc;
            uint32_t r_base = tr_start * QMX_TILE_ROWS;
            uint32_t sub_rows = orig_rows - r_base;
            if (sub_rows > chunk_tr * QMX_TILE_ROWS) sub_rows = chunk_tr * QMX_TILE_ROWS;

            size_t d_bytes = (size_t)chunk_tiles * tile_bytes;
            uint64_t d_pos = pay_base + w->data_offset + (uint64_t)tile_idx * (uint64_t)tile_bytes;
            const float *chunk_scales = NULL;

            if (base_scales != NULL) {
                if (qmx_stream_read_bytes(model, d_pos, s->stream_data_buf, d_bytes) != 0) {
                    if (!accumulate) {
                        for (int b = 0; b < batch_size; b++) {
                            memset(dst_batch[b] + r_base, 0, (size_t)sub_rows * sizeof(float));
                        }
                    }
                    continue;
                }
                chunk_scales = (const float *)((const uint8_t *)base_scales + (size_t)tile_idx * sc_tile_bytes);
            } else {
                size_t s_bytes = (size_t)chunk_tiles * sc_tile_bytes;
                uint64_t s_pos = pay_base + w->scales_offset + (uint64_t)tile_idx * (uint64_t)sc_tile_bytes;
                if (qmx_stream_read_bytes(model, s_pos, s->stream_scales_buf, s_bytes) != 0 ||
                    qmx_stream_read_bytes(model, d_pos, s->stream_data_buf, d_bytes) != 0) {
                    if (!accumulate) {
                        for (int b = 0; b < batch_size; b++) {
                            memset(dst_batch[b] + r_base, 0, (size_t)sub_rows * sizeof(float));
                        }
                    }
                    continue;
                }
                chunk_scales = s->stream_scales_buf;
            }

            QmxTensor sub_w;
            memset(&sub_w, 0, sizeof(sub_w));
            sub_w.data       = s->stream_data_buf;
            sub_w.scales     = chunk_scales;
            sub_w.rows       = sub_rows;
            sub_w.cols       = orig_cols;
            sub_w.quant_type = w->quant_type;
            sub_w.tile_rows  = QMX_TILE_ROWS;

            float *chunk_dst[QMX_PREFILL_MAX_BATCH];
            for (int b = 0; b < batch_size; b++) chunk_dst[b] = dst_batch[b] + r_base;
            dispatch_matmul_batch_parallel(model, chunk_dst, xq_batch, xq16_batch, xq_eo_batch, xsums_batch, xscales_batch,
                                           batch_size, &sub_w, accumulate);
        }
        return;
    }

    for (uint32_t tr = 0; tr < n_tr; tr++) {
        uint32_t r_base = tr * QMX_TILE_ROWS;
        uint32_t sub_rows = orig_rows - r_base;
        if (sub_rows > QMX_TILE_ROWS) sub_rows = QMX_TILE_ROWS;

        for (uint32_t tc_start = 0; tc_start < n_tc; tc_start += cap) {
            uint32_t chunk_tc = n_tc - tc_start;
            if (chunk_tc > cap) chunk_tc = cap;

            uint32_t tile_idx = tr * n_tc + tc_start;
            size_t d_bytes = (size_t)chunk_tc * tile_bytes;
            uint64_t d_pos = pay_base + w->data_offset + (uint64_t)tile_idx * (uint64_t)tile_bytes;
            const float *chunk_scales = NULL;

            if (base_scales != NULL) {
                if (qmx_stream_read_bytes(model, d_pos, s->stream_data_buf, d_bytes) != 0) {
                    if (!accumulate && tc_start == 0) {
                        for (int b = 0; b < batch_size; b++) {
                            memset(dst_batch[b] + r_base, 0, (size_t)sub_rows * sizeof(float));
                        }
                    }
                    continue;
                }
                chunk_scales = (const float *)((const uint8_t *)base_scales + (size_t)tile_idx * sc_tile_bytes);
            } else {
                size_t s_bytes = (size_t)chunk_tc * sc_tile_bytes;
                uint64_t s_pos = pay_base + w->scales_offset + (uint64_t)tile_idx * (uint64_t)sc_tile_bytes;
                if (qmx_stream_read_bytes(model, s_pos, s->stream_scales_buf, s_bytes) != 0 ||
                    qmx_stream_read_bytes(model, d_pos, s->stream_data_buf, d_bytes) != 0) {
                    if (!accumulate && tc_start == 0) {
                        for (int b = 0; b < batch_size; b++) {
                            memset(dst_batch[b] + r_base, 0, (size_t)sub_rows * sizeof(float));
                        }
                    }
                    continue;
                }
                chunk_scales = s->stream_scales_buf;
            }

            uint32_t col_start = tc_start * QMX_TILE_COLS;
            uint32_t sub_cols = orig_cols - col_start;
            if (sub_cols > chunk_tc * QMX_TILE_COLS) sub_cols = chunk_tc * QMX_TILE_COLS;

            QmxTensor sub_w;
            memset(&sub_w, 0, sizeof(sub_w));
            sub_w.data       = s->stream_data_buf;
            sub_w.scales     = chunk_scales;
            sub_w.rows       = sub_rows;
            sub_w.cols       = sub_cols;
            sub_w.quant_type = w->quant_type;
            sub_w.tile_rows  = QMX_TILE_ROWS;

            int sub_acc = (accumulate || (tc_start > 0)) ? 1 : 0;
            int use_eo = (ops->matmul_tiled_eo != NULL && xq_eo_batch != NULL && xsums_batch != NULL);
            for (int b = 0; b < batch_size; b++) {
                if (use_eo && xq_eo_batch[b] != NULL && xsums_batch[b] != NULL) {
                    ops->matmul_tiled_eo(dst_batch[b] + r_base,
                                         xq_batch[b] + col_start,
                                         xq_eo_batch[b] + col_start,
                                         xsums_batch[b] + tc_start,
                                         xscales_batch[b] + tc_start,
                                         &sub_w, sub_acc);
                } else {
                    ops->matmul_tiled(dst_batch[b] + r_base,
                                      xq_batch[b] + col_start,
                                      xq16_batch[b] + col_start,
                                      xscales_batch[b] + tc_start,
                                      &sub_w, sub_acc);
                }
            }
        }
    }
}

void qmx_model_matmul(QmxModel *model, float * QMX_RESTRICT out,
                      const QmxTensor *w, int32_t layer_idx, int accumulate) {
    QmxRunState *s = &model->state;
    float *dst_arr[1] = { out };
    int8_t *xq_arr[1] = { s->xq };
    int16_t *xq16_arr[1] = { s->xq16 };
    int8_t *xq_eo_arr[1] = { s->xq_eo };
    int32_t *xsums_arr[1] = { s->x_sums };
    float *xs_arr[1] = { s->x_scales };
    qmx_model_matmul_batch(model, dst_arr, xq_arr, xq16_arr, xq_eo_arr, xsums_arr, xs_arr, 1, w, layer_idx, accumulate);
}

void qmx_load_token_embedding(QmxModel *model, float * QMX_RESTRICT dst, int token) {
    if (!model || !dst) return;
    if (token < 0 || (uint32_t)token >= model->header.vocab_size) token = 0;
    uint32_t dim = model->header.hidden_dim;
    const QmxTensor *emb = model->tok_emb_tensor ? model->tok_emb_tensor : qmx_find_tensor(model, 0xFFFFu, QMX_TID_TOKEN_EMB);
    if (!emb) {
        memset(dst, 0, (size_t)dim * sizeof(float));
        return;
    }
    if (emb->data != NULL && emb->scales != NULL) {
        qmx_dequantize_tiled_row(dst, emb, (uint32_t)token);
        if (model->emb_scale != 1.0f) {
            float es = model->emb_scale;
            for (uint32_t i = 0; i < dim; i++) dst[i] *= es;
        }
        return;
    }

    QmxRunState *s = &model->state;
    uint32_t emb_cap = s->emb_cache_cap ? s->emb_cache_cap : QMX_EMB_CACHE_SIZE;
    uint32_t victim = 0;
    uint32_t min_lru = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < emb_cap; i++) {
        if (s->emb_cache_tokens[i] == token) {
            model->emb_cache_hits++;
            s->emb_cache_lru[i] = ++model->emb_lru_clock;
            memcpy(dst, s->emb_cache_vecs + (size_t)i * dim, (size_t)dim * sizeof(float));
            return;
        }
        if (s->emb_cache_tokens[i] == -1) {
            victim = i;
            min_lru = 0;
        } else if (s->emb_cache_lru[i] < min_lru) {
            min_lru = s->emb_cache_lru[i];
            victim = i;
        }
    }

    model->emb_cache_misses++;
    uint32_t tr = (uint32_t)token / QMX_TILE_ROWS;
    uint32_t lr = (uint32_t)token % QMX_TILE_ROWS;
    uint32_t pad_c = (uint32_t)qmx_align_up(dim, QMX_TILE_COLS);
    uint32_t n_tc = pad_c / QMX_TILE_COLS;
    size_t tile_bytes = qmx_tile_data_bytes(emb->quant_type);
    size_t sc_tile_bytes = qmx_scale_tile_bytes(emb->quant_type);
    uint32_t cap = (emb->scales != NULL && tile_bytes > 0u)
                 ? (uint32_t)(((size_t)s->stream_tile_cap * 512u) / tile_bytes)
                 : s->stream_tile_cap;
    uint64_t pay_base = model->payload_offset_64;
    float *cached_vec = s->emb_cache_vecs + (size_t)victim * dim;

    for (uint32_t tc_start = 0; tc_start < n_tc; tc_start += cap) {
        uint32_t chunk_tc = n_tc - tc_start;
        if (chunk_tc > cap) chunk_tc = cap;

        uint32_t tile_idx = tr * n_tc + tc_start;
        size_t d_bytes = (size_t)chunk_tc * tile_bytes;
        uint64_t d_pos = pay_base + emb->data_offset + (uint64_t)tile_idx * (uint64_t)tile_bytes;
        const float *chunk_scales = NULL;

        if (emb->scales != NULL) {
            if (qmx_stream_read_bytes(model, d_pos, s->stream_data_buf, d_bytes) != 0) {
                memset(cached_vec, 0, (size_t)dim * sizeof(float));
                break;
            }
            chunk_scales = (const float *)((const uint8_t *)emb->scales + (size_t)tile_idx * sc_tile_bytes);
        } else {
            size_t s_bytes = (size_t)chunk_tc * sc_tile_bytes;
            uint64_t s_pos = pay_base + emb->scales_offset + (uint64_t)tile_idx * (uint64_t)sc_tile_bytes;
            if (qmx_stream_read_bytes(model, s_pos, s->stream_scales_buf, s_bytes) != 0 ||
                qmx_stream_read_bytes(model, d_pos, s->stream_data_buf, d_bytes) != 0) {
                memset(cached_vec, 0, (size_t)dim * sizeof(float));
                break;
            }
            chunk_scales = s->stream_scales_buf;
        }

        uint32_t col_start = tc_start * QMX_TILE_COLS;
        uint32_t sub_cols = dim - col_start;
        if (sub_cols > chunk_tc * QMX_TILE_COLS) sub_cols = chunk_tc * QMX_TILE_COLS;

        QmxTensor sub_emb;
        memset(&sub_emb, 0, sizeof(sub_emb));
        sub_emb.data       = s->stream_data_buf;
        sub_emb.scales     = chunk_scales;
        sub_emb.rows       = lr + 1u;
        sub_emb.cols       = sub_cols;
        sub_emb.quant_type = emb->quant_type;
        sub_emb.tile_rows  = QMX_TILE_ROWS;

        qmx_dequantize_tiled_row(cached_vec + col_start, &sub_emb, lr);
    }

    if (model->emb_scale != 1.0f) {
        float es = model->emb_scale;
        for (uint32_t i = 0; i < dim; i++) cached_vec[i] *= es;
    }

    s->emb_cache_tokens[victim] = token;
    s->emb_cache_lru[victim] = ++model->emb_lru_clock;
    memcpy(dst, cached_vec, (size_t)dim * sizeof(float));
}

int qmx_shift_kv_cache_ex(QmxModel *model, int cur_pos, int shift, int n_sink) {
    if (!model || shift <= 0 || shift >= cur_pos) return 0;
    QmxHeader *h = &model->header;
    QmxRunState *s = &model->state;
    int keep = cur_pos - shift;
    if (n_sink < 0 || n_sink >= keep) n_sink = 0;
    int tail_keep = keep - n_sink;
    int src_start = n_sink + shift;

    uint32_t n_layers = h->num_layers;
    uint32_t n_kv_heads = h->num_kv_heads;
    uint32_t head_size = model->head_dim;
    uint32_t rotary_dim = model->rotary_dim;
    uint32_t half_rot = rotary_dim / 2u;
    uint32_t kv_dim = model->kv_dim;
    int split_half = (h->flags & QMX_FLAG_ROPE_SPLIT_HALF) != 0;

    for (uint32_t idx = 0; idx < half_rot; idx++) {
        float val = -(float)shift * s->rope_inv_freq[idx];
        qmx_sincosf(val, &s->rope_sin[idx], &s->rope_cos[idx]);
    }

    for (uint32_t l = 0; l < n_layers; l++) {
        size_t loff = (size_t)l * h->max_seq_len * kv_dim;
        if (model->kv_type == QMX_KV_Q8) {
            size_t soff_base = (size_t)l * h->max_seq_len * n_kv_heads;
            memmove(s->key_cache_q8 + loff + (size_t)n_sink * kv_dim,
                    s->key_cache_q8 + loff + (size_t)src_start * kv_dim,
                    (size_t)tail_keep * kv_dim * sizeof(int8_t));
            memmove(s->value_cache_q8 + loff + (size_t)n_sink * kv_dim,
                    s->value_cache_q8 + loff + (size_t)src_start * kv_dim,
                    (size_t)tail_keep * kv_dim * sizeof(int8_t));
            memmove(s->key_scales + soff_base + (size_t)n_sink * n_kv_heads,
                    s->key_scales + soff_base + (size_t)src_start * n_kv_heads,
                    (size_t)tail_keep * n_kv_heads * sizeof(float));
            memmove(s->value_scales + soff_base + (size_t)n_sink * n_kv_heads,
                    s->value_scales + soff_base + (size_t)src_start * n_kv_heads,
                    (size_t)tail_keep * n_kv_heads * sizeof(float));

            float *tmp_k = s->slots[QMX_BUF_K];
            for (int t = n_sink; t < keep; t++) {
                int8_t *k_q8 = s->key_cache_q8 + loff + (size_t)t * kv_dim;
                float *ks = s->key_scales + soff_base + (size_t)t * n_kv_heads;
                for (uint32_t kh = 0; kh < n_kv_heads; kh++) {
                    float sc = ks[kh];
                    for (uint32_t i = 0; i < head_size; i++) {
                        tmp_k[kh * head_size + i] = (float)k_q8[kh * head_size + i] * sc;
                    }
                }
                qmx_rope_apply_partial(tmp_k, NULL, s->rope_cos, s->rope_sin, kv_dim, 0, head_size, rotary_dim, split_half);
                for (uint32_t kh = 0; kh < n_kv_heads; kh++) {
                    quantize_head_q8(tmp_k + kh * head_size, k_q8 + kh * head_size, &ks[kh], head_size);
                }
            }
        } else if (model->kv_type == QMX_KV_FP16) {
            memmove(s->key_cache_f16 + loff + (size_t)n_sink * kv_dim,
                    s->key_cache_f16 + loff + (size_t)src_start * kv_dim,
                    (size_t)tail_keep * kv_dim * sizeof(uint16_t));
            memmove(s->value_cache_f16 + loff + (size_t)n_sink * kv_dim,
                    s->value_cache_f16 + loff + (size_t)src_start * kv_dim,
                    (size_t)tail_keep * kv_dim * sizeof(uint16_t));

            const QmxBackendOps *ops = model->ops ? model->ops : &qmx_ops_scalar;
            float *tmp_k = s->slots[QMX_BUF_K];
            for (int t = n_sink; t < keep; t++) {
                uint16_t *k_f16 = s->key_cache_f16 + loff + (size_t)t * kv_dim;
                if (ops->f16_to_f32) ops->f16_to_f32(tmp_k, k_f16, kv_dim);
                else qmx_f16_to_f32_scalar(tmp_k, k_f16, kv_dim);
                qmx_rope_apply_partial(tmp_k, NULL, s->rope_cos, s->rope_sin, kv_dim, 0, head_size, rotary_dim, split_half);
                if (ops->f32_to_f16) ops->f32_to_f16(k_f16, tmp_k, kv_dim);
                else qmx_f32_to_f16_scalar(k_f16, tmp_k, kv_dim);
            }
        } else {
            memmove(s->key_cache + loff + (size_t)n_sink * kv_dim,
                    s->key_cache + loff + (size_t)src_start * kv_dim,
                    (size_t)tail_keep * kv_dim * sizeof(float));
            memmove(s->value_cache + loff + (size_t)n_sink * kv_dim,
                    s->value_cache + loff + (size_t)src_start * kv_dim,
                    (size_t)tail_keep * kv_dim * sizeof(float));
            for (int t = n_sink; t < keep; t++) {
                float *k_vec = s->key_cache + loff + (size_t)t * kv_dim;
                qmx_rope_apply_partial(k_vec, NULL, s->rope_cos, s->rope_sin, kv_dim, 0, head_size, rotary_dim, split_half);
            }
        }
    }
    return keep;
}

int qmx_shift_kv_cache(QmxModel *model, int cur_pos, int shift) {
    int sink = (cur_pos >= 16 && (cur_pos - shift) > 4) ? 4 : 0;
    return qmx_shift_kv_cache_ex(model, cur_pos, shift, sink);
}

static void apply_tanh_softcap(float *vec, uint32_t n, float cap) {
    if (cap <= 0.0f) return;
    float inv_cap = 1.0f / cap;
    for (uint32_t i = 0; i < n; i++) {
        float z = vec[i] * inv_cap;
        float th;
        if (z > 20.0f) th = 1.0f;
        else if (z < -20.0f) th = -1.0f;
        else {
            float e2z = qmx_expf(2.0f * z);
            th = (e2z - 1.0f) / (e2z + 1.0f);
        }
        vec[i] = cap * th;
    }
}

static inline float qmx_softcap_scalar(float v, float cap, float inv_cap) {
    float z = v * inv_cap;
    float th;
    if (z > 20.0f) th = 1.0f;
    else if (z < -20.0f) th = -1.0f;
    else {
        float e2z = qmx_expf(2.0f * z);
        th = (e2z - 1.0f) / (e2z + 1.0f);
    }
    return cap * th;
}

typedef struct {
    QmxModel    *model;
    uint32_t     layer_idx;
    int          pos;
    int          t_start;
    uint32_t     att_len;
    const float *q_vec;
    float       *out_vec;
    float        inv_sqrt_head;
    uint32_t     att_group_cap;
} QmxAttnJob;

static void exec_attn_heads_job(void *raw_ctx, int tid, int n_threads) {
    const QmxAttnJob *aj = (const QmxAttnJob *)raw_ctx;
    QmxModel *model = aj->model;
    const QmxHeader *h = &model->header;
    const QmxRunState *s = &model->state;
    const QmxBackendOps *ops = model->ops ? model->ops : &qmx_ops_scalar;

    uint32_t n_heads = h->num_heads;
    uint32_t n_kv_heads = h->num_kv_heads;
    uint32_t head_size = model->head_dim;
    uint32_t kv_dim = model->kv_dim;
    uint32_t kv_mul = n_heads / n_kv_heads;
    size_t loff = (size_t)aj->layer_idx * h->max_seq_len * kv_dim;
    float attn_cap = model->attn_softcap;
    float inv_cap = (attn_cap > 0.0f) ? (1.0f / attn_cap) : 0.0f;
    int t_start = aj->t_start;
    int pos = aj->pos;
    float inv_sqrt_head = aj->inv_sqrt_head;
    uint32_t group_cap = aj->att_group_cap ? aj->att_group_cap : 1u;
    if (group_cap > QMX_GQA_MAX_GROUP) group_cap = QMX_GQA_MAX_GROUP;

    uint32_t hd_start = ((uint32_t)tid * n_heads) / (uint32_t)n_threads;
    uint32_t hd_end   = (((uint32_t)tid + 1u) * n_heads) / (uint32_t)n_threads;
    if (hd_start >= hd_end) return;

    float scores[QMX_GQA_MAX_GROUP][QMX_FLASH_BLOCK];
    float m_run[QMX_GQA_MAX_GROUP];
    float l_run[QMX_GQA_MAX_GROUP];

    if (model->kv_type == QMX_KV_Q8) {
        uint32_t q_stride = (uint32_t)qmx_align_up(head_size, QMX_TILE_COLS);
        uint32_t n_groups = q_stride / QMX_TILE_COLS;
        uint32_t g_lim = (q_stride <= 128u) ? 4u : 1u;
        if (g_lim > group_cap) g_lim = group_cap;

        QMX_ALIGNED(64) int8_t  q_xq[512];
        QMX_ALIGNED(64) int16_t q_xq16[512];
        float   q_xs[64];
        float   v_f32[512];

        for (uint32_t hd = hd_start; hd < hd_end; ) {
            uint32_t kv_h = hd / kv_mul;
            uint32_t kv_head_off = kv_h * head_size;
            uint32_t g_cnt = 1u;
            while (hd + g_cnt < hd_end && g_cnt < g_lim && ((hd + g_cnt) / kv_mul) == kv_h) {
                g_cnt++;
            }

            float *xbs[QMX_GQA_MAX_GROUP];
            for (uint32_t g = 0; g < g_cnt; g++) {
                m_run[g] = -3.402823466e+38f;
                l_run[g] = 0.0f;
                xbs[g]   = aj->out_vec + (hd + g) * head_size;
                memset(xbs[g], 0, (size_t)head_size * sizeof(float));
                ops->prepare_act(aj->q_vec + (hd + g) * head_size,
                                 q_xq + g * q_stride,
                                 q_xq16 + g * q_stride,
                                 q_xs + g * n_groups,
                                 head_size);
            }

            for (int tb = t_start; tb <= pos; tb += (int)QMX_FLASH_BLOCK) {
                int tb_end = tb + (int)QMX_FLASH_BLOCK - 1;
                if (tb_end > pos) tb_end = pos;
                uint32_t blen = (uint32_t)(tb_end - tb + 1);

                for (uint32_t k = 0; k < blen; k++) {
                    int t = tb + (int)k;
                    const int8_t *k8 = s->key_cache_q8 + loff + (size_t)t * kv_dim + kv_head_off;
                    float k_sc = s->key_scales[((size_t)aj->layer_idx * h->max_seq_len + (size_t)t) * n_kv_heads + kv_h] * inv_sqrt_head;
                    for (uint32_t g = 0; g < g_cnt; g++) {
                        const int8_t  *g_xq   = q_xq + g * q_stride;
                        const int16_t *g_xq16 = q_xq16 + g * q_stride;
                        const float   *g_xs   = q_xs + g * n_groups;
                        float dot_f = 0.0f;
                        for (uint32_t c = 0; c < n_groups; c++) {
                            uint32_t off = c * QMX_TILE_COLS;
                            uint32_t len = head_size - off;
                            if (len > QMX_TILE_COLS) len = QMX_TILE_COLS;
                            int32_t idot = ops->dot_q8(g_xq + off, g_xq16 + off, k8 + off, len);
                            dot_f += (float)idot * g_xs[c];
                        }
                        float sc = dot_f * k_sc;
                        if (attn_cap > 0.0f) sc = qmx_softcap_scalar(sc, attn_cap, inv_cap);
                        scores[g][k] = sc;
                    }
                }

                for (uint32_t g = 0; g < g_cnt; g++) {
                    float bmax = scores[g][0];
                    for (uint32_t k = 1u; k < blen; k++) {
                        if (scores[g][k] > bmax) bmax = scores[g][k];
                    }
                    float m_new = (bmax > m_run[g]) ? bmax : m_run[g];
                    if (l_run[g] > 0.0f && m_new > m_run[g]) {
                        float alpha = qmx_expf(m_run[g] - m_new);
                        float *xb = xbs[g];
                        for (uint32_t i = 0; i < head_size; i++) xb[i] *= alpha;
                        l_run[g] *= alpha;
                    }
                    m_run[g] = m_new;
                    float bsum = 0.0f;
                    for (uint32_t k = 0; k < blen; k++) {
                        float w = qmx_expf(scores[g][k] - m_new);
                        scores[g][k] = w;
                        bsum += w;
                    }
                    l_run[g] += bsum;
                }

                for (uint32_t k = 0; k < blen; k++) {
                    int t = tb + (int)k;
                    const int8_t *v8 = s->value_cache_q8 + loff + (size_t)t * kv_dim + kv_head_off;
                    float v_sc = s->value_scales[((size_t)aj->layer_idx * h->max_seq_len + (size_t)t) * n_kv_heads + kv_h];
                    for (uint32_t i = 0; i < head_size; i++) v_f32[i] = (float)v8[i];
                    for (uint32_t g = 0; g < g_cnt; g++) {
                        ops->saxpy_f32(xbs[g], scores[g][k] * v_sc, v_f32, head_size);
                    }
                }
            }

            for (uint32_t g = 0; g < g_cnt; g++) {
                float inv_l = (l_run[g] > 0.0f) ? (1.0f / l_run[g]) : 0.0f;
                float *xb = xbs[g];
                for (uint32_t i = 0; i < head_size; i++) xb[i] *= inv_l;
            }
            hd += g_cnt;
        }
    } else if (model->kv_type == QMX_KV_FP16) {
        float (*dot_f16_fn)(const float * QMX_RESTRICT, const uint16_t * QMX_RESTRICT, uint32_t) =
            ops->dot_f32_f16 ? ops->dot_f32_f16 : qmx_dot_f32_f16_scalar;
        void (*saxpy_f16_fn)(float * QMX_RESTRICT, float, const uint16_t * QMX_RESTRICT, uint32_t) =
            ops->saxpy_f32_f16 ? ops->saxpy_f32_f16 : qmx_saxpy_f32_f16_scalar;

        for (uint32_t hd = hd_start; hd < hd_end; ) {
            uint32_t kv_h = hd / kv_mul;
            uint32_t kv_head_off = kv_h * head_size;
            uint32_t g_cnt = 1u;
            while (hd + g_cnt < hd_end && g_cnt < group_cap && ((hd + g_cnt) / kv_mul) == kv_h) {
                g_cnt++;
            }

            const float *qhs[QMX_GQA_MAX_GROUP];
            float *xbs[QMX_GQA_MAX_GROUP];
            for (uint32_t g = 0; g < g_cnt; g++) {
                m_run[g] = -3.402823466e+38f;
                l_run[g] = 0.0f;
                qhs[g]   = aj->q_vec + (hd + g) * head_size;
                xbs[g]   = aj->out_vec + (hd + g) * head_size;
                memset(xbs[g], 0, (size_t)head_size * sizeof(float));
            }

            for (int tb = t_start; tb <= pos; tb += (int)QMX_FLASH_BLOCK) {
                int tb_end = tb + (int)QMX_FLASH_BLOCK - 1;
                if (tb_end > pos) tb_end = pos;
                uint32_t blen = (uint32_t)(tb_end - tb + 1);

                for (uint32_t k = 0; k < blen; k++) {
                    int t = tb + (int)k;
                    const uint16_t *kh16 = s->key_cache_f16 + loff + (size_t)t * kv_dim + kv_head_off;
                    for (uint32_t g = 0; g < g_cnt; g++) {
                        float sc = dot_f16_fn(qhs[g], kh16, head_size) * inv_sqrt_head;
                        if (attn_cap > 0.0f) sc = qmx_softcap_scalar(sc, attn_cap, inv_cap);
                        scores[g][k] = sc;
                    }
                }

                for (uint32_t g = 0; g < g_cnt; g++) {
                    float bmax = scores[g][0];
                    for (uint32_t k = 1u; k < blen; k++) {
                        if (scores[g][k] > bmax) bmax = scores[g][k];
                    }
                    float m_new = (bmax > m_run[g]) ? bmax : m_run[g];
                    if (l_run[g] > 0.0f && m_new > m_run[g]) {
                        float alpha = qmx_expf(m_run[g] - m_new);
                        float *xb = xbs[g];
                        for (uint32_t i = 0; i < head_size; i++) xb[i] *= alpha;
                        l_run[g] *= alpha;
                    }
                    m_run[g] = m_new;
                    float bsum = 0.0f;
                    for (uint32_t k = 0; k < blen; k++) {
                        float w = qmx_expf(scores[g][k] - m_new);
                        scores[g][k] = w;
                        bsum += w;
                    }
                    l_run[g] += bsum;
                }

                for (uint32_t k = 0; k < blen; k++) {
                    int t = tb + (int)k;
                    const uint16_t *vh16 = s->value_cache_f16 + loff + (size_t)t * kv_dim + kv_head_off;
                    for (uint32_t g = 0; g < g_cnt; g++) {
                        saxpy_f16_fn(xbs[g], scores[g][k], vh16, head_size);
                    }
                }
            }

            for (uint32_t g = 0; g < g_cnt; g++) {
                float inv_l = (l_run[g] > 0.0f) ? (1.0f / l_run[g]) : 0.0f;
                float *xb = xbs[g];
                for (uint32_t i = 0; i < head_size; i++) xb[i] *= inv_l;
            }
            hd += g_cnt;
        }
    } else {
        for (uint32_t hd = hd_start; hd < hd_end; ) {
            uint32_t kv_h = hd / kv_mul;
            uint32_t kv_head_off = kv_h * head_size;
            uint32_t g_cnt = 1u;
            while (hd + g_cnt < hd_end && g_cnt < group_cap && ((hd + g_cnt) / kv_mul) == kv_h) {
                g_cnt++;
            }

            const float *qhs[QMX_GQA_MAX_GROUP];
            float *xbs[QMX_GQA_MAX_GROUP];
            for (uint32_t g = 0; g < g_cnt; g++) {
                m_run[g] = -3.402823466e+38f;
                l_run[g] = 0.0f;
                qhs[g]   = aj->q_vec + (hd + g) * head_size;
                xbs[g]   = aj->out_vec + (hd + g) * head_size;
                memset(xbs[g], 0, (size_t)head_size * sizeof(float));
            }

            for (int tb = t_start; tb <= pos; tb += (int)QMX_FLASH_BLOCK) {
                int tb_end = tb + (int)QMX_FLASH_BLOCK - 1;
                if (tb_end > pos) tb_end = pos;
                uint32_t blen = (uint32_t)(tb_end - tb + 1);

                for (uint32_t k = 0; k < blen; k++) {
                    int t = tb + (int)k;
                    const float *kh = s->key_cache + loff + (size_t)t * kv_dim + kv_head_off;
                    for (uint32_t g = 0; g < g_cnt; g++) {
                        float sc = ops->dot_f32(qhs[g], kh, head_size) * inv_sqrt_head;
                        if (attn_cap > 0.0f) sc = qmx_softcap_scalar(sc, attn_cap, inv_cap);
                        scores[g][k] = sc;
                    }
                }

                for (uint32_t g = 0; g < g_cnt; g++) {
                    float bmax = scores[g][0];
                    for (uint32_t k = 1u; k < blen; k++) {
                        if (scores[g][k] > bmax) bmax = scores[g][k];
                    }
                    float m_new = (bmax > m_run[g]) ? bmax : m_run[g];
                    if (l_run[g] > 0.0f && m_new > m_run[g]) {
                        float alpha = qmx_expf(m_run[g] - m_new);
                        float *xb = xbs[g];
                        for (uint32_t i = 0; i < head_size; i++) xb[i] *= alpha;
                        l_run[g] *= alpha;
                    }
                    m_run[g] = m_new;
                    float bsum = 0.0f;
                    for (uint32_t k = 0; k < blen; k++) {
                        float w = qmx_expf(scores[g][k] - m_new);
                        scores[g][k] = w;
                        bsum += w;
                    }
                    l_run[g] += bsum;
                }

                for (uint32_t k = 0; k < blen; k++) {
                    int t = tb + (int)k;
                    const float *vh = s->value_cache + loff + (size_t)t * kv_dim + kv_head_off;
                    for (uint32_t g = 0; g < g_cnt; g++) {
                        ops->saxpy_f32(xbs[g], scores[g][k], vh, head_size);
                    }
                }
            }

            for (uint32_t g = 0; g < g_cnt; g++) {
                float inv_l = (l_run[g] > 0.0f) ? (1.0f / l_run[g]) : 0.0f;
                float *xb = xbs[g];
                for (uint32_t i = 0; i < head_size; i++) xb[i] *= inv_l;
            }
            hd += g_cnt;
        }
    }
}

static void run_causal_attention(QmxModel *model, uint32_t layer_idx, int pos,
                                 const float *q_vec, const float *k_src, const float *v_src,
                                 float *out_vec, float inv_sqrt_head) {
    QmxHeader *h = &model->header;
    QmxRunState *s = &model->state;
    const QmxBackendOps *ops = model->ops ? model->ops : &qmx_ops_scalar;
    uint32_t n_heads = h->num_heads;
    uint32_t n_kv_heads = h->num_kv_heads;
    uint32_t head_size = model->head_dim;
    uint32_t kv_dim = model->kv_dim;
    size_t loff = (size_t)layer_idx * h->max_seq_len * kv_dim;

    if (model->kv_type == QMX_KV_Q8) {
        size_t soff = ((size_t)layer_idx * h->max_seq_len + (size_t)pos) * n_kv_heads;
        int8_t *k_dst = s->key_cache_q8 + loff + (size_t)pos * kv_dim;
        int8_t *v_dst = s->value_cache_q8 + loff + (size_t)pos * kv_dim;
        for (uint32_t kh = 0; kh < n_kv_heads; kh++) {
            quantize_head_q8(k_src + kh * head_size, k_dst + kh * head_size, &s->key_scales[soff + kh], head_size);
            quantize_head_q8(v_src + kh * head_size, v_dst + kh * head_size, &s->value_scales[soff + kh], head_size);
        }
    } else if (model->kv_type == QMX_KV_FP16) {
        uint16_t *k_dst = s->key_cache_f16 + loff + (size_t)pos * kv_dim;
        uint16_t *v_dst = s->value_cache_f16 + loff + (size_t)pos * kv_dim;
        if (ops->f32_to_f16) {
            ops->f32_to_f16(k_dst, k_src, kv_dim);
            ops->f32_to_f16(v_dst, v_src, kv_dim);
        } else {
            qmx_f32_to_f16_scalar(k_dst, k_src, kv_dim);
            qmx_f32_to_f16_scalar(v_dst, v_src, kv_dim);
        }
    }

    int t_start = 0;
    if (model->sliding_window > 0u && pos >= (int)model->sliding_window) {
        t_start = pos - (int)model->sliding_window + 1;
    }
    uint32_t att_len = (uint32_t)(pos - t_start + 1);
    uint32_t kv_mul = n_heads / n_kv_heads;
    uint32_t desired_g = (kv_mul > QMX_GQA_MAX_GROUP) ? QMX_GQA_MAX_GROUP : kv_mul;

    QmxAttnJob aj;
    aj.model         = model;
    aj.layer_idx     = layer_idx;
    aj.pos           = pos;
    aj.t_start       = t_start;
    aj.att_len       = att_len;
    aj.q_vec         = q_vec;
    aj.out_vec       = out_vec;
    aj.inv_sqrt_head = inv_sqrt_head;
    aj.att_group_cap = desired_g;

    int n_th = model->num_threads;
    if (model->pool && n_th > 1 && n_heads >= 4u && att_len >= 64u) {
        int active = qmx_pool_size(model->pool);
        if (active > n_th) active = n_th;
        if ((uint32_t)active > n_heads) active = (int)n_heads;
        if (active > 1) {
            qmx_pool_run(model->pool, active, exec_attn_heads_job, &aj);
            return;
        }
    }

    exec_attn_heads_job(&aj, 0, 1);
}

static inline float *vm_resolve_slot(QmxModel *model, float *b_slots[][8], int b,
                                     uint16_t slot, uint32_t layer_idx, int p_b) {
    if (model->kv_type == QMX_KV_FP32) {
        if (slot == QMX_BUF_K) {
            return model->state.key_cache + ((size_t)layer_idx * model->header.max_seq_len + (size_t)p_b) * model->kv_dim;
        }
        if (slot == QMX_BUF_V) {
            return model->state.value_cache + ((size_t)layer_idx * model->header.max_seq_len + (size_t)p_b) * model->kv_dim;
        }
    }
    return b_slots[b][slot];
}

static void qmx_vm_exec_chunk(QmxModel *model,
                              float *b_slots[][8],
                              int8_t **b_xq,
                              int16_t **b_xq16,
                              int8_t **b_xq_eo,
                              int32_t **b_xsums,
                              float **b_xs,
                              float **b_cos,
                              float **b_sin,
                              int B,
                              int base_pos) {
    QmxHeader *h = &model->header;
    const QmxBackendOps *ops = model->ops ? model->ops : &qmx_ops_scalar;

    uint32_t dim = h->hidden_dim;
    uint32_t inter_dim = h->intermediate_dim;
    uint32_t n_layers = h->num_layers;
    uint32_t head_size = model->head_dim;
    uint32_t rotary_dim = model->rotary_dim;
    uint32_t q_dim = model->q_dim;
    uint32_t kv_dim = model->kv_dim;
    float eps = (h->rms_norm_eps > 1e-9f && h->rms_norm_eps < 1e-1f) ? h->rms_norm_eps : 1e-5f;
    float inv_sqrt_head = 1.0f / qmx_sqrtf((float)head_size);

    qmx_plugin_emit_hook(QMX_HOOK_STEP_BEGIN, 0u, B, base_pos, -1);
    for (uint32_t l = 0; l < n_layers; l++) {
        qmx_plugin_emit_hook(QMX_HOOK_LAYER_BEGIN, (uint32_t)l, B, base_pos, -1);
        if (qmx_plugin_exec_layer(model, model->backend, (uint32_t)l, B, base_pos) == 0) {
            qmx_plugin_emit_hook(QMX_HOOK_LAYER_END, (uint32_t)l, B, base_pos, -1);
            continue;
        }

        const QmxTensor * const *layer_tensors = model->recipe_tensors + (size_t)l * h->recipe_count;

        for (uint32_t pc = 0; pc < h->recipe_count; pc++) {
            const QmxInstruction *inst = &model->recipe[pc];
            uint16_t sb = inst->src_buf;
            uint16_t db = inst->dst_buf;
            if (sb >= 8u || db >= 8u) continue;
            const QmxTensor *w = layer_tensors[pc];

            switch (inst->op) {
                case QMX_OP_NOP:
                    break;

                case QMX_OP_RMSNORM:
                    if (w && w->data) {
                        if (model->recipe_fuse && model->recipe_fuse[pc] == QMX_FUSE_RMSNORM_QUANT && ops->rmsnorm_quant) {
                            int need16 = qmx_backend_needs_xq16(model->backend);
                            for (int b = 0; b < B; b++) {
                                ops->rmsnorm_quant(b_slots[b][sb], (const float *)w->data,
                                                   b_xq[b], need16 ? b_xq16[b] : NULL, b_xs[b], w->cols, eps);
                                if (ops->prepare_act_eo && b_xq_eo[b] && b_xsums[b]) {
                                    ops->prepare_act_eo(b_xq[b], b_xq_eo[b], b_xsums[b], w->cols);
                                }
                            }
                            pc++;
                        } else {
                            for (int b = 0; b < B; b++) {
                                ops->rmsnorm(b_slots[b][db], b_slots[b][sb], (const float *)w->data, w->cols, eps);
                            }
                        }
                    }
                    break;

                case QMX_OP_SUBNORM: {
                    uint32_t sz = (w != NULL) ? w->cols : ((sb == QMX_BUF_HB_GATE || sb == QMX_BUF_HB_UP) ? inter_dim : dim);
                    for (int b = 0; b < B; b++) {
                        if (w != NULL && w->data != NULL) {
                            ops->rmsnorm(b_slots[b][db], b_slots[b][sb], (const float *)w->data, sz, eps);
                        } else {
                            qmx_subnorm_scalar(b_slots[b][db], b_slots[b][sb], NULL, sz, eps);
                        }
                    }
                    break;
                }

                case QMX_OP_LAYERNORM: {
                    uint32_t sz = (w != NULL) ? w->cols : dim;
                    for (int b = 0; b < B; b++) {
                        qmx_layernorm_scalar(b_slots[b][db], b_slots[b][sb],
                                             (w && w->data) ? (const float *)w->data : NULL, sz, eps);
                    }
                    break;
                }

                case QMX_OP_QK_NORM:
                    if (w && w->data) {
                        uint32_t vec_dim = (db == QMX_BUF_K) ? kv_dim : q_dim;
                        for (int b = 0; b < B; b++) {
                            float *dst_b = vm_resolve_slot(model, b_slots, b, db, l, base_pos + b);
                            qmx_qk_rmsnorm_scalar(dst_b, (const float *)w->data, vec_dim, head_size, eps);
                        }
                    }
                    break;

                case QMX_OP_SCALE_MUL:
                    if (w && w->data) {
                        const float *sc_w = (const float *)w->data;
                        for (int b = 0; b < B; b++) {
                            for (uint32_t i = 0; i < w->cols; i++) {
                                b_slots[b][db][i] = b_slots[b][sb][i] * sc_w[i];
                            }
                        }
                    }
                    break;

                case QMX_OP_QUANT_ACT: {
                    uint32_t qcols = model->recipe_quant_cols[pc];
                    for (int b = 0; b < B; b++) {
                        ops->prepare_act(b_slots[b][sb], b_xq[b], b_xq16[b], b_xs[b], qcols);
                        if (ops->prepare_act_eo && b_xq_eo[b] && b_xsums[b]) {
                            ops->prepare_act_eo(b_xq[b], b_xq_eo[b], b_xsums[b], qcols);
                        }
                    }
                    break;
                }

                case QMX_OP_MATMUL:
                case QMX_OP_MATMUL_ADD:
                    if (inst->op == QMX_OP_MATMUL && model->qkv_fuse_valid &&
                        pc == (uint32_t)model->qkv_fuse_start &&
                        model->pool && model->num_threads > 1) {
                        const QmxTensor *w0 = layer_tensors[model->qkv_fuse_pc[0]];
                        const QmxTensor *w1 = layer_tensors[model->qkv_fuse_pc[1]];
                        const QmxTensor *w2 = layer_tensors[model->qkv_fuse_pc[2]];
                        if (w0 && w1 && w2 && w0->data && w0->scales &&
                            w1->data && w1->scales && w2->data && w2->scales) {
                            QmxQkvJob qj;
                            memset(&qj, 0, sizeof(qj));
                            qj.ops           = ops;
                            qj.xq_batch      = b_xq;
                            qj.xq16_batch    = b_xq16;
                            qj.xq_eo_batch   = b_xq_eo;
                            qj.xsums_batch   = b_xsums;
                            qj.xscales_batch = b_xs;
                            qj.batch_size    = B;
                            qj.w[0] = w0; qj.w[1] = w1; qj.w[2] = w2;
                            size_t total_work = 0u;
                            for (int m = 0; m < 3; m++) {
                                uint16_t mdst = model->recipe[model->qkv_fuse_pc[m]].dst_buf;
                                for (int b = 0; b < B; b++) {
                                    qj.dst[m][b] = vm_resolve_slot(model, b_slots, b, mdst, l, base_pos + b);
                                }
                                qj.n_tr[m]          = (uint32_t)qmx_align_up(qj.w[m]->rows, QMX_TILE_ROWS) / QMX_TILE_ROWS;
                                qj.n_tc[m]          = (uint32_t)qmx_align_up(qj.w[m]->cols, QMX_TILE_COLS) / QMX_TILE_COLS;
                                qj.tile_bytes[m]    = qmx_tile_data_bytes(qj.w[m]->quant_type);
                                qj.sc_tile_bytes[m] = qmx_scale_tile_bytes(qj.w[m]->quant_type);
                                qj.total_tr        += qj.n_tr[m];
                                total_work         += (size_t)qj.w[m]->rows * (size_t)qj.w[m]->cols * (size_t)B;
                            }
                            int n_th = model->num_threads;
                            int pool_n = qmx_pool_size(model->pool);
                            if (n_th > pool_n) n_th = pool_n;
                            if ((uint32_t)n_th > qj.total_tr / 2u) n_th = (int)(qj.total_tr / 2u);
                            if (n_th > 1 && total_work >= 65536u) {
                                qj.chunk_tr = qmx_pick_ws_chunk(qj.total_tr, n_th);
                                qj.next_tr  = 0u;
                                qmx_pool_run(model->pool, n_th, exec_qkv_slice_job, &qj);
                                for (uint32_t k = pc; k <= (uint32_t)model->qkv_fuse_end; k++) {
                                    const QmxInstruction *sub = &model->recipe[k];
                                    const QmxTensor *sw = layer_tensors[k];
                                    if (sub->op == QMX_OP_ADD_BIAS && sw && sw->data) {
                                        const float *bias = (const float *)sw->data;
                                        for (int b = 0; b < B; b++) {
                                            float *dst_b = vm_resolve_slot(model, b_slots, b, sub->dst_buf, l, base_pos + b);
                                            for (uint32_t i = 0; i < sw->cols; i++) dst_b[i] += bias[i];
                                        }
                                    } else if (sub->op == QMX_OP_QK_NORM && sw && sw->data) {
                                        uint32_t vec_dim = (sub->dst_buf == QMX_BUF_K) ? kv_dim : q_dim;
                                        for (int b = 0; b < B; b++) {
                                            float *dst_b = vm_resolve_slot(model, b_slots, b, sub->dst_buf, l, base_pos + b);
                                            qmx_qk_rmsnorm_scalar(dst_b, (const float *)sw->data, vec_dim, head_size, eps);
                                        }
                                    }
                                }
                                pc = (uint32_t)model->qkv_fuse_end;
                                break;
                            }
                        }
                    }

                    if (B == 1 && inst->op == QMX_OP_MATMUL && model->recipe_fuse &&
                        model->recipe_fuse[pc] == QMX_FUSE_FFN_GATE_UP_SWIGLU) {
                        const QmxTensor *wg = layer_tensors[pc];
                        const QmxTensor *wu = layer_tensors[pc + 1u];
                        if (wg && wu && wg->data && wg->scales && wu->data && wu->scales) {
                            uint32_t pad_in = (uint32_t)qmx_align_up(wg->cols, QMX_TILE_COLS);
                            uint32_t in_tc  = pad_in / QMX_TILE_COLS;
                            size_t al_in    = qmx_align_up((size_t)pad_in, QMX_CACHE_ALIGN);
                            size_t al_sc    = qmx_align_up((size_t)in_tc * sizeof(float), QMX_CACHE_ALIGN);
                            size_t need_g   = al_in + (size_t)pad_in * sizeof(int16_t);
                            size_t need_u   = al_in + 2u * al_sc;
                            size_t slot_b   = (size_t)inter_dim * sizeof(float);
                            if (need_g <= slot_b && need_u <= slot_b) {
                                int need16 = qmx_backend_needs_xq16(model->backend);
                                uint16_t g_slot = model->recipe[pc].dst_buf;
                                uint16_t u_slot = model->recipe[pc + 1u].dst_buf;
                                QmxFfnFusedJob fj;
                                memset(&fj, 0, sizeof(fj));
                                fj.ops          = ops;
                                fj.w_gate       = wg;
                                fj.w_up         = wu;
                                fj.out_xq       = b_xq;
                                fj.out_xq16     = b_xq16;
                                fj.out_xq_eo    = b_xq_eo;
                                fj.out_xsums    = b_xsums;
                                fj.out_xs       = b_xs;
                                fj.batch_size   = B;
                                fj.need_xq16    = need16;
                                fj.n_tr         = (uint32_t)qmx_align_up(wg->rows, QMX_TILE_ROWS) / QMX_TILE_ROWS;
                                fj.n_tc         = in_tc;
                                fj.g_tile_bytes = qmx_tile_data_bytes(wg->quant_type);
                                fj.g_sc_bytes   = qmx_scale_tile_bytes(wg->quant_type);
                                fj.u_tile_bytes = qmx_tile_data_bytes(wu->quant_type);
                                fj.u_sc_bytes   = qmx_scale_tile_bytes(wu->quant_type);

                                for (int b = 0; b < B; b++) {
                                    uint8_t *gb = (uint8_t *)b_slots[b][g_slot];
                                    uint8_t *ub = (uint8_t *)b_slots[b][u_slot];
                                    fj.in_xq[b] = (int8_t *)gb;
                                    memcpy(fj.in_xq[b], b_xq[b], (size_t)pad_in);
                                    if (need16 && b_xq16[b]) {
                                        fj.in_xq16[b] = (int16_t *)(gb + al_in);
                                        memcpy(fj.in_xq16[b], b_xq16[b], (size_t)pad_in * sizeof(int16_t));
                                    }
                                    if (b_xq_eo[b] && b_xsums[b]) {
                                        fj.in_xq_eo[b] = (int8_t *)ub;
                                        memcpy(fj.in_xq_eo[b], b_xq_eo[b], (size_t)pad_in);
                                        fj.in_xsums[b] = (int32_t *)(ub + al_in + al_sc);
                                        memcpy(fj.in_xsums[b], b_xsums[b], (size_t)in_tc * sizeof(int32_t));
                                    }
                                    fj.in_xs[b] = (float *)(ub + al_in);
                                    memcpy(fj.in_xs[b], b_xs[b], (size_t)in_tc * sizeof(float));
                                }

                                int n_th = model->num_threads;
                                size_t work_elems = 2u * (size_t)wg->rows * (size_t)wg->cols * (size_t)B;
                                if (model->pool && n_th > 1 && fj.n_tr >= 8u && work_elems >= 65536u) {
                                    int pool_n = qmx_pool_size(model->pool);
                                    if (n_th > pool_n) n_th = pool_n;
                                    if ((uint32_t)n_th > fj.n_tr / 2u) n_th = (int)(fj.n_tr / 2u);
                                    if (n_th > 1) {
                                        fj.chunk_tr = qmx_pick_ws_chunk(fj.n_tr, n_th);
                                        fj.next_tr  = 0u;
                                        qmx_pool_run(model->pool, n_th, exec_ffn_fused_job, &fj);
                                    } else {
                                        exec_ffn_fused_job(&fj, 0, 1);
                                    }
                                } else {
                                    exec_ffn_fused_job(&fj, 0, 1);
                                }
                                pc += 3u;
                                break;
                            }
                        }
                    }

                    if (w) {
                        float *dst_ptrs[QMX_PREFILL_MAX_BATCH];
                        for (int b = 0; b < B; b++) {
                            dst_ptrs[b] = vm_resolve_slot(model, b_slots, b, db, l, base_pos + b);
                        }
                        qmx_model_matmul_batch(model, dst_ptrs, b_xq, b_xq16, b_xq_eo, b_xsums, b_xs, B, w, (int32_t)l,
                                               (inst->op == QMX_OP_MATMUL_ADD) ? 1 : 0);
                    }
                    break;

                case QMX_OP_ADD_BIAS:
                    if (w && w->data) {
                        const float *bias = (const float *)w->data;
                        for (int b = 0; b < B; b++) {
                            float *dst_b = vm_resolve_slot(model, b_slots, b, db, l, base_pos + b);
                            for (uint32_t i = 0; i < w->cols; i++) {
                                dst_b[i] += bias[i];
                            }
                        }
                    }
                    break;

                case QMX_OP_ROPE: {
                    int split_half = ((inst->flags & 1u) != 0) || ((h->flags & QMX_FLAG_ROPE_SPLIT_HALF) != 0);
                    for (int b = 0; b < B; b++) {
                        float *q_ptr = vm_resolve_slot(model, b_slots, b, sb, l, base_pos + b);
                        float *k_ptr = vm_resolve_slot(model, b_slots, b, db, l, base_pos + b);
                        qmx_rope_apply_partial(q_ptr, k_ptr, b_cos[b], b_sin[b],
                                               q_dim, kv_dim, head_size, rotary_dim, split_half);
                    }
                    break;
                }

                case QMX_OP_ATTN:
                    for (int b = 0; b < B; b++) {
                        run_causal_attention(model, l, base_pos + b,
                                             b_slots[b][sb],
                                             b_slots[b][QMX_BUF_K],
                                             b_slots[b][QMX_BUF_V],
                                             b_slots[b][db],
                                             inv_sqrt_head);
                    }
                    break;

                case QMX_OP_SWIGLU:
                    if (model->recipe_fuse && model->recipe_fuse[pc] == QMX_FUSE_SWIGLU_QUANT) {
                        int need16 = qmx_backend_needs_xq16(model->backend);
                        if (ops->swiglu_quant) {
                            for (int b = 0; b < B; b++) {
                                ops->swiglu_quant(b_slots[b][sb], b_slots[b][db],
                                                  b_xq[b], need16 ? b_xq16[b] : NULL, b_xs[b], inter_dim);
                                if (ops->prepare_act_eo && b_xq_eo[b] && b_xsums[b]) {
                                    ops->prepare_act_eo(b_xq[b], b_xq_eo[b], b_xsums[b], inter_dim);
                                }
                            }
                        } else {
                            for (int b = 0; b < B; b++) {
                                qmx_swiglu_quant_scalar(b_slots[b][sb], b_slots[b][db],
                                                        b_xq[b], need16 ? b_xq16[b] : NULL, b_xs[b], inter_dim);
                                if (ops->prepare_act_eo && b_xq_eo[b] && b_xsums[b]) {
                                    ops->prepare_act_eo(b_xq[b], b_xq_eo[b], b_xsums[b], inter_dim);
                                }
                            }
                        }
                        pc++;
                    } else {
                        for (int b = 0; b < B; b++) {
                            float *src_b = b_slots[b][sb];
                            float *dst_b = b_slots[b][db];
                            for (uint32_t i = 0; i < inter_dim; i++) {
                                float v = src_b[i];
                                v *= (1.0f / (1.0f + qmx_expf(-v)));
                                dst_b[i] = v * dst_b[i];
                            }
                        }
                    }
                    break;

                case QMX_OP_SILU:
                    for (int b = 0; b < B; b++) {
                        float *src_b = b_slots[b][sb];
                        float *dst_b = b_slots[b][db];
                        for (uint32_t i = 0; i < inter_dim; i++) {
                            float v = src_b[i];
                            dst_b[i] = v * (1.0f / (1.0f + qmx_expf(-v)));
                        }
                    }
                    break;

                case QMX_OP_GEGLU:
                    for (int b = 0; b < B; b++) {
                        float *src_b = b_slots[b][sb];
                        float *dst_b = b_slots[b][db];
                        for (uint32_t i = 0; i < inter_dim; i++) {
                            float x = src_b[i];
                            float z = 0.7978845608f * (x + 0.044715f * x * x * x);
                            float th;
                            if (z > 20.0f) th = 1.0f;
                            else if (z < -20.0f) th = -1.0f;
                            else {
                                float e2z = qmx_expf(2.0f * z);
                                th = (e2z - 1.0f) / (e2z + 1.0f);
                            }
                            dst_b[i] = 0.5f * x * (1.0f + th) * dst_b[i];
                        }
                    }
                    break;

                case QMX_OP_RELU2:
                    for (int b = 0; b < B; b++) {
                        float *src_b = b_slots[b][sb];
                        float *dst_b = b_slots[b][db];
                        for (uint32_t i = 0; i < inter_dim; i++) {
                            float v = (src_b[i] > 0.0f) ? (src_b[i] * src_b[i]) : 0.0f;
                            dst_b[i] = v * dst_b[i];
                        }
                    }
                    break;

                case QMX_OP_RESIDUAL_ADD:
                    for (int b = 0; b < B; b++) {
                        float *src_b = b_slots[b][sb];
                        float *dst_b = b_slots[b][db];
                        for (uint32_t i = 0; i < dim; i++) {
                            dst_b[i] += src_b[i];
                        }
                    }
                    break;

                default:
                    break;
            }
        }
    
        qmx_plugin_emit_hook(QMX_HOOK_LAYER_END, (uint32_t)l, B, base_pos, -1);
    }
    qmx_plugin_emit_hook(QMX_HOOK_STEP_END, (uint32_t)(n_layers), B, base_pos, -1);
}

static void compute_logits_for_hidden(QmxModel *model, const float *x_vec) {
    QmxHeader *h = &model->header;
    QmxRunState *s = &model->state;
    const QmxBackendOps *ops = model->ops ? model->ops : &qmx_ops_scalar;
    uint32_t dim = h->hidden_dim;
    float eps = (h->rms_norm_eps > 1e-9f && h->rms_norm_eps < 1e-1f) ? h->rms_norm_eps : 1e-5f;

    const QmxTensor *w_final = model->rms_final_tensor;
    const QmxTensor *b_final = model->rms_final_bias_tensor;
    if (w_final && w_final->data && !(b_final && b_final->data) && ops->rmsnorm_quant) {
        int need16 = qmx_backend_needs_xq16(model->backend);
        ops->rmsnorm_quant(x_vec, (const float *)w_final->data, s->xq, need16 ? s->xq16 : NULL, s->x_scales, dim, eps);
    } else {
        if (w_final && w_final->data) {
            if (b_final && b_final->data) {
                qmx_layernorm_scalar(s->slots[QMX_BUF_XB], x_vec, (const float *)w_final->data, dim, eps);
                const float *bf = (const float *)b_final->data;
                for (uint32_t i = 0; i < dim; i++) s->slots[QMX_BUF_XB][i] += bf[i];
            } else {
                ops->rmsnorm(s->slots[QMX_BUF_XB], x_vec, (const float *)w_final->data, dim, eps);
            }
        } else {
            memcpy(s->slots[QMX_BUF_XB], x_vec, (size_t)dim * sizeof(float));
        }
        ops->prepare_act(s->slots[QMX_BUF_XB], s->xq, s->xq16, s->x_scales, dim);
    }
    if (ops->prepare_act_eo && s->xq_eo && s->x_sums) {
        ops->prepare_act_eo(s->xq, s->xq_eo, s->x_sums, dim);
    }
    const QmxTensor *w_cls = model->cls_tensor;
    if (w_cls) {
        qmx_model_matmul(model, s->logits, w_cls, (int32_t)h->num_layers, 0);
    }
    if (model->logit_softcap > 0.0f) {
        apply_tanh_softcap(s->logits, h->vocab_size, model->logit_softcap);
    }
}

static float *qmx_forward_internal(QmxModel *model, int token, int pos, int compute_logits) {
    if (!model) return NULL;
    QmxHeader *h = &model->header;
    QmxRunState *s = &model->state;

    if (token < 0 || (uint32_t)token >= h->vocab_size) token = 0;
    if (pos < 0 || (uint32_t)pos >= h->max_seq_len) pos = (int)h->max_seq_len - 1;

    uint32_t half_rot = model->rotary_dim / 2u;
    qmx_load_token_embedding(model, s->slots[QMX_BUF_X], token);
    for (uint32_t idx = 0; idx < half_rot; idx++) {
        float val = (float)pos * s->rope_inv_freq[idx];
        qmx_sincosf(val, &s->rope_sin[idx], &s->rope_cos[idx]);
    }

    float   *b_slots[1][8];
    int8_t  *b_xq[1]    = { s->xq };
    int16_t *b_xq16[1]  = { s->xq16 };
    int8_t  *b_xq_eo[1] = { s->xq_eo };
    int32_t *b_xsums[1] = { s->x_sums };
    float   *b_xs[1]    = { s->x_scales };
    float   *b_cos[1]   = { s->rope_cos };
    float   *b_sin[1]   = { s->rope_sin };
    for (int sl = 0; sl < 8; sl++) b_slots[0][sl] = s->slots[sl];

    qmx_vm_exec_chunk(model, b_slots, b_xq, b_xq16, b_xq_eo, b_xsums, b_xs, b_cos, b_sin, 1, pos);

    if (compute_logits) {
        compute_logits_for_hidden(model, s->slots[QMX_BUF_X]);
    }
    return s->logits;
}

float *qmx_forward(QmxModel *model, int token, int pos) {
    return qmx_forward_internal(model, token, pos, 1);
}

static float qmx_calc_step0_checksum(const float *logits, uint32_t vocab_size) {
    float chk = 0.0f;
    for (uint32_t i = 0; i < vocab_size; i++) {
        chk += logits[i] * (float)((int)(i % 7u) - 3);
    }
    return chk;
}

static int qmx_ensure_pos_capacity(QmxModel *model, int cur_pos, int add_tokens, int *out_shifted) {
    int max_seq = (int)model->header.max_seq_len;
    if (cur_pos < 0) cur_pos = 0;
    if (cur_pos > max_seq) cur_pos = max_seq;
    if (cur_pos + add_tokens <= max_seq) return cur_pos;
    int needed = (cur_pos + add_tokens) - max_seq;
    int shift = (max_seq / 2 > needed) ? (max_seq / 2) : needed;
    if (shift > cur_pos) shift = cur_pos;
    if (shift < 1) shift = 1;
    if (cur_pos > 0) {
        cur_pos = qmx_shift_kv_cache(model, cur_pos, shift);
        if (out_shifted) *out_shifted = 1;
    }
    return cur_pos;
}

float *qmx_forward_batch_ex(QmxModel *model, const int *tokens, int n_tokens,
                            int *io_pos, float *step0_chk_out, int *out_shifted) {
    if (out_shifted) *out_shifted = 0;
    if (!model || !tokens || n_tokens <= 0) return NULL;
    QmxHeader *h = &model->header;
    QmxRunState *s = &model->state;

    int cur_pos = io_pos ? *io_pos : 0;
    if (cur_pos < 0) cur_pos = 0;

    if (n_tokens == 1) {
        cur_pos = qmx_ensure_pos_capacity(model, cur_pos, 1, out_shifted);
        float *l = qmx_forward_internal(model, tokens[0], cur_pos, 1);
        cur_pos++;
        if (io_pos) *io_pos = cur_pos;
        if (step0_chk_out && l) {
            *step0_chk_out = qmx_calc_step0_checksum(l, h->vocab_size);
        }
        return l;
    }

#ifndef QMX_NO_MALLOC
    int no_batch_ws = model->arena.is_external || (model->stream_mode && s->stream_tile_cap <= 16u);
#else
    int no_batch_ws = 1;
#endif
    if (no_batch_ws) {
        for (int i = 0; i < n_tokens; i++) {
            cur_pos = qmx_ensure_pos_capacity(model, cur_pos, 1, out_shifted);
            int want_logits = (i == n_tokens - 1) || (i == 0 && step0_chk_out != NULL);
            float *l = qmx_forward_internal(model, tokens[i], cur_pos, want_logits);
            cur_pos++;
            if (i == 0 && step0_chk_out != NULL && l != NULL) {
                *step0_chk_out = qmx_calc_step0_checksum(l, h->vocab_size);
            }
        }
        if (io_pos) *io_pos = cur_pos;
        return s->logits;
    }

#ifndef QMX_NO_MALLOC
    uint32_t dim = h->hidden_dim;
    uint32_t inter_dim = h->intermediate_dim;
    uint32_t half_rot = model->rotary_dim / 2u;
    uint32_t kv_dim = model->kv_dim;
    uint32_t att_buf_dim = model->att_buf_dim;
    uint32_t kv_slot_d = (kv_dim > dim) ? kv_dim : dim;

    uint32_t max_cols = (inter_dim > att_buf_dim) ? inter_dim : att_buf_dim;
    size_t pad_act_cols = qmx_align_up(max_cols, QMX_TILE_COLS);
    uint32_t max_tiles_c = (uint32_t)(pad_act_cols / QMX_TILE_COLS);

    int max_b = (n_tokens < QMX_PREFILL_MAX_BATCH) ? n_tokens : QMX_PREFILL_MAX_BATCH;
    uint64_t per_tok64 = ((uint64_t)2u * dim + (uint64_t)2u * att_buf_dim + (uint64_t)2u * kv_slot_d +
                          (uint64_t)2u * inter_dim + (uint64_t)2u * half_rot) * (uint64_t)sizeof(float)
                       + (uint64_t)pad_act_cols * (uint64_t)(2u * sizeof(int8_t) + sizeof(int16_t))
                       + (uint64_t)max_tiles_c * (uint64_t)(sizeof(float) + sizeof(int32_t)) + 1280u;
    uint64_t need_ws64 = 0u;
    size_t need_ws = 0u;
    if (qmx_checked_mul_u64((uint64_t)max_b, per_tok64, &need_ws64) == 0 &&
        qmx_checked_add_u64(need_ws64, (uint64_t)QMX_CACHE_ALIGN, &need_ws64) == 0 &&
        need_ws64 <= (uint64_t)(~(size_t)0)) {
        need_ws = (size_t)need_ws64;
    }

    if (need_ws > 0u && need_ws > model->batch_ws_cap) {
        uint8_t *nws = (uint8_t *)realloc(model->batch_ws, need_ws);
        if (nws) {
            model->batch_ws = nws;
            model->batch_ws_cap = need_ws;
        }
    }

    if (need_ws == 0u || !model->batch_ws || model->batch_ws_cap < need_ws) {
        for (int i = 0; i < n_tokens; i++) {
            cur_pos = qmx_ensure_pos_capacity(model, cur_pos, 1, out_shifted);
            int want_logits = (i == n_tokens - 1) || (i == 0 && step0_chk_out != NULL);
            float *l = qmx_forward_internal(model, tokens[i], cur_pos, want_logits);
            cur_pos++;
            if (i == 0 && step0_chk_out != NULL && l != NULL) {
                *step0_chk_out = qmx_calc_step0_checksum(l, h->vocab_size);
            }
        }
        if (io_pos) *io_pos = cur_pos;
        return s->logits;
    }

    uintptr_t al_ws = ((uintptr_t)model->batch_ws + (QMX_CACHE_ALIGN - 1u)) & ~(uintptr_t)(QMX_CACHE_ALIGN - 1u);
    uint8_t *ws_ptr = (uint8_t *)al_ws;

    float   *b_slots[QMX_PREFILL_MAX_BATCH][8];
    int8_t  *b_xq[QMX_PREFILL_MAX_BATCH];
    int16_t *b_xq16[QMX_PREFILL_MAX_BATCH];
    int8_t  *b_xq_eo[QMX_PREFILL_MAX_BATCH];
    int32_t *b_xsums[QMX_PREFILL_MAX_BATCH];
    float   *b_xs[QMX_PREFILL_MAX_BATCH];
    float   *b_cos[QMX_PREFILL_MAX_BATCH];
    float   *b_sin[QMX_PREFILL_MAX_BATCH];

    const uint32_t slot_dims[8] = {
        dim, att_buf_dim, dim, att_buf_dim, kv_slot_d, kv_slot_d, inter_dim, inter_dim
    };

    size_t ws_off = 0;
    for (int b = 0; b < max_b; b++) {
        for (int sl = 0; sl < 8; sl++) {
            ws_off = qmx_align_up(ws_off, QMX_CACHE_ALIGN);
            b_slots[b][sl] = (float *)(ws_ptr + ws_off);
            ws_off += (size_t)slot_dims[sl] * sizeof(float);
        }
        ws_off = qmx_align_up(ws_off, QMX_CACHE_ALIGN);
        b_xq[b] = (int8_t *)(ws_ptr + ws_off);
        ws_off += pad_act_cols * sizeof(int8_t);

        ws_off = qmx_align_up(ws_off, QMX_CACHE_ALIGN);
        b_xq16[b] = (int16_t *)(ws_ptr + ws_off);
        ws_off += pad_act_cols * sizeof(int16_t);

        ws_off = qmx_align_up(ws_off, QMX_CACHE_ALIGN);
        b_xq_eo[b] = (int8_t *)(ws_ptr + ws_off);
        ws_off += pad_act_cols * sizeof(int8_t);

        ws_off = qmx_align_up(ws_off, QMX_CACHE_ALIGN);
        b_xsums[b] = (int32_t *)(ws_ptr + ws_off);
        ws_off += (size_t)max_tiles_c * sizeof(int32_t);

        ws_off = qmx_align_up(ws_off, QMX_CACHE_ALIGN);
        b_xs[b] = (float *)(ws_ptr + ws_off);
        ws_off += (size_t)max_tiles_c * sizeof(float);

        ws_off = qmx_align_up(ws_off, QMX_CACHE_ALIGN);
        b_cos[b] = (float *)(ws_ptr + ws_off);
        ws_off += (size_t)half_rot * sizeof(float);

        ws_off = qmx_align_up(ws_off, QMX_CACHE_ALIGN);
        b_sin[b] = (float *)(ws_ptr + ws_off);
        ws_off += (size_t)half_rot * sizeof(float);
    }

    int chunk_start = 0;
    while (chunk_start < n_tokens) {
        int B = n_tokens - chunk_start;
        if (B > max_b) B = max_b;
        cur_pos = qmx_ensure_pos_capacity(model, cur_pos, B, out_shifted);
        if (cur_pos + B > (int)h->max_seq_len) {
            B = (int)h->max_seq_len - cur_pos;
            if (B <= 0) break;
        }

        int base_pos = cur_pos;
        for (int b = 0; b < B; b++) {
            int tok = tokens[chunk_start + b];
            if (tok < 0 || (uint32_t)tok >= h->vocab_size) tok = 0;
            qmx_load_token_embedding(model, b_slots[b][QMX_BUF_X], tok);

            int p_b = base_pos + b;
            for (uint32_t idx = 0; idx < half_rot; idx++) {
                float val = (float)p_b * s->rope_inv_freq[idx];
                qmx_sincosf(val, &b_sin[b][idx], &b_cos[b][idx]);
            }
        }

        qmx_vm_exec_chunk(model, b_slots, b_xq, b_xq16, b_xq_eo, b_xsums, b_xs, b_cos, b_sin, B, base_pos);
        cur_pos += B;

        if (chunk_start == 0 && step0_chk_out != NULL) {
            compute_logits_for_hidden(model, b_slots[0][QMX_BUF_X]);
            *step0_chk_out = qmx_calc_step0_checksum(s->logits, h->vocab_size);
        }

        chunk_start += B;
        if (chunk_start == n_tokens) {
            if (!(chunk_start == B && B == 1 && step0_chk_out != NULL)) {
                compute_logits_for_hidden(model, b_slots[B - 1][QMX_BUF_X]);
            }
        }
    }
    if (io_pos) *io_pos = cur_pos;
#endif

    return s->logits;
}

float *qmx_forward_batch(QmxModel *model, const int *tokens, int n_tokens,
                         int start_pos, float *step0_chk_out) {
    int pos = start_pos;
    return qmx_forward_batch_ex(model, tokens, n_tokens, &pos, step0_chk_out, NULL);
}
