/*
 * Copyright 2026 1WhiteFox
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/qmx_runtime.h"
#include "../include/qmx_kernels.h"
#ifndef QMX_NO_MALLOC
#include <stdlib.h>
#endif
#include <string.h>

#define QMX_PREFILL_MAX_BATCH 16
#define QMX_GEMM_ROW_STRIP    4u
#define QMX_GQA_MAX_GROUP     8u

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

typedef struct {
    const QmxBackendOps *ops;
    float * const       *dst_batch;
    int8_t * const      *xq_batch;
    int16_t * const     *xq16_batch;
    float * const       *xscales_batch;
    int                  batch_size;
    const QmxTensor     *w;
    int                  accumulate;
    uint32_t             n_tr;
    uint32_t             n_tc;
    size_t               tile_bytes;
    size_t               sc_tile_bytes;
} QmxMatmulJob;

static void exec_matmul_slice_job(void *raw_ctx, int tid, int n_threads) {
    const QmxMatmulJob *job = (const QmxMatmulJob *)raw_ctx;
    uint32_t n_tr = job->n_tr;
    uint32_t n_th = (uint32_t)n_threads;
    uint32_t tr_start = ((uint32_t)tid * n_tr) / n_th;
    uint32_t tr_end   = (((uint32_t)tid + 1u) * n_tr) / n_th;
    if (tr_start >= tr_end) return;

    uint32_t step = (job->batch_size > 1) ? QMX_GEMM_ROW_STRIP : (tr_end - tr_start);
    if (step == 0u) step = 1u;

    for (uint32_t s_tr = tr_start; s_tr < tr_end; s_tr += step) {
        uint32_t strip_tr = tr_end - s_tr;
        if (strip_tr > step) strip_tr = step;

        uint32_t r_base = s_tr * QMX_TILE_ROWS;
        uint32_t max_rows = strip_tr * QMX_TILE_ROWS;
        uint32_t sub_rows = job->w->rows - r_base;
        if (sub_rows > max_rows) sub_rows = max_rows;

        uint32_t tile_off = s_tr * job->n_tc;
        QmxTensor sub_w;
        memset(&sub_w, 0, sizeof(sub_w));
        sub_w.data       = (const uint8_t *)job->w->data + (size_t)tile_off * job->tile_bytes;
        sub_w.scales     = (const float *)((const uint8_t *)job->w->scales + (size_t)tile_off * job->sc_tile_bytes);
        sub_w.rows       = sub_rows;
        sub_w.cols       = job->w->cols;
        sub_w.quant_type = job->w->quant_type;
        sub_w.tile_rows  = QMX_TILE_ROWS;

        for (int b = 0; b < job->batch_size; b++) {
            job->ops->matmul_tiled(job->dst_batch[b] + r_base,
                                   job->xq_batch[b],
                                   job->xq16_batch[b],
                                   job->xscales_batch[b],
                                   &sub_w,
                                   job->accumulate);
        }
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
    job.xscales_batch = xscales_batch;
    job.batch_size    = batch_size;
    job.w             = w;
    job.accumulate    = accumulate;
    job.n_tr          = n_tr;
    job.n_tc          = n_tc;
    job.tile_bytes    = qmx_tile_data_bytes(w->quant_type);
    job.sc_tile_bytes = qmx_scale_tile_bytes(w->quant_type);

    int n_th = model->num_threads;
    size_t work_elems = (size_t)w->rows * (size_t)w->cols * (size_t)batch_size;
    if (model->pool && n_th > 1 && n_tr >= 8u && work_elems >= 65536u) {
        int pool_n = qmx_pool_size(model->pool);
        if (n_th > pool_n) n_th = pool_n;
        if ((uint32_t)n_th > n_tr / 2u) n_th = (int)(n_tr / 2u);
        if (n_th > 1) {
            qmx_pool_run(model->pool, n_th, exec_matmul_slice_job, &job);
            return;
        }
    }

    if (batch_size > 1 && n_tr > QMX_GEMM_ROW_STRIP) {
        exec_matmul_slice_job(&job, 0, 1);
        return;
    }

    for (int b = 0; b < batch_size; b++) {
        ops->matmul_tiled(dst_batch[b], xq_batch[b], xq16_batch[b], xscales_batch[b], w, accumulate);
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
                                   float * const *xscales_batch,
                                   int batch_size,
                                   const QmxTensor *w,
                                   int32_t layer_idx,
                                   int accumulate) {
    if (w->data != NULL && w->scales != NULL) {
        dispatch_matmul_batch_parallel(model, dst_batch, xq_batch, xq16_batch, xscales_batch,
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
                        dj.mm_job.xscales_batch = xscales_batch;
                        dj.mm_job.batch_size    = batch_size;
                        dj.mm_job.w             = &sub_w;
                        dj.mm_job.accumulate    = accumulate;
                        dj.mm_job.n_tr          = chunk_tr;
                        dj.mm_job.n_tc          = n_tc;
                        dj.mm_job.tile_bytes    = tile_bytes;
                        dj.mm_job.sc_tile_bytes = sc_tile_bytes;
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
                        dispatch_matmul_batch_parallel(model, chunk_dst, xq_batch, xq16_batch, xscales_batch,
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
            dispatch_matmul_batch_parallel(model, chunk_dst, xq_batch, xq16_batch, xscales_batch,
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
            for (int b = 0; b < batch_size; b++) {
                ops->matmul_tiled(dst_batch[b] + r_base,
                                  xq_batch[b] + col_start,
                                  xq16_batch[b] + col_start,
                                  xscales_batch[b] + tc_start,
                                  &sub_w, sub_acc);
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
    float *xs_arr[1] = { s->x_scales };
    qmx_model_matmul_batch(model, dst_arr, xq_arr, xq16_arr, xs_arr, 1, w, layer_idx, accumulate);
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

typedef struct {
    QmxModel    *model;
    uint32_t     layer_idx;
    int          pos;
    int          t_start;
    uint32_t     att_len;
    const float *q_vec;
    float       *out_vec;
    float        inv_sqrt_head;
    float       *att_buf;
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
    int t_start = aj->t_start;
    int pos = aj->pos;
    uint32_t att_len = aj->att_len;
    float inv_sqrt_head = aj->inv_sqrt_head;
    uint32_t group_cap = aj->att_group_cap ? aj->att_group_cap : 1u;
    if (group_cap > QMX_GQA_MAX_GROUP) group_cap = QMX_GQA_MAX_GROUP;

    uint32_t hd_start = ((uint32_t)tid * n_heads) / (uint32_t)n_threads;
    uint32_t hd_end   = (((uint32_t)tid + 1u) * n_heads) / (uint32_t)n_threads;
    if (hd_start >= hd_end) return;

    float *att_tid = aj->att_buf + (size_t)tid * (size_t)group_cap * (size_t)h->max_seq_len;

    if (model->kv_type == QMX_KV_Q8) {
        uint32_t q_stride = (uint32_t)qmx_align_up(head_size, QMX_TILE_COLS);
        uint32_t n_groups = q_stride / QMX_TILE_COLS;
        uint32_t g_lim = (q_stride <= 128u) ? 4u : 1u;
        if (g_lim > group_cap) g_lim = group_cap;

        int8_t  q_xq[512] QMX_ALIGNED(64);
        int16_t q_xq16[512] QMX_ALIGNED(64);
        float   q_xs[64];
        float   v_f32[512];

        for (uint32_t hd = hd_start; hd < hd_end; ) {
            uint32_t kv_h = hd / kv_mul;
            uint32_t kv_head_off = kv_h * head_size;
            uint32_t g_cnt = 1u;
            while (hd + g_cnt < hd_end && g_cnt < g_lim && ((hd + g_cnt) / kv_mul) == kv_h) {
                g_cnt++;
            }

            float *atts[4];
            float *xbs[4];
            for (uint32_t g = 0; g < g_cnt; g++) {
                atts[g] = att_tid + (size_t)g * h->max_seq_len;
                xbs[g]  = aj->out_vec + (hd + g) * head_size;
                memset(xbs[g], 0, (size_t)head_size * sizeof(float));
                ops->prepare_act(aj->q_vec + (hd + g) * head_size,
                                 q_xq + g * q_stride,
                                 q_xq16 + g * q_stride,
                                 q_xs + g * n_groups,
                                 head_size);
            }

            for (int t = t_start; t <= pos; t++) {
                const int8_t *k8 = s->key_cache_q8 + loff + (size_t)t * kv_dim + kv_head_off;
                float k_sc = s->key_scales[((size_t)aj->layer_idx * h->max_seq_len + (size_t)t) * n_kv_heads + kv_h] * inv_sqrt_head;
                int tidx = t - t_start;
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
                    atts[g][tidx] = dot_f * k_sc;
                }
            }
            for (uint32_t g = 0; g < g_cnt; g++) {
                if (attn_cap > 0.0f) apply_tanh_softcap(atts[g], att_len, attn_cap);
                qmx_softmax_scalar(atts[g], att_len);
            }

            for (int t = t_start; t <= pos; t++) {
                const int8_t *v8 = s->value_cache_q8 + loff + (size_t)t * kv_dim + kv_head_off;
                float v_sc = s->value_scales[((size_t)aj->layer_idx * h->max_seq_len + (size_t)t) * n_kv_heads + kv_h];
                for (uint32_t i = 0; i < head_size; i++) v_f32[i] = (float)v8[i];
                int tidx = t - t_start;
                for (uint32_t g = 0; g < g_cnt; g++) {
                    ops->saxpy_f32(xbs[g], atts[g][tidx] * v_sc, v_f32, head_size);
                }
            }
            hd += g_cnt;
        }
    } else if (model->kv_type == QMX_KV_FP16) {
        void (*cvt_fn)(float * QMX_RESTRICT, const uint16_t * QMX_RESTRICT, uint32_t) =
            ops->f16_to_f32 ? ops->f16_to_f32 : qmx_f16_to_f32_scalar;
        float kv_f32[512] QMX_ALIGNED(64);

        for (uint32_t hd = hd_start; hd < hd_end; ) {
            uint32_t kv_h = hd / kv_mul;
            uint32_t kv_head_off = kv_h * head_size;
            uint32_t g_cnt = 1u;
            while (hd + g_cnt < hd_end && g_cnt < group_cap && ((hd + g_cnt) / kv_mul) == kv_h) {
                g_cnt++;
            }

            float *atts[QMX_GQA_MAX_GROUP];
            const float *qhs[QMX_GQA_MAX_GROUP];
            float *xbs[QMX_GQA_MAX_GROUP];
            for (uint32_t g = 0; g < g_cnt; g++) {
                atts[g] = att_tid + (size_t)g * h->max_seq_len;
                qhs[g]  = aj->q_vec + (hd + g) * head_size;
                xbs[g]  = aj->out_vec + (hd + g) * head_size;
                memset(xbs[g], 0, (size_t)head_size * sizeof(float));
            }

            for (int t = t_start; t <= pos; t++) {
                const uint16_t *kh16 = s->key_cache_f16 + loff + (size_t)t * kv_dim + kv_head_off;
                cvt_fn(kv_f32, kh16, head_size);
                int tidx = t - t_start;
                for (uint32_t g = 0; g < g_cnt; g++) {
                    atts[g][tidx] = ops->dot_f32(qhs[g], kv_f32, head_size) * inv_sqrt_head;
                }
            }
            for (uint32_t g = 0; g < g_cnt; g++) {
                if (attn_cap > 0.0f) apply_tanh_softcap(atts[g], att_len, attn_cap);
                qmx_softmax_scalar(atts[g], att_len);
            }

            for (int t = t_start; t <= pos; t++) {
                const uint16_t *vh16 = s->value_cache_f16 + loff + (size_t)t * kv_dim + kv_head_off;
                cvt_fn(kv_f32, vh16, head_size);
                int tidx = t - t_start;
                for (uint32_t g = 0; g < g_cnt; g++) {
                    ops->saxpy_f32(xbs[g], atts[g][tidx], kv_f32, head_size);
                }
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

            float *atts[QMX_GQA_MAX_GROUP];
            const float *qhs[QMX_GQA_MAX_GROUP];
            float *xbs[QMX_GQA_MAX_GROUP];
            for (uint32_t g = 0; g < g_cnt; g++) {
                atts[g] = att_tid + (size_t)g * h->max_seq_len;
                qhs[g]  = aj->q_vec + (hd + g) * head_size;
                xbs[g]  = aj->out_vec + (hd + g) * head_size;
                memset(xbs[g], 0, (size_t)head_size * sizeof(float));
            }

            for (int t = t_start; t <= pos; t++) {
                const float *kh = s->key_cache + loff + (size_t)t * kv_dim + kv_head_off;
                int tidx = t - t_start;
                for (uint32_t g = 0; g < g_cnt; g++) {
                    atts[g][tidx] = ops->dot_f32(qhs[g], kh, head_size) * inv_sqrt_head;
                }
            }
            for (uint32_t g = 0; g < g_cnt; g++) {
                if (attn_cap > 0.0f) apply_tanh_softcap(atts[g], att_len, attn_cap);
                qmx_softmax_scalar(atts[g], att_len);
            }

            for (int t = t_start; t <= pos; t++) {
                const float *vh = s->value_cache + loff + (size_t)t * kv_dim + kv_head_off;
                int tidx = t - t_start;
                for (uint32_t g = 0; g < g_cnt; g++) {
                    ops->saxpy_f32(xbs[g], atts[g][tidx], vh, head_size);
                }
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

    QmxAttnJob aj;
    aj.model         = model;
    aj.layer_idx     = layer_idx;
    aj.pos           = pos;
    aj.t_start       = t_start;
    aj.att_len       = att_len;
    aj.q_vec         = q_vec;
    aj.out_vec       = out_vec;
    aj.inv_sqrt_head = inv_sqrt_head;
    aj.att_buf       = s->att;
    aj.att_group_cap = 1u;

#ifndef QMX_NO_MALLOC
    uint32_t kv_mul = n_heads / n_kv_heads;
    uint32_t desired_g = (kv_mul > QMX_GQA_MAX_GROUP) ? QMX_GQA_MAX_GROUP : kv_mul;
    int active = 1;
    int n_th = model->num_threads;
    if (!model->arena.is_external && model->pool && n_th > 1 && n_heads >= 4u && att_len >= 64u) {
        active = qmx_pool_size(model->pool);
        if (active > n_th) active = n_th;
        if ((uint32_t)active > n_heads) active = (int)n_heads;
        if (active < 1) active = 1;
    }
    if (!model->arena.is_external && (active > 1 || desired_g > 1u)) {
        size_t need = (size_t)active * (size_t)desired_g * (size_t)h->max_seq_len;
        if (need > model->mt_att_cap) {
            float *nb = (float *)realloc(model->mt_att_buf, need * sizeof(float));
            if (nb) {
                model->mt_att_buf = nb;
                model->mt_att_cap = need;
            }
        }
        if (model->mt_att_buf && model->mt_att_cap >= need) {
            aj.att_buf       = model->mt_att_buf;
            aj.att_group_cap = desired_g;
            if (active > 1) {
                qmx_pool_run(model->pool, active, exec_attn_heads_job, &aj);
                return;
            }
        }
    }
#else
    (void)n_heads;
#endif

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

    for (uint32_t l = 0; l < n_layers; l++) {
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
                    }
                    break;
                }

                case QMX_OP_MATMUL:
                case QMX_OP_MATMUL_ADD:
                    if (w) {
                        float *dst_ptrs[QMX_PREFILL_MAX_BATCH];
                        for (int b = 0; b < B; b++) {
                            dst_ptrs[b] = vm_resolve_slot(model, b_slots, b, db, l, base_pos + b);
                        }
                        qmx_model_matmul_batch(model, dst_ptrs, b_xq, b_xq16, b_xs, B, w, (int32_t)l,
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
                            }
                        } else {
                            for (int b = 0; b < B; b++) {
                                qmx_swiglu_quant_scalar(b_slots[b][sb], b_slots[b][db],
                                                        b_xq[b], need16 ? b_xq16[b] : NULL, b_xs[b], inter_dim);
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
    }
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
    int8_t  *b_xq[1]   = { s->xq };
    int16_t *b_xq16[1] = { s->xq16 };
    float   *b_xs[1]   = { s->x_scales };
    float   *b_cos[1]  = { s->rope_cos };
    float   *b_sin[1]  = { s->rope_sin };
    for (int sl = 0; sl < 8; sl++) b_slots[0][sl] = s->slots[sl];

    qmx_vm_exec_chunk(model, b_slots, b_xq, b_xq16, b_xs, b_cos, b_sin, 1, pos);

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
    size_t per_tok_bytes = (size_t)(2u * dim + 2u * att_buf_dim + 2u * kv_slot_d + 2u * inter_dim + 2u * half_rot) * sizeof(float)
                         + pad_act_cols * (sizeof(int8_t) + sizeof(int16_t))
                         + (size_t)max_tiles_c * sizeof(float) + 1024u;
    size_t need_ws = (size_t)max_b * per_tok_bytes + QMX_CACHE_ALIGN;

    if (need_ws > model->batch_ws_cap) {
        uint8_t *nws = (uint8_t *)realloc(model->batch_ws, need_ws);
        if (nws) {
            model->batch_ws = nws;
            model->batch_ws_cap = need_ws;
        }
    }

    if (!model->batch_ws || model->batch_ws_cap < need_ws) {
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

        qmx_vm_exec_chunk(model, b_slots, b_xq, b_xq16, b_xs, b_cos, b_sin, B, base_pos);
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
