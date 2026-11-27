/*
 * Copyright 2026 1WhiteFox
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/qmx_kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int g_failures = 0;

static void check_near(const char *tag, float got, float ref, float tol) {
    float diff = fabsf(got - ref);
    if (diff > tol || isnan(got) || isinf(got)) {
        fprintf(stderr, "[FAIL] %s: got=%.6f ref=%.6f diff=%.6f (tol=%.6f)\n",
                tag, (double)got, (double)ref, (double)diff, (double)tol);
        g_failures++;
    }
}

static uint16_t simd_fp32_to_fp16(float f) {
    QmxFloatBits fb;
    fb.f = f;
    uint32_t sign = (fb.u >> 16u) & 0x8000u;
    int32_t exp = (int32_t)((fb.u >> 23u) & 0xFFu) - 127 + 15;
    uint32_t mant = fb.u & 0x007FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant = (mant | 0x00800000u) >> (uint32_t)(1 - exp);
        return (uint16_t)(sign | ((mant + 0x00001000u) >> 13u));
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
    uint32_t rounded = ((uint32_t)exp << 10u) + ((mant + 0x00001000u) >> 13u);
    if (rounded >= 0x7C00u) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | rounded);
}

static void pack_random_matrix(const float *mat, uint32_t rows, uint32_t cols,
                               uint8_t qtype, float **out_scales, uint8_t **out_data) {
    uint32_t pad_r = (uint32_t)qmx_align_up(rows, QMX_TILE_ROWS);
    uint32_t pad_c = (uint32_t)qmx_align_up(cols, QMX_TILE_COLS);
    uint32_t n_tr = pad_r / QMX_TILE_ROWS;
    uint32_t n_tc = pad_c / QMX_TILE_COLS;
    uint32_t num_tiles = n_tr * n_tc;
    size_t tile_bytes = qmx_tile_data_bytes(qtype);
    size_t row_bytes = qmx_tile_row_bytes(qtype);
    int is_row_scaled = (qtype == QMX_QUANT_Q4_ROW32 || qtype == QMX_QUANT_Q8_ROW32);

    *out_scales = (float *)calloc(is_row_scaled ? ((size_t)num_tiles * 8u) : (size_t)num_tiles, sizeof(float));
    *out_data = (uint8_t *)calloc(num_tiles, tile_bytes);

    for (uint32_t tr = 0; tr < n_tr; tr++) {
        for (uint32_t tc = 0; tc < n_tc; tc++) {
            uint32_t tidx = tr * n_tc + tc;
            uint8_t *tptr = *out_data + (size_t)tidx * tile_bytes;

            if (is_row_scaled) {
                uint16_t *sc16 = (uint16_t *)(void *)(*out_scales) + (size_t)tidx * QMX_TILE_ROWS;
                for (uint32_t lr = 0; lr < QMX_TILE_ROWS; lr++) {
                    float rmax = 0.0f;
                    uint32_t r = tr * QMX_TILE_ROWS + lr;
                    for (uint32_t lc = 0; lc < QMX_TILE_COLS; lc++) {
                        uint32_t c = tc * QMX_TILE_COLS + lc;
                        float v = (r < rows && c < cols) ? mat[r * cols + c] : 0.0f;
                        float av = fabsf(v);
                        if (av > rmax) rmax = av;
                    }
                    float sc = (qtype == QMX_QUANT_Q8_ROW32) ? (rmax / 127.0f) : (rmax / 7.0f);
                    uint16_t h16 = simd_fp32_to_fp16(sc);
                    sc16[lr] = h16;
                    float eff_sc = qmx_fp16_to_fp32(h16);
                    float inv_sc = (eff_sc == 0.0f) ? 0.0f : (1.0f / eff_sc);
                    uint8_t *rptr = tptr + (size_t)lr * row_bytes;
                    for (uint32_t lc = 0; lc < QMX_TILE_COLS; lc++) {
                        uint32_t c = tc * QMX_TILE_COLS + lc;
                        float v = (r < rows && c < cols) ? mat[r * cols + c] : 0.0f;
                        if (qtype == QMX_QUANT_Q4_ROW32) {
                            int32_t q = qmx_round_nearest_i32(v * inv_sc);
                            if (q > 7) q = 7;
                            if (q < -8) q = -8;
                            uint8_t u = (uint8_t)(q + 8);
                            if ((lc & 1u) == 0u) rptr[lc / 2u] |= u;
                            else rptr[lc / 2u] |= (uint8_t)(u << 4u);
                        } else {
                            int32_t q = qmx_round_nearest_i32(v * inv_sc);
                            if (q > 127) q = 127;
                            if (q < -127) q = -127;
                            ((int8_t *)rptr)[lc] = (int8_t)q;
                        }
                    }
                }
                continue;
            }

            float max_abs = 0.0f;
            float sum_abs = 0.0f;
            for (uint32_t lr = 0; lr < QMX_TILE_ROWS; lr++) {
                for (uint32_t lc = 0; lc < QMX_TILE_COLS; lc++) {
                    uint32_t r = tr * QMX_TILE_ROWS + lr;
                    uint32_t c = tc * QMX_TILE_COLS + lc;
                    float v = (r < rows && c < cols) ? mat[r * cols + c] : 0.0f;
                    float av = fabsf(v);
                    if (av > max_abs) max_abs = av;
                    sum_abs += av;
                }
            }
            float sc = (qtype == QMX_QUANT_Q8_SYM) ? (max_abs / 127.0f)
                     : (qtype == QMX_QUANT_Q4_SYM) ? (max_abs / 7.0f)
                     : (sum_abs / 512.0f + 1e-8f);
            (*out_scales)[tidx] = sc;
            float inv_sc = (sc == 0.0f) ? 0.0f : (1.0f / sc);

            for (uint32_t lr = 0; lr < QMX_TILE_ROWS; lr++) {
                uint8_t *rptr = tptr + (size_t)lr * row_bytes;
                for (uint32_t lc = 0; lc < QMX_TILE_COLS; lc++) {
                    uint32_t r = tr * QMX_TILE_ROWS + lr;
                    uint32_t c = tc * QMX_TILE_COLS + lc;
                    float v = (r < rows && c < cols) ? mat[r * cols + c] : 0.0f;
                    if (qtype == QMX_QUANT_BINARY_1BIT) {
                        if (v >= 0.0f) rptr[lc / 8u] |= (uint8_t)(1u << (lc % 8u));
                    } else if (qtype == QMX_QUANT_TERNARY_158) {
                        int32_t q = qmx_round_nearest_i32(v * inv_sc);
                        uint8_t code = (q >= 1) ? 1u : ((q <= -1) ? 2u : 0u);
                        rptr[lc / 4u] |= (uint8_t)(code << ((lc % 4u) * 2u));
                    } else if (qtype == QMX_QUANT_Q4_SYM) {
                        int32_t q = qmx_round_nearest_i32(v * inv_sc);
                        if (q > 7) q = 7;
                        if (q < -8) q = -8;
                        uint8_t u = (uint8_t)(q + 8);
                        if ((lc & 1u) == 0u) rptr[lc / 2u] |= u;
                        else rptr[lc / 2u] |= (uint8_t)(u << 4u);
                    } else if (qtype == QMX_QUANT_Q8_SYM) {
                        int32_t q = qmx_round_nearest_i32(v * inv_sc);
                        if (q > 127) q = 127;
                        if (q < -127) q = -127;
                        ((int8_t *)rptr)[lc] = (int8_t)q;
                    }
                }
            }
        }
    }
}

static void verify_backend_ops(QmxBackend backend, const QmxBackendOps *ops, const char *model_path) {
    const uint32_t rows = 19;
    const uint32_t cols = 45;
    float mat[19 * 45];
    float x[64];
    float w_norm[64];
    int8_t k8[64];

    uint32_t seed = 2026u;
    for (uint32_t i = 0; i < rows * cols; i++) {
        seed = seed * 1664525u + 1013904223u;
        mat[i] = ((float)((int32_t)(seed >> 8) - 8388608)) / 8388608.0f;
    }
    for (uint32_t i = 0; i < 64u; i++) {
        seed = seed * 1664525u + 1013904223u;
        x[i] = ((float)((int32_t)(seed >> 8) - 8388608)) / 8388608.0f;
        seed = seed * 1664525u + 1013904223u;
        w_norm[i] = 0.5f + ((float)(seed >> 16)) / 65536.0f;
        k8[i] = (int8_t)((int32_t)(seed % 200u) - 100);
    }

    float rms_ref[45], rms_got[45];
    qmx_ops_scalar.rmsnorm(rms_ref, x, w_norm, cols, 1e-5f);
    ops->rmsnorm(rms_got, x, w_norm, cols, 1e-5f);
    for (uint32_t i = 0; i < cols; i++) {
        check_near(ops->name, rms_got[i], rms_ref[i], 1e-4f);
    }

    float dot_ref = qmx_ops_scalar.dot_f32(x, w_norm, cols);
    float dot_got = ops->dot_f32(x, w_norm, cols);
    check_near(ops->name, dot_got, dot_ref, 1e-4f);

    float y_ref[45], y_got[45];
    memcpy(y_ref, w_norm, cols * sizeof(float));
    memcpy(y_got, w_norm, cols * sizeof(float));
    qmx_ops_scalar.saxpy_f32(y_ref, 0.375f, x, cols);
    ops->saxpy_f32(y_got, 0.375f, x, cols);
    for (uint32_t i = 0; i < cols; i++) {
        check_near(ops->name, y_got[i], y_ref[i], 1e-4f);
    }

    QMX_ALIGNED(64) int8_t xq_ref[64];
    QMX_ALIGNED(64) int16_t xq16_ref[64];
    float xs_ref[2];
    qmx_ops_scalar.prepare_act(x, xq_ref, xq16_ref, xs_ref, cols);

    QMX_ALIGNED(64) int8_t xq_got[64];
    QMX_ALIGNED(64) int16_t xq16_got[64];
    QMX_ALIGNED(64) int8_t xq_eo_got[64];
    int32_t xsums_got[2];
    float xs_got[2];
    ops->prepare_act(x, xq_got, xq16_got, xs_got, cols);
    if (ops->prepare_act_eo) {
        ops->prepare_act_eo(xq_got, xq_eo_got, xsums_got, cols);
    }

    int32_t dq8_ref = qmx_ops_scalar.dot_q8(xq_ref, xq16_ref, k8, cols);
    int32_t dq8_got = ops->dot_q8(xq_got, xq16_got, k8, cols);
    if (dq8_ref != dq8_got) {
        fprintf(stderr, "[FAIL] %s dot_q8: got=%d ref=%d\n", ops->name, dq8_got, dq8_ref);
        g_failures++;
    }

    if (ops->f16_to_f32 && ops->f32_to_f16) {
        uint16_t h_ref[64], h_got[64];
        float f_ref[64], f_got[64];
        qmx_f32_to_f16_scalar(h_ref, x, cols);
        ops->f32_to_f16(h_got, x, cols);
        qmx_f16_to_f32_scalar(f_ref, h_ref, cols);
        ops->f16_to_f32(f_got, h_got, cols);
        for (uint32_t i = 0; i < cols; i++) {
            check_near(ops->name, f_got[i], f_ref[i], 1e-3f);
        }
        if (ops->dot_f32_f16) {
            float df16_ref = qmx_dot_f32_f16_scalar(w_norm, h_ref, cols);
            float df16_got = ops->dot_f32_f16(w_norm, h_ref, cols);
            check_near(ops->name, df16_got, df16_ref, 1e-3f);
        }
        if (ops->saxpy_f32_f16) {
            float sy_ref[45], sy_got[45];
            memcpy(sy_ref, w_norm, cols * sizeof(float));
            memcpy(sy_got, w_norm, cols * sizeof(float));
            qmx_saxpy_f32_f16_scalar(sy_ref, 0.375f, h_ref, cols);
            ops->saxpy_f32_f16(sy_got, 0.375f, h_ref, cols);
            for (uint32_t i = 0; i < cols; i++) {
                check_near(ops->name, sy_got[i], sy_ref[i], 1e-3f);
            }
        }
    }

    if (ops->rmsnorm_quant) {
        QMX_ALIGNED(64) int8_t  rq_ref[64];
        QMX_ALIGNED(64) int8_t  rq_got[64];
        QMX_ALIGNED(64) int16_t rq16_ref[64];
        QMX_ALIGNED(64) int16_t rq16_got[64];
        float rqs_ref[2], rqs_got[2];
        qmx_ops_scalar.prepare_act(rms_ref, rq_ref, rq16_ref, rqs_ref, cols);
        ops->rmsnorm_quant(x, w_norm, rq_got, rq16_got, rqs_got, cols, 1e-5f);
        for (uint32_t i = 0; i < cols; i++) {
            float vf = (float)rq_ref[i] * rqs_ref[i / QMX_TILE_COLS];
            float vg = (float)rq_got[i] * rqs_got[i / QMX_TILE_COLS];
            check_near(ops->name, vg, vf, 0.03f);
        }
    }

    if (ops->swiglu_quant) {
        float sw_ref[45];
        for (uint32_t i = 0; i < cols; i++) {
            float gv = x[i];
            sw_ref[i] = gv * (1.0f / (1.0f + qmx_expf(-gv))) * w_norm[i];
        }
        QMX_ALIGNED(64) int8_t  sq_ref[64];
        QMX_ALIGNED(64) int8_t  sq_got[64];
        QMX_ALIGNED(64) int16_t sq16_ref[64];
        QMX_ALIGNED(64) int16_t sq16_got[64];
        float sqs_ref[2], sqs_got[2];
        qmx_ops_scalar.prepare_act(sw_ref, sq_ref, sq16_ref, sqs_ref, cols);
        ops->swiglu_quant(x, w_norm, sq_got, sq16_got, sqs_got, cols);
        for (uint32_t i = 0; i < cols; i++) {
            float vf = (float)sq_ref[i] * sqs_ref[i / QMX_TILE_COLS];
            float vg = (float)sq_got[i] * sqs_got[i / QMX_TILE_COLS];
            check_near(ops->name, vg, vf, 0.03f);
        }
    }

    uint8_t qtypes[6] = {
        QMX_QUANT_Q8_SYM,
        QMX_QUANT_Q4_SYM,
        QMX_QUANT_TERNARY_158,
        QMX_QUANT_BINARY_1BIT,
        QMX_QUANT_Q4_ROW32,
        QMX_QUANT_Q8_ROW32
    };

    for (int t = 0; t < 6; t++) {
        float *scales = NULL;
        uint8_t *data = NULL;
        pack_random_matrix(mat, rows, cols, qtypes[t], &scales, &data);

        QmxTensor w;
        memset(&w, 0, sizeof(w));
        w.data = data;
        w.scales = scales;
        w.rows = rows;
        w.cols = cols;
        w.quant_type = qtypes[t];
        w.tile_rows = QMX_TILE_ROWS;

        float out_ref[19], out_got[19];
        for (uint32_t r = 0; r < rows; r++) {
            out_ref[r] = 1.25f;
            out_got[r] = 1.25f;
        }
        qmx_ops_scalar.matmul_tiled(out_ref, xq_ref, xq16_ref, xs_ref, &w, 1);
        ops->matmul_tiled(out_got, xq_got, xq16_got, xs_got, &w, 1);

        for (uint32_t r = 0; r < rows; r++) {
            check_near(ops->name, out_got[r], out_ref[r], 1e-4f);
        }

        if (ops->prepare_act_eo && ops->matmul_tiled_eo) {
            float out_eo[19];
            for (uint32_t r = 0; r < rows; r++) out_eo[r] = 1.25f;
            ops->matmul_tiled_eo(out_eo, xq_got, xq_eo_got, xsums_got, xs_got, &w, 1);
            for (uint32_t r = 0; r < rows; r++) {
                check_near(ops->name, out_eo[r], out_ref[r], 1e-4f);
            }
        }

        if (ops->prepare_act_eo && ops->matmul_tiled_batch) {
            enum { TB = 13 };
            QMX_ALIGNED(64) int8_t  bxq[TB][64];
            QMX_ALIGNED(64) int16_t bxq16[TB][64];
            QMX_ALIGNED(64) int8_t  bxq_eo[TB][64];
            int32_t bxsums[TB][2];
            float   bxs[TB][2];
            float   bout_ref[TB][19];
            float   bout_got[TB][19];
            float   *dst_p[TB];
            int8_t  *xq_p[TB];
            int8_t  *eo_p[TB];
            int32_t *sum_p[TB];
            float   *xs_p[TB];

            for (int b = 0; b < TB; b++) {
                float xb[64];
                for (uint32_t i = 0; i < 64u; i++) {
                    seed = seed * 1664525u + 1013904223u;
                    xb[i] = ((float)((int32_t)(seed >> 8) - 8388608)) / 8388608.0f;
                }
                ops->prepare_act(xb, bxq[b], bxq16[b], bxs[b], cols);
                ops->prepare_act_eo(bxq[b], bxq_eo[b], bxsums[b], cols);
                dst_p[b] = bout_got[b];
                xq_p[b]  = bxq[b];
                eo_p[b]  = bxq_eo[b];
                sum_p[b] = bxsums[b];
                xs_p[b]  = bxs[b];
            }

            const int test_bs[7] = { 1, 2, 4, 5, 8, 9, 13 };
            for (int acc_m = 0; acc_m < 2; acc_m++) {
                for (int bi = 0; bi < 7; bi++) {
                    int B = test_bs[bi];
                    for (int b = 0; b < B; b++) {
                        for (uint32_t r = 0; r < rows; r++) {
                            float init_v = acc_m ? (0.25f * (float)(b + 1) - 0.1f * (float)r) : 99.0f;
                            bout_ref[b][r] = init_v;
                            bout_got[b][r] = init_v;
                        }
                        qmx_ops_scalar.matmul_tiled(bout_ref[b], bxq[b], bxq16[b], bxs[b], &w, acc_m);
                    }
                    ops->matmul_tiled_batch(dst_p, xq_p, eo_p, sum_p, xs_p, B, &w, acc_m);
                    for (int b = 0; b < B; b++) {
                        for (uint32_t r = 0; r < rows; r++) {
                            check_near(ops->name, bout_got[b][r], bout_ref[b][r], 1e-4f);
                        }
                    }
                }
            }
        }

        free(scales);
        free(data);
    }

    if (model_path) {
        QmxModel m_ref, m_got;
        if (qmx_load(&m_ref, model_path, 0, QMX_BACKEND_SCALAR, 0, 0, 0) == 0 &&
            qmx_load(&m_got, model_path, 0, backend, 0, 0, 0) == 0) {
            int prompt[3] = { 1, 15, 77 };
            float *l_ref = NULL;
            float *l_got = NULL;
            for (int p = 0; p < 3; p++) {
                l_ref = qmx_forward(&m_ref, prompt[p], p);
                l_got = qmx_forward(&m_got, prompt[p], p);
            }
            for (uint32_t i = 0; i < m_ref.header.vocab_size; i++) {
                check_near(ops->name, l_got[i], l_ref[i], 1e-3f);
            }
            qmx_free(&m_ref);
            qmx_free(&m_got);
        }
    }
}

int main(int argc, char **argv) {
    const char *model_path = (argc > 1) ? argv[1] : "dummy.qmx";
    QmxBackend detected = qmx_detect_backend();
    printf("[Info] Auto-detected best backend: %s\n", qmx_backend_name(detected));


    int tested = 0;
    for (int b = 1; b < (int)QMX_BACKEND_COUNT; b++) {
        QmxModel dummy_m;
        memset(&dummy_m, 0, sizeof(dummy_m));
        qmx_bind_backend_ops(&dummy_m, (QmxBackend)b);
        if (dummy_m.backend == (QmxBackend)b && dummy_m.ops != &qmx_ops_scalar) {
            printf("  -> Verifying backend: %s ...\n", dummy_m.ops->name);
            verify_backend_ops((QmxBackend)b, dummy_m.ops, model_path);
            tested++;
        }
    }

    if (tested == 0) {
        fprintf(stderr, "[FAIL] No x86 SIMD backends were registered!\n");
        return 1;
    }

    if (g_failures == 0) {
        printf("[PASS] Stage 3.2 x86 SIMD verification passed across %d backend(s)!\n", tested);
        return 0;
    }
    fprintf(stderr, "[FAIL] Total SIMD assertion failures: %d\n", g_failures);
    return 1;
}