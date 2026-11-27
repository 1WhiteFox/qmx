/*
 * Copyright 2026 1WhiteFox
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/qmx_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int g_failures = 0;

static void check_near(const char *name, float got, float expected, float tol) {
    float diff = fabsf(got - expected);
    if (diff > tol || isnan(got) || isinf(got)) {
        fprintf(stderr, "[FAIL] %s: got=%.6f expected=%.6f diff=%.6f (tol=%.6f)\n",
                name, (double)got, (double)expected, (double)diff, (double)tol);
        g_failures++;
    }
}

static void test_crc32(void) {
    const char *vec = "123456789";
    uint32_t crc = qmx_crc32(vec, 9);
    if (crc != 0xCBF43926u) {
        fprintf(stderr, "[FAIL] crc32: got=0x%08X expected=0xCBF43926\n", crc);
        g_failures++;
    }
    uint32_t sc = 0xFFFFFFFFu;
    sc = qmx_crc32_update(sc, vec, 4);
    sc = qmx_crc32_update(sc, vec + 4, 5);
    if (~sc != 0xCBF43926u) {
        fprintf(stderr, "[FAIL] crc32_update: got=0x%08X expected=0xCBF43926\n", ~sc);
        g_failures++;
    }
}

static void test_rmsnorm_and_softmax(void) {
    float x[4] = { 1.0f, -2.0f, 3.0f, -4.0f };
    float w[4] = { 0.5f, 1.0f, 1.5f, 2.0f };
    float out[4] = { 0 };
    qmx_rmsnorm_scalar(out, x, w, 4, 1e-5f);
    float rms = sqrtf((1.0f + 4.0f + 9.0f + 16.0f) / 4.0f + 1e-5f);
    for (int i = 0; i < 4; i++) {
        check_near("rmsnorm", out[i], (x[i] / rms) * w[i], 1e-5f);
    }

    float s[4] = { 2.0f, 1.0f, 0.0f, -1.0f };
    qmx_softmax_scalar(s, 4);
    float sum = s[0] + s[1] + s[2] + s[3];
    check_near("softmax_sum", sum, 1.0f, 1e-5f);
    float denom = expf(2.0f) + expf(1.0f) + expf(0.0f) + expf(-1.0f);
    check_near("softmax_0", s[0], expf(2.0f) / denom, 1e-5f);
}

static void test_rope_modes(void) {
    float q_inter[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    float k_inter[4] = { 0.5f, -0.5f, 1.5f, -1.5f };
    float fcr[2] = { 0.8f, 0.6f };
    float fci[2] = { 0.6f, 0.8f };
    qmx_rope_apply(q_inter, k_inter, fcr, fci, 4, 4, 4, 0);
    check_near("rope_inter_q0", q_inter[0], 1.0f * 0.8f - 2.0f * 0.6f, 1e-5f);
    check_near("rope_inter_q1", q_inter[1], 1.0f * 0.6f + 2.0f * 0.8f, 1e-5f);
    check_near("rope_inter_q2", q_inter[2], 3.0f * 0.6f - 4.0f * 0.8f, 1e-5f);
    check_near("rope_inter_q3", q_inter[3], 3.0f * 0.8f + 4.0f * 0.6f, 1e-5f);

    float q_split[4] = { 1.0f, 3.0f, 2.0f, 4.0f };
    qmx_rope_apply(q_split, NULL, fcr, fci, 4, 0, 4, 1);
    check_near("rope_split_q0", q_split[0], 1.0f * 0.8f - 2.0f * 0.6f, 1e-5f);
    check_near("rope_split_q2", q_split[2], 1.0f * 0.6f + 2.0f * 0.8f, 1e-5f);
    check_near("rope_split_q1", q_split[1], 3.0f * 0.6f - 4.0f * 0.8f, 1e-5f);
    check_near("rope_split_q3", q_split[3], 3.0f * 0.8f + 4.0f * 0.6f, 1e-5f);
}

static uint16_t test_fp32_to_fp16(float f) {
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

static void pack_test_matrix(const float *mat, uint32_t rows, uint32_t cols,
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
                    uint16_t h16 = test_fp32_to_fp16(sc);
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
            float sc = 0.0f;
            if (qtype == QMX_QUANT_BINARY_1BIT || qtype == QMX_QUANT_TERNARY_158) {
                sc = sum_abs / 512.0f;
                if (sc < 1e-8f) sc = 1e-8f;
            } else if (qtype == QMX_QUANT_Q4_SYM) {
                sc = max_abs / 7.0f;
            } else if (qtype == QMX_QUANT_Q8_SYM) {
                sc = max_abs / 127.0f;
            }
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

static void test_tiled_matmul_and_row_dequant(void) {
    const uint32_t rows = 19;
    const uint32_t cols = 45;
    float mat[19 * 45];
    float x[45];

    uint32_t seed = 1337u;
    for (uint32_t i = 0; i < rows * cols; i++) {
        seed = seed * 1664525u + 1013904223u;
        mat[i] = ((float)((int32_t)(seed >> 8) - 8388608)) / 8388608.0f;
    }
    for (uint32_t c = 0; c < cols; c++) {
        seed = seed * 1664525u + 1013904223u;
        x[c] = ((float)((int32_t)(seed >> 8) - 8388608)) / 8388608.0f;
    }

    float ref_fp32[19];
    for (uint32_t r = 0; r < rows; r++) {
        float acc = 0.0f;
        for (uint32_t c = 0; c < cols; c++) {
            acc += mat[r * cols + c] * x[c];
        }
        ref_fp32[r] = acc;
    }

    int8_t xq[64];
    float x_scales[2];
    qmx_quantize_activations_q8(x, xq, x_scales, cols);

    float x_dequant[45];
    for (uint32_t c = 0; c < cols; c++) {
        x_dequant[c] = (float)xq[c] * x_scales[c / QMX_TILE_COLS];
    }

    uint8_t qtypes[6] = {
        QMX_QUANT_Q8_SYM,
        QMX_QUANT_Q4_SYM,
        QMX_QUANT_TERNARY_158,
        QMX_QUANT_BINARY_1BIT,
        QMX_QUANT_Q4_ROW32,
        QMX_QUANT_Q8_ROW32
    };
    float fp32_tols[6] = { 0.15f, 0.85f, 3.50f, 4.50f, 0.70f, 0.15f };

    for (int t = 0; t < 6; t++) {
        float *scales = NULL;
        uint8_t *data = NULL;
        pack_test_matrix(mat, rows, cols, qtypes[t], &scales, &data);

        QmxTensor w;
        memset(&w, 0, sizeof(w));
        w.data = data;
        w.scales = scales;
        w.rows = rows;
        w.cols = cols;
        w.quant_type = qtypes[t];
        w.tile_rows = QMX_TILE_ROWS;

        float out[19];
        qmx_matmul_tiled_scalar(out, xq, x_scales, &w, 0);

        for (uint32_t r = 0; r < rows; r++) {
            float row_w[45];
            qmx_dequantize_tiled_row(row_w, &w, r);
            float exact_quant_dot = 0.0f;
            for (uint32_t c = 0; c < cols; c++) {
                exact_quant_dot += row_w[c] * x_dequant[c];
            }
            check_near("matmul_exact_quant", out[r], exact_quant_dot, 1e-4f);
            check_near("matmul_vs_fp32", out[r], ref_fp32[r], fp32_tols[t]);
        }

        free(scales);
        free(data);
    }
}
static void test_fp16_roundtrip(void) {
    const float vals[] = { 0.0f, -0.0f, 1.0f, -1.0f, 0.5f, -3.25f, 65504.0f, 1e-4f, -1e-4f, 0.000030517578125f, 123.5f };
    for (size_t i = 0; i < sizeof(vals) / sizeof(vals[0]); i++) {
        uint16_t h = qmx_fp32_to_fp16(vals[i]);
        float r = qmx_fp16_to_fp32(h);
        float tol = fabsf(vals[i]) * 2e-3f + 1e-5f;
        check_near("fp16_roundtrip", r, vals[i], tol);
    }
}

int main(void) {
    test_crc32();
    test_fp16_roundtrip();
    test_rmsnorm_and_softmax();
    test_rope_modes();
    test_tiled_matmul_and_row_dequant();

    if (g_failures == 0) {
        printf("[PASS] All QMX Stage 2.1 scalar ops & 16x32 tiled tests passed!\n");
        return 0;
    }
    fprintf(stderr, "[FAIL] Total failures: %d\n", g_failures);
    return 1;
}