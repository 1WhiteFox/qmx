/*
 * Copyright 2026 1WhiteFox
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/qmx_kernels.h"

#if defined(QMX_ENABLE_BACKEND_MCU32)
#include <string.h>

static void mcu32_prepare_act(const float * QMX_RESTRICT x,
                              int8_t      * QMX_RESTRICT xq,
                              int16_t     * QMX_RESTRICT xq16,
                              float       * QMX_RESTRICT x_scales,
                              uint32_t cols) {
    (void)xq16;
    qmx_quantize_activations_q8(x, xq, x_scales, cols);
}

static void mcu32_dot_2rows_32(const uint8_t * QMX_RESTRICT r0,
                               const uint8_t * QMX_RESTRICT r1,
                               const int8_t  * QMX_RESTRICT xp,
                               uint8_t qtype,
                               int32_t * QMX_RESTRICT out0,
                               int32_t * QMX_RESTRICT out1) {
    int32_t a0 = 0;
    int32_t a1 = 0;
    if (qtype == QMX_QUANT_Q8_SYM) {
        const int8_t *w0 = (const int8_t *)r0;
        const int8_t *w1 = (const int8_t *)r1;
        for (uint32_t i = 0; i < 32u; i += 4u) {
            int32_t x0 = xp[i + 0u];
            int32_t x1 = xp[i + 1u];
            int32_t x2 = xp[i + 2u];
            int32_t x3 = xp[i + 3u];
            a0 += (int32_t)w0[i + 0u] * x0 + (int32_t)w0[i + 1u] * x1
                + (int32_t)w0[i + 2u] * x2 + (int32_t)w0[i + 3u] * x3;
            a1 += (int32_t)w1[i + 0u] * x0 + (int32_t)w1[i + 1u] * x1
                + (int32_t)w1[i + 2u] * x2 + (int32_t)w1[i + 3u] * x3;
        }
    } else if (qtype == QMX_QUANT_Q4_SYM) {
        for (uint32_t b = 0; b < 16u; b += 2u) {
            int32_t x0 = xp[b * 2u + 0u];
            int32_t x1 = xp[b * 2u + 1u];
            int32_t x2 = xp[b * 2u + 2u];
            int32_t x3 = xp[b * 2u + 3u];
            uint8_t p00 = r0[b + 0u];
            uint8_t p01 = r0[b + 1u];
            uint8_t p10 = r1[b + 0u];
            uint8_t p11 = r1[b + 1u];
            a0 += ((int32_t)(p00 & 0x0Fu) - 8) * x0 + ((int32_t)(p00 >> 4u) - 8) * x1
                + ((int32_t)(p01 & 0x0Fu) - 8) * x2 + ((int32_t)(p01 >> 4u) - 8) * x3;
            a1 += ((int32_t)(p10 & 0x0Fu) - 8) * x0 + ((int32_t)(p10 >> 4u) - 8) * x1
                + ((int32_t)(p11 & 0x0Fu) - 8) * x2 + ((int32_t)(p11 >> 4u) - 8) * x3;
        }
    } else if (qtype == QMX_QUANT_TERNARY_158) {
        for (uint32_t b = 0; b < 8u; b++) {
            int32_t x0 = xp[b * 4u + 0u];
            int32_t x1 = xp[b * 4u + 1u];
            int32_t x2 = xp[b * 4u + 2u];
            int32_t x3 = xp[b * 4u + 3u];
            const int8_t *l0 = qmx_lut_ternary_s8[r0[b]];
            const int8_t *l1 = qmx_lut_ternary_s8[r1[b]];
            a0 += (int32_t)l0[0] * x0 + (int32_t)l0[1] * x1 + (int32_t)l0[2] * x2 + (int32_t)l0[3] * x3;
            a1 += (int32_t)l1[0] * x0 + (int32_t)l1[1] * x1 + (int32_t)l1[2] * x2 + (int32_t)l1[3] * x3;
        }
    } else if (qtype == QMX_QUANT_BINARY_1BIT) {
        for (uint32_t b = 0; b < 4u; b++) {
            const int8_t *x0 = xp + b * 8u;
            uint8_t p0 = r0[b];
            uint8_t p1 = r1[b];
            const int8_t *l0a = qmx_lut_binary_s8[p0 & 0x0Fu];
            const int8_t *l0b = qmx_lut_binary_s8[p0 >> 4u];
            const int8_t *l1a = qmx_lut_binary_s8[p1 & 0x0Fu];
            const int8_t *l1b = qmx_lut_binary_s8[p1 >> 4u];
            a0 += (int32_t)l0a[0] * x0[0] + (int32_t)l0a[1] * x0[1]
                + (int32_t)l0a[2] * x0[2] + (int32_t)l0a[3] * x0[3]
                + (int32_t)l0b[0] * x0[4] + (int32_t)l0b[1] * x0[5]
                + (int32_t)l0b[2] * x0[6] + (int32_t)l0b[3] * x0[7];
            a1 += (int32_t)l1a[0] * x0[0] + (int32_t)l1a[1] * x0[1]
                + (int32_t)l1a[2] * x0[2] + (int32_t)l1a[3] * x0[3]
                + (int32_t)l1b[0] * x0[4] + (int32_t)l1b[1] * x0[5]
                + (int32_t)l1b[2] * x0[6] + (int32_t)l1b[3] * x0[7];
        }
    }
    *out0 = a0;
    *out1 = a1;
}

static void mcu32_matmul_tiled(float         * QMX_RESTRICT out,
                               const int8_t  * QMX_RESTRICT xq,
                               const int16_t * QMX_RESTRICT xq16,
                               const float   * QMX_RESTRICT x_scales,
                               const QmxTensor *w,
                               int accumulate) {
    (void)xq16;
    uint32_t orig_rows = w->rows;
    uint32_t orig_cols = w->cols;
    uint32_t pad_r = (uint32_t)qmx_align_up(orig_rows, QMX_TILE_ROWS);
    uint32_t pad_c = (uint32_t)qmx_align_up(orig_cols, QMX_TILE_COLS);
    uint32_t n_tr = pad_r / QMX_TILE_ROWS;
    uint32_t n_tc = pad_c / QMX_TILE_COLS;
    uint8_t qtype = w->quant_type;
    int is_row_scaled = (qtype == QMX_QUANT_Q4_ROW32 || qtype == QMX_QUANT_Q8_ROW32);
    uint8_t base_qtype = (qtype == QMX_QUANT_Q4_ROW32) ? QMX_QUANT_Q4_SYM
                       : (qtype == QMX_QUANT_Q8_ROW32) ? QMX_QUANT_Q8_SYM : qtype;
    size_t tile_bytes = qmx_tile_data_bytes(qtype);
    size_t row_bytes = qmx_tile_row_bytes(qtype);
    const uint8_t *base_data = (const uint8_t *)w->data;
    const uint16_t *scales_f16 = (const uint16_t *)(const void *)w->scales;

    float tile_acc[QMX_TILE_ROWS];

    for (uint32_t tr = 0; tr < n_tr; tr++) {
        uint32_t r_base = tr * QMX_TILE_ROWS;
        for (uint32_t lr = 0; lr < QMX_TILE_ROWS; lr++) {
            tile_acc[lr] = (accumulate && (r_base + lr < orig_rows)) ? out[r_base + lr] : 0.0f;
        }

        for (uint32_t tc = 0; tc < n_tc; tc++) {
            uint32_t tile_idx = tr * n_tc + tc;
            float x_sc = x_scales[tc];
            float tile_comb_scale = is_row_scaled ? 0.0f : (w->scales[tile_idx] * x_sc);
            const uint16_t *row_sc = is_row_scaled ? (scales_f16 + (size_t)tile_idx * QMX_TILE_ROWS) : NULL;
            const uint8_t *tile_ptr = base_data + (size_t)tile_idx * tile_bytes;
            const int8_t *xq_ptr = xq + (size_t)tc * QMX_TILE_COLS;

            for (uint32_t lr = 0; lr < QMX_TILE_ROWS; lr += 2u) {
                if (r_base + lr >= orig_rows) break;
                int32_t d0 = 0;
                int32_t d1 = 0;
                const uint8_t *rp0 = tile_ptr + (size_t)lr * row_bytes;
                const uint8_t *rp1 = rp0 + row_bytes;
                mcu32_dot_2rows_32(rp0, rp1, xq_ptr, base_qtype, &d0, &d1);
                float cs0 = is_row_scaled ? (qmx_fp16_to_fp32(row_sc[lr]) * x_sc) : tile_comb_scale;
                tile_acc[lr] += (float)d0 * cs0;
                if (r_base + lr + 1u < orig_rows) {
                    float cs1 = is_row_scaled ? (qmx_fp16_to_fp32(row_sc[lr + 1u]) * x_sc) : tile_comb_scale;
                    tile_acc[lr + 1u] += (float)d1 * cs1;
                }
            }
        }

        for (uint32_t lr = 0; lr < QMX_TILE_ROWS; lr++) {
            if (r_base + lr < orig_rows) {
                out[r_base + lr] = tile_acc[lr];
            }
        }
    }
}

static float mcu32_dot_f32(const float * QMX_RESTRICT a,
                           const float * QMX_RESTRICT b,
                           uint32_t n) {
    float s0 = 0.0f;
    float s1 = 0.0f;
    uint32_t i = 0;
    for (; i + 3u < n; i += 4u) {
        s0 += a[i] * b[i] + a[i + 1u] * b[i + 1u];
        s1 += a[i + 2u] * b[i + 2u] + a[i + 3u] * b[i + 3u];
    }
    for (; i < n; i++) s0 += a[i] * b[i];
    return s0 + s1;
}

static void mcu32_saxpy_f32(float       * QMX_RESTRICT y,
                            float a,
                            const float * QMX_RESTRICT x,
                            uint32_t n) {
    uint32_t i = 0;
    for (; i + 3u < n; i += 4u) {
        y[i]      += a * x[i];
        y[i + 1u] += a * x[i + 1u];
        y[i + 2u] += a * x[i + 2u];
        y[i + 3u] += a * x[i + 3u];
    }
    for (; i < n; i++) y[i] += a * x[i];
}

static int32_t mcu32_dot_q8(const int8_t  * QMX_RESTRICT q8,
                            const int16_t * QMX_RESTRICT q16,
                            const int8_t  * QMX_RESTRICT k8,
                            uint32_t n) {
    (void)q16;
    int32_t acc0 = 0;
    int32_t acc1 = 0;
    uint32_t i = 0;
    for (; i + 3u < n; i += 4u) {
        acc0 += (int32_t)q8[i] * k8[i] + (int32_t)q8[i + 1u] * k8[i + 1u];
        acc1 += (int32_t)q8[i + 2u] * k8[i + 2u] + (int32_t)q8[i + 3u] * k8[i + 3u];
    }
    for (; i < n; i++) acc0 += (int32_t)q8[i] * k8[i];
    return acc0 + acc1;
}

const QmxBackendOps qmx_ops_mcu32 = {
    "MCU32",
    mcu32_prepare_act,
    mcu32_matmul_tiled,
    qmx_rmsnorm_scalar,
    mcu32_dot_f32,
    mcu32_saxpy_f32,
    mcu32_dot_q8,
    qmx_rmsnorm_quant_scalar,
    qmx_swiglu_quant_scalar,
    qmx_f16_to_f32_scalar,
    qmx_f32_to_f16_scalar
};

#endif
