/*
 * Copyright 2026 1WhiteFox
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/qmx_kernels.h"
#include <string.h>

#if defined(QMX_ENABLE_BACKEND_ARM_DSP) && defined(__ARM_FEATURE_DSP) && (defined(__GNUC__) || defined(__clang__))



static inline uint32_t pack_i16_pair(int16_t lo, int16_t hi) {
    return (uint32_t)(uint16_t)lo | ((uint32_t)(uint16_t)hi << 16u);
}

static inline int32_t arm_smlad(int32_t acc, uint32_t w_pair, uint32_t x_pair) {
    __asm__ volatile ("smlad %0, %1, %2, %0" : "+r"(acc) : "r"(w_pair), "r"(x_pair));
    return acc;
}

static void arm_dsp_prepare_act(const float * QMX_RESTRICT x,
                                int8_t      * QMX_RESTRICT xq,
                                int16_t     * QMX_RESTRICT xq16,
                                float       * QMX_RESTRICT x_scales,
                                uint32_t cols) {
    qmx_quantize_activations_q8(x, xq, x_scales, cols);
    uint32_t pad_cols = (uint32_t)qmx_align_up(cols, QMX_TILE_COLS);
    for (uint32_t i = 0; i < pad_cols; i++) {
        xq16[i] = (int16_t)xq[i];
    }
}

static int32_t arm_dsp_dot_row32(const uint8_t  * QMX_RESTRICT rp,
                                 const uint32_t * QMX_RESTRICT xp_pairs,
                                 uint8_t qtype) {
    int32_t acc = 0;
    if (qtype == QMX_QUANT_Q8_SYM) {
        const int8_t *wp = (const int8_t *)rp;
        for (uint32_t i = 0; i < 16u; i++) {
            uint32_t w_pair = pack_i16_pair((int16_t)wp[i * 2u], (int16_t)wp[i * 2u + 1u]);
            acc = arm_smlad(acc, w_pair, xp_pairs[i]);
        }
    } else if (qtype == QMX_QUANT_Q4_SYM) {
        for (uint32_t b = 0; b < 16u; b++) {
            uint8_t p = rp[b];
            int16_t w0 = (int16_t)((int32_t)(p & 0x0Fu) - 8);
            int16_t w1 = (int16_t)((int32_t)(p >> 4u) - 8);
            acc = arm_smlad(acc, pack_i16_pair(w0, w1), xp_pairs[b]);
        }
    } else if (qtype == QMX_QUANT_TERNARY_158) {
        for (uint32_t b = 0; b < 8u; b++) {
            const uint32_t *lut = qmx_lut_ternary_u32[rp[b]];
            acc = arm_smlad(acc, lut[0], xp_pairs[b * 2u + 0u]);
            acc = arm_smlad(acc, lut[1], xp_pairs[b * 2u + 1u]);
        }
    } else if (qtype == QMX_QUANT_BINARY_1BIT) {
        for (uint32_t b = 0; b < 4u; b++) {
            uint8_t p = rp[b];
            const uint32_t *l0 = qmx_lut_binary_u32[p & 0x0Fu];
            const uint32_t *l1 = qmx_lut_binary_u32[p >> 4u];
            acc = arm_smlad(acc, l0[0], xp_pairs[b * 4u + 0u]);
            acc = arm_smlad(acc, l0[1], xp_pairs[b * 4u + 1u]);
            acc = arm_smlad(acc, l1[0], xp_pairs[b * 4u + 2u]);
            acc = arm_smlad(acc, l1[1], xp_pairs[b * 4u + 3u]);
        }
    }
    return acc;
}

static void arm_dsp_matmul_tiled(float         * QMX_RESTRICT out,
                                 const int8_t  * QMX_RESTRICT xq,
                                 const int16_t * QMX_RESTRICT xq16,
                                 const float   * QMX_RESTRICT x_scales,
                                 const QmxTensor *w,
                                 int accumulate) {
    (void)xq;
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
    uint32_t xp_pairs[16];

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
            memcpy(xp_pairs, xq16 + (size_t)tc * QMX_TILE_COLS, sizeof(xp_pairs));

            for (uint32_t lr = 0; lr < QMX_TILE_ROWS; lr++) {
                if (r_base + lr >= orig_rows) break;
                int32_t dot = arm_dsp_dot_row32(tile_ptr + (size_t)lr * row_bytes, xp_pairs, base_qtype);
                float comb_scale = is_row_scaled ? (qmx_fp16_to_fp32(row_sc[lr]) * x_sc) : tile_comb_scale;
                tile_acc[lr] += (float)dot * comb_scale;
            }
        }

        for (uint32_t lr = 0; lr < QMX_TILE_ROWS; lr++) {
            if (r_base + lr < orig_rows) {
                out[r_base + lr] = tile_acc[lr];
            }
        }
    }
}

static float arm_dsp_dot_f32(const float * QMX_RESTRICT a,
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

static void arm_dsp_saxpy_f32(float       * QMX_RESTRICT y,
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

static int32_t arm_dsp_dot_q8(const int8_t  * QMX_RESTRICT q8,
                              const int16_t * QMX_RESTRICT q16,
                              const int8_t  * QMX_RESTRICT k8,
                              uint32_t n) {
    (void)q8;
    int32_t acc = 0;
    uint32_t i = 0;
    for (; i + 3u < n; i += 4u) {
        uint32_t q_p0, q_p1;
        memcpy(&q_p0, q16 + i, sizeof(uint32_t));
        memcpy(&q_p1, q16 + i + 2u, sizeof(uint32_t));
        uint32_t k_p0 = pack_i16_pair((int16_t)k8[i], (int16_t)k8[i + 1u]);
        uint32_t k_p1 = pack_i16_pair((int16_t)k8[i + 2u], (int16_t)k8[i + 3u]);
        acc = arm_smlad(acc, k_p0, q_p0);
        acc = arm_smlad(acc, k_p1, q_p1);
    }
    for (; i < n; i++) acc += (int32_t)q16[i] * (int32_t)k8[i];
    return acc;
}

const QmxBackendOps qmx_ops_arm_dsp = {
    "ARM-DSP",
    arm_dsp_prepare_act,
    arm_dsp_matmul_tiled,
    qmx_rmsnorm_scalar,
    arm_dsp_dot_f32,
    arm_dsp_saxpy_f32,
    arm_dsp_dot_q8,
    qmx_rmsnorm_quant_scalar,
    qmx_swiglu_quant_scalar,
    qmx_f16_to_f32_scalar,
    qmx_f32_to_f16_scalar,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL
};
#endif
