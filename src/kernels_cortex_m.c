#include "../include/qmx_kernels.h"

#if defined(QMX_ENABLE_BACKEND_CORTEX_M)
#include <string.h>

#if defined(__ARM_FEATURE_MVE)
#include <arm_mve.h>
#endif


static inline uint32_t cm_pack_i16_pair(int16_t lo, int16_t hi) {
    return (uint32_t)(uint16_t)lo | ((uint32_t)(uint16_t)hi << 16u);
}

static void cm_prepare_act_q8(const float * QMX_RESTRICT x,
                              int8_t      * QMX_RESTRICT xq,
                              int16_t     * QMX_RESTRICT xq16,
                              float       * QMX_RESTRICT x_scales,
                              uint32_t cols) {
    (void)xq16;
    qmx_quantize_activations_q8(x, xq, x_scales, cols);
}

static void cm_prepare_act_q8_q16(const float * QMX_RESTRICT x,
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

/* ============================================================================
 * 1. Cortex-M0 / M0+ (ARMv6-M) & Cortex-M23 (ARMv8-M.baseline)
 *    Optimized for low-register pressure (r0..r7) and single-cycle 32x32 MULS
 * ============================================================================ */

static int32_t cm0_dot_row32(const uint8_t * QMX_RESTRICT rp,
                             const int8_t  * QMX_RESTRICT xp,
                             uint8_t qtype) {
    int32_t acc = 0;
    if (qtype == QMX_QUANT_Q8_SYM) {
        const int8_t *wp = (const int8_t *)rp;
        for (uint32_t i = 0; i < 32u; i += 4u) {
            acc += (int32_t)wp[i + 0u] * (int32_t)xp[i + 0u]
                 + (int32_t)wp[i + 1u] * (int32_t)xp[i + 1u]
                 + (int32_t)wp[i + 2u] * (int32_t)xp[i + 2u]
                 + (int32_t)wp[i + 3u] * (int32_t)xp[i + 3u];
        }
    } else if (qtype == QMX_QUANT_Q4_SYM) {
        for (uint32_t b = 0; b < 16u; b += 2u) {
            uint32_t p0 = rp[b + 0u];
            uint32_t p1 = rp[b + 1u];
            acc += ((int32_t)(p0 & 0x0Fu) - 8) * (int32_t)xp[b * 2u + 0u]
                 + ((int32_t)(p0 >> 4u)   - 8) * (int32_t)xp[b * 2u + 1u]
                 + ((int32_t)(p1 & 0x0Fu) - 8) * (int32_t)xp[b * 2u + 2u]
                 + ((int32_t)(p1 >> 4u)   - 8) * (int32_t)xp[b * 2u + 3u];
        }
    } else if (qtype == QMX_QUANT_TERNARY_158) {
        for (uint32_t b = 0; b < 8u; b++) {
            const int8_t *l = qmx_lut_ternary_s8[rp[b]];
            const int8_t *x = xp + b * 4u;
            acc += (int32_t)l[0] * x[0] + (int32_t)l[1] * x[1]
                 + (int32_t)l[2] * x[2] + (int32_t)l[3] * x[3];
        }
    } else {
        for (uint32_t b = 0; b < 4u; b++) {
            uint8_t p = rp[b];
            const int8_t *l0 = qmx_lut_binary_s8[p & 0x0Fu];
            const int8_t *l1 = qmx_lut_binary_s8[p >> 4u];
            const int8_t *x  = xp + b * 8u;
            acc += (int32_t)l0[0] * x[0] + (int32_t)l0[1] * x[1]
                 + (int32_t)l0[2] * x[2] + (int32_t)l0[3] * x[3]
                 + (int32_t)l1[0] * x[4] + (int32_t)l1[1] * x[5]
                 + (int32_t)l1[2] * x[6] + (int32_t)l1[3] * x[7];
        }
    }
    return acc;
}

static void cm0_matmul_tiled(float         * QMX_RESTRICT out,
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
            for (uint32_t lr = 0; lr < QMX_TILE_ROWS; lr++) {
                if (r_base + lr >= orig_rows) break;
                int32_t dot = cm0_dot_row32(tile_ptr + (size_t)lr * row_bytes, xq_ptr, base_qtype);
                float comb_scale = is_row_scaled ? (qmx_fp16_to_fp32(row_sc[lr]) * x_sc) : tile_comb_scale;
                tile_acc[lr] += (float)dot * comb_scale;
            }
        }
        for (uint32_t lr = 0; lr < QMX_TILE_ROWS; lr++) {
            if (r_base + lr < orig_rows) out[r_base + lr] = tile_acc[lr];
        }
    }
}

/* ============================================================================
 * 2. Cortex-M3 (ARMv7-M Thumb-2: hardware UBFX/SBFX bitfield extract + MLA)
 * ============================================================================ */

static inline int32_t cm3_mla(int32_t acc, int32_t a, int32_t b) {
#if (defined(__ARM_ARCH_7M__) || defined(__ARM_ARCH_7EM__) || defined(__ARM_ARCH_8M_MAIN__)) && (defined(__GNUC__) || defined(__clang__))
    __asm__ ("mla %0, %1, %2, %0" : "+r"(acc) : "r"(a), "r"(b));
    return acc;
#else
    return acc + a * b;
#endif
}

static inline uint32_t cm3_ubfx4(uint32_t val, uint32_t lsb) {
#if (defined(__ARM_ARCH_7M__) || defined(__ARM_ARCH_7EM__) || defined(__ARM_ARCH_8M_MAIN__)) && (defined(__GNUC__) || defined(__clang__))
    uint32_t res;
    switch (lsb) {
        case 0:  __asm__ ("ubfx %0, %1, #0, #4"  : "=r"(res) : "r"(val)); return res;
        case 4:  __asm__ ("ubfx %0, %1, #4, #4"  : "=r"(res) : "r"(val)); return res;
        case 8:  __asm__ ("ubfx %0, %1, #8, #4"  : "=r"(res) : "r"(val)); return res;
        case 12: __asm__ ("ubfx %0, %1, #12, #4" : "=r"(res) : "r"(val)); return res;
        case 16: __asm__ ("ubfx %0, %1, #16, #4" : "=r"(res) : "r"(val)); return res;
        case 20: __asm__ ("ubfx %0, %1, #20, #4" : "=r"(res) : "r"(val)); return res;
        case 24: __asm__ ("ubfx %0, %1, #24, #4" : "=r"(res) : "r"(val)); return res;
        default: __asm__ ("ubfx %0, %1, #28, #4" : "=r"(res) : "r"(val)); return res;
    }
#else
    return (val >> lsb) & 0x0Fu;
#endif
}

static void cm3_dot_2rows_32(const uint8_t * QMX_RESTRICT r0,
                             const uint8_t * QMX_RESTRICT r1,
                             const int8_t  * QMX_RESTRICT xp,
                             uint8_t qtype,
                             int32_t * QMX_RESTRICT out0,
                             int32_t * QMX_RESTRICT out1) {
    int32_t a0 = 0, a1 = 0;
    if (qtype == QMX_QUANT_Q8_SYM) {
        const int8_t *w0 = (const int8_t *)r0;
        const int8_t *w1 = (const int8_t *)r1;
        for (uint32_t i = 0; i < 32u; i += 4u) {
            int32_t x0 = xp[i + 0u], x1 = xp[i + 1u], x2 = xp[i + 2u], x3 = xp[i + 3u];
            a0 = cm3_mla(a0, w0[i + 0u], x0); a0 = cm3_mla(a0, w0[i + 1u], x1);
            a0 = cm3_mla(a0, w0[i + 2u], x2); a0 = cm3_mla(a0, w0[i + 3u], x3);
            a1 = cm3_mla(a1, w1[i + 0u], x0); a1 = cm3_mla(a1, w1[i + 1u], x1);
            a1 = cm3_mla(a1, w1[i + 2u], x2); a1 = cm3_mla(a1, w1[i + 3u], x3);
        }
    } else if (qtype == QMX_QUANT_Q4_SYM) {
        for (uint32_t b = 0; b < 16u; b += 4u) {
            uint32_t w0_32 = 0, w1_32 = 0;
            memcpy(&w0_32, r0 + b, 4);
            memcpy(&w1_32, r1 + b, 4);
            const int8_t *x = xp + b * 2u;
            a0 = cm3_mla(a0, (int32_t)cm3_ubfx4(w0_32,  0) - 8, x[0]);
            a0 = cm3_mla(a0, (int32_t)cm3_ubfx4(w0_32,  4) - 8, x[1]);
            a0 = cm3_mla(a0, (int32_t)cm3_ubfx4(w0_32,  8) - 8, x[2]);
            a0 = cm3_mla(a0, (int32_t)cm3_ubfx4(w0_32, 12) - 8, x[3]);
            a0 = cm3_mla(a0, (int32_t)cm3_ubfx4(w0_32, 16) - 8, x[4]);
            a0 = cm3_mla(a0, (int32_t)cm3_ubfx4(w0_32, 20) - 8, x[5]);
            a0 = cm3_mla(a0, (int32_t)cm3_ubfx4(w0_32, 24) - 8, x[6]);
            a0 = cm3_mla(a0, (int32_t)cm3_ubfx4(w0_32, 28) - 8, x[7]);

            a1 = cm3_mla(a1, (int32_t)cm3_ubfx4(w1_32,  0) - 8, x[0]);
            a1 = cm3_mla(a1, (int32_t)cm3_ubfx4(w1_32,  4) - 8, x[1]);
            a1 = cm3_mla(a1, (int32_t)cm3_ubfx4(w1_32,  8) - 8, x[2]);
            a1 = cm3_mla(a1, (int32_t)cm3_ubfx4(w1_32, 12) - 8, x[3]);
            a1 = cm3_mla(a1, (int32_t)cm3_ubfx4(w1_32, 16) - 8, x[4]);
            a1 = cm3_mla(a1, (int32_t)cm3_ubfx4(w1_32, 20) - 8, x[5]);
            a1 = cm3_mla(a1, (int32_t)cm3_ubfx4(w1_32, 24) - 8, x[6]);
            a1 = cm3_mla(a1, (int32_t)cm3_ubfx4(w1_32, 28) - 8, x[7]);
        }
    } else if (qtype == QMX_QUANT_TERNARY_158) {
        for (uint32_t b = 0; b < 8u; b++) {
            const int8_t *l0 = qmx_lut_ternary_s8[r0[b]];
            const int8_t *l1 = qmx_lut_ternary_s8[r1[b]];
            const int8_t *x = xp + b * 4u;
            a0 = cm3_mla(a0, l0[0], x[0]); a0 = cm3_mla(a0, l0[1], x[1]);
            a0 = cm3_mla(a0, l0[2], x[2]); a0 = cm3_mla(a0, l0[3], x[3]);
            a1 = cm3_mla(a1, l1[0], x[0]); a1 = cm3_mla(a1, l1[1], x[1]);
            a1 = cm3_mla(a1, l1[2], x[2]); a1 = cm3_mla(a1, l1[3], x[3]);
        }
    } else {
        for (uint32_t b = 0; b < 4u; b++) {
            uint8_t p0 = r0[b], p1 = r1[b];
            const int8_t *l0a = qmx_lut_binary_s8[p0 & 0x0Fu], *l0b = qmx_lut_binary_s8[p0 >> 4u];
            const int8_t *l1a = qmx_lut_binary_s8[p1 & 0x0Fu], *l1b = qmx_lut_binary_s8[p1 >> 4u];
            const int8_t *x = xp + b * 8u;
            for (int k = 0; k < 4; k++) {
                a0 = cm3_mla(a0, l0a[k], x[k]);
                a0 = cm3_mla(a0, l0b[k], x[k + 4]);
                a1 = cm3_mla(a1, l1a[k], x[k]);
                a1 = cm3_mla(a1, l1b[k], x[k + 4]);
            }
        }
    }
    *out0 = a0;
    *out1 = a1;
}

static void cm3_matmul_tiled(float         * QMX_RESTRICT out,
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
                int32_t d0 = 0, d1 = 0;
                const uint8_t *rp0 = tile_ptr + (size_t)lr * row_bytes;
                cm3_dot_2rows_32(rp0, rp0 + row_bytes, xq_ptr, base_qtype, &d0, &d1);
                float cs0 = is_row_scaled ? (qmx_fp16_to_fp32(row_sc[lr]) * x_sc) : tile_comb_scale;
                tile_acc[lr] += (float)d0 * cs0;
                if (r_base + lr + 1u < orig_rows) {
                    float cs1 = is_row_scaled ? (qmx_fp16_to_fp32(row_sc[lr + 1u]) * x_sc) : tile_comb_scale;
                    tile_acc[lr + 1u] += (float)d1 * cs1;
                }
            }
        }
        for (uint32_t lr = 0; lr < QMX_TILE_ROWS; lr++) {
            if (r_base + lr < orig_rows) out[r_base + lr] = tile_acc[lr];
        }
    }
}

/* ============================================================================
 * 3. Cortex-M4 / M7 (ARMv7E-M) & Cortex-M33 (ARMv8-M.mainline)
 *    Uses single-cycle SXTB16 dual byte sign-extend + SMLAD dual 16x16 MAC
 * ============================================================================ */

static inline int32_t cm4_smlad(int32_t acc, uint32_t w_pair, uint32_t x_pair) {
#if defined(__ARM_FEATURE_DSP) && (defined(__GNUC__) || defined(__clang__))
    __asm__ ("smlad %0, %1, %2, %0" : "+r"(acc) : "r"(w_pair), "r"(x_pair));
    return acc;
#else
    int16_t w0 = (int16_t)(w_pair & 0xFFFFu);
    int16_t w1 = (int16_t)(w_pair >> 16u);
    int16_t x0 = (int16_t)(x_pair & 0xFFFFu);
    int16_t x1 = (int16_t)(x_pair >> 16u);
    return acc + (int32_t)w0 * x0 + (int32_t)w1 * x1;
#endif
}

static inline uint32_t cm4_sxtb16(uint32_t v) {
#if defined(__ARM_FEATURE_DSP) && (defined(__GNUC__) || defined(__clang__))
    uint32_t res;
    __asm__ ("sxtb16 %0, %1" : "=r"(res) : "r"(v));
    return res;
#else
    int16_t lo = (int8_t)(v & 0xFFu);
    int16_t hi = (int8_t)((v >> 16u) & 0xFFu);
    return cm_pack_i16_pair(lo, hi);
#endif
}

static int32_t cm4_dot_row32(const uint8_t  * QMX_RESTRICT rp,
                             const uint32_t * QMX_RESTRICT xp_pairs,
                             uint8_t qtype) {
    int32_t acc = 0;
    if (qtype == QMX_QUANT_Q8_SYM) {
        for (uint32_t i = 0; i < 8u; i++) {
            uint32_t w4 = 0;
            memcpy(&w4, rp + i * 4u, 4);
            /* bytes [0, 2] and [1, 3] via sxtb16 -> repack or pair extract */
            uint32_t p02 = cm4_sxtb16(w4);
            uint32_t p13 = cm4_sxtb16(w4 >> 8u);
            uint32_t p01 = (p02 & 0xFFFFu) | (p13 << 16u);
            uint32_t p23 = (p02 >> 16u) | (p13 & 0xFFFF0000u);
            acc = cm4_smlad(acc, p01, xp_pairs[i * 2u + 0u]);
            acc = cm4_smlad(acc, p23, xp_pairs[i * 2u + 1u]);
        }
    } else if (qtype == QMX_QUANT_Q4_SYM) {
        for (uint32_t b = 0; b < 16u; b++) {
            uint8_t p = rp[b];
            int16_t w0 = (int16_t)((int32_t)(p & 0x0Fu) - 8);
            int16_t w1 = (int16_t)((int32_t)(p >> 4u)   - 8);
            acc = cm4_smlad(acc, cm_pack_i16_pair(w0, w1), xp_pairs[b]);
        }
    } else if (qtype == QMX_QUANT_TERNARY_158) {
        for (uint32_t b = 0; b < 8u; b++) {
            const uint32_t *lut = qmx_lut_ternary_u32[rp[b]];
            acc = cm4_smlad(acc, lut[0], xp_pairs[b * 2u + 0u]);
            acc = cm4_smlad(acc, lut[1], xp_pairs[b * 2u + 1u]);
        }
    } else {
        for (uint32_t b = 0; b < 4u; b++) {
            uint8_t p = rp[b];
            const uint32_t *l0 = qmx_lut_binary_u32[p & 0x0Fu];
            const uint32_t *l1 = qmx_lut_binary_u32[p >> 4u];
            acc = cm4_smlad(acc, l0[0], xp_pairs[b * 4u + 0u]);
            acc = cm4_smlad(acc, l0[1], xp_pairs[b * 4u + 1u]);
            acc = cm4_smlad(acc, l1[0], xp_pairs[b * 4u + 2u]);
            acc = cm4_smlad(acc, l1[1], xp_pairs[b * 4u + 3u]);
        }
    }
    return acc;
}

static void cm4_matmul_tiled(float         * QMX_RESTRICT out,
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
                int32_t dot = cm4_dot_row32(tile_ptr + (size_t)lr * row_bytes, xp_pairs, base_qtype);
                float comb_scale = is_row_scaled ? (qmx_fp16_to_fp32(row_sc[lr]) * x_sc) : tile_comb_scale;
                tile_acc[lr] += (float)dot * comb_scale;
            }
        }
        for (uint32_t lr = 0; lr < QMX_TILE_ROWS; lr++) {
            if (r_base + lr < orig_rows) out[r_base + lr] = tile_acc[lr];
        }
    }
}

static int32_t cm4_dot_q8(const int8_t  * QMX_RESTRICT q8,
                          const int16_t * QMX_RESTRICT q16,
                          const int8_t  * QMX_RESTRICT k8,
                          uint32_t n) {
    (void)q8;
    int32_t acc = 0;
    uint32_t i = 0;
    for (; i + 3u < n; i += 4u) {
        uint32_t q_p0, q_p1;
        memcpy(&q_p0, q16 + i, 4);
        memcpy(&q_p1, q16 + i + 2u, 4);
        uint32_t k_p0 = cm_pack_i16_pair((int16_t)k8[i], (int16_t)k8[i + 1u]);
        uint32_t k_p1 = cm_pack_i16_pair((int16_t)k8[i + 2u], (int16_t)k8[i + 3u]);
        acc = cm4_smlad(acc, k_p0, q_p0);
        acc = cm4_smlad(acc, k_p1, q_p1);
    }
    for (; i < n; i++) acc += (int32_t)q16[i] * (int32_t)k8[i];
    return acc;
}

/* ============================================================================
 * 4. Cortex-M55 (ARMv8.1-M Helium MVE: 128-bit vmladavaq_s8 vector MAC)
 * ============================================================================ */

static int32_t cm55_dot_q8(const int8_t  * QMX_RESTRICT q8,
                           const int16_t * QMX_RESTRICT q16,
                           const int8_t  * QMX_RESTRICT k8,
                           uint32_t n) {
    (void)q16;
    int32_t sum = 0;
    uint32_t i = 0;
#if defined(__ARM_FEATURE_MVE)
    for (; i + 15u < n; i += 16u) {
        int8x16_t vq = vldrbq_s8(q8 + i);
        int8x16_t vk = vldrbq_s8(k8 + i);
        sum = vmladavaq_s8(sum, vq, vk);
    }
#endif
    for (; i < n; i++) sum += (int32_t)q8[i] * (int32_t)k8[i];
    return sum;
}

static void cm55_matmul_tiled(float         * QMX_RESTRICT out,
                              const int8_t  * QMX_RESTRICT xq,
                              const int16_t * QMX_RESTRICT xq16,
                              const float   * QMX_RESTRICT x_scales,
                              const QmxTensor *w,
                              int accumulate) {
#if defined(__ARM_FEATURE_MVE)
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
            const int8_t *xp = xq + (size_t)tc * QMX_TILE_COLS;
            int8x16_t x0 = vldrbq_s8(xp);
            int8x16_t x1 = vldrbq_s8(xp + 16u);
            for (uint32_t lr = 0; lr < QMX_TILE_ROWS; lr++) {
                if (r_base + lr >= orig_rows) break;
                int32_t dot = 0;
                const uint8_t *rp = tile_ptr + (size_t)lr * row_bytes;
                if (base_qtype == QMX_QUANT_Q8_SYM) {
                    dot = vmladavq_s8(vldrbq_s8((const int8_t *)rp), x0);
                    dot = vmladavaq_s8(dot, vldrbq_s8((const int8_t *)rp + 16u), x1);
                } else if (base_qtype == QMX_QUANT_Q4_SYM) {
                    int8_t wtmp[32];
                    for (uint32_t b = 0; b < 16u; b++) {
                        uint8_t p = rp[b];
                        wtmp[b * 2u + 0u] = (int8_t)((int32_t)(p & 0x0Fu) - 8);
                        wtmp[b * 2u + 1u] = (int8_t)((int32_t)(p >> 4u) - 8);
                    }
                    dot = vmladavq_s8(vldrbq_s8(wtmp), x0);
                    dot = vmladavaq_s8(dot, vldrbq_s8(wtmp + 16u), x1);
                } else {
                    dot = cm0_dot_row32(rp, xp, base_qtype);
                }
                float comb_scale = is_row_scaled ? (qmx_fp16_to_fp32(row_sc[lr]) * x_sc) : tile_comb_scale;
                tile_acc[lr] += (float)dot * comb_scale;
            }
        }
        for (uint32_t lr = 0; lr < QMX_TILE_ROWS; lr++) {
            if (r_base + lr < orig_rows) out[r_base + lr] = tile_acc[lr];
        }
    }
#else
    cm3_matmul_tiled(out, xq, xq16, x_scales, w, accumulate);
#endif
}


static float cm_dot_f32(const float * QMX_RESTRICT a, const float * QMX_RESTRICT b, uint32_t n) {
    float s0 = 0.0f, s1 = 0.0f;
    uint32_t i = 0;
    for (; i + 3u < n; i += 4u) {
        s0 += a[i] * b[i] + a[i + 1u] * b[i + 1u];
        s1 += a[i + 2u] * b[i + 2u] + a[i + 3u] * b[i + 3u];
    }
    for (; i < n; i++) s0 += a[i] * b[i];
    return s0 + s1;
}

static void cm_saxpy_f32(float * QMX_RESTRICT y, float a, const float * QMX_RESTRICT x, uint32_t n) {
    uint32_t i = 0;
    for (; i + 3u < n; i += 4u) {
        y[i + 0u] += a * x[i + 0u];
        y[i + 1u] += a * x[i + 1u];
        y[i + 2u] += a * x[i + 2u];
        y[i + 3u] += a * x[i + 3u];
    }
    for (; i < n; i++) y[i] += a * x[i];
}

const QmxBackendOps qmx_ops_cortex_m0 = {
    "Cortex-M0/M0+", cm_prepare_act_q8, cm0_matmul_tiled, qmx_rmsnorm_scalar, cm_dot_f32, cm_saxpy_f32, cm55_dot_q8, qmx_rmsnorm_quant_scalar, qmx_swiglu_quant_scalar, qmx_f16_to_f32_scalar, qmx_f32_to_f16_scalar
};

const QmxBackendOps qmx_ops_cortex_m23 = {
    "Cortex-M23", cm_prepare_act_q8, cm0_matmul_tiled, qmx_rmsnorm_scalar, cm_dot_f32, cm_saxpy_f32, cm55_dot_q8, qmx_rmsnorm_quant_scalar, qmx_swiglu_quant_scalar, qmx_f16_to_f32_scalar, qmx_f32_to_f16_scalar
};

const QmxBackendOps qmx_ops_cortex_m3 = {
    "Cortex-M3", cm_prepare_act_q8, cm3_matmul_tiled, qmx_rmsnorm_scalar, cm_dot_f32, cm_saxpy_f32, cm55_dot_q8, qmx_rmsnorm_quant_scalar, qmx_swiglu_quant_scalar, qmx_f16_to_f32_scalar, qmx_f32_to_f16_scalar
};

const QmxBackendOps qmx_ops_cortex_m4 = {
    "Cortex-M4/M7", cm_prepare_act_q8_q16, cm4_matmul_tiled, qmx_rmsnorm_scalar, cm_dot_f32, cm_saxpy_f32, cm4_dot_q8, qmx_rmsnorm_quant_scalar, qmx_swiglu_quant_scalar, qmx_f16_to_f32_scalar, qmx_f32_to_f16_scalar
};

const QmxBackendOps qmx_ops_cortex_m33 = {
    "Cortex-M33", cm_prepare_act_q8_q16, cm4_matmul_tiled, qmx_rmsnorm_scalar, cm_dot_f32, cm_saxpy_f32, cm4_dot_q8, qmx_rmsnorm_quant_scalar, qmx_swiglu_quant_scalar, qmx_f16_to_f32_scalar, qmx_f32_to_f16_scalar
};

const QmxBackendOps qmx_ops_cortex_m55 = {
    "Cortex-M55", cm_prepare_act_q8, cm55_matmul_tiled, qmx_rmsnorm_scalar, cm_dot_f32, cm_saxpy_f32, cm55_dot_q8, qmx_rmsnorm_quant_scalar, qmx_swiglu_quant_scalar, qmx_f16_to_f32_scalar, qmx_f32_to_f16_scalar
};

#endif
