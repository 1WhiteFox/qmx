/*
 * Copyright 2026 1WhiteFox
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/qmx_kernels.h"

#if defined(QMX_ENABLE_BACKEND_NEON) && (defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__))
#include <arm_neon.h>

static void neon_prepare_act(const float * QMX_RESTRICT x,
                             int8_t      * QMX_RESTRICT xq,
                             int16_t     * QMX_RESTRICT xq16,
                             float       * QMX_RESTRICT x_scales,
                             uint32_t cols) {
    (void)xq16;
    uint32_t full_groups = cols / QMX_TILE_COLS;
    const int8x16_t min_i8 = vdupq_n_s8(-127);
#if !defined(__aarch64__)
    const uint32x4_t sign_mask = vdupq_n_u32(0x80000000u);
    const float32x4_t half_f = vdupq_n_f32(0.5f);
#endif
    for (uint32_t g = 0; g < full_groups; g++) {
        const float *xp = x + (size_t)g * QMX_TILE_COLS;
        float32x4_t v0 = vld1q_f32(xp + 0u);
        float32x4_t v1 = vld1q_f32(xp + 4u);
        float32x4_t v2 = vld1q_f32(xp + 8u);
        float32x4_t v3 = vld1q_f32(xp + 12u);
        float32x4_t v4 = vld1q_f32(xp + 16u);
        float32x4_t v5 = vld1q_f32(xp + 20u);
        float32x4_t v6 = vld1q_f32(xp + 24u);
        float32x4_t v7 = vld1q_f32(xp + 28u);

        float32x4_t m0 = vmaxq_f32(vmaxq_f32(vabsq_f32(v0), vabsq_f32(v1)),
                                   vmaxq_f32(vabsq_f32(v2), vabsq_f32(v3)));
        float32x4_t m1 = vmaxq_f32(vmaxq_f32(vabsq_f32(v4), vabsq_f32(v5)),
                                   vmaxq_f32(vabsq_f32(v6), vabsq_f32(v7)));
        float32x4_t m4 = vmaxq_f32(m0, m1);
        float32x2_t m2 = vpmax_f32(vget_low_f32(m4), vget_high_f32(m4));
        m2 = vpmax_f32(m2, m2);
        float max_abs = vget_lane_f32(m2, 0);

        x_scales[g] = max_abs / 127.0f;
        float inv_scale = (max_abs == 0.0f) ? 0.0f : (127.0f / max_abs);
        float32x4_t v_inv = vdupq_n_f32(inv_scale);

        v0 = vmulq_f32(v0, v_inv); v1 = vmulq_f32(v1, v_inv);
        v2 = vmulq_f32(v2, v_inv); v3 = vmulq_f32(v3, v_inv);
        v4 = vmulq_f32(v4, v_inv); v5 = vmulq_f32(v5, v_inv);
        v6 = vmulq_f32(v6, v_inv); v7 = vmulq_f32(v7, v_inv);
#if defined(__aarch64__)
        int32x4_t i0 = vcvtnq_s32_f32(v0), i1 = vcvtnq_s32_f32(v1);
        int32x4_t i2 = vcvtnq_s32_f32(v2), i3 = vcvtnq_s32_f32(v3);
        int32x4_t i4 = vcvtnq_s32_f32(v4), i5 = vcvtnq_s32_f32(v5);
        int32x4_t i6 = vcvtnq_s32_f32(v6), i7 = vcvtnq_s32_f32(v7);
#else
        int32x4_t i0 = vcvtq_s32_f32(vaddq_f32(v0, vbslq_f32(sign_mask, v0, half_f)));
        int32x4_t i1 = vcvtq_s32_f32(vaddq_f32(v1, vbslq_f32(sign_mask, v1, half_f)));
        int32x4_t i2 = vcvtq_s32_f32(vaddq_f32(v2, vbslq_f32(sign_mask, v2, half_f)));
        int32x4_t i3 = vcvtq_s32_f32(vaddq_f32(v3, vbslq_f32(sign_mask, v3, half_f)));
        int32x4_t i4 = vcvtq_s32_f32(vaddq_f32(v4, vbslq_f32(sign_mask, v4, half_f)));
        int32x4_t i5 = vcvtq_s32_f32(vaddq_f32(v5, vbslq_f32(sign_mask, v5, half_f)));
        int32x4_t i6 = vcvtq_s32_f32(vaddq_f32(v6, vbslq_f32(sign_mask, v6, half_f)));
        int32x4_t i7 = vcvtq_s32_f32(vaddq_f32(v7, vbslq_f32(sign_mask, v7, half_f)));
#endif
        int16x8_t s0 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
        int16x8_t s1 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
        int16x8_t s2 = vcombine_s16(vqmovn_s32(i4), vqmovn_s32(i5));
        int16x8_t s3 = vcombine_s16(vqmovn_s32(i6), vqmovn_s32(i7));

        int8x16_t b0 = vmaxq_s8(vcombine_s8(vqmovn_s16(s0), vqmovn_s16(s1)), min_i8);
        int8x16_t b1 = vmaxq_s8(vcombine_s8(vqmovn_s16(s2), vqmovn_s16(s3)), min_i8);
        int8_t *dst = xq + (size_t)g * QMX_TILE_COLS;
        vst1q_s8(dst + 0u, b0);
        vst1q_s8(dst + 16u, b1);
    }
    uint32_t tail_base = full_groups * QMX_TILE_COLS;
    if (tail_base < cols) {
        qmx_quantize_activations_q8(x + tail_base, xq + tail_base, x_scales + full_groups, cols - tail_base);
    }
}

static inline int32_t neon_hsum_s32(int32x4_t v) {
    int32x2_t sum2 = vadd_s32(vget_low_s32(v), vget_high_s32(v));
    sum2 = vpadd_s32(sum2, sum2);
    return vget_lane_s32(sum2, 0);
}

static inline int32x4_t neon_hsum4_s32(int32x4_t r0, int32x4_t r1, int32x4_t r2, int32x4_t r3) {
    int32x2_t s0 = vadd_s32(vget_low_s32(r0), vget_high_s32(r0));
    int32x2_t s1 = vadd_s32(vget_low_s32(r1), vget_high_s32(r1));
    int32x2_t s2 = vadd_s32(vget_low_s32(r2), vget_high_s32(r2));
    int32x2_t s3 = vadd_s32(vget_low_s32(r3), vget_high_s32(r3));
    return vcombine_s32(vpadd_s32(s0, s1), vpadd_s32(s2, s3));
}

static inline float neon_hsum_f32(float32x4_t v) {
    float32x2_t sum2 = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    sum2 = vpadd_f32(sum2, sum2);
    return vget_lane_f32(sum2, 0);
}

static inline int32x4_t neon_dot_s8x32_vec(int8x16_t w0, int8x16_t w1,
                                           int8x16_t x0, int8x16_t x1) {
    int16x8_t p0 = vmull_s8(vget_low_s8(w0), vget_low_s8(x0));
    p0 = vmlal_s8(p0, vget_high_s8(w0), vget_high_s8(x0));
    int16x8_t p1 = vmull_s8(vget_low_s8(w1), vget_low_s8(x1));
    p1 = vmlal_s8(p1, vget_high_s8(w1), vget_high_s8(x1));
    int32x4_t s32 = vaddl_s16(vget_low_s16(p0), vget_high_s16(p0));
    return vpadalq_s16(s32, p1);
}

static inline int32x4_t neon_dot_row_vec(const uint8_t *tile_ptr, uint32_t lr, uint8_t qtype,
                                         int8x16_t x0, int8x16_t x1,
                                         uint8x16_t mask0f, int8x16_t bias8,
                                         uint8x8_t mask03_8, uint8x16_t one_u8, uint8x16_t two_u8) {
    if (qtype == QMX_QUANT_Q8_SYM) {
        const int8_t *rp = (const int8_t *)(tile_ptr + lr * 32u);
        int8x16_t w0 = vld1q_s8(rp);
        int8x16_t w1 = vld1q_s8(rp + 16u);
        return neon_dot_s8x32_vec(w0, w1, x0, x1);
    } else if (qtype == QMX_QUANT_Q4_SYM) {
        const uint8_t *rp = tile_ptr + lr * 16u;
        uint8x16_t raw = vld1q_u8(rp);
        uint8x16_t lo_nib = vandq_u8(raw, mask0f);
        uint8x16_t hi_nib = vshrq_n_u8(raw, 4);
        uint8x16x2_t zipped = vzipq_u8(lo_nib, hi_nib);
        int8x16_t w0 = vsubq_s8(vreinterpretq_s8_u8(zipped.val[0]), bias8);
        int8x16_t w1 = vsubq_s8(vreinterpretq_s8_u8(zipped.val[1]), bias8);
        return neon_dot_s8x32_vec(w0, w1, x0, x1);
    } else if (qtype == QMX_QUANT_TERNARY_158) {
        const uint8_t *rp = tile_ptr + lr * 8u;
        uint8x8_t raw = vld1_u8(rp);
        uint8x8_t c0 = vand_u8(raw, mask03_8);
        uint8x8_t c1 = vand_u8(vshr_n_u8(raw, 2), mask03_8);
        uint8x8_t c2 = vand_u8(vshr_n_u8(raw, 4), mask03_8);
        uint8x8_t c3 = vshr_n_u8(raw, 6);
        uint8x8x2_t z02 = vzip_u8(c0, c2);
        uint8x8x2_t z13 = vzip_u8(c1, c3);
        uint8x8x2_t lo16 = vzip_u8(z02.val[0], z13.val[0]);
        uint8x8x2_t hi16 = vzip_u8(z02.val[1], z13.val[1]);
        uint8x16_t u0 = vcombine_u8(lo16.val[0], lo16.val[1]);
        uint8x16_t u1 = vcombine_u8(hi16.val[0], hi16.val[1]);
        int8x16_t w0 = vsubq_s8(vreinterpretq_s8_u8(vceqq_u8(u0, two_u8)),
                                vreinterpretq_s8_u8(vceqq_u8(u0, one_u8)));
        int8x16_t w1 = vsubq_s8(vreinterpretq_s8_u8(vceqq_u8(u1, two_u8)),
                                vreinterpretq_s8_u8(vceqq_u8(u1, one_u8)));
        return neon_dot_s8x32_vec(w0, w1, x0, x1);
    } else {
        const uint8_t *rp = tile_ptr + lr * 4u;
        int8x16_t w0 = vcombine_s8(vld1_s8(qmx_lut_binary_s8_8[rp[0]]),
                                   vld1_s8(qmx_lut_binary_s8_8[rp[1]]));
        int8x16_t w1 = vcombine_s8(vld1_s8(qmx_lut_binary_s8_8[rp[2]]),
                                   vld1_s8(qmx_lut_binary_s8_8[rp[3]]));
        return neon_dot_s8x32_vec(w0, w1, x0, x1);
    }
}

static inline float32x4_t neon_load_row_scales4(const uint16_t *sc16, float x_sc) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) || defined(__ARM_FP16_FORMAT_IEEE)
    float16x4_t h4 = vld1_f16((const __fp16 *)(const void *)sc16);
    return vmulq_n_f32(vcvt_f32_f16(h4), x_sc);
#else
    float s[4];
    s[0] = qmx_fp16_to_fp32(sc16[0]);
    s[1] = qmx_fp16_to_fp32(sc16[1]);
    s[2] = qmx_fp16_to_fp32(sc16[2]);
    s[3] = qmx_fp16_to_fp32(sc16[3]);
    return vmulq_n_f32(vld1q_f32(s), x_sc);
#endif
}

static void neon_matmul_tiled(float         * QMX_RESTRICT out,
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
    const uint8_t *base_data = (const uint8_t *)w->data;
    const uint16_t *scales_f16 = (const uint16_t *)(const void *)w->scales;

    uint8x16_t mask0f = vdupq_n_u8(0x0F);
    int8x16_t bias8 = vdupq_n_s8(8);
    uint8x8_t mask03_8 = vdup_n_u8(0x03);
    uint8x16_t one_u8 = vdupq_n_u8(1);
    uint8x16_t two_u8 = vdupq_n_u8(2);

    QMX_ALIGNED(16) float tile_acc[QMX_TILE_ROWS];

    for (uint32_t tr = 0; tr < n_tr; tr++) {
        uint32_t r_base = tr * QMX_TILE_ROWS;
        uint32_t rem_rows = (r_base < orig_rows) ? (orig_rows - r_base) : 0u;
        if (rem_rows > QMX_TILE_ROWS) rem_rows = QMX_TILE_ROWS;

        for (uint32_t lr = 0; lr < QMX_TILE_ROWS; lr++) {
            tile_acc[lr] = (accumulate && (lr < rem_rows)) ? out[r_base + lr] : 0.0f;
        }

        float32x4_t vacc0 = vld1q_f32(tile_acc + 0u);
        float32x4_t vacc1 = vld1q_f32(tile_acc + 4u);
        float32x4_t vacc2 = vld1q_f32(tile_acc + 8u);
        float32x4_t vacc3 = vld1q_f32(tile_acc + 12u);

        for (uint32_t tc = 0; tc < n_tc; tc++) {
            uint32_t tile_idx = tr * n_tc + tc;
            float x_sc = x_scales[tc];
            float32x4_t v_scale = is_row_scaled ? vdupq_n_f32(0.0f) : vdupq_n_f32(w->scales[tile_idx] * x_sc);
            const uint16_t *row_sc = is_row_scaled ? (scales_f16 + (size_t)tile_idx * QMX_TILE_ROWS) : NULL;
            const uint8_t *tile_ptr = base_data + (size_t)tile_idx * tile_bytes;
            const int8_t *xp = xq + (size_t)tc * QMX_TILE_COLS;
            int8x16_t x0 = vld1q_s8(xp);
            int8x16_t x1 = vld1q_s8(xp + 16u);

            if (rem_rows > 0u) {
                int32x4_t r0 = neon_dot_row_vec(tile_ptr, 0u, base_qtype, x0, x1, mask0f, bias8, mask03_8, one_u8, two_u8);
                int32x4_t r1 = neon_dot_row_vec(tile_ptr, 1u, base_qtype, x0, x1, mask0f, bias8, mask03_8, one_u8, two_u8);
                int32x4_t r2 = neon_dot_row_vec(tile_ptr, 2u, base_qtype, x0, x1, mask0f, bias8, mask03_8, one_u8, two_u8);
                int32x4_t r3 = neon_dot_row_vec(tile_ptr, 3u, base_qtype, x0, x1, mask0f, bias8, mask03_8, one_u8, two_u8);
                float32x4_t vs = is_row_scaled ? neon_load_row_scales4(row_sc + 0u, x_sc) : v_scale;
                vacc0 = vmlaq_f32(vacc0, vcvtq_f32_s32(neon_hsum4_s32(r0, r1, r2, r3)), vs);
            }
            if (rem_rows > 4u) {
                int32x4_t r0 = neon_dot_row_vec(tile_ptr, 4u, base_qtype, x0, x1, mask0f, bias8, mask03_8, one_u8, two_u8);
                int32x4_t r1 = neon_dot_row_vec(tile_ptr, 5u, base_qtype, x0, x1, mask0f, bias8, mask03_8, one_u8, two_u8);
                int32x4_t r2 = neon_dot_row_vec(tile_ptr, 6u, base_qtype, x0, x1, mask0f, bias8, mask03_8, one_u8, two_u8);
                int32x4_t r3 = neon_dot_row_vec(tile_ptr, 7u, base_qtype, x0, x1, mask0f, bias8, mask03_8, one_u8, two_u8);
                float32x4_t vs = is_row_scaled ? neon_load_row_scales4(row_sc + 4u, x_sc) : v_scale;
                vacc1 = vmlaq_f32(vacc1, vcvtq_f32_s32(neon_hsum4_s32(r0, r1, r2, r3)), vs);
            }
            if (rem_rows > 8u) {
                int32x4_t r0 = neon_dot_row_vec(tile_ptr, 8u,  base_qtype, x0, x1, mask0f, bias8, mask03_8, one_u8, two_u8);
                int32x4_t r1 = neon_dot_row_vec(tile_ptr, 9u,  base_qtype, x0, x1, mask0f, bias8, mask03_8, one_u8, two_u8);
                int32x4_t r2 = neon_dot_row_vec(tile_ptr, 10u, base_qtype, x0, x1, mask0f, bias8, mask03_8, one_u8, two_u8);
                int32x4_t r3 = neon_dot_row_vec(tile_ptr, 11u, base_qtype, x0, x1, mask0f, bias8, mask03_8, one_u8, two_u8);
                float32x4_t vs = is_row_scaled ? neon_load_row_scales4(row_sc + 8u, x_sc) : v_scale;
                vacc2 = vmlaq_f32(vacc2, vcvtq_f32_s32(neon_hsum4_s32(r0, r1, r2, r3)), vs);
            }
            if (rem_rows > 12u) {
                int32x4_t r0 = neon_dot_row_vec(tile_ptr, 12u, base_qtype, x0, x1, mask0f, bias8, mask03_8, one_u8, two_u8);
                int32x4_t r1 = neon_dot_row_vec(tile_ptr, 13u, base_qtype, x0, x1, mask0f, bias8, mask03_8, one_u8, two_u8);
                int32x4_t r2 = neon_dot_row_vec(tile_ptr, 14u, base_qtype, x0, x1, mask0f, bias8, mask03_8, one_u8, two_u8);
                int32x4_t r3 = neon_dot_row_vec(tile_ptr, 15u, base_qtype, x0, x1, mask0f, bias8, mask03_8, one_u8, two_u8);
                float32x4_t vs = is_row_scaled ? neon_load_row_scales4(row_sc + 12u, x_sc) : v_scale;
                vacc3 = vmlaq_f32(vacc3, vcvtq_f32_s32(neon_hsum4_s32(r0, r1, r2, r3)), vs);
            }
        }

        vst1q_f32(tile_acc + 0u,  vacc0);
        vst1q_f32(tile_acc + 4u,  vacc1);
        vst1q_f32(tile_acc + 8u,  vacc2);
        vst1q_f32(tile_acc + 12u, vacc3);

        for (uint32_t lr = 0; lr < rem_rows; lr++) {
            out[r_base + lr] = tile_acc[lr];
        }
    }
}

static void neon_rmsnorm(float       * QMX_RESTRICT out,
                         const float * QMX_RESTRICT x,
                         const float * QMX_RESTRICT weight,
                         uint32_t size,
                         float eps) {
    float32x4_t sum4 = vdupq_n_f32(0.0f);
    uint32_t i = 0;
    for (; i + 3u < size; i += 4u) {
        float32x4_t v = vld1q_f32(x + i);
        sum4 = vmlaq_f32(sum4, v, v);
    }
    float ss = neon_hsum_f32(sum4);
    for (; i < size; i++) ss += x[i] * x[i];
    float inv_rms = 1.0f / qmx_sqrtf(ss / (float)size + eps);
    float32x4_t v_inv = vdupq_n_f32(inv_rms);
    i = 0;
    for (; i + 3u < size; i += 4u) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t vw = vld1q_f32(weight + i);
        vst1q_f32(out + i, vmulq_f32(vmulq_f32(vx, v_inv), vw));
    }
    for (; i < size; i++) out[i] = weight[i] * (inv_rms * x[i]);
}

static float neon_dot_f32(const float * QMX_RESTRICT a,
                          const float * QMX_RESTRICT b,
                          uint32_t n) {
    float32x4_t sum4 = vdupq_n_f32(0.0f);
    uint32_t i = 0;
    for (; i + 3u < n; i += 4u) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        sum4 = vmlaq_f32(sum4, va, vb);
    }
    float sum = neon_hsum_f32(sum4);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

static void neon_saxpy_f32(float       * QMX_RESTRICT y,
                           float a,
                           const float * QMX_RESTRICT x,
                           uint32_t n) {
    float32x4_t va = vdupq_n_f32(a);
    uint32_t i = 0;
    for (; i + 3u < n; i += 4u) {
        float32x4_t vy = vld1q_f32(y + i);
        float32x4_t vx = vld1q_f32(x + i);
        vst1q_f32(y + i, vmlaq_f32(vy, va, vx));
    }
    for (; i < n; i++) y[i] += a * x[i];
}

static int32_t neon_dot_q8(const int8_t  * QMX_RESTRICT q8,
                           const int16_t * QMX_RESTRICT q16,
                           const int8_t  * QMX_RESTRICT k8,
                           uint32_t n) {
    (void)q16;
    int32x4_t acc = vdupq_n_s32(0);
    uint32_t i = 0;
    for (; i + 15u < n; i += 16u) {
        int8x16_t vq = vld1q_s8(q8 + i);
        int8x16_t vk = vld1q_s8(k8 + i);
        int16x8_t p_lo = vmull_s8(vget_low_s8(vq), vget_low_s8(vk));
        int16x8_t p_hi = vmull_s8(vget_high_s8(vq), vget_high_s8(vk));
        acc = vpadalq_s16(acc, p_lo);
        acc = vpadalq_s16(acc, p_hi);
    }
    int32_t sum = neon_hsum_s32(acc);
    for (; i < n; i++) sum += (int32_t)q8[i] * (int32_t)k8[i];
    return sum;
}

static float neon_dot_f32_f16(const float * QMX_RESTRICT a,
                              const uint16_t * QMX_RESTRICT b,
                              uint32_t n) {
    uint32_t i = 0;
    float sum = 0.0f;
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) || defined(__ARM_FP16_FORMAT_IEEE)
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (; i + 3u < n; i += 4u) {
        float32x4_t vb = vcvt_f32_f16(vld1_f16((const __fp16 *)(const void *)(b + i)));
        acc = vmlaq_f32(acc, vld1q_f32(a + i), vb);
    }
    sum = neon_hsum_f32(acc);
#endif
    for (; i < n; i++) sum += a[i] * qmx_fp16_to_fp32(b[i]);
    return sum;
}

static void neon_saxpy_f32_f16(float * QMX_RESTRICT y,
                               float a,
                               const uint16_t * QMX_RESTRICT x,
                               uint32_t n) {
    uint32_t i = 0;
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) || defined(__ARM_FP16_FORMAT_IEEE)
    float32x4_t va = vdupq_n_f32(a);
    for (; i + 3u < n; i += 4u) {
        float32x4_t vx = vcvt_f32_f16(vld1_f16((const __fp16 *)(const void *)(x + i)));
        vst1q_f32(y + i, vmlaq_f32(vld1q_f32(y + i), va, vx));
    }
#endif
    for (; i < n; i++) y[i] += a * qmx_fp16_to_fp32(x[i]);
}

#if defined(QMX_ENABLE_BACKEND_NEON_DOT) && defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
#define QMX_HAS_NEON_DOT 1
#if !defined(__ARM_FEATURE_DOTPROD)
    #define QMX_NEON_DOT_TARGET __attribute__((target("+dotprod")))
#else
    #define QMX_NEON_DOT_TARGET
#endif
QMX_NEON_DOT_TARGET
static inline int32x4_t neon_sdot_s32(int32x4_t acc, int8x16_t a, int8x16_t b) {
    __asm__ ("sdot %0.4s, %1.16b, %2.16b" : "+w"(acc) : "w"(a), "w"(b));
    return acc;
}
#elif defined(QMX_ENABLE_BACKEND_NEON_DOT) && defined(__ARM_FEATURE_DOTPROD)
#define QMX_HAS_NEON_DOT 1
#define QMX_NEON_DOT_TARGET
static inline int32x4_t neon_sdot_s32(int32x4_t acc, int8x16_t a, int8x16_t b) {
    return vdotq_s32(acc, a, b);
}
#else
#define QMX_HAS_NEON_DOT 0
#endif

#if QMX_HAS_NEON_DOT
QMX_NEON_DOT_TARGET
static inline int32x4_t neon_dot_s8x32_sdot(int8x16_t w0, int8x16_t w1,
                                            int8x16_t x0, int8x16_t x1) {
    int32x4_t acc = neon_sdot_s32(vdupq_n_s32(0), w0, x0);
    return neon_sdot_s32(acc, w1, x1);
}

QMX_NEON_DOT_TARGET
static inline int32x4_t neon_dot_row_sdot(const uint8_t *tile_ptr, uint32_t lr, uint8_t qtype,
                                          int8x16_t x0, int8x16_t x1,
                                          uint8x16_t mask0f, int8x16_t bias8,
                                          uint8x8_t mask03_8, uint8x16_t one_u8, uint8x16_t two_u8) {
    if (qtype == QMX_QUANT_Q8_SYM) {
        const int8_t *rp = (const int8_t *)(tile_ptr + lr * 32u);
        return neon_dot_s8x32_sdot(vld1q_s8(rp), vld1q_s8(rp + 16u), x0, x1);
    } else if (qtype == QMX_QUANT_Q4_SYM) {
        const uint8_t *rp = tile_ptr + lr * 16u;
        uint8x16_t raw = vld1q_u8(rp);
        uint8x16x2_t zipped = vzipq_u8(vandq_u8(raw, mask0f), vshrq_n_u8(raw, 4));
        int8x16_t w0 = vsubq_s8(vreinterpretq_s8_u8(zipped.val[0]), bias8);
        int8x16_t w1 = vsubq_s8(vreinterpretq_s8_u8(zipped.val[1]), bias8);
        return neon_dot_s8x32_sdot(w0, w1, x0, x1);
    } else if (qtype == QMX_QUANT_TERNARY_158) {
        const uint8_t *rp = tile_ptr + lr * 8u;
        uint8x8_t raw = vld1_u8(rp);
        uint8x8_t c0 = vand_u8(raw, mask03_8);
        uint8x8_t c1 = vand_u8(vshr_n_u8(raw, 2), mask03_8);
        uint8x8_t c2 = vand_u8(vshr_n_u8(raw, 4), mask03_8);
        uint8x8_t c3 = vshr_n_u8(raw, 6);
        uint8x8x2_t z02 = vzip_u8(c0, c2);
        uint8x8x2_t z13 = vzip_u8(c1, c3);
        uint8x8x2_t lo16 = vzip_u8(z02.val[0], z13.val[0]);
        uint8x8x2_t hi16 = vzip_u8(z02.val[1], z13.val[1]);
        uint8x16_t u0 = vcombine_u8(lo16.val[0], lo16.val[1]);
        uint8x16_t u1 = vcombine_u8(hi16.val[0], hi16.val[1]);
        int8x16_t w0 = vsubq_s8(vreinterpretq_s8_u8(vceqq_u8(u0, two_u8)),
                                vreinterpretq_s8_u8(vceqq_u8(u0, one_u8)));
        int8x16_t w1 = vsubq_s8(vreinterpretq_s8_u8(vceqq_u8(u1, two_u8)),
                                vreinterpretq_s8_u8(vceqq_u8(u1, one_u8)));
        return neon_dot_s8x32_sdot(w0, w1, x0, x1);
    } else {
        const uint8_t *rp = tile_ptr + lr * 4u;
        int8x16_t w0 = vcombine_s8(vld1_s8(qmx_lut_binary_s8_8[rp[0]]),
                                   vld1_s8(qmx_lut_binary_s8_8[rp[1]]));
        int8x16_t w1 = vcombine_s8(vld1_s8(qmx_lut_binary_s8_8[rp[2]]),
                                   vld1_s8(qmx_lut_binary_s8_8[rp[3]]));
        return neon_dot_s8x32_sdot(w0, w1, x0, x1);
    }
}

QMX_NEON_DOT_TARGET
static void neon_dot_matmul_tiled(float         * QMX_RESTRICT out,
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
    const uint8_t *base_data = (const uint8_t *)w->data;
    const uint16_t *scales_f16 = (const uint16_t *)(const void *)w->scales;

    uint8x16_t mask0f = vdupq_n_u8(0x0F);
    int8x16_t bias8 = vdupq_n_s8(8);
    uint8x8_t mask03_8 = vdup_n_u8(0x03);
    uint8x16_t one_u8 = vdupq_n_u8(1);
    uint8x16_t two_u8 = vdupq_n_u8(2);

    QMX_ALIGNED(16) float tile_acc[QMX_TILE_ROWS];

    for (uint32_t tr = 0; tr < n_tr; tr++) {
        uint32_t r_base = tr * QMX_TILE_ROWS;
        uint32_t rem_rows = (r_base < orig_rows) ? (orig_rows - r_base) : 0u;
        if (rem_rows > QMX_TILE_ROWS) rem_rows = QMX_TILE_ROWS;

        for (uint32_t lr = 0; lr < QMX_TILE_ROWS; lr++) {
            tile_acc[lr] = (accumulate && (lr < rem_rows)) ? out[r_base + lr] : 0.0f;
        }

        float32x4_t vacc0 = vld1q_f32(tile_acc + 0u);
        float32x4_t vacc1 = vld1q_f32(tile_acc + 4u);
        float32x4_t vacc2 = vld1q_f32(tile_acc + 8u);
        float32x4_t vacc3 = vld1q_f32(tile_acc + 12u);

        for (uint32_t tc = 0; tc < n_tc; tc++) {
            uint32_t tile_idx = tr * n_tc + tc;
            float x_sc = x_scales[tc];
            float32x4_t v_scale = is_row_scaled ? vdupq_n_f32(0.0f) : vdupq_n_f32(w->scales[tile_idx] * x_sc);
            const uint16_t *row_sc = is_row_scaled ? (scales_f16 + (size_t)tile_idx * QMX_TILE_ROWS) : NULL;
            const uint8_t *tile_ptr = base_data + (size_t)tile_idx * tile_bytes;
            const int8_t *xp = xq + (size_t)tc * QMX_TILE_COLS;
            int8x16_t x0 = vld1q_s8(xp);
            int8x16_t x1 = vld1q_s8(xp + 16u);

            if (rem_rows > 0u) {
                int32x4_t r0 = neon_dot_row_sdot(tile_ptr, 0u, base_qtype, x0, x1, mask0f, bias8, mask03_8, one_u8, two_u8);
                int32x4_t r1 = neon_dot_row_sdot(tile_ptr, 1u, base_qtype, x0, x1, mask0f, bias8, mask03_8, one_u8, two_u8);
                int32x4_t r2 = neon_dot_row_sdot(tile_ptr, 2u, base_qtype, x0, x1, mask0f, bias8, mask03_8, one_u8, two_u8);
                int32x4_t r3 = neon_dot_row_sdot(tile_ptr, 3u, base_qtype, x0, x1, mask0f, bias8, mask03_8, one_u8, two_u8);
                float32x4_t vs = is_row_scaled ? neon_load_row_scales4(row_sc + 0u, x_sc) : v_scale;
                vacc0 = vmlaq_f32(vacc0, vcvtq_f32_s32(neon_hsum4_s32(r0, r1, r2, r3)), vs);
            }
            if (rem_rows > 4u) {
                int32x4_t r0 = neon_dot_row_sdot(tile_ptr, 4u, base_qtype, x0, x1, mask0f, bias8, mask03_8, one_u8, two_u8);
                int32x4_t r1 = neon_dot_row_sdot(tile_ptr, 5u, base_qtype, x0, x1, mask0f, bias8, mask03_8, one_u8, two_u8);
                int32x4_t r2 = neon_dot_row_sdot(tile_ptr, 6u, base_qtype, x0, x1, mask0f, bias8, mask03_8, one_u8, two_u8);
                int32x4_t r3 = neon_dot_row_sdot(tile_ptr, 7u, base_qtype, x0, x1, mask0f, bias8, mask03_8, one_u8, two_u8);
                float32x4_t vs = is_row_scaled ? neon_load_row_scales4(row_sc + 4u, x_sc) : v_scale;
                vacc1 = vmlaq_f32(vacc1, vcvtq_f32_s32(neon_hsum4_s32(r0, r1, r2, r3)), vs);
            }
            if (rem_rows > 8u) {
                int32x4_t r0 = neon_dot_row_sdot(tile_ptr, 8u,  base_qtype, x0, x1, mask0f, bias8, mask03_8, one_u8, two_u8);
                int32x4_t r1 = neon_dot_row_sdot(tile_ptr, 9u,  base_qtype, x0, x1, mask0f, bias8, mask03_8, one_u8, two_u8);
                int32x4_t r2 = neon_dot_row_sdot(tile_ptr, 10u, base_qtype, x0, x1, mask0f, bias8, mask03_8, one_u8, two_u8);
                int32x4_t r3 = neon_dot_row_sdot(tile_ptr, 11u, base_qtype, x0, x1, mask0f, bias8, mask03_8, one_u8, two_u8);
                float32x4_t vs = is_row_scaled ? neon_load_row_scales4(row_sc + 8u, x_sc) : v_scale;
                vacc2 = vmlaq_f32(vacc2, vcvtq_f32_s32(neon_hsum4_s32(r0, r1, r2, r3)), vs);
            }
            if (rem_rows > 12u) {
                int32x4_t r0 = neon_dot_row_sdot(tile_ptr, 12u, base_qtype, x0, x1, mask0f, bias8, mask03_8, one_u8, two_u8);
                int32x4_t r1 = neon_dot_row_sdot(tile_ptr, 13u, base_qtype, x0, x1, mask0f, bias8, mask03_8, one_u8, two_u8);
                int32x4_t r2 = neon_dot_row_sdot(tile_ptr, 14u, base_qtype, x0, x1, mask0f, bias8, mask03_8, one_u8, two_u8);
                int32x4_t r3 = neon_dot_row_sdot(tile_ptr, 15u, base_qtype, x0, x1, mask0f, bias8, mask03_8, one_u8, two_u8);
                float32x4_t vs = is_row_scaled ? neon_load_row_scales4(row_sc + 12u, x_sc) : v_scale;
                vacc3 = vmlaq_f32(vacc3, vcvtq_f32_s32(neon_hsum4_s32(r0, r1, r2, r3)), vs);
            }
        }

        vst1q_f32(tile_acc + 0u,  vacc0);
        vst1q_f32(tile_acc + 4u,  vacc1);
        vst1q_f32(tile_acc + 8u,  vacc2);
        vst1q_f32(tile_acc + 12u, vacc3);

        for (uint32_t lr = 0; lr < rem_rows; lr++) {
            out[r_base + lr] = tile_acc[lr];
        }
    }
}

QMX_NEON_DOT_TARGET
static int32_t neon_dot_dot_q8(const int8_t  * QMX_RESTRICT q8,
                               const int16_t * QMX_RESTRICT q16,
                               const int8_t  * QMX_RESTRICT k8,
                               uint32_t n) {
    (void)q16;
    int32x4_t acc = vdupq_n_s32(0);
    uint32_t i = 0;
    for (; i + 15u < n; i += 16u) {
        acc = neon_sdot_s32(acc, vld1q_s8(q8 + i), vld1q_s8(k8 + i));
    }
    int32_t sum = neon_hsum_s32(acc);
    for (; i < n; i++) sum += (int32_t)q8[i] * (int32_t)k8[i];
    return sum;
}

static void neon_f16_to_f32(float * QMX_RESTRICT dst, const uint16_t * QMX_RESTRICT src, uint32_t n) {
    uint32_t i = 0;
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) || defined(__ARM_FP16_FORMAT_IEEE)
    for (; i + 3u < n; i += 4u) {
        vst1q_f32(dst + i, vcvt_f32_f16(vld1_f16((const __fp16 *)(const void *)(src + i))));
    }
#endif
    for (; i < n; i++) dst[i] = qmx_fp16_to_fp32(src[i]);
}

static void neon_f32_to_f16(uint16_t * QMX_RESTRICT dst, const float * QMX_RESTRICT src, uint32_t n) {
    uint32_t i = 0;
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) || defined(__ARM_FP16_FORMAT_IEEE)
    for (; i + 3u < n; i += 4u) {
        vst1_f16((__fp16 *)(void *)(dst + i), vcvt_f16_f32(vld1q_f32(src + i)));
    }
#endif
    for (; i < n; i++) dst[i] = qmx_fp32_to_fp16(src[i]);
}

const QmxBackendOps qmx_ops_neon_dot = {
    "NEON-Dot",
    neon_prepare_act,
    neon_dot_matmul_tiled,
    neon_rmsnorm,
    neon_dot_f32,
    neon_saxpy_f32,
    neon_dot_dot_q8,
    NULL,
    qmx_swiglu_quant_scalar,
    neon_f16_to_f32,
    neon_f32_to_f16,
    NULL,
    NULL,
    NULL,
    neon_dot_f32_f16,
    neon_saxpy_f32_f16
};
#endif

const QmxBackendOps qmx_ops_neon = {
    "NEON",
    neon_prepare_act,
    neon_matmul_tiled,
    neon_rmsnorm,
    neon_dot_f32,
    neon_saxpy_f32,
    neon_dot_q8,
    NULL,
    qmx_swiglu_quant_scalar,
    neon_f16_to_f32,
    neon_f32_to_f16,
    NULL,
    NULL,
    NULL,
    neon_dot_f32_f16,
    neon_saxpy_f32_f16
};
#endif
