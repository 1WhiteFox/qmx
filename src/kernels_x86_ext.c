/*
 * Copyright 2026 1WhiteFox
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/qmx_kernels.h"
#include <string.h>

#if defined(QMX_ENABLE_BACKEND_X86_EXT) && !defined(__DJGPP__) && \
    (((defined(__i386__) || defined(__x86_64__)) && (defined(__GNUC__) || defined(__clang__))) || \
     (defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))))
#include <immintrin.h>

#if defined(__GNUC__) || defined(__clang__)
    #define QMX_X86_TARGET(t) __attribute__((target(t)))
#else
    #define QMX_X86_TARGET(t)
#endif

#ifndef QMX_MSVC_ZEROUPPER
    #if defined(_MSC_VER)
        #define QMX_MSVC_ZEROUPPER() _mm256_zeroupper()
    #else
        #define QMX_MSVC_ZEROUPPER() ((void)0)
    #endif
#endif

static void x86_ext_prepare_act_q8(const float * QMX_RESTRICT x,
                                   int8_t      * QMX_RESTRICT xq,
                                   int16_t     * QMX_RESTRICT xq16,
                                   float       * QMX_RESTRICT x_scales,
                                   uint32_t cols) {
    (void)xq16;
    qmx_quantize_activations_q8(x, xq, x_scales, cols);
}

QMX_X86_TARGET("sse2")
static void x86_ext_prepare_act_q8_q16(const float * QMX_RESTRICT x,
                                       int8_t      * QMX_RESTRICT xq,
                                       int16_t     * QMX_RESTRICT xq16,
                                       float       * QMX_RESTRICT x_scales,
                                       uint32_t cols) {
    qmx_quantize_activations_q8(x, xq, x_scales, cols);
    uint32_t pad_cols = (uint32_t)qmx_align_up(cols, QMX_TILE_COLS);
    __m128i zero = _mm_setzero_si128();
    for (uint32_t i = 0; i < pad_cols; i += 16u) {
        __m128i v8 = _mm_loadu_si128((const __m128i *)(xq + i));
        __m128i sign = _mm_cmpgt_epi8(zero, v8);
        _mm_storeu_si128((__m128i *)(xq16 + i), _mm_unpacklo_epi8(v8, sign));
        _mm_storeu_si128((__m128i *)(xq16 + i + 8u), _mm_unpackhi_epi8(v8, sign));
    }
}

/* ============================================================================
 * 1. SSE (Pentium III / Athlon XP: 128-bit XMM FP32 + Prefetch)
 * ============================================================================ */

QMX_X86_TARGET("sse")
static inline float sse1_hsum_ps(__m128 v) {
    __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
    v = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, v);
    v = _mm_add_ss(v, shuf);
    return _mm_cvtss_f32(v);
}

QMX_X86_TARGET("sse")
static void sse1_rmsnorm(float       * QMX_RESTRICT out,
                         const float * QMX_RESTRICT x,
                         const float * QMX_RESTRICT weight,
                         uint32_t size,
                         float eps) {
    __m128 sum4 = _mm_setzero_ps();
    uint32_t i = 0;
    for (; i + 3u < size; i += 4u) {
        __m128 v = _mm_loadu_ps(x + i);
        sum4 = _mm_add_ps(sum4, _mm_mul_ps(v, v));
    }
    float ss = sse1_hsum_ps(sum4);
    for (; i < size; i++) ss += x[i] * x[i];
    float mean_eps = ss / (float)size + eps;
    __m128 v_me = _mm_set_ss(mean_eps);
    __m128 r0 = _mm_rsqrt_ss(v_me);
    /* One Newton-Raphson refinement step for full FP32 accuracy */
    __m128 half = _mm_set_ss(0.5f);
    __m128 three = _mm_set_ss(3.0f);
    __m128 r1 = _mm_mul_ss(_mm_mul_ss(r0, half), _mm_sub_ss(three, _mm_mul_ss(v_me, _mm_mul_ss(r0, r0))));
    float inv_rms = _mm_cvtss_f32(r1);
    __m128 v_inv = _mm_set1_ps(inv_rms);
    i = 0;
    for (; i + 3u < size; i += 4u) {
        __m128 vx = _mm_loadu_ps(x + i);
        __m128 vw = _mm_loadu_ps(weight + i);
        _mm_storeu_ps(out + i, _mm_mul_ps(_mm_mul_ps(vx, v_inv), vw));
    }
    for (; i < size; i++) out[i] = weight[i] * (inv_rms * x[i]);
}

QMX_X86_TARGET("sse")
static float sse1_dot_f32(const float * QMX_RESTRICT a,
                          const float * QMX_RESTRICT b,
                          uint32_t n) {
    __m128 sum4 = _mm_setzero_ps();
    uint32_t i = 0;
    for (; i + 3u < n; i += 4u) {
        sum4 = _mm_add_ps(sum4, _mm_mul_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
    }
    float sum = sse1_hsum_ps(sum4);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

QMX_X86_TARGET("sse")
static void sse1_saxpy_f32(float       * QMX_RESTRICT y,
                           float a,
                           const float * QMX_RESTRICT x,
                           uint32_t n) {
    __m128 va = _mm_set1_ps(a);
    uint32_t i = 0;
    for (; i + 3u < n; i += 4u) {
        _mm_storeu_ps(y + i, _mm_add_ps(_mm_loadu_ps(y + i), _mm_mul_ps(va, _mm_loadu_ps(x + i))));
    }
    for (; i < n; i++) y[i] += a * x[i];
}

QMX_X86_TARGET("sse")
static void sse1_matmul_tiled(float         * QMX_RESTRICT out,
                              const int8_t  * QMX_RESTRICT xq,
                              const int16_t * QMX_RESTRICT xq16,
                              const float   * QMX_RESTRICT x_scales,
                              const QmxTensor *w,
                              int accumulate) {
#if defined(QMX_ENABLE_BACKEND_MMX)
    qmx_ops_mmx.matmul_tiled(out, xq, xq16, x_scales, w, accumulate);
#else
    (void)xq16;
    qmx_matmul_tiled_scalar(out, xq, x_scales, w, accumulate);
#endif
}

static void sse1_prepare_act(const float * QMX_RESTRICT x,
                             int8_t      * QMX_RESTRICT xq,
                             int16_t     * QMX_RESTRICT xq16,
                             float       * QMX_RESTRICT x_scales,
                             uint32_t cols) {
#if defined(QMX_ENABLE_BACKEND_MMX)
    qmx_ops_mmx.prepare_act(x, xq, xq16, x_scales, cols);
#else
    (void)xq16;
    qmx_quantize_activations_q8(x, xq, x_scales, cols);
#endif
}

static int32_t sse1_dot_q8(const int8_t  * QMX_RESTRICT q8,
                           const int16_t * QMX_RESTRICT q16,
                           const int8_t  * QMX_RESTRICT k8,
                           uint32_t n) {
#if defined(QMX_ENABLE_BACKEND_MMX)
    return qmx_ops_mmx.dot_q8(q8, q16, k8, n);
#else
    return qmx_ops_scalar.dot_q8(q8, q16, k8, n);
#endif
}

/* ============================================================================
 * 2. SSE3 (Prescott / Yonah: lddqu + haddps horizontal reduction, NO SSSE3!)
 * ============================================================================ */

QMX_X86_TARGET("sse3")
static inline float sse3_hsum_ps(__m128 v) {
    v = _mm_hadd_ps(v, v);
    v = _mm_hadd_ps(v, v);
    return _mm_cvtss_f32(v);
}

QMX_X86_TARGET("sse3")
static inline __m128 sse3_load_row_scales4(const uint16_t *sc16, float x_sc) {
    __m128 s = _mm_set_ps(qmx_fp16_to_fp32(sc16[3]),
                          qmx_fp16_to_fp32(sc16[2]),
                          qmx_fp16_to_fp32(sc16[1]),
                          qmx_fp16_to_fp32(sc16[0]));
    return _mm_mul_ps(s, _mm_set1_ps(x_sc));
}

QMX_X86_TARGET("sse3")
static inline __m128i sse3_dot_row_vec(const uint8_t *tile_ptr, uint32_t lr, uint8_t base_qtype,
                                       __m128i x0, __m128i x1, __m128i x2, __m128i x3,
                                       __m128i zero, __m128i mask0f, __m128i bias8) {
    if (base_qtype == QMX_QUANT_Q8_SYM) {
        const uint8_t *rp = tile_ptr + lr * 32u;
        __m128i w0 = _mm_lddqu_si128((const __m128i *)rp);
        __m128i w1 = _mm_lddqu_si128((const __m128i *)(rp + 16u));
        __m128i s0 = _mm_cmpgt_epi8(zero, w0);
        __m128i s1 = _mm_cmpgt_epi8(zero, w1);
        __m128i acc = _mm_madd_epi16(_mm_unpacklo_epi8(w0, s0), x0);
        acc = _mm_add_epi32(acc, _mm_madd_epi16(_mm_unpackhi_epi8(w0, s0), x1));
        acc = _mm_add_epi32(acc, _mm_madd_epi16(_mm_unpacklo_epi8(w1, s1), x2));
        return _mm_add_epi32(acc, _mm_madd_epi16(_mm_unpackhi_epi8(w1, s1), x3));
    } else if (base_qtype == QMX_QUANT_Q4_SYM) {
        const uint8_t *rp = tile_ptr + lr * 16u;
        __m128i packed = _mm_lddqu_si128((const __m128i *)rp);
        __m128i lo_nib = _mm_and_si128(packed, mask0f);
        __m128i hi_nib = _mm_and_si128(_mm_srli_epi16(packed, 4), mask0f);
        __m128i u0 = _mm_unpacklo_epi8(lo_nib, hi_nib);
        __m128i u1 = _mm_unpackhi_epi8(lo_nib, hi_nib);
        __m128i w0 = _mm_sub_epi16(_mm_unpacklo_epi8(u0, zero), bias8);
        __m128i w1 = _mm_sub_epi16(_mm_unpackhi_epi8(u0, zero), bias8);
        __m128i w2 = _mm_sub_epi16(_mm_unpacklo_epi8(u1, zero), bias8);
        __m128i w3 = _mm_sub_epi16(_mm_unpackhi_epi8(u1, zero), bias8);
        __m128i acc = _mm_madd_epi16(w0, x0);
        acc = _mm_add_epi32(acc, _mm_madd_epi16(w1, x1));
        acc = _mm_add_epi32(acc, _mm_madd_epi16(w2, x2));
        return _mm_add_epi32(acc, _mm_madd_epi16(w3, x3));
    } else if (base_qtype == QMX_QUANT_TERNARY_158) {
        const uint8_t *rp = tile_ptr + lr * 8u;
        __m128i w0 = _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i *)qmx_lut_ternary_i16[rp[0]]),
                                        _mm_loadl_epi64((const __m128i *)qmx_lut_ternary_i16[rp[1]]));
        __m128i w1 = _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i *)qmx_lut_ternary_i16[rp[2]]),
                                        _mm_loadl_epi64((const __m128i *)qmx_lut_ternary_i16[rp[3]]));
        __m128i w2 = _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i *)qmx_lut_ternary_i16[rp[4]]),
                                        _mm_loadl_epi64((const __m128i *)qmx_lut_ternary_i16[rp[5]]));
        __m128i w3 = _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i *)qmx_lut_ternary_i16[rp[6]]),
                                        _mm_loadl_epi64((const __m128i *)qmx_lut_ternary_i16[rp[7]]));
        __m128i acc = _mm_madd_epi16(w0, x0);
        acc = _mm_add_epi32(acc, _mm_madd_epi16(w1, x1));
        acc = _mm_add_epi32(acc, _mm_madd_epi16(w2, x2));
        return _mm_add_epi32(acc, _mm_madd_epi16(w3, x3));
    } else {
        const uint8_t *rp = tile_ptr + lr * 4u;
        __m128i w0 = _mm_load_si128((const __m128i *)qmx_lut_binary_i16_8[rp[0]]);
        __m128i w1 = _mm_load_si128((const __m128i *)qmx_lut_binary_i16_8[rp[1]]);
        __m128i w2 = _mm_load_si128((const __m128i *)qmx_lut_binary_i16_8[rp[2]]);
        __m128i w3 = _mm_load_si128((const __m128i *)qmx_lut_binary_i16_8[rp[3]]);
        __m128i acc = _mm_madd_epi16(w0, x0);
        acc = _mm_add_epi32(acc, _mm_madd_epi16(w1, x1));
        acc = _mm_add_epi32(acc, _mm_madd_epi16(w2, x2));
        return _mm_add_epi32(acc, _mm_madd_epi16(w3, x3));
    }
}

QMX_X86_TARGET("sse3")
static void sse3_matmul_tiled(float         * QMX_RESTRICT out,
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
    const uint8_t *base_data = (const uint8_t *)w->data;
    const uint16_t *scales_f16 = (const uint16_t *)(const void *)w->scales;

    __m128i zero = _mm_setzero_si128();
    __m128i mask0f = _mm_set1_epi8(0x0F);
    __m128i bias8 = _mm_set1_epi16(8);
    QMX_ALIGNED(16) float tile_acc[QMX_TILE_ROWS];

    for (uint32_t tr = 0; tr < n_tr; tr++) {
        uint32_t r_base = tr * QMX_TILE_ROWS;
        uint32_t rem_rows = (r_base < orig_rows) ? (orig_rows - r_base) : 0u;
        if (rem_rows > QMX_TILE_ROWS) rem_rows = QMX_TILE_ROWS;
        for (uint32_t lr = 0; lr < QMX_TILE_ROWS; lr++) {
            tile_acc[lr] = (accumulate && (lr < rem_rows)) ? out[r_base + lr] : 0.0f;
        }

        __m128 vacc0 = _mm_load_ps(tile_acc + 0u);
        __m128 vacc1 = _mm_load_ps(tile_acc + 4u);
        __m128 vacc2 = _mm_load_ps(tile_acc + 8u);
        __m128 vacc3 = _mm_load_ps(tile_acc + 12u);

        for (uint32_t tc = 0; tc < n_tc; tc++) {
            uint32_t tile_idx = tr * n_tc + tc;
            float x_sc = x_scales[tc];
            __m128 v_scale = is_row_scaled ? _mm_setzero_ps() : _mm_set1_ps(w->scales[tile_idx] * x_sc);
            const uint16_t *row_sc = is_row_scaled ? (scales_f16 + (size_t)tile_idx * QMX_TILE_ROWS) : NULL;
            const uint8_t *tile_ptr = base_data + (size_t)tile_idx * tile_bytes;
            const int16_t *xp = xq16 + (size_t)tc * QMX_TILE_COLS;

            __m128i x0 = _mm_lddqu_si128((const __m128i *)(xp + 0u));
            __m128i x1 = _mm_lddqu_si128((const __m128i *)(xp + 8u));
            __m128i x2 = _mm_lddqu_si128((const __m128i *)(xp + 16u));
            __m128i x3 = _mm_lddqu_si128((const __m128i *)(xp + 24u));

            for (uint32_t g = 0; g < 4u; g++) {
                uint32_t base_lr = g * 4u;
                if (rem_rows <= base_lr) break;
                __m128i r0 = sse3_dot_row_vec(tile_ptr, base_lr + 0u, base_qtype, x0, x1, x2, x3, zero, mask0f, bias8);
                __m128i r1 = sse3_dot_row_vec(tile_ptr, base_lr + 1u, base_qtype, x0, x1, x2, x3, zero, mask0f, bias8);
                __m128i r2 = sse3_dot_row_vec(tile_ptr, base_lr + 2u, base_qtype, x0, x1, x2, x3, zero, mask0f, bias8);
                __m128i r3 = sse3_dot_row_vec(tile_ptr, base_lr + 3u, base_qtype, x0, x1, x2, x3, zero, mask0f, bias8);
                __m128i t0 = _mm_unpacklo_epi32(r0, r1);
                __m128i t1 = _mm_unpackhi_epi32(r0, r1);
                __m128i t2 = _mm_unpacklo_epi32(r2, r3);
                __m128i t3 = _mm_unpackhi_epi32(r2, r3);
                __m128i s01 = _mm_add_epi32(t0, t1);
                __m128i s23 = _mm_add_epi32(t2, t3);
                __m128i sum4 = _mm_add_epi32(_mm_unpacklo_epi64(s01, s23), _mm_unpackhi_epi64(s01, s23));
                __m128 vs = is_row_scaled ? sse3_load_row_scales4(row_sc + base_lr, x_sc) : v_scale;
                __m128 d4 = _mm_mul_ps(_mm_cvtepi32_ps(sum4), vs);
                if (g == 0u) vacc0 = _mm_add_ps(vacc0, d4);
                else if (g == 1u) vacc1 = _mm_add_ps(vacc1, d4);
                else if (g == 2u) vacc2 = _mm_add_ps(vacc2, d4);
                else vacc3 = _mm_add_ps(vacc3, d4);
            }
        }

        _mm_store_ps(tile_acc + 0u,  vacc0);
        _mm_store_ps(tile_acc + 4u,  vacc1);
        _mm_store_ps(tile_acc + 8u,  vacc2);
        _mm_store_ps(tile_acc + 12u, vacc3);
        for (uint32_t lr = 0; lr < rem_rows; lr++) out[r_base + lr] = tile_acc[lr];
    }
}

QMX_X86_TARGET("sse3")
static void sse3_rmsnorm(float       * QMX_RESTRICT out,
                         const float * QMX_RESTRICT x,
                         const float * QMX_RESTRICT weight,
                         uint32_t size,
                         float eps) {
    __m128 sum4 = _mm_setzero_ps();
    uint32_t i = 0;
    for (; i + 3u < size; i += 4u) {
        __m128 v = _mm_loadu_ps(x + i);
        sum4 = _mm_add_ps(sum4, _mm_mul_ps(v, v));
    }
    float ss = sse3_hsum_ps(sum4);
    for (; i < size; i++) ss += x[i] * x[i];
    float inv_rms = 1.0f / qmx_sqrtf(ss / (float)size + eps);
    __m128 v_inv = _mm_set1_ps(inv_rms);
    i = 0;
    for (; i + 3u < size; i += 4u) {
        _mm_storeu_ps(out + i, _mm_mul_ps(_mm_mul_ps(_mm_loadu_ps(x + i), v_inv), _mm_loadu_ps(weight + i)));
    }
    for (; i < size; i++) out[i] = weight[i] * (inv_rms * x[i]);
}

QMX_X86_TARGET("sse3")
static float sse3_dot_f32(const float * QMX_RESTRICT a,
                          const float * QMX_RESTRICT b,
                          uint32_t n) {
    __m128 sum4 = _mm_setzero_ps();
    uint32_t i = 0;
    for (; i + 3u < n; i += 4u) {
        sum4 = _mm_add_ps(sum4, _mm_mul_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
    }
    float sum = sse3_hsum_ps(sum4);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

QMX_X86_TARGET("sse3")
static int32_t sse3_dot_q8(const int8_t  * QMX_RESTRICT q8,
                           const int16_t * QMX_RESTRICT q16,
                           const int8_t  * QMX_RESTRICT k8,
                           uint32_t n) {
    (void)q8;
    __m128i zero = _mm_setzero_si128();
    __m128i acc = _mm_setzero_si128();
    uint32_t i = 0;
    for (; i + 15u < n; i += 16u) {
        __m128i kv = _mm_lddqu_si128((const __m128i *)(k8 + i));
        __m128i ks = _mm_cmpgt_epi8(zero, kv);
        __m128i q_lo = _mm_lddqu_si128((const __m128i *)(q16 + i));
        __m128i q_hi = _mm_lddqu_si128((const __m128i *)(q16 + i + 8u));
        acc = _mm_add_epi32(acc, _mm_madd_epi16(_mm_unpacklo_epi8(kv, ks), q_lo));
        acc = _mm_add_epi32(acc, _mm_madd_epi16(_mm_unpackhi_epi8(kv, ks), q_hi));
    }
    __m128i hi64 = _mm_unpackhi_epi64(acc, acc);
    acc = _mm_add_epi32(acc, hi64);
    __m128i hi32 = _mm_shuffle_epi32(acc, _MM_SHUFFLE(1, 1, 1, 1));
    acc = _mm_add_epi32(acc, hi32);
    int32_t sum = _mm_cvtsi128_si32(acc);
    for (; i < n; i++) sum += (int32_t)q16[i] * (int32_t)k8[i];
    return sum;
}

/* ============================================================================
 * 3. SSSE3 (Core 2 / Atom: pshufb + pmaddubsw + psignb + phaddd on 128-bit XMM)
 * ============================================================================ */

QMX_X86_TARGET("ssse3")
static inline __m128 ssse3_load_row_scales4(const uint16_t *sc16, float x_sc) {
    __m128 s = _mm_set_ps(qmx_fp16_to_fp32(sc16[3]),
                          qmx_fp16_to_fp32(sc16[2]),
                          qmx_fp16_to_fp32(sc16[1]),
                          qmx_fp16_to_fp32(sc16[0]));
    return _mm_mul_ps(s, _mm_set1_ps(x_sc));
}

QMX_X86_TARGET("ssse3")
static inline __m128i ssse3_dot_row_vec(const uint8_t *tile_ptr, uint32_t lr, uint8_t base_qtype,
                                        __m128i x0, __m128i x1, __m128i x_bias0, __m128i x_bias1,
                                        __m128i ones16, __m128i mask0f,
                                        __m128i tern_shuf, __m128i tern_m0, __m128i tern_m1,
                                        __m128i tern_m2, __m128i tern_m3, __m128i tern_lut,
                                        __m128i bin_shuf_lo, __m128i bin_shuf_hi,
                                        __m128i bin_bits, __m128i two_u8, __m128i one_u8) {
    if (base_qtype == QMX_QUANT_Q8_SYM) {
        const uint8_t *rp = tile_ptr + lr * 32u;
        __m128i w0 = _mm_lddqu_si128((const __m128i *)rp);
        __m128i w1 = _mm_lddqu_si128((const __m128i *)(rp + 16u));
        __m128i p0 = _mm_maddubs_epi16(_mm_abs_epi8(w0), _mm_sign_epi8(x0, w0));
        __m128i p1 = _mm_maddubs_epi16(_mm_abs_epi8(w1), _mm_sign_epi8(x1, w1));
        return _mm_add_epi32(_mm_madd_epi16(p0, ones16), _mm_madd_epi16(p1, ones16));
    } else if (base_qtype == QMX_QUANT_Q4_SYM) {
        __m128i raw16 = _mm_lddqu_si128((const __m128i *)(tile_ptr + lr * 16u));
        __m128i lo_nib = _mm_and_si128(raw16, mask0f);
        __m128i hi_nib = _mm_and_si128(_mm_srli_epi16(raw16, 4), mask0f);
        __m128i u0 = _mm_unpacklo_epi8(lo_nib, hi_nib);
        __m128i u1 = _mm_unpackhi_epi8(lo_nib, hi_nib);
        __m128i p0 = _mm_sub_epi16(_mm_maddubs_epi16(u0, x0), x_bias0);
        __m128i p1 = _mm_sub_epi16(_mm_maddubs_epi16(u1, x1), x_bias1);
        return _mm_add_epi32(_mm_madd_epi16(p0, ones16), _mm_madd_epi16(p1, ones16));
    } else if (base_qtype == QMX_QUANT_TERNARY_158) {
        const uint8_t *rp = tile_ptr + lr * 8u;
        uint32_t r_lo = 0, r_hi = 0;
        memcpy(&r_lo, rp, 4);
        memcpy(&r_hi, rp + 4, 4);
        __m128i b0 = _mm_shuffle_epi8(_mm_set1_epi32((int)r_lo), tern_shuf);
        __m128i b1 = _mm_shuffle_epi8(_mm_set1_epi32((int)r_hi), tern_shuf);
        __m128i c0 = _mm_or_si128(_mm_or_si128(_mm_and_si128(b0, tern_m0), _mm_and_si128(_mm_srli_epi16(b0, 2), tern_m1)),
                                  _mm_or_si128(_mm_and_si128(_mm_srli_epi16(b0, 4), tern_m2), _mm_and_si128(_mm_srli_epi16(b0, 6), tern_m3)));
        __m128i c1 = _mm_or_si128(_mm_or_si128(_mm_and_si128(b1, tern_m0), _mm_and_si128(_mm_srli_epi16(b1, 2), tern_m1)),
                                  _mm_or_si128(_mm_and_si128(_mm_srli_epi16(b1, 4), tern_m2), _mm_and_si128(_mm_srli_epi16(b1, 6), tern_m3)));
        __m128i w0 = _mm_shuffle_epi8(tern_lut, c0);
        __m128i w1 = _mm_shuffle_epi8(tern_lut, c1);
        __m128i p0 = _mm_maddubs_epi16(_mm_abs_epi8(w0), _mm_sign_epi8(x0, w0));
        __m128i p1 = _mm_maddubs_epi16(_mm_abs_epi8(w1), _mm_sign_epi8(x1, w1));
        return _mm_add_epi32(_mm_madd_epi16(p0, ones16), _mm_madd_epi16(p1, ones16));
    } else {
        const uint8_t *rp = tile_ptr + lr * 4u;
        uint32_t raw32 = 0;
        memcpy(&raw32, rp, 4);
        __m128i vr = _mm_set1_epi32((int)raw32);
        __m128i b0 = _mm_shuffle_epi8(vr, bin_shuf_lo);
        __m128i b1 = _mm_shuffle_epi8(vr, bin_shuf_hi);
        __m128i w0 = _mm_sub_epi8(_mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(b0, bin_bits), bin_bits), two_u8), one_u8);
        __m128i w1 = _mm_sub_epi8(_mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(b1, bin_bits), bin_bits), two_u8), one_u8);
        __m128i p0 = _mm_maddubs_epi16(one_u8, _mm_sign_epi8(x0, w0));
        __m128i p1 = _mm_maddubs_epi16(one_u8, _mm_sign_epi8(x1, w1));
        return _mm_add_epi32(_mm_madd_epi16(p0, ones16), _mm_madd_epi16(p1, ones16));
    }
}

QMX_X86_TARGET("ssse3")
static void ssse3_matmul_tiled(float         * QMX_RESTRICT out,
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

    __m128i ones16 = _mm_set1_epi16(1);
    __m128i mask0f = _mm_set1_epi8(0x0F);
    __m128i eight_u8 = _mm_set1_epi8(8);
    __m128i tern_shuf = _mm_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);
    __m128i tern_m0 = _mm_set1_epi32(0x00000003);
    __m128i tern_m1 = _mm_set1_epi32(0x00000300);
    __m128i tern_m2 = _mm_set1_epi32(0x00030000);
    __m128i tern_m3 = _mm_set1_epi32(0x03000000);
    __m128i tern_lut = _mm_setr_epi8(0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, 0);
    __m128i bin_shuf_lo = _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
    __m128i bin_shuf_hi = _mm_setr_epi8(2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    __m128i bin_bits = _mm_set1_epi64x((long long)0x8040201008040201ULL);
    __m128i two_u8 = _mm_set1_epi8(2);
    __m128i one_u8 = _mm_set1_epi8(1);

    QMX_ALIGNED(16) float tile_acc[QMX_TILE_ROWS];

    for (uint32_t tr = 0; tr < n_tr; tr++) {
        uint32_t r_base = tr * QMX_TILE_ROWS;
        uint32_t rem_rows = (r_base < orig_rows) ? (orig_rows - r_base) : 0u;
        if (rem_rows > QMX_TILE_ROWS) rem_rows = QMX_TILE_ROWS;
        for (uint32_t lr = 0; lr < QMX_TILE_ROWS; lr++) {
            tile_acc[lr] = (accumulate && (lr < rem_rows)) ? out[r_base + lr] : 0.0f;
        }

        __m128 vacc0 = _mm_load_ps(tile_acc + 0u);
        __m128 vacc1 = _mm_load_ps(tile_acc + 4u);
        __m128 vacc2 = _mm_load_ps(tile_acc + 8u);
        __m128 vacc3 = _mm_load_ps(tile_acc + 12u);

        for (uint32_t tc = 0; tc < n_tc; tc++) {
            uint32_t tile_idx = tr * n_tc + tc;
            float x_sc = x_scales[tc];
            __m128 v_scale = is_row_scaled ? _mm_setzero_ps() : _mm_set1_ps(w->scales[tile_idx] * x_sc);
            const uint16_t *row_sc = is_row_scaled ? (scales_f16 + (size_t)tile_idx * QMX_TILE_ROWS) : NULL;
            const uint8_t *tile_ptr = base_data + (size_t)tile_idx * tile_bytes;
            const int8_t *xp = xq + (size_t)tc * QMX_TILE_COLS;
            __m128i x0 = _mm_lddqu_si128((const __m128i *)xp);
            __m128i x1 = _mm_lddqu_si128((const __m128i *)(xp + 16u));
            __m128i xb0 = (base_qtype == QMX_QUANT_Q4_SYM) ? _mm_maddubs_epi16(eight_u8, x0) : _mm_setzero_si128();
            __m128i xb1 = (base_qtype == QMX_QUANT_Q4_SYM) ? _mm_maddubs_epi16(eight_u8, x1) : _mm_setzero_si128();

            for (uint32_t g = 0; g < 4u; g++) {
                uint32_t base_lr = g * 4u;
                if (rem_rows <= base_lr) break;
                __m128i r0 = ssse3_dot_row_vec(tile_ptr, base_lr + 0u, base_qtype, x0, x1, xb0, xb1, ones16, mask0f, tern_shuf, tern_m0, tern_m1, tern_m2, tern_m3, tern_lut, bin_shuf_lo, bin_shuf_hi, bin_bits, two_u8, one_u8);
                __m128i r1 = ssse3_dot_row_vec(tile_ptr, base_lr + 1u, base_qtype, x0, x1, xb0, xb1, ones16, mask0f, tern_shuf, tern_m0, tern_m1, tern_m2, tern_m3, tern_lut, bin_shuf_lo, bin_shuf_hi, bin_bits, two_u8, one_u8);
                __m128i r2 = ssse3_dot_row_vec(tile_ptr, base_lr + 2u, base_qtype, x0, x1, xb0, xb1, ones16, mask0f, tern_shuf, tern_m0, tern_m1, tern_m2, tern_m3, tern_lut, bin_shuf_lo, bin_shuf_hi, bin_bits, two_u8, one_u8);
                __m128i r3 = ssse3_dot_row_vec(tile_ptr, base_lr + 3u, base_qtype, x0, x1, xb0, xb1, ones16, mask0f, tern_shuf, tern_m0, tern_m1, tern_m2, tern_m3, tern_lut, bin_shuf_lo, bin_shuf_hi, bin_bits, two_u8, one_u8);
                __m128i sum4 = _mm_hadd_epi32(_mm_hadd_epi32(r0, r1), _mm_hadd_epi32(r2, r3));
                __m128 vs = is_row_scaled ? ssse3_load_row_scales4(row_sc + base_lr, x_sc) : v_scale;
                __m128 d4 = _mm_mul_ps(_mm_cvtepi32_ps(sum4), vs);
                if (g == 0u) vacc0 = _mm_add_ps(vacc0, d4);
                else if (g == 1u) vacc1 = _mm_add_ps(vacc1, d4);
                else if (g == 2u) vacc2 = _mm_add_ps(vacc2, d4);
                else vacc3 = _mm_add_ps(vacc3, d4);
            }
        }

        _mm_store_ps(tile_acc + 0u,  vacc0);
        _mm_store_ps(tile_acc + 4u,  vacc1);
        _mm_store_ps(tile_acc + 8u,  vacc2);
        _mm_store_ps(tile_acc + 12u, vacc3);
        for (uint32_t lr = 0; lr < rem_rows; lr++) out[r_base + lr] = tile_acc[lr];
    }
}

QMX_X86_TARGET("ssse3")
static int32_t ssse3_dot_q8(const int8_t  * QMX_RESTRICT q8,
                            const int16_t * QMX_RESTRICT q16,
                            const int8_t  * QMX_RESTRICT k8,
                            uint32_t n) {
    (void)q16;
    __m128i ones16 = _mm_set1_epi16(1);
    __m128i acc = _mm_setzero_si128();
    uint32_t i = 0;
    for (; i + 15u < n; i += 16u) {
        __m128i vq = _mm_lddqu_si128((const __m128i *)(q8 + i));
        __m128i vk = _mm_lddqu_si128((const __m128i *)(k8 + i));
        __m128i p16 = _mm_maddubs_epi16(_mm_abs_epi8(vq), _mm_sign_epi8(vk, vq));
        acc = _mm_add_epi32(acc, _mm_madd_epi16(p16, ones16));
    }
    acc = _mm_hadd_epi32(acc, acc);
    acc = _mm_hadd_epi32(acc, acc);
    int32_t sum = _mm_cvtsi128_si32(acc);
    for (; i < n; i++) sum += (int32_t)q8[i] * (int32_t)k8[i];
    return sum;
}

/* ============================================================================
 * 4. SSE4.1 (Penryn / Nehalem: dpps + pmovsxbw + pshufb/pmaddubsw)
 * ============================================================================ */

QMX_X86_TARGET("sse4.1")
static float sse41_dot_f32(const float * QMX_RESTRICT a,
                           const float * QMX_RESTRICT b,
                           uint32_t n) {
    __m128 acc = _mm_setzero_ps();
    uint32_t i = 0;
    for (; i + 3u < n; i += 4u) {
        acc = _mm_add_ss(acc, _mm_dp_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i), 0xF1));
    }
    float sum = _mm_cvtss_f32(acc);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

QMX_X86_TARGET("sse4.1")
static int32_t sse41_dot_q8(const int8_t  * QMX_RESTRICT q8,
                            const int16_t * QMX_RESTRICT q16,
                            const int8_t  * QMX_RESTRICT k8,
                            uint32_t n) {
    return ssse3_dot_q8(q8, q16, k8, n);
}

/* ============================================================================
 * 5. SSE4.2 (Nehalem / Westmere: SSE4.2 + POPCNT)
 * ============================================================================ */

QMX_X86_TARGET("sse4.2,popcnt")
static int32_t sse42_dot_q8(const int8_t  * QMX_RESTRICT q8,
                            const int16_t * QMX_RESTRICT q16,
                            const int8_t  * QMX_RESTRICT k8,
                            uint32_t n) {
    return ssse3_dot_q8(q8, q16, k8, n);
}

/* ============================================================================
 * 6. AVX (Sandy Bridge / Ivy Bridge / Bulldozer: 256-bit YMM FP32 + VEX XMM)
 * ============================================================================ */

QMX_X86_TARGET("avx")
static inline float avx1_hsum_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

QMX_X86_TARGET("avx")
static void avx1_rmsnorm(float       * QMX_RESTRICT out,
                         const float * QMX_RESTRICT x,
                         const float * QMX_RESTRICT weight,
                         uint32_t size,
                         float eps) {
    __m256 sum8 = _mm256_setzero_ps();
    uint32_t i = 0;
    for (; i + 7u < size; i += 8u) {
        __m256 v = _mm256_loadu_ps(x + i);
        sum8 = _mm256_add_ps(sum8, _mm256_mul_ps(v, v));
    }
    float ss = avx1_hsum_ps(sum8);
    for (; i < size; i++) ss += x[i] * x[i];
    float inv_rms = 1.0f / qmx_sqrtf(ss / (float)size + eps);
    __m256 v_inv = _mm256_set1_ps(inv_rms);
    i = 0;
    for (; i + 7u < size; i += 8u) {
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + i), v_inv), _mm256_loadu_ps(weight + i)));
    }
    _mm256_zeroupper();
    for (; i < size; i++) out[i] = weight[i] * (inv_rms * x[i]);
}

QMX_X86_TARGET("avx")
static float avx1_dot_f32(const float * QMX_RESTRICT a,
                          const float * QMX_RESTRICT b,
                          uint32_t n) {
    __m256 sum8 = _mm256_setzero_ps();
    uint32_t i = 0;
    for (; i + 7u < n; i += 8u) {
        sum8 = _mm256_add_ps(sum8, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
    float sum = avx1_hsum_ps(sum8);
    _mm256_zeroupper();
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

QMX_X86_TARGET("avx")
static void avx1_saxpy_f32(float       * QMX_RESTRICT y,
                           float a,
                           const float * QMX_RESTRICT x,
                           uint32_t n) {
    __m256 va = _mm256_set1_ps(a);
    uint32_t i = 0;
    for (; i + 7u < n; i += 8u) {
        _mm256_storeu_ps(y + i, _mm256_add_ps(_mm256_loadu_ps(y + i), _mm256_mul_ps(va, _mm256_loadu_ps(x + i))));
    }
    _mm256_zeroupper();
    for (; i < n; i++) y[i] += a * x[i];
}

QMX_X86_TARGET("avx")
static void avx1_matmul_tiled(float         * QMX_RESTRICT out,
                              const int8_t  * QMX_RESTRICT xq,
                              const int16_t * QMX_RESTRICT xq16,
                              const float   * QMX_RESTRICT x_scales,
                              const QmxTensor *w,
                              int accumulate) {
    ssse3_matmul_tiled(out, xq, xq16, x_scales, w, accumulate);
}

/* ============================================================================
 * 7. AVX-512 VNNI (Ice Lake / Zen 4 / Zen 5: 512-bit ZMM + vpdpbusd)
 * ============================================================================ */

#if defined(QMX_ENABLE_BACKEND_AVX512)

QMX_X86_TARGET("avx512f,avx512bw,avx512vl,avx512vnni")
static inline float avx512_hsum_ps(__m512 v) {
    __m256 lo = _mm512_castps512_ps256(v);
    __m256 hi = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(v), 1));
    __m256 s8 = _mm256_add_ps(lo, hi);
    __m128 s4 = _mm_add_ps(_mm256_castps256_ps128(s8), _mm256_extractf128_ps(s8, 1));
    __m128 sh = _mm_movehdup_ps(s4);
    s4 = _mm_add_ps(s4, sh);
    sh = _mm_movehl_ps(sh, s4);
    s4 = _mm_add_ss(s4, sh);
    return _mm_cvtss_f32(s4);
}

QMX_X86_TARGET("avx512f,avx512bw,avx512vl,avx512vnni")
static inline float avx512_hmax_ps(__m512 v) {
    __m256 lo = _mm512_castps512_ps256(v);
    __m256 hi = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(v), 1));
    __m256 m8 = _mm256_max_ps(lo, hi);
    __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(m8), _mm256_extractf128_ps(m8, 1));
    m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    m4 = _mm_max_ss(m4, _mm_shuffle_ps(m4, m4, 1));
    return _mm_cvtss_f32(m4);
}

QMX_X86_TARGET("avx512f,avx512bw,avx512vl,avx512vnni")
static inline int32_t avx512_hsum_epi32(__m512i v) {
    __m256i lo = _mm512_castsi512_si256(v);
    __m256i hi = _mm512_extracti64x4_epi64(v, 1);
    __m256i s8 = _mm256_add_epi32(lo, hi);
    __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(s8), _mm256_extracti128_si256(s8, 1));
    s4 = _mm_hadd_epi32(s4, s4);
    s4 = _mm_hadd_epi32(s4, s4);
    return _mm_cvtsi128_si32(s4);
}

QMX_X86_TARGET("avx512f,avx512bw,avx512vl,avx512vnni")
static void avx512_rmsnorm(float       * QMX_RESTRICT out,
                           const float * QMX_RESTRICT x,
                           const float * QMX_RESTRICT weight,
                           uint32_t size,
                           float eps) {
    __m512 sum16 = _mm512_setzero_ps();
    uint32_t i = 0;
    for (; i + 15u < size; i += 16u) {
        __m512 v = _mm512_loadu_ps(x + i);
        sum16 = _mm512_fmadd_ps(v, v, sum16);
    }
    float ss = avx512_hsum_ps(sum16);
    for (; i < size; i++) ss += x[i] * x[i];
    float inv_rms = 1.0f / qmx_sqrtf(ss / (float)size + eps);
    __m512 v_inv = _mm512_set1_ps(inv_rms);
    i = 0;
    for (; i + 15u < size; i += 16u) {
        _mm512_storeu_ps(out + i, _mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(x + i), v_inv), _mm512_loadu_ps(weight + i)));
    }
    QMX_MSVC_ZEROUPPER();
    for (; i < size; i++) out[i] = weight[i] * (inv_rms * x[i]);
}

QMX_X86_TARGET("avx512f,avx512bw,avx512vl,avx512vnni")
static float avx512_dot_f32(const float * QMX_RESTRICT a,
                            const float * QMX_RESTRICT b,
                            uint32_t n) {
    __m512 sum16 = _mm512_setzero_ps();
    uint32_t i = 0;
    for (; i + 15u < n; i += 16u) {
        sum16 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), sum16);
    }
    float sum = avx512_hsum_ps(sum16);
    QMX_MSVC_ZEROUPPER();
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

QMX_X86_TARGET("avx512f,avx512bw,avx512vl,avx512vnni")
static void avx512_saxpy_f32(float       * QMX_RESTRICT y,
                             float a,
                             const float * QMX_RESTRICT x,
                             uint32_t n) {
    __m512 va = _mm512_set1_ps(a);
    uint32_t i = 0;
    for (; i + 15u < n; i += 16u) {
        _mm512_storeu_ps(y + i, _mm512_fmadd_ps(va, _mm512_loadu_ps(x + i), _mm512_loadu_ps(y + i)));
    }
    QMX_MSVC_ZEROUPPER();
    for (; i < n; i++) y[i] += a * x[i];
}

QMX_X86_TARGET("avx512f,avx512bw,avx512vl,avx512vnni")
static int32_t avx512_dot_q8(const int8_t  * QMX_RESTRICT q8,
                             const int16_t * QMX_RESTRICT q16,
                             const int8_t  * QMX_RESTRICT k8,
                             uint32_t n) {
    (void)q16;
    uint32_t i = 0;
    int32_t sum = 0;
    if (n >= 64u) {
        __m512i acc512      = _mm512_setzero_si512();
        __m512i ksum512     = _mm512_setzero_si512();
        __m512i bias128_512 = _mm512_set1_epi8((char)0x80);
        __m512i ones_512    = _mm512_set1_epi8(1);
        for (; i + 63u < n; i += 64u) {
            __m512i vq_u8 = _mm512_xor_si512(_mm512_loadu_si512((const void *)(q8 + i)), bias128_512);
            __m512i vk_s8 = _mm512_loadu_si512((const void *)(k8 + i));
            acc512  = _mm512_dpbusd_epi32(acc512, vq_u8, vk_s8);
            ksum512 = _mm512_dpbusd_epi32(ksum512, ones_512, vk_s8);
        }
        __m512i corr512 = _mm512_sub_epi32(acc512, _mm512_slli_epi32(ksum512, 7));
        sum += avx512_hsum_epi32(corr512);
    }
    if (i + 31u < n) {
        __m256i bias128 = _mm256_set1_epi8((char)0x80);
        __m256i ones_u8 = _mm256_set1_epi8(1);
        __m256i vq_u8 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(q8 + i)), bias128);
        __m256i vk_s8 = _mm256_loadu_si256((const __m256i *)(k8 + i));
        __m256i acc   = _mm256_dpbusd_epi32(_mm256_setzero_si256(), vq_u8, vk_s8);
        __m256i k_sum = _mm256_dpbusd_epi32(_mm256_setzero_si256(), ones_u8, vk_s8);
        __m256i corr  = _mm256_sub_epi32(acc, _mm256_slli_epi32(k_sum, 7));
        __m128i lo = _mm256_castsi256_si128(corr);
        __m128i hi = _mm256_extracti128_si256(corr, 1);
        lo = _mm_add_epi32(lo, hi);
        lo = _mm_hadd_epi32(lo, lo);
        lo = _mm_hadd_epi32(lo, lo);
        sum += _mm_cvtsi128_si32(lo);
        i += 32u;
    }
    QMX_MSVC_ZEROUPPER();
    for (; i < n; i++) sum += (int32_t)q8[i] * (int32_t)k8[i];
    return sum;
}

QMX_X86_TARGET("avx512f,avx512bw,avx512vl,avx512vnni,f16c,fma")
static void avx512_prepare_act(const float * QMX_RESTRICT x,
                               int8_t      * QMX_RESTRICT xq,
                               int16_t     * QMX_RESTRICT xq16,
                               float       * QMX_RESTRICT x_scales,
                               uint32_t cols) {
    (void)xq16;
    uint32_t full_groups = cols / QMX_TILE_COLS;
    const __m512i abs_mask = _mm512_set1_epi32(0x7FFFFFFF);
    const __m256i min_i8   = _mm256_set1_epi8(-127);

    for (uint32_t g = 0; g < full_groups; g++) {
        const float *xp = x + (size_t)g * QMX_TILE_COLS;
        __m512 v0 = _mm512_loadu_ps(xp + 0u);
        __m512 v1 = _mm512_loadu_ps(xp + 16u);
        __m512 a0 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(v0), abs_mask));
        __m512 a1 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(v1), abs_mask));
        float max_abs = avx512_hmax_ps(_mm512_max_ps(a0, a1));

        x_scales[g] = max_abs / 127.0f;
        float inv_scale = (max_abs == 0.0f) ? 0.0f : (127.0f / max_abs);
        __m512 v_inv = _mm512_set1_ps(inv_scale);

        __m512i i0 = _mm512_cvtps_epi32(_mm512_mul_ps(v0, v_inv));
        __m512i i1 = _mm512_cvtps_epi32(_mm512_mul_ps(v1, v_inv));
        __m128i b0 = _mm512_cvtsepi32_epi8(i0);
        __m128i b1 = _mm512_cvtsepi32_epi8(i1);
        __m256i b32 = _mm256_max_epi8(_mm256_inserti128_si256(_mm256_castsi128_si256(b0), b1, 1), min_i8);
        _mm256_storeu_si256((__m256i *)(xq + (size_t)g * QMX_TILE_COLS), b32);
    }

    QMX_MSVC_ZEROUPPER();
    uint32_t tail_base = full_groups * QMX_TILE_COLS;
    if (tail_base < cols) {
        qmx_quantize_activations_q8(x + tail_base, xq + tail_base, x_scales + full_groups, cols - tail_base);
    }
}

QMX_X86_TARGET("avx512f,avx512bw,avx512vl,avx512vnni,f16c,fma")
static inline __m128 avx512_load_row_scales4(const uint16_t *sc16, float x_sc) {
    __m128i h4 = _mm_loadl_epi64((const __m128i *)sc16);
    return _mm_mul_ps(_mm_cvtph_ps(h4), _mm_set1_ps(x_sc));
}

QMX_X86_TARGET("avx512f,avx512bw,avx512vl,avx512vnni")
static inline __m128i avx512_hsum4_2x512(__m512i r01, __m512i r23) {
    __m256i r0 = _mm512_castsi512_si256(r01);
    __m256i r1 = _mm512_extracti64x4_epi64(r01, 1);
    __m256i r2 = _mm512_castsi512_si256(r23);
    __m256i r3 = _mm512_extracti64x4_epi64(r23, 1);
    __m256i h01 = _mm256_hadd_epi32(r0, r1);
    __m256i h23 = _mm256_hadd_epi32(r2, r3);
    __m256i h0123 = _mm256_hadd_epi32(h01, h23);
    return _mm_add_epi32(_mm256_castsi256_si128(h0123), _mm256_extracti128_si256(h0123, 1));
}

QMX_X86_TARGET("avx512f,avx512bw,avx512vl,avx512vnni")
static inline __m128i avx512_hsum4_128lanes(__m512i acc4) {
    __m256i lo = _mm512_castsi512_si256(acc4);
    __m256i hi = _mm512_extracti64x4_epi64(acc4, 1);
    __m256i h1 = _mm256_hadd_epi32(lo, hi);
    __m256i h2 = _mm256_hadd_epi32(h1, h1);
    return _mm_unpacklo_epi32(_mm256_castsi256_si128(h2), _mm256_extracti128_si256(h2, 1));
}

QMX_X86_TARGET("avx512f,avx512bw,avx512vl,avx512vnni")
static inline int32_t avx512_xsum32(__m256i vx256, __m256i ones_u8) {
    __m256i s8 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), ones_u8, vx256);
    __m128i lo = _mm_add_epi32(_mm256_castsi256_si128(s8), _mm256_extracti128_si256(s8, 1));
    lo = _mm_hadd_epi32(lo, lo);
    lo = _mm_hadd_epi32(lo, lo);
    return _mm_cvtsi128_si32(lo);
}

QMX_X86_TARGET("avx512f,avx512bw,avx512vl,avx512vnni")
static inline void avx512_split_x32_eo(const int8_t *xp, __m128i shuf_eo, __m256i ones_u8,
                                       __m128i *out_even, __m128i *out_odd, int32_t *out_sum) {
    __m256i vx256 = _mm256_loadu_si256((const __m256i *)xp);
    *out_sum = avx512_xsum32(vx256, ones_u8);
    __m128i t0 = _mm_shuffle_epi8(_mm256_castsi256_si128(vx256), shuf_eo);
    __m128i t1 = _mm_shuffle_epi8(_mm256_extracti128_si256(vx256, 1), shuf_eo);
    *out_even = _mm_unpacklo_epi64(t0, t1);
    *out_odd  = _mm_unpackhi_epi64(t0, t1);
}

QMX_X86_TARGET("avx512f,avx512bw,avx512vl,avx512vnni")
static inline __m128i avx512_dot_q4_4rows(const uint8_t *tile_ptr, uint32_t lr,
                                          __m512i vxe, __m512i vxo, __m512i mask0f, __m128i vbias) {
    __m512i raw64 = _mm512_loadu_si512((const void *)(tile_ptr + lr * 16u));
    __m512i lo64 = _mm512_and_si512(raw64, mask0f);
    __m512i hi64 = _mm512_and_si512(_mm512_srli_epi16(raw64, 4), mask0f);
    __m512i acc4 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), lo64, vxe);
    acc4 = _mm512_dpbusd_epi32(acc4, hi64, vxo);
    return _mm_sub_epi32(avx512_hsum4_128lanes(acc4), vbias);
}

QMX_X86_TARGET("avx512f,avx512bw,avx512vl,avx512vnni")
static void avx512_prepare_act_eo(const int8_t * QMX_RESTRICT xq,
                                  int8_t       * QMX_RESTRICT xq_eo,
                                  int32_t      * QMX_RESTRICT x_sums,
                                  uint32_t cols) {
    uint32_t n_tc = (uint32_t)qmx_align_up(cols, QMX_TILE_COLS) / QMX_TILE_COLS;
    __m256i ones_u8 = _mm256_set1_epi8(1);
    __m128i shuf_eo = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
    for (uint32_t tc = 0; tc < n_tc; tc++) {
        __m128i xe, xo;
        int32_t xs = 0;
        avx512_split_x32_eo(xq + (size_t)tc * QMX_TILE_COLS, shuf_eo, ones_u8, &xe, &xo, &xs);
        _mm_storeu_si128((__m128i *)(xq_eo + (size_t)tc * QMX_TILE_COLS + 0u), xe);
        _mm_storeu_si128((__m128i *)(xq_eo + (size_t)tc * QMX_TILE_COLS + 16u), xo);
        x_sums[tc] = xs;
    }
    QMX_MSVC_ZEROUPPER();
}

QMX_X86_TARGET("avx512f,avx512bw,avx512vl,avx512vnni,f16c,fma")
QMX_INLINE void avx512_matmul_q4(
        float * QMX_RESTRICT out, const int8_t * QMX_RESTRICT xq,
        const int8_t * QMX_RESTRICT xq_eo, const int32_t * QMX_RESTRICT x_sums,
        const float * QMX_RESTRICT x_scales, const QmxTensor *w,
        int accumulate, int is_row_scaled, int use_eo) {
    uint32_t orig_rows = w->rows;
    uint32_t n_tr = (uint32_t)qmx_align_up(orig_rows, QMX_TILE_ROWS) / QMX_TILE_ROWS;
    uint32_t n_tc = (uint32_t)qmx_align_up(w->cols, QMX_TILE_COLS) / QMX_TILE_COLS;
    const uint8_t *base_data = (const uint8_t *)w->data;
    const uint16_t *scales_f16 = (const uint16_t *)(const void *)w->scales;
    __m512i mask0f = _mm512_set1_epi8(0x0F);
    __m256i ones_u8 = _mm256_set1_epi8(1);
    __m128i shuf_eo = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
    (void)xq; (void)xq_eo; (void)x_sums; (void)ones_u8; (void)shuf_eo;

    uint32_t tr = 0;
    for (; tr + 1u < n_tr && (tr + 2u) * QMX_TILE_ROWS <= orig_rows; tr += 2u) {
        uint32_t r_base0 = tr * QMX_TILE_ROWS;
        uint32_t r_base1 = r_base0 + QMX_TILE_ROWS;
        __m128 va0 = accumulate ? _mm_loadu_ps(out + r_base0 + 0u)  : _mm_setzero_ps();
        __m128 va1 = accumulate ? _mm_loadu_ps(out + r_base0 + 4u)  : _mm_setzero_ps();
        __m128 va2 = accumulate ? _mm_loadu_ps(out + r_base0 + 8u)  : _mm_setzero_ps();
        __m128 va3 = accumulate ? _mm_loadu_ps(out + r_base0 + 12u) : _mm_setzero_ps();
        __m128 vb0 = accumulate ? _mm_loadu_ps(out + r_base1 + 0u)  : _mm_setzero_ps();
        __m128 vb1 = accumulate ? _mm_loadu_ps(out + r_base1 + 4u)  : _mm_setzero_ps();
        __m128 vb2 = accumulate ? _mm_loadu_ps(out + r_base1 + 8u)  : _mm_setzero_ps();
        __m128 vb3 = accumulate ? _mm_loadu_ps(out + r_base1 + 12u) : _mm_setzero_ps();

        uint32_t tbase0 = tr * n_tc;
        uint32_t tbase1 = tbase0 + n_tc;

        for (uint32_t tc = 0; tc < n_tc; tc++) {
            __m128i xe128, xo128;
            int32_t bias_i32;
            if (use_eo) {
                xe128 = _mm_loadu_si128((const __m128i *)(xq_eo + (size_t)tc * QMX_TILE_COLS + 0u));
                xo128 = _mm_loadu_si128((const __m128i *)(xq_eo + (size_t)tc * QMX_TILE_COLS + 16u));
                bias_i32 = x_sums[tc] * 8;
            } else {
                int32_t xs = 0;
                avx512_split_x32_eo(xq + (size_t)tc * QMX_TILE_COLS, shuf_eo, ones_u8, &xe128, &xo128, &xs);
                bias_i32 = xs * 8;
            }
            __m512i vxe = _mm512_broadcast_i32x4(xe128);
            __m512i vxo = _mm512_broadcast_i32x4(xo128);
            __m128i vbias = _mm_set1_epi32(bias_i32);
            float x_sc = x_scales[tc];

            uint32_t tid0 = tbase0 + tc;
            uint32_t tid1 = tbase1 + tc;
            const uint8_t *tp0 = base_data + (size_t)tid0 * 256u;
            const uint8_t *tp1 = base_data + (size_t)tid1 * 256u;
            if (tc + 2u < n_tc) {
                _mm_prefetch((const char *)(tp0 + 512u), _MM_HINT_T0);
                _mm_prefetch((const char *)(tp1 + 512u), _MM_HINT_T0);
            }

            if (is_row_scaled) {
                const uint16_t *rs0 = scales_f16 + (size_t)tid0 * QMX_TILE_ROWS;
                const uint16_t *rs1 = scales_f16 + (size_t)tid1 * QMX_TILE_ROWS;
                va0 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx512_dot_q4_4rows(tp0, 0u,  vxe, vxo, mask0f, vbias)), avx512_load_row_scales4(rs0 + 0u,  x_sc), va0);
                va1 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx512_dot_q4_4rows(tp0, 4u,  vxe, vxo, mask0f, vbias)), avx512_load_row_scales4(rs0 + 4u,  x_sc), va1);
                va2 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx512_dot_q4_4rows(tp0, 8u,  vxe, vxo, mask0f, vbias)), avx512_load_row_scales4(rs0 + 8u,  x_sc), va2);
                va3 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx512_dot_q4_4rows(tp0, 12u, vxe, vxo, mask0f, vbias)), avx512_load_row_scales4(rs0 + 12u, x_sc), va3);
                vb0 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx512_dot_q4_4rows(tp1, 0u,  vxe, vxo, mask0f, vbias)), avx512_load_row_scales4(rs1 + 0u,  x_sc), vb0);
                vb1 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx512_dot_q4_4rows(tp1, 4u,  vxe, vxo, mask0f, vbias)), avx512_load_row_scales4(rs1 + 4u,  x_sc), vb1);
                vb2 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx512_dot_q4_4rows(tp1, 8u,  vxe, vxo, mask0f, vbias)), avx512_load_row_scales4(rs1 + 8u,  x_sc), vb2);
                vb3 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx512_dot_q4_4rows(tp1, 12u, vxe, vxo, mask0f, vbias)), avx512_load_row_scales4(rs1 + 12u, x_sc), vb3);
            } else {
                __m128 vs0 = _mm_set1_ps(w->scales[tid0] * x_sc);
                __m128 vs1 = _mm_set1_ps(w->scales[tid1] * x_sc);
                va0 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx512_dot_q4_4rows(tp0, 0u,  vxe, vxo, mask0f, vbias)), vs0, va0);
                va1 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx512_dot_q4_4rows(tp0, 4u,  vxe, vxo, mask0f, vbias)), vs0, va1);
                va2 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx512_dot_q4_4rows(tp0, 8u,  vxe, vxo, mask0f, vbias)), vs0, va2);
                va3 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx512_dot_q4_4rows(tp0, 12u, vxe, vxo, mask0f, vbias)), vs0, va3);
                vb0 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx512_dot_q4_4rows(tp1, 0u,  vxe, vxo, mask0f, vbias)), vs1, vb0);
                vb1 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx512_dot_q4_4rows(tp1, 4u,  vxe, vxo, mask0f, vbias)), vs1, vb1);
                vb2 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx512_dot_q4_4rows(tp1, 8u,  vxe, vxo, mask0f, vbias)), vs1, vb2);
                vb3 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx512_dot_q4_4rows(tp1, 12u, vxe, vxo, mask0f, vbias)), vs1, vb3);
            }
        }

        _mm_storeu_ps(out + r_base0 + 0u,  va0);
        _mm_storeu_ps(out + r_base0 + 4u,  va1);
        _mm_storeu_ps(out + r_base0 + 8u,  va2);
        _mm_storeu_ps(out + r_base0 + 12u, va3);
        _mm_storeu_ps(out + r_base1 + 0u,  vb0);
        _mm_storeu_ps(out + r_base1 + 4u,  vb1);
        _mm_storeu_ps(out + r_base1 + 8u,  vb2);
        _mm_storeu_ps(out + r_base1 + 12u, vb3);
    }

    QMX_ALIGNED(64) float tile_acc[QMX_TILE_ROWS];
    for (; tr < n_tr; tr++) {
        uint32_t r_base = tr * QMX_TILE_ROWS;
        uint32_t rem_rows = (r_base < orig_rows) ? (orig_rows - r_base) : 0u;
        if (rem_rows > QMX_TILE_ROWS) rem_rows = QMX_TILE_ROWS;
        for (uint32_t lr = 0; lr < QMX_TILE_ROWS; lr++) {
            tile_acc[lr] = (accumulate && (lr < rem_rows)) ? out[r_base + lr] : 0.0f;
        }
        __m128 vacc0 = _mm_load_ps(tile_acc + 0u);
        __m128 vacc1 = _mm_load_ps(tile_acc + 4u);
        __m128 vacc2 = _mm_load_ps(tile_acc + 8u);
        __m128 vacc3 = _mm_load_ps(tile_acc + 12u);

        for (uint32_t tc = 0; tc < n_tc; tc++) {
            uint32_t tile_idx = tr * n_tc + tc;
            float x_sc = x_scales[tc];
            __m128 v_scale = is_row_scaled ? _mm_setzero_ps() : _mm_set1_ps(w->scales[tile_idx] * x_sc);
            const uint16_t *row_sc = is_row_scaled ? (scales_f16 + (size_t)tile_idx * QMX_TILE_ROWS) : NULL;
            const uint8_t *tile_ptr = base_data + (size_t)tile_idx * 256u;

            __m128i xe128, xo128;
            int32_t bias_i32;
            if (use_eo) {
                xe128 = _mm_loadu_si128((const __m128i *)(xq_eo + (size_t)tc * QMX_TILE_COLS + 0u));
                xo128 = _mm_loadu_si128((const __m128i *)(xq_eo + (size_t)tc * QMX_TILE_COLS + 16u));
                bias_i32 = x_sums[tc] * 8;
            } else {
                int32_t xs = 0;
                avx512_split_x32_eo(xq + (size_t)tc * QMX_TILE_COLS, shuf_eo, ones_u8, &xe128, &xo128, &xs);
                bias_i32 = xs * 8;
            }
            __m512i vxe = _mm512_broadcast_i32x4(xe128);
            __m512i vxo = _mm512_broadcast_i32x4(xo128);
            __m128i vbias = _mm_set1_epi32(bias_i32);

            if (rem_rows > 0u) {
                __m128 vs = is_row_scaled ? avx512_load_row_scales4(row_sc + 0u, x_sc) : v_scale;
                vacc0 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx512_dot_q4_4rows(tile_ptr, 0u, vxe, vxo, mask0f, vbias)), vs, vacc0);
            }
            if (rem_rows > 4u) {
                __m128 vs = is_row_scaled ? avx512_load_row_scales4(row_sc + 4u, x_sc) : v_scale;
                vacc1 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx512_dot_q4_4rows(tile_ptr, 4u, vxe, vxo, mask0f, vbias)), vs, vacc1);
            }
            if (rem_rows > 8u) {
                __m128 vs = is_row_scaled ? avx512_load_row_scales4(row_sc + 8u, x_sc) : v_scale;
                vacc2 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx512_dot_q4_4rows(tile_ptr, 8u, vxe, vxo, mask0f, vbias)), vs, vacc2);
            }
            if (rem_rows > 12u) {
                __m128 vs = is_row_scaled ? avx512_load_row_scales4(row_sc + 12u, x_sc) : v_scale;
                vacc3 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx512_dot_q4_4rows(tile_ptr, 12u, vxe, vxo, mask0f, vbias)), vs, vacc3);
            }
        }

        _mm_store_ps(tile_acc + 0u,  vacc0);
        _mm_store_ps(tile_acc + 4u,  vacc1);
        _mm_store_ps(tile_acc + 8u,  vacc2);
        _mm_store_ps(tile_acc + 12u, vacc3);
        for (uint32_t lr = 0; lr < rem_rows; lr++) out[r_base + lr] = tile_acc[lr];
    }
}

QMX_X86_TARGET("avx512f,avx512bw,avx512vl,avx512vnni,f16c,fma")
static inline __m128i avx512_dot_q8_4rows(const uint8_t *tile_ptr, uint32_t lr,
                                          __m512i vx512, __m512i bias128_512, __m128i vbias) {
    __m512i r01 = _mm512_dpbusd_epi32(_mm512_setzero_si512(),
                                      _mm512_xor_si512(_mm512_loadu_si512((const void *)(tile_ptr + lr * 32u)), bias128_512),
                                      vx512);
    __m512i r23 = _mm512_dpbusd_epi32(_mm512_setzero_si512(),
                                      _mm512_xor_si512(_mm512_loadu_si512((const void *)(tile_ptr + (lr + 2u) * 32u)), bias128_512),
                                      vx512);
    return _mm_sub_epi32(avx512_hsum4_2x512(r01, r23), vbias);
}

QMX_X86_TARGET("avx512f,avx512bw,avx512vl,avx512vnni,f16c,fma")
static void avx512_matmul_q8(float * QMX_RESTRICT out, const int8_t * QMX_RESTRICT xq,
                             const int32_t * QMX_RESTRICT x_sums,
                             const float * QMX_RESTRICT x_scales, const QmxTensor *w,
                             int accumulate, int is_row_scaled) {
    uint32_t orig_rows = w->rows;
    uint32_t n_tr = (uint32_t)qmx_align_up(orig_rows, QMX_TILE_ROWS) / QMX_TILE_ROWS;
    uint32_t n_tc = (uint32_t)qmx_align_up(w->cols, QMX_TILE_COLS) / QMX_TILE_COLS;
    const uint8_t *base_data = (const uint8_t *)w->data;
    const uint16_t *scales_f16 = (const uint16_t *)(const void *)w->scales;
    __m256i ones_u8 = _mm256_set1_epi8(1);
    __m512i bias128_512 = _mm512_set1_epi8((char)0x80);
    (void)ones_u8;

    uint32_t tr = 0;
    for (; tr + 1u < n_tr && (tr + 2u) * QMX_TILE_ROWS <= orig_rows; tr += 2u) {
        uint32_t r_base0 = tr * QMX_TILE_ROWS;
        uint32_t r_base1 = r_base0 + QMX_TILE_ROWS;
        __m128 va0 = accumulate ? _mm_loadu_ps(out + r_base0 + 0u)  : _mm_setzero_ps();
        __m128 va1 = accumulate ? _mm_loadu_ps(out + r_base0 + 4u)  : _mm_setzero_ps();
        __m128 va2 = accumulate ? _mm_loadu_ps(out + r_base0 + 8u)  : _mm_setzero_ps();
        __m128 va3 = accumulate ? _mm_loadu_ps(out + r_base0 + 12u) : _mm_setzero_ps();
        __m128 vb0 = accumulate ? _mm_loadu_ps(out + r_base1 + 0u)  : _mm_setzero_ps();
        __m128 vb1 = accumulate ? _mm_loadu_ps(out + r_base1 + 4u)  : _mm_setzero_ps();
        __m128 vb2 = accumulate ? _mm_loadu_ps(out + r_base1 + 8u)  : _mm_setzero_ps();
        __m128 vb3 = accumulate ? _mm_loadu_ps(out + r_base1 + 12u) : _mm_setzero_ps();

        uint32_t tbase0 = tr * n_tc;
        uint32_t tbase1 = tbase0 + n_tc;

        for (uint32_t tc = 0; tc < n_tc; tc++) {
            __m256i vx256 = _mm256_loadu_si256((const __m256i *)(xq + (size_t)tc * QMX_TILE_COLS));
            __m512i vx512 = _mm512_broadcast_i64x4(vx256);
            int32_t bias_i32 = (x_sums != NULL) ? (x_sums[tc] * 128) : (avx512_xsum32(vx256, ones_u8) * 128);
            __m128i vbias = _mm_set1_epi32(bias_i32);
            float x_sc = x_scales[tc];

            uint32_t tid0 = tbase0 + tc;
            uint32_t tid1 = tbase1 + tc;
            const uint8_t *tp0 = base_data + (size_t)tid0 * 512u;
            const uint8_t *tp1 = base_data + (size_t)tid1 * 512u;
            if (tc + 1u < n_tc) {
                _mm_prefetch((const char *)(tp0 + 512u), _MM_HINT_T0);
                _mm_prefetch((const char *)(tp1 + 512u), _MM_HINT_T0);
            }

            if (is_row_scaled) {
                const uint16_t *rs0 = scales_f16 + (size_t)tid0 * QMX_TILE_ROWS;
                const uint16_t *rs1 = scales_f16 + (size_t)tid1 * QMX_TILE_ROWS;
                va0 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx512_dot_q8_4rows(tp0, 0u,  vx512, bias128_512, vbias)), avx512_load_row_scales4(rs0 + 0u,  x_sc), va0);
                va1 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx512_dot_q8_4rows(tp0, 4u,  vx512, bias128_512, vbias)), avx512_load_row_scales4(rs0 + 4u,  x_sc), va1);
                va2 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx512_dot_q8_4rows(tp0, 8u,  vx512, bias128_512, vbias)), avx512_load_row_scales4(rs0 + 8u,  x_sc), va2);
                va3 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx512_dot_q8_4rows(tp0, 12u, vx512, bias128_512, vbias)), avx512_load_row_scales4(rs0 + 12u, x_sc), va3);
                vb0 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx512_dot_q8_4rows(tp1, 0u,  vx512, bias128_512, vbias)), avx512_load_row_scales4(rs1 + 0u,  x_sc), vb0);
                vb1 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx512_dot_q8_4rows(tp1, 4u,  vx512, bias128_512, vbias)), avx512_load_row_scales4(rs1 + 4u,  x_sc), vb1);
                vb2 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx512_dot_q8_4rows(tp1, 8u,  vx512, bias128_512, vbias)), avx512_load_row_scales4(rs1 + 8u,  x_sc), vb2);
                vb3 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx512_dot_q8_4rows(tp1, 12u, vx512, bias128_512, vbias)), avx512_load_row_scales4(rs1 + 12u, x_sc), vb3);
            } else {
                __m128 vs0 = _mm_set1_ps(w->scales[tid0] * x_sc);
                __m128 vs1 = _mm_set1_ps(w->scales[tid1] * x_sc);
                va0 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx512_dot_q8_4rows(tp0, 0u,  vx512, bias128_512, vbias)), vs0, va0);
                va1 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx512_dot_q8_4rows(tp0, 4u,  vx512, bias128_512, vbias)), vs0, va1);
                va2 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx512_dot_q8_4rows(tp0, 8u,  vx512, bias128_512, vbias)), vs0, va2);
                va3 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx512_dot_q8_4rows(tp0, 12u, vx512, bias128_512, vbias)), vs0, va3);
                vb0 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx512_dot_q8_4rows(tp1, 0u,  vx512, bias128_512, vbias)), vs1, vb0);
                vb1 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx512_dot_q8_4rows(tp1, 4u,  vx512, bias128_512, vbias)), vs1, vb1);
                vb2 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx512_dot_q8_4rows(tp1, 8u,  vx512, bias128_512, vbias)), vs1, vb2);
                vb3 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx512_dot_q8_4rows(tp1, 12u, vx512, bias128_512, vbias)), vs1, vb3);
            }
        }

        _mm_storeu_ps(out + r_base0 + 0u,  va0);
        _mm_storeu_ps(out + r_base0 + 4u,  va1);
        _mm_storeu_ps(out + r_base0 + 8u,  va2);
        _mm_storeu_ps(out + r_base0 + 12u, va3);
        _mm_storeu_ps(out + r_base1 + 0u,  vb0);
        _mm_storeu_ps(out + r_base1 + 4u,  vb1);
        _mm_storeu_ps(out + r_base1 + 8u,  vb2);
        _mm_storeu_ps(out + r_base1 + 12u, vb3);
    }

    QMX_ALIGNED(64) float tile_acc[QMX_TILE_ROWS];
    for (; tr < n_tr; tr++) {
        uint32_t r_base = tr * QMX_TILE_ROWS;
        uint32_t rem_rows = (r_base < orig_rows) ? (orig_rows - r_base) : 0u;
        if (rem_rows > QMX_TILE_ROWS) rem_rows = QMX_TILE_ROWS;
        for (uint32_t lr = 0; lr < QMX_TILE_ROWS; lr++) {
            tile_acc[lr] = (accumulate && (lr < rem_rows)) ? out[r_base + lr] : 0.0f;
        }
        __m128 vacc0 = _mm_load_ps(tile_acc + 0u);
        __m128 vacc1 = _mm_load_ps(tile_acc + 4u);
        __m128 vacc2 = _mm_load_ps(tile_acc + 8u);
        __m128 vacc3 = _mm_load_ps(tile_acc + 12u);

        for (uint32_t tc = 0; tc < n_tc; tc++) {
            uint32_t tile_idx = tr * n_tc + tc;
            float x_sc = x_scales[tc];
            __m128 v_scale = is_row_scaled ? _mm_setzero_ps() : _mm_set1_ps(w->scales[tile_idx] * x_sc);
            const uint16_t *row_sc = is_row_scaled ? (scales_f16 + (size_t)tile_idx * QMX_TILE_ROWS) : NULL;
            const uint8_t *tile_ptr = base_data + (size_t)tile_idx * 512u;

            __m256i vx256 = _mm256_loadu_si256((const __m256i *)(xq + (size_t)tc * QMX_TILE_COLS));
            __m512i vx512 = _mm512_broadcast_i64x4(vx256);
            int32_t bias_i32 = (x_sums != NULL) ? (x_sums[tc] * 128) : (avx512_xsum32(vx256, ones_u8) * 128);
            __m128i vbias = _mm_set1_epi32(bias_i32);

            if (rem_rows > 0u) {
                __m128 vs = is_row_scaled ? avx512_load_row_scales4(row_sc + 0u, x_sc) : v_scale;
                vacc0 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx512_dot_q8_4rows(tile_ptr, 0u, vx512, bias128_512, vbias)), vs, vacc0);
            }
            if (rem_rows > 4u) {
                __m128 vs = is_row_scaled ? avx512_load_row_scales4(row_sc + 4u, x_sc) : v_scale;
                vacc1 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx512_dot_q8_4rows(tile_ptr, 4u, vx512, bias128_512, vbias)), vs, vacc1);
            }
            if (rem_rows > 8u) {
                __m128 vs = is_row_scaled ? avx512_load_row_scales4(row_sc + 8u, x_sc) : v_scale;
                vacc2 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx512_dot_q8_4rows(tile_ptr, 8u, vx512, bias128_512, vbias)), vs, vacc2);
            }
            if (rem_rows > 12u) {
                __m128 vs = is_row_scaled ? avx512_load_row_scales4(row_sc + 12u, x_sc) : v_scale;
                vacc3 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx512_dot_q8_4rows(tile_ptr, 12u, vx512, bias128_512, vbias)), vs, vacc3);
            }
        }

        _mm_store_ps(tile_acc + 0u,  vacc0);
        _mm_store_ps(tile_acc + 4u,  vacc1);
        _mm_store_ps(tile_acc + 8u,  vacc2);
        _mm_store_ps(tile_acc + 12u, vacc3);
        for (uint32_t lr = 0; lr < rem_rows; lr++) out[r_base + lr] = tile_acc[lr];
    }
}

QMX_X86_TARGET("avx512f,avx512bw,avx512vl,avx512vnni")
static inline __m512i avx512_dot_2rows_lowbit(const uint8_t *tile_ptr, uint32_t lr, uint8_t qtype,
                                              __m512i vx512,
                                              __m512i tern_shuf, __m512i tern_m0, __m512i tern_m1,
                                              __m512i tern_m2, __m512i tern_m3, __m512i tern_lut_u8,
                                              __m512i bin_shuf, __m512i bin_bits, __m512i two_u8) {
    if (qtype == QMX_QUANT_TERNARY_158) {
        const uint8_t *rp = tile_ptr + lr * 8u;
        uint64_t r0 = 0, r1 = 0;
        memcpy(&r0, rp, 8);
        memcpy(&r1, rp + 8u, 8);
        __m256i v0 = _mm256_set1_epi64x((long long)r0);
        __m256i v1 = _mm256_set1_epi64x((long long)r1);
        __m512i vb = _mm512_shuffle_epi8(_mm512_inserti64x4(_mm512_castsi256_si512(v0), v1, 1), tern_shuf);
        __m512i c0 = _mm512_and_si512(vb, tern_m0);
        __m512i c1 = _mm512_and_si512(_mm512_srli_epi16(vb, 2), tern_m1);
        __m512i c2 = _mm512_and_si512(_mm512_srli_epi16(vb, 4), tern_m2);
        __m512i c3 = _mm512_and_si512(_mm512_srli_epi16(vb, 6), tern_m3);
        __m512i codes = _mm512_or_si512(_mm512_or_si512(c0, c1), _mm512_or_si512(c2, c3));
        __m512i vu = _mm512_shuffle_epi8(tern_lut_u8, codes);
        return _mm512_dpbusd_epi32(_mm512_setzero_si512(), vu, vx512);
    }
    const uint8_t *rp = tile_ptr + lr * 4u;
    uint32_t r0 = 0, r1 = 0;
    memcpy(&r0, rp, 4);
    memcpy(&r1, rp + 4u, 4);
    __m256i v0 = _mm256_set1_epi32((int)r0);
    __m256i v1 = _mm256_set1_epi32((int)r1);
    __m512i vb = _mm512_shuffle_epi8(_mm512_inserti64x4(_mm512_castsi256_si512(v0), v1, 1), bin_shuf);
    __mmask64 m = _mm512_test_epi8_mask(vb, bin_bits);
    __m512i vu = _mm512_maskz_mov_epi8(m, two_u8);
    return _mm512_dpbusd_epi32(_mm512_setzero_si512(), vu, vx512);
}

QMX_X86_TARGET("avx512f,avx512bw,avx512vl,avx512vnni,f16c,fma")
static void avx512_matmul_lowbit(float * QMX_RESTRICT out, const int8_t * QMX_RESTRICT xq,
                                 const int32_t * QMX_RESTRICT x_sums,
                                 const float * QMX_RESTRICT x_scales, const QmxTensor *w,
                                 int accumulate, uint8_t qtype) {
    uint32_t orig_rows = w->rows;
    uint32_t n_tr = (uint32_t)qmx_align_up(orig_rows, QMX_TILE_ROWS) / QMX_TILE_ROWS;
    uint32_t n_tc = (uint32_t)qmx_align_up(w->cols, QMX_TILE_COLS) / QMX_TILE_COLS;
    size_t tile_bytes = qmx_tile_data_bytes(qtype);
    const uint8_t *base_data = (const uint8_t *)w->data;

    __m256i ones_u8 = _mm256_set1_epi8(1);
    (void)ones_u8;
    __m256i t_shuf256 = _mm256_setr_epi8(
        0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
        4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7
    );
    __m512i tern_shuf = _mm512_broadcast_i64x4(t_shuf256);
    __m512i tern_m0   = _mm512_set1_epi32(0x00000003);
    __m512i tern_m1   = _mm512_set1_epi32(0x00000300);
    __m512i tern_m2   = _mm512_set1_epi32(0x00030000);
    __m512i tern_m3   = _mm512_set1_epi32(0x03000000);
    __m256i t_lut256  = _mm256_setr_epi8(
        1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1,
        1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1
    );
    __m512i tern_lut_u8 = _mm512_broadcast_i64x4(t_lut256);
    __m256i b_shuf256 = _mm256_setr_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3
    );
    __m512i bin_shuf = _mm512_broadcast_i64x4(b_shuf256);
    __m512i bin_bits = _mm512_set1_epi64((long long)0x8040201008040201ULL);
    __m512i two_u8   = _mm512_set1_epi8(2);



    QMX_ALIGNED(64) float tile_acc[QMX_TILE_ROWS];
    for (uint32_t tr = 0; tr < n_tr; tr++) {
        uint32_t r_base = tr * QMX_TILE_ROWS;
        uint32_t rem_rows = (r_base < orig_rows) ? (orig_rows - r_base) : 0u;
        if (rem_rows > QMX_TILE_ROWS) rem_rows = QMX_TILE_ROWS;
        for (uint32_t lr = 0; lr < QMX_TILE_ROWS; lr++) {
            tile_acc[lr] = (accumulate && (lr < rem_rows)) ? out[r_base + lr] : 0.0f;
        }
        __m128 vacc0 = _mm_load_ps(tile_acc + 0u);
        __m128 vacc1 = _mm_load_ps(tile_acc + 4u);
        __m128 vacc2 = _mm_load_ps(tile_acc + 8u);
        __m128 vacc3 = _mm_load_ps(tile_acc + 12u);

        for (uint32_t tc = 0; tc < n_tc; tc++) {
            uint32_t tile_idx = tr * n_tc + tc;
            __m128 v_scale = _mm_set1_ps(w->scales[tile_idx] * x_scales[tc]);
            const uint8_t *tile_ptr = base_data + (size_t)tile_idx * tile_bytes;
            __m256i vx256 = _mm256_loadu_si256((const __m256i *)(xq + (size_t)tc * QMX_TILE_COLS));
            __m512i vx512 = _mm512_broadcast_i64x4(vx256);
            int32_t bias_i32 = (x_sums != NULL) ? x_sums[tc] : avx512_xsum32(vx256, ones_u8);
            __m128i vbias = _mm_set1_epi32(bias_i32);

            if (rem_rows > 0u) {
                __m512i r01 = avx512_dot_2rows_lowbit(tile_ptr, 0u, qtype, vx512, tern_shuf, tern_m0, tern_m1, tern_m2, tern_m3, tern_lut_u8, bin_shuf, bin_bits, two_u8);
                __m512i r23 = avx512_dot_2rows_lowbit(tile_ptr, 2u, qtype, vx512, tern_shuf, tern_m0, tern_m1, tern_m2, tern_m3, tern_lut_u8, bin_shuf, bin_bits, two_u8);
                vacc0 = _mm_fmadd_ps(_mm_cvtepi32_ps(_mm_sub_epi32(avx512_hsum4_2x512(r01, r23), vbias)), v_scale, vacc0);
            }
            if (rem_rows > 4u) {
                __m512i r01 = avx512_dot_2rows_lowbit(tile_ptr, 4u, qtype, vx512, tern_shuf, tern_m0, tern_m1, tern_m2, tern_m3, tern_lut_u8, bin_shuf, bin_bits, two_u8);
                __m512i r23 = avx512_dot_2rows_lowbit(tile_ptr, 6u, qtype, vx512, tern_shuf, tern_m0, tern_m1, tern_m2, tern_m3, tern_lut_u8, bin_shuf, bin_bits, two_u8);
                vacc1 = _mm_fmadd_ps(_mm_cvtepi32_ps(_mm_sub_epi32(avx512_hsum4_2x512(r01, r23), vbias)), v_scale, vacc1);
            }
            if (rem_rows > 8u) {
                __m512i r01 = avx512_dot_2rows_lowbit(tile_ptr, 8u, qtype, vx512, tern_shuf, tern_m0, tern_m1, tern_m2, tern_m3, tern_lut_u8, bin_shuf, bin_bits, two_u8);
                __m512i r23 = avx512_dot_2rows_lowbit(tile_ptr, 10u, qtype, vx512, tern_shuf, tern_m0, tern_m1, tern_m2, tern_m3, tern_lut_u8, bin_shuf, bin_bits, two_u8);
                vacc2 = _mm_fmadd_ps(_mm_cvtepi32_ps(_mm_sub_epi32(avx512_hsum4_2x512(r01, r23), vbias)), v_scale, vacc2);
            }
            if (rem_rows > 12u) {
                __m512i r01 = avx512_dot_2rows_lowbit(tile_ptr, 12u, qtype, vx512, tern_shuf, tern_m0, tern_m1, tern_m2, tern_m3, tern_lut_u8, bin_shuf, bin_bits, two_u8);
                __m512i r23 = avx512_dot_2rows_lowbit(tile_ptr, 14u, qtype, vx512, tern_shuf, tern_m0, tern_m1, tern_m2, tern_m3, tern_lut_u8, bin_shuf, bin_bits, two_u8);
                vacc3 = _mm_fmadd_ps(_mm_cvtepi32_ps(_mm_sub_epi32(avx512_hsum4_2x512(r01, r23), vbias)), v_scale, vacc3);
            }
        }

        _mm_store_ps(tile_acc + 0u,  vacc0);
        _mm_store_ps(tile_acc + 4u,  vacc1);
        _mm_store_ps(tile_acc + 8u,  vacc2);
        _mm_store_ps(tile_acc + 12u, vacc3);
        for (uint32_t lr = 0; lr < rem_rows; lr++) out[r_base + lr] = tile_acc[lr];
    }
}

QMX_X86_TARGET("avx512f,avx512bw,avx512vl,avx512vnni,f16c,fma")
static void avx512_matmul_tiled(float         * QMX_RESTRICT out,
                                const int8_t  * QMX_RESTRICT xq,
                                const int16_t * QMX_RESTRICT xq16,
                                const float   * QMX_RESTRICT x_scales,
                                const QmxTensor *w,
                                int accumulate) {
    (void)xq16;
    uint8_t qt = w->quant_type;
    if (qt == QMX_QUANT_Q4_SYM) {
        avx512_matmul_q4(out, xq, NULL, NULL, x_scales, w, accumulate, 0, 0);
    } else if (qt == QMX_QUANT_Q4_ROW32) {
        avx512_matmul_q4(out, xq, NULL, NULL, x_scales, w, accumulate, 1, 0);
    } else if (qt == QMX_QUANT_Q8_SYM) {
        avx512_matmul_q8(out, xq, NULL, x_scales, w, accumulate, 0);
    } else if (qt == QMX_QUANT_Q8_ROW32) {
        avx512_matmul_q8(out, xq, NULL, x_scales, w, accumulate, 1);
    } else {
        avx512_matmul_lowbit(out, xq, NULL, x_scales, w, accumulate, qt);
    }
    QMX_MSVC_ZEROUPPER();
}

QMX_X86_TARGET("avx512f,avx512bw,avx512vl,avx512vnni,f16c,fma")
static void avx512_matmul_tiled_eo(float         * QMX_RESTRICT out,
                                   const int8_t  * QMX_RESTRICT xq,
                                   const int8_t  * QMX_RESTRICT xq_eo,
                                   const int32_t * QMX_RESTRICT x_sums,
                                   const float   * QMX_RESTRICT x_scales,
                                   const QmxTensor *w,
                                   int accumulate) {
    if (!xq_eo || !x_sums) {
        avx512_matmul_tiled(out, xq, NULL, x_scales, w, accumulate);
        return;
    }
    uint8_t qt = w->quant_type;
    if (qt == QMX_QUANT_Q4_SYM) {
        avx512_matmul_q4(out, xq, xq_eo, x_sums, x_scales, w, accumulate, 0, 1);
    } else if (qt == QMX_QUANT_Q4_ROW32) {
        avx512_matmul_q4(out, xq, xq_eo, x_sums, x_scales, w, accumulate, 1, 1);
    } else if (qt == QMX_QUANT_Q8_SYM) {
        avx512_matmul_q8(out, xq, x_sums, x_scales, w, accumulate, 0);
    } else if (qt == QMX_QUANT_Q8_ROW32) {
        avx512_matmul_q8(out, xq, x_sums, x_scales, w, accumulate, 1);
    } else {
        avx512_matmul_lowbit(out, xq, x_sums, x_scales, w, accumulate, qt);
    }
    QMX_MSVC_ZEROUPPER();
}

QMX_X86_TARGET("avx512f,avx512bw,avx512vl,avx512vnni")
static inline __m512i avx512_reduce4x4_epi32(__m512i d0, __m512i d1, __m512i d2, __m512i d3) {
    __m512i t0  = _mm512_unpacklo_epi32(d0, d1);
    __m512i t1  = _mm512_unpackhi_epi32(d0, d1);
    __m512i t2  = _mm512_unpacklo_epi32(d2, d3);
    __m512i t3  = _mm512_unpackhi_epi32(d2, d3);
    __m512i s01 = _mm512_add_epi32(t0, t1);
    __m512i s23 = _mm512_add_epi32(t2, t3);
    return _mm512_add_epi32(_mm512_unpacklo_epi64(s01, s23),
                            _mm512_unpackhi_epi64(s01, s23));
}

QMX_X86_TARGET("avx512f,avx512bw,avx512vl,avx512vnni,f16c,fma")
static inline __m512 avx512_load_ws4_bcast(const uint16_t *rs, __m512i ws_idx) {
    __m128 ws4 = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)rs));
    return _mm512_permutexvar_ps(ws_idx, _mm512_castps128_ps512(ws4));
}

QMX_X86_TARGET("avx512f,avx512bw,avx512vl,avx512vnni,f16c,fma")
static inline void avx512_store_zmm4x4(float * const *dst4, uint32_t row_off,
                                       uint32_t rem_rows, __m512 acc) {
    QMX_ALIGNED(64) float tmp[16];
    _mm512_store_ps(tmp, acc);
    for (uint32_t k = 0; k < 4u && row_off + k < rem_rows; k++) {
        dst4[0][row_off + k] = tmp[k * 4u + 0u];
        dst4[1][row_off + k] = tmp[k * 4u + 1u];
        dst4[2][row_off + k] = tmp[k * 4u + 2u];
        dst4[3][row_off + k] = tmp[k * 4u + 3u];
    }
}

QMX_X86_TARGET("avx512f,avx512bw,avx512vl,avx512vnni,f16c,fma")
static inline __m512 avx512_load_zmm4x4(float * const *dst4, uint32_t row_off,
                                        uint32_t rem_rows, int accumulate) {
    if (!accumulate) return _mm512_setzero_ps();
    QMX_ALIGNED(64) float tmp[16];
    for (uint32_t k = 0; k < 4u; k++) {
        if (row_off + k < rem_rows) {
            tmp[k * 4u + 0u] = dst4[0][row_off + k];
            tmp[k * 4u + 1u] = dst4[1][row_off + k];
            tmp[k * 4u + 2u] = dst4[2][row_off + k];
            tmp[k * 4u + 3u] = dst4[3][row_off + k];
        } else {
            tmp[k * 4u + 0u] = 0.0f; tmp[k * 4u + 1u] = 0.0f;
            tmp[k * 4u + 2u] = 0.0f; tmp[k * 4u + 3u] = 0.0f;
        }
    }
    return _mm512_load_ps(tmp);
}

QMX_X86_TARGET("avx512f,avx512bw,avx512vl,avx512vnni,f16c,fma")
static void avx512_matmul_q4_batch8(float * const *dst_batch,
                                    int8_t * const *xq_eo_batch,
                                    int32_t * const *xsums_batch,
                                    float * const *xscales_batch,
                                    const QmxTensor *w,
                                    int accumulate,
                                    int is_row_scaled) {
    uint32_t orig_rows = w->rows;
    uint32_t n_tr = (uint32_t)qmx_align_up(orig_rows, QMX_TILE_ROWS) / QMX_TILE_ROWS;
    uint32_t n_tc = (uint32_t)qmx_align_up(w->cols, QMX_TILE_COLS) / QMX_TILE_COLS;
    const uint8_t *base_data = (const uint8_t *)w->data;
    const uint16_t *scales_f16 = (const uint16_t *)(const void *)w->scales;
    const __m512i mask0f = _mm512_set1_epi8(0x0F);
    const __m512i ws_idx = _mm512_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);

    for (uint32_t tr = 0; tr < n_tr; tr++) {
        uint32_t r_base = tr * QMX_TILE_ROWS;
        uint32_t rem_rows = (r_base < orig_rows) ? (orig_rows - r_base) : 0u;
        if (rem_rows > QMX_TILE_ROWS) rem_rows = QMX_TILE_ROWS;

        float *dA[4] = { dst_batch[0] + r_base, dst_batch[1] + r_base, dst_batch[2] + r_base, dst_batch[3] + r_base };
        float *dB[4] = { dst_batch[4] + r_base, dst_batch[5] + r_base, dst_batch[6] + r_base, dst_batch[7] + r_base };

        __m512 a0 = avx512_load_zmm4x4(dA,  0u, rem_rows, accumulate);
        __m512 a1 = avx512_load_zmm4x4(dA,  4u, rem_rows, accumulate);
        __m512 a2 = avx512_load_zmm4x4(dA,  8u, rem_rows, accumulate);
        __m512 a3 = avx512_load_zmm4x4(dA, 12u, rem_rows, accumulate);
        __m512 b0 = avx512_load_zmm4x4(dB,  0u, rem_rows, accumulate);
        __m512 b1 = avx512_load_zmm4x4(dB,  4u, rem_rows, accumulate);
        __m512 b2 = avx512_load_zmm4x4(dB,  8u, rem_rows, accumulate);
        __m512 b3 = avx512_load_zmm4x4(dB, 12u, rem_rows, accumulate);

        uint32_t tbase = tr * n_tc;
        for (uint32_t tc = 0; tc < n_tc; tc++) {
            uint32_t tid = tbase + tc;
            const uint8_t *tp = base_data + (size_t)tid * 256u;
            if (tc + 2u < n_tc) _mm_prefetch((const char *)(tp + 512u), _MM_HINT_T0);

            size_t eo_off = (size_t)tc * QMX_TILE_COLS;
            const int8_t *e0 = xq_eo_batch[0] + eo_off;
            const int8_t *e1 = xq_eo_batch[1] + eo_off;
            const int8_t *e2 = xq_eo_batch[2] + eo_off;
            const int8_t *e3 = xq_eo_batch[3] + eo_off;
            const int8_t *e4 = xq_eo_batch[4] + eo_off;
            const int8_t *e5 = xq_eo_batch[5] + eo_off;
            const int8_t *e6 = xq_eo_batch[6] + eo_off;
            const int8_t *e7 = xq_eo_batch[7] + eo_off;

            __m512i biasA = _mm512_broadcast_i32x4(_mm_setr_epi32(
                xsums_batch[0][tc] * 8, xsums_batch[1][tc] * 8,
                xsums_batch[2][tc] * 8, xsums_batch[3][tc] * 8));
            __m512i biasB = _mm512_broadcast_i32x4(_mm_setr_epi32(
                xsums_batch[4][tc] * 8, xsums_batch[5][tc] * 8,
                xsums_batch[6][tc] * 8, xsums_batch[7][tc] * 8));
            __m512 xsA = _mm512_broadcast_f32x4(_mm_setr_ps(
                xscales_batch[0][tc], xscales_batch[1][tc],
                xscales_batch[2][tc], xscales_batch[3][tc]));
            __m512 xsB = _mm512_broadcast_f32x4(_mm_setr_ps(
                xscales_batch[4][tc], xscales_batch[5][tc],
                xscales_batch[6][tc], xscales_batch[7][tc]));

            const uint16_t *rs = is_row_scaled ? (scales_f16 + (size_t)tid * QMX_TILE_ROWS) : NULL;
            __m512 w_tile = is_row_scaled ? _mm512_setzero_ps() : _mm512_set1_ps(w->scales[tid]);

            #define Q4_ZMM_B8_GRP(LR, ACC_A, ACC_B) do { \
                __m512i raw = _mm512_loadu_si512((const void *)(tp + (LR) * 16u)); \
                __m512i lo  = _mm512_and_si512(raw, mask0f); \
                __m512i hi  = _mm512_and_si512(_mm512_srli_epi16(raw, 4), mask0f); \
                __m512  vws = is_row_scaled ? avx512_load_ws4_bcast(rs + (LR), ws_idx) : w_tile; \
                __m512i p0 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), lo, _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(e0 + 0u)))), hi, _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(e0 + 16u)))); \
                __m512i p1 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), lo, _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(e1 + 0u)))), hi, _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(e1 + 16u)))); \
                __m512i p2 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), lo, _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(e2 + 0u)))), hi, _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(e2 + 16u)))); \
                __m512i p3 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), lo, _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(e3 + 0u)))), hi, _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(e3 + 16u)))); \
                __m512i sA = _mm512_sub_epi32(avx512_reduce4x4_epi32(p0, p1, p2, p3), biasA); \
                ACC_A = _mm512_fmadd_ps(_mm512_cvtepi32_ps(sA), _mm512_mul_ps(vws, xsA), ACC_A); \
                __m512i p4 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), lo, _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(e4 + 0u)))), hi, _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(e4 + 16u)))); \
                __m512i p5 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), lo, _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(e5 + 0u)))), hi, _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(e5 + 16u)))); \
                __m512i p6 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), lo, _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(e6 + 0u)))), hi, _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(e6 + 16u)))); \
                __m512i p7 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), lo, _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(e7 + 0u)))), hi, _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(e7 + 16u)))); \
                __m512i sB = _mm512_sub_epi32(avx512_reduce4x4_epi32(p4, p5, p6, p7), biasB); \
                ACC_B = _mm512_fmadd_ps(_mm512_cvtepi32_ps(sB), _mm512_mul_ps(vws, xsB), ACC_B); \
            } while (0)

            Q4_ZMM_B8_GRP( 0u, a0, b0);
            Q4_ZMM_B8_GRP( 4u, a1, b1);
            Q4_ZMM_B8_GRP( 8u, a2, b2);
            Q4_ZMM_B8_GRP(12u, a3, b3);
            #undef Q4_ZMM_B8_GRP
        }

        avx512_store_zmm4x4(dA,  0u, rem_rows, a0);
        avx512_store_zmm4x4(dA,  4u, rem_rows, a1);
        avx512_store_zmm4x4(dA,  8u, rem_rows, a2);
        avx512_store_zmm4x4(dA, 12u, rem_rows, a3);
        avx512_store_zmm4x4(dB,  0u, rem_rows, b0);
        avx512_store_zmm4x4(dB,  4u, rem_rows, b1);
        avx512_store_zmm4x4(dB,  8u, rem_rows, b2);
        avx512_store_zmm4x4(dB, 12u, rem_rows, b3);
    }
}

QMX_X86_TARGET("avx512f,avx512bw,avx512vl,avx512vnni,f16c,fma")
static void avx512_matmul_q4_batch4(float * const *dst_batch,
                                    int8_t * const *xq_eo_batch,
                                    int32_t * const *xsums_batch,
                                    float * const *xscales_batch,
                                    const QmxTensor *w,
                                    int accumulate,
                                    int is_row_scaled) {
    uint32_t orig_rows = w->rows;
    uint32_t n_tr = (uint32_t)qmx_align_up(orig_rows, QMX_TILE_ROWS) / QMX_TILE_ROWS;
    uint32_t n_tc = (uint32_t)qmx_align_up(w->cols, QMX_TILE_COLS) / QMX_TILE_COLS;
    const uint8_t *base_data = (const uint8_t *)w->data;
    const uint16_t *scales_f16 = (const uint16_t *)(const void *)w->scales;
    const __m512i mask0f = _mm512_set1_epi8(0x0F);
    const __m512i ws_idx = _mm512_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);

    for (uint32_t tr = 0; tr < n_tr; tr++) {
        uint32_t r_base = tr * QMX_TILE_ROWS;
        uint32_t rem_rows = (r_base < orig_rows) ? (orig_rows - r_base) : 0u;
        if (rem_rows > QMX_TILE_ROWS) rem_rows = QMX_TILE_ROWS;

        float *dA[4] = { dst_batch[0] + r_base, dst_batch[1] + r_base, dst_batch[2] + r_base, dst_batch[3] + r_base };
        __m512 a0 = avx512_load_zmm4x4(dA,  0u, rem_rows, accumulate);
        __m512 a1 = avx512_load_zmm4x4(dA,  4u, rem_rows, accumulate);
        __m512 a2 = avx512_load_zmm4x4(dA,  8u, rem_rows, accumulate);
        __m512 a3 = avx512_load_zmm4x4(dA, 12u, rem_rows, accumulate);

        uint32_t tbase = tr * n_tc;
        for (uint32_t tc = 0; tc < n_tc; tc++) {
            uint32_t tid = tbase + tc;
            const uint8_t *tp = base_data + (size_t)tid * 256u;
            if (tc + 2u < n_tc) _mm_prefetch((const char *)(tp + 512u), _MM_HINT_T0);

            size_t eo_off = (size_t)tc * QMX_TILE_COLS;
            __m512i vxe0 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(xq_eo_batch[0] + eo_off + 0u)));
            __m512i vxo0 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(xq_eo_batch[0] + eo_off + 16u)));
            __m512i vxe1 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(xq_eo_batch[1] + eo_off + 0u)));
            __m512i vxo1 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(xq_eo_batch[1] + eo_off + 16u)));
            __m512i vxe2 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(xq_eo_batch[2] + eo_off + 0u)));
            __m512i vxo2 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(xq_eo_batch[2] + eo_off + 16u)));
            __m512i vxe3 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(xq_eo_batch[3] + eo_off + 0u)));
            __m512i vxo3 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(xq_eo_batch[3] + eo_off + 16u)));

            __m512i biasA = _mm512_broadcast_i32x4(_mm_setr_epi32(
                xsums_batch[0][tc] * 8, xsums_batch[1][tc] * 8,
                xsums_batch[2][tc] * 8, xsums_batch[3][tc] * 8));
            __m512 xsA = _mm512_broadcast_f32x4(_mm_setr_ps(
                xscales_batch[0][tc], xscales_batch[1][tc],
                xscales_batch[2][tc], xscales_batch[3][tc]));

            const uint16_t *rs = is_row_scaled ? (scales_f16 + (size_t)tid * QMX_TILE_ROWS) : NULL;
            __m512 w_tile = is_row_scaled ? _mm512_setzero_ps() : _mm512_set1_ps(w->scales[tid]);

            #define Q4_ZMM_B4_GRP(LR, ACC_A) do { \
                __m512i raw = _mm512_loadu_si512((const void *)(tp + (LR) * 16u)); \
                __m512i lo  = _mm512_and_si512(raw, mask0f); \
                __m512i hi  = _mm512_and_si512(_mm512_srli_epi16(raw, 4), mask0f); \
                __m512  vws = is_row_scaled ? avx512_load_ws4_bcast(rs + (LR), ws_idx) : w_tile; \
                __m512i p0 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), lo, vxe0), hi, vxo0); \
                __m512i p1 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), lo, vxe1), hi, vxo1); \
                __m512i p2 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), lo, vxe2), hi, vxo2); \
                __m512i p3 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), lo, vxe3), hi, vxo3); \
                __m512i sA = _mm512_sub_epi32(avx512_reduce4x4_epi32(p0, p1, p2, p3), biasA); \
                ACC_A = _mm512_fmadd_ps(_mm512_cvtepi32_ps(sA), _mm512_mul_ps(vws, xsA), ACC_A); \
            } while (0)

            Q4_ZMM_B4_GRP( 0u, a0);
            Q4_ZMM_B4_GRP( 4u, a1);
            Q4_ZMM_B4_GRP( 8u, a2);
            Q4_ZMM_B4_GRP(12u, a3);
            #undef Q4_ZMM_B4_GRP
        }

        avx512_store_zmm4x4(dA,  0u, rem_rows, a0);
        avx512_store_zmm4x4(dA,  4u, rem_rows, a1);
        avx512_store_zmm4x4(dA,  8u, rem_rows, a2);
        avx512_store_zmm4x4(dA, 12u, rem_rows, a3);
    }
}

QMX_X86_TARGET("avx512f,avx512bw,avx512vl,avx512vnni,f16c,fma")
static void avx512_matmul_q8_batch4(float * const *dst_batch,
                                    int8_t * const *xq_batch,
                                    int32_t * const *xsums_batch,
                                    float * const *xscales_batch,
                                    const QmxTensor *w,
                                    int accumulate,
                                    int is_row_scaled) {
    uint32_t orig_rows = w->rows;
    uint32_t n_tr = (uint32_t)qmx_align_up(orig_rows, QMX_TILE_ROWS) / QMX_TILE_ROWS;
    uint32_t n_tc = (uint32_t)qmx_align_up(w->cols, QMX_TILE_COLS) / QMX_TILE_COLS;
    const uint8_t *base_data = (const uint8_t *)w->data;
    const uint16_t *scales_f16 = (const uint16_t *)(const void *)w->scales;
    const __m512i bias128_512 = _mm512_set1_epi8((char)0x80);
    const __m512i ws_idx = _mm512_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);

    for (uint32_t tr = 0; tr < n_tr; tr++) {
        uint32_t r_base = tr * QMX_TILE_ROWS;
        uint32_t rem_rows = (r_base < orig_rows) ? (orig_rows - r_base) : 0u;
        if (rem_rows > QMX_TILE_ROWS) rem_rows = QMX_TILE_ROWS;

        float *dA[4] = { dst_batch[0] + r_base, dst_batch[1] + r_base, dst_batch[2] + r_base, dst_batch[3] + r_base };
        __m512 a0 = avx512_load_zmm4x4(dA,  0u, rem_rows, accumulate);
        __m512 a1 = avx512_load_zmm4x4(dA,  4u, rem_rows, accumulate);
        __m512 a2 = avx512_load_zmm4x4(dA,  8u, rem_rows, accumulate);
        __m512 a3 = avx512_load_zmm4x4(dA, 12u, rem_rows, accumulate);

        uint32_t tbase = tr * n_tc;
        for (uint32_t tc = 0; tc < n_tc; tc++) {
            uint32_t tid = tbase + tc;
            const uint8_t *tp = base_data + (size_t)tid * 512u;
            if (tc + 1u < n_tc) _mm_prefetch((const char *)(tp + 512u), _MM_HINT_T0);

            size_t x_off = (size_t)tc * QMX_TILE_COLS;
            __m512i vx0 = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *)(xq_batch[0] + x_off)));
            __m512i vx1 = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *)(xq_batch[1] + x_off)));
            __m512i vx2 = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *)(xq_batch[2] + x_off)));
            __m512i vx3 = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *)(xq_batch[3] + x_off)));

            __m512i biasA = _mm512_broadcast_i32x4(_mm_setr_epi32(
                xsums_batch[0][tc] * 128, xsums_batch[1][tc] * 128,
                xsums_batch[2][tc] * 128, xsums_batch[3][tc] * 128));
            __m512 xsA = _mm512_broadcast_f32x4(_mm_setr_ps(
                xscales_batch[0][tc], xscales_batch[1][tc],
                xscales_batch[2][tc], xscales_batch[3][tc]));

            const uint16_t *rs = is_row_scaled ? (scales_f16 + (size_t)tid * QMX_TILE_ROWS) : NULL;
            __m512 w_tile = is_row_scaled ? _mm512_setzero_ps() : _mm512_set1_ps(w->scales[tid]);

            #define Q8_ZMM_FOLD(P01, P23) _mm512_add_epi32(_mm512_shuffle_i32x4((P01), (P23), 0x88), _mm512_shuffle_i32x4((P01), (P23), 0xDD))
            #define Q8_ZMM_B4_GRP(LR, ACC_A) do { \
                __m512i w01 = _mm512_xor_si512(_mm512_loadu_si512((const void *)(tp + (LR) * 32u)), bias128_512); \
                __m512i w23 = _mm512_xor_si512(_mm512_loadu_si512((const void *)(tp + ((LR) + 2u) * 32u)), bias128_512); \
                __m512  vws = is_row_scaled ? avx512_load_ws4_bcast(rs + (LR), ws_idx) : w_tile; \
                __m512i d0 = Q8_ZMM_FOLD(_mm512_dpbusd_epi32(_mm512_setzero_si512(), w01, vx0), _mm512_dpbusd_epi32(_mm512_setzero_si512(), w23, vx0)); \
                __m512i d1 = Q8_ZMM_FOLD(_mm512_dpbusd_epi32(_mm512_setzero_si512(), w01, vx1), _mm512_dpbusd_epi32(_mm512_setzero_si512(), w23, vx1)); \
                __m512i d2 = Q8_ZMM_FOLD(_mm512_dpbusd_epi32(_mm512_setzero_si512(), w01, vx2), _mm512_dpbusd_epi32(_mm512_setzero_si512(), w23, vx2)); \
                __m512i d3 = Q8_ZMM_FOLD(_mm512_dpbusd_epi32(_mm512_setzero_si512(), w01, vx3), _mm512_dpbusd_epi32(_mm512_setzero_si512(), w23, vx3)); \
                __m512i sA = _mm512_sub_epi32(avx512_reduce4x4_epi32(d0, d1, d2, d3), biasA); \
                ACC_A = _mm512_fmadd_ps(_mm512_cvtepi32_ps(sA), _mm512_mul_ps(vws, xsA), ACC_A); \
            } while (0)

            Q8_ZMM_B4_GRP( 0u, a0);
            Q8_ZMM_B4_GRP( 4u, a1);
            Q8_ZMM_B4_GRP( 8u, a2);
            Q8_ZMM_B4_GRP(12u, a3);
            #undef Q8_ZMM_B4_GRP
            #undef Q8_ZMM_FOLD
        }

        avx512_store_zmm4x4(dA,  0u, rem_rows, a0);
        avx512_store_zmm4x4(dA,  4u, rem_rows, a1);
        avx512_store_zmm4x4(dA,  8u, rem_rows, a2);
        avx512_store_zmm4x4(dA, 12u, rem_rows, a3);
    }
}

QMX_X86_TARGET("avx512f,avx512bw,avx512vl,avx512vnni,f16c,fma")
static void avx512_matmul_tiled_batch(float   * const *dst_batch,
                                      int8_t  * const *xq_batch,
                                      int8_t  * const *xq_eo_batch,
                                      int32_t * const *xsums_batch,
                                      float   * const *xscales_batch,
                                      int batch_size,
                                      const QmxTensor *w,
                                      int accumulate) {
    uint8_t qt = w->quant_type;
    int is_q4 = (qt == QMX_QUANT_Q4_SYM || qt == QMX_QUANT_Q4_ROW32);
    int is_q8 = (qt == QMX_QUANT_Q8_SYM || qt == QMX_QUANT_Q8_ROW32);
    int is_rs = (qt == QMX_QUANT_Q4_ROW32 || qt == QMX_QUANT_Q8_ROW32);

    int b = 0;
    if (is_q4 && xq_eo_batch && xsums_batch) {
        while (b + 8 <= batch_size) {
            int ok = 1;
            for (int k = 0; k < 8; k++) {
                if (!xq_eo_batch[b + k] || !xsums_batch[b + k]) { ok = 0; break; }
            }
            if (!ok) break;
            avx512_matmul_q4_batch8(dst_batch + b, xq_eo_batch + b, xsums_batch + b,
                                    xscales_batch + b, w, accumulate, is_rs);
            b += 8;
        }
        while (b + 4 <= batch_size) {
            if (xq_eo_batch[b] && xq_eo_batch[b + 1] && xq_eo_batch[b + 2] && xq_eo_batch[b + 3] &&
                xsums_batch[b] && xsums_batch[b + 1] && xsums_batch[b + 2] && xsums_batch[b + 3]) {
                avx512_matmul_q4_batch4(dst_batch + b, xq_eo_batch + b, xsums_batch + b,
                                        xscales_batch + b, w, accumulate, is_rs);
                b += 4;
            } else {
                break;
            }
        }
    } else if (is_q8 && xsums_batch) {
        while (b + 4 <= batch_size) {
            if (xsums_batch[b] && xsums_batch[b + 1] && xsums_batch[b + 2] && xsums_batch[b + 3]) {
                avx512_matmul_q8_batch4(dst_batch + b, xq_batch + b, xsums_batch + b,
                                        xscales_batch + b, w, accumulate, is_rs);
                b += 4;
            } else {
                break;
            }
        }
    }

    for (; b < batch_size; b++) {
        if (xq_eo_batch && xsums_batch && xq_eo_batch[b] && xsums_batch[b]) {
            avx512_matmul_tiled_eo(dst_batch[b], xq_batch[b], xq_eo_batch[b], xsums_batch[b],
                                   xscales_batch[b], w, accumulate);
        } else {
            avx512_matmul_tiled(dst_batch[b], xq_batch[b], NULL, xscales_batch[b], w, accumulate);
        }
    }
    QMX_MSVC_ZEROUPPER();
}

QMX_X86_TARGET("avx512f,avx512bw,avx512vl,avx512vnni,f16c,fma")
static inline void avx512_quant_group32(__m512 v0, __m512 v1,
                                        int8_t * QMX_RESTRICT dst, float * QMX_RESTRICT scale_out) {
    const __m512i abs_mask = _mm512_set1_epi32(0x7FFFFFFF);
    const __m256i min_i8   = _mm256_set1_epi8(-127);

    __m512 a0 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(v0), abs_mask));
    __m512 a1 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(v1), abs_mask));
    float max_abs = avx512_hmax_ps(_mm512_max_ps(a0, a1));

    *scale_out = max_abs / 127.0f;
    float inv_scale = (max_abs == 0.0f) ? 0.0f : (127.0f / max_abs);
    __m512 v_inv = _mm512_set1_ps(inv_scale);

    __m128i b0 = _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(_mm512_mul_ps(v0, v_inv)));
    __m128i b1 = _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(_mm512_mul_ps(v1, v_inv)));
    __m256i b32 = _mm256_max_epi8(_mm256_inserti128_si256(_mm256_castsi128_si256(b0), b1, 1), min_i8);
    _mm256_storeu_si256((__m256i *)dst, b32);
}

QMX_X86_TARGET("avx512f,avx512bw,avx512vl,avx512vnni,f16c,fma")
static inline __m512 avx512_expf_ps(__m512 x) {
    const __m512 hi = _mm512_set1_ps(88.0f);
    const __m512 lo = _mm512_set1_ps(-87.3f);
    __mmask16 valid = _mm512_cmp_ps_mask(x, lo, _CMP_GE_OQ);
    x = _mm512_min_ps(_mm512_max_ps(x, lo), hi);
    __m512i ki = _mm512_cvtps_epi32(_mm512_mul_ps(x, _mm512_set1_ps(1.4426950408889634f)));
    __m512 kf = _mm512_cvtepi32_ps(ki);
    __m512 r = _mm512_fnmadd_ps(kf, _mm512_set1_ps(0.693145751953125f), x);
    r = _mm512_fnmadd_ps(kf, _mm512_set1_ps(1.428606765330187e-6f), r);
    __m512 p = _mm512_set1_ps(1.38888889e-3f);
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(8.33333333e-3f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(4.16666667e-2f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.66666667e-1f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(5.0e-1f));
    p = _mm512_fmadd_ps(p, _mm512_mul_ps(r, r), _mm512_add_ps(r, _mm512_set1_ps(1.0f)));
    __m512 res = _mm512_castsi512_ps(_mm512_add_epi32(_mm512_castps_si512(p), _mm512_slli_epi32(ki, 23)));
    return _mm512_maskz_mov_ps(valid, res);
}

QMX_X86_TARGET("avx512f,avx512bw,avx512vl,avx512vnni,f16c,fma")
static void avx512_rmsnorm_quant(const float * QMX_RESTRICT x,
                                 const float * QMX_RESTRICT weight,
                                 int8_t      * QMX_RESTRICT xq,
                                 int16_t     * QMX_RESTRICT xq16,
                                 float       * QMX_RESTRICT x_scales,
                                 uint32_t size,
                                 float eps) {
    (void)xq16;
    __m512 sum16 = _mm512_setzero_ps();
    uint32_t i = 0;
    for (; i + 15u < size; i += 16u) {
        __m512 v = _mm512_loadu_ps(x + i);
        sum16 = _mm512_fmadd_ps(v, v, sum16);
    }
    float ss = avx512_hsum_ps(sum16);
    for (; i < size; i++) ss += x[i] * x[i];
    float inv_rms = 1.0f / qmx_sqrtf(ss / (float)size + eps);
    __m512 v_inv = _mm512_set1_ps(inv_rms);

    uint32_t full_groups = size / QMX_TILE_COLS;
    for (uint32_t g = 0; g < full_groups; g++) {
        uint32_t base = g * QMX_TILE_COLS;
        __m512 v0 = _mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(x + base + 0u),  v_inv), _mm512_loadu_ps(weight + base + 0u));
        __m512 v1 = _mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(x + base + 16u), v_inv), _mm512_loadu_ps(weight + base + 16u));
        avx512_quant_group32(v0, v1, xq + base, &x_scales[g]);
    }

    QMX_MSVC_ZEROUPPER();
    uint32_t tail_base = full_groups * QMX_TILE_COLS;
    if (tail_base < size) {
        uint32_t rem = size - tail_base;
        float tmp[QMX_TILE_COLS];
        for (uint32_t t = 0; t < rem; t++) {
            tmp[t] = weight[tail_base + t] * (inv_rms * x[tail_base + t]);
        }
        qmx_quantize_activations_q8(tmp, xq + tail_base, x_scales + full_groups, rem);
    }
}

QMX_X86_TARGET("avx512f,avx512bw,avx512vl,avx512vnni,f16c,fma")
static void avx512_swiglu_quant(const float * QMX_RESTRICT gate,
                                const float * QMX_RESTRICT up,
                                int8_t      * QMX_RESTRICT xq,
                                int16_t     * QMX_RESTRICT xq16,
                                float       * QMX_RESTRICT x_scales,
                                uint32_t size) {
    (void)xq16;
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512i sign_bit = _mm512_set1_epi32((int)0x80000000u);
    uint32_t full_groups = size / QMX_TILE_COLS;

    for (uint32_t g = 0; g < full_groups; g++) {
        uint32_t base = g * QMX_TILE_COLS;
        __m512 g0 = _mm512_loadu_ps(gate + base + 0u);
        __m512 g1 = _mm512_loadu_ps(gate + base + 16u);
        __m512 ng0 = _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(g0), sign_bit));
        __m512 ng1 = _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(g1), sign_bit));

        const __m512 two = _mm512_set1_ps(2.0f);
        __m512 d0 = _mm512_add_ps(one, avx512_expf_ps(ng0));
        __m512 d1 = _mm512_add_ps(one, avx512_expf_ps(ng1));
        __m512 r0 = _mm512_rcp14_ps(d0); r0 = _mm512_mul_ps(r0, _mm512_fnmadd_ps(d0, r0, two));
        __m512 r1 = _mm512_rcp14_ps(d1); r1 = _mm512_mul_ps(r1, _mm512_fnmadd_ps(d1, r1, two));
        __m512 s0 = _mm512_mul_ps(g0, r0);
        __m512 s1 = _mm512_mul_ps(g1, r1);

        __m512 v0 = _mm512_mul_ps(s0, _mm512_loadu_ps(up + base + 0u));
        __m512 v1 = _mm512_mul_ps(s1, _mm512_loadu_ps(up + base + 16u));
        avx512_quant_group32(v0, v1, xq + base, &x_scales[g]);
    }

    QMX_MSVC_ZEROUPPER();
    uint32_t tail_base = full_groups * QMX_TILE_COLS;
    if (tail_base < size) {
        uint32_t rem = size - tail_base;
        float tmp[QMX_TILE_COLS];
        for (uint32_t t = 0; t < rem; t++) {
            float gv = gate[tail_base + t];
            tmp[t] = gv * (1.0f / (1.0f + qmx_expf(-gv))) * up[tail_base + t];
        }
        qmx_quantize_activations_q8(tmp, xq + tail_base, x_scales + full_groups, rem);
    }
}

QMX_X86_TARGET("avx512f,avx512bw,avx512vl,avx512vnni,f16c,fma")
static void avx512_f16_to_f32(float * QMX_RESTRICT dst, const uint16_t * QMX_RESTRICT src, uint32_t n) {
    uint32_t i = 0;
    for (; i + 15u < n; i += 16u) {
        __m256i h16 = _mm256_loadu_si256((const __m256i *)(src + i));
        _mm512_storeu_ps(dst + i, _mm512_cvtph_ps(h16));
    }
    if (i + 7u < n) {
        __m128i h8 = _mm_loadu_si128((const __m128i *)(src + i));
        _mm256_storeu_ps(dst + i, _mm256_cvtph_ps(h8));
        i += 8u;
    }
    QMX_MSVC_ZEROUPPER();
    if (i + 3u < n) {
        __m128i h4 = _mm_loadl_epi64((const __m128i *)(src + i));
        _mm_storeu_ps(dst + i, _mm_cvtph_ps(h4));
        i += 4u;
    }
    for (; i < n; i++) dst[i] = qmx_fp16_to_fp32(src[i]);
}

QMX_X86_TARGET("avx512f,avx512bw,avx512vl,avx512vnni,f16c,fma")
static void avx512_f32_to_f16(uint16_t * QMX_RESTRICT dst, const float * QMX_RESTRICT src, uint32_t n) {
    uint32_t i = 0;
    for (; i + 15u < n; i += 16u) {
        __m512 v16 = _mm512_loadu_ps(src + i);
        _mm256_storeu_si256((__m256i *)(dst + i), _mm512_cvtps_ph(v16, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }
    if (i + 7u < n) {
        __m256 v8 = _mm256_loadu_ps(src + i);
        _mm_storeu_si128((__m128i *)(dst + i), _mm256_cvtps_ph(v8, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        i += 8u;
    }
    QMX_MSVC_ZEROUPPER();
    if (i + 3u < n) {
        __m128 v4 = _mm_loadu_ps(src + i);
        _mm_storel_epi64((__m128i *)(dst + i), _mm_cvtps_ph(v4, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        i += 4u;
    }
    for (; i < n; i++) dst[i] = qmx_fp32_to_fp16(src[i]);
}

QMX_X86_TARGET("avx512f,avx512bw,avx512vl,avx512vnni,f16c,fma")
static float avx512_dot_f32_f16(const float * QMX_RESTRICT a,
                                const uint16_t * QMX_RESTRICT b,
                                uint32_t n) {
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    uint32_t i = 0;
    for (; i + 31u < n; i += 32u) {
        __m512 vb0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b + i)));
        __m512 vb1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b + i + 16u)));
        sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), vb0, sum0);
        sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16u), vb1, sum1);
    }
    sum0 = _mm512_add_ps(sum0, sum1);
    for (; i + 15u < n; i += 16u) {
        __m512 vb = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b + i)));
        sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), vb, sum0);
    }
    float sum = avx512_hsum_ps(sum0);
    QMX_MSVC_ZEROUPPER();
    for (; i < n; i++) sum += a[i] * qmx_fp16_to_fp32(b[i]);
    return sum;
}

QMX_X86_TARGET("avx512f,avx512bw,avx512vl,avx512vnni,f16c,fma")
static void avx512_saxpy_f32_f16(float * QMX_RESTRICT y,
                                 float a,
                                 const uint16_t * QMX_RESTRICT x,
                                 uint32_t n) {
    __m512 va = _mm512_set1_ps(a);
    uint32_t i = 0;
    for (; i + 31u < n; i += 32u) {
        __m512 vx0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(x + i)));
        __m512 vx1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(x + i + 16u)));
        _mm512_storeu_ps(y + i,       _mm512_fmadd_ps(va, vx0, _mm512_loadu_ps(y + i)));
        _mm512_storeu_ps(y + i + 16u, _mm512_fmadd_ps(va, vx1, _mm512_loadu_ps(y + i + 16u)));
    }
    for (; i + 15u < n; i += 16u) {
        __m512 vx = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(x + i)));
        _mm512_storeu_ps(y + i, _mm512_fmadd_ps(va, vx, _mm512_loadu_ps(y + i)));
    }
    QMX_MSVC_ZEROUPPER();
    for (; i < n; i++) y[i] += a * qmx_fp16_to_fp32(x[i]);
}

#endif

const QmxBackendOps qmx_ops_sse = {
    "SSE", sse1_prepare_act, sse1_matmul_tiled, sse1_rmsnorm, sse1_dot_f32, sse1_saxpy_f32, sse1_dot_q8, NULL, qmx_swiglu_quant_scalar, qmx_f16_to_f32_scalar, qmx_f32_to_f16_scalar, NULL, NULL, NULL, NULL, NULL
};

const QmxBackendOps qmx_ops_sse3 = {
    "SSE3", x86_ext_prepare_act_q8_q16, sse3_matmul_tiled, sse3_rmsnorm, sse3_dot_f32, sse1_saxpy_f32, sse3_dot_q8, NULL, qmx_swiglu_quant_scalar, qmx_f16_to_f32_scalar, qmx_f32_to_f16_scalar, NULL, NULL, NULL, NULL, NULL
};

const QmxBackendOps qmx_ops_ssse3 = {
    "SSSE3", x86_ext_prepare_act_q8, ssse3_matmul_tiled, sse3_rmsnorm, sse3_dot_f32, sse1_saxpy_f32, ssse3_dot_q8, NULL, qmx_swiglu_quant_scalar, qmx_f16_to_f32_scalar, qmx_f32_to_f16_scalar, NULL, NULL, NULL, NULL, NULL
};

const QmxBackendOps qmx_ops_sse41 = {
    "SSE4.1", x86_ext_prepare_act_q8, ssse3_matmul_tiled, sse3_rmsnorm, sse41_dot_f32, sse1_saxpy_f32, sse41_dot_q8, NULL, qmx_swiglu_quant_scalar, qmx_f16_to_f32_scalar, qmx_f32_to_f16_scalar, NULL, NULL, NULL, NULL, NULL
};

const QmxBackendOps qmx_ops_sse42 = {
    "SSE4.2", x86_ext_prepare_act_q8, ssse3_matmul_tiled, sse3_rmsnorm, sse41_dot_f32, sse1_saxpy_f32, sse42_dot_q8, NULL, qmx_swiglu_quant_scalar, qmx_f16_to_f32_scalar, qmx_f32_to_f16_scalar, NULL, NULL, NULL, NULL, NULL
};

const QmxBackendOps qmx_ops_avx = {
    "AVX", x86_ext_prepare_act_q8, avx1_matmul_tiled, avx1_rmsnorm, avx1_dot_f32, avx1_saxpy_f32, sse42_dot_q8, NULL, qmx_swiglu_quant_scalar, qmx_f16_to_f32_scalar, qmx_f32_to_f16_scalar, NULL, NULL, NULL, NULL, NULL
};

#if defined(QMX_ENABLE_BACKEND_AVX512)
const QmxBackendOps qmx_ops_avx512 = {
    "AVX-512", avx512_prepare_act, avx512_matmul_tiled, avx512_rmsnorm, avx512_dot_f32, avx512_saxpy_f32, avx512_dot_q8, avx512_rmsnorm_quant, avx512_swiglu_quant, avx512_f16_to_f32, avx512_f32_to_f16, avx512_prepare_act_eo, avx512_matmul_tiled_eo, avx512_matmul_tiled_batch, avx512_dot_f32_f16, avx512_saxpy_f32_f16
};
#endif
#endif
