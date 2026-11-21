#include "../include/qmx_kernels.h"

#if defined(QMX_ENABLE_BACKEND_SSE2)
#if defined(__GNUC__) || defined(__clang__)
    #define QMX_SSE2_TARGET __attribute__((target("sse2")))
#else
    #define QMX_SSE2_TARGET
#endif
#include <emmintrin.h>

QMX_SSE2_TARGET
static void sse2_prepare_act(const float * QMX_RESTRICT x,
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
        __m128i lo16 = _mm_unpacklo_epi8(v8, sign);
        __m128i hi16 = _mm_unpackhi_epi8(v8, sign);
        _mm_storeu_si128((__m128i *)(xq16 + i), lo16);
        _mm_storeu_si128((__m128i *)(xq16 + i + 8u), hi16);
    }
}

QMX_SSE2_TARGET
static inline int32_t sse2_hsum_epi32(__m128i v) {
    __m128i hi64 = _mm_unpackhi_epi64(v, v);
    v = _mm_add_epi32(v, hi64);
    __m128i hi32 = _mm_shuffle_epi32(v, _MM_SHUFFLE(1, 1, 1, 1));
    v = _mm_add_epi32(v, hi32);
    return _mm_cvtsi128_si32(v);
}

QMX_SSE2_TARGET
static inline __m128i sse2_hsum4_epi32(__m128i r0, __m128i r1, __m128i r2, __m128i r3) {
    __m128i t0 = _mm_unpacklo_epi32(r0, r1);
    __m128i t1 = _mm_unpackhi_epi32(r0, r1);
    __m128i t2 = _mm_unpacklo_epi32(r2, r3);
    __m128i t3 = _mm_unpackhi_epi32(r2, r3);
    __m128i s01 = _mm_add_epi32(t0, t1);
    __m128i s23 = _mm_add_epi32(t2, t3);
    __m128i lo = _mm_unpacklo_epi64(s01, s23);
    __m128i hi = _mm_unpackhi_epi64(s01, s23);
    return _mm_add_epi32(lo, hi);
}

QMX_SSE2_TARGET
static inline float sse2_hsum_ps(__m128 v) {
    __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
    v = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, v);
    v = _mm_add_ss(v, shuf);
    return _mm_cvtss_f32(v);
}

QMX_SSE2_TARGET
static inline __m128 sse2_load_row_scales4(const uint16_t *sc16, float x_sc) {
    __m128 s = _mm_set_ps(qmx_fp16_to_fp32(sc16[3]),
                          qmx_fp16_to_fp32(sc16[2]),
                          qmx_fp16_to_fp32(sc16[1]),
                          qmx_fp16_to_fp32(sc16[0]));
    return _mm_mul_ps(s, _mm_set1_ps(x_sc));
}

QMX_SSE2_TARGET
static inline __m128i sse2_dot_row_vec(const uint8_t *tile_ptr, uint32_t lr, uint8_t base_qtype,
                                       __m128i x0, __m128i x1, __m128i x2, __m128i x3,
                                       __m128i zero, __m128i mask0f, __m128i bias8) {
    if (base_qtype == QMX_QUANT_Q8_SYM) {
        const uint8_t *rp = tile_ptr + lr * 32u;
        __m128i w0 = _mm_loadu_si128((const __m128i *)rp);
        __m128i w1 = _mm_loadu_si128((const __m128i *)(rp + 16u));
        __m128i s0 = _mm_cmpgt_epi8(zero, w0);
        __m128i s1 = _mm_cmpgt_epi8(zero, w1);
        __m128i acc = _mm_madd_epi16(_mm_unpacklo_epi8(w0, s0), x0);
        acc = _mm_add_epi32(acc, _mm_madd_epi16(_mm_unpackhi_epi8(w0, s0), x1));
        acc = _mm_add_epi32(acc, _mm_madd_epi16(_mm_unpacklo_epi8(w1, s1), x2));
        return _mm_add_epi32(acc, _mm_madd_epi16(_mm_unpackhi_epi8(w1, s1), x3));
    } else if (base_qtype == QMX_QUANT_Q4_SYM) {
        const uint8_t *rp = tile_ptr + lr * 16u;
        __m128i packed = _mm_loadu_si128((const __m128i *)rp);
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

QMX_SSE2_TARGET
static void sse2_matmul_tiled(float         * QMX_RESTRICT out,
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

    float tile_acc[QMX_TILE_ROWS] QMX_ALIGNED(16);

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

            __m128i x0 = _mm_loadu_si128((const __m128i *)(xp + 0u));
            __m128i x1 = _mm_loadu_si128((const __m128i *)(xp + 8u));
            __m128i x2 = _mm_loadu_si128((const __m128i *)(xp + 16u));
            __m128i x3 = _mm_loadu_si128((const __m128i *)(xp + 24u));

            if (rem_rows > 0u) {
                __m128i r0 = sse2_dot_row_vec(tile_ptr, 0u, base_qtype, x0, x1, x2, x3, zero, mask0f, bias8);
                __m128i r1 = sse2_dot_row_vec(tile_ptr, 1u, base_qtype, x0, x1, x2, x3, zero, mask0f, bias8);
                __m128i r2 = sse2_dot_row_vec(tile_ptr, 2u, base_qtype, x0, x1, x2, x3, zero, mask0f, bias8);
                __m128i r3 = sse2_dot_row_vec(tile_ptr, 3u, base_qtype, x0, x1, x2, x3, zero, mask0f, bias8);
                __m128 vs = is_row_scaled ? sse2_load_row_scales4(row_sc + 0u, x_sc) : v_scale;
                vacc0 = _mm_add_ps(vacc0, _mm_mul_ps(_mm_cvtepi32_ps(sse2_hsum4_epi32(r0, r1, r2, r3)), vs));
            }
            if (rem_rows > 4u) {
                __m128i r0 = sse2_dot_row_vec(tile_ptr, 4u, base_qtype, x0, x1, x2, x3, zero, mask0f, bias8);
                __m128i r1 = sse2_dot_row_vec(tile_ptr, 5u, base_qtype, x0, x1, x2, x3, zero, mask0f, bias8);
                __m128i r2 = sse2_dot_row_vec(tile_ptr, 6u, base_qtype, x0, x1, x2, x3, zero, mask0f, bias8);
                __m128i r3 = sse2_dot_row_vec(tile_ptr, 7u, base_qtype, x0, x1, x2, x3, zero, mask0f, bias8);
                __m128 vs = is_row_scaled ? sse2_load_row_scales4(row_sc + 4u, x_sc) : v_scale;
                vacc1 = _mm_add_ps(vacc1, _mm_mul_ps(_mm_cvtepi32_ps(sse2_hsum4_epi32(r0, r1, r2, r3)), vs));
            }
            if (rem_rows > 8u) {
                __m128i r0 = sse2_dot_row_vec(tile_ptr, 8u,  base_qtype, x0, x1, x2, x3, zero, mask0f, bias8);
                __m128i r1 = sse2_dot_row_vec(tile_ptr, 9u,  base_qtype, x0, x1, x2, x3, zero, mask0f, bias8);
                __m128i r2 = sse2_dot_row_vec(tile_ptr, 10u, base_qtype, x0, x1, x2, x3, zero, mask0f, bias8);
                __m128i r3 = sse2_dot_row_vec(tile_ptr, 11u, base_qtype, x0, x1, x2, x3, zero, mask0f, bias8);
                __m128 vs = is_row_scaled ? sse2_load_row_scales4(row_sc + 8u, x_sc) : v_scale;
                vacc2 = _mm_add_ps(vacc2, _mm_mul_ps(_mm_cvtepi32_ps(sse2_hsum4_epi32(r0, r1, r2, r3)), vs));
            }
            if (rem_rows > 12u) {
                __m128i r0 = sse2_dot_row_vec(tile_ptr, 12u, base_qtype, x0, x1, x2, x3, zero, mask0f, bias8);
                __m128i r1 = sse2_dot_row_vec(tile_ptr, 13u, base_qtype, x0, x1, x2, x3, zero, mask0f, bias8);
                __m128i r2 = sse2_dot_row_vec(tile_ptr, 14u, base_qtype, x0, x1, x2, x3, zero, mask0f, bias8);
                __m128i r3 = sse2_dot_row_vec(tile_ptr, 15u, base_qtype, x0, x1, x2, x3, zero, mask0f, bias8);
                __m128 vs = is_row_scaled ? sse2_load_row_scales4(row_sc + 12u, x_sc) : v_scale;
                vacc3 = _mm_add_ps(vacc3, _mm_mul_ps(_mm_cvtepi32_ps(sse2_hsum4_epi32(r0, r1, r2, r3)), vs));
            }
        }

        _mm_store_ps(tile_acc + 0u,  vacc0);
        _mm_store_ps(tile_acc + 4u,  vacc1);
        _mm_store_ps(tile_acc + 8u,  vacc2);
        _mm_store_ps(tile_acc + 12u, vacc3);

        for (uint32_t lr = 0; lr < rem_rows; lr++) {
            out[r_base + lr] = tile_acc[lr];
        }
    }
}

QMX_SSE2_TARGET
static void sse2_rmsnorm(float       * QMX_RESTRICT out,
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
    float ss = sse2_hsum_ps(sum4);
    for (; i < size; i++) ss += x[i] * x[i];
    float inv_rms = 1.0f / qmx_sqrtf(ss / (float)size + eps);
    __m128 v_inv = _mm_set1_ps(inv_rms);
    i = 0;
    for (; i + 3u < size; i += 4u) {
        __m128 vx = _mm_loadu_ps(x + i);
        __m128 vw = _mm_loadu_ps(weight + i);
        _mm_storeu_ps(out + i, _mm_mul_ps(_mm_mul_ps(vx, v_inv), vw));
    }
    for (; i < size; i++) out[i] = weight[i] * (inv_rms * x[i]);
}

QMX_SSE2_TARGET
static float sse2_dot_f32(const float * QMX_RESTRICT a,
                          const float * QMX_RESTRICT b,
                          uint32_t n) {
    __m128 sum4 = _mm_setzero_ps();
    uint32_t i = 0;
    for (; i + 3u < n; i += 4u) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        sum4 = _mm_add_ps(sum4, _mm_mul_ps(va, vb));
    }
    float sum = sse2_hsum_ps(sum4);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

QMX_SSE2_TARGET
static void sse2_saxpy_f32(float       * QMX_RESTRICT y,
                           float a,
                           const float * QMX_RESTRICT x,
                           uint32_t n) {
    __m128 va = _mm_set1_ps(a);
    uint32_t i = 0;
    for (; i + 3u < n; i += 4u) {
        __m128 vy = _mm_loadu_ps(y + i);
        __m128 vx = _mm_loadu_ps(x + i);
        _mm_storeu_ps(y + i, _mm_add_ps(vy, _mm_mul_ps(va, vx)));
    }
    for (; i < n; i++) y[i] += a * x[i];
}

QMX_SSE2_TARGET
static int32_t sse2_dot_q8(const int8_t  * QMX_RESTRICT q8,
                           const int16_t * QMX_RESTRICT q16,
                           const int8_t  * QMX_RESTRICT k8,
                           uint32_t n) {
    (void)q8;
    __m128i zero = _mm_setzero_si128();
    __m128i acc = _mm_setzero_si128();
    uint32_t i = 0;
    for (; i + 15u < n; i += 16u) {
        __m128i kv = _mm_loadu_si128((const __m128i *)(k8 + i));
        __m128i ks = _mm_cmpgt_epi8(zero, kv);
        __m128i q_lo = _mm_loadu_si128((const __m128i *)(q16 + i));
        __m128i q_hi = _mm_loadu_si128((const __m128i *)(q16 + i + 8u));
        acc = _mm_add_epi32(acc, _mm_madd_epi16(_mm_unpacklo_epi8(kv, ks), q_lo));
        acc = _mm_add_epi32(acc, _mm_madd_epi16(_mm_unpackhi_epi8(kv, ks), q_hi));
    }
    int32_t sum = sse2_hsum_epi32(acc);
    for (; i < n; i++) sum += (int32_t)q16[i] * (int32_t)k8[i];
    return sum;
}

const QmxBackendOps qmx_ops_sse2 = {
    "SSE2",
    sse2_prepare_act,
    sse2_matmul_tiled,
    sse2_rmsnorm,
    sse2_dot_f32,
    sse2_saxpy_f32,
    sse2_dot_q8,
    NULL,
    qmx_swiglu_quant_scalar,
    qmx_f16_to_f32_scalar,
    qmx_f32_to_f16_scalar
};
#endif
