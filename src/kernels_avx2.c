/*
 * Copyright 2026 1WhiteFox
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/qmx_kernels.h"
#include <string.h>

#if defined(QMX_ENABLE_BACKEND_AVX2)
#if defined(__GNUC__) || defined(__clang__)
    #define QMX_AVX2_TARGET __attribute__((target("avx2,fma,f16c")))
#else
    #define QMX_AVX2_TARGET
#endif
#include <immintrin.h>

QMX_AVX2_TARGET
static void avx2_prepare_act(const float * QMX_RESTRICT x,
                             int8_t      * QMX_RESTRICT xq,
                             int16_t     * QMX_RESTRICT xq16,
                             float       * QMX_RESTRICT x_scales,
                             uint32_t cols) {
    (void)xq16;
    uint32_t full_groups = cols / QMX_TILE_COLS;
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const __m256i min_i8  = _mm256_set1_epi8(-127);

    for (uint32_t g = 0; g < full_groups; g++) {
        const float *xp = x + (size_t)g * QMX_TILE_COLS;
        __m256 v0 = _mm256_loadu_ps(xp + 0u);
        __m256 v1 = _mm256_loadu_ps(xp + 8u);
        __m256 v2 = _mm256_loadu_ps(xp + 16u);
        __m256 v3 = _mm256_loadu_ps(xp + 24u);

        __m256 m01 = _mm256_max_ps(_mm256_and_ps(v0, abs_mask), _mm256_and_ps(v1, abs_mask));
        __m256 m23 = _mm256_max_ps(_mm256_and_ps(v2, abs_mask), _mm256_and_ps(v3, abs_mask));
        __m256 m8  = _mm256_max_ps(m01, m23);
        __m128 m4  = _mm_max_ps(_mm256_castps256_ps128(m8), _mm256_extractf128_ps(m8, 1));
        m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        m4 = _mm_max_ss(m4, _mm_shuffle_ps(m4, m4, 1));
        float max_abs = _mm_cvtss_f32(m4);

        x_scales[g] = max_abs / 127.0f;
        float inv_scale = (max_abs == 0.0f) ? 0.0f : (127.0f / max_abs);
        __m256 v_inv = _mm256_set1_ps(inv_scale);

        __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(v0, v_inv));
        __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(v1, v_inv));
        __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(v2, v_inv));
        __m256i i3 = _mm256_cvtps_epi32(_mm256_mul_ps(v3, v_inv));

        __m256i p01 = _mm256_permute4x64_epi64(_mm256_packs_epi32(i0, i1), 0xD8);
        __m256i p23 = _mm256_permute4x64_epi64(_mm256_packs_epi32(i2, i3), 0xD8);
        __m256i b32 = _mm256_permute4x64_epi64(_mm256_packs_epi16(p01, p23), 0xD8);
        b32 = _mm256_max_epi8(b32, min_i8);
        _mm256_storeu_si256((__m256i *)(xq + (size_t)g * QMX_TILE_COLS), b32);
    }

    uint32_t tail_base = full_groups * QMX_TILE_COLS;
    if (tail_base < cols) {
        qmx_quantize_activations_q8(x + tail_base, xq + tail_base, x_scales + full_groups, cols - tail_base);
    }
}

QMX_AVX2_TARGET
static inline int32_t avx2_hsum_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    lo = _mm_add_epi32(lo, hi);
    __m128i hi64 = _mm_unpackhi_epi64(lo, lo);
    lo = _mm_add_epi32(lo, hi64);
    __m128i hi32 = _mm_shuffle_epi32(lo, _MM_SHUFFLE(1, 1, 1, 1));
    lo = _mm_add_epi32(lo, hi32);
    return _mm_cvtsi128_si32(lo);
}

QMX_AVX2_TARGET
static inline __m128i avx2_hsum4_epi32(__m256i r0, __m256i r1, __m256i r2, __m256i r3) {
    __m256i h01 = _mm256_hadd_epi32(r0, r1);
    __m256i h23 = _mm256_hadd_epi32(r2, r3);
    __m256i h0123 = _mm256_hadd_epi32(h01, h23);
    __m128i lo = _mm256_castsi256_si128(h0123);
    __m128i hi = _mm256_extracti128_si256(h0123, 1);
    return _mm_add_epi32(lo, hi);
}

QMX_AVX2_TARGET
static inline float avx2_hsum_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    lo = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, lo);
    lo = _mm_add_ss(lo, shuf);
    return _mm_cvtss_f32(lo);
}

QMX_AVX2_TARGET
static inline __m128 avx2_load_row_scales4(const uint16_t *sc16, __m128 vx_sc) {
    __m128i h4 = _mm_loadl_epi64((const __m128i *)sc16);
    return _mm_mul_ps(_mm_cvtph_ps(h4), vx_sc);
}

QMX_AVX2_TARGET
static inline __m128i avx2_hsum4_2x256(__m256i r01, __m256i r23) {
    __m256i h1 = _mm256_hadd_epi32(r01, r23);
    __m256i h2 = _mm256_hadd_epi32(h1, h1);
    return _mm_unpacklo_epi32(_mm256_castsi256_si128(h2), _mm256_extracti128_si256(h2, 1));
}

QMX_AVX2_TARGET
static inline void avx2_split_x32_eo(const int8_t *xp, __m128i shuf_eo,
                                     __m128i *out_even, __m128i *out_odd, int32_t *out_sum) {
    __m128i x0 = _mm_loadu_si128((const __m128i *)(xp + 0u));
    __m128i x1 = _mm_loadu_si128((const __m128i *)(xp + 16u));
    __m128i t0 = _mm_shuffle_epi8(x0, shuf_eo);
    __m128i t1 = _mm_shuffle_epi8(x1, shuf_eo);
    __m128i xe = _mm_unpacklo_epi64(t0, t1);
    __m128i xo = _mm_unpackhi_epi64(t0, t1);
    *out_even = xe;
    *out_odd  = xo;
    __m128i ones8 = _mm_set1_epi8(1);
    __m128i s16 = _mm_add_epi16(_mm_maddubs_epi16(ones8, xe), _mm_maddubs_epi16(ones8, xo));
    __m128i s32 = _mm_madd_epi16(s16, _mm_set1_epi16(1));
    s32 = _mm_hadd_epi32(s32, s32);
    s32 = _mm_hadd_epi32(s32, s32);
    *out_sum = _mm_cvtsi128_si32(s32);
}

QMX_AVX2_TARGET
static inline __m256i avx2_dot_q4_2rows(const uint8_t *tile_ptr, uint32_t lr,
                                        __m256i vx_even, __m256i vx_odd,
                                        __m256i ones16, __m256i mask0f) {
    __m256i raw32 = _mm256_loadu_si256((const __m256i *)(tile_ptr + lr * 16u));
    __m256i lo = _mm256_and_si256(raw32, mask0f);
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(raw32, 4), mask0f);
    __m256i p16 = _mm256_add_epi16(_mm256_maddubs_epi16(lo, vx_even),
                                   _mm256_maddubs_epi16(hi, vx_odd));
    return _mm256_madd_epi16(p16, ones16);
}

QMX_AVX2_TARGET
static inline __m256i avx2_dot_q8_row(const uint8_t *tile_ptr, uint32_t lr,
                                      __m256i vx, __m256i ones16) {
    __m256i vw = _mm256_loadu_si256((const __m256i *)(tile_ptr + lr * 32u));
    __m256i aw = _mm256_abs_epi8(vw);
    __m256i sx = _mm256_sign_epi8(vx, vw);
    __m256i p16 = _mm256_maddubs_epi16(aw, sx);
    return _mm256_madd_epi16(p16, ones16);
}

QMX_AVX2_TARGET
static inline __m256i avx2_dot_tern_row(const uint8_t *tile_ptr, uint32_t lr, __m256i vx, __m256i ones16,
                                        __m256i tern_shuf, __m256i tern_m0, __m256i tern_m1,
                                        __m256i tern_m2, __m256i tern_m3, __m256i tern_lut) {
    uint64_t raw64 = 0;
    memcpy(&raw64, tile_ptr + lr * 8u, sizeof(raw64));
    __m256i vb = _mm256_shuffle_epi8(_mm256_set1_epi64x((long long)raw64), tern_shuf);
    __m256i c0 = _mm256_and_si256(vb, tern_m0);
    __m256i c1 = _mm256_and_si256(_mm256_srli_epi16(vb, 2), tern_m1);
    __m256i c2 = _mm256_and_si256(_mm256_srli_epi16(vb, 4), tern_m2);
    __m256i c3 = _mm256_and_si256(_mm256_srli_epi16(vb, 6), tern_m3);
    __m256i codes = _mm256_or_si256(_mm256_or_si256(c0, c1), _mm256_or_si256(c2, c3));
    __m256i vw = _mm256_shuffle_epi8(tern_lut, codes);
    __m256i aw = _mm256_abs_epi8(vw);
    __m256i sx = _mm256_sign_epi8(vx, vw);
    __m256i p16 = _mm256_maddubs_epi16(aw, sx);
    return _mm256_madd_epi16(p16, ones16);
}

QMX_AVX2_TARGET
static inline __m256i avx2_dot_bin_row(const uint8_t *tile_ptr, uint32_t lr, __m256i vx, __m256i ones16,
                                       __m256i bin_shuf, __m256i bin_bits, __m256i two_u8, __m256i one_u8) {
    uint32_t raw32 = 0;
    memcpy(&raw32, tile_ptr + lr * 4u, sizeof(raw32));
    __m256i vb = _mm256_shuffle_epi8(_mm256_set1_epi32((int)raw32), bin_shuf);
    __m256i eq = _mm256_cmpeq_epi8(_mm256_and_si256(vb, bin_bits), bin_bits);
    __m256i vw = _mm256_sub_epi8(_mm256_and_si256(eq, two_u8), one_u8);
    __m256i sx = _mm256_sign_epi8(vx, vw);
    __m128i sx_lo = _mm256_castsi256_si128(sx);
    __m128i sx_hi = _mm256_extracti128_si256(sx, 1);
    __m256i s16 = _mm256_add_epi16(_mm256_cvtepi8_epi16(sx_lo), _mm256_cvtepi8_epi16(sx_hi));
    return _mm256_madd_epi16(s16, ones16);
}

QMX_AVX2_TARGET
static void avx2_matmul_q4(float * QMX_RESTRICT out, const int8_t * QMX_RESTRICT xq,
                           const float * QMX_RESTRICT x_scales, const QmxTensor *w,
                           int accumulate, int is_row_scaled) {
    uint32_t orig_rows = w->rows;
    uint32_t n_tr = (uint32_t)qmx_align_up(orig_rows, QMX_TILE_ROWS) / QMX_TILE_ROWS;
    uint32_t n_tc = (uint32_t)qmx_align_up(w->cols, QMX_TILE_COLS) / QMX_TILE_COLS;
    const uint8_t *base_data = (const uint8_t *)w->data;
    const uint16_t *scales_f16 = (const uint16_t *)(const void *)w->scales;
    __m256i ones16 = _mm256_set1_epi16(1);
    __m256i mask0f = _mm256_set1_epi8(0x0F);
    __m128i shuf_eo = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);

    __m128i pre_xe[512];
    __m128i pre_xo[512];
    int32_t pre_bias[512];
    uint32_t pre_cap = (n_tc <= 512u) ? n_tc : 512u;
    for (uint32_t tc = 0; tc < pre_cap; tc++) {
        int32_t xs = 0;
        avx2_split_x32_eo(xq + (size_t)tc * QMX_TILE_COLS, shuf_eo, &pre_xe[tc], &pre_xo[tc], &xs);
        pre_bias[tc] = xs << 3;
    }

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
            if (tc < pre_cap) {
                xe128 = pre_xe[tc];
                xo128 = pre_xo[tc];
                bias_i32 = pre_bias[tc];
            } else {
                int32_t xs = 0;
                avx2_split_x32_eo(xq + (size_t)tc * QMX_TILE_COLS, shuf_eo, &xe128, &xo128, &xs);
                bias_i32 = xs << 3;
            }
            __m256i vxe = _mm256_broadcastsi128_si256(xe128);
            __m256i vxo = _mm256_broadcastsi128_si256(xo128);
            __m128i vbias = _mm_set1_epi32(bias_i32);
            __m128 vx_sc = _mm_set1_ps(x_scales[tc]);

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
                __m128i s0 = _mm_sub_epi32(avx2_hsum4_2x256(avx2_dot_q4_2rows(tp0, 0u, vxe, vxo, ones16, mask0f),
                                                            avx2_dot_q4_2rows(tp0, 2u, vxe, vxo, ones16, mask0f)), vbias);
                __m128i s1 = _mm_sub_epi32(avx2_hsum4_2x256(avx2_dot_q4_2rows(tp0, 4u, vxe, vxo, ones16, mask0f),
                                                            avx2_dot_q4_2rows(tp0, 6u, vxe, vxo, ones16, mask0f)), vbias);
                __m128i s2 = _mm_sub_epi32(avx2_hsum4_2x256(avx2_dot_q4_2rows(tp0, 8u, vxe, vxo, ones16, mask0f),
                                                            avx2_dot_q4_2rows(tp0, 10u, vxe, vxo, ones16, mask0f)), vbias);
                __m128i s3 = _mm_sub_epi32(avx2_hsum4_2x256(avx2_dot_q4_2rows(tp0, 12u, vxe, vxo, ones16, mask0f),
                                                            avx2_dot_q4_2rows(tp0, 14u, vxe, vxo, ones16, mask0f)), vbias);
                va0 = _mm_fmadd_ps(_mm_cvtepi32_ps(s0), avx2_load_row_scales4(rs0 + 0u, vx_sc), va0);
                va1 = _mm_fmadd_ps(_mm_cvtepi32_ps(s1), avx2_load_row_scales4(rs0 + 4u, vx_sc), va1);
                va2 = _mm_fmadd_ps(_mm_cvtepi32_ps(s2), avx2_load_row_scales4(rs0 + 8u, vx_sc), va2);
                va3 = _mm_fmadd_ps(_mm_cvtepi32_ps(s3), avx2_load_row_scales4(rs0 + 12u, vx_sc), va3);

                __m128i t0 = _mm_sub_epi32(avx2_hsum4_2x256(avx2_dot_q4_2rows(tp1, 0u, vxe, vxo, ones16, mask0f),
                                                            avx2_dot_q4_2rows(tp1, 2u, vxe, vxo, ones16, mask0f)), vbias);
                __m128i t1 = _mm_sub_epi32(avx2_hsum4_2x256(avx2_dot_q4_2rows(tp1, 4u, vxe, vxo, ones16, mask0f),
                                                            avx2_dot_q4_2rows(tp1, 6u, vxe, vxo, ones16, mask0f)), vbias);
                __m128i t2 = _mm_sub_epi32(avx2_hsum4_2x256(avx2_dot_q4_2rows(tp1, 8u, vxe, vxo, ones16, mask0f),
                                                            avx2_dot_q4_2rows(tp1, 10u, vxe, vxo, ones16, mask0f)), vbias);
                __m128i t3 = _mm_sub_epi32(avx2_hsum4_2x256(avx2_dot_q4_2rows(tp1, 12u, vxe, vxo, ones16, mask0f),
                                                            avx2_dot_q4_2rows(tp1, 14u, vxe, vxo, ones16, mask0f)), vbias);
                vb0 = _mm_fmadd_ps(_mm_cvtepi32_ps(t0), avx2_load_row_scales4(rs1 + 0u, vx_sc), vb0);
                vb1 = _mm_fmadd_ps(_mm_cvtepi32_ps(t1), avx2_load_row_scales4(rs1 + 4u, vx_sc), vb1);
                vb2 = _mm_fmadd_ps(_mm_cvtepi32_ps(t2), avx2_load_row_scales4(rs1 + 8u, vx_sc), vb2);
                vb3 = _mm_fmadd_ps(_mm_cvtepi32_ps(t3), avx2_load_row_scales4(rs1 + 12u, vx_sc), vb3);
            } else {
                __m128 vs0 = _mm_mul_ps(_mm_set1_ps(w->scales[tid0]), vx_sc);
                __m128 vs1 = _mm_mul_ps(_mm_set1_ps(w->scales[tid1]), vx_sc);
                __m128i s0 = _mm_sub_epi32(avx2_hsum4_2x256(avx2_dot_q4_2rows(tp0, 0u, vxe, vxo, ones16, mask0f),
                                                            avx2_dot_q4_2rows(tp0, 2u, vxe, vxo, ones16, mask0f)), vbias);
                __m128i s1 = _mm_sub_epi32(avx2_hsum4_2x256(avx2_dot_q4_2rows(tp0, 4u, vxe, vxo, ones16, mask0f),
                                                            avx2_dot_q4_2rows(tp0, 6u, vxe, vxo, ones16, mask0f)), vbias);
                __m128i s2 = _mm_sub_epi32(avx2_hsum4_2x256(avx2_dot_q4_2rows(tp0, 8u, vxe, vxo, ones16, mask0f),
                                                            avx2_dot_q4_2rows(tp0, 10u, vxe, vxo, ones16, mask0f)), vbias);
                __m128i s3 = _mm_sub_epi32(avx2_hsum4_2x256(avx2_dot_q4_2rows(tp0, 12u, vxe, vxo, ones16, mask0f),
                                                            avx2_dot_q4_2rows(tp0, 14u, vxe, vxo, ones16, mask0f)), vbias);
                va0 = _mm_fmadd_ps(_mm_cvtepi32_ps(s0), vs0, va0);
                va1 = _mm_fmadd_ps(_mm_cvtepi32_ps(s1), vs0, va1);
                va2 = _mm_fmadd_ps(_mm_cvtepi32_ps(s2), vs0, va2);
                va3 = _mm_fmadd_ps(_mm_cvtepi32_ps(s3), vs0, va3);

                __m128i t0 = _mm_sub_epi32(avx2_hsum4_2x256(avx2_dot_q4_2rows(tp1, 0u, vxe, vxo, ones16, mask0f),
                                                            avx2_dot_q4_2rows(tp1, 2u, vxe, vxo, ones16, mask0f)), vbias);
                __m128i t1 = _mm_sub_epi32(avx2_hsum4_2x256(avx2_dot_q4_2rows(tp1, 4u, vxe, vxo, ones16, mask0f),
                                                            avx2_dot_q4_2rows(tp1, 6u, vxe, vxo, ones16, mask0f)), vbias);
                __m128i t2 = _mm_sub_epi32(avx2_hsum4_2x256(avx2_dot_q4_2rows(tp1, 8u, vxe, vxo, ones16, mask0f),
                                                            avx2_dot_q4_2rows(tp1, 10u, vxe, vxo, ones16, mask0f)), vbias);
                __m128i t3 = _mm_sub_epi32(avx2_hsum4_2x256(avx2_dot_q4_2rows(tp1, 12u, vxe, vxo, ones16, mask0f),
                                                            avx2_dot_q4_2rows(tp1, 14u, vxe, vxo, ones16, mask0f)), vbias);
                vb0 = _mm_fmadd_ps(_mm_cvtepi32_ps(t0), vs1, vb0);
                vb1 = _mm_fmadd_ps(_mm_cvtepi32_ps(t1), vs1, vb1);
                vb2 = _mm_fmadd_ps(_mm_cvtepi32_ps(t2), vs1, vb2);
                vb3 = _mm_fmadd_ps(_mm_cvtepi32_ps(t3), vs1, vb3);
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

    float tile_acc[QMX_TILE_ROWS] QMX_ALIGNED(32);
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
            __m128 vx_sc = _mm_set1_ps(x_scales[tc]);
            __m128 v_scale = is_row_scaled ? _mm_setzero_ps() : _mm_mul_ps(_mm_set1_ps(w->scales[tile_idx]), vx_sc);
            const uint16_t *row_sc = is_row_scaled ? (scales_f16 + (size_t)tile_idx * QMX_TILE_ROWS) : NULL;
            const uint8_t *tile_ptr = base_data + (size_t)tile_idx * 256u;

            __m128i xe128, xo128;
            int32_t bias_i32;
            if (tc < pre_cap) {
                xe128 = pre_xe[tc];
                xo128 = pre_xo[tc];
                bias_i32 = pre_bias[tc];
            } else {
                int32_t xs = 0;
                avx2_split_x32_eo(xq + (size_t)tc * QMX_TILE_COLS, shuf_eo, &xe128, &xo128, &xs);
                bias_i32 = xs << 3;
            }
            __m256i vxe = _mm256_broadcastsi128_si256(xe128);
            __m256i vxo = _mm256_broadcastsi128_si256(xo128);
            __m128i vbias = _mm_set1_epi32(bias_i32);

            if (rem_rows > 0u) {
                __m128i s0 = _mm_sub_epi32(avx2_hsum4_2x256(avx2_dot_q4_2rows(tile_ptr, 0u, vxe, vxo, ones16, mask0f),
                                                            avx2_dot_q4_2rows(tile_ptr, 2u, vxe, vxo, ones16, mask0f)), vbias);
                __m128 vs = is_row_scaled ? avx2_load_row_scales4(row_sc + 0u, vx_sc) : v_scale;
                vacc0 = _mm_fmadd_ps(_mm_cvtepi32_ps(s0), vs, vacc0);
            }
            if (rem_rows > 4u) {
                __m128i s1 = _mm_sub_epi32(avx2_hsum4_2x256(avx2_dot_q4_2rows(tile_ptr, 4u, vxe, vxo, ones16, mask0f),
                                                            avx2_dot_q4_2rows(tile_ptr, 6u, vxe, vxo, ones16, mask0f)), vbias);
                __m128 vs = is_row_scaled ? avx2_load_row_scales4(row_sc + 4u, vx_sc) : v_scale;
                vacc1 = _mm_fmadd_ps(_mm_cvtepi32_ps(s1), vs, vacc1);
            }
            if (rem_rows > 8u) {
                __m128i s2 = _mm_sub_epi32(avx2_hsum4_2x256(avx2_dot_q4_2rows(tile_ptr, 8u, vxe, vxo, ones16, mask0f),
                                                            avx2_dot_q4_2rows(tile_ptr, 10u, vxe, vxo, ones16, mask0f)), vbias);
                __m128 vs = is_row_scaled ? avx2_load_row_scales4(row_sc + 8u, vx_sc) : v_scale;
                vacc2 = _mm_fmadd_ps(_mm_cvtepi32_ps(s2), vs, vacc2);
            }
            if (rem_rows > 12u) {
                __m128i s3 = _mm_sub_epi32(avx2_hsum4_2x256(avx2_dot_q4_2rows(tile_ptr, 12u, vxe, vxo, ones16, mask0f),
                                                            avx2_dot_q4_2rows(tile_ptr, 14u, vxe, vxo, ones16, mask0f)), vbias);
                __m128 vs = is_row_scaled ? avx2_load_row_scales4(row_sc + 12u, vx_sc) : v_scale;
                vacc3 = _mm_fmadd_ps(_mm_cvtepi32_ps(s3), vs, vacc3);
            }
        }

        _mm_store_ps(tile_acc + 0u,  vacc0);
        _mm_store_ps(tile_acc + 4u,  vacc1);
        _mm_store_ps(tile_acc + 8u,  vacc2);
        _mm_store_ps(tile_acc + 12u, vacc3);
        for (uint32_t lr = 0; lr < rem_rows; lr++) out[r_base + lr] = tile_acc[lr];
    }
}

QMX_AVX2_TARGET
static void avx2_matmul_q8(float * QMX_RESTRICT out, const int8_t * QMX_RESTRICT xq,
                           const float * QMX_RESTRICT x_scales, const QmxTensor *w,
                           int accumulate, int is_row_scaled) {
    uint32_t orig_rows = w->rows;
    uint32_t n_tr = (uint32_t)qmx_align_up(orig_rows, QMX_TILE_ROWS) / QMX_TILE_ROWS;
    uint32_t n_tc = (uint32_t)qmx_align_up(w->cols, QMX_TILE_COLS) / QMX_TILE_COLS;
    const uint8_t *base_data = (const uint8_t *)w->data;
    const uint16_t *scales_f16 = (const uint16_t *)(const void *)w->scales;
    __m256i ones16 = _mm256_set1_epi16(1);
    float tile_acc[QMX_TILE_ROWS] QMX_ALIGNED(32);

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
            __m128 vx_sc = _mm_set1_ps(x_scales[tc]);
            __m128 v_scale = is_row_scaled ? _mm_setzero_ps() : _mm_mul_ps(_mm_set1_ps(w->scales[tile_idx]), vx_sc);
            const uint16_t *row_sc = is_row_scaled ? (scales_f16 + (size_t)tile_idx * QMX_TILE_ROWS) : NULL;
            const uint8_t *tile_ptr = base_data + (size_t)tile_idx * 512u;
            __m256i vx = _mm256_loadu_si256((const __m256i *)(xq + (size_t)tc * QMX_TILE_COLS));

            if (rem_rows > 0u) {
                __m256i r0 = avx2_dot_q8_row(tile_ptr, 0u, vx, ones16);
                __m256i r1 = avx2_dot_q8_row(tile_ptr, 1u, vx, ones16);
                __m256i r2 = avx2_dot_q8_row(tile_ptr, 2u, vx, ones16);
                __m256i r3 = avx2_dot_q8_row(tile_ptr, 3u, vx, ones16);
                __m128 vs = is_row_scaled ? avx2_load_row_scales4(row_sc + 0u, vx_sc) : v_scale;
                vacc0 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx2_hsum4_epi32(r0, r1, r2, r3)), vs, vacc0);
            }
            if (rem_rows > 4u) {
                __m256i r0 = avx2_dot_q8_row(tile_ptr, 4u, vx, ones16);
                __m256i r1 = avx2_dot_q8_row(tile_ptr, 5u, vx, ones16);
                __m256i r2 = avx2_dot_q8_row(tile_ptr, 6u, vx, ones16);
                __m256i r3 = avx2_dot_q8_row(tile_ptr, 7u, vx, ones16);
                __m128 vs = is_row_scaled ? avx2_load_row_scales4(row_sc + 4u, vx_sc) : v_scale;
                vacc1 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx2_hsum4_epi32(r0, r1, r2, r3)), vs, vacc1);
            }
            if (rem_rows > 8u) {
                __m256i r0 = avx2_dot_q8_row(tile_ptr, 8u,  vx, ones16);
                __m256i r1 = avx2_dot_q8_row(tile_ptr, 9u,  vx, ones16);
                __m256i r2 = avx2_dot_q8_row(tile_ptr, 10u, vx, ones16);
                __m256i r3 = avx2_dot_q8_row(tile_ptr, 11u, vx, ones16);
                __m128 vs = is_row_scaled ? avx2_load_row_scales4(row_sc + 8u, vx_sc) : v_scale;
                vacc2 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx2_hsum4_epi32(r0, r1, r2, r3)), vs, vacc2);
            }
            if (rem_rows > 12u) {
                __m256i r0 = avx2_dot_q8_row(tile_ptr, 12u, vx, ones16);
                __m256i r1 = avx2_dot_q8_row(tile_ptr, 13u, vx, ones16);
                __m256i r2 = avx2_dot_q8_row(tile_ptr, 14u, vx, ones16);
                __m256i r3 = avx2_dot_q8_row(tile_ptr, 15u, vx, ones16);
                __m128 vs = is_row_scaled ? avx2_load_row_scales4(row_sc + 12u, vx_sc) : v_scale;
                vacc3 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx2_hsum4_epi32(r0, r1, r2, r3)), vs, vacc3);
            }
        }

        _mm_store_ps(tile_acc + 0u,  vacc0);
        _mm_store_ps(tile_acc + 4u,  vacc1);
        _mm_store_ps(tile_acc + 8u,  vacc2);
        _mm_store_ps(tile_acc + 12u, vacc3);
        for (uint32_t lr = 0; lr < rem_rows; lr++) out[r_base + lr] = tile_acc[lr];
    }
}

QMX_AVX2_TARGET
static void avx2_matmul_lowbit(float * QMX_RESTRICT out, const int8_t * QMX_RESTRICT xq,
                               const float * QMX_RESTRICT x_scales, const QmxTensor *w,
                               int accumulate, uint8_t qtype) {
    uint32_t orig_rows = w->rows;
    uint32_t n_tr = (uint32_t)qmx_align_up(orig_rows, QMX_TILE_ROWS) / QMX_TILE_ROWS;
    uint32_t n_tc = (uint32_t)qmx_align_up(w->cols, QMX_TILE_COLS) / QMX_TILE_COLS;
    size_t tile_bytes = qmx_tile_data_bytes(qtype);
    const uint8_t *base_data = (const uint8_t *)w->data;
    __m256i ones16 = _mm256_set1_epi16(1);

    __m256i tern_shuf = _mm256_setr_epi8(
        0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
        4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7
    );
    __m256i tern_m0 = _mm256_set1_epi32(0x00000003);
    __m256i tern_m1 = _mm256_set1_epi32(0x00000300);
    __m256i tern_m2 = _mm256_set1_epi32(0x00030000);
    __m256i tern_m3 = _mm256_set1_epi32(0x03000000);
    __m256i tern_lut = _mm256_setr_epi8(
        0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, 0,
        0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, 0
    );
    __m256i bin_shuf = _mm256_setr_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3
    );
    __m256i bin_bits = _mm256_set1_epi64x((long long)0x8040201008040201ULL);
    __m256i two_u8 = _mm256_set1_epi8(2);
    __m256i one_u8 = _mm256_set1_epi8(1);
    float tile_acc[QMX_TILE_ROWS] QMX_ALIGNED(32);

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
            __m256i vx = _mm256_loadu_si256((const __m256i *)(xq + (size_t)tc * QMX_TILE_COLS));

            if (qtype == QMX_QUANT_TERNARY_158) {
                if (rem_rows > 0u) {
                    __m256i r0 = avx2_dot_tern_row(tile_ptr, 0u, vx, ones16, tern_shuf, tern_m0, tern_m1, tern_m2, tern_m3, tern_lut);
                    __m256i r1 = avx2_dot_tern_row(tile_ptr, 1u, vx, ones16, tern_shuf, tern_m0, tern_m1, tern_m2, tern_m3, tern_lut);
                    __m256i r2 = avx2_dot_tern_row(tile_ptr, 2u, vx, ones16, tern_shuf, tern_m0, tern_m1, tern_m2, tern_m3, tern_lut);
                    __m256i r3 = avx2_dot_tern_row(tile_ptr, 3u, vx, ones16, tern_shuf, tern_m0, tern_m1, tern_m2, tern_m3, tern_lut);
                    vacc0 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx2_hsum4_epi32(r0, r1, r2, r3)), v_scale, vacc0);
                }
                if (rem_rows > 4u) {
                    __m256i r0 = avx2_dot_tern_row(tile_ptr, 4u, vx, ones16, tern_shuf, tern_m0, tern_m1, tern_m2, tern_m3, tern_lut);
                    __m256i r1 = avx2_dot_tern_row(tile_ptr, 5u, vx, ones16, tern_shuf, tern_m0, tern_m1, tern_m2, tern_m3, tern_lut);
                    __m256i r2 = avx2_dot_tern_row(tile_ptr, 6u, vx, ones16, tern_shuf, tern_m0, tern_m1, tern_m2, tern_m3, tern_lut);
                    __m256i r3 = avx2_dot_tern_row(tile_ptr, 7u, vx, ones16, tern_shuf, tern_m0, tern_m1, tern_m2, tern_m3, tern_lut);
                    vacc1 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx2_hsum4_epi32(r0, r1, r2, r3)), v_scale, vacc1);
                }
                if (rem_rows > 8u) {
                    __m256i r0 = avx2_dot_tern_row(tile_ptr, 8u,  vx, ones16, tern_shuf, tern_m0, tern_m1, tern_m2, tern_m3, tern_lut);
                    __m256i r1 = avx2_dot_tern_row(tile_ptr, 9u,  vx, ones16, tern_shuf, tern_m0, tern_m1, tern_m2, tern_m3, tern_lut);
                    __m256i r2 = avx2_dot_tern_row(tile_ptr, 10u, vx, ones16, tern_shuf, tern_m0, tern_m1, tern_m2, tern_m3, tern_lut);
                    __m256i r3 = avx2_dot_tern_row(tile_ptr, 11u, vx, ones16, tern_shuf, tern_m0, tern_m1, tern_m2, tern_m3, tern_lut);
                    vacc2 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx2_hsum4_epi32(r0, r1, r2, r3)), v_scale, vacc2);
                }
                if (rem_rows > 12u) {
                    __m256i r0 = avx2_dot_tern_row(tile_ptr, 12u, vx, ones16, tern_shuf, tern_m0, tern_m1, tern_m2, tern_m3, tern_lut);
                    __m256i r1 = avx2_dot_tern_row(tile_ptr, 13u, vx, ones16, tern_shuf, tern_m0, tern_m1, tern_m2, tern_m3, tern_lut);
                    __m256i r2 = avx2_dot_tern_row(tile_ptr, 14u, vx, ones16, tern_shuf, tern_m0, tern_m1, tern_m2, tern_m3, tern_lut);
                    __m256i r3 = avx2_dot_tern_row(tile_ptr, 15u, vx, ones16, tern_shuf, tern_m0, tern_m1, tern_m2, tern_m3, tern_lut);
                    vacc3 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx2_hsum4_epi32(r0, r1, r2, r3)), v_scale, vacc3);
                }
            } else {
                if (rem_rows > 0u) {
                    __m256i r0 = avx2_dot_bin_row(tile_ptr, 0u, vx, ones16, bin_shuf, bin_bits, two_u8, one_u8);
                    __m256i r1 = avx2_dot_bin_row(tile_ptr, 1u, vx, ones16, bin_shuf, bin_bits, two_u8, one_u8);
                    __m256i r2 = avx2_dot_bin_row(tile_ptr, 2u, vx, ones16, bin_shuf, bin_bits, two_u8, one_u8);
                    __m256i r3 = avx2_dot_bin_row(tile_ptr, 3u, vx, ones16, bin_shuf, bin_bits, two_u8, one_u8);
                    vacc0 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx2_hsum4_epi32(r0, r1, r2, r3)), v_scale, vacc0);
                }
                if (rem_rows > 4u) {
                    __m256i r0 = avx2_dot_bin_row(tile_ptr, 4u, vx, ones16, bin_shuf, bin_bits, two_u8, one_u8);
                    __m256i r1 = avx2_dot_bin_row(tile_ptr, 5u, vx, ones16, bin_shuf, bin_bits, two_u8, one_u8);
                    __m256i r2 = avx2_dot_bin_row(tile_ptr, 6u, vx, ones16, bin_shuf, bin_bits, two_u8, one_u8);
                    __m256i r3 = avx2_dot_bin_row(tile_ptr, 7u, vx, ones16, bin_shuf, bin_bits, two_u8, one_u8);
                    vacc1 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx2_hsum4_epi32(r0, r1, r2, r3)), v_scale, vacc1);
                }
                if (rem_rows > 8u) {
                    __m256i r0 = avx2_dot_bin_row(tile_ptr, 8u,  vx, ones16, bin_shuf, bin_bits, two_u8, one_u8);
                    __m256i r1 = avx2_dot_bin_row(tile_ptr, 9u,  vx, ones16, bin_shuf, bin_bits, two_u8, one_u8);
                    __m256i r2 = avx2_dot_bin_row(tile_ptr, 10u, vx, ones16, bin_shuf, bin_bits, two_u8, one_u8);
                    __m256i r3 = avx2_dot_bin_row(tile_ptr, 11u, vx, ones16, bin_shuf, bin_bits, two_u8, one_u8);
                    vacc2 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx2_hsum4_epi32(r0, r1, r2, r3)), v_scale, vacc2);
                }
                if (rem_rows > 12u) {
                    __m256i r0 = avx2_dot_bin_row(tile_ptr, 12u, vx, ones16, bin_shuf, bin_bits, two_u8, one_u8);
                    __m256i r1 = avx2_dot_bin_row(tile_ptr, 13u, vx, ones16, bin_shuf, bin_bits, two_u8, one_u8);
                    __m256i r2 = avx2_dot_bin_row(tile_ptr, 14u, vx, ones16, bin_shuf, bin_bits, two_u8, one_u8);
                    __m256i r3 = avx2_dot_bin_row(tile_ptr, 15u, vx, ones16, bin_shuf, bin_bits, two_u8, one_u8);
                    vacc3 = _mm_fmadd_ps(_mm_cvtepi32_ps(avx2_hsum4_epi32(r0, r1, r2, r3)), v_scale, vacc3);
                }
            }
        }

        _mm_store_ps(tile_acc + 0u,  vacc0);
        _mm_store_ps(tile_acc + 4u,  vacc1);
        _mm_store_ps(tile_acc + 8u,  vacc2);
        _mm_store_ps(tile_acc + 12u, vacc3);
        for (uint32_t lr = 0; lr < rem_rows; lr++) out[r_base + lr] = tile_acc[lr];
    }
}

QMX_AVX2_TARGET
static void avx2_matmul_tiled(float         * QMX_RESTRICT out,
                              const int8_t  * QMX_RESTRICT xq,
                              const int16_t * QMX_RESTRICT xq16,
                              const float   * QMX_RESTRICT x_scales,
                              const QmxTensor *w,
                              int accumulate) {
    (void)xq16;
    uint8_t qt = w->quant_type;
    if (qt == QMX_QUANT_Q4_SYM) {
        avx2_matmul_q4(out, xq, x_scales, w, accumulate, 0);
    } else if (qt == QMX_QUANT_Q4_ROW32) {
        avx2_matmul_q4(out, xq, x_scales, w, accumulate, 1);
    } else if (qt == QMX_QUANT_Q8_SYM) {
        avx2_matmul_q8(out, xq, x_scales, w, accumulate, 0);
    } else if (qt == QMX_QUANT_Q8_ROW32) {
        avx2_matmul_q8(out, xq, x_scales, w, accumulate, 1);
    } else {
        avx2_matmul_lowbit(out, xq, x_scales, w, accumulate, qt);
    }
}

QMX_AVX2_TARGET
static void avx2_rmsnorm(float       * QMX_RESTRICT out,
                         const float * QMX_RESTRICT x,
                         const float * QMX_RESTRICT weight,
                         uint32_t size,
                         float eps) {
    __m256 sum8 = _mm256_setzero_ps();
    uint32_t i = 0;
    for (; i + 7u < size; i += 8u) {
        __m256 v = _mm256_loadu_ps(x + i);
        sum8 = _mm256_fmadd_ps(v, v, sum8);
    }
    float ss = avx2_hsum_ps(sum8);
    for (; i < size; i++) ss += x[i] * x[i];
    float inv_rms = 1.0f / qmx_sqrtf(ss / (float)size + eps);
    __m256 v_inv = _mm256_set1_ps(inv_rms);
    i = 0;
    for (; i + 7u < size; i += 8u) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vw = _mm256_loadu_ps(weight + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(vx, v_inv), vw));
    }
    for (; i < size; i++) out[i] = weight[i] * (inv_rms * x[i]);
}

QMX_AVX2_TARGET
static float avx2_dot_f32(const float * QMX_RESTRICT a,
                          const float * QMX_RESTRICT b,
                          uint32_t n) {
    __m256 sum8 = _mm256_setzero_ps();
    uint32_t i = 0;
    for (; i + 7u < n; i += 8u) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum8 = _mm256_fmadd_ps(va, vb, sum8);
    }
    float sum = avx2_hsum_ps(sum8);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

QMX_AVX2_TARGET
static void avx2_saxpy_f32(float       * QMX_RESTRICT y,
                           float a,
                           const float * QMX_RESTRICT x,
                           uint32_t n) {
    __m256 va = _mm256_set1_ps(a);
    uint32_t i = 0;
    for (; i + 7u < n; i += 8u) {
        __m256 vy = _mm256_loadu_ps(y + i);
        __m256 vx = _mm256_loadu_ps(x + i);
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, vx, vy));
    }
    for (; i < n; i++) y[i] += a * x[i];
}

QMX_AVX2_TARGET
static int32_t avx2_dot_q8(const int8_t  * QMX_RESTRICT q8,
                           const int16_t * QMX_RESTRICT q16,
                           const int8_t  * QMX_RESTRICT k8,
                           uint32_t n) {
    (void)q16;
    __m256i ones16 = _mm256_set1_epi16(1);
    __m256i acc = _mm256_setzero_si256();
    uint32_t i = 0;
    for (; i + 31u < n; i += 32u) {
        __m256i vq = _mm256_loadu_si256((const __m256i *)(q8 + i));
        __m256i vk = _mm256_loadu_si256((const __m256i *)(k8 + i));
        __m256i aq = _mm256_abs_epi8(vq);
        __m256i sk = _mm256_sign_epi8(vk, vq);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(aq, sk), ones16));
    }
    int32_t sum = avx2_hsum_epi32(acc);
    for (; i < n; i++) sum += (int32_t)q8[i] * (int32_t)k8[i];
    return sum;
}

QMX_AVX2_TARGET
static inline void avx2_quant_group32(__m256 v0, __m256 v1, __m256 v2, __m256 v3,
                                      int8_t * QMX_RESTRICT dst, float * QMX_RESTRICT scale_out) {
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const __m256i min_i8  = _mm256_set1_epi8(-127);

    __m256 m01 = _mm256_max_ps(_mm256_and_ps(v0, abs_mask), _mm256_and_ps(v1, abs_mask));
    __m256 m23 = _mm256_max_ps(_mm256_and_ps(v2, abs_mask), _mm256_and_ps(v3, abs_mask));
    __m256 m8  = _mm256_max_ps(m01, m23);
    __m128 m4  = _mm_max_ps(_mm256_castps256_ps128(m8), _mm256_extractf128_ps(m8, 1));
    m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    m4 = _mm_max_ss(m4, _mm_shuffle_ps(m4, m4, 1));
    float max_abs = _mm_cvtss_f32(m4);

    *scale_out = max_abs / 127.0f;
    float inv_scale = (max_abs == 0.0f) ? 0.0f : (127.0f / max_abs);
    __m256 v_inv = _mm256_set1_ps(inv_scale);

    __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(v0, v_inv));
    __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(v1, v_inv));
    __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(v2, v_inv));
    __m256i i3 = _mm256_cvtps_epi32(_mm256_mul_ps(v3, v_inv));

    __m256i p01 = _mm256_permute4x64_epi64(_mm256_packs_epi32(i0, i1), 0xD8);
    __m256i p23 = _mm256_permute4x64_epi64(_mm256_packs_epi32(i2, i3), 0xD8);
    __m256i b32 = _mm256_permute4x64_epi64(_mm256_packs_epi16(p01, p23), 0xD8);
    b32 = _mm256_max_epi8(b32, min_i8);
    _mm256_storeu_si256((__m256i *)dst, b32);
}

QMX_AVX2_TARGET
static inline __m256 avx2_expf_ps(__m256 x) {
    const __m256 hi = _mm256_set1_ps(88.0f);
    const __m256 lo = _mm256_set1_ps(-87.3f);
    __m256 under = _mm256_cmp_ps(x, lo, _CMP_LT_OQ);
    x = _mm256_min_ps(_mm256_max_ps(x, lo), hi);
    __m256 kf = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.4426950408889634f)),
                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256i ki = _mm256_cvtps_epi32(kf);
    __m256 r = _mm256_fnmadd_ps(kf, _mm256_set1_ps(0.693145751953125f), x);
    r = _mm256_fnmadd_ps(kf, _mm256_set1_ps(1.428606765330187e-6f), r);
    __m256 p = _mm256_set1_ps(1.38888889e-3f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.33333333e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.16666667e-2f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.66666667e-1f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.0e-1f));
    p = _mm256_fmadd_ps(p, _mm256_mul_ps(r, r), _mm256_add_ps(r, _mm256_set1_ps(1.0f)));
    __m256 res = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(p), _mm256_slli_epi32(ki, 23)));
    return _mm256_andnot_ps(under, res);
}

QMX_AVX2_TARGET
static void avx2_rmsnorm_quant(const float * QMX_RESTRICT x,
                               const float * QMX_RESTRICT weight,
                               int8_t      * QMX_RESTRICT xq,
                               int16_t     * QMX_RESTRICT xq16,
                               float       * QMX_RESTRICT x_scales,
                               uint32_t size,
                               float eps) {
    (void)xq16;
    __m256 sum8 = _mm256_setzero_ps();
    uint32_t i = 0;
    for (; i + 7u < size; i += 8u) {
        __m256 v = _mm256_loadu_ps(x + i);
        sum8 = _mm256_fmadd_ps(v, v, sum8);
    }
    float ss = avx2_hsum_ps(sum8);
    for (; i < size; i++) ss += x[i] * x[i];
    float inv_rms = 1.0f / qmx_sqrtf(ss / (float)size + eps);
    __m256 v_inv = _mm256_set1_ps(inv_rms);

    uint32_t full_groups = size / QMX_TILE_COLS;
    for (uint32_t g = 0; g < full_groups; g++) {
        uint32_t base = g * QMX_TILE_COLS;
        __m256 v0 = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + base + 0u),  v_inv), _mm256_loadu_ps(weight + base + 0u));
        __m256 v1 = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + base + 8u),  v_inv), _mm256_loadu_ps(weight + base + 8u));
        __m256 v2 = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + base + 16u), v_inv), _mm256_loadu_ps(weight + base + 16u));
        __m256 v3 = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + base + 24u), v_inv), _mm256_loadu_ps(weight + base + 24u));
        avx2_quant_group32(v0, v1, v2, v3, xq + base, &x_scales[g]);
    }

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

QMX_AVX2_TARGET
static void avx2_swiglu_quant(const float * QMX_RESTRICT gate,
                              const float * QMX_RESTRICT up,
                              int8_t      * QMX_RESTRICT xq,
                              int16_t     * QMX_RESTRICT xq16,
                              float       * QMX_RESTRICT x_scales,
                              uint32_t size) {
    (void)xq16;
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 neg = _mm256_set1_ps(-0.0f);
    uint32_t full_groups = size / QMX_TILE_COLS;

    for (uint32_t g = 0; g < full_groups; g++) {
        uint32_t base = g * QMX_TILE_COLS;
        __m256 g0 = _mm256_loadu_ps(gate + base + 0u);
        __m256 g1 = _mm256_loadu_ps(gate + base + 8u);
        __m256 g2 = _mm256_loadu_ps(gate + base + 16u);
        __m256 g3 = _mm256_loadu_ps(gate + base + 24u);

        const __m256 two = _mm256_set1_ps(2.0f);
        __m256 d0 = _mm256_add_ps(one, avx2_expf_ps(_mm256_xor_ps(g0, neg)));
        __m256 d1 = _mm256_add_ps(one, avx2_expf_ps(_mm256_xor_ps(g1, neg)));
        __m256 d2 = _mm256_add_ps(one, avx2_expf_ps(_mm256_xor_ps(g2, neg)));
        __m256 d3 = _mm256_add_ps(one, avx2_expf_ps(_mm256_xor_ps(g3, neg)));
        __m256 r0 = _mm256_rcp_ps(d0); r0 = _mm256_mul_ps(r0, _mm256_fnmadd_ps(d0, r0, two));
        __m256 r1 = _mm256_rcp_ps(d1); r1 = _mm256_mul_ps(r1, _mm256_fnmadd_ps(d1, r1, two));
        __m256 r2 = _mm256_rcp_ps(d2); r2 = _mm256_mul_ps(r2, _mm256_fnmadd_ps(d2, r2, two));
        __m256 r3 = _mm256_rcp_ps(d3); r3 = _mm256_mul_ps(r3, _mm256_fnmadd_ps(d3, r3, two));
        __m256 s0 = _mm256_mul_ps(g0, r0);
        __m256 s1 = _mm256_mul_ps(g1, r1);
        __m256 s2 = _mm256_mul_ps(g2, r2);
        __m256 s3 = _mm256_mul_ps(g3, r3);

        __m256 v0 = _mm256_mul_ps(s0, _mm256_loadu_ps(up + base + 0u));
        __m256 v1 = _mm256_mul_ps(s1, _mm256_loadu_ps(up + base + 8u));
        __m256 v2 = _mm256_mul_ps(s2, _mm256_loadu_ps(up + base + 16u));
        __m256 v3 = _mm256_mul_ps(s3, _mm256_loadu_ps(up + base + 24u));
        avx2_quant_group32(v0, v1, v2, v3, xq + base, &x_scales[g]);
    }

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

QMX_AVX2_TARGET
static void avx2_f16_to_f32(float * QMX_RESTRICT dst, const uint16_t * QMX_RESTRICT src, uint32_t n) {
    uint32_t i = 0;
    for (; i + 7u < n; i += 8u) {
        __m128i h8 = _mm_loadu_si128((const __m128i *)(src + i));
        _mm256_storeu_ps(dst + i, _mm256_cvtph_ps(h8));
    }
    if (i + 3u < n) {
        __m128i h4 = _mm_loadl_epi64((const __m128i *)(src + i));
        _mm_storeu_ps(dst + i, _mm_cvtph_ps(h4));
        i += 4u;
    }
    for (; i < n; i++) dst[i] = qmx_fp16_to_fp32(src[i]);
}

QMX_AVX2_TARGET
static void avx2_f32_to_f16(uint16_t * QMX_RESTRICT dst, const float * QMX_RESTRICT src, uint32_t n) {
    uint32_t i = 0;
    for (; i + 7u < n; i += 8u) {
        __m256 v8 = _mm256_loadu_ps(src + i);
        _mm_storeu_si128((__m128i *)(dst + i), _mm256_cvtps_ph(v8, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }
    if (i + 3u < n) {
        __m128 v4 = _mm_loadu_ps(src + i);
        _mm_storel_epi64((__m128i *)(dst + i), _mm_cvtps_ph(v4, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        i += 4u;
    }
    for (; i < n; i++) dst[i] = qmx_fp32_to_fp16(src[i]);
}

const QmxBackendOps qmx_ops_avx2 = {
    "AVX2",
    avx2_prepare_act,
    avx2_matmul_tiled,
    avx2_rmsnorm,
    avx2_dot_f32,
    avx2_saxpy_f32,
    avx2_dot_q8,
    avx2_rmsnorm_quant,
    avx2_swiglu_quant,
    avx2_f16_to_f32,
    avx2_f32_to_f16
};
#endif
