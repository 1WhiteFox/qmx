/*
 * Copyright 2026 1WhiteFox
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/qmx_kernels.h"

#if defined(QMX_ENABLE_BACKEND_3DNOW) && (defined(__i386__) || defined(__x86_64__)) && (defined(__3dNOW__) || defined(__3dnow__))


static const uint64_t g_3dn_q4_mask_0f   QMX_ALIGNED(8) = 0x0F0F0F0F0F0F0F0FULL;
static const uint64_t g_3dn_q4_bias_08   QMX_ALIGNED(8) = 0x0808080808080808ULL;


#define QMX_FEMMS() \
    __asm__ volatile ("femms" : : : "st", "st(1)", "st(2)", "st(3)", "st(4)", "st(5)", "st(6)", "st(7)", "memory")

static void threednow_prepare_act(const float * QMX_RESTRICT x,
                                  int8_t      * QMX_RESTRICT xq,
                                  int16_t     * QMX_RESTRICT xq16,
                                  float       * QMX_RESTRICT x_scales,
                                  uint32_t cols) {
    qmx_quantize_activations_q8(x, xq, x_scales, cols);
    uint32_t pad_cols = (uint32_t)qmx_align_up(cols, QMX_TILE_COLS);
    for (uint32_t i = 0; i < pad_cols; i += 8u) {
        const int8_t *src = xq + i;
        int16_t *dst = xq16 + i;
        __asm__ volatile (
            "movq (%[src]), %%mm2\n\t"
            "movq %%mm2, %%mm0\n\t"
            "movq %%mm2, %%mm1\n\t"
            "punpcklbw %%mm0, %%mm0\n\t"
            "punpckhbw %%mm1, %%mm1\n\t"
            "psraw $8, %%mm0\n\t"
            "psraw $8, %%mm1\n\t"
            "movq %%mm0, (%[dst])\n\t"
            "movq %%mm1, 8(%[dst])\n\t"
            : : [src] "r"(src), [dst] "r"(dst) : "mm0", "mm1", "mm2", "memory"
        );
    }
    QMX_FEMMS();
}

static void threednow_rmsnorm(float       * QMX_RESTRICT out,
                              const float * QMX_RESTRICT x,
                              const float * QMX_RESTRICT weight,
                              uint32_t size,
                              float eps) {
    float ss = 0.0f;
    uint32_t j = 0;
    uint64_t acc_f2 = 0;
    for (; j + 3u < size; j += 4u) {
        const float *xp = x + j;
        __asm__ volatile (
            "movq (%[xp]), %%mm1\n\t"
            "movq 8(%[xp]), %%mm2\n\t"
            "movq %[acc], %%mm0\n\t"
            "pfmul %%mm1, %%mm1\n\t"
            "pfmul %%mm2, %%mm2\n\t"
            "pfadd %%mm1, %%mm0\n\t"
            "pfadd %%mm2, %%mm0\n\t"
            "movq %%mm0, %[acc]\n\t"
            : [acc] "+m"(acc_f2) : [xp] "r"(xp) : "mm0", "mm1", "mm2", "memory"
        );
    }
    __asm__ volatile (
        "movq %[acc], %%mm0\n\t"
        "pfacc %%mm0, %%mm0\n\t"
        "movd %%mm0, %[ss]\n\t"
        "femms\n\t"
        : [ss] "=r"(ss) : [acc] "m"(acc_f2)
        : "mm0", "st", "st(1)", "st(2)", "st(3)", "st(4)", "st(5)", "st(6)", "st(7)", "memory"
    );
    for (; j < size; j++) ss += x[j] * x[j];
    ss = ss / (float)size + eps;

    float inv_sqrt = 0.0f;
    __asm__ volatile (
        "movd %[ss], %%mm0\n\t"
        "punpckldq %%mm0, %%mm0\n\t"
        "pfrsqrt %%mm0, %%mm1\n\t"
        "movq %%mm1, %%mm2\n\t"
        "pfmul %%mm1, %%mm1\n\t"
        "pfrsqit1 %%mm0, %%mm1\n\t"
        "pfrcpit2 %%mm2, %%mm1\n\t"
        "movd %%mm1, %[inv]\n\t"
        : [inv] "=m"(inv_sqrt) : [ss] "m"(ss) : "mm0", "mm1", "mm2", "memory"
    );
    j = 0;
    for (; j + 1u < size; j += 2u) {
        const float *xp = x + j;
        const float *wp = weight + j;
        float *op = out + j;
        __asm__ volatile (
            "movd %[inv], %%mm1\n\t"
            "punpckldq %%mm1, %%mm1\n\t"
            "movq (%[xp]), %%mm3\n\t"
            "movq (%[wp]), %%mm4\n\t"
            "pfmul %%mm1, %%mm3\n\t"
            "pfmul %%mm4, %%mm3\n\t"
            "movq %%mm3, (%[op])\n\t"
            : : [xp] "r"(xp), [wp] "r"(wp), [op] "r"(op), [inv] "m"(inv_sqrt)
            : "mm1", "mm3", "mm4", "memory"
        );
    }
    QMX_FEMMS();
    for (; j < size; j++) out[j] = weight[j] * (inv_sqrt * x[j]);
}

static float threednow_dot_f32(const float * QMX_RESTRICT a,
                               const float * QMX_RESTRICT b,
                               uint32_t n) {
    float sum = 0.0f;
    uint32_t i = 0;
    uint64_t acc_f2 = 0;
    for (; i + 3u < n; i += 4u) {
        const float *ap = a + i;
        const float *bp = b + i;
        __asm__ volatile (
            "movq (%[ap]), %%mm1\n\t"
            "movq 8(%[ap]), %%mm2\n\t"
            "movq %[acc], %%mm0\n\t"
            "pfmul (%[bp]), %%mm1\n\t"
            "pfmul 8(%[bp]), %%mm2\n\t"
            "pfadd %%mm1, %%mm0\n\t"
            "pfadd %%mm2, %%mm0\n\t"
            "movq %%mm0, %[acc]\n\t"
            : [acc] "+m"(acc_f2) : [ap] "r"(ap), [bp] "r"(bp) : "mm0", "mm1", "mm2", "memory"
        );
    }
    __asm__ volatile (
        "movq %[acc], %%mm0\n\tpfacc %%mm0, %%mm0\n\tmovd %%mm0, %[sum]\n\tfemms\n\t"
        : [sum] "=r"(sum) : [acc] "m"(acc_f2)
        : "mm0", "st", "st(1)", "st(2)", "st(3)", "st(4)", "st(5)", "st(6)", "st(7)", "memory"
    );
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

static void threednow_saxpy_f32(float       * QMX_RESTRICT y,
                                float a,
                                const float * QMX_RESTRICT x,
                                uint32_t n) {
    uint32_t i = 0;
    for (; i + 3u < n; i += 4u) {
        float *yp = y + i;
        const float *xp = x + i;
        __asm__ volatile (
            "movd %[a], %%mm0\n\t"
            "punpckldq %%mm0, %%mm0\n\t"
            "movq (%[xp]), %%mm1\n\t"
            "movq 8(%[xp]), %%mm2\n\t"
            "movq (%[yp]), %%mm3\n\t"
            "movq 8(%[yp]), %%mm4\n\t"
            "pfmul %%mm0, %%mm1\n\t"
            "pfmul %%mm0, %%mm2\n\t"
            "pfadd %%mm1, %%mm3\n\t"
            "pfadd %%mm2, %%mm4\n\t"
            "movq %%mm3, (%[yp])\n\t"
            "movq %%mm4, 8(%[yp])\n\t"
            : : [xp] "r"(xp), [yp] "r"(yp), [a] "m"(a)
            : "mm0", "mm1", "mm2", "mm3", "mm4", "memory"
        );
    }
    QMX_FEMMS();
    for (; i < n; i++) y[i] += a * x[i];
}

static int32_t threednow_dot_q8(const int8_t  * QMX_RESTRICT q8,
                                const int16_t * QMX_RESTRICT q16,
                                const int8_t  * QMX_RESTRICT k8,
                                uint32_t n) {
    (void)q8;
    int32_t sum = 0;
    uint32_t i = 0;
    for (; i + 7u < n; i += 8u) {
        const int16_t *qp = q16 + i;
        const int8_t  *kp = k8 + i;
        int32_t blk;
        __asm__ volatile (
            "movq (%[k]), %%mm2\n\t"
            "movq %%mm2, %%mm3\n\t"
            "punpcklbw %%mm2, %%mm2\n\t"
            "punpckhbw %%mm3, %%mm3\n\t"
            "psraw $8, %%mm2\n\t"
            "psraw $8, %%mm3\n\t"
            "pmaddwd (%[q]), %%mm2\n\t"
            "pmaddwd 8(%[q]), %%mm3\n\t"
            "paddd %%mm3, %%mm2\n\t"
            "movq %%mm2, %%mm0\n\t"
            "psrlq $32, %%mm0\n\t"
            "paddd %%mm0, %%mm2\n\t"
            "movd %%mm2, %[blk]\n\t"
            : [blk] "=r"(blk)
            : [q] "r"(qp), [k] "r"(kp)
            : "mm0", "mm2", "mm3"
        );
        sum += blk;
    }
    QMX_FEMMS();
    for (; i < n; i++) sum += (int32_t)q16[i] * (int32_t)k8[i];
    return sum;
}

QMX_INLINE void threednow_lut_step_2rows(const int16_t *x_ptr,
                                         const int16_t *l0a, const int16_t *l0b,
                                         const int16_t *l1a, const int16_t *l1b,
                                         uint64_t *acc0, uint64_t *acc1) {
    __asm__ volatile (
        "movq (%[x]), %%mm0\n\t"
        "movq 8(%[x]), %%mm1\n\t"
        "movq (%[l0a]), %%mm2\n\t"
        "movq (%[l0b]), %%mm3\n\t"
        "pmaddwd %%mm0, %%mm2\n\t"
        "pmaddwd %%mm1, %%mm3\n\t"
        "movq (%[l1a]), %%mm4\n\t"
        "movq (%[l1b]), %%mm5\n\t"
        "pmaddwd %%mm0, %%mm4\n\t"
        "pmaddwd %%mm1, %%mm5\n\t"
        "paddd %%mm3, %%mm2\n\t"
        "paddd %%mm5, %%mm4\n\t"
        "paddd %[a0], %%mm2\n\t"
        "paddd %[a1], %%mm4\n\t"
        "movq %%mm2, %[a0]\n\t"
        "movq %%mm4, %[a1]\n\t"
        : [a0] "+m"(*acc0), [a1] "+m"(*acc1)
        : [x] "r"(x_ptr), [l0a] "r"(l0a), [l0b] "r"(l0b), [l1a] "r"(l1a), [l1b] "r"(l1b)
        : "mm0", "mm1", "mm2", "mm3", "mm4", "mm5", "memory"
    );
}

static inline uint32_t threednow_fp16_to_f32_bits(uint16_t h) {
    uint32_t sign = ((uint32_t)(h & 0x8000u)) << 16u;
    uint32_t exp  = (uint32_t)((h >> 10u) & 0x1Fu);
    uint32_t mant = (uint32_t)(h & 0x03FFu);
    if (exp == 0u) {
        if (mant == 0u) return sign;
        exp = 1u;
        while ((mant & 0x0400u) == 0u) {
            mant <<= 1u;
            exp--;
        }
        mant &= 0x03FFu;
        return sign | ((exp + 112u) << 23u) | (mant << 13u);
    }
    if (exp == 31u) {
        return sign | 0x7F800000u | (mant << 13u);
    }
    return sign | ((exp + 112u) << 23u) | (mant << 13u);
}

QMX_INLINE void threednow_scale_accum_2rows(float *acc_pair_ptr,
                                            const uint64_t *acc0,
                                            const uint64_t *acc1,
                                            const float *w_scale_ptr,
                                            const float *x_scale_ptr) {
    __asm__ volatile (
        "movd (%[ws]), %%mm4\n\t"
        "movd (%[xs]), %%mm3\n\t"
        "pfmul %%mm3, %%mm4\n\t"
        "punpckldq %%mm4, %%mm4\n\t"
        "movq %[a0], %%mm5\n\t"
        "movq %[a1], %%mm6\n\t"
        "movq %%mm5, %%mm0\n\tpsrlq $32, %%mm0\n\tpaddd %%mm0, %%mm5\n\t"
        "movq %%mm6, %%mm1\n\tpsrlq $32, %%mm1\n\tpaddd %%mm1, %%mm6\n\t"
        "punpckldq %%mm6, %%mm5\n\t"
        "pi2fd %%mm5, %%mm5\n\t"
        "pfmul %%mm4, %%mm5\n\t"
        "pfadd (%[ap]), %%mm5\n\t"
        "movq %%mm5, (%[ap])\n\t"
        : : [ap] "r"(acc_pair_ptr), [a0] "m"(*acc0), [a1] "m"(*acc1),
            [ws] "r"(w_scale_ptr), [xs] "r"(x_scale_ptr)
        : "mm0", "mm1", "mm3", "mm4", "mm5", "mm6", "memory"
    );
}

QMX_INLINE void threednow_scale_accum_2rows_pair(float *acc_pair_ptr,
                                                 const uint64_t *acc0,
                                                 const uint64_t *acc1,
                                                 const uint32_t *w_scale_pair,
                                                 const float *x_scale_ptr) {
    __asm__ volatile (
        "movq (%[ws]), %%mm4\n\t"
        "movd (%[xs]), %%mm3\n\t"
        "punpckldq %%mm3, %%mm3\n\t"
        "pfmul %%mm3, %%mm4\n\t"
        "movq %[a0], %%mm5\n\t"
        "movq %[a1], %%mm6\n\t"
        "movq %%mm5, %%mm0\n\tpsrlq $32, %%mm0\n\tpaddd %%mm0, %%mm5\n\t"
        "movq %%mm6, %%mm1\n\tpsrlq $32, %%mm1\n\tpaddd %%mm1, %%mm6\n\t"
        "punpckldq %%mm6, %%mm5\n\t"
        "pi2fd %%mm5, %%mm5\n\t"
        "pfmul %%mm4, %%mm5\n\t"
        "pfadd (%[ap]), %%mm5\n\t"
        "movq %%mm5, (%[ap])\n\t"
        : : [ap] "r"(acc_pair_ptr), [a0] "m"(*acc0), [a1] "m"(*acc1),
            [ws] "r"(w_scale_pair), [xs] "r"(x_scale_ptr)
        : "mm0", "mm1", "mm3", "mm4", "mm5", "mm6", "memory"
    );
}

static void threednow_accum_tile(float *tile_acc16,
                                 const uint8_t *tile_ptr,
                                 const int16_t *xq16_32,
                                 uint8_t base_qtype,
                                 const float *w_scale_ptr,
                                 const uint32_t *row_scales_u32,
                                 const float *x_scale_ptr) {
    if (base_qtype == QMX_QUANT_BINARY_1BIT) {
        for (uint32_t lr = 0; lr < 16u; lr += 2u) {
            const uint8_t *r0 = tile_ptr + lr * 4u;
            const uint8_t *r1 = r0 + 4u;
            uint64_t acc0 = 0, acc1 = 0;
            for (uint32_t b = 0; b < 4u; b++) {
                uint8_t p0 = r0[b];
                uint8_t p1 = r1[b];
                threednow_lut_step_2rows(xq16_32 + b * 8u,
                                         qmx_lut_binary_i16_4[p0 & 0x0Fu], qmx_lut_binary_i16_4[p0 >> 4u],
                                         qmx_lut_binary_i16_4[p1 & 0x0Fu], qmx_lut_binary_i16_4[p1 >> 4u],
                                         &acc0, &acc1);
            }
            threednow_scale_accum_2rows(tile_acc16 + lr, &acc0, &acc1, w_scale_ptr, x_scale_ptr);
        }
    } else if (base_qtype == QMX_QUANT_TERNARY_158) {
        for (uint32_t lr = 0; lr < 16u; lr += 2u) {
            const uint8_t *r0 = tile_ptr + lr * 8u;
            const uint8_t *r1 = r0 + 8u;
            uint64_t acc0 = 0, acc1 = 0;
            for (uint32_t b = 0; b < 8u; b += 2u) {
                threednow_lut_step_2rows(xq16_32 + b * 4u,
                                         qmx_lut_ternary_i16[r0[b]], qmx_lut_ternary_i16[r0[b + 1u]],
                                         qmx_lut_ternary_i16[r1[b]], qmx_lut_ternary_i16[r1[b + 1u]],
                                         &acc0, &acc1);
            }
            threednow_scale_accum_2rows(tile_acc16 + lr, &acc0, &acc1, w_scale_ptr, x_scale_ptr);
        }
    } else if (base_qtype == QMX_QUANT_Q4_SYM) {
        for (uint32_t lr = 0; lr < 16u; lr += 2u) {
            const uint8_t *r0 = tile_ptr + lr * 16u;
            const uint8_t *r1 = r0 + 16u;
            uint64_t acc0 = 0, acc1 = 0;
            for (uint32_t b = 0; b < 16u; b += 4u) {
                const uint8_t *w0p = r0 + b;
                const uint8_t *w1p = r1 + b;
                const int16_t *xp = xq16_32 + b * 2u;
                __asm__ volatile (
                    "movq (%[x]), %%mm0\n\tmovq 8(%[x]), %%mm1\n\t"
                    "movd (%[w0]), %%mm2\n\tmovq %%mm2, %%mm3\n\tpsrlq $4, %%mm3\n\t"
                    "punpcklbw %%mm3, %%mm2\n\tpand %[m0f], %%mm2\n\tpsubb %[b08], %%mm2\n\t"
                    "movq %%mm2, %%mm3\n\tpunpcklbw %%mm2, %%mm2\n\tpunpckhbw %%mm3, %%mm3\n\t"
                    "psraw $8, %%mm2\n\tpsraw $8, %%mm3\n\t"
                    "pmaddwd %%mm0, %%mm2\n\tpmaddwd %%mm1, %%mm3\n\t"
                    "paddd %%mm3, %%mm2\n\tpaddd %[a0], %%mm2\n\tmovq %%mm2, %[a0]\n\t"
                    "movd (%[w1]), %%mm2\n\tmovq %%mm2, %%mm3\n\tpsrlq $4, %%mm3\n\t"
                    "punpcklbw %%mm3, %%mm2\n\tpand %[m0f], %%mm2\n\tpsubb %[b08], %%mm2\n\t"
                    "movq %%mm2, %%mm3\n\tpunpcklbw %%mm2, %%mm2\n\tpunpckhbw %%mm3, %%mm3\n\t"
                    "psraw $8, %%mm2\n\tpsraw $8, %%mm3\n\t"
                    "pmaddwd %%mm0, %%mm2\n\tpmaddwd %%mm1, %%mm3\n\t"
                    "paddd %%mm3, %%mm2\n\tpaddd %[a1], %%mm2\n\tmovq %%mm2, %[a1]\n\t"
                    : [a0] "+m"(acc0), [a1] "+m"(acc1)
                    : [x] "r"(xp), [w0] "r"(w0p), [w1] "r"(w1p),
                      [m0f] "m"(g_3dn_q4_mask_0f), [b08] "m"(g_3dn_q4_bias_08)
                    : "mm0", "mm1", "mm2", "mm3", "memory"
                );
            }
            if (row_scales_u32 != NULL) {
                threednow_scale_accum_2rows_pair(tile_acc16 + lr, &acc0, &acc1, row_scales_u32 + lr, x_scale_ptr);
            } else {
                threednow_scale_accum_2rows(tile_acc16 + lr, &acc0, &acc1, w_scale_ptr, x_scale_ptr);
            }
        }
    } else if (base_qtype == QMX_QUANT_Q8_SYM) {
        for (uint32_t lr = 0; lr < 16u; lr += 2u) {
            const int8_t *r0 = (const int8_t *)(tile_ptr + lr * 32u);
            const int8_t *r1 = r0 + 32u;
            uint64_t acc0 = 0, acc1 = 0;
            for (uint32_t i = 0; i < 32u; i += 8u) {
                const int8_t  *w0p = r0 + i;
                const int8_t  *w1p = r1 + i;
                const int16_t *xp = xq16_32 + i;
                __asm__ volatile (
                    "movq (%[x]), %%mm0\n\tmovq 8(%[x]), %%mm1\n\t"
                    "movq (%[w0]), %%mm2\n\tmovq %%mm2, %%mm3\n\t"
                    "punpcklbw %%mm2, %%mm2\n\tpunpckhbw %%mm3, %%mm3\n\t"
                    "psraw $8, %%mm2\n\tpsraw $8, %%mm3\n\t"
                    "pmaddwd %%mm0, %%mm2\n\tpmaddwd %%mm1, %%mm3\n\t"
                    "paddd %%mm3, %%mm2\n\tpaddd %[a0], %%mm2\n\tmovq %%mm2, %[a0]\n\t"
                    "movq (%[w1]), %%mm2\n\tmovq %%mm2, %%mm3\n\t"
                    "punpcklbw %%mm2, %%mm2\n\tpunpckhbw %%mm3, %%mm3\n\t"
                    "psraw $8, %%mm2\n\tpsraw $8, %%mm3\n\t"
                    "pmaddwd %%mm0, %%mm2\n\tpmaddwd %%mm1, %%mm3\n\t"
                    "paddd %%mm3, %%mm2\n\tpaddd %[a1], %%mm2\n\tmovq %%mm2, %[a1]\n\t"
                    : [a0] "+m"(acc0), [a1] "+m"(acc1)
                    : [x] "r"(xp), [w0] "r"(w0p), [w1] "r"(w1p)
                    : "mm0", "mm1", "mm2", "mm3", "memory"
                );
            }
            if (row_scales_u32 != NULL) {
                threednow_scale_accum_2rows_pair(tile_acc16 + lr, &acc0, &acc1, row_scales_u32 + lr, x_scale_ptr);
            } else {
                threednow_scale_accum_2rows(tile_acc16 + lr, &acc0, &acc1, w_scale_ptr, x_scale_ptr);
            }
        }
    }
}

static void threednow_matmul_tiled(float         * QMX_RESTRICT out,
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

    uint32_t tile_acc_u32[QMX_TILE_ROWS] QMX_ALIGNED(8);
    uint32_t row_sc_u32[QMX_TILE_ROWS] QMX_ALIGNED(8);
    float *tile_acc = (float *)(void *)tile_acc_u32;
    const uint32_t *out_u32 = (const uint32_t *)(const void *)out;
    uint32_t *out_u32_mut = (uint32_t *)(void *)out;

    for (uint32_t tr = 0; tr < n_tr; tr++) {
        uint32_t r_base = tr * QMX_TILE_ROWS;
        for (uint32_t lr = 0; lr < QMX_TILE_ROWS; lr++) {
            tile_acc_u32[lr] = (accumulate && (r_base + lr < orig_rows)) ? out_u32[r_base + lr] : 0u;
        }

        for (uint32_t tc = 0; tc < n_tc; tc++) {
            uint32_t tile_idx = tr * n_tc + tc;
            const uint8_t *tile_ptr = base_data + (size_t)tile_idx * tile_bytes;
            const int16_t *xq16_ptr = xq16 + (size_t)tc * QMX_TILE_COLS;
            if (is_row_scaled) {
                const uint16_t *rs = scales_f16 + (size_t)tile_idx * QMX_TILE_ROWS;
                for (uint32_t lr = 0; lr < QMX_TILE_ROWS; lr++) {
                    row_sc_u32[lr] = threednow_fp16_to_f32_bits(rs[lr]);
                }
                threednow_accum_tile(tile_acc, tile_ptr, xq16_ptr, base_qtype, NULL, row_sc_u32, &x_scales[tc]);
            } else {
                threednow_accum_tile(tile_acc, tile_ptr, xq16_ptr, base_qtype, &w->scales[tile_idx], NULL, &x_scales[tc]);
            }
        }

        for (uint32_t lr = 0; lr < QMX_TILE_ROWS; lr++) {
            if (r_base + lr < orig_rows) {
                out_u32_mut[r_base + lr] = tile_acc_u32[lr];
            }
        }
    }
    QMX_FEMMS();
}

const QmxBackendOps qmx_ops_3dnow = {
    "3DNow!",
    threednow_prepare_act,
    threednow_matmul_tiled,
    threednow_rmsnorm,
    threednow_dot_f32,
    threednow_saxpy_f32,
    threednow_dot_q8,
    NULL,
    qmx_swiglu_quant_scalar,
    qmx_f16_to_f32_scalar,
    qmx_f32_to_f16_scalar
};
#endif
