/*
 * Copyright 2026 1WhiteFox
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/qmx_kernels.h"

#if defined(QMX_ENABLE_BACKEND_MMX) && (defined(__i386__) || defined(__x86_64__)) && (defined(__GNUC__) || defined(__clang__))

#if !defined(__MMX__) && (defined(__clang__) || (defined(__GNUC__) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 9))))
    #define QMX_MMX_TARGET __attribute__((target("mmx")))
#else
    #define QMX_MMX_TARGET
#endif


static QMX_ALIGNED(8) const uint64_t g_mmx_q4_mask_0f = 0x0F0F0F0F0F0F0F0FULL;
static QMX_ALIGNED(8) const uint64_t g_mmx_q4_bias_08 = 0x0808080808080808ULL;


#define QMX_EMMS() \
    __asm__ volatile ("emms" : : : "st", "st(1)", "st(2)", "st(3)", "st(4)", "st(5)", "st(6)", "st(7)", "memory")

QMX_MMX_TARGET
static void mmx_prepare_act(const float * QMX_RESTRICT x,
                            int8_t      * QMX_RESTRICT xq,
                            int16_t     * QMX_RESTRICT xq16,
                            float       * QMX_RESTRICT x_scales,
                            uint32_t cols) {
    qmx_quantize_activations_q8(x, xq, x_scales, cols);
    uint32_t pad_cols = (cols + (QMX_TILE_COLS - 1u)) & ~(QMX_TILE_COLS - 1u);
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
    QMX_EMMS();
}

QMX_MMX_TARGET
QMX_INLINE void mmx_lut_step_2rows(const int16_t *x_ptr,
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

QMX_MMX_TARGET
QMX_INLINE void mmx_reduce_2rows(const uint64_t *acc0, const uint64_t *acc1,
                                 int32_t *s0, int32_t *s1) {
    __asm__ volatile (
        "movq %[a0], %%mm6\n\t"
        "movq %[a1], %%mm7\n\t"
        "movq %%mm6, %%mm0\n\t"
        "psrlq $32, %%mm0\n\t"
        "paddd %%mm0, %%mm6\n\t"
        "movd %%mm6, %[s0]\n\t"
        "movq %%mm7, %%mm1\n\t"
        "psrlq $32, %%mm1\n\t"
        "paddd %%mm1, %%mm7\n\t"
        "movd %%mm7, %[s1]\n\t"
        : [s0] "=r"(*s0), [s1] "=r"(*s1)
        : [a0] "m"(*acc0), [a1] "m"(*acc1)
        : "mm0", "mm1", "mm6", "mm7"
    );
}

#define MMX_Q8_BLOCK8_2ROWS(X_OFF, W_OFF) \
    "movq " #X_OFF "(%[x]), %%mm0\n\t" \
    "movq " #X_OFF "+8(%[x]), %%mm1\n\t" \
    "movq " #W_OFF "(%[w0]), %%mm2\n\t" \
    "movq " #W_OFF "(%[w1]), %%mm4\n\t" \
    "movq %%mm2, %%mm3\n\t" \
    "movq %%mm4, %%mm5\n\t" \
    "punpcklbw %%mm2, %%mm2\n\t" \
    "punpckhbw %%mm3, %%mm3\n\t" \
    "punpcklbw %%mm4, %%mm4\n\t" \
    "punpckhbw %%mm5, %%mm5\n\t" \
    "psraw $8, %%mm2\n\t" \
    "psraw $8, %%mm3\n\t" \
    "psraw $8, %%mm4\n\t" \
    "psraw $8, %%mm5\n\t" \
    "pmaddwd %%mm0, %%mm2\n\t" \
    "pmaddwd %%mm1, %%mm3\n\t" \
    "pmaddwd %%mm0, %%mm4\n\t" \
    "paddd %%mm2, %%mm6\n\t" \
    "pmaddwd %%mm1, %%mm5\n\t" \
    "paddd %%mm3, %%mm6\n\t" \
    "paddd %%mm4, %%mm7\n\t" \
    "paddd %%mm5, %%mm7\n\t"

#define MMX_Q4_BLOCK8_2ROWS(X_OFF, W_OFF) \
    "movd " #W_OFF "(%[w0]), %%mm2\n\t" \
    "movd " #W_OFF "(%[w1]), %%mm0\n\t" \
    "movq %%mm2, %%mm3\n\t" \
    "movq %%mm0, %%mm1\n\t" \
    "psrlq $4, %%mm3\n\t" \
    "psrlq $4, %%mm1\n\t" \
    "punpcklbw %%mm3, %%mm2\n\t" \
    "punpcklbw %%mm1, %%mm0\n\t" \
    "pand %%mm4, %%mm2\n\t" \
    "pand %%mm4, %%mm0\n\t" \
    "psubb %%mm5, %%mm2\n\t" \
    "psubb %%mm5, %%mm0\n\t" \
    "movq %%mm2, %%mm3\n\t" \
    "movq %%mm0, %%mm1\n\t" \
    "punpcklbw %%mm2, %%mm2\n\t" \
    "punpckhbw %%mm3, %%mm3\n\t" \
    "punpcklbw %%mm0, %%mm0\n\t" \
    "punpckhbw %%mm1, %%mm1\n\t" \
    "psraw $8, %%mm2\n\t" \
    "psraw $8, %%mm3\n\t" \
    "psraw $8, %%mm0\n\t" \
    "psraw $8, %%mm1\n\t" \
    "pmaddwd " #X_OFF "(%[x]), %%mm2\n\t" \
    "pmaddwd " #X_OFF "+8(%[x]), %%mm3\n\t" \
    "pmaddwd " #X_OFF "(%[x]), %%mm0\n\t" \
    "paddd %%mm2, %%mm6\n\t" \
    "pmaddwd " #X_OFF "+8(%[x]), %%mm1\n\t" \
    "paddd %%mm3, %%mm6\n\t" \
    "paddd %%mm0, %%mm7\n\t" \
    "paddd %%mm1, %%mm7\n\t"

QMX_MMX_TARGET
static void mmx_dot_tile_16x32(const uint8_t *tile_ptr,
                               const int16_t *xq16_32,
                               uint8_t qtype,
                               int32_t *dots16) {
    if (qtype == QMX_QUANT_BINARY_1BIT) {
        for (uint32_t lr = 0; lr < 16u; lr += 2u) {
            const uint8_t *r0 = tile_ptr + lr * 4u;
            const uint8_t *r1 = r0 + 4u;
            uint64_t acc0 = 0, acc1 = 0;
            int32_t s0, s1;
            for (uint32_t b = 0; b < 4u; b++) {
                uint8_t p0 = r0[b];
                uint8_t p1 = r1[b];
                mmx_lut_step_2rows(xq16_32 + b * 8u,
                                   qmx_lut_binary_i16_4[p0 & 0x0Fu], qmx_lut_binary_i16_4[p0 >> 4u],
                                   qmx_lut_binary_i16_4[p1 & 0x0Fu], qmx_lut_binary_i16_4[p1 >> 4u],
                                   &acc0, &acc1);
            }
            mmx_reduce_2rows(&acc0, &acc1, &s0, &s1);
            dots16[lr] = s0;
            dots16[lr + 1u] = s1;
        }
    } else if (qtype == QMX_QUANT_TERNARY_158) {
        for (uint32_t lr = 0; lr < 16u; lr += 2u) {
            const uint8_t *r0 = tile_ptr + lr * 8u;
            const uint8_t *r1 = r0 + 8u;
            uint64_t acc0 = 0, acc1 = 0;
            int32_t s0, s1;
            mmx_lut_step_2rows(xq16_32 +  0u, qmx_lut_ternary_i16[r0[0]], qmx_lut_ternary_i16[r0[1]], qmx_lut_ternary_i16[r1[0]], qmx_lut_ternary_i16[r1[1]], &acc0, &acc1);
            mmx_lut_step_2rows(xq16_32 +  8u, qmx_lut_ternary_i16[r0[2]], qmx_lut_ternary_i16[r0[3]], qmx_lut_ternary_i16[r1[2]], qmx_lut_ternary_i16[r1[3]], &acc0, &acc1);
            mmx_lut_step_2rows(xq16_32 + 16u, qmx_lut_ternary_i16[r0[4]], qmx_lut_ternary_i16[r0[5]], qmx_lut_ternary_i16[r1[4]], qmx_lut_ternary_i16[r1[5]], &acc0, &acc1);
            mmx_lut_step_2rows(xq16_32 + 24u, qmx_lut_ternary_i16[r0[6]], qmx_lut_ternary_i16[r0[7]], qmx_lut_ternary_i16[r1[6]], qmx_lut_ternary_i16[r1[7]], &acc0, &acc1);
            mmx_reduce_2rows(&acc0, &acc1, &s0, &s1);
            dots16[lr] = s0;
            dots16[lr + 1u] = s1;
        }
    } else if (qtype == QMX_QUANT_Q4_SYM) {
        for (uint32_t lr = 0; lr < 16u; lr += 2u) {
            const uint8_t *r0 = tile_ptr + lr * 16u;
            const uint8_t *r1 = r0 + 16u;
            int32_t s0, s1;
            __asm__ volatile (
                "movq %[m0f], %%mm4\n\tmovq %[b08], %%mm5\n\t"
                "pxor %%mm6, %%mm6\n\tpxor %%mm7, %%mm7\n\t"
                MMX_Q4_BLOCK8_2ROWS(0,  0)
                MMX_Q4_BLOCK8_2ROWS(16, 4)
                MMX_Q4_BLOCK8_2ROWS(32, 8)
                MMX_Q4_BLOCK8_2ROWS(48, 12)
                "movq %%mm6, %%mm0\n\tpsrlq $32, %%mm0\n\tpaddd %%mm0, %%mm6\n\tmovd %%mm6, %[s0]\n\t"
                "movq %%mm7, %%mm1\n\tpsrlq $32, %%mm1\n\tpaddd %%mm1, %%mm7\n\tmovd %%mm7, %[s1]\n\t"
                : [s0] "=r"(s0), [s1] "=r"(s1)
                : [x] "r"(xq16_32), [w0] "r"(r0), [w1] "r"(r1),
                    [m0f] "m"(g_mmx_q4_mask_0f), [b08] "m"(g_mmx_q4_bias_08)
                : "mm0", "mm1", "mm2", "mm3", "mm4", "mm5", "mm6", "mm7"
            );
            dots16[lr] = s0;
            dots16[lr + 1u] = s1;
        }
    } else if (qtype == QMX_QUANT_Q8_SYM) {
        for (uint32_t lr = 0; lr < 16u; lr += 2u) {
            const int8_t *r0 = (const int8_t *)(tile_ptr + lr * 32u);
            const int8_t *r1 = r0 + 32u;
            int32_t s0, s1;
            __asm__ volatile (
                "pxor %%mm6, %%mm6\n\tpxor %%mm7, %%mm7\n\t"
                MMX_Q8_BLOCK8_2ROWS(0,  0)
                MMX_Q8_BLOCK8_2ROWS(16, 8)
                MMX_Q8_BLOCK8_2ROWS(32, 16)
                MMX_Q8_BLOCK8_2ROWS(48, 24)
                "movq %%mm6, %%mm0\n\tpsrlq $32, %%mm0\n\tpaddd %%mm0, %%mm6\n\tmovd %%mm6, %[s0]\n\t"
                "movq %%mm7, %%mm1\n\tpsrlq $32, %%mm1\n\tpaddd %%mm1, %%mm7\n\tmovd %%mm7, %[s1]\n\t"
                : [s0] "=r"(s0), [s1] "=r"(s1)
                : [x] "r"(xq16_32), [w0] "r"(r0), [w1] "r"(r1)
                : "mm0", "mm1", "mm2", "mm3", "mm4", "mm5", "mm6", "mm7"
            );
            dots16[lr] = s0;
            dots16[lr + 1u] = s1;
        }
    }
    QMX_EMMS();
}

static void mmx_matmul_tiled(float         * QMX_RESTRICT out,
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

    float tile_acc[QMX_TILE_ROWS];
    int32_t dots16[QMX_TILE_ROWS];

    for (uint32_t tr = 0; tr < n_tr; tr++) {
        uint32_t r_base = tr * QMX_TILE_ROWS;
        for (uint32_t lr = 0; lr < QMX_TILE_ROWS; lr++) {
            tile_acc[lr] = (accumulate && (r_base + lr < orig_rows)) ? out[r_base + lr] : 0.0f;
        }

        for (uint32_t tc = 0; tc < n_tc; tc++) {
            uint32_t tile_idx = tr * n_tc + tc;
            float x_sc = x_scales[tc];
            const uint8_t *tile_ptr = base_data + (size_t)tile_idx * tile_bytes;
            const int16_t *xq16_ptr = xq16 + (size_t)tc * QMX_TILE_COLS;

            mmx_dot_tile_16x32(tile_ptr, xq16_ptr, base_qtype, dots16);

            if (is_row_scaled) {
                const uint16_t *row_sc = scales_f16 + (size_t)tile_idx * QMX_TILE_ROWS;
                for (uint32_t lr = 0; lr < QMX_TILE_ROWS; lr++) {
                    tile_acc[lr] += (float)dots16[lr] * (qmx_fp16_to_fp32(row_sc[lr]) * x_sc);
                }
            } else {
                float comb_scale = w->scales[tile_idx] * x_sc;
                for (uint32_t lr = 0; lr < QMX_TILE_ROWS; lr++) {
                    tile_acc[lr] += (float)dots16[lr] * comb_scale;
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

static float mmx_dot_f32(const float * QMX_RESTRICT a,
                         const float * QMX_RESTRICT b,
                         uint32_t n) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

static void mmx_saxpy_f32(float       * QMX_RESTRICT y,
                          float a,
                          const float * QMX_RESTRICT x,
                          uint32_t n) {
    for (uint32_t i = 0; i < n; i++) y[i] += a * x[i];
}

QMX_MMX_TARGET
static int32_t mmx_dot_q8(const int8_t  * QMX_RESTRICT q8,
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
    QMX_EMMS();
    for (; i < n; i++) sum += (int32_t)q16[i] * (int32_t)k8[i];
    return sum;
}

const QmxBackendOps qmx_ops_mmx = {
    "MMX",
    mmx_prepare_act,
    mmx_matmul_tiled,
    qmx_rmsnorm_scalar,
    mmx_dot_f32,
    mmx_saxpy_f32,
    mmx_dot_q8,
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
