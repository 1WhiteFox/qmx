#include "../include/qmx_kernels.h"

#if defined(QMX_ENABLE_BACKEND_P54C)

static void p54c_prepare_act(const float * QMX_RESTRICT x,
                             int8_t      * QMX_RESTRICT xq,
                             int16_t     * QMX_RESTRICT xq16,
                             float       * QMX_RESTRICT x_scales,
                             uint32_t cols) {
    (void)xq16;
    qmx_quantize_activations_q8(x, xq, x_scales, cols);
}

static int32_t p54c_dot_row32(const uint8_t * QMX_RESTRICT rp,
                              const int8_t  * QMX_RESTRICT xp,
                              uint8_t qtype) {
    int32_t acc0 = 0;
    int32_t acc1 = 0;
    if (qtype == QMX_QUANT_BINARY_1BIT) {
        for (uint32_t b = 0; b < 4u; b++) {
            uint8_t p = rp[b];
            const int8_t *l0 = qmx_lut_binary_s8[p & 0x0Fu];
            const int8_t *l1 = qmx_lut_binary_s8[p >> 4u];
            const int8_t *x0 = xp + b * 8u;
            acc0 += (int32_t)l0[0] * x0[0] + (int32_t)l0[1] * x0[1]
                  + (int32_t)l0[2] * x0[2] + (int32_t)l0[3] * x0[3];
            acc1 += (int32_t)l1[0] * x0[4] + (int32_t)l1[1] * x0[5]
                  + (int32_t)l1[2] * x0[6] + (int32_t)l1[3] * x0[7];
        }
    } else if (qtype == QMX_QUANT_TERNARY_158) {
        for (uint32_t b = 0; b < 8u; b += 2u) {
            const int8_t *l0 = qmx_lut_ternary_s8[rp[b]];
            const int8_t *l1 = qmx_lut_ternary_s8[rp[b + 1u]];
            const int8_t *x0 = xp + b * 4u;
            acc0 += (int32_t)l0[0] * x0[0] + (int32_t)l0[1] * x0[1]
                  + (int32_t)l0[2] * x0[2] + (int32_t)l0[3] * x0[3];
            acc1 += (int32_t)l1[0] * x0[4] + (int32_t)l1[1] * x0[5]
                  + (int32_t)l1[2] * x0[6] + (int32_t)l1[3] * x0[7];
        }
    } else if (qtype == QMX_QUANT_Q4_SYM) {
        for (uint32_t b = 0; b < 16u; b += 2u) {
            uint8_t p0 = rp[b];
            uint8_t p1 = rp[b + 1u];
            const int8_t *x0 = xp + b * 2u;
            acc0 += ((int32_t)(p0 & 0x0Fu) - 8) * x0[0]
                  + ((int32_t)(p0 >> 4u)   - 8) * x0[1];
            acc1 += ((int32_t)(p1 & 0x0Fu) - 8) * x0[2]
                  + ((int32_t)(p1 >> 4u)   - 8) * x0[3];
        }
    } else if (qtype == QMX_QUANT_Q8_SYM) {
        const int8_t *wp = (const int8_t *)rp;
        for (uint32_t i = 0; i < 32u; i += 8u) {
            acc0 += (int32_t)wp[i + 0u] * xp[i + 0u] + (int32_t)wp[i + 1u] * xp[i + 1u]
                  + (int32_t)wp[i + 2u] * xp[i + 2u] + (int32_t)wp[i + 3u] * xp[i + 3u];
            acc1 += (int32_t)wp[i + 4u] * xp[i + 4u] + (int32_t)wp[i + 5u] * xp[i + 5u]
                  + (int32_t)wp[i + 6u] * xp[i + 6u] + (int32_t)wp[i + 7u] * xp[i + 7u];
        }
    }
    return acc0 + acc1;
}

static void p54c_matmul_tiled(float         * QMX_RESTRICT out,
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
                int32_t dot = p54c_dot_row32(tile_ptr + (size_t)lr * row_bytes, xq_ptr, base_qtype);
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

static float p54c_dot_f32(const float * QMX_RESTRICT a,
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

static void p54c_saxpy_f32(float       * QMX_RESTRICT y,
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

static int32_t p54c_dot_q8(const int8_t  * QMX_RESTRICT q8,
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

const QmxBackendOps qmx_ops_p54c = {
    "P54C",
    p54c_prepare_act,
    p54c_matmul_tiled,
    qmx_rmsnorm_scalar,
    p54c_dot_f32,
    p54c_saxpy_f32,
    p54c_dot_q8,
    qmx_rmsnorm_quant_scalar,
    qmx_swiglu_quant_scalar,
    qmx_f16_to_f32_scalar,
    qmx_f32_to_f16_scalar
};

#endif
