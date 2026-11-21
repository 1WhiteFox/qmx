#if !defined(_WIN32) && !defined(__DJGPP__)
    #ifndef _DEFAULT_SOURCE
        #define _DEFAULT_SOURCE
    #endif
    #ifndef _POSIX_C_SOURCE
        #define _POSIX_C_SOURCE 200809L
    #endif
#endif

#include "../include/qmx_runtime.h"
#include <limits.h>
#ifndef QMX_NO_MALLOC
#include <stdlib.h>
#endif
#include <string.h>

#ifndef QMX_NO_STDIO
int qmx_fseek64(FILE *fp, int64_t offset, int origin) {
    if (!fp) return -1;
#if defined(_WIN64)
    return _fseeki64(fp, (__int64)offset, origin);
#elif defined(_WIN32) || defined(__DJGPP__)
    if (offset < -(int64_t)LONG_MAX || offset > (int64_t)LONG_MAX) return -1;
    return fseek(fp, (long)offset, origin);
#else
    return fseeko(fp, (off_t)offset, origin);
#endif
}

int64_t qmx_ftell64(FILE *fp) {
    if (!fp) return -1;
#if defined(_WIN64)
    return (int64_t)_ftelli64(fp);
#elif defined(_WIN32) || defined(__DJGPP__)
    return (int64_t)ftell(fp);
#else
    return (int64_t)ftello(fp);
#endif
}

int64_t qmx_file_size64(FILE *fp) {
    if (!fp) return -1;
    int64_t cur = qmx_ftell64(fp);
    if (qmx_fseek64(fp, 0, SEEK_END) != 0) return -1;
    int64_t sz = qmx_ftell64(fp);
    if (cur >= 0) (void)qmx_fseek64(fp, cur, SEEK_SET);
    return sz;
}
#endif

uint32_t qmx_crc32(const void *data, size_t len) {
    const uint8_t *p = (const uint8_t *)data;
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; i++) {
        crc ^= (uint32_t)p[i];
        for (int b = 0; b < 8; b++) {
            uint32_t mask = -(int32_t)(crc & 1u);
            crc = (crc >> 1) ^ (0xEDB88320u & mask);
        }
    }
    return ~crc;
}

size_t qmx_tile_data_bytes(uint8_t quant_type) {
    if (quant_type == QMX_QUANT_BINARY_1BIT) return 64u;
    if (quant_type == QMX_QUANT_TERNARY_158) return 128u;
    if (quant_type == QMX_QUANT_Q4_SYM || quant_type == QMX_QUANT_Q4_ROW32) return 256u;
    if (quant_type == QMX_QUANT_Q8_SYM || quant_type == QMX_QUANT_Q8_ROW32) return 512u;
    return 0u;
}

size_t qmx_tile_row_bytes(uint8_t quant_type) {
    if (quant_type == QMX_QUANT_BINARY_1BIT) return 4u;
    if (quant_type == QMX_QUANT_TERNARY_158) return 8u;
    if (quant_type == QMX_QUANT_Q4_SYM || quant_type == QMX_QUANT_Q4_ROW32) return 16u;
    if (quant_type == QMX_QUANT_Q8_SYM || quant_type == QMX_QUANT_Q8_ROW32) return 32u;
    return 0u;
}

uint64_t qmx_matrix_data_bytes(uint32_t rows, uint32_t cols, uint8_t quant_type) {
    if (quant_type == QMX_QUANT_FP32) {
        return (uint64_t)rows * (uint64_t)cols * (uint64_t)sizeof(float);
    }
    uint64_t pad_r = (uint64_t)qmx_align_up(rows, QMX_TILE_ROWS);
    uint64_t pad_c = (uint64_t)qmx_align_up(cols, QMX_TILE_COLS);
    uint64_t num_tiles = (pad_r / QMX_TILE_ROWS) * (pad_c / QMX_TILE_COLS);
    return num_tiles * (uint64_t)qmx_tile_data_bytes(quant_type);
}

size_t qmx_matrix_scales_bytes(uint32_t rows, uint32_t cols) {
    size_t pad_r = qmx_align_up(rows, QMX_TILE_ROWS);
    size_t pad_c = qmx_align_up(cols, QMX_TILE_COLS);
    size_t num_tiles = (pad_r / QMX_TILE_ROWS) * (pad_c / QMX_TILE_COLS);
    return num_tiles * sizeof(float);
}

uint64_t qmx_matrix_scales_bytes_ex(uint32_t rows, uint32_t cols, uint8_t quant_type) {
    if (quant_type == QMX_QUANT_FP32) return 0u;
    uint64_t pad_r = (uint64_t)qmx_align_up(rows, QMX_TILE_ROWS);
    uint64_t pad_c = (uint64_t)qmx_align_up(cols, QMX_TILE_COLS);
    uint64_t num_tiles = (pad_r / QMX_TILE_ROWS) * (pad_c / QMX_TILE_COLS);
    if (quant_type == QMX_QUANT_Q4_ROW32 || quant_type == QMX_QUANT_Q8_ROW32) {
        return num_tiles * (uint64_t)QMX_TILE_ROWS * (uint64_t)sizeof(uint16_t);
    }
    return num_tiles * (uint64_t)sizeof(float);
}

int qmx_arena_init(QmxArena *arena, size_t capacity) {
#ifdef QMX_NO_MALLOC
    (void)arena;
    (void)capacity;
    return -1;
#else
    arena->raw_ptr = (uint8_t *)malloc(capacity + QMX_CACHE_ALIGN);
    if (!arena->raw_ptr) return -1;
    uintptr_t addr = (uintptr_t)arena->raw_ptr;
    uintptr_t aligned = (addr + (QMX_CACHE_ALIGN - 1u)) & ~(uintptr_t)(QMX_CACHE_ALIGN - 1u);
    arena->base_ptr = (uint8_t *)aligned;
    arena->capacity = capacity;
    arena->offset = 0;
    arena->is_external = 0;
    memset(arena->base_ptr, 0, capacity);
    return 0;
#endif
}

int qmx_arena_init_external(QmxArena *arena, void *buf, size_t capacity) {
    if (!arena || !buf || capacity == 0u) return -1;
    uintptr_t addr = (uintptr_t)buf;
    uintptr_t aligned = (addr + (QMX_CACHE_ALIGN - 1u)) & ~(uintptr_t)(QMX_CACHE_ALIGN - 1u);
    size_t diff = (size_t)(aligned - addr);
    if (diff >= capacity) return -1;
    arena->raw_ptr = NULL;
    arena->base_ptr = (uint8_t *)aligned;
    arena->capacity = capacity - diff;
    arena->offset = 0;
    arena->is_external = 1;
    memset(arena->base_ptr, 0, arena->capacity);
    return 0;
}

void *qmx_arena_alloc(QmxArena *arena, size_t size, size_t alignment) {
    if (alignment == 0) alignment = 1;
    size_t aligned_off = qmx_align_up(arena->offset, alignment);
    if (aligned_off > arena->capacity || size > arena->capacity - aligned_off) {
        return NULL;
    }
    void *ptr = arena->base_ptr + aligned_off;
    arena->offset = aligned_off + size;
    return ptr;
}

void qmx_arena_free(QmxArena *arena) {
    if (!arena) return;
#ifndef QMX_NO_MALLOC
    if (arena->raw_ptr && !arena->is_external) {
        free(arena->raw_ptr);
    }
#endif
    arena->raw_ptr = NULL;
    arena->base_ptr = NULL;
    arena->capacity = 0;
    arena->offset = 0;
    arena->is_external = 0;
}

void qmx_rmsnorm_scalar(float * QMX_RESTRICT out, const float * QMX_RESTRICT x,
                        const float * QMX_RESTRICT weight, uint32_t size, float eps) {
    float ss = 0.0f;
    for (uint32_t j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss = 1.0f / qmx_sqrtf(ss / (float)size + eps);
    for (uint32_t j = 0; j < size; j++) {
        out[j] = weight[j] * (ss * x[j]);
    }
}

void qmx_subnorm_scalar(float * QMX_RESTRICT out, const float * QMX_RESTRICT x,
                        const float * QMX_RESTRICT weight, uint32_t size, float eps) {
    float ss = 0.0f;
    for (uint32_t j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss = 1.0f / qmx_sqrtf(ss / (float)size + eps);
    if (weight != NULL) {
        for (uint32_t j = 0; j < size; j++) {
            out[j] = weight[j] * (ss * x[j]);
        }
    } else {
        for (uint32_t j = 0; j < size; j++) {
            out[j] = ss * x[j];
        }
    }
}

void qmx_layernorm_scalar(float * QMX_RESTRICT out, const float * QMX_RESTRICT x,
                          const float * QMX_RESTRICT weight, uint32_t size, float eps) {
    if (size == 0) return;
    float mean = 0.0f;
    for (uint32_t j = 0; j < size; j++) {
        mean += x[j];
    }
    mean /= (float)size;
    float var = 0.0f;
    for (uint32_t j = 0; j < size; j++) {
        float d = x[j] - mean;
        var += d * d;
    }
    float inv_std = 1.0f / qmx_sqrtf(var / (float)size + eps);
    if (weight != NULL) {
        for (uint32_t j = 0; j < size; j++) {
            out[j] = ((x[j] - mean) * inv_std) * weight[j];
        }
    } else {
        for (uint32_t j = 0; j < size; j++) {
            out[j] = (x[j] - mean) * inv_std;
        }
    }
}

void qmx_qk_rmsnorm_scalar(float * QMX_RESTRICT vec, const float * QMX_RESTRICT weight,
                           uint32_t total_dim, uint32_t head_size, float eps) {
    if (!vec || !weight || head_size == 0) return;
    for (uint32_t base = 0; base < total_dim; base += head_size) {
        float *h = vec + base;
        float ss = 0.0f;
        for (uint32_t j = 0; j < head_size; j++) {
            ss += h[j] * h[j];
        }
        ss = 1.0f / qmx_sqrtf(ss / (float)head_size + eps);
        for (uint32_t j = 0; j < head_size; j++) {
            h[j] = weight[j] * (ss * h[j]);
        }
    }
}

void qmx_softmax_scalar(float * QMX_RESTRICT x, uint32_t size) {
    if (size == 0) return;
    if (size == 1) {
        x[0] = 1.0f;
        return;
    }
    float max_val = x[0];
    for (uint32_t i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (uint32_t i = 0; i < size; i++) {
        float e = qmx_expf(x[i] - max_val);
        x[i] = e;
        sum += e;
    }
    float inv_sum = (sum > 0.0f) ? (1.0f / sum) : (1.0f / (float)size);
    for (uint32_t i = 0; i < size; i++) {
        x[i] *= inv_sum;
    }
}

void qmx_rope_apply_partial(float * QMX_RESTRICT q, float * QMX_RESTRICT k,
                            const float * QMX_RESTRICT fcr, const float * QMX_RESTRICT fci,
                            uint32_t q_dim, uint32_t kv_dim, uint32_t head_size,
                            uint32_t rotary_dim, int split_half) {
    if (head_size == 0u) return;
    if (rotary_dim == 0u || rotary_dim > head_size) rotary_dim = head_size;
    rotary_dim &= ~1u;
    uint32_t half_rot = rotary_dim / 2u;
    if (half_rot == 0u) return;

    if (split_half) {
        if (q != NULL) {
            for (uint32_t base = 0; base < q_dim; base += head_size) {
                float *qh = q + base;
                for (uint32_t i = 0; i < half_rot; i++) {
                    float cr = fcr[i];
                    float ci = fci[i];
                    float q0 = qh[i];
                    float q1 = qh[i + half_rot];
                    qh[i]            = q0 * cr - q1 * ci;
                    qh[i + half_rot] = q0 * ci + q1 * cr;
                }
            }
        }
        if (k != NULL) {
            for (uint32_t base = 0; base < kv_dim; base += head_size) {
                float *kh = k + base;
                for (uint32_t i = 0; i < half_rot; i++) {
                    float cr = fcr[i];
                    float ci = fci[i];
                    float k0 = kh[i];
                    float k1 = kh[i + half_rot];
                    kh[i]            = k0 * cr - k1 * ci;
                    kh[i + half_rot] = k0 * ci + k1 * cr;
                }
            }
        }
    } else {
        if (q != NULL) {
            for (uint32_t base = 0; base < q_dim; base += head_size) {
                float *qh = q + base;
                for (uint32_t hd = 0, idx = 0; hd < rotary_dim; hd += 2u, idx++) {
                    float cr = fcr[idx];
                    float ci = fci[idx];
                    float q0 = qh[hd];
                    float q1 = qh[hd + 1u];
                    qh[hd]      = q0 * cr - q1 * ci;
                    qh[hd + 1u] = q0 * ci + q1 * cr;
                }
            }
        }
        if (k != NULL) {
            for (uint32_t base = 0; base < kv_dim; base += head_size) {
                float *kh = k + base;
                for (uint32_t hd = 0, idx = 0; hd < rotary_dim; hd += 2u, idx++) {
                    float cr = fcr[idx];
                    float ci = fci[idx];
                    float k0 = kh[hd];
                    float k1 = kh[hd + 1u];
                    kh[hd]      = k0 * cr - k1 * ci;
                    kh[hd + 1u] = k0 * ci + k1 * cr;
                }
            }
        }
    }
}

void qmx_rope_apply(float * QMX_RESTRICT q, float * QMX_RESTRICT k,
                    const float * QMX_RESTRICT fcr, const float * QMX_RESTRICT fci,
                    uint32_t dim, uint32_t kv_dim, uint32_t head_size, int split_half) {
    qmx_rope_apply_partial(q, k, fcr, fci, dim, kv_dim, head_size, head_size, split_half);
}

void qmx_quantize_activations_q8(const float * QMX_RESTRICT x,
                                 int8_t      * QMX_RESTRICT xq,
                                 float       * QMX_RESTRICT x_scales,
                                 uint32_t cols) {
    uint32_t pad_cols = (uint32_t)qmx_align_up(cols, QMX_TILE_COLS);
    uint32_t num_groups = pad_cols / QMX_TILE_COLS;
    for (uint32_t g = 0; g < num_groups; g++) {
        uint32_t base = g * QMX_TILE_COLS;
        int8_t *qg = xq + base;
        float max_abs = 0.0f;
        for (uint32_t i = 0; i < QMX_TILE_COLS; i++) {
            float v = (base + i < cols) ? x[base + i] : 0.0f;
            float av = (v < 0.0f) ? -v : v;
            if (av > max_abs) max_abs = av;
        }
        float scale = max_abs / 127.0f;
        float inv_scale = (max_abs == 0.0f) ? 0.0f : (127.0f / max_abs);
        x_scales[g] = scale;
        for (uint32_t i = 0; i < QMX_TILE_COLS; i++) {
            if (base + i < cols) {
                int32_t iv = qmx_round_nearest_i32(x[base + i] * inv_scale);
                if (iv > 127) iv = 127;
                if (iv < -127) iv = -127;
                qg[i] = (int8_t)iv;
            } else {
                qg[i] = 0;
            }
        }
    }
}

void qmx_rmsnorm_quant_scalar(const float * QMX_RESTRICT x,
                              const float * QMX_RESTRICT weight,
                              int8_t      * QMX_RESTRICT xq,
                              int16_t     * QMX_RESTRICT xq16,
                              float       * QMX_RESTRICT x_scales,
                              uint32_t size,
                              float eps) {
    float ss = 0.0f;
    for (uint32_t j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    float inv_rms = 1.0f / qmx_sqrtf(ss / (float)size + eps);

    uint32_t pad_cols = (uint32_t)qmx_align_up(size, QMX_TILE_COLS);
    uint32_t num_groups = pad_cols / QMX_TILE_COLS;
    float tmp[QMX_TILE_COLS];

    for (uint32_t g = 0; g < num_groups; g++) {
        uint32_t base = g * QMX_TILE_COLS;
        int8_t *qg = xq + base;
        int16_t *qg16 = xq16 ? (xq16 + base) : NULL;
        float max_abs = 0.0f;
        for (uint32_t i = 0; i < QMX_TILE_COLS; i++) {
            uint32_t idx = base + i;
            float v = (idx < size) ? (weight[idx] * (inv_rms * x[idx])) : 0.0f;
            tmp[i] = v;
            float av = (v < 0.0f) ? -v : v;
            if (av > max_abs) max_abs = av;
        }
        x_scales[g] = max_abs / 127.0f;
        float inv_scale = (max_abs == 0.0f) ? 0.0f : (127.0f / max_abs);
        for (uint32_t i = 0; i < QMX_TILE_COLS; i++) {
            int8_t qv = 0;
            if (base + i < size) {
                int32_t iv = qmx_round_nearest_i32(tmp[i] * inv_scale);
                if (iv > 127) iv = 127;
                if (iv < -127) iv = -127;
                qv = (int8_t)iv;
            }
            qg[i] = qv;
            if (qg16) qg16[i] = (int16_t)qv;
        }
    }
}

void qmx_swiglu_quant_scalar(const float * QMX_RESTRICT gate,
                             const float * QMX_RESTRICT up,
                             int8_t      * QMX_RESTRICT xq,
                             int16_t     * QMX_RESTRICT xq16,
                             float       * QMX_RESTRICT x_scales,
                             uint32_t size) {
    uint32_t pad_cols = (uint32_t)qmx_align_up(size, QMX_TILE_COLS);
    uint32_t num_groups = pad_cols / QMX_TILE_COLS;
    float tmp[QMX_TILE_COLS];

    for (uint32_t g = 0; g < num_groups; g++) {
        uint32_t base = g * QMX_TILE_COLS;
        int8_t *qg = xq + base;
        int16_t *qg16 = xq16 ? (xq16 + base) : NULL;
        float max_abs = 0.0f;
        for (uint32_t i = 0; i < QMX_TILE_COLS; i++) {
            uint32_t idx = base + i;
            float v = 0.0f;
            if (idx < size) {
                float gv = gate[idx];
                v = gv * (1.0f / (1.0f + qmx_expf(-gv))) * up[idx];
            }
            tmp[i] = v;
            float av = (v < 0.0f) ? -v : v;
            if (av > max_abs) max_abs = av;
        }
        x_scales[g] = max_abs / 127.0f;
        float inv_scale = (max_abs == 0.0f) ? 0.0f : (127.0f / max_abs);
        for (uint32_t i = 0; i < QMX_TILE_COLS; i++) {
            int8_t qv = 0;
            if (base + i < size) {
                int32_t iv = qmx_round_nearest_i32(tmp[i] * inv_scale);
                if (iv > 127) iv = 127;
                if (iv < -127) iv = -127;
                qv = (int8_t)iv;
            }
            qg[i] = qv;
            if (qg16) qg16[i] = (int16_t)qv;
        }
    }
}

void qmx_f16_to_f32_scalar(float * QMX_RESTRICT dst, const uint16_t * QMX_RESTRICT src, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        dst[i] = qmx_fp16_to_fp32(src[i]);
    }
}

void qmx_f32_to_f16_scalar(uint16_t * QMX_RESTRICT dst, const float * QMX_RESTRICT src, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        dst[i] = qmx_fp32_to_fp16(src[i]);
    }
}

void qmx_dequantize_tiled_row(float * QMX_RESTRICT out,
                              const QmxTensor *w,
                              uint32_t row_idx) {
    if (!w || !w->data || !w->scales || row_idx >= w->rows) return;
    uint32_t cols = w->cols;
    uint32_t pad_c = (uint32_t)qmx_align_up(cols, QMX_TILE_COLS);
    uint32_t n_tc = pad_c / QMX_TILE_COLS;
    uint32_t tr = row_idx / QMX_TILE_ROWS;
    uint32_t lr = row_idx % QMX_TILE_ROWS;
    uint8_t qtype = w->quant_type;
    size_t tile_bytes = qmx_tile_data_bytes(qtype);
    size_t row_bytes = qmx_tile_row_bytes(qtype);
    const uint8_t *base_data = (const uint8_t *)w->data;
    const uint16_t *scales_f16 = (const uint16_t *)(const void *)w->scales;

    for (uint32_t tc = 0; tc < n_tc; tc++) {
        uint32_t tile_idx = tr * n_tc + tc;
        float sc = (qtype == QMX_QUANT_Q4_ROW32 || qtype == QMX_QUANT_Q8_ROW32)
                 ? qmx_fp16_to_fp32(scales_f16[(size_t)tile_idx * QMX_TILE_ROWS + lr])
                 : w->scales[tile_idx];
        const uint8_t *row_ptr = base_data + (size_t)tile_idx * tile_bytes + (size_t)lr * row_bytes;
        uint32_t col_base = tc * QMX_TILE_COLS;

        if (qtype == QMX_QUANT_BINARY_1BIT) {
            for (uint32_t b = 0; b < 4u; b++) {
                uint8_t p = row_ptr[b];
                for (uint32_t k = 0; k < 8u; k++) {
                    uint32_t c = col_base + b * 8u + k;
                    if (c < cols) out[c] = ((p >> k) & 1u) ? sc : -sc;
                }
            }
        } else if (qtype == QMX_QUANT_TERNARY_158) {
            for (uint32_t b = 0; b < 8u; b++) {
                uint8_t p = row_ptr[b];
                for (uint32_t k = 0; k < 4u; k++) {
                    uint32_t c = col_base + b * 4u + k;
                    if (c < cols) {
                        uint8_t code = (p >> (k * 2u)) & 0x03u;
                        float v = (code == 1u) ? 1.0f : ((code == 2u) ? -1.0f : 0.0f);
                        out[c] = v * sc;
                    }
                }
            }
        } else if (qtype == QMX_QUANT_Q4_SYM || qtype == QMX_QUANT_Q4_ROW32) {
            for (uint32_t b = 0; b < 16u; b++) {
                uint8_t p = row_ptr[b];
                uint32_t c0 = col_base + b * 2u;
                uint32_t c1 = c0 + 1u;
                if (c0 < cols) out[c0] = (float)((int32_t)(p & 0x0Fu) - 8) * sc;
                if (c1 < cols) out[c1] = (float)((int32_t)(p >> 4u) - 8) * sc;
            }
        } else if (qtype == QMX_QUANT_Q8_SYM || qtype == QMX_QUANT_Q8_ROW32) {
            const int8_t *s_ptr = (const int8_t *)row_ptr;
            for (uint32_t i = 0; i < QMX_TILE_COLS; i++) {
                uint32_t c = col_base + i;
                if (c < cols) out[c] = (float)s_ptr[i] * sc;
            }
        }
    }
}

static int32_t dot_tile_row_32(const uint8_t *row_ptr, const int8_t *xq_32, uint8_t qtype) {
    int32_t acc = 0;
    if (qtype == QMX_QUANT_BINARY_1BIT) {
        for (uint32_t b = 0; b < 4u; b++) {
            uint8_t p = row_ptr[b];
            const int8_t *xp = xq_32 + b * 8u;
            for (uint32_t k = 0; k < 8u; k++) {
                acc += ((p >> k) & 1u) ? (int32_t)xp[k] : -(int32_t)xp[k];
            }
        }
    } else if (qtype == QMX_QUANT_TERNARY_158) {
        for (uint32_t b = 0; b < 8u; b++) {
            uint8_t p = row_ptr[b];
            const int8_t *xp = xq_32 + b * 4u;
            uint8_t c0 = p & 0x03u;
            uint8_t c1 = (p >> 2u) & 0x03u;
            uint8_t c2 = (p >> 4u) & 0x03u;
            uint8_t c3 = (p >> 6u) & 0x03u;
            if (c0 == 1u) acc += xp[0]; else if (c0 == 2u) acc -= xp[0];
            if (c1 == 1u) acc += xp[1]; else if (c1 == 2u) acc -= xp[1];
            if (c2 == 1u) acc += xp[2]; else if (c2 == 2u) acc -= xp[2];
            if (c3 == 1u) acc += xp[3]; else if (c3 == 2u) acc -= xp[3];
        }
    } else if (qtype == QMX_QUANT_Q4_SYM || qtype == QMX_QUANT_Q4_ROW32) {
        for (uint32_t b = 0; b < 16u; b++) {
            uint8_t p = row_ptr[b];
            int32_t w0 = (int32_t)(p & 0x0Fu) - 8;
            int32_t w1 = (int32_t)(p >> 4u) - 8;
            acc += w0 * (int32_t)xq_32[b * 2u] + w1 * (int32_t)xq_32[b * 2u + 1u];
        }
    } else if (qtype == QMX_QUANT_Q8_SYM || qtype == QMX_QUANT_Q8_ROW32) {
        const int8_t *wp = (const int8_t *)row_ptr;
        for (uint32_t i = 0; i < QMX_TILE_COLS; i++) {
            acc += (int32_t)wp[i] * (int32_t)xq_32[i];
        }
    }
    return acc;
}

void qmx_matmul_tiled_scalar(float * QMX_RESTRICT out,
                             const int8_t * QMX_RESTRICT xq,
                             const float  * QMX_RESTRICT x_scales,
                             const QmxTensor *w,
                             int accumulate) {
    uint32_t orig_rows = w->rows;
    uint32_t orig_cols = w->cols;
    uint32_t pad_r = (uint32_t)qmx_align_up(orig_rows, QMX_TILE_ROWS);
    uint32_t pad_c = (uint32_t)qmx_align_up(orig_cols, QMX_TILE_COLS);
    uint32_t n_tr = pad_r / QMX_TILE_ROWS;
    uint32_t n_tc = pad_c / QMX_TILE_COLS;
    uint8_t qtype = w->quant_type;
    int is_row_scaled = (qtype == QMX_QUANT_Q4_ROW32 || qtype == QMX_QUANT_Q8_ROW32);
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
            const uint16_t *row_sc_ptr = is_row_scaled ? (scales_f16 + (size_t)tile_idx * QMX_TILE_ROWS) : NULL;
            const uint8_t *tile_ptr = base_data + (size_t)tile_idx * tile_bytes;
            const int8_t *xq_ptr = xq + (size_t)tc * QMX_TILE_COLS;

            for (uint32_t lr = 0; lr < QMX_TILE_ROWS; lr++) {
                if (r_base + lr >= orig_rows) break;
                int32_t dot = dot_tile_row_32(tile_ptr + (size_t)lr * row_bytes, xq_ptr, qtype);
                float comb_scale = is_row_scaled ? (qmx_fp16_to_fp32(row_sc_ptr[lr]) * x_sc) : tile_comb_scale;
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

const int8_t qmx_lut_ternary_s8[256][4] QMX_ALIGNED(32) = {
    { 0, 0, 0, 0 },
    { 1, 0, 0, 0 },
    { -1, 0, 0, 0 },
    { 0, 0, 0, 0 },
    { 0, 1, 0, 0 },
    { 1, 1, 0, 0 },
    { -1, 1, 0, 0 },
    { 0, 1, 0, 0 },
    { 0, -1, 0, 0 },
    { 1, -1, 0, 0 },
    { -1, -1, 0, 0 },
    { 0, -1, 0, 0 },
    { 0, 0, 0, 0 },
    { 1, 0, 0, 0 },
    { -1, 0, 0, 0 },
    { 0, 0, 0, 0 },
    { 0, 0, 1, 0 },
    { 1, 0, 1, 0 },
    { -1, 0, 1, 0 },
    { 0, 0, 1, 0 },
    { 0, 1, 1, 0 },
    { 1, 1, 1, 0 },
    { -1, 1, 1, 0 },
    { 0, 1, 1, 0 },
    { 0, -1, 1, 0 },
    { 1, -1, 1, 0 },
    { -1, -1, 1, 0 },
    { 0, -1, 1, 0 },
    { 0, 0, 1, 0 },
    { 1, 0, 1, 0 },
    { -1, 0, 1, 0 },
    { 0, 0, 1, 0 },
    { 0, 0, -1, 0 },
    { 1, 0, -1, 0 },
    { -1, 0, -1, 0 },
    { 0, 0, -1, 0 },
    { 0, 1, -1, 0 },
    { 1, 1, -1, 0 },
    { -1, 1, -1, 0 },
    { 0, 1, -1, 0 },
    { 0, -1, -1, 0 },
    { 1, -1, -1, 0 },
    { -1, -1, -1, 0 },
    { 0, -1, -1, 0 },
    { 0, 0, -1, 0 },
    { 1, 0, -1, 0 },
    { -1, 0, -1, 0 },
    { 0, 0, -1, 0 },
    { 0, 0, 0, 0 },
    { 1, 0, 0, 0 },
    { -1, 0, 0, 0 },
    { 0, 0, 0, 0 },
    { 0, 1, 0, 0 },
    { 1, 1, 0, 0 },
    { -1, 1, 0, 0 },
    { 0, 1, 0, 0 },
    { 0, -1, 0, 0 },
    { 1, -1, 0, 0 },
    { -1, -1, 0, 0 },
    { 0, -1, 0, 0 },
    { 0, 0, 0, 0 },
    { 1, 0, 0, 0 },
    { -1, 0, 0, 0 },
    { 0, 0, 0, 0 },
    { 0, 0, 0, 1 },
    { 1, 0, 0, 1 },
    { -1, 0, 0, 1 },
    { 0, 0, 0, 1 },
    { 0, 1, 0, 1 },
    { 1, 1, 0, 1 },
    { -1, 1, 0, 1 },
    { 0, 1, 0, 1 },
    { 0, -1, 0, 1 },
    { 1, -1, 0, 1 },
    { -1, -1, 0, 1 },
    { 0, -1, 0, 1 },
    { 0, 0, 0, 1 },
    { 1, 0, 0, 1 },
    { -1, 0, 0, 1 },
    { 0, 0, 0, 1 },
    { 0, 0, 1, 1 },
    { 1, 0, 1, 1 },
    { -1, 0, 1, 1 },
    { 0, 0, 1, 1 },
    { 0, 1, 1, 1 },
    { 1, 1, 1, 1 },
    { -1, 1, 1, 1 },
    { 0, 1, 1, 1 },
    { 0, -1, 1, 1 },
    { 1, -1, 1, 1 },
    { -1, -1, 1, 1 },
    { 0, -1, 1, 1 },
    { 0, 0, 1, 1 },
    { 1, 0, 1, 1 },
    { -1, 0, 1, 1 },
    { 0, 0, 1, 1 },
    { 0, 0, -1, 1 },
    { 1, 0, -1, 1 },
    { -1, 0, -1, 1 },
    { 0, 0, -1, 1 },
    { 0, 1, -1, 1 },
    { 1, 1, -1, 1 },
    { -1, 1, -1, 1 },
    { 0, 1, -1, 1 },
    { 0, -1, -1, 1 },
    { 1, -1, -1, 1 },
    { -1, -1, -1, 1 },
    { 0, -1, -1, 1 },
    { 0, 0, -1, 1 },
    { 1, 0, -1, 1 },
    { -1, 0, -1, 1 },
    { 0, 0, -1, 1 },
    { 0, 0, 0, 1 },
    { 1, 0, 0, 1 },
    { -1, 0, 0, 1 },
    { 0, 0, 0, 1 },
    { 0, 1, 0, 1 },
    { 1, 1, 0, 1 },
    { -1, 1, 0, 1 },
    { 0, 1, 0, 1 },
    { 0, -1, 0, 1 },
    { 1, -1, 0, 1 },
    { -1, -1, 0, 1 },
    { 0, -1, 0, 1 },
    { 0, 0, 0, 1 },
    { 1, 0, 0, 1 },
    { -1, 0, 0, 1 },
    { 0, 0, 0, 1 },
    { 0, 0, 0, -1 },
    { 1, 0, 0, -1 },
    { -1, 0, 0, -1 },
    { 0, 0, 0, -1 },
    { 0, 1, 0, -1 },
    { 1, 1, 0, -1 },
    { -1, 1, 0, -1 },
    { 0, 1, 0, -1 },
    { 0, -1, 0, -1 },
    { 1, -1, 0, -1 },
    { -1, -1, 0, -1 },
    { 0, -1, 0, -1 },
    { 0, 0, 0, -1 },
    { 1, 0, 0, -1 },
    { -1, 0, 0, -1 },
    { 0, 0, 0, -1 },
    { 0, 0, 1, -1 },
    { 1, 0, 1, -1 },
    { -1, 0, 1, -1 },
    { 0, 0, 1, -1 },
    { 0, 1, 1, -1 },
    { 1, 1, 1, -1 },
    { -1, 1, 1, -1 },
    { 0, 1, 1, -1 },
    { 0, -1, 1, -1 },
    { 1, -1, 1, -1 },
    { -1, -1, 1, -1 },
    { 0, -1, 1, -1 },
    { 0, 0, 1, -1 },
    { 1, 0, 1, -1 },
    { -1, 0, 1, -1 },
    { 0, 0, 1, -1 },
    { 0, 0, -1, -1 },
    { 1, 0, -1, -1 },
    { -1, 0, -1, -1 },
    { 0, 0, -1, -1 },
    { 0, 1, -1, -1 },
    { 1, 1, -1, -1 },
    { -1, 1, -1, -1 },
    { 0, 1, -1, -1 },
    { 0, -1, -1, -1 },
    { 1, -1, -1, -1 },
    { -1, -1, -1, -1 },
    { 0, -1, -1, -1 },
    { 0, 0, -1, -1 },
    { 1, 0, -1, -1 },
    { -1, 0, -1, -1 },
    { 0, 0, -1, -1 },
    { 0, 0, 0, -1 },
    { 1, 0, 0, -1 },
    { -1, 0, 0, -1 },
    { 0, 0, 0, -1 },
    { 0, 1, 0, -1 },
    { 1, 1, 0, -1 },
    { -1, 1, 0, -1 },
    { 0, 1, 0, -1 },
    { 0, -1, 0, -1 },
    { 1, -1, 0, -1 },
    { -1, -1, 0, -1 },
    { 0, -1, 0, -1 },
    { 0, 0, 0, -1 },
    { 1, 0, 0, -1 },
    { -1, 0, 0, -1 },
    { 0, 0, 0, -1 },
    { 0, 0, 0, 0 },
    { 1, 0, 0, 0 },
    { -1, 0, 0, 0 },
    { 0, 0, 0, 0 },
    { 0, 1, 0, 0 },
    { 1, 1, 0, 0 },
    { -1, 1, 0, 0 },
    { 0, 1, 0, 0 },
    { 0, -1, 0, 0 },
    { 1, -1, 0, 0 },
    { -1, -1, 0, 0 },
    { 0, -1, 0, 0 },
    { 0, 0, 0, 0 },
    { 1, 0, 0, 0 },
    { -1, 0, 0, 0 },
    { 0, 0, 0, 0 },
    { 0, 0, 1, 0 },
    { 1, 0, 1, 0 },
    { -1, 0, 1, 0 },
    { 0, 0, 1, 0 },
    { 0, 1, 1, 0 },
    { 1, 1, 1, 0 },
    { -1, 1, 1, 0 },
    { 0, 1, 1, 0 },
    { 0, -1, 1, 0 },
    { 1, -1, 1, 0 },
    { -1, -1, 1, 0 },
    { 0, -1, 1, 0 },
    { 0, 0, 1, 0 },
    { 1, 0, 1, 0 },
    { -1, 0, 1, 0 },
    { 0, 0, 1, 0 },
    { 0, 0, -1, 0 },
    { 1, 0, -1, 0 },
    { -1, 0, -1, 0 },
    { 0, 0, -1, 0 },
    { 0, 1, -1, 0 },
    { 1, 1, -1, 0 },
    { -1, 1, -1, 0 },
    { 0, 1, -1, 0 },
    { 0, -1, -1, 0 },
    { 1, -1, -1, 0 },
    { -1, -1, -1, 0 },
    { 0, -1, -1, 0 },
    { 0, 0, -1, 0 },
    { 1, 0, -1, 0 },
    { -1, 0, -1, 0 },
    { 0, 0, -1, 0 },
    { 0, 0, 0, 0 },
    { 1, 0, 0, 0 },
    { -1, 0, 0, 0 },
    { 0, 0, 0, 0 },
    { 0, 1, 0, 0 },
    { 1, 1, 0, 0 },
    { -1, 1, 0, 0 },
    { 0, 1, 0, 0 },
    { 0, -1, 0, 0 },
    { 1, -1, 0, 0 },
    { -1, -1, 0, 0 },
    { 0, -1, 0, 0 },
    { 0, 0, 0, 0 },
    { 1, 0, 0, 0 },
    { -1, 0, 0, 0 },
    { 0, 0, 0, 0 }
};

const int8_t qmx_lut_binary_s8[16][4] QMX_ALIGNED(32) = {
    { -1, -1, -1, -1 },
    { 1, -1, -1, -1 },
    { -1, 1, -1, -1 },
    { 1, 1, -1, -1 },
    { -1, -1, 1, -1 },
    { 1, -1, 1, -1 },
    { -1, 1, 1, -1 },
    { 1, 1, 1, -1 },
    { -1, -1, -1, 1 },
    { 1, -1, -1, 1 },
    { -1, 1, -1, 1 },
    { 1, 1, -1, 1 },
    { -1, -1, 1, 1 },
    { 1, -1, 1, 1 },
    { -1, 1, 1, 1 },
    { 1, 1, 1, 1 }
};

const uint32_t qmx_lut_ternary_u32[256][2] QMX_ALIGNED(32) = {
    { 0x00000000u, 0x00000000u },
    { 0x00000001u, 0x00000000u },
    { 0x0000FFFFu, 0x00000000u },
    { 0x00000000u, 0x00000000u },
    { 0x00010000u, 0x00000000u },
    { 0x00010001u, 0x00000000u },
    { 0x0001FFFFu, 0x00000000u },
    { 0x00010000u, 0x00000000u },
    { 0xFFFF0000u, 0x00000000u },
    { 0xFFFF0001u, 0x00000000u },
    { 0xFFFFFFFFu, 0x00000000u },
    { 0xFFFF0000u, 0x00000000u },
    { 0x00000000u, 0x00000000u },
    { 0x00000001u, 0x00000000u },
    { 0x0000FFFFu, 0x00000000u },
    { 0x00000000u, 0x00000000u },
    { 0x00000000u, 0x00000001u },
    { 0x00000001u, 0x00000001u },
    { 0x0000FFFFu, 0x00000001u },
    { 0x00000000u, 0x00000001u },
    { 0x00010000u, 0x00000001u },
    { 0x00010001u, 0x00000001u },
    { 0x0001FFFFu, 0x00000001u },
    { 0x00010000u, 0x00000001u },
    { 0xFFFF0000u, 0x00000001u },
    { 0xFFFF0001u, 0x00000001u },
    { 0xFFFFFFFFu, 0x00000001u },
    { 0xFFFF0000u, 0x00000001u },
    { 0x00000000u, 0x00000001u },
    { 0x00000001u, 0x00000001u },
    { 0x0000FFFFu, 0x00000001u },
    { 0x00000000u, 0x00000001u },
    { 0x00000000u, 0x0000FFFFu },
    { 0x00000001u, 0x0000FFFFu },
    { 0x0000FFFFu, 0x0000FFFFu },
    { 0x00000000u, 0x0000FFFFu },
    { 0x00010000u, 0x0000FFFFu },
    { 0x00010001u, 0x0000FFFFu },
    { 0x0001FFFFu, 0x0000FFFFu },
    { 0x00010000u, 0x0000FFFFu },
    { 0xFFFF0000u, 0x0000FFFFu },
    { 0xFFFF0001u, 0x0000FFFFu },
    { 0xFFFFFFFFu, 0x0000FFFFu },
    { 0xFFFF0000u, 0x0000FFFFu },
    { 0x00000000u, 0x0000FFFFu },
    { 0x00000001u, 0x0000FFFFu },
    { 0x0000FFFFu, 0x0000FFFFu },
    { 0x00000000u, 0x0000FFFFu },
    { 0x00000000u, 0x00000000u },
    { 0x00000001u, 0x00000000u },
    { 0x0000FFFFu, 0x00000000u },
    { 0x00000000u, 0x00000000u },
    { 0x00010000u, 0x00000000u },
    { 0x00010001u, 0x00000000u },
    { 0x0001FFFFu, 0x00000000u },
    { 0x00010000u, 0x00000000u },
    { 0xFFFF0000u, 0x00000000u },
    { 0xFFFF0001u, 0x00000000u },
    { 0xFFFFFFFFu, 0x00000000u },
    { 0xFFFF0000u, 0x00000000u },
    { 0x00000000u, 0x00000000u },
    { 0x00000001u, 0x00000000u },
    { 0x0000FFFFu, 0x00000000u },
    { 0x00000000u, 0x00000000u },
    { 0x00000000u, 0x00010000u },
    { 0x00000001u, 0x00010000u },
    { 0x0000FFFFu, 0x00010000u },
    { 0x00000000u, 0x00010000u },
    { 0x00010000u, 0x00010000u },
    { 0x00010001u, 0x00010000u },
    { 0x0001FFFFu, 0x00010000u },
    { 0x00010000u, 0x00010000u },
    { 0xFFFF0000u, 0x00010000u },
    { 0xFFFF0001u, 0x00010000u },
    { 0xFFFFFFFFu, 0x00010000u },
    { 0xFFFF0000u, 0x00010000u },
    { 0x00000000u, 0x00010000u },
    { 0x00000001u, 0x00010000u },
    { 0x0000FFFFu, 0x00010000u },
    { 0x00000000u, 0x00010000u },
    { 0x00000000u, 0x00010001u },
    { 0x00000001u, 0x00010001u },
    { 0x0000FFFFu, 0x00010001u },
    { 0x00000000u, 0x00010001u },
    { 0x00010000u, 0x00010001u },
    { 0x00010001u, 0x00010001u },
    { 0x0001FFFFu, 0x00010001u },
    { 0x00010000u, 0x00010001u },
    { 0xFFFF0000u, 0x00010001u },
    { 0xFFFF0001u, 0x00010001u },
    { 0xFFFFFFFFu, 0x00010001u },
    { 0xFFFF0000u, 0x00010001u },
    { 0x00000000u, 0x00010001u },
    { 0x00000001u, 0x00010001u },
    { 0x0000FFFFu, 0x00010001u },
    { 0x00000000u, 0x00010001u },
    { 0x00000000u, 0x0001FFFFu },
    { 0x00000001u, 0x0001FFFFu },
    { 0x0000FFFFu, 0x0001FFFFu },
    { 0x00000000u, 0x0001FFFFu },
    { 0x00010000u, 0x0001FFFFu },
    { 0x00010001u, 0x0001FFFFu },
    { 0x0001FFFFu, 0x0001FFFFu },
    { 0x00010000u, 0x0001FFFFu },
    { 0xFFFF0000u, 0x0001FFFFu },
    { 0xFFFF0001u, 0x0001FFFFu },
    { 0xFFFFFFFFu, 0x0001FFFFu },
    { 0xFFFF0000u, 0x0001FFFFu },
    { 0x00000000u, 0x0001FFFFu },
    { 0x00000001u, 0x0001FFFFu },
    { 0x0000FFFFu, 0x0001FFFFu },
    { 0x00000000u, 0x0001FFFFu },
    { 0x00000000u, 0x00010000u },
    { 0x00000001u, 0x00010000u },
    { 0x0000FFFFu, 0x00010000u },
    { 0x00000000u, 0x00010000u },
    { 0x00010000u, 0x00010000u },
    { 0x00010001u, 0x00010000u },
    { 0x0001FFFFu, 0x00010000u },
    { 0x00010000u, 0x00010000u },
    { 0xFFFF0000u, 0x00010000u },
    { 0xFFFF0001u, 0x00010000u },
    { 0xFFFFFFFFu, 0x00010000u },
    { 0xFFFF0000u, 0x00010000u },
    { 0x00000000u, 0x00010000u },
    { 0x00000001u, 0x00010000u },
    { 0x0000FFFFu, 0x00010000u },
    { 0x00000000u, 0x00010000u },
    { 0x00000000u, 0xFFFF0000u },
    { 0x00000001u, 0xFFFF0000u },
    { 0x0000FFFFu, 0xFFFF0000u },
    { 0x00000000u, 0xFFFF0000u },
    { 0x00010000u, 0xFFFF0000u },
    { 0x00010001u, 0xFFFF0000u },
    { 0x0001FFFFu, 0xFFFF0000u },
    { 0x00010000u, 0xFFFF0000u },
    { 0xFFFF0000u, 0xFFFF0000u },
    { 0xFFFF0001u, 0xFFFF0000u },
    { 0xFFFFFFFFu, 0xFFFF0000u },
    { 0xFFFF0000u, 0xFFFF0000u },
    { 0x00000000u, 0xFFFF0000u },
    { 0x00000001u, 0xFFFF0000u },
    { 0x0000FFFFu, 0xFFFF0000u },
    { 0x00000000u, 0xFFFF0000u },
    { 0x00000000u, 0xFFFF0001u },
    { 0x00000001u, 0xFFFF0001u },
    { 0x0000FFFFu, 0xFFFF0001u },
    { 0x00000000u, 0xFFFF0001u },
    { 0x00010000u, 0xFFFF0001u },
    { 0x00010001u, 0xFFFF0001u },
    { 0x0001FFFFu, 0xFFFF0001u },
    { 0x00010000u, 0xFFFF0001u },
    { 0xFFFF0000u, 0xFFFF0001u },
    { 0xFFFF0001u, 0xFFFF0001u },
    { 0xFFFFFFFFu, 0xFFFF0001u },
    { 0xFFFF0000u, 0xFFFF0001u },
    { 0x00000000u, 0xFFFF0001u },
    { 0x00000001u, 0xFFFF0001u },
    { 0x0000FFFFu, 0xFFFF0001u },
    { 0x00000000u, 0xFFFF0001u },
    { 0x00000000u, 0xFFFFFFFFu },
    { 0x00000001u, 0xFFFFFFFFu },
    { 0x0000FFFFu, 0xFFFFFFFFu },
    { 0x00000000u, 0xFFFFFFFFu },
    { 0x00010000u, 0xFFFFFFFFu },
    { 0x00010001u, 0xFFFFFFFFu },
    { 0x0001FFFFu, 0xFFFFFFFFu },
    { 0x00010000u, 0xFFFFFFFFu },
    { 0xFFFF0000u, 0xFFFFFFFFu },
    { 0xFFFF0001u, 0xFFFFFFFFu },
    { 0xFFFFFFFFu, 0xFFFFFFFFu },
    { 0xFFFF0000u, 0xFFFFFFFFu },
    { 0x00000000u, 0xFFFFFFFFu },
    { 0x00000001u, 0xFFFFFFFFu },
    { 0x0000FFFFu, 0xFFFFFFFFu },
    { 0x00000000u, 0xFFFFFFFFu },
    { 0x00000000u, 0xFFFF0000u },
    { 0x00000001u, 0xFFFF0000u },
    { 0x0000FFFFu, 0xFFFF0000u },
    { 0x00000000u, 0xFFFF0000u },
    { 0x00010000u, 0xFFFF0000u },
    { 0x00010001u, 0xFFFF0000u },
    { 0x0001FFFFu, 0xFFFF0000u },
    { 0x00010000u, 0xFFFF0000u },
    { 0xFFFF0000u, 0xFFFF0000u },
    { 0xFFFF0001u, 0xFFFF0000u },
    { 0xFFFFFFFFu, 0xFFFF0000u },
    { 0xFFFF0000u, 0xFFFF0000u },
    { 0x00000000u, 0xFFFF0000u },
    { 0x00000001u, 0xFFFF0000u },
    { 0x0000FFFFu, 0xFFFF0000u },
    { 0x00000000u, 0xFFFF0000u },
    { 0x00000000u, 0x00000000u },
    { 0x00000001u, 0x00000000u },
    { 0x0000FFFFu, 0x00000000u },
    { 0x00000000u, 0x00000000u },
    { 0x00010000u, 0x00000000u },
    { 0x00010001u, 0x00000000u },
    { 0x0001FFFFu, 0x00000000u },
    { 0x00010000u, 0x00000000u },
    { 0xFFFF0000u, 0x00000000u },
    { 0xFFFF0001u, 0x00000000u },
    { 0xFFFFFFFFu, 0x00000000u },
    { 0xFFFF0000u, 0x00000000u },
    { 0x00000000u, 0x00000000u },
    { 0x00000001u, 0x00000000u },
    { 0x0000FFFFu, 0x00000000u },
    { 0x00000000u, 0x00000000u },
    { 0x00000000u, 0x00000001u },
    { 0x00000001u, 0x00000001u },
    { 0x0000FFFFu, 0x00000001u },
    { 0x00000000u, 0x00000001u },
    { 0x00010000u, 0x00000001u },
    { 0x00010001u, 0x00000001u },
    { 0x0001FFFFu, 0x00000001u },
    { 0x00010000u, 0x00000001u },
    { 0xFFFF0000u, 0x00000001u },
    { 0xFFFF0001u, 0x00000001u },
    { 0xFFFFFFFFu, 0x00000001u },
    { 0xFFFF0000u, 0x00000001u },
    { 0x00000000u, 0x00000001u },
    { 0x00000001u, 0x00000001u },
    { 0x0000FFFFu, 0x00000001u },
    { 0x00000000u, 0x00000001u },
    { 0x00000000u, 0x0000FFFFu },
    { 0x00000001u, 0x0000FFFFu },
    { 0x0000FFFFu, 0x0000FFFFu },
    { 0x00000000u, 0x0000FFFFu },
    { 0x00010000u, 0x0000FFFFu },
    { 0x00010001u, 0x0000FFFFu },
    { 0x0001FFFFu, 0x0000FFFFu },
    { 0x00010000u, 0x0000FFFFu },
    { 0xFFFF0000u, 0x0000FFFFu },
    { 0xFFFF0001u, 0x0000FFFFu },
    { 0xFFFFFFFFu, 0x0000FFFFu },
    { 0xFFFF0000u, 0x0000FFFFu },
    { 0x00000000u, 0x0000FFFFu },
    { 0x00000001u, 0x0000FFFFu },
    { 0x0000FFFFu, 0x0000FFFFu },
    { 0x00000000u, 0x0000FFFFu },
    { 0x00000000u, 0x00000000u },
    { 0x00000001u, 0x00000000u },
    { 0x0000FFFFu, 0x00000000u },
    { 0x00000000u, 0x00000000u },
    { 0x00010000u, 0x00000000u },
    { 0x00010001u, 0x00000000u },
    { 0x0001FFFFu, 0x00000000u },
    { 0x00010000u, 0x00000000u },
    { 0xFFFF0000u, 0x00000000u },
    { 0xFFFF0001u, 0x00000000u },
    { 0xFFFFFFFFu, 0x00000000u },
    { 0xFFFF0000u, 0x00000000u },
    { 0x00000000u, 0x00000000u },
    { 0x00000001u, 0x00000000u },
    { 0x0000FFFFu, 0x00000000u },
    { 0x00000000u, 0x00000000u }
};

const uint32_t qmx_lut_binary_u32[16][2] QMX_ALIGNED(32) = {
    { 0xFFFFFFFFu, 0xFFFFFFFFu },
    { 0xFFFF0001u, 0xFFFFFFFFu },
    { 0x0001FFFFu, 0xFFFFFFFFu },
    { 0x00010001u, 0xFFFFFFFFu },
    { 0xFFFFFFFFu, 0xFFFF0001u },
    { 0xFFFF0001u, 0xFFFF0001u },
    { 0x0001FFFFu, 0xFFFF0001u },
    { 0x00010001u, 0xFFFF0001u },
    { 0xFFFFFFFFu, 0x0001FFFFu },
    { 0xFFFF0001u, 0x0001FFFFu },
    { 0x0001FFFFu, 0x0001FFFFu },
    { 0x00010001u, 0x0001FFFFu },
    { 0xFFFFFFFFu, 0x00010001u },
    { 0xFFFF0001u, 0x00010001u },
    { 0x0001FFFFu, 0x00010001u },
    { 0x00010001u, 0x00010001u }
};

#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
const int16_t qmx_lut_ternary_i16[256][4] QMX_ALIGNED(64) = {
    { 0, 0, 0, 0 },
    { 1, 0, 0, 0 },
    { -1, 0, 0, 0 },
    { 0, 0, 0, 0 },
    { 0, 1, 0, 0 },
    { 1, 1, 0, 0 },
    { -1, 1, 0, 0 },
    { 0, 1, 0, 0 },
    { 0, -1, 0, 0 },
    { 1, -1, 0, 0 },
    { -1, -1, 0, 0 },
    { 0, -1, 0, 0 },
    { 0, 0, 0, 0 },
    { 1, 0, 0, 0 },
    { -1, 0, 0, 0 },
    { 0, 0, 0, 0 },
    { 0, 0, 1, 0 },
    { 1, 0, 1, 0 },
    { -1, 0, 1, 0 },
    { 0, 0, 1, 0 },
    { 0, 1, 1, 0 },
    { 1, 1, 1, 0 },
    { -1, 1, 1, 0 },
    { 0, 1, 1, 0 },
    { 0, -1, 1, 0 },
    { 1, -1, 1, 0 },
    { -1, -1, 1, 0 },
    { 0, -1, 1, 0 },
    { 0, 0, 1, 0 },
    { 1, 0, 1, 0 },
    { -1, 0, 1, 0 },
    { 0, 0, 1, 0 },
    { 0, 0, -1, 0 },
    { 1, 0, -1, 0 },
    { -1, 0, -1, 0 },
    { 0, 0, -1, 0 },
    { 0, 1, -1, 0 },
    { 1, 1, -1, 0 },
    { -1, 1, -1, 0 },
    { 0, 1, -1, 0 },
    { 0, -1, -1, 0 },
    { 1, -1, -1, 0 },
    { -1, -1, -1, 0 },
    { 0, -1, -1, 0 },
    { 0, 0, -1, 0 },
    { 1, 0, -1, 0 },
    { -1, 0, -1, 0 },
    { 0, 0, -1, 0 },
    { 0, 0, 0, 0 },
    { 1, 0, 0, 0 },
    { -1, 0, 0, 0 },
    { 0, 0, 0, 0 },
    { 0, 1, 0, 0 },
    { 1, 1, 0, 0 },
    { -1, 1, 0, 0 },
    { 0, 1, 0, 0 },
    { 0, -1, 0, 0 },
    { 1, -1, 0, 0 },
    { -1, -1, 0, 0 },
    { 0, -1, 0, 0 },
    { 0, 0, 0, 0 },
    { 1, 0, 0, 0 },
    { -1, 0, 0, 0 },
    { 0, 0, 0, 0 },
    { 0, 0, 0, 1 },
    { 1, 0, 0, 1 },
    { -1, 0, 0, 1 },
    { 0, 0, 0, 1 },
    { 0, 1, 0, 1 },
    { 1, 1, 0, 1 },
    { -1, 1, 0, 1 },
    { 0, 1, 0, 1 },
    { 0, -1, 0, 1 },
    { 1, -1, 0, 1 },
    { -1, -1, 0, 1 },
    { 0, -1, 0, 1 },
    { 0, 0, 0, 1 },
    { 1, 0, 0, 1 },
    { -1, 0, 0, 1 },
    { 0, 0, 0, 1 },
    { 0, 0, 1, 1 },
    { 1, 0, 1, 1 },
    { -1, 0, 1, 1 },
    { 0, 0, 1, 1 },
    { 0, 1, 1, 1 },
    { 1, 1, 1, 1 },
    { -1, 1, 1, 1 },
    { 0, 1, 1, 1 },
    { 0, -1, 1, 1 },
    { 1, -1, 1, 1 },
    { -1, -1, 1, 1 },
    { 0, -1, 1, 1 },
    { 0, 0, 1, 1 },
    { 1, 0, 1, 1 },
    { -1, 0, 1, 1 },
    { 0, 0, 1, 1 },
    { 0, 0, -1, 1 },
    { 1, 0, -1, 1 },
    { -1, 0, -1, 1 },
    { 0, 0, -1, 1 },
    { 0, 1, -1, 1 },
    { 1, 1, -1, 1 },
    { -1, 1, -1, 1 },
    { 0, 1, -1, 1 },
    { 0, -1, -1, 1 },
    { 1, -1, -1, 1 },
    { -1, -1, -1, 1 },
    { 0, -1, -1, 1 },
    { 0, 0, -1, 1 },
    { 1, 0, -1, 1 },
    { -1, 0, -1, 1 },
    { 0, 0, -1, 1 },
    { 0, 0, 0, 1 },
    { 1, 0, 0, 1 },
    { -1, 0, 0, 1 },
    { 0, 0, 0, 1 },
    { 0, 1, 0, 1 },
    { 1, 1, 0, 1 },
    { -1, 1, 0, 1 },
    { 0, 1, 0, 1 },
    { 0, -1, 0, 1 },
    { 1, -1, 0, 1 },
    { -1, -1, 0, 1 },
    { 0, -1, 0, 1 },
    { 0, 0, 0, 1 },
    { 1, 0, 0, 1 },
    { -1, 0, 0, 1 },
    { 0, 0, 0, 1 },
    { 0, 0, 0, -1 },
    { 1, 0, 0, -1 },
    { -1, 0, 0, -1 },
    { 0, 0, 0, -1 },
    { 0, 1, 0, -1 },
    { 1, 1, 0, -1 },
    { -1, 1, 0, -1 },
    { 0, 1, 0, -1 },
    { 0, -1, 0, -1 },
    { 1, -1, 0, -1 },
    { -1, -1, 0, -1 },
    { 0, -1, 0, -1 },
    { 0, 0, 0, -1 },
    { 1, 0, 0, -1 },
    { -1, 0, 0, -1 },
    { 0, 0, 0, -1 },
    { 0, 0, 1, -1 },
    { 1, 0, 1, -1 },
    { -1, 0, 1, -1 },
    { 0, 0, 1, -1 },
    { 0, 1, 1, -1 },
    { 1, 1, 1, -1 },
    { -1, 1, 1, -1 },
    { 0, 1, 1, -1 },
    { 0, -1, 1, -1 },
    { 1, -1, 1, -1 },
    { -1, -1, 1, -1 },
    { 0, -1, 1, -1 },
    { 0, 0, 1, -1 },
    { 1, 0, 1, -1 },
    { -1, 0, 1, -1 },
    { 0, 0, 1, -1 },
    { 0, 0, -1, -1 },
    { 1, 0, -1, -1 },
    { -1, 0, -1, -1 },
    { 0, 0, -1, -1 },
    { 0, 1, -1, -1 },
    { 1, 1, -1, -1 },
    { -1, 1, -1, -1 },
    { 0, 1, -1, -1 },
    { 0, -1, -1, -1 },
    { 1, -1, -1, -1 },
    { -1, -1, -1, -1 },
    { 0, -1, -1, -1 },
    { 0, 0, -1, -1 },
    { 1, 0, -1, -1 },
    { -1, 0, -1, -1 },
    { 0, 0, -1, -1 },
    { 0, 0, 0, -1 },
    { 1, 0, 0, -1 },
    { -1, 0, 0, -1 },
    { 0, 0, 0, -1 },
    { 0, 1, 0, -1 },
    { 1, 1, 0, -1 },
    { -1, 1, 0, -1 },
    { 0, 1, 0, -1 },
    { 0, -1, 0, -1 },
    { 1, -1, 0, -1 },
    { -1, -1, 0, -1 },
    { 0, -1, 0, -1 },
    { 0, 0, 0, -1 },
    { 1, 0, 0, -1 },
    { -1, 0, 0, -1 },
    { 0, 0, 0, -1 },
    { 0, 0, 0, 0 },
    { 1, 0, 0, 0 },
    { -1, 0, 0, 0 },
    { 0, 0, 0, 0 },
    { 0, 1, 0, 0 },
    { 1, 1, 0, 0 },
    { -1, 1, 0, 0 },
    { 0, 1, 0, 0 },
    { 0, -1, 0, 0 },
    { 1, -1, 0, 0 },
    { -1, -1, 0, 0 },
    { 0, -1, 0, 0 },
    { 0, 0, 0, 0 },
    { 1, 0, 0, 0 },
    { -1, 0, 0, 0 },
    { 0, 0, 0, 0 },
    { 0, 0, 1, 0 },
    { 1, 0, 1, 0 },
    { -1, 0, 1, 0 },
    { 0, 0, 1, 0 },
    { 0, 1, 1, 0 },
    { 1, 1, 1, 0 },
    { -1, 1, 1, 0 },
    { 0, 1, 1, 0 },
    { 0, -1, 1, 0 },
    { 1, -1, 1, 0 },
    { -1, -1, 1, 0 },
    { 0, -1, 1, 0 },
    { 0, 0, 1, 0 },
    { 1, 0, 1, 0 },
    { -1, 0, 1, 0 },
    { 0, 0, 1, 0 },
    { 0, 0, -1, 0 },
    { 1, 0, -1, 0 },
    { -1, 0, -1, 0 },
    { 0, 0, -1, 0 },
    { 0, 1, -1, 0 },
    { 1, 1, -1, 0 },
    { -1, 1, -1, 0 },
    { 0, 1, -1, 0 },
    { 0, -1, -1, 0 },
    { 1, -1, -1, 0 },
    { -1, -1, -1, 0 },
    { 0, -1, -1, 0 },
    { 0, 0, -1, 0 },
    { 1, 0, -1, 0 },
    { -1, 0, -1, 0 },
    { 0, 0, -1, 0 },
    { 0, 0, 0, 0 },
    { 1, 0, 0, 0 },
    { -1, 0, 0, 0 },
    { 0, 0, 0, 0 },
    { 0, 1, 0, 0 },
    { 1, 1, 0, 0 },
    { -1, 1, 0, 0 },
    { 0, 1, 0, 0 },
    { 0, -1, 0, 0 },
    { 1, -1, 0, 0 },
    { -1, -1, 0, 0 },
    { 0, -1, 0, 0 },
    { 0, 0, 0, 0 },
    { 1, 0, 0, 0 },
    { -1, 0, 0, 0 },
    { 0, 0, 0, 0 }
};

const int16_t qmx_lut_binary_i16_4[16][4] QMX_ALIGNED(64) = {
    { -1, -1, -1, -1 },
    { 1, -1, -1, -1 },
    { -1, 1, -1, -1 },
    { 1, 1, -1, -1 },
    { -1, -1, 1, -1 },
    { 1, -1, 1, -1 },
    { -1, 1, 1, -1 },
    { 1, 1, 1, -1 },
    { -1, -1, -1, 1 },
    { 1, -1, -1, 1 },
    { -1, 1, -1, 1 },
    { 1, 1, -1, 1 },
    { -1, -1, 1, 1 },
    { 1, -1, 1, 1 },
    { -1, 1, 1, 1 },
    { 1, 1, 1, 1 }
};

const int16_t qmx_lut_binary_i16_8[256][8] QMX_ALIGNED(64) = {
    { -1, -1, -1, -1, -1, -1, -1, -1 },
    { 1, -1, -1, -1, -1, -1, -1, -1 },
    { -1, 1, -1, -1, -1, -1, -1, -1 },
    { 1, 1, -1, -1, -1, -1, -1, -1 },
    { -1, -1, 1, -1, -1, -1, -1, -1 },
    { 1, -1, 1, -1, -1, -1, -1, -1 },
    { -1, 1, 1, -1, -1, -1, -1, -1 },
    { 1, 1, 1, -1, -1, -1, -1, -1 },
    { -1, -1, -1, 1, -1, -1, -1, -1 },
    { 1, -1, -1, 1, -1, -1, -1, -1 },
    { -1, 1, -1, 1, -1, -1, -1, -1 },
    { 1, 1, -1, 1, -1, -1, -1, -1 },
    { -1, -1, 1, 1, -1, -1, -1, -1 },
    { 1, -1, 1, 1, -1, -1, -1, -1 },
    { -1, 1, 1, 1, -1, -1, -1, -1 },
    { 1, 1, 1, 1, -1, -1, -1, -1 },
    { -1, -1, -1, -1, 1, -1, -1, -1 },
    { 1, -1, -1, -1, 1, -1, -1, -1 },
    { -1, 1, -1, -1, 1, -1, -1, -1 },
    { 1, 1, -1, -1, 1, -1, -1, -1 },
    { -1, -1, 1, -1, 1, -1, -1, -1 },
    { 1, -1, 1, -1, 1, -1, -1, -1 },
    { -1, 1, 1, -1, 1, -1, -1, -1 },
    { 1, 1, 1, -1, 1, -1, -1, -1 },
    { -1, -1, -1, 1, 1, -1, -1, -1 },
    { 1, -1, -1, 1, 1, -1, -1, -1 },
    { -1, 1, -1, 1, 1, -1, -1, -1 },
    { 1, 1, -1, 1, 1, -1, -1, -1 },
    { -1, -1, 1, 1, 1, -1, -1, -1 },
    { 1, -1, 1, 1, 1, -1, -1, -1 },
    { -1, 1, 1, 1, 1, -1, -1, -1 },
    { 1, 1, 1, 1, 1, -1, -1, -1 },
    { -1, -1, -1, -1, -1, 1, -1, -1 },
    { 1, -1, -1, -1, -1, 1, -1, -1 },
    { -1, 1, -1, -1, -1, 1, -1, -1 },
    { 1, 1, -1, -1, -1, 1, -1, -1 },
    { -1, -1, 1, -1, -1, 1, -1, -1 },
    { 1, -1, 1, -1, -1, 1, -1, -1 },
    { -1, 1, 1, -1, -1, 1, -1, -1 },
    { 1, 1, 1, -1, -1, 1, -1, -1 },
    { -1, -1, -1, 1, -1, 1, -1, -1 },
    { 1, -1, -1, 1, -1, 1, -1, -1 },
    { -1, 1, -1, 1, -1, 1, -1, -1 },
    { 1, 1, -1, 1, -1, 1, -1, -1 },
    { -1, -1, 1, 1, -1, 1, -1, -1 },
    { 1, -1, 1, 1, -1, 1, -1, -1 },
    { -1, 1, 1, 1, -1, 1, -1, -1 },
    { 1, 1, 1, 1, -1, 1, -1, -1 },
    { -1, -1, -1, -1, 1, 1, -1, -1 },
    { 1, -1, -1, -1, 1, 1, -1, -1 },
    { -1, 1, -1, -1, 1, 1, -1, -1 },
    { 1, 1, -1, -1, 1, 1, -1, -1 },
    { -1, -1, 1, -1, 1, 1, -1, -1 },
    { 1, -1, 1, -1, 1, 1, -1, -1 },
    { -1, 1, 1, -1, 1, 1, -1, -1 },
    { 1, 1, 1, -1, 1, 1, -1, -1 },
    { -1, -1, -1, 1, 1, 1, -1, -1 },
    { 1, -1, -1, 1, 1, 1, -1, -1 },
    { -1, 1, -1, 1, 1, 1, -1, -1 },
    { 1, 1, -1, 1, 1, 1, -1, -1 },
    { -1, -1, 1, 1, 1, 1, -1, -1 },
    { 1, -1, 1, 1, 1, 1, -1, -1 },
    { -1, 1, 1, 1, 1, 1, -1, -1 },
    { 1, 1, 1, 1, 1, 1, -1, -1 },
    { -1, -1, -1, -1, -1, -1, 1, -1 },
    { 1, -1, -1, -1, -1, -1, 1, -1 },
    { -1, 1, -1, -1, -1, -1, 1, -1 },
    { 1, 1, -1, -1, -1, -1, 1, -1 },
    { -1, -1, 1, -1, -1, -1, 1, -1 },
    { 1, -1, 1, -1, -1, -1, 1, -1 },
    { -1, 1, 1, -1, -1, -1, 1, -1 },
    { 1, 1, 1, -1, -1, -1, 1, -1 },
    { -1, -1, -1, 1, -1, -1, 1, -1 },
    { 1, -1, -1, 1, -1, -1, 1, -1 },
    { -1, 1, -1, 1, -1, -1, 1, -1 },
    { 1, 1, -1, 1, -1, -1, 1, -1 },
    { -1, -1, 1, 1, -1, -1, 1, -1 },
    { 1, -1, 1, 1, -1, -1, 1, -1 },
    { -1, 1, 1, 1, -1, -1, 1, -1 },
    { 1, 1, 1, 1, -1, -1, 1, -1 },
    { -1, -1, -1, -1, 1, -1, 1, -1 },
    { 1, -1, -1, -1, 1, -1, 1, -1 },
    { -1, 1, -1, -1, 1, -1, 1, -1 },
    { 1, 1, -1, -1, 1, -1, 1, -1 },
    { -1, -1, 1, -1, 1, -1, 1, -1 },
    { 1, -1, 1, -1, 1, -1, 1, -1 },
    { -1, 1, 1, -1, 1, -1, 1, -1 },
    { 1, 1, 1, -1, 1, -1, 1, -1 },
    { -1, -1, -1, 1, 1, -1, 1, -1 },
    { 1, -1, -1, 1, 1, -1, 1, -1 },
    { -1, 1, -1, 1, 1, -1, 1, -1 },
    { 1, 1, -1, 1, 1, -1, 1, -1 },
    { -1, -1, 1, 1, 1, -1, 1, -1 },
    { 1, -1, 1, 1, 1, -1, 1, -1 },
    { -1, 1, 1, 1, 1, -1, 1, -1 },
    { 1, 1, 1, 1, 1, -1, 1, -1 },
    { -1, -1, -1, -1, -1, 1, 1, -1 },
    { 1, -1, -1, -1, -1, 1, 1, -1 },
    { -1, 1, -1, -1, -1, 1, 1, -1 },
    { 1, 1, -1, -1, -1, 1, 1, -1 },
    { -1, -1, 1, -1, -1, 1, 1, -1 },
    { 1, -1, 1, -1, -1, 1, 1, -1 },
    { -1, 1, 1, -1, -1, 1, 1, -1 },
    { 1, 1, 1, -1, -1, 1, 1, -1 },
    { -1, -1, -1, 1, -1, 1, 1, -1 },
    { 1, -1, -1, 1, -1, 1, 1, -1 },
    { -1, 1, -1, 1, -1, 1, 1, -1 },
    { 1, 1, -1, 1, -1, 1, 1, -1 },
    { -1, -1, 1, 1, -1, 1, 1, -1 },
    { 1, -1, 1, 1, -1, 1, 1, -1 },
    { -1, 1, 1, 1, -1, 1, 1, -1 },
    { 1, 1, 1, 1, -1, 1, 1, -1 },
    { -1, -1, -1, -1, 1, 1, 1, -1 },
    { 1, -1, -1, -1, 1, 1, 1, -1 },
    { -1, 1, -1, -1, 1, 1, 1, -1 },
    { 1, 1, -1, -1, 1, 1, 1, -1 },
    { -1, -1, 1, -1, 1, 1, 1, -1 },
    { 1, -1, 1, -1, 1, 1, 1, -1 },
    { -1, 1, 1, -1, 1, 1, 1, -1 },
    { 1, 1, 1, -1, 1, 1, 1, -1 },
    { -1, -1, -1, 1, 1, 1, 1, -1 },
    { 1, -1, -1, 1, 1, 1, 1, -1 },
    { -1, 1, -1, 1, 1, 1, 1, -1 },
    { 1, 1, -1, 1, 1, 1, 1, -1 },
    { -1, -1, 1, 1, 1, 1, 1, -1 },
    { 1, -1, 1, 1, 1, 1, 1, -1 },
    { -1, 1, 1, 1, 1, 1, 1, -1 },
    { 1, 1, 1, 1, 1, 1, 1, -1 },
    { -1, -1, -1, -1, -1, -1, -1, 1 },
    { 1, -1, -1, -1, -1, -1, -1, 1 },
    { -1, 1, -1, -1, -1, -1, -1, 1 },
    { 1, 1, -1, -1, -1, -1, -1, 1 },
    { -1, -1, 1, -1, -1, -1, -1, 1 },
    { 1, -1, 1, -1, -1, -1, -1, 1 },
    { -1, 1, 1, -1, -1, -1, -1, 1 },
    { 1, 1, 1, -1, -1, -1, -1, 1 },
    { -1, -1, -1, 1, -1, -1, -1, 1 },
    { 1, -1, -1, 1, -1, -1, -1, 1 },
    { -1, 1, -1, 1, -1, -1, -1, 1 },
    { 1, 1, -1, 1, -1, -1, -1, 1 },
    { -1, -1, 1, 1, -1, -1, -1, 1 },
    { 1, -1, 1, 1, -1, -1, -1, 1 },
    { -1, 1, 1, 1, -1, -1, -1, 1 },
    { 1, 1, 1, 1, -1, -1, -1, 1 },
    { -1, -1, -1, -1, 1, -1, -1, 1 },
    { 1, -1, -1, -1, 1, -1, -1, 1 },
    { -1, 1, -1, -1, 1, -1, -1, 1 },
    { 1, 1, -1, -1, 1, -1, -1, 1 },
    { -1, -1, 1, -1, 1, -1, -1, 1 },
    { 1, -1, 1, -1, 1, -1, -1, 1 },
    { -1, 1, 1, -1, 1, -1, -1, 1 },
    { 1, 1, 1, -1, 1, -1, -1, 1 },
    { -1, -1, -1, 1, 1, -1, -1, 1 },
    { 1, -1, -1, 1, 1, -1, -1, 1 },
    { -1, 1, -1, 1, 1, -1, -1, 1 },
    { 1, 1, -1, 1, 1, -1, -1, 1 },
    { -1, -1, 1, 1, 1, -1, -1, 1 },
    { 1, -1, 1, 1, 1, -1, -1, 1 },
    { -1, 1, 1, 1, 1, -1, -1, 1 },
    { 1, 1, 1, 1, 1, -1, -1, 1 },
    { -1, -1, -1, -1, -1, 1, -1, 1 },
    { 1, -1, -1, -1, -1, 1, -1, 1 },
    { -1, 1, -1, -1, -1, 1, -1, 1 },
    { 1, 1, -1, -1, -1, 1, -1, 1 },
    { -1, -1, 1, -1, -1, 1, -1, 1 },
    { 1, -1, 1, -1, -1, 1, -1, 1 },
    { -1, 1, 1, -1, -1, 1, -1, 1 },
    { 1, 1, 1, -1, -1, 1, -1, 1 },
    { -1, -1, -1, 1, -1, 1, -1, 1 },
    { 1, -1, -1, 1, -1, 1, -1, 1 },
    { -1, 1, -1, 1, -1, 1, -1, 1 },
    { 1, 1, -1, 1, -1, 1, -1, 1 },
    { -1, -1, 1, 1, -1, 1, -1, 1 },
    { 1, -1, 1, 1, -1, 1, -1, 1 },
    { -1, 1, 1, 1, -1, 1, -1, 1 },
    { 1, 1, 1, 1, -1, 1, -1, 1 },
    { -1, -1, -1, -1, 1, 1, -1, 1 },
    { 1, -1, -1, -1, 1, 1, -1, 1 },
    { -1, 1, -1, -1, 1, 1, -1, 1 },
    { 1, 1, -1, -1, 1, 1, -1, 1 },
    { -1, -1, 1, -1, 1, 1, -1, 1 },
    { 1, -1, 1, -1, 1, 1, -1, 1 },
    { -1, 1, 1, -1, 1, 1, -1, 1 },
    { 1, 1, 1, -1, 1, 1, -1, 1 },
    { -1, -1, -1, 1, 1, 1, -1, 1 },
    { 1, -1, -1, 1, 1, 1, -1, 1 },
    { -1, 1, -1, 1, 1, 1, -1, 1 },
    { 1, 1, -1, 1, 1, 1, -1, 1 },
    { -1, -1, 1, 1, 1, 1, -1, 1 },
    { 1, -1, 1, 1, 1, 1, -1, 1 },
    { -1, 1, 1, 1, 1, 1, -1, 1 },
    { 1, 1, 1, 1, 1, 1, -1, 1 },
    { -1, -1, -1, -1, -1, -1, 1, 1 },
    { 1, -1, -1, -1, -1, -1, 1, 1 },
    { -1, 1, -1, -1, -1, -1, 1, 1 },
    { 1, 1, -1, -1, -1, -1, 1, 1 },
    { -1, -1, 1, -1, -1, -1, 1, 1 },
    { 1, -1, 1, -1, -1, -1, 1, 1 },
    { -1, 1, 1, -1, -1, -1, 1, 1 },
    { 1, 1, 1, -1, -1, -1, 1, 1 },
    { -1, -1, -1, 1, -1, -1, 1, 1 },
    { 1, -1, -1, 1, -1, -1, 1, 1 },
    { -1, 1, -1, 1, -1, -1, 1, 1 },
    { 1, 1, -1, 1, -1, -1, 1, 1 },
    { -1, -1, 1, 1, -1, -1, 1, 1 },
    { 1, -1, 1, 1, -1, -1, 1, 1 },
    { -1, 1, 1, 1, -1, -1, 1, 1 },
    { 1, 1, 1, 1, -1, -1, 1, 1 },
    { -1, -1, -1, -1, 1, -1, 1, 1 },
    { 1, -1, -1, -1, 1, -1, 1, 1 },
    { -1, 1, -1, -1, 1, -1, 1, 1 },
    { 1, 1, -1, -1, 1, -1, 1, 1 },
    { -1, -1, 1, -1, 1, -1, 1, 1 },
    { 1, -1, 1, -1, 1, -1, 1, 1 },
    { -1, 1, 1, -1, 1, -1, 1, 1 },
    { 1, 1, 1, -1, 1, -1, 1, 1 },
    { -1, -1, -1, 1, 1, -1, 1, 1 },
    { 1, -1, -1, 1, 1, -1, 1, 1 },
    { -1, 1, -1, 1, 1, -1, 1, 1 },
    { 1, 1, -1, 1, 1, -1, 1, 1 },
    { -1, -1, 1, 1, 1, -1, 1, 1 },
    { 1, -1, 1, 1, 1, -1, 1, 1 },
    { -1, 1, 1, 1, 1, -1, 1, 1 },
    { 1, 1, 1, 1, 1, -1, 1, 1 },
    { -1, -1, -1, -1, -1, 1, 1, 1 },
    { 1, -1, -1, -1, -1, 1, 1, 1 },
    { -1, 1, -1, -1, -1, 1, 1, 1 },
    { 1, 1, -1, -1, -1, 1, 1, 1 },
    { -1, -1, 1, -1, -1, 1, 1, 1 },
    { 1, -1, 1, -1, -1, 1, 1, 1 },
    { -1, 1, 1, -1, -1, 1, 1, 1 },
    { 1, 1, 1, -1, -1, 1, 1, 1 },
    { -1, -1, -1, 1, -1, 1, 1, 1 },
    { 1, -1, -1, 1, -1, 1, 1, 1 },
    { -1, 1, -1, 1, -1, 1, 1, 1 },
    { 1, 1, -1, 1, -1, 1, 1, 1 },
    { -1, -1, 1, 1, -1, 1, 1, 1 },
    { 1, -1, 1, 1, -1, 1, 1, 1 },
    { -1, 1, 1, 1, -1, 1, 1, 1 },
    { 1, 1, 1, 1, -1, 1, 1, 1 },
    { -1, -1, -1, -1, 1, 1, 1, 1 },
    { 1, -1, -1, -1, 1, 1, 1, 1 },
    { -1, 1, -1, -1, 1, 1, 1, 1 },
    { 1, 1, -1, -1, 1, 1, 1, 1 },
    { -1, -1, 1, -1, 1, 1, 1, 1 },
    { 1, -1, 1, -1, 1, 1, 1, 1 },
    { -1, 1, 1, -1, 1, 1, 1, 1 },
    { 1, 1, 1, -1, 1, 1, 1, 1 },
    { -1, -1, -1, 1, 1, 1, 1, 1 },
    { 1, -1, -1, 1, 1, 1, 1, 1 },
    { -1, 1, -1, 1, 1, 1, 1, 1 },
    { 1, 1, -1, 1, 1, 1, 1, 1 },
    { -1, -1, 1, 1, 1, 1, 1, 1 },
    { 1, -1, 1, 1, 1, 1, 1, 1 },
    { -1, 1, 1, 1, 1, 1, 1, 1 },
    { 1, 1, 1, 1, 1, 1, 1, 1 }
};
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__)
const int8_t qmx_lut_binary_s8_8[256][8] QMX_ALIGNED(64) = {
    { -1, -1, -1, -1, -1, -1, -1, -1 },
    { 1, -1, -1, -1, -1, -1, -1, -1 },
    { -1, 1, -1, -1, -1, -1, -1, -1 },
    { 1, 1, -1, -1, -1, -1, -1, -1 },
    { -1, -1, 1, -1, -1, -1, -1, -1 },
    { 1, -1, 1, -1, -1, -1, -1, -1 },
    { -1, 1, 1, -1, -1, -1, -1, -1 },
    { 1, 1, 1, -1, -1, -1, -1, -1 },
    { -1, -1, -1, 1, -1, -1, -1, -1 },
    { 1, -1, -1, 1, -1, -1, -1, -1 },
    { -1, 1, -1, 1, -1, -1, -1, -1 },
    { 1, 1, -1, 1, -1, -1, -1, -1 },
    { -1, -1, 1, 1, -1, -1, -1, -1 },
    { 1, -1, 1, 1, -1, -1, -1, -1 },
    { -1, 1, 1, 1, -1, -1, -1, -1 },
    { 1, 1, 1, 1, -1, -1, -1, -1 },
    { -1, -1, -1, -1, 1, -1, -1, -1 },
    { 1, -1, -1, -1, 1, -1, -1, -1 },
    { -1, 1, -1, -1, 1, -1, -1, -1 },
    { 1, 1, -1, -1, 1, -1, -1, -1 },
    { -1, -1, 1, -1, 1, -1, -1, -1 },
    { 1, -1, 1, -1, 1, -1, -1, -1 },
    { -1, 1, 1, -1, 1, -1, -1, -1 },
    { 1, 1, 1, -1, 1, -1, -1, -1 },
    { -1, -1, -1, 1, 1, -1, -1, -1 },
    { 1, -1, -1, 1, 1, -1, -1, -1 },
    { -1, 1, -1, 1, 1, -1, -1, -1 },
    { 1, 1, -1, 1, 1, -1, -1, -1 },
    { -1, -1, 1, 1, 1, -1, -1, -1 },
    { 1, -1, 1, 1, 1, -1, -1, -1 },
    { -1, 1, 1, 1, 1, -1, -1, -1 },
    { 1, 1, 1, 1, 1, -1, -1, -1 },
    { -1, -1, -1, -1, -1, 1, -1, -1 },
    { 1, -1, -1, -1, -1, 1, -1, -1 },
    { -1, 1, -1, -1, -1, 1, -1, -1 },
    { 1, 1, -1, -1, -1, 1, -1, -1 },
    { -1, -1, 1, -1, -1, 1, -1, -1 },
    { 1, -1, 1, -1, -1, 1, -1, -1 },
    { -1, 1, 1, -1, -1, 1, -1, -1 },
    { 1, 1, 1, -1, -1, 1, -1, -1 },
    { -1, -1, -1, 1, -1, 1, -1, -1 },
    { 1, -1, -1, 1, -1, 1, -1, -1 },
    { -1, 1, -1, 1, -1, 1, -1, -1 },
    { 1, 1, -1, 1, -1, 1, -1, -1 },
    { -1, -1, 1, 1, -1, 1, -1, -1 },
    { 1, -1, 1, 1, -1, 1, -1, -1 },
    { -1, 1, 1, 1, -1, 1, -1, -1 },
    { 1, 1, 1, 1, -1, 1, -1, -1 },
    { -1, -1, -1, -1, 1, 1, -1, -1 },
    { 1, -1, -1, -1, 1, 1, -1, -1 },
    { -1, 1, -1, -1, 1, 1, -1, -1 },
    { 1, 1, -1, -1, 1, 1, -1, -1 },
    { -1, -1, 1, -1, 1, 1, -1, -1 },
    { 1, -1, 1, -1, 1, 1, -1, -1 },
    { -1, 1, 1, -1, 1, 1, -1, -1 },
    { 1, 1, 1, -1, 1, 1, -1, -1 },
    { -1, -1, -1, 1, 1, 1, -1, -1 },
    { 1, -1, -1, 1, 1, 1, -1, -1 },
    { -1, 1, -1, 1, 1, 1, -1, -1 },
    { 1, 1, -1, 1, 1, 1, -1, -1 },
    { -1, -1, 1, 1, 1, 1, -1, -1 },
    { 1, -1, 1, 1, 1, 1, -1, -1 },
    { -1, 1, 1, 1, 1, 1, -1, -1 },
    { 1, 1, 1, 1, 1, 1, -1, -1 },
    { -1, -1, -1, -1, -1, -1, 1, -1 },
    { 1, -1, -1, -1, -1, -1, 1, -1 },
    { -1, 1, -1, -1, -1, -1, 1, -1 },
    { 1, 1, -1, -1, -1, -1, 1, -1 },
    { -1, -1, 1, -1, -1, -1, 1, -1 },
    { 1, -1, 1, -1, -1, -1, 1, -1 },
    { -1, 1, 1, -1, -1, -1, 1, -1 },
    { 1, 1, 1, -1, -1, -1, 1, -1 },
    { -1, -1, -1, 1, -1, -1, 1, -1 },
    { 1, -1, -1, 1, -1, -1, 1, -1 },
    { -1, 1, -1, 1, -1, -1, 1, -1 },
    { 1, 1, -1, 1, -1, -1, 1, -1 },
    { -1, -1, 1, 1, -1, -1, 1, -1 },
    { 1, -1, 1, 1, -1, -1, 1, -1 },
    { -1, 1, 1, 1, -1, -1, 1, -1 },
    { 1, 1, 1, 1, -1, -1, 1, -1 },
    { -1, -1, -1, -1, 1, -1, 1, -1 },
    { 1, -1, -1, -1, 1, -1, 1, -1 },
    { -1, 1, -1, -1, 1, -1, 1, -1 },
    { 1, 1, -1, -1, 1, -1, 1, -1 },
    { -1, -1, 1, -1, 1, -1, 1, -1 },
    { 1, -1, 1, -1, 1, -1, 1, -1 },
    { -1, 1, 1, -1, 1, -1, 1, -1 },
    { 1, 1, 1, -1, 1, -1, 1, -1 },
    { -1, -1, -1, 1, 1, -1, 1, -1 },
    { 1, -1, -1, 1, 1, -1, 1, -1 },
    { -1, 1, -1, 1, 1, -1, 1, -1 },
    { 1, 1, -1, 1, 1, -1, 1, -1 },
    { -1, -1, 1, 1, 1, -1, 1, -1 },
    { 1, -1, 1, 1, 1, -1, 1, -1 },
    { -1, 1, 1, 1, 1, -1, 1, -1 },
    { 1, 1, 1, 1, 1, -1, 1, -1 },
    { -1, -1, -1, -1, -1, 1, 1, -1 },
    { 1, -1, -1, -1, -1, 1, 1, -1 },
    { -1, 1, -1, -1, -1, 1, 1, -1 },
    { 1, 1, -1, -1, -1, 1, 1, -1 },
    { -1, -1, 1, -1, -1, 1, 1, -1 },
    { 1, -1, 1, -1, -1, 1, 1, -1 },
    { -1, 1, 1, -1, -1, 1, 1, -1 },
    { 1, 1, 1, -1, -1, 1, 1, -1 },
    { -1, -1, -1, 1, -1, 1, 1, -1 },
    { 1, -1, -1, 1, -1, 1, 1, -1 },
    { -1, 1, -1, 1, -1, 1, 1, -1 },
    { 1, 1, -1, 1, -1, 1, 1, -1 },
    { -1, -1, 1, 1, -1, 1, 1, -1 },
    { 1, -1, 1, 1, -1, 1, 1, -1 },
    { -1, 1, 1, 1, -1, 1, 1, -1 },
    { 1, 1, 1, 1, -1, 1, 1, -1 },
    { -1, -1, -1, -1, 1, 1, 1, -1 },
    { 1, -1, -1, -1, 1, 1, 1, -1 },
    { -1, 1, -1, -1, 1, 1, 1, -1 },
    { 1, 1, -1, -1, 1, 1, 1, -1 },
    { -1, -1, 1, -1, 1, 1, 1, -1 },
    { 1, -1, 1, -1, 1, 1, 1, -1 },
    { -1, 1, 1, -1, 1, 1, 1, -1 },
    { 1, 1, 1, -1, 1, 1, 1, -1 },
    { -1, -1, -1, 1, 1, 1, 1, -1 },
    { 1, -1, -1, 1, 1, 1, 1, -1 },
    { -1, 1, -1, 1, 1, 1, 1, -1 },
    { 1, 1, -1, 1, 1, 1, 1, -1 },
    { -1, -1, 1, 1, 1, 1, 1, -1 },
    { 1, -1, 1, 1, 1, 1, 1, -1 },
    { -1, 1, 1, 1, 1, 1, 1, -1 },
    { 1, 1, 1, 1, 1, 1, 1, -1 },
    { -1, -1, -1, -1, -1, -1, -1, 1 },
    { 1, -1, -1, -1, -1, -1, -1, 1 },
    { -1, 1, -1, -1, -1, -1, -1, 1 },
    { 1, 1, -1, -1, -1, -1, -1, 1 },
    { -1, -1, 1, -1, -1, -1, -1, 1 },
    { 1, -1, 1, -1, -1, -1, -1, 1 },
    { -1, 1, 1, -1, -1, -1, -1, 1 },
    { 1, 1, 1, -1, -1, -1, -1, 1 },
    { -1, -1, -1, 1, -1, -1, -1, 1 },
    { 1, -1, -1, 1, -1, -1, -1, 1 },
    { -1, 1, -1, 1, -1, -1, -1, 1 },
    { 1, 1, -1, 1, -1, -1, -1, 1 },
    { -1, -1, 1, 1, -1, -1, -1, 1 },
    { 1, -1, 1, 1, -1, -1, -1, 1 },
    { -1, 1, 1, 1, -1, -1, -1, 1 },
    { 1, 1, 1, 1, -1, -1, -1, 1 },
    { -1, -1, -1, -1, 1, -1, -1, 1 },
    { 1, -1, -1, -1, 1, -1, -1, 1 },
    { -1, 1, -1, -1, 1, -1, -1, 1 },
    { 1, 1, -1, -1, 1, -1, -1, 1 },
    { -1, -1, 1, -1, 1, -1, -1, 1 },
    { 1, -1, 1, -1, 1, -1, -1, 1 },
    { -1, 1, 1, -1, 1, -1, -1, 1 },
    { 1, 1, 1, -1, 1, -1, -1, 1 },
    { -1, -1, -1, 1, 1, -1, -1, 1 },
    { 1, -1, -1, 1, 1, -1, -1, 1 },
    { -1, 1, -1, 1, 1, -1, -1, 1 },
    { 1, 1, -1, 1, 1, -1, -1, 1 },
    { -1, -1, 1, 1, 1, -1, -1, 1 },
    { 1, -1, 1, 1, 1, -1, -1, 1 },
    { -1, 1, 1, 1, 1, -1, -1, 1 },
    { 1, 1, 1, 1, 1, -1, -1, 1 },
    { -1, -1, -1, -1, -1, 1, -1, 1 },
    { 1, -1, -1, -1, -1, 1, -1, 1 },
    { -1, 1, -1, -1, -1, 1, -1, 1 },
    { 1, 1, -1, -1, -1, 1, -1, 1 },
    { -1, -1, 1, -1, -1, 1, -1, 1 },
    { 1, -1, 1, -1, -1, 1, -1, 1 },
    { -1, 1, 1, -1, -1, 1, -1, 1 },
    { 1, 1, 1, -1, -1, 1, -1, 1 },
    { -1, -1, -1, 1, -1, 1, -1, 1 },
    { 1, -1, -1, 1, -1, 1, -1, 1 },
    { -1, 1, -1, 1, -1, 1, -1, 1 },
    { 1, 1, -1, 1, -1, 1, -1, 1 },
    { -1, -1, 1, 1, -1, 1, -1, 1 },
    { 1, -1, 1, 1, -1, 1, -1, 1 },
    { -1, 1, 1, 1, -1, 1, -1, 1 },
    { 1, 1, 1, 1, -1, 1, -1, 1 },
    { -1, -1, -1, -1, 1, 1, -1, 1 },
    { 1, -1, -1, -1, 1, 1, -1, 1 },
    { -1, 1, -1, -1, 1, 1, -1, 1 },
    { 1, 1, -1, -1, 1, 1, -1, 1 },
    { -1, -1, 1, -1, 1, 1, -1, 1 },
    { 1, -1, 1, -1, 1, 1, -1, 1 },
    { -1, 1, 1, -1, 1, 1, -1, 1 },
    { 1, 1, 1, -1, 1, 1, -1, 1 },
    { -1, -1, -1, 1, 1, 1, -1, 1 },
    { 1, -1, -1, 1, 1, 1, -1, 1 },
    { -1, 1, -1, 1, 1, 1, -1, 1 },
    { 1, 1, -1, 1, 1, 1, -1, 1 },
    { -1, -1, 1, 1, 1, 1, -1, 1 },
    { 1, -1, 1, 1, 1, 1, -1, 1 },
    { -1, 1, 1, 1, 1, 1, -1, 1 },
    { 1, 1, 1, 1, 1, 1, -1, 1 },
    { -1, -1, -1, -1, -1, -1, 1, 1 },
    { 1, -1, -1, -1, -1, -1, 1, 1 },
    { -1, 1, -1, -1, -1, -1, 1, 1 },
    { 1, 1, -1, -1, -1, -1, 1, 1 },
    { -1, -1, 1, -1, -1, -1, 1, 1 },
    { 1, -1, 1, -1, -1, -1, 1, 1 },
    { -1, 1, 1, -1, -1, -1, 1, 1 },
    { 1, 1, 1, -1, -1, -1, 1, 1 },
    { -1, -1, -1, 1, -1, -1, 1, 1 },
    { 1, -1, -1, 1, -1, -1, 1, 1 },
    { -1, 1, -1, 1, -1, -1, 1, 1 },
    { 1, 1, -1, 1, -1, -1, 1, 1 },
    { -1, -1, 1, 1, -1, -1, 1, 1 },
    { 1, -1, 1, 1, -1, -1, 1, 1 },
    { -1, 1, 1, 1, -1, -1, 1, 1 },
    { 1, 1, 1, 1, -1, -1, 1, 1 },
    { -1, -1, -1, -1, 1, -1, 1, 1 },
    { 1, -1, -1, -1, 1, -1, 1, 1 },
    { -1, 1, -1, -1, 1, -1, 1, 1 },
    { 1, 1, -1, -1, 1, -1, 1, 1 },
    { -1, -1, 1, -1, 1, -1, 1, 1 },
    { 1, -1, 1, -1, 1, -1, 1, 1 },
    { -1, 1, 1, -1, 1, -1, 1, 1 },
    { 1, 1, 1, -1, 1, -1, 1, 1 },
    { -1, -1, -1, 1, 1, -1, 1, 1 },
    { 1, -1, -1, 1, 1, -1, 1, 1 },
    { -1, 1, -1, 1, 1, -1, 1, 1 },
    { 1, 1, -1, 1, 1, -1, 1, 1 },
    { -1, -1, 1, 1, 1, -1, 1, 1 },
    { 1, -1, 1, 1, 1, -1, 1, 1 },
    { -1, 1, 1, 1, 1, -1, 1, 1 },
    { 1, 1, 1, 1, 1, -1, 1, 1 },
    { -1, -1, -1, -1, -1, 1, 1, 1 },
    { 1, -1, -1, -1, -1, 1, 1, 1 },
    { -1, 1, -1, -1, -1, 1, 1, 1 },
    { 1, 1, -1, -1, -1, 1, 1, 1 },
    { -1, -1, 1, -1, -1, 1, 1, 1 },
    { 1, -1, 1, -1, -1, 1, 1, 1 },
    { -1, 1, 1, -1, -1, 1, 1, 1 },
    { 1, 1, 1, -1, -1, 1, 1, 1 },
    { -1, -1, -1, 1, -1, 1, 1, 1 },
    { 1, -1, -1, 1, -1, 1, 1, 1 },
    { -1, 1, -1, 1, -1, 1, 1, 1 },
    { 1, 1, -1, 1, -1, 1, 1, 1 },
    { -1, -1, 1, 1, -1, 1, 1, 1 },
    { 1, -1, 1, 1, -1, 1, 1, 1 },
    { -1, 1, 1, 1, -1, 1, 1, 1 },
    { 1, 1, 1, 1, -1, 1, 1, 1 },
    { -1, -1, -1, -1, 1, 1, 1, 1 },
    { 1, -1, -1, -1, 1, 1, 1, 1 },
    { -1, 1, -1, -1, 1, 1, 1, 1 },
    { 1, 1, -1, -1, 1, 1, 1, 1 },
    { -1, -1, 1, -1, 1, 1, 1, 1 },
    { 1, -1, 1, -1, 1, 1, 1, 1 },
    { -1, 1, 1, -1, 1, 1, 1, 1 },
    { 1, 1, 1, -1, 1, 1, 1, 1 },
    { -1, -1, -1, 1, 1, 1, 1, 1 },
    { 1, -1, -1, 1, 1, 1, 1, 1 },
    { -1, 1, -1, 1, 1, 1, 1, 1 },
    { 1, 1, -1, 1, 1, 1, 1, 1 },
    { -1, -1, 1, 1, 1, 1, 1, 1 },
    { 1, -1, 1, 1, 1, 1, 1, 1 },
    { -1, 1, 1, 1, 1, 1, 1, 1 },
    { 1, 1, 1, 1, 1, 1, 1, 1 }
};
#endif
