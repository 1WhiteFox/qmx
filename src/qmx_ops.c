/*
 * Copyright 2026 1WhiteFox
 * SPDX-License-Identifier: Apache-2.0
 */

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
#if defined(_WIN64) || (defined(_WIN32) && !defined(QMX_WIN98_COMPAT))
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
#if defined(_WIN64) || (defined(_WIN32) && !defined(QMX_WIN98_COMPAT))
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

#ifndef QMX_NO_MALLOC
static const uint32_t s_qmx_crc32_tab[256] = {
    0x00000000u, 0x77073096u, 0xEE0E612Cu, 0x990951BAu, 0x076DC419u, 0x706AF48Fu, 0xE963A535u, 0x9E6495A3u,
    0x0EDB8832u, 0x79DCB8A4u, 0xE0D5E91Eu, 0x97D2D988u, 0x09B64C2Bu, 0x7EB17CBDu, 0xE7B82D07u, 0x90BF1D91u,
    0x1DB71064u, 0x6AB020F2u, 0xF3B97148u, 0x84BE41DEu, 0x1ADAD47Du, 0x6DDDE4EBu, 0xF4D4B551u, 0x83D385C7u,
    0x136C9856u, 0x646BA8C0u, 0xFD62F97Au, 0x8A65C9ECu, 0x14015C4Fu, 0x63066CD9u, 0xFA0F3D63u, 0x8D080DF5u,
    0x3B6E20C8u, 0x4C69105Eu, 0xD56041E4u, 0xA2677172u, 0x3C03E4D1u, 0x4B04D447u, 0xD20D85FDu, 0xA50AB56Bu,
    0x35B5A8FAu, 0x42B2986Cu, 0xDBBBC9D6u, 0xACBCF940u, 0x32D86CE3u, 0x45DF5C75u, 0xDCD60DCFu, 0xABD13D59u,
    0x26D930ACu, 0x51DE003Au, 0xC8D75180u, 0xBFD06116u, 0x21B4F4B5u, 0x56B3C423u, 0xCFBA9599u, 0xB8BDA50Fu,
    0x2802B89Eu, 0x5F058808u, 0xC60CD9B2u, 0xB10BE924u, 0x2F6F7C87u, 0x58684C11u, 0xC1611DABu, 0xB6662D3Du,
    0x76DC4190u, 0x01DB7106u, 0x98D220BCu, 0xEFD5102Au, 0x71B18589u, 0x06B6B51Fu, 0x9FBFE4A5u, 0xE8B8D433u,
    0x7807C9A2u, 0x0F00F934u, 0x9609A88Eu, 0xE10E9818u, 0x7F6A0DBBu, 0x086D3D2Du, 0x91646C97u, 0xE6635C01u,
    0x6B6B51F4u, 0x1C6C6162u, 0x856530D8u, 0xF262004Eu, 0x6C0695EDu, 0x1B01A57Bu, 0x8208F4C1u, 0xF50FC457u,
    0x65B0D9C6u, 0x12B7E950u, 0x8BBEB8EAu, 0xFCB9887Cu, 0x62DD1DDFu, 0x15DA2D49u, 0x8CD37CF3u, 0xFBD44C65u,
    0x4DB26158u, 0x3AB551CEu, 0xA3BC0074u, 0xD4BB30E2u, 0x4ADFA541u, 0x3DD895D7u, 0xA4D1C46Du, 0xD3D6F4FBu,
    0x4369E96Au, 0x346ED9FCu, 0xAD678846u, 0xDA60B8D0u, 0x44042D73u, 0x33031DE5u, 0xAA0A4C5Fu, 0xDD0D7CC9u,
    0x5005713Cu, 0x270241AAu, 0xBE0B1010u, 0xC90C2086u, 0x5768B525u, 0x206F85B3u, 0xB966D409u, 0xCE61E49Fu,
    0x5EDEF90Eu, 0x29D9C998u, 0xB0D09822u, 0xC7D7A8B4u, 0x59B33D17u, 0x2EB40D81u, 0xB7BD5C3Bu, 0xC0BA6CADu,
    0xEDB88320u, 0x9ABFB3B6u, 0x03B6E20Cu, 0x74B1D29Au, 0xEAD54739u, 0x9DD277AFu, 0x04DB2615u, 0x73DC1683u,
    0xE3630B12u, 0x94643B84u, 0x0D6D6A3Eu, 0x7A6A5AA8u, 0xE40ECF0Bu, 0x9309FF9Du, 0x0A00AE27u, 0x7D079EB1u,
    0xF00F9344u, 0x8708A3D2u, 0x1E01F268u, 0x6906C2FEu, 0xF762575Du, 0x806567CBu, 0x196C3671u, 0x6E6B06E7u,
    0xFED41B76u, 0x89D32BE0u, 0x10DA7A5Au, 0x67DD4ACCu, 0xF9B9DF6Fu, 0x8EBEEFF9u, 0x17B7BE43u, 0x60B08ED5u,
    0xD6D6A3E8u, 0xA1D1937Eu, 0x38D8C2C4u, 0x4FDFF252u, 0xD1BB67F1u, 0xA6BC5767u, 0x3FB506DDu, 0x48B2364Bu,
    0xD80D2BDAu, 0xAF0A1B4Cu, 0x36034AF6u, 0x41047A60u, 0xDF60EFC3u, 0xA867DF55u, 0x316E8EEFu, 0x4669BE79u,
    0xCB61B38Cu, 0xBC66831Au, 0x256FD2A0u, 0x5268E236u, 0xCC0C7795u, 0xBB0B4703u, 0x220216B9u, 0x5505262Fu,
    0xC5BA3BBEu, 0xB2BD0B28u, 0x2BB45A92u, 0x5CB36A04u, 0xC2D7FFA7u, 0xB5D0CF31u, 0x2CD99E8Bu, 0x5BDEAE1Du,
    0x9B64C2B0u, 0xEC63F226u, 0x756AA39Cu, 0x026D930Au, 0x9C0906A9u, 0xEB0E363Fu, 0x72076785u, 0x05005713u,
    0x95BF4A82u, 0xE2B87A14u, 0x7BB12BAEu, 0x0CB61B38u, 0x92D28E9Bu, 0xE5D5BE0Du, 0x7CDCEFB7u, 0x0BDBDF21u,
    0x86D3D2D4u, 0xF1D4E242u, 0x68DDB3F8u, 0x1FDA836Eu, 0x81BE16CDu, 0xF6B9265Bu, 0x6FB077E1u, 0x18B74777u,
    0x88085AE6u, 0xFF0F6A70u, 0x66063BCAu, 0x11010B5Cu, 0x8F659EFFu, 0xF862AE69u, 0x616BFFD3u, 0x166CCF45u,
    0xA00AE278u, 0xD70DD2EEu, 0x4E048354u, 0x3903B3C2u, 0xA7672661u, 0xD06016F7u, 0x4969474Du, 0x3E6E77DBu,
    0xAED16A4Au, 0xD9D65ADCu, 0x40DF0B66u, 0x37D83BF0u, 0xA9BCAE53u, 0xDEBB9EC5u, 0x47B2CF7Fu, 0x30B5FFE9u,
    0xBDBDF21Cu, 0xCABAC28Au, 0x53B39330u, 0x24B4A3A6u, 0xBAD03605u, 0xCDD70693u, 0x54DE5729u, 0x23D967BFu,
    0xB3667A2Eu, 0xC4614AB8u, 0x5D681B02u, 0x2A6F2B94u, 0xB40BBE37u, 0xC30C8EA1u, 0x5A05DF1Bu, 0x2D02EF8Du
};
#endif

uint32_t qmx_crc32_update(uint32_t crc, const void *data, size_t len) {
    const uint8_t *p = (const uint8_t *)data;
#ifndef QMX_NO_MALLOC
    for (size_t i = 0; i < len; i++) {
        crc = s_qmx_crc32_tab[(crc ^ (uint32_t)p[i]) & 0xFFu] ^ (crc >> 8u);
    }
#else
    for (size_t i = 0; i < len; i++) {
        crc ^= (uint32_t)p[i];
        for (int b = 0; b < 8; b++) {
            uint32_t mask = 0u - (crc & 1u);
            crc = (crc >> 1u) ^ (0xEDB88320u & mask);
        }
    }
#endif
    return crc;
}

uint32_t qmx_crc32(const void *data, size_t len) {
    return ~qmx_crc32_update(0xFFFFFFFFu, data, len);
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
    if (!arena || capacity > (~(size_t)0) - QMX_CACHE_ALIGN) return -1;
    arena->raw_ptr = (uint8_t *)malloc(capacity + QMX_CACHE_ALIGN);
    if (!arena->raw_ptr) return -1;
    uintptr_t addr = (uintptr_t)arena->raw_ptr;
    uintptr_t aligned = (addr + (QMX_CACHE_ALIGN - 1u)) & ~(uintptr_t)(QMX_CACHE_ALIGN - 1u);
    arena->base_ptr = (uint8_t *)aligned;
    arena->capacity = capacity;
    arena->offset = 0;
    arena->is_external = 0;
    arena->failed = 0;
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
    arena->failed = 0;
    memset(arena->base_ptr, 0, arena->capacity);
    return 0;
}

void *qmx_arena_alloc(QmxArena *arena, size_t size, size_t alignment) {
    if (!arena || arena->failed) return NULL;
    if (alignment == 0u) alignment = 1u;
    uint64_t aligned_off64 = 0u;
    if (qmx_align_up64_checked((uint64_t)arena->offset, (uint64_t)alignment, &aligned_off64) != 0 ||
        aligned_off64 > (uint64_t)arena->capacity ||
        (uint64_t)size > (uint64_t)arena->capacity - aligned_off64) {
        arena->failed = 1;
        return NULL;
    }
    size_t aligned_off = (size_t)aligned_off64;
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
    arena->failed = 0;
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

float qmx_dot_f32_f16_scalar(const float * QMX_RESTRICT a, const uint16_t * QMX_RESTRICT b, uint32_t n) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        sum += a[i] * qmx_fp16_to_fp32(b[i]);
    }
    return sum;
}

void qmx_saxpy_f32_f16_scalar(float * QMX_RESTRICT y, float a, const uint16_t * QMX_RESTRICT x, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        y[i] += a * qmx_fp16_to_fp32(x[i]);
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

        if (0) { (void)0; }
#if QMX_HAS_QUANT(QMX_QID_BINARY_1BIT)
        else if (qtype == QMX_QUANT_BINARY_1BIT) {
            for (uint32_t b = 0; b < 4u; b++) {
                uint8_t p = row_ptr[b];
                for (uint32_t k = 0; k < 8u; k++) {
                    uint32_t c = col_base + b * 8u + k;
                    if (c < cols) out[c] = ((p >> k) & 1u) ? sc : -sc;
                }
            }
        }
#endif
#if QMX_HAS_QUANT(QMX_QID_TERNARY_158)
        else if (qtype == QMX_QUANT_TERNARY_158) {
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
        }
#endif
#if QMX_HAS_QUANT(QMX_QID_Q4_SYM) || QMX_HAS_QUANT(QMX_QID_Q4_ROW32)
        else if (qtype == QMX_QUANT_Q4_SYM || qtype == QMX_QUANT_Q4_ROW32) {
            for (uint32_t b = 0; b < 16u; b++) {
                uint8_t p = row_ptr[b];
                uint32_t c0 = col_base + b * 2u;
                uint32_t c1 = c0 + 1u;
                if (c0 < cols) out[c0] = (float)((int32_t)(p & 0x0Fu) - 8) * sc;
                if (c1 < cols) out[c1] = (float)((int32_t)(p >> 4u) - 8) * sc;
            }
        }
#endif
#if QMX_HAS_QUANT(QMX_QID_Q8_SYM) || QMX_HAS_QUANT(QMX_QID_Q8_ROW32)
        else if (qtype == QMX_QUANT_Q8_SYM || qtype == QMX_QUANT_Q8_ROW32) {
            const int8_t *s_ptr = (const int8_t *)row_ptr;
            for (uint32_t i = 0; i < QMX_TILE_COLS; i++) {
                uint32_t c = col_base + i;
                if (c < cols) out[c] = (float)s_ptr[i] * sc;
            }
        }
#endif

    }
}

static int32_t dot_tile_row_32(const uint8_t *row_ptr, const int8_t *xq_32, uint8_t qtype) {
    int32_t acc = 0;
    if (0) { (void)0; }
#if QMX_HAS_QUANT(QMX_QID_BINARY_1BIT)
    else if (qtype == QMX_QUANT_BINARY_1BIT) {
        for (uint32_t b = 0; b < 4u; b++) {
            uint8_t p = row_ptr[b];
            const int8_t *xp = xq_32 + b * 8u;
            for (uint32_t k = 0; k < 8u; k++) {
                acc += ((p >> k) & 1u) ? (int32_t)xp[k] : -(int32_t)xp[k];
            }
        }
    }
#endif
#if QMX_HAS_QUANT(QMX_QID_TERNARY_158)
    else if (qtype == QMX_QUANT_TERNARY_158) {
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
    }
#endif
#if QMX_HAS_QUANT(QMX_QID_Q4_SYM) || QMX_HAS_QUANT(QMX_QID_Q4_ROW32)
    else if (qtype == QMX_QUANT_Q4_SYM || qtype == QMX_QUANT_Q4_ROW32) {
        for (uint32_t b = 0; b < 16u; b++) {
            uint8_t p = row_ptr[b];
            int32_t w0 = (int32_t)(p & 0x0Fu) - 8;
            int32_t w1 = (int32_t)(p >> 4u) - 8;
            acc += w0 * (int32_t)xq_32[b * 2u] + w1 * (int32_t)xq_32[b * 2u + 1u];
        }
    }
#endif
#if QMX_HAS_QUANT(QMX_QID_Q8_SYM) || QMX_HAS_QUANT(QMX_QID_Q8_ROW32)
    else if (qtype == QMX_QUANT_Q8_SYM || qtype == QMX_QUANT_Q8_ROW32) {
        const int8_t *wp = (const int8_t *)row_ptr;
        for (uint32_t i = 0; i < QMX_TILE_COLS; i++) {
            acc += (int32_t)wp[i] * (int32_t)xq_32[i];
        }
    }
#endif

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

#if QMX_HAS_QUANT(QMX_QID_TERNARY_158) || defined(QMX_ENABLE_BACKEND_P54C) || defined(QMX_ENABLE_BACKEND_MMX) || defined(QMX_ENABLE_BACKEND_3DNOW) || defined(QMX_ENABLE_BACKEND_SSE2) || defined(QMX_ENABLE_BACKEND_AVX2) || defined(QMX_ENABLE_BACKEND_X86_EXT) || defined(QMX_ENABLE_BACKEND_NEON) || defined(QMX_ENABLE_BACKEND_ARM_DSP) || defined(QMX_ENABLE_BACKEND_MCU32)
QMX_ALIGNED(32) const int8_t qmx_lut_ternary_s8[256][4] = {
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

#endif

#if QMX_HAS_QUANT(QMX_QID_BINARY_1BIT) || defined(QMX_ENABLE_BACKEND_P54C) || defined(QMX_ENABLE_BACKEND_MMX) || defined(QMX_ENABLE_BACKEND_3DNOW) || defined(QMX_ENABLE_BACKEND_SSE2) || defined(QMX_ENABLE_BACKEND_AVX2) || defined(QMX_ENABLE_BACKEND_X86_EXT) || defined(QMX_ENABLE_BACKEND_NEON) || defined(QMX_ENABLE_BACKEND_ARM_DSP) || defined(QMX_ENABLE_BACKEND_MCU32) || defined(QMX_ENABLE_BACKEND_CORTEX_M3) || defined(QMX_ENABLE_BACKEND_CORTEX_M0) || defined(QMX_ENABLE_BACKEND_CORTEX_M4)
QMX_ALIGNED(32) const int8_t qmx_lut_binary_s8[16][4] = {
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

#endif

#if QMX_HAS_QUANT(QMX_QID_TERNARY_158) || defined(QMX_ENABLE_BACKEND_P54C) || defined(QMX_ENABLE_BACKEND_MMX) || defined(QMX_ENABLE_BACKEND_3DNOW) || defined(QMX_ENABLE_BACKEND_SSE2) || defined(QMX_ENABLE_BACKEND_AVX2) || defined(QMX_ENABLE_BACKEND_X86_EXT) || defined(QMX_ENABLE_BACKEND_NEON) || defined(QMX_ENABLE_BACKEND_ARM_DSP) || defined(QMX_ENABLE_BACKEND_MCU32)
QMX_ALIGNED(32) const uint32_t qmx_lut_ternary_u32[256][2] = {
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

#endif

#if QMX_HAS_QUANT(QMX_QID_BINARY_1BIT) || defined(QMX_ENABLE_BACKEND_P54C) || defined(QMX_ENABLE_BACKEND_MMX) || defined(QMX_ENABLE_BACKEND_3DNOW) || defined(QMX_ENABLE_BACKEND_SSE2) || defined(QMX_ENABLE_BACKEND_AVX2) || defined(QMX_ENABLE_BACKEND_X86_EXT) || defined(QMX_ENABLE_BACKEND_NEON) || defined(QMX_ENABLE_BACKEND_ARM_DSP) || defined(QMX_ENABLE_BACKEND_MCU32) || defined(QMX_ENABLE_BACKEND_CORTEX_M3) || defined(QMX_ENABLE_BACKEND_CORTEX_M0) || defined(QMX_ENABLE_BACKEND_CORTEX_M4)
QMX_ALIGNED(32) const uint32_t qmx_lut_binary_u32[16][2] = {
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

#endif

#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
QMX_ALIGNED(64) const int16_t qmx_lut_ternary_i16[256][4] = {
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

QMX_ALIGNED(64) const int16_t qmx_lut_binary_i16_4[16][4] = {
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

QMX_ALIGNED(64) const int16_t qmx_lut_binary_i16_8[256][8] = {
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
QMX_ALIGNED(64) const int8_t qmx_lut_binary_s8_8[256][8] = {
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
