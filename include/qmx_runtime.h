#ifndef QMX_RUNTIME_H
#define QMX_RUNTIME_H

#include "qmx.h"
#include "qmx_thread.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_MSC_VER) && (defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1))
    #include <xmmintrin.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
    #define QMX_ALIGNED(x) __attribute__((aligned(x)))
    #define QMX_RESTRICT   __restrict__
    #define QMX_INLINE     static inline __attribute__((always_inline))
#elif defined(_MSC_VER)
    #define QMX_ALIGNED(x) __declspec(align(x))
    #define QMX_RESTRICT   __restrict
    #define QMX_INLINE     static __forceinline
#else
    #define QMX_ALIGNED(x)
    #define QMX_RESTRICT
    #define QMX_INLINE     static inline
#endif

typedef struct QmxBackendOps QmxBackendOps;

typedef union {
    float    f;
    uint32_t u;
    int32_t  i;
} QmxFloatBits;

QMX_INLINE size_t qmx_align_up(size_t val, size_t align) {
    size_t rem = val % align;
    return (rem == 0) ? val : (val + (align - rem));
}

QMX_INLINE int32_t qmx_round_nearest_i32(float v) {
    QmxFloatBits fb;
    fb.f = v + 12582912.0f;
    return (fb.i & 0x007FFFFF) - 0x00400000;
}

QMX_INLINE float qmx_fp16_to_fp32(uint16_t h) {
    uint32_t sign = ((uint32_t)(h & 0x8000u)) << 16u;
    uint32_t exp  = (uint32_t)((h >> 10u) & 0x1Fu);
    uint32_t mant = (uint32_t)(h & 0x03FFu);
    QmxFloatBits fb;
    if (exp == 0u) {
        if (mant == 0u) {
            fb.u = sign;
            return fb.f;
        }
        exp = 1u;
        while ((mant & 0x0400u) == 0u) {
            mant <<= 1u;
            exp--;
        }
        mant &= 0x03FFu;
        fb.u = sign | ((exp + 112u) << 23u) | (mant << 13u);
        return fb.f;
    }
    if (exp == 31u) {
        fb.u = sign | 0x7F800000u | (mant << 13u);
        return fb.f;
    }
    fb.u = sign | ((exp + 112u) << 23u) | (mant << 13u);
    return fb.f;
}

QMX_INLINE uint16_t qmx_fp32_to_fp16(float f) {
    QmxFloatBits fb;
    fb.f = f;
    uint32_t sign = (fb.u >> 16u) & 0x8000u;
    int32_t exp = (int32_t)((fb.u >> 23u) & 0xFFu) - 127 + 15;
    uint32_t mant = fb.u & 0x007FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant = (mant | 0x00800000u) >> (uint32_t)(1 - exp);
        return (uint16_t)(sign | ((mant + 0x00001000u) >> 13u));
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
    uint32_t rounded = ((uint32_t)exp << 10u) + ((mant + 0x00001000u) >> 13u);
    if (rounded >= 0x7C00u) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | rounded);
}

QMX_INLINE float qmx_sqrtf(float x) {
    if (x <= 0.0f) return 0.0f;
#if defined(_MSC_VER) && (defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1))
    return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x)));
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__SSE__))
    __asm__ ("sqrtss %1, %0" : "=x"(x) : "x"(x));
    return x;
#elif (defined(__GNUC__) || defined(__clang__)) && defined(__i386__)
    __asm__ ("fsqrt" : "+t"(x));
    return x;
#elif (defined(__GNUC__) || defined(__clang__)) && defined(__aarch64__)
    __asm__ ("fsqrt %s0, %s0" : "+w"(x));
    return x;
#elif (defined(__GNUC__) || defined(__clang__)) && defined(__arm__) && defined(__VFP_FP__) && !defined(__SOFTFP__)
    __asm__ ("vsqrt.f32 %0, %0" : "+t"(x));
    return x;
#else
    QmxFloatBits u;
    u.f = x;
    u.u = 0x5f3759dfu - (u.u >> 1);
    float y = u.f;
    y = y * (1.5f - 0.5f * x * y * y);
    y = y * (1.5f - 0.5f * x * y * y);
    return x * y;
#endif
}

QMX_INLINE float qmx_expf(float x) {
    if (x > 88.0f) return 3.402823466e+38f;
    if (x < -87.3f) return 0.0f;
    QmxFloatBits kd, res;
    kd.f = x * 1.4426950408889634f + 12582912.0f;
    int32_t k = (kd.i & 0x007FFFFF) - 0x00400000;
    float kf = (float)k;
    float r = (x - kf * 0.693145751953125f) - kf * 1.428606765330187e-6f;
    float p = 1.38888889e-3f;
    p = p * r + 8.33333333e-3f;
    p = p * r + 4.16666667e-2f;
    p = p * r + 1.66666667e-1f;
    p = p * r + 5.0f * 1e-1f;
    p = p * (r * r) + r + 1.0f;
    res.f = p;
    res.i += (k << 23);
    return res.f;
}

QMX_INLINE void qmx_sincosf(float val, float * QMX_RESTRICT s_out, float * QMX_RESTRICT c_out) {
#if (defined(__GNUC__) || defined(__clang__)) && defined(__i386__) && !defined(__SSE2__)
    float s, c;
    __asm__ ("fsincos" : "=t"(c), "=u"(s) : "0"(val));
    *s_out = s;
    *c_out = c;
#else
    int32_t ki = qmx_round_nearest_i32(val * 0.3183098861837907f);
    float kf = (float)ki;
    float x = (val - kf * 3.141592025756836f) - kf * 6.2781803655e-7f;
    float x2 = x * x;
    float s = x * (1.0f - x2 * (0.16666667f - x2 * (0.0083333310f - x2 * 0.00019840874f)));
    float c = 1.0f - x2 * (0.5f - x2 * (0.041666638f - x2 * (0.0013888378f - x2 * 0.0000247609f)));
    if (ki & 1) {
        s = -s;
        c = -c;
    }
    *s_out = s;
    *c_out = c;
#endif
}

typedef struct {
    const void  *data;
    const float *scales;
    uint64_t     scales_offset;
    uint64_t     data_offset;
    uint64_t     data_size;
    uint32_t     rows;
    uint32_t     cols;
    uint16_t     tensor_id;
    uint16_t     layer_idx;
    uint8_t      quant_type;
    uint8_t      pin_priority;
    uint16_t     tile_rows;
} QmxTensor;

typedef struct {
    uint8_t *raw_ptr;
    uint8_t *base_ptr;
    size_t   capacity;
    size_t   offset;
    int      is_external;
} QmxArena;

typedef struct {
    float    *slots[QMX_MAX_ARENA_SLOTS];
    int8_t   *xq;
    int16_t  *xq16;
    float    *x_scales;
    float    *rope_inv_freq;
    float    *rope_cos;
    float    *rope_sin;
    float    *att;
    float    *logits;
    float    *key_cache;
    float    *value_cache;
    uint16_t *key_cache_f16;
    uint16_t *value_cache_f16;
    int8_t   *key_cache_q8;
    int8_t   *value_cache_q8;
    float    *key_scales;
    float    *value_scales;
    uint32_t  stream_tile_cap;
    float    *stream_scales_buf;
    uint8_t  *stream_data_buf;
    uint8_t  *stream_data_buf2;
    uint32_t  emb_cache_cap;
    int32_t  *emb_cache_tokens;
    uint32_t *emb_cache_lru;
    float    *emb_cache_vecs;
} QmxRunState;

typedef struct {
    QmxHeader            header;
    uint32_t             orig_max_seq_len;
    uint32_t             max_dim;
    uint32_t             head_dim;
    uint32_t             rotary_dim;
    uint32_t             q_dim;
    uint32_t             kv_dim;
    uint32_t             att_buf_dim;
    uint32_t             sliding_window;
    float                emb_scale;
    float                logit_softcap;
    float                attn_softcap;
    uint64_t             payload_offset_64;
    uint64_t             payload_size_64;
    QmxInstruction      *recipe;
    QmxTensor           *tensors;
    const QmxTensor    **recipe_tensors;
    uint32_t            *recipe_quant_cols;
    uint8_t             *recipe_fuse;
    const QmxTensor     *tok_emb_tensor;
    const QmxTensor     *rms_final_tensor;
    const QmxTensor     *rms_final_bias_tensor;
    const QmxTensor     *cls_tensor;
    QmxRunState          state;
    QmxArena             arena;
    const QmxBackendOps *ops;
    void                *payload_mem;
    size_t               payload_mem_size;
    int                  is_mmap;
    int                  is_xip;
    int                  use_kv8;
    int                  kv_type;
    QmxBackend           backend;
    int                  stream_mode;
    QmxReader            reader;
    uint8_t             *pin_pool;
    size_t               pin_pool_size;
    uint32_t             pinned_tensors;
    uint32_t             emb_lru_clock;
    uint32_t             emb_cache_hits;
    uint32_t             emb_cache_misses;
    uint32_t             disk_reads;
    uint64_t             disk_bytes_read;
    int32_t              last_stream_layer;
    int                  num_threads;
    QmxThreadPool       *pool;
    uint8_t             *batch_ws;
    size_t               batch_ws_cap;
    float               *mt_att_buf;
    size_t               mt_att_cap;
} QmxModel;

#ifndef QMX_NO_STDIO
int     qmx_fseek64(FILE *fp, int64_t offset, int origin);
int64_t qmx_ftell64(FILE *fp);
int64_t qmx_file_size64(FILE *fp);
#endif

uint32_t qmx_crc32(const void *data, size_t len);
size_t   qmx_tile_data_bytes(uint8_t quant_type);
size_t   qmx_tile_row_bytes(uint8_t quant_type);
uint64_t qmx_matrix_data_bytes(uint32_t rows, uint32_t cols, uint8_t quant_type);
size_t   qmx_matrix_scales_bytes(uint32_t rows, uint32_t cols);
uint64_t qmx_matrix_scales_bytes_ex(uint32_t rows, uint32_t cols, uint8_t quant_type);

int   qmx_arena_init(QmxArena *arena, size_t capacity);
int   qmx_arena_init_external(QmxArena *arena, void *buf, size_t capacity);
void *qmx_arena_alloc(QmxArena *arena, size_t size, size_t alignment);
void  qmx_arena_free(QmxArena *arena);

void qmx_rmsnorm_scalar(float * QMX_RESTRICT out, const float * QMX_RESTRICT x,
                        const float * QMX_RESTRICT weight, uint32_t size, float eps);
void qmx_subnorm_scalar(float * QMX_RESTRICT out, const float * QMX_RESTRICT x,
                        const float * QMX_RESTRICT weight, uint32_t size, float eps);
void qmx_layernorm_scalar(float * QMX_RESTRICT out, const float * QMX_RESTRICT x,
                          const float * QMX_RESTRICT weight, uint32_t size, float eps);
void qmx_qk_rmsnorm_scalar(float * QMX_RESTRICT vec, const float * QMX_RESTRICT weight,
                           uint32_t total_dim, uint32_t head_size, float eps);
void qmx_softmax_scalar(float * QMX_RESTRICT x, uint32_t size);
void qmx_rope_apply(float * QMX_RESTRICT q, float * QMX_RESTRICT k,
                    const float * QMX_RESTRICT fcr, const float * QMX_RESTRICT fci,
                    uint32_t dim, uint32_t kv_dim, uint32_t head_size, int split_half);
void qmx_rope_apply_partial(float * QMX_RESTRICT q, float * QMX_RESTRICT k,
                            const float * QMX_RESTRICT fcr, const float * QMX_RESTRICT fci,
                            uint32_t q_dim, uint32_t kv_dim, uint32_t head_size,
                            uint32_t rotary_dim, int split_half);

#define QMX_FUSE_NONE          0u
#define QMX_FUSE_RMSNORM_QUANT 1u
#define QMX_FUSE_SWIGLU_QUANT  2u

void qmx_quantize_activations_q8(const float * QMX_RESTRICT x,
                                 int8_t      * QMX_RESTRICT xq,
                                 float       * QMX_RESTRICT x_scales,
                                 uint32_t cols);

void qmx_rmsnorm_quant_scalar(const float * QMX_RESTRICT x,
                              const float * QMX_RESTRICT weight,
                              int8_t      * QMX_RESTRICT xq,
                              int16_t     * QMX_RESTRICT xq16,
                              float       * QMX_RESTRICT x_scales,
                              uint32_t size,
                              float eps);

void qmx_swiglu_quant_scalar(const float * QMX_RESTRICT gate,
                             const float * QMX_RESTRICT up,
                             int8_t      * QMX_RESTRICT xq,
                             int16_t     * QMX_RESTRICT xq16,
                             float       * QMX_RESTRICT x_scales,
                             uint32_t size);

void qmx_f16_to_f32_scalar(float * QMX_RESTRICT dst, const uint16_t * QMX_RESTRICT src, uint32_t n);
void qmx_f32_to_f16_scalar(uint16_t * QMX_RESTRICT dst, const float * QMX_RESTRICT src, uint32_t n);

void qmx_dequantize_tiled_row(float * QMX_RESTRICT out,
                              const QmxTensor *w,
                              uint32_t row_idx);

void qmx_matmul_tiled_scalar(float * QMX_RESTRICT out,
                             const int8_t * QMX_RESTRICT xq,
                             const float  * QMX_RESTRICT x_scales,
                             const QmxTensor *w,
                             int accumulate);

const QmxTensor *qmx_find_tensor(const QmxModel *model, uint16_t layer_idx, uint16_t tensor_id);

QmxLoadParams qmx_load_params_default(void);

size_t qmx_calc_arena_size(const QmxReader *reader, const QmxLoadParams *params, int is_xip);
size_t qmx_calc_arena_size_memory(const void *cartridge_mem, size_t cartridge_size, const QmxLoadParams *params);

int qmx_load_from_reader(QmxModel *model, QmxReader reader, const QmxLoadParams *params);

int qmx_load_from_memory_ex(QmxModel *model, const void *cartridge_mem, size_t cartridge_size,
                            const QmxLoadParams *params);

int qmx_load_from_memory(QmxModel *model, const void *cartridge_mem, size_t cartridge_size,
                         uint32_t override_max_seq, QmxBackend backend, int kv_mode);

#ifndef QMX_NO_STDIO
int qmx_load_ex(QmxModel *model, const char *path, uint64_t base_offset, uint64_t slice_size,
                const QmxLoadParams *params);

int qmx_load(QmxModel *model, const char *path, uint32_t override_max_seq,
             QmxBackend backend, int kv_mode, int force_stream, size_t max_ram_bytes);

int qmx_load_from_offset(QmxModel *model, const char *path, uint64_t base_offset, uint64_t slice_size,
                         uint32_t override_max_seq, QmxBackend backend,
                         int kv_mode, int force_stream, size_t max_ram_bytes);
#endif

void qmx_model_matmul(QmxModel *model, float * QMX_RESTRICT out,
                      const QmxTensor *w, int32_t layer_idx, int accumulate);

void qmx_load_token_embedding(QmxModel *model, float * QMX_RESTRICT dst, int token);

int qmx_shift_kv_cache(QmxModel *model, int cur_pos, int shift);
int qmx_shift_kv_cache_ex(QmxModel *model, int cur_pos, int shift, int n_sink);

float *qmx_forward(QmxModel *model, int token, int pos);

float *qmx_forward_batch_ex(QmxModel *model, const int *tokens, int n_tokens,
                            int *io_pos, float *step0_chk_out, int *out_shifted);

float *qmx_forward_batch(QmxModel *model, const int *tokens, int n_tokens,
                         int start_pos, float *step0_chk_out);

void qmx_get_model_info(const QmxModel *model, QmxModelInfo *out);
void qmx_get_stats(const QmxModel *model, QmxStats *out);
void qmx_reset_stats(QmxModel *model);

void qmx_set_num_threads(QmxModel *model, int num_threads);

void qmx_free(QmxModel *model);

#ifdef __cplusplus
}
#endif

#endif
