#ifndef QMX_H
#define QMX_H

#if !defined(_WIN32) && !defined(__DJGPP__)
    #ifndef _FILE_OFFSET_BITS
        #define _FILE_OFFSET_BITS 64
    #endif
    #if !defined(_POSIX_C_SOURCE) && !defined(_XOPEN_SOURCE) && !defined(_DEFAULT_SOURCE)
        #define _POSIX_C_SOURCE 200809L
    #endif
#endif

#include <stdint.h>
#include <stddef.h>
#ifndef QMX_NO_STDIO
#include <stdio.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define QMX_MAGIC               0x31584D51u
#define QMX_TOKENIZER_MAGIC     0x4E4B4F54u
#define QMX_VERSION_1_0         0x0100u
#define QMX_VERSION_1_1         0x0101u
#define QMX_VERSION             QMX_VERSION_1_1
#define QMX_RUNTIME_VERSION_STR "v1.2b"

#define QMX_CACHE_ALIGN         64u
#define QMX_SECTOR_ALIGN        512u
#define QMX_PAGE_ALIGN          4096u

#define QMX_TILE_ROWS           16u
#define QMX_TILE_COLS           32u
#define QMX_TILE_ELEMS          (QMX_TILE_ROWS * QMX_TILE_COLS)

#define QMX_MAX_DIM             8192u
#define QMX_MAX_INTER_DIM       65536u
#define QMX_MAX_ARENA_SLOTS     16u
#define QMX_EMB_CACHE_SIZE      64u

#define QMX_FLAG_TILE_MAJOR         (1u << 0)
#define QMX_FLAG_SHARED_CLASSIFIER  (1u << 1)
#define QMX_FLAG_ROPE_SPLIT_HALF    (1u << 2)
#define QMX_FLAG_LLAMA3_ROPE_SCALE  (1u << 3)
#define QMX_FLAG_HAS_BIAS           (1u << 4)
#define QMX_FLAG_LOGIT_SOFTCAP      (1u << 5)
#define QMX_FLAG_COMPACT_ALIGN      (1u << 6)
#define QMX_FLAG_ATTN_SOFTCAP       (1u << 7)

typedef enum {
    QMX_KV_FP32 = 0,
    QMX_KV_Q8   = 1,
    QMX_KV_FP16 = 2
} QmxKvType;

typedef enum {
    QMX_ROPE_SCALE_NONE   = 0,
    QMX_ROPE_SCALE_LINEAR = 1,
    QMX_ROPE_SCALE_LLAMA3 = 2,
    QMX_ROPE_SCALE_YARN   = 3
} QmxRopeScaleType;

typedef enum {
    QMX_QUANT_BINARY_1BIT = 0x01,
    QMX_QUANT_TERNARY_158 = 0x02,
    QMX_QUANT_Q4_SYM      = 0x03,
    QMX_QUANT_Q8_SYM      = 0x04,
    QMX_QUANT_FP32        = 0x05,
    QMX_QUANT_Q4_ROW32    = 0x06,
    QMX_QUANT_Q8_ROW32    = 0x07
} QmxQuantType;

typedef enum {
    QMX_PIN_CRITICAL     = 0x00,
    QMX_PIN_LATENCY_SENS = 0x01,
    QMX_PIN_STREAM_ONLY  = 0x02
} QmxPinPriority;

typedef enum {
    QMX_BACKEND_SCALAR     = 0,
    QMX_BACKEND_P54C       = 1,
    QMX_BACKEND_MMX        = 2,
    QMX_BACKEND_3DNOW      = 3,
    QMX_BACKEND_SSE2       = 4,
    QMX_BACKEND_AVX2       = 5,
    QMX_BACKEND_NEON       = 6,
    QMX_BACKEND_ARM_DSP    = 7,
    QMX_BACKEND_MCU32      = 8,
    QMX_BACKEND_SSE        = 9,
    QMX_BACKEND_SSE3       = 10,
    QMX_BACKEND_SSSE3      = 11,
    QMX_BACKEND_SSE41      = 12,
    QMX_BACKEND_SSE42      = 13,
    QMX_BACKEND_AVX        = 14,
    QMX_BACKEND_AVX512     = 15,
    QMX_BACKEND_CORTEX_M0  = 16,
    QMX_BACKEND_CORTEX_M23 = 17,
    QMX_BACKEND_CORTEX_M3  = 18,
    QMX_BACKEND_CORTEX_M4  = 19,
    QMX_BACKEND_CORTEX_M33 = 20,
    QMX_BACKEND_CORTEX_M55 = 21,
    QMX_BACKEND_NEON_DOT   = 22,
    QMX_BACKEND_COUNT      = 23,
    QMX_BACKEND_AUTO       = 255
} QmxBackend;

typedef enum {
    QMX_OP_NOP          = 0x00,
    QMX_OP_RMSNORM      = 0x01,
    QMX_OP_SUBNORM      = 0x02,
    QMX_OP_QUANT_ACT    = 0x03,
    QMX_OP_MATMUL       = 0x04,
    QMX_OP_MATMUL_ADD   = 0x05,
    QMX_OP_ADD_BIAS     = 0x06,
    QMX_OP_ROPE         = 0x07,
    QMX_OP_ATTN         = 0x08,
    QMX_OP_SWIGLU       = 0x09,
    QMX_OP_GEGLU        = 0x0A,
    QMX_OP_RELU2        = 0x0B,
    QMX_OP_RESIDUAL_ADD = 0x0C,
    QMX_OP_LAYERNORM    = 0x0D,
    QMX_OP_QK_NORM      = 0x0E,
    QMX_OP_SCALE_MUL    = 0x0F,
    QMX_OP_SILU         = 0x10
} QmxOpCode;

typedef enum {
    QMX_BUF_X         = 0,
    QMX_BUF_XB        = 1,
    QMX_BUF_XB2       = 2,
    QMX_BUF_Q         = 3,
    QMX_BUF_K         = 4,
    QMX_BUF_V         = 5,
    QMX_BUF_HB_GATE   = 6,
    QMX_BUF_HB_UP     = 7
} QmxArenaSlot;

typedef enum {
    QMX_TID_NONE           = 0xFFFFu,
    QMX_TID_TOKEN_EMB      = 0x0001u,
    QMX_TID_RMS_FINAL      = 0x0002u,
    QMX_TID_LM_HEAD        = 0x0003u,
    QMX_TID_RMS_FINAL_BIAS = 0x0004u,

    QMX_TID_ATTN_NORM      = 0x0100u,
    QMX_TID_ATTN_Q         = 0x0101u,
    QMX_TID_ATTN_K         = 0x0102u,
    QMX_TID_ATTN_V         = 0x0103u,
    QMX_TID_ATTN_WO        = 0x0104u,
    QMX_TID_ATTN_SUBNORM   = 0x0105u,
    QMX_TID_ATTN_Q_BIAS    = 0x0106u,
    QMX_TID_ATTN_K_BIAS    = 0x0107u,
    QMX_TID_ATTN_V_BIAS    = 0x0108u,
    QMX_TID_ATTN_O_BIAS    = 0x0109u,
    QMX_TID_ATTN_NORM_BIAS = 0x010Au,
    QMX_TID_ATTN_Q_NORM    = 0x010Bu,
    QMX_TID_ATTN_K_NORM    = 0x010Cu,
    QMX_TID_ATTN_POST_NORM = 0x010Du,

    QMX_TID_FFN_NORM       = 0x0200u,
    QMX_TID_FFN_GATE       = 0x0201u,
    QMX_TID_FFN_UP         = 0x0202u,
    QMX_TID_FFN_DOWN       = 0x0203u,
    QMX_TID_FFN_SUBNORM    = 0x0204u,
    QMX_TID_FFN_NORM_BIAS  = 0x0205u,
    QMX_TID_FFN_POST_NORM  = 0x0206u
} QmxTensorId;

typedef int  (*QmxReadFn)(void *ctx, uint64_t offset, void *dst, size_t size);
typedef void (*QmxCloseFn)(void *ctx);

typedef struct {
    QmxReadFn  read;
    QmxCloseFn close;
    void      *ctx;
    uint64_t   size;
} QmxReader;

typedef struct {
    uint32_t   override_max_seq;
    QmxBackend backend;
    QmxKvType  kv_type;
    int        force_stream;
    size_t     max_ram_bytes;
    void      *arena_buf;
    size_t     arena_size;
} QmxLoadParams;

typedef struct {
    uint32_t    hidden_dim;
    uint32_t    intermediate_dim;
    uint32_t    num_layers;
    uint32_t    num_heads;
    uint32_t    num_kv_heads;
    uint32_t    head_dim;
    uint32_t    rotary_dim;
    uint32_t    vocab_size;
    uint32_t    max_seq_len;
    uint32_t    orig_max_seq_len;
    uint32_t    sliding_window;
    uint16_t    primary_quant;
    QmxKvType   kv_type;
    QmxBackend  backend;
    const char *backend_name;
    int         stream_mode;
    int         is_mmap;
    int         is_xip;
    uint32_t    tensor_count;
    uint32_t    tokenizer_offset;
    uint32_t    tokenizer_size;
    int         num_threads;
} QmxModelInfo;

typedef struct {
    size_t   arena_used;
    size_t   arena_capacity;
    size_t   kv_cache_bytes;
    size_t   pin_pool_bytes;
    uint32_t pinned_tensors;
    uint32_t tensor_count;
    uint32_t disk_reads;
    uint64_t disk_bytes_read;
    uint32_t emb_cache_hits;
    uint32_t emb_cache_misses;
    int32_t  last_stream_layer;
} QmxStats;

typedef enum {
    QMX_STEP_ERROR   = -1,
    QMX_STEP_DONE    = 0,
    QMX_STEP_PREFILL = 1,
    QMX_STEP_TOKEN   = 2,
    QMX_STEP_EOS     = 3,
    QMX_STEP_STOP    = 4
} QmxStepStatus;

typedef struct {
    float    temp;
    int      top_k;
    float    top_p;
    float    min_p;
    float    rep_penalty;
    uint32_t seed;
    int      max_tokens;
    int      prefill_chunk;
    int      check_qa_stop;
} QmxGenParams;

typedef struct {
    int         token;
    const char *piece;
    int         pos;
    int         prefill_done;
    int         prefill_total;
    int         generated;
    int         kv_shifted;
} QmxStepResult;

#pragma pack(push, 1)

typedef struct {
    uint32_t magic;
    uint16_t version;
    uint16_t flags;
    uint32_t header_crc32;
    uint32_t file_crc32;

    uint32_t recipe_offset;
    uint32_t recipe_count;
    uint32_t tensor_dir_offset;
    uint32_t tensor_count;
    uint32_t tokenizer_offset;
    uint32_t tokenizer_size;
    uint32_t payload_offset;
    uint32_t payload_size;

    uint32_t vocab_size;
    uint32_t max_seq_len;
    uint32_t hidden_dim;
    uint32_t intermediate_dim;
    uint16_t num_layers;
    uint16_t num_heads;
    uint16_t num_kv_heads;
    uint16_t primary_quant;
    float    rms_norm_eps;
    float    rope_theta;
    uint32_t min_ram_bytes;

    uint32_t payload_offset_hi;
    uint32_t payload_size_hi;
    uint16_t head_dim;
    uint16_t rotary_dim;
    float    emb_scale;
    float    logit_softcap;
    float    attn_softcap;
    float    rope_scale_factor;
    uint8_t  rope_scale_type;
    uint8_t  norm_type;
    uint16_t sliding_window;
    uint8_t  reserved[12];
} QmxHeader;

typedef struct {
    char     name[32];
    uint16_t tensor_id;
    uint16_t layer_idx;
    uint8_t  quant_type;
    uint8_t  pin_priority;
    uint16_t tile_rows;
    uint32_t rows;
    uint32_t cols;
    uint32_t scales_offset;
    uint32_t data_offset;
    uint32_t data_size;
    uint16_t scales_offset_hi;
    uint16_t data_offset_hi;
} QmxTensorDesc;

typedef struct {
    uint8_t  op;
    uint8_t  flags;
    uint16_t tensor_id;
    uint16_t src_buf;
    uint16_t dst_buf;
} QmxInstruction;

typedef struct {
    uint32_t magic;
    uint32_t vocab_size;
    uint32_t merges_count;
    uint32_t max_token_len;
    int32_t  bos_id;
    int32_t  eos_id;
    int32_t  eot_id;
    uint32_t flags;
} QmxTokenizerHeader;

typedef struct {
    uint32_t left_tok;
    uint32_t right_tok;
    uint32_t target_tok;
    uint32_t priority;
} QmxBpeMergeEntry;

#pragma pack(pop)

typedef char assert_qmx_header_128b[(sizeof(QmxHeader) == 128) ? 1 : -1];
typedef char assert_qmx_hdr_rope_off[(offsetof(QmxHeader, rope_scale_type) == 112) ? 1 : -1];
typedef char assert_qmx_hdr_norm_off[(offsetof(QmxHeader, norm_type) == 113) ? 1 : -1];
typedef char assert_qmx_hdr_swa_off[(offsetof(QmxHeader, sliding_window) == 114) ? 1 : -1];
typedef char assert_qmx_desc_64b[(sizeof(QmxTensorDesc) == 64) ? 1 : -1];
typedef char assert_qmx_inst_8b[(sizeof(QmxInstruction) == 8) ? 1 : -1];
typedef char assert_qmx_tok_hdr_32b[(sizeof(QmxTokenizerHeader) == 32) ? 1 : -1];
typedef char assert_qmx_merge_16b[(sizeof(QmxBpeMergeEntry) == 16) ? 1 : -1];

#ifdef __cplusplus
}
#endif

#endif
