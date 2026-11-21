#if !defined(_WIN32) && !defined(__DJGPP__)
    #ifndef _POSIX_C_SOURCE
        #define _POSIX_C_SOURCE 200809L
    #endif
#endif

#include "../include/qmx_runtime.h"
#include "../include/qmx_kernels.h"
#ifndef QMX_NO_MALLOC
#include <stdlib.h>
#endif
#include <string.h>

#if defined(QMX_USE_MMAP) && !defined(QMX_NO_MMAP) && !defined(QMX_NO_STDIO) && (defined(__unix__) || defined(__APPLE__))
    #define QMX_HAS_MMAP 1
    #include <sys/mman.h>
    #include <fcntl.h>
    #include <unistd.h>
#else
    #define QMX_HAS_MMAP 0
#endif

#define QMX_DESC_BATCH 16u

typedef struct {
    QmxReader      reader;
    const uint8_t *mem;
    uint64_t       total_slice;
    const char    *mmap_path;
    uint64_t       mmap_base_offset;
} QmxSource;

typedef struct {
    QmxHeader header;
    int       is_v11;
    uint64_t  payload_offset_64;
    uint64_t  payload_size_64;
    uint32_t  orig_max_seq_len;
    uint32_t  head_dim;
    uint32_t  rotary_dim;
    uint32_t  q_dim;
    uint32_t  kv_dim;
    uint32_t  att_buf_dim;
    uint32_t  kv_slot_d;
    uint32_t  max_dim;
    uint32_t  sliding_window;
    float     emb_scale;
    float     logit_softcap;
    float     attn_softcap;
    size_t    recipe_bytes;
    size_t    recipe_map_bytes;
    size_t    quant_cols_bytes;
    uint32_t  max_tensor_cols;
    uint32_t  max_tiles_c;
    size_t    pad_act_cols;
    size_t    kv_entries;
    size_t    kv_scale_entries;
    uint64_t  critical_bytes64;
    uint64_t  all_scales_bytes64;
    uint32_t  max_matrix_tiles;
    int       need_xq16;
    size_t    base_arena_bytes;
    uint32_t  stream_tile_cap;
    uint32_t  emb_cache_cap;
    int       pin_all_scales_in_arena;
    int       use_double_buf;
    size_t    stream_extra_bytes;
} QmxLayoutPlan;

static int qmx_plan_needs_xq16(QmxBackend requested, size_t max_ram_bytes, int is_external_arena) {
#if defined(QMX_NO_SIMD)
    (void)requested;
    (void)max_ram_bytes;
    (void)is_external_arena;
    return 0;
#else
    if (is_external_arena || (max_ram_bytes > 0u && max_ram_bytes <= 262144u)) {
        QmxBackend b = (requested == QMX_BACKEND_AUTO) ? qmx_detect_backend() : requested;
        return qmx_backend_needs_xq16(b);
    }
    return 1;
#endif
}

static int qmx_range_valid64(uint64_t off, uint64_t len, uint64_t limit) {
    return (off <= limit) && (len <= limit - off);
}

static int qmx_src_read(const QmxSource *src, uint64_t rel_off, void *dst, size_t len) {
    if (!qmx_range_valid64(rel_off, (uint64_t)len, src->total_slice)) return -1;
    if (len == 0u) return 0;
    if (src->mem != NULL) {
        memcpy(dst, src->mem + (size_t)rel_off, len);
        return 0;
    }
    if (!src->reader.read) return -1;
    return src->reader.read(src->reader.ctx, rel_off, dst, len);
}

static int qmx_valid_quant_type(uint8_t qt) {
    return (qt >= QMX_QUANT_BINARY_1BIT && qt <= QMX_QUANT_Q8_ROW32);
}

static int qmx_normalize_kv_type(int kv_mode) {
    if (kv_mode == (int)QMX_KV_FP16) return (int)QMX_KV_FP16;
    return kv_mode ? (int)QMX_KV_Q8 : (int)QMX_KV_FP32;
}

static uint32_t qmx_slot_capacity(const QmxModel *model, uint16_t slot) {
    const QmxHeader *h = &model->header;
    uint32_t kv_cap = (model->kv_dim > h->hidden_dim) ? model->kv_dim : h->hidden_dim;
    switch (slot) {
        case QMX_BUF_X:
        case QMX_BUF_XB2:
            return h->hidden_dim;
        case QMX_BUF_XB:
        case QMX_BUF_Q:
            return model->att_buf_dim;
        case QMX_BUF_K:
        case QMX_BUF_V:
            return kv_cap;
        case QMX_BUF_HB_GATE:
        case QMX_BUF_HB_UP:
            return h->intermediate_dim;
        default:
            return 0u;
    }
}

QmxLoadParams qmx_load_params_default(void) {
    QmxLoadParams p;
    memset(&p, 0, sizeof(p));
    p.backend = QMX_BACKEND_AUTO;
    p.kv_type = QMX_KV_FP32;
    return p;
}

const QmxTensor *qmx_find_tensor(const QmxModel *model, uint16_t layer_idx, uint16_t tensor_id) {
    if (!model || !model->tensors || tensor_id == QMX_TID_NONE) return NULL;
    uint32_t count = model->header.tensor_count;
    for (uint32_t i = 0; i < count; i++) {
        if (model->tensors[i].layer_idx == layer_idx && model->tensors[i].tensor_id == tensor_id) {
            return &model->tensors[i];
        }
    }
    return NULL;
}

static int pin_tensor_scales_from_reader(const QmxReader *r, uint64_t pay_base,
                                         uint8_t *pool_base, QmxTensor *t,
                                         size_t *pool_off, size_t pool_cap) {
    if (t->quant_type == QMX_QUANT_FP32 || t->scales != NULL) return 0;
    if (!r || !r->read) return -1;
    uint64_t s_bytes64 = qmx_matrix_scales_bytes_ex(t->rows, t->cols, t->quant_type);
    if (s_bytes64 > (uint64_t)pool_cap) return -1;
    size_t s_bytes = (size_t)s_bytes64;
    size_t off = qmx_align_up(*pool_off, QMX_CACHE_ALIGN);
    if (!qmx_range_valid64((uint64_t)off, (uint64_t)s_bytes, (uint64_t)pool_cap)) return -1;

    float *s_dst = (float *)(pool_base + off);
    if (r->read(r->ctx, pay_base + t->scales_offset, s_dst, s_bytes) != 0) {
        return -1;
    }
    t->scales = s_dst;
    *pool_off = off + s_bytes;
    return 0;
}

static int pin_tensor_from_reader(QmxModel *model, const QmxReader *r, uint64_t pay_base,
                                  uint8_t *pool_base, QmxTensor *t,
                                  size_t *pool_off, size_t pool_cap) {
    if (!r || !r->read) return -1;
    if (t->data_size > (uint64_t)pool_cap) return -1;
    size_t d_bytes = (size_t)t->data_size;

    if (t->quant_type == QMX_QUANT_FP32) {
        size_t off = qmx_align_up(*pool_off, QMX_CACHE_ALIGN);
        if (!qmx_range_valid64((uint64_t)off, (uint64_t)d_bytes, (uint64_t)pool_cap)) return -1;
        uint8_t *dst = pool_base + off;
        if (r->read(r->ctx, pay_base + t->data_offset, dst, d_bytes) != 0) {
            return -1;
        }
        t->data = dst;
        t->scales = NULL;
        *pool_off = off + d_bytes;
        model->pinned_tensors++;
        return 0;
    }

    if (t->scales == NULL) {
        uint64_t s_bytes64 = qmx_matrix_scales_bytes_ex(t->rows, t->cols, t->quant_type);
        if (s_bytes64 > (uint64_t)pool_cap) return -1;
        size_t s_bytes = (size_t)s_bytes64;
        size_t off1 = qmx_align_up(*pool_off, QMX_CACHE_ALIGN);
        if (!qmx_range_valid64((uint64_t)off1, (uint64_t)s_bytes, (uint64_t)pool_cap)) return -1;
        size_t off2 = qmx_align_up(off1 + s_bytes, QMX_CACHE_ALIGN);
        if (!qmx_range_valid64((uint64_t)off2, (uint64_t)d_bytes, (uint64_t)pool_cap)) return -1;

        float   *s_dst = (float *)(pool_base + off1);
        uint8_t *d_dst = pool_base + off2;
        if (r->read(r->ctx, pay_base + t->scales_offset, s_dst, s_bytes) != 0 ||
            r->read(r->ctx, pay_base + t->data_offset, d_dst, d_bytes) != 0) {
            return -1;
        }
        t->scales = s_dst;
        t->data = d_dst;
        *pool_off = off2 + d_bytes;
        model->pinned_tensors++;
        return 0;
    }

    size_t off = qmx_align_up(*pool_off, QMX_CACHE_ALIGN);
    if (!qmx_range_valid64((uint64_t)off, (uint64_t)d_bytes, (uint64_t)pool_cap)) return -1;
    uint8_t *d_dst = pool_base + off;
    if (r->read(r->ctx, pay_base + t->data_offset, d_dst, d_bytes) != 0) {
        return -1;
    }
    t->data = d_dst;
    *pool_off = off + d_bytes;
    model->pinned_tensors++;
    return 0;
}

static uint32_t resolve_recipe_quant_cols(const QmxModel *model, uint32_t pc, uint16_t src_buf) {
    for (uint32_t look = pc + 1u; look < model->header.recipe_count; look++) {
        uint8_t op = model->recipe[look].op;
        if (op == QMX_OP_MATMUL || op == QMX_OP_MATMUL_ADD) {
            const QmxTensor *w = qmx_find_tensor(model, 0, model->recipe[look].tensor_id);
            if (w) return w->cols;
        }
    }
    if (src_buf == QMX_BUF_HB_GATE || src_buf == QMX_BUF_HB_UP) {
        return model->header.intermediate_dim;
    }
    if (src_buf == QMX_BUF_Q) {
        return model->q_dim;
    }
    return model->header.hidden_dim;
}

static int qmx_slot_dead_after_quant(const QmxInstruction *recipe, uint32_t count,
                                     uint32_t start_pc, uint16_t slot) {
    if (slot == QMX_BUF_X) return 0;
    for (uint32_t k = start_pc; k < count; k++) {
        uint8_t op = recipe[k].op;
        uint16_t sb = recipe[k].src_buf;
        uint16_t db = recipe[k].dst_buf;
        if (op == QMX_OP_NOP) continue;
        if (op == QMX_OP_MATMUL) {
            if (db == slot) return 1;
            continue;
        }
        if (op == QMX_OP_MATMUL_ADD) {
            if (db == slot) return 0;
            continue;
        }
        if (op == QMX_OP_ATTN) {
            if (sb == slot || slot == QMX_BUF_K || slot == QMX_BUF_V) return 0;
            if (db == slot) return 1;
            continue;
        }
        if (op == QMX_OP_ROPE || op == QMX_OP_QK_NORM || op == QMX_OP_ADD_BIAS ||
            op == QMX_OP_RESIDUAL_ADD || op == QMX_OP_SWIGLU || op == QMX_OP_GEGLU || op == QMX_OP_RELU2) {
            if (sb == slot || db == slot) return 0;
            continue;
        }
        if (sb == slot) return 0;
        if (db == slot) return 1;
    }
    return 1;
}

static int validate_and_bind_recipe(QmxModel *model, uint32_t max_tensor_cols) {
    const QmxHeader *h = &model->header;
    uint32_t dim = h->hidden_dim;
    uint32_t inter_dim = h->intermediate_dim;
    uint32_t q_dim = model->q_dim;
    uint32_t kv_dim = model->kv_dim;
    uint32_t head_size = model->head_dim;

    model->tok_emb_tensor        = qmx_find_tensor(model, 0xFFFFu, QMX_TID_TOKEN_EMB);
    model->rms_final_tensor      = qmx_find_tensor(model, 0xFFFFu, QMX_TID_RMS_FINAL);
    model->rms_final_bias_tensor = qmx_find_tensor(model, 0xFFFFu, QMX_TID_RMS_FINAL_BIAS);
    model->cls_tensor            = qmx_find_tensor(model, 0xFFFFu, QMX_TID_LM_HEAD);
    if (!model->cls_tensor) {
        model->cls_tensor = model->tok_emb_tensor;
    }

    if (model->tok_emb_tensor) {
        if (model->tok_emb_tensor->rows < h->vocab_size || model->tok_emb_tensor->cols != dim) {
            return -1;
        }
    }
    if (model->rms_final_tensor) {
        if (model->rms_final_tensor->quant_type != QMX_QUANT_FP32 || model->rms_final_tensor->cols != dim) {
            return -1;
        }
    }
    if (model->rms_final_bias_tensor) {
        if (model->rms_final_bias_tensor->quant_type != QMX_QUANT_FP32 || model->rms_final_bias_tensor->cols != dim) {
            return -1;
        }
    }
    if (model->cls_tensor) {
        if (model->cls_tensor->rows < h->vocab_size || model->cls_tensor->cols != dim) {
            return -1;
        }
    }

    for (uint32_t pc = 0; pc < h->recipe_count; pc++) {
        const QmxInstruction *inst = &model->recipe[pc];
        uint32_t src_cap = qmx_slot_capacity(model, inst->src_buf);
        uint32_t dst_cap = qmx_slot_capacity(model, inst->dst_buf);
        if (inst->op != QMX_OP_NOP && (src_cap == 0u || dst_cap == 0u)) {
            return -1;
        }

        uint32_t qcols = resolve_recipe_quant_cols(model, pc, inst->src_buf);
        if (inst->op == QMX_OP_QUANT_ACT && (qcols > src_cap || qcols > max_tensor_cols)) {
            return -1;
        }
        model->recipe_quant_cols[pc] = qcols;

        switch (inst->op) {
            case QMX_OP_ROPE:
                if (q_dim > src_cap || kv_dim > dst_cap) return -1;
                break;
            case QMX_OP_ATTN:
                if (q_dim > src_cap || q_dim > dst_cap) return -1;
                break;
            case QMX_OP_SWIGLU:
            case QMX_OP_GEGLU:
            case QMX_OP_RELU2:
            case QMX_OP_SILU:
                if (inter_dim > src_cap || inter_dim > dst_cap) return -1;
                break;
            case QMX_OP_RESIDUAL_ADD:
                if (dim > src_cap || dim > dst_cap) return -1;
                break;
            default:
                break;
        }

        for (uint32_t l = 0; l < h->num_layers; l++) {
            const QmxTensor *w = (inst->tensor_id != QMX_TID_NONE)
                               ? qmx_find_tensor(model, (uint16_t)l, inst->tensor_id)
                               : NULL;
            model->recipe_tensors[(size_t)l * h->recipe_count + pc] = w;

            if (inst->op == QMX_OP_RMSNORM || inst->op == QMX_OP_LAYERNORM || inst->op == QMX_OP_SCALE_MUL) {
                if (!w || w->quant_type != QMX_QUANT_FP32 || w->cols > src_cap || w->cols > dst_cap) {
                    return -1;
                }
            } else if (inst->op == QMX_OP_QK_NORM) {
                if (!w || w->quant_type != QMX_QUANT_FP32 || (w->cols != head_size && w->cols != dst_cap)) {
                    return -1;
                }
            } else if (inst->op == QMX_OP_SUBNORM) {
                if (w != NULL) {
                    if (w->quant_type != QMX_QUANT_FP32 || w->cols > src_cap || w->cols > dst_cap) {
                        return -1;
                    }
                } else {
                    uint32_t sz = (inst->src_buf == QMX_BUF_HB_GATE || inst->src_buf == QMX_BUF_HB_UP) ? inter_dim : dim;
                    if (sz > src_cap || sz > dst_cap) return -1;
                }
            } else if (inst->op == QMX_OP_MATMUL || inst->op == QMX_OP_MATMUL_ADD) {
                if (!w || w->quant_type == QMX_QUANT_FP32 || w->rows > dst_cap || w->cols > max_tensor_cols) {
                    return -1;
                }
            } else if (inst->op == QMX_OP_ADD_BIAS) {
                if (!w || w->quant_type != QMX_QUANT_FP32 || w->cols > dst_cap) {
                    return -1;
                }
            }
        }
    }

    if (model->recipe_fuse) {
        memset(model->recipe_fuse, QMX_FUSE_NONE, (size_t)h->recipe_count);
        for (uint32_t pc = 0; pc + 1u < h->recipe_count; pc++) {
            const QmxInstruction *a = &model->recipe[pc];
            const QmxInstruction *b = &model->recipe[pc + 1u];
            if (b->op != QMX_OP_QUANT_ACT || a->dst_buf != b->src_buf) continue;
            if (!qmx_slot_dead_after_quant(model->recipe, h->recipe_count, pc + 2u, a->dst_buf)) continue;
            uint32_t qcols = model->recipe_quant_cols[pc + 1u];
            if (a->op == QMX_OP_RMSNORM) {
                int ok = 1;
                for (uint32_t l = 0; l < h->num_layers; l++) {
                    const QmxTensor *w = model->recipe_tensors[(size_t)l * h->recipe_count + pc];
                    if (!w || w->cols != qcols) { ok = 0; break; }
                }
                if (ok) {
                    model->recipe_fuse[pc] = QMX_FUSE_RMSNORM_QUANT;
                    pc++;
                }
            } else if (a->op == QMX_OP_SWIGLU && inter_dim == qcols) {
                model->recipe_fuse[pc] = QMX_FUSE_SWIGLU_QUANT;
                pc++;
            }
        }
    }
    return 0;
}

static void init_rope_inv_freq(QmxModel *model) {
    const QmxHeader *h = &model->header;
    QmxRunState *s = &model->state;
    uint32_t dim = h->hidden_dim;
    uint32_t rot_dim = model->rotary_dim;

    float theta = (h->rope_theta > 1.0f) ? h->rope_theta : 10000.0f;
    float log_theta = 9.210340371976184f;
    if (theta >= 490000.0f && theta <= 510000.0f) {
        log_theta = 13.122363377404328f;
    } else if (theta >= 900000.0f && theta <= 1100000.0f) {
        log_theta = 13.815510557964274f;
    } else if (theta != 10000.0f) {
        float x = theta;
        int e = 0;
        while (x >= 2.0f) { x *= 0.5f; e++; }
        while (x < 1.0f)  { x *= 2.0f; e--; }
        float y = (x - 1.0f) / (x + 1.0f);
        float y2 = y * y;
        float ln_x = 2.0f * y * (1.0f + y2 * (0.33333333f + y2 * (0.2f + y2 * 0.14285714f)));
        log_theta = (float)e * 0.6931471805599453f + ln_x;
    }

    uint8_t scale_type = (h->version >= QMX_VERSION_1_1) ? h->rope_scale_type : QMX_ROPE_SCALE_NONE;
    if (scale_type == QMX_ROPE_SCALE_NONE && (h->flags & QMX_FLAG_LLAMA3_ROPE_SCALE) != 0) {
        scale_type = QMX_ROPE_SCALE_LLAMA3;
    }
    float scale_factor = (h->version >= QMX_VERSION_1_1 && h->rope_scale_factor > 0.0f)
                       ? h->rope_scale_factor
                       : ((dim <= 3072u) ? 32.0f : 8.0f);

    for (uint32_t hd = 0, idx = 0; hd < rot_dim; hd += 2u, idx++) {
        float freq = qmx_expf(-(float)hd * (log_theta / (float)rot_dim));
        if (freq > 0.0f) {
            if (scale_type == QMX_ROPE_SCALE_LINEAR) {
                freq /= scale_factor;
            } else if (scale_type == QMX_ROPE_SCALE_LLAMA3) {
                float wavelen = 6.283185307179586f / freq;
                if (wavelen > 8192.0f) {
                    freq /= scale_factor;
                } else if (wavelen > 2048.0f) {
                    float smooth = (8192.0f / wavelen - 1.0f) * (1.0f / 3.0f);
                    freq = (1.0f - smooth) * (freq / scale_factor) + smooth * freq;
                }
            } else if (scale_type == QMX_ROPE_SCALE_YARN && scale_factor > 1.0f) {
                float orig_ctx = (model->orig_max_seq_len > 0u)
                               ? ((float)model->orig_max_seq_len / scale_factor)
                               : 2048.0f;
                if (orig_ctx < 256.0f) orig_ctx = 256.0f;
                float wavelen = 6.283185307179586f / freq;
                float low_wl  = orig_ctx * (1.0f / 32.0f);
                float high_wl = orig_ctx;
                if (wavelen > high_wl) {
                    freq /= scale_factor;
                } else if (wavelen > low_wl) {
                    float smooth = (orig_ctx / wavelen - 1.0f) * (1.0f / 31.0f);
                    if (smooth < 0.0f) smooth = 0.0f;
                    if (smooth > 1.0f) smooth = 1.0f;
                    freq = (1.0f - smooth) * (freq / scale_factor) + smooth * freq;
                }
            }
        }
        s->rope_inv_freq[idx] = freq;
    }
}

static int qmx_plan_base(const QmxSource *src, uint32_t override_max_seq, int kv_type, int need_xq16, QmxLayoutPlan *plan) {
    memset(plan, 0, sizeof(*plan));
    if (qmx_src_read(src, 0u, &plan->header, sizeof(QmxHeader)) != 0) {
        return -1;
    }

    QmxHeader *h = &plan->header;
    if (h->magic != QMX_MAGIC ||
        (h->version != QMX_VERSION_1_0 && h->version != QMX_VERSION_1_1)) {
        return -1;
    }

    if (h->header_crc32 != 0u) {
        QmxHeader tmp_hdr = *h;
        tmp_hdr.header_crc32 = 0u;
        if (qmx_crc32(&tmp_hdr, sizeof(QmxHeader)) != h->header_crc32) {
            return -1;
        }
    }

    int is_v11 = (h->version >= QMX_VERSION_1_1);
    plan->is_v11 = is_v11;
    uint64_t payload_off64 = is_v11
                           ? (((uint64_t)h->payload_offset_hi << 32u) | (uint64_t)h->payload_offset)
                           : (uint64_t)h->payload_offset;
    uint64_t payload_sz64  = is_v11
                           ? (((uint64_t)h->payload_size_hi << 32u) | (uint64_t)h->payload_size)
                           : (uint64_t)h->payload_size;
    plan->payload_offset_64 = payload_off64;
    plan->payload_size_64   = payload_sz64;

    if (h->hidden_dim == 0u || h->hidden_dim > QMX_MAX_DIM ||
        h->intermediate_dim == 0u || h->intermediate_dim > QMX_MAX_INTER_DIM ||
        h->num_layers == 0u || h->num_layers > 128u ||
        h->num_heads == 0u || h->num_kv_heads == 0u || (h->num_heads % h->num_kv_heads) != 0u ||
        h->vocab_size == 0u || h->vocab_size > 512000u ||
        h->max_seq_len == 0u || h->max_seq_len > 131072u ||
        h->recipe_count == 0u || h->recipe_count > 128u ||
        h->tensor_count == 0u || h->tensor_count > 8192u) {
        return -1;
    }

    uint32_t head_dim = (is_v11 && h->head_dim > 0u) ? (uint32_t)h->head_dim : (h->hidden_dim / h->num_heads);
    if ((!is_v11 || h->head_dim == 0u) && (h->hidden_dim % h->num_heads) != 0u) {
        return -1;
    }
    if (head_dim == 0u || head_dim > 512u || (head_dim % 2u) != 0u) {
        return -1;
    }

    uint32_t rotary_dim = (is_v11 && h->rotary_dim > 0u) ? (uint32_t)h->rotary_dim : head_dim;
    if (rotary_dim == 0u || rotary_dim > head_dim || (rotary_dim % 2u) != 0u) {
        return -1;
    }

    uint32_t q_dim  = (uint32_t)h->num_heads * head_dim;
    uint32_t kv_dim = (uint32_t)h->num_kv_heads * head_dim;
    if (q_dim == 0u || q_dim > 16384u || kv_dim == 0u || kv_dim > 16384u) {
        return -1;
    }

    plan->head_dim       = head_dim;
    plan->rotary_dim     = rotary_dim;
    plan->q_dim          = q_dim;
    plan->kv_dim         = kv_dim;
    plan->att_buf_dim    = (q_dim > h->hidden_dim) ? q_dim : h->hidden_dim;
    plan->sliding_window = is_v11 ? (uint32_t)h->sliding_window : 0u;
    plan->emb_scale      = (is_v11 && h->emb_scale > 0.0f) ? h->emb_scale : 1.0f;
    plan->logit_softcap  = (h->flags & QMX_FLAG_LOGIT_SOFTCAP)
                         ? ((is_v11 && h->logit_softcap > 0.0f) ? h->logit_softcap : 30.0f)
                         : 0.0f;
    plan->attn_softcap   = (h->flags & QMX_FLAG_ATTN_SOFTCAP)
                         ? ((is_v11 && h->attn_softcap > 0.0f) ? h->attn_softcap : 50.0f)
                         : 0.0f;

    size_t recipe_bytes = (size_t)h->recipe_count * sizeof(QmxInstruction);
    size_t tdir_bytes   = (size_t)h->tensor_count * sizeof(QmxTensorDesc);
    if (!qmx_range_valid64((uint64_t)h->recipe_offset, (uint64_t)recipe_bytes, src->total_slice) ||
        !qmx_range_valid64((uint64_t)h->tensor_dir_offset, (uint64_t)tdir_bytes, src->total_slice) ||
        !qmx_range_valid64(payload_off64, payload_sz64, src->total_slice)) {
        return -1;
    }
    if (h->tokenizer_size > 0u &&
        !qmx_range_valid64((uint64_t)h->tokenizer_offset, (uint64_t)h->tokenizer_size, src->total_slice)) {
        return -1;
    }

    plan->orig_max_seq_len = h->max_seq_len;
    if (override_max_seq > 0u && override_max_seq < h->max_seq_len) {
        h->max_seq_len = override_max_seq;
    }

    uint32_t dim        = h->hidden_dim;
    uint32_t inter_dim  = h->intermediate_dim;
    uint32_t n_layers   = h->num_layers;
    uint32_t n_kv_heads = h->num_kv_heads;
    uint32_t vocab_size = h->vocab_size;
    uint32_t seq_len    = h->max_seq_len;
    uint32_t half_rot   = rotary_dim / 2u;
    uint32_t kv_slot_d  = (kv_dim > dim) ? kv_dim : dim;
    uint32_t max_dim    = (inter_dim > plan->att_buf_dim) ? inter_dim : plan->att_buf_dim;
    if (vocab_size > max_dim) max_dim = vocab_size;
    plan->kv_slot_d = kv_slot_d;
    plan->max_dim   = max_dim;

    uint32_t max_tensor_cols = (inter_dim > plan->att_buf_dim) ? inter_dim : plan->att_buf_dim;
    uint64_t critical_bytes64 = 0u;
    uint64_t all_scales_bytes64 = 0u;
    uint32_t max_matrix_tiles = 1u;

    QmxTensorDesc batch[QMX_DESC_BATCH];
    for (uint32_t base_i = 0; base_i < h->tensor_count; base_i += QMX_DESC_BATCH) {
        uint32_t cnt = h->tensor_count - base_i;
        if (cnt > QMX_DESC_BATCH) cnt = QMX_DESC_BATCH;
        uint64_t off = (uint64_t)h->tensor_dir_offset + (uint64_t)base_i * (uint64_t)sizeof(QmxTensorDesc);
        if (qmx_src_read(src, off, batch, (size_t)cnt * sizeof(QmxTensorDesc)) != 0) {
            return -1;
        }
        for (uint32_t k = 0; k < cnt; k++) {
            uint8_t qt = batch[k].quant_type;
            uint32_t rows = batch[k].rows;
            uint32_t cols = batch[k].cols;
            if (!qmx_valid_quant_type(qt) || rows == 0u || cols == 0u ||
                rows > 512000u || cols > QMX_MAX_INTER_DIM) {
                return -1;
            }
            uint64_t d_off64 = is_v11
                             ? (((uint64_t)batch[k].data_offset_hi << 32u) | (uint64_t)batch[k].data_offset)
                             : (uint64_t)batch[k].data_offset;
            uint64_t s_off64 = is_v11
                             ? (((uint64_t)batch[k].scales_offset_hi << 32u) | (uint64_t)batch[k].scales_offset)
                             : (uint64_t)batch[k].scales_offset;
            uint64_t d_sz64  = (uint64_t)batch[k].data_size;
            uint64_t need_data = qmx_matrix_data_bytes(rows, cols, qt);
            if (d_sz64 < need_data || !qmx_range_valid64(d_off64, d_sz64, payload_sz64)) {
                return -1;
            }
            if (qt != QMX_QUANT_FP32) {
                uint64_t need_scales = qmx_matrix_scales_bytes_ex(rows, cols, qt);
                if (!qmx_range_valid64(s_off64, need_scales, payload_sz64)) {
                    return -1;
                }
                if (batch[k].pin_priority != QMX_PIN_CRITICAL) {
                    all_scales_bytes64 += qmx_align_up((size_t)need_scales, QMX_CACHE_ALIGN);
                }
                uint64_t nt = ((uint64_t)qmx_align_up(rows, QMX_TILE_ROWS) / QMX_TILE_ROWS)
                            * ((uint64_t)qmx_align_up(cols, QMX_TILE_COLS) / QMX_TILE_COLS);
                if (nt > (uint64_t)max_matrix_tiles) {
                    max_matrix_tiles = (nt > 65536u) ? 65536u : (uint32_t)nt;
                }
            }
            if (cols > max_tensor_cols) max_tensor_cols = cols;
            if (batch[k].pin_priority == QMX_PIN_CRITICAL) {
                if (qt != QMX_QUANT_FP32) {
                    critical_bytes64 += (uint64_t)qmx_align_up((size_t)qmx_matrix_scales_bytes_ex(rows, cols, qt), QMX_CACHE_ALIGN);
                }
                critical_bytes64 += (uint64_t)qmx_align_up((size_t)d_sz64, QMX_CACHE_ALIGN);
            }
        }
    }

    uint32_t max_tiles_c = (uint32_t)(qmx_align_up(max_tensor_cols, QMX_TILE_COLS) / QMX_TILE_COLS);
    size_t pad_act_cols = qmx_align_up(max_tensor_cols, QMX_TILE_COLS);

    size_t kv_entries       = (size_t)n_layers * seq_len * kv_dim;
    size_t kv_scale_entries = (size_t)n_layers * seq_len * n_kv_heads;
    size_t recipe_map_bytes = (size_t)n_layers * (size_t)h->recipe_count * sizeof(const QmxTensor *);
    size_t quant_cols_bytes = (size_t)h->recipe_count * sizeof(uint32_t);

    size_t base_arena_bytes = 0;
    base_arena_bytes += qmx_align_up(recipe_bytes, QMX_CACHE_ALIGN);
    base_arena_bytes += qmx_align_up((size_t)h->tensor_count * sizeof(QmxTensor), QMX_CACHE_ALIGN);
    base_arena_bytes += qmx_align_up(recipe_map_bytes, QMX_CACHE_ALIGN);
    base_arena_bytes += qmx_align_up(quant_cols_bytes, QMX_CACHE_ALIGN);
    base_arena_bytes += qmx_align_up((size_t)h->recipe_count, QMX_CACHE_ALIGN);
    base_arena_bytes += 2u * qmx_align_up((size_t)dim * sizeof(float), QMX_CACHE_ALIGN);
    base_arena_bytes += 2u * qmx_align_up((size_t)plan->att_buf_dim * sizeof(float), QMX_CACHE_ALIGN);
    base_arena_bytes += 2u * qmx_align_up((size_t)kv_slot_d * sizeof(float), QMX_CACHE_ALIGN);
    base_arena_bytes += 2u * qmx_align_up((size_t)inter_dim * sizeof(float), QMX_CACHE_ALIGN);
    base_arena_bytes += qmx_align_up(pad_act_cols * sizeof(int8_t), QMX_CACHE_ALIGN);
    if (need_xq16) {
        base_arena_bytes += qmx_align_up(pad_act_cols * sizeof(int16_t), QMX_CACHE_ALIGN);
    }
    base_arena_bytes += qmx_align_up((size_t)max_tiles_c * sizeof(float), QMX_CACHE_ALIGN);
    base_arena_bytes += 3u * qmx_align_up((size_t)half_rot * sizeof(float), QMX_CACHE_ALIGN);
    base_arena_bytes += qmx_align_up((size_t)seq_len * sizeof(float), QMX_CACHE_ALIGN);
    base_arena_bytes += qmx_align_up((size_t)vocab_size * sizeof(float), QMX_CACHE_ALIGN);
    if (kv_type == (int)QMX_KV_Q8) {
        base_arena_bytes += 2u * qmx_align_up(kv_entries * sizeof(int8_t), QMX_CACHE_ALIGN);
        base_arena_bytes += 2u * qmx_align_up(kv_scale_entries * sizeof(float), QMX_CACHE_ALIGN);
    } else if (kv_type == (int)QMX_KV_FP16) {
        base_arena_bytes += 2u * qmx_align_up(kv_entries * sizeof(uint16_t), QMX_CACHE_ALIGN);
    } else {
        base_arena_bytes += 2u * qmx_align_up(kv_entries * sizeof(float), QMX_CACHE_ALIGN);
    }
    base_arena_bytes += 1024u;

    plan->recipe_bytes       = recipe_bytes;
    plan->recipe_map_bytes   = recipe_map_bytes;
    plan->quant_cols_bytes   = quant_cols_bytes;
    plan->max_tensor_cols    = max_tensor_cols;
    plan->max_tiles_c        = max_tiles_c;
    plan->pad_act_cols       = pad_act_cols;
    plan->kv_entries         = kv_entries;
    plan->kv_scale_entries   = kv_scale_entries;
    plan->critical_bytes64   = critical_bytes64;
    plan->all_scales_bytes64 = all_scales_bytes64;
    plan->max_matrix_tiles   = max_matrix_tiles;
    plan->need_xq16          = need_xq16;
    plan->base_arena_bytes   = base_arena_bytes;
    return 0;
}

static void qmx_plan_stream(QmxLayoutPlan *plan, int want_stream, size_t max_ram_bytes, int is_external) {
    uint32_t stream_tile_cap = 256u;
    int pin_all_scales_in_arena = 0;
    int use_double_buf = 0;
#if defined(QMX_ENABLE_THREADS) && !defined(QMX_NO_THREADS) && !defined(QMX_NO_MALLOC)
    if (want_stream && !is_external && (max_ram_bytes == 0u || max_ram_bytes >= 16u * 1024u * 1024u)) {
        use_double_buf = 1;
    }
#else
    (void)is_external;
#endif

    if (max_ram_bytes > 0u && max_ram_bytes <= 49152u) {
        stream_tile_cap = 2u;
    } else if (max_ram_bytes > 0u && max_ram_bytes <= 196608u) {
        stream_tile_cap = 16u;
        pin_all_scales_in_arena = 1;
    } else if (max_ram_bytes > 0u && max_ram_bytes < 524288u) {
        stream_tile_cap = 64u;
        pin_all_scales_in_arena = 1;
    } else {
        pin_all_scales_in_arena = 1;
        if (max_ram_bytes >= 16u * 1024u * 1024u) {
            size_t overhead = plan->base_arena_bytes + (size_t)plan->critical_bytes64;
            size_t avail_b = (max_ram_bytes > overhead) ? (max_ram_bytes - overhead) : 0u;
            size_t div_b = use_double_buf ? 16u : 8u;
            uint32_t max_cap = use_double_buf ? 16384u : 32768u;
            uint32_t dyn_cap = (uint32_t)((avail_b / div_b) / 544u);
            if (dyn_cap > max_cap) dyn_cap = max_cap;
            if (dyn_cap > plan->max_matrix_tiles) dyn_cap = plan->max_matrix_tiles;
            if (dyn_cap > stream_tile_cap) stream_tile_cap = dyn_cap;
        } else if (max_ram_bytes == 0u) {
            uint32_t big_cap = use_double_buf ? 8192u : 16384u;
            if (plan->payload_size_64 >= 64ULL * 1024ULL * 1024ULL) {
                stream_tile_cap = (plan->max_matrix_tiles < big_cap) ? plan->max_matrix_tiles : big_cap;
            } else if (plan->max_matrix_tiles < stream_tile_cap) {
                stream_tile_cap = plan->max_matrix_tiles;
            }
        }
    }
    if (stream_tile_cap < plan->max_tiles_c && (max_ram_bytes == 0u || max_ram_bytes > 49152u)) {
        stream_tile_cap = plan->max_tiles_c;
    }
    if (stream_tile_cap == 0u) stream_tile_cap = 1u;

    if (pin_all_scales_in_arena && max_ram_bytes > 0u &&
        plan->all_scales_bytes64 > (uint64_t)(max_ram_bytes / 2u)) {
        pin_all_scales_in_arena = 0;
    }
    if (pin_all_scales_in_arena && sizeof(size_t) < 8u && plan->all_scales_bytes64 > 0x10000000ULL) {
        pin_all_scales_in_arena = 0;
    }

    uint32_t emb_cache_cap = QMX_EMB_CACHE_SIZE;
    if (max_ram_bytes > 0u && max_ram_bytes <= 65536u) {
        emb_cache_cap = 4u;
    } else if (max_ram_bytes > 0u && max_ram_bytes <= 262144u) {
        emb_cache_cap = 16u;
    }

    size_t stream_extra_bytes = 0u;
    if (want_stream) {
        stream_extra_bytes += qmx_align_up((size_t)plan->critical_bytes64, QMX_CACHE_ALIGN);
        if (pin_all_scales_in_arena) {
            stream_extra_bytes += qmx_align_up((size_t)plan->all_scales_bytes64, QMX_CACHE_ALIGN);
        }
        stream_extra_bytes += qmx_align_up((size_t)stream_tile_cap * 32u, QMX_CACHE_ALIGN);
        stream_extra_bytes += qmx_align_up((size_t)stream_tile_cap * 512u, QMX_CACHE_ALIGN);
        if (use_double_buf) {
            stream_extra_bytes += qmx_align_up((size_t)stream_tile_cap * 512u, QMX_CACHE_ALIGN);
        }
        stream_extra_bytes += qmx_align_up((size_t)emb_cache_cap * sizeof(int32_t), QMX_CACHE_ALIGN);
        stream_extra_bytes += qmx_align_up((size_t)emb_cache_cap * sizeof(uint32_t), QMX_CACHE_ALIGN);
        stream_extra_bytes += qmx_align_up((size_t)emb_cache_cap * plan->header.hidden_dim * sizeof(float), QMX_CACHE_ALIGN);
        stream_extra_bytes += 512u;
    }

    plan->stream_tile_cap         = stream_tile_cap;
    plan->emb_cache_cap           = emb_cache_cap;
    plan->pin_all_scales_in_arena = pin_all_scales_in_arena;
    plan->use_double_buf          = use_double_buf;
    plan->stream_extra_bytes      = stream_extra_bytes;
}

size_t qmx_calc_arena_size(const QmxReader *reader, const QmxLoadParams *params, int is_xip) {
    if (!reader || !reader->read || reader->size < sizeof(QmxHeader)) return 0u;
    QmxLoadParams p = params ? *params : qmx_load_params_default();
    QmxSource src;
    memset(&src, 0, sizeof(src));
    src.reader      = *reader;
    src.total_slice = reader->size;

    QmxLayoutPlan plan;
    int kv_type = qmx_normalize_kv_type((int)p.kv_type);
    int need_xq16 = qmx_plan_needs_xq16(p.backend, p.max_ram_bytes, 1);
    if (qmx_plan_base(&src, p.override_max_seq, kv_type, need_xq16, &plan) != 0) return 0u;
    qmx_plan_stream(&plan, is_xip ? 0 : 1, p.max_ram_bytes, 1);
    return plan.base_arena_bytes + plan.stream_extra_bytes + QMX_CACHE_ALIGN;
}

size_t qmx_calc_arena_size_memory(const void *cartridge_mem, size_t cartridge_size, const QmxLoadParams *params) {
    if (!cartridge_mem || cartridge_size < sizeof(QmxHeader)) return 0u;
    QmxLoadParams p = params ? *params : qmx_load_params_default();
    QmxSource src;
    memset(&src, 0, sizeof(src));
    src.mem         = (const uint8_t *)cartridge_mem;
    src.total_slice = (uint64_t)cartridge_size;

    QmxLayoutPlan plan;
    int kv_type = qmx_normalize_kv_type((int)p.kv_type);
    int need_xq16 = qmx_plan_needs_xq16(p.backend, p.max_ram_bytes, 1);
    if (qmx_plan_base(&src, p.override_max_seq, kv_type, need_xq16, &plan) != 0) return 0u;
    qmx_plan_stream(&plan, 0, p.max_ram_bytes, 1);
    return plan.base_arena_bytes + QMX_CACHE_ALIGN;
}

static int qmx_load_internal(QmxModel *model, QmxSource *src, const QmxLoadParams *params) {
    QmxLoadParams p = params ? *params : qmx_load_params_default();
    memset(model, 0, sizeof(QmxModel));
    model->kv_type = qmx_normalize_kv_type((int)p.kv_type);
    model->use_kv8 = (model->kv_type == (int)QMX_KV_Q8) ? 1 : 0;
    model->last_stream_layer = -1;
    model->num_threads = 1;
    qmx_bind_backend_ops(model, p.backend);

    QmxLayoutPlan plan;
    int need_xq16 = qmx_plan_needs_xq16(model->backend, p.max_ram_bytes, p.arena_buf != NULL);
    if (qmx_plan_base(src, p.override_max_seq, model->kv_type, need_xq16, &plan) != 0) {
        return -1;
    }

    model->header            = plan.header;
    model->orig_max_seq_len  = plan.orig_max_seq_len;
    model->max_dim           = plan.max_dim;
    model->head_dim          = plan.head_dim;
    model->rotary_dim        = plan.rotary_dim;
    model->q_dim             = plan.q_dim;
    model->kv_dim            = plan.kv_dim;
    model->att_buf_dim       = plan.att_buf_dim;
    model->sliding_window    = plan.sliding_window;
    model->emb_scale         = plan.emb_scale;
    model->logit_softcap     = plan.logit_softcap;
    model->attn_softcap      = plan.attn_softcap;
    model->payload_offset_64 = plan.payload_offset_64;
    model->payload_size_64   = plan.payload_size_64;

    QmxHeader *h = &model->header;
    int want_stream = 0;

    if (src->mem != NULL) {
        model->payload_mem      = (void *)(uintptr_t)(src->mem + (size_t)plan.payload_offset_64);
        model->payload_mem_size = (size_t)plan.payload_size_64;
        model->is_xip           = 1;
        want_stream             = 0;
    } else {
        want_stream = p.force_stream ? 1 : 0;
        if (p.arena_buf != NULL) {
            want_stream = 1;
        }
        if (!want_stream && p.max_ram_bytes > 0u &&
            (plan.payload_size_64 + (uint64_t)plan.base_arena_bytes > (uint64_t)p.max_ram_bytes)) {
            want_stream = 1;
        }
#if QMX_HAS_MMAP || !defined(QMX_NO_MALLOC)
        int fits_va = (sizeof(size_t) >= 8u) || (plan.payload_size_64 <= 0x40000000ULL);
        if (!want_stream && fits_va) {
#if QMX_HAS_MMAP
            if (src->mmap_path != NULL && src->mmap_base_offset == 0u &&
                (plan.payload_offset_64 % QMX_PAGE_ALIGN) == 0u) {
                int fd = open(src->mmap_path, O_RDONLY);
                if (fd >= 0) {
                    void *mapped = mmap(NULL, (size_t)plan.payload_size_64, PROT_READ, MAP_PRIVATE,
                                        fd, (off_t)plan.payload_offset_64);
                    close(fd);
                    if (mapped != MAP_FAILED) {
#if defined(POSIX_MADV_WILLNEED)
                        (void)posix_madvise(mapped, (size_t)plan.payload_size_64, POSIX_MADV_WILLNEED);
#endif
                        model->payload_mem = mapped;
                        model->payload_mem_size = (size_t)plan.payload_size_64;
                        model->is_mmap = 1;
                    }
                }
            }
#endif
#ifndef QMX_NO_MALLOC
            if (!model->payload_mem && plan.payload_size_64 <= (uint64_t)(~(size_t)0)) {
                model->payload_mem = malloc((size_t)plan.payload_size_64);
                if (model->payload_mem) {
                    if (qmx_src_read(src, plan.payload_offset_64, model->payload_mem, (size_t)plan.payload_size_64) == 0) {
                        model->payload_mem_size = (size_t)plan.payload_size_64;
                    } else {
                        free(model->payload_mem);
                        model->payload_mem = NULL;
                    }
                }
            }
#endif
            if (!model->payload_mem) {
                want_stream = 1;
            }
        } else if (!fits_va) {
            want_stream = 1;
        }
#else
        want_stream = 1;
#endif
    }

    model->stream_mode = want_stream;
    qmx_plan_stream(&plan, want_stream, p.max_ram_bytes, p.arena_buf != NULL);

    if (model->stream_mode) {
        model->reader = src->reader;
        src->reader.close = NULL;
    }

    size_t need_arena = plan.base_arena_bytes + plan.stream_extra_bytes;
    if (p.arena_buf != NULL) {
        if (qmx_arena_init_external(&model->arena, p.arena_buf, p.arena_size) != 0 ||
            model->arena.capacity < need_arena) {
            qmx_free(model);
            return -1;
        }
    } else {
        if (qmx_arena_init(&model->arena, need_arena) != 0) {
            qmx_free(model);
            return -1;
        }
    }

    model->recipe = (QmxInstruction *)qmx_arena_alloc(&model->arena, plan.recipe_bytes, QMX_CACHE_ALIGN);
    model->tensors = (QmxTensor *)qmx_arena_alloc(&model->arena, (size_t)h->tensor_count * sizeof(QmxTensor), QMX_CACHE_ALIGN);
    model->recipe_tensors = (const QmxTensor **)qmx_arena_alloc(&model->arena, plan.recipe_map_bytes, QMX_CACHE_ALIGN);
    model->recipe_quant_cols = (uint32_t *)qmx_arena_alloc(&model->arena, plan.quant_cols_bytes, QMX_CACHE_ALIGN);
    model->recipe_fuse = (uint8_t *)qmx_arena_alloc(&model->arena, (size_t)h->recipe_count, QMX_CACHE_ALIGN);
    if (!model->recipe || !model->tensors || !model->recipe_tensors || !model->recipe_quant_cols || !model->recipe_fuse) {
        qmx_free(model);
        return -1;
    }

    if (qmx_src_read(src, (uint64_t)h->recipe_offset, model->recipe, plan.recipe_bytes) != 0) {
        qmx_free(model);
        return -1;
    }

    const uint8_t *pay_base_ptr = (const uint8_t *)model->payload_mem;
    QmxTensorDesc batch[QMX_DESC_BATCH];
    for (uint32_t base_i = 0; base_i < h->tensor_count; base_i += QMX_DESC_BATCH) {
        uint32_t cnt = h->tensor_count - base_i;
        if (cnt > QMX_DESC_BATCH) cnt = QMX_DESC_BATCH;
        uint64_t off = (uint64_t)h->tensor_dir_offset + (uint64_t)base_i * (uint64_t)sizeof(QmxTensorDesc);
        if (qmx_src_read(src, off, batch, (size_t)cnt * sizeof(QmxTensorDesc)) != 0) {
            qmx_free(model);
            return -1;
        }
        for (uint32_t k = 0; k < cnt; k++) {
            QmxTensor *t = &model->tensors[base_i + k];
            t->tensor_id     = batch[k].tensor_id;
            t->layer_idx     = batch[k].layer_idx;
            t->quant_type    = batch[k].quant_type;
            t->pin_priority  = batch[k].pin_priority;
            t->tile_rows     = batch[k].tile_rows;
            t->rows          = batch[k].rows;
            t->cols          = batch[k].cols;
            t->scales_offset = plan.is_v11
                             ? (((uint64_t)batch[k].scales_offset_hi << 32u) | (uint64_t)batch[k].scales_offset)
                             : (uint64_t)batch[k].scales_offset;
            t->data_offset   = plan.is_v11
                             ? (((uint64_t)batch[k].data_offset_hi << 32u) | (uint64_t)batch[k].data_offset)
                             : (uint64_t)batch[k].data_offset;
            t->data_size     = (uint64_t)batch[k].data_size;
            if (pay_base_ptr != NULL) {
                t->scales = (t->quant_type == QMX_QUANT_FP32) ? NULL : (const float *)(pay_base_ptr + (size_t)t->scales_offset);
                t->data = (const void *)(pay_base_ptr + (size_t)t->data_offset);
                model->pinned_tensors++;
            } else {
                t->scales = NULL;
                t->data = NULL;
            }
        }
    }

    if (validate_and_bind_recipe(model, plan.max_tensor_cols) != 0) {
        qmx_free(model);
        return -1;
    }

    uint32_t dim       = h->hidden_dim;
    uint32_t inter_dim = h->intermediate_dim;
    uint32_t half_rot  = model->rotary_dim / 2u;
    uint32_t seq_len   = h->max_seq_len;
    uint32_t vocab_sz  = h->vocab_size;

    QmxRunState *s = &model->state;
    s->slots[QMX_BUF_X]       = (float *)qmx_arena_alloc(&model->arena, (size_t)dim * sizeof(float), QMX_CACHE_ALIGN);
    s->slots[QMX_BUF_XB]      = (float *)qmx_arena_alloc(&model->arena, (size_t)model->att_buf_dim * sizeof(float), QMX_CACHE_ALIGN);
    s->slots[QMX_BUF_XB2]     = (float *)qmx_arena_alloc(&model->arena, (size_t)dim * sizeof(float), QMX_CACHE_ALIGN);
    s->slots[QMX_BUF_Q]       = (float *)qmx_arena_alloc(&model->arena, (size_t)model->att_buf_dim * sizeof(float), QMX_CACHE_ALIGN);
    s->slots[QMX_BUF_K]       = (float *)qmx_arena_alloc(&model->arena, (size_t)plan.kv_slot_d * sizeof(float), QMX_CACHE_ALIGN);
    s->slots[QMX_BUF_V]       = (float *)qmx_arena_alloc(&model->arena, (size_t)plan.kv_slot_d * sizeof(float), QMX_CACHE_ALIGN);
    s->slots[QMX_BUF_HB_GATE] = (float *)qmx_arena_alloc(&model->arena, (size_t)inter_dim * sizeof(float), QMX_CACHE_ALIGN);
    s->slots[QMX_BUF_HB_UP]   = (float *)qmx_arena_alloc(&model->arena, (size_t)inter_dim * sizeof(float), QMX_CACHE_ALIGN);

    s->xq            = (int8_t *)qmx_arena_alloc(&model->arena, plan.pad_act_cols * sizeof(int8_t), QMX_CACHE_ALIGN);
    s->xq16          = plan.need_xq16
                     ? (int16_t *)qmx_arena_alloc(&model->arena, plan.pad_act_cols * sizeof(int16_t), QMX_CACHE_ALIGN)
                     : NULL;
    s->x_scales      = (float *)qmx_arena_alloc(&model->arena, (size_t)plan.max_tiles_c * sizeof(float), QMX_CACHE_ALIGN);
    s->rope_inv_freq = (float *)qmx_arena_alloc(&model->arena, (size_t)half_rot * sizeof(float), QMX_CACHE_ALIGN);
    s->rope_cos      = (float *)qmx_arena_alloc(&model->arena, (size_t)half_rot * sizeof(float), QMX_CACHE_ALIGN);
    s->rope_sin      = (float *)qmx_arena_alloc(&model->arena, (size_t)half_rot * sizeof(float), QMX_CACHE_ALIGN);
    s->att           = (float *)qmx_arena_alloc(&model->arena, (size_t)seq_len * sizeof(float), QMX_CACHE_ALIGN);
    s->logits        = (float *)qmx_arena_alloc(&model->arena, (size_t)vocab_sz * sizeof(float), QMX_CACHE_ALIGN);

    if (model->kv_type == (int)QMX_KV_Q8) {
        s->key_cache_q8   = (int8_t *)qmx_arena_alloc(&model->arena, plan.kv_entries * sizeof(int8_t), QMX_CACHE_ALIGN);
        s->value_cache_q8 = (int8_t *)qmx_arena_alloc(&model->arena, plan.kv_entries * sizeof(int8_t), QMX_CACHE_ALIGN);
        s->key_scales     = (float *)qmx_arena_alloc(&model->arena, plan.kv_scale_entries * sizeof(float), QMX_CACHE_ALIGN);
        s->value_scales   = (float *)qmx_arena_alloc(&model->arena, plan.kv_scale_entries * sizeof(float), QMX_CACHE_ALIGN);
        if (!s->value_scales) { qmx_free(model); return -1; }
    } else if (model->kv_type == (int)QMX_KV_FP16) {
        s->key_cache_f16   = (uint16_t *)qmx_arena_alloc(&model->arena, plan.kv_entries * sizeof(uint16_t), QMX_CACHE_ALIGN);
        s->value_cache_f16 = (uint16_t *)qmx_arena_alloc(&model->arena, plan.kv_entries * sizeof(uint16_t), QMX_CACHE_ALIGN);
        if (!s->value_cache_f16) { qmx_free(model); return -1; }
    } else {
        s->key_cache   = (float *)qmx_arena_alloc(&model->arena, plan.kv_entries * sizeof(float), QMX_CACHE_ALIGN);
        s->value_cache = (float *)qmx_arena_alloc(&model->arena, plan.kv_entries * sizeof(float), QMX_CACHE_ALIGN);
        if (!s->value_cache) { qmx_free(model); return -1; }
    }

    if (model->stream_mode) {
        s->stream_tile_cap   = plan.stream_tile_cap;
        s->stream_scales_buf = (float *)qmx_arena_alloc(&model->arena, (size_t)plan.stream_tile_cap * 32u, QMX_CACHE_ALIGN);
        s->stream_data_buf   = (uint8_t *)qmx_arena_alloc(&model->arena, (size_t)plan.stream_tile_cap * 512u, QMX_CACHE_ALIGN);
        s->stream_data_buf2  = plan.use_double_buf
                             ? (uint8_t *)qmx_arena_alloc(&model->arena, (size_t)plan.stream_tile_cap * 512u, QMX_CACHE_ALIGN)
                             : NULL;
        s->emb_cache_cap     = plan.emb_cache_cap;
        s->emb_cache_tokens  = (int32_t *)qmx_arena_alloc(&model->arena, (size_t)plan.emb_cache_cap * sizeof(int32_t), QMX_CACHE_ALIGN);
        s->emb_cache_lru     = (uint32_t *)qmx_arena_alloc(&model->arena, (size_t)plan.emb_cache_cap * sizeof(uint32_t), QMX_CACHE_ALIGN);
        s->emb_cache_vecs    = (float *)qmx_arena_alloc(&model->arena, (size_t)plan.emb_cache_cap * dim * sizeof(float), QMX_CACHE_ALIGN);
        if (!s->emb_cache_vecs) { qmx_free(model); return -1; }

        for (uint32_t i = 0; i < plan.emb_cache_cap; i++) {
            s->emb_cache_tokens[i] = -1;
            s->emb_cache_lru[i] = 0;
        }

        uint64_t pay_base = model->payload_offset_64;
        for (uint32_t i = 0; i < h->tensor_count; i++) {
            if (model->tensors[i].pin_priority == QMX_PIN_CRITICAL) {
                if (pin_tensor_from_reader(model, &model->reader, pay_base, model->arena.base_ptr,
                                           &model->tensors[i], &model->arena.offset, model->arena.capacity) != 0) {
                    qmx_free(model);
                    return -1;
                }
            }
        }

        if (plan.pin_all_scales_in_arena) {
            for (uint32_t i = 0; i < h->tensor_count; i++) {
                if (model->tensors[i].scales == NULL && model->tensors[i].quant_type != QMX_QUANT_FP32) {
                    if (pin_tensor_scales_from_reader(&model->reader, pay_base, model->arena.base_ptr,
                                                      &model->tensors[i], &model->arena.offset, model->arena.capacity) != 0) {
                        qmx_free(model);
                        return -1;
                    }
                }
            }
        }

#ifndef QMX_NO_MALLOC
        if (p.arena_buf == NULL && p.max_ram_bytes > model->arena.offset + 2048u) {
            size_t spare_budget = p.max_ram_bytes - model->arena.offset;
            uint64_t unpinned_need64 = 0u;
            for (uint32_t i = 0; i < h->tensor_count; i++) {
                if (model->tensors[i].data == NULL) {
                    if (model->tensors[i].quant_type != QMX_QUANT_FP32 && model->tensors[i].scales == NULL) {
                        unpinned_need64 = (uint64_t)qmx_align_up((size_t)unpinned_need64, QMX_CACHE_ALIGN)
                                        + qmx_matrix_scales_bytes_ex(model->tensors[i].rows, model->tensors[i].cols, model->tensors[i].quant_type);
                    }
                    unpinned_need64 = (uint64_t)qmx_align_up((size_t)unpinned_need64, QMX_CACHE_ALIGN)
                                    + model->tensors[i].data_size;
                    if (unpinned_need64 > (uint64_t)spare_budget) {
                        unpinned_need64 = (uint64_t)spare_budget;
                        break;
                    }
                }
            }
            size_t unpinned_need = (size_t)unpinned_need64;
            if (unpinned_need > 0u && spare_budget > unpinned_need + QMX_CACHE_ALIGN) {
                spare_budget = unpinned_need + QMX_CACHE_ALIGN;
            }
            model->pin_pool = (unpinned_need > 0u) ? (uint8_t *)malloc(spare_budget + QMX_CACHE_ALIGN) : NULL;
            if (model->pin_pool) {
                uintptr_t raw_addr = (uintptr_t)model->pin_pool;
                uintptr_t al_addr = (raw_addr + (QMX_CACHE_ALIGN - 1u)) & ~(uintptr_t)(QMX_CACHE_ALIGN - 1u);
                uint8_t *pin_base = (uint8_t *)al_addr;
                size_t pool_off = 0;
                for (uint8_t prio = QMX_PIN_LATENCY_SENS; prio <= QMX_PIN_STREAM_ONLY; prio++) {
                    for (uint32_t i = 0; i < h->tensor_count; i++) {
                        if (model->tensors[i].data == NULL && model->tensors[i].pin_priority == prio) {
                            (void)pin_tensor_from_reader(model, &model->reader, pay_base, pin_base,
                                                         &model->tensors[i], &pool_off, spare_budget);
                        }
                    }
                }
                model->pin_pool_size = pool_off;
            }
        }
#endif
    }

    init_rope_inv_freq(model);
    return 0;
}

int qmx_load_from_reader(QmxModel *model, QmxReader reader, const QmxLoadParams *params) {
    if (!model || !reader.read || reader.size < sizeof(QmxHeader)) {
        if (reader.close) reader.close(reader.ctx);
        return -1;
    }
    QmxSource src;
    memset(&src, 0, sizeof(src));
    src.reader      = reader;
    src.total_slice = reader.size;
    int rc = qmx_load_internal(model, &src, params);
    if (src.reader.close) {
        src.reader.close(src.reader.ctx);
    }
    return rc;
}

int qmx_load_from_memory_ex(QmxModel *model, const void *cartridge_mem, size_t cartridge_size,
                            const QmxLoadParams *params) {
    if (!model || !cartridge_mem || cartridge_size < sizeof(QmxHeader)) return -1;
    QmxSource src;
    memset(&src, 0, sizeof(src));
    src.mem         = (const uint8_t *)cartridge_mem;
    src.total_slice = (uint64_t)cartridge_size;
    return qmx_load_internal(model, &src, params);
}

int qmx_load_from_memory(QmxModel *model, const void *cartridge_mem, size_t cartridge_size,
                         uint32_t override_max_seq, QmxBackend backend, int kv_mode) {
    QmxLoadParams p = qmx_load_params_default();
    p.override_max_seq = override_max_seq;
    p.backend          = backend;
    p.kv_type          = (QmxKvType)qmx_normalize_kv_type(kv_mode);
    return qmx_load_from_memory_ex(model, cartridge_mem, cartridge_size, &p);
}

#ifndef QMX_NO_STDIO

typedef struct {
    FILE    *fp;
    uint64_t base_offset;
    uint64_t size;
    uint64_t cur_pos;
    int      pos_valid;
} QmxFileReaderCtx;

static int qmx_file_reader_read(void *ctx, uint64_t offset, void *dst, size_t size) {
    QmxFileReaderCtx *fc = (QmxFileReaderCtx *)ctx;
    if (!fc || !fc->fp) return -1;
    if (!qmx_range_valid64(offset, (uint64_t)size, fc->size)) return -1;
    if (size == 0u) return 0;
    uint64_t abs_off = fc->base_offset + offset;
    if (!fc->pos_valid || fc->cur_pos != abs_off) {
        if (qmx_fseek64(fc->fp, (int64_t)abs_off, SEEK_SET) != 0) {
            fc->pos_valid = 0;
            return -1;
        }
        fc->cur_pos = abs_off;
        fc->pos_valid = 1;
    }
    if (fread(dst, 1, size, fc->fp) != size) {
        fc->pos_valid = 0;
        return -1;
    }
    fc->cur_pos += (uint64_t)size;
    return 0;
}

static void qmx_file_reader_close(void *ctx) {
    QmxFileReaderCtx *fc = (QmxFileReaderCtx *)ctx;
    if (!fc) return;
    if (fc->fp) fclose(fc->fp);
#ifndef QMX_NO_MALLOC
    free(fc);
#endif
}

int qmx_load_ex(QmxModel *model, const char *path, uint64_t base_offset, uint64_t slice_size,
                const QmxLoadParams *params) {
#ifdef QMX_NO_MALLOC
    (void)model; (void)path; (void)base_offset; (void)slice_size; (void)params;
    return -1;
#else
    if (!model || !path) return -1;
    QmxLoadParams p = params ? *params : qmx_load_params_default();
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    size_t init_vbuf = 65536u;
    if (p.max_ram_bytes > 0u && p.max_ram_bytes <= 65536u) {
        init_vbuf = 4096u;
    } else if (p.max_ram_bytes > 0u && p.max_ram_bytes <= 262144u) {
        init_vbuf = 16384u;
    } else if (p.max_ram_bytes >= 16u * 1024u * 1024u) {
        init_vbuf = 1048576u;
    }
    setvbuf(f, NULL, _IOFBF, init_vbuf);

    int64_t fsz = qmx_file_size64(f);
    if (fsz < 0 || !qmx_range_valid64(base_offset, (uint64_t)sizeof(QmxHeader), (uint64_t)fsz)) {
        fclose(f);
        return -1;
    }

    uint64_t total_slice = (slice_size > 0u && qmx_range_valid64(base_offset, slice_size, (uint64_t)fsz))
                         ? slice_size
                         : ((uint64_t)fsz - base_offset);

    QmxFileReaderCtx *fc = (QmxFileReaderCtx *)calloc(1, sizeof(QmxFileReaderCtx));
    if (!fc) {
        fclose(f);
        return -1;
    }
    fc->fp          = f;
    fc->base_offset = base_offset;
    fc->size        = total_slice;
    fc->pos_valid   = 0;

    QmxSource src;
    memset(&src, 0, sizeof(src));
    src.reader.read      = qmx_file_reader_read;
    src.reader.close     = qmx_file_reader_close;
    src.reader.ctx       = fc;
    src.reader.size      = total_slice;
    src.total_slice      = total_slice;
    src.mmap_path        = path;
    src.mmap_base_offset = base_offset;

    int rc = qmx_load_internal(model, &src, &p);
    if (src.reader.close) {
        src.reader.close(src.reader.ctx);
    }

    if (rc == 0 && model->stream_mode && model->state.stream_tile_cap >= 2048u && init_vbuf < 1048576u) {
        FILE *f2 = fopen(path, "rb");
        if (f2) {
            setvbuf(f2, NULL, _IOFBF, 1048576u);
            QmxFileReaderCtx *active_fc = (QmxFileReaderCtx *)model->reader.ctx;
            fclose(active_fc->fp);
            active_fc->fp = f2;
            active_fc->pos_valid = 0;
        }
    }
    return rc;
#endif
}

int qmx_load(QmxModel *model, const char *path, uint32_t override_max_seq,
             QmxBackend backend, int kv_mode, int force_stream, size_t max_ram_bytes) {
    return qmx_load_from_offset(model, path, 0u, 0u, override_max_seq, backend, kv_mode, force_stream, max_ram_bytes);
}

int qmx_load_from_offset(QmxModel *model, const char *path, uint64_t base_offset, uint64_t slice_size,
                         uint32_t override_max_seq, QmxBackend backend,
                         int kv_mode, int force_stream, size_t max_ram_bytes) {
    QmxLoadParams p = qmx_load_params_default();
    p.override_max_seq = override_max_seq;
    p.backend          = backend;
    p.kv_type          = (QmxKvType)qmx_normalize_kv_type(kv_mode);
    p.force_stream     = force_stream;
    p.max_ram_bytes    = max_ram_bytes;
    return qmx_load_ex(model, path, base_offset, slice_size, &p);
}

#endif

void qmx_get_model_info(const QmxModel *model, QmxModelInfo *out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    if (!model) return;
    out->hidden_dim       = model->header.hidden_dim;
    out->intermediate_dim = model->header.intermediate_dim;
    out->num_layers       = model->header.num_layers;
    out->num_heads        = model->header.num_heads;
    out->num_kv_heads     = model->header.num_kv_heads;
    out->head_dim         = model->head_dim;
    out->rotary_dim       = model->rotary_dim;
    out->vocab_size       = model->header.vocab_size;
    out->max_seq_len      = model->header.max_seq_len;
    out->orig_max_seq_len = model->orig_max_seq_len;
    out->sliding_window   = model->sliding_window;
    out->primary_quant    = model->header.primary_quant;
    out->kv_type          = (QmxKvType)model->kv_type;
    out->backend          = model->backend;
    out->backend_name     = qmx_backend_name(model->backend);
    out->stream_mode      = model->stream_mode;
    out->is_mmap          = model->is_mmap;
    out->is_xip           = model->is_xip;
    out->tensor_count     = model->header.tensor_count;
    out->tokenizer_offset = model->header.tokenizer_offset;
    out->tokenizer_size   = model->header.tokenizer_size;
    out->num_threads      = model->num_threads;
}

void qmx_get_stats(const QmxModel *model, QmxStats *out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    if (!model) return;
    size_t kv_entries = (size_t)model->header.num_layers * model->header.max_seq_len * model->kv_dim;
    size_t kv_bytes = 0u;
    if (model->kv_type == (int)QMX_KV_Q8) {
        size_t sc_entries = (size_t)model->header.num_layers * model->header.max_seq_len * model->header.num_kv_heads;
        kv_bytes = 2u * kv_entries * sizeof(int8_t) + 2u * sc_entries * sizeof(float);
    } else if (model->kv_type == (int)QMX_KV_FP16) {
        kv_bytes = 2u * kv_entries * sizeof(uint16_t);
    } else {
        kv_bytes = 2u * kv_entries * sizeof(float);
    }
    out->arena_used        = model->arena.offset;
    out->arena_capacity    = model->arena.capacity;
    out->kv_cache_bytes    = kv_bytes;
    out->pin_pool_bytes    = model->pin_pool_size;
    out->pinned_tensors    = model->pinned_tensors;
    out->tensor_count      = model->header.tensor_count;
    out->disk_reads        = model->disk_reads;
    out->disk_bytes_read   = model->disk_bytes_read;
    out->emb_cache_hits    = model->emb_cache_hits;
    out->emb_cache_misses  = model->emb_cache_misses;
    out->last_stream_layer = model->last_stream_layer;
}

void qmx_reset_stats(QmxModel *model) {
    if (!model) return;
    model->disk_reads        = 0u;
    model->disk_bytes_read   = 0u;
    model->emb_cache_hits    = 0u;
    model->emb_cache_misses  = 0u;
    model->last_stream_layer = -1;
}

void qmx_free(QmxModel *model) {
    if (!model) return;
    if (model->pool) {
        qmx_pool_destroy(model->pool);
        model->pool = NULL;
    }
#ifndef QMX_NO_MALLOC
    if (model->batch_ws) {
        free(model->batch_ws);
        model->batch_ws = NULL;
        model->batch_ws_cap = 0;
    }
    if (model->mt_att_buf) {
        free(model->mt_att_buf);
        model->mt_att_buf = NULL;
        model->mt_att_cap = 0;
    }
#endif
    qmx_arena_free(&model->arena);
#ifndef QMX_NO_MALLOC
    if (model->pin_pool) {
        free(model->pin_pool);
        model->pin_pool = NULL;
    }
#endif
    if (model->reader.close) {
        model->reader.close(model->reader.ctx);
    }
    memset(&model->reader, 0, sizeof(model->reader));
    if (model->payload_mem && !model->is_xip) {
#if QMX_HAS_MMAP
        if (model->is_mmap) munmap(model->payload_mem, model->payload_mem_size);
#ifndef QMX_NO_MALLOC
        else free(model->payload_mem);
#endif
#elif !defined(QMX_NO_MALLOC)
        free(model->payload_mem);
#endif
    }
    model->payload_mem = NULL;
}
