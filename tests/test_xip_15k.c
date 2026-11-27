/*
 * Copyright 2026 1WhiteFox
 * SPDX-License-Identifier: Apache-2.0
 */

#ifdef QMX_TEST_AMALGAMATED
#include "qmx.h"
#else
#include "../include/qmx_gen.h"
#endif
#include <stdio.h>
#include <string.h>
#include <math.h>

static int g_failures = 0;

#define EXPECT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "[FAIL] %s (line %d)\n", msg, __LINE__); \
        g_failures++; \
    } \
} while (0)

static QMX_ALIGNED(64) uint8_t s_flash_rom[24u * 1024u];
static QMX_ALIGNED(64) uint8_t s_mcu_sram[32u * 1024u];

static size_t align64(size_t v) {
    return (v + 63u) & ~(size_t)63u;
}

static size_t build_15k_xip_cartridge(uint8_t *rom, size_t cap, uint32_t *out_params) {
    const uint32_t dim       = 32u;
    const uint32_t inter_dim = 64u;
    const uint32_t n_layers  = 1u;
    const uint32_t n_heads   = 4u;
    const uint32_t n_kv      = 2u;
    const uint32_t head_dim  = 8u;
    const uint32_t kv_dim    = n_kv * head_dim;
    const uint32_t vocab     = 192u;
    const uint32_t max_seq   = 64u;

    memset(rom, 0, cap);

    static const QmxInstruction recipe[18] = {
        { QMX_OP_RMSNORM,      0, QMX_TID_ATTN_NORM, QMX_BUF_X,       QMX_BUF_XB      },
        { QMX_OP_QUANT_ACT,    0, QMX_TID_NONE,      QMX_BUF_XB,      QMX_BUF_XB      },
        { QMX_OP_MATMUL,       0, QMX_TID_ATTN_Q,    QMX_BUF_XB,      QMX_BUF_Q       },
        { QMX_OP_MATMUL,       0, QMX_TID_ATTN_K,    QMX_BUF_XB,      QMX_BUF_K       },
        { QMX_OP_MATMUL,       0, QMX_TID_ATTN_V,    QMX_BUF_XB,      QMX_BUF_V       },
        { QMX_OP_ROPE,         0, QMX_TID_NONE,      QMX_BUF_Q,       QMX_BUF_K       },
        { QMX_OP_ATTN,         0, QMX_TID_NONE,      QMX_BUF_Q,       QMX_BUF_XB      },
        { QMX_OP_QUANT_ACT,    0, QMX_TID_NONE,      QMX_BUF_XB,      QMX_BUF_XB      },
        { QMX_OP_MATMUL,       0, QMX_TID_ATTN_WO,   QMX_BUF_XB,      QMX_BUF_XB2     },
        { QMX_OP_RESIDUAL_ADD, 0, QMX_TID_NONE,      QMX_BUF_XB2,     QMX_BUF_X       },
        { QMX_OP_RMSNORM,      0, QMX_TID_FFN_NORM,  QMX_BUF_X,       QMX_BUF_XB      },
        { QMX_OP_QUANT_ACT,    0, QMX_TID_NONE,      QMX_BUF_XB,      QMX_BUF_XB      },
        { QMX_OP_MATMUL,       0, QMX_TID_FFN_GATE,  QMX_BUF_XB,      QMX_BUF_HB_GATE },
        { QMX_OP_MATMUL,       0, QMX_TID_FFN_UP,    QMX_BUF_XB,      QMX_BUF_HB_UP   },
        { QMX_OP_SWIGLU,       0, QMX_TID_NONE,      QMX_BUF_HB_GATE, QMX_BUF_HB_UP   },
        { QMX_OP_QUANT_ACT,    0, QMX_TID_NONE,      QMX_BUF_HB_UP,   QMX_BUF_HB_UP   },
        { QMX_OP_MATMUL,       0, QMX_TID_FFN_DOWN,  QMX_BUF_HB_UP,   QMX_BUF_XB2     },
        { QMX_OP_RESIDUAL_ADD, 0, QMX_TID_NONE,      QMX_BUF_XB2,     QMX_BUF_X       }
    };

    typedef struct {
        const char *name;
        uint16_t    tid;
        uint16_t    layer;
        uint8_t     qtype;
        uint8_t     prio;
        uint32_t    rows;
        uint32_t    cols;
    } Spec;

    const Spec specs[11] = {
        { "token_emb.weight", QMX_TID_TOKEN_EMB, 0xFFFFu, QMX_QUANT_Q4_SYM, QMX_PIN_LATENCY_SENS, vocab,     dim       },
        { "rms_final.weight", QMX_TID_RMS_FINAL, 0xFFFFu, QMX_QUANT_FP32,   QMX_PIN_CRITICAL,     1u,        dim       },
        { "blk.0.attn_norm",  QMX_TID_ATTN_NORM, 0u,      QMX_QUANT_FP32,   QMX_PIN_CRITICAL,     1u,        dim       },
        { "blk.0.attn_q",     QMX_TID_ATTN_Q,    0u,      QMX_QUANT_Q4_SYM, QMX_PIN_LATENCY_SENS, dim,       dim       },
        { "blk.0.attn_k",     QMX_TID_ATTN_K,    0u,      QMX_QUANT_Q4_SYM, QMX_PIN_LATENCY_SENS, kv_dim,    dim       },
        { "blk.0.attn_v",     QMX_TID_ATTN_V,    0u,      QMX_QUANT_Q4_SYM, QMX_PIN_LATENCY_SENS, kv_dim,    dim       },
        { "blk.0.attn_wo",    QMX_TID_ATTN_WO,   0u,      QMX_QUANT_Q4_SYM, QMX_PIN_LATENCY_SENS, dim,       dim       },
        { "blk.0.ffn_norm",   QMX_TID_FFN_NORM,  0u,      QMX_QUANT_FP32,   QMX_PIN_CRITICAL,     1u,        dim       },
        { "blk.0.ffn_gate",   QMX_TID_FFN_GATE,  0u,      QMX_QUANT_Q4_SYM, QMX_PIN_STREAM_ONLY,  inter_dim, dim       },
        { "blk.0.ffn_up",     QMX_TID_FFN_UP,    0u,      QMX_QUANT_Q4_SYM, QMX_PIN_STREAM_ONLY,  inter_dim, dim       },
        { "blk.0.ffn_down",   QMX_TID_FFN_DOWN,  0u,      QMX_QUANT_Q4_SYM, QMX_PIN_STREAM_ONLY,  dim,       inter_dim }
    };

    size_t recipe_off = align64(sizeof(QmxHeader));
    memcpy(rom + recipe_off, recipe, sizeof(recipe));

    size_t tdir_off = align64(recipe_off + sizeof(recipe));
    size_t tok_off  = align64(tdir_off + 11u * sizeof(QmxTensorDesc));

    QmxTokenizerHeader th;
    memset(&th, 0, sizeof(th));
    th.magic         = QMX_TOKENIZER_MAGIC;
    th.vocab_size    = vocab;
    th.merges_count  = 0u;
    th.max_token_len = 1u;
    th.bos_id        = 1;
    th.eos_id        = 2;
    th.eot_id        = -1;
    th.flags         = 0u;

    size_t tpos = tok_off;
    memcpy(rom + tpos, &th, sizeof(th));
    tpos += sizeof(th);
    for (uint32_t i = 0; i < vocab; i++) {
        uint32_t off = i * 2u;
        memcpy(rom + tpos, &off, sizeof(uint32_t));
        tpos += sizeof(uint32_t);
    }
    uint32_t pool_sz = vocab * 2u;
    memcpy(rom + tpos, &pool_sz, sizeof(uint32_t));
    tpos += sizeof(uint32_t);
    for (uint32_t i = 0; i < vocab; i++) {
        rom[tpos++] = (uint8_t)i;
        rom[tpos++] = 0u;
    }
    size_t tok_sz = tpos - tok_off;

    size_t pay_off = align64(tpos);
    size_t cur = pay_off;
    uint32_t total_params = 0u;
    uint32_t rng = 1234567u;

    for (uint32_t i = 0; i < 11u; i++) {
        const Spec *sp = &specs[i];
        total_params += sp->rows * sp->cols;
        cur = align64(cur);

        QmxTensorDesc d;
        memset(&d, 0, sizeof(d));
        strncpy(d.name, sp->name, sizeof(d.name) - 1u);
        d.tensor_id    = sp->tid;
        d.layer_idx    = sp->layer;
        d.quant_type   = sp->qtype;
        d.pin_priority = sp->prio;
        d.rows         = sp->rows;
        d.cols         = sp->cols;

        if (sp->qtype == QMX_QUANT_FP32) {
            d.tile_rows   = 1u;
            d.data_offset = (uint32_t)(cur - pay_off);
            d.data_size   = sp->cols * (uint32_t)sizeof(float);
            float *dst = (float *)(void *)(rom + cur);
            for (uint32_t j = 0; j < sp->cols; j++) dst[j] = 1.0f;
            cur += d.data_size;
        } else {
            uint32_t n_tr = (uint32_t)(qmx_align_up(sp->rows, QMX_TILE_ROWS) / QMX_TILE_ROWS);
            uint32_t n_tc = (uint32_t)(qmx_align_up(sp->cols, QMX_TILE_COLS) / QMX_TILE_COLS);
            uint32_t nt = n_tr * n_tc;
            d.tile_rows     = QMX_TILE_ROWS;
            d.scales_offset = (uint32_t)(cur - pay_off);
            float *sc = (float *)(void *)(rom + cur);
            for (uint32_t t = 0; t < nt; t++) sc[t] = 0.04f;
            cur = align64(cur + nt * sizeof(float));

            d.data_offset = (uint32_t)(cur - pay_off);
            d.data_size   = nt * 256u;
            for (uint32_t b = 0; b < d.data_size; b++) {
                rng = rng * 1664525u + 1013904223u;
                uint8_t lo = (uint8_t)((rng >> 24u) & 0x0Fu);
                uint8_t hi = (uint8_t)((rng >> 20u) & 0x0Fu);
                rom[cur + b] = (uint8_t)(lo | (hi << 4u));
            }
            cur += d.data_size;
        }
        cur = align64(cur);
        memcpy(rom + tdir_off + i * sizeof(QmxTensorDesc), &d, sizeof(d));
    }

    size_t pay_sz = cur - pay_off;
    if (cur > cap) return 0u;

    QmxHeader h;
    memset(&h, 0, sizeof(h));
    h.magic             = QMX_MAGIC;
    h.version           = QMX_VERSION_1_1;
    h.flags             = QMX_FLAG_TILE_MAJOR | QMX_FLAG_SHARED_CLASSIFIER | QMX_FLAG_COMPACT_ALIGN;
    h.recipe_offset     = (uint32_t)recipe_off;
    h.recipe_count      = 18u;
    h.tensor_dir_offset = (uint32_t)tdir_off;
    h.tensor_count      = 11u;
    h.tokenizer_offset  = (uint32_t)tok_off;
    h.tokenizer_size    = (uint32_t)tok_sz;
    h.payload_offset    = (uint32_t)pay_off;
    h.payload_size      = (uint32_t)pay_sz;
    h.vocab_size        = vocab;
    h.max_seq_len       = max_seq;
    h.hidden_dim        = dim;
    h.intermediate_dim  = inter_dim;
    h.num_layers        = (uint16_t)n_layers;
    h.num_heads         = (uint16_t)n_heads;
    h.num_kv_heads      = (uint16_t)n_kv;
    h.primary_quant     = QMX_QUANT_Q4_SYM;
    h.rms_norm_eps      = 1e-5f;
    h.rope_theta        = 10000.0f;
    h.head_dim          = (uint16_t)head_dim;
    h.rotary_dim        = (uint16_t)head_dim;
    h.emb_scale         = 1.0f;
    h.header_crc32      = qmx_crc32(&h, sizeof(h));
    memcpy(rom, &h, sizeof(h));

    if (out_params) *out_params = total_params;
    return cur;
}

int main(void) {
    uint32_t params = 0u;
    size_t flash_bytes = build_15k_xip_cartridge(s_flash_rom, sizeof(s_flash_rom), &params);
    EXPECT(flash_bytes > 0u && flash_bytes <= 16384u, "15K XIP cartridge fits within 16 KB Flash");
    EXPECT(params >= 14000u && params <= 17000u, "Micro-model parameter count is ~15K");

    QmxLoadParams lp = qmx_load_params_default();
    lp.override_max_seq = 64u;
    lp.backend          = QMX_BACKEND_SCALAR;
    lp.kv_type          = QMX_KV_Q8;

    size_t need_sram = qmx_calc_arena_size_memory(s_flash_rom, flash_bytes, &lp);
    EXPECT(need_sram > 0u && need_sram <= sizeof(s_mcu_sram), "15K XIP SRAM arena fits within 32 KB static buffer");

    lp.arena_buf  = s_mcu_sram;
    lp.arena_size = need_sram;

    QmxModel model;
    EXPECT(qmx_load_from_memory_ex(&model, s_flash_rom, flash_bytes, &lp) == 0, "Zero-Copy XIP 15K model load");

    QmxModelInfo info;
    QmxStats stats;
    qmx_get_model_info(&model, &info);
    qmx_get_stats(&model, &stats);
    EXPECT(info.is_xip == 1 && info.stream_mode == 0 && stats.disk_reads == 0u, "Pure Zero-Copy XIP active");

    QmxTokenizer tok;
    EXPECT(qmx_tokenizer_load_from_memory(&tok, s_flash_rom + (size_t)info.tokenizer_offset, (size_t)info.tokenizer_size) == 0,
           "Zero-Copy XIP tokenizer load");
    EXPECT(tok.is_xip == 1 && tok.merges_owned == 0, "XIP tokenizer uses 0 heap");

    int prompt_ids[16];
    int n_prompt = 0;
    qmx_tokenizer_encode(&tok, "MCU!", prompt_ids, &n_prompt, 16, 1);
    EXPECT(n_prompt == 5 && prompt_ids[0] == 1 && prompt_ids[1] == 'M', "XIP tokenizer encodes prompt");

    QmxGenParams gp = qmx_gen_params_default();
    gp.temp          = 0.0f;
    gp.max_tokens    = 12;
    gp.prefill_chunk = 1;

    QmxGenState gen;
    EXPECT(qmx_gen_init(&gen, &model, &tok, &gp) == 0, "qmx_gen_init on 15K XIP");
    EXPECT(qmx_gen_begin(&gen, prompt_ids, n_prompt, &gp) == 0, "qmx_gen_begin on 15K XIP");

    int prefill_steps = 0;
    int gen_tokens = 0;
    QmxStepStatus st;
    QmxStepResult res;
    while ((st = qmx_step(&gen, &res)) > 0) {
        if (st == QMX_STEP_PREFILL) prefill_steps++;
        else if (st == QMX_STEP_TOKEN) gen_tokens++;
        else break;
    }
    EXPECT(prefill_steps == 5 && gen_tokens > 0, "15K XIP step-by-step prefill and decode succeeded");
    EXPECT(!isnan(gen.step0_checksum) && !isinf(gen.step0_checksum), "15K XIP logits finite");

    qmx_tokenizer_free(&tok);
    qmx_free(&model);

    if (g_failures == 0) {
        printf("[PASS] 15K MCU Zero-Copy XIP test passed! (params=%u, Flash=%.2f KB, SRAM/Arena=%.2f KB, chk=%.4f)\n",
               params, (double)flash_bytes / 1024.0, (double)stats.arena_used / 1024.0, (double)gen.step0_checksum);
        return 0;
    }
    fprintf(stderr, "[FAIL] test_xip_15k failures: %d\n", g_failures);
    return 1;
}
