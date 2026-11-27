/*
 * Copyright 2026 1WhiteFox
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/qmx_runtime.h"
#include "../include/qmx_kernels.h"
#include "../include/qmx_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures = 0;

#define EXPECT_TRUE(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "[FAIL] %s (line %d)\n", msg, __LINE__); \
        g_failures++; \
    } \
} while (0)

static int copy_file_bytes(const char *src, const char *dst, uint8_t **out_buf, size_t *out_len) {
    FILE *f = fopen(src, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return -1; }
    uint8_t *buf = (uint8_t *)malloc((size_t)sz);
    if (!buf) { fclose(f); return -1; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        fclose(f);
        return -1;
    }
    fclose(f);
    if (dst) {
        FILE *wf = fopen(dst, "wb");
        if (!wf) { free(buf); return -1; }
        fwrite(buf, 1, (size_t)sz, wf);
        fclose(wf);
    }
    *out_buf = buf;
    *out_len = (size_t)sz;
    return 0;
}

static void rehash_header_crc(uint8_t *buf) {
    QmxHeader *h = (QmxHeader *)buf;
    h->header_crc32 = 0;
    h->header_crc32 = qmx_crc32(h, sizeof(QmxHeader));
}

static void test_malformed_cartridges(const char *valid_qmx) {
    const char *tmp_path = "test_corrupt_tmp.qmx";
    uint8_t *orig = NULL;
    size_t len = 0;
    if (copy_file_bytes(valid_qmx, NULL, &orig, &len) != 0) {
        fprintf(stderr, "[FAIL] Cannot read %s for security tests\n", valid_qmx);
        g_failures++;
        return;
    }

    uint8_t *mut = (uint8_t *)malloc(len);
    QmxModel m;

    /* 1. Повреждение CRC32 заголовка */
    memcpy(mut, orig, len);
    ((QmxHeader *)mut)->header_crc32 ^= 0xDEADBEEFu;
    FILE *f = fopen(tmp_path, "wb"); fwrite(mut, 1, len, f); fclose(f);
    EXPECT_TRUE(qmx_load(&m, tmp_path, 0, QMX_BACKEND_SCALAR, 0, 0, 0) != 0,
                "Reject corrupted header_crc32");

    /* 2. Выход scales_offset за пределы payload_size (OOB Read exploit) */
    memcpy(mut, orig, len);
    QmxHeader *h = (QmxHeader *)mut;
    QmxTensorDesc *descs = (QmxTensorDesc *)(mut + h->tensor_dir_offset);
    descs[0].scales_offset = h->payload_size - 4u;
    rehash_header_crc(mut);
    f = fopen(tmp_path, "wb"); fwrite(mut, 1, len, f); fclose(f);
    EXPECT_TRUE(qmx_load(&m, tmp_path, 0, QMX_BACKEND_SCALAR, 0, 0, 0) != 0,
                "Reject out-of-bounds scales_offset");

    /* 3. Целочисленное переполнение data_offset + data_size (32-bit wrap-around) */
    memcpy(mut, orig, len);
    h = (QmxHeader *)mut;
    descs = (QmxTensorDesc *)(mut + h->tensor_dir_offset);
    descs[0].data_offset = 0xFFFFFF00u;
    descs[0].data_size   = 0x00000200u;
    rehash_header_crc(mut);
    f = fopen(tmp_path, "wb"); fwrite(mut, 1, len, f); fclose(f);
    EXPECT_TRUE(qmx_load(&m, tmp_path, 0, QMX_BACKEND_SCALAR, 0, 0, 0) != 0,
                "Reject 32-bit overflow in data_offset + data_size");

    /* 4. Превышение размерности выходного слота арены (rows > slot capacity) */
    memcpy(mut, orig, len);
    h = (QmxHeader *)mut;
    descs = (QmxTensorDesc *)(mut + h->tensor_dir_offset);
    for (uint32_t i = 0; i < h->tensor_count; i++) {
        if (descs[i].tensor_id == QMX_TID_ATTN_Q) {
            descs[i].rows = h->hidden_dim + 64u;
            break;
        }
    }
    rehash_header_crc(mut);
    f = fopen(tmp_path, "wb"); fwrite(mut, 1, len, f); fclose(f);
    EXPECT_TRUE(qmx_load(&m, tmp_path, 0, QMX_BACKEND_SCALAR, 0, 0, 0) != 0,
                "Reject tensor rows exceeding destination arena slot capacity");

    /* 5. Выход 64-битного tokenizer_offset_hi за пределы файла */
    memcpy(mut, orig, len);
    h = (QmxHeader *)mut;
    h->tokenizer_offset_hi = 1u;
    rehash_header_crc(mut);
    f = fopen(tmp_path, "wb"); fwrite(mut, 1, len, f); fclose(f);
    EXPECT_TRUE(qmx_load(&m, tmp_path, 0, QMX_BACKEND_SCALAR, 0, 0, 0) != 0,
                "Reject out-of-bounds 64-bit tokenizer_offset_hi");

    /* 6. Невыровненный scales_offset (защита от HardFault/UB на ARM/MCU) */
    memcpy(mut, orig, len);
    h = (QmxHeader *)mut;
    descs = (QmxTensorDesc *)(mut + h->tensor_dir_offset);
    descs[0].scales_offset += 2u;
    rehash_header_crc(mut);
    f = fopen(tmp_path, "wb"); fwrite(mut, 1, len, f); fclose(f);
    EXPECT_TRUE(qmx_load(&m, tmp_path, 0, QMX_BACKEND_SCALAR, 0, 0, 0) != 0,
                "Reject misaligned scales_offset (< 4-byte alignment)");

    remove(tmp_path);
    free(mut);
    free(orig);
}

static void test_contractions_and_long_bpe(void) {
    const char *tmp_tok = "test_contractions_tok.bin";
    const char *vocab[] = {
        "<unk>", "<s>", "</s>", "<|eot_id|>",
        " ", "d", "o", "n", "'", "t", "w", "e", "r",
        "do", "don", "'t", "we", "'r", "'re"
    };
    uint32_t vocab_size = (uint32_t)(sizeof(vocab) / sizeof(vocab[0]));
    uint32_t offsets[32];
    char pool[256];
    uint32_t pool_pos = 0;
    for (uint32_t i = 0; i < vocab_size; i++) {
        offsets[i] = pool_pos;
        size_t l = strlen(vocab[i]);
        memcpy(pool + pool_pos, vocab[i], l + 1u);
        pool_pos += (uint32_t)(l + 1u);
    }

    QmxBpeMergeEntry merges[] = {
        { 5,  6,  13, 0 },
        { 13, 7,  14, 1 },
        { 8,  9,  15, 2 },
        { 10, 11, 16, 3 },
        { 8,  12, 17, 4 },
        { 17, 11, 18, 5 }
    };
    uint32_t merges_count = (uint32_t)(sizeof(merges) / sizeof(merges[0]));
    for (uint32_t i = 0; i < merges_count; i++) {
        for (uint32_t j = i + 1u; j < merges_count; j++) {
            uint64_t ki = ((uint64_t)merges[i].left_tok << 32) | merges[i].right_tok;
            uint64_t kj = ((uint64_t)merges[j].left_tok << 32) | merges[j].right_tok;
            if (kj < ki) {
                QmxBpeMergeEntry t = merges[i];
                merges[i] = merges[j];
                merges[j] = t;
            }
        }
    }

    QmxTokenizerHeader hdr = { QMX_TOKENIZER_MAGIC, vocab_size, merges_count, 4, 1, 2, 3, 1 };
    FILE *f = fopen(tmp_tok, "wb");
    fwrite(&hdr, 1, sizeof(hdr), f);
    fwrite(offsets, sizeof(uint32_t), vocab_size, f);
    fwrite(&pool_pos, sizeof(uint32_t), 1, f);
    fwrite(pool, 1, pool_pos, f);
    fwrite(merges, sizeof(QmxBpeMergeEntry), merges_count, f);
    fclose(f);

    QmxTokenizer tok;
    EXPECT_TRUE(qmx_tokenizer_load(&tok, tmp_tok, 0, 0) == 0, "Load contraction tokenizer");

    int ids[32];
    int n = 0;
    qmx_tokenizer_encode(&tok, "don't we're", ids, &n, 32, 0);
    int exp[5] = { 14, 15, 4, 16, 18 };
    EXPECT_TRUE(n == 5, "Contraction token count == 5");
    if (n == 5) {
        for (int i = 0; i < 5; i++) {
            EXPECT_TRUE(ids[i] == exp[i], "Contraction token ID match");
        }
    }

    char long_text[1024];
    size_t pos = 0;
    for (int rep = 0; rep < 80; rep++) {
        memcpy(long_text + pos, "don't we're ", 12);
        pos += 12;
    }
    long_text[pos] = '\0';
    int long_ids[1024];
    int long_n = 0;
    qmx_tokenizer_encode(&tok, long_text, long_ids, &long_n, 1024, 1);
    EXPECT_TRUE(long_n == 1 + 80 * 6, "Long BPE prompt encoded accurately");

    qmx_tokenizer_free(&tok);
    remove(tmp_tok);
}

static void test_sequential_stream_reads(const char *valid_qmx) {
    QmxModel m_unpinned, m_pinned_scales;
    EXPECT_TRUE(qmx_load(&m_unpinned, valid_qmx, 64, QMX_BACKEND_SCALAR, 1, 1, 49152u) == 0,
                "Load stream model with 48KB cap");
    EXPECT_TRUE(qmx_load(&m_pinned_scales, valid_qmx, 64, QMX_BACKEND_SCALAR, 1, 1, 0) == 0,
                "Load stream model with sequential scale-cached reads");

    float *l1 = qmx_forward(&m_unpinned, 1, 0);
    float *l2 = qmx_forward(&m_pinned_scales, 1, 0);
    EXPECT_TRUE(l1 != NULL && l2 != NULL, "Forward non-null in both stream modes");
    EXPECT_TRUE(m_pinned_scales.disk_reads < m_unpinned.disk_reads,
                "Sequential stream mode performs fewer disk reads than unpinned chunk mode");

    for (uint32_t i = 0; i < m_unpinned.header.vocab_size; i++) {
        float diff = l1[i] - l2[i];
        if (diff < 0.0f) diff = -diff;
        if (diff > 1e-5f) {
            fprintf(stderr, "[FAIL] Stream modes logit mismatch at %u: %f vs %f\n", i, (double)l1[i], (double)l2[i]);
            g_failures++;
            break;
        }
    }
    qmx_free(&m_unpinned);
    qmx_free(&m_pinned_scales);

    /* 3. Проверка эквивалентности пакетного префилла qmx_forward_batch и одиночного qmx_forward */
    QmxModel m_single, m_batch;
    EXPECT_TRUE(qmx_load(&m_single, valid_qmx, 64, QMX_BACKEND_SCALAR, 0, 1, 0) == 0, "Load single model");
    EXPECT_TRUE(qmx_load(&m_batch, valid_qmx, 64, QMX_BACKEND_SCALAR, 0, 1, 0) == 0, "Load batch model");
    int p_toks[4] = { 1, 12, 34, 56 };
    float *ls = NULL;
    for (int i = 0; i < 4; i++) ls = qmx_forward(&m_single, p_toks[i], i);
    float chk_b = 0.0f;
    float *lb = qmx_forward_batch(&m_batch, p_toks, 4, 0, &chk_b);
    EXPECT_TRUE(m_batch.disk_reads * 2 < m_single.disk_reads, "Batched prefill cuts stream disk reads by >2x");
    for (uint32_t i = 0; i < m_single.header.vocab_size; i++) {
        float d = ls[i] - lb[i];
        if (d < 0.0f) d = -d;
        if (d > 1e-5f) {
            fprintf(stderr, "[FAIL] Batched prefill mismatch at %u: %f vs %f\n", i, (double)ls[i], (double)lb[i]);
            g_failures++;
            break;
        }
    }

    /* 4. Проверка Zero-Copy XIP (qmx_load_from_memory) и новых опкодов LayerNorm / QK_Norm */
    uint8_t *xip_rom = NULL;
    size_t xip_len = 0;
    if (copy_file_bytes(valid_qmx, NULL, &xip_rom, &xip_len) == 0) {
        QmxModel m_xip;
        EXPECT_TRUE(qmx_load_from_memory(&m_xip, xip_rom, xip_len, 64, QMX_BACKEND_SCALAR, 0) == 0,
                    "Load model in Zero-Copy XIP mode (qmx_load_from_memory)");
        float chk_xip = 0.0f;
        float *l_xip = qmx_forward_batch(&m_xip, p_toks, 4, 0, &chk_xip);
        EXPECT_TRUE(l_xip != NULL && m_xip.disk_reads == 0 && m_xip.is_xip == 1,
                    "XIP forward performs 0 disk reads and uses ROM pointers directly");
        for (uint32_t i = 0; i < m_xip.header.vocab_size; i++) {
            float d = ls[i] - l_xip[i];
            if (d < 0.0f) d = -d;
            if (d > 1e-5f) {
                fprintf(stderr, "[FAIL] XIP logit mismatch at %u\n", i);
                g_failures++;
                break;
            }
        }
        qmx_free(&m_xip);
        free(xip_rom);
    }

    qmx_free(&m_single);
    qmx_free(&m_batch);

    float ln_in[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    float ln_w[4]  = { 1.0f, 1.0f, 1.0f, 1.0f };
    float ln_out[4] = { 0 };
    qmx_layernorm_scalar(ln_out, ln_in, ln_w, 4, 1e-5f);
    float ln_sum = ln_out[0] + ln_out[1] + ln_out[2] + ln_out[3];
    if (ln_sum < 0.0f) ln_sum = -ln_sum;
    EXPECT_TRUE(ln_sum < 1e-5f && ln_out[0] < 0.0f && ln_out[3] > 0.0f,
                "LayerNorm zero-mean unit-variance verification");

    /* 5. Регрессионный тест v1.01a: BUG-02 (расширение пула потоков 1 -> 2 -> 4) и BUG-03 (границы контекста в batch) */
    QmxModel m_mt;
    if (qmx_load(&m_mt, valid_qmx, 16, QMX_BACKEND_SCALAR, 0, 0, 0) == 0) {
        qmx_set_num_threads(&m_mt, 2);
        int b_toks[8] = { 1, 5, 10, 15, 20, 25, 30, 35 };
        float *l_mt1 = qmx_forward_batch(&m_mt, b_toks, 8, 0, NULL);
        qmx_set_num_threads(&m_mt, 4);
        /* Подача пачки B=8 (128*64*8 = 65536 элементов -> реальный вход в pthread spin-pool) за пределы max_seq_len=16 */
        float *l_mt2 = qmx_forward_batch(&m_mt, b_toks, 8, 12, NULL);
        EXPECT_TRUE(l_mt1 != NULL && l_mt2 != NULL,
                    "v1.01a thread pool expansion (2->4) and batch context window clamping");
        qmx_free(&m_mt);
    }

    QmxModel m_f32, m_f16;
    if (qmx_load(&m_f32, valid_qmx, 32, QMX_BACKEND_SCALAR, QMX_KV_FP32, 0, 0) == 0 &&
        qmx_load(&m_f16, valid_qmx, 32, QMX_BACKEND_SCALAR, QMX_KV_FP16, 0, 0) == 0) {
        qmx_set_num_threads(&m_f32, 2);
        qmx_set_num_threads(&m_f16, 2);
        int seq_t[8] = { 1, 7, 13, 19, 25, 31, 37, 43 };
        float *lf32 = qmx_forward_batch(&m_f32, seq_t, 8, 0, NULL);
        float *lf16 = qmx_forward_batch(&m_f16, seq_t, 8, 0, NULL);
        float max_d = 0.0f;
        for (uint32_t i = 0; i < m_f32.header.vocab_size; i++) {
            float d = lf32[i] - lf16[i];
            if (d < 0.0f) d = -d;
            if (d > max_d) max_d = d;
        }
        EXPECT_TRUE(max_d < 0.025f, "FP16 KV-cache parity with FP32 KV-cache");
        int new_p = qmx_shift_kv_cache_ex(&m_f16, 8, 2, 2);
        EXPECT_TRUE(new_p == 6, "Attention Sinks KV shift keeps sinks and shifts tail");
        float *lafter = qmx_forward(&m_f16, 55, new_p);
        EXPECT_TRUE(lafter != NULL, "Forward after Attention Sinks FP16 shift succeeds");
        qmx_free(&m_f32);
        qmx_free(&m_f16);
    }

    QmxModel m_dbuf, m_ref_db;
    if (qmx_load(&m_ref_db, valid_qmx, 32, QMX_BACKEND_SCALAR, QMX_KV_FP32, 0, 0) == 0 &&
        qmx_load(&m_dbuf, valid_qmx, 32, QMX_BACKEND_SCALAR, QMX_KV_FP32, 1, 0) == 0) {
        qmx_set_num_threads(&m_dbuf, 4);
        if (m_dbuf.state.stream_tile_cap > 4u) {
            m_dbuf.state.stream_tile_cap = 4u;
        }
        int seq_t[4] = { 1, 11, 22, 33 };
        float *l_ref = qmx_forward_batch(&m_ref_db, seq_t, 4, 0, NULL);
        float *l_db  = qmx_forward_batch(&m_dbuf, seq_t, 4, 0, NULL);
        float db_diff = 0.0f;
        for (uint32_t i = 0; i < m_ref_db.header.vocab_size; i++) {
            float d = l_ref[i] - l_db[i];
            if (d < 0.0f) d = -d;
            if (d > db_diff) db_diff = d;
        }
        EXPECT_TRUE(m_dbuf.state.stream_data_buf2 != NULL, "Double-buffer allocated in MT stream mode");
        EXPECT_TRUE(db_diff < 1e-5f && m_dbuf.disk_reads > 0u, "Multi-threaded Double-Buffer Stream matches RAM bit-accurately");
        qmx_free(&m_dbuf);
        qmx_free(&m_ref_db);
    }
}

static void test_v11_header_and_xip_tok_align(const char *valid_qmx) {
    const char *tmp_path = "test_v11_hdr_tmp.qmx";
    uint8_t *buf = NULL;
    size_t len = 0;
    if (copy_file_bytes(valid_qmx, NULL, &buf, &len) != 0) {
        fprintf(stderr, "[FAIL] Cannot read %s for v1.1 header test\n", valid_qmx);
        g_failures++;
        return;
    }

    QmxHeader *h = (QmxHeader *)buf;
    h->rope_scale_type   = QMX_ROPE_SCALE_LINEAR;
    h->rope_scale_factor = 2.0f;
    h->norm_type         = 0u;
    h->sliding_window    = 32u;
    rehash_header_crc(buf);

    FILE *f = fopen(tmp_path, "wb");
    if (f) {
        fwrite(buf, 1, len, f);
        fclose(f);
    }

    QmxModel m;
    int rc = qmx_load(&m, tmp_path, 64, QMX_BACKEND_SCALAR, QMX_KV_FP32, 0, 0);
    EXPECT_TRUE(rc == 0, "Load v1.1 cartridge with rope_scale_type + sliding_window");
    if (rc == 0) {
        EXPECT_TRUE(m.header.rope_scale_type == QMX_ROPE_SCALE_LINEAR, "Header rope_scale_type preserved");
        EXPECT_TRUE(m.sliding_window == 32u, "Header sliding_window == 32 preserved");
        float *l = qmx_forward(&m, 1, 0);
        EXPECT_TRUE(l != NULL, "Forward succeeds with v1.1 header extensions");
        qmx_free(&m);
    }

    h->rope_scale_type   = QMX_ROPE_SCALE_YARN;
    h->rope_scale_factor = 4.0f;
    rehash_header_crc(buf);
    f = fopen(tmp_path, "wb");
    if (f) {
        fwrite(buf, 1, len, f);
        fclose(f);
    }
    rc = qmx_load(&m, tmp_path, 64, QMX_BACKEND_SCALAR, QMX_KV_FP32, 0, 0);
    EXPECT_TRUE(rc == 0, "Load v1.1 cartridge with YaRN RoPE scaling");
    if (rc == 0) {
        float *l_yarn = qmx_forward(&m, 1, 5);
        EXPECT_TRUE(l_yarn != NULL && m.state.rope_inv_freq[0] > 0.0f, "YaRN RoPE forward succeeds");
        qmx_free(&m);
    }
    remove(tmp_path);
    free(buf);

    const char *vocab[] = { "<unk>", "<s>", "</s>", "a", "b", "ab" };
    uint32_t vocab_size = 6u;
    uint32_t offsets[6];
    uint8_t pool[32];
    uint32_t pool_pos = 0u;
    for (uint32_t i = 0; i < vocab_size; i++) {
        offsets[i] = pool_pos;
        size_t sl = strlen(vocab[i]);
        memcpy(pool + pool_pos, vocab[i], sl + 1u);
        pool_pos += (uint32_t)(sl + 1u);
    }
    while ((pool_pos & 3u) != 0u) pool[pool_pos++] = 0u;

    QmxBpeMergeEntry merge0 = { 3u, 4u, 5u, 0u };
    QmxTokenizerHeader th = { QMX_TOKENIZER_MAGIC, vocab_size, 1u, 5u, 1, 2, -1, 0u };

    QMX_ALIGNED(16) uint8_t xip_tok_buf[256];
    size_t pos = 0;
    memcpy(xip_tok_buf + pos, &th, sizeof(th)); pos += sizeof(th);
    memcpy(xip_tok_buf + pos, offsets, sizeof(offsets)); pos += sizeof(offsets);
    memcpy(xip_tok_buf + pos, &pool_pos, sizeof(pool_pos)); pos += sizeof(pool_pos);
    memcpy(xip_tok_buf + pos, pool, pool_pos); pos += pool_pos;
    memcpy(xip_tok_buf + pos, &merge0, sizeof(merge0)); pos += sizeof(merge0);

    QmxTokenizer xtok;
    EXPECT_TRUE(qmx_tokenizer_load_from_memory(&xtok, xip_tok_buf, pos) == 0, "Load aligned XIP tokenizer");
    EXPECT_TRUE(xtok.is_xip == 1 && xtok.merges_owned == 0, "XIP tokenizer uses zero-copy merges without malloc");
    int ids[8];
    int n_ids = 0;
    qmx_tokenizer_encode(&xtok, "ab", ids, &n_ids, 8, 0);
    EXPECT_TRUE(n_ids == 1 && ids[0] == 5, "Aligned XIP tokenizer BPE merge works");
    qmx_tokenizer_free(&xtok);
}

typedef struct {
    const uint8_t *buf;
    size_t         len;
} TestCrcMemCtx;

static int test_crc_mem_read(void *ctx, uint64_t offset, void *dst, size_t size) {
    TestCrcMemCtx *m = (TestCrcMemCtx *)ctx;
    if (!m || !m->buf || offset > (uint64_t)m->len || size > m->len - (size_t)offset) return -1;
    if (size > 0u) memcpy(dst, m->buf + (size_t)offset, size);
    return 0;
}

static void test_file_crc32_verification(const char *valid_qmx) {
    const char *tmp_path = "test_crc_tmp.qmx";
    uint8_t *orig = NULL;
    size_t len = 0;
    if (copy_file_bytes(valid_qmx, NULL, &orig, &len) != 0) {
        fprintf(stderr, "[FAIL] Cannot read %s for file_crc32 test\n", valid_qmx);
        g_failures++;
        return;
    }

    uint32_t stored = 0u, calc = 0u;
    EXPECT_TRUE(qmx_verify_crc_file(valid_qmx, 0u, 0u, &stored, &calc) == 0 && stored != 0u && stored == calc,
                "Freshly packed cartridge passes qmx_verify_crc_file");
    EXPECT_TRUE(qmx_verify_crc_memory(orig, len, &stored, &calc) == 0 && stored == calc,
                "Freshly packed cartridge passes qmx_verify_crc_memory");

    TestCrcMemCtx mctx = { orig, len };
    QmxReader rd = { test_crc_mem_read, NULL, &mctx, (uint64_t)len };
    EXPECT_TRUE(qmx_verify_crc_reader(&rd, &stored, &calc) == 0 && stored == calc,
                "Freshly packed cartridge passes qmx_verify_crc_reader");

    uint8_t *mut = (uint8_t *)malloc(len);
    memcpy(mut, orig, len);
    mut[len - 17u] ^= 0x01u;
    EXPECT_TRUE(qmx_verify_crc_memory(mut, len, &stored, &calc) < 0 && stored != calc,
                "Reject 1-bit payload corruption via qmx_verify_crc_memory");

    FILE *f = fopen(tmp_path, "wb");
    if (f) { fwrite(mut, 1, len, f); fclose(f); }
    EXPECT_TRUE(qmx_verify_crc_file(tmp_path, 0u, 0u, &stored, &calc) < 0 && stored != calc,
                "Reject 1-bit payload corruption via qmx_verify_crc_file");

    EXPECT_TRUE(qmx_verify_crc_memory(orig, len - 64u, NULL, NULL) < 0,
                "Reject truncated cartridge via qmx_verify_crc_memory");

    memcpy(mut, orig, len);
    ((QmxHeader *)mut)->file_crc32 = 0u;
    rehash_header_crc(mut);
    EXPECT_TRUE(qmx_verify_crc_memory(mut, len, &stored, &calc) == 1 && stored == 0u,
                "Legacy cartridge with file_crc32==0 returns 1 (graceful skip)");

    ((QmxHeader *)mut)->header_crc32 ^= 0x12345678u;
    EXPECT_TRUE(qmx_verify_crc_memory(mut, len, NULL, NULL) < 0,
                "Corrupted header_crc32 rejected by qmx_verify_crc_memory");

    remove(tmp_path);
    free(mut);
    free(orig);
}

int main(int argc, char **argv) {
    const char *qmx_path = (argc > 1) ? argv[1] : "test_q4.qmx";
    test_malformed_cartridges(qmx_path);
    test_contractions_and_long_bpe();
    test_sequential_stream_reads(qmx_path);
    test_v11_header_and_xip_tok_align(qmx_path);
    test_file_crc32_verification(qmx_path);

    if (g_failures == 0) {
        printf("[PASS] All security, BPE contraction, XIP & sequential streaming stress tests passed!\n");
        return 0;
    }
    fprintf(stderr, "[FAIL] Total stress test failures: %d\n", g_failures);
    return 1;
}