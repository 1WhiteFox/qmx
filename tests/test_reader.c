/*
 * Copyright 2026 1WhiteFox
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/qmx_runtime.h"
#include "../include/qmx_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int g_failures = 0;

#define EXPECT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "[FAIL] %s (line %d)\n", msg, __LINE__); \
        g_failures++; \
    } \
} while (0)

typedef struct {
    const uint8_t *buf;
    uint64_t       size;
    uint32_t       read_calls;
    uint64_t       bytes_read;
    int            close_calls;
} MockSpiFlash;

static int mock_spi_read(void *ctx, uint64_t offset, void *dst, size_t size) {
    MockSpiFlash *f = (MockSpiFlash *)ctx;
    if (!f || !f->buf) return -1;
    if (offset > f->size || (uint64_t)size > f->size - offset) return -1;
    if (size > 0u) {
        memcpy(dst, f->buf + (size_t)offset, size);
    }
    f->read_calls++;
    f->bytes_read += (uint64_t)size;
    return 0;
}

static void mock_spi_close(void *ctx) {
    MockSpiFlash *f = (MockSpiFlash *)ctx;
    if (f) f->close_calls++;
}

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "test_q4.qmx";
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "[FAIL] Cannot open %s\n", path);
        return 1;
    }
    int64_t fsz = qmx_file_size64(fp);
    EXPECT(fsz > (int64_t)sizeof(QmxHeader), "Cartridge file size valid");
    uint8_t *raw = (uint8_t *)malloc((size_t)fsz);
    EXPECT(raw != NULL && fread(raw, 1, (size_t)fsz, fp) == (size_t)fsz, "Read cartridge bytes");
    fclose(fp);
    if (g_failures > 0) { free(raw); return 1; }

    QmxModel m_ref;
    EXPECT(qmx_load(&m_ref, path, 64, QMX_BACKEND_SCALAR, QMX_KV_FP32, 0, 0) == 0, "Load reference model");
    int prompt[4] = { 1, 65, 66, 67 };
    float *l_ref = NULL;
    for (int i = 0; i < 4; i++) l_ref = qmx_forward(&m_ref, prompt[i], i);
    uint32_t vocab = m_ref.header.vocab_size;
    float *ref_copy = (float *)malloc((size_t)vocab * sizeof(float));
    memcpy(ref_copy, l_ref, (size_t)vocab * sizeof(float));
    qmx_free(&m_ref);

    MockSpiFlash spi;
    memset(&spi, 0, sizeof(spi));
    spi.buf  = raw;
    spi.size = (uint64_t)fsz;

    QmxReader borrowed;
    borrowed.read  = mock_spi_read;
    borrowed.close = NULL;
    borrowed.ctx   = &spi;
    borrowed.size  = spi.size;

    QmxLoadParams p = qmx_load_params_default();
    p.override_max_seq = 64u;
    p.backend          = QMX_BACKEND_SCALAR;
    p.kv_type          = QMX_KV_FP32;
    p.force_stream     = 1;
    p.max_ram_bytes    = 49152u;

    size_t need_arena = qmx_calc_arena_size(&borrowed, &p, 0);
    EXPECT(need_arena > 0u, "qmx_calc_arena_size returned non-zero");

    uint8_t *ext_arena = (uint8_t *)malloc(need_arena);
    p.arena_buf  = ext_arena;
    p.arena_size = need_arena;

    QmxModel m_spi;
    EXPECT(qmx_load_from_reader(&m_spi, borrowed, &p) == 0, "qmx_load_from_reader with external arena");
    EXPECT(m_spi.arena.is_external == 1 && m_spi.stream_mode == 1, "External arena + stream mode active");
    EXPECT(m_spi.pin_pool == NULL && m_spi.batch_ws == NULL && m_spi.state.xq16 == NULL, "No extra heap pools or unused xq16 with external arena");

    float *l_spi = NULL;
    for (int i = 0; i < 4; i++) l_spi = qmx_forward(&m_spi, prompt[i], i);
    float max_diff = 0.0f;
    for (uint32_t i = 0; i < vocab; i++) {
        float d = fabsf(l_spi[i] - ref_copy[i]);
        if (d > max_diff) max_diff = d;
    }
    EXPECT(max_diff < 1e-5f, "SPI reader stream logits match reference bit-accurately");
    EXPECT(spi.read_calls > 0u && spi.close_calls == 0, "SPI reader read called and borrowed close not called");

    QmxTokenizer tok;
    EXPECT(qmx_tokenizer_load_reader(&tok, &borrowed,
                                     m_spi.header.tokenizer_offset,
                                     m_spi.header.tokenizer_size) == 0,
           "qmx_tokenizer_load_reader over borrowed SPI reader");
    EXPECT(spi.close_calls == 0, "Tokenizer never calls reader.close");

    int ids[16];
    int n_ids = 0;
    qmx_tokenizer_encode(&tok, "QMX", ids, &n_ids, 16, 1);
    EXPECT(n_ids == 4 && ids[0] == 1 && ids[1] == 'Q' && ids[2] == 'M' && ids[3] == 'X',
           "SPI-loaded tokenizer encodes accurately");
    qmx_tokenizer_free(&tok);

    qmx_free(&m_spi);
    EXPECT(spi.close_calls == 0, "qmx_free respects close == NULL");
    free(ext_arena);

    QmxReader owned = borrowed;
    owned.close = mock_spi_close;
    QmxLoadParams p_own = qmx_load_params_default();
    p_own.override_max_seq = 64u;
    p_own.backend          = QMX_BACKEND_SCALAR;
    p_own.force_stream     = 1;
    QmxModel m_own;
    EXPECT(qmx_load_from_reader(&m_own, owned, &p_own) == 0, "Load owned streaming reader");
    EXPECT(spi.close_calls == 0, "Owned streaming reader kept open while model alive");
    qmx_free(&m_own);
    EXPECT(spi.close_calls == 1, "Owned streaming reader closed exactly once in qmx_free");

    QmxLoadParams p_xip = qmx_load_params_default();
    p_xip.override_max_seq = 64u;
    p_xip.backend          = QMX_BACKEND_SCALAR;
    size_t xip_arena_sz = qmx_calc_arena_size_memory(raw, (size_t)fsz, &p_xip);
    EXPECT(xip_arena_sz > 0u, "qmx_calc_arena_size_memory returned non-zero");
    uint8_t *xip_arena = (uint8_t *)malloc(xip_arena_sz);
    p_xip.arena_buf  = xip_arena;
    p_xip.arena_size = xip_arena_sz;

    QmxModel m_xip;
    EXPECT(qmx_load_from_memory_ex(&m_xip, raw, (size_t)fsz, &p_xip) == 0, "XIP load with external arena");
    float *l_xip = qmx_forward_batch(&m_xip, prompt, 4, 0, NULL);
    float xip_diff = 0.0f;
    for (uint32_t i = 0; i < vocab; i++) {
        float d = fabsf(l_xip[i] - ref_copy[i]);
        if (d > xip_diff) xip_diff = d;
    }
    EXPECT(xip_diff < 1e-5f && m_xip.batch_ws == NULL && m_xip.state.xq16 == NULL, "XIP external-arena batch forward matches with 0 heap ws and no xq16");
    qmx_free(&m_xip);
    free(xip_arena);

    free(ref_copy);
    free(raw);

    if (g_failures == 0) {
        printf("[PASS] QmxReader, external arena & SPI tokenizer tests passed! (spi_diff=%.2e, xip_diff=%.2e)\n",
               (double)max_diff, (double)xip_diff);
        return 0;
    }
    fprintf(stderr, "[FAIL] test_reader failures: %d\n", g_failures);
    return 1;
}
