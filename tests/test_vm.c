/*
 * Copyright 2026 1WhiteFox
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/qmx_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float logits_checksum(const float *logits, uint32_t n) {
    float chk = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        chk += logits[i] * (float)((int)(i % 7u) - 3);
    }
    return chk;
}

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "dummy.qmx";

    QmxModel m_ram;
    if (qmx_load(&m_ram, path, 0, QMX_BACKEND_SCALAR, 0, 0, 0) != 0) {
        fprintf(stderr, "[FAIL] Could not load %s in RAM mode\n", path);
        return 1;
    }

    uint32_t vocab = m_ram.header.vocab_size;
    float *ref_logits = (float *)malloc((size_t)vocab * sizeof(float));
    if (!ref_logits) {
        qmx_free(&m_ram);
        return 1;
    }

    int prompt[4] = { 1, 42, 77, 13 };
    float *out_ram = NULL;
    for (int p = 0; p < 4; p++) {
        out_ram = qmx_forward(&m_ram, prompt[p], p);
        if (!out_ram) {
            fprintf(stderr, "[FAIL] qmx_forward returned NULL at pos=%d\n", p);
            free(ref_logits);
            qmx_free(&m_ram);
            return 1;
        }
    }
    memcpy(ref_logits, out_ram, (size_t)vocab * sizeof(float));
    float chk_ram = logits_checksum(ref_logits, vocab);
    if (isnan(chk_ram) || isinf(chk_ram) || chk_ram == 0.0f) {
        fprintf(stderr, "[FAIL] Invalid RAM checksum: %f\n", (double)chk_ram);
        free(ref_logits);
        qmx_free(&m_ram);
        return 1;
    }
    qmx_free(&m_ram);

    QmxModel m_stream;
    if (qmx_load(&m_stream, path, 0, QMX_BACKEND_SCALAR, 0, 1, 49152u) != 0) {
        fprintf(stderr, "[FAIL] Could not load %s in STREAM mode\n", path);
        free(ref_logits);
        return 1;
    }
    if (!m_stream.stream_mode || m_stream.state.stream_tile_cap != 2u) {
        fprintf(stderr, "[FAIL] Unexpected stream config: stream=%d cap=%u\n",
                m_stream.stream_mode, m_stream.state.stream_tile_cap);
        free(ref_logits);
        qmx_free(&m_stream);
        return 1;
    }

    float *out_stream = NULL;
    for (int p = 0; p < 4; p++) {
        out_stream = qmx_forward(&m_stream, prompt[p], p);
    }
    float max_diff_stream = 0.0f;
    for (uint32_t i = 0; i < vocab; i++) {
        float d = fabsf(out_stream[i] - ref_logits[i]);
        if (d > max_diff_stream) max_diff_stream = d;
    }
    if (max_diff_stream > 1e-5f || m_stream.disk_reads == 0) {
        fprintf(stderr, "[FAIL] Stream mismatch: max_diff=%.6f disk_reads=%u\n",
                (double)max_diff_stream, m_stream.disk_reads);
        free(ref_logits);
        qmx_free(&m_stream);
        return 1;
    }
    qmx_free(&m_stream);

    QmxModel m_kv8;
    if (qmx_load(&m_kv8, path, 0, QMX_BACKEND_SCALAR, 1, 0, 0) != 0) {
        fprintf(stderr, "[FAIL] Could not load %s in KV8 mode\n", path);
        free(ref_logits);
        return 1;
    }
    float *out_kv8 = NULL;
    for (int p = 0; p < 4; p++) {
        out_kv8 = qmx_forward(&m_kv8, prompt[p], p);
    }
    float max_diff_kv8 = 0.0f;
    for (uint32_t i = 0; i < vocab; i++) {
        float d = fabsf(out_kv8[i] - ref_logits[i]);
        if (d > max_diff_kv8) max_diff_kv8 = d;
    }
    if (max_diff_kv8 > 0.05f) {
        fprintf(stderr, "[FAIL] KV8 mismatch: max_diff=%.6f\n", (double)max_diff_kv8);
        free(ref_logits);
        qmx_free(&m_kv8);
        return 1;
    }

    int shifted_pos = qmx_shift_kv_cache(&m_kv8, 4, 2);
    if (shifted_pos != 2) {
        fprintf(stderr, "[FAIL] KV shift returned pos=%d expected 2\n", shifted_pos);
        free(ref_logits);
        qmx_free(&m_kv8);
        return 1;
    }
    float *out_after_shift = qmx_forward(&m_kv8, 99, shifted_pos);
    float chk_shift = logits_checksum(out_after_shift, vocab);
    if (isnan(chk_shift) || isinf(chk_shift)) {
        fprintf(stderr, "[FAIL] Invalid logits after KV shift\n");
        free(ref_logits);
        qmx_free(&m_kv8);
        return 1;
    }
    qmx_free(&m_kv8);

    QmxModel m_long_f32, m_long_f16;
    if (qmx_load(&m_long_f32, path, 128, QMX_BACKEND_AUTO, QMX_KV_FP32, 0, 0) != 0 ||
        qmx_load(&m_long_f16, path, 128, QMX_BACKEND_AUTO, QMX_KV_FP16, 0, 0) != 0) {
        fprintf(stderr, "[FAIL] Could not load models for multi-block FlashAttention check\n");
        free(ref_logits);
        return 1;
    }
    qmx_set_num_threads(&m_long_f32, 2);
    qmx_set_num_threads(&m_long_f16, 2);
    int long_seq[80];
    for (int i = 0; i < 80; i++) long_seq[i] = 1 + ((i * 37) % (int)(vocab - 1u));
    float *lf32 = qmx_forward_batch(&m_long_f32, long_seq, 80, 0, NULL);
    float *lf16 = qmx_forward_batch(&m_long_f16, long_seq, 80, 0, NULL);
    float max_diff_flash = 0.0f;
    for (uint32_t i = 0; i < vocab; i++) {
        float d = fabsf(lf32[i] - lf16[i]);
        if (d > max_diff_flash) max_diff_flash = d;
    }
    if (isnan(max_diff_flash) || max_diff_flash > 0.02f || m_long_f16.mt_att_buf != NULL) {
        fprintf(stderr, "[FAIL] Multi-block FlashAttention mismatch: diff=%.6f\n", (double)max_diff_flash);
        qmx_free(&m_long_f32);
        qmx_free(&m_long_f16);
        free(ref_logits);
        return 1;
    }
    qmx_free(&m_long_f32);
    qmx_free(&m_long_f16);
    free(ref_logits);

    printf("[PASS] Stage 2.2 QMX VM test passed! (chk=%.6f, stream_diff=%.2e, kv8_diff=%.4f)\n",
           (double)chk_ram, (double)max_diff_stream, (double)max_diff_kv8);
    return 0;
}