/*
 * Copyright 2026 1WhiteFox
 * SPDX-License-Identifier: Apache-2.0
 */

#include "qmx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const uint8_t *data;
    uint64_t       size;
} MemReaderCtx;

static int mem_read(void *ctx, uint64_t offset, void *dst, size_t size) {
    MemReaderCtx *m = (MemReaderCtx *)ctx;
    if (!m || offset > m->size || (uint64_t)size > m->size - offset) return -1;
    if (size > 0u) memcpy(dst, m->data + (size_t)offset, size);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.qmx> [prompt] [--reader]\n", argv[0]);
        return 1;
    }
    const char *path = argv[1];
    const char *prompt = (argc >= 3 && strcmp(argv[2], "--reader") != 0) ? argv[2] : "Hello";
    int use_reader = 0;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--reader") == 0) use_reader = 1;
    }

    QmxModel model;
    QmxTokenizer tok;
    int has_tok = 0;
    uint8_t *file_buf = NULL;
    uint8_t *ext_arena = NULL;
    MemReaderCtx mctx = { NULL, 0u };

    QmxLoadParams lp = qmx_load_params_default();
    lp.override_max_seq = 128u;
    lp.backend          = QMX_BACKEND_AUTO;

    if (use_reader) {
        FILE *fp = fopen(path, "rb");
        if (!fp) return 1;
        int64_t fsz = qmx_file_size64(fp);
        if (fsz <= 0) { fclose(fp); return 1; }
        file_buf = (uint8_t *)malloc((size_t)fsz);
        if (!file_buf || fread(file_buf, 1, (size_t)fsz, fp) != (size_t)fsz) {
            free(file_buf);
            fclose(fp);
            return 1;
        }
        fclose(fp);

        mctx.data = file_buf;
        mctx.size = (uint64_t)fsz;
        QmxReader reader  = { mem_read, NULL, &mctx, (uint64_t)fsz };
        lp.force_stream   = 1;
        lp.max_ram_bytes  = 49152u;
        size_t need_arena = qmx_calc_arena_size(&reader, &lp, 0);
        ext_arena         = (uint8_t *)malloc(need_arena);
        lp.arena_buf      = ext_arena;
        lp.arena_size     = need_arena;

        if (qmx_load_from_reader(&model, reader, &lp) != 0) {
            free(ext_arena);
            free(file_buf);
            return 1;
        }
        QmxModelInfo info;
        qmx_get_model_info(&model, &info);
        if (info.tokenizer_size > 0u) {
            has_tok = (qmx_tokenizer_load_reader(&tok, &reader, info.tokenizer_offset, info.tokenizer_size) == 0);
        }
    } else {
        if (qmx_load_ex(&model, path, 0u, 0u, &lp) != 0) return 1;
        QmxModelInfo info;
        qmx_get_model_info(&model, &info);
        if (info.tokenizer_size > 0u) {
            has_tok = (qmx_tokenizer_load(&tok, path, info.tokenizer_offset, info.tokenizer_size) == 0);
        }
    }

    int prompt_ids[64];
    int n_prompt = 0;
    if (has_tok) {
        qmx_tokenizer_encode(&tok, prompt, prompt_ids, &n_prompt, 64, 1);
    }
    if (n_prompt <= 0) {
        prompt_ids[0] = 1;
        n_prompt = 1;
    }

    QmxGenParams gp = qmx_gen_params_default();
    gp.temp          = 0.0f;
    gp.max_tokens    = 16;
    gp.prefill_chunk = use_reader ? 1 : 8;

    QmxGenState gen;
    qmx_gen_init(&gen, &model, has_tok ? &tok : NULL, &gp);
    qmx_gen_begin(&gen, prompt_ids, n_prompt, &gp);

    QmxStepStatus st;
    QmxStepResult res;
    while ((st = qmx_step(&gen, &res)) > 0) {
        if (st == QMX_STEP_TOKEN && res.piece) {
            fputs(res.piece, stdout);
            fflush(stdout);
        } else if (st != QMX_STEP_PREFILL) {
            break;
        }
    }
    putchar('\n');

    QmxModelInfo info;
    qmx_get_model_info(&model, &info);
    fprintf(stderr, "[Amalg] backend=%s | mode=%s | prefill=%u | decode=%u | step0_checksum=%.6f\n",
            info.backend_name, info.stream_mode ? "STREAM" : "RAM",
            gen.prefill_tokens, gen.decode_tokens, (double)gen.step0_checksum);

    if (has_tok) qmx_tokenizer_free(&tok);
    qmx_free(&model);
    free(ext_arena);
    free(file_buf);
    return 0;
}
