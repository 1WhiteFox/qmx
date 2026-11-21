/*
 * Copyright 2026 1WhiteFox
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef QMX_GEN_H
#define QMX_GEN_H

#include "qmx_runtime.h"
#include "qmx_tokenizer.h"
#include "qmx_sampler.h"

#ifdef __cplusplus
extern "C" {
#endif

#define QMX_GEN_RECENT_CAP 64
#define QMX_GEN_TAIL_CAP   128

typedef struct {
    QmxModel           *model;
    const QmxTokenizer *tok;
    QmxGenParams        params;
    int                 pos;
    int                 pending_token;
    int                 prev_token;
    const int          *prompt_tokens;
    int                 prompt_len;
    int                 prompt_off;
    int                 generated;
    int                 finished;
    int                 recent_tokens[QMX_GEN_RECENT_CAP];
    int                 recent_cnt;
    char                tail_window[QMX_GEN_TAIL_CAP];
    float              *last_logits;
    float               step0_checksum;
    int                 has_step0_checksum;
    uint32_t            prefill_tokens;
    uint32_t            decode_tokens;
    uint32_t            rng_state;
    int                 bos_id;
    int                 eos_id;
    int                 eot_id;
} QmxGenState;

QmxGenParams qmx_gen_params_default(void);

int qmx_gen_init(QmxGenState *gen, QmxModel *model, const QmxTokenizer *tok, const QmxGenParams *params);
void qmx_gen_reset(QmxGenState *gen);
void qmx_gen_set_tokenizer(QmxGenState *gen, const QmxTokenizer *tok);

int qmx_gen_begin(QmxGenState *gen, const int *prompt_tokens, int n_tokens, const QmxGenParams *params);
QmxStepStatus qmx_step(QmxGenState *gen, QmxStepResult *out);
int qmx_gen_flush(QmxGenState *gen);

int qmx_gen_is_stop_token(const QmxTokenizer *tok, int token_id);

#ifdef __cplusplus
}
#endif

#endif
