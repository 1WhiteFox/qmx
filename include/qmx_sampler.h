/*
 * Copyright 2026 1WhiteFox
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef QMX_SAMPLER_H
#define QMX_SAMPLER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QMX_MAX_TOP_K 256

uint32_t qmx_xorshift32(uint32_t *state);

void qmx_push_recent_token(int *recent, int *recent_cnt, int cap, int token);

void qmx_apply_repetition_penalty(float *logits, uint32_t vocab_size,
                                  const int *recent, int recent_cnt,
                                  float penalty, int bos_id, int eos_id, int eot_id);

int qmx_sample_argmax(const float *logits, uint32_t n);

int qmx_sample_ex(const float *logits, uint32_t n,
                  float temp, int top_k, float top_p, float min_p, uint32_t *rng);

int qmx_sample(const float *logits, uint32_t n,
               float temp, int top_k, float top_p, uint32_t *rng);

int qmx_check_stop_sequence(char *tail_window, size_t window_cap,
                            const char *piece, int check_qa);

#ifdef __cplusplus
}
#endif

#endif