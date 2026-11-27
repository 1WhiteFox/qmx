/*
 * Copyright 2026 1WhiteFox
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/qmx_sampler.h"
#include "../include/qmx_runtime.h"
#include <string.h>

static inline int qmx_is_finite_f32(float x) {
    QmxFloatBits fb;
    fb.f = x;
    return ((fb.u & 0x7F800000u) != 0x7F800000u);
}

uint32_t qmx_xorshift32(uint32_t *state) {
    uint32_t x = *state;
    if (x == 0u) x = 2463534242u;
    x ^= x << 13u;
    x ^= x >> 17u;
    x ^= x << 5u;
    *state = x;
    return x;
}

void qmx_push_recent_token(int *recent, int *recent_cnt, int cap, int token) {
    if (!recent || !recent_cnt || cap <= 0) return;
    if (*recent_cnt < cap) {
        recent[(*recent_cnt)++] = token;
    } else {
        memmove(recent, recent + 1, (size_t)(cap - 1) * sizeof(int));
        recent[cap - 1] = token;
    }
}

void qmx_apply_repetition_penalty(float *logits, uint32_t vocab_size,
                                  const int *recent, int recent_cnt,
                                  float penalty, int bos_id, int eos_id, int eot_id) {
    if (!logits || !recent || penalty <= 1.0f || recent_cnt <= 0) return;
    for (int i = 0; i < recent_cnt; i++) {
        int tok = recent[i];
        if (tok < 0 || (uint32_t)tok >= vocab_size) continue;
        if (tok == bos_id || tok == eos_id || tok == eot_id) continue;
        int already = 0;
        for (int j = 0; j < i; j++) {
            if (recent[j] == tok) {
                already = 1;
                break;
            }
        }
        if (already) continue;
        if (!qmx_is_finite_f32(logits[tok])) continue;
        if (logits[tok] > 0.0f) logits[tok] /= penalty;
        else logits[tok] *= penalty;
    }
}

int qmx_sample_argmax(const float *logits, uint32_t n) {
    if (!logits || n == 0u) return 0;
    int max_i = 0;
    float max_p = -3.402823466e+38f;
    int found = 0;
    for (uint32_t i = 0u; i < n; i++) {
        float v = logits[i];
        if (!qmx_is_finite_f32(v)) continue;
        if (!found || v > max_p) {
            max_i = (int)i;
            max_p = v;
            found = 1;
        }
    }
    return max_i;
}

int qmx_sample_ex(const float *logits, uint32_t n,
                  float temp, int top_k, float top_p, float min_p, uint32_t *rng) {
    if (!logits || n == 0u) return 0;
    if (temp <= 0.0f || top_k == 1 || n <= 1u) {
        return qmx_sample_argmax(logits, n);
    }

    int k = top_k;
    if (k <= 0) k = QMX_MAX_TOP_K;
    if (k > QMX_MAX_TOP_K) k = QMX_MAX_TOP_K;
    if ((uint32_t)k > n) k = (int)n;

    int cand_id[QMX_MAX_TOP_K];
    float cand_val[QMX_MAX_TOP_K];
    int count = 0;

    for (uint32_t i = 0u; i < n; i++) {
        float v = logits[i];
        if (!qmx_is_finite_f32(v)) continue;
        if (count == k && v <= cand_val[k - 1]) continue;
        int pos = count;
        if (count < k) count++;
        else pos = k - 1;
        while (pos > 0 && v > cand_val[pos - 1]) {
            cand_val[pos] = cand_val[pos - 1];
            cand_id[pos]  = cand_id[pos - 1];
            pos--;
        }
        cand_val[pos] = v;
        cand_id[pos]  = (int)i;
    }

    if (count == 0) return 0;

    float inv_temp = 1.0f / temp;
    float max_l = cand_val[0] * inv_temp;
    float sum = 0.0f;
    for (int i = 0; i < count; i++) {
        float e = qmx_expf(cand_val[i] * inv_temp - max_l);
        cand_val[i] = e;
        sum += e;
    }
    if (sum <= 0.0f) return cand_id[0];

    float inv_sum = 1.0f / sum;
    for (int i = 0; i < count; i++) {
        cand_val[i] *= inv_sum;
    }

    if (min_p > 0.0f && min_p < 1.0f) {
        float thresh = cand_val[0] * min_p;
        int keep = 1;
        float keep_sum = cand_val[0];
        for (int i = 1; i < count; i++) {
            if (cand_val[i] < thresh) break;
            keep_sum += cand_val[i];
            keep++;
        }
        if (keep < count && keep_sum > 0.0f) {
            count = keep;
            float renorm = 1.0f / keep_sum;
            for (int i = 0; i < count; i++) {
                cand_val[i] *= renorm;
            }
        }
    }

    if (top_p > 0.0f && top_p < 1.0f) {
        float cum = 0.0f;
        int cutoff = count;
        for (int i = 0; i < count; i++) {
            cum += cand_val[i];
            if (cum >= top_p) {
                cutoff = i + 1;
                break;
            }
        }
        if (cutoff < count && cum > 0.0f) {
            count = cutoff;
            float renorm = 1.0f / cum;
            for (int i = 0; i < count; i++) {
                cand_val[i] *= renorm;
            }
        }
    }

    uint32_t r = rng ? qmx_xorshift32(rng) : 123456789u;
    float coin = (float)(r >> 8u) * (1.0f / 16777216.0f);
    float cdf = 0.0f;
    for (int i = 0; i < count; i++) {
        cdf += cand_val[i];
        if (coin < cdf) return cand_id[i];
    }
    return cand_id[count - 1];
}

int qmx_sample(const float *logits, uint32_t n,
               float temp, int top_k, float top_p, uint32_t *rng) {
    return qmx_sample_ex(logits, n, temp, top_k, top_p, 0.0f, rng);
}

int qmx_check_stop_sequence(char *tail_window, size_t window_cap,
                            const char *piece, int check_qa) {
    if (!tail_window || window_cap < 16u || !piece || piece[0] == '\0') return 0;
    size_t tlen = strlen(tail_window);
    size_t plen = strlen(piece);
    if (plen >= window_cap) {
        piece += (plen - (window_cap - 1u));
        plen = strlen(piece);
        tail_window[0] = '\0';
        tlen = 0;
    } else if (tlen + plen >= window_cap) {
        size_t keep = window_cap / 2u;
        if (keep > tlen) keep = tlen;
        memmove(tail_window, tail_window + tlen - keep, keep);
        tail_window[keep] = '\0';
    }
    strncat(tail_window, piece, window_cap - strlen(tail_window) - 1u);

    if (strstr(tail_window, "<|eot_id|>") ||
        strstr(tail_window, "<|start_header_id|>") ||
        strstr(tail_window, "<|im_end|>") ||
        strstr(tail_window, "<|endoftext|>") ||
        strstr(tail_window, "<end_of_turn>") ||
        strstr(tail_window, "<|end|>") ||
        strstr(tail_window, "</s>")) {
        return 1;
    }
    if (check_qa &&
        (strstr(tail_window, "\nQuestion:") ||
         strstr(tail_window, "\nQ:") ||
         strstr(tail_window, "\nUser:"))) {
        return 1;
    }
    return 0;
}