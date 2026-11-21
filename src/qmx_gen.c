#include "../include/qmx_gen.h"
#include <string.h>

QmxGenParams qmx_gen_params_default(void) {
    QmxGenParams p;
    p.temp          = 0.0f;
    p.top_k         = 40;
    p.top_p         = 0.90f;
    p.min_p         = 0.0f;
    p.rep_penalty   = 1.0f;
    p.seed          = 42u;
    p.max_tokens    = 64;
    p.prefill_chunk = 16;
    p.check_qa_stop = 0;
    return p;
}

int qmx_gen_is_stop_token(const QmxTokenizer *tok, int token_id) {
    if (tok != NULL) {
        if (token_id == tok->header.bos_id ||
            token_id == tok->header.eos_id ||
            (tok->header.eot_id >= 0 && token_id == tok->header.eot_id)) {
            return 1;
        }
        const char *p = qmx_tokenizer_piece(tok, token_id);
        if (strcmp(p, "<|eot_id|>") == 0 ||
            strcmp(p, "<|im_end|>") == 0 ||
            strcmp(p, "<|endoftext|>") == 0 ||
            strcmp(p, "<end_of_turn>") == 0 ||
            strcmp(p, "</s>") == 0) {
            return 1;
        }
    } else if (token_id == 1 || token_id == 2) {
        return 1;
    }
    return 0;
}

void qmx_gen_set_tokenizer(QmxGenState *gen, const QmxTokenizer *tok) {
    if (!gen) return;
    gen->tok    = tok;
    gen->bos_id = tok ? tok->header.bos_id : 1;
    gen->eos_id = tok ? tok->header.eos_id : 2;
    gen->eot_id = tok ? tok->header.eot_id : -1;
}

void qmx_gen_reset(QmxGenState *gen) {
    if (!gen) return;
    gen->pos                = 0;
    gen->pending_token      = -1;
    gen->prev_token         = -1;
    gen->prompt_tokens      = NULL;
    gen->prompt_len         = 0;
    gen->prompt_off         = 0;
    gen->generated          = 0;
    gen->finished           = 1;
    gen->recent_cnt         = 0;
    gen->tail_window[0]     = '\0';
    gen->last_logits        = NULL;
    gen->step0_checksum     = 0.0f;
    gen->has_step0_checksum = 0;
    gen->prefill_tokens     = 0u;
    gen->decode_tokens      = 0u;
}

int qmx_gen_init(QmxGenState *gen, QmxModel *model, const QmxTokenizer *tok, const QmxGenParams *params) {
    if (!gen || !model) return -1;
    memset(gen, 0, sizeof(*gen));
    gen->model     = model;
    gen->params    = params ? *params : qmx_gen_params_default();
    gen->rng_state = gen->params.seed ? gen->params.seed : 42u;
    qmx_gen_set_tokenizer(gen, tok);
    qmx_gen_reset(gen);
    return 0;
}

int qmx_gen_flush(QmxGenState *gen) {
    if (!gen || !gen->model) return -1;
    if (gen->pending_token < 0) return gen->pos;

    int max_seq = (int)gen->model->header.max_seq_len;
    if (gen->pos >= max_seq) {
        int shift = max_seq / 2;
        if (shift < 8) shift = gen->pos;
        if (shift > gen->pos) shift = gen->pos;
        if (shift > 0) {
            gen->pos = qmx_shift_kv_cache(gen->model, gen->pos, shift);
        }
    }

    int tok = gen->pending_token;
    gen->last_logits = qmx_forward(gen->model, tok, gen->pos);
    gen->pos++;
    gen->decode_tokens++;
    qmx_push_recent_token(gen->recent_tokens, &gen->recent_cnt, QMX_GEN_RECENT_CAP, tok);
    gen->prev_token    = tok;
    gen->pending_token = -1;
    return gen->pos;
}

int qmx_gen_begin(QmxGenState *gen, const int *prompt_tokens, int n_tokens, const QmxGenParams *params) {
    if (!gen || !gen->model || !prompt_tokens || n_tokens <= 0) return -1;
    if (gen->pending_token >= 0) {
        (void)qmx_gen_flush(gen);
    }
    if (params) {
        uint32_t prev_seed = gen->params.seed;
        gen->params = *params;
        if (params->seed != prev_seed) {
            gen->rng_state = params->seed ? params->seed : 42u;
        }
    }
    gen->prompt_tokens      = prompt_tokens;
    gen->prompt_len         = n_tokens;
    gen->prompt_off         = 0;
    gen->generated          = 0;
    gen->finished           = 0;
    gen->tail_window[0]     = '\0';
    gen->last_logits        = NULL;
    gen->step0_checksum     = 0.0f;
    gen->has_step0_checksum = 0;
    gen->prefill_tokens     = 0u;
    gen->decode_tokens      = 0u;
    return 0;
}

QmxStepStatus qmx_step(QmxGenState *gen, QmxStepResult *out) {
    if (out) memset(out, 0, sizeof(*out));
    if (!gen || !gen->model) return QMX_STEP_ERROR;
    if (gen->finished) return QMX_STEP_DONE;

    QmxModel *model = gen->model;
    int max_seq = (int)model->header.max_seq_len;
    int shifted = 0;

    if (gen->prompt_off < gen->prompt_len) {
        if (!gen->prompt_tokens) {
            gen->finished = 1;
            return QMX_STEP_ERROR;
        }

        if (gen->prompt_off == 0 && gen->pos > 0 && gen->pos + gen->prompt_len + 16 >= max_seq) {
            int want_room = gen->params.max_tokens > 0 ? gen->params.max_tokens : 16;
            if (want_room > max_seq / 2) want_room = max_seq / 2;
            int needed = (gen->pos + gen->prompt_len + want_room) - max_seq;
            int shift = (gen->pos / 2 > needed) ? (gen->pos / 2) : needed;
            if (shift > gen->pos) shift = gen->pos;
            if (shift < 8) shift = gen->pos;
            if (shift > 0) {
                gen->pos = qmx_shift_kv_cache(model, gen->pos, shift);
                shifted = 1;
            }
        }

        int chunk = gen->prompt_len - gen->prompt_off;
        int limit = gen->params.prefill_chunk;
        if (limit <= 0) {
            limit = (model->arena.is_external || (model->stream_mode && model->state.stream_tile_cap <= 16u)) ? 1 : 16;
        }
        if (chunk > limit) chunk = limit;

        const int *slice = gen->prompt_tokens + gen->prompt_off;
        for (int i = 0; i < chunk; i++) {
            qmx_push_recent_token(gen->recent_tokens, &gen->recent_cnt, QMX_GEN_RECENT_CAP, slice[i]);
        }

        int batch_shifted = 0;
        float *chk_ptr = (gen->prompt_off == 0) ? &gen->step0_checksum : NULL;
        gen->last_logits = qmx_forward_batch_ex(model, slice, chunk, &gen->pos, chk_ptr, &batch_shifted);
        if (!gen->last_logits) {
            gen->finished = 1;
            return QMX_STEP_ERROR;
        }
        if (chk_ptr) gen->has_step0_checksum = 1;
        if (batch_shifted) shifted = 1;

        gen->prev_token     = slice[chunk - 1];
        gen->prompt_off    += chunk;
        gen->prefill_tokens += (uint32_t)chunk;

        if (out) {
            out->token         = gen->prev_token;
            out->piece         = NULL;
            out->pos           = gen->pos;
            out->prefill_done  = gen->prompt_off;
            out->prefill_total = gen->prompt_len;
            out->generated     = gen->generated;
            out->kv_shifted    = shifted;
        }
        return QMX_STEP_PREFILL;
    }

    if (gen->params.max_tokens > 0 && gen->generated >= gen->params.max_tokens) {
        gen->finished = 1;
        if (out) {
            out->pos           = gen->pos;
            out->prefill_done  = gen->prompt_len;
            out->prefill_total = gen->prompt_len;
            out->generated     = gen->generated;
        }
        return QMX_STEP_DONE;
    }

    if (gen->last_logits == NULL) {
        if (gen->pending_token < 0) {
            gen->finished = 1;
            return QMX_STEP_ERROR;
        }
        if (gen->pos >= max_seq) {
            int shift = max_seq / 2;
            if (shift < 8) shift = gen->pos;
            if (shift > gen->pos) shift = gen->pos;
            if (shift > 0) {
                gen->pos = qmx_shift_kv_cache(model, gen->pos, shift);
                shifted = 1;
            }
        }
        int tok_in = gen->pending_token;
        gen->last_logits = qmx_forward(model, tok_in, gen->pos);
        if (!gen->last_logits) {
            gen->finished = 1;
            return QMX_STEP_ERROR;
        }
        gen->pos++;
        gen->decode_tokens++;
        qmx_push_recent_token(gen->recent_tokens, &gen->recent_cnt, QMX_GEN_RECENT_CAP, tok_in);
        gen->prev_token    = tok_in;
        gen->pending_token = -1;
    }

    if (gen->params.rep_penalty > 1.0f) {
        qmx_apply_repetition_penalty(gen->last_logits, model->header.vocab_size,
                                     gen->recent_tokens, gen->recent_cnt,
                                     gen->params.rep_penalty,
                                     gen->bos_id, gen->eos_id, gen->eot_id);
    }

    int next = qmx_sample_ex(gen->last_logits, model->header.vocab_size,
                             gen->params.temp, gen->params.top_k,
                             gen->params.top_p, gen->params.min_p,
                             &gen->rng_state);
    gen->last_logits = NULL;

    if (qmx_gen_is_stop_token(gen->tok, next)) {
        int stop_tok = (gen->eot_id >= 0) ? gen->eot_id : ((gen->eos_id >= 0) ? gen->eos_id : next);
        gen->pending_token = stop_tok;
        gen->finished      = 1;
        if (out) {
            out->token         = next;
            out->piece         = gen->tok ? qmx_tokenizer_piece(gen->tok, next) : NULL;
            out->pos           = gen->pos;
            out->prefill_done  = gen->prompt_len;
            out->prefill_total = gen->prompt_len;
            out->generated     = gen->generated;
            out->kv_shifted    = shifted;
        }
        return QMX_STEP_EOS;
    }

    const char *piece = gen->tok ? qmx_tokenizer_decode(gen->tok, gen->prev_token, next) : NULL;
    if (piece && qmx_check_stop_sequence(gen->tail_window, sizeof(gen->tail_window),
                                         piece, gen->params.check_qa_stop)) {
        gen->pending_token = next;
        gen->finished      = 1;
        if (out) {
            out->token         = next;
            out->piece         = piece;
            out->pos           = gen->pos;
            out->prefill_done  = gen->prompt_len;
            out->prefill_total = gen->prompt_len;
            out->generated     = gen->generated;
            out->kv_shifted    = shifted;
        }
        return QMX_STEP_STOP;
    }

    gen->pending_token = next;
    gen->generated++;

    if (out) {
        out->token         = next;
        out->piece         = piece;
        out->pos           = gen->pos;
        out->prefill_done  = gen->prompt_len;
        out->prefill_total = gen->prompt_len;
        out->generated     = gen->generated;
        out->kv_shifted    = shifted;
    }
    return QMX_STEP_TOKEN;
}
