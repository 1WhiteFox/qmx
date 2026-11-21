#include "../include/qmx_gen.h"
#include "../include/qmx_kernels.h"
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

static void run_ref_greedy(const char *path, const int *prompt, int n_prompt, int n_gen,
                           int *out_tokens, float *out_step0_chk) {
    QmxModel m;
    EXPECT(qmx_load(&m, path, 64u, QMX_BACKEND_SCALAR, QMX_KV_FP32, 0, 0) == 0, "Load ref model");
    float *logits = NULL;
    for (int i = 0; i < n_prompt; i++) {
        logits = qmx_forward(&m, prompt[i], i);
        if (i == 0 && out_step0_chk) {
            float chk = 0.0f;
            for (uint32_t v = 0; v < m.header.vocab_size; v++) {
                chk += logits[v] * (float)((int)(v % 7u) - 3);
            }
            *out_step0_chk = chk;
        }
    }
    int pos = n_prompt;
    for (int g = 0; g < n_gen; g++) {
        int next = qmx_sample_argmax(logits, m.header.vocab_size);
        out_tokens[g] = next;
        if (g + 1 < n_gen) {
            logits = qmx_forward(&m, next, pos++);
        }
    }
    qmx_free(&m);
}

static void test_step_parity_and_chunks(const char *path) {
    int prompt[10] = { 1, 72, 101, 108, 108, 111, 32, 81, 77, 88 };
    int ref_toks[12];
    float ref_chk = 0.0f;
    run_ref_greedy(path, prompt, 10, 12, ref_toks, &ref_chk);

    int chunk_modes[2] = { 1, 8 };
    for (int cm = 0; cm < 2; cm++) {
        QmxModel m;
        QmxTokenizer tok;
        EXPECT(qmx_load(&m, path, 64u, QMX_BACKEND_SCALAR, QMX_KV_FP32, 0, 0) == 0, "Load step parity model");

        QmxModelInfo info;
        qmx_get_model_info(&m, &info);
        EXPECT(info.hidden_dim == 64u && info.num_layers == 2u && info.max_seq_len == 64u, "qmx_get_model_info fields valid");
        EXPECT(info.backend == QMX_BACKEND_SCALAR && strcmp(info.backend_name, "Scalar") == 0, "qmx_get_model_info backend valid");

        EXPECT(qmx_tokenizer_load(&tok, path, info.tokenizer_offset, info.tokenizer_size) == 0, "Load embedded tokenizer via info");

        QmxGenParams gp = qmx_gen_params_default();
        gp.temp          = 0.0f;
        gp.max_tokens    = 12;
        gp.prefill_chunk = chunk_modes[cm];

        QmxGenState gen;
        EXPECT(qmx_gen_init(&gen, &m, &tok, &gp) == 0, "qmx_gen_init");
        EXPECT(qmx_gen_begin(&gen, prompt, 10, &gp) == 0, "qmx_gen_begin");

        int prefill_steps = 0;
        int got_toks[12];
        int n_got = 0;
        QmxStepStatus st;
        QmxStepResult res;
        while ((st = qmx_step(&gen, &res)) > 0) {
            if (st == QMX_STEP_PREFILL) {
                prefill_steps++;
            } else if (st == QMX_STEP_TOKEN) {
                EXPECT(n_got < 12, "Generated token count within max_tokens");
                if (n_got < 12) got_toks[n_got++] = res.token;
            } else {
                break;
            }
        }
        EXPECT(st == QMX_STEP_DONE, "Terminated cleanly with QMX_STEP_DONE");
        EXPECT(prefill_steps == (chunk_modes[cm] == 1 ? 10 : 2), "Prefill yielded expected step count");
        EXPECT(n_got == 12, "Generated exact max_tokens count");
        EXPECT(gen.prefill_tokens == 10u && gen.decode_tokens == 11u, "Prefill/Decode counters accurate");
        EXPECT(gen.has_step0_checksum && fabsf(gen.step0_checksum - ref_chk) < 1e-4f, "step0_checksum matches reference");

        for (int i = 0; i < 12; i++) {
            EXPECT(got_toks[i] == ref_toks[i], "qmx_step token matches manual qmx_forward");
        }

        qmx_tokenizer_free(&tok);
        qmx_free(&m);
    }
}

static void test_multiturn_flush_and_window_shift(const char *path) {
    int turn1[4] = { 1, 65, 66, 67 };
    int turn2[4] = { 68, 69, 70, 71 };

    QmxModel m_step, m_mono;
    EXPECT(qmx_load(&m_step, path, 64u, QMX_BACKEND_SCALAR, QMX_KV_FP32, 0, 0) == 0, "Load multi-turn model");
    EXPECT(qmx_load(&m_mono, path, 64u, QMX_BACKEND_SCALAR, QMX_KV_FP32, 0, 0) == 0, "Load monolithic ref model");

    QmxGenParams gp = qmx_gen_params_default();
    gp.temp          = 0.0f;
    gp.max_tokens    = 10;
    gp.prefill_chunk = 4;

    QmxGenState gen;
    qmx_gen_init(&gen, &m_step, NULL, &gp);
    qmx_gen_begin(&gen, turn1, 4, &gp);

    int t1_toks[3];
    int n_t1 = 0;
    QmxStepResult res;
    while (n_t1 < 3) {
        QmxStepStatus st = qmx_step(&gen, &res);
        if (st == QMX_STEP_TOKEN) t1_toks[n_t1++] = res.token;
        else if (st != QMX_STEP_PREFILL) break;
    }
    EXPECT(n_t1 == 3, "Interrupted turn 1 after 3 tokens");
    int flushed_pos = qmx_gen_flush(&gen);
    EXPECT(flushed_pos == 7, "qmx_gen_flush committed pending 3rd token to pos=7");

    qmx_gen_begin(&gen, turn2, 4, &gp);
    int t2_got[4];
    int n_t2 = 0;
    while (n_t2 < 4) {
        QmxStepStatus st = qmx_step(&gen, &res);
        if (st == QMX_STEP_TOKEN) t2_got[n_t2++] = res.token;
        else if (st != QMX_STEP_PREFILL) break;
    }
    EXPECT(n_t2 == 4, "Turn 2 produced 4 tokens");

    int full_seq[11];
    memcpy(full_seq, turn1, 4 * sizeof(int));
    memcpy(full_seq + 4, t1_toks, 3 * sizeof(int));
    memcpy(full_seq + 7, turn2, 4 * sizeof(int));
    float *l_ref = NULL;
    for (int i = 0; i < 11; i++) l_ref = qmx_forward(&m_mono, full_seq[i], i);
    int mono_pos = 11;
    for (int g = 0; g < 4; g++) {
        int exp = qmx_sample_argmax(l_ref, m_mono.header.vocab_size);
        EXPECT(t2_got[g] == exp, "Multi-turn after flush matches monolithic KV history");
        if (g + 1 < 4) l_ref = qmx_forward(&m_mono, exp, mono_pos++);
    }
    qmx_free(&m_step);
    qmx_free(&m_mono);

    QmxModel m_win;
    EXPECT(qmx_load(&m_win, path, 16u, QMX_BACKEND_SCALAR, QMX_KV_FP16, 1, 0) == 0, "Load narrow-ctx stream model");

    QmxStats st0;
    qmx_get_stats(&m_win, &st0);
    EXPECT(st0.arena_used > 0u && st0.arena_used <= st0.arena_capacity, "Stats arena usage valid");
    EXPECT(st0.kv_cache_bytes > 0u && st0.tensor_count > 0u, "Stats KV and tensor counts valid");
    qmx_reset_stats(&m_win);

    int long_prompt[20];
    for (int i = 0; i < 20; i++) long_prompt[i] = 40 + i;
    QmxGenParams gp_win = qmx_gen_params_default();
    gp_win.temp          = 0.0f;
    gp_win.max_tokens    = 10;
    gp_win.prefill_chunk = 8;

    QmxGenState gen_win;
    qmx_gen_init(&gen_win, &m_win, NULL, &gp_win);
    qmx_gen_begin(&gen_win, long_prompt, 20, &gp_win);

    int saw_shift = 0;
    int gen_cnt = 0;
    QmxStepStatus wst;
    while ((wst = qmx_step(&gen_win, &res)) > 0) {
        if (res.kv_shifted) saw_shift = 1;
        EXPECT(res.pos >= 0 && res.pos <= 16, "pos stays strictly within max_seq_len=16");
        if (wst == QMX_STEP_TOKEN) gen_cnt++;
        else if (wst != QMX_STEP_PREFILL) break;
    }
    EXPECT(saw_shift == 1, "KV sliding window shift triggered on 20-token prompt in ctx=16");
    EXPECT(gen_cnt == 10, "Generated full 10 tokens across multiple window shifts");

    QmxStats st1;
    qmx_get_stats(&m_win, &st1);
    EXPECT(st1.disk_reads > 0u && st1.disk_bytes_read > 0u, "Stream stats tracked reads and bytes");
    qmx_free(&m_win);
}

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "test_q4.qmx";
    test_step_parity_and_chunks(path);
    test_multiturn_flush_and_window_shift(path);

    if (g_failures == 0) {
        printf("[PASS] qmx_step, multi-turn flush, KV window pos & telemetry tests passed!\n");
        return 0;
    }
    fprintf(stderr, "[FAIL] test_gen failures: %d\n", g_failures);
    return 1;
}
