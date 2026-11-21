#include "../include/qmx_runtime.h"
#include "../include/qmx_tokenizer.h"
#include "../include/qmx_sampler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int g_failures = 0;

static void write_synthetic_bpe_section(const char *path) {
    const char *vocab[] = {
        "<unk>", "<s>", "</s>", "<|eot_id|>",
        " ", "h", "e", "l", "o", "w", "r", "d",
        "he", "lo", "hel", "hello", "wo", "ld", "wor", "world",
        "hello world", "'", "t", "'t"
    };
    uint32_t vocab_size = (uint32_t)(sizeof(vocab) / sizeof(vocab[0]));

    uint32_t offsets[32];
    char pool[512];
    uint32_t pool_pos = 0;
    uint32_t max_len = 1;
    for (uint32_t i = 0; i < vocab_size; i++) {
        offsets[i] = pool_pos;
        size_t len = strlen(vocab[i]);
        if (len > max_len) max_len = (uint32_t)len;
        memcpy(pool + pool_pos, vocab[i], len + 1u);
        pool_pos += (uint32_t)(len + 1u);
    }

    QmxBpeMergeEntry merges[] = {
        { 5,  6,  12, 0 },
        { 7,  8,  13, 1 },
        { 12, 7,  14, 2 },
        { 14, 13, 15, 3 },
        { 9,  8,  16, 4 },
        { 7,  11, 17, 5 },
        { 16, 10, 18, 6 },
        { 18, 17, 19, 7 },
        { 21, 22, 23, 8 }
    };
    uint32_t merges_count = (uint32_t)(sizeof(merges) / sizeof(merges[0]));

    for (uint32_t i = 0; i < merges_count; i++) {
        for (uint32_t j = i + 1u; j < merges_count; j++) {
            uint64_t ki = ((uint64_t)merges[i].left_tok << 32) | merges[i].right_tok;
            uint64_t kj = ((uint64_t)merges[j].left_tok << 32) | merges[j].right_tok;
            if (kj < ki) {
                QmxBpeMergeEntry tmp = merges[i];
                merges[i] = merges[j];
                merges[j] = tmp;
            }
        }
    }

    QmxTokenizerHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = QMX_TOKENIZER_MAGIC;
    hdr.vocab_size = vocab_size;
    hdr.merges_count = merges_count;
    hdr.max_token_len = max_len;
    hdr.bos_id = 1;
    hdr.eos_id = 2;
    hdr.eot_id = 3;
    hdr.flags = 1;

    FILE *f = fopen(path, "wb");
    fwrite(&hdr, 1, sizeof(hdr), f);
    fwrite(offsets, sizeof(uint32_t), vocab_size, f);
    fwrite(&pool_pos, sizeof(uint32_t), 1, f);
    fwrite(pool, 1, pool_pos, f);
    fwrite(merges, sizeof(QmxBpeMergeEntry), merges_count, f);
    fclose(f);
}

static void test_bpe_tokenizer(const char *dummy_qmx_path) {
    const char *tmp_tok = "test_bpe_section.bin";
    write_synthetic_bpe_section(tmp_tok);

    QmxTokenizer tok;
    if (qmx_tokenizer_load(&tok, tmp_tok, 0, 0) != 0) {
        fprintf(stderr, "[FAIL] Could not load synthetic tokenizer\n");
        g_failures++;
        return;
    }

    int tokens[32];
    int n_tok = 0;
    qmx_tokenizer_encode(&tok, "hello world't<|eot_id|>", tokens, &n_tok, 32, 1);

    int expected[] = { 1, 15, 4, 19, 23, 3 };
    if (n_tok != 6) {
        fprintf(stderr, "[FAIL] BPE encode count: got=%d expected=6\n", n_tok);
        g_failures++;
    } else {
        for (int i = 0; i < 6; i++) {
            if (tokens[i] != expected[i]) {
                fprintf(stderr, "[FAIL] BPE token[%d]: got=%d expected=%d\n", i, tokens[i], expected[i]);
                g_failures++;
            }
        }
    }

    char decoded[64] = {0};
    int prev = -1;
    for (int i = 1; i < n_tok - 1; i++) {
        strcat(decoded, qmx_tokenizer_decode(&tok, prev, tokens[i]));
        prev = tokens[i];
    }
    if (strcmp(decoded, "hello world't") != 0) {
        fprintf(stderr, "[FAIL] BPE decode mismatch: got='%s'\n", decoded);
        g_failures++;
    }
    qmx_tokenizer_free(&tok);

    FILE *xf = fopen(tmp_tok, "rb");
    if (xf) {
        fseek(xf, 0, SEEK_END);
        long xsz = ftell(xf);
        fseek(xf, 0, SEEK_SET);
        uint8_t *xbuf = (uint8_t *)malloc((size_t)xsz);
        if (xbuf && fread(xbuf, 1, (size_t)xsz, xf) == (size_t)xsz) {
            QmxTokenizer xtok;
            if (qmx_tokenizer_load_from_memory(&xtok, xbuf, (size_t)xsz) != 0) {
                fprintf(stderr, "[FAIL] XIP tokenizer load failed\n");
                g_failures++;
            } else {
                int xids[32];
                int xn = 0;
                qmx_tokenizer_encode(&xtok, "hello world't<|eot_id|>", xids, &xn, 32, 1);
                if (xn != 6 || xids[1] != 15 || xids[4] != 23 || xids[5] != 3) {
                    fprintf(stderr, "[FAIL] XIP tokenizer encode mismatch (xn=%d)\n", xn);
                    g_failures++;
                }
                qmx_tokenizer_free(&xtok);
            }
        }
        free(xbuf);
        fclose(xf);
    }
    remove(tmp_tok);

    QmxModel m;
    if (qmx_load(&m, dummy_qmx_path, 0, QMX_BACKEND_SCALAR, 0, 0, 0) == 0) {
        QmxTokenizer dtok;
        if (qmx_tokenizer_load(&dtok, dummy_qmx_path, m.header.tokenizer_offset, m.header.tokenizer_size) != 0) {
            fprintf(stderr, "[FAIL] Could not load embedded tokenizer from %s\n", dummy_qmx_path);
            g_failures++;
        } else {
            int dt[16];
            int dn = 0;
            qmx_tokenizer_encode(&dtok, "QMX", dt, &dn, 16, 1);
            if (dn != 4 || dt[0] != 1 || dt[1] != 'Q' || dt[2] != 'M' || dt[3] != 'X') {
                fprintf(stderr, "[FAIL] Dummy cartridge encode mismatch (dn=%d)\n", dn);
                g_failures++;
            }
            qmx_tokenizer_free(&dtok);
        }
        qmx_free(&m);
    }
}

static void test_sampler_and_stop(void) {
    float logits[8] = { -1.0f, 0.5f, 4.0f, 2.0f, -2.0f, 0.1f, 3.5f, 0.0f };
    if (qmx_sample_argmax(logits, 8) != 2) {
        fprintf(stderr, "[FAIL] Argmax mismatch\n");
        g_failures++;
    }

    int recent[4] = { 2, 6, 2, 0 };
    qmx_apply_repetition_penalty(logits, 8, recent, 3, 2.0f, 1, 7, -1);
    if (fabsf(logits[2] - 2.0f) > 1e-5f || fabsf(logits[6] - 1.75f) > 1e-5f) {
        fprintf(stderr, "[FAIL] Repetition penalty mismatch: logits[2]=%f logits[6]=%f\n",
                (double)logits[2], (double)logits[6]);
        g_failures++;
    }

    uint32_t rng = 42u;
    int counts[8] = {0};
    for (int i = 0; i < 200; i++) {
        int s = qmx_sample(logits, 8, 0.8f, 3, 0.9f, &rng);
        if (s < 0 || s >= 8) {
            fprintf(stderr, "[FAIL] Sample out of range: %d\n", s);
            g_failures++;
            break;
        }
        counts[s]++;
    }
    if (counts[0] != 0 || counts[4] != 0) {
        fprintf(stderr, "[FAIL] Top-K=3 leaked low-probability tokens\n");
        g_failures++;
    }

    float nan_logits[4] = { 0.0f / 0.0f, -1.0f, 3.0f, 0.0f / 0.0f };
    if (qmx_sample_argmax(nan_logits, 4) != 2) {
        fprintf(stderr, "[FAIL] Argmax failed to ignore NaN logits\n");
        g_failures++;
    }
    float mp_logits[4] = { 10.0f, 1.0f, 0.0f, -5.0f };
    for (int i = 0; i < 64; i++) {
        int s = qmx_sample_ex(mp_logits, 4, 1.0f, 0, 1.0f, 0.20f, &rng);
        if (s != 0) {
            fprintf(stderr, "[FAIL] Min-P leaked low-prob token %d\n", s);
            g_failures++;
            break;
        }
    }

    char tail[64] = {0};
    if (qmx_check_stop_sequence(tail, sizeof(tail), "Hello <|eot", 1) != 0) {
        fprintf(stderr, "[FAIL] Premature stop trigger\n");
        g_failures++;
    }
    if (qmx_check_stop_sequence(tail, sizeof(tail), "_id|>", 1) != 1) {
        fprintf(stderr, "[FAIL] Failed to detect split <|eot_id|>\n");
        g_failures++;
    }
}

int main(int argc, char **argv) {
    const char *dummy_qmx = (argc > 1) ? argv[1] : "dummy.qmx";
    test_bpe_tokenizer(dummy_qmx);
    test_sampler_and_stop();

    if (g_failures == 0) {
        printf("[PASS] Stage 4 BPE O(log M) tokenizer & sampler tests passed!\n");
        return 0;
    }
    fprintf(stderr, "[FAIL] Stage 4 failures: %d\n", g_failures);
    return 1;
}