#include "../include/qmx_runtime.h"
#include "../include/qmx_tokenizer.h"
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static void fuzz_one_input(const uint8_t *data, size_t size) {
    QmxLoadParams lp = qmx_load_params_default();
    lp.override_max_seq = 16u;
    lp.backend          = QMX_BACKEND_SCALAR;
    lp.kv_type          = QMX_KV_FP32;
    lp.max_ram_bytes    = 1024u * 1024u;

    (void)qmx_verify_crc_memory(data, size, NULL, NULL);

    QmxModel m;
    if (qmx_load_from_memory_ex(&m, data, size, &lp) == 0) {
        if (m.header.vocab_size <= 4096u && m.header.num_layers <= 4u &&
            m.header.hidden_dim <= 256u && m.header.intermediate_dim <= 512u) {
            (void)qmx_forward(&m, 1, 0);
        }
        qmx_free(&m);
    }

    QmxTokenizer tok;
    if (qmx_tokenizer_load_from_memory(&tok, data, size) == 0) {
        int ids[32];
        int n = 0;
        qmx_tokenizer_encode(&tok, "Hello <|end|> QMX!", ids, &n, 32, 1);
        qmx_tokenizer_free(&tok);
    }
}

#ifdef QMX_LIBFUZZER

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    fuzz_one_input(data, size);
    return 0;
}

#else

static uint32_t fuzz_rng(uint32_t *st) {
    uint32_t x = *st;
    x ^= x << 13u;
    x ^= x >> 17u;
    x ^= x << 5u;
    *st = x;
    return x;
}

int main(int argc, char **argv) {
    const char *seed_path = (argc > 1) ? argv[1] : "test_q4.qmx";
    FILE *fp = fopen(seed_path, "rb");
    if (!fp) {
        fprintf(stderr, "[FAIL] fuzz_loader: cannot open seed %s\n", seed_path);
        return 1;
    }
    int64_t fsz = qmx_file_size64(fp);
    if (fsz <= 0) {
        fclose(fp);
        return 1;
    }
    uint8_t *seed = (uint8_t *)malloc((size_t)fsz);
    uint8_t *mut  = (uint8_t *)malloc((size_t)fsz);
    if (!seed || !mut || fread(seed, 1, (size_t)fsz, fp) != (size_t)fsz) {
        free(seed);
        free(mut);
        fclose(fp);
        return 1;
    }
    fclose(fp);

    uint32_t rng = 0xC0FFEE13u;
    const size_t len = (size_t)fsz;

    /* 1. Valid seed + boundary truncations */
    fuzz_one_input(seed, len);
    fuzz_one_input(seed, 0u);
    fuzz_one_input(seed, 1u);
    fuzz_one_input(seed, 64u);
    fuzz_one_input(seed, 128u);

    /* 2. Targeted 32-bit/64-bit header & tokenizer mutations */
    for (int it = 0; it < 1500; it++) {
        memcpy(mut, seed, len);
        uint32_t mode = fuzz_rng(&rng) % 5u;
        size_t cur_len = len;
        if (mode == 0u) {
            /* Corrupt random bytes in header/tdir and rehash header CRC */
            int flips = 1 + (int)(fuzz_rng(&rng) % 6u);
            for (int f = 0; f < flips; f++) {
                size_t pos = (size_t)(fuzz_rng(&rng) % (len < 2048u ? len : 2048u));
                mut[pos] ^= (uint8_t)(1u << (fuzz_rng(&rng) & 7u));
            }
            if (cur_len >= sizeof(QmxHeader)) {
                QmxHeader *h = (QmxHeader *)mut;
                h->header_crc32 = 0u;
                h->header_crc32 = qmx_crc32(h, sizeof(QmxHeader));
            }
        } else if (mode == 1u) {
            /* Random 32-bit overwrite inside header */
            size_t off = (size_t)(fuzz_rng(&rng) % (sizeof(QmxHeader) - 4u));
            uint32_t val = fuzz_rng(&rng);
            if ((val & 3u) == 0u) val = 0xFFFFFFFFu;
            else if ((val & 3u) == 1u) val = 0x80000000u;
            memcpy(mut + off, &val, 4u);
            QmxHeader *h = (QmxHeader *)mut;
            h->header_crc32 = 0u;
            h->header_crc32 = qmx_crc32(h, sizeof(QmxHeader));
        } else if (mode == 2u) {
            /* Random truncation */
            cur_len = (size_t)(fuzz_rng(&rng) % len);
        } else if (mode == 3u) {
            /* Tokenizer slice mutation */
            const QmxHeader *h = (const QmxHeader *)seed;
            if (h->tokenizer_size > 0u && (size_t)h->tokenizer_offset + (size_t)h->tokenizer_size <= len) {
                size_t tsz = (size_t)h->tokenizer_size;
                uint8_t *tptr = mut + h->tokenizer_offset;
                int flips = 1 + (int)(fuzz_rng(&rng) % 4u);
                for (int f = 0; f < flips; f++) {
                    tptr[fuzz_rng(&rng) % tsz] ^= (uint8_t)(fuzz_rng(&rng) & 0xFFu);
                }
                fuzz_one_input(tptr, tsz);
                continue;
            }
        } else {
            /* Pure random noise */
            cur_len = (size_t)(fuzz_rng(&rng) % 512u);
            for (size_t i = 0; i < cur_len; i++) mut[i] = (uint8_t)(fuzz_rng(&rng) & 0xFFu);
        }
        fuzz_one_input(mut, cur_len);
    }

    free(seed);
    free(mut);
    printf("[PASS] Loader & tokenizer mutation fuzzer passed (1500+ mutated inputs, 0 crashes/UB)!\n");
    return 0;
}

#endif
