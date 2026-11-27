/*
 * Copyright 2026 1WhiteFox
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef QMX_TOKENIZER_H
#define QMX_TOKENIZER_H

#include "qmx.h"

#ifdef __cplusplus
extern "C" {
#endif

#define QMX_MAX_SPECIAL_TOKENS 256u

typedef struct {
    QmxTokenizerHeader  header;
    uint32_t           *vocab_offsets;
    char               *string_pool;
    uint32_t            string_pool_size;
    QmxBpeMergeEntry   *merges;
    int                 is_xip;
    int                 merges_owned;
    int32_t             byte_to_tok[256];
    int32_t             space_tok_id;
    int32_t             special_ids[QMX_MAX_SPECIAL_TOKENS];
    uint32_t            special_count;
    unsigned char       byte_pieces[512];
} QmxTokenizer;

#ifdef QMX_NO_TOKENIZER
static inline int qmx_tokenizer_load_reader(QmxTokenizer *tok, const QmxReader *reader, uint64_t offset, uint64_t slice_bytes) {
    (void)tok; (void)reader; (void)offset; (void)slice_bytes; return -1;
}
#ifndef QMX_NO_STDIO
static inline int qmx_tokenizer_load(QmxTokenizer *tok, const char *path, uint64_t offset, uint64_t slice_bytes) {
    (void)tok; (void)path; (void)offset; (void)slice_bytes; return -1;
}
#endif
static inline int qmx_tokenizer_load_from_memory(QmxTokenizer *tok, const void *buf, size_t size) {
    (void)tok; (void)buf; (void)size; return -1;
}
static inline void qmx_tokenizer_free(QmxTokenizer *tok) {
    (void)tok;
}
static inline const char *qmx_tokenizer_piece(const QmxTokenizer *tok, int token) {
    (void)tok; (void)token; return "";
}
static inline const char *qmx_tokenizer_decode(const QmxTokenizer *tok, int prev_token, int token) {
    (void)tok; (void)prev_token; (void)token; return "";
}
static inline const QmxBpeMergeEntry *qmx_bpe_find_merge(const QmxTokenizer *tok, uint32_t left, uint32_t right) {
    (void)tok; (void)left; (void)right; return NULL;
}
static inline void qmx_tokenizer_encode(const QmxTokenizer *tok, const char *text,
                                        int *tokens, int *n_tokens, int max_tokens, int add_bos) {
    (void)tok; (void)text; (void)tokens; (void)max_tokens; (void)add_bos;
    if (n_tokens) *n_tokens = 0;
}
#else
int qmx_tokenizer_load_reader(QmxTokenizer *tok, const QmxReader *reader, uint64_t offset, uint64_t slice_bytes);
#ifndef QMX_NO_STDIO
int qmx_tokenizer_load(QmxTokenizer *tok, const char *path, uint64_t offset, uint64_t slice_bytes);
#endif
int qmx_tokenizer_load_from_memory(QmxTokenizer *tok, const void *buf, size_t size);
void qmx_tokenizer_free(QmxTokenizer *tok);

const char *qmx_tokenizer_piece(const QmxTokenizer *tok, int token);
const char *qmx_tokenizer_decode(const QmxTokenizer *tok, int prev_token, int token);

const QmxBpeMergeEntry *qmx_bpe_find_merge(const QmxTokenizer *tok, uint32_t left, uint32_t right);

void qmx_tokenizer_encode(const QmxTokenizer *tok, const char *text,
                          int *tokens, int *n_tokens, int max_tokens, int add_bos);

#endif
#ifdef __cplusplus
}
#endif

#endif
