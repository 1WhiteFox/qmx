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

#ifdef __cplusplus
}
#endif

#endif
