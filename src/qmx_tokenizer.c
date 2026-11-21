/*
 * Copyright 2026 1WhiteFox
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/qmx_tokenizer.h"
#include "../include/qmx_runtime.h"
#ifndef QMX_NO_MALLOC
#include <stdlib.h>
#endif
#include <string.h>

static int hex_nibble_val(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
}

static int is_special_id(const QmxTokenizer *tok, int id) {
    for (uint32_t i = 0; i < tok->special_count; i++) {
        if (tok->special_ids[i] == id) return 1;
    }
    return 0;
}

static int byte_category(unsigned char b) {
    if (b == '\n' || b == '\r') return 1;
    if (b == ' ' || b == '\t')  return 2;
    if (b >= '0' && b <= '9')   return 3;
    if ((b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z') || b >= 0x80u) return 4;
    return 5;
}

static int is_utf8_punct_piece(const char *s, size_t len) {
    if (len >= 2u && (unsigned char)s[0] == 0xC2u) {
        unsigned char b1 = (unsigned char)s[1];
        if (b1 == 0xA0u || b1 == 0xABu || b1 == 0xBBu) return 1;
    }
    if (len >= 3u && (unsigned char)s[0] == 0xE2u && (unsigned char)s[1] == 0x80u) {
        return 1;
    }
    return 0;
}

static int is_contraction_tail(const char *s, size_t len) {
    if (len == 1u) {
        char c = (char)(s[0] | 0x20);
        return (c == 's' || c == 't' || c == 'm' || c == 'd' ||
                c == 'r' || c == 'v' || c == 'l');
    }
    if (len == 2u) {
        char c0 = (char)(s[0] | 0x20);
        char c1 = (char)(s[1] | 0x20);
        return (c0 == 'r' && c1 == 'e') ||
               (c0 == 'v' && c1 == 'e') ||
               (c0 == 'l' && c1 == 'l');
    }
    return 0;
}

static int can_bpe_merge_bytelevel(const char *s1, const char *s2) {
    size_t l1 = strlen(s1);
    size_t l2 = strlen(s2);
    if (l1 == 0 || l2 == 0) return 0;
    unsigned char f1 = (unsigned char)s1[0];
    unsigned char e1 = (unsigned char)s1[l1 - 1u];
    unsigned char f2 = (unsigned char)s2[0];

    if (f1 == '\'' && l1 == 1u && is_contraction_tail(s2, l2)) {
        return 1;
    }
    if (f1 == '\'' && l1 == 2u && l2 == 1u) {
        char pair[2];
        pair[0] = s1[1];
        pair[1] = s2[0];
        if (is_contraction_tail(pair, 2u)) return 1;
    }
    if (f2 == '\'') {
        return (e1 == ' ' && l1 == 1u);
    }

    int c_e1 = is_utf8_punct_piece(s1, l1) ? 5 : byte_category(e1);
    int c_f2 = is_utf8_punct_piece(s2, l2) ? 5 : byte_category(f2);
    if (c_e1 == 1 || c_f2 == 1) return (c_e1 == 1 && c_f2 == 1);
    if (c_f2 == 2) return (c_e1 == 2);
    if (e1 == ' ' && l1 == 1u && (c_f2 == 4 || c_f2 == 5)) return 1;
    if (f1 == ' ' && l1 > 1u) {
        int c_second = byte_category((unsigned char)s1[1]);
        return (c_second == c_f2 && c_f2 != 3);
    }
    if (c_e1 == 3 && c_f2 == 3) return (l1 + l2 <= 3u);
    return (c_e1 == c_f2);
}

static uint32_t tokenizer_token_merge_score(const QmxTokenizer *tok, uint32_t id) {
    if (!tok->merges || tok->header.merges_count == 0u) return 0u;
    uint32_t n = tok->header.merges_count;
    uint32_t lo = 0u, hi = n;
    while (lo < hi) {
        uint32_t mid = lo + ((hi - lo) >> 1u);
        if (tok->merges[mid].left_tok < id) lo = mid + 1u;
        else hi = mid;
    }
    if (lo < n && tok->merges[lo].left_tok == id) {
        uint32_t start = lo;
        hi = n;
        while (lo < hi) {
            uint32_t mid = lo + ((hi - lo) >> 1u);
            if (tok->merges[mid].left_tok <= id) lo = mid + 1u;
            else hi = mid;
        }
        return (lo - start) + 1u;
    }
    for (uint32_t k = 0u; k < n; k++) {
        if (tok->merges[k].right_tok == id) return 1u;
    }
    return 0u;
}

static void tokenizer_register_byte(QmxTokenizer *tok, unsigned char b, uint32_t id) {
    int32_t cur = tok->byte_to_tok[b];
    if (cur == -1) {
        tok->byte_to_tok[b] = (int32_t)id;
        if (b == ' ' && tok->space_tok_id == -1) tok->space_tok_id = (int32_t)id;
        return;
    }
    if (tok->merges && tok->header.merges_count > 0u) {
        uint32_t sc_cur = tokenizer_token_merge_score(tok, (uint32_t)cur);
        uint32_t sc_new = tokenizer_token_merge_score(tok, id);
        if (sc_new > sc_cur) {
            tok->byte_to_tok[b] = (int32_t)id;
            if (b == ' ') tok->space_tok_id = (int32_t)id;
        }
    }
}

static int tokenizer_build_indices(QmxTokenizer *tok) {
    QmxTokenizerHeader *h = &tok->header;
    uint32_t pool_sz = tok->string_pool_size;
    for (uint32_t i = 0; i < h->vocab_size; i++) {
        uint32_t off = tok->vocab_offsets[i];
        if (off > pool_sz || (pool_sz > 0u && off == pool_sz && tok->is_xip)) {
            return -1;
        }
        const char *s = tok->string_pool + off;
        size_t len = strlen(s);
        if (len == 1u) {
            tokenizer_register_byte(tok, (unsigned char)s[0], i);
        } else if (len == 6u && s[0] == '<' && s[1] == '0' && (s[2] == 'x' || s[2] == 'X') && s[5] == '>') {
            int hi = hex_nibble_val(s[3]);
            int lo = hex_nibble_val(s[4]);
            if (hi >= 0 && lo >= 0) {
                tokenizer_register_byte(tok, (unsigned char)((hi << 4) | lo), i);
            }
        }

        int is_sp = 0;
        if ((int32_t)i == h->bos_id || (int32_t)i == h->eos_id || (int32_t)i == h->eot_id) {
            is_sp = (len > 0);
        } else if (len >= 5u && s[0] == '<' && s[1] == '|' && s[len - 2u] == '|' && s[len - 1u] == '>') {
            is_sp = 1;
        } else if (strcmp(s, "<s>") == 0 || strcmp(s, "</s>") == 0 ||
                   strcmp(s, "<unk>") == 0 || strcmp(s, "<pad>") == 0 ||
                   strcmp(s, "<bos>") == 0 || strcmp(s, "<eos>") == 0 ||
                   strcmp(s, "<start_of_turn>") == 0 || strcmp(s, "<end_of_turn>") == 0 ||
                   strcmp(s, "<think>") == 0 || strcmp(s, "</think>") == 0) {
            is_sp = 1;
        }

        if (is_sp && tok->special_count < QMX_MAX_SPECIAL_TOKENS) {
            tok->special_ids[tok->special_count++] = (int32_t)i;
        }
    }
    return 0;
}

static void tokenizer_init_base(QmxTokenizer *tok) {
    memset(tok, 0, sizeof(QmxTokenizer));
    tok->space_tok_id = -1;
    for (int i = 0; i < 256; i++) {
        tok->byte_to_tok[i] = -1;
        tok->byte_pieces[i * 2] = (unsigned char)i;
        tok->byte_pieces[i * 2 + 1] = '\0';
    }
}

int qmx_tokenizer_load_from_memory(QmxTokenizer *tok, const void *buf, size_t size) {
    if (!tok || !buf || size < sizeof(QmxTokenizerHeader) + sizeof(uint32_t)) return -1;
    tokenizer_init_base(tok);

    const uint8_t *base = (const uint8_t *)buf;
    memcpy(&tok->header, base, sizeof(QmxTokenizerHeader));
    QmxTokenizerHeader *h = &tok->header;
    if (h->magic != QMX_TOKENIZER_MAGIC ||
        h->vocab_size == 0u || h->vocab_size > 512000u ||
        h->merges_count > 2000000u) {
        return -1;
    }

    size_t offsets_bytes   = (size_t)h->vocab_size * sizeof(uint32_t);
    size_t merges_bytes    = (size_t)h->merges_count * sizeof(QmxBpeMergeEntry);
    size_t fixed_hdr_bytes = sizeof(QmxTokenizerHeader) + sizeof(uint32_t);
    if (offsets_bytes > size - fixed_hdr_bytes ||
        merges_bytes > size - fixed_hdr_bytes - offsets_bytes) {
        return -1;
    }

    size_t off_pos = sizeof(QmxTokenizerHeader);
    size_t psz_pos = off_pos + offsets_bytes;
    uint32_t pool_sz = 0;
    memcpy(&pool_sz, base + psz_pos, sizeof(uint32_t));
    if (pool_sz > 64u * 1024u * 1024u ||
        (size_t)pool_sz > size - fixed_hdr_bytes - offsets_bytes - merges_bytes) {
        return -1;
    }
    size_t pool_pos = psz_pos + sizeof(uint32_t);
    if (pool_sz == 0u || base[pool_pos + (size_t)pool_sz - 1u] != 0u) {
        return -1;
    }

    tok->is_xip = 1;
    tok->string_pool_size = pool_sz;
    if ((((uintptr_t)(base + off_pos)) & 3u) == 0u) {
        tok->vocab_offsets = (uint32_t *)(uintptr_t)(base + off_pos);
    } else {
#ifdef QMX_NO_MALLOC
        return -1;
#else
        tok->vocab_offsets = (uint32_t *)malloc(offsets_bytes);
        if (!tok->vocab_offsets) return -1;
        memcpy(tok->vocab_offsets, base + off_pos, offsets_bytes);
        tok->is_xip = 0;
#endif
    }

    if (tok->is_xip) {
        tok->string_pool = (char *)(uintptr_t)(base + pool_pos);
    } else {
#ifdef QMX_NO_MALLOC
        return -1;
#else
        tok->string_pool = (char *)malloc((size_t)pool_sz + 1u);
        if (!tok->string_pool) {
            qmx_tokenizer_free(tok);
            return -1;
        }
        memcpy(tok->string_pool, base + pool_pos, pool_sz);
        tok->string_pool[pool_sz] = '\0';
#endif
    }

    size_t merges_pos = pool_pos + (size_t)pool_sz;
    if (h->merges_count > 0u) {
        if (tok->is_xip && ((((uintptr_t)(base + merges_pos)) & 3u) == 0u)) {
            tok->merges = (QmxBpeMergeEntry *)(uintptr_t)(base + merges_pos);
            tok->merges_owned = 0;
        } else {
#ifdef QMX_NO_MALLOC
            qmx_tokenizer_free(tok);
            return -1;
#else
            tok->merges = (QmxBpeMergeEntry *)malloc(merges_bytes);
            if (!tok->merges) {
                qmx_tokenizer_free(tok);
                return -1;
            }
            memcpy(tok->merges, base + merges_pos, merges_bytes);
            tok->merges_owned = 1;
#endif
        }
    }

    if (tokenizer_build_indices(tok) != 0) {
        qmx_tokenizer_free(tok);
        return -1;
    }
    return 0;
}

int qmx_tokenizer_load_reader(QmxTokenizer *tok, const QmxReader *reader, uint64_t offset, uint64_t slice_bytes) {
#ifdef QMX_NO_MALLOC
    (void)tok; (void)reader; (void)offset; (void)slice_bytes;
    return -1;
#else
    if (!tok || !reader || !reader->read) return -1;
    tokenizer_init_base(tok);

    uint64_t rsz = reader->size;
    if (offset > rsz || (uint64_t)sizeof(QmxTokenizerHeader) > rsz - offset) {
        return -1;
    }

    uint64_t avail64 = (slice_bytes > 0u && slice_bytes <= rsz - offset)
                     ? slice_bytes
                     : (rsz - offset);
    size_t avail = (avail64 > (uint64_t)(~(size_t)0)) ? (~(size_t)0) : (size_t)avail64;

    uint64_t cur = offset;
    if (reader->read(reader->ctx, cur, &tok->header, sizeof(QmxTokenizerHeader)) != 0) {
        return -1;
    }
    cur += sizeof(QmxTokenizerHeader);

    QmxTokenizerHeader *h = &tok->header;
    if (h->magic != QMX_TOKENIZER_MAGIC ||
        h->vocab_size == 0u || h->vocab_size > 512000u ||
        h->merges_count > 2000000u) {
        return -1;
    }

    size_t offsets_bytes   = (size_t)h->vocab_size * sizeof(uint32_t);
    size_t merges_bytes    = (size_t)h->merges_count * sizeof(QmxBpeMergeEntry);
    size_t fixed_hdr_bytes = sizeof(QmxTokenizerHeader) + sizeof(uint32_t);
    if (avail < fixed_hdr_bytes ||
        offsets_bytes > avail - fixed_hdr_bytes ||
        merges_bytes > avail - fixed_hdr_bytes - offsets_bytes) {
        return -1;
    }

    tok->vocab_offsets = (uint32_t *)malloc(offsets_bytes);
    if (!tok->vocab_offsets) return -1;
    if (reader->read(reader->ctx, cur, tok->vocab_offsets, offsets_bytes) != 0) {
        qmx_tokenizer_free(tok);
        return -1;
    }
    cur += (uint64_t)offsets_bytes;

    uint32_t pool_sz = 0;
    if (reader->read(reader->ctx, cur, &pool_sz, sizeof(uint32_t)) != 0) {
        qmx_tokenizer_free(tok);
        return -1;
    }
    cur += sizeof(uint32_t);

    if (pool_sz > 64u * 1024u * 1024u ||
        (size_t)pool_sz > avail - fixed_hdr_bytes - offsets_bytes - merges_bytes) {
        qmx_tokenizer_free(tok);
        return -1;
    }

    tok->string_pool_size = pool_sz;
    tok->string_pool = (char *)malloc((size_t)pool_sz + 1u);
    if (!tok->string_pool) {
        qmx_tokenizer_free(tok);
        return -1;
    }
    if (pool_sz > 0u && reader->read(reader->ctx, cur, tok->string_pool, pool_sz) != 0) {
        qmx_tokenizer_free(tok);
        return -1;
    }
    tok->string_pool[pool_sz] = '\0';
    cur += (uint64_t)pool_sz;

    if (h->merges_count > 0u) {
        tok->merges = (QmxBpeMergeEntry *)malloc(merges_bytes);
        if (!tok->merges || reader->read(reader->ctx, cur, tok->merges, merges_bytes) != 0) {
            qmx_tokenizer_free(tok);
            return -1;
        }
        tok->merges_owned = 1;
    }

    if (tokenizer_build_indices(tok) != 0) {
        qmx_tokenizer_free(tok);
        return -1;
    }
    return 0;
#endif
}

#ifndef QMX_NO_STDIO

typedef struct {
    FILE    *fp;
    uint64_t size;
    uint64_t cur_pos;
    int      pos_valid;
} QmxTokFileCtx;

static int qmx_tok_file_read(void *ctx, uint64_t offset, void *dst, size_t size) {
    QmxTokFileCtx *fc = (QmxTokFileCtx *)ctx;
    if (!fc || !fc->fp || offset > fc->size || (uint64_t)size > fc->size - offset) return -1;
    if (size == 0u) return 0;
    if (!fc->pos_valid || fc->cur_pos != offset) {
        if (qmx_fseek64(fc->fp, (int64_t)offset, SEEK_SET) != 0) {
            fc->pos_valid = 0;
            return -1;
        }
        fc->cur_pos = offset;
        fc->pos_valid = 1;
    }
    if (fread(dst, 1, size, fc->fp) != size) {
        fc->pos_valid = 0;
        return -1;
    }
    fc->cur_pos += (uint64_t)size;
    return 0;
}

int qmx_tokenizer_load(QmxTokenizer *tok, const char *path, uint64_t offset, uint64_t slice_bytes) {
    if (!tok || !path) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    int64_t fsz = qmx_file_size64(f);
    if (fsz < 0) {
        fclose(f);
        return -1;
    }
    QmxTokFileCtx fc;
    fc.fp        = f;
    fc.size      = (uint64_t)fsz;
    fc.cur_pos   = 0u;
    fc.pos_valid = 0;

    QmxReader r;
    r.read  = qmx_tok_file_read;
    r.close = NULL;
    r.ctx   = &fc;
    r.size  = (uint64_t)fsz;

    int rc = qmx_tokenizer_load_reader(tok, &r, offset, slice_bytes);
    fclose(f);
    return rc;
}

#endif

void qmx_tokenizer_free(QmxTokenizer *tok) {
    if (!tok) return;
#ifndef QMX_NO_MALLOC
    if (!tok->is_xip) {
        if (tok->vocab_offsets) free(tok->vocab_offsets);
        if (tok->string_pool)   free(tok->string_pool);
    }
    if (tok->merges && (!tok->is_xip || tok->merges_owned)) {
        free(tok->merges);
    }
#endif
    tok->vocab_offsets = NULL;
    tok->string_pool   = NULL;
    tok->merges        = NULL;
    tok->is_xip        = 0;
    tok->merges_owned  = 0;
}

const char *qmx_tokenizer_piece(const QmxTokenizer *tok, int token) {
    if (!tok || !tok->string_pool || !tok->vocab_offsets) return "";
    if (token < 0 || (uint32_t)token >= tok->header.vocab_size) return "";
    return tok->string_pool + tok->vocab_offsets[(uint32_t)token];
}

const char *qmx_tokenizer_decode(const QmxTokenizer *tok, int prev_token, int token) {
    const char *piece = qmx_tokenizer_piece(tok, token);
    if (!piece || piece[0] == '\0') return "";
    int is_byte_level = (tok->header.flags & 1u) != 0;
    if (!is_byte_level && tok->header.vocab_size < 100000u &&
        prev_token == tok->header.bos_id && piece[0] == ' ') {
        piece++;
    }
    if (piece[0] == '<' && piece[1] == '0' && (piece[2] == 'x' || piece[2] == 'X') &&
        piece[5] == '>' && piece[6] == '\0') {
        int hi = hex_nibble_val(piece[3]);
        int lo = hex_nibble_val(piece[4]);
        if (hi >= 0 && lo >= 0) {
            unsigned int byte_val = (unsigned int)((hi << 4) | lo);
            return (const char *)tok->byte_pieces + byte_val * 2u;
        }
    }
    return piece;
}

const QmxBpeMergeEntry *qmx_bpe_find_merge(const QmxTokenizer *tok, uint32_t left, uint32_t right) {
    if (!tok || !tok->merges || tok->header.merges_count == 0u) return NULL;
    uint64_t target = ((uint64_t)left << 32) | (uint64_t)right;
    uint32_t low = 0u;
    uint32_t high = tok->header.merges_count;
    while (low < high) {
        uint32_t mid = low + ((high - low) >> 1);
        uint64_t key = ((uint64_t)tok->merges[mid].left_tok << 32) | (uint64_t)tok->merges[mid].right_tok;
        if (key == target) return &tok->merges[mid];
        if (key < target) low = mid + 1u;
        else high = mid;
    }
    return NULL;
}

static void eval_bpe_pair(const QmxTokenizer *tok, int is_byte_level,
                          int l_id, int r_id,
                          uint32_t *out_prio, uint32_t *out_target) {
    *out_prio = 0xFFFFFFFFu;
    *out_target = 0u;
    if (is_special_id(tok, l_id) || is_special_id(tok, r_id)) return;
    if (is_byte_level) {
        const char *s1 = qmx_tokenizer_piece(tok, l_id);
        const char *s2 = qmx_tokenizer_piece(tok, r_id);
        if (!can_bpe_merge_bytelevel(s1, s2)) return;
    }
    const QmxBpeMergeEntry *m = qmx_bpe_find_merge(tok, (uint32_t)l_id, (uint32_t)r_id);
    if (m != NULL) {
        *out_prio = m->priority;
        *out_target = m->target_tok;
    }
}

#ifndef QMX_NO_MALLOC

typedef struct {
    uint32_t prio;
    uint32_t target;
    int      idx;
    int      left_tok;
    int      right_tok;
} QmxBpeHeapItem;

static inline int bpe_heap_less(const QmxBpeHeapItem *a, const QmxBpeHeapItem *b) {
    if (a->prio != b->prio) return a->prio < b->prio;
    return a->idx < b->idx;
}

static void bpe_heap_push(QmxBpeHeapItem *heap, int *size, int cap, QmxBpeHeapItem item) {
    if (*size >= cap) return;
    int i = (*size)++;
    while (i > 0) {
        int parent = (i - 1) >> 1;
        if (!bpe_heap_less(&item, &heap[parent])) break;
        heap[i] = heap[parent];
        i = parent;
    }
    heap[i] = item;
}

static QmxBpeHeapItem bpe_heap_pop(QmxBpeHeapItem *heap, int *size) {
    QmxBpeHeapItem top = heap[0];
    int n = --(*size);
    if (n > 0) {
        QmxBpeHeapItem last = heap[n];
        int i = 0;
        for (;;) {
            int l = (i << 1) + 1;
            int r = l + 1;
            if (l >= n) break;
            int sm = (r < n && bpe_heap_less(&heap[r], &heap[l])) ? r : l;
            if (!bpe_heap_less(&heap[sm], &last)) break;
            heap[i] = heap[sm];
            i = sm;
        }
        heap[i] = last;
    }
    return top;
}

static int bpe_merge_heap(const QmxTokenizer *tok, int is_byte_level,
                          int *work_tokens, int count, int merge_start) {
    int n = count;
    int heap_cap = n * 3 + 16;

    int *prev_idx = (int *)malloc((size_t)n * sizeof(int));
    int *next_idx = (int *)malloc((size_t)n * sizeof(int));
    uint8_t *alive = (uint8_t *)malloc((size_t)n * sizeof(uint8_t));
    QmxBpeHeapItem *heap = (QmxBpeHeapItem *)malloc((size_t)heap_cap * sizeof(QmxBpeHeapItem));

    if (!prev_idx || !next_idx || !alive || !heap) {
        free(prev_idx);
        free(next_idx);
        free(alive);
        free(heap);
        return -1;
    }

    for (int i = 0; i < n; i++) {
        prev_idx[i] = i - 1;
        next_idx[i] = (i + 1 < n) ? (i + 1) : -1;
        alive[i] = 1u;
    }

    int heap_sz = 0;
    for (int i = merge_start; i < n - 1; i++) {
        uint32_t prio, target;
        eval_bpe_pair(tok, is_byte_level, work_tokens[i], work_tokens[i + 1], &prio, &target);
        if (prio != 0xFFFFFFFFu) {
            QmxBpeHeapItem it;
            it.prio      = prio;
            it.target    = target;
            it.idx       = i;
            it.left_tok  = work_tokens[i];
            it.right_tok = work_tokens[i + 1];
            bpe_heap_push(heap, &heap_sz, heap_cap, it);
        }
    }

    while (heap_sz > 0) {
        QmxBpeHeapItem top = bpe_heap_pop(heap, &heap_sz);
        int l = top.idx;
        if (!alive[l]) continue;
        int r = next_idx[l];
        if (r < 0 || !alive[r]) continue;
        if (work_tokens[l] != top.left_tok || work_tokens[r] != top.right_tok) continue;

        work_tokens[l] = (int)top.target;
        alive[r] = 0u;
        int nr = next_idx[r];
        next_idx[l] = nr;
        if (nr >= 0) prev_idx[nr] = l;

        int pl = prev_idx[l];
        if (pl >= merge_start && alive[pl]) {
            uint32_t prio, target;
            eval_bpe_pair(tok, is_byte_level, work_tokens[pl], work_tokens[l], &prio, &target);
            if (prio != 0xFFFFFFFFu) {
                QmxBpeHeapItem it;
                it.prio      = prio;
                it.target    = target;
                it.idx       = pl;
                it.left_tok  = work_tokens[pl];
                it.right_tok = work_tokens[l];
                bpe_heap_push(heap, &heap_sz, heap_cap, it);
            }
        }
        if (nr >= 0 && alive[nr]) {
            uint32_t prio, target;
            eval_bpe_pair(tok, is_byte_level, work_tokens[l], work_tokens[nr], &prio, &target);
            if (prio != 0xFFFFFFFFu) {
                QmxBpeHeapItem it;
                it.prio      = prio;
                it.target    = target;
                it.idx       = l;
                it.left_tok  = work_tokens[l];
                it.right_tok = work_tokens[nr];
                bpe_heap_push(heap, &heap_sz, heap_cap, it);
            }
        }
    }

    int out_cnt = 0;
    for (int i = 0; i < n; i++) {
        if (alive[i]) {
            work_tokens[out_cnt++] = work_tokens[i];
        }
    }

    free(prev_idx);
    free(next_idx);
    free(alive);
    free(heap);
    return out_cnt;
}

#endif

void qmx_tokenizer_encode(const QmxTokenizer *tok, const char *text,
                          int *tokens, int *n_tokens, int max_tokens, int add_bos) {
    if (!n_tokens) return;
    *n_tokens = 0;
    if (!tok || !tokens || max_tokens < 1) return;

    if (!text || text[0] == '\0') {
        if (add_bos && tok->header.bos_id >= 0 && (uint32_t)tok->header.bos_id < tok->header.vocab_size) {
            tokens[(*n_tokens)++] = tok->header.bos_id;
        }
        return;
    }

    int *work_tokens = tokens;
    int work_cap = max_tokens;
#ifndef QMX_NO_MALLOC
    size_t text_len = strlen(text);
    size_t max_raw_cap = (size_t)max_tokens * 4u + 4u;
    if (text_len + 4u < max_raw_cap) max_raw_cap = text_len + 4u;
    int *heap_work = NULL;

    if ((int)max_raw_cap > max_tokens && tok->merges && tok->header.merges_count > 0u) {
        heap_work = (int *)malloc(max_raw_cap * sizeof(int));
        if (heap_work) {
            work_tokens = heap_work;
            work_cap = (int)max_raw_cap;
        }
    }
#endif

    int count = 0;
    if (add_bos && tok->header.bos_id >= 0 && (uint32_t)tok->header.bos_id < tok->header.vocab_size) {
        work_tokens[count++] = tok->header.bos_id;
    }

    int merge_start = count;
    int is_byte_level = (tok->header.flags & 1u) != 0;

    if (!is_byte_level && !add_bos && tok->header.vocab_size >= 32000u &&
        text[0] != ' ' && text[0] != '\n' &&
        tok->space_tok_id >= 0 && count < work_cap) {
        work_tokens[count++] = tok->space_tok_id;
    }

    const char *p = text;
    while (*p != '\0' && count < work_cap) {
        if (*p == '<') {
            int matched_special = 0;
            for (uint32_t s = 0; s < tok->special_count; s++) {
                int sp_id = tok->special_ids[s];
                const char *sp_str = qmx_tokenizer_piece(tok, sp_id);
                size_t sp_len = strlen(sp_str);
                if (sp_len > 0u && strncmp(p, sp_str, sp_len) == 0) {
                    work_tokens[count++] = sp_id;
                    p += sp_len;
                    matched_special = 1;
                    break;
                }
            }
            if (matched_special) continue;
        }

        unsigned char b = (unsigned char)(*p++);
        int tid = tok->byte_to_tok[b];
        if (tid < 0) {
            tid = ((uint32_t)b < tok->header.vocab_size) ? (int)b : 0;
        }
        work_tokens[count++] = tid;
    }

    if (tok->merges && tok->header.merges_count > 0u && count - merge_start >= 2) {
        int merged = -1;
#ifndef QMX_NO_MALLOC
        merged = bpe_merge_heap(tok, is_byte_level, work_tokens, count, merge_start);
#endif
        if (merged >= 0) {
            count = merged;
        } else {
            while (1) {
                uint32_t best_prio = 0xFFFFFFFFu;
                int best_idx = -1;
                uint32_t best_target = 0u;
                for (int i = merge_start; i < count - 1; i++) {
                    uint32_t prio, target;
                    eval_bpe_pair(tok, is_byte_level, work_tokens[i], work_tokens[i + 1], &prio, &target);
                    if (prio < best_prio) {
                        best_prio = prio;
                        best_idx = i;
                        best_target = target;
                    }
                }
                if (best_idx < 0) break;
                work_tokens[best_idx] = (int)best_target;
                for (int i = best_idx + 1; i < count - 1; i++) {
                    work_tokens[i] = work_tokens[i + 1];
                }
                count--;
            }
        }
    }

    if (count > max_tokens) count = max_tokens;
#ifndef QMX_NO_MALLOC
    if (work_tokens != tokens) {
        memcpy(tokens, work_tokens, (size_t)count * sizeof(int));
        free(heap_work);
    }
#endif
    *n_tokens = count;
}
