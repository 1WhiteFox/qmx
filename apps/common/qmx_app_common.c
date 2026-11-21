/*
 * Copyright 2026 1WhiteFox
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 500
#endif
#if defined(_WIN32) && !defined(__USE_MINGW_ANSI_STDIO)
#define __USE_MINGW_ANSI_STDIO 0
#endif

#include "qmx_app_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#if !defined(__DJGPP__)
#include <sys/time.h>
#endif
#endif

#if defined(_WIN32) && defined(__i386__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((used, noinline)) void __qmx_i586_trampoline_cave(void) {
    __asm__ volatile (".fill 560, 1, 0x90\n\t");
}
#endif

int qmx_app_utf8_expected_len(unsigned char lead) {
    if (lead < 0x80) return 1;
    if ((lead & 0xE0) == 0xC0) return 2;
    if ((lead & 0xF0) == 0xE0) return 3;
    if ((lead & 0xF8) == 0xF0) return 4;
    return 1;
}

void qmx_app_utf8_printer_reset(QmxUtf8Printer *p) {
    if (!p) return;
    p->carry_len = 0;
}

void qmx_app_safe_print(QmxUtf8Printer *p, const char *piece) {
    if (!p || !piece || piece[0] == '\0') return;
    if (piece[1] == '\0' && p->carry_len == 0) {
        unsigned char b = (unsigned char)piece[0];
        if (b < 0x20 && b != '\n' && b != '\r' && b != '\t') return;
        if (b >= 0x7F && b < 0xC0) return;
    }
    char merged[512];
    int m_len = 0;
    if (p->carry_len > 0) {
        memcpy(merged, p->carry, (size_t)p->carry_len);
        m_len = p->carry_len;
        p->carry_len = 0;
    }
    for (const char *s = piece; *s != '\0' && m_len < (int)sizeof(merged) - 2; s++) {
        merged[m_len++] = *s;
    }
    merged[m_len] = '\0';
    int check_back = (m_len < 3) ? m_len : 3;
    for (int back = 1; back <= check_back; back++) {
        int idx = m_len - back;
        unsigned char b = (unsigned char)merged[idx];
        if ((b & 0xC0) == 0x80) continue;
        if (b >= 0xC0) {
            int need = qmx_app_utf8_expected_len(b);
            if (back < need) {
                memcpy(p->carry, merged + idx, (size_t)back);
                p->carry_len = back;
                m_len = idx;
                merged[m_len] = '\0';
            }
        }
        break;
    }
    if (m_len > 0) {
        fwrite(merged, 1, (size_t)m_len, stdout);
        fflush(stdout);
    }
}

float qmx_app_time_ms(void) {
#if defined(_WIN32)
    static LARGE_INTEGER freq;
    static LARGE_INTEGER t0;
    static int init = 0;
    LARGE_INTEGER now;
    if (!init) {
        if (QueryPerformanceFrequency(&freq) && freq.QuadPart > 0) {
            QueryPerformanceCounter(&t0);
            init = 1;
        } else {
            init = -1;
        }
    }
    if (init == 1 && QueryPerformanceCounter(&now)) {
        return (float)((double)(now.QuadPart - t0.QuadPart) * 1000.0 / (double)freq.QuadPart);
    }
    return (float)GetTickCount();
#elif !defined(__DJGPP__)
    static struct timeval t0;
    static int init = 0;
    struct timeval tv;
    if (gettimeofday(&tv, NULL) == 0) {
        if (!init) {
            t0 = tv;
            init = 1;
        }
        return (float)((double)(tv.tv_sec - t0.tv_sec) * 1000.0 +
                       (double)(tv.tv_usec - t0.tv_usec) * 0.001);
    }
    return (float)clock() * (1000.0f / (float)CLOCKS_PER_SEC);
#else
    static clock_t c0;
    static int init = 0;
    clock_t now = clock();
    if (!init) {
        c0 = now;
        init = 1;
    }
    return (float)(now - c0) * (1000.0f / (float)CLOCKS_PER_SEC);
#endif
}

const char *qmx_app_quant_label(uint16_t q) {
    if (q == QMX_QUANT_BINARY_1BIT) return "1-bit";
    if (q == QMX_QUANT_TERNARY_158) return "1.58-bit";
    if (q == QMX_QUANT_Q4_SYM)      return "Q4";
    if (q == QMX_QUANT_Q8_SYM)      return "Q8";
    if (q == QMX_QUANT_Q4_ROW32)    return "Q4_R32";
    if (q == QMX_QUANT_Q8_ROW32)    return "Q8_R32";
    return "FP32";
}

QmxBackend qmx_app_parse_backend(const char *s) {
    if (!s) return QMX_BACKEND_AUTO;
    if (strcmp(s, "scalar") == 0)     return QMX_BACKEND_SCALAR;
    if (strcmp(s, "p54c") == 0)       return QMX_BACKEND_P54C;
    if (strcmp(s, "mmx") == 0)        return QMX_BACKEND_MMX;
    if (strcmp(s, "3dnow") == 0)      return QMX_BACKEND_3DNOW;
    if (strcmp(s, "sse") == 0)        return QMX_BACKEND_SSE;
    if (strcmp(s, "sse2") == 0)       return QMX_BACKEND_SSE2;
    if (strcmp(s, "sse3") == 0)       return QMX_BACKEND_SSE3;
    if (strcmp(s, "ssse3") == 0)      return QMX_BACKEND_SSSE3;
    if (strcmp(s, "sse4.1") == 0 || strcmp(s, "sse41") == 0) return QMX_BACKEND_SSE41;
    if (strcmp(s, "sse4.2") == 0 || strcmp(s, "sse42") == 0) return QMX_BACKEND_SSE42;
    if (strcmp(s, "avx") == 0)        return QMX_BACKEND_AVX;
    if (strcmp(s, "avx2") == 0)       return QMX_BACKEND_AVX2;
    if (strcmp(s, "avx512") == 0)     return QMX_BACKEND_AVX512;
    if (strcmp(s, "neon") == 0)       return QMX_BACKEND_NEON;
    if (strcmp(s, "neon_dot") == 0)   return QMX_BACKEND_NEON_DOT;
    if (strcmp(s, "arm_dsp") == 0)    return QMX_BACKEND_ARM_DSP;
    if (strcmp(s, "mcu32") == 0)      return QMX_BACKEND_MCU32;
    if (strcmp(s, "cortex-m0") == 0  || strcmp(s, "m0") == 0)  return QMX_BACKEND_CORTEX_M0;
    if (strcmp(s, "cortex-m23") == 0 || strcmp(s, "m23") == 0) return QMX_BACKEND_CORTEX_M23;
    if (strcmp(s, "cortex-m3") == 0  || strcmp(s, "m3") == 0)  return QMX_BACKEND_CORTEX_M3;
    if (strcmp(s, "cortex-m4") == 0  || strcmp(s, "m4") == 0 || strcmp(s, "m7") == 0) return QMX_BACKEND_CORTEX_M4;
    if (strcmp(s, "cortex-m33") == 0 || strcmp(s, "m33") == 0) return QMX_BACKEND_CORTEX_M33;
    if (strcmp(s, "cortex-m55") == 0 || strcmp(s, "m55") == 0) return QMX_BACKEND_CORTEX_M55;
    return QMX_BACKEND_AUTO;
}

QmxBackend qmx_app_cycle_backend(QmxBackend cur) {
    int total = (int)QMX_BACKEND_COUNT;
    for (int step = 1; step <= total; step++) {
        QmxBackend next = (QmxBackend)(((int)cur + step) % total);
        if (next == QMX_BACKEND_SCALAR) return next;
        if (qmx_get_backend_ops(next) != &qmx_ops_scalar) return next;
    }
    return QMX_BACKEND_SCALAR;
}

QmxChatTemplate qmx_app_detect_chat_template(const QmxTokenizer *tok, int has_tok) {
    if (!has_tok || !tok) return QMX_CHAT_NONE;
    for (uint32_t i = 0; i < tok->special_count; i++) {
        const char *sp = qmx_tokenizer_piece(tok, tok->special_ids[i]);
        if (strcmp(sp, "<|start_header_id|>") == 0) return QMX_CHAT_LLAMA3;
        if (strcmp(sp, "<|im_start|>") == 0)        return QMX_CHAT_CHATML;
        if (strcmp(sp, "<start_of_turn>") == 0)     return QMX_CHAT_GEMMA2;
        if (strcmp(sp, "<|assistant|>") == 0)       return QMX_CHAT_PHI3;
    }
    return QMX_CHAT_NONE;
}

int qmx_app_load_cfg_file(const char *path, QmxRuntimeCfg *out_cfg) {
    if (!path || !out_cfg) return -1;
    memset(out_cfg, 0, sizeof(*out_cfg));
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;
    int ok = (fread(out_cfg, 1, sizeof(*out_cfg), fp) == sizeof(*out_cfg) &&
              memcmp(out_cfg->magic, "QMXCFG", 6) == 0);
    fclose(fp);
    if (!ok) {
        memset(out_cfg, 0, sizeof(*out_cfg));
        return -1;
    }
    return 0;
}

int qmx_app_save_cfg_file(const char *path, const QmxModelInfo *info, const char *model_path,
                          int steps, float temp, int lang_ru, size_t max_ram_bytes) {
    if (!path || !info) return -1;
    QmxRuntimeCfg cfg;
    memset(&cfg, 0, sizeof(cfg));
    memcpy(cfg.magic, "QMXCFG\0\0", 8);
    cfg.steps        = (uint32_t)steps;
    cfg.ctx_limit    = info->max_seq_len;
    cfg.temp         = temp;
    cfg.backend      = (uint8_t)info->backend;
    cfg.use_kv8      = (uint8_t)info->kv_type;
    cfg.force_stream = (uint8_t)(info->stream_mode ? 1 : 0);
    cfg.lang         = (uint8_t)(lang_ru ? 1 : 0);
    cfg.max_ram_kb   = (uint32_t)(max_ram_bytes / 1024u);
    if (model_path && model_path[0] != '\0') {
        strncpy(cfg.model_file, model_path, sizeof(cfg.model_file) - 1);
    }
    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;
    size_t wr = fwrite(&cfg, 1, sizeof(cfg), fp);
    fclose(fp);
    return (wr == sizeof(cfg)) ? 0 : -1;
}

int qmx_app_probe_self_bundle(const char *exe_path, uint64_t *out_mod_off, uint64_t *out_mod_sz,
                              QmxRuntimeCfg *out_cfg) {
    FILE *f = fopen(exe_path, "rb");
    if (!f) return 0;
    int64_t fsz = qmx_file_size64(f);
    if (fsz < (int64_t)(sizeof(QmxBundleTrailer) + sizeof(QmxHeader))) { fclose(f); return 0; }
    if (qmx_fseek64(f, fsz - (int64_t)sizeof(QmxBundleTrailer), SEEK_SET) != 0) { fclose(f); return 0; }
    QmxBundleTrailer tr;
    if (fread(&tr, 1, sizeof(tr), f) != sizeof(tr) || memcmp(tr.magic, "QMXP", 4) != 0) {
        fclose(f);
        return 0;
    }
    if ((uint64_t)tr.bundle_offset > (uint64_t)fsz ||
        (uint64_t)tr.bundle_size > (uint64_t)fsz - (uint64_t)tr.bundle_offset) {
        fclose(f);
        return 0;
    }
    if (out_cfg && tr.cfg_offset > 0u &&
        (uint64_t)tr.cfg_offset <= (uint64_t)fsz - (uint64_t)sizeof(QmxRuntimeCfg)) {
        if (qmx_fseek64(f, (int64_t)tr.cfg_offset, SEEK_SET) == 0) {
            if (fread(out_cfg, 1, sizeof(QmxRuntimeCfg), f) != sizeof(QmxRuntimeCfg)) {
                memset(out_cfg, 0, sizeof(QmxRuntimeCfg));
            }
        }
    }
    fclose(f);
    *out_mod_off = (uint64_t)tr.bundle_offset;
    *out_mod_sz  = (uint64_t)tr.bundle_size;
    return 1;
}

void qmx_app_reload_tokenizer(const QmxModelInfo *info, const char *active_path, uint64_t base_offset,
                              const char *fallback_tok_path, QmxTokenizer *tok, int *has_tok) {
    if (*has_tok) {
        qmx_tokenizer_free(tok);
        *has_tok = 0;
    }
    if (info && info->tokenizer_offset > 0u && info->tokenizer_size > 0u) {
        *has_tok = (qmx_tokenizer_load(tok, active_path,
                                       base_offset + (uint64_t)info->tokenizer_offset,
                                       (uint64_t)info->tokenizer_size) == 0);
    }
    if (!(*has_tok) && fallback_tok_path && fallback_tok_path[0] != '\0') {
        *has_tok = (qmx_tokenizer_load(tok, fallback_tok_path, 0u, 0u) == 0);
    }
}

int qmx_app_run_cli(QmxModel *model, const QmxTokenizer *tok, int has_tok,
                    const char *prompt, const QmxGenParams *params, int debug_tokens) {
    if (!model || !params) return -1;
    QmxModelInfo info;
    QmxStats stats;
    qmx_get_model_info(model, &info);
    qmx_get_stats(model, &stats);

    const char *kv_str = (info.kv_type == QMX_KV_Q8) ? "int8"
                       : ((info.kv_type == QMX_KV_FP16) ? "fp16" : "fp32");
    fprintf(stderr, "[QMX-Runtime] dim=%u | layers=%u | heads=%u/%u | vocab=%u | quant=0x%02x | backend=%s | kv=%s | mode=%s | arena=%.2f KB\n",
            info.hidden_dim, info.num_layers,
            info.num_heads, info.num_kv_heads,
            info.vocab_size, info.primary_quant,
            info.backend_name, kv_str,
            info.stream_mode ? "STREAM" : "RAM",
            (double)((float)stats.arena_used / 1024.0f));

    if (!prompt) prompt = "";
    char *cli_prompt_heap = NULL;
    const char *effective_prompt = prompt;
    int is_chat_wrapped = 0;
    QmxChatTemplate cli_tpl = qmx_app_detect_chat_template(tok, has_tok);
    if (cli_tpl != QMX_CHAT_NONE && prompt[0] != '\0' && prompt[0] != '<' && prompt[0] != '!') {
        size_t need_cap = strlen(prompt) + 192u;
        cli_prompt_heap = (char *)malloc(need_cap);
        if (cli_prompt_heap) {
            if (cli_tpl == QMX_CHAT_LLAMA3) {
                snprintf(cli_prompt_heap, need_cap,
                         "<|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", prompt);
            } else if (cli_tpl == QMX_CHAT_CHATML) {
                snprintf(cli_prompt_heap, need_cap,
                         "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", prompt);
            } else if (cli_tpl == QMX_CHAT_PHI3) {
                snprintf(cli_prompt_heap, need_cap,
                         "<|user|>\n%s<|end|>\n<|assistant|>\n", prompt);
            } else {
                snprintf(cli_prompt_heap, need_cap,
                         "<start_of_turn>user\n%s<end_of_turn>\n<start_of_turn>model\n", prompt);
            }
            effective_prompt = cli_prompt_heap;
            is_chat_wrapped = 1;
        }
    } else if (prompt[0] == '!') {
        effective_prompt = prompt + 1;
    }

    int max_prompt_tokens = (int)info.max_seq_len;
    int *prompt_tokens = (int *)malloc((size_t)max_prompt_tokens * sizeof(int));
    if (!prompt_tokens) {
        free(cli_prompt_heap);
        return -1;
    }

    int num_prompt_tokens = 0;
    if (has_tok) {
        qmx_tokenizer_encode(tok, effective_prompt, prompt_tokens, &num_prompt_tokens, max_prompt_tokens, 1);
    } else {
        prompt_tokens[0] = 1;
        num_prompt_tokens = 1;
    }
    if (num_prompt_tokens <= 0) {
        prompt_tokens[0] = 1;
        num_prompt_tokens = 1;
    }

    if (debug_tokens && has_tok) {
        fprintf(stderr, "[Debug-Tokens] encoded %d token(s):\n  ", num_prompt_tokens);
        for (int i = 0; i < num_prompt_tokens; i++) {
            fprintf(stderr, "%d:\"%s\" ", prompt_tokens[i], qmx_tokenizer_piece(tok, prompt_tokens[i]));
        }
        fprintf(stderr, "\n");
    }

    QmxUtf8Printer printer;
    qmx_app_utf8_printer_reset(&printer);
    int bos_id = has_tok ? tok->header.bos_id : 1;
    int eos_id = has_tok ? tok->header.eos_id : 2;
    for (int i = 1; i < num_prompt_tokens; i++) {
        if (has_tok && !is_chat_wrapped && prompt_tokens[i] != bos_id && prompt_tokens[i] != eos_id) {
            qmx_app_safe_print(&printer, qmx_tokenizer_decode(tok, prompt_tokens[i - 1], prompt_tokens[i]));
        }
    }

    QmxGenParams gp = *params;
    gp.check_qa_stop = is_chat_wrapped;
    if (gp.temp <= 0.0f) gp.rep_penalty = 1.0f;

    QmxGenState gen;
    qmx_gen_init(&gen, model, has_tok ? tok : NULL, &gp);
    qmx_gen_begin(&gen, prompt_tokens, num_prompt_tokens, &gp);

    float t_start = qmx_app_time_ms();
    float t_prefill_end = t_start;
    int prefill_marked = 0;

    QmxStepStatus st;
    QmxStepResult res;
    while ((st = qmx_step(&gen, &res)) > 0) {
        if (st == QMX_STEP_PREFILL) {
            if (res.prefill_done >= res.prefill_total) {
                t_prefill_end = qmx_app_time_ms();
                prefill_marked = 1;
            }
        } else if (st == QMX_STEP_TOKEN) {
            if (has_tok) {
                qmx_app_safe_print(&printer, res.piece);
            } else {
                printf("[%d]", res.token);
            }
        } else {
            break;
        }
    }
    printf("\n");

    float t_end = qmx_app_time_ms();
    if (!prefill_marked) t_prefill_end = t_end;

    int total_forwards = (int)(gen.prefill_tokens + gen.decode_tokens);
    float elapsed_ms = t_end - t_start;
    float tok_s = (elapsed_ms > 0.0f) ? ((float)total_forwards * 1000.0f / elapsed_ms) : 0.0f;

    float prefill_ms = t_prefill_end - t_start;
    float prefill_tps = (prefill_ms > 0.0f && gen.prefill_tokens > 0u)
                      ? ((float)gen.prefill_tokens * 1000.0f / prefill_ms) : 0.0f;
    float decode_ms = t_end - t_prefill_end;
    float decode_tps = (decode_ms > 0.0f && gen.decode_tokens > 0u)
                     ? ((float)gen.decode_tokens * 1000.0f / decode_ms) : 0.0f;

    fprintf(stderr, "[Stats] %d tokens in %.2f ms (%.2f tok/s) | step0_checksum=%.6f\n",
            total_forwards, (double)elapsed_ms, (double)tok_s, (double)gen.step0_checksum);
    fprintf(stderr, "[Perf] prefill=%u tok in %.2f ms (%.2f tok/s) | decode=%u tok in %.2f ms (%.2f tok/s)\n",
            gen.prefill_tokens, (double)prefill_ms, (double)prefill_tps,
            gen.decode_tokens, (double)decode_ms, (double)decode_tps);

    if (info.stream_mode) {
        qmx_get_stats(model, &stats);
        uint32_t total_emb = stats.emb_cache_hits + stats.emb_cache_misses;
        float hit_pct = (total_emb > 0u) ? ((float)stats.emb_cache_hits * 100.0f / (float)total_emb) : 0.0f;
        fprintf(stderr, "[Stream] disk_reads=%u (%.1f KB) | emb_lru=%u/%u hits (%.0f%%) | pinned=%u/%u tensors (%.1f KB)\n",
                stats.disk_reads,
                (double)((float)stats.disk_bytes_read / 1024.0f),
                stats.emb_cache_hits, total_emb, (double)hit_pct,
                stats.pinned_tensors, stats.tensor_count,
                (double)((float)stats.pin_pool_bytes / 1024.0f));
    }

    free(prompt_tokens);
    free(cli_prompt_heap);
    return 0;
}
