#if defined(_WIN32) && !defined(__USE_MINGW_ANSI_STDIO)
#define __USE_MINGW_ANSI_STDIO 0
#endif

#include "qmx_profiles.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char prof_tolower(char c) {
    return (c >= 'A' && c <= 'Z') ? (char)(c + ('a' - 'A')) : c;
}

static int prof_strcasecmp(const char *a, const char *b) {
    while (*a && *b) {
        char ca = prof_tolower(*a++);
        char cb = prof_tolower(*b++);
        if (ca != cb) return (int)(unsigned char)ca - (int)(unsigned char)cb;
    }
    return (int)(unsigned char)prof_tolower(*a) - (int)(unsigned char)prof_tolower(*b);
}

static int prof_strncasecmp(const char *a, const char *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        char ca = prof_tolower(a[i]);
        char cb = prof_tolower(b[i]);
        if (ca != cb || ca == '\0') return (int)(unsigned char)ca - (int)(unsigned char)cb;
    }
    return 0;
}

static int prof_contains_ci(const char *haystack, const char *needle) {
    if (!haystack || !needle || needle[0] == '\0') return 0;
    size_t nlen = strlen(needle);
    for (const char *p = haystack; *p != '\0'; p++) {
        if (prof_strncasecmp(p, needle, nlen) == 0) return 1;
    }
    return 0;
}

static const char *prof_basename(const char *path) {
    if (!path) return "";
    const char *last = path;
    for (const char *p = path; *p != '\0'; p++) {
        if (*p == '/' || *p == '\\') last = p + 1;
    }
    return last;
}

static char *prof_trim(char *s) {
    while (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n') s++;
    size_t len = strlen(s);
    while (len > 0u && (s[len - 1u] == ' ' || s[len - 1u] == '\t' ||
                        s[len - 1u] == '\r' || s[len - 1u] == '\n')) {
        s[--len] = '\0';
    }
    return s;
}

static void prof_strip_inline_comment(char *val) {
    int in_quotes = 0;
    for (size_t i = 0; val[i] != '\0'; i++) {
        if (val[i] == '"') {
            in_quotes = !in_quotes;
        } else if (!in_quotes && (val[i] == '#' || val[i] == ';')) {
            if (i == 0u || val[i - 1u] == ' ' || val[i - 1u] == '\t') {
                val[i] = '\0';
                break;
            }
        }
    }
}

static char *prof_unquote(char *s) {
    s = prof_trim(s);
    size_t len = strlen(s);
    if (len >= 2u && ((s[0] == '"' && s[len - 1u] == '"') ||
                      (s[0] == '\'' && s[len - 1u] == '\''))) {
        s[len - 1u] = '\0';
        return s + 1;
    }
    return s;
}

static int prof_parse_bool(const char *s, int fallback) {
    if (!s || s[0] == '\0') return fallback;
    if (prof_strcasecmp(s, "1") == 0 || prof_strcasecmp(s, "true") == 0 ||
        prof_strcasecmp(s, "yes") == 0 || prof_strcasecmp(s, "on") == 0) return 1;
    if (prof_strcasecmp(s, "0") == 0 || prof_strcasecmp(s, "false") == 0 ||
        prof_strcasecmp(s, "no") == 0 || prof_strcasecmp(s, "off") == 0) return 0;
    return fallback;
}

static void prof_copy_str(char *dst, size_t cap, const char *src) {
    if (!dst || cap == 0u) return;
    if (!src) { dst[0] = '\0'; return; }
    strncpy(dst, src, cap - 1u);
    dst[cap - 1u] = '\0';
}

void qmx_profile_init(QmxProfile *p) {
    if (!p) return;
    memset(p, 0, sizeof(*p));
    p->backend       = QMX_BACKEND_AUTO;
    p->kv_type       = QMX_KV_FP32;
    p->threads       = 0;
    p->ctx           = 0u;
    p->steps         = 0;
    p->temp          = 0.0f;
    p->top_k         = 40;
    p->top_p         = 0.90f;
    p->min_p         = 0.0f;
    p->rep_penalty   = 1.15f;
    p->seed          = 42u;
    p->prefill_chunk = 32;
}

void qmx_profile_merge(QmxProfile *dst, const QmxProfile *src) {
    if (!dst || !src) return;
    uint32_t m = src->mask;
    if (m & QMX_PROF_F_MODEL)         prof_copy_str(dst->model, sizeof(dst->model), src->model);
    if (m & QMX_PROF_F_TOKENIZER)     prof_copy_str(dst->tokenizer, sizeof(dst->tokenizer), src->tokenizer);
    if (m & QMX_PROF_F_BACKEND)       dst->backend       = src->backend;
    if (m & QMX_PROF_F_KV)            dst->kv_type       = src->kv_type;
    if (m & QMX_PROF_F_THREADS)       dst->threads       = src->threads;
    if (m & QMX_PROF_F_CTX)           dst->ctx           = src->ctx;
    if (m & QMX_PROF_F_STEPS)         dst->steps         = src->steps;
    if (m & QMX_PROF_F_TEMP)          dst->temp          = src->temp;
    if (m & QMX_PROF_F_TOP_K)         dst->top_k         = src->top_k;
    if (m & QMX_PROF_F_TOP_P)         dst->top_p         = src->top_p;
    if (m & QMX_PROF_F_MIN_P)         dst->min_p         = src->min_p;
    if (m & QMX_PROF_F_REP_PENALTY)   dst->rep_penalty   = src->rep_penalty;
    if (m & QMX_PROF_F_SEED)          dst->seed          = src->seed;
    if (m & QMX_PROF_F_PREFILL_CHUNK) dst->prefill_chunk = src->prefill_chunk;
    if (m & QMX_PROF_F_STREAM)        dst->force_stream  = src->force_stream;
    if (m & QMX_PROF_F_MAX_RAM)       dst->max_ram_bytes = src->max_ram_bytes;
    if (m & QMX_PROF_F_NO_AUTOTUNE)   dst->no_autotune   = src->no_autotune;
    if (m & QMX_PROF_F_LANG_RU)       dst->lang_ru       = src->lang_ru;
    if (m & QMX_PROF_F_PLUGIN)        prof_copy_str(dst->plugin, sizeof(dst->plugin), src->plugin);
    if (m & QMX_PROF_F_PLUGIN_DIR)    prof_copy_str(dst->plugin_dir, sizeof(dst->plugin_dir), src->plugin_dir);
    dst->mask |= m;
}

void qmx_profile_from_cfg(QmxProfile *dst, const QmxRuntimeCfg *cfg) {
    qmx_profile_init(dst);
    if (!dst || !cfg || memcmp(cfg->magic, "QMXCFG", 6) != 0) return;
    if (cfg->steps > 0u) {
        dst->steps = (int)cfg->steps;
        dst->mask |= QMX_PROF_F_STEPS;
    }
    if (cfg->ctx_limit > 0u) {
        dst->ctx = cfg->ctx_limit;
        dst->mask |= QMX_PROF_F_CTX;
    }
    dst->temp = cfg->temp;
    dst->mask |= QMX_PROF_F_TEMP;
    if (cfg->backend < (uint8_t)QMX_BACKEND_COUNT || cfg->backend == (uint8_t)QMX_BACKEND_AUTO) {
        dst->backend = (QmxBackend)cfg->backend;
        dst->mask |= QMX_PROF_F_BACKEND;
    }
    dst->kv_type = (cfg->use_kv8 <= 2u) ? (QmxKvType)cfg->use_kv8 : QMX_KV_Q8;
    dst->mask |= QMX_PROF_F_KV;
    dst->force_stream = cfg->force_stream ? 1 : 0;
    dst->mask |= QMX_PROF_F_STREAM;
    dst->lang_ru = cfg->lang ? 1 : 0;
    dst->mask |= QMX_PROF_F_LANG_RU;
    if (cfg->max_ram_kb > 0u) {
        dst->max_ram_bytes = (size_t)cfg->max_ram_kb * 1024u;
        dst->mask |= QMX_PROF_F_MAX_RAM;
    }
    if (cfg->model_file[0] != '\0') {
        prof_copy_str(dst->model, sizeof(dst->model), cfg->model_file);
        dst->mask |= QMX_PROF_F_MODEL;
    }
}

static void prof_apply_kv(QmxProfile *p, const char *key, const char *val) {
    if (prof_strcasecmp(key, "model") == 0) {
        prof_copy_str(p->model, sizeof(p->model), val);
        p->mask |= QMX_PROF_F_MODEL;
    } else if (prof_strcasecmp(key, "tokenizer") == 0) {
        prof_copy_str(p->tokenizer, sizeof(p->tokenizer), val);
        p->mask |= QMX_PROF_F_TOKENIZER;
    } else if (prof_strcasecmp(key, "backend") == 0) {
        char low[32];
        size_t i = 0;
        for (; val[i] != '\0' && i + 1u < sizeof(low); i++) low[i] = prof_tolower(val[i]);
        low[i] = '\0';
        p->backend = qmx_app_parse_backend(low);
        p->mask |= QMX_PROF_F_BACKEND;
    } else if (prof_strcasecmp(key, "kv") == 0 || prof_strcasecmp(key, "kv_type") == 0) {
        if (prof_strcasecmp(val, "fp16") == 0 || prof_strcasecmp(val, "f16") == 0) p->kv_type = QMX_KV_FP16;
        else if (prof_strcasecmp(val, "q8") == 0 || prof_strcasecmp(val, "int8") == 0 || prof_strcasecmp(val, "i8") == 0) p->kv_type = QMX_KV_Q8;
        else p->kv_type = QMX_KV_FP32;
        p->mask |= QMX_PROF_F_KV;
    } else if (prof_strcasecmp(key, "threads") == 0) {
        p->threads = (prof_strcasecmp(val, "auto") == 0) ? 0 : qmx_app_parse_i32(val, 1, QMX_MAX_THREADS, 0);
        p->mask |= QMX_PROF_F_THREADS;
    } else if (prof_strcasecmp(key, "ctx") == 0 || prof_strcasecmp(key, "context") == 0) {
        p->ctx = qmx_app_parse_u32(val, 131072u, 0u);
        p->mask |= QMX_PROF_F_CTX;
    } else if (prof_strcasecmp(key, "steps") == 0 || prof_strcasecmp(key, "max_tokens") == 0) {
        p->steps = qmx_app_parse_i32(val, 1, 131072, 64);
        p->mask |= QMX_PROF_F_STEPS;
    } else if (prof_strcasecmp(key, "temp") == 0 || prof_strcasecmp(key, "temperature") == 0) {
        p->temp = qmx_app_parse_f32(val, 0.0f, 10.0f, 0.0f);
        p->mask |= QMX_PROF_F_TEMP;
    } else if (prof_strcasecmp(key, "top_k") == 0) {
        p->top_k = qmx_app_parse_i32(val, 0, QMX_MAX_TOP_K, 40);
        p->mask |= QMX_PROF_F_TOP_K;
    } else if (prof_strcasecmp(key, "top_p") == 0) {
        p->top_p = qmx_app_parse_f32(val, 0.0f, 1.0f, 0.90f);
        p->mask |= QMX_PROF_F_TOP_P;
    } else if (prof_strcasecmp(key, "min_p") == 0) {
        p->min_p = qmx_app_parse_f32(val, 0.0f, 1.0f, 0.0f);
        p->mask |= QMX_PROF_F_MIN_P;
    } else if (prof_strcasecmp(key, "rep_penalty") == 0) {
        p->rep_penalty = qmx_app_parse_f32(val, 1.0f, 10.0f, 1.15f);
        p->mask |= QMX_PROF_F_REP_PENALTY;
    } else if (prof_strcasecmp(key, "seed") == 0) {
        p->seed = qmx_app_parse_u32(val, 0xFFFFFFFFu, 42u);
        p->mask |= QMX_PROF_F_SEED;
    } else if (prof_strcasecmp(key, "prefill_chunk") == 0) {
        p->prefill_chunk = qmx_app_parse_i32(val, 1, 64, 32);
        p->mask |= QMX_PROF_F_PREFILL_CHUNK;
    } else if (prof_strcasecmp(key, "stream") == 0) {
        p->force_stream = prof_parse_bool(val, 0);
        p->mask |= QMX_PROF_F_STREAM;
    } else if (prof_strcasecmp(key, "max_ram_mb") == 0 || prof_strcasecmp(key, "max_ram") == 0) {
        uint64_t mb = qmx_app_parse_u64(val, (uint64_t)(~(size_t)0) / (1024u * 1024u), 0u);
        p->max_ram_bytes = (size_t)(mb * 1024ULL * 1024ULL);
        p->mask |= QMX_PROF_F_MAX_RAM;
    } else if (prof_strcasecmp(key, "max_ram_kb") == 0) {
        uint64_t kb = qmx_app_parse_u64(val, (uint64_t)(~(size_t)0) / 1024u, 0u);
        p->max_ram_bytes = (size_t)(kb * 1024ULL);
        p->mask |= QMX_PROF_F_MAX_RAM;
    } else if (prof_strcasecmp(key, "no_autotune") == 0) {
        p->no_autotune = prof_parse_bool(val, 0);
        p->mask |= QMX_PROF_F_NO_AUTOTUNE;
    } else if (prof_strcasecmp(key, "autotune") == 0) {
        p->no_autotune = prof_parse_bool(val, 1) ? 0 : 1;
        p->mask |= QMX_PROF_F_NO_AUTOTUNE;
    } else if (prof_strcasecmp(key, "lang") == 0) {
        p->lang_ru = (prof_strcasecmp(val, "ru") == 0) ? 1 : 0;
        p->mask |= QMX_PROF_F_LANG_RU;
    } else if (prof_strcasecmp(key, "plugin") == 0) {
        prof_copy_str(p->plugin, sizeof(p->plugin), val);
        p->mask |= QMX_PROF_F_PLUGIN;
    } else if (prof_strcasecmp(key, "plugin_dir") == 0) {
        prof_copy_str(p->plugin_dir, sizeof(p->plugin_dir), val);
        p->mask |= QMX_PROF_F_PLUGIN_DIR;
    }
}

int qmx_profile_load_ini(const char *ini_path, const char *profile_name,
                         const char *model_hint, QmxProfile *out_prof) {
    if (!out_prof) return -1;
    qmx_profile_init(out_prof);
    if (!ini_path || ini_path[0] == '\0') return -1;

    FILE *fp = fopen(ini_path, "rb");
    if (!fp) return -1;

    QmxProfile p_def;
    QmxProfile p_mod;
    QmxProfile p_prof;
    qmx_profile_init(&p_def);
    qmx_profile_init(&p_mod);
    qmx_profile_init(&p_prof);

    const char *model_base = prof_basename(model_hint);
    QmxProfile *cur = NULL;
    int found_profile = 0;

    char line[512];
    while (fgets(line, sizeof(line), fp) != NULL) {
        char *s = prof_trim(line);
        if (s[0] == '\0' || s[0] == '#' || s[0] == ';') continue;

        if (s[0] == '[') {
            char *rb = strchr(s + 1, ']');
            if (!rb) { cur = NULL; continue; }
            *rb = '\0';
            char *sec = prof_trim(s + 1);
            if (prof_strcasecmp(sec, "default") == 0) {
                cur = &p_def;
            } else if (prof_strncasecmp(sec, "profile.", 8u) == 0) {
                const char *pname = sec + 8;
                if (profile_name && profile_name[0] != '\0' && prof_strcasecmp(pname, profile_name) == 0) {
                    cur = &p_prof;
                    found_profile = 1;
                } else {
                    cur = NULL;
                }
            } else if (prof_strncasecmp(sec, "model.", 6u) == 0) {
                const char *mpat = sec + 6;
                if (model_base[0] != '\0' && prof_contains_ci(model_base, mpat)) {
                    cur = &p_mod;
                } else {
                    cur = NULL;
                }
            } else {
                cur = NULL;
            }
            continue;
        }

        if (!cur) continue;
        char *eq = strchr(s, '=');
        if (!eq) continue;
        *eq = '\0';
        char *k = prof_trim(s);
        char *v = eq + 1;
        prof_strip_inline_comment(v);
        v = prof_unquote(v);
        if (k[0] != '\0') prof_apply_kv(cur, k, v);
    }

    fclose(fp);

    qmx_profile_merge(out_prof, &p_def);
    qmx_profile_merge(out_prof, &p_mod);
    qmx_profile_merge(out_prof, &p_prof);

    if (profile_name && profile_name[0] != '\0' && !found_profile) {
        return -2;
    }
    return 0;
}

static const char *prof_backend_key(QmxBackend b) {
    switch (b) {
        case QMX_BACKEND_SCALAR:     return "scalar";
        case QMX_BACKEND_P54C:       return "p54c";
        case QMX_BACKEND_MMX:        return "mmx";
        case QMX_BACKEND_3DNOW:      return "3dnow";
        case QMX_BACKEND_SSE:        return "sse";
        case QMX_BACKEND_SSE2:       return "sse2";
        case QMX_BACKEND_SSE3:       return "sse3";
        case QMX_BACKEND_SSSE3:      return "ssse3";
        case QMX_BACKEND_SSE41:      return "sse4.1";
        case QMX_BACKEND_SSE42:      return "sse4.2";
        case QMX_BACKEND_AVX:        return "avx";
        case QMX_BACKEND_AVX2:       return "avx2";
        case QMX_BACKEND_AVX512:     return "avx512";
        case QMX_BACKEND_NEON:       return "neon";
        case QMX_BACKEND_NEON_DOT:   return "neon_dot";
        case QMX_BACKEND_ARM_DSP:    return "arm_dsp";
        case QMX_BACKEND_MCU32:      return "mcu32";
        case QMX_BACKEND_CORTEX_M0:  return "cortex-m0";
        case QMX_BACKEND_CORTEX_M23: return "cortex-m23";
        case QMX_BACKEND_CORTEX_M3:  return "cortex-m3";
        case QMX_BACKEND_CORTEX_M4:  return "cortex-m4";
        case QMX_BACKEND_CORTEX_M33: return "cortex-m33";
        case QMX_BACKEND_CORTEX_M55: return "cortex-m55";
        case QMX_BACKEND_AUTO:
        default:                     return "auto";
    }
}

int qmx_profile_save_ini(const char *ini_path, const char *section_name,
                         const QmxProfile *prof) {
    if (!ini_path || ini_path[0] == '\0' || !prof) return -1;
    const char *sec = (section_name && section_name[0] != '\0') ? section_name : "default";

    char *kept = NULL;
    size_t kept_len = 0u;
    size_t kept_cap = 0u;

    FILE *rfp = fopen(ini_path, "rb");
    if (rfp) {
        int skipping = 0;
        char line[512];
        while (fgets(line, sizeof(line), rfp) != NULL) {
            char probe[512];
            strncpy(probe, line, sizeof(probe) - 1u);
            probe[sizeof(probe) - 1u] = '\0';
            char *s = prof_trim(probe);
            if (s[0] == '[') {
                char *rb = strchr(s + 1, ']');
                if (rb) {
                    *rb = '\0';
                    char *cur_sec = prof_trim(s + 1);
                    skipping = (prof_strcasecmp(cur_sec, sec) == 0);
                } else {
                    skipping = 0;
                }
            }
            if (skipping) continue;
            size_t llen = strlen(line);
            if (kept_len + llen + 2u > kept_cap) {
                size_t ncap = (kept_cap == 0u) ? 4096u : (kept_cap * 2u);
                while (ncap < kept_len + llen + 2u) ncap *= 2u;
                if (ncap > 1024u * 1024u) { free(kept); fclose(rfp); return -1; }
                char *nb = (char *)realloc(kept, ncap);
                if (!nb) { free(kept); fclose(rfp); return -1; }
                kept = nb;
                kept_cap = ncap;
            }
            memcpy(kept + kept_len, line, llen);
            kept_len += llen;
            kept[kept_len] = '\0';
        }
        fclose(rfp);
    }

    FILE *wfp = fopen(ini_path, "wb");
    if (!wfp) {
        free(kept);
        return -1;
    }

    if (kept && kept_len > 0u) {
        while (kept_len > 0u && (kept[kept_len - 1u] == '\n' || kept[kept_len - 1u] == '\r')) {
            kept_len--;
        }
        if (kept_len > 0u) {
            fwrite(kept, 1, kept_len, wfp);
            fputs("\n\n", wfp);
        }
    }
    free(kept);

    uint32_t m = prof->mask;
    fprintf(wfp, "[%s]\n", sec);
    if ((m & QMX_PROF_F_MODEL) && prof->model[0] != '\0') {
        fprintf(wfp, "model = \"%s\"\n", prof->model);
    }
    if ((m & QMX_PROF_F_TOKENIZER) && prof->tokenizer[0] != '\0') {
        fprintf(wfp, "tokenizer = \"%s\"\n", prof->tokenizer);
    }
    if (m & QMX_PROF_F_BACKEND) {
        fprintf(wfp, "backend = %s\n", prof_backend_key(prof->backend));
    }
    if (m & QMX_PROF_F_KV) {
        const char *kv = (prof->kv_type == QMX_KV_FP16) ? "fp16" : ((prof->kv_type == QMX_KV_Q8) ? "q8" : "fp32");
        fprintf(wfp, "kv = %s\n", kv);
    }
    if (m & QMX_PROF_F_THREADS) {
        if (prof->threads <= 0) fputs("threads = auto\n", wfp);
        else fprintf(wfp, "threads = %d\n", prof->threads);
    }
    if (m & QMX_PROF_F_CTX) {
        fprintf(wfp, "ctx = %u\n", prof->ctx);
    }
    if (m & QMX_PROF_F_STEPS) {
        fprintf(wfp, "steps = %d\n", prof->steps);
    }
    if (m & QMX_PROF_F_TEMP) {
        fprintf(wfp, "temp = %.3f\n", (double)prof->temp);
    }
    if (m & QMX_PROF_F_TOP_K) {
        fprintf(wfp, "top_k = %d\n", prof->top_k);
    }
    if (m & QMX_PROF_F_TOP_P) {
        fprintf(wfp, "top_p = %.3f\n", (double)prof->top_p);
    }
    if (m & QMX_PROF_F_MIN_P) {
        fprintf(wfp, "min_p = %.3f\n", (double)prof->min_p);
    }
    if (m & QMX_PROF_F_REP_PENALTY) {
        fprintf(wfp, "rep_penalty = %.3f\n", (double)prof->rep_penalty);
    }
    if (m & QMX_PROF_F_SEED) {
        fprintf(wfp, "seed = %u\n", prof->seed);
    }
    if (m & QMX_PROF_F_PREFILL_CHUNK) {
        fprintf(wfp, "prefill_chunk = %d\n", prof->prefill_chunk);
    }
    if (m & QMX_PROF_F_STREAM) {
        fprintf(wfp, "stream = %s\n", prof->force_stream ? "true" : "false");
    }
    if (m & QMX_PROF_F_MAX_RAM) {
        if (prof->max_ram_bytes > 0u && (prof->max_ram_bytes % (1024u * 1024u)) == 0u) {
            fprintf(wfp, "max_ram_mb = %lu\n", (unsigned long)(prof->max_ram_bytes / (1024u * 1024u)));
        } else {
            fprintf(wfp, "max_ram_kb = %lu\n", (unsigned long)(prof->max_ram_bytes / 1024u));
        }
    }
    if (m & QMX_PROF_F_NO_AUTOTUNE) {
        fprintf(wfp, "no_autotune = %s\n", prof->no_autotune ? "true" : "false");
    }
    if (m & QMX_PROF_F_LANG_RU) {
        fprintf(wfp, "lang = %s\n", prof->lang_ru ? "ru" : "en");
    }
    if ((m & QMX_PROF_F_PLUGIN) && prof->plugin[0] != '\0') {
        fprintf(wfp, "plugin = \"%s\"\n", prof->plugin);
    }
    if ((m & QMX_PROF_F_PLUGIN_DIR) && prof->plugin_dir[0] != '\0') {
        fprintf(wfp, "plugin_dir = \"%s\"\n", prof->plugin_dir);
    }

    int err = ferror(wfp);
    fclose(wfp);
    return err ? -1 : 0;
}

