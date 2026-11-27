#if defined(_WIN32) && !defined(__USE_MINGW_ANSI_STDIO)
#define __USE_MINGW_ANSI_STDIO 0
#endif

#include "../common/qmx_app_common.h"
#include "../common/qmx_profiles.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double ppl_token_nll(const float *logits, uint32_t vocab_size, int target) {
    if (!logits || vocab_size == 0u || target < 0 || (uint32_t)target >= vocab_size) return 0.0;
    double max_l = (double)logits[0];
    for (uint32_t i = 1u; i < vocab_size; i++) {
        double v = (double)logits[i];
        if (v > max_l) max_l = v;
    }
    double sum_exp = 0.0;
    for (uint32_t i = 0u; i < vocab_size; i++) {
        sum_exp += exp((double)logits[i] - max_l);
    }
    double log_z = max_l + log(sum_exp > 0.0 ? sum_exp : 1e-30);
    return log_z - (double)logits[target];
}

int main(int argc, char **argv) {
    const char *text_inline = NULL;
    const char *text_file = NULL;
    const char *profile_name = NULL;
    const char *profiles_file = "qmx_profiles.ini";
    int profiles_file_explicit = 0;
    uint32_t stride_opt = 0u;
    int max_chunks = 0;
    int quiet = 0;
    int verify_crc = 0;
    const char *cli_plugins[16];
    int cli_plugin_cnt = 0;
    const char *cli_plugin_dirs[8];
    int cli_plugin_dir_cnt = 0;

    QmxProfile cli_p;
    qmx_profile_init(&cli_p);

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            strncpy(cli_p.model, argv[++i], sizeof(cli_p.model) - 1u);
            cli_p.model[sizeof(cli_p.model) - 1u] = '\0';
            cli_p.mask |= QMX_PROF_F_MODEL;
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            strncpy(cli_p.tokenizer, argv[++i], sizeof(cli_p.tokenizer) - 1u);
            cli_p.tokenizer[sizeof(cli_p.tokenizer) - 1u] = '\0';
            cli_p.mask |= QMX_PROF_F_TOKENIZER;
        } else if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) text_file = argv[++i];
        else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) text_inline = argv[++i];
        else if (strcmp(argv[i], "--profile") == 0 && i + 1 < argc) profile_name = argv[++i];
        else if (strcmp(argv[i], "--profiles-file") == 0 && i + 1 < argc) {
            profiles_file = argv[++i];
            profiles_file_explicit = 1;
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            cli_p.ctx = qmx_app_parse_u32(argv[++i], 131072u, 512u);
            cli_p.mask |= QMX_PROF_F_CTX;
        } else if (strcmp(argv[i], "--stride") == 0 && i + 1 < argc) {
            stride_opt = qmx_app_parse_u32(argv[++i], 131072u, 0u);
        } else if (strcmp(argv[i], "--chunks") == 0 && i + 1 < argc) {
            max_chunks = qmx_app_parse_i32(argv[++i], 0, 1000000, 0);
        } else if (strcmp(argv[i], "-q") == 0 || strcmp(argv[i], "--quiet") == 0) {
            quiet = 1;
        } else if (strcmp(argv[i], "--verify-crc") == 0) {
            verify_crc = 1;
        } else if ((strcmp(argv[i], "-j") == 0 || strcmp(argv[i], "--threads") == 0) && i + 1 < argc) {
            const char *jv = argv[++i];
            cli_p.threads = (strcmp(jv, "auto") == 0) ? 0 : qmx_app_parse_i32(jv, 1, QMX_MAX_THREADS, 1);
            cli_p.mask |= QMX_PROF_F_THREADS;
        } else if (strcmp(argv[i], "--no-autotune") == 0) {
            cli_p.no_autotune = 1;
            cli_p.mask |= QMX_PROF_F_NO_AUTOTUNE;
        } else if (strcmp(argv[i], "--plugin") == 0 && i + 1 < argc) {
            if (cli_plugin_cnt < 16) cli_plugins[cli_plugin_cnt++] = argv[++i];
            else i++;
        } else if (strcmp(argv[i], "--plugin-dir") == 0 && i + 1 < argc) {
            if (cli_plugin_dir_cnt < 8) cli_plugin_dirs[cli_plugin_dir_cnt++] = argv[++i];
            else i++;
        } else if (strcmp(argv[i], "--kv8") == 0) {
            cli_p.kv_type = QMX_KV_Q8;
            cli_p.mask |= QMX_PROF_F_KV;
        } else if (strcmp(argv[i], "--kv16") == 0) {
            cli_p.kv_type = QMX_KV_FP16;
            cli_p.mask |= QMX_PROF_F_KV;
        } else if (strcmp(argv[i], "--stream") == 0 || strcmp(argv[i], "-s") == 0) {
            cli_p.force_stream = 1;
            cli_p.mask |= QMX_PROF_F_STREAM;
        } else if (strcmp(argv[i], "--max-ram") == 0 && i + 1 < argc) {
            uint64_t mb = qmx_app_parse_u64(argv[++i], (uint64_t)(~(size_t)0) / (1024u * 1024u), 0u);
            cli_p.max_ram_bytes = (size_t)(mb * 1024ULL * 1024ULL);
            cli_p.mask |= QMX_PROF_F_MAX_RAM;
        } else if (strcmp(argv[i], "--max-ram-kb") == 0 && i + 1 < argc) {
            uint64_t kb = qmx_app_parse_u64(argv[++i], (uint64_t)(~(size_t)0) / 1024u, 0u);
            cli_p.max_ram_bytes = (size_t)(kb * 1024ULL);
            cli_p.mask |= QMX_PROF_F_MAX_RAM;
        } else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            cli_p.backend = qmx_app_parse_backend(argv[++i]);
            cli_p.mask |= QMX_PROF_F_BACKEND;
        }
    }

    QmxProfile resolved;
    qmx_profile_init(&resolved);
    const char *hint_model = (cli_p.mask & QMX_PROF_F_MODEL) ? cli_p.model : NULL;
    QmxProfile ini_p;
    int ini_rc = qmx_profile_load_ini(profiles_file, profile_name, hint_model, &ini_p);
    if (ini_rc == -1 && (profiles_file_explicit || profile_name != NULL)) {
        fprintf(stderr, "Error: failed to open profiles file %s\n", profiles_file);
        return 1;
    }
    if (ini_rc == -2) {
        fprintf(stderr, "Error: profile '%s' not found in %s\n", profile_name, profiles_file);
        return 1;
    }
    if (ini_rc == 0) qmx_profile_merge(&resolved, &ini_p);
    qmx_profile_merge(&resolved, &cli_p);

    const char *model_path = ((resolved.mask & QMX_PROF_F_MODEL) && resolved.model[0] != '\0') ? resolved.model : NULL;
    if (!model_path || (!text_file && !text_inline)) {
        fprintf(stderr, "Usage: %s -m model.qmx (-f corpus.txt | -p \"text\") [-c ctx] [--stride S] [--chunks N] [-j threads|auto] [-b auto|...] [--kv8|--kv16] [--stream]\n", argv[0]);
        return 1;
    }

    if ((resolved.mask & QMX_PROF_F_PLUGIN_DIR) && resolved.plugin_dir[0] != '\0') {
        (void)qmx_plugin_load_dir(resolved.plugin_dir);
    }
    if ((resolved.mask & QMX_PROF_F_PLUGIN) && resolved.plugin[0] != '\0') {
        (void)qmx_plugin_load_shared(resolved.plugin);
    }
    for (int i = 0; i < cli_plugin_dir_cnt; i++) (void)qmx_plugin_load_dir(cli_plugin_dirs[i]);
    for (int i = 0; i < cli_plugin_cnt; i++) (void)qmx_plugin_load_shared(cli_plugins[i]);

    char *text_heap = NULL;
    const char *corpus = text_inline;
    if (text_file) {
        FILE *fp = fopen(text_file, "rb");
        if (!fp) {
            fprintf(stderr, "Error: cannot open text file %s\n", text_file);
            qmx_plugin_unload_all_shared();
            return 1;
        }
        int64_t fsz = qmx_file_size64(fp);
        if (fsz <= 0 || fsz > 64 * 1024 * 1024) {
            fclose(fp);
            fprintf(stderr, "Error: invalid text file size for %s\n", text_file);
            qmx_plugin_unload_all_shared();
            return 1;
        }
        text_heap = (char *)malloc((size_t)fsz + 1u);
        if (!text_heap) {
            fclose(fp);
            qmx_plugin_unload_all_shared();
            return 1;
        }
        size_t rd = fread(text_heap, 1, (size_t)fsz, fp);
        text_heap[rd] = '\0';
        fclose(fp);
        corpus = text_heap;
    }

    uint32_t ctx_override = (resolved.ctx > 0u) ? resolved.ctx : 512u;
    QmxLoadParams lp = qmx_load_params_default();
    lp.override_max_seq = ctx_override;
    lp.backend          = resolved.backend;
    lp.kv_type          = resolved.kv_type;
    lp.force_stream     = resolved.force_stream;
    lp.max_ram_bytes    = resolved.max_ram_bytes;

    QmxModel model;
    QmxTokenizer tok;
    int has_tok = 0;
    int used_uri = 0;
    if (strstr(model_path, "://") != NULL) {
        if (qmx_app_load_uri_model_ex(model_path, &lp, verify_crc, &model, &tok, &has_tok) != 0) {
            fprintf(stderr, "Error: failed to load QMX URI %s\n", model_path);
            free(text_heap);
            qmx_plugin_unload_all_shared();
            return 1;
        }
        used_uri = 1;
    } else {
        if (verify_crc) {
            uint32_t stored_crc = 0u, calc_crc = 0u;
            int crc_rc = qmx_verify_crc_file(model_path, 0u, 0u, &stored_crc, &calc_crc);
            if (crc_rc < 0) {
                fprintf(stderr, "Error: CRC32 verification failed for %s (stored=0x%08X, calc=0x%08X)\n",
                        model_path, stored_crc, calc_crc);
                free(text_heap);
                qmx_plugin_unload_all_shared();
                return 1;
            } else if (crc_rc == 1) {
                fprintf(stderr, "[CRC] Warning: %s has no file_crc32 (legacy cartridge); header CRC OK.\n", model_path);
            } else if (!quiet) {
                fprintf(stderr, "[CRC] Verified OK: %s (file_crc32=0x%08X)\n", model_path, stored_crc);
            }
        }
        if (qmx_load_ex(&model, model_path, 0u, 0u, &lp) != 0) {
            fprintf(stderr, "Error: failed to load QMX cartridge %s\n", model_path);
            free(text_heap);
            qmx_plugin_unload_all_shared();
            return 1;
        }
    }

    QmxModelInfo info;
    qmx_get_model_info(&model, &info);
    if (resolved.backend == QMX_BACKEND_AUTO && !resolved.no_autotune && info.hidden_dim >= 256u) {
        QmxBackend tuned = qmx_autotune_backend(info.hidden_dim, info.intermediate_dim);
        if (tuned != info.backend) {
            qmx_bind_backend_ops(&model, tuned);
            qmx_get_model_info(&model, &info);
        }
    }
    int num_threads = resolved.threads;
    if (num_threads <= 0) {
        num_threads = resolved.no_autotune ? 1 : qmx_recommend_threads(&model);
    }
    qmx_set_num_threads(&model, num_threads);
    qmx_get_model_info(&model, &info);

    const char *tok_path = (resolved.mask & QMX_PROF_F_TOKENIZER) ? resolved.tokenizer : "";
    if (used_uri) {
        if (!has_tok && tok_path[0] != '\0') {
            has_tok = (qmx_tokenizer_load(&tok, tok_path, 0u, 0u) == 0);
        }
    } else {
        qmx_app_reload_tokenizer(&info, model_path, 0u, tok_path, &tok, &has_tok);
    }

    if (!has_tok) {
        fprintf(stderr, "Error: tokenizer is required for perplexity evaluation\n");
        qmx_free(&model);
        free(text_heap);
        qmx_plugin_unload_all_shared();
        return 1;
    }

    size_t clen = strlen(corpus);
    size_t tok_cap = clen + 8u;
    if (tok_cap < 64u) tok_cap = 64u;
    int *all_tokens = (int *)malloc(tok_cap * sizeof(int));
    if (!all_tokens) {
        qmx_tokenizer_free(&tok);
        qmx_free(&model);
        free(text_heap);
        qmx_plugin_unload_all_shared();
        return 1;
    }

    int n_tokens = 0;
    qmx_tokenizer_encode(&tok, corpus, all_tokens, &n_tokens, (int)tok_cap, 1);
    if (n_tokens < 2) {
        fprintf(stderr, "Error: corpus encoded to fewer than 2 tokens (%d)\n", n_tokens);
        free(all_tokens);
        qmx_tokenizer_free(&tok);
        qmx_free(&model);
        free(text_heap);
        qmx_plugin_unload_all_shared();
        return 1;
    }

    int W = (int)info.max_seq_len;
    if (W < 2) W = 2;
    int S = (stride_opt > 0u && (int)stride_opt <= W) ? (int)stride_opt : ((W >= 2) ? (W / 2) : 1);
    if (S < 1) S = 1;

    const char *kv_str = (info.kv_type == QMX_KV_Q8) ? "int8"
                       : ((info.kv_type == QMX_KV_FP16) ? "fp16" : "fp32");
    fprintf(stderr, "[QMX-PPL] tokens=%d | ctx=%d | stride=%d | backend=%s | kv=%s | threads=%d\n",
            n_tokens, W, S, info.backend_name, kv_str, info.num_threads);

    double total_nll = 0.0;
    double total_nll_sq = 0.0;
    uint64_t scored_tokens = 0u;
    uint64_t total_eval_forwards = 0u;
    int chunks_done = 0;
    int start = 0;
    int scored_until = 0;

    float t0 = qmx_app_time_ms();
    while (scored_until < n_tokens - 1) {
        int chunk_len = n_tokens - start;
        if (chunk_len > W) chunk_len = W;
        if (chunk_len < 2) break;

        int score_from = (chunks_done == 0) ? 0 : (scored_until - start);
        if (score_from < 0) score_from = 0;
        if (score_from >= chunk_len - 1) break;

        if (score_from > 0) {
            (void)qmx_forward_batch(&model, all_tokens + start, score_from, 0, NULL);
            total_eval_forwards += (uint64_t)score_from;
        }

        for (int pos = score_from; pos < chunk_len - 1; pos++) {
            float *logits = qmx_forward(&model, all_tokens[start + pos], pos);
            total_eval_forwards++;
            if (!logits) break;
            double nll = ppl_token_nll(logits, info.vocab_size, all_tokens[start + pos + 1]);
            total_nll += nll;
            total_nll_sq += nll * nll;
            scored_tokens++;
        }

        scored_until = start + chunk_len - 1;
        chunks_done++;

        if (!quiet && scored_tokens > 0u) {
            double cur_ppl = exp(total_nll / (double)scored_tokens);
            fprintf(stderr, "[%d] ppl=%.4f (scored=%llu/%d)\n",
                    chunks_done, cur_ppl, (unsigned long long)scored_tokens, n_tokens - 1);
        }

        if (max_chunks > 0 && chunks_done >= max_chunks) break;
        if (start + chunk_len >= n_tokens) break;
        start += S;
    }
    float elapsed_ms = qmx_app_time_ms() - t0;

    double avg_nll = (scored_tokens > 0u) ? (total_nll / (double)scored_tokens) : 0.0;
    double ppl = (scored_tokens > 0u) ? exp(avg_nll) : 0.0;
    double var = (scored_tokens > 1u)
               ? ((total_nll_sq / (double)scored_tokens) - (avg_nll * avg_nll))
               : 0.0;
    if (var < 0.0) var = 0.0;
    double ppl_err = (scored_tokens > 1u) ? (ppl * sqrt(var / (double)scored_tokens)) : 0.0;
    double tok_s = (elapsed_ms > 0.0f) ? ((double)total_eval_forwards * 1000.0 / (double)elapsed_ms) : 0.0;

    printf("PPL = %.4f +/- %.4f (avg_nll=%.4f, scored=%llu tokens, chunks=%d)\n",
           ppl, ppl_err, avg_nll, (unsigned long long)scored_tokens, chunks_done);
    fprintf(stderr, "[PPL] tokens=%llu | chunks=%d | avg_nll=%.4f | ppl=%.4f +/- %.4f | time=%.2f ms (%.2f tok/s)\n",
            (unsigned long long)scored_tokens, chunks_done, avg_nll, ppl, ppl_err, (double)elapsed_ms, tok_s);

    free(all_tokens);
    qmx_tokenizer_free(&tok);
    qmx_free(&model);
    free(text_heap);
    qmx_plugin_unload_all_shared();
    return (scored_tokens > 0u && !isnan(ppl) && !isinf(ppl)) ? 0 : 1;
}
