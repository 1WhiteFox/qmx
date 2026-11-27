#if defined(_WIN32) && !defined(__USE_MINGW_ANSI_STDIO)
#define __USE_MINGW_ANSI_STDIO 0
#endif

#include "../common/qmx_app_common.h"
#include "../common/qmx_profiles.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__DJGPP__)
unsigned _stklen = 65536U;
#endif

int main(int argc, char **argv) {
    const char *prompt = "";
    const char *prompt_file = NULL;
    char *prompt_file_buf = NULL;
    const char *profile_name = NULL;
    const char *profiles_file = "qmx_profiles.ini";
    int profiles_file_explicit = 0;
    int debug_tokens = 0;
    int verify_crc = 0;
    uint64_t bundle_offset = 0;
    uint64_t bundle_size = 0;
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
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) prompt_file = argv[++i];
        else if (strcmp(argv[i], "--profile") == 0 && i + 1 < argc) profile_name = argv[++i];
        else if (strcmp(argv[i], "--profiles-file") == 0 && i + 1 < argc) {
            profiles_file = argv[++i];
            profiles_file_explicit = 1;
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            cli_p.steps = qmx_app_parse_i32(argv[++i], 1, 131072, 32);
            cli_p.mask |= QMX_PROF_F_STEPS;
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
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            cli_p.ctx = qmx_app_parse_u32(argv[++i], 131072u, 0u);
            cli_p.mask |= QMX_PROF_F_CTX;
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            cli_p.temp = qmx_app_parse_f32(argv[++i], 0.0f, 10.0f, 0.0f);
            cli_p.mask |= QMX_PROF_F_TEMP;
        } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
            cli_p.top_k = qmx_app_parse_i32(argv[++i], 0, QMX_MAX_TOP_K, 40);
            cli_p.mask |= QMX_PROF_F_TOP_K;
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            cli_p.top_p = qmx_app_parse_f32(argv[++i], 0.0f, 1.0f, 0.90f);
            cli_p.mask |= QMX_PROF_F_TOP_P;
        } else if (strcmp(argv[i], "--min-p") == 0 && i + 1 < argc) {
            cli_p.min_p = qmx_app_parse_f32(argv[++i], 0.0f, 1.0f, 0.0f);
            cli_p.mask |= QMX_PROF_F_MIN_P;
        } else if (strcmp(argv[i], "--rep-penalty") == 0 && i + 1 < argc) {
            cli_p.rep_penalty = qmx_app_parse_f32(argv[++i], 1.0f, 10.0f, 1.15f);
            cli_p.mask |= QMX_PROF_F_REP_PENALTY;
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            cli_p.seed = qmx_app_parse_u32(argv[++i], 0xFFFFFFFFu, 42u);
            cli_p.mask |= QMX_PROF_F_SEED;
        } else if (strcmp(argv[i], "--prefill-chunk") == 0 && i + 1 < argc) {
            cli_p.prefill_chunk = qmx_app_parse_i32(argv[++i], 1, 64, 32);
            cli_p.mask |= QMX_PROF_F_PREFILL_CHUNK;
        } else if (strcmp(argv[i], "--kv8") == 0) {
            cli_p.kv_type = QMX_KV_Q8;
            cli_p.mask |= QMX_PROF_F_KV;
        } else if (strcmp(argv[i], "--kv16") == 0) {
            cli_p.kv_type = QMX_KV_FP16;
            cli_p.mask |= QMX_PROF_F_KV;
        } else if (strcmp(argv[i], "--debug-tokens") == 0) {
            debug_tokens = 1;
        } else if (strcmp(argv[i], "--verify-crc") == 0) {
            verify_crc = 1;
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

    QmxRuntimeCfg rt_cfg;
    if (qmx_app_load_cfg_file("QMX.CFG", &rt_cfg) == 0) {
        QmxProfile cfg_p;
        qmx_profile_from_cfg(&cfg_p, &rt_cfg);
        cfg_p.mask &= ~QMX_PROF_F_TEMP;
        qmx_profile_merge(&resolved, &cfg_p);
    }

    const char *hint_model = (cli_p.mask & QMX_PROF_F_MODEL) ? cli_p.model
                           : ((resolved.mask & QMX_PROF_F_MODEL) ? resolved.model : NULL);
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
    if (ini_rc == 0) {
        if (!hint_model && (ini_p.mask & QMX_PROF_F_MODEL) && ini_p.model[0] != '\0') {
            char discovered[256];
            strncpy(discovered, ini_p.model, sizeof(discovered) - 1u);
            discovered[sizeof(discovered) - 1u] = '\0';
            (void)qmx_profile_load_ini(profiles_file, profile_name, discovered, &ini_p);
        }
        qmx_profile_merge(&resolved, &ini_p);
    }

    qmx_profile_merge(&resolved, &cli_p);

    const char *model_path = (resolved.mask & QMX_PROF_F_MODEL) && resolved.model[0] != '\0' ? resolved.model : NULL;
    if (!model_path && argv[0] && qmx_app_probe_self_bundle(argv[0], &bundle_offset, &bundle_size, &rt_cfg)) {
        model_path = argv[0];
        if (memcmp(rt_cfg.magic, "QMXCFG", 6) == 0) {
            QmxProfile bcfg_p;
            qmx_profile_from_cfg(&bcfg_p, &rt_cfg);
            bcfg_p.mask &= ~(QMX_PROF_F_MODEL | QMX_PROF_F_TEMP);
            qmx_profile_merge(&resolved, &bcfg_p);
            qmx_profile_merge(&resolved, &cli_p);
        }
    }

    if (!model_path) {
        fprintf(stderr, "Usage: %s -m model.qmx [-z tok.bin] [-p prompt | -f file] [--profile name] [--profiles-file ini] [-c ctx] [-n steps] [-j threads|auto] [-t temp] [-k top_k] [--top-p P] [--min-p P] [--stream] [--kv8 | --kv16] [-b auto|scalar|...]\n", argv[0]);
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

    const char *tok_path = (resolved.mask & QMX_PROF_F_TOKENIZER) ? resolved.tokenizer : "";
    int steps = resolved.steps;
    uint32_t ctx_override = resolved.ctx;
    float temp = resolved.temp;
    int top_k = resolved.top_k;
    float top_p = resolved.top_p;
    float min_p = resolved.min_p;
    float rep_penalty = resolved.rep_penalty;
    uint32_t seed = resolved.seed;
    QmxKvType kv_type = resolved.kv_type;
    int force_stream = resolved.force_stream;
    int no_autotune = resolved.no_autotune;
    int num_threads = resolved.threads;
    int prefill_chunk = resolved.prefill_chunk;
    size_t max_ram_bytes = resolved.max_ram_bytes;
    QmxBackend requested_backend = resolved.backend;

    if (prompt_file) {
        FILE *pf = fopen(prompt_file, "rb");
        if (pf) {
            int64_t psz = qmx_file_size64(pf);
            if (psz > 0 && psz < 4 * 1024 * 1024) {
                prompt_file_buf = (char *)malloc((size_t)psz + 1u);
                if (prompt_file_buf) {
                    size_t rd = fread(prompt_file_buf, 1, (size_t)psz, pf);
                    prompt_file_buf[rd] = '\0';
                    prompt = prompt_file_buf;
                }
            }
            fclose(pf);
        }
    }

    if (steps <= 0) steps = 32;
    if (ctx_override == 0) {
        size_t plen = strlen(prompt);
        uint32_t est_p = (plen > 128u) ? (uint32_t)(plen + 64u) : 128u;
        ctx_override = (prompt[0] != '\0') ? ((uint32_t)steps + est_p) : (uint32_t)steps;
    }

    if (!force_stream && max_ram_bytes == 0u && !no_autotune && bundle_size == 0u && strstr(model_path, "://") == NULL) {
        uint64_t avail_ram = qmx_detect_avail_ram_bytes();
        if (avail_ram > 0u) {
            FILE *szf = fopen(model_path, "rb");
            if (szf) {
                int64_t msz = qmx_file_size64(szf);
                fclose(szf);
                if (msz > (int64_t)(64ULL * 1024ULL * 1024ULL) &&
                    (uint64_t)msz + (256ULL * 1024ULL * 1024ULL) > avail_ram) {
                    force_stream = 1;
                    uint64_t cap = avail_ram / 2u;
                    if (cap < 64ULL * 1024ULL * 1024ULL) cap = 64ULL * 1024ULL * 1024ULL;
                    if (cap > (uint64_t)(~(size_t)0)) cap = (uint64_t)(~(size_t)0);
                    max_ram_bytes = (size_t)cap;
                }
            }
        }
    }

    QmxLoadParams lp = qmx_load_params_default();
    lp.override_max_seq = ctx_override;
    lp.backend          = requested_backend;
    lp.kv_type          = kv_type;
    lp.force_stream     = force_stream;
    lp.max_ram_bytes    = max_ram_bytes;

    QmxModel model;
    QmxTokenizer tokenizer;
    int has_tok = 0;
    int used_uri_reader = 0;
    if (strstr(model_path, "://") != NULL) {
        if (qmx_app_load_uri_model_ex(model_path, &lp, verify_crc, &model, &tokenizer, &has_tok) != 0) {
            fprintf(stderr, "Error: failed to load QMX URI %s\n", model_path);
            free(prompt_file_buf);
            qmx_plugin_unload_all_shared();
            return 1;
        }
        used_uri_reader = 1;
    } else {
        if (verify_crc) {
            uint32_t stored_crc = 0u, calc_crc = 0u;
            int crc_rc = qmx_verify_crc_file(model_path, bundle_offset, bundle_size, &stored_crc, &calc_crc);
            if (crc_rc < 0) {
                fprintf(stderr, "Error: CRC32 verification failed for %s (stored=0x%08X, calc=0x%08X)\n",
                        model_path, stored_crc, calc_crc);
                free(prompt_file_buf);
                qmx_plugin_unload_all_shared();
                return 1;
            } else if (crc_rc == 1) {
                fprintf(stderr, "[CRC] Warning: %s has no file_crc32 (legacy cartridge); header CRC OK.\n", model_path);
            } else {
                fprintf(stderr, "[CRC] Verified OK: %s (file_crc32=0x%08X)\n", model_path, stored_crc);
            }
        }
        if (qmx_load_ex(&model, model_path, bundle_offset, bundle_size, &lp) != 0) {
            fprintf(stderr, "Error: failed to load QMX cartridge %s\n", model_path);
            free(prompt_file_buf);
            qmx_plugin_unload_all_shared();
            return 1;
        }
    }


    QmxModelInfo info;
    qmx_get_model_info(&model, &info);
    if (requested_backend == QMX_BACKEND_AUTO && !no_autotune && info.hidden_dim >= 256u) {
        QmxBackend tuned = qmx_autotune_backend(info.hidden_dim, info.intermediate_dim);
        if (tuned != info.backend) {
            qmx_bind_backend_ops(&model, tuned);
            qmx_get_model_info(&model, &info);
        }
    }
    if (num_threads <= 0) {
        num_threads = no_autotune ? 1 : qmx_recommend_threads(&model);
    }
    qmx_set_num_threads(&model, num_threads);
    qmx_get_model_info(&model, &info);

    if (used_uri_reader) {
        if (!has_tok && tok_path[0] != '\0') {
            has_tok = (qmx_tokenizer_load(&tokenizer, tok_path, 0u, 0u) == 0);
        }
    } else {
        qmx_app_reload_tokenizer(&info, model_path, bundle_offset, tok_path, &tokenizer, &has_tok);
    }

    QmxGenParams gp = qmx_gen_params_default();
    gp.temp          = temp;
    gp.top_k         = top_k;
    gp.top_p         = top_p;
    gp.min_p         = min_p;
    gp.rep_penalty   = rep_penalty;
    gp.seed          = seed;
    gp.max_tokens    = steps;
    gp.prefill_chunk = prefill_chunk;

    int rc = qmx_app_run_cli(&model, &tokenizer, has_tok, prompt, &gp, debug_tokens);

    free(prompt_file_buf);
    if (has_tok) qmx_tokenizer_free(&tokenizer);
    qmx_free(&model);
    qmx_plugin_unload_all_shared();
    return (rc == 0) ? 0 : 1;
}
