#if defined(_WIN32) && !defined(__USE_MINGW_ANSI_STDIO)
#define __USE_MINGW_ANSI_STDIO 0
#endif

#include "../common/qmx_app_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__DJGPP__)
unsigned _stklen = 65536U;
#endif

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *tok_path = "";
    const char *prompt = "";
    const char *prompt_file = NULL;
    char *prompt_file_buf = NULL;
    int steps = 0;
    uint32_t ctx_override = 0;
    float temp = 0.0f;
    int top_k = 40;
    float top_p = 0.90f;
    float min_p = 0.0f;
    float rep_penalty = 1.15f;
    uint32_t seed = 42u;
    QmxKvType kv_type = QMX_KV_FP32;
    int force_stream = 0;
    int debug_tokens = 0;
    int num_threads = 1;
    int prefill_chunk = 16;
    size_t max_ram_bytes = 0;
    uint64_t bundle_offset = 0;
    uint64_t bundle_size = 0;
    QmxBackend requested_backend = QMX_BACKEND_AUTO;
    char cfg_model_path[256] = {0};

    QmxRuntimeCfg rt_cfg;
    if (qmx_app_load_cfg_file("QMX.CFG", &rt_cfg) == 0) {
        if (rt_cfg.steps > 0) steps = (int)rt_cfg.steps;
        if (rt_cfg.ctx_limit > 0) ctx_override = rt_cfg.ctx_limit;
        if (rt_cfg.backend < (uint8_t)QMX_BACKEND_COUNT || rt_cfg.backend == (uint8_t)QMX_BACKEND_AUTO) {
            requested_backend = (QmxBackend)rt_cfg.backend;
        }
        kv_type = (rt_cfg.use_kv8 <= 2u) ? (QmxKvType)rt_cfg.use_kv8 : QMX_KV_Q8;
        force_stream = rt_cfg.force_stream ? 1 : 0;
        if (rt_cfg.max_ram_kb > 0) max_ram_bytes = (size_t)rt_cfg.max_ram_kb * 1024u;
        if (rt_cfg.model_file[0] != '\0') {
            strncpy(cfg_model_path, rt_cfg.model_file, sizeof(cfg_model_path) - 1);
            model_path = cfg_model_path;
        }
    }

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) tok_path = argv[++i];
        else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) prompt_file = argv[++i];
        else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) steps = atoi(argv[++i]);
        else if ((strcmp(argv[i], "-j") == 0 || strcmp(argv[i], "--threads") == 0) && i + 1 < argc) num_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) ctx_override = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) temp = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) top_p = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--min-p") == 0 && i + 1 < argc) min_p = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--rep-penalty") == 0 && i + 1 < argc) rep_penalty = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--prefill-chunk") == 0 && i + 1 < argc) prefill_chunk = atoi(argv[++i]);
        else if (strcmp(argv[i], "--kv8") == 0) kv_type = QMX_KV_Q8;
        else if (strcmp(argv[i], "--kv16") == 0) kv_type = QMX_KV_FP16;
        else if (strcmp(argv[i], "--debug-tokens") == 0) debug_tokens = 1;
        else if (strcmp(argv[i], "--stream") == 0 || strcmp(argv[i], "-s") == 0) force_stream = 1;
        else if (strcmp(argv[i], "--max-ram") == 0 && i + 1 < argc) {
            max_ram_bytes = (size_t)atoi(argv[++i]) * 1024u * 1024u;
        } else if (strcmp(argv[i], "--max-ram-kb") == 0 && i + 1 < argc) {
            max_ram_bytes = (size_t)atoi(argv[++i]) * 1024u;
        } else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            requested_backend = qmx_app_parse_backend(argv[++i]);
        }
    }

    if (!model_path && argv[0] && qmx_app_probe_self_bundle(argv[0], &bundle_offset, &bundle_size, &rt_cfg)) {
        model_path = argv[0];
        if (memcmp(rt_cfg.magic, "QMXCFG", 6) == 0) {
            if (rt_cfg.steps > 0) steps = (int)rt_cfg.steps;
            if (rt_cfg.ctx_limit > 0) ctx_override = rt_cfg.ctx_limit;
            if (rt_cfg.backend < (uint8_t)QMX_BACKEND_COUNT || rt_cfg.backend == (uint8_t)QMX_BACKEND_AUTO) {
                requested_backend = (QmxBackend)rt_cfg.backend;
            }
            kv_type = (rt_cfg.use_kv8 <= 2u) ? (QmxKvType)rt_cfg.use_kv8 : QMX_KV_Q8;
            force_stream = rt_cfg.force_stream ? 1 : 0;
            if (rt_cfg.max_ram_kb > 0) max_ram_bytes = (size_t)rt_cfg.max_ram_kb * 1024u;
        }
    }

    if (!model_path) {
        fprintf(stderr, "Usage: %s -m model.qmx [-z tok.bin] [-p prompt | -f file] [-c ctx] [-n steps] [-j threads] [-t temp] [-k top_k] [--top-p P] [--min-p P] [--stream] [--kv8 | --kv16] [-b auto|scalar|...]\n", argv[0]);
        return 1;
    }

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

    QmxLoadParams lp = qmx_load_params_default();
    lp.override_max_seq = ctx_override;
    lp.backend          = requested_backend;
    lp.kv_type          = kv_type;
    lp.force_stream     = force_stream;
    lp.max_ram_bytes    = max_ram_bytes;

    QmxModel model;
    if (qmx_load_ex(&model, model_path, bundle_offset, bundle_size, &lp) != 0) {
        fprintf(stderr, "Error: failed to load QMX cartridge %s\n", model_path);
        free(prompt_file_buf);
        return 1;
    }
    qmx_set_num_threads(&model, num_threads);

    QmxModelInfo info;
    qmx_get_model_info(&model, &info);

    QmxTokenizer tokenizer;
    int has_tok = 0;
    qmx_app_reload_tokenizer(&info, model_path, bundle_offset, tok_path, &tokenizer, &has_tok);

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
    return (rc == 0) ? 0 : 1;
}
