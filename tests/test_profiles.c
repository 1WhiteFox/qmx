#include "../apps/common/qmx_profiles.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

static int g_failures = 0;

#define EXPECT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "[FAIL] %s (line %d)\n", (msg), __LINE__); \
        g_failures++; \
    } \
} while (0)

int main(void) {
    const char *ini_path = "test_profiles_tmp.ini";
    FILE *fp = fopen(ini_path, "wb");
    EXPECT(fp != NULL, "Create temporary INI file");
    if (!fp) return 1;

    const char *ini_content =
        "# top comment\r\n"
        "; another comment\r\n"
        "[default]\r\n"
        "  threads = 6 # inline comment\r\n"
        "  kv = fp16\r\n"
        "  temp = 0.3\r\n"
        "  ctx = 512\r\n"
        "  prefill_chunk = 16\r\n"
        "\r\n"
        "[model.qwen3]\r\n"
        "  ctx = 2048\r\n"
        "  temp = 0.65\r\n"
        "  top_p = 0.95\r\n"
        "\r\n"
        "[profile.low_ram]\r\n"
        "  model = \"models/qwen3_4b_q4_r32.qmx\"\r\n"
        "  stream = true\r\n"
        "  max_ram_mb = 256\r\n"
        "  kv = q8\r\n"
        "  threads = 4\r\n"
        "  no_autotune = yes\r\n"
        "  lang = ru\r\n"
        "  backend = avx2\r\n";

    fwrite(ini_content, 1, strlen(ini_content), fp);
    fclose(fp);

    QmxProfile p;
    EXPECT(qmx_profile_load_ini("nonexistent_qmx_12345.ini", NULL, NULL, &p) == -1,
           "Missing INI returns -1");

    EXPECT(qmx_profile_load_ini(ini_path, "no_such_profile", NULL, &p) == -2,
           "Unknown requested profile returns -2");

    EXPECT(qmx_profile_load_ini(ini_path, NULL, "C:\\models\\Qwen3_4B.qmx", &p) == 0,
           "Load default + model.qwen3 sections");
    EXPECT(p.threads == 6, "[default] threads=6");
    EXPECT(p.kv_type == QMX_KV_FP16, "[default] kv=fp16");
    EXPECT(p.ctx == 2048u, "[model.qwen3] overrides ctx=2048");
    EXPECT(fabsf(p.temp - 0.65f) < 1e-4f, "[model.qwen3] overrides temp=0.65");
    EXPECT(fabsf(p.top_p - 0.95f) < 1e-4f, "[model.qwen3] sets top_p=0.95");
    EXPECT(p.prefill_chunk == 16, "[default] prefill_chunk=16 preserved");

    EXPECT(qmx_profile_load_ini(ini_path, "LOW_RAM", "models/qwen3_4b_q4_r32.qmx", &p) == 0,
           "Load default + model.qwen3 + profile.low_ram (case-insensitive)");
    EXPECT(strcmp(p.model, "models/qwen3_4b_q4_r32.qmx") == 0, "Unquoted model path parsed");
    EXPECT(p.force_stream == 1, "[profile.low_ram] stream=true");
    EXPECT(p.max_ram_bytes == (size_t)(256ULL * 1024ULL * 1024ULL), "[profile.low_ram] max_ram_mb=256");
    EXPECT(p.kv_type == QMX_KV_Q8, "[profile.low_ram] overrides kv=q8");
    EXPECT(p.threads == 4, "[profile.low_ram] overrides threads=4");
    EXPECT(p.ctx == 2048u, "[model.qwen3] ctx=2048 inherited under profile");
    EXPECT(p.no_autotune == 1, "[profile.low_ram] no_autotune=yes");
    EXPECT(p.lang_ru == 1, "[profile.low_ram] lang=ru");
    EXPECT(p.backend == QMX_BACKEND_AVX2, "[profile.low_ram] backend=avx2");

    QmxProfile cli_ovr;
    qmx_profile_init(&cli_ovr);
    cli_ovr.threads = 8;
    cli_ovr.mask |= QMX_PROF_F_THREADS;
    cli_ovr.temp = 0.0f;
    cli_ovr.mask |= QMX_PROF_F_TEMP;
    qmx_profile_merge(&p, &cli_ovr);
    EXPECT(p.threads == 8, "CLI overrides profile threads");
    EXPECT(fabsf(p.temp) < 1e-6f, "CLI overrides model temp");
    EXPECT(p.force_stream == 1 && p.kv_type == QMX_KV_Q8, "Unset CLI fields keep profile values");

    QmxRuntimeCfg cfg;
    memset(&cfg, 0, sizeof(cfg));
    memcpy(cfg.magic, "QMXCFG\0\0", 8);
    cfg.steps = 96u;
    cfg.ctx_limit = 384u;
    cfg.temp = 0.75f;
    cfg.backend = (uint8_t)QMX_BACKEND_MMX;
    cfg.use_kv8 = (uint8_t)QMX_KV_Q8;
    cfg.force_stream = 1u;
    cfg.lang = 1u;
    cfg.max_ram_kb = 128u;
    strncpy(cfg.model_file, "stories260k.qmx", sizeof(cfg.model_file) - 1u);

    QmxProfile from_cfg;
    qmx_profile_from_cfg(&from_cfg, &cfg);
    EXPECT(from_cfg.steps == 96 && from_cfg.ctx == 384u, "QMX.CFG steps & ctx imported");
    EXPECT(from_cfg.backend == QMX_BACKEND_MMX && from_cfg.kv_type == QMX_KV_Q8, "QMX.CFG backend & kv imported");
    EXPECT(from_cfg.max_ram_bytes == 128u * 1024u, "QMX.CFG max_ram_kb imported");
    EXPECT(strcmp(from_cfg.model, "stories260k.qmx") == 0, "QMX.CFG model_file imported");

    QmxProfile to_save;
    qmx_profile_init(&to_save);
    strncpy(to_save.model, "models/phi3_mini.qmx", sizeof(to_save.model) - 1u);
    to_save.backend = QMX_BACKEND_AVX512;
    to_save.kv_type = QMX_KV_FP16;
    to_save.threads = 8;
    to_save.ctx = 4096u;
    to_save.steps = 256;
    to_save.temp = 0.45f;
    to_save.force_stream = 1;
    to_save.max_ram_bytes = 512ULL * 1024ULL * 1024ULL;
    to_save.lang_ru = 1;
    to_save.mask = QMX_PROF_F_MODEL | QMX_PROF_F_BACKEND | QMX_PROF_F_KV | QMX_PROF_F_THREADS |
                   QMX_PROF_F_CTX | QMX_PROF_F_STEPS | QMX_PROF_F_TEMP | QMX_PROF_F_STREAM |
                   QMX_PROF_F_MAX_RAM | QMX_PROF_F_LANG_RU;
    EXPECT(qmx_profile_save_ini(ini_path, "profile.saved_tui", &to_save) == 0, "Save new profile section to INI");
    to_save.threads = 6;
    EXPECT(qmx_profile_save_ini(ini_path, "profile.saved_tui", &to_save) == 0, "Update existing profile section in INI");

    QmxProfile reloaded;
    EXPECT(qmx_profile_load_ini(ini_path, "saved_tui", NULL, &reloaded) == 0, "Reload saved INI profile");
    EXPECT(strcmp(reloaded.model, "models/phi3_mini.qmx") == 0, "Saved INI model round-trip");
    EXPECT(reloaded.backend == QMX_BACKEND_AVX512 && reloaded.kv_type == QMX_KV_FP16, "Saved INI backend & kv round-trip");
    EXPECT(reloaded.threads == 6 && reloaded.ctx == 4096u && reloaded.steps == 256, "Saved INI updated threads/ctx/steps round-trip");
    EXPECT(reloaded.max_ram_bytes == (size_t)(512ULL * 1024ULL * 1024ULL) && reloaded.lang_ru == 1, "Saved INI RAM & lang round-trip");
    EXPECT(qmx_profile_load_ini(ini_path, "low_ram", NULL, &reloaded) == 0 && reloaded.threads == 4,
           "Saving new INI section preserved existing sections");

    remove(ini_path);

    if (g_failures == 0) {
        printf("[PASS] INI profiles parser, precedence ([default] -> [model.*] -> [profile.*] -> CLI) & QMX.CFG tests passed!\n");
        return 0;
    }
    fprintf(stderr, "[FAIL] test_profiles had %d failure(s)\n", g_failures);
    return 1;
}
