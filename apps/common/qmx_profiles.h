#ifndef QMX_PROFILES_H
#define QMX_PROFILES_H

#include "qmx_app_common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define QMX_PROF_F_MODEL         (1u << 0)
#define QMX_PROF_F_TOKENIZER     (1u << 1)
#define QMX_PROF_F_BACKEND       (1u << 2)
#define QMX_PROF_F_KV            (1u << 3)
#define QMX_PROF_F_THREADS       (1u << 4)
#define QMX_PROF_F_CTX           (1u << 5)
#define QMX_PROF_F_STEPS         (1u << 6)
#define QMX_PROF_F_TEMP          (1u << 7)
#define QMX_PROF_F_TOP_K         (1u << 8)
#define QMX_PROF_F_TOP_P         (1u << 9)
#define QMX_PROF_F_MIN_P         (1u << 10)
#define QMX_PROF_F_REP_PENALTY   (1u << 11)
#define QMX_PROF_F_SEED          (1u << 12)
#define QMX_PROF_F_PREFILL_CHUNK (1u << 13)
#define QMX_PROF_F_STREAM        (1u << 14)
#define QMX_PROF_F_MAX_RAM       (1u << 15)
#define QMX_PROF_F_NO_AUTOTUNE   (1u << 16)
#define QMX_PROF_F_LANG_RU       (1u << 17)
#define QMX_PROF_F_PLUGIN        (1u << 18)
#define QMX_PROF_F_PLUGIN_DIR    (1u << 19)

typedef struct {
    uint32_t   mask;
    char       model[256];
    char       tokenizer[256];
    char       plugin[256];
    char       plugin_dir[256];
    QmxBackend backend;
    QmxKvType  kv_type;
    int        threads;
    uint32_t   ctx;
    int        steps;
    float      temp;
    int        top_k;
    float      top_p;
    float      min_p;
    float      rep_penalty;
    uint32_t   seed;
    int        prefill_chunk;
    int        force_stream;
    size_t     max_ram_bytes;
    int        no_autotune;
    int        lang_ru;
} QmxProfile;

void qmx_profile_init(QmxProfile *p);
void qmx_profile_merge(QmxProfile *dst, const QmxProfile *src);
void qmx_profile_from_cfg(QmxProfile *dst, const QmxRuntimeCfg *cfg);
int  qmx_profile_load_ini(const char *ini_path, const char *profile_name,
                          const char *model_hint, QmxProfile *out_prof);
int  qmx_profile_save_ini(const char *ini_path, const char *section_name,
                          const QmxProfile *prof);

#ifdef __cplusplus
}
#endif

#endif
