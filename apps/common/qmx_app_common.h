#ifndef QMX_APP_COMMON_H
#define QMX_APP_COMMON_H

#include "../../include/qmx.h"
#include "../../include/qmx_runtime.h"
#include "../../include/qmx_kernels.h"
#include "../../include/qmx_tokenizer.h"
#include "../../include/qmx_gen.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma pack(push, 1)
typedef struct {
    char     magic[8];
    uint32_t steps;
    uint32_t ctx_limit;
    float    temp;
    uint8_t  backend;
    uint8_t  use_kv8;
    uint8_t  force_stream;
    uint8_t  lang;
    uint32_t max_ram_kb;
    char     model_file[36];
} QmxRuntimeCfg;

typedef struct {
    uint32_t bundle_offset;
    uint32_t bundle_size;
    uint32_t cfg_offset;
    char     magic[4];
} QmxBundleTrailer;
#pragma pack(pop)

typedef char assert_qmx_rt_cfg_64b[(sizeof(QmxRuntimeCfg) == 64) ? 1 : -1];
typedef char assert_qmx_trailer_16b[(sizeof(QmxBundleTrailer) == 16) ? 1 : -1];

typedef enum {
    QMX_CHAT_NONE   = 0,
    QMX_CHAT_LLAMA3 = 1,
    QMX_CHAT_CHATML = 2,
    QMX_CHAT_GEMMA2 = 3,
    QMX_CHAT_PHI3   = 4
} QmxChatTemplate;

typedef struct {
    char carry[4];
    int  carry_len;
} QmxUtf8Printer;

int             qmx_app_utf8_expected_len(unsigned char lead);
void            qmx_app_utf8_printer_reset(QmxUtf8Printer *p);
void            qmx_app_safe_print(QmxUtf8Printer *p, const char *piece);
float           qmx_app_time_ms(void);
const char     *qmx_app_quant_label(uint16_t q);
QmxBackend      qmx_app_parse_backend(const char *s);
QmxBackend      qmx_app_cycle_backend(QmxBackend cur);
QmxChatTemplate qmx_app_detect_chat_template(const QmxTokenizer *tok, int has_tok);

int qmx_app_load_cfg_file(const char *path, QmxRuntimeCfg *out_cfg);
int qmx_app_save_cfg_file(const char *path, const QmxModelInfo *info, const char *model_path,
                          int steps, float temp, int lang_ru, size_t max_ram_bytes);
int qmx_app_probe_self_bundle(const char *exe_path, uint64_t *out_mod_off, uint64_t *out_mod_sz,
                              QmxRuntimeCfg *out_cfg);

void qmx_app_reload_tokenizer(const QmxModelInfo *info, const char *active_path, uint64_t base_offset,
                              const char *fallback_tok_path, QmxTokenizer *tok, int *has_tok);

int qmx_app_run_cli(QmxModel *model, const QmxTokenizer *tok, int has_tok,
                    const char *prompt, const QmxGenParams *params, int debug_tokens);

#ifdef __cplusplus
}
#endif

#endif
