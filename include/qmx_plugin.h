#ifndef QMX_PLUGIN_H
#define QMX_PLUGIN_H

#include "qmx.h"
#include "qmx_kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

#define QMX_PLUGIN_ABI_VERSION  0x00010000u
#define QMX_MAX_PLUGINS         32
#define QMX_PLUGIN_ENTRY_SYMBOL "qmx_plugin_entry"

#if defined(_WIN32)
    #define QMX_PLUGIN_EXPORT __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
    #define QMX_PLUGIN_EXPORT __attribute__((visibility("default")))
#else
    #define QMX_PLUGIN_EXPORT
#endif

typedef enum {
    QMX_PLUGIN_READER  = 1,
    QMX_PLUGIN_BACKEND = 2,
    QMX_PLUGIN_SAMPLER = 3,
    QMX_PLUGIN_HOOK    = 4
} QmxPluginKind;

typedef enum {
    QMX_HOOK_STEP_BEGIN    = 1,
    QMX_HOOK_STEP_END      = 2,
    QMX_HOOK_LAYER_BEGIN   = 3,
    QMX_HOOK_LAYER_END     = 4,
    QMX_HOOK_TOKEN_SAMPLED = 5
} QmxPluginEventKind;

typedef struct {
    const char *scheme;
    int (*open)(const char *uri, QmxReader *out_reader, void *plugin_ctx);
} QmxPluginReaderApi;

typedef struct {
    QmxBackend slot;
    const QmxBackendOps *ops;
    int (*exec_layer)(void *model_ptr, uint32_t layer_idx, int batch_size, int base_pos, void *plugin_ctx);
} QmxPluginBackendApi;

typedef struct {
    void (*filter_logits)(float *logits, uint32_t vocab_size, const int *recent, int recent_cnt, void *plugin_ctx);
} QmxPluginSamplerApi;

typedef struct {
    void (*on_event)(QmxPluginEventKind ev, uint32_t layer_idx, int batch_size, int pos, int token_id, void *plugin_ctx);
} QmxPluginHookApi;

typedef struct {
    uint32_t      abi_version;
    uint32_t      struct_size;
    QmxPluginKind kind;
    const char   *name;
    const char   *version;
    void         *plugin_ctx;
    void        (*destroy)(void *plugin_ctx);
    union {
        QmxPluginReaderApi  reader;
        QmxPluginBackendApi backend;
        QmxPluginSamplerApi sampler;
        QmxPluginHookApi    hook;
    } api;
} QmxPluginDesc;

typedef const QmxPluginDesc *(*QmxPluginEntryFn)(uint32_t host_abi_version);

int                  qmx_plugin_register(const QmxPluginDesc *desc);
int                  qmx_plugin_unregister(const char *name);
void                 qmx_plugin_clear_all(void);
int                  qmx_plugin_count(void);
const QmxPluginDesc *qmx_plugin_get(int index);
const QmxPluginDesc *qmx_plugin_find(const char *name);

int  qmx_plugin_open_reader(const char *uri, QmxReader *out_reader);
int  qmx_plugin_exec_layer(void *model_ptr, QmxBackend backend, uint32_t layer_idx, int batch_size, int base_pos);
void qmx_plugin_run_samplers(float *logits, uint32_t vocab_size, const int *recent, int recent_cnt);
void qmx_plugin_emit_hook(QmxPluginEventKind ev, uint32_t layer_idx, int batch_size, int pos, int token_id);

int  qmx_plugin_load_shared(const char *path);
int  qmx_plugin_load_dir(const char *dir_path);
void qmx_plugin_unload_all_shared(void);

#ifdef __cplusplus
}
#endif

#endif
