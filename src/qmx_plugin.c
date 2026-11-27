#if !defined(_WIN32) && !defined(__DJGPP__)
    #ifndef _DEFAULT_SOURCE
        #define _DEFAULT_SOURCE
    #endif
    #ifndef _POSIX_C_SOURCE
        #define _POSIX_C_SOURCE 200809L
    #endif
#endif

#include "../include/qmx_plugin.h"
#include "../include/qmx_runtime.h"
#include <string.h>

#if defined(QMX_ENABLE_DYNAMIC_PLUGINS) && !defined(QMX_NO_PLUGINS) && \
    !defined(QMX_NO_MALLOC) && !defined(QMX_NO_STDIO) && \
    !defined(__DJGPP__) && !defined(QMX_WIN98_COMPAT)
    #define QMX_HAS_DYNAMIC_PLUGINS 1
    #include <stdio.h>
    #if defined(_WIN32)
        #include <windows.h>
    #else
        #include <dlfcn.h>
        #include <dirent.h>
    #endif
#else
    #define QMX_HAS_DYNAMIC_PLUGINS 0
#endif

#if !defined(QMX_NO_PLUGINS)

static const QmxPluginDesc *g_qmx_plg_list[QMX_MAX_PLUGINS];
static int g_qmx_plg_cnt = 0;
static int g_qmx_plg_hook_cnt = 0;
static int g_qmx_plg_sampler_cnt = 0;
static int g_qmx_plg_layer_cnt = 0;

int qmx_plugin_register(const QmxPluginDesc *desc) {
    if (!desc) return -1;
    if (desc->abi_version != QMX_PLUGIN_ABI_VERSION) return -1;
    if (desc->struct_size != (uint32_t)sizeof(QmxPluginDesc)) return -1;
    if (!desc->name || desc->name[0] == '\0') return -1;
    if (g_qmx_plg_cnt >= QMX_MAX_PLUGINS) return -1;

    for (int i = 0; i < g_qmx_plg_cnt; i++) {
        if (strcmp(g_qmx_plg_list[i]->name, desc->name) == 0) return -1;
    }

    if (desc->kind == QMX_PLUGIN_READER) {
        if (!desc->api.reader.scheme || desc->api.reader.scheme[0] == '\0' || !desc->api.reader.open) {
            return -1;
        }
    } else if (desc->kind == QMX_PLUGIN_BACKEND) {
        if (!desc->api.backend.ops && !desc->api.backend.exec_layer) return -1;
        if (desc->api.backend.ops) {
#ifdef QMX_NO_CUSTOM_BACKENDS
            return -1;
#else
            if ((unsigned)desc->api.backend.slot >= (unsigned)QMX_BACKEND_COUNT) {
                return -1;
            }
            qmx_register_backend(desc->api.backend.slot, desc->api.backend.ops);
#endif
        }
    } else if (desc->kind == QMX_PLUGIN_SAMPLER) {
        if (!desc->api.sampler.filter_logits) return -1;
    } else if (desc->kind == QMX_PLUGIN_HOOK) {
        if (!desc->api.hook.on_event) return -1;
    } else {
        return -1;
    }

    g_qmx_plg_list[g_qmx_plg_cnt++] = desc;
    if (desc->kind == QMX_PLUGIN_HOOK) g_qmx_plg_hook_cnt++;
    else if (desc->kind == QMX_PLUGIN_SAMPLER) g_qmx_plg_sampler_cnt++;
    else if (desc->kind == QMX_PLUGIN_BACKEND && desc->api.backend.exec_layer) g_qmx_plg_layer_cnt++;
    return 0;
}

int qmx_plugin_unregister(const char *name) {
    if (!name || name[0] == '\0') return -1;
    for (int i = 0; i < g_qmx_plg_cnt; i++) {
        const QmxPluginDesc *d = g_qmx_plg_list[i];
        if (strcmp(d->name, name) == 0) {
            if (d->kind == QMX_PLUGIN_HOOK && g_qmx_plg_hook_cnt > 0) g_qmx_plg_hook_cnt--;
            else if (d->kind == QMX_PLUGIN_SAMPLER && g_qmx_plg_sampler_cnt > 0) g_qmx_plg_sampler_cnt--;
            else if (d->kind == QMX_PLUGIN_BACKEND && d->api.backend.exec_layer && g_qmx_plg_layer_cnt > 0) {
                g_qmx_plg_layer_cnt--;
            }
            if (d->destroy) d->destroy(d->plugin_ctx);
            for (int j = i + 1; j < g_qmx_plg_cnt; j++) {
                g_qmx_plg_list[j - 1] = g_qmx_plg_list[j];
            }
            g_qmx_plg_cnt--;
            g_qmx_plg_list[g_qmx_plg_cnt] = NULL;
            return 0;
        }
    }
    return -1;
}

void qmx_plugin_clear_all(void) {
    while (g_qmx_plg_cnt > 0) {
        const QmxPluginDesc *d = g_qmx_plg_list[g_qmx_plg_cnt - 1];
        g_qmx_plg_cnt--;
        g_qmx_plg_list[g_qmx_plg_cnt] = NULL;
        if (d && d->destroy) d->destroy(d->plugin_ctx);
    }
    g_qmx_plg_hook_cnt = 0;
    g_qmx_plg_sampler_cnt = 0;
    g_qmx_plg_layer_cnt = 0;
}

int qmx_plugin_count(void) {
    return g_qmx_plg_cnt;
}

const QmxPluginDesc *qmx_plugin_get(int index) {
    if (index < 0 || index >= g_qmx_plg_cnt) return NULL;
    return g_qmx_plg_list[index];
}

const QmxPluginDesc *qmx_plugin_find(const char *name) {
    if (!name || name[0] == '\0') return NULL;
    for (int i = 0; i < g_qmx_plg_cnt; i++) {
        if (strcmp(g_qmx_plg_list[i]->name, name) == 0) return g_qmx_plg_list[i];
    }
    return NULL;
}

int qmx_plugin_open_reader(const char *uri, QmxReader *out_reader) {
    if (!uri || !out_reader) return -1;
    for (int i = 0; i < g_qmx_plg_cnt; i++) {
        const QmxPluginDesc *d = g_qmx_plg_list[i];
        if (d->kind != QMX_PLUGIN_READER) continue;
        size_t slen = strlen(d->api.reader.scheme);
        if (slen > 0u && strncmp(uri, d->api.reader.scheme, slen) == 0) {
            memset(out_reader, 0, sizeof(*out_reader));
            return d->api.reader.open(uri, out_reader, d->plugin_ctx);
        }
    }
    return -1;
}

int qmx_plugin_exec_layer(void *model_ptr, QmxBackend backend, uint32_t layer_idx, int batch_size, int base_pos) {
    if (g_qmx_plg_layer_cnt <= 0) return -1;
    for (int i = 0; i < g_qmx_plg_cnt; i++) {
        const QmxPluginDesc *d = g_qmx_plg_list[i];
        if (d->kind == QMX_PLUGIN_BACKEND && d->api.backend.slot == backend && d->api.backend.exec_layer) {
            if (d->api.backend.exec_layer(model_ptr, layer_idx, batch_size, base_pos, d->plugin_ctx) == 0) {
                return 0;
            }
        }
    }
    return -1;
}

void qmx_plugin_run_samplers(float *logits, uint32_t vocab_size, const int *recent, int recent_cnt) {
    if (g_qmx_plg_sampler_cnt <= 0 || !logits || vocab_size == 0u) return;
    for (int i = 0; i < g_qmx_plg_cnt; i++) {
        const QmxPluginDesc *d = g_qmx_plg_list[i];
        if (d->kind == QMX_PLUGIN_SAMPLER && d->api.sampler.filter_logits) {
            d->api.sampler.filter_logits(logits, vocab_size, recent, recent_cnt, d->plugin_ctx);
        }
    }
}

void qmx_plugin_emit_hook(QmxPluginEventKind ev, uint32_t layer_idx, int batch_size, int pos, int token_id) {
    if (g_qmx_plg_hook_cnt <= 0) return;
    for (int i = 0; i < g_qmx_plg_cnt; i++) {
        const QmxPluginDesc *d = g_qmx_plg_list[i];
        if (d->kind == QMX_PLUGIN_HOOK && d->api.hook.on_event) {
            d->api.hook.on_event(ev, layer_idx, batch_size, pos, token_id, d->plugin_ctx);
        }
    }
}

#else

int qmx_plugin_register(const QmxPluginDesc *desc) { (void)desc; return -1; }
int qmx_plugin_unregister(const char *name) { (void)name; return -1; }
void qmx_plugin_clear_all(void) {}
int qmx_plugin_count(void) { return 0; }
const QmxPluginDesc *qmx_plugin_get(int index) { (void)index; return NULL; }
const QmxPluginDesc *qmx_plugin_find(const char *name) { (void)name; return NULL; }
int qmx_plugin_open_reader(const char *uri, QmxReader *out_reader) { (void)uri; (void)out_reader; return -1; }
int qmx_plugin_exec_layer(void *model_ptr, QmxBackend backend, uint32_t layer_idx, int batch_size, int base_pos) {
    (void)model_ptr; (void)backend; (void)layer_idx; (void)batch_size; (void)base_pos;
    return -1;
}
void qmx_plugin_run_samplers(float *logits, uint32_t vocab_size, const int *recent, int recent_cnt) {
    (void)logits; (void)vocab_size; (void)recent; (void)recent_cnt;
}
void qmx_plugin_emit_hook(QmxPluginEventKind ev, uint32_t layer_idx, int batch_size, int pos, int token_id) {
    (void)ev; (void)layer_idx; (void)batch_size; (void)pos; (void)token_id;
}

#endif

#if QMX_HAS_DYNAMIC_PLUGINS

typedef struct {
#if defined(_WIN32)
    HMODULE handle;
#else
    void   *handle;
#endif
    char    name[64];
} QmxSharedPluginSlot;

static QmxSharedPluginSlot g_qmx_plg_shared[QMX_MAX_PLUGINS];
static int g_qmx_plg_shared_cnt = 0;

int qmx_plugin_load_shared(const char *path) {
    if (!path || path[0] == '\0' || g_qmx_plg_shared_cnt >= QMX_MAX_PLUGINS) return -1;
#if defined(_WIN32)
    HMODULE h = LoadLibraryA(path);
    if (!h) return -1;
    FARPROC sym = GetProcAddress(h, QMX_PLUGIN_ENTRY_SYMBOL);
    if (!sym) {
        FreeLibrary(h);
        return -1;
    }
    QmxPluginEntryFn entry = NULL;
    memcpy(&entry, &sym, sizeof(entry));
    const QmxPluginDesc *desc = entry(QMX_PLUGIN_ABI_VERSION);
    if (!desc || qmx_plugin_register(desc) != 0) {
        FreeLibrary(h);
        return -1;
    }
    g_qmx_plg_shared[g_qmx_plg_shared_cnt].handle = h;
    strncpy(g_qmx_plg_shared[g_qmx_plg_shared_cnt].name, desc->name, 63u);
    g_qmx_plg_shared[g_qmx_plg_shared_cnt].name[63] = '\0';
    g_qmx_plg_shared_cnt++;
    return 0;
#else
    void *h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!h) return -1;
    void *sym = dlsym(h, QMX_PLUGIN_ENTRY_SYMBOL);
    if (!sym) {
        dlclose(h);
        return -1;
    }
    QmxPluginEntryFn entry = NULL;
    memcpy(&entry, &sym, sizeof(entry));
    const QmxPluginDesc *desc = entry(QMX_PLUGIN_ABI_VERSION);
    if (!desc || qmx_plugin_register(desc) != 0) {
        dlclose(h);
        return -1;
    }
    g_qmx_plg_shared[g_qmx_plg_shared_cnt].handle = h;
    strncpy(g_qmx_plg_shared[g_qmx_plg_shared_cnt].name, desc->name, 63u);
    g_qmx_plg_shared[g_qmx_plg_shared_cnt].name[63] = '\0';
    g_qmx_plg_shared_cnt++;
    return 0;
#endif
}

int qmx_plugin_load_dir(const char *dir_path) {
    if (!dir_path || dir_path[0] == '\0') return -1;
    int loaded = 0;
#if defined(_WIN32)
    char pattern[512];
    snprintf(pattern, sizeof(pattern), "%s\\*.dll", dir_path);
    WIN32_FIND_DATAA ffd;
    HANDLE hf = FindFirstFileA(pattern, &ffd);
    if (hf == INVALID_HANDLE_VALUE) return -1;
    do {
        if (!(ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
            char full[512];
            snprintf(full, sizeof(full), "%s\\%s", dir_path, ffd.cFileName);
            if (qmx_plugin_load_shared(full) == 0) loaded++;
        }
    } while (FindNextFileA(hf, &ffd));
    FindClose(hf);
    return loaded;
#else
    DIR *dp = opendir(dir_path);
    if (!dp) return -1;
    #if defined(__APPLE__)
    const char *ext = ".dylib";
    const size_t ext_len = 6u;
    #else
    const char *ext = ".so";
    const size_t ext_len = 3u;
    #endif
    struct dirent *de;
    while ((de = readdir(dp)) != NULL) {
        size_t nlen = strlen(de->d_name);
        if (nlen > ext_len && strcmp(de->d_name + nlen - ext_len, ext) == 0) {
            char full[512];
            snprintf(full, sizeof(full), "%s/%s", dir_path, de->d_name);
            if (qmx_plugin_load_shared(full) == 0) loaded++;
        }
    }
    closedir(dp);
    return loaded;
#endif
}

void qmx_plugin_unload_all_shared(void) {
    while (g_qmx_plg_shared_cnt > 0) {
        g_qmx_plg_shared_cnt--;
        qmx_plugin_unregister(g_qmx_plg_shared[g_qmx_plg_shared_cnt].name);
#if defined(_WIN32)
        if (g_qmx_plg_shared[g_qmx_plg_shared_cnt].handle) {
            FreeLibrary(g_qmx_plg_shared[g_qmx_plg_shared_cnt].handle);
            g_qmx_plg_shared[g_qmx_plg_shared_cnt].handle = NULL;
        }
#else
        if (g_qmx_plg_shared[g_qmx_plg_shared_cnt].handle) {
            dlclose(g_qmx_plg_shared[g_qmx_plg_shared_cnt].handle);
            g_qmx_plg_shared[g_qmx_plg_shared_cnt].handle = NULL;
        }
#endif
        g_qmx_plg_shared[g_qmx_plg_shared_cnt].name[0] = '\0';
    }
}

#else

int qmx_plugin_load_shared(const char *path) { (void)path; return -1; }
int qmx_plugin_load_dir(const char *dir_path) { (void)dir_path; return -1; }
void qmx_plugin_unload_all_shared(void) {}

#endif
