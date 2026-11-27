#include "../include/qmx.h"
#include "../include/qmx_runtime.h"
#include "../include/qmx_tokenizer.h"
#include "../include/qmx_gen.h"
#include "../include/qmx_plugin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

const QmxPluginDesc *qmx_plugin_profiler_desc(void);
uint64_t             qmx_plugin_profiler_steps(void);
uint64_t             qmx_plugin_profiler_tokens(void);
uint64_t             qmx_plugin_profiler_layer_calls(uint32_t layer_idx);

void                 qmx_plugin_token_ban_reset(void);
int                  qmx_plugin_token_ban_add(int token_id);
uint64_t             qmx_plugin_token_ban_calls(void);
const QmxPluginDesc *qmx_plugin_token_ban_desc(void);

static int g_failures = 0;

#define EXPECT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "[FAIL] %s (line %d)\n", (msg), __LINE__); \
        g_failures++; \
    } \
} while (0)

typedef struct {
    const uint8_t *buf;
    size_t         len;
    int            closed;
} MemReaderCtx;

static MemReaderCtx g_mem_ctx;

static int mem_read(void *ctx, uint64_t offset, void *dst, size_t size) {
    MemReaderCtx *m = (MemReaderCtx *)ctx;
    if (!m || !m->buf || offset > (uint64_t)m->len || size > m->len - (size_t)offset) return -1;
    memcpy(dst, m->buf + (size_t)offset, size);
    return 0;
}

static void mem_close(void *ctx) {
    MemReaderCtx *m = (MemReaderCtx *)ctx;
    if (m) m->closed = 1;
}

static int mem_plugin_open(const char *uri, QmxReader *out_reader, void *plugin_ctx) {
    (void)uri;
    MemReaderCtx *m = (MemReaderCtx *)plugin_ctx;
    if (!m || !m->buf || !out_reader) return -1;
    m->closed = 0;
    out_reader->ctx   = m;
    out_reader->read  = mem_read;
    out_reader->size  = (uint64_t)m->len;
    out_reader->close = mem_close;
    return 0;
}

static void dummy_hook_fn(QmxPluginEventKind ev, uint32_t l, int b, int p, int t, void *c) {
    (void)ev; (void)l; (void)b; (void)p; (void)t; (void)c;
}

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "test_q4.qmx";

    qmx_plugin_clear_all();

    QmxPluginDesc bad_abi = {
        QMX_PLUGIN_ABI_VERSION + 1u,
        (uint32_t)sizeof(QmxPluginDesc),
        QMX_PLUGIN_HOOK,
        "bad_abi",
        "0.1",
        NULL,
        NULL,
        { .hook = { dummy_hook_fn } }
    };
    EXPECT(qmx_plugin_register(&bad_abi) != 0, "Reject mismatched abi_version");

    QmxPluginDesc bad_sz = bad_abi;
    bad_sz.abi_version = QMX_PLUGIN_ABI_VERSION;
    bad_sz.struct_size = (uint32_t)sizeof(QmxPluginDesc) - 4u;
    bad_sz.name = "bad_sz";
    EXPECT(qmx_plugin_register(&bad_sz) != 0, "Reject mismatched struct_size");

    QmxPluginDesc bad_name = bad_abi;
    bad_name.abi_version = QMX_PLUGIN_ABI_VERSION;
    bad_name.name = "";
    EXPECT(qmx_plugin_register(&bad_name) != 0, "Reject empty plugin name");

    QmxModel m;
    QmxLoadParams lp = qmx_load_params_default();
    lp.backend = QMX_BACKEND_SCALAR;
    lp.override_max_seq = 64u;
    EXPECT(qmx_load_ex(&m, path, 0u, 0u, &lp) == 0, "Load test_q4.qmx");
    if (g_failures) return 1;

    QmxModelInfo info;
    qmx_get_model_info(&m, &info);
    QmxTokenizer tok;
    int has_tok = (qmx_tokenizer_load(&tok, path, info.tokenizer_offset, info.tokenizer_size) == 0);
    EXPECT(has_tok, "Load embedded tokenizer");

    int prompt_ids[3] = { 1, 72, 101 };
    QmxGenParams gp = qmx_gen_params_default();
    gp.temp = 0.0f;
    gp.rep_penalty = 1.0f;
    gp.max_tokens = 4;

    QmxGenState gen;
    qmx_gen_init(&gen, &m, &tok, &gp);
    qmx_gen_begin(&gen, prompt_ids, 3, &gp);
    QmxStepStatus st;
    QmxStepResult res;
    int first_base_tok = -1;
    while ((st = qmx_step(&gen, &res)) > 0) {
        if (st == QMX_STEP_TOKEN && first_base_tok < 0) first_base_tok = res.token;
    }
    EXPECT(first_base_tok >= 0, "Baseline first token sampled");

    const QmxPluginDesc *prof = qmx_plugin_profiler_desc();
    EXPECT(qmx_plugin_register(prof) == 0, "Register static profiler plugin");
    EXPECT(qmx_plugin_register(prof) != 0, "Reject duplicate plugin name");

    qmx_plugin_token_ban_reset();
    EXPECT(qmx_plugin_token_ban_add(first_base_tok) == 0, "Add banned token");
    EXPECT(qmx_plugin_register(qmx_plugin_token_ban_desc()) == 0, "Register static token_ban plugin");
    EXPECT(qmx_plugin_count() == 2, "Two static plugins active");

    qmx_gen_reset(&gen);
    qmx_gen_begin(&gen, prompt_ids, 3, &gp);
    int first_banned_tok = -1;
    while ((st = qmx_step(&gen, &res)) > 0) {
        if (st == QMX_STEP_TOKEN && first_banned_tok < 0) first_banned_tok = res.token;
    }
    EXPECT(first_banned_tok >= 0 && first_banned_tok != first_base_tok, "Sampler plugin suppressed baseline token");
    EXPECT(qmx_plugin_token_ban_calls() >= 4u, "Sampler filter invoked on every decode step");
    EXPECT(qmx_plugin_profiler_steps() >= 4u, "Profiler observed steps");
    EXPECT(qmx_plugin_profiler_tokens() >= 4u, "Profiler observed sampled tokens");
    EXPECT(qmx_plugin_profiler_layer_calls(0u) == qmx_plugin_profiler_steps(), "Layer 0 call count matches steps");
    EXPECT(qmx_plugin_profiler_layer_calls(1u) == qmx_plugin_profiler_steps(), "Layer 1 call count matches steps");

    EXPECT(qmx_plugin_unregister("token_ban") == 0, "Unregister token_ban");
    EXPECT(qmx_plugin_unregister("layer_profiler") == 0, "Unregister layer_profiler");
    EXPECT(qmx_plugin_count() == 0, "Zero plugins after unregister");

    FILE *fp = fopen(path, "rb");
    EXPECT(fp != NULL, "Open file for memtest:// reader plugin");
    if (fp) {
        int64_t fsz = qmx_file_size64(fp);
        uint8_t *buf = (uint8_t *)malloc((size_t)fsz);
        EXPECT(buf != NULL && fread(buf, 1, (size_t)fsz, fp) == (size_t)fsz, "Read cartridge buffer");
        fclose(fp);
        if (buf) {
            g_mem_ctx.buf = buf;
            g_mem_ctx.len = (size_t)fsz;
            g_mem_ctx.closed = 0;
            QmxPluginDesc reader_desc = {
                QMX_PLUGIN_ABI_VERSION,
                (uint32_t)sizeof(QmxPluginDesc),
                QMX_PLUGIN_READER,
                "memtest_reader",
                "1.3b",
                &g_mem_ctx,
                NULL,
                { .reader = { "memtest://", mem_plugin_open } }
            };
            EXPECT(qmx_plugin_register(&reader_desc) == 0, "Register memtest:// reader plugin");
            QmxReader rd;
            EXPECT(qmx_plugin_open_reader("memtest://test_q4.qmx", &rd) == 0, "Open memtest:// URI");
            QmxModel m_uri;
            lp.force_stream = 1;
            EXPECT(qmx_load_from_reader(&m_uri, rd, &lp) == 0, "Load model from plugin reader");
            float *l_ref = qmx_forward(&m, 1, 0);
            float *l_uri = qmx_forward(&m_uri, 1, 0);
            float max_diff = 0.0f;
            for (uint32_t i = 0; i < info.vocab_size; i++) {
                float d = fabsf(l_ref[i] - l_uri[i]);
                if (d > max_diff) max_diff = d;
            }
            EXPECT(max_diff < 1e-5f, "Plugin reader logits match file load");
            qmx_free(&m_uri);
            EXPECT(g_mem_ctx.closed == 1, "Plugin reader close callback executed on qmx_free");
            EXPECT(qmx_plugin_unregister("memtest_reader") == 0, "Unregister memtest_reader");
            free(buf);
        }
    }

#if defined(QMX_ENABLE_DYNAMIC_PLUGINS)
    {
        const char *pdir = (argc > 2) ? argv[2] : "build/plugins";
        int n_dyn = qmx_plugin_load_dir(pdir);
        EXPECT(n_dyn == 2, "Loaded 2 dynamic plugins from build/plugins");
        EXPECT(qmx_plugin_find("layer_profiler") != NULL, "Dynamic layer_profiler registered");
        EXPECT(qmx_plugin_find("token_ban") != NULL, "Dynamic token_ban registered");
        qmx_gen_reset(&gen);
        qmx_gen_begin(&gen, prompt_ids, 3, &gp);
        while (qmx_step(&gen, &res) > 0) {}
        qmx_plugin_unload_all_shared();
        EXPECT(qmx_plugin_count() == 0, "Dynamic plugins cleanly unloaded");
    }
#endif

    if (has_tok) qmx_tokenizer_free(&tok);
    qmx_free(&m);

    if (g_failures == 0) {
        printf("[PASS] Plugin ABI (static + dynamic .so, reader/sampler/hook, ABI guards) passed!\n");
        return 0;
    }
    fprintf(stderr, "[FAIL] test_plugin had %d failure(s)\n", g_failures);
    return 1;
}
