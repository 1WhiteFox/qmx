#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 500
#endif
#if defined(_WIN32) && !defined(__USE_MINGW_ANSI_STDIO)
#define __USE_MINGW_ANSI_STDIO 0
#endif

#include "../common/qmx_app_common.h"
#include "../common/qmx_profiles.h"
#include "../common/qmx_session.h"
#include "tui.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__DJGPP__)
#include <dos.h>
#else
#include <unistd.h>
#endif

#if defined(__DJGPP__)
unsigned _stklen = 65536U;
#endif

#define CHAT_VIEW_LINES  14
#define CHAT_COLS        76
#define MODAL_CFG_ITEMS  5

static int calc_chat_view_lines(const TuiEngine *tui) {
    int v = (tui ? tui->rows : TUI_ROWS) - 11;
    return (v >= 4) ? v : 4;
}

static void tui_draw_hsep(TuiEngine *tui, int y, uint8_t attr) {
    int cols = tui->cols;
    tui_put_cell(tui, 0, y, 0x251Cu, attr);
    for (int i = 1; i < cols - 1; i++) tui_put_cell(tui, i, y, 0x2500u, attr);
    tui_put_cell(tui, cols - 1, y, 0x2524u, attr);
}

static void tui_render_screen(TuiEngine *tui, const QmxModel *model, const QmxSession *sess,
                              int pos, float temp, int top_k, float top_p, float rep_pen,
                              int tok_cur, int tok_total, float tps, float ms_per_tok, float step0_chk,
                              int is_generating, int modal_open, int modal_sel,
                              const char *active_profile, const char *profiles_file, int plugin_count) {
    QmxModelInfo info;
    QmxStats stats;
    qmx_get_model_info(model, &info);
    qmx_get_stats(model, &stats);

    int cols = tui->cols;
    int rows = tui->rows;
    int col1 = (cols == 80) ? 25 : (cols / 3);
    int col2 = (cols == 80) ? 52 : ((cols * 2) / 3);
    int chat_lines = calc_chat_view_lines(tui);
    int input_sep_y = rows - 5;
    int input_y     = rows - 4;
    int prog_sep_y  = rows - 3;
    int prog_y      = rows - 2;
    int fkey_y      = rows - 1;

    tui_clear(tui, TUI_ATTR_NORMAL);
    tui_draw_box(tui, 0, 0, cols, rows - 1, TUI_ATTR_BORDER, tui_str(tui, STR_SYS_TITLE));
    {
        char badge[64];
        if (active_profile && active_profile[0] != '\0' && plugin_count > 0) {
            snprintf(badge, sizeof(badge), "[P:%.20s | PLG:%d]", active_profile, plugin_count);
        } else if (active_profile && active_profile[0] != '\0') {
            snprintf(badge, sizeof(badge), "[P:%.24s]", active_profile);
        } else if (plugin_count > 0) {
            snprintf(badge, sizeof(badge), "[P:default | PLG:%d]", plugin_count);
        } else {
            snprintf(badge, sizeof(badge), "[P:default]");
        }
        int bcols = tui_utf8_cols(badge);
        int tcols = tui_utf8_cols(tui_str(tui, STR_SYS_TITLE));
        if (cols - 2 - bcols > tcols + 5) {
            tui_print(tui, cols - 2 - bcols, 0, TUI_ATTR_WARN, badge);
        }
    }
    tui_printf(tui, 2, 1, TUI_ATTR_NORMAL, "%s:%s | %s:%s | %s | T:%.1f K:%d | J:%d",
               tui_str(tui, STR_BACKEND), info.backend_name,
               tui_str(tui, STR_MODE), qmx_app_quant_label(info.primary_quant),
               info.stream_mode ? tui_str(tui, STR_MEM_STREAM) : tui_str(tui, STR_MEM_RAM),
               (double)temp, top_k, info.num_threads);

    tui_draw_hsep(tui, 2, TUI_ATTR_BORDER);
    tui_put_cell(tui, col1, 2, 0x252Cu, TUI_ATTR_BORDER);
    tui_put_cell(tui, col2, 2, 0x252Cu, TUI_ATTR_BORDER);

    float arena_kb = (float)stats.arena_used / 1024.0f;
    float arena_max_kb = (float)stats.arena_capacity / 1024.0f;
    if (arena_max_kb >= 1024.0f) {
        tui_printf(tui, 2, 3, TUI_ATTR_NORMAL, "%s: %.1fM/%.1fM",
                   tui_str(tui, STR_ARENA), (double)(arena_kb / 1024.0f), (double)(arena_max_kb / 1024.0f));
    } else {
        tui_printf(tui, 2, 3, TUI_ATTR_NORMAL, "%s: %.1fK/%.0fK",
                   tui_str(tui, STR_ARENA), (double)arena_kb, (double)arena_max_kb);
    }

    const char *kv_tag = (info.kv_type == QMX_KV_Q8) ? "I8" : ((info.kv_type == QMX_KV_FP16) ? "F16" : "F32");
    float kv_kb = (float)stats.kv_cache_bytes / 1024.0f;
    if (kv_kb >= 1024.0f) {
        tui_printf(tui, 2, 4, TUI_ATTR_NORMAL, "%s(%s):%.1fM[%d/%u]",
                   tui_str(tui, STR_KV_CACHE), kv_tag,
                   (double)(kv_kb / 1024.0f), pos, info.max_seq_len);
    } else {
        tui_printf(tui, 2, 4, TUI_ATTR_NORMAL, "%s(%s):%.1fK[%d/%u]",
                   tui_str(tui, STR_KV_CACHE), kv_tag,
                   (double)kv_kb, pos, info.max_seq_len);
    }

    tui_put_cell(tui, col1, 3, 0x2502u, TUI_ATTR_BORDER);
    tui_put_cell(tui, col1, 4, 0x2502u, TUI_ATTR_BORDER);
    tui_put_cell(tui, col2, 3, 0x2502u, TUI_ATTR_BORDER);
    tui_put_cell(tui, col2, 4, 0x2502u, TUI_ATTR_BORDER);

    if (info.stream_mode) {
        if (is_generating && stats.last_stream_layer >= 0) {
            tui_printf(tui, col1 + 2, 3, TUI_ATTR_WARN, "[IO: READ L%02d]", stats.last_stream_layer);
        } else {
            tui_print(tui, col1 + 2, 3, TUI_ATTR_NORMAL, tui_str(tui, STR_HDD_IDLE));
        }
        uint32_t total_emb = stats.emb_cache_hits + stats.emb_cache_misses;
        float hit_pct = (total_emb > 0u) ? ((float)stats.emb_cache_hits * 100.0f / (float)total_emb) : 0.0f;
        tui_printf(tui, col1 + 2, 4, TUI_ATTR_NORMAL, "LRU: %u/%u (%.0f%%)", stats.emb_cache_hits, total_emb, (double)hit_pct);
    } else {
        tui_print(tui, col1 + 2, 3, TUI_ATTR_NORMAL, tui_str(tui, STR_HDD_IDLE));
        tui_printf(tui, col1 + 2, 4, TUI_ATTR_NORMAL, "%s: %u/%u",
                   tui_str(tui, STR_PINNED), stats.pinned_tensors, stats.tensor_count);
    }

    tui_printf(tui, col2 + 2, 3, TUI_ATTR_WARN, "%s: %.1f tok/s", tui_str(tui, STR_SPEED), (double)tps);
    tui_printf(tui, col2 + 2, 4, TUI_ATTR_NORMAL, "%s: %.2f ms", tui_str(tui, STR_LATENCY), (double)ms_per_tok);

    tui_draw_hsep(tui, 5, TUI_ATTR_BORDER);
    tui_put_cell(tui, col1, 5, 0x2534u, TUI_ATTR_BORDER);
    tui_put_cell(tui, col2, 5, 0x2534u, TUI_ATTR_BORDER);

    int start_line = 0;
    if (sess->line_count > chat_lines) {
        start_line = sess->line_count - chat_lines - sess->scroll_offset;
        if (start_line < 0) start_line = 0;
    }
    int view_cnt = sess->line_count - start_line;
    if (view_cnt > chat_lines) view_cnt = chat_lines;
    for (int i = 0; i < view_cnt; i++) {
        int li = start_line + i;
        uint8_t role = sess->line_roles[li];
        uint8_t attr = (role == (uint8_t)QMX_LINE_USER) ? TUI_ATTR_USER
                     : ((role == (uint8_t)QMX_LINE_WARN) ? TUI_ATTR_WARN : TUI_ATTR_NORMAL);
        tui_print(tui, 2, 6 + i, attr, sess->lines[li]);
    }
    if (sess->scroll_offset > 0 && cols > 20) {
        tui_printf(tui, cols - 18, 5, TUI_ATTR_WARN, "[PgUp/Dn: -%d]", sess->scroll_offset);
    }

    if (modal_open) {
        int mw = (cols - 6 < 52) ? (cols - 6) : 52;
        if (mw < 32) mw = 32;
        int total_items = MODAL_CFG_ITEMS + (sess->model_count > 0 ? sess->model_count : 1);
        int mh = total_items + 4;
        if (mh > chat_lines) mh = chat_lines;
        if (mh < 9 && chat_lines >= 9) mh = 9;
        int mx = (cols - mw) / 2;
        int my = 6;
        if (my + mh > input_sep_y) mh = input_sep_y - my;
        if (mh < 4) mh = 4;
        for (int ry = my; ry < my + mh && ry < input_sep_y; ry++) {
            for (int rx = mx; rx < mx + mw && rx < cols - 1; rx++) {
                tui_put_cell(tui, rx, ry, ' ', TUI_ATTR_NORMAL);
            }
        }
        tui_draw_box(tui, mx, my, mw, mh, TUI_ATTR_WARN, tui_str(tui, STR_SELECT_MODEL));
        uint8_t a0 = (modal_sel == 0) ? TUI_ATTR_HOTKEY_LBL : TUI_ATTR_NORMAL;
        uint8_t a1 = (modal_sel == 1) ? TUI_ATTR_HOTKEY_LBL : TUI_ATTR_NORMAL;
        uint8_t a2 = (modal_sel == 2) ? TUI_ATTR_HOTKEY_LBL : TUI_ATTR_NORMAL;
        uint8_t a3 = (modal_sel == 3) ? TUI_ATTR_HOTKEY_LBL : TUI_ATTR_WARN;
        uint8_t a4 = (modal_sel == 4) ? TUI_ATTR_HOTKEY_LBL : TUI_ATTR_WARN;
        const char *ini_base = (profiles_file && profiles_file[0] != '\0') ? profiles_file : "qmx_profiles.ini";
        const char *sl = strrchr(ini_base, '/');
        const char *bs = strrchr(ini_base, '\\');
        if (sl && (!bs || sl > bs)) ini_base = sl + 1;
        else if (bs) ini_base = bs + 1;
        tui_printf(tui, mx + 2, my + 1, a0, "%s Max Gen Steps : < %d tok >", (modal_sel == 0) ? ">" : " ", tok_total);
        tui_printf(tui, mx + 2, my + 2, a1, "%s Top-K / Top-P : < K=%d P=%.2f >", (modal_sel == 1) ? ">" : " ", top_k, (double)top_p);
        if (my + 3 < my + mh - 1) tui_printf(tui, mx + 2, my + 3, a2, "%s Rep. Penalty  : < %.2f >", (modal_sel == 2) ? ">" : " ", (double)rep_pen);
        if (my + 4 < my + mh - 1) tui_printf(tui, mx + 2, my + 4, a3, "%s [Save Settings to QMX.CFG]", (modal_sel == 3) ? ">" : " ");
        if (my + 5 < my + mh - 1) tui_printf(tui, mx + 2, my + 5, a4, "%s [Save INI -> %-20.20s]", (modal_sel == 4) ? ">" : " ", ini_base);
        if (my + 6 < my + mh - 1) {
            for (int i = 1; i < mw - 1; i++) {
                tui_put_cell(tui, mx + i, my + 6, 0x2500u, TUI_ATTR_BORDER);
            }
        }
        int max_mod_rows = mh - 8;
        if (max_mod_rows > 0) {
            if (sess->model_count == 0) {
                uint8_t am = (modal_sel == MODAL_CFG_ITEMS) ? TUI_ATTR_HOTKEY_LBL : TUI_ATTR_WARN;
                tui_print(tui, mx + 2, my + 7, am, "No *.qmx files found in current dir");
            } else {
                int mod_sel = modal_sel - MODAL_CFG_ITEMS;
                int m_start = 0;
                if (mod_sel >= max_mod_rows) m_start = mod_sel - max_mod_rows + 1;
                for (int m = 0; m < max_mod_rows && (m_start + m) < sess->model_count; m++) {
                    int idx = m_start + m;
                    int item_idx = MODAL_CFG_ITEMS + idx;
                    uint8_t mattr = (item_idx == modal_sel) ? TUI_ATTR_HOTKEY_LBL : TUI_ATTR_NORMAL;
                    tui_printf(tui, mx + 2, my + 7 + m, mattr, "%s [QMX] %-34.34s",
                               (item_idx == modal_sel) ? ">" : " ", sess->models[idx]);
                }
            }
        }
    }

    tui_draw_hsep(tui, input_sep_y, TUI_ATTR_BORDER);
    tui_printf(tui, 2, input_y, TUI_ATTR_USER, "> %s_", sess->input_buf);

    tui_draw_hsep(tui, prog_sep_y, TUI_ATTR_BORDER);
    tui_printf(tui, 2, prog_y, TUI_ATTR_NORMAL, "%s: [", tui_str(tui, STR_GEN_PROG));
    tui_draw_bar(tui, 15, prog_y, 20, tok_cur, tok_total > 0 ? tok_total : 1, TUI_ATTR_TITLE, TUI_ATTR_BORDER);
    tui_printf(tui, 36, prog_y, TUI_ATTR_NORMAL, "] %d/%d tok | chk: %.4f", tok_cur, tok_total, (double)step0_chk);

    for (int i = 0; i < cols; i++) {
        tui_put_cell(tui, i, fkey_y, ' ', TUI_ATTR_BAR);
    }
    const TuiStringId fkeys[] = {
        STR_F1_HELP, STR_F2_BACKEND, STR_F3_KV8, STR_F4_STREAM,
        STR_F5_RESET, STR_F6_CTX, STR_F7_TEMP, STR_F8_LANG, STR_F9_MODELS, STR_F10_QUIT
    };
    int fx = 1;
    for (size_t k = 0; k < sizeof(fkeys) / sizeof(fkeys[0]); k++) {
        const char *label = tui_str(tui, fkeys[k]);
        int lcols = tui_utf8_cols(label);
        if (fx + lcols >= cols) break;
        tui_print(tui, fx, fkey_y, TUI_ATTR_HOTKEY_NUM, label);
        fx += lcols + 2;
    }
    tui_flip(tui);
}

static int reload_model_safe(QmxModel *model, int num_threads,
                             const char *new_path, uint64_t new_off, uint64_t new_slice,
                             const QmxLoadParams *new_p,
                             const char *fb_path, uint64_t fb_off, uint64_t fb_slice,
                             const QmxLoadParams *fb_p) {
    qmx_free(model);
    if (qmx_load_ex(model, new_path, new_off, new_slice, new_p) == 0) {
        qmx_set_num_threads(model, num_threads);
        return 0;
    }
    if (qmx_load_ex(model, fb_path, fb_off, fb_slice, fb_p) == 0) {
        qmx_set_num_threads(model, num_threads);
        return -1;
    }
    return -2;
}

static int calc_tui_steps_for_ctx(uint32_t max_seq_len, int requested_steps) {
    int target = requested_steps;
    if (target > (int)max_seq_len - 16) {
        target = (int)max_seq_len / 2;
    }
    if (target < 16) target = 16;
    return target;
}

static void adjust_modal_setting(int modal_sel, int dir, uint32_t max_seq_len,
                                 int *steps, int *top_k, float *top_p, float *rep_pen) {
    if (modal_sel == 0) {
        int step_opts[] = { 32, 64, 128, 256, 384 };
        int n_opts = 5;
        int cur_i = 0;
        for (int i = 0; i < n_opts; i++) {
            if (*steps >= step_opts[i]) cur_i = i;
        }
        cur_i = (cur_i + dir + n_opts) % n_opts;
        *steps = calc_tui_steps_for_ctx(max_seq_len, step_opts[cur_i]);
    } else if (modal_sel == 1) {
        int   k_opts[] = { 1, 20, 40, 64 };
        float p_opts[] = { 1.0f, 0.85f, 0.90f, 0.95f };
        int   n_opts = 4;
        int   cur_i = 2;
        for (int i = 0; i < n_opts; i++) {
            if (*top_k == k_opts[i]) { cur_i = i; break; }
        }
        cur_i = (cur_i + dir + n_opts) % n_opts;
        *top_k = k_opts[cur_i];
        *top_p = p_opts[cur_i];
    } else if (modal_sel == 2) {
        float r_opts[] = { 1.00f, 1.10f, 1.15f, 1.25f };
        int   n_opts = 4;
        int   cur_i = 2;
        for (int i = 0; i < n_opts; i++) {
            if (*rep_pen >= r_opts[i] - 0.02f && *rep_pen <= r_opts[i] + 0.02f) { cur_i = i; break; }
        }
        cur_i = (cur_i + dir + n_opts) % n_opts;
        *rep_pen = r_opts[cur_i];
    }
}

static void run_tui_session(QmxModel *model, int num_threads,
                            const char *init_model_path, uint64_t base_offset, uint64_t slice_size,
                            const char *tok_path, QmxTokenizer *tokenizer, int *has_tok,
                            int steps, float temp, int top_k, float top_p, float min_p, float rep_pen,
                            uint32_t seed, size_t max_ram_bytes, TuiLang init_lang,
                            const char *active_profile, const char *profiles_file, int plugin_count) {
    TuiEngine tui;
    tui_init(&tui, init_lang);

    QmxSession *sess = (QmxSession *)malloc(sizeof(QmxSession));
    if (!sess) {
        tui_shutdown(&tui);
        return;
    }
    qmx_session_init(sess, (tui.cols > 4) ? (tui.cols - 4) : CHAT_COLS);

    QmxModelInfo info;
    qmx_get_model_info(model, &info);
    steps = calc_tui_steps_for_ctx(info.max_seq_len, steps);

    char active_model_path[256];
    strncpy(active_model_path, init_model_path, sizeof(active_model_path) - 1);
    active_model_path[sizeof(active_model_path) - 1] = '\0';

    size_t prompt_buf_cap = (size_t)info.orig_max_seq_len;
    if (prompt_buf_cap < 256u) prompt_buf_cap = 256u;
    if (prompt_buf_cap > 32768u) prompt_buf_cap = 32768u;
    int *prompt_tokens = (int *)malloc(prompt_buf_cap * sizeof(int));
    if (!prompt_tokens) {
        free(sess);
        tui_shutdown(&tui);
        return;
    }

    QmxGenParams gp = qmx_gen_params_default();
    gp.temp          = temp;
    gp.top_k         = top_k;
    gp.top_p         = top_p;
    gp.min_p         = min_p;
    gp.rep_penalty   = rep_pen;
    gp.seed          = seed;
    gp.max_tokens    = steps;
    gp.prefill_chunk = 8;

    QmxGenState gen;
    qmx_gen_init(&gen, model, (*has_tok) ? tokenizer : NULL, &gp);

    qmx_session_push_line(sess, QMX_LINE_SYSTEM, tui_str(&tui, STR_WELCOME));
    qmx_session_push_line(sess, QMX_LINE_SYSTEM, "");

    float last_tps   = 0.0f;
    float last_ms    = 0.0f;
    float last_chk   = 0.0f;
    int   last_gen   = 0;
    int   modal_open = 0;
    int   modal_sel  = 0;

    tui_render_screen(&tui, model, sess, gen.pos, temp, top_k, top_p, rep_pen,
                      last_gen, steps, last_tps, last_ms, last_chk, 0, modal_open, modal_sel, active_profile, profiles_file, plugin_count);

    int running = 1;
    while (running) {
        if (tui_check_resize(&tui)) {
            qmx_session_set_wrap_cols(sess, (tui.cols > 4) ? (tui.cols - 4) : CHAT_COLS);
            tui_render_screen(&tui, model, sess, gen.pos, temp, top_k, top_p, rep_pen,
                              last_gen, steps, last_tps, last_ms, last_chk, 0, modal_open, modal_sel, active_profile, profiles_file, plugin_count);
        }
        int key = tui_poll_key();
        if (key == -1) {
#if defined(_WIN32)
            Sleep(15);
#elif defined(__DJGPP__)
            delay(15);
#else
            usleep(15000);
#endif
            continue;
        }

        qmx_get_model_info(model, &info);

        if (modal_open) {
            int max_items = MODAL_CFG_ITEMS + sess->model_count;
            if (key == TUI_KEY_ESC || key == TUI_KEY_F9) {
                modal_open = 0;
            } else if (key == TUI_KEY_UP && modal_sel > 0) {
                modal_sel--;
            } else if (key == TUI_KEY_DOWN && modal_sel + 1 < max_items) {
                modal_sel++;
            } else if (key == TUI_KEY_LEFT && modal_sel < 3) {
                adjust_modal_setting(modal_sel, -1, info.max_seq_len, &steps, &top_k, &top_p, &rep_pen);
            } else if (key == TUI_KEY_RIGHT && modal_sel < 3) {
                adjust_modal_setting(modal_sel, +1, info.max_seq_len, &steps, &top_k, &top_p, &rep_pen);
            } else if (key == '\r' || key == '\n') {
                if (modal_sel < 3) {
                    adjust_modal_setting(modal_sel, +1, info.max_seq_len, &steps, &top_k, &top_p, &rep_pen);
                } else if (modal_sel == 3) {
                    if (qmx_app_save_cfg_file("QMX.CFG", &info, active_model_path, steps, temp,
                                              tui.lang == TUI_LANG_RU, max_ram_bytes) == 0) {
                        qmx_session_push_line(sess, QMX_LINE_SYSTEM, "[OK] Saved runtime settings to QMX.CFG");
                    } else {
                        qmx_session_push_line(sess, QMX_LINE_WARN, "[Error] Failed to write QMX.CFG");
                    }
                    modal_open = 0;
                } else if (modal_sel == 4) {
                    const char *ini_dst = (profiles_file && profiles_file[0] != '\0') ? profiles_file : "qmx_profiles.ini";
                    char sec_name[96];
                    if (active_profile && active_profile[0] != '\0') {
                        snprintf(sec_name, sizeof(sec_name), "profile.%.80s", active_profile);
                    } else {
                        strncpy(sec_name, "default", sizeof(sec_name) - 1u);
                        sec_name[sizeof(sec_name) - 1u] = '\0';
                    }
                    QmxProfile sp;
                    qmx_profile_init(&sp);
                    strncpy(sp.model, active_model_path, sizeof(sp.model) - 1u);
                    sp.model[sizeof(sp.model) - 1u] = '\0';
                    if (tok_path && tok_path[0] != '\0') {
                        strncpy(sp.tokenizer, tok_path, sizeof(sp.tokenizer) - 1u);
                        sp.tokenizer[sizeof(sp.tokenizer) - 1u] = '\0';
                        sp.mask |= QMX_PROF_F_TOKENIZER;
                    }
                    sp.backend       = info.backend;
                    sp.kv_type       = info.kv_type;
                    sp.threads       = num_threads;
                    sp.ctx           = info.max_seq_len;
                    sp.steps         = steps;
                    sp.temp          = temp;
                    sp.top_k         = top_k;
                    sp.top_p         = top_p;
                    sp.min_p         = min_p;
                    sp.rep_penalty   = rep_pen;
                    sp.seed          = seed;
                    sp.force_stream  = info.stream_mode;
                    sp.max_ram_bytes = max_ram_bytes;
                    sp.lang_ru       = (tui.lang == TUI_LANG_RU) ? 1 : 0;
                    sp.mask |= QMX_PROF_F_MODEL | QMX_PROF_F_BACKEND | QMX_PROF_F_KV | QMX_PROF_F_THREADS |
                               QMX_PROF_F_CTX | QMX_PROF_F_STEPS | QMX_PROF_F_TEMP | QMX_PROF_F_TOP_K |
                               QMX_PROF_F_TOP_P | QMX_PROF_F_MIN_P | QMX_PROF_F_REP_PENALTY | QMX_PROF_F_SEED |
                               QMX_PROF_F_STREAM | QMX_PROF_F_LANG_RU;
                    if (max_ram_bytes > 0u) sp.mask |= QMX_PROF_F_MAX_RAM;
                    char msg[320];
                    if (qmx_profile_save_ini(ini_dst, sec_name, &sp) == 0) {
                        snprintf(msg, sizeof(msg), "[OK] Saved [%s] to %.160s", sec_name, ini_dst);
                        qmx_session_push_line(sess, QMX_LINE_SYSTEM, msg);
                    } else {
                        snprintf(msg, sizeof(msg), "[Error] Failed to write %.160s", ini_dst);
                        qmx_session_push_line(sess, QMX_LINE_WARN, msg);
                    }
                    modal_open = 0;
                } else if (sess->model_count > 0) {
                    int m_idx = modal_sel - MODAL_CFG_ITEMS;
                    QmxLoadParams lp = qmx_load_params_default();
                    lp.override_max_seq = info.max_seq_len;
                    lp.backend          = info.backend;
                    lp.kv_type          = info.kv_type;
                    lp.force_stream     = info.stream_mode;
                    lp.max_ram_bytes    = max_ram_bytes;

                    int rc = reload_model_safe(model, num_threads,
                                               sess->models[m_idx], 0u, 0u, &lp,
                                               active_model_path, base_offset, slice_size, &lp);
                    if (rc == 0) {
                        strncpy(active_model_path, sess->models[m_idx], sizeof(active_model_path) - 1);
                        active_model_path[sizeof(active_model_path) - 1] = '\0';
                        base_offset = 0u;
                        slice_size  = 0u;
                        qmx_get_model_info(model, &info);
                        qmx_app_reload_tokenizer(&info, active_model_path, 0u, tok_path, tokenizer, has_tok);
                        qmx_gen_init(&gen, model, (*has_tok) ? tokenizer : NULL, &gp);
                        steps = calc_tui_steps_for_ctx(info.max_seq_len, steps);
                        last_gen = 0;
                        if ((size_t)info.max_seq_len > prompt_buf_cap) {
                            size_t new_cap = (size_t)info.max_seq_len;
                            int *new_pt = (int *)realloc(prompt_tokens, new_cap * sizeof(int));
                            if (new_pt) {
                                prompt_tokens = new_pt;
                                prompt_buf_cap = new_cap;
                            }
                        }
                        qmx_session_push_line(sess, QMX_LINE_SYSTEM, active_model_path);
                    } else if (rc == -1) {
                        qmx_session_push_line(sess, QMX_LINE_WARN, "[Error] Failed to load selected .qmx; restored previous model");
                    } else {
                        running = 0;
                        break;
                    }
                    modal_open = 0;
                }
            }
            tui_render_screen(&tui, model, sess, gen.pos, temp, top_k, top_p, rep_pen,
                              last_gen, steps, last_tps, last_ms, last_chk, 0, modal_open, modal_sel, active_profile, profiles_file, plugin_count);
            continue;
        }

        if (key == TUI_KEY_ESC || key == TUI_KEY_F10) {
            running = 0;
            break;
        } else if (key == TUI_KEY_PGUP) {
            qmx_session_scroll(sess, +4, calc_chat_view_lines(&tui));
        } else if (key == TUI_KEY_PGDN) {
            qmx_session_scroll(sess, -4, calc_chat_view_lines(&tui));
        } else if (key == TUI_KEY_F1) {
            qmx_session_push_line(sess, QMX_LINE_SYSTEM, tui_str(&tui, STR_HELP_INFO));
        } else if (key == TUI_KEY_F2) {
            QmxBackend next_b = qmx_app_cycle_backend(info.backend);
            qmx_bind_backend_ops(model, next_b);
        } else if (key == TUI_KEY_F3) {
            QmxLoadParams old_p = qmx_load_params_default();
            old_p.override_max_seq = info.max_seq_len;
            old_p.backend          = info.backend;
            old_p.kv_type          = info.kv_type;
            old_p.force_stream     = info.stream_mode;
            old_p.max_ram_bytes    = max_ram_bytes;
            QmxLoadParams new_p = old_p;
            new_p.kv_type = (info.kv_type == QMX_KV_FP32) ? QMX_KV_FP16
                          : ((info.kv_type == QMX_KV_FP16) ? QMX_KV_Q8 : QMX_KV_FP32);
            int rc = reload_model_safe(model, num_threads,
                                       active_model_path, base_offset, slice_size, &new_p,
                                       active_model_path, base_offset, slice_size, &old_p);
            if (rc == -2) { running = 0; break; }
            qmx_gen_reset(&gen);
        } else if (key == TUI_KEY_F4) {
            QmxLoadParams old_p = qmx_load_params_default();
            old_p.override_max_seq = info.max_seq_len;
            old_p.backend          = info.backend;
            old_p.kv_type          = info.kv_type;
            old_p.force_stream     = info.stream_mode;
            old_p.max_ram_bytes    = max_ram_bytes;
            QmxLoadParams new_p = old_p;
            new_p.force_stream = !info.stream_mode;
            int rc = reload_model_safe(model, num_threads,
                                       active_model_path, base_offset, slice_size, &new_p,
                                       active_model_path, base_offset, slice_size, &old_p);
            if (rc == -2) { running = 0; break; }
            qmx_gen_reset(&gen);
        } else if (key == TUI_KEY_F5) {
            qmx_gen_reset(&gen);
            qmx_session_push_line(sess, QMX_LINE_SYSTEM, tui_str(&tui, STR_CTX_RESET));
        } else if (key == TUI_KEY_F6) {
            uint32_t next_ctx = info.max_seq_len * 2u;
            if (next_ctx > info.orig_max_seq_len) next_ctx = 64u;
            if (next_ctx > info.orig_max_seq_len) next_ctx = info.orig_max_seq_len;
            QmxLoadParams old_p = qmx_load_params_default();
            old_p.override_max_seq = info.max_seq_len;
            old_p.backend          = info.backend;
            old_p.kv_type          = info.kv_type;
            old_p.force_stream     = info.stream_mode;
            old_p.max_ram_bytes    = max_ram_bytes;
            QmxLoadParams new_p = old_p;
            new_p.override_max_seq = next_ctx;
            int rc = reload_model_safe(model, num_threads,
                                       active_model_path, base_offset, slice_size, &new_p,
                                       active_model_path, base_offset, slice_size, &old_p);
            if (rc == -2) { running = 0; break; }
            qmx_get_model_info(model, &info);
            steps = calc_tui_steps_for_ctx(info.max_seq_len, (int)info.max_seq_len / 2);
            qmx_gen_reset(&gen);
        } else if (key == TUI_KEY_F7) {
            if (temp < 0.1f) temp = 0.5f;
            else if (temp < 0.6f) temp = 0.8f;
            else if (temp < 0.9f) temp = 1.0f;
            else temp = 0.0f;
        } else if (key == TUI_KEY_F8) {
            tui_toggle_lang(&tui);
            qmx_session_clear(sess);
            qmx_session_push_line(sess, QMX_LINE_SYSTEM, tui_str(&tui, STR_WELCOME));
            qmx_session_push_line(sess, QMX_LINE_SYSTEM, "");
        } else if (key == TUI_KEY_F9) {
            qmx_session_scan_models(sess, ".");
            modal_sel = 0;
            modal_open = 1;
        } else if (key == '\b' || key == 127) {
            qmx_session_input_backspace(sess);
        } else if (key == '\r' || key == '\n') {
            if (sess->input_len == 0) continue;
            char line_prompt[320];
            snprintf(line_prompt, sizeof(line_prompt), "%s > %s", tui_str(&tui, STR_YOU), sess->input_buf);
            qmx_session_push_line(sess, QMX_LINE_USER, line_prompt);

            char formatted_prompt[512];
            int is_qa_wrapped = 0;
            (void)qmx_session_format_prompt(tokenizer, *has_tok, sess->input_buf,
                                            (gen.pos == 0), 1,
                                            formatted_prompt, sizeof(formatted_prompt),
                                            &is_qa_wrapped);

            int max_prompt_cap = (int)info.max_seq_len - 16;
            if (max_prompt_cap < 16) max_prompt_cap = 16;
            if ((size_t)max_prompt_cap > prompt_buf_cap) max_prompt_cap = (int)prompt_buf_cap;
            int num_p = 0;
            if (*has_tok) {
                qmx_tokenizer_encode(tokenizer, formatted_prompt, prompt_tokens, &num_p, max_prompt_cap, (gen.pos == 0));
            } else {
                prompt_tokens[0] = 1;
                num_p = 1;
            }
            if (num_p <= 0) {
                prompt_tokens[0] = 1;
                num_p = 1;
            }

            qmx_session_input_clear(sess);

            char resp_line[300];
            snprintf(resp_line, sizeof(resp_line), "%s > ", tui_str(&tui, STR_ASSISTANT));
            qmx_session_begin_response(sess, resp_line);

            gp.temp          = temp;
            gp.top_k         = top_k;
            gp.top_p         = top_p;
            gp.min_p         = min_p;
            gp.rep_penalty   = rep_pen;
            gp.max_tokens    = steps;
            gp.prefill_chunk = 8;
            gp.check_qa_stop = is_qa_wrapped;
            qmx_gen_begin(&gen, prompt_tokens, num_p, &gp);

            float t_start = qmx_app_time_ms();
            int prefill_aborted = 0;
            QmxStepStatus st;
            QmxStepResult res;

            while ((st = qmx_step(&gen, &res)) > 0) {
                if (res.kv_shifted) {
                    qmx_session_push_line(sess, QMX_LINE_WARN, tui_str(&tui, STR_CTX_SHIFTED));
                }
                if (st == QMX_STEP_PREFILL) {
                    float dt = qmx_app_time_ms() - t_start;
                    int done_p = res.prefill_done;
                    last_tps = (dt > 0.0f) ? ((float)done_p * 1000.0f / dt) : 0.0f;
                    last_ms  = (dt > 0.0f && done_p > 0) ? (dt / (float)done_p) : 0.0f;
                    last_chk = gen.step0_checksum;
                    tui_render_screen(&tui, model, sess, res.pos, temp, top_k, top_p, rep_pen,
                                      0, steps, last_tps, last_ms, last_chk, 1, modal_open, modal_sel, active_profile, profiles_file, plugin_count);
                    if (tui_poll_key() == TUI_KEY_ESC) {
                        prefill_aborted = 1;
                        break;
                    }
                } else if (st == QMX_STEP_TOKEN) {
                    const char *piece = res.piece;
                    char fallback_tok[16];
                    if (!piece && !(*has_tok)) {
                        snprintf(fallback_tok, sizeof(fallback_tok), "[%d]", res.token);
                        piece = fallback_tok;
                    }
                    qmx_session_append_piece(sess, piece);
                    float dt = qmx_app_time_ms() - t_start;
                    int total_f = (int)(gen.prefill_tokens + gen.decode_tokens);
                    if (total_f < 1) total_f = 1;
                    last_tps = (dt > 0.0f) ? ((float)total_f * 1000.0f / dt) : 0.0f;
                    last_ms  = (dt > 0.0f) ? (dt / (float)total_f) : 0.0f;
                    last_gen = res.generated;
                    last_chk = gen.step0_checksum;
                    tui_render_screen(&tui, model, sess, res.pos, temp, top_k, top_p, rep_pen,
                                      last_gen, steps, last_tps, last_ms, last_chk, 1, modal_open, modal_sel, active_profile, profiles_file, plugin_count);
                    if (tui_poll_key() == TUI_KEY_ESC) {
                        break;
                    }
                } else {
                    break;
                }
            }
            if (!prefill_aborted) {
                (void)qmx_gen_flush(&gen);
            }
            qmx_session_push_line(sess, QMX_LINE_SYSTEM, "");
        } else if (key >= 32 && key < 256 && sess->input_len < QMX_SESSION_INPUT_CAP - 5) {
            (void)qmx_session_input_append_byte(sess, (char)key);
            int need = qmx_app_utf8_expected_len((unsigned char)key);
            for (int u = 1; u < need; u++) {
                int nb = tui_poll_key();
                if (nb >= 0x80 && nb < 0xC0) {
                    (void)qmx_session_input_append_byte(sess, (char)nb);
                } else {
                    break;
                }
            }
        }

        tui_render_screen(&tui, model, sess, gen.pos, temp, top_k, top_p, rep_pen,
                          last_gen, steps, last_tps, last_ms, last_chk, 0, modal_open, modal_sel, active_profile, profiles_file, plugin_count);
    }

    free(prompt_tokens);
    free(sess);
    tui_shutdown(&tui);
}

static char *read_stdin_prompt(void) {
    size_t cap = 4096u;
    size_t len = 0u;
    char *buf = (char *)malloc(cap);
    if (!buf) return NULL;
    for (;;) {
        if (len + 1024u >= cap) {
            if (cap >= 65536u) break;
            size_t ncap = cap * 2u;
            char *nb = (char *)realloc(buf, ncap);
            if (!nb) break;
            buf = nb;
            cap = ncap;
        }
        size_t n = fread(buf + len, 1, cap - len - 1u, stdin);
        len += n;
        if (n == 0u) break;
    }
    while (len > 0u && (buf[len - 1u] == '\n' || buf[len - 1u] == '\r')) len--;
    buf[len] = '\0';
    return buf;
}

int main(int argc, char **argv) {
    const char *prompt = NULL;
    const char *prompt_file = NULL;
    char *prompt_file_buf = NULL;
    const char *profile_name = NULL;
    const char *profiles_file = "qmx_profiles.ini";
    int profiles_file_explicit = 0;
    int force_tui = 0;
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
            cli_p.steps = qmx_app_parse_i32(argv[++i], 1, 131072, 128);
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
            cli_p.temp = qmx_app_parse_f32(argv[++i], 0.0f, 10.0f, 0.5f);
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
        } else if (strcmp(argv[i], "--kv8") == 0) {
            cli_p.kv_type = QMX_KV_Q8;
            cli_p.mask |= QMX_PROF_F_KV;
        } else if (strcmp(argv[i], "--kv16") == 0) {
            cli_p.kv_type = QMX_KV_FP16;
            cli_p.mask |= QMX_PROF_F_KV;
        } else if (strcmp(argv[i], "--tui") == 0 || strcmp(argv[i], "--repl") == 0) {
            force_tui = 1;
        } else if (strcmp(argv[i], "--ru") == 0) {
            cli_p.lang_ru = 1;
            cli_p.mask |= QMX_PROF_F_LANG_RU;
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
    resolved.temp = 0.5f;

    QmxRuntimeCfg rt_cfg;
    if (qmx_app_load_cfg_file("QMX.CFG", &rt_cfg) == 0) {
        QmxProfile cfg_p;
        qmx_profile_from_cfg(&cfg_p, &rt_cfg);
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
        force_tui = 1;
        if (memcmp(rt_cfg.magic, "QMXCFG", 6) == 0) {
            QmxProfile bcfg_p;
            qmx_profile_from_cfg(&bcfg_p, &rt_cfg);
            bcfg_p.mask &= ~QMX_PROF_F_MODEL;
            qmx_profile_merge(&resolved, &bcfg_p);
            qmx_profile_merge(&resolved, &cli_p);
        }
    }

    if (!model_path) {
        const char *candidates[] = { "model.qmx", "stories260k.qmx", "stories15m.qmx", "bitnet.qmx", "dummy.qmx" };
        for (size_t c = 0; c < sizeof(candidates) / sizeof(candidates[0]); c++) {
            FILE *probe = fopen(candidates[c], "rb");
            if (probe) {
                fclose(probe);
                model_path = candidates[c];
                break;
            }
        }
    }

    if (!model_path) {
        fprintf(stderr, "Usage: %s [-m model.qmx] [-z tok.bin] [--profile name] [--profiles-file ini] [--ru] [-c ctx] [-n steps] [-j threads|auto] [-t temp] [--kv8 | --kv16] [--stream] [-b auto|...] [-p prompt]\n", argv[0]);
        return 1;
    }

    int plugin_count = 0;
    if ((resolved.mask & QMX_PROF_F_PLUGIN_DIR) && resolved.plugin_dir[0] != '\0') {
        int n = qmx_plugin_load_dir(resolved.plugin_dir);
        if (n > 0) plugin_count += n;
    }
    if ((resolved.mask & QMX_PROF_F_PLUGIN) && resolved.plugin[0] != '\0') {
        if (qmx_plugin_load_shared(resolved.plugin) == 0) plugin_count++;
    }
    for (int i = 0; i < cli_plugin_dir_cnt; i++) {
        int n = qmx_plugin_load_dir(cli_plugin_dirs[i]);
        if (n > 0) plugin_count += n;
    }
    for (int i = 0; i < cli_plugin_cnt; i++) {
        if (qmx_plugin_load_shared(cli_plugins[i]) == 0) plugin_count++;
    }

    const char *tok_path = (resolved.mask & QMX_PROF_F_TOKENIZER) ? resolved.tokenizer : "";
    int steps = resolved.steps;
    uint32_t ctx_override = resolved.ctx;
    float temp = resolved.temp;
    int temp_set = (resolved.mask & QMX_PROF_F_TEMP) ? 1 : 0;
    int top_k = resolved.top_k;
    float top_p = resolved.top_p;
    float min_p = resolved.min_p;
    float rep_penalty = resolved.rep_penalty;
    uint32_t seed = resolved.seed;
    QmxKvType kv_type = resolved.kv_type;
    int force_stream = resolved.force_stream;
    int no_autotune = resolved.no_autotune;
    int num_threads = resolved.threads;
    size_t max_ram_bytes = resolved.max_ram_bytes;
    TuiLang default_lang = resolved.lang_ru ? TUI_LANG_RU : TUI_LANG_EN;
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

    int cli_mode = (!force_tui && (prompt != NULL || !qmx_session_is_interactive_tty()));
    if (cli_mode && prompt == NULL) {
        prompt_file_buf = read_stdin_prompt();
        prompt = prompt_file_buf ? prompt_file_buf : "";
    }
    if (cli_mode && !temp_set) temp = 0.0f;
    if (steps <= 0) steps = cli_mode ? 32 : 128;
    if (ctx_override == 0) {
        if (cli_mode) {
            size_t plen = strlen(prompt);
            uint32_t est_p = (plen > 128u) ? (uint32_t)(plen + 64u) : 128u;
            ctx_override = (prompt[0] != '\0') ? ((uint32_t)steps + est_p) : (uint32_t)steps;
        } else {
            ctx_override = 256u;
        }
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
            } else if (cli_mode) {
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

    int rc = 0;
    if (cli_mode) {
        QmxGenParams gp = qmx_gen_params_default();
        gp.temp          = temp;
        gp.top_k         = top_k;
        gp.top_p         = top_p;
        gp.min_p         = min_p;
        gp.rep_penalty   = rep_penalty;
        gp.seed          = seed;
        gp.max_tokens    = steps;
        gp.prefill_chunk = (resolved.mask & QMX_PROF_F_PREFILL_CHUNK) ? resolved.prefill_chunk : 32;
        rc = qmx_app_run_cli(&model, &tokenizer, has_tok, prompt, &gp, debug_tokens);
    } else {
        run_tui_session(&model, num_threads, model_path, bundle_offset, bundle_size, tok_path,
                        &tokenizer, &has_tok, steps, temp, top_k, top_p, min_p, rep_penalty,
                        seed, max_ram_bytes, default_lang,
                        profile_name, profiles_file, plugin_count);
    }

    free(prompt_file_buf);
    if (has_tok) qmx_tokenizer_free(&tokenizer);
    qmx_free(&model);
    qmx_plugin_unload_all_shared();
    return (rc == 0) ? 0 : 1;
}
