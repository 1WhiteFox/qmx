#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif
#if defined(_WIN32) && !defined(__USE_MINGW_ANSI_STDIO)
#define __USE_MINGW_ANSI_STDIO 0
#endif

#include "qmx_session.h"
#include <stdio.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#include <dirent.h>
#endif

int qmx_session_utf8_cols(const char *s) {
    if (!s) return 0;
    int cols = 0;
    for (const unsigned char *p = (const unsigned char *)s; *p != '\0'; p++) {
        if ((*p & 0xC0u) == 0x80u) continue;
        if (*p < 0x20u && *p != '\t') continue;
        cols++;
    }
    return cols;
}

int qmx_session_is_interactive_tty(void) {
#if defined(__DJGPP__)
    return 1;
#elif defined(_WIN32)
    HANDLE hin  = GetStdHandle(STD_INPUT_HANDLE);
    HANDLE hout = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hin == INVALID_HANDLE_VALUE || hout == INVALID_HANDLE_VALUE || !hin || !hout) return 0;
    return (GetFileType(hin) == FILE_TYPE_CHAR && GetFileType(hout) == FILE_TYPE_CHAR) ? 1 : 0;
#else
    return (isatty(STDIN_FILENO) && isatty(STDOUT_FILENO)) ? 1 : 0;
#endif
}

void qmx_session_init(QmxSession *sess, int wrap_cols) {
    if (!sess) return;
    memset(sess, 0, sizeof(*sess));
    sess->wrap_cols = (wrap_cols >= 20) ? wrap_cols : 76;
}

static void sess_push_visual_line(QmxSession *sess, QmxLineRole role, const char *line) {
    if (!sess) return;
    if (!line) line = "";
    int idx;
    if (sess->line_count < QMX_SESSION_HISTORY_CAP) {
        idx = sess->line_count++;
    } else {
        for (int i = 0; i < QMX_SESSION_HISTORY_CAP - 1; i++) {
            memcpy(sess->lines[i], sess->lines[i + 1], QMX_SESSION_LINE_BYTES);
            sess->line_roles[i] = sess->line_roles[i + 1];
        }
        idx = QMX_SESSION_HISTORY_CAP - 1;
    }
    strncpy(sess->lines[idx], line, QMX_SESSION_LINE_BYTES - 1u);
    sess->lines[idx][QMX_SESSION_LINE_BYTES - 1u] = '\0';
    sess->line_roles[idx] = (uint8_t)role;
}

static void sess_append_last(QmxSession *sess, const char *str) {
    if (!sess || sess->line_count <= 0 || !str) return;
    char *dst = sess->lines[sess->line_count - 1];
    size_t cur = strlen(dst);
    if (cur + 1u >= QMX_SESSION_LINE_BYTES) return;
    size_t add = strlen(str);
    if (cur + add >= QMX_SESSION_LINE_BYTES) add = QMX_SESSION_LINE_BYTES - cur - 1u;
    memcpy(dst + cur, str, add);
    dst[cur + add] = '\0';
}

static void sess_push_turn(QmxSession *sess, QmxLineRole role, const char *text) {
    if (!sess) return;
    if (!text) text = "";
    int idx;
    if (sess->turn_count < QMX_SESSION_TURN_CAP) {
        idx = sess->turn_count++;
    } else {
        for (int i = 0; i < QMX_SESSION_TURN_CAP - 1; i++) {
            memcpy(sess->turns[i], sess->turns[i + 1], QMX_SESSION_TURN_BYTES);
            sess->turn_roles[i] = sess->turn_roles[i + 1];
        }
        idx = QMX_SESSION_TURN_CAP - 1;
    }
    strncpy(sess->turns[idx], text, QMX_SESSION_TURN_BYTES - 1u);
    sess->turns[idx][QMX_SESSION_TURN_BYTES - 1u] = '\0';
    sess->turn_roles[idx] = (uint8_t)role;
}

static void sess_append_turn_raw(QmxSession *sess, const char *text, int len) {
    if (!sess || !text || len <= 0) return;
    if (sess->turn_count <= 0) {
        sess_push_turn(sess, QMX_LINE_ASSISTANT, "");
    }
    char *dst = sess->turns[sess->turn_count - 1];
    size_t cur = strlen(dst);
    if (cur + 1u >= QMX_SESSION_TURN_BYTES) return;
    size_t add = (size_t)len;
    if (cur + add >= QMX_SESSION_TURN_BYTES) add = QMX_SESSION_TURN_BYTES - cur - 1u;
    memcpy(dst + cur, text, add);
    dst[cur + add] = '\0';
}

static void sess_render_wrapped_turn(QmxSession *sess, QmxLineRole role, const char *text) {
    int wrap = (sess->wrap_cols >= 20) ? sess->wrap_cols : 76;
    const char *p = text ? text : "";
    int first_line = 1;
    char line_buf[QMX_SESSION_LINE_BYTES];

    if (*p == '\0') {
        sess_push_visual_line(sess, role, "");
        sess->resp_cols = 0;
        return;
    }

    while (*p != '\0') {
        const char *nl = strchr(p, '\n');
        const char *seg_end = nl ? nl : (p + strlen(p));
        if (p == seg_end) {
            if (!first_line) {
                sess_push_visual_line(sess, role, "  ");
                sess->resp_cols = 2;
            } else {
                sess_push_visual_line(sess, role, "");
                sess->resp_cols = 0;
            }
            first_line = 0;
            p = nl ? (nl + 1) : seg_end;
            continue;
        }

        const char *cur = p;
        while (cur < seg_end) {
            int out_len = 0;
            int cols = 0;
            if (!first_line) {
                line_buf[out_len++] = ' ';
                line_buf[out_len++] = ' ';
                cols = 2;
                while (cur < seg_end && *cur == ' ') cur++;
                if (cur >= seg_end) break;
            }

            const char *scan = cur;
            const char *last_space_src = NULL;
            int last_space_out = -1;
            int last_space_cols = 0;

            while (scan < seg_end) {
                unsigned char lead = (unsigned char)*scan;
                if (lead == '\r') { scan++; continue; }
                int clen = qmx_app_utf8_expected_len(lead);
                if (scan + clen > seg_end) clen = (int)(seg_end - scan);
                if (lead < 0x20u && lead != '\t') { scan += clen; continue; }

                if (cols + 1 >= wrap && out_len > (first_line ? 0 : 2)) {
                    if (last_space_src != NULL && last_space_out > (first_line ? 0 : 2)) {
                        out_len = last_space_out;
                        cols = last_space_cols;
                        scan = last_space_src + 1;
                    }
                    break;
                }
                if (out_len + clen >= (int)sizeof(line_buf) - 1) break;
                if (lead == ' ') {
                    last_space_src = scan;
                    last_space_out = out_len;
                    last_space_cols = cols;
                }
                memcpy(line_buf + out_len, scan, (size_t)clen);
                out_len += clen;
                cols++;
                scan += clen;
            }

            line_buf[out_len] = '\0';
            sess_push_visual_line(sess, role, line_buf);
            sess->resp_cols = cols;
            first_line = 0;
            if (scan == cur) scan++;
            cur = scan;
        }
        p = nl ? (nl + 1) : seg_end;
    }
}

void qmx_session_reflow(QmxSession *sess, int new_wrap_cols) {
    if (!sess) return;
    if (new_wrap_cols < 20) new_wrap_cols = 20;
    if (new_wrap_cols > QMX_SESSION_LINE_BYTES - 8) new_wrap_cols = QMX_SESSION_LINE_BYTES - 8;
    sess->wrap_cols = new_wrap_cols;
    sess->line_count = 0;
    sess->resp_cols = 0;
    for (int i = 0; i < QMX_SESSION_HISTORY_CAP; i++) {
        sess->lines[i][0] = '\0';
        sess->line_roles[i] = (uint8_t)QMX_LINE_SYSTEM;
    }
    for (int t = 0; t < sess->turn_count; t++) {
        sess_render_wrapped_turn(sess, (QmxLineRole)sess->turn_roles[t], sess->turns[t]);
    }
    if (sess->scroll_offset >= sess->line_count) {
        sess->scroll_offset = (sess->line_count > 0) ? (sess->line_count - 1) : 0;
    }
}

void qmx_session_set_wrap_cols(QmxSession *sess, int wrap_cols) {
    if (!sess) return;
    if (wrap_cols < 20) wrap_cols = 20;
    if (wrap_cols > QMX_SESSION_LINE_BYTES - 8) wrap_cols = QMX_SESSION_LINE_BYTES - 8;
    if (sess->wrap_cols == wrap_cols) return;
    if (sess->turn_count > 0) {
        qmx_session_reflow(sess, wrap_cols);
    } else {
        sess->wrap_cols = wrap_cols;
    }
}

void qmx_session_clear(QmxSession *sess) {
    if (!sess) return;
    sess->line_count     = 0;
    sess->turn_count     = 0;
    sess->scroll_offset  = 0;
    sess->resp_cols      = 0;
    sess->utf8_carry_len = 0;
    for (int i = 0; i < QMX_SESSION_HISTORY_CAP; i++) {
        sess->lines[i][0]   = '\0';
        sess->line_roles[i] = (uint8_t)QMX_LINE_SYSTEM;
    }
    for (int i = 0; i < QMX_SESSION_TURN_CAP; i++) {
        sess->turns[i][0]   = '\0';
        sess->turn_roles[i] = (uint8_t)QMX_LINE_SYSTEM;
    }
}

void qmx_session_push_line(QmxSession *sess, QmxLineRole role, const char *line) {
    if (!sess) return;
    if (!line) line = "";
    sess->scroll_offset = 0;
    sess_push_turn(sess, role, line);
    if (qmx_session_utf8_cols(line) < sess->wrap_cols && strchr(line, '\n') == NULL) {
        sess_push_visual_line(sess, role, line);
        sess->resp_cols = qmx_session_utf8_cols(line);
    } else {
        sess_render_wrapped_turn(sess, role, line);
    }
}

void qmx_session_begin_response(QmxSession *sess, const char *prefix) {
    if (!sess) return;
    sess->utf8_carry_len = 0;
    sess->scroll_offset = 0;
    const char *pfx = prefix ? prefix : "";
    sess_push_turn(sess, QMX_LINE_ASSISTANT, pfx);
    sess_push_visual_line(sess, QMX_LINE_ASSISTANT, pfx);
    sess->resp_cols = qmx_session_utf8_cols(pfx);
}

void qmx_session_append_piece(QmxSession *sess, const char *piece) {
    if (!sess || !piece || piece[0] == '\0') return;
    sess->scroll_offset = 0;
    if (sess->line_count <= 0) {
        sess_push_turn(sess, QMX_LINE_ASSISTANT, "");
        sess_push_visual_line(sess, QMX_LINE_ASSISTANT, "");
        sess->resp_cols = 0;
    }

    char merged_buf[320];
    int m_len = 0;
    if (sess->utf8_carry_len > 0) {
        memcpy(merged_buf, sess->utf8_carry, (size_t)sess->utf8_carry_len);
        m_len = sess->utf8_carry_len;
        sess->utf8_carry_len = 0;
    }
    for (const char *p = piece; *p != '\0' && m_len < (int)sizeof(merged_buf) - 2; p++) {
        merged_buf[m_len++] = *p;
    }
    merged_buf[m_len] = '\0';

    int check_back = (m_len < 3) ? m_len : 3;
    for (int back = 1; back <= check_back; back++) {
        int idx = m_len - back;
        unsigned char b = (unsigned char)merged_buf[idx];
        if ((b & 0xC0u) == 0x80u) continue;
        if (b >= 0xC0u) {
            int need = qmx_app_utf8_expected_len(b);
            if (back < need) {
                memcpy(sess->utf8_carry, merged_buf + idx, (size_t)back);
                sess->utf8_carry_len = back;
                m_len = idx;
                merged_buf[m_len] = '\0';
            }
        }
        break;
    }

    if (m_len > 0) sess_append_turn_raw(sess, merged_buf, m_len);
    int wrap = (sess->wrap_cols >= 20) ? sess->wrap_cols : 76;
    char chunk[300];
    int c_idx = 0;

    for (int i = 0; i < m_len; i++) {
        unsigned char b = (unsigned char)merged_buf[i];
        if (b == '\r') continue;
        if (b == '\n') {
            if (c_idx == 0 && sess->resp_cols <= 8) continue;
            if (c_idx > 0) {
                chunk[c_idx] = '\0';
                int pcols = qmx_session_utf8_cols(chunk);
                if (sess->resp_cols + pcols >= wrap) {
                    char wrap_line[320];
                    snprintf(wrap_line, sizeof(wrap_line), "  %s", chunk);
                    sess_push_visual_line(sess, QMX_LINE_ASSISTANT, wrap_line);
                } else {
                    sess_append_last(sess, chunk);
                }
                c_idx = 0;
            }
            sess_push_visual_line(sess, QMX_LINE_ASSISTANT, "  ");
            sess->resp_cols = 2;
            continue;
        }
        if (b < 0x20u && b != '\t') continue;
        if (c_idx < (int)sizeof(chunk) - 2) {
            chunk[c_idx++] = (char)b;
        }
    }

    if (c_idx > 0) {
        chunk[c_idx] = '\0';
        int pcols = qmx_session_utf8_cols(chunk);
        if (sess->resp_cols + pcols >= wrap) {
            char wrap_line[320];
            snprintf(wrap_line, sizeof(wrap_line), "  %s", chunk);
            sess_push_visual_line(sess, QMX_LINE_ASSISTANT, wrap_line);
            sess->resp_cols = qmx_session_utf8_cols(wrap_line);
        } else {
            sess_append_last(sess, chunk);
            sess->resp_cols += pcols;
        }
    }
}

void qmx_session_scroll(QmxSession *sess, int delta, int view_lines) {
    if (!sess || view_lines <= 0) return;
    int max_sc = sess->line_count - view_lines;
    if (max_sc < 0) max_sc = 0;
    int next = sess->scroll_offset + delta;
    if (next < 0) next = 0;
    if (next > max_sc) next = max_sc;
    sess->scroll_offset = next;
}

void qmx_session_input_clear(QmxSession *sess) {
    if (!sess) return;
    sess->input_buf[0] = '\0';
    sess->input_len = 0;
}

void qmx_session_input_backspace(QmxSession *sess) {
    if (!sess || sess->input_len <= 0) return;
    while (sess->input_len > 0 && ((unsigned char)sess->input_buf[sess->input_len - 1] & 0xC0u) == 0x80u) {
        sess->input_buf[--sess->input_len] = '\0';
    }
    if (sess->input_len > 0) {
        sess->input_buf[--sess->input_len] = '\0';
    }
}

int qmx_session_input_append_byte(QmxSession *sess, char ch) {
    if (!sess || sess->input_len >= QMX_SESSION_INPUT_CAP - 2) return -1;
    sess->input_buf[sess->input_len++] = ch;
    sess->input_buf[sess->input_len] = '\0';
    return 0;
}

int qmx_session_scan_models(QmxSession *sess, const char *dir_path) {
    if (!sess) return 0;
    sess->model_count = 0;
    if (!dir_path || dir_path[0] == '\0') dir_path = ".";
#if defined(_WIN32)
    char pattern[300];
    snprintf(pattern, sizeof(pattern), "%s\\*.qmx", dir_path);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pattern, &fd);
    if (h != INVALID_HANDLE_VALUE) {
        do {
            if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) && sess->model_count < QMX_SESSION_MAX_MODELS) {
                strncpy(sess->models[sess->model_count], fd.cFileName, sizeof(sess->models[0]) - 1u);
                sess->models[sess->model_count][sizeof(sess->models[0]) - 1u] = '\0';
                sess->model_count++;
            }
        } while (FindNextFileA(h, &fd) && sess->model_count < QMX_SESSION_MAX_MODELS);
        FindClose(h);
    }
#else
    DIR *d = opendir(dir_path);
    if (d) {
        struct dirent *de;
        while ((de = readdir(d)) != NULL && sess->model_count < QMX_SESSION_MAX_MODELS) {
            size_t len = strlen(de->d_name);
            const char *ext = de->d_name + (len > 4u ? len - 4u : 0u);
            if (len > 4u && ext[0] == '.' &&
                (ext[1] == 'q' || ext[1] == 'Q') &&
                (ext[2] == 'm' || ext[2] == 'M') &&
                (ext[3] == 'x' || ext[3] == 'X')) {
                strncpy(sess->models[sess->model_count], de->d_name, sizeof(sess->models[0]) - 1u);
                sess->models[sess->model_count][sizeof(sess->models[0]) - 1u] = '\0';
                sess->model_count++;
            }
        }
        closedir(d);
    }
#endif
    return sess->model_count;
}

size_t qmx_session_format_prompt(const QmxTokenizer *tok, int has_tok,
                                 const char *raw_input, int is_first_turn,
                                 int allow_qa_prefix,
                                 char *out_buf, size_t out_cap,
                                 int *out_wrapped) {
    if (out_wrapped) *out_wrapped = 0;
    if (!out_buf || out_cap == 0u) return 0u;
    if (!raw_input) raw_input = "";

    QmxChatTemplate tpl = qmx_app_detect_chat_template(tok, has_tok);
    int n = 0;
    if (raw_input[0] == '!') {
        n = snprintf(out_buf, out_cap, "%s", raw_input + 1);
    } else if (allow_qa_prefix && raw_input[0] == '?') {
        n = snprintf(out_buf, out_cap, "%sQuestion: %s\nAnswer:", is_first_turn ? "" : "\n", raw_input + 1);
        if (out_wrapped) *out_wrapped = 1;
    } else if (raw_input[0] != '\0' && raw_input[0] != '<' && tpl != QMX_CHAT_NONE) {
        if (tpl == QMX_CHAT_LLAMA3) {
            n = snprintf(out_buf, out_cap,
                         "<|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                         raw_input);
        } else if (tpl == QMX_CHAT_CHATML) {
            n = snprintf(out_buf, out_cap,
                         "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
                         raw_input);
        } else if (tpl == QMX_CHAT_PHI3) {
            n = snprintf(out_buf, out_cap,
                         "<|user|>\n%s<|end|>\n<|assistant|>\n",
                         raw_input);
        } else {
            n = snprintf(out_buf, out_cap,
                         "<start_of_turn>user\n%s<end_of_turn>\n<start_of_turn>model\n",
                         raw_input);
        }
        if (out_wrapped) *out_wrapped = 1;
    } else {
        n = snprintf(out_buf, out_cap, "%s", raw_input);
    }

    if (n < 0) {
        out_buf[0] = '\0';
        return 0u;
    }
    return ((size_t)n >= out_cap) ? (out_cap - 1u) : (size_t)n;
}
