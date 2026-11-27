#ifndef QMX_SESSION_H
#define QMX_SESSION_H

#include "qmx_app_common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define QMX_SESSION_HISTORY_CAP 256
#define QMX_SESSION_LINE_BYTES  512
#define QMX_SESSION_TURN_CAP    64
#define QMX_SESSION_TURN_BYTES  1024
#define QMX_SESSION_MAX_MODELS  32
#define QMX_SESSION_INPUT_CAP   256

typedef enum {
    QMX_LINE_SYSTEM    = 0,
    QMX_LINE_USER      = 1,
    QMX_LINE_ASSISTANT = 2,
    QMX_LINE_WARN      = 3
} QmxLineRole;

typedef struct {
    char    lines[QMX_SESSION_HISTORY_CAP][QMX_SESSION_LINE_BYTES];
    uint8_t line_roles[QMX_SESSION_HISTORY_CAP];
    int     line_count;
    int     scroll_offset;
    int     wrap_cols;
    int     resp_cols;
    char    utf8_carry[4];
    int     utf8_carry_len;
    char    input_buf[QMX_SESSION_INPUT_CAP];
    int     input_len;
    char    models[QMX_SESSION_MAX_MODELS][256];
    int     model_count;
    char    turns[QMX_SESSION_TURN_CAP][QMX_SESSION_TURN_BYTES];
    uint8_t turn_roles[QMX_SESSION_TURN_CAP];
    int     turn_count;
} QmxSession;

int  qmx_session_utf8_cols(const char *s);
int  qmx_session_is_interactive_tty(void);

void qmx_session_init(QmxSession *sess, int wrap_cols);
void qmx_session_set_wrap_cols(QmxSession *sess, int wrap_cols);
void qmx_session_reflow(QmxSession *sess, int new_wrap_cols);
void qmx_session_clear(QmxSession *sess);

void qmx_session_push_line(QmxSession *sess, QmxLineRole role, const char *line);
void qmx_session_begin_response(QmxSession *sess, const char *prefix);
void qmx_session_append_piece(QmxSession *sess, const char *piece);
void qmx_session_scroll(QmxSession *sess, int delta, int view_lines);

void qmx_session_input_clear(QmxSession *sess);
void qmx_session_input_backspace(QmxSession *sess);
int  qmx_session_input_append_byte(QmxSession *sess, char ch);

int  qmx_session_scan_models(QmxSession *sess, const char *dir_path);

size_t qmx_session_format_prompt(const QmxTokenizer *tok, int has_tok,
                                 const char *raw_input, int is_first_turn,
                                 int allow_qa_prefix,
                                 char *out_buf, size_t out_cap,
                                 int *out_wrapped);

#ifdef __cplusplus
}
#endif

#endif
