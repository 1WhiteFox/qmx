#include "../apps/common/qmx_session.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures = 0;

#define EXPECT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "[FAIL] %s (line %d)\n", (msg), __LINE__); \
        g_failures++; \
    } \
} while (0)

int main(void) {
    QmxSession *sess = (QmxSession *)malloc(sizeof(QmxSession));
    EXPECT(sess != NULL, "Allocate QmxSession");
    if (!sess) return 1;

    qmx_session_init(sess, 24);
    EXPECT(qmx_session_utf8_cols("Привет, QMX!") == 12, "UTF-8 column count for Cyrillic + ASCII");

    qmx_session_begin_response(sess, "AI > ");
    const char split1[] = { (char)0xD0, '\0' };
    const char split2[] = { (char)0x9F, (char)0xD1, (char)0x80, (char)0xD0, (char)0xB8, '\0' };
    qmx_session_append_piece(sess, split1);
    EXPECT(sess->utf8_carry_len == 1, "Incomplete UTF-8 byte held in carry buffer");
    qmx_session_append_piece(sess, split2);
    EXPECT(sess->utf8_carry_len == 0, "Carry buffer flushed on next UTF-8 chunk");
    EXPECT(strcmp(sess->lines[0], "AI > При") == 0, "Split UTF-8 multibyte sequence reassembled");

    qmx_session_append_piece(sess, "вет мир очень длинная строка");
    EXPECT(sess->line_count >= 2, "Long UTF-8 response wrapped across lines");

    qmx_session_scroll(sess, 10, 1);
    EXPECT(sess->scroll_offset == sess->line_count - 1, "Scroll clamped to max offset");
    qmx_session_scroll(sess, -100, 1);
    EXPECT(sess->scroll_offset == 0, "Scroll clamped to 0");

    qmx_session_input_clear(sess);
    qmx_session_input_append_byte(sess, 'A');
    qmx_session_input_append_byte(sess, (char)0xD0);
    qmx_session_input_append_byte(sess, (char)0xAF);
    qmx_session_input_backspace(sess);
    EXPECT(sess->input_len == 1 && strcmp(sess->input_buf, "A") == 0, "UTF-8 backspace removes whole code point");

    char fmt_out[256];
    int wrapped = 0;
    qmx_session_format_prompt(NULL, 0, "!raw text", 1, 1, fmt_out, sizeof(fmt_out), &wrapped);
    EXPECT(wrapped == 0 && strcmp(fmt_out, "raw text") == 0, "Bang prefix passes raw prompt");

    qmx_session_format_prompt(NULL, 0, "?2+2", 1, 1, fmt_out, sizeof(fmt_out), &wrapped);
    EXPECT(wrapped == 1 && strcmp(fmt_out, "Question: 2+2\nAnswer:") == 0, "QA prefix formats question");

    int lines_narrow = sess->line_count;
    qmx_session_set_wrap_cols(sess, 64);
    EXPECT(sess->line_count == 1, "Reflow to wider viewport collapses wrapped turn to 1 line");
    EXPECT(strcmp(sess->lines[0], "AI > Привет мир очень длинная строка") == 0,
           "Reflow preserves full UTF-8 logical turn content");
    qmx_session_reflow(sess, 20);
    EXPECT(sess->line_count >= lines_narrow && sess->line_count >= 2,
           "Reflow to narrower viewport re-wraps turn cleanly");

    qmx_session_clear(sess);
    EXPECT(sess->line_count == 0 && sess->turn_count == 0 && sess->scroll_offset == 0, "Session clear resets history & turns");

    free(sess);

    if (g_failures == 0) {
        printf("[PASS] QmxSession (UTF-8 carry/wrap, scroll, backspace, prompt formatting) passed!\n");
        return 0;
    }
    fprintf(stderr, "[FAIL] test_session had %d failure(s)\n", g_failures);
    return 1;
}
