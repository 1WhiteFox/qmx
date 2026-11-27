#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 500
#endif

#include "tui.h"
#include "../../include/qmx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#if defined(_WIN32)
    #include <windows.h>
    #include <conio.h>
    static HANDLE g_hStdout = INVALID_HANDLE_VALUE;
    static HANDLE g_hStdin  = INVALID_HANDLE_VALUE;
    static UINT   g_orig_out_cp = 0;
    static UINT   g_orig_in_cp  = 0;
#elif defined(__DJGPP__)
    #include <conio.h>
    #include <dos.h>
    #include <sys/farptr.h>
    #include <go32.h>
    #define DOS_VRAM_ADDR 0xB8000UL
#else
    #include <termios.h>
    #include <unistd.h>
    #include <sys/select.h>
    #include <sys/ioctl.h>
    #include <signal.h>
    static struct termios g_orig_termios;
    static int g_termios_saved = 0;
    static int g_is_utf8 = 0;
    static volatile sig_atomic_t g_tui_resized = 0;
    static void tui_sigwinch_handler(int sig) {
        (void)sig;
        g_tui_resized = 1;
    }
#endif

#if defined(_WIN32) || defined(__DJGPP__)
static unsigned char g_key_utf8_queue[8];
static int g_key_utf8_head = 0;
static int g_key_utf8_tail = 0;

static void queue_utf8_codepoint(uint32_t cp) {
    if (cp < 0x80u) {
        g_key_utf8_queue[g_key_utf8_tail++] = (unsigned char)cp;
    } else if (cp < 0x800u) {
        g_key_utf8_queue[g_key_utf8_tail++] = (unsigned char)(0xC0u | (cp >> 6u));
        g_key_utf8_queue[g_key_utf8_tail++] = (unsigned char)(0x80u | (cp & 0x3Fu));
    } else if (cp < 0x10000u) {
        g_key_utf8_queue[g_key_utf8_tail++] = (unsigned char)(0xE0u | (cp >> 12u));
        g_key_utf8_queue[g_key_utf8_tail++] = (unsigned char)(0x80u | ((cp >> 6u) & 0x3Fu));
        g_key_utf8_queue[g_key_utf8_tail++] = (unsigned char)(0x80u | (cp & 0x3Fu));
    }
}

#if defined(__DJGPP__)
static uint32_t cp866_to_unicode(uint8_t b) {
    if (b < 0x80u) return b;
    if (b >= 0x80u && b <= 0xAFu) return 0x0410u + (uint32_t)(b - 0x80u);
    if (b >= 0xE0u && b <= 0xEFu) return 0x0440u + (uint32_t)(b - 0xE0u);
    if (b == 0xF0u) return 0x0401u;
    if (b == 0xF1u) return 0x0451u;
    if (b == 0xFCu) return 0x2116u;
    return 0u;
}
#endif
#endif

static const char * const g_strings[2][STR_TOTAL_COUNT] = {
    {
        "qmx-term | QMX Runtime " QMX_RUNTIME_VERSION_STR,
        "Backend",
        "Mode",
        "RAM RESIDENT",
        "DISK STREAM",
        "Arena",
        "KV",
        "[IO: IDLE]",
        "[IO: READ]",
        "Pinned",
        "Speed",
        "Latency",
        "You",
        "QMX",
        "Generating",
        "1Help",
        "2Back",
        "3KV",
        "4Strm",
        "5Rst",
        "6Ctx",
        "7Tmp",
        "8Lng",
        "9Cfg",
        "10Quit",
        "Welcome to qmx-term (" QMX_RUNTIME_VERSION_STR ")" "! Type a prompt and press Enter. (PgUp/PgDn: scroll)",
        "[Help] F2:Back F3:KV F4:Stream F5:Reset F6:Ctx F7:Temp F8:Lang F9:Cfg/Models",
        "[Context reset to pos=0]",
        "[Sliding Window: KV-cache shifted with RoPE phase correction]",
        "Settings & Cartridges (Arrows / Enter)"
    },
    {
        "qmx-term | QMX Runtime " QMX_RUNTIME_VERSION_STR,
        "Бэкенд",
        "Режим",
        "ОЗУ: РЕЗИДЕНТ",
        "ДИСК: СТРИМ",
        "Арена",
        "КВ",
        "[ДИСК: ПОКОЙ]",
        "[ДИСК: ЧТЕНИЕ]",
        "Закреп",
        "Скорость",
        "Задержка",
        "Вы",
        "QMX",
        "Генерация",
        "1Пом",
        "2Бэк",
        "3KV",
        "4Стр",
        "5Сбр",
        "6Кон",
        "7Тмп",
        "8Язк",
        "9Наст",
        "10Вых",
        "Добро пожаловать в qmx-term (" QMX_RUNTIME_VERSION_STR ")" "! Введите запрос и нажмите Enter.",
        "[Справка] F2:Бэк F3:KV F4:Стрим F5:Сброс F6:Конт F7:Темп F8:Язк F9:Настр/Мод",
        "[Контекст сброшен на pos=0]",
        "[Скользящее окно: KV-кэш сдвинут с фазовой коррекцией RoPE]",
        "Настройки и картриджи .qmx (Стрелки / Enter)"
    }
};

static int decode_utf8_cp(const unsigned char *s, uint32_t *out_cp) {
    if (*s < 0x80) {
        *out_cp = *s;
        return 1;
    } else if ((*s & 0xE0) == 0xC0) {
        if ((s[1] & 0xC0) == 0x80) {
            *out_cp = ((uint32_t)(s[0] & 0x1F) << 6) | (uint32_t)(s[1] & 0x3F);
            return 2;
        }
        *out_cp = *s;
        return 1;
    } else if ((*s & 0xF0) == 0xE0) {
        if ((s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80) {
            *out_cp = ((uint32_t)(s[0] & 0x0F) << 12) |
                      ((uint32_t)(s[1] & 0x3F) << 6)  |
                      (uint32_t)(s[2] & 0x3F);
            return 3;
        }
        *out_cp = *s;
        return 1;
    } else if ((*s & 0xF8) == 0xF0) {
        if ((s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80 && (s[3] & 0xC0) == 0x80) {
            *out_cp = ((uint32_t)(s[0] & 0x07) << 18) |
                      ((uint32_t)(s[1] & 0x3F) << 12) |
                      ((uint32_t)(s[2] & 0x3F) << 6)  |
                      (uint32_t)(s[3] & 0x3F);
            return 4;
        }
        *out_cp = *s;
        return 1;
    }
    *out_cp = *s;
    return 1;
}

int tui_utf8_cols(const char *s) {
    if (!s) return 0;
    int count = 0;
    const unsigned char *p = (const unsigned char *)s;
    while (*p) {
        uint32_t cp = 0;
        int len = decode_utf8_cp(p, &cp);
        if (cp == 0x2026) count += 3;
        else count += 1;
        p += len;
    }
    return count;
}

#if TUI_RETRO_FIXED || (!defined(_WIN32) && !defined(__DJGPP__))
static uint8_t unicode_to_cp866(uint32_t cp) {
    if (cp < 0x80u) return (uint8_t)cp;
    if (cp >= 0x0410u && cp <= 0x043Fu) return (uint8_t)(0x80u + (cp - 0x0410u));
    if (cp >= 0x0440u && cp <= 0x044Fu) return (uint8_t)(0xE0u + (cp - 0x0440u));
    switch (cp) {
        case 0x0401u: return 0xF0u;
        case 0x0451u: return 0xF1u;
        case 0x2116u: return 0xFCu;
        case 0x2500u: return 0xC4u;
        case 0x2502u: return 0xB3u;
        case 0x250Cu: return 0xDAu;
        case 0x2510u: return 0xBFu;
        case 0x2514u: return 0xC0u;
        case 0x2518u: return 0xD9u;
        case 0x251Cu: return 0xC3u;
        case 0x2524u: return 0xB4u;
        case 0x252Cu: return 0xC2u;
        case 0x2534u: return 0xC1u;
        case 0x253Cu: return 0xC5u;
        case 0x2588u: return 0xDBu;
        case 0x2591u: return 0xB0u;
        case 0x2592u: return 0xB1u;
        case 0x2593u: return 0xB2u;
        default: break;
    }
    if (cp == 0x00A0u || cp == 0x202Fu || cp == 0x2009u) return ' ';
    if (cp == 0x2018u || cp == 0x2019u || cp == 0x201Bu || cp == 0x00B4u || cp == 0x02BCu) return '\'';
    if (cp == 0x201Cu || cp == 0x201Du || cp == 0x201Eu || cp == 0x201Fu || cp == 0x00ABu || cp == 0x00BBu) return '"';
    if (cp == 0x2010u || cp == 0x2011u || cp == 0x2012u || cp == 0x2013u || cp == 0x2014u || cp == 0x2015u || cp == 0x2212u) return '-';
    if (cp == 0x2022u || cp == 0x2023u || cp == 0x25AAu || cp == 0x25CFu) return '*';
    return '?';
}
#endif

#if !TUI_RETRO_FIXED
static void tui_query_term_size(int *out_cols, int *out_rows) {
    int c = TUI_COLS;
    int r = TUI_ROWS;
#if defined(_WIN32)
    if (g_hStdout != INVALID_HANDLE_VALUE) {
        CONSOLE_SCREEN_BUFFER_INFO csbi;
        if (GetConsoleScreenBufferInfo(g_hStdout, &csbi)) {
            int w = (int)(csbi.srWindow.Right - csbi.srWindow.Left + 1);
            int h = (int)(csbi.srWindow.Bottom - csbi.srWindow.Top + 1);
            if (w >= TUI_MIN_COLS) c = w;
            if (h >= TUI_MIN_ROWS) r = h;
        }
    }
#else
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0) {
        if (ws.ws_col >= TUI_MIN_COLS) c = (int)ws.ws_col;
        if (ws.ws_row >= TUI_MIN_ROWS) r = (int)ws.ws_row;
    }
#endif
    if (c < TUI_MIN_COLS) c = TUI_MIN_COLS;
    if (c > TUI_MAX_COLS) c = TUI_MAX_COLS;
    if (r < TUI_MIN_ROWS) r = TUI_MIN_ROWS;
    if (r > TUI_MAX_ROWS) r = TUI_MAX_ROWS;
    *out_cols = c;
    *out_rows = r;
}

static void tui_resize_buffers(TuiEngine *tui, int cols, int rows) {
    int need = cols * rows;
    if (need > tui->buf_cap) {
        TuiCell *nb = (TuiCell *)realloc(tui->backbuffer, (size_t)need * sizeof(TuiCell));
        TuiCell *nf = (TuiCell *)realloc(tui->frontbuffer, (size_t)need * sizeof(TuiCell));
        if (!nb || !nf) {
            if (nb) tui->backbuffer = nb;
            if (nf) tui->frontbuffer = nf;
            return;
        }
        tui->backbuffer  = nb;
        tui->frontbuffer = nf;
        tui->buf_cap     = need;
    }
    tui->cols = cols;
    tui->rows = rows;
    for (int i = 0; i < need; i++) {
        tui->frontbuffer[i].ch   = 0xFFFFFFFFu;
        tui->frontbuffer[i].attr = 0xFFu;
    }
}
#endif

const char *tui_str(TuiEngine *tui, TuiStringId id) {
    if (id < 0 || id >= STR_TOTAL_COUNT) return "";
    return g_strings[tui->lang][id];
}

void tui_toggle_lang(TuiEngine *tui) {
    tui->lang = (tui->lang == TUI_LANG_EN) ? TUI_LANG_RU : TUI_LANG_EN;
}

void tui_init(TuiEngine *tui, TuiLang default_lang) {
    memset(tui, 0, sizeof(TuiEngine));
    tui->lang = default_lang;
    tui->rows = TUI_ROWS;
    tui->cols = TUI_COLS;
#if defined(_WIN32)
    g_hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
    g_hStdin  = GetStdHandle(STD_INPUT_HANDLE);
    g_orig_out_cp = GetConsoleOutputCP();
    g_orig_in_cp  = GetConsoleCP();
#if TUI_RETRO_FIXED
    SetConsoleOutputCP(866);
    SetConsoleCP(866);
#endif
    g_key_utf8_head = 0;
    g_key_utf8_tail = 0;
    CONSOLE_CURSOR_INFO cci;
    cci.dwSize = 25;
    cci.bVisible = FALSE;
    SetConsoleCursorInfo(g_hStdout, &cci);
#elif defined(__DJGPP__)
    g_key_utf8_head = 0;
    g_key_utf8_tail = 0;
    _setcursortype(_NOCURSOR);
#else
    if (!g_termios_saved) {
        tcgetattr(STDIN_FILENO, &g_orig_termios);
        g_termios_saved = 1;
    }
    struct termios raw = g_orig_termios;
    raw.c_lflag &= ~(tcflag_t)(ECHO | ICANON);
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
    const char *lang = getenv("LC_ALL");
    if (!lang) lang = getenv("LC_CTYPE");
    if (!lang) lang = getenv("LANG");
    if (lang && (strstr(lang, "UTF-8") || strstr(lang, "utf-8") || strstr(lang, "UTF8") || strstr(lang, "utf8"))) {
        g_is_utf8 = 1;
    } else {
        g_is_utf8 = 0;
    }
    g_tui_resized = 0;
    signal(SIGWINCH, tui_sigwinch_handler);
    fputs("\033[?1049h\033[?25l\033[2J", stdout);
    fflush(stdout);
#endif
#if !TUI_RETRO_FIXED
    int c = TUI_COLS, r = TUI_ROWS;
    tui_query_term_size(&c, &r);
    tui_resize_buffers(tui, c, r);
#endif
    tui_clear(tui, TUI_ATTR_NORMAL);
}

void tui_shutdown(TuiEngine *tui) {
#if !TUI_RETRO_FIXED
    if (tui) {
        free(tui->backbuffer);
        free(tui->frontbuffer);
        tui->backbuffer  = NULL;
        tui->frontbuffer = NULL;
        tui->buf_cap     = 0;
    }
#else
    (void)tui;
#endif
#if defined(_WIN32)
    if (g_orig_out_cp != 0) SetConsoleOutputCP(g_orig_out_cp);
    if (g_orig_in_cp != 0)  SetConsoleCP(g_orig_in_cp);
    if (g_hStdout != INVALID_HANDLE_VALUE) {
        CONSOLE_CURSOR_INFO cci;
        cci.dwSize = 25;
        cci.bVisible = TRUE;
        SetConsoleCursorInfo(g_hStdout, &cci);
    }
#elif defined(__DJGPP__)
    _setcursortype(_NORMALCURSOR);
#else
    signal(SIGWINCH, SIG_DFL);
    if (g_termios_saved) {
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &g_orig_termios);
        g_termios_saved = 0;
    }
    fputs("\033[0m\033[?1049l\033[?25h", stdout);
    fflush(stdout);
#endif
}

int tui_check_resize(TuiEngine *tui) {
#if TUI_RETRO_FIXED
    (void)tui;
    return 0;
#elif defined(_WIN32)
    if (!tui) return 0;
    int c = tui->cols, r = tui->rows;
    tui_query_term_size(&c, &r);
    if (c != tui->cols || r != tui->rows) {
        tui_resize_buffers(tui, c, r);
        return 1;
    }
    return 0;
#else
    if (!tui || !g_tui_resized) return 0;
    g_tui_resized = 0;
    int c = tui->cols, r = tui->rows;
    tui_query_term_size(&c, &r);
    if (c != tui->cols || r != tui->rows) {
        tui_resize_buffers(tui, c, r);
        fputs("\033[2J", stdout);
        fflush(stdout);
        return 1;
    }
    return 0;
#endif
}

void tui_put_cell(TuiEngine *tui, int x, int y, uint32_t cp, uint8_t attr) {
    if (!tui || x < 0 || y < 0 || x >= tui->cols || y >= tui->rows) return;
#if !TUI_RETRO_FIXED
    if (!tui->backbuffer) return;
    int idx = y * tui->cols + x;
    tui->backbuffer[idx].ch   = cp;
    tui->backbuffer[idx].attr = attr;
#else
    int idx = y * TUI_COLS + x;
    tui->backbuffer[idx].ch   = unicode_to_cp866(cp);
    tui->backbuffer[idx].attr = attr;
#endif
}

void tui_clear(TuiEngine *tui, uint8_t attr) {
    if (!tui) return;
#if !TUI_RETRO_FIXED
    if (!tui->backbuffer) return;
#endif
    int total = tui->rows * tui->cols;
    for (int i = 0; i < total; i++) {
        tui->backbuffer[i].ch   = ' ';
        tui->backbuffer[i].attr = attr;
    }
}

void tui_print(TuiEngine *tui, int x, int y, uint8_t attr, const char *str) {
    if (!tui || y < 0 || y >= tui->rows || !str) return;
    int cur_x = x;
    const unsigned char *s = (const unsigned char *)str;
    while (*s && cur_x < tui->cols) {
        uint32_t cp = 0;
        int len = decode_utf8_cp(s, &cp);
        if (cp == 0x2026u) {
            for (int d = 0; d < 3 && cur_x < tui->cols; d++) {
                if (cur_x >= 0) tui_put_cell(tui, cur_x, y, '.', attr);
                cur_x++;
            }
        } else {
            if (cur_x >= 0) tui_put_cell(tui, cur_x, y, cp, attr);
            cur_x++;
        }
        s += len;
    }
}

void tui_printf(TuiEngine *tui, int x, int y, uint8_t attr, const char *fmt, ...) {
    char buf[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    tui_print(tui, x, y, attr, buf);
}

void tui_draw_box(TuiEngine *tui, int x, int y, int w, int h, uint8_t attr, const char *title) {
    if (!tui || x < 0 || y < 0 || x + w > tui->cols || y + h > tui->rows || w < 2 || h < 2) return;
    tui_put_cell(tui, x, y, 0x250Cu, attr);
    tui_put_cell(tui, x + w - 1, y, 0x2510u, attr);
    tui_put_cell(tui, x, y + h - 1, 0x2514u, attr);
    tui_put_cell(tui, x + w - 1, y + h - 1, 0x2518u, attr);
    for (int i = 1; i < w - 1; i++) {
        tui_put_cell(tui, x + i, y, 0x2500u, attr);
        tui_put_cell(tui, x + i, y + h - 1, 0x2500u, attr);
    }
    for (int j = 1; j < h - 1; j++) {
        tui_put_cell(tui, x, y + j, 0x2502u, attr);
        tui_put_cell(tui, x + w - 1, y + j, 0x2502u, attr);
    }
    if (title && title[0] != '\0') {
        int char_count = tui_utf8_cols(title);
        if (char_count + 4 < w) {
            int tx = x + 2;
            tui_put_cell(tui, tx - 1, y, ' ', attr);
            tui_print(tui, tx, y, TUI_ATTR_TITLE, title);
            tui_put_cell(tui, tx + char_count, y, ' ', attr);
        }
    }
}

void tui_draw_bar(TuiEngine *tui, int x, int y, int width, int current, int max, uint8_t fill_attr, uint8_t empty_attr) {
    if (!tui || width <= 0 || max <= 0) return;
    int filled = (current * width) / max;
    if (filled > width) filled = width;
    for (int i = 0; i < width; i++) {
        if (i < filled) {
            tui_put_cell(tui, x + i, y, 0x2588u, fill_attr);
        } else {
            tui_put_cell(tui, x + i, y, 0x2591u, empty_attr);
        }
    }
}

#if !defined(_WIN32) && !defined(__DJGPP__)
static void tui_emit_utf8_cp(uint32_t cp) {
    if (cp < 0x80u) {
        putchar((int)cp);
    } else if (cp < 0x800u) {
        putchar((int)(0xC0u | (cp >> 6u)));
        putchar((int)(0x80u | (cp & 0x3Fu)));
    } else if (cp < 0x10000u) {
        putchar((int)(0xE0u | (cp >> 12u)));
        putchar((int)(0x80u | ((cp >> 6u) & 0x3Fu)));
        putchar((int)(0x80u | (cp & 0x3Fu)));
    } else if (cp <= 0x10FFFFu) {
        putchar((int)(0xF0u | (cp >> 18u)));
        putchar((int)(0x80u | ((cp >> 12u) & 0x3Fu)));
        putchar((int)(0x80u | ((cp >> 6u) & 0x3Fu)));
        putchar((int)(0x80u | (cp & 0x3Fu)));
    } else {
        putchar('?');
    }
}
#endif

void tui_flip(TuiEngine *tui) {
    if (!tui) return;
#if defined(_WIN32)
    #if TUI_RETRO_FIXED
    CHAR_INFO win_buf[TUI_ROWS * TUI_COLS];
    COORD buf_size = { TUI_COLS, TUI_ROWS };
    COORD buf_coord = { 0, 0 };
    SMALL_RECT write_region = { 0, 0, TUI_COLS - 1, TUI_ROWS - 1 };
    int changed = 0;
    for (int i = 0; i < TUI_ROWS * TUI_COLS; i++) {
        if (tui->backbuffer[i].ch != tui->frontbuffer[i].ch ||
            tui->backbuffer[i].attr != tui->frontbuffer[i].attr) {
            changed = 1;
        }
        win_buf[i].Char.AsciiChar = (char)tui->backbuffer[i].ch;
        win_buf[i].Attributes     = tui->backbuffer[i].attr;
        tui->frontbuffer[i]       = tui->backbuffer[i];
    }
    if (changed && g_hStdout != INVALID_HANDLE_VALUE) {
        WriteConsoleOutputA(g_hStdout, win_buf, buf_size, buf_coord, &write_region);
    }
    #else
    if (!tui->backbuffer || !tui->frontbuffer) return;
    int total = tui->rows * tui->cols;
    int changed = 0;
    for (int i = 0; i < total; i++) {
        if (tui->backbuffer[i].ch != tui->frontbuffer[i].ch ||
            tui->backbuffer[i].attr != tui->frontbuffer[i].attr) {
            changed = 1;
            break;
        }
    }
    if (!changed || g_hStdout == INVALID_HANDLE_VALUE) return;
    CHAR_INFO *win_buf = (CHAR_INFO *)malloc((size_t)total * sizeof(CHAR_INFO));
    if (!win_buf) return;
    for (int i = 0; i < total; i++) {
        uint32_t cp = tui->backbuffer[i].ch;
        win_buf[i].Char.UnicodeChar = (WCHAR)((cp <= 0xFFFFu) ? cp : '?');
        win_buf[i].Attributes       = tui->backbuffer[i].attr;
        tui->frontbuffer[i]         = tui->backbuffer[i];
    }
    COORD buf_size = { (SHORT)tui->cols, (SHORT)tui->rows };
    COORD buf_coord = { 0, 0 };
    SMALL_RECT write_region = { 0, 0, (SHORT)(tui->cols - 1), (SHORT)(tui->rows - 1) };
    WriteConsoleOutputW(g_hStdout, win_buf, buf_size, buf_coord, &write_region);
    free(win_buf);
    #endif
#elif defined(__DJGPP__)
    for (int y = 0; y < TUI_ROWS; y++) {
        for (int x = 0; x < TUI_COLS; x++) {
            int idx = y * TUI_COLS + x;
            if (tui->backbuffer[idx].ch != tui->frontbuffer[idx].ch ||
                tui->backbuffer[idx].attr != tui->frontbuffer[idx].attr) {
                int off = idx * 2;
                _farpokeb(_dos_ds, DOS_VRAM_ADDR + (unsigned long)off, tui->backbuffer[idx].ch);
                _farpokeb(_dos_ds, DOS_VRAM_ADDR + (unsigned long)off + 1UL, tui->backbuffer[idx].attr);
                tui->frontbuffer[idx] = tui->backbuffer[idx];
            }
        }
    }
#else
    if (!tui->backbuffer || !tui->frontbuffer) return;
    for (int y = 0; y < tui->rows; y++) {
        for (int x = 0; x < tui->cols; x++) {
            int idx = y * tui->cols + x;
            if (tui->backbuffer[idx].ch != tui->frontbuffer[idx].ch ||
                tui->backbuffer[idx].attr != tui->frontbuffer[idx].attr) {
                uint8_t attr = tui->backbuffer[idx].attr;
                uint8_t fg = attr & 0x0Fu;
                uint8_t bg = (attr >> 4) & 0x07u;
                printf("\033[%d;%dH\033[%d;%dm", y + 1, x + 1,
                       30 + (fg % 8) + (fg >= 8 ? 60 : 0),
                       40 + bg);
                if (g_is_utf8) {
                    tui_emit_utf8_cp(tui->backbuffer[idx].ch);
                } else {
                    putchar(unicode_to_cp866(tui->backbuffer[idx].ch));
                }
                tui->frontbuffer[idx] = tui->backbuffer[idx];
            }
        }
    }
    fflush(stdout);
#endif
}

int tui_poll_key(void) {
#if defined(_WIN32)
    if (g_key_utf8_head < g_key_utf8_tail) {
        int b = g_key_utf8_queue[g_key_utf8_head++];
        if (g_key_utf8_head >= g_key_utf8_tail) {
            g_key_utf8_head = 0;
            g_key_utf8_tail = 0;
        }
        return b;
    }
    if (g_hStdin == INVALID_HANDLE_VALUE) {
        g_hStdin = GetStdHandle(STD_INPUT_HANDLE);
    }
    DWORD events = 0;
    while (GetNumberOfConsoleInputEvents(g_hStdin, &events) && events > 0) {
        INPUT_RECORD rec;
        DWORD read_cnt = 0;
        if (!ReadConsoleInputW(g_hStdin, &rec, 1, &read_cnt) || read_cnt == 0) break;
        if (rec.EventType != KEY_EVENT || !rec.Event.KeyEvent.bKeyDown) continue;

        WORD vk = rec.Event.KeyEvent.wVirtualKeyCode;
        WCHAR wc = rec.Event.KeyEvent.uChar.UnicodeChar;
        if (vk >= VK_F1 && vk <= VK_F10) return TUI_KEY_F1 + (int)(vk - VK_F1);
        if (vk == VK_UP)    return TUI_KEY_UP;
        if (vk == VK_DOWN)  return TUI_KEY_DOWN;
        if (vk == VK_LEFT)  return TUI_KEY_LEFT;
        if (vk == VK_RIGHT) return TUI_KEY_RIGHT;
        if (vk == VK_PRIOR) return TUI_KEY_PGUP;
        if (vk == VK_NEXT)  return TUI_KEY_PGDN;
        if (vk == VK_ESCAPE) return TUI_KEY_ESC;
        if (vk == VK_BACK)   return '\b';
        if (vk == VK_RETURN) return '\r';

        if (wc >= 32) {
            queue_utf8_codepoint((uint32_t)wc);
            if (g_key_utf8_head < g_key_utf8_tail) {
                int b = g_key_utf8_queue[g_key_utf8_head++];
                if (g_key_utf8_head >= g_key_utf8_tail) {
                    g_key_utf8_head = 0;
                    g_key_utf8_tail = 0;
                }
                return b;
            }
        }
    }
    return -1;
#elif defined(__DJGPP__)
    if (g_key_utf8_head < g_key_utf8_tail) {
        int b = g_key_utf8_queue[g_key_utf8_head++];
        if (g_key_utf8_head >= g_key_utf8_tail) {
            g_key_utf8_head = 0;
            g_key_utf8_tail = 0;
        }
        return b;
    }
    if (kbhit()) {
        int c = getch();
        if (c == 0 || (c == 0xE0 && kbhit())) {
            return 0x100 | getch();
        }
        if (c >= 0x80 && c <= 0xFF) {
            uint32_t cp = cp866_to_unicode((uint8_t)c);
            if (cp > 0) {
                queue_utf8_codepoint(cp);
                int b = g_key_utf8_queue[g_key_utf8_head++];
                if (g_key_utf8_head >= g_key_utf8_tail) {
                    g_key_utf8_head = 0;
                    g_key_utf8_tail = 0;
                }
                return b;
            }
        }
        return c;
    }
    return -1;
#else
    struct timeval tv = { 0, 0 };
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(STDIN_FILENO, &fds);
    if (select(STDIN_FILENO + 1, &fds, NULL, NULL, &tv) > 0) {
        unsigned char ch;
        if (read(STDIN_FILENO, &ch, 1) == 1) {
            if (ch == 27) {
                struct timeval esc_tv = { 0, 25000 };
                fd_set esc_fds;
                FD_ZERO(&esc_fds);
                FD_SET(STDIN_FILENO, &esc_fds);
                if (select(STDIN_FILENO + 1, &esc_fds, NULL, NULL, &esc_tv) > 0) {
                    unsigned char seq[8];
                    int n = (int)read(STDIN_FILENO, seq, sizeof(seq) - 1);
                    if (n > 0) {
                        seq[n] = '\0';
                        if (seq[0] == 'O') {
                            if (seq[1] == 'P') return TUI_KEY_F1;
                            if (seq[1] == 'Q') return TUI_KEY_F2;
                            if (seq[1] == 'R') return TUI_KEY_F3;
                            if (seq[1] == 'S') return TUI_KEY_F4;
                        } else if (seq[0] == '[') {
                            if (seq[1] == 'A') return TUI_KEY_UP;
                            if (seq[1] == 'B') return TUI_KEY_DOWN;
                            if (seq[1] == 'C') return TUI_KEY_RIGHT;
                            if (seq[1] == 'D') return TUI_KEY_LEFT;
                            if (strncmp((char *)seq + 1, "5~", 2) == 0) return TUI_KEY_PGUP;
                            if (strncmp((char *)seq + 1, "6~", 2) == 0) return TUI_KEY_PGDN;
                            if (seq[1] == '[') {
                                if (seq[2] == 'A') return TUI_KEY_F1;
                                if (seq[2] == 'B') return TUI_KEY_F2;
                                if (seq[2] == 'C') return TUI_KEY_F3;
                                if (seq[2] == 'D') return TUI_KEY_F4;
                                if (seq[2] == 'E') return TUI_KEY_F5;
                            } else if (strncmp((char *)seq + 1, "11~", 3) == 0) {
                                return TUI_KEY_F1;
                            } else if (strncmp((char *)seq + 1, "12~", 3) == 0) {
                                return TUI_KEY_F2;
                            } else if (strncmp((char *)seq + 1, "13~", 3) == 0) {
                                return TUI_KEY_F3;
                            } else if (strncmp((char *)seq + 1, "14~", 3) == 0) {
                                return TUI_KEY_F4;
                            } else if (strncmp((char *)seq + 1, "15~", 3) == 0) {
                                return TUI_KEY_F5;
                            } else if (strncmp((char *)seq + 1, "17~", 3) == 0) {
                                return TUI_KEY_F6;
                            } else if (strncmp((char *)seq + 1, "18~", 3) == 0) {
                                return TUI_KEY_F7;
                            } else if (strncmp((char *)seq + 1, "19~", 3) == 0) {
                                return TUI_KEY_F8;
                            } else if (strncmp((char *)seq + 1, "20~", 3) == 0) {
                                return TUI_KEY_F9;
                            } else if (strncmp((char *)seq + 1, "21~", 3) == 0) {
                                return TUI_KEY_F10;
                            }
                        }
                    }
                }
            }
            return (int)ch;
        }
    }
    return -1;
#endif
}
