.RECIPEPREFIX = >
CC         ?= gcc
AR         ?= ar
WIN32_CC   ?= i686-w64-mingw32-gcc
WIN32_OBJD ?= i686-w64-mingw32-objdump
WIN64_CC   ?= x86_64-w64-mingw32-gcc
DOS_CC     ?= i586-pc-msdosdjgpp-gcc
DOS_OBJD   ?= i586-pc-msdosdjgpp-objdump

CORE_SRCS   = src/qmx_ops.c src/qmx_thread.c src/qmx_dispatch.c src/qmx_model.c src/qmx_vm.c src/qmx_tokenizer.c src/qmx_sampler.c src/qmx_gen.c src/qmx_autotune.c src/qmx_plugin.c
KERNEL_SRCS = src/kernels_p54c.c src/kernels_mmx.c src/kernels_3dnow.c src/kernels_sse2.c src/kernels_avx2.c src/kernels_x86_ext.c src/kernels_neon.c src/kernels_arm_dsp.c src/kernels_mcu32.c src/kernels_cortex_m.c
APP_COMMON  = apps/common/qmx_app_common.c apps/common/qmx_profiles.c apps/common/qmx_session.c
CLI_SRCS    = $(APP_COMMON) apps/qmx-cli/main.c
TERM_SRCS   = $(APP_COMMON) apps/qmx-term/tui.c apps/qmx-term/main.c
PPL_SRCS    = $(APP_COMMON) apps/qmx-ppl/main.c
RETRO_SRCS  = $(CORE_SRCS) $(TERM_SRCS)

HEADERS     = $(wildcard include/*.h)
CORE_OBJS   = $(patsubst src/%.c,build/host/%.o,$(CORE_SRCS))
KERNEL_OBJS = $(patsubst src/%.c,build/host/%.o,$(KERNEL_SRCS))
LIBQMX      = build/host/libqmx.a
SAN_OBJS    = $(patsubst src/%.c,build/san/%.o,$(CORE_SRCS) $(KERNEL_SRCS))
LIBQMX_SAN  = build/san/libqmx_san.a
WIN64_OBJS  = $(patsubst src/%.c,build/win64/%.o,$(CORE_SRCS) $(KERNEL_SRCS))

CFLAGS_STD  = -std=c99 -Wall -Wextra -Iinclude -fno-builtin
HOST_ARCH_FLAGS ?= $(if $(filter x86_64,$(shell uname -m 2>/dev/null)),-march=x86-64,)
HOST_FLAGS  = -DQMX_USE_MMAP -DQMX_ENABLE_THREADS -DQMX_ENABLE_DYNAMIC_PLUGINS -pthread
HOST_LIBS   = -ldl

RETRO_486_FLAGS       = -march=i486 -mtune=i486 -mno-mmx -mno-3dnow -mno-sse -mno-sse2 -DQMX_CUSTOM_BACKEND_SET -DQMX_ENABLE_BACKEND_P54C
RETRO_DOS_MULTI_FLAGS = -march=i486 -mtune=pentium -mno-sse -mno-sse2 -DQMX_DOS_MULTI_ISA -DQMX_CUSTOM_BACKEND_SET -DQMX_ENABLE_BACKEND_P54C -DQMX_ENABLE_BACKEND_MMX -DQMX_ENABLE_BACKEND_3DNOW
RETRO_P54C_FLAGS      = -march=pentium -mtune=pentium -mno-mmx -mno-3dnow -mno-sse -mno-sse2 -DQMX_CUSTOM_BACKEND_SET -DQMX_ENABLE_BACKEND_P54C
RETRO_MMX_FLAGS   = -march=pentium-mmx -mtune=pentium-mmx -mmmx -mno-3dnow -mno-sse -mno-sse2 -DQMX_CUSTOM_BACKEND_SET -DQMX_ENABLE_BACKEND_P54C -DQMX_ENABLE_BACKEND_MMX
RETRO_3DNOW_FLAGS = -march=k6-2 -mtune=k6-2 -m3dnow -mno-sse -mno-sse2 -DQMX_CUSTOM_BACKEND_SET -DQMX_ENABLE_BACKEND_P54C -DQMX_ENABLE_BACKEND_MMX -DQMX_ENABLE_BACKEND_3DNOW
WIN98_DEFS        = -D__USE_MINGW_ANSI_STDIO=0 -DQMX_WIN98_COMPAT
SAN_FLAGS         = -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer -g -O1

.PHONY: all libqmx host cli term ppl plugins amalgamate test-amalgamate check-m32 x86-p54c x86-mmx k6-3dnow win98 win98-p54c win98-mmx win98-3dnow win64 dos dos-486 dos-mmx dos-3dnow dos-multi test test-san fuzz test-mcu-nomalloc test-isa retro-dist tui clean

all: host

build/host:
> mkdir -p build/host

build/host/qmx_%.o: src/qmx_%.c $(HEADERS) | build/host
> $(CC) $(HOST_ARCH_FLAGS) $(CFLAGS_STD) $(HOST_FLAGS) -O3 -c $< -o $@

build/host/kernels_%.o: src/kernels_%.c $(HEADERS) | build/host
> $(CC) $(HOST_ARCH_FLAGS) $(CFLAGS_STD) -O3 -c $< -o $@

build/san:
> mkdir -p build/san

build/san/qmx_%.o: src/qmx_%.c $(HEADERS) | build/san
> $(CC) $(HOST_ARCH_FLAGS) $(CFLAGS_STD) $(HOST_FLAGS) $(SAN_FLAGS) -c $< -o $@

build/san/kernels_%.o: src/kernels_%.c $(HEADERS) | build/san
> $(CC) $(HOST_ARCH_FLAGS) $(CFLAGS_STD) $(SAN_FLAGS) -c $< -o $@

$(LIBQMX_SAN): $(SAN_OBJS)
> rm -f $@
> $(AR) rcs $@ $(SAN_OBJS)

$(LIBQMX): $(CORE_OBJS) $(KERNEL_OBJS)
> rm -f $@
> $(AR) rcs $@ $(CORE_OBJS) $(KERNEL_OBJS)

libqmx: $(LIBQMX)

cli: $(LIBQMX) $(CLI_SRCS)
> $(CC) $(CFLAGS_STD) $(HOST_FLAGS) -O3 $(CLI_SRCS) $(LIBQMX) $(HOST_LIBS) -o qmx

term: $(LIBQMX) $(TERM_SRCS)
> $(CC) $(CFLAGS_STD) $(HOST_FLAGS) -O3 $(TERM_SRCS) $(LIBQMX) $(HOST_LIBS) -o qmx-term

ppl: $(LIBQMX) $(PPL_SRCS)
> $(CC) $(CFLAGS_STD) $(HOST_FLAGS) -O3 $(PPL_SRCS) $(LIBQMX) -lm $(HOST_LIBS) -o qmx-ppl

plugins: $(HEADERS)
> mkdir -p build/plugins
> $(CC) $(CFLAGS_STD) -O3 -fPIC -shared -DQMX_PLUGIN_SHARED plugins/plugin_layer_profiler.c -o build/plugins/libqmx_profiler.so
> $(CC) $(CFLAGS_STD) -O3 -fPIC -shared -DQMX_PLUGIN_SHARED plugins/plugin_token_ban.c -o build/plugins/libqmx_token_ban.so

host: cli term ppl

check-m32:
> @printf "#include <stdint.h>\nint main(void){return 0;}\n" | $(CC) -m32 -x c - -o /dev/null 2>/dev/null || \
  (echo "[SKIP/ERROR] 32-bit headers missing for 'gcc -m32'. Run: sudo apt install gcc-multilib" && exit 1)

x86-p54c: check-m32 $(RETRO_SRCS) src/kernels_p54c.c
> $(CC) $(CFLAGS_STD) -m32 $(RETRO_P54C_FLAGS) -O3 $(RETRO_SRCS) src/kernels_p54c.c -o qmx_p54c
> @! objdump -d qmx_p54c | grep -E '\bcmov|\bfcomi|\bfucomi|\bmm[0-7]|\bxmm|\bymm' || (echo "[ERROR] Forbidden opcode found!" && exit 1)
> @echo "[OK] qmx_p54c: 100% pure Pentium Classic P54C"

x86-mmx: check-m32 $(RETRO_SRCS) src/kernels_p54c.c src/kernels_mmx.c
> $(CC) $(CFLAGS_STD) -m32 $(RETRO_MMX_FLAGS) -O3 $(RETRO_SRCS) src/kernels_p54c.c src/kernels_mmx.c -o qmx_mmx
> @! objdump -d qmx_mmx | grep -E '\bcmov|\bfcomi|\bfucomi|\bxmm|\bymm' || (echo "[ERROR] Forbidden opcode found!" && exit 1)
> @echo "[OK] qmx_mmx: 100% pure Pentium MMX"

k6-3dnow: check-m32 $(RETRO_SRCS) src/kernels_p54c.c src/kernels_mmx.c src/kernels_3dnow.c
> $(CC) $(CFLAGS_STD) -m32 $(RETRO_3DNOW_FLAGS) -O3 $(RETRO_SRCS) src/kernels_p54c.c src/kernels_mmx.c src/kernels_3dnow.c -o qmx_3dnow
> @! objdump -d qmx_3dnow | grep -E '\bcmov|\bfcomi|\bfucomi|\bxmm|\bymm' || (echo "[ERROR] Forbidden opcode found!" && exit 1)
> @objdump -d qmx_3dnow | grep -qE '\bpfmul\b' && objdump -d qmx_3dnow | grep -qE '\bpi2fd\b' || (echo "[ERROR] Missing 3DNow! opcodes in qmx_3dnow!" && exit 1)
> @echo "[OK] qmx_3dnow: 100% pure AMD K6-2 3DNow!"

win98: win98-p54c win98-mmx win98-3dnow

win98-p54c: $(RETRO_SRCS) src/kernels_p54c.c
> $(WIN32_CC) $(CFLAGS_STD) $(WIN98_DEFS) $(RETRO_P54C_FLAGS) -O3 -s $(RETRO_SRCS) src/kernels_p54c.c -o qmx_p54c.exe
> @python3 tools/sanitize_i586_pe.py qmx_p54c.exe "$(WIN32_OBJD)"

win98-mmx: $(RETRO_SRCS) src/kernels_p54c.c src/kernels_mmx.c
> $(WIN32_CC) $(CFLAGS_STD) $(WIN98_DEFS) $(RETRO_MMX_FLAGS) -O3 -s $(RETRO_SRCS) src/kernels_p54c.c src/kernels_mmx.c -o qmx_mmx.exe
> @python3 tools/sanitize_i586_pe.py qmx_mmx.exe "$(WIN32_OBJD)"

win98-3dnow: $(RETRO_SRCS) src/kernels_p54c.c src/kernels_mmx.c src/kernels_3dnow.c
> $(WIN32_CC) $(CFLAGS_STD) $(WIN98_DEFS) $(RETRO_3DNOW_FLAGS) -O3 -s $(RETRO_SRCS) src/kernels_p54c.c src/kernels_mmx.c src/kernels_3dnow.c -o qmx_3dnow.exe
> @python3 tools/sanitize_i586_pe.py qmx_3dnow.exe "$(WIN32_OBJD)"

win64:
> @if ! command -v $(WIN64_CC) >/dev/null 2>&1; then \
    echo "[SKIP] '$(WIN64_CC)' not installed (install: sudo apt install gcc-mingw-w64-x86-64)."; \
  else \
    mkdir -p build/win64 && \
    for s in $(CORE_SRCS); do \
      base=$$(basename $$s .c); \
      $(WIN64_CC) $(CFLAGS_STD) -DQMX_ENABLE_THREADS -DQMX_ENABLE_DYNAMIC_PLUGINS -O3 -c $$s -o build/win64/$$base.o || exit 1; \
    done && \
    for s in $(KERNEL_SRCS); do \
      base=$$(basename $$s .c); \
      $(WIN64_CC) $(CFLAGS_STD) -O3 -c $$s -o build/win64/$$base.o || exit 1; \
    done && \
    $(WIN64_CC) $(CFLAGS_STD) -DQMX_ENABLE_THREADS -DQMX_ENABLE_DYNAMIC_PLUGINS -O3 $(CLI_SRCS) $(WIN64_OBJS) -o qmx_win64.exe && \
    $(WIN64_CC) $(CFLAGS_STD) -DQMX_ENABLE_THREADS -DQMX_ENABLE_DYNAMIC_PLUGINS -O3 $(TERM_SRCS) $(WIN64_OBJS) -o qmx_term_win64.exe && \
    $(WIN64_CC) $(CFLAGS_STD) -DQMX_ENABLE_THREADS -DQMX_ENABLE_DYNAMIC_PLUGINS -O3 $(PPL_SRCS) $(WIN64_OBJS) -lm -o qmx_ppl_win64.exe && \
    echo "[OK] Built native Win64 binaries: qmx_win64.exe, qmx_term_win64.exe & qmx_ppl_win64.exe"; \
  fi

dos-486: $(RETRO_SRCS) src/kernels_p54c.c
> @command -v $(DOS_CC) >/dev/null 2>&1 || (echo "[SKIP/ERROR] DJGPP '$(DOS_CC)' not found." && exit 1)
> $(DOS_CC) $(CFLAGS_STD) $(RETRO_486_FLAGS) -O3 -s $(RETRO_SRCS) src/kernels_p54c.c -o qmx_486.exe
> @! $(DOS_OBJD) -d qmx_486.exe | grep -E '\bcmov|\bfcomi|\bfucomi|\bmm[0-7]|\bxmm|\bymm|\bpfmul\b' || (echo "[ERROR] Forbidden opcode in qmx_486.exe!" && exit 1)
> @echo "[OK] qmx_486.exe: 100% pure i486 DOS"

dos-mmx: $(RETRO_SRCS) src/kernels_p54c.c src/kernels_mmx.c
> @command -v $(DOS_CC) >/dev/null 2>&1 || (echo "[SKIP/ERROR] DJGPP '$(DOS_CC)' not found." && exit 1)
> $(DOS_CC) $(CFLAGS_STD) $(RETRO_MMX_FLAGS) -O3 -s $(RETRO_SRCS) src/kernels_p54c.c src/kernels_mmx.c -o qmx_dos_mmx.exe
> @! $(DOS_OBJD) -d qmx_dos_mmx.exe | grep -E '\bcmov|\bfcomi|\bfucomi|\bxmm|\bymm' || (echo "[ERROR] Forbidden opcode in qmx_dos_mmx.exe!" && exit 1)
> @echo "[OK] qmx_dos_mmx.exe: 100% pure Pentium MMX DOS"

dos-3dnow: $(RETRO_SRCS) src/kernels_p54c.c src/kernels_mmx.c src/kernels_3dnow.c
> @command -v $(DOS_CC) >/dev/null 2>&1 || (echo "[SKIP/ERROR] DJGPP '$(DOS_CC)' not found." && exit 1)
> $(DOS_CC) $(CFLAGS_STD) $(RETRO_3DNOW_FLAGS) -O3 -s $(RETRO_SRCS) src/kernels_p54c.c src/kernels_mmx.c src/kernels_3dnow.c -o qmx_dos_3dnow.exe
> @! $(DOS_OBJD) -d qmx_dos_3dnow.exe | grep -E '\bcmov|\bfcomi|\bfucomi|\bxmm|\bymm' || (echo "[ERROR] Forbidden opcode in qmx_dos_3dnow.exe!" && exit 1)
> @$(DOS_OBJD) -d qmx_dos_3dnow.exe | grep -qE '\bpfmul\b' || (echo "[ERROR] Missing 3DNow! opcodes in qmx_dos_3dnow.exe!" && exit 1)
> @echo "[OK] qmx_dos_3dnow.exe: 100% pure AMD K6-2 3DNow! DOS"

dos-multi: $(RETRO_SRCS) src/kernels_p54c.c src/kernels_mmx.c src/kernels_3dnow.c
> @command -v $(DOS_CC) >/dev/null 2>&1 || (echo "[SKIP/ERROR] DJGPP '$(DOS_CC)' not found." && exit 1)
> $(DOS_CC) $(CFLAGS_STD) $(RETRO_DOS_MULTI_FLAGS) -O3 -s $(RETRO_SRCS) src/kernels_p54c.c src/kernels_mmx.c src/kernels_3dnow.c -o qmx_dos.exe
> @! $(DOS_OBJD) -d qmx_dos.exe | grep -E '\bcmov|\bfcomi|\bfucomi|\bxmm|\bymm' || (echo "[ERROR] Forbidden opcode in qmx_dos.exe!" && exit 1)
> @$(DOS_OBJD) -d qmx_dos.exe | grep -qE '\bpmaddwd\b' && $(DOS_OBJD) -d qmx_dos.exe | grep -qE '\bpfmul\b' || (echo "[ERROR] Missing MMX/3DNow! opcodes in qmx_dos.exe!" && exit 1)
> @echo "[OK] qmx_dos.exe: multi-ISA DOS (i486 base + runtime MMX / 3DNow!)"

dos: dos-multi

retro-dist: win98
> python3 tools/qmx_forge.py --fetch stories260k -q q4 -m stories260k_q4.qmx --exe qmx_mmx.exe --dist-dir dist_retro --kv8 --ctx 256
> cp -f qmx_p54c.exe qmx_3dnow.exe dist_retro/
> @if [ -f stories260k_ternary.qmx ]; then cp -f stories260k_ternary.qmx dist_retro/; fi
> @if [ -f stories260k_binary.qmx ]; then cp -f stories260k_binary.qmx dist_retro/; fi
> @echo "[OK] Ready-to-run Socket 7 / Win98 distribution created in dist_retro/"

test: host plugins
> python3 tools/qmx_pack.py --dummy -q q4 -o test_q4.qmx
> python3 tools/qmx_pack.py --dummy -q ternary -o test_ternary.qmx
> python3 tools/qmx_pack.py --dummy -q q8 -o test_q8.qmx
> python3 tools/qmx_pack.py --info test_q4.qmx
> python3 tests/test_packer.py
> $(CC) $(HOST_ARCH_FLAGS) $(CFLAGS_STD) -O2 tests/test_ops.c $(LIBQMX) -lm -pthread $(HOST_LIBS) -o build/host/test_ops
> ./build/host/test_ops
> $(CC) $(HOST_ARCH_FLAGS) $(CFLAGS_STD) $(HOST_FLAGS) -O2 tests/test_vm.c $(LIBQMX) -lm $(HOST_LIBS) -o build/host/test_vm
> ./build/host/test_vm test_q4.qmx
> $(CC) $(HOST_ARCH_FLAGS) $(CFLAGS_STD) $(HOST_FLAGS) -O2 tests/test_simd.c $(LIBQMX) -lm $(HOST_LIBS) -o build/host/test_simd
> ./build/host/test_simd test_q4.qmx
> $(CC) $(HOST_ARCH_FLAGS) $(CFLAGS_STD) $(HOST_FLAGS) -O2 tests/test_tokenizer.c $(LIBQMX) -lm $(HOST_LIBS) -o build/host/test_tokenizer
> ./build/host/test_tokenizer test_q4.qmx
> $(CC) $(HOST_ARCH_FLAGS) $(CFLAGS_STD) $(HOST_FLAGS) -O2 tests/test_security_and_stress.c $(LIBQMX) -lm $(HOST_LIBS) -o build/host/test_security_and_stress
> ./build/host/test_security_and_stress test_q4.qmx
> $(CC) $(HOST_ARCH_FLAGS) $(CFLAGS_STD) $(HOST_FLAGS) -O2 tests/test_reader.c $(LIBQMX) -lm $(HOST_LIBS) -o build/host/test_reader
> ./build/host/test_reader test_q4.qmx
> $(CC) $(HOST_ARCH_FLAGS) $(CFLAGS_STD) $(HOST_FLAGS) -O2 tests/test_gen.c $(LIBQMX) -lm $(HOST_LIBS) -o build/host/test_gen
> ./build/host/test_gen test_q4.qmx
> $(CC) $(HOST_ARCH_FLAGS) $(CFLAGS_STD) $(HOST_FLAGS) -O2 tests/test_xip_15k.c $(LIBQMX) -lm $(HOST_LIBS) -o build/host/test_xip_15k
> ./build/host/test_xip_15k
> $(CC) $(HOST_ARCH_FLAGS) $(CFLAGS_STD) $(HOST_FLAGS) -O2 tests/test_plugin.c plugins/plugin_layer_profiler.c plugins/plugin_token_ban.c $(LIBQMX) -lm $(HOST_LIBS) -o build/host/test_plugin
> ./build/host/test_plugin test_q4.qmx build/plugins
> $(CC) $(HOST_ARCH_FLAGS) $(CFLAGS_STD) $(HOST_FLAGS) -O2 tests/test_profiles.c $(APP_COMMON) $(LIBQMX) -lm $(HOST_LIBS) -o build/host/test_profiles
> ./build/host/test_profiles
> $(CC) $(HOST_ARCH_FLAGS) $(CFLAGS_STD) $(HOST_FLAGS) -O2 tests/test_session.c $(APP_COMMON) $(LIBQMX) -lm $(HOST_LIBS) -o build/host/test_session
> ./build/host/test_session
> ./qmx -m test_q4.qmx -p "Hello" -n 16 -b auto --verify-crc
> ./qmx -m test_ternary.qmx -p "Hello" -n 16 -b mmx --stream --kv8
> ./qmx -m test_q8.qmx -p "Hello" -n 16 -b scalar
> ./qmx-term -m test_q4.qmx -p "Hello" -n 16 -b auto
> ./qmx-ppl -m test_q4.qmx -p "Hello world, QMX FlashAttention perplexity smoke test!" -c 32 --kv16 --verify-crc -q
> python3 tools/qmx_forge.py -m test_q4.qmx -e ./qmx-term --out-exe build/host/qmx_bundle --backend auto --kv8 --ctx 64
> chmod +x build/host/qmx_bundle

test-san: $(LIBQMX_SAN) plugins
> @test -f test_q4.qmx || python3 tools/qmx_pack.py --dummy -q q4 -o test_q4.qmx
> $(CC) $(HOST_ARCH_FLAGS) $(CFLAGS_STD) $(SAN_FLAGS) tests/test_ops.c $(LIBQMX_SAN) -lm -pthread $(HOST_LIBS) -o build/san/test_ops
> ./build/san/test_ops
> $(CC) $(HOST_ARCH_FLAGS) $(CFLAGS_STD) $(HOST_FLAGS) $(SAN_FLAGS) tests/test_vm.c $(LIBQMX_SAN) -lm $(HOST_LIBS) -o build/san/test_vm
> ./build/san/test_vm test_q4.qmx
> $(CC) $(HOST_ARCH_FLAGS) $(CFLAGS_STD) $(HOST_FLAGS) $(SAN_FLAGS) tests/test_simd.c $(LIBQMX_SAN) -lm $(HOST_LIBS) -o build/san/test_simd
> ./build/san/test_simd test_q4.qmx
> $(CC) $(HOST_ARCH_FLAGS) $(CFLAGS_STD) $(HOST_FLAGS) $(SAN_FLAGS) tests/test_tokenizer.c $(LIBQMX_SAN) -lm $(HOST_LIBS) -o build/san/test_tokenizer
> ./build/san/test_tokenizer test_q4.qmx
> $(CC) $(HOST_ARCH_FLAGS) $(CFLAGS_STD) $(HOST_FLAGS) $(SAN_FLAGS) tests/test_security_and_stress.c $(LIBQMX_SAN) -lm $(HOST_LIBS) -o build/san/test_security_and_stress
> ./build/san/test_security_and_stress test_q4.qmx
> $(CC) $(HOST_ARCH_FLAGS) $(CFLAGS_STD) $(HOST_FLAGS) $(SAN_FLAGS) tests/test_reader.c $(LIBQMX_SAN) -lm $(HOST_LIBS) -o build/san/test_reader
> ./build/san/test_reader test_q4.qmx
> $(CC) $(HOST_ARCH_FLAGS) $(CFLAGS_STD) $(HOST_FLAGS) $(SAN_FLAGS) tests/test_gen.c $(LIBQMX_SAN) -lm $(HOST_LIBS) -o build/san/test_gen
> ./build/san/test_gen test_q4.qmx
> $(CC) $(HOST_ARCH_FLAGS) $(CFLAGS_STD) $(HOST_FLAGS) $(SAN_FLAGS) tests/test_xip_15k.c $(LIBQMX_SAN) -lm $(HOST_LIBS) -o build/san/test_xip_15k
> ./build/san/test_xip_15k
> $(CC) $(HOST_ARCH_FLAGS) $(CFLAGS_STD) $(HOST_FLAGS) $(SAN_FLAGS) tests/test_plugin.c plugins/plugin_layer_profiler.c plugins/plugin_token_ban.c $(LIBQMX_SAN) -lm $(HOST_LIBS) -o build/san/test_plugin
> ./build/san/test_plugin test_q4.qmx build/plugins
> $(CC) $(HOST_ARCH_FLAGS) $(CFLAGS_STD) $(HOST_FLAGS) $(SAN_FLAGS) tests/test_profiles.c $(APP_COMMON) $(LIBQMX_SAN) -lm $(HOST_LIBS) -o build/san/test_profiles
> ./build/san/test_profiles
> $(CC) $(HOST_ARCH_FLAGS) $(CFLAGS_STD) $(HOST_FLAGS) $(SAN_FLAGS) tests/test_session.c $(APP_COMMON) $(LIBQMX_SAN) -lm $(HOST_LIBS) -o build/san/test_session
> ./build/san/test_session
> $(CC) $(HOST_ARCH_FLAGS) $(CFLAGS_STD) $(HOST_FLAGS) $(SAN_FLAGS) tests/fuzz_loader.c $(LIBQMX_SAN) -lm $(HOST_LIBS) -o build/san/fuzz_loader
> ./build/san/fuzz_loader test_q4.qmx
> @echo "[PASS] ASan + UBSan suite passed across all 11 test binaries + 1500-case loader fuzzer!"

FUZZ_TIME ?= 15

fuzz:
> @if ! command -v clang >/dev/null 2>&1; then \
    echo "[SKIP] 'clang' not installed (needed for libFuzzer)."; \
  else \
    test -f test_q4.qmx || python3 tools/qmx_pack.py --dummy -q q4 -o test_q4.qmx; \
    mkdir -p build/fuzz/corpus && \
    cp -f test_q4.qmx build/fuzz/corpus/ && \
    clang -std=c99 -Wall -Wextra -Iinclude -fno-builtin -DQMX_LIBFUZZER -DQMX_NO_THREADS -DQMX_NO_SIMD -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all -g -O1 tests/fuzz_loader.c $(CORE_SRCS) -lm -o build/fuzz/fuzz_loader && \
    ./build/fuzz/fuzz_loader build/fuzz/corpus -max_total_time=$(FUZZ_TIME); \
  fi

test-mcu-nomalloc:
> mkdir -p build/mcu_nomalloc
> for s in $(CORE_SRCS); do \
    base=$$(basename $$s .c); \
    $(CC) $(CFLAGS_STD) -DQMX_NO_STDIO -DQMX_NO_MALLOC -DQMX_NO_THREADS -DQMX_NO_MMAP -DQMX_NO_SIMD -DQMX_NO_CUSTOM_BACKENDS -DQMX_NO_PLUGINS -O2 -c $$s -o build/mcu_nomalloc/$$base.o || exit 1; \
  done
> @! nm -u build/mcu_nomalloc/*.o | grep -E '\b(malloc|calloc|realloc|free|fopen|fread|fseek|fseeko|ftell|ftello|printf|fprintf)\b' || (echo "[ERROR] Forbidden heap/stdio symbol in QMX_NO_MALLOC/QMX_NO_STDIO build!" && exit 1)
> @echo "[PASS] MCU 0-malloc / 0-stdio verification passed (0 heap & stdio symbols)!"

amalgamate:
> python3 tools/amalgamate.py dist

test-amalgamate: amalgamate
> @test -f test_q4.qmx || python3 tools/qmx_pack.py --dummy -q q4 -o test_q4.qmx
> mkdir -p build/amalg
> $(CC) $(HOST_ARCH_FLAGS) -std=c99 -Wall -Wextra -Idist -fno-builtin -DQMX_USE_MMAP -DQMX_ENABLE_THREADS -pthread -O3 -c dist/qmx.c -o build/amalg/qmx_host.o
> $(CC) $(HOST_ARCH_FLAGS) -std=c99 -Wall -Wextra -Idist -fno-builtin -DQMX_USE_MMAP -DQMX_ENABLE_THREADS -pthread -O3 examples/minimal_step.c build/amalg/qmx_host.o -o build/amalg/minimal_step
> ./build/amalg/minimal_step test_q4.qmx "Hello"
> ./build/amalg/minimal_step test_q4.qmx "Hello" --reader
> $(CC) $(HOST_ARCH_FLAGS) $(CFLAGS_STD) $(HOST_FLAGS) -O2 tests/test_vm.c build/amalg/qmx_host.o -lm $(HOST_LIBS) -o build/amalg/test_vm
> ./build/amalg/test_vm test_q4.qmx
> $(CC) $(HOST_ARCH_FLAGS) $(CFLAGS_STD) $(HOST_FLAGS) -O2 tests/test_simd.c build/amalg/qmx_host.o -lm $(HOST_LIBS) -o build/amalg/test_simd
> ./build/amalg/test_simd test_q4.qmx
> $(CC) $(HOST_ARCH_FLAGS) $(CFLAGS_STD) $(HOST_FLAGS) -O2 tests/test_reader.c build/amalg/qmx_host.o -lm $(HOST_LIBS) -o build/amalg/test_reader
> ./build/amalg/test_reader test_q4.qmx
> $(CC) $(HOST_ARCH_FLAGS) $(CFLAGS_STD) $(HOST_FLAGS) -O2 tests/test_gen.c build/amalg/qmx_host.o -lm $(HOST_LIBS) -o build/amalg/test_gen
> ./build/amalg/test_gen test_q4.qmx
> $(CC) -std=c99 -Wall -Wextra -Idist -fno-builtin -DQMX_NO_STDIO -DQMX_NO_MALLOC -DQMX_NO_THREADS -DQMX_NO_MMAP -DQMX_NO_SIMD -DQMX_NO_CUSTOM_BACKENDS -DQMX_NO_PLUGINS -O2 -c dist/qmx.c -o build/amalg/qmx_mcu.o
> @! nm -u build/amalg/qmx_mcu.o | grep -E '\b(malloc|calloc|realloc|free|fopen|fread|fseek|fseeko|ftell|ftello|printf|fprintf)\b' || (echo "[ERROR] Forbidden heap/stdio symbol in amalgamated MCU build!" && exit 1)
> $(CC) -std=c99 -Wall -Wextra -Idist -fno-builtin -DQMX_TEST_AMALGAMATED -DQMX_NO_STDIO -DQMX_NO_MALLOC -DQMX_NO_THREADS -DQMX_NO_MMAP -DQMX_NO_SIMD -DQMX_NO_CUSTOM_BACKENDS -DQMX_NO_PLUGINS -O2 tests/test_xip_15k.c build/amalg/qmx_mcu.o -lm -o build/amalg/test_xip_15k_mcu
> ./build/amalg/test_xip_15k_mcu
> $(CC) -std=c99 -Wall -Wextra -Idist -fno-builtin -m32 $(RETRO_P54C_FLAGS) -O3 -c dist/qmx.c -o build/amalg/qmx_p54c.o
> @! objdump -d build/amalg/qmx_p54c.o | grep -E '\bcmov|\bfcomi|\bfucomi|\bmm[0-7]|\bxmm|\bymm' || (echo "[ERROR] Forbidden opcode in amalgamated P54C build!" && exit 1)
> @echo "[PASS] Amalgamation (dist/qmx.h + dist/qmx.c) verified across Host, SIMD/VM/Reader/Gen suites, MCU 0-malloc & Pure P54C!"

test-isa: host
> python3 tools/test_all_isa.py

tui: term
> @test -f test_q4.qmx || python3 tools/qmx_pack.py --dummy -q q4 -o test_q4.qmx
> ./qmx-term -m test_q4.qmx --ru

clean:
> rm -rf build dist dist_retro qmx qmx-term qmx-ppl qmx_p54c qmx_mmx qmx_3dnow qmx*.exe test_*.qmx dummy.qmx test_ops test_vm test_simd test_tokenizer test_security_and_stress build_*.o QMX.CFG
