.RECIPEPREFIX = >
CC         ?= gcc
AR         ?= ar
WIN32_CC   ?= i686-w64-mingw32-gcc
WIN32_OBJD ?= i686-w64-mingw32-objdump
WIN64_CC   ?= x86_64-w64-mingw32-gcc
DOS_CC     ?= i586-pc-msdosdjgpp-gcc

CORE_SRCS   = src/qmx_ops.c src/qmx_thread.c src/qmx_dispatch.c src/qmx_model.c src/qmx_vm.c src/qmx_tokenizer.c src/qmx_sampler.c src/qmx_gen.c
KERNEL_SRCS = src/kernels_p54c.c src/kernels_mmx.c src/kernels_3dnow.c src/kernels_sse2.c src/kernels_avx2.c src/kernels_x86_ext.c src/kernels_neon.c src/kernels_arm_dsp.c src/kernels_mcu32.c src/kernels_cortex_m.c
APP_COMMON  = apps/common/qmx_app_common.c
CLI_SRCS    = $(APP_COMMON) apps/qmx-cli/main.c
TERM_SRCS   = $(APP_COMMON) apps/qmx-term/tui.c apps/qmx-term/main.c
RETRO_SRCS  = $(CORE_SRCS) $(TERM_SRCS)

CORE_OBJS   = $(patsubst src/%.c,build/host/%.o,$(CORE_SRCS))
KERNEL_OBJS = $(patsubst src/%.c,build/host/%.o,$(KERNEL_SRCS))
LIBQMX      = build/host/libqmx.a

CFLAGS_STD  = -std=c99 -Wall -Wextra -Iinclude -fno-builtin
HOST_FLAGS  = -DQMX_USE_MMAP -DQMX_ENABLE_THREADS -pthread

RETRO_P54C_FLAGS  = -march=pentium -mtune=pentium -mno-mmx -mno-3dnow -mno-sse -mno-sse2 -DQMX_CUSTOM_BACKEND_SET -DQMX_ENABLE_BACKEND_P54C
RETRO_MMX_FLAGS   = -march=pentium-mmx -mtune=pentium-mmx -mmmx -mno-3dnow -mno-sse -mno-sse2 -DQMX_CUSTOM_BACKEND_SET -DQMX_ENABLE_BACKEND_P54C -DQMX_ENABLE_BACKEND_MMX
RETRO_3DNOW_FLAGS = -march=k6-2 -mtune=k6-2 -m3dnow -mno-sse -mno-sse2 -DQMX_CUSTOM_BACKEND_SET -DQMX_ENABLE_BACKEND_P54C -DQMX_ENABLE_BACKEND_MMX -DQMX_ENABLE_BACKEND_3DNOW
WIN98_DEFS        = -D__USE_MINGW_ANSI_STDIO=0

.PHONY: all libqmx host cli term amalgamate test-amalgamate check-m32 x86-p54c x86-mmx k6-3dnow win98 win98-p54c win98-mmx win98-3dnow win64 dos test test-mcu-nomalloc test-isa retro-dist tui clean

all: host

build/host:
> mkdir -p build/host

build/host/qmx_%.o: src/qmx_%.c | build/host
> $(CC) $(CFLAGS_STD) $(HOST_FLAGS) -O3 -c $< -o $@

build/host/kernels_%.o: src/kernels_%.c | build/host
> $(CC) $(CFLAGS_STD) -O3 -c $< -o $@

$(LIBQMX): $(CORE_OBJS) $(KERNEL_OBJS)
> rm -f $@
> $(AR) rcs $@ $(CORE_OBJS) $(KERNEL_OBJS)

libqmx: $(LIBQMX)

cli: $(LIBQMX) $(CLI_SRCS)
> $(CC) $(CFLAGS_STD) $(HOST_FLAGS) -O3 $(CLI_SRCS) $(LIBQMX) -o qmx

term: $(LIBQMX) $(TERM_SRCS)
> $(CC) $(CFLAGS_STD) $(HOST_FLAGS) -O3 $(TERM_SRCS) $(LIBQMX) -o qmx-term

host: cli term

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
> @command -v $(WIN64_CC) >/dev/null 2>&1 || (echo "[SKIP] '$(WIN64_CC)' not installed (install: sudo apt install gcc-mingw-w64-x86-64)." && exit 0)
> mkdir -p build/win64
> for s in $(CORE_SRCS); do \
    base=$$(basename $$s .c); \
    $(WIN64_CC) $(CFLAGS_STD) -DQMX_ENABLE_THREADS -O3 -c $$s -o build/win64/$$base.o || exit 1; \
  done
> for s in $(KERNEL_SRCS); do \
    base=$$(basename $$s .c); \
    $(WIN64_CC) $(CFLAGS_STD) -O3 -c $$s -o build/win64/$$base.o || exit 1; \
  done
> $(WIN64_CC) $(CFLAGS_STD) -DQMX_ENABLE_THREADS -O3 $(CLI_SRCS) build/win64/*.o -o qmx_win64.exe
> $(WIN64_CC) $(CFLAGS_STD) -DQMX_ENABLE_THREADS -O3 $(TERM_SRCS) build/win64/*.o -o qmx_term_win64.exe
> @echo "[OK] Built native Win64 binaries: qmx_win64.exe & qmx_term_win64.exe"

dos: $(RETRO_SRCS) src/kernels_p54c.c src/kernels_mmx.c
> @command -v $(DOS_CC) >/dev/null 2>&1 || (echo "[SKIP/ERROR] DJGPP '$(DOS_CC)' not found." && exit 1)
> $(DOS_CC) $(CFLAGS_STD) $(RETRO_MMX_FLAGS) -O3 -s $(RETRO_SRCS) src/kernels_p54c.c src/kernels_mmx.c -o qmx_dos.exe

retro-dist: win98
> python3 tools/qmx_forge.py --fetch stories260k -q q4 -m stories260k_q4.qmx --exe qmx_mmx.exe --dist-dir dist_retro --kv8 --ctx 256
> cp -f qmx_p54c.exe qmx_3dnow.exe dist_retro/
> @if [ -f stories260k_ternary.qmx ]; then cp -f stories260k_ternary.qmx dist_retro/; fi
> @if [ -f stories260k_binary.qmx ]; then cp -f stories260k_binary.qmx dist_retro/; fi
> @echo "[OK] Ready-to-run Socket 7 / Win98 distribution created in dist_retro/"

test: host
> python3 tools/qmx_pack.py --dummy -q q4 -o test_q4.qmx
> python3 tools/qmx_pack.py --dummy -q ternary -o test_ternary.qmx
> python3 tools/qmx_pack.py --dummy -q q8 -o test_q8.qmx
> python3 tools/qmx_pack.py --info test_q4.qmx
> $(CC) $(CFLAGS_STD) -O2 tests/test_ops.c $(LIBQMX) -lm -pthread -o build/host/test_ops
> ./build/host/test_ops
> $(CC) $(CFLAGS_STD) $(HOST_FLAGS) -O2 tests/test_vm.c $(LIBQMX) -lm -o build/host/test_vm
> ./build/host/test_vm test_q4.qmx
> $(CC) $(CFLAGS_STD) $(HOST_FLAGS) -O2 tests/test_simd.c $(LIBQMX) -lm -o build/host/test_simd
> ./build/host/test_simd test_q4.qmx
> $(CC) $(CFLAGS_STD) $(HOST_FLAGS) -O2 tests/test_tokenizer.c $(LIBQMX) -lm -o build/host/test_tokenizer
> ./build/host/test_tokenizer test_q4.qmx
> $(CC) $(CFLAGS_STD) $(HOST_FLAGS) -O2 tests/test_security_and_stress.c $(LIBQMX) -lm -o build/host/test_security_and_stress
> ./build/host/test_security_and_stress test_q4.qmx
> $(CC) $(CFLAGS_STD) $(HOST_FLAGS) -O2 tests/test_reader.c $(LIBQMX) -lm -o build/host/test_reader
> ./build/host/test_reader test_q4.qmx
> $(CC) $(CFLAGS_STD) $(HOST_FLAGS) -O2 tests/test_gen.c $(LIBQMX) -lm -o build/host/test_gen
> ./build/host/test_gen test_q4.qmx
> $(CC) $(CFLAGS_STD) $(HOST_FLAGS) -O2 tests/test_xip_15k.c $(LIBQMX) -lm -o build/host/test_xip_15k
> ./build/host/test_xip_15k
> ./qmx -m test_q4.qmx -p "Hello" -n 16 -b auto
> ./qmx -m test_ternary.qmx -p "Hello" -n 16 -b mmx --stream --kv8
> ./qmx -m test_q8.qmx -p "Hello" -n 16 -b scalar
> ./qmx-term -m test_q4.qmx -p "Hello" -n 16 -b auto
> python3 tools/qmx_forge.py -m test_q4.qmx -e ./qmx-term --out-exe build/host/qmx_bundle --backend auto --kv8 --ctx 64
> chmod +x build/host/qmx_bundle

test-mcu-nomalloc:
> mkdir -p build/mcu_nomalloc
> for s in $(CORE_SRCS); do \
    base=$$(basename $$s .c); \
    $(CC) $(CFLAGS_STD) -DQMX_NO_STDIO -DQMX_NO_MALLOC -DQMX_NO_THREADS -DQMX_NO_MMAP -DQMX_NO_SIMD -DQMX_NO_CUSTOM_BACKENDS -O2 -c $$s -o build/mcu_nomalloc/$$base.o || exit 1; \
  done
> @! nm -u build/mcu_nomalloc/*.o | grep -E '\b(malloc|calloc|realloc|free|fopen|fread|fseek|fseeko|ftell|ftello|printf|fprintf)\b' || (echo "[ERROR] Forbidden heap/stdio symbol in QMX_NO_MALLOC/QMX_NO_STDIO build!" && exit 1)
> @echo "[PASS] MCU 0-malloc / 0-stdio verification passed (0 heap & stdio symbols)!"

amalgamate:
> python3 tools/amalgamate.py dist

test-amalgamate: amalgamate
> @test -f test_q4.qmx || python3 tools/qmx_pack.py --dummy -q q4 -o test_q4.qmx
> mkdir -p build/amalg
> $(CC) -std=c99 -Wall -Wextra -Idist -fno-builtin -DQMX_USE_MMAP -DQMX_ENABLE_THREADS -pthread -O3 -c dist/qmx.c -o build/amalg/qmx_host.o
> $(CC) -std=c99 -Wall -Wextra -Idist -fno-builtin -DQMX_USE_MMAP -DQMX_ENABLE_THREADS -pthread -O3 examples/minimal_step.c build/amalg/qmx_host.o -o build/amalg/minimal_step
> ./build/amalg/minimal_step test_q4.qmx "Hello"
> ./build/amalg/minimal_step test_q4.qmx "Hello" --reader
> $(CC) $(CFLAGS_STD) $(HOST_FLAGS) -O2 tests/test_vm.c build/amalg/qmx_host.o -lm -o build/amalg/test_vm
> ./build/amalg/test_vm test_q4.qmx
> $(CC) $(CFLAGS_STD) $(HOST_FLAGS) -O2 tests/test_simd.c build/amalg/qmx_host.o -lm -o build/amalg/test_simd
> ./build/amalg/test_simd test_q4.qmx
> $(CC) $(CFLAGS_STD) $(HOST_FLAGS) -O2 tests/test_reader.c build/amalg/qmx_host.o -lm -o build/amalg/test_reader
> ./build/amalg/test_reader test_q4.qmx
> $(CC) $(CFLAGS_STD) $(HOST_FLAGS) -O2 tests/test_gen.c build/amalg/qmx_host.o -lm -o build/amalg/test_gen
> ./build/amalg/test_gen test_q4.qmx
> $(CC) -std=c99 -Wall -Wextra -Idist -fno-builtin -DQMX_NO_STDIO -DQMX_NO_MALLOC -DQMX_NO_THREADS -DQMX_NO_MMAP -DQMX_NO_SIMD -DQMX_NO_CUSTOM_BACKENDS -O2 -c dist/qmx.c -o build/amalg/qmx_mcu.o
> @! nm -u build/amalg/qmx_mcu.o | grep -E '\b(malloc|calloc|realloc|free|fopen|fread|fseek|fseeko|ftell|ftello|printf|fprintf)\b' || (echo "[ERROR] Forbidden heap/stdio symbol in amalgamated MCU build!" && exit 1)
> $(CC) -std=c99 -Wall -Wextra -Idist -fno-builtin -DQMX_TEST_AMALGAMATED -DQMX_NO_STDIO -DQMX_NO_MALLOC -DQMX_NO_THREADS -DQMX_NO_MMAP -DQMX_NO_SIMD -DQMX_NO_CUSTOM_BACKENDS -O2 tests/test_xip_15k.c build/amalg/qmx_mcu.o -lm -o build/amalg/test_xip_15k_mcu
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
> rm -rf build dist dist_retro qmx qmx-term qmx_p54c qmx_mmx qmx_3dnow qmx*.exe test_*.qmx dummy.qmx test_ops test_vm test_simd test_tokenizer test_security_and_stress build_*.o QMX.CFG
