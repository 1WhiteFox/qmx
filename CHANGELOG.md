# Changelog — QMX Runtime

All notable changes to the **QMX Runtime & Forge** project are documented in this file.

## [v1.2b] - 2026-11-21
### Added
- **Compact Classifier Quantization Controls (`tools/qmx_pack.py`, `tools/qmx_forge.py`)**: Added `--lm-head-quant` (`auto|binary|ternary|q4|q8|q4_r32|q8_r32`) alongside `--no-hybrid-emb` to reduce DRAM traffic on large vocabularies (128K–152K, saving ~195 MB/token on Qwen3-4B when packing tied embeddings/classifier in `Q4_ROW32`) while keeping hybrid `Q8` default for compact models (`260K`/`15M`).
- **YaRN RoPE Frequency Scaling (`src/qmx_model.c`, `tests/test_security_and_stress.c`)**: Implemented `QMX_ROPE_SCALE_YARN` wavelength-smoothed interpolation in `init_rope_inv_freq` with unit-test coverage.
- **15K MCU Zero-Copy XIP Verification & Native Win64 Target (`tests/test_xip_15k.c`, `Makefile`)**: Added formal `~15K`-parameter micro-model test running directly from a `<16 KB` Flash image in a `<32 KB` static SRAM arena against the `0-malloc / 0-stdio` amalgamated `qmx_mcu.o` object, plus a `make win64` cross-compilation target (`qmx_win64.exe` / `qmx_term_win64.exe`) for native Windows x64 benchmarking against `llama.cpp`.
- **Phi-3 Chat Template & EOT Support (`apps/common/qmx_app_common.h`, `apps/common/qmx_app_common.c`, `tools/qmx_pack.py`)**: Added automatic `QMX_CHAT_PHI3` detection (`<|user|>...<|end|>\n<|assistant|>\n`) and `<|end|>` `eot_id` recognition during HuggingFace tokenizer packing.

### Optimized
- **Zero-Shuffle Even/Odd AVX-512 & AVX2 Microkernels (`src/kernels_x86_ext.c`, `src/kernels_avx2.c`)**: Replaced nibble-unpacking `unpack`/`permutexvar` pipelines with pre-split even/odd activation vectors (`avx512_dot_q4_4rows` loading 4 Q4 rows per 512-bit `vmovdqu64` + two `vpdpbusd` with zero shuffles; `avx2_dot_q4_2rows` loading 2 rows per 256-bit load), eliminated the 16 KB stack `pre_init256[512]` VNNI cache in favor of scalar `x_sum` correction, added 32-row (`2x` tile-row) unrolling with `_mm_prefetch` for `Q4` and `Q8`, and expanded `QMX_PREFILL_MAX_BATCH` / default `prefill_chunk` from `8` to `16`.
- **Peephole Operator Fusion & Fast Vector SwiGLU (`src/qmx_model.c`, `src/qmx_vm.c`, `src/qmx_ops.c`, `src/kernels_avx2.c`, `src/kernels_x86_ext.c`)**: Added automatic load-time fusion of `(RMSNORM + QUANT_ACT)` and `(SWIGLU + QUANT_ACT)` (plus final `rmsnorm_quant` before `lm_head`) without modifying the `.qmx v1.1` format, backed by 8-wide AVX2 and 16-wide AVX-512 fused kernels using fast reciprocal Newton-Raphson refinement (`_mm512_rcp14_ps` / `_mm256_rcp_ps`).
- **GQA KV-Reuse & Hardware SIMD FP16 KV Conversion (`src/qmx_vm.c`, `include/qmx_kernels.h`, `src/kernels_x86_ext.c`, `src/kernels_avx2.c`, `src/kernels_neon.c`)**: Grouped Q-heads sharing a KV-head (`QMX_GQA_MAX_GROUP = 8`) in `exec_attn_heads_job` to cut KV-cache memory reads and dequantizations by up to `4x` on 4:1 GQA models (Qwen3-4B), and wired hardware `vcvtph2ps`/`vcvtps2ph` (`AVX-512`/`AVX2`) and `vcvt_f32_f16` (`NEON`) into `QMX_KV_FP16` write, attention, and RoPE window-shift paths (`stories15m -j 4 --kv16` pure decode reached **5,444 tok/s**).
- **Asynchronous Double-Buffer Streaming (`src/qmx_model.c`, `src/qmx_vm.c`, `tests/test_security_and_stress.c`)**: Added dual streaming buffer `stream_data_buf2` for large-budget multi-threaded streaming (`max_ram == 0` or `>= 16 MB`) with `exec_stream_dbuf_job` overlapping chunk `k+1` reads over `QmxReader` with chunk `k` parallel matmul across the remaining pool workers.
- **Lean Arena Allocation (`src/qmx_model.c`, `src/qmx_dispatch.c`, `src/qmx_vm.c`)**: Added `qmx_backend_needs_xq16` to skip allocating/writing `s->xq16` on constrained/external arenas when unused by the active backend, and scaled `emb_cache_cap` (`4` / `16` / `64` slots) to RAM budget.

### Fixed
- **SentencePiece / Metaspace Duplicate-Byte BPE Priority (`src/qmx_tokenizer.c`)**: Fixed a tokenizer indexing bug on SentencePiece models (`Phi-3`, `LLaMA-2`, `Mistral`, `Gemma`) where byte-fallback tokens (`<0x00>..<0xFF>`, IDs `3..258`) shadowed real single-byte vocabulary pieces in `byte_to_tok[]` and prevented BPE merges. Implemented zero-malloc `O(log M)` merge-score resolution (`tokenizer_token_merge_score` / `tokenizer_register_byte`), restoring proper tokenization (`Phi-3` prompt compressed from 85 raw bytes to 22 BPE tokens without repacking).
- **Backend Binding in SIMD Test Harness (`tests/test_simd.c`)**: Switched `verify_backend_ops` to load `m_got` via `qmx_load(..., backend, ...)` so backend-specific arena flags (`xq16`) are accurately exercised across all 18 host SIMD backends.

### Verified
- **Test & Build Matrix (`0 warnings`, `-std=c99 -Wall -Wextra`)**: Passed all 8 host unit/stress suites, `test-mcu-nomalloc` (`0` heap/stdio symbols), `test-amalgamate` (`dist/qmx.h` + `dist/qmx.c` across Host, MCU `0-malloc`, and Pure P54C), `test-isa` (bit-accurate parity across all 19 backends, `Checksum=30.0950`), `retro-dist` (`0 CMOV` Win98 PE), and `win64`.
- **15K MCU Zero-Copy XIP (`tests/test_xip_15k.c`)**: Verified `15,456`-parameter micro-model directly on `qmx_mcu.o` (`0-malloc / 0-stdio`) in **10.56 KB Flash** and **6.75 KB SRAM**.
- **5-Model Native Win64 Matrix (`qmx_win64.exe`, Windows 10 IoT LTSC, AMD Ryzen 7 H 255, FP16 KV)**:
  - `TinyStories-260K` (`q4_r32`): **41,554 tok/s** decode (RAM) / **9,766 tok/s** (DOS-Stream, `161.8 KB` arena).
  - `SmolLM2-135M-Instruct` (`q4_r32_all`, `-j 4`): **521.92 tok/s** decode (RAM) / **60.44 tok/s** (`Stream 32 MB`, bit-exact).
  - `Qwen2.5-0.5B-Instruct` (`q4_r32_lm4`, `-j 8`, GQA 7:1 + QKV Bias): **188.17–189.12 tok/s** decode (EN/RU) / **27.15 tok/s** (`Stream 128 MB`, bit-exact).
  - `Phi-3-mini-4k-instruct` (`3.82B`, `q4_r32_lm4`, `-j 8`, MHA, fused QKV/GateUp): **26.19 tok/s** decode (`AVX2`) / **2.40 tok/s** (`Stream 256 MB`, bit-exact).
  - `Qwen3-4B` (`4.02B`, `-j 8`, GQA 4:1 + `QK_NORM`): **23.31–23.48 tok/s** (`hyb-Q8`, `2.293 GiB`) and **24.77–25.19 tok/s** (`lm4-Q4`, `2.112 GiB`) decode (EN/RU `<think>`), reaching the ~55 GB/s DRAM bandwidth ceiling.
- **Head-to-Head Median Check (`Qwen3-4B`, Windows 10 IoT LTSC, `-t 8 / -j 8`, `-c 1024`, FP16 KV, `temp=0.0`, 1 warmup + 3 runs)**:
  - `llama.cpp 2.80.1` (`Q4_K_M`, `2.326 GiB`, AVX2 build): **64.94 tok/s** prefill | **20.49 tok/s** decode (`47.7 GiB/s` eff. DRAM).
  - `QMX v1.2b` (`hyb-Q8`, `2.293 GiB`, `AVX2`): **56.54 tok/s** prefill | **21.73 tok/s** decode (`49.8 GiB/s`).
  - `QMX v1.2b` (`hyb-Q8`, `2.293 GiB`, `AVX-512`): **60.78 tok/s** prefill | **22.68 tok/s** decode (`52.0 GiB/s`).
  - `QMX v1.2b` (`lm4-Q4`, `2.112 GiB`, `AVX2`): **61.12 tok/s** prefill | **23.73 tok/s** decode (`50.1 GiB/s`).
  - *Methodology note*: `llama.cpp` reference binary was AVX2-compiled (`Q4_K_M` uses mixed 4/6-bit superblocks with asymmetric scales) and remained faster on prefill (`+6–15%`), while QMX achieved higher decode bandwidth via simpler symmetric `16x32` `Q4_ROW32` tiles and runtime AVX-512 dispatch; minor OS background activity was present during runs.

---

## [v1.2a] - 2026-11-21
### Added
- **Abstract I/O Callback Interface (`include/qmx.h`, `include/qmx_runtime.h`, `src/qmx_model.c`, `src/qmx_vm.c`, `src/qmx_tokenizer.c`)**: Introduced `QmxReader` (`QmxReadFn` / `QmxCloseFn` + `size`), `QmxLoadParams` (`qmx_load_params_default`), `qmx_load_from_reader`, `qmx_load_ex`, `qmx_load_from_memory_ex`, and `qmx_tokenizer_load_reader`. All weight loading, tensor pinning, adaptive streaming, embedding lookup, and tokenizer loading now operate over offset-based callbacks (enabling SPI Flash, raw SD sectors, or network streams without modifying core code), while preserving zero-copy `mmap` (`POSIX_MADV_WILLNEED`) and XIP fast paths. Enforced explicit ownership rules: `qmx_tokenizer_load_reader` strictly borrows the reader and never calls `close()`, while `qmx_free` invokes `reader.close()` only when non-NULL.
- **Zero-Malloc & External Arena MCU Support (`include/qmx_runtime.h`, `src/qmx_ops.c`, `src/qmx_model.c`, `src/qmx_vm.c`, `src/qmx_tokenizer.c`)**: Added `qmx_arena_init_external`, `qmx_calc_arena_size`, and `qmx_calc_arena_size_memory` alongside `arena_buf` / `arena_size` in `QmxLoadParams`. Converted `QmxTensorDesc` validation and loading to 16-entry stack chunks (removing temporary heap allocations) and added compile-time stripping macros (`QMX_NO_STDIO`, `QMX_NO_MALLOC`, `QMX_NO_THREADS`, `QMX_NO_MMAP`, `QMX_NO_SIMD`, `QMX_NO_CUSTOM_BACKENDS`), verified by `make test-mcu-nomalloc` (`0` heap and stdio symbols).
- **Step-by-Step Generation Engine (`include/qmx_gen.h`, `src/qmx_gen.c`)**: Added non-blocking token-by-token generation API (`QmxGenState`, `QmxGenParams`, `QmxStepResult`, `QmxStepStatus`, `qmx_gen_init`, `qmx_gen_reset`, `qmx_gen_begin`, `qmx_step`, `qmx_gen_flush`) with configurable chunked prefill (`prefill_chunk`), zero heap allocations, automatic `pending_token` flushing, and built-in `prefill_tokens` / `decode_tokens` counters for RTOS/UART and interactive frontends.
- **Public Read-Only Telemetry API (`include/qmx.h`, `include/qmx_runtime.h`, `src/qmx_model.c`)**: Added `QmxModelInfo`, `QmxStats`, `qmx_get_model_info`, `qmx_get_stats`, and `qmx_reset_stats` so external applications no longer inspect private `QmxModel` fields directly.
- **Static Library & Single-Pair Amalgamation (`Makefile`, `tools/amalgamate.py`, `examples/minimal_step.c`)**: Added `make libqmx` (`build/host/libqmx.a`), `make amalgamate` (`dist/qmx.h` + `dist/qmx.c`, SQLite/stb drop-in format), `make test-amalgamate`, and the standalone `examples/minimal_step.c` embedding example.
- **Dedicated Reference Clients & Split Prefill/Decode Telemetry (`apps/common/`, `apps/qmx-cli/`, `apps/qmx-term/`)**: Extracted the 80x25 retro terminal interface into a standalone `qmx-term` client (`apps/qmx-term/`) and minimal CLI runner `qmx` (`apps/qmx-cli/`), both built strictly over the public `libqmx` API and `qmx_step`, with separate `[Perf] prefill=... | decode=...` timings in CLI output.
- **Expanded Verification Suite (`tests/test_reader.c`, `tests/test_gen.c`, `tests/test_security_and_stress.c`)**: Added unit and integration tests for mock-SPI `QmxReader` + external arena + borrowed tokenizer loading, `qmx_step` parity (`prefill_chunk=1` and `8`), multi-turn `qmx_gen_flush`, narrow-window KV shift `pos` consistency, `.qmx v1.1` header extension fields, and 4-byte aligned zero-copy XIP tokenizer merges.

### Changed
- **Static `.rodata` Backend Dispatch Table (`include/qmx_kernels.h`, `src/qmx_dispatch.c`, `src/kernels_*.c`)**: Replaced `__attribute__((constructor))` registration across all 10 kernel modules with a compile-time `static const QmxBackendOps * const g_builtin_backends[QMX_BACKEND_COUNT]` table controlled by `QMX_ENABLE_BACKEND_*`, eliminating static-linker stripping in `libqmx.a` (no `--whole-archive`), thread-init races, and MCU SRAM overhead.
- **64-Bit Offset Signatures (`include/qmx_runtime.h`, `include/qmx_tokenizer.h`)**: Widened `base_offset` and `slice_size` parameters in `qmx_load_from_offset` and `qmx_tokenizer_load` from `size_t` to `uint64_t`.
- **MSVC & Cross-Compiler Portability (`include/qmx_runtime.h`, `src/qmx_dispatch.c`, `src/kernels_sse2.c`, `src/kernels_avx2.c`)**: Added `_MSC_VER` support for alignment, restrict/inline macros, `_mm_sqrt_ss`, `__cpuidex`, `_xgetbv`, and target-attribute guards.
- **Retro Build Packaging (`Makefile`, `tools/qmx_forge.py`)**: Host builds now emit both `qmx` (CLI) and `qmx-term` (TUI), while retro targets (`x86-p54c`, `x86-mmx`, `k6-3dnow`, `win98`, `dos`, `retro-dist`) and self-bundles package a single compact `qmx-term` binary supporting both TUI and `-p` CLI benchmarking.

### Fixed
- **`.qmx v1.1` Header Layout Synchronization (`include/qmx.h`)**: Aligned `QmxHeader` bytes 112..127 (`rope_scale_type`, `norm_type`, `sliding_window`, `reserved[12]`) with `tools/qmx_pack.py` while keeping all existing v1.1 cartridges valid, and added compile-time `offsetof` assertions to prevent future layout drift.
- **AMD K6-2 3DNow! Macro Detection (`include/qmx_kernels.h`, `src/kernels_3dnow.c`)**: Fixed case-sensitive macro check (`__3dNOW__` in addition to `__3dnow__`), restoring native `pfmul` / `pfadd` / `pfrsqrt` / `pi2fd` code generation in `qmx_3dnow` and `qmx_3dnow.exe`.
- **KV-Cache Position Consistency in Batched Prefill (`src/qmx_vm.c`)**: Added `qmx_forward_batch_ex` to propagate updated `pos` and `kv_shifted` state back to callers whenever an automatic sliding-window shift occurs mid-prefill.
- **Zero-Copy XIP Tokenizer Alignment (`tools/qmx_pack.py`)**: Padded `string_pool` to a 4-byte boundary in `serialize_qmx_tokenizer` so `merges` maps directly from Flash/ROM with zero heap allocation.
- **Per-Function Kernel Attributes & TUI Fixes (`src/kernels_mmx.c`, `src/kernels_neon.c`, `apps/qmx-term/`, `Makefile`)**: Migrated `kernels_mmx.c` to per-function `__attribute__((target("mmx")))` (removing file-wide `-mmmx` and unused `Makefile` variables), propagated `target("+dotprod")` to `neon_dot_*` callers, enabled `--min-p` (`qmx_sample_ex`) in `qmx-term`, and updated the F3 hotkey label from `3KV8` to `3KV`.

---

## [v1.1b] - 2026-11-21
### Added
- **Native Pure-NumPy SafeTensors Reader (`tools/qmx_pack.py`)**: Implemented `NativeSafeTensors` for direct BF16, F16, and F32 tensor loading without PyTorch or `safetensors` dependencies, added packing progress output for large models, and wired exact `bos_token_id` / `eos_token_id` parsing from `config.json` (required for Qwen-2/3).

### Optimized
- **Zero-Math-Impact `mmap` Page Cache Hint (`src/qmx_model.c`)**: Added `posix_madvise(mapped, payload_sz64, POSIX_MADV_WILLNEED)` under `_POSIX_C_SOURCE 200809L`, reaching **15.01 tok/s** on Qwen3-4B (2.35 GB, native ext4, `-j 8 --kv16`, AVX-512) with bit-identical `step0_checksum=45.308704`.
- **Adaptive Streaming Buffer for 4B Models (`src/qmx_model.c`)**: Dynamically scaled `stream_tile_cap` up to 32,768 tiles (~17 MB) and stdio buffering up to 1 MB when `--max-ram >= 16 MB` or `payload >= 64 MB`, reducing disk read calls on Qwen3-4B by **84.1x** and achieving **7.53 tok/s** within a strict 256 MB RAM cap.

---

## [v1.1a] - 2026-11-21
### Added
- **`.qmx v1.1` Container Format & 64-Bit File I/O (`include/qmx.h`, `include/qmx_runtime.h`, `src/qmx_ops.c`, `src/qmx_model.c`, `tools/qmx_pack.py`)**: Extended `QmxHeader` (128 B) with 64-bit `payload_offset_hi` / `payload_size_hi`, explicit `head_dim` and `rotary_dim` (Partial RoPE), `emb_scale`, `logit_softcap`, `attn_softcap`, `rope_scale_type` / `rope_scale_factor`, `sliding_window`, and `QMX_FLAG_COMPACT_ALIGN` (64 B alignment for micro-models < 100K params). Extended `QmxTensorDesc` (64 B) with 48-bit offsets (up to 256 TB) while keeping full read compatibility with `.qmx v1.0`. Replaced 32-bit `fseek`/`ftell` with `qmx_fseek64` / `qmx_ftell64` / `qmx_file_size64`, unified loaders into `qmx_load_internal`, and added streaming single-pass packing for Qwen-3, Gemma-2/3 (`+1.0` RMSNorm, 4 norms/layer), and Phi-3/4 (fused `qkv_proj` / `gate_up_proj` slicing).
- **Per-Row Quantization Profiles `Q4_ROW32` & `Q8_ROW32` (`src/qmx_ops.c`, `src/kernels_*.c`, `tools/qmx_pack.py`, `tools/qmx_forge.py`)**: Added 16x FP16 per-row scales per 16x32 tile across all 23 compute backends (using hardware `_mm_cvtph_ps` on AVX2/AVX-512, `vcvt_f32_f16` on NEON, and integer FP16 unpacking without x87 touch on 3DNow!), plus `qmx_fp16_to_fp32` / `qmx_fp32_to_fp16` helpers.
- **FP16 KV-Cache & Attention Sinks (`src/qmx_model.c`, `src/qmx_vm.c`, `src/main.c`)**: Added `QMX_KV_FP16` (`--kv16`, 2x memory reduction; F3 cycle `F32 -> F16 -> I8` in TUI) and `qmx_shift_kv_cache_ex` preserving the first 4 sink tokens without RoPE rotation during sliding-window shifts.
- **Encapsulated Cross-Platform Thread Pool & Multi-Threaded Attention (`include/qmx_thread.h`, `src/qmx_thread.c`, `src/qmx_vm.c`)**: Replaced global thread state with per-model `QmxThreadPool` supporting POSIX (`pthreads`) and native Win32 (`CreateThread` / `CreateEventA` / `Interlocked*`), adaptive spin-wait (`QMX_SPIN_ITERS = 4096`), clean shutdown in `qmx_free`, and parallel attention heads (`exec_attn_heads_job`) for `att_len >= 64`.
- **$O(N \log N)$ Min-Heap BPE, XIP Tokenizer & Min-P Sampler (`include/qmx_tokenizer.h`, `src/qmx_tokenizer.c`, `include/qmx_sampler.h`, `src/qmx_sampler.c`, `src/main.c`)**: Rewrote BPE merging using a doubly-linked list and binary min-heap (`bpe_merge_heap`), added Zero-Copy XIP `qmx_tokenizer_load_from_memory`, added Min-P sampling (`qmx_sample_ex`, `--min-p`), NaN/Inf logit protection, raised `QMX_MAX_TOP_K` to 256, removed the 1 KB CLI prompt cap, and added `-f <prompt_file>`.

### Fixed
- **ISA Safety & SIGILL Elimination (`src/kernels_x86_ext.c`, `src/kernels_avx2.c`, `src/kernels_sse2.c`, `Makefile`, `tools/test_all_isa.py`)**: Implemented a pure `sse3_dot_q8` without SSSE3 instructions (verified in `test_all_isa.py`), switched `kernels_avx2.c` and `kernels_sse2.c` to per-function `__attribute__((target(...)))`, removed file-wide `-mavx2`/`-msse2` flags, and added missing `_mm256_zeroupper()` calls in AVX kernels.

### Optimized
- **Unified VM & Cache-Blocked GEMM Prefill (`src/qmx_vm.c`)**: Merged single-token and batched execution into `qmx_vm_exec_chunk` with persistent `model->batch_ws` (zero `malloc`/`free` during generation), hoisted RoPE phase calculation outside the layer loop, and added 4-tile-row strip blocking (`QMX_GEMM_ROW_STRIP = 4`).
- **SIMD Activation Quantization & Shared `.rodata` LUTs (`src/kernels_*.c`, `src/qmx_ops.c`)**: Added hardware FMA3/F16C, vectorized `avx2_prepare_act` / `avx512_prepare_act` / `neon_prepare_act`, expanded VNNI pre-cache up to `cols <= 16384`, vectorized Cortex-M55 Q4 (Helium MVE), and consolidated duplicate LUTs from 9 kernel files into shared `.rodata` tables (0 bytes SRAM on MCU).

---

## [v1.01b] - 2026-11-18
### Fixed
- **Thread Pool Synchronization (`src/qmx_vm.c`)**: Pre-initialized per-worker generation counters (`g_worker_init_gen[t]`) under `g_pool_mtx` prior to `pthread_create` to prevent asynchronous startup race conditions when expanding the thread pool immediately before a parallel dispatch. Lowered multi-threading threshold to `65536` work elements so 8-token batched prefills engage worker threads on compact models.
- **TUI Multi-Turn KV-Cache Consistency (`src/main.c`)**: Added `pending_kv_tok` tracking in `run_tui_session` to ensure the final sampled token is always flushed through `qmx_forward` into the KV-cache when generation terminates via the `steps` limit, `ESC`, or a stop-sequence match.

### Optimized
- **AVX-512 VNNI Accumulator Pipeline (`src/kernels_x86_ext.c`)**: Replaced per-tile cross-lane `128 -> 256 -> 512` integer repacking (`_mm256_inserti128_si256` / `_mm512_inserti64x4`) in `avx512_matmul_tiled` with four direct 128-bit `FP32` vector accumulators (`vacc0..vacc3`), eliminating lane-crossing overhead on small matrices.

---

## [v1.01a] - 2026-11-18
### Added
- **Native 512-bit AVX-512 VNNI Tiled MatMul (`src/kernels_x86_ext.c`)**: Implemented full 512-bit `ZMM` `_mm512_dpbusd_epi32` (`vpdpbusd`) kernels for `Q8`, `Q4`, `1.58-bit Ternary`, and `1-bit Binary` quantization profiles with analytical sign-bias compensation.
- **Full ARMv8.2-A NEON-DotProd Backend (`src/kernels_neon.c`)**: Implemented dedicated `sdot`-accelerated `neon_dot_matmul_tiled` across all quantization profiles and enabled automatic Apple Silicon (`aarch64`) detection.
- **Chat Template Auto-Detection (`src/main.c`)**: Added automatic prompt formatting for `LLaMA-3`, `ChatML`, and `Gemma-2` special tokens in both CLI and TUI modes.

### Fixed
- **BUG-01 (`src/kernels_neon.c`, `src/kernels_cortex_m.c`)**: Moved `qmx_ops_neon_dot` out of `kernels_cortex_m.c` and bound it to vectorized NEON routines instead of Cortex-M scalar fallbacks.
- **BUG-02 (`src/qmx_vm.c`)**: Fixed worker `seen_gen` initialization on dynamic thread pool expansion (`qmx_set_num_threads`).
- **BUG-03 (`src/qmx_vm.c`)**: Added context window boundary checks and automatic `qmx_shift_kv_cache` invocation inside `qmx_forward_batch`.
- **BUG-04 (`src/main.c`, `tools/qmx_forge.py`)**: Expanded `QMX.CFG` and bundle loader backend validation from `<= 8` to all 23 registered `QmxBackend` IDs.
- **BUG-05 (`tools/qmx_pack.py`, `src/qmx_model.c`, `src/qmx_vm.c`)**: Synchronized `QMX_OP_LAYERNORM`, `QMX_OP_QK_NORM`, `QMX_FLAG_LOGIT_SOFTCAP`, and `TID_RMS_FINAL_BIAS` across the Python packer and C runtime.
- **BUG-06 (`include/qmx_tokenizer.h`, `src/qmx_tokenizer.c`)**: Raised `QMX_MAX_SPECIAL_TOKENS` from `64` to `256` and added UTF-8 multi-byte punctuation filtering (`is_utf8_punct_piece`) in Byte-Level BPE.
- **BUG-07 (`src/qmx_vm.c`)**: Added seamless fallback to `fseek` + `fread` chunk repositioning when sequential `data_stream_ready` reads return fewer bytes than requested.
- **BUG-08 (`src/main.c`)**: Preserved active `cur_ctx` when switching `.qmx` models via the `F9` modal and switched TUI prompt prefill to 8-token `qmx_forward_batch` chunks with live progress rendering and `ESC` abort support.

### Optimized
- **Batched Prefill in RAM Mode (`src/qmx_vm.c`)**: Enabled 8-token mini-GEMM batching in `qmx_forward_batch` for RAM-resident, `mmap`, and `XIP` models.
- **Hardware Math Primitives (`include/qmx_runtime.h`)**: Added hardware `sqrtss` (x86_64/SSE), `fsqrt` (AArch64), and `vsqrt.f32` (ARM VFP) to `qmx_sqrtf`, and replaced microcoded x87 `fsincos` on SSE2/x86_64 with a fast half-period polynomial reduction in `qmx_sincosf`.
- **Stream Pin Pool Sizing (`src/qmx_model.c`)**: Capped `pin_pool` allocation to the exact byte size of remaining unpinned tensors (`unpinned_need`).

---

## [v1.0] - 2026-11-17
### Added
- Initial release of the `.qmx` v1.0 Tile-Major (16x32) cartridge format, Layer Recipe VM bytecode interpreter, deterministic `QmxArena` allocator, `FP32`/`INT8` KV-cache with RoPE phase-corrected sliding window, $O(\log M)$ BPE tokenizer, dual-language CP866/UTF-8 retro TUI, Socket 7 `CMOV` binary sanitizer (`sanitize_i586_pe.py`), and 23 hardware ISA backends.
