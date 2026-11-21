# qmx

[Русская версия](README_RU.md)

qmx (Quantized Model eXchange) is a zero-dependency C99 inference runtime, embeddable SDK, and binary model format (.qmx) for quantized transformer LLMs.

It is an ideological successor to Tunguzka-LLM — growing out of a retro Socket 7 inference engine into a single C99 codebase that runs models from ~15K parameters on bare-metal microcontrollers up to 4B models (Qwen3-4B, Phi-3-mini, Gemma-2/3, SmolLM2) on modern x86 and ARM CPUs.

## What it does and why

Most LLM runtimes focus either on multi-gigabyte desktop/server setups or small single-purpose demos. qmx tests how far one self-contained C99 core with no external libraries can stretch across both ends:

- Binary format (.qmx v1.1): stores 16x32 tiled quantized weights, an embedded BPE tokenizer, and a compact Layer Recipe VM bytecode for the transformer block.
- Three memory modes: RAM/mmap for desktop speed, Zero-Copy XIP from Flash/ROM with an external static arena (0 malloc, 0 stdio for MCUs), and adaptive streaming over a generic QmxReader callback to run gigabyte models inside a strict RAM cap.
- 23 ISA backends: Scalar C99, Pentium P54C/MMX, AMD 3DNow!, SSE through AVX-512 VNNI, ARM NEON / NEON-DotProd, ARM-DSP, MCU32, and Cortex-M0 through Cortex-M55.
- Embeddable SDK: compiles to libqmx.a or a single amalgamated pair (dist/qmx.h + dist/qmx.c), with two reference clients (qmx CLI and qmx-term 80x25 TUI) built on a non-blocking qmx_step() API.
- Pure-NumPy packer: tools/qmx_pack.py converts HuggingFace SafeTensors (BF16/F16/F32) and llama2.c .bin files directly into .qmx without PyTorch.

## Performance overview

In local tests on an AMD Ryzen 7 H 255 (Windows x64, FP16 KV-cache), qmx scales across the full target range:

- Micro-model (~15.5K params, 0-malloc MCU XIP): fits in ~10.6 KB Flash and ~6.8 KB SRAM.
- Compact models (260K to 0.5B): from ~188 tok/s on Qwen2.5-0.5B up to 40K+ tok/s on TinyStories-260K.
- 4B models (Qwen3-4B, Phi-3-mini): ~22–26 tok/s decode in RAM mode, approaching the DRAM bandwidth ceiling and edging out an AVX2 llama.cpp build on decode (while remaining slower on prompt prefill). In streaming mode, 4B models still run inside a strict 256 MB RAM budget.

Detailed benchmark tables, exact hardware/software flags, and methodology notes will be published in a separate file.

## Quick start

Build the CLI, TUI, and static library:

```bash
make host
make test
make amalgamate
```

Pack a model:

```bash
# Quick test with TinyStories-260K
python3 tools/qmx_forge.py --fetch stories260k -q q4_r32 -m stories260k.qmx

# Or convert a HuggingFace model (local folder or repo ID)
python3 tools/qmx_pack.py --hf Qwen/Qwen3-4B -o qwen3_4b.qmx -q q4_r32 --lm-head-quant q4_r32
```

Run:

```bash
# CLI
./qmx -m stories260k.qmx -p "Once upon a time" -n 64

# Low-RAM streaming (256 MB cap)
./qmx -m qwen3_4b.qmx -p "Hello" -n 64 -j 8 --kv16 --stream --max-ram 256

# 80x25 terminal UI
./qmx-term -m stories260k.qmx
```

For embedding examples, see examples/minimal_step.c and tests/test_xip_15k.c.

## Disclaimer and AI disclosure

I work in a non-CS field (BI & BA) and build this purely as a hobby in my free time. The code is provided as-is, with no guarantees or ongoing support commitments.

AI usage: The overall idea (continuing Tunguzka-LLM), runtime architecture, .qmx format, and streaming/XIP design are mine, while AI wrote the majority of the C99, SIMD/ASM, and Python code under my direction and testing.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
