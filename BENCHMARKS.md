# QMX Runtime Benchmarks & Quality Methodology

Verified performance and footprint measurements for **QMX Runtime** with reproducible commands (`tools/bench_suite.py`, `qmx`, `qmx-ppl`).

## 1. Testbed & Methodology

* **Host CPU:** AMD Ryzen 7 H 255 (8C/16T, AVX-512 VNNI / AVX2 / FMA / F16C)
* **Host OS:** Windows 10 IoT LTSC (native `x86_64-w64-mingw32-gcc -O3` Win64 binaries) & Linux / WSL2 (`gcc -std=c99 -O3`)
* **Protocol:** deterministic greedy decoding (`-t 0.0`), 1 warmup + 3 timed runs (median), separated `[Perf] prefill` and `decode` timings, bit-exact `step0_checksum` between RAM and Streaming modes.

```bash
python3 tools/bench_suite.py \
  --qmx-bin ./qmx --ppl-bin ./qmx-ppl \
  --models model_a.qmx model_b.qmx \
  --contexts 512 2048 4096 8192 \
  --kv fp16 --threads auto --stream-mb 256 \
  --ppl-file wiki.test.raw --ppl-ctx 512 --ppl-stride 256 --verify-crc
```

## 2. Bare-Metal MCU Zero-Copy XIP (`0-malloc / 0-stdio`)

| Model | Parameters | Quant | Flash Image | Static SRAM Arena | Heap / Stdio |
| :--- | ---: | :--- | ---: | ---: | :--- |
| Micro-Transformer (`test_xip_15k`) | `15,456` | `Q4_SYM` + `Q8 KV` | **10.56 KB** | **6.75 KB** | `0` symbols |

## 3. 5-Model Native Win64 Matrix (`qmx_win64.exe`, FP16 KV)

| Model | Params | Quant Profile | Threads | RAM Decode (tok/s) | Stream Cap | Stream Decode (tok/s) |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: |
| `TinyStories-260K` | 0.26M | `q4_r32` | 1 | **41,554** | 161.8 KB | **9,766** |
| `SmolLM2-135M-Instruct` | 135M | `q4_r32` (`all`) | 4 | **521.92** | 32 MB | **60.44** |
| `Qwen2.5-0.5B-Instruct` | 0.50B | `q4_r32` (`lm4`) | 8 | **188.17-189.12** | 128 MB | **27.15** |
| `Phi-3-mini-4k-instruct` | 3.82B | `q4_r32` (`lm4`) | 8 | **26.24** | 256 MB | **2.36** |
| `Qwen3-4B` (`hyb-Q8`) | 4.02B | `q4_r32` + `Q8 emb` | 8 | **23.31-23.48** | - | - |
| `Qwen3-4B` (`lm4-Q4`) | 4.02B | `q4_r32` | 8 | **24.77-25.19** | 256 MB | **2.76** |

## 4. ZMM/YMM Batch GEMM Prefill (`Phi-3-mini-3.8B`, `-j 8`, `--kv16`, 128-tok prompt)

| Backend | Prefill Batch GEMM | Gain vs Pre-Batch | Decode (tok/s) | Step0 Checksum |
| :--- | ---: | ---: | ---: | ---: |
| `AVX-512` (`B=8`/`B=4` ZMM) | **101.61 tok/s** (~126.6 steady) | **+64.7%** | **26.09** | `2483.347656` |
| `AVX2` (`B=4` YMM) | **71.18 tok/s** | **+25.6%** | **25.79** | `2488.346680` |

## 5. WikiText-2 Perplexity Validation (`qmx-ppl`)

```bash
python3 tools/qmx_pack.py --gguf model_f16.gguf -q q4_r32 --imatrix imatrix.gguf -o model_q4_r32_im.qmx
./qmx-ppl -m model_q4_r32_im.qmx -f wiki.test.raw -c 512 --stride 256 --kv16 -j auto --verify-crc
```
