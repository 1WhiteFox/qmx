#!/usr/bin/env python3
# Copyright 2026 1WhiteFox
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import shutil
import struct
import sys
import urllib.request
import qmx_pack as pack

ALIGN = 64

BACKEND_MAP = {
    "scalar": 0,
    "p54c": 1,
    "mmx": 2,
    "3dnow": 3,
    "sse2": 4,
    "avx2": 5,
    "neon": 6,
    "arm_dsp": 7,
    "mcu32": 8,
    "sse": 9,
    "sse3": 10,
    "ssse3": 11,
    "sse4.1": 12,
    "sse41": 12,
    "sse4.2": 13,
    "sse42": 13,
    "avx": 14,
    "avx512": 15,
    "cortex-m0": 16,
    "m0": 16,
    "cortex-m23": 17,
    "m23": 17,
    "cortex-m3": 18,
    "m3": 18,
    "cortex-m4": 19,
    "m4": 19,
    "m7": 19,
    "cortex-m33": 20,
    "m33": 20,
    "cortex-m55": 21,
    "m55": 21,
    "neon_dot": 22,
    "auto": 255,
}

PRESET_MODELS = {
    "stories260k": {
        "model_url": "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/stories260K.bin",
        "tok_url": "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/tok512.bin",
        "bin_file": "stories260K.bin",
        "tok_file": "tok512.bin",
    },
    "stories15m": {
        "model_url": "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin",
        "tok_url": "https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.bin",
        "bin_file": "stories15M.bin",
        "tok_file": "tokenizer.bin",
    },
}


def download_if_missing(url: str, dst_path: str):
    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 1024:
        print(f"[Forge] Cached file found: {dst_path} ({os.path.getsize(dst_path) / 1024.0:.1f} KB)")
        return
    print(f"[Forge] Downloading {url} -> {dst_path} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "QMX-Forge/1.0"})
    with urllib.request.urlopen(req) as resp, open(dst_path, "wb") as out:
        shutil.copyfileobj(resp, out)
    print(f"[Forge] Downloaded {dst_path} ({os.path.getsize(dst_path) / 1024.0:.1f} KB)")


def build_runtime_cfg(
    model_file: str = "model.qmx",
    steps: int = 128,
    ctx_limit: int = 256,
    temp: float = 0.8,
    backend: str = "auto",
    use_kv8: bool = False,
    force_stream: bool = False,
    lang: str = "ru",
    max_ram_kb: int = 0,
) -> bytes:
    mf_bytes = model_file.encode("ascii", errors="ignore")[:35]
    mf_padded = mf_bytes + b"\x00" * (36 - len(mf_bytes))
    cfg = struct.pack(
        "<8sIIfBBBBI36s",
        b"QMXCFG\x00\x00",
        steps,
        ctx_limit,
        temp,
        BACKEND_MAP.get(backend, 255),
        1 if use_kv8 else 0,
        1 if force_stream else 0,
        1 if lang == "ru" else 0,
        max_ram_kb,
        mf_padded,
    )
    assert len(cfg) == 64
    return cfg


def pack_standalone_exe(exe_path: str, qmx_path: str, cfg_bytes: bytes, out_exe: str):
    with open(exe_path, "rb") as f:
        exe_data = bytearray(f.read())
    with open(qmx_path, "rb") as f:
        qmx_data = f.read()

    rem = len(exe_data) % ALIGN
    if rem != 0:
        exe_data.extend(b"\x00" * (ALIGN - rem))
    bundle_offset = len(exe_data)
    bundle_size = len(qmx_data)
    exe_data.extend(qmx_data)

    rem = len(exe_data) % ALIGN
    if rem != 0:
        exe_data.extend(b"\x00" * (ALIGN - rem))
    cfg_offset = len(exe_data)
    exe_data.extend(cfg_bytes)

    trailer = struct.pack("<III4s", bundle_offset, bundle_size, cfg_offset, b"QMXP")
    exe_data.extend(trailer)

    out_dir = os.path.dirname(os.path.abspath(out_exe))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_exe, "wb") as f:
        f.write(exe_data)
    try:
        os.chmod(out_exe, 0o755)
    except OSError:
        pass
    print(f"[Forge] Packed standalone executable: {out_exe} ({len(exe_data) / 1024.0:.1f} KB)")


def prepare_retro_dist(dist_dir: str, exe_path: str, qmx_path: str, cfg_bytes: bytes):
    os.makedirs(dist_dir, exist_ok=True)
    dst_qmx = os.path.join(dist_dir, os.path.basename(qmx_path))
    shutil.copy2(qmx_path, dst_qmx)
    exe_name = os.path.basename(exe_path)
    dst_exe = os.path.join(dist_dir, exe_name)
    shutil.copy2(exe_path, dst_exe)
    cfg_path = os.path.join(dist_dir, "QMX.CFG")
    with open(cfg_path, "wb") as f:
        f.write(cfg_bytes)

    bat_bench = os.path.join(dist_dir, "BENCH.BAT")
    with open(bat_bench, "w", encoding="ascii") as f:
        f.write("@ECHO OFF\r\n")
        f.write("CLS\r\n")
        f.write("ECHO === QMX Runtime v1.2b Benchmark ===\r\n")
        f.write(f'{exe_name} -m {os.path.basename(qmx_path)} -p "Once upon a time" -n 128 -b auto\r\n')
        f.write("PAUSE\r\n")

    bat_tui = os.path.join(dist_dir, "RUN_TUI.BAT")
    with open(bat_tui, "w", encoding="ascii") as f:
        f.write("@ECHO OFF\r\n")
        f.write(f"{exe_name} -m {os.path.basename(qmx_path)} --tui --ru\r\n")
    print(f"[Forge] Prepared retro distribution in '{dist_dir}/'")


def main():
    parser = argparse.ArgumentParser(description="QMX Forge: Fetch, Convert, Inspect & Bundle .qmx Packages")
    parser.add_argument("--info", type=str, help="Inspect metadata and verify CRC32 of a .qmx file")
    parser.add_argument("--fetch", choices=["stories260k", "stories15m"], help="Download & convert TinyStories model")
    parser.add_argument("--hf", type=str, help="Convert HuggingFace repo or local dir into .qmx")
    parser.add_argument("--quant", "-q", choices=["binary", "ternary", "q8", "q4", "q4_r32", "q8_r32"], default="q4")
    parser.add_argument("--model", "-m", type=str, help="Input/Output .qmx model")
    parser.add_argument("--exe", "-e", type=str, help="Base binary executable")
    parser.add_argument("--out-exe", type=str, help="Output self-contained single-click executable")
    parser.add_argument("--out-cfg", type=str, help="Output 64-byte QMX.CFG file")
    parser.add_argument("--dist-dir", type=str, help="Output directory for Retro PC / 86Box distribution")
    parser.add_argument("--ctx", type=int, default=256, help="Default context limit")
    parser.add_argument("--steps", type=int, default=128, help="Default generation steps")
    parser.add_argument("--temp", type=float, default=0.8, help="Default sampling temperature")
    parser.add_argument("--backend", choices=list(BACKEND_MAP.keys()), default="auto")
    parser.add_argument("--kv8", action="store_true", help="Enable INT8 KV-Cache by default")
    parser.add_argument("--stream", action="store_true", help="Enable Disk Streaming by default")
    parser.add_argument("--lang", choices=["en", "ru"], default="ru", help="Default UI language")
    parser.add_argument("--max-ram-kb", type=int, default=0)
    parser.add_argument("--no-hybrid-emb", action="store_true", help="Quantize embeddings/lm_head with primary quant")
    parser.add_argument("--lm-head-quant", choices=["auto", "binary", "ternary", "q4", "q8", "q4_r32", "q8_r32"], default="auto")
    args = parser.parse_args()

    if args.info:
        pack.inspect_qmx(args.info)
        return

    q_map = {
        "binary": pack.QMX_QUANT_BINARY_1BIT,
        "ternary": pack.QMX_QUANT_TERNARY_158,
        "q8": pack.QMX_QUANT_Q8_SYM,
        "q4": pack.QMX_QUANT_Q4_SYM,
        "q4_r32": pack.QMX_QUANT_Q4_ROW32,
        "q8_r32": pack.QMX_QUANT_Q8_ROW32,
    }
    qmx_path = args.model

    if args.fetch:
        preset = PRESET_MODELS[args.fetch]
        download_if_missing(preset["model_url"], preset["bin_file"])
        download_if_missing(preset["tok_url"], preset["tok_file"])
        if not qmx_path:
            qmx_path = f"{args.fetch}_{args.quant}.qmx"
        cfg, t_iter = pack.load_karpathy_bin(preset["bin_file"], preset["tok_file"])
        lm_q = None if args.lm_head_quant == "auto" else q_map[args.lm_head_quant]
        pack.export_qmx(qmx_path, cfg, t_iter, q_map[args.quant], not args.no_hybrid_emb, lm_head_quant=lm_q)
    elif args.hf:
        if not qmx_path:
            qmx_path = "model_hf.qmx"
        cfg, t_iter = pack.load_hf_safetensors(args.hf)
        lm_q = None if args.lm_head_quant == "auto" else q_map[args.lm_head_quant]
        pack.export_qmx(qmx_path, cfg, t_iter, q_map[args.quant], not args.no_hybrid_emb, lm_head_quant=lm_q)

    if not qmx_path:
        sys.exit("Error: specify --model <file.qmx>, --fetch {stories260k,stories15m}, --hf <repo>, or --info <file.qmx>")

    cfg_bytes = build_runtime_cfg(
        model_file=os.path.basename(qmx_path),
        steps=args.steps,
        ctx_limit=args.ctx,
        temp=args.temp,
        backend=args.backend,
        use_kv8=args.kv8,
        force_stream=args.stream,
        lang=args.lang,
        max_ram_kb=args.max_ram_kb,
    )

    if args.out_cfg:
        with open(args.out_cfg, "wb") as f:
            f.write(cfg_bytes)
        print(f"[Forge] Saved 64-byte binary config: {args.out_cfg}")

    if args.dist_dir:
        if not args.exe:
            sys.exit("Error: --exe is required when building --dist-dir")
        prepare_retro_dist(args.dist_dir, args.exe, qmx_path, cfg_bytes)

    if args.out_exe:
        if not args.exe:
            sys.exit("Error: --exe is required when building --out-exe")
        pack_standalone_exe(args.exe, qmx_path, cfg_bytes, args.out_exe)


if __name__ == "__main__":
    main()