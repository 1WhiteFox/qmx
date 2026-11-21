#!/usr/bin/env python3
# Copyright 2026 1WhiteFox
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import re
import subprocess
import sys

STATS_RE = re.compile(
    r"\[Stats\]\s+(\d+)\s+tokens in\s+([\d.]+)\s+ms\s+\(([\d.]+)\s+tok/s\)\s+\|\s+step0_checksum=([-\d.]+)"
)
RUNTIME_RE = re.compile(r"arena=([\d.]+)\s+KB")
STREAM_RE = re.compile(r"disk_reads=(\d+)\s+\(([\d.]+)\s+KB\)")


def run_cmd(cmd, check=True):
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and res.returncode != 0:
        print(f"[ERROR] Command failed: {' '.join(cmd)}", file=sys.stderr)
        print(res.stdout, file=sys.stderr)
        print(res.stderr, file=sys.stderr)
        sys.exit(res.returncode)
    return res


def parse_qmx_run(exe_path, model_path, backend, steps=64, ctx=256, kv8=False, stream=False, max_ram_kb=0, prompt="Once upon a time"):
    cmd = [
        exe_path,
        "-m", model_path,
        "-p", prompt,
        "-n", str(steps),
        "-c", str(ctx),
        "-t", "0.0",
        "-b", backend,
    ]
    if kv8:
        cmd.append("--kv8")
    if stream:
        cmd.append("--stream")
    if max_ram_kb > 0:
        cmd.extend(["--max-ram-kb", str(max_ram_kb)])

    res = run_cmd(cmd)
    err = res.stderr
    m_stat = STATS_RE.search(err)
    m_rt = RUNTIME_RE.search(err)
    m_st = STREAM_RE.search(err)

    tok_s = float(m_stat.group(3)) if m_stat else 0.0
    chk = float(m_stat.group(4)) if m_stat else 0.0
    arena_kb = float(m_rt.group(1)) if m_rt else 0.0
    disk_reads = int(m_st.group(1)) if m_st else 0
    disk_kb = float(m_st.group(2)) if m_st else 0.0
    sample_text = res.stdout.strip().replace("\n", " ")[:70]

    return {
        "file_kb": os.path.getsize(model_path) / 1024.0,
        "arena_kb": arena_kb,
        "tok_s": tok_s,
        "chk": chk,
        "disk_reads": disk_reads,
        "disk_kb": disk_kb,
        "text": sample_text,
    }


def main():
    parser = argparse.ArgumentParser(description="QMX Synthetic & Real-World Benchmark Suite")
    parser.add_argument("--full", action="store_true", help="Also download and benchmark stories15m (15M params)")
    parser.add_argument("--skip-fetch", action="store_true", help="Run only local synthetic tests without network")
    args = parser.parse_args()

    print("=== [1/3] Building QMX Host Runtime & Running Synthetic Test Suite ===")
    run_cmd(["make", "test"])

    stress_bin = "build/host/test_security_and_stress"
    run_cmd([
        "gcc", "-std=c99", "-Wall", "-Wextra", "-Iinclude", "-DQMX_ENABLE_THREADS", "-pthread", "-O2",
        "tests/test_security_and_stress.c", "build/host/libqmx.a", "-lm", "-o", stress_bin
    ])
    res_stress = run_cmd([stress_bin, "test_q4.qmx"])
    print(res_stress.stdout.strip())

    if args.skip_fetch:
        return

    print("\n=== [2/3] Packing Real TinyStories-260K Cartridges (Q8 / Q4 / Ternary / Binary) ===")
    for q in ("q8", "q4", "ternary", "binary"):
        out_qmx = f"stories260k_{q}.qmx"
        run_cmd([sys.executable, "tools/qmx_forge.py", "--fetch", "stories260k", "-q", q, "-m", out_qmx])

    print("\n=== [3/3] Real-World Benchmark: QMX vs DOS 540KB Budget (stories260k) ===")
    print(f"{'Profile / Mode':<34} | {'File KB':>8} | {'Arena KB':>8} | {'IO Reads':>8} | {'Speed (tok/s)':>13} | {'Checksum':>10}")
    print("-" * 96)

    configs = [
        ("Q8  RAM  (FP32 KV, Auto SIMD)", "stories260k_q8.qmx", "auto", False, False, 0),
        ("Q4  RAM  (FP32 KV, Scalar)",    "stories260k_q4.qmx", "scalar", False, False, 0),
        ("Q4  RAM  (FP32 KV, P54C)",      "stories260k_q4.qmx", "p54c", False, False, 0),
        ("Q4  RAM  (FP32 KV, MMX)",       "stories260k_q4.qmx", "mmx", False, False, 0),
        ("Q4  RAM  (FP32 KV, SSE2)",      "stories260k_q4.qmx", "sse2", False, False, 0),
        ("Q4  RAM  (FP32 KV, Auto SIMD)", "stories260k_q4.qmx", "auto", False, False, 0),
        ("Q4  DOS-Stream (KV8, 128KB cap)", "stories260k_q4.qmx", "auto", True, True, 128),
        ("1.58b Ternary Stream (KV8)",    "stories260k_ternary.qmx", "auto", True, True, 96),
        ("1.0b  Binary  Stream (KV8)",    "stories260k_binary.qmx", "auto", True, True, 96),
    ]

    sample_out = ""
    for label, mpath, bkend, kv8, strm, ram_kb in configs:
        r = parse_qmx_run("./qmx", mpath, bkend, steps=64, ctx=256, kv8=kv8, stream=strm, max_ram_kb=ram_kb)
        if label.startswith("Q4  RAM  (FP32 KV, Auto"):
            sample_out = r["text"]
        print(
            f"{label:<34} | {r['file_kb']:>7.1f}K | {r['arena_kb']:>7.1f}K | "
            f"{r['disk_reads']:>8d} | {r['tok_s']:>13.1f} | {r['chk']:>10.4f}"
        )

    print("-" * 96)
    print(f"Sample Generation (stories260k Q4): \"{sample_out}...\"")

    if args.full:
        print("\n=== [Bonus] Real-World 15M Model in ~1.1 MB RAM (stories15m Q4) ===")
        out_15m = "stories15m_q4.qmx"
        run_cmd([sys.executable, "tools/qmx_forge.py", "--fetch", "stories15m", "-q", "q4", "-m", out_15m])
        r_ram = parse_qmx_run("./qmx", out_15m, "auto", steps=32, ctx=256, kv8=False, stream=False)
        r_str = parse_qmx_run("./qmx", out_15m, "auto", steps=32, ctx=256, kv8=True, stream=True, max_ram_kb=1200)
        print(f"15M Q4 RAM Resident : File={r_ram['file_kb']:.1f} KB | Arena={r_ram['arena_kb']:.1f} KB | Speed={r_ram['tok_s']:.1f} tok/s")
        print(f"15M Q4 Disk Stream  : File={r_str['file_kb']:.1f} KB | Arena={r_str['arena_kb']:.1f} KB | IO={r_str['disk_reads']} reads | Speed={r_str['tok_s']:.1f} tok/s")
        print(f"Sample Generation (stories15m Q4): \"{r_str['text']}...\"")


if __name__ == "__main__":
    main()