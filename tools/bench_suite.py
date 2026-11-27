#!/usr/bin/env python3
import argparse
import os
import platform
import re
import statistics
import subprocess

PERF_RE = re.compile(
    r"\[Perf\]\s+prefill=(\d+)\s+tok in\s+([\d.]+)\s+ms\s+\(([\d.]+)\s+tok/s\)\s+\|\s+decode=(\d+)\s+tok in\s+([\d.]+)\s+ms\s+\(([\d.]+)\s+tok/s\)"
)
STATS_RE = re.compile(r"step0_checksum=([-\d.]+)")
RUNTIME_RE = re.compile(
    r"backend=([^|]+)\|.*kv=([^|]+)\|.*mode=([^|]+)\|.*threads=(\d+)\s+\|\s+arena=([\d.]+)\s+KB"
)
PPL_RE = re.compile(
    r"\[PPL\]\s+tokens=(\d+)\s+\|\s+chunks=(\d+)\s+\|\s+avg_nll=([\d.]+)\s+\|\s+ppl=([\d.]+)\s+\+/-\s+([\d.]+)\s+\|\s+time=([\d.]+)\s+ms\s+\(([\d.]+)\s+tok/s\)"
)


def detect_cpu_name() -> str:
    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                if ln.lower().startswith("model name"):
                    return ln.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def detect_cc_version(cc: str = "gcc") -> str:
    try:
        out = subprocess.check_output([cc, "--version"], text=True, stderr=subprocess.STDOUT)
        return out.splitlines()[0].strip()
    except Exception:
        return "unknown"


def make_synthetic_prompt(approx_tokens: int) -> str:
    unit = "The quick brown fox jumps over the lazy dog near the quiet river bank. "
    reps = max(1, (approx_tokens // 14) + 2)
    return (unit * reps).strip()


def run_speed_case(qmx_bin: str, model: str, ctx: int, gen_steps: int,
                   threads: str, backend: str, kv: str, stream_mb: int,
                   prefill_chunk: int, warmup: int, runs: int, prompt_text: str):
    cmd = [
        qmx_bin, "-m", model, "-p", "!" + prompt_text,
        "-c", str(ctx), "-n", str(gen_steps),
        "-t", "0.0", "-j", str(threads), "-b", backend,
        "--prefill-chunk", str(prefill_chunk),
    ]
    if kv == "fp16":
        cmd.append("--kv16")
    elif kv == "q8":
        cmd.append("--kv8")
    if stream_mb > 0:
        cmd.extend(["--stream", "--max-ram", str(stream_mb)])

    cmd_str = " ".join(cmd)
    pp_vals, tg_vals = [], []
    meta = {}
    for r in range(warmup + runs):
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Benchmark failed ({cmd_str}):\n{res.stderr}")
        m_p = PERF_RE.search(res.stderr)
        m_s = STATS_RE.search(res.stderr)
        m_r = RUNTIME_RE.search(res.stderr)
        if not m_p:
            raise RuntimeError(f"Could not parse [Perf] line:\n{res.stderr}")
        if r >= warmup:
            pp_vals.append(float(m_p.group(3)))
            tg_vals.append(float(m_p.group(6)))
        meta = {
            "pp_tok": int(m_p.group(1)),
            "tg_tok": int(m_p.group(4)),
            "chk": float(m_s.group(1)) if m_s else 0.0,
            "backend": m_r.group(1).strip() if m_r else backend,
            "kv": m_r.group(2).strip() if m_r else kv,
            "mode": m_r.group(3).strip() if m_r else ("STREAM" if stream_mb else "RAM"),
            "threads": int(m_r.group(4)) if m_r else 0,
            "arena_kb": float(m_r.group(5)) if m_r else 0.0,
        }
    meta["pp_tps"] = statistics.median(pp_vals)
    meta["tg_tps"] = statistics.median(tg_vals)
    meta["size_gib"] = os.path.getsize(model) / (1024.0 ** 3)
    return meta


def run_ppl_case(ppl_bin: str, model: str, corpus_file: str, ctx: int, stride: int,
                 threads: str, backend: str, kv: str, chunks: int, verify_crc: bool):
    cmd = [
        ppl_bin, "-m", model, "-f", corpus_file,
        "-c", str(ctx), "--stride", str(stride),
        "-j", str(threads), "-b", backend, "-q",
    ]
    if kv == "fp16":
        cmd.append("--kv16")
    elif kv == "q8":
        cmd.append("--kv8")
    if chunks > 0:
        cmd.extend(["--chunks", str(chunks)])
    if verify_crc:
        cmd.append("--verify-crc")

    cmd_str = " ".join(cmd)
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"PPL run failed ({cmd_str}):\n{res.stderr}")
    m = PPL_RE.search(res.stderr)
    if not m:
        raise RuntimeError(f"Could not parse [PPL] line:\n{res.stderr}")
    return {
        "tokens": int(m.group(1)),
        "chunks": int(m.group(2)),
        "avg_nll": float(m.group(3)),
        "ppl": float(m.group(4)),
        "ppl_err": float(m.group(5)),
        "time_ms": float(m.group(6)),
        "tok_s": float(m.group(7)),
        "size_mb": os.path.getsize(model) / (1024.0 ** 2),
    }


def main():
    parser = argparse.ArgumentParser(description="QMX Reproducible Speed & WikiText-2 PPL Suite")
    parser.add_argument("--qmx-bin", default="./qmx", help="Path to qmx binary")
    parser.add_argument("--ppl-bin", default="./qmx-ppl", help="Path to qmx-ppl binary")
    parser.add_argument("--models", nargs="+", required=True, help="List of .qmx cartridge paths")
    parser.add_argument("--contexts", type=int, nargs="+", default=[512], help="Context sizes")
    parser.add_argument("--prompt-tokens", type=int, default=128, help="Approx prompt tokens")
    parser.add_argument("--gen-steps", type=int, default=64, help="Decode tokens per run")
    parser.add_argument("--threads", default="auto", help="Thread count or 'auto'")
    parser.add_argument("--backend", default="auto", help="Backend override or 'auto'")
    parser.add_argument("--kv", choices=["fp32", "fp16", "q8"], default="fp16", help="KV precision")
    parser.add_argument("--stream-mb", type=int, default=0, help="Stream RAM cap in MB (0=off)")
    parser.add_argument("--prefill-chunk", type=int, default=32, help="Prefill batch chunk size")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs")
    parser.add_argument("--runs", type=int, default=3, help="Measured runs (median)")
    parser.add_argument("--ppl-file", type=str, help="Corpus text file for qmx-ppl")
    parser.add_argument("--ppl-ctx", type=int, default=512, help="Context window for PPL")
    parser.add_argument("--ppl-stride", type=int, default=256, help="Sliding stride for PPL")
    parser.add_argument("--ppl-chunks", type=int, default=0, help="Limit PPL chunks (0=all)")
    parser.add_argument("--verify-crc", action="store_true", help="Pass --verify-crc to qmx-ppl")
    args = parser.parse_args()

    print("### Environment")
    print(f"- **OS:** `{platform.system()} {platform.release()} ({platform.machine()})`")
    print(f"- **CPU:** `{detect_cpu_name()}`")
    print(f"- **Compiler:** `{detect_cc_version()}`")
    print(f"- **Settings:** `threads={args.threads}`, `backend={args.backend}`, `kv={args.kv}`, `prefill_chunk={args.prefill_chunk}`, `warmup={args.warmup}`, `runs={args.runs}`\n")

    prompt_txt = make_synthetic_prompt(args.prompt_tokens)
    print("### Throughput (Prefill & Decode Median)")
    print("| Model | Size (GiB) | Mode | Ctx | Backend | Prefill (tok/s) | Decode (tok/s) | Arena (KB) | Step0 Checksum |")
    print("| :--- | ---: | :--- | ---: | :--- | ---: | ---: | ---: | ---: |")
    for mpath in args.models:
        mname = os.path.basename(mpath)
        for ctx in args.contexts:
            r = run_speed_case(
                args.qmx_bin, mpath, ctx, args.gen_steps, args.threads,
                args.backend, args.kv, 0, args.prefill_chunk, args.warmup, args.runs, prompt_txt,
            )
            print(
                f"| `{mname}` | {r['size_gib']:.3f} | {r['mode']} | {ctx} | `{r['backend']}` | "
                f"**{r['pp_tps']:.2f}** | **{r['tg_tps']:.2f}** | {r['arena_kb']:.1f} | `{r['chk']:.4f}` |"
            )
        if args.stream_mb > 0:
            rs = run_speed_case(
                args.qmx_bin, mpath, args.contexts[0], args.gen_steps, args.threads,
                args.backend, args.kv, args.stream_mb, args.prefill_chunk, args.warmup, args.runs, prompt_txt,
            )
            print(
                f"| `{mname}` | {rs['size_gib']:.3f} | STREAM ({args.stream_mb} MB) | {args.contexts[0]} | `{rs['backend']}` | "
                f"**{rs['pp_tps']:.2f}** | **{rs['tg_tps']:.2f}** | {rs['arena_kb']:.1f} | `{rs['chk']:.4f}` |"
            )

    if args.ppl_file:
        print("\n### Perplexity (`qmx-ppl`)")
        print("| Model | Size (MB) | Scored Tokens | Chunks | Avg NLL | PPL | Eval Speed (tok/s) |")
        print("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for mpath in args.models:
            rp = run_ppl_case(
                args.ppl_bin, mpath, args.ppl_file, args.ppl_ctx, args.ppl_stride,
                args.threads, args.backend, args.kv, args.ppl_chunks, args.verify_crc,
            )
            print(
                f"| `{os.path.basename(mpath)}` | {rp['size_mb']:.2f} | {rp['tokens']} | {rp['chunks']} | "
                f"{rp['avg_nll']:.4f} | **{rp['ppl']:.4f} +/- {rp['ppl_err']:.4f}** | {rp['tok_s']:.2f} |"
            )


if __name__ == "__main__":
    main()
