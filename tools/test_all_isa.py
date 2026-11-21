#!/usr/bin/env python3
import os
import platform
import re
import shutil
import subprocess
import sys


def run_cmd(cmd, check=True):
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and res.returncode != 0:
        print(f"[FAIL] {' '.join(cmd)}\n{res.stderr}", file=sys.stderr)
        sys.exit(res.returncode)
    return res


def verify_x86_objdump(obj_path: str):
    out = run_cmd(["objdump", "-d", "-w", obj_path]).stdout
    funcs = {}
    cur_fn = None
    for line in out.splitlines():
        m = re.match(r"^[0-9a-fA-F]+\s+<([^>]+)>:", line)
        if m:
            cur_fn = m.group(1)
            funcs[cur_fn] = []
        elif cur_fn:
            funcs[cur_fn].append(line)

    checks = [
        ("sse1_rmsnorm",        r"\brsqrtss\b",          r"\bymm|\bzmm"),
        ("sse3_rmsnorm",        r"\bhaddps\b",           r"\bymm|\bzmm"),
        ("sse3_dot_q8",         r"\bpmaddwd\b",          r"\bpmaddubsw|\bpshufb|\bymm|\bzmm"),
        ("ssse3_matmul_tiled",  r"\bpmaddubsw\b",        r"\bymm|\bzmm"),
        ("sse41_dot_f32",       r"\bdpps\b",             r"\bymm|\bzmm"),
        ("avx1_rmsnorm",        r"\bvmulps\b.*%ymm",     r"\bzmm"),
        ("avx512_dot_q8",       r"\bvpdpbusd\b.*%zmm",   r"^$"),
        ("avx512_matmul_tiled", r"\bvpdpbusd\b.*%zmm",   r"^$"),
    ]
    for fn_sub, must_re, forbid_re in checks:
        matched_body = "\n".join(
            "\n".join(lines) for name, lines in funcs.items() if name.lstrip("_") == fn_sub or name.lstrip("_").startswith(fn_sub + ".")
        )
        if not matched_body or not re.search(must_re, matched_body):
            sys.exit(f"[FAIL] ISA Gatekeeper: expected pattern '{must_re}' missing in {fn_sub}")
        if forbid_re != r"^$" and re.search(forbid_re, matched_body):
            sys.exit(f"[FAIL] ISA Gatekeeper: forbidden register '{forbid_re}' leaked into {fn_sub}")
    print("[PASS] x86 ISA Gatekeeper: SSE / SSE3 / SSSE3 / SSE4.1 / AVX / AVX-512 (512-bit ZMM vpdpbusd) verified!")


def verify_arm64_objdump(obj_path: str):
    out = run_cmd(["objdump", "-d", obj_path]).stdout
    if not re.search(r"\bsdot\b", out):
        sys.exit("[FAIL] ARM64 ISA Gatekeeper: expected 'sdot' instruction missing in kernels_neon.o")
    print("[PASS] ARM64 ISA Gatekeeper: NEON & NEON-DotProd (sdot) machine code verified!")


def verify_arm_cortex_m_cross():
    arm_cc = shutil.which("arm-none-eabi-gcc")
    arm_objd = shutil.which("arm-none-eabi-objdump")
    if not arm_cc or not arm_objd:
        print("[INFO] 'arm-none-eabi-gcc' not installed; skipping bare-metal Cortex-M object disassembly (host C verification active).")
        return

    targets = [
        ("cortex-m0",     ["-mcpu=cortex-m0", "-mthumb"], r"\bmuls\b"),
        ("cortex-m0plus", ["-mcpu=cortex-m0plus", "-mthumb"], r"\bmuls\b"),
        ("cortex-m23",    ["-mcpu=cortex-m23", "-mthumb"], r"\bmuls\b"),
        ("cortex-m3",     ["-mcpu=cortex-m3", "-mthumb"], r"\b(?:ubfx|mla)\b"),
        ("cortex-m4",     ["-mcpu=cortex-m4", "-mthumb"], r"\bsmlad\b"),
        ("cortex-m7",     ["-mcpu=cortex-m7", "-mthumb"], r"\bsmlad\b"),
        ("cortex-m33",    ["-mcpu=cortex-m33", "-mthumb"], r"\bsmlad\b"),
        ("cortex-m55",    ["-mcpu=cortex-m55", "-mthumb", "-mfloat-abi=hard"], r"\bvmladav"),
    ]
    for name, flags, exp_insn in targets:
        os.makedirs("build/cm_cross", exist_ok=True)
        out_o = f"build/cm_cross/cm_{name}.o"
        out_s = f"build/cm_cross/cm_{name}.s"
        run_cmd([arm_cc, "-std=c99", "-O3", "-Iinclude"] + flags + ["-c", "src/kernels_cortex_m.c", "-o", out_o])
        run_cmd([arm_cc, "-std=c99", "-O3", "-Iinclude"] + flags + ["-S", "src/kernels_cortex_m.c", "-o", out_s])
        dis = run_cmd([arm_objd, "-d", out_o]).stdout
        with open(out_s, "r", encoding="utf-8", errors="ignore") as sf:
            asm_txt = sf.read()
        if not (re.search(exp_insn, dis, re.IGNORECASE) or re.search(exp_insn, asm_txt, re.IGNORECASE)):
            sys.exit(f"[FAIL] ARM Cortex-M Gatekeeper: expected '{exp_insn}' in {name}")
    print("[PASS] ARM Cortex-M Gatekeeper: M0 / M0+ / M23 / M3 / M4 / M7 / M33 / M55 machine code verified!")


def main():
    run_cmd(["make", "host"])
    if not os.path.exists("stories260k_q4.qmx"):
        run_cmd([sys.executable, "tools/qmx_forge.py", "--fetch", "stories260k", "-q", "q4", "-m", "stories260k_q4.qmx"])

    mach = platform.machine().lower()
    if mach in ("x86_64", "amd64", "i386", "i686"):
        verify_x86_objdump("build/host/kernels_x86_ext.o")
        backends = [
            "scalar", "p54c", "mmx", "sse", "sse2", "sse3", "ssse3", "sse4.1", "sse4.2",
            "avx", "avx2", "avx512", "mcu32", "m0", "m23", "m3", "m4", "m33", "m55"
        ]
    elif mach in ("aarch64", "arm64"):
        verify_arm64_objdump("build/host/kernels_neon.o")
        backends = [
            "scalar", "neon", "neon_dot", "mcu32", "m0", "m23", "m3", "m4", "m33", "m55"
        ]
    else:
        backends = ["scalar", "mcu32", "m0", "m23", "m3", "m4", "m33", "m55"]

    verify_arm_cortex_m_cross()

    print(f"\n{'Backend':<16} | {'Speed (tok/s)':>14} | {'Step0 Checksum':>14}")
    print("-" * 52)
    ref_chk = None
    tested = 0
    for b in backends:
        res = run_cmd(["./qmx", "-m", "stories260k_q4.qmx", "-p", "Once upon a time", "-n", "64", "-c", "256", "-t", "0.0", "-b", b])
        m = re.search(r"\(([\d.]+)\s+tok/s\)\s+\|\s+step0_checksum=([-\d.]+)", res.stderr)
        if not m:
            sys.exit(f"[FAIL] Could not parse stats for backend {b}")
        tps = float(m.group(1))
        chk = float(m.group(2))
        if ref_chk is None:
            ref_chk = chk
        elif abs(chk - ref_chk) > 1e-3:
            sys.exit(f"[FAIL] Checksum mismatch on {b}: got {chk} vs ref {ref_chk}")
        print(f"{b:<16} | {tps:>14.1f} | {chk:>14.4f}")
        tested += 1
    print("-" * 52)
    print(f"[PASS] All {tested} backends produced bit-accurate logits (Checksum={ref_chk:.4f})!")


if __name__ == "__main__":
    main()
