#!/usr/bin/env python3
import json
import os
import struct
import subprocess
import sys
import tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import qmx_pack as pack


def dequantize_tiled(scales_b: bytes, data_b: bytes, rows: int, cols: int, qtype: int) -> np.ndarray:
    pad_r = pack.align_up(rows, pack.TILE_ROWS)
    pad_c = pack.align_up(cols, pack.TILE_COLS)
    n_tr = pad_r // pack.TILE_ROWS
    n_tc = pad_c // pack.TILE_COLS
    num_tiles = n_tr * n_tc

    if qtype in (pack.QMX_QUANT_Q4_ROW32, pack.QMX_QUANT_Q8_ROW32):
        sc = np.frombuffer(scales_b, dtype=np.float16).astype(np.float32).reshape(num_tiles, pack.TILE_ROWS, 1)
    else:
        sc = np.frombuffer(scales_b, dtype=np.float32).reshape(num_tiles, 1, 1)

    raw = np.frombuffer(data_b, dtype=np.uint8)
    if qtype in (pack.QMX_QUANT_Q4_SYM, pack.QMX_QUANT_Q4_ROW32):
        raw = raw.reshape(num_tiles, pack.TILE_ROWS, 16)
        lo = (raw & 0x0F).astype(np.int32) - 8
        hi = ((raw >> 4) & 0x0F).astype(np.int32) - 8
        q = np.empty((num_tiles, pack.TILE_ROWS, pack.TILE_COLS), dtype=np.float32)
        q[:, :, 0::2] = lo
        q[:, :, 1::2] = hi
    elif qtype == pack.QMX_QUANT_TERNARY_158:
        raw = raw.reshape(num_tiles, pack.TILE_ROWS, 8)
        lut = np.array([0.0, 1.0, -1.0, 0.0], dtype=np.float32)
        q = np.empty((num_tiles, pack.TILE_ROWS, pack.TILE_COLS), dtype=np.float32)
        for k in range(4):
            q[:, :, k::4] = lut[(raw >> (2 * k)) & 0x03]
    elif qtype in (pack.QMX_QUANT_Q8_SYM, pack.QMX_QUANT_Q8_ROW32):
        q = raw.view(np.int8).astype(np.float32).reshape(num_tiles, pack.TILE_ROWS, pack.TILE_COLS)
    else:
        raise ValueError(f"Unsupported qtype {qtype}")

    dq = (q * sc).reshape(n_tr, n_tc, pack.TILE_ROWS, pack.TILE_COLS).transpose(0, 2, 1, 3)
    return dq.reshape(pad_r, pad_c)[:rows, :cols]


def test_imatrix_weighted_error_reduction():
    rng = np.random.default_rng(2026)
    rows, cols = 32, 64
    mat = rng.normal(0.0, 0.35, size=(rows, cols)).astype(np.float32)
    imp = np.full(cols, 100.0, dtype=np.float32)
    for c in (0, 32):
        mat[:, c] = rng.choice([-4.5, 4.5], size=rows).astype(np.float32)
        imp[c] = 1e-3

    for qtype, name in (
        (pack.QMX_QUANT_Q4_ROW32, "Q4_ROW32"),
        (pack.QMX_QUANT_Q4_SYM, "Q4_SYM"),
        (pack.QMX_QUANT_TERNARY_158, "TERNARY_158"),
    ):
        sc_u, dat_u, _, _ = pack.quantize_matrix_tiled(mat, qtype, False, importance=None)
        sc_w, dat_w, _, _ = pack.quantize_matrix_tiled(mat, qtype, False, importance=imp)
        dq_u = dequantize_tiled(sc_u, dat_u, rows, cols, qtype)
        dq_w = dequantize_tiled(sc_w, dat_w, rows, cols, qtype)
        err_u = float(np.sum(((mat - dq_u) ** 2) * imp[None, :]))
        err_w = float(np.sum(((mat - dq_w) ** 2) * imp[None, :]))
        assert err_w < err_u * 0.5, f"{name}: imatrix did not reduce weighted error ({err_w:.4f} vs {err_u:.4f})"


def test_imatrix_formats_and_aliases():
    with tempfile.TemporaryDirectory() as td:
        vec = np.linspace(1.0, 64.0, 64, dtype=np.float32)

        jp = os.path.join(td, "im.json")
        with open(jp, "w", encoding="utf-8") as f:
            json.dump({
                "model.layers.0.self_attn.q_proj.weight": vec.tolist(),
                "blk.0.attn_output.weight": {"values": (vec * 4.0).tolist(), "ncall": 4},
                "output.weight": vec.tolist(),
            }, f)
        im_j = pack.load_imatrix(jp)
        r_q = pack.resolve_imatrix_vector(im_j, "blk.0.attn_q", 64)
        r_o = pack.resolve_imatrix_vector(im_j, "blk.0.attn_wo", 64)
        r_h = pack.resolve_imatrix_vector(im_j, "lm_head.weight", 64)
        assert r_q is not None and np.allclose(r_q, vec)
        assert r_o is not None and np.allclose(r_o, vec)
        assert r_h is not None and np.allclose(r_h, vec)
        assert pack.resolve_imatrix_vector(im_j, "blk.0.attn_q", 32) is None

        dp = os.path.join(td, "im.dat")
        name_b = b"blk.0.ffn_down.weight"
        with open(dp, "wb") as f:
            f.write(struct.pack("<i", 1))
            f.write(struct.pack("<i", len(name_b)))
            f.write(name_b)
            f.write(struct.pack("<ii", 2, 64))
            f.write((vec * 2.0).tobytes())
        im_d = pack.load_imatrix(dp)
        r_d = pack.resolve_imatrix_vector(im_d, "blk.0.ffn_down", 64)
        assert r_d is not None and np.allclose(r_d, vec)

        npzp = os.path.join(td, "im.npz")
        np.savez(npzp, **{"blk.1.attn_k": vec})
        im_n = pack.load_imatrix(npzp)
        assert np.allclose(pack.resolve_imatrix_vector(im_n, "blk.1.attn_k", 64), vec)

        tp = os.path.join(td, "im.txt")
        with open(tp, "w", encoding="utf-8") as f:
            f.write("blk.1.attn_v: " + " ".join(f"{x:.4f}" for x in vec) + "\n")
        im_t = pack.load_imatrix(tp)
        assert np.allclose(pack.resolve_imatrix_vector(im_t, "blk.1.attn_v", 64), vec, atol=1e-3)

        out_qmx = os.path.join(td, "dummy_im.qmx")
        subprocess.run(
            [sys.executable, "tools/qmx_pack.py", "--dummy", "-q", "q4_r32", "--imatrix", jp, "-o", out_qmx],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if os.path.exists("./qmx"):
            res = subprocess.run(
                ["./qmx", "-m", out_qmx, "-p", "Hello", "-n", "8", "--verify-crc"],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            assert "[CRC] Verified OK" in res.stderr


def _gguf_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _gguf_kv(key: str, vtype: int, val) -> bytes:
    out = bytearray(_gguf_str(key))
    out.extend(struct.pack("<I", vtype))
    if vtype == pack.GGUF_VAL_STRING:
        out.extend(_gguf_str(val))
    elif vtype == pack.GGUF_VAL_UINT32:
        out.extend(struct.pack("<I", int(val)))
    elif vtype == pack.GGUF_VAL_FLOAT32:
        out.extend(struct.pack("<f", float(val)))
    elif vtype == pack.GGUF_VAL_ARRAY:
        etype, items = val
        out.extend(struct.pack("<IQ", etype, len(items)))
        if etype == pack.GGUF_VAL_STRING:
            for it in items:
                out.extend(_gguf_str(it))
        elif etype == pack.GGUF_VAL_INT32:
            out.extend(np.asarray(items, dtype=np.int32).tobytes())
        elif etype == pack.GGUF_VAL_FLOAT32:
            out.extend(np.asarray(items, dtype=np.float32).tobytes())
        else:
            raise ValueError(f"Unsupported test array etype {etype}")
    else:
        raise ValueError(f"Unsupported test kv type {vtype}")
    return bytes(out)


def _encode_ggml(arr: np.ndarray, ggml_type: int) -> bytes:
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    if ggml_type == pack.GGML_TYPE_F32:
        return arr.tobytes()
    if ggml_type == pack.GGML_TYPE_F16:
        return arr.astype(np.float16).tobytes()
    if ggml_type == pack.GGML_TYPE_BF16:
        u32 = arr.view(np.uint32)
        return ((u32 >> 16) & 0xFFFF).astype(np.uint16).tobytes()
    flat = arr.reshape(-1, 32)
    if ggml_type == pack.GGML_TYPE_Q8_0:
        max_abs = np.max(np.abs(flat), axis=1, keepdims=True)
        sc = (max_abs / 127.0).astype(np.float16)
        sc_f32 = sc.astype(np.float32)
        inv = np.where(sc_f32 == 0.0, 0.0, 1.0 / sc_f32)
        qs = np.clip(np.round(flat * inv), -127.0, 127.0).astype(np.int8)
        out = np.empty((flat.shape[0], 34), dtype=np.uint8)
        out[:, :2] = sc.view(np.uint8)
        out[:, 2:] = qs.view(np.uint8)
        return out.tobytes()
    if ggml_type == pack.GGML_TYPE_Q4_0:
        max_abs = np.max(np.abs(flat), axis=1, keepdims=True)
        sc = (max_abs / 7.0).astype(np.float16)
        sc_f32 = sc.astype(np.float32)
        inv = np.where(sc_f32 == 0.0, 0.0, 1.0 / sc_f32)
        q = np.clip(np.round(flat * inv), -8.0, 7.0).astype(np.int32) + 8
        lo = q[:, :16].astype(np.uint8)
        hi = q[:, 16:].astype(np.uint8)
        out = np.empty((flat.shape[0], 18), dtype=np.uint8)
        out[:, :2] = sc.view(np.uint8)
        out[:, 2:] = lo | (hi << 4)
        return out.tobytes()
    raise ValueError(f"Unsupported test ggml_type {ggml_type}")


def write_test_gguf(path: str, kvs: list, tensor_list: list, alignment: int = 32):
    blob = bytearray()
    infos = []
    for name, arr, gtype in tensor_list:
        rem = len(blob) % alignment
        if rem:
            blob.extend(b"\x00" * (alignment - rem))
        off = len(blob)
        blob.extend(_encode_ggml(arr, gtype))
        dims = tuple(reversed(arr.shape))
        infos.append((name, dims, gtype, off))

    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<IQQ", 3, len(infos), len(kvs)))
        for k, vt, val in kvs:
            f.write(_gguf_kv(k, vt, val))
        for name, dims, gtype, off in infos:
            f.write(_gguf_str(name))
            f.write(struct.pack("<I", len(dims)))
            for d in dims:
                f.write(struct.pack("<Q", int(d)))
            f.write(struct.pack("<IQ", gtype, off))
        rem = f.tell() % alignment
        if rem:
            f.write(b"\x00" * (alignment - rem))
        f.write(blob)


def test_gguf_import_all_types_and_e2e():
    rng = np.random.default_rng(777)
    dim, inter, n_layers, n_heads, n_kv, vocab, ctx = 64, 128, 2, 4, 4, 256, 128

    tokens = [f"<0x{i:02X}>" if i < 32 or i >= 127 else chr(i) for i in range(vocab)]
    tokens[0], tokens[1], tokens[2] = "<unk>", "<s>", "</s>"
    tok_types = [1] * vocab
    tok_types[0] = tok_types[1] = tok_types[2] = 3

    kvs = [
        ("general.architecture", pack.GGUF_VAL_STRING, "llama"),
        ("general.alignment", pack.GGUF_VAL_UINT32, 32),
        ("llama.embedding_length", pack.GGUF_VAL_UINT32, dim),
        ("llama.feed_forward_length", pack.GGUF_VAL_UINT32, inter),
        ("llama.block_count", pack.GGUF_VAL_UINT32, n_layers),
        ("llama.attention.head_count", pack.GGUF_VAL_UINT32, n_heads),
        ("llama.attention.head_count_kv", pack.GGUF_VAL_UINT32, n_kv),
        ("llama.context_length", pack.GGUF_VAL_UINT32, ctx),
        ("llama.rope.freq_base", pack.GGUF_VAL_FLOAT32, 10000.0),
        ("llama.attention.layer_norm_rms_epsilon", pack.GGUF_VAL_FLOAT32, 1e-5),
        ("tokenizer.ggml.model", pack.GGUF_VAL_STRING, "llama"),
        ("tokenizer.ggml.tokens", pack.GGUF_VAL_ARRAY, (pack.GGUF_VAL_STRING, tokens)),
        ("tokenizer.ggml.token_type", pack.GGUF_VAL_ARRAY, (pack.GGUF_VAL_INT32, tok_types)),
        ("tokenizer.ggml.bos_token_id", pack.GGUF_VAL_UINT32, 1),
        ("tokenizer.ggml.eos_token_id", pack.GGUF_VAL_UINT32, 2),
    ]

    emb = rng.normal(0.0, 0.05, (vocab, dim)).astype(np.float32)
    rms_f = np.ones(dim, dtype=np.float32)
    tensors = [
        ("token_embd.weight", emb, pack.GGML_TYPE_F16),
        ("output_norm.weight", rms_f, pack.GGML_TYPE_F32),
    ]
    ref_mats = {}
    type_cycle = [pack.GGML_TYPE_F32, pack.GGML_TYPE_F16, pack.GGML_TYPE_BF16, pack.GGML_TYPE_Q8_0, pack.GGML_TYPE_Q4_0]
    t_i = 0
    for l in range(n_layers):
        tensors.append((f"blk.{l}.attn_norm.weight", np.ones(dim, dtype=np.float32), pack.GGML_TYPE_F32))
        tensors.append((f"blk.{l}.ffn_norm.weight", np.ones(dim, dtype=np.float32), pack.GGML_TYPE_F32))
        for role, shp in (
            ("attn_q", (dim, dim)),
            ("attn_k", (dim, dim)),
            ("attn_v", (dim, dim)),
            ("attn_output", (dim, dim)),
            ("ffn_gate", (inter, dim)),
            ("ffn_up", (inter, dim)),
            ("ffn_down", (dim, inter)),
        ):
            m = rng.normal(0.0, 0.05, shp).astype(np.float32)
            gt = type_cycle[t_i % len(type_cycle)]
            t_i += 1
            tname = f"blk.{l}.{role}.weight"
            tensors.append((tname, m, gt))
            ref_mats[tname] = (m, gt)

    with tempfile.TemporaryDirectory() as td:
        gguf_path = os.path.join(td, "micro.gguf")
        write_test_gguf(gguf_path, kvs, tensors)

        g = pack.NativeGGUF(gguf_path)
        for tname, (orig, gt) in ref_mats.items():
            got = g.get_tensor(tname)
            assert got.shape == orig.shape, f"Shape mismatch for {tname}: {got.shape} vs {orig.shape}"
            tol = {
                pack.GGML_TYPE_F32: 1e-6,
                pack.GGML_TYPE_F16: 1e-3,
                pack.GGML_TYPE_BF16: 2e-3,
                pack.GGML_TYPE_Q8_0: 3e-3,
                pack.GGML_TYPE_Q4_0: 3.5e-2,
            }[gt]
            max_err = float(np.max(np.abs(got - orig)))
            assert max_err < tol, f"Dequant error too high for {tname} (type={gt}): {max_err} >= {tol}"

        im_gguf = os.path.join(td, "im.gguf")
        im_vec = np.linspace(1.0, 64.0, dim, dtype=np.float32)
        write_test_gguf(
            im_gguf,
            [("general.architecture", pack.GGUF_VAL_STRING, "imatrix")],
            [
                ("blk.0.attn_q.weight.in_sum2", im_vec * 3.0, pack.GGML_TYPE_F32),
                ("blk.0.attn_q.weight.counts", np.full(dim, 3.0, dtype=np.float32), pack.GGML_TYPE_F32),
            ],
        )
        im_loaded = pack.load_imatrix(im_gguf)
        res_v = pack.resolve_imatrix_vector(im_loaded, "blk.0.attn_q", dim)
        assert res_v is not None and np.allclose(res_v, im_vec, atol=1e-5)

        out_qmx = os.path.join(td, "from_gguf.qmx")
        subprocess.run(
            [sys.executable, "tools/qmx_pack.py", "--gguf", gguf_path, "-q", "q4_r32", "--imatrix", im_gguf, "-o", out_qmx],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if os.path.exists("./qmx"):
            res = subprocess.run(
                ["./qmx", "-m", out_qmx, "-p", "Hello", "-n", "8", "-t", "0.0", "--verify-crc"],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            assert "[CRC] Verified OK" in res.stderr
            assert "step0_checksum=" in res.stderr


def main():
    test_imatrix_weighted_error_reduction()
    test_imatrix_formats_and_aliases()
    test_gguf_import_all_types_and_e2e()
    print("[PASS] Packer imatrix + GGUF (F32/F16/BF16/Q8_0/Q4_0, imatrix.gguf, e2e CRC) tests passed!")


if __name__ == "__main__":
    main()
