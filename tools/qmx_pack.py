#!/usr/bin/env python3
"""
QMX Forge Packer v1.1a
Converts HuggingFace SafeTensors (LLaMA, SmolLM, BitNet b1.58, Qwen-2/3, Gemma-2/3, Phi-3/4),
Karpathy llama2.c .bin checkpoints, or synthetic test models into hardware-aligned
.qmx v1.1 cartridges with 64-bit streaming payload writes, Tile-Major 16x32 layout,
per-tile FP32 or per-row FP16 scales, sorted 32-bit BPE merge tables, and Layer Recipe VM bytecode.
"""

import argparse
import gc
import json
import math
import os
import struct
import sys
import zlib
import numpy as np

QMX_MAGIC = 0x31584D51           # "QMX1"
QMX_TOKENIZER_MAGIC = 0x4E4B4F54 # "TOKN"
QMX_VERSION_1_0 = 0x0100
QMX_VERSION_1_1 = 0x0101
QMX_VERSION = QMX_VERSION_1_1

CACHE_ALIGN = 64
SECTOR_ALIGN = 512
PAGE_ALIGN = 4096

TILE_ROWS = 16
TILE_COLS = 32

# Global cartridge flags
QMX_FLAG_TILE_MAJOR        = 1 << 0
QMX_FLAG_SHARED_CLASSIFIER = 1 << 1
QMX_FLAG_ROPE_SPLIT_HALF   = 1 << 2
QMX_FLAG_LLAMA3_ROPE_SCALE = 1 << 3
QMX_FLAG_HAS_BIAS          = 1 << 4
QMX_FLAG_LOGIT_SOFTCAP     = 1 << 5
QMX_FLAG_COMPACT_ALIGN     = 1 << 6
QMX_FLAG_ATTN_SOFTCAP      = 1 << 7

# Quantization profiles
QMX_QUANT_BINARY_1BIT = 0x01
QMX_QUANT_TERNARY_158 = 0x02
QMX_QUANT_Q4_SYM      = 0x03
QMX_QUANT_Q8_SYM      = 0x04
QMX_QUANT_FP32        = 0x05
QMX_QUANT_Q4_ROW32    = 0x06
QMX_QUANT_Q8_ROW32    = 0x07

QMX_ROPE_SCALE_NONE   = 0
QMX_ROPE_SCALE_LINEAR = 1
QMX_ROPE_SCALE_LLAMA3 = 2
QMX_ROPE_SCALE_YARN   = 3

QUANT_NAMES = {
    QMX_QUANT_BINARY_1BIT: "1.0-bit Binary (64B/tile)",
    QMX_QUANT_TERNARY_158: "1.58-bit Ternary (128B/tile)",
    QMX_QUANT_Q4_SYM:      "4-bit Symmetric Q4 (256B/tile)",
    QMX_QUANT_Q8_SYM:      "8-bit Symmetric Q8 (512B/tile)",
    QMX_QUANT_FP32:        "32-bit Float (Vector)",
    QMX_QUANT_Q4_ROW32:    "4-bit Row-Scaled Q4_R32 (256B + 16xF16/tile)",
    QMX_QUANT_Q8_ROW32:    "8-bit Row-Scaled Q8_R32 (512B + 16xF16/tile)",
}

# Memory retention tiers
QMX_PIN_CRITICAL     = 0x00
QMX_PIN_LATENCY_SENS = 0x01
QMX_PIN_STREAM_ONLY  = 0x02

# Layer Recipe VM Opcodes
QMX_OP_NOP          = 0x00
QMX_OP_RMSNORM      = 0x01
QMX_OP_SUBNORM      = 0x02
QMX_OP_QUANT_ACT    = 0x03
QMX_OP_MATMUL       = 0x04
QMX_OP_MATMUL_ADD   = 0x05
QMX_OP_ADD_BIAS     = 0x06
QMX_OP_ROPE         = 0x07
QMX_OP_ATTN         = 0x08
QMX_OP_SWIGLU       = 0x09
QMX_OP_GEGLU        = 0x0A
QMX_OP_RELU2        = 0x0B
QMX_OP_RESIDUAL_ADD = 0x0C
QMX_OP_LAYERNORM    = 0x0D
QMX_OP_QK_NORM      = 0x0E
QMX_OP_SCALE_MUL    = 0x0F
QMX_OP_SILU         = 0x10

OPCODE_NAMES = {
    QMX_OP_NOP:          "NOP",
    QMX_OP_RMSNORM:      "RMSNORM",
    QMX_OP_SUBNORM:      "SUBNORM",
    QMX_OP_QUANT_ACT:    "QUANT_ACT",
    QMX_OP_MATMUL:       "MATMUL",
    QMX_OP_MATMUL_ADD:   "MATMUL_ADD",
    QMX_OP_ADD_BIAS:     "ADD_BIAS",
    QMX_OP_ROPE:         "ROPE",
    QMX_OP_ATTN:         "ATTN",
    QMX_OP_SWIGLU:       "SWIGLU",
    QMX_OP_GEGLU:        "GEGLU",
    QMX_OP_RELU2:        "RELU2",
    QMX_OP_RESIDUAL_ADD: "RESIDUAL_ADD",
    QMX_OP_LAYERNORM:    "LAYERNORM",
    QMX_OP_QK_NORM:      "QK_NORM",
    QMX_OP_SCALE_MUL:    "SCALE_MUL",
    QMX_OP_SILU:         "SILU",
}

# Arena slots
BUF_X       = 0
BUF_XB      = 1
BUF_XB2     = 2
BUF_Q       = 3
BUF_K       = 4
BUF_V       = 5
BUF_HB_GATE = 6
BUF_HB_UP   = 7

# Standard Tensor IDs
TID_NONE           = 0xFFFF
TID_TOKEN_EMB      = 0x0001
TID_RMS_FINAL      = 0x0002
TID_LM_HEAD        = 0x0003
TID_RMS_FINAL_BIAS = 0x0004

TID_ATTN_NORM      = 0x0100
TID_ATTN_Q         = 0x0101
TID_ATTN_K         = 0x0102
TID_ATTN_V         = 0x0103
TID_ATTN_WO        = 0x0104
TID_ATTN_SUBNORM   = 0x0105
TID_ATTN_Q_BIAS    = 0x0106
TID_ATTN_K_BIAS    = 0x0107
TID_ATTN_V_BIAS    = 0x0108
TID_ATTN_O_BIAS    = 0x0109
TID_ATTN_NORM_BIAS = 0x010A
TID_ATTN_Q_NORM    = 0x010B
TID_ATTN_K_NORM    = 0x010C
TID_ATTN_POST_NORM = 0x010D

TID_FFN_NORM       = 0x0200
TID_FFN_GATE       = 0x0201
TID_FFN_UP         = 0x0202
TID_FFN_DOWN       = 0x0203
TID_FFN_SUBNORM    = 0x0204
TID_FFN_NORM_BIAS  = 0x0205
TID_FFN_POST_NORM  = 0x0206

HEADER_FMT = "<IHHIIIIIIIIIIIIIIHHHHffIIIHHffffBBH12s"
DESC_FMT = "<32sHHBBHIIIIIHH"
assert struct.calcsize(HEADER_FMT) == 128
assert struct.calcsize(DESC_FMT) == 64


def align_up(val: int, alignment: int) -> int:
    rem = val % alignment
    return val if rem == 0 else val + (alignment - rem)


def pad_file(fp, alignment: int) -> int:
    pos = fp.tell()
    rem = pos % alignment
    if rem != 0:
        fp.write(b"\x00" * (alignment - rem))
        return pos + (alignment - rem)
    return pos


def quantize_matrix_tiled(mat: np.ndarray, quant_type: int, per_tensor_scale: bool = False):
    """
    Pads a 2D float32 matrix to (16, 32) multiples, arranges into Tile-Major 16x32
    blocks, and returns (scales_bytes, data_bytes, padded_rows, padded_cols).
    """
    assert mat.ndim == 2
    orig_rows, orig_cols = mat.shape
    pad_r = (TILE_ROWS - (orig_rows % TILE_ROWS)) % TILE_ROWS
    pad_c = (TILE_COLS - (orig_cols % TILE_COLS)) % TILE_COLS
    if pad_r > 0 or pad_c > 0:
        mat = np.pad(mat, ((0, pad_r), (0, pad_c)), mode="constant")

    rows, cols = mat.shape
    n_tr = rows // TILE_ROWS
    n_tc = cols // TILE_COLS

    # Reshape into (n_tr, n_tc, 16, 32) -> (num_tiles, 512)
    tiles = mat.reshape(n_tr, TILE_ROWS, n_tc, TILE_COLS).transpose(0, 2, 1, 3).reshape(-1, 512)
    num_tiles = tiles.shape[0]

    global_gamma = max(float(np.mean(np.abs(tiles))), 1e-6) if per_tensor_scale else 0.0

    scales_list = []
    data_list = []
    chunk_tiles = 4096

    for start in range(0, num_tiles, chunk_tiles):
        blk = tiles[start : start + chunk_tiles].astype(np.float32, copy=False)

        if quant_type == QMX_QUANT_BINARY_1BIT:
            if per_tensor_scale:
                sc = np.full((blk.shape[0], 1), global_gamma, dtype=np.float32)
            else:
                sc = np.maximum(np.mean(np.abs(blk), axis=1, keepdims=True), 1e-8).astype(np.float32)
            bits = (blk >= 0.0).astype(np.uint8).reshape(-1, 8)
            packed = np.zeros(bits.shape[0], dtype=np.uint8)
            for k in range(8):
                packed |= (bits[:, k] << k)
            scales_list.append(sc.reshape(-1).tobytes())
            data_list.append(packed.tobytes())

        elif quant_type == QMX_QUANT_TERNARY_158:
            if per_tensor_scale:
                sc = np.full((blk.shape[0], 1), global_gamma, dtype=np.float32)
                best_q = np.clip(np.round(blk / sc), -1.0, 1.0).astype(np.int8)
            else:
                abs_blk = np.abs(blk)
                max_abs = np.max(abs_blk, axis=1, keepdims=True)
                best_score = np.full((blk.shape[0], 1), -1.0, dtype=np.float32)
                best_sc = np.maximum(np.mean(abs_blk, axis=1, keepdims=True), 1e-8).astype(np.float32)
                best_q = np.clip(np.round(blk / best_sc), -1.0, 1.0).astype(np.int8)

                for beta in np.linspace(0.14, 0.62, 13, dtype=np.float32):
                    thr = max_abs * beta
                    mask = abs_blk > thr
                    nnz = np.sum(mask, axis=1, keepdims=True).astype(np.float32)
                    wq = np.sum(abs_blk * mask, axis=1, keepdims=True)
                    sc_cand = np.where(nnz > 0.0, wq / nnz, 0.0)
                    score = wq * sc_cand
                    improve = score > best_score
                    if np.any(improve):
                        best_score = np.where(improve, score, best_score)
                        best_sc = np.where(improve, sc_cand, best_sc)
                        q_cand = np.where(blk > thr, 1, np.where(blk < -thr, -1, 0)).astype(np.int8)
                        best_q = np.where(improve, q_cand, best_q)

                sc = np.maximum(best_sc, 1e-8).astype(np.float32)

            codes = np.zeros_like(best_q, dtype=np.uint8)
            codes[best_q == 1] = 1
            codes[best_q == -1] = 2
            codes = codes.reshape(-1, 4)
            packed = (
                codes[:, 0]
                | (codes[:, 1] << 2)
                | (codes[:, 2] << 4)
                | (codes[:, 3] << 6)
            ).astype(np.uint8)
            scales_list.append(sc.reshape(-1).tobytes())
            data_list.append(packed.tobytes())

        elif quant_type == QMX_QUANT_Q4_SYM:
            with np.errstate(divide="ignore", invalid="ignore"):
                max_abs = np.max(np.abs(blk), axis=1, keepdims=True)
                best_score = np.full((blk.shape[0], 1), -1.0, dtype=np.float32)
                best_sc = (max_abs / 7.0).astype(np.float32)
                inv_init = np.where(best_sc == 0.0, 0.0, 1.0 / best_sc)
                best_q = np.clip(np.round(blk * inv_init), -8.0, 7.0).astype(np.int8)

                for alpha in np.linspace(0.68, 1.00, 13, dtype=np.float32):
                    sc_try = (max_abs * alpha) / 7.5
                    inv_try = np.where(sc_try == 0.0, 0.0, 1.0 / sc_try)
                    q_try = np.clip(np.round(blk * inv_try), -8.0, 7.0)
                    wq = np.sum(blk * q_try, axis=1, keepdims=True)
                    qq = np.sum(q_try * q_try, axis=1, keepdims=True)
                    sc_opt = np.where(qq > 0.0, wq / qq, 0.0)
                    score = wq * sc_opt
                    improve = score > best_score
                    if np.any(improve):
                        best_score = np.where(improve, score, best_score)
                        best_sc = np.where(improve, sc_opt, best_sc)
                        best_q = np.where(improve, q_try.astype(np.int8), best_q)

            u = (best_q + 8).astype(np.uint8).reshape(-1, 2)
            packed = (u[:, 0] | (u[:, 1] << 4)).astype(np.uint8)
            scales_list.append(best_sc.astype(np.float32).reshape(-1).tobytes())
            data_list.append(packed.tobytes())

        elif quant_type == QMX_QUANT_Q4_ROW32:
            with np.errstate(divide="ignore", invalid="ignore"):
                rows32 = blk.reshape(-1, TILE_COLS)
                max_abs = np.max(np.abs(rows32), axis=1, keepdims=True)
                best_score = np.full((rows32.shape[0], 1), -1.0, dtype=np.float32)
                best_sc = (max_abs / 7.0).astype(np.float32)
                inv_init = np.where(best_sc == 0.0, 0.0, 1.0 / best_sc)
                best_q = np.clip(np.round(rows32 * inv_init), -8.0, 7.0).astype(np.int8)

                for alpha in np.linspace(0.72, 1.00, 9, dtype=np.float32):
                    sc_try = (max_abs * alpha) / 7.5
                    inv_try = np.where(sc_try == 0.0, 0.0, 1.0 / sc_try)
                    q_try = np.clip(np.round(rows32 * inv_try), -8.0, 7.0)
                    wq = np.sum(rows32 * q_try, axis=1, keepdims=True)
                    qq = np.sum(q_try * q_try, axis=1, keepdims=True)
                    sc_opt = np.where(qq > 0.0, wq / qq, 0.0)
                    score = wq * sc_opt
                    improve = score > best_score
                    if np.any(improve):
                        best_score = np.where(improve, score, best_score)
                        best_sc = np.where(improve, sc_opt, best_sc)
                        best_q = np.where(improve, q_try.astype(np.int8), best_q)

            u = (best_q + 8).astype(np.uint8).reshape(-1, 2)
            packed = (u[:, 0] | (u[:, 1] << 4)).astype(np.uint8)
            scales_list.append(best_sc.astype(np.float16).reshape(-1).tobytes())
            data_list.append(packed.tobytes())

        elif quant_type == QMX_QUANT_Q8_SYM:
            with np.errstate(divide="ignore", invalid="ignore"):
                max_abs = np.max(np.abs(blk), axis=1, keepdims=True)
                sc = (max_abs / 127.0).astype(np.float32)
                inv_sc = np.where(sc == 0.0, 0.0, 1.0 / sc)
                q_f = np.clip(np.round(blk * inv_sc), -127.0, 127.0)
                wq = np.sum(blk * q_f, axis=1, keepdims=True)
                qq = np.sum(q_f * q_f, axis=1, keepdims=True)
                sc_opt = np.where(qq > 0.0, wq / qq, sc).astype(np.float32)
            q = q_f.astype(np.int8)
            scales_list.append(sc_opt.reshape(-1).tobytes())
            data_list.append(q.tobytes())

        elif quant_type == QMX_QUANT_Q8_ROW32:
            with np.errstate(divide="ignore", invalid="ignore"):
                rows32 = blk.reshape(-1, TILE_COLS)
                max_abs = np.max(np.abs(rows32), axis=1, keepdims=True)
                sc = (max_abs / 127.0).astype(np.float32)
                inv_sc = np.where(sc == 0.0, 0.0, 1.0 / sc)
                q_f = np.clip(np.round(rows32 * inv_sc), -127.0, 127.0)
                wq = np.sum(rows32 * q_f, axis=1, keepdims=True)
                qq = np.sum(q_f * q_f, axis=1, keepdims=True)
                sc_opt = np.where(qq > 0.0, wq / qq, sc).astype(np.float16)
            q = q_f.astype(np.int8)
            scales_list.append(sc_opt.reshape(-1).tobytes())
            data_list.append(q.tobytes())

        else:
            raise ValueError(f"Unsupported tiled quant_type: {quant_type}")

    return b"".join(scales_list), b"".join(data_list), rows, cols


def build_layer_recipe(config: dict):
    """
    Compiles the 8-byte Layer Recipe VM instruction stream for a transformer layer.
    """
    has_qkv_bias = bool(config.get("has_qkv_bias", False))
    has_o_bias = bool(config.get("has_o_bias", False))
    has_subnorm = bool(config.get("has_subnorm", False))
    has_qk_norm = bool(config.get("has_qk_norm", False))
    has_post_norm = bool(config.get("has_post_norm", False))
    use_layernorm = bool(config.get("use_layernorm", False))
    has_norm_bias = bool(config.get("has_norm_bias", False))
    act_type = config.get("activation", "swiglu")
    rope_flags = 1 if config.get("rope_split_half", False) else 0
    norm_op = QMX_OP_LAYERNORM if use_layernorm else QMX_OP_RMSNORM

    ops = []
    # 1. Attention Norm + Q/K/V Projections
    ops.append((norm_op, 0, TID_ATTN_NORM, BUF_X, BUF_XB))
    if has_norm_bias:
        ops.append((QMX_OP_ADD_BIAS, 0, TID_ATTN_NORM_BIAS, BUF_XB, BUF_XB))
    ops.append((QMX_OP_QUANT_ACT, 0, TID_NONE, BUF_XB, BUF_XB))
    ops.append((QMX_OP_MATMUL, 0, TID_ATTN_Q, BUF_XB, BUF_Q))
    if has_qkv_bias:
        ops.append((QMX_OP_ADD_BIAS, 0, TID_ATTN_Q_BIAS, BUF_Q, BUF_Q))
    if has_qk_norm:
        ops.append((QMX_OP_QK_NORM, 0, TID_ATTN_Q_NORM, BUF_Q, BUF_Q))
    ops.append((QMX_OP_MATMUL, 0, TID_ATTN_K, BUF_XB, BUF_K))
    if has_qkv_bias:
        ops.append((QMX_OP_ADD_BIAS, 0, TID_ATTN_K_BIAS, BUF_K, BUF_K))
    if has_qk_norm:
        ops.append((QMX_OP_QK_NORM, 0, TID_ATTN_K_NORM, BUF_K, BUF_K))
    ops.append((QMX_OP_MATMUL, 0, TID_ATTN_V, BUF_XB, BUF_V))
    if has_qkv_bias:
        ops.append((QMX_OP_ADD_BIAS, 0, TID_ATTN_V_BIAS, BUF_V, BUF_V))

    # 2. RoPE + Causal GQA Attention
    ops.append((QMX_OP_ROPE, rope_flags, TID_NONE, BUF_Q, BUF_K))
    ops.append((QMX_OP_ATTN, 0, TID_NONE, BUF_Q, BUF_XB))

    # 3. Optional BitNet SubNorm before Wo + Output Projection + Optional Post-Attn Norm + Residual
    if has_subnorm:
        sub_tid = TID_ATTN_SUBNORM if config.get("has_subnorm_weights", False) else TID_NONE
        ops.append((QMX_OP_SUBNORM, 0, sub_tid, BUF_XB, BUF_XB))
    ops.append((QMX_OP_QUANT_ACT, 0, TID_NONE, BUF_XB, BUF_XB))
    ops.append((QMX_OP_MATMUL, 0, TID_ATTN_WO, BUF_XB, BUF_XB2))
    if has_o_bias:
        ops.append((QMX_OP_ADD_BIAS, 0, TID_ATTN_O_BIAS, BUF_XB2, BUF_XB2))
    if has_post_norm:
        ops.append((norm_op, 0, TID_ATTN_POST_NORM, BUF_XB2, BUF_XB2))
    ops.append((QMX_OP_RESIDUAL_ADD, 0, TID_NONE, BUF_XB2, BUF_X))

    # 4. FFN Norm + Gate/Up + Non-linearity
    ops.append((norm_op, 0, TID_FFN_NORM, BUF_X, BUF_XB))
    if has_norm_bias:
        ops.append((QMX_OP_ADD_BIAS, 0, TID_FFN_NORM_BIAS, BUF_XB, BUF_XB))
    ops.append((QMX_OP_QUANT_ACT, 0, TID_NONE, BUF_XB, BUF_XB))
    ops.append((QMX_OP_MATMUL, 0, TID_FFN_GATE, BUF_XB, BUF_HB_GATE))
    ops.append((QMX_OP_MATMUL, 0, TID_FFN_UP, BUF_XB, BUF_HB_UP))

    if act_type == "relu2":
        ops.append((QMX_OP_RELU2, 0, TID_NONE, BUF_HB_GATE, BUF_HB_UP))
    elif act_type == "geglu":
        ops.append((QMX_OP_GEGLU, 0, TID_NONE, BUF_HB_GATE, BUF_HB_UP))
    else:
        ops.append((QMX_OP_SWIGLU, 0, TID_NONE, BUF_HB_GATE, BUF_HB_UP))

    # 5. Optional BitNet SubNorm before W2 + Down Projection + Optional Post-FFN Norm + Residual
    if has_subnorm:
        sub_tid = TID_FFN_SUBNORM if config.get("has_subnorm_weights", False) else TID_NONE
        ops.append((QMX_OP_SUBNORM, 0, sub_tid, BUF_HB_UP, BUF_HB_UP))
    ops.append((QMX_OP_QUANT_ACT, 0, TID_NONE, BUF_HB_UP, BUF_HB_UP))
    ops.append((QMX_OP_MATMUL, 0, TID_FFN_DOWN, BUF_HB_UP, BUF_XB2))
    if has_post_norm:
        ops.append((norm_op, 0, TID_FFN_POST_NORM, BUF_XB2, BUF_XB2))
    ops.append((QMX_OP_RESIDUAL_ADD, 0, TID_NONE, BUF_XB2, BUF_X))

    recipe_bytes = bytearray()
    for op, flg, tid, src, dst in ops:
        recipe_bytes.extend(struct.pack("<BBHHH", op, flg, tid, src, dst))
    return bytes(recipe_bytes), len(ops)


def _gpt2_bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def serialize_qmx_tokenizer(vocab_pieces: list, merges_tuples: list, bos_id: int, eos_id: int, eot_id: int, is_byte_level: bool) -> bytes:
    vocab_size = len(vocab_pieces)
    offsets = bytearray()
    pool = bytearray()
    max_len = 1

    for p in vocab_pieces:
        if len(p) > max_len:
            max_len = len(p)
        offsets.extend(struct.pack("<I", len(pool)))
        pool.extend(p)
        pool.append(0)
    while (len(pool) % 4) != 0:
        pool.append(0)

    merges_sorted = sorted(merges_tuples, key=lambda m: (m[0] << 32) | m[1])
    merges_buf = bytearray()
    for left_id, right_id, target_id, rank in merges_sorted:
        merges_buf.extend(struct.pack("<IIII", left_id, right_id, target_id, rank))

    flags = 1 if is_byte_level else 0
    hdr = struct.pack(
        "<IIIIiiiI",
        QMX_TOKENIZER_MAGIC,
        vocab_size,
        len(merges_sorted),
        max_len,
        bos_id,
        eos_id,
        eot_id,
        flags,
    )
    assert len(hdr) == 32

    out = bytearray(hdr)
    out.extend(offsets)
    out.extend(struct.pack("<I", len(pool)))
    out.extend(pool)
    out.extend(merges_buf)
    return bytes(out)


class NativeSafeTensors:
    def __init__(self, path: str):
        self.fp = open(path, "rb")
        hdr_len = struct.unpack("<Q", self.fp.read(8))[0]
        self.header = json.loads(self.fp.read(hdr_len).decode("utf-8"))
        self.data_base = 8 + hdr_len

    def keys(self):
        return [k for k in self.header.keys() if k != "__metadata__"]

    def get_tensor(self, name: str) -> np.ndarray:
        meta = self.header[name]
        dtype = meta["dtype"]
        shape = tuple(meta["shape"])
        start, end = meta["data_offsets"]
        self.fp.seek(self.data_base + start)
        raw = self.fp.read(end - start)
        if dtype == "F32":
            return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
        if dtype == "F16":
            return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
        if dtype == "BF16":
            u32 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
            return u32.view(np.float32).reshape(shape)
        raise ValueError(f"Unsupported safetensors dtype {dtype} for {name}")


def build_tokenizer_from_hf_json(tok_json_path: str, vocab_size: int, cfg_bos=None, cfg_eos=None) -> bytes:
    with open(tok_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model_obj = data.get("model", {})
    vocab_raw = model_obj.get("vocab", {})
    dec_str = json.dumps(data.get("decoder", {})) + json.dumps(data.get("pre_tokenizer", {}))
    is_byte_level = ("ByteLevel" in dec_str) or (len(vocab_raw) >= 49000)
    u2b = _gpt2_bytes_to_unicode() if is_byte_level else None

    def piece_to_bytes(piece_str: str) -> bytes:
        if piece_str.startswith("<0x") and piece_str.endswith(">") and len(piece_str) == 6:
            try:
                return bytes([int(piece_str[3:5], 16)])
            except ValueError:
                return piece_str.encode("utf-8")
        if is_byte_level and u2b is not None:
            out_b = bytearray()
            for ch in piece_str:
                if ch in u2b:
                    out_b.append(u2b[ch])
                else:
                    out_b.extend(ch.encode("utf-8"))
            return bytes(out_b)
        return piece_str.replace("\u2581", " ").replace("\u0120", " ").encode("utf-8")

    id_to_raw = {}
    str_to_id = {}
    if isinstance(vocab_raw, list):
        for idx, item in enumerate(vocab_raw):
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                s = str(item[0])
                id_to_raw[idx] = s
                str_to_id[s] = idx
    elif isinstance(vocab_raw, dict):
        for piece_str, idx in vocab_raw.items():
            i = int(idx)
            s = str(piece_str)
            id_to_raw[i] = s
            str_to_id[s] = i

    bos_id = int(cfg_bos) if isinstance(cfg_bos, int) else -1
    eos_id = -1
    eot_id = -1
    if isinstance(cfg_eos, int):
        eos_id = int(cfg_eos)
    elif isinstance(cfg_eos, list) and len(cfg_eos) > 0:
        eos_id = int(cfg_eos[0])
        if len(cfg_eos) > 1:
            eot_id = int(cfg_eos[1])

    special_contents = set()
    for added in data.get("added_tokens", []):
        aid = int(added.get("id", -1))
        content = str(added.get("content", ""))
        if 0 <= aid < vocab_size:
            id_to_raw[aid] = content
            str_to_id[content] = aid
            if added.get("special", False):
                special_contents.add(content)
        if bos_id < 0 and content in ("<s>", "<|begin_of_text|>", "<bos>"):
            bos_id = aid
        elif content in ("</s>", "<|end_of_text|>", "<|endoftext|>", "<eos>"):
            if eos_id < 0:
                eos_id = aid
        elif content in ("<|eot_id|>", "<|im_end|>", "<end_of_turn>", "<|end|>"):
            eot_id = aid
    if eos_id < 0:
        eos_id = eot_id if eot_id >= 0 else 2

    vocab_pieces = []
    for i in range(vocab_size):
        raw_s = id_to_raw.get(i, "")
        if raw_s == "":
            vocab_pieces.append(b"")
        elif raw_s in special_contents or (raw_s.startswith("<|") and raw_s.endswith("|>")) or raw_s in (
            "<s>", "</s>", "<unk>", "<pad>", "<bos>", "<eos>", "<start_of_turn>", "<end_of_turn>", "<think>", "</think>"
        ):
            vocab_pieces.append(raw_s.encode("utf-8"))
        else:
            vocab_pieces.append(piece_to_bytes(raw_s))

    merges_tuples = []
    seen_pairs = set()
    for rank, m in enumerate(model_obj.get("merges", [])):
        if isinstance(m, str):
            parts = m.split(" ", 1)
            if len(parts) != 2:
                continue
            p1, p2 = parts[0], parts[1]
        elif isinstance(m, (list, tuple)) and len(m) >= 2:
            p1, p2 = str(m[0]), str(m[1])
        else:
            continue

        id1 = str_to_id.get(p1)
        id2 = str_to_id.get(p2)
        idt = str_to_id.get(p1 + p2)
        if id1 is not None and id2 is not None and idt is not None:
            pair_key = (id1 << 32) | id2
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                merges_tuples.append((id1, id2, idt, rank))

    return serialize_qmx_tokenizer(vocab_pieces, merges_tuples, bos_id, eos_id, eot_id, is_byte_level)


def build_tokenizer_from_karpathy_bin(bin_path: str, vocab_size: int) -> bytes:
    with open(bin_path, "rb") as f:
        _max_len = struct.unpack("<I", f.read(4))[0]
        raw_items = []
        for _ in range(vocab_size):
            hdr = f.read(8)
            if len(hdr) < 8:
                raw_items.append((b"", -1e10))
                continue
            score, slen = struct.unpack("<fI", hdr)
            sbytes = f.read(slen)
            raw_items.append((sbytes, score))

    vocab_pieces = []
    bytes_to_id = {}
    for i, (sbytes, _) in enumerate(raw_items):
        if len(sbytes) == 6 and sbytes.startswith(b"<0x") and sbytes.endswith(b">"):
            try:
                b_val = bytes([int(sbytes[3:5], 16)])
                vocab_pieces.append(b_val)
                if b_val not in bytes_to_id:
                    bytes_to_id[b_val] = i
                continue
            except ValueError:
                pass
        vocab_pieces.append(sbytes)
        if sbytes and sbytes not in bytes_to_id:
            bytes_to_id[sbytes] = i

    merges_candidates = []
    for target_id, ((raw_s, score), p_bytes) in enumerate(zip(raw_items, vocab_pieces)):
        if len(p_bytes) < 2 or raw_s.startswith(b"<0x"):
            continue
        for split in range(1, len(p_bytes)):
            left_b = p_bytes[:split]
            right_b = p_bytes[split:]
            lid = bytes_to_id.get(left_b)
            rid = bytes_to_id.get(right_b)
            if lid is not None and rid is not None:
                merges_candidates.append((-score, lid, rid, target_id))

    merges_candidates.sort(key=lambda x: x[0])
    merges_tuples = []
    seen_pairs = set()
    for rank, (_, lid, rid, tid) in enumerate(merges_candidates):
        pk = (lid << 32) | rid
        if pk not in seen_pairs:
            seen_pairs.add(pk)
            merges_tuples.append((lid, rid, tid, rank))

    return serialize_qmx_tokenizer(vocab_pieces, merges_tuples, 1, 2, -1, False)


def load_hf_safetensors(repo_or_dir: str, max_seq_override: int = 0):
    if os.path.isdir(repo_or_dir):
        model_dir = repo_or_dir
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            sys.exit("Error: install huggingface_hub (`pip install huggingface_hub`) for --hf")
        model_dir = snapshot_download(repo_id=repo_or_dir)

    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Gemma-3 multimodal wraps text config under text_config
    text_cfg = cfg.get("text_config", cfg)

    archs = " ".join(cfg.get("architectures", [])).lower()
    model_type = str(text_cfg.get("model_type", cfg.get("model_type", ""))).lower()
    is_bitnet = ("bitnet" in archs) or ("bitnet" in model_type)
    is_gemma = ("gemma" in model_type) or ("gemma" in archs)

    hidden_act = str(text_cfg.get("hidden_act", text_cfg.get("hidden_activation", text_cfg.get("activation_function", "silu")))).lower()
    if "relu2" in hidden_act or "squared" in hidden_act:
        activation = "relu2"
    elif "gelu" in hidden_act:
        activation = "geglu"
    else:
        activation = "swiglu"

    rms_eps = float(text_cfg.get("rms_norm_eps", text_cfg.get("layer_norm_eps", 1e-5)))
    rope_theta = float(text_cfg.get("rope_theta", 10000.0))

    rope_scaling = text_cfg.get("rope_scaling")
    rope_scale_type = QMX_ROPE_SCALE_NONE
    rope_scale_factor = 1.0
    llama3_rope_scale = False
    if isinstance(rope_scaling, dict):
        rtype = str(rope_scaling.get("rope_type", rope_scaling.get("type", ""))).lower()
        rope_scale_factor = float(rope_scaling.get("factor", 1.0))
        if rtype == "llama3":
            rope_scale_type = QMX_ROPE_SCALE_LLAMA3
            llama3_rope_scale = True
        elif rtype == "linear":
            rope_scale_type = QMX_ROPE_SCALE_LINEAR
        elif rtype == "yarn":
            rope_scale_type = QMX_ROPE_SCALE_YARN

    dim = int(text_cfg["hidden_size"])
    hidden_dim = int(text_cfg["intermediate_size"])
    n_layers = int(text_cfg["num_hidden_layers"])
    n_heads = int(text_cfg["num_attention_heads"])
    n_kv_heads = int(text_cfg.get("num_key_value_heads", n_heads))
    head_dim = int(text_cfg.get("head_dim", dim // n_heads))
    partial_rotary = float(text_cfg.get("partial_rotary_factor", 1.0))
    rotary_dim = int(head_dim * partial_rotary) & ~1
    if rotary_dim <= 0 or rotary_dim > head_dim:
        rotary_dim = head_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim

    vocab_size = int(text_cfg["vocab_size"])
    native_seq_len = int(text_cfg.get("max_position_embeddings", 4096))
    seq_len = max_seq_override if max_seq_override > 0 else min(native_seq_len, 4096)

    st_files = sorted([os.path.join(model_dir, fn) for fn in os.listdir(model_dir) if fn.endswith(".safetensors")])
    if not st_files:
        sys.exit(f"Error: no .safetensors files found in {model_dir}")
    handles = [NativeSafeTensors(p) for p in st_files]
    key_to_handle = {}
    for h in handles:
        for k in h.keys():
            key_to_handle[k] = h

    prefix = "language_model.model." if "language_model.model.embed_tokens.weight" in key_to_handle else "model."

    def has_t(name: str) -> bool:
        return name in key_to_handle

    def get_t(name: str):
        h = key_to_handle.get(name)
        if h is None:
            return None
        return h.get_tensor(name).astype(np.float32)

    def get_norm_t(name: str):
        arr = get_t(name)
        if arr is not None and is_gemma:
            arr = arr + 1.0
        return arr

    fused_qkv = (not has_t(f"{prefix}layers.0.self_attn.q_proj.weight")) and has_t(f"{prefix}layers.0.self_attn.qkv_proj.weight")
    fused_gate_up = (not has_t(f"{prefix}layers.0.mlp.gate_proj.weight")) and has_t(f"{prefix}layers.0.mlp.gate_up_proj.weight")

    has_qkv_bias = has_t(f"{prefix}layers.0.self_attn.q_proj.bias")
    has_o_bias = has_t(f"{prefix}layers.0.self_attn.o_proj.bias")
    has_qk_norm = has_t(f"{prefix}layers.0.self_attn.q_norm.weight") and has_t(f"{prefix}layers.0.self_attn.k_norm.weight")
    has_norm_bias = has_t(f"{prefix}layers.0.input_layernorm.bias")
    has_final_bias = has_t(f"{prefix}norm.bias")
    use_layernorm = has_norm_bias or ("gpt" in model_type)
    has_post_norm = has_t(f"{prefix}layers.0.pre_feedforward_layernorm.weight") and has_t(f"{prefix}layers.0.post_feedforward_layernorm.weight")

    logit_softcap = float(text_cfg.get("final_logit_softcapping") or 0.0)
    attn_softcap = float(text_cfg.get("attn_logit_softcapping") or 0.0)
    emb_scale = math.sqrt(float(dim)) if is_gemma else 1.0
    sliding_window = 0
    if not is_gemma and text_cfg.get("use_sliding_window", False) and text_cfg.get("sliding_window"):
        sliding_window = int(text_cfg["sliding_window"])

    has_subnorm_weights = has_t(f"{prefix}layers.0.self_attn.inner_attn_ln.weight") and has_t(f"{prefix}layers.0.mlp.ffn_layernorm.weight")
    if has_subnorm_weights:
        is_bitnet = True

    lm_head_key = "language_model.lm_head.weight" if has_t("language_model.lm_head.weight") else "lm_head.weight"
    shared_classifier = (not has_t(lm_head_key)) or bool(text_cfg.get("tie_word_embeddings", False))

    tok_bytes = None
    tok_json_path = os.path.join(model_dir, "tokenizer.json")
    if os.path.exists(tok_json_path):
        tok_bytes = build_tokenizer_from_hf_json(
            tok_json_path, vocab_size,
            cfg_bos=text_cfg.get("bos_token_id"),
            cfg_eos=text_cfg.get("eos_token_id")
        )

    def make_slice_loader(key: str, r0: int, r1: int):
        def _load():
            arr = get_t(key)
            return np.ascontiguousarray(arr[r0:r1, :])
        return _load

    specs = []
    specs.append(("token_emb.weight", TID_TOKEN_EMB, 0xFFFF, True, QMX_PIN_LATENCY_SENS, lambda: get_t(f"{prefix}embed_tokens.weight")))
    specs.append(("rms_final.weight", TID_RMS_FINAL, 0xFFFF, False, QMX_PIN_CRITICAL, lambda: get_norm_t(f"{prefix}norm.weight")))
    if has_final_bias:
        specs.append(("rms_final.bias", TID_RMS_FINAL_BIAS, 0xFFFF, False, QMX_PIN_CRITICAL, lambda: get_t(f"{prefix}norm.bias")))
    if not shared_classifier:
        specs.append(("lm_head.weight", TID_LM_HEAD, 0xFFFF, True, QMX_PIN_STREAM_ONLY, lambda: get_t(lm_head_key)))

    for l in range(n_layers):
        lp = f"{prefix}layers.{l}."
        specs.append((f"blk.{l}.attn_norm", TID_ATTN_NORM, l, False, QMX_PIN_CRITICAL, lambda k=f"{lp}input_layernorm.weight": get_norm_t(k)))
        if has_norm_bias:
            specs.append((f"blk.{l}.attn_norm_bias", TID_ATTN_NORM_BIAS, l, False, QMX_PIN_CRITICAL, lambda k=f"{lp}input_layernorm.bias": get_t(k)))

        if fused_qkv:
            qk = f"{lp}self_attn.qkv_proj.weight"
            specs.append((f"blk.{l}.attn_q", TID_ATTN_Q, l, True, QMX_PIN_LATENCY_SENS, make_slice_loader(qk, 0, q_dim)))
            specs.append((f"blk.{l}.attn_k", TID_ATTN_K, l, True, QMX_PIN_LATENCY_SENS, make_slice_loader(qk, q_dim, q_dim + kv_dim)))
            specs.append((f"blk.{l}.attn_v", TID_ATTN_V, l, True, QMX_PIN_LATENCY_SENS, make_slice_loader(qk, q_dim + kv_dim, q_dim + 2 * kv_dim)))
        else:
            specs.append((f"blk.{l}.attn_q", TID_ATTN_Q, l, True, QMX_PIN_LATENCY_SENS, lambda k=f"{lp}self_attn.q_proj.weight": get_t(k)))
            if has_qkv_bias:
                specs.append((f"blk.{l}.attn_q_bias", TID_ATTN_Q_BIAS, l, False, QMX_PIN_CRITICAL, lambda k=f"{lp}self_attn.q_proj.bias": get_t(k)))
            specs.append((f"blk.{l}.attn_k", TID_ATTN_K, l, True, QMX_PIN_LATENCY_SENS, lambda k=f"{lp}self_attn.k_proj.weight": get_t(k)))
            if has_qkv_bias:
                specs.append((f"blk.{l}.attn_k_bias", TID_ATTN_K_BIAS, l, False, QMX_PIN_CRITICAL, lambda k=f"{lp}self_attn.k_proj.bias": get_t(k)))
            specs.append((f"blk.{l}.attn_v", TID_ATTN_V, l, True, QMX_PIN_LATENCY_SENS, lambda k=f"{lp}self_attn.v_proj.weight": get_t(k)))
            if has_qkv_bias:
                specs.append((f"blk.{l}.attn_v_bias", TID_ATTN_V_BIAS, l, False, QMX_PIN_CRITICAL, lambda k=f"{lp}self_attn.v_proj.bias": get_t(k)))

        if has_qk_norm:
            specs.append((f"blk.{l}.attn_q_norm", TID_ATTN_Q_NORM, l, False, QMX_PIN_CRITICAL, lambda k=f"{lp}self_attn.q_norm.weight": get_norm_t(k)))
            specs.append((f"blk.{l}.attn_k_norm", TID_ATTN_K_NORM, l, False, QMX_PIN_CRITICAL, lambda k=f"{lp}self_attn.k_norm.weight": get_norm_t(k)))
        if has_subnorm_weights:
            specs.append((f"blk.{l}.attn_subnorm", TID_ATTN_SUBNORM, l, False, QMX_PIN_CRITICAL, lambda k=f"{lp}self_attn.inner_attn_ln.weight": get_t(k)))

        specs.append((f"blk.{l}.attn_wo", TID_ATTN_WO, l, True, QMX_PIN_LATENCY_SENS, lambda k=f"{lp}self_attn.o_proj.weight": get_t(k)))
        if has_o_bias:
            specs.append((f"blk.{l}.attn_o_bias", TID_ATTN_O_BIAS, l, False, QMX_PIN_CRITICAL, lambda k=f"{lp}self_attn.o_proj.bias": get_t(k)))

        if has_post_norm:
            specs.append((f"blk.{l}.attn_post_norm", TID_ATTN_POST_NORM, l, False, QMX_PIN_CRITICAL, lambda k=f"{lp}post_attention_layernorm.weight": get_norm_t(k)))
            specs.append((f"blk.{l}.ffn_norm", TID_FFN_NORM, l, False, QMX_PIN_CRITICAL, lambda k=f"{lp}pre_feedforward_layernorm.weight": get_norm_t(k)))
        else:
            specs.append((f"blk.{l}.ffn_norm", TID_FFN_NORM, l, False, QMX_PIN_CRITICAL, lambda k=f"{lp}post_attention_layernorm.weight": get_norm_t(k)))
            if has_norm_bias:
                specs.append((f"blk.{l}.ffn_norm_bias", TID_FFN_NORM_BIAS, l, False, QMX_PIN_CRITICAL, lambda k=f"{lp}post_attention_layernorm.bias": get_t(k)))

        if fused_gate_up:
            gk = f"{lp}mlp.gate_up_proj.weight"
            specs.append((f"blk.{l}.ffn_gate", TID_FFN_GATE, l, True, QMX_PIN_STREAM_ONLY, make_slice_loader(gk, 0, hidden_dim)))
            specs.append((f"blk.{l}.ffn_up", TID_FFN_UP, l, True, QMX_PIN_STREAM_ONLY, make_slice_loader(gk, hidden_dim, 2 * hidden_dim)))
        else:
            specs.append((f"blk.{l}.ffn_gate", TID_FFN_GATE, l, True, QMX_PIN_STREAM_ONLY, lambda k=f"{lp}mlp.gate_proj.weight": get_t(k)))
            specs.append((f"blk.{l}.ffn_up", TID_FFN_UP, l, True, QMX_PIN_STREAM_ONLY, lambda k=f"{lp}mlp.up_proj.weight": get_t(k)))

        if has_subnorm_weights:
            specs.append((f"blk.{l}.ffn_subnorm", TID_FFN_SUBNORM, l, False, QMX_PIN_CRITICAL, lambda k=f"{lp}mlp.ffn_layernorm.weight": get_t(k)))
        specs.append((f"blk.{l}.ffn_down", TID_FFN_DOWN, l, True, QMX_PIN_STREAM_ONLY, lambda k=f"{lp}mlp.down_proj.weight": get_t(k)))
        if has_post_norm:
            specs.append((f"blk.{l}.ffn_post_norm", TID_FFN_POST_NORM, l, False, QMX_PIN_CRITICAL, lambda k=f"{lp}post_feedforward_layernorm.weight": get_norm_t(k)))

    config = {
        "vocab_size": vocab_size,
        "max_seq_len": seq_len,
        "hidden_dim": dim,
        "intermediate_dim": hidden_dim,
        "num_layers": n_layers,
        "num_heads": n_heads,
        "num_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "rotary_dim": rotary_dim,
        "rms_norm_eps": rms_eps,
        "rope_theta": rope_theta,
        "rope_scale_type": rope_scale_type,
        "rope_scale_factor": rope_scale_factor,
        "emb_scale": emb_scale,
        "logit_softcap": logit_softcap,
        "attn_softcap": attn_softcap,
        "sliding_window": sliding_window,
        "shared_classifier": shared_classifier,
        "rope_split_half": True,
        "llama3_rope_scale": llama3_rope_scale,
        "has_qkv_bias": has_qkv_bias,
        "has_o_bias": has_o_bias,
        "has_qk_norm": has_qk_norm,
        "has_norm_bias": has_norm_bias,
        "has_post_norm": has_post_norm,
        "use_layernorm": use_layernorm,
        "has_subnorm": is_bitnet,
        "has_subnorm_weights": has_subnorm_weights,
        "activation": activation,
        "is_bitnet": is_bitnet,
        "tokenizer_bytes": tok_bytes,
    }
    return config, specs


def load_karpathy_bin(bin_path: str, tok_path: str = None):
    with open(bin_path, "rb") as f:
        dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = struct.unpack("<7i", f.read(28))
        shared_classifier = vocab_size > 0
        vocab_size = abs(vocab_size)
        kv_dim = (dim * n_kv_heads) // n_heads

        def read_mat(r, c):
            return np.frombuffer(f.read(r * c * 4), dtype=np.float32).reshape(r, c).copy()

        def read_vec(n):
            return np.frombuffer(f.read(n * 4), dtype=np.float32).copy()

        tok_emb = read_mat(vocab_size, dim)
        rms_att = [read_vec(dim) for _ in range(n_layers)]
        wq = [read_mat(dim, dim) for _ in range(n_layers)]
        wk = [read_mat(kv_dim, dim) for _ in range(n_layers)]
        wv = [read_mat(kv_dim, dim) for _ in range(n_layers)]
        wo = [read_mat(dim, dim) for _ in range(n_layers)]
        rms_ffn = [read_vec(dim) for _ in range(n_layers)]
        w1 = [read_mat(hidden_dim, dim) for _ in range(n_layers)]
        w2 = [read_mat(dim, hidden_dim) for _ in range(n_layers)]
        w3 = [read_mat(hidden_dim, dim) for _ in range(n_layers)]
        rms_final = read_vec(dim)
        f.seek(seq_len * (dim // n_heads) * 4, os.SEEK_CUR)
        wcls = tok_emb if shared_classifier else read_mat(vocab_size, dim)

    tok_bytes = None
    if tok_path and os.path.exists(tok_path):
        tok_bytes = build_tokenizer_from_karpathy_bin(tok_path, vocab_size)

    specs = [
        ("token_emb.weight", TID_TOKEN_EMB, 0xFFFF, True, QMX_PIN_LATENCY_SENS, tok_emb),
        ("rms_final.weight", TID_RMS_FINAL, 0xFFFF, False, QMX_PIN_CRITICAL, rms_final),
    ]
    if not shared_classifier:
        specs.append(("lm_head.weight", TID_LM_HEAD, 0xFFFF, True, QMX_PIN_STREAM_ONLY, wcls))
    for l in range(n_layers):
        specs.extend([
            (f"blk.{l}.attn_norm", TID_ATTN_NORM, l, False, QMX_PIN_CRITICAL, rms_att[l]),
            (f"blk.{l}.attn_q", TID_ATTN_Q, l, True, QMX_PIN_LATENCY_SENS, wq[l]),
            (f"blk.{l}.attn_k", TID_ATTN_K, l, True, QMX_PIN_LATENCY_SENS, wk[l]),
            (f"blk.{l}.attn_v", TID_ATTN_V, l, True, QMX_PIN_LATENCY_SENS, wv[l]),
            (f"blk.{l}.attn_wo", TID_ATTN_WO, l, True, QMX_PIN_LATENCY_SENS, wo[l]),
            (f"blk.{l}.ffn_norm", TID_FFN_NORM, l, False, QMX_PIN_CRITICAL, rms_ffn[l]),
            (f"blk.{l}.ffn_gate", TID_FFN_GATE, l, True, QMX_PIN_STREAM_ONLY, w1[l]),
            (f"blk.{l}.ffn_up", TID_FFN_UP, l, True, QMX_PIN_STREAM_ONLY, w3[l]),
            (f"blk.{l}.ffn_down", TID_FFN_DOWN, l, True, QMX_PIN_STREAM_ONLY, w2[l]),
        ])

    config = {
        "vocab_size": vocab_size,
        "max_seq_len": min(seq_len, 512),
        "hidden_dim": dim,
        "intermediate_dim": hidden_dim,
        "num_layers": n_layers,
        "num_heads": n_heads,
        "num_kv_heads": n_kv_heads,
        "head_dim": dim // n_heads,
        "rotary_dim": dim // n_heads,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "shared_classifier": shared_classifier,
        "rope_split_half": False,
        "llama3_rope_scale": False,
        "has_qkv_bias": False,
        "has_o_bias": False,
        "has_subnorm": False,
        "has_subnorm_weights": False,
        "activation": "swiglu",
        "is_bitnet": False,
        "tokenizer_bytes": tok_bytes,
    }
    return config, specs


def build_dummy_model():
    rng = np.random.default_rng(42)
    dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = 64, 128, 2, 4, 4, 256, 128

    vocab_pieces = [bytes([i]) for i in range(vocab_size)]
    tok_bytes = serialize_qmx_tokenizer(vocab_pieces, [], 1, 2, -1, False)

    specs = [
        ("token_emb.weight", TID_TOKEN_EMB, 0xFFFF, True, QMX_PIN_LATENCY_SENS, rng.normal(0, 0.05, (vocab_size, dim)).astype(np.float32)),
        ("rms_final.weight", TID_RMS_FINAL, 0xFFFF, False, QMX_PIN_CRITICAL, np.ones(dim, dtype=np.float32)),
    ]
    for l in range(n_layers):
        specs.extend([
            (f"blk.{l}.attn_norm", TID_ATTN_NORM, l, False, QMX_PIN_CRITICAL, np.ones(dim, dtype=np.float32)),
            (f"blk.{l}.attn_q", TID_ATTN_Q, l, True, QMX_PIN_LATENCY_SENS, rng.normal(0, 0.05, (dim, dim)).astype(np.float32)),
            (f"blk.{l}.attn_k", TID_ATTN_K, l, True, QMX_PIN_LATENCY_SENS, rng.normal(0, 0.05, (dim, dim)).astype(np.float32)),
            (f"blk.{l}.attn_v", TID_ATTN_V, l, True, QMX_PIN_LATENCY_SENS, rng.normal(0, 0.05, (dim, dim)).astype(np.float32)),
            (f"blk.{l}.attn_wo", TID_ATTN_WO, l, True, QMX_PIN_LATENCY_SENS, rng.normal(0, 0.05, (dim, dim)).astype(np.float32)),
            (f"blk.{l}.ffn_norm", TID_FFN_NORM, l, False, QMX_PIN_CRITICAL, np.ones(dim, dtype=np.float32)),
            (f"blk.{l}.ffn_gate", TID_FFN_GATE, l, True, QMX_PIN_STREAM_ONLY, rng.normal(0, 0.05, (hidden_dim, dim)).astype(np.float32)),
            (f"blk.{l}.ffn_up", TID_FFN_UP, l, True, QMX_PIN_STREAM_ONLY, rng.normal(0, 0.05, (hidden_dim, dim)).astype(np.float32)),
            (f"blk.{l}.ffn_down", TID_FFN_DOWN, l, True, QMX_PIN_STREAM_ONLY, rng.normal(0, 0.05, (dim, hidden_dim)).astype(np.float32)),
        ])

    config = {
        "vocab_size": vocab_size,
        "max_seq_len": seq_len,
        "hidden_dim": dim,
        "intermediate_dim": hidden_dim,
        "num_layers": n_layers,
        "num_heads": n_heads,
        "num_kv_heads": n_kv_heads,
        "head_dim": dim // n_heads,
        "rotary_dim": dim // n_heads,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "shared_classifier": True,
        "rope_split_half": False,
        "llama3_rope_scale": False,
        "has_qkv_bias": False,
        "has_o_bias": False,
        "has_subnorm": False,
        "has_subnorm_weights": False,
        "activation": "swiglu",
        "is_bitnet": False,
        "tokenizer_bytes": tok_bytes,
    }
    return config, specs


def estimate_param_count(config: dict) -> int:
    dim = config["hidden_dim"]
    inter = config["intermediate_dim"]
    layers = config["num_layers"]
    head_dim = config.get("head_dim", dim // config["num_heads"])
    q_dim = config["num_heads"] * head_dim
    kv_dim = config["num_kv_heads"] * head_dim
    per_layer = dim * (2 * q_dim + 2 * kv_dim + 3 * inter)
    emb = config["vocab_size"] * dim * (1 if config.get("shared_classifier", True) else 2)
    return emb + layers * per_layer


def estimate_min_ram(config: dict, critical_bytes: int) -> int:
    dim = config["hidden_dim"]
    inter = config["intermediate_dim"]
    layers = config["num_layers"]
    seq = config["max_seq_len"]
    head_dim = config.get("head_dim", dim // config["num_heads"])
    kv_dim = config["num_kv_heads"] * head_dim
    kv_bytes = 2 * layers * seq * kv_dim + 2 * layers * seq * config["num_kv_heads"] * 4
    act_bytes = (6 * dim + 2 * inter + config["vocab_size"]) * 4
    lru_bytes = 64 * dim * 4 + 32768
    return min(critical_bytes + kv_bytes + act_bytes + lru_bytes, 0xFFFFFFFF)


def export_qmx(output_path: str, config: dict, tensor_specs, primary_quant: int,
               hybrid_emb_q8: bool, compact_align: bool = None, lm_head_quant: int = None):
    if not isinstance(tensor_specs, (list, tuple)):
        tensor_specs = list(tensor_specs)
    expected_count = len(tensor_specs)

    est_params = estimate_param_count(config)
    if compact_align is None:
        compact_align = (est_params < 100000)

    tensor_align = CACHE_ALIGN if compact_align else SECTOR_ALIGN
    payload_align = CACHE_ALIGN if compact_align else PAGE_ALIGN

    flags = QMX_FLAG_TILE_MAJOR
    if config.get("shared_classifier", True):
        flags |= QMX_FLAG_SHARED_CLASSIFIER
    if config.get("rope_split_half", False):
        flags |= QMX_FLAG_ROPE_SPLIT_HALF
    if config.get("llama3_rope_scale", False):
        flags |= QMX_FLAG_LLAMA3_ROPE_SCALE
    if config.get("has_qkv_bias", False) or config.get("has_o_bias", False) or config.get("has_norm_bias", False):
        flags |= QMX_FLAG_HAS_BIAS
    logit_softcap = float(config.get("logit_softcap", 0.0) or 0.0)
    attn_softcap = float(config.get("attn_softcap", 0.0) or 0.0)
    if logit_softcap > 0.0:
        flags |= QMX_FLAG_LOGIT_SOFTCAP
    if attn_softcap > 0.0:
        flags |= QMX_FLAG_ATTN_SOFTCAP
    if compact_align:
        flags |= QMX_FLAG_COMPACT_ALIGN

    recipe_bytes, recipe_count = build_layer_recipe(config)
    per_tensor_scale = bool(config.get("is_bitnet", False))
    tok_bytes = config.get("tokenizer_bytes") or b""

    tensor_dir = bytearray()
    tensor_count = 0
    critical_bytes = 0

    with open(output_path, "wb") as f:
        # 1. Placeholder header (128 bytes)
        f.write(b"\x00" * 128)

        # 2. Recipe
        recipe_offset = pad_file(f, CACHE_ALIGN)
        f.write(recipe_bytes)

        # 3. Placeholder tensor directory (written at the end)
        tensor_dir_offset = pad_file(f, CACHE_ALIGN)
        f.write(b"\x00" * (expected_count * 64))

        # 4. Tokenizer
        tokenizer_size = len(tok_bytes)
        if tokenizer_size > 0:
            tokenizer_offset = pad_file(f, CACHE_ALIGN)
            f.write(tok_bytes)
        else:
            tokenizer_offset = 0

        # 5. Stream payload directly to disk
        payload_offset = pad_file(f, payload_align)

        emb_qtype = QMX_QUANT_Q8_ROW32 if primary_quant in (QMX_QUANT_Q4_ROW32, QMX_QUANT_Q8_ROW32) else QMX_QUANT_Q8_SYM

        for name, tid, layer_idx, is_2d, pin_prio, arr_or_fn in tensor_specs:
            arr = arr_or_fn() if callable(arr_or_fn) else arr_or_fn
            if arr is None:
                raise RuntimeError(f"Missing required tensor: {name}")

            name_raw = name.encode("ascii", errors="ignore")[:31]
            name_padded = name_raw + b"\x00" * (32 - len(name_raw))

            cur_abs = pad_file(f, tensor_align)

            if is_2d:
                assert arr.ndim == 2
                orig_rows, orig_cols = arr.shape
                if tid == TID_LM_HEAD or (tid == TID_TOKEN_EMB and config.get("shared_classifier", True)):
                    if lm_head_quant is not None:
                        use_qtype = lm_head_quant
                    else:
                        use_qtype = emb_qtype if hybrid_emb_q8 else primary_quant
                elif tid == TID_TOKEN_EMB:
                    use_qtype = emb_qtype if hybrid_emb_q8 else primary_quant
                else:
                    use_qtype = primary_quant
                scales_b, data_b, _, _ = quantize_matrix_tiled(arr, use_qtype, per_tensor_scale)

                scales_off = cur_abs - payload_offset
                f.write(scales_b)
                cur_abs = pad_file(f, CACHE_ALIGN)

                data_off = cur_abs - payload_offset
                f.write(data_b)
                data_sz = len(data_b)
                pad_file(f, CACHE_ALIGN)
                tile_rows = TILE_ROWS
                del scales_b, data_b
            else:
                flat = arr.astype(np.float32).reshape(-1)
                orig_rows = 1
                orig_cols = flat.shape[0]
                use_qtype = QMX_QUANT_FP32
                scales_off = 0
                data_off = cur_abs - payload_offset
                raw_f32 = flat.tobytes()
                f.write(raw_f32)
                data_sz = len(raw_f32)
                pad_file(f, CACHE_ALIGN)
                tile_rows = 1
                del flat, raw_f32

            if pin_prio == QMX_PIN_CRITICAL:
                critical_bytes += align_up(data_sz, CACHE_ALIGN)

            if scales_off >= (1 << 48) or data_off >= (1 << 48) or data_sz >= (1 << 32):
                raise OverflowError(f"Tensor '{name}' exceeds .qmx v1.1 descriptor bounds")

            desc = struct.pack(
                DESC_FMT,
                name_padded,
                tid,
                layer_idx,
                use_qtype,
                pin_prio,
                tile_rows,
                orig_rows,
                orig_cols,
                scales_off & 0xFFFFFFFF,
                data_off & 0xFFFFFFFF,
                data_sz,
                (scales_off >> 32) & 0xFFFF,
                (data_off >> 32) & 0xFFFF,
            )
            tensor_dir.extend(desc)
            tensor_count += 1
            if expected_count >= 100 and (tensor_count % 16 == 0 or tensor_count == expected_count):
                print(f"\r[Pack] {tensor_count}/{expected_count} tensors ({f.tell() / (1024.0 * 1024.0):.1f} MB)...", end="", flush=True)
            del arr
            if tensor_count % 32 == 0:
                gc.collect()
        if expected_count >= 100:
            print()

        file_end = f.tell()
        payload_size = file_end - payload_offset
        assert tensor_count == expected_count

        min_ram = estimate_min_ram(config, critical_bytes)
        head_dim = int(config.get("head_dim", config["hidden_dim"] // config["num_heads"]))
        rotary_dim = int(config.get("rotary_dim", head_dim))
        emb_scale = float(config.get("emb_scale", 1.0))
        rope_scale_type = int(config.get("rope_scale_type", QMX_ROPE_SCALE_LLAMA3 if config.get("llama3_rope_scale") else QMX_ROPE_SCALE_NONE))
        rope_scale_factor = float(config.get("rope_scale_factor", 1.0))
        sliding_window = int(config.get("sliding_window", 0))

        hdr_zero_crc = struct.pack(
            HEADER_FMT,
            QMX_MAGIC,
            QMX_VERSION_1_1,
            flags,
            0,
            0,
            recipe_offset,
            recipe_count,
            tensor_dir_offset,
            tensor_count,
            tokenizer_offset,
            tokenizer_size,
            payload_offset & 0xFFFFFFFF,
            payload_size & 0xFFFFFFFF,
            config["vocab_size"],
            config["max_seq_len"],
            config["hidden_dim"],
            config["intermediate_dim"],
            config["num_layers"],
            config["num_heads"],
            config["num_kv_heads"],
            primary_quant,
            float(config["rms_norm_eps"]),
            float(config["rope_theta"]),
            min_ram,
            (payload_offset >> 32) & 0xFFFFFFFF,
            (payload_size >> 32) & 0xFFFFFFFF,
            head_dim,
            rotary_dim,
            emb_scale,
            logit_softcap,
            attn_softcap,
            rope_scale_factor,
            rope_scale_type,
            1 if config.get("use_layernorm", False) else 0,
            sliding_window,
            b"\x00" * 12,
        )
        hdr_crc = zlib.crc32(hdr_zero_crc) & 0xFFFFFFFF
        hdr_final = hdr_zero_crc[:8] + struct.pack("<I", hdr_crc) + hdr_zero_crc[12:]

        f.seek(tensor_dir_offset)
        f.write(tensor_dir)
        f.seek(0)
        f.write(hdr_final)

    size_mb = file_end / (1024.0 * 1024.0)
    size_str = f"{size_mb:.2f} MB" if size_mb >= 1.0 else f"{file_end / 1024.0:.1f} KB"
    print(
        f"[OK] Packed {output_path} ({size_str}) | v1.1 | "
        f"quant={QUANT_NAMES[primary_quant]} | layers={config['num_layers']} | "
        f"dim={config['hidden_dim']} | head_dim={head_dim} | min_ram={min_ram / 1024.0:.1f} KB"
    )


def inspect_qmx(path: str):
    if not os.path.exists(path):
        sys.exit(f"Error: file not found: {path}")
    fsize = os.path.getsize(path)
    with open(path, "rb") as f:
        hdr = f.read(128)
        if len(hdr) < 128:
            sys.exit("Error: truncated QMX header")

        unpacked = struct.unpack(HEADER_FMT, hdr)
        (
            magic, version, flags, stored_crc, _file_crc,
            recipe_off, recipe_cnt, tdir_off, t_cnt,
            tok_off, tok_sz, pay_off_lo, pay_sz_lo,
            vocab_sz, max_seq, dim, inter_dim,
            n_layers, n_heads, n_kv_heads, p_quant,
            rms_eps, rope_theta, min_ram,
            pay_off_hi, pay_sz_hi, head_dim, rotary_dim,
            emb_scale, logit_softcap, attn_softcap, rope_scale_factor,
            rope_scale_type, _norm_type, sliding_window, _res
        ) = unpacked

        if magic != QMX_MAGIC:
            sys.exit(f"Error: invalid QMX magic 0x{magic:08X}")

        is_v11 = version >= QMX_VERSION_1_1
        pay_off = ((pay_off_hi << 32) | pay_off_lo) if is_v11 else pay_off_lo
        pay_sz = ((pay_sz_hi << 32) | pay_sz_lo) if is_v11 else pay_sz_lo
        eff_head_dim = head_dim if (is_v11 and head_dim > 0) else (dim // n_heads)
        eff_rot_dim = rotary_dim if (is_v11 and rotary_dim > 0) else eff_head_dim

        zeroed = hdr[:8] + b"\x00\x00\x00\x00" + hdr[12:]
        calc_crc = zlib.crc32(zeroed) & 0xFFFFFFFF
        crc_str = f"0x{stored_crc:08X} ({'VALID' if calc_crc == stored_crc else f'MISMATCH expected 0x{calc_crc:08X}'})"

        print(f"=== QMX Cartridge Info: {os.path.basename(path)} ===")
        print(f"  File Size      : {fsize / 1024.0:.1f} KB ({fsize} bytes)")
        print(f"  Format Version : v{version >> 8}.{version & 0xFF}")
        print(f"  Header CRC32   : {crc_str}")
        print(f"  Primary Quant  : 0x{p_quant:02X} ({QUANT_NAMES.get(p_quant, 'Unknown')})")
        print(f"  Topology       : dim={dim}, inter={inter_dim}, layers={n_layers}, heads={n_heads}/{n_kv_heads}, head_dim={eff_head_dim}, rot_dim={eff_rot_dim}")
        print(f"  Vocab / Ctx    : vocab={vocab_sz}, max_seq={max_seq}, swa={sliding_window}")
        print(f"  RoPE / Norm    : theta={rope_theta:.1f}, scale_type={rope_scale_type} (x{rope_scale_factor:.2f}), eps={rms_eps:.1e}")
        if is_v11:
            print(f"  Scaling / Cap  : emb_scale={emb_scale:.4f}, logit_cap={logit_softcap:.1f}, attn_cap={attn_softcap:.1f}")
        print(f"  Min Stream RAM : {min_ram / 1024.0:.1f} KB")
        print(f"  Flags          : 0x{flags:04X}")
        print(f"  Tokenizer      : {'Embedded (' + str(tok_sz) + ' B @ 0x' + hex(tok_off)[2:] + ')' if tok_sz else 'None'}")
        print(f"  Payload Offset : 0x{pay_off:08X} ({pay_sz / 1024.0:.1f} KB, page_aligned={pay_off % 4096 == 0})")

        f.seek(recipe_off)
        print(f"\n  Layer Recipe VM ({recipe_cnt} instructions):")
        for idx in range(recipe_cnt):
            op, flg, tid, src, dst = struct.unpack("<BBHHH", f.read(8))
            print(f"    {idx:02d}: {OPCODE_NAMES.get(op, f'OP_{op:02X}'):13s} flags=0x{flg:02X} tid=0x{tid:04X} buf[{src}] -> buf[{dst}]")

        f.seek(tdir_off)
        print(f"\n  Tensor Directory ({t_cnt} tensors, showing first 12):")
        for idx in range(min(t_cnt, 12)):
            raw_d = f.read(64)
            tname, tid, lidx, qt, prio, trows, rows, cols, soff_lo, doff_lo, dsz, soff_hi, doff_hi = struct.unpack(DESC_FMT, raw_d)
            clean_name = tname.split(b"\x00", 1)[0].decode("ascii", errors="ignore")
            doff = ((doff_hi << 32) | doff_lo) if is_v11 else doff_lo
            print(f"    [{idx:02d}] {clean_name:22s} L={lidx if lidx != 0xFFFF else -1:2d} q=0x{qt:02X} shape={rows}x{cols} tile_r={trows} off=0x{doff:X} data={dsz}B")


def main():
    parser = argparse.ArgumentParser(description="QMX Forge v1.2b: Package & Inspect .qmx v1.1 Cartridges")
    parser.add_argument("--info", type=str, help="Inspect an existing .qmx cartridge")
    parser.add_argument("--hf", type=str, help="HuggingFace repo ID or local model directory")
    parser.add_argument("--input", "-i", type=str, help="Karpathy llama2.c .bin checkpoint")
    parser.add_argument("--tokenizer", "-z", type=str, help="External tokenizer.bin (for Karpathy .bin)")
    parser.add_argument("--dummy", action="store_true", help="Generate synthetic micro-model for testing")
    parser.add_argument("--output", "-o", type=str, help="Output .qmx file path")
    parser.add_argument("--quant", "-q", choices=["binary", "ternary", "q4", "q8", "q4_r32", "q8_r32"], default="q4")
    parser.add_argument("--max-seq", type=int, default=0, help="Override max_seq_len stored in cartridge header")
    parser.add_argument("--compact", action="store_true", help="Force compact 64B alignment (best for 15K-500K MCU models)")
    parser.add_argument("--no-compact", action="store_true", help="Force 512B/4096B page alignment (best for mmap)")
    parser.add_argument("--no-hybrid-emb", action="store_true", help="Quantize embeddings and lm_head with primary quant instead of Q8")
    parser.add_argument("--lm-head-quant", choices=["auto", "binary", "ternary", "q4", "q8", "q4_r32", "q8_r32"], default="auto", help="Override quantization profile for lm_head (and tied token_emb)")
    args = parser.parse_args()

    if args.info:
        inspect_qmx(args.info)
        return

    if not args.output:
        sys.exit("Error: specify --output <model.qmx> or --info <model.qmx>")

    q_map = {
        "binary": QMX_QUANT_BINARY_1BIT,
        "ternary": QMX_QUANT_TERNARY_158,
        "q4": QMX_QUANT_Q4_SYM,
        "q8": QMX_QUANT_Q8_SYM,
        "q4_r32": QMX_QUANT_Q4_ROW32,
        "q8_r32": QMX_QUANT_Q8_ROW32,
    }
    p_quant = q_map[args.quant]

    compact = True if args.compact else (False if args.no_compact else None)

    if args.dummy:
        cfg, t_specs = build_dummy_model()
        hybrid = False
        if compact is None:
            compact = False
    elif args.hf:
        cfg, t_specs = load_hf_safetensors(args.hf, args.max_seq)
        hybrid = not args.no_hybrid_emb
    elif args.input:
        cfg, t_specs = load_karpathy_bin(args.input, args.tokenizer)
        hybrid = not args.no_hybrid_emb
    else:
        sys.exit("Error: specify --hf <repo_or_dir>, --input <file.bin>, --dummy, or --info <file.qmx>")

    lm_q = None if args.lm_head_quant == "auto" else q_map[args.lm_head_quant]
    export_qmx(args.output, cfg, t_specs, p_quant, hybrid, compact_align=compact, lm_head_quant=lm_q)


if __name__ == "__main__":
    main()