"""Multi-precision TurboQuant (`16-bit`, `8-bit`, `4-bit`, `2-bit`) for MLM v2.

Implements Google Research ICLR 2026 TurboQuant (`arXiv:2504.19874`,
`tonbistudio/turboquant-pytorch` & `ryancodrai/turbovec`):
1. Blockwise Fast Walsh-Hadamard Orthogonal Rotation (`FWHT`, `B=64`) with deterministic Rademacher
   sign-flipping (`D_s`) to concentrate weight coordinates into an outlier-free Gaussian/Beta law.
2. Optimal Gaussian Lloyd-Max scalar codebooks (`4-bit` 16-centroid and `2-bit` 4-centroid).
3. 1-Bit Quantized Johnson-Lindenstrauss (`QJL`) sign residual correction for `2-bit` & `4-bit`
   inner-product preservation with zero calibration data.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from safetensors.numpy import save as save_safetensors_bytes

BLOCK_SIZE = 64

# Optimal unit-variance Gaussian N(0, 1) Lloyd-Max centroids
LLOYD_MAX_2BIT = np.array([-1.5104, -0.4528, 0.4528, 1.5104], dtype=np.float32)
LLOYD_MAX_2BIT_BOUNDS = np.array([-0.9816, 0.0, 0.9816], dtype=np.float32)

LLOYD_MAX_4BIT = np.array(
    [
        -2.7326,
        -2.0690,
        -1.6180,
        -1.2562,
        -0.9423,
        -0.6568,
        -0.3880,
        -0.1284,
        0.1284,
        0.3880,
        0.6568,
        0.9423,
        1.2562,
        1.6180,
        2.0690,
        2.7326,
    ],
    dtype=np.float32,
)
LLOYD_MAX_4BIT_BOUNDS = 0.5 * (LLOYD_MAX_4BIT[:-1] + LLOYD_MAX_4BIT[1:])


def _rademacher_signs(block_size: int = BLOCK_SIZE, seed: int = 202604) -> np.ndarray:
    """Deterministic `{-1, +1}` Rademacher diagonal signs for randomized Hadamard rotation."""
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(1, block_size))


RADEMACHER_64 = _rademacher_signs(BLOCK_SIZE)


def _fwht_64(blocks: np.ndarray) -> np.ndarray:
    """Normalized orthogonal Fast Walsh-Hadamard Transform (`H_64 / 8`) over `(N, 64)` blocks."""
    x = blocks.astype(np.float32, copy=True)
    n_blocks = x.shape[0]
    h = 1
    while h < BLOCK_SIZE:
        x = x.reshape(n_blocks, BLOCK_SIZE // (2 * h), 2, h)
        a = x[:, :, 0, :]
        b = x[:, :, 1, :]
        x = np.stack([a + b, a - b], axis=2)
        h *= 2
    return x.reshape(n_blocks, BLOCK_SIZE) * (1.0 / 8.0)


def _rotate_blocks(flat: np.ndarray) -> tuple[np.ndarray, int]:
    """Pad 1D array to a multiple of 64 and apply randomized Hadamard rotation `FWHT(x * s)`."""
    orig_len = int(flat.size)
    rem = orig_len % BLOCK_SIZE
    pad_len = 0 if rem == 0 else (BLOCK_SIZE - rem)
    if pad_len > 0:
        padded = np.pad(flat.astype(np.float32), (0, pad_len))
    else:
        padded = flat.astype(np.float32, copy=False)
    blocks = padded.reshape(-1, BLOCK_SIZE)
    rotated = _fwht_64(blocks * RADEMACHER_64)
    return rotated, orig_len


def _unrotate_blocks(rotated: np.ndarray, orig_len: int) -> np.ndarray:
    """Apply inverse randomized Hadamard rotation `FWHT(u) * s` and slice to `orig_len`."""
    unrotated = _fwht_64(rotated) * RADEMACHER_64
    return unrotated.reshape(-1)[:orig_len]


def _should_quantize_tensor(name: str, arr: np.ndarray) -> bool:
    """Keep 1D norms/biases/gates, low-rank graph edges, and BLT table in exact float32/float16."""
    if name == "blt_token_ngrams" or arr.ndim < 2 or arr.size < 256:
        return False
    if name.endswith(
        (
            ".res",
            ".pre",
            ".post",
            ".router_bias",
            ".edge_scale",
            ".edge_u",
            ".edge_v",
            ".edge_proj_h",
            ".edge_proj_x",
            ".edge_bias",
            ".msg_down",
            ".msg_up",
            ".collab_gate",
        )
    ):
        return False
    return True


def quantize_params_turboquant(
    params: dict[str, jnp.ndarray | np.ndarray],
    bits: int = 4,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Quantize MLM v2 parameters into `16`, `8`, `4`, `2`, or `1` bit TurboQuant representation."""
    if bits not in (16, 8, 4, 2, 1):
        raise ValueError(
            f"Unsupported quantization bit-width: {bits}. Choose from 16, 8, 4, 2, 1."
        )

    q_tensors: dict[str, np.ndarray] = {}
    tensor_meta: dict[str, dict[str, Any]] = {}

    for k, v in params.items():
        arr = np.asarray(v, dtype=np.float32)
        if not _should_quantize_tensor(k, arr):
            dt = np.float16 if (bits == 16 and k != "blt_token_ngrams") else np.float32
            q_tensors[k] = arr.astype(dt)
            tensor_meta[k] = {
                "mode": "raw",
                "shape": list(arr.shape),
                "dtype": str(q_tensors[k].dtype),
            }
            continue

        if bits == 16:
            q_tensors[k] = arr.astype(np.float16)
            tensor_meta[k] = {"mode": "fp16", "shape": list(arr.shape)}
            continue

        rot_blocks, orig_len = _rotate_blocks(arr.reshape(-1))
        n_blocks = rot_blocks.shape[0]

        if bits == 8:
            abs_max = np.maximum(np.max(np.abs(rot_blocks), axis=-1, keepdims=True), 1e-6)
            scales = (abs_max / 127.0).astype(np.float16)
            safe_scales = np.maximum(scales.astype(np.float32), 1e-7)
            q_int8 = np.clip(
                np.round(rot_blocks / safe_scales), -127, 127
            ).astype(np.int8)
            q_tensors[f"{k}.qweight"] = q_int8
            q_tensors[f"{k}.scales"] = scales[:, 0]
            tensor_meta[k] = {
                "mode": "turboquant_8bit",
                "shape": list(arr.shape),
                "orig_len": orig_len,
                "n_blocks": n_blocks,
            }
        elif bits == 4:
            rms = np.sqrt(np.mean(np.square(rot_blocks), axis=-1, keepdims=True) + 1e-12)
            z = rot_blocks / rms
            idx = np.searchsorted(LLOYD_MAX_4BIT_BOUNDS, z).astype(np.uint8)  # (N, 64) in 0..15
            # Pack 2 4-bit indices per uint8 byte -> (N, 32)
            packed_4b = (idx[:, 0::2] & 0x0F) | ((idx[:, 1::2] & 0x0F) << 4)
            # 1-bit QJL residual sign correction + exact L2-norm preservation
            recon_z = LLOYD_MAX_4BIT[idx]
            res = z - recon_z
            res_scale = np.mean(np.abs(res), axis=-1, keepdims=True).astype(np.float16)
            qjl_dir = np.where(res >= 0.0, 1.0, -1.0).astype(np.float32)
            raw_z = recon_z + res_scale.astype(np.float32) * qjl_dir
            recon_rms = np.sqrt(np.mean(np.square(raw_z), axis=-1, keepdims=True) + 1e-12)
            qjl_bits = np.packbits((res >= 0.0).astype(np.uint8), axis=-1)  # (N, 8)
            q_tensors[f"{k}.qweight"] = packed_4b
            q_tensors[f"{k}.scales"] = (rms[:, 0] / recon_rms[:, 0]).astype(np.float16)
            q_tensors[f"{k}.qjl_bits"] = qjl_bits
            q_tensors[f"{k}.qjl_scales"] = res_scale[:, 0]
            tensor_meta[k] = {
                "mode": "turboquant_4bit_qjl",
                "shape": list(arr.shape),
                "orig_len": orig_len,
                "n_blocks": n_blocks,
            }
        elif bits == 2:
            is_salient_trunk = (
                k == "tok_embeddings.weight"
                or k.endswith(".router.weight")
                or ".nodes." not in k
            ) and k not in ("k", "v")
            if is_salient_trunk:
                rms = np.sqrt(np.mean(np.square(rot_blocks), axis=-1, keepdims=True) + 1e-12)
                z = rot_blocks / rms
                idx = np.searchsorted(LLOYD_MAX_4BIT_BOUNDS, z).astype(np.uint8)
                packed_4b = (idx[:, 0::2] & 0x0F) | ((idx[:, 1::2] & 0x0F) << 4)
                recon_z = LLOYD_MAX_4BIT[idx]
                res = z - recon_z
                res_scale = np.mean(np.abs(res), axis=-1, keepdims=True).astype(np.float16)
                qjl_dir = np.where(res >= 0.0, 1.0, -1.0).astype(np.float32)
                raw_z = recon_z + res_scale.astype(np.float32) * qjl_dir
                recon_rms = np.sqrt(np.mean(np.square(raw_z), axis=-1, keepdims=True) + 1e-12)
                qjl_bits = np.packbits((res >= 0.0).astype(np.uint8), axis=-1)
                q_tensors[f"{k}.qweight"] = packed_4b
                q_tensors[f"{k}.scales"] = (rms[:, 0] / recon_rms[:, 0]).astype(np.float16)
                q_tensors[f"{k}.qjl_bits"] = qjl_bits
                q_tensors[f"{k}.qjl_scales"] = res_scale[:, 0]
                tensor_meta[k] = {
                    "mode": "turboquant_4bit_qjl",
                    "shape": list(arr.shape),
                    "orig_len": orig_len,
                    "n_blocks": n_blocks,
                }
            else:
                rot_g = rot_blocks.reshape(n_blocks, 8, 8)
                rms_g = np.sqrt(np.mean(np.square(rot_g), axis=-1, keepdims=True) + 1e-12)
                z = (rot_g / rms_g).reshape(n_blocks, BLOCK_SIZE)
                idx = np.searchsorted(LLOYD_MAX_2BIT_BOUNDS, z).astype(np.uint8)  # (N, 64) in 0..3
                packed_2b = (
                    (idx[:, 0::4] & 0x03)
                    | ((idx[:, 1::4] & 0x03) << 2)
                    | ((idx[:, 2::4] & 0x03) << 4)
                    | ((idx[:, 3::4] & 0x03) << 6)
                )
                recon_z = LLOYD_MAX_2BIT[idx]
                res = z - recon_z
                res_scale = np.mean(np.abs(res), axis=-1, keepdims=True).astype(np.float16)
                qjl_dir = np.where(res >= 0.0, 1.0, -1.0).astype(np.float32)
                w_hat_g = (recon_z + res_scale.astype(np.float32) * qjl_dir).reshape(
                    n_blocks, 8, 8
                )
                w_rms_g = np.sqrt(np.mean(np.square(w_hat_g), axis=-1, keepdims=True) + 1e-12)
                opt_env_g = (rms_g / w_rms_g)[:, :, 0]
                env_max = np.maximum(np.max(opt_env_g, axis=-1, keepdims=True), 1e-7)
                u4 = np.clip(np.round((opt_env_g / env_max) * 15.0), 1, 15).astype(np.uint8)
                packed_u4 = (u4[:, 0::2] & 0x0F) | ((u4[:, 1::2] & 0x0F) << 4)
                env_hat = ((u4.astype(np.float32) / 15.0) * env_max)[:, :, None]
                raw_rot = (w_hat_g * env_hat).reshape(n_blocks, BLOCK_SIZE)
                orig_rms = np.sqrt(np.mean(np.square(rot_blocks), axis=-1) + 1e-12)
                recon_rms = np.sqrt(np.mean(np.square(raw_rot), axis=-1) + 1e-12)
                gain = orig_rms / recon_rms
                qjl_bits = np.packbits((res >= 0.0).astype(np.uint8), axis=-1)  # (N, 8)
                q_tensors[f"{k}.qweight"] = packed_2b
                q_tensors[f"{k}.scales"] = (env_max[:, 0] * gain).astype(np.float16)
                q_tensors[f"{k}.sub_scales"] = packed_u4
                q_tensors[f"{k}.qjl_bits"] = qjl_bits
                q_tensors[f"{k}.qjl_scales"] = res_scale[:, 0]
                tensor_meta[k] = {
                    "mode": "turboquant_2bit_qjl",
                    "shape": list(arr.shape),
                    "orig_len": orig_len,
                    "n_blocks": n_blocks,
                }
        elif bits == 1:
            # PB-LLM / BiLLM Hybrid 1-Bit + QJL with Shared G=8 4-Bit Sub-Group Envelope:
            # - Shared Core Trunk / Embeddings / Router (salient 13% of params) use 4-bit + QJL.
            # - Specialist Graph Node Banks (`.nodes.*`, 87% of params) and KV cache use
            #   1-Bit Sign (`qweight`) + 1-Bit QJL (`qjl_bits`) + 4-bit G=8 RMS envelope.
            is_salient_trunk = (
                k == "tok_embeddings.weight"
                or k.endswith(".router.weight")
                or ".nodes." not in k
            ) and k not in ("k", "v")
            if is_salient_trunk:
                rms = np.sqrt(np.mean(np.square(rot_blocks), axis=-1, keepdims=True) + 1e-12)
                z = rot_blocks / rms
                idx = np.searchsorted(LLOYD_MAX_4BIT_BOUNDS, z).astype(np.uint8)
                packed_4b = (idx[:, 0::2] & 0x0F) | ((idx[:, 1::2] & 0x0F) << 4)
                recon_z = LLOYD_MAX_4BIT[idx]
                res = z - recon_z
                res_scale = np.mean(np.abs(res), axis=-1, keepdims=True).astype(np.float16)
                qjl_dir = np.where(res >= 0.0, 1.0, -1.0).astype(np.float32)
                raw_z = recon_z + res_scale.astype(np.float32) * qjl_dir
                recon_rms = np.sqrt(np.mean(np.square(raw_z), axis=-1, keepdims=True) + 1e-12)
                qjl_bits = np.packbits((res >= 0.0).astype(np.uint8), axis=-1)
                q_tensors[f"{k}.qweight"] = packed_4b
                q_tensors[f"{k}.scales"] = (rms[:, 0] / recon_rms[:, 0]).astype(np.float16)
                q_tensors[f"{k}.qjl_bits"] = qjl_bits
                q_tensors[f"{k}.qjl_scales"] = res_scale[:, 0]
                tensor_meta[k] = {
                    "mode": "turboquant_4bit_qjl",
                    "shape": list(arr.shape),
                    "orig_len": orig_len,
                    "n_blocks": n_blocks,
                }
            else:
                rot_g = rot_blocks.reshape(n_blocks, 8, 8)
                rms_g = np.sqrt(np.mean(np.square(rot_g), axis=-1, keepdims=True) + 1e-12)
                a1_g = 0.9816 * rms_g
                b1_bits = np.packbits((rot_blocks >= 0.0).astype(np.uint8), axis=-1)
                b1_dir = np.where(rot_g >= 0.0, 1.0, -1.0).astype(np.float32)
                res_g = rot_g - a1_g * b1_dir
                b2_dir = np.where(res_g >= 0.0, 1.0, -1.0).astype(np.float32)
                a2_g = np.mean(res_g * b2_dir, axis=-1, keepdims=True)
                raw_g = a1_g * b1_dir + a2_g * b2_dir
                recon_rms_g = np.sqrt(np.mean(np.square(raw_g), axis=-1, keepdims=True) + 1e-12)
                gain_g = rms_g / recon_rms_g
                a1_g = a1_g * gain_g
                a2_g = a2_g * gain_g

                # Pack 8 sub-group relative RMS scales into 4 uint8 bytes (4-bit nibbles)
                env_max = np.maximum(np.max(rms_g[:, :, 0], axis=-1, keepdims=True), 1e-7)
                u4 = np.clip(np.round((rms_g[:, :, 0] / env_max) * 15.0), 1, 15).astype(np.uint8)
                packed_u4 = (u4[:, 0::2] & 0x0F) | ((u4[:, 1::2] & 0x0F) << 4)  # (N, 4)
                s1_max = (env_max[:, 0] * float(np.mean(a1_g / rms_g))).astype(np.float16)
                s2_max = (env_max[:, 0] * float(np.mean(a2_g / rms_g))).astype(np.float16)
                res_flat = res_g.reshape(n_blocks, BLOCK_SIZE)
                qjl_bits = np.packbits((res_flat >= 0.0).astype(np.uint8), axis=-1)
                q_tensors[f"{k}.qweight"] = b1_bits
                q_tensors[f"{k}.scales"] = s1_max
                q_tensors[f"{k}.sub_scales"] = packed_u4
                q_tensors[f"{k}.qjl_bits"] = qjl_bits
                q_tensors[f"{k}.qjl_scales"] = s2_max
                tensor_meta[k] = {
                    "mode": "turboquant_1bit_qjl",
                    "shape": list(arr.shape),
                    "orig_len": orig_len,
                    "n_blocks": n_blocks,
                }

    quant_info = {
        "bits": bits,
        "method": "fp16" if bits == 16 else f"turboquant_{bits}bit_fwht64_qjl",
        "block_size": BLOCK_SIZE,
        "tensors": tensor_meta,
    }
    return q_tensors, quant_info


def dequantize_params_turboquant(
    q_tensors: dict[str, np.ndarray],
    quant_info: dict[str, Any],
) -> dict[str, jnp.ndarray]:
    """Reconstruct `float32` JAX parameter dictionary from a TurboQuant tensor bundle."""
    tensor_meta = quant_info.get("tensors", {})
    params: dict[str, jnp.ndarray] = {}

    for k, meta in tensor_meta.items():
        mode = meta.get("mode", "raw")
        shape = tuple(meta["shape"])
        if mode in ("raw", "fp16"):
            params[k] = jnp.asarray(q_tensors[k], dtype=jnp.float32).reshape(shape)
            continue

        orig_len = int(meta["orig_len"])
        if mode == "turboquant_8bit":
            q_int8 = q_tensors[f"{k}.qweight"].astype(np.float32)
            scales = q_tensors[f"{k}.scales"].astype(np.float32)[:, None]
            rot_blocks = q_int8 * scales
            flat = _unrotate_blocks(rot_blocks, orig_len)
            params[k] = jnp.asarray(flat.reshape(shape), dtype=jnp.float32)
        elif mode == "turboquant_4bit_qjl":
            packed = q_tensors[f"{k}.qweight"]
            n_blocks = packed.shape[0]
            idx = np.empty((n_blocks, BLOCK_SIZE), dtype=np.uint8)
            idx[:, 0::2] = packed & 0x0F
            idx[:, 1::2] = (packed >> 4) & 0x0F
            z_hat = LLOYD_MAX_4BIT[idx]
            if f"{k}.qjl_bits" in q_tensors:
                qjl_signs = np.unpackbits(q_tensors[f"{k}.qjl_bits"], axis=-1)[:, :BLOCK_SIZE]
                qjl_dir = np.where(qjl_signs > 0, 1.0, -1.0).astype(np.float32)
                r_scale = q_tensors[f"{k}.qjl_scales"].astype(np.float32)[:, None]
                z_hat = z_hat + r_scale * qjl_dir
            scales = q_tensors[f"{k}.scales"].astype(np.float32)[:, None]
            rot_blocks = z_hat * scales
            flat = _unrotate_blocks(rot_blocks, orig_len)
            params[k] = jnp.asarray(flat.reshape(shape), dtype=jnp.float32)
        elif mode == "turboquant_2bit_qjl":
            packed = q_tensors[f"{k}.qweight"]
            n_blocks = packed.shape[0]
            idx = np.empty((n_blocks, BLOCK_SIZE), dtype=np.uint8)
            idx[:, 0::4] = packed & 0x03
            idx[:, 1::4] = (packed >> 2) & 0x03
            idx[:, 2::4] = (packed >> 4) & 0x03
            idx[:, 3::4] = (packed >> 6) & 0x03
            z_hat = LLOYD_MAX_2BIT[idx]
            if f"{k}.qjl_bits" in q_tensors:
                qjl_signs = np.unpackbits(q_tensors[f"{k}.qjl_bits"], axis=-1)[:, :BLOCK_SIZE]
                qjl_dir = np.where(qjl_signs > 0, 1.0, -1.0).astype(np.float32)
                r_scale = q_tensors[f"{k}.qjl_scales"].astype(np.float32)[:, None]
                z_hat = z_hat + r_scale * qjl_dir
            scales = q_tensors[f"{k}.scales"].astype(np.float32)
            if f"{k}.sub_scales" in q_tensors:
                packed_u4 = q_tensors[f"{k}.sub_scales"]
                u4 = np.empty((n_blocks, 8), dtype=np.uint8)
                u4[:, 0::2] = packed_u4 & 0x0F
                u4[:, 1::2] = (packed_u4 >> 4) & 0x0F
                env_hat = (u4.astype(np.float32) / 15.0) * scales[:, None]
                rot_blocks = (z_hat.reshape(n_blocks, 8, 8) * env_hat[:, :, None]).reshape(
                    n_blocks, BLOCK_SIZE
                )
            else:
                rot_blocks = z_hat * scales[:, None]
            flat = _unrotate_blocks(rot_blocks, orig_len)
            params[k] = jnp.asarray(flat.reshape(shape), dtype=jnp.float32)
        elif mode == "turboquant_1bit_qjl":
            b1_signs = np.unpackbits(q_tensors[f"{k}.qweight"], axis=-1)[:, :BLOCK_SIZE]
            n_blocks = b1_signs.shape[0]
            b1_dir = np.where(b1_signs > 0, 1.0, -1.0).astype(np.float32).reshape(n_blocks, 8, 8)
            raw_s1 = q_tensors[f"{k}.scales"].astype(np.float32)
            if f"{k}.sub_scales" in q_tensors:
                packed_u4 = q_tensors[f"{k}.sub_scales"]
                u4 = np.empty((n_blocks, 8), dtype=np.uint8)
                u4[:, 0::2] = packed_u4 & 0x0F
                u4[:, 1::2] = (packed_u4 >> 4) & 0x0F
                env_hat = (u4.astype(np.float32) / 15.0)[:, :, None]
            else:
                env_hat = np.ones((n_blocks, 8, 1), dtype=np.float32)
            a1 = (raw_s1[:, :, None] if raw_s1.ndim == 2 else raw_s1[:, None, None]) * env_hat
            rot_g = a1 * b1_dir
            if f"{k}.qjl_bits" in q_tensors:
                qjl_signs = np.unpackbits(q_tensors[f"{k}.qjl_bits"], axis=-1)[:, :BLOCK_SIZE]
                qjl_flat = np.where(qjl_signs > 0, 1.0, -1.0).astype(np.float32)
                qjl_dir = qjl_flat.reshape(n_blocks, 8, 8)
                raw_s2 = q_tensors[f"{k}.qjl_scales"].astype(np.float32)
                s2_exp = raw_s2[:, :, None] if raw_s2.ndim == 2 else raw_s2[:, None, None]
                r_scale = s2_exp * env_hat
                rot_g = rot_g + r_scale * qjl_dir
            flat = _unrotate_blocks(rot_g.reshape(n_blocks, BLOCK_SIZE), orig_len)
            params[k] = jnp.asarray(flat.reshape(shape), dtype=jnp.float32)
        else:
            raise ValueError(f"Unknown TurboQuant tensor mode: {mode}")

    return params


def _encode_cachegen_anchor_delta(
    arr: np.ndarray, anchor_stride: int = 8
) -> tuple[np.ndarray, int]:
    """LMCache / CacheGen Anchor-Token Temporal Delta Encoding along the sequence axis.

    Divides token sequence into contiguous chunks of `anchor_stride` tokens:
      - Anchor token `t = g * G` stores absolute value `K_{g*G}`
      - Subsequent tokens `t = g * G + r` (`r = 1..G-1`) store temporal delta `K_t - K_{g*G}`
    """
    if arr.ndim == 4 and arr.shape[1] >= 2:
        seq_axis = 1
    elif arr.ndim == 3 and arr.shape[1] >= 2:
        seq_axis = 1
    else:
        return arr, 0
    seq_len = arr.shape[seq_axis]
    stride = max(2, min(int(anchor_stride), seq_len))
    out = arr.copy()
    for t in range(seq_len):
        anchor_t = (t // stride) * stride
        if t != anchor_t:
            if seq_axis == 1:
                out[:, t] = arr[:, t] - arr[:, anchor_t]
    return out, stride


def _decode_cachegen_anchor_delta(
    arr: jnp.ndarray, seq_axis: int, stride: int
) -> jnp.ndarray:
    """Reconstruct full KV sequence from CacheGen Anchor-Token + Temporal Delta representation."""
    if stride <= 1 or seq_axis != 1:
        return arr
    out_np = np.asarray(arr, dtype=np.float32).copy()
    seq_len = out_np.shape[1]
    for t in range(seq_len):
        anchor_t = (t // stride) * stride
        if t != anchor_t:
            out_np[:, t] = out_np[:, anchor_t] + out_np[:, t]
    return jnp.asarray(out_np, dtype=jnp.float32)


def quantize_kv_cache_turboquant(
    k: np.ndarray | jnp.ndarray,
    v: np.ndarray | jnp.ndarray,
    bits: int = 4,
    anchor_stride: int = 8,
    use_anchor_delta: bool = True,
) -> dict[str, Any]:
    """Bit-pack `(K, V)` KV Cache tensors into `TurboQuant + CacheGen + QJL` format.

    Combines Google Research ICLR 2026 `TurboQuant` (`arXiv:2504.19874`) with `LMCache / CacheGen`
    Anchor-Token Temporal Delta Encoding:
      - CacheGen Anchor-Token Temporal Delta (`Delta K_t = K_t - K_{anchor}`) along token axis
      - Fast Walsh-Hadamard Orthogonal Rotation (`FWHT`)
      - Gaussian Lloyd-Max `4-bit` (16-centroid) or `2-bit` (4-centroid) scalar quantization
      - `1-Bit QJL` (Quantized Johnson-Lindenstrauss) residual signs for unbiased inner products
    """
    if bits not in (1, 2, 4):
        raise ValueError(f"Unsupported KV cache bits={bits}; expected 4, 2, or 1.")
    k_np = np.asarray(k, dtype=np.float32)
    v_np = np.asarray(v, dtype=np.float32)
    raw_bytes = k_np.nbytes + v_np.nbytes
    if use_anchor_delta:
        k_enc, k_stride = _encode_cachegen_anchor_delta(k_np, anchor_stride=anchor_stride)
        v_enc, _ = _encode_cachegen_anchor_delta(v_np, anchor_stride=anchor_stride)
    else:
        k_enc, k_stride = k_np, 0
        v_enc = v_np
    q_tensors, meta = quantize_params_turboquant({"k": k_enc, "v": v_enc}, bits=bits)
    meta["cachegen_anchor_stride"] = int(k_stride)
    meta["cachegen_seq_axis"] = 1 if k_stride > 0 else -1
    packed_bytes = sum(int(arr.nbytes) for arr in q_tensors.values())
    method_sfx = "_cachegen_delta" if k_stride > 0 else ""
    return {
        "bits": bits,
        "method": f"turboquant_kv_{bits}bit_fwht_qjl{method_sfx}",
        "anchor_stride": int(k_stride),
        "tensors": q_tensors,
        "meta": meta,
        "raw_fp32_bytes": raw_bytes,
        "raw_fp16_bytes": raw_bytes // 2,
        "packed_bytes": packed_bytes,
        "compression_vs_fp32": round(raw_bytes / max(packed_bytes, 1), 2),
        "compression_vs_fp16": round((raw_bytes / 2.0) / max(packed_bytes, 1), 2),
    }


def dequantize_kv_cache_turboquant(
    packed_kv: dict[str, Any],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Dequantize a `TurboQuant + CacheGen` bit-packed KV Cache back into `(K, V)`."""
    meta = packed_kv["meta"]
    deq = dequantize_params_turboquant(packed_kv["tensors"], meta)
    stride = int(meta.get("cachegen_anchor_stride", 0))
    seq_axis = int(meta.get("cachegen_seq_axis", -1))
    if stride > 1 and seq_axis == 1:
        k_out = _decode_cachegen_anchor_delta(deq["k"], seq_axis=seq_axis, stride=stride)
        v_out = _decode_cachegen_anchor_delta(deq["v"], seq_axis=seq_axis, stride=stride)
        return k_out, v_out
    return deq["k"], deq["v"]


def save_quantized_checkpoint_v2(
    path: Path | str,
    params: dict[str, jnp.ndarray],
    config: Any,
    tokenizer: Any,
    *,
    bits: int = 4,
    kv_cache_bits: int | None = None,
    step: int = 0,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Save a quantized (`16-bit`, `8-bit`, `4-bit`, or `2-bit`) `.mlm2` checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    q_tensors, quant_info = quantize_params_turboquant(params, bits=bits)
    st_bytes = save_safetensors_bytes(q_tensors)

    tok_tmp = path.with_suffix(".tokenizer.tmp.json")
    tokenizer.save(tok_tmp)
    tok_json = tok_tmp.read_text(encoding="utf-8")
    tok_tmp.unlink(missing_ok=True)

    eff_kv_bits = (
        int(kv_cache_bits)
        if kv_cache_bits is not None
        else (1 if bits == 1 else (2 if bits == 2 else 4))
    )
    cfg_dump = config.model_dump()
    cfg_dump["kv_cache_bits"] = eff_kv_bits

    meta = dict(metadata or {})
    meta["quantization_bits"] = bits
    meta["quantization_method"] = quant_info["method"]
    meta["kv_cache_bits"] = eff_kv_bits

    step_val = int(step.get("step", 0)) if isinstance(step, dict) else int(step)
    manifest = {
        "format": 2,
        "step": step_val,
        "config": cfg_dump,
        "metadata": meta,
        "quantization": quant_info,
    }
    tmp_zip = path.with_suffix(".mlm2.tmp")
    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        zf.writestr("tokenizer.json", tok_json)
        zf.writestr("model.safetensors", st_bytes)
    tmp_zip.replace(path)

    return {
        "path": str(path),
        "bits": bits,
        "kv_cache_bits": eff_kv_bits,
        "method": quant_info["method"],
        "file_size_bytes": path.stat().st_size,
        "file_size_mib": round(path.stat().st_size / (1024.0 * 1024.0), 3),
    }


def export_all_precisions_v2(
    base_checkpoint: Path | str,
    out_dir: Path | str | None = None,
    stem: str | None = None,
    export_gguf: bool = True,
    kv_cache_bits_map: dict[int, int] | None = None,
    bits_list: tuple[int, ...] = (16, 8, 4, 2, 1),
) -> dict[str, dict[str, Any]]:
    """Export `16-bit`, `8-bit`, `4-bit`, `2-bit`, and `1-bit` `.mlm2` and `.gguf` variants."""
    from mlm.gguf_export import export_gguf_v2
    from mlm.model_v2 import load_checkpoint_v2

    base_path = Path(base_checkpoint)
    target_dir = Path(out_dir) if out_dir is not None else (base_path.parent / "quantized")
    target_dir.mkdir(parents=True, exist_ok=True)
    name_stem = stem or base_path.stem

    params, cfg, tok, raw_meta = load_checkpoint_v2(base_path)
    step_num = int(raw_meta.get("step", 0)) if isinstance(raw_meta, dict) else int(raw_meta)
    default_kv_map = {16: 4, 8: 4, 4: 4, 2: 2, 1: 1}
    if kv_cache_bits_map:
        default_kv_map.update(kv_cache_bits_map)
    results: dict[str, dict[str, Any]] = {}
    for bits in bits_list:
        mlm2_p = target_dir / f"{name_stem}_{bits}bit.mlm2"
        kv_b = default_kv_map.get(bits, 4)
        info = save_quantized_checkpoint_v2(
            mlm2_p,
            params,
            cfg,
            tok,
            bits=bits,
            kv_cache_bits=kv_b,
            step=step_num,
        )
        info["mlm2_path"] = str(mlm2_p)
        if export_gguf:
            gguf_p = target_dir / f"{name_stem}_{bits}bit.gguf"
            gguf_info = export_gguf_v2(gguf_p, params, cfg, tok, bits=bits)
            info["gguf_path"] = str(gguf_p)
            info["gguf_size_mib"] = round(gguf_info["file_size_bytes"] / (1024.0 * 1024.0), 3)
        results[f"{bits}bit"] = info
    return results
