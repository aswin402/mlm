"""GGUF v3 exporter and `llama.cpp` verification utilities for MLM v2."""

from __future__ import annotations

import struct
import subprocess
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from mlm.model_v2 import ConfigV2
from mlm.tokenizer import BPETokenizer

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32

# GGUF metadata value types
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10

# GGML tensor element types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q2_K = 10
GGML_TYPE_TQ1_0 = 34


def _pack_ggml_tq1_0(arr: np.ndarray) -> bytes:
    """Pack float32 array into GGML `TQ1_0` 1-bit super-blocks (256 weights -> 50 bytes)."""
    flat = arr.reshape(-1).astype(np.float32)
    rem = flat.size % 256
    if rem != 0:
        flat = np.pad(flat, (0, 256 - rem))
    sblocks = flat.reshape(-1, 256)
    a1 = np.mean(np.abs(sblocks), axis=-1, keepdims=True)
    b1_bits = np.packbits((sblocks >= 0.0).astype(np.uint8), axis=-1)  # (N, 32)
    b1_dir = np.where(sblocks >= 0.0, 1.0, -1.0).astype(np.float32)
    res = sblocks - a1 * b1_dir
    qjl_bits = np.packbits((res >= 0.0).astype(np.uint8), axis=-1)[:, :16]  # (N, 16)
    dt = np.dtype([("qs", "u1", (32,)), ("qh", "u1", (16,)), ("d", "<f2")])
    out = np.zeros(sblocks.shape[0], dtype=dt)
    out["qs"] = b1_bits
    out["qh"] = qjl_bits
    out["d"] = a1[:, 0].astype(np.float16)
    return out.tobytes()


def _pack_ggml_q8_0(arr: np.ndarray) -> bytes:
    """Pack float32 array into GGML `Q8_0` blocks (32 weights: 2B fp16 scale + 32B int8)."""
    flat = arr.reshape(-1).astype(np.float32)
    rem = flat.size % 32
    if rem != 0:
        flat = np.pad(flat, (0, 32 - rem))
    blocks = flat.reshape(-1, 32)
    amax = np.max(np.abs(blocks), axis=-1, keepdims=True)
    d = (amax / 127.0).astype(np.float16)
    inv_d = np.where(d.astype(np.float32) > 0, 1.0 / d.astype(np.float32), 0.0)
    qs = np.clip(np.round(blocks * inv_d), -127, 127).astype(np.int8)
    dt = np.dtype([("d", "<f2"), ("qs", "i1", (32,))])
    out = np.empty(blocks.shape[0], dtype=dt)
    out["d"] = d[:, 0]
    out["qs"] = qs
    return out.tobytes()


def _pack_ggml_q4_0(arr: np.ndarray) -> bytes:
    """Pack float32 array into GGML `Q4_0` blocks (32 weights: 2B fp16 scale + 16B nibbles)."""
    flat = arr.reshape(-1).astype(np.float32)
    rem = flat.size % 32
    if rem != 0:
        flat = np.pad(flat, (0, 32 - rem))
    blocks = flat.reshape(-1, 32)
    amax = np.max(np.abs(blocks), axis=-1, keepdims=True)
    d = (amax / 7.0).astype(np.float16)
    inv_d = np.where(d.astype(np.float32) > 0, 1.0 / d.astype(np.float32), 0.0)
    qi = np.clip(np.round(blocks * inv_d) + 8.0, 0, 15).astype(np.uint8)
    packed = (qi[:, :16] & 0x0F) | ((qi[:, 16:] & 0x0F) << 4)
    dt = np.dtype([("d", "<f2"), ("qs", "u1", (16,))])
    out = np.empty(blocks.shape[0], dtype=dt)
    out["d"] = d[:, 0]
    out["qs"] = packed
    return out.tobytes()


def _pack_ggml_q2_k(arr: np.ndarray) -> bytes:
    """Pack float32 array into GGML `Q2_K` super-blocks (256 weights -> 84 bytes)."""
    flat = arr.reshape(-1).astype(np.float32)
    rem = flat.size % 256
    if rem != 0:
        flat = np.pad(flat, (0, 256 - rem))
    sblocks = flat.reshape(-1, 256)
    fmin = np.min(sblocks, axis=-1, keepdims=True)
    fmax = np.max(sblocks, axis=-1, keepdims=True)
    d = np.maximum((fmax - fmin) / 3.0, 1e-8).astype(np.float16)
    dmin = (-fmin).astype(np.float16)
    qi = np.clip(np.round((sblocks - fmin) / d.astype(np.float32)), 0, 3).astype(np.uint8)
    packed = (
        (qi[:, 0::4] & 0x03)
        | ((qi[:, 1::4] & 0x03) << 2)
        | ((qi[:, 2::4] & 0x03) << 4)
        | ((qi[:, 3::4] & 0x03) << 6)
    )
    dt = np.dtype([("scales", "u1", (16,)), ("qs", "u1", (64,)), ("d", "<f2"), ("dmin", "<f2")])
    out = np.zeros(sblocks.shape[0], dtype=dt)
    out["scales"] = 0x11
    out["qs"] = packed
    out["d"] = d[:, 0]
    out["dmin"] = dmin[:, 0]
    return out.tobytes()


def _pack_string(s: str) -> bytes:
    raw = s.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _pack_kv(key: str, val_type: int, value: Any) -> bytes:
    out = bytearray(_pack_string(key))
    out.extend(struct.pack("<I", val_type))
    if val_type == GGUF_TYPE_UINT32:
        out.extend(struct.pack("<I", int(value)))
    elif val_type == GGUF_TYPE_INT32:
        out.extend(struct.pack("<i", int(value)))
    elif val_type == GGUF_TYPE_UINT64:
        out.extend(struct.pack("<Q", int(value)))
    elif val_type == GGUF_TYPE_FLOAT32:
        out.extend(struct.pack("<f", float(value)))
    elif val_type == GGUF_TYPE_BOOL:
        out.extend(struct.pack("<?", bool(value)))
    elif val_type == GGUF_TYPE_STRING:
        out.extend(_pack_string(str(value)))
    elif val_type == GGUF_TYPE_ARRAY:
        elem_type, items = value
        out.extend(struct.pack("<IQ", elem_type, len(items)))
        if elem_type == GGUF_TYPE_STRING:
            for item in items:
                out.extend(_pack_string(str(item)))
        elif elem_type == GGUF_TYPE_FLOAT32:
            out.extend(struct.pack(f"<{len(items)}f", *(float(x) for x in items)))
        elif elem_type == GGUF_TYPE_INT32:
            out.extend(struct.pack(f"<{len(items)}i", *(int(x) for x in items)))
        elif elem_type == GGUF_TYPE_UINT32:
            out.extend(struct.pack(f"<{len(items)}I", *(int(x) for x in items)))
        else:
            raise ValueError(f"Unsupported GGUF array element type: {elem_type}")
    else:
        raise ValueError(f"Unsupported GGUF value type: {val_type}")
    return bytes(out)


def _align_offset(offset: int, alignment: int = GGUF_DEFAULT_ALIGNMENT) -> int:
    rem = offset % alignment
    return offset if rem == 0 else offset + (alignment - rem)


def _encode_ggml_tensor(
    name: str, arr: np.ndarray, bits: int
) -> tuple[tuple[int, ...], bytes, int]:
    """Encode a contiguous float32 array into GGML raw or block-quantized bytes."""
    can_quant = (
        arr.ndim >= 2
        and arr.size >= 256
        and name != "blt_token_ngrams"
        and not name.endswith((".res", ".pre", ".post"))
    )
    if not can_quant or bits == 32:
        f32 = arr.astype(np.float32)
        return f32.shape, f32.tobytes(), GGML_TYPE_F32
    if bits == 16:
        f16 = arr.astype(np.float16)
        return f16.shape, f16.tobytes(), GGML_TYPE_F16
    if bits == 8 and arr.shape[-1] % 32 == 0:
        return arr.shape, _pack_ggml_q8_0(arr), GGML_TYPE_Q8_0
    if bits == 4 and arr.shape[-1] % 32 == 0:
        return arr.shape, _pack_ggml_q4_0(arr), GGML_TYPE_Q4_0
    if bits == 2 and arr.size % 256 == 0:
        return arr.shape, _pack_ggml_q2_k(arr), GGML_TYPE_Q2_K
    if bits == 1 and arr.size % 256 == 0:
        return arr.shape, _pack_ggml_tq1_0(arr), GGML_TYPE_TQ1_0
    f16 = arr.astype(np.float16)
    return f16.shape, f16.tobytes(), GGML_TYPE_F16


def export_gguf_v2(
    path: Path | str,
    params: dict[str, jnp.ndarray],
    config: ConfigV2,
    tokenizer: BPETokenizer,
    *,
    use_f16: bool = False,
    bits: int = 32,
) -> dict[str, Any]:
    """Export an MLM v2 model and BPE tokenizer to a `llama.cpp`-compatible `.gguf` v3 file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    eff_bits = 16 if (use_f16 and bits == 32) else int(bits)

    arch = "llama" if config.arch_version == "v2_llama" else "lfm2"
    tokens, scores, token_types = tokenizer.get_vocab_tokens_and_scores()

    kvs: list[tuple[str, int, Any]] = [
        ("general.architecture", GGUF_TYPE_STRING, arch),
        ("general.name", GGUF_TYPE_STRING, f"mlm-{config.arch_version}"),
        ("general.alignment", GGUF_TYPE_UINT32, GGUF_DEFAULT_ALIGNMENT),
        ("general.quantization_bits", GGUF_TYPE_UINT32, eff_bits),
        (f"{arch}.context_length", GGUF_TYPE_UINT32, config.max_seq_len),
        (f"{arch}.embedding_length", GGUF_TYPE_UINT32, config.d_model),
        (f"{arch}.block_count", GGUF_TYPE_UINT32, config.n_layers),
        (f"{arch}.feed_forward_length", GGUF_TYPE_UINT32, config.ffn_dim),
        (f"{arch}.attention.head_count", GGUF_TYPE_UINT32, config.n_heads),
        (f"{arch}.attention.head_count_kv", GGUF_TYPE_UINT32, config.n_kv_heads),
        (f"{arch}.attention.layer_norm_rms_epsilon", GGUF_TYPE_FLOAT32, config.rms_eps),
        (f"{arch}.rope.freq_base", GGUF_TYPE_FLOAT32, config.rope_theta),
        (f"{arch}.rope.dimension_count", GGUF_TYPE_UINT32, config.head_dim),
        ("tokenizer.ggml.model", GGUF_TYPE_STRING, "gpt2"),
        ("tokenizer.ggml.pre", GGUF_TYPE_STRING, "gpt-2"),
        ("tokenizer.ggml.tokens", GGUF_TYPE_ARRAY, (GGUF_TYPE_STRING, tokens)),
        ("tokenizer.ggml.scores", GGUF_TYPE_ARRAY, (GGUF_TYPE_FLOAT32, scores)),
        ("tokenizer.ggml.token_type", GGUF_TYPE_ARRAY, (GGUF_TYPE_INT32, token_types)),
        ("tokenizer.ggml.bos_token_id", GGUF_TYPE_UINT32, tokenizer.bos_id),
        ("tokenizer.ggml.eos_token_id", GGUF_TYPE_UINT32, tokenizer.eos_id),
        ("tokenizer.ggml.padding_token_id", GGUF_TYPE_UINT32, tokenizer.pad_id),
        ("tokenizer.ggml.add_bos_token", GGUF_TYPE_BOOL, True),
    ]
    if arch == "lfm2":
        kvs.append((f"{arch}.shortconv.l_cache", GGUF_TYPE_UINT32, config.conv_kernel))

    # Prepare GGML tensors (converting JAX `(in, out)` linear matrices to GGML `(out, in)`)
    gguf_tensors: list[tuple[str, tuple[int, ...], bytes, int]] = []
    for k, v in sorted(params.items()):
        arr = np.asarray(v, dtype=np.float32)
        name = "token_embd.weight" if k == "tok_embeddings.weight" else k
        if arr.ndim == 2 and name not in ("token_embd.weight", "output.weight"):
            arr = np.ascontiguousarray(arr.T)
        else:
            arr = np.ascontiguousarray(arr)

        t_shape, raw_bytes, ttype = _encode_ggml_tensor(name, arr, eff_bits)
        gguf_tensors.append((name, t_shape, raw_bytes, ttype))

    if config.tie_embeddings and "output.weight" not in params:
        emb = np.ascontiguousarray(np.asarray(params["tok_embeddings.weight"], dtype=np.float32))
        t_shape, raw_bytes, ttype = _encode_ggml_tensor("output.weight", emb, eff_bits)
        gguf_tensors.append(("output.weight", t_shape, raw_bytes, ttype))

    header_And_meta = bytearray()
    header_And_meta.extend(GGUF_MAGIC)
    header_And_meta.extend(
        struct.pack("<IQQ", GGUF_VERSION, len(gguf_tensors), len(kvs))
    )
    for k_name, k_type, k_val in kvs:
        header_And_meta.extend(_pack_kv(k_name, k_type, k_val))

    tensor_infos = bytearray()
    data_offsets: list[int] = []
    running_offset = 0
    for name, t_shape, raw_bytes, ttype in gguf_tensors:
        running_offset = _align_offset(running_offset, GGUF_DEFAULT_ALIGNMENT)
        data_offsets.append(running_offset)
        tensor_infos.extend(_pack_string(name))
        n_dims = len(t_shape)
        tensor_infos.extend(struct.pack("<I", n_dims))
        # GGML dimensions are ordered innermost-first (reverse of NumPy shape)
        for dim in reversed(t_shape):
            tensor_infos.extend(struct.pack("<Q", int(dim)))
        tensor_infos.extend(struct.pack("<IQ", ttype, running_offset))
        running_offset += len(raw_bytes)

    pre_data_len = len(header_And_meta) + len(tensor_infos)
    aligned_pre_len = _align_offset(pre_data_len, GGUF_DEFAULT_ALIGNMENT)
    pad_header = b"\x00" * (aligned_pre_len - pre_data_len)

    tmp_path = path.with_suffix(".gguf.tmp")
    with tmp_path.open("wb") as f:
        f.write(header_And_meta)
        f.write(tensor_infos)
        f.write(pad_header)
        curr_data_pos = 0
        for (_, _, raw_bytes, _), target_off in zip(gguf_tensors, data_offsets, strict=True):
            if target_off > curr_data_pos:
                f.write(b"\x00" * (target_off - curr_data_pos))
                curr_data_pos = target_off
            f.write(raw_bytes)
            curr_data_pos += len(raw_bytes)
    tmp_path.replace(path)

    return {
        "path": str(path),
        "architecture": arch,
        "bits": eff_bits,
        "tensor_count": len(gguf_tensors),
        "metadata_kv_count": len(kvs),
        "file_size_bytes": path.stat().st_size,
    }


def verify_gguf_with_rust(path: Path | str) -> dict[str, Any]:
    """Verify a `.gguf` file using the native Rust `mlm-gguf-inspect` binary."""
    import json

    rust_bin = (
        Path(__file__).resolve().parents[2]
        / "crates"
        / "mlm_core"
        / "target"
        / "release"
        / "mlm-gguf-inspect"
    )
    if rust_bin.exists():
        proc = subprocess.run(
            [str(rust_bin), str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(proc.stdout.strip())

    # Fallback binary header check if Rust release binary has not been compiled yet
    p = Path(path)
    with p.open("rb") as f:
        hdr = f.read(24)
    magic = hdr[:4]
    if magic != GGUF_MAGIC:
        raise ValueError("Invalid GGUF magic")
    version, t_count, kv_count = struct.unpack("<IQQ", hdr[4:24])
    return {
        "valid": version == GGUF_VERSION,
        "version": version,
        "tensor_count": t_count,
        "metadata_kv_count": kv_count,
        "file_size_bytes": p.stat().st_size,
    }
