"""Single-file atomic checkpoints with parameters and full Adam state."""

import json
import os
import tempfile
import zipfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from mlm.config import Config
from mlm.graph import model_topology
from mlm.training import create_state


def quantize_params_b8(params: dict) -> dict:
    """Quantize float weight tensors to symmetric per-channel 8-bit (`int8`) with `fp16` scales."""
    packed = {}
    for k, v in params.items():
        arr = np.asarray(v, dtype=np.float32)
        if arr.ndim >= 2 and arr.size >= 64:
            reduce_axes = tuple(range(arr.ndim - 1))
            max_abs = np.amax(np.abs(arr), axis=reduce_axes, keepdims=True)
            scale = np.maximum(max_abs / 127.0, 1e-8).astype(np.float16)
            q = np.clip(np.round(arr / scale.astype(np.float32)), -127, 127).astype(np.int8)
            packed[k] = {"__b8__": True, "q": q, "scale": scale}
        else:
            packed[k] = arr.astype(np.float16) if arr.size >= 16 else arr
    return packed


def dequantize_params_b8(packed: dict) -> dict:
    """Restore `fp32` JAX arrays from a `b8` (`int8`) quantized parameter dictionary."""
    restored = {}
    for k, v in packed.items():
        if isinstance(v, dict) and v.get("__b8__"):
            q = np.asarray(v["q"], dtype=np.float32)
            scale = np.asarray(v["scale"], dtype=np.float32)
            restored[k] = jnp.asarray(q * scale, dtype=jnp.float32)
        else:
            restored[k] = jnp.asarray(v, dtype=jnp.float32)
    return restored


def save_checkpoint(
    path: Path,
    state,
    config: Config,
    metadata: dict,
    quantize: str = "none",
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": 1,
        "quantize": quantize,
        "config": config.model_dump(),
        "topology": model_topology(config),
        "metadata": metadata,
    }
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        comp = zipfile.ZIP_DEFLATED if quantize == "b8" else zipfile.ZIP_STORED
        with zipfile.ZipFile(temporary, "w", compression=comp) as archive:
            archive.writestr("manifest.json", json.dumps(manifest, allow_nan=False))
            if quantize == "b8":
                b8_payload = {
                    "step": int(state.step),
                    "params": quantize_params_b8(state.params),
                }
                archive.writestr("params_b8.msgpack", serialization.to_bytes(b8_payload))
            else:
                archive.writestr("state.msgpack", serialization.to_bytes(state))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path: Path):
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest["format"] != 1:
            raise ValueError("Unsupported checkpoint version")
        config = Config.model_validate(manifest["config"])
        if manifest["topology"] != model_topology(config):
            raise ValueError("Checkpoint topology does not match this implementation")
        base_state = create_state(config)
        if manifest.get("quantize") == "b8":
            raw_b8 = serialization.msgpack_restore(archive.read("params_b8.msgpack"))
            params = dequantize_params_b8(raw_b8["params"])
            state = base_state.replace(params=params, step=int(raw_b8.get("step", 0)))
        else:
            state = serialization.from_bytes(base_state, archive.read("state.msgpack"))
    return jax.device_put(state), config, manifest["metadata"]
