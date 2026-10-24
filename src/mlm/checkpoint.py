"""Single-file atomic checkpoints with parameters and full Adam state."""

import json
import os
import tempfile
import zipfile
from pathlib import Path

import jax
from flax import serialization

from mlm.config import Config
from mlm.graph import model_topology
from mlm.training import create_state


def save_checkpoint(path: Path, state, config: Config, metadata: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": 1,
        "config": config.model_dump(),
        "topology": model_topology(config),
        "metadata": metadata,
    }
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("manifest.json", json.dumps(manifest, allow_nan=False))
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
        state = serialization.from_bytes(create_state(config), archive.read("state.msgpack"))
    return jax.device_put(state), config, manifest["metadata"]
