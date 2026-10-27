import hashlib
import json

import pytest

from mlm.future_replication_run import PILOT_JSON, validated_pilot_paths


def test_pilot_artifact_validation_rejects_missing_and_mutated_files(tmp_path):
    published = tmp_path / "published"
    raw = tmp_path / "raw"

    def sha(data):
        return hashlib.sha256(data).hexdigest()

    def write(root, name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    records = {}
    for arm in ["control", "future_byte"]:
        for name in ["final.mlm", "head.msgpack"]:
            p = raw / arm / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes((arm + name).encode())
        records[arm] = {
            "final_backbone_sha256": sha((raw / arm / "final.mlm").read_bytes()),
            "head_sha256": sha((raw / arm / "head.msgpack").read_bytes()),
        }
    for root in [published, raw]:
        for name in PILOT_JSON:
            write(root, name, {})
        write(root, "protocol.json", {"settings": {}})
        write(root, "continuation-protocol.json", {"settings": {}})
        write(root, "summary.json", {"arms": records})
        for arm, r in records.items():
            write(root, arm + "/summary.json", r)
        write(
            root,
            "execution.json",
            {"protocol_sha256": sha((root / "protocol.json").read_bytes()), "settings": {}},
        )
        write(
            root,
            "continuation-execution.json",
            {
                "protocol_sha256": sha((root / "continuation-protocol.json").read_bytes()),
                "settings": {},
            },
        )
    manifest = {name: sha((published / name).read_bytes()) for name in PILOT_JSON}
    for root in [published, raw]:
        write(root, "artifact-sha256.json", manifest)
    paths = validated_pilot_paths(published, raw)
    assert (
        raw / "future_byte/final.mlm" in paths
        and published / "continuation-execution.json" in paths
    )
    missing = raw / "future_byte/head.msgpack"
    saved = missing.read_bytes()
    missing.unlink()
    with pytest.raises(ValueError):
        validated_pilot_paths(published, raw)
    missing.write_bytes(saved + b"bad")
    with pytest.raises(ValueError):
        validated_pilot_paths(published, raw)
    missing.write_bytes(saved)
    history = published / "control/history.json"
    history.write_text('[{"changed":true}]')
    with pytest.raises(ValueError):
        validated_pilot_paths(published, raw)
