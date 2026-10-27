import numpy as np
import pytest

from mlm.context_contrastive_run import gates
from mlm.context_replication_run import check_integrity, expected_exposure, validate_exposure
from mlm.interference_run import digest


def test_independent_exposure_accounts_mask_eos_and_auxiliary():
    qa = {
        "mask": np.array([[1.0, 1.0, 1.0, 0.0]]),
        "loss_mask": np.array([[0.0, 1.0, 1.0, 1.0]]),
        "targets": np.array([[10, 20, 257, 30]]),
    }
    english = {"mask": np.array([[1.0, 1.0, 0.0]]), "targets": np.array([[40, 257, 50]])}
    candidates = {"answer_mask": np.ones((8, 4)), "gold": np.array([0, 1, 1, 0])}
    counts = expected_exposure([qa], [candidates], [0, 0], [english, english], 100)
    assert counts == dict(
        qa_updates=2,
        english_updates=2,
        qa_targets=4,
        english_targets=4,
        qa_bytes=2,
        english_bytes=2,
        candidate_positions=64,
        context_terms=8,
    )
    control = expected_exposure([qa], [candidates], [0, 0], [english, english], 0)
    assert control["candidate_positions"] == control["context_terms"] == 0
    assert all(
        control[k] == v
        for k, v in counts.items()
        if k not in ["candidate_positions", "context_terms"]
    )


def test_exposure_guard_rejects_count_and_visit_mismatches():
    expected = dict(
        qa_updates=128, english_updates=128, candidate_positions=4096, context_terms=512
    )
    visits = dict.fromkeys(["a", "b", "c", "d"], 32)
    validate_exposure(expected, expected, visits)
    for key in expected:
        bad = dict(expected)
        bad[key] -= 1
        with pytest.raises(ValueError):
            validate_exposure(bad, expected, visits)
    with pytest.raises(ValueError):
        validate_exposure(expected, expected, {**visits, "d": 31})
    with pytest.raises(ValueError):
        validate_exposure(expected, expected, {})


def test_protocol_input_and_settings_tampering_are_rejected(tmp_path, monkeypatch):
    import mlm.context_replication_run as runner

    protocol = tmp_path / "protocol.json"
    source = tmp_path / "source"
    protocol.write_text("{}")
    source.write_text("original")
    settings = {"numerical": "fixed"}
    monkeypatch.setattr(runner, "execution_settings", lambda: settings.copy())
    protocol_hash = digest(protocol)
    frozen = {str(source): digest(source)}
    check_integrity(protocol, protocol_hash, settings, frozen)
    protocol.write_text('{"changed":true}')
    with pytest.raises(ValueError):
        check_integrity(protocol, protocol_hash, settings, frozen)
    protocol.write_text("{}")
    source.write_text("changed")
    with pytest.raises(ValueError):
        check_integrity(protocol, protocol_hash, settings, frozen)
    source.write_text("original")
    monkeypatch.setattr(runner, "execution_settings", lambda: {"numerical": "changed"})
    with pytest.raises(ValueError):
        check_integrity(protocol, protocol_hash, settings, frozen)


def test_replication_gate_requires_control_comparison_and_english():
    control = dict(correct=8, groups_correct=0, english_ce=1.0)
    active = dict(correct=16, groups_correct=4, english_ce=1.04)
    assert gates(active, control) == dict(primary=True, english=False, joint=False)
    assert not gates(active, active)["primary"]


def test_prior_raw_history_and_summary_are_required_and_immutable(tmp_path):
    from mlm.context_replication_run import prior_raw_jsons

    raw, published = tmp_path / "raw", tmp_path / "published"
    raw.mkdir()
    published.mkdir()
    for name in [
        "protocol.json",
        "gradient.json",
        "initial.json",
        "summary.json",
        "improvement-protocol.json",
    ]:
        (raw / name).write_text("{}")
        (published / name).write_text("{}")
    for w in [0, 1, 100]:
        (raw / f"weight-{w}").mkdir()
        for name in ["summary", "history"]:
            (raw / f"weight-{w}/{name}.json").write_text("{}")
            (published / f"weight-{w}-{name}.json").write_text("{}")
    assert len(prior_raw_jsons(raw, published)) == 22
    history = raw / "weight-100/history.json"
    history.write_text("[1]")
    with pytest.raises(ValueError):
        prior_raw_jsons(raw, published)
    history.write_text("{}")
    (raw / "weight-0/summary.json").unlink()
    with pytest.raises(ValueError):
        prior_raw_jsons(raw, published)
