import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from mlm.config import Config
from mlm.future_byte import create_head
from mlm.future_byte_run import serialized_values_equal


def test_auxiliary_roundtrip_compares_named_values_not_mapping_order():
    template = create_head(Config())
    trained = template.apply_gradients(grads=jax.tree.map(jnp.ones_like, template.params))
    data = serialization.to_bytes(trained)
    restored = serialization.from_bytes(template, data)
    raw = serialization.msgpack_restore(data)
    assert list(raw["params"]) != list(serialization.to_state_dict(restored)["params"])
    assert serialization.to_bytes(restored) != data
    assert serialized_values_equal(raw, serialization.to_state_dict(restored))
    changed = serialization.to_state_dict(restored)
    changed["params"]["bias"] = np.asarray(changed["params"]["bias"]).copy()
    changed["params"]["bias"][0] += 1
    assert not serialized_values_equal(raw, changed)
    assert not serialized_values_equal(raw, {**raw, "step": jnp.asarray(999, dtype=jnp.int32)})
