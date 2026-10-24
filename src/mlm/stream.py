"""Full-document lanes, truncated BPTT, and frozen full-document evaluation."""

import copy
import math

import jax
import jax.numpy as jnp
import numpy as np
import optax

from mlm.data import EOS, corpus_hash, encode, supervision_hash
from mlm.model import initial_state


class DocumentStream:
    def __init__(
        self, documents, batch_size, length, seed=42, repeat=True, shuffle=True, loss_masks=None
    ):
        if not documents or batch_size < 1 or length < 1:
            raise ValueError("Nonempty documents and positive dimensions required")
        self.tokens = [encode(doc) for doc in documents]
        self.identity = corpus_hash(documents)
        self.loss_masks = (
            None if loss_masks is None else [np.asarray(m, np.float32) for m in loss_masks]
        )
        if self.loss_masks is not None:
            if len(self.loss_masks) != len(self.tokens):
                raise ValueError("Loss mask/document count mismatch")
            for tokens, mask in zip(self.tokens, self.loss_masks, strict=True):
                if (
                    mask.shape != (len(tokens) - 1,)
                    or not np.isin(mask, [0, 1]).all()
                    or not mask.any()
                ):
                    raise ValueError("Invalid loss mask")
        self.loss_identity = supervision_hash(self.loss_masks)
        self.batch_size, self.length = batch_size, length
        self.repeat, self.shuffle = repeat, shuffle
        self.rng = np.random.default_rng(seed)
        self.order = list(range(len(documents)))
        if shuffle:
            self.rng.shuffle(self.order)
        self.position = 0
        self.epochs = 0
        self.lanes = [None] * batch_size

    def _assign(self):
        if self.position == len(self.order):
            if not self.repeat:
                return None
            self.epochs += 1
            self.position = 0
            if self.shuffle:
                self.rng.shuffle(self.order)
        document = self.order[self.position]
        self.position += 1
        return {"document": document, "offset": 0}

    def next_batch(self):
        shape = (self.batch_size, self.length)
        batch = {
            "inputs": np.full(shape, EOS, np.int32),
            "targets": np.full(shape, EOS, np.int32),
            "mask": np.zeros(shape, np.float32),
            "resets": np.zeros(shape, bool),
        }
        if self.loss_masks is not None:
            batch["loss_mask"] = np.zeros(shape, np.float32)
        for row in range(self.batch_size):
            if self.lanes[row] is None:
                self.lanes[row] = self._assign()
            lane = self.lanes[row]
            if lane is None:
                continue
            tokens = self.tokens[lane["document"]]
            offset = lane["offset"]
            count = min(self.length, len(tokens) - 1 - offset)
            batch["inputs"][row, :count] = tokens[offset : offset + count]
            batch["targets"][row, :count] = tokens[offset + 1 : offset + count + 1]
            batch["mask"][row, :count] = 1
            if self.loss_masks is not None:
                batch["loss_mask"][row, :count] = self.loss_masks[lane["document"]][
                    offset : offset + count
                ]
            batch["resets"][row, 0] = offset == 0
            lane["offset"] += count
            if lane["offset"] == len(tokens) - 1:
                self.lanes[row] = None
        return batch if batch["mask"].any() else None

    def snapshot(self):
        return copy.deepcopy(
            {
                "corpus": self.identity,
                "order": self.order,
                "position": self.position,
                "epochs": self.epochs,
                "lanes": self.lanes,
                "rng": self.rng.bit_generator.state,
                "batch_size": self.batch_size,
                "length": self.length,
                "repeat": self.repeat,
                "shuffle": self.shuffle,
                **({"loss_mask_hash": self.loss_identity} if self.loss_masks is not None else {}),
            }
        )

    def restore(self, snapshot):
        if snapshot.get("loss_mask_hash") != self.loss_identity:
            raise ValueError("Stream checkpoint loss mask mismatch")
        for key, expected in (
            ("corpus", self.identity),
            ("batch_size", self.batch_size),
            ("length", self.length),
            ("repeat", self.repeat),
            ("shuffle", self.shuffle),
        ):
            if snapshot[key] != expected:
                raise ValueError(f"Stream checkpoint mismatch: {key}")
        self.order = list(snapshot["order"])
        self.position, self.epochs = snapshot["position"], snapshot["epochs"]
        self.lanes = copy.deepcopy(snapshot["lanes"])
        self.rng.bit_generator.state = snapshot["rng"]


def stream_objective(params, apply_fn, batch, carry, messages=True):
    logits, final = apply_fn(
        {"params": params},
        batch["inputs"],
        carry,
        batch["resets"],
        messages=messages,
        valid=batch["mask"],
    )
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, batch["targets"])
    loss_mask = batch["mask"] * batch.get("loss_mask", batch["mask"])
    count = loss_mask.sum()
    total = (losses * loss_mask).sum()
    payload = loss_mask * (batch["targets"] < 256)
    metrics = {
        "loss": total / jnp.maximum(count, 1),
        "nll": total,
        "positions": count,
        "byte_nll": (losses * payload).sum(),
        "bytes": payload.sum(),
        "consumed_bytes": (batch["mask"] * (batch["targets"] < 256)).sum(),
    }
    return metrics["loss"], (final, metrics)


def make_stream_step(config, messages=True):
    @jax.jit
    def step(state, carry, batch):
        (_, (final, metrics)), gradients = jax.value_and_grad(stream_objective, has_aux=True)(
            state.params, state.apply_fn, batch, jax.lax.stop_gradient(carry), messages
        )
        candidate = state.apply_gradients(grads=gradients)
        leaves = jax.tree.leaves((candidate.params, candidate.opt_state, gradients, final, metrics))
        finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in leaves]))
        updated = finite & (metrics["positions"] > 0)
        new_state = jax.lax.cond(updated, lambda: candidate, lambda: state)
        new_carry = jax.lax.cond(finite, lambda: final, lambda: carry)
        return (
            new_state,
            new_carry,
            {
                **metrics,
                "finite": finite,
                "updated": updated,
                "grad_norm": optax.tree.norm(gradients),
            },
        )

    return step


def evaluate_stream(state, config, documents, messages=True, loss_masks=None):
    stream = DocumentStream(
        documents,
        config.batch_size,
        config.length,
        repeat=False,
        shuffle=False,
        loss_masks=loss_masks,
    )
    carry = initial_state(config, config.batch_size)
    compute = jax.jit(lambda params, c, b: stream_objective(params, state.apply_fn, b, c, messages))
    totals = {"nll": 0.0, "positions": 0.0, "byte_nll": 0.0, "bytes": 0.0, "consumed_bytes": 0.0}
    while (batch := stream.next_batch()) is not None:
        _, (carry, metrics) = compute(state.params, carry, jax.tree.map(jnp.asarray, batch))
        metrics = jax.device_get(metrics)
        for key in totals:
            totals[key] += float(metrics[key])
    if totals["bytes"] <= 0:
        raise ValueError("Evaluation requires payload bytes")
    loss = totals["nll"] / totals["positions"]
    return {
        "loss": loss,
        "bits_per_byte": totals["byte_nll"] / totals["bytes"] / math.log(2),
        "perplexity_with_markers": math.exp(loss),
        "positions": int(totals["positions"]),
        "payload_bytes": int(totals["bytes"]),
        "consumed_payload_bytes": int(totals["consumed_bytes"]),
        "loss_scope": "explicit target spans" if loss_masks is not None else "all targets",
        "documents": len(documents),
        "coverage": "complete documents",
        "chunk_length": config.length,
    }
