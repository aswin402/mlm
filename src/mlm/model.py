"""Causal graph-GRU scan with optional state-conditioned node readout; no growth."""

import math

import jax
import jax.numpy as jnp
from flax import linen as nn

from mlm.associative import delta_step
from mlm.config import Config
from mlm.data import VOCAB_SIZE
from mlm.graph import aggregate, topology


def initial_state(config: Config, batch_size: int):
    nodes = 1 if config.architecture == "gru" else config.nodes
    if config.memory != "none":
        nodes += config.memory_keys
    return jnp.zeros((batch_size, nodes, config.width), dtype=jnp.float32)


class GraphLM(nn.Module):
    config: Config

    @nn.compact
    def __call__(
        self, tokens, state, resets, messages: bool = True, valid=None, return_routes=False
    ):
        c = self.config
        d, n = c.width, c.nodes
        normal = nn.initializers.normal
        embedding = self.param("embedding", normal(0.1), (VOCAB_SIZE, d))
        msg_kernel = self.param("message_kernel", normal(1 / math.sqrt(d)), (d, d))
        x_kernel = self.param("input_kernel", normal(1 / math.sqrt(2 * d)), (n, 2 * d, 3 * d))
        h_kernel = self.param("hidden_kernel", normal(1 / math.sqrt(d)), (n, d, 3 * d))
        bias = self.param("gru_bias", nn.initializers.zeros, (n, 3 * d))
        read_kernel = self.param("read_kernel", normal(1 / math.sqrt(d)), (n, d, d))
        scale = self.param("norm_scale", nn.initializers.ones, (d,))
        norm_bias = self.param("norm_bias", nn.initializers.zeros, (d,))
        vocab_kernel = self.param("vocab_kernel", normal(1 / math.sqrt(d)), (d, VOCAB_SIZE))
        vocab_bias = self.param("vocab_bias", nn.initializers.zeros, (VOCAB_SIZE,))
        if c.edge_aggregation == "attention":
            edge_query = self.param("edge_query", normal(1 / math.sqrt(d)), (d, d))
            edge_key = self.param("edge_key", normal(1 / math.sqrt(d)), (d, d))
            edge_vector = self.param("edge_vector", nn.initializers.zeros, (d,))
        if c.node_readout == "attention":
            attention_query = self.param("attention_query", normal(1 / math.sqrt(d)), (d, d))
            attention_key = self.param("attention_key", normal(1 / math.sqrt(d)), (d, d))
            attention_vector = self.param("attention_vector", nn.initializers.zeros, (d,))
        if c.node_update == "hierarchical":
            groups, children = c.groups, n // c.groups
            group_kernel = self.param("router_group_kernel", nn.initializers.zeros, (2 * d, groups))
            group_bias = self.param("router_group_bias", nn.initializers.zeros, (groups,))
            child_kernel = self.param(
                "router_child_kernel", nn.initializers.zeros, (groups, 2 * d, children)
            )
            child_bias = self.param("router_child_bias", nn.initializers.zeros, (groups, children))
        elif c.node_update == "timescale":
            timescale_kernel = self.param(
                "router_timescale_kernel", normal(1 / math.sqrt(2 * d)), (2 * d, n)
            )
            surprise_kernel = self.param("router_surprise_kernel", nn.initializers.zeros, (n, d, 1))
            timescale_bias = self.param(
                "router_timescale_bias",
                lambda key, shape, dtype=jnp.float32: jnp.linspace(
                    2.0, -2.0, shape[0], dtype=dtype
                ),
                (n,),
            )
        if c.memory == "delta":
            feature_width = 2 * d
            memory_key = self.param(
                "memory_key_kernel",
                normal(1 / math.sqrt(feature_width)),
                (feature_width, c.memory_keys),
            )
            memory_query = self.param(
                "memory_query_kernel",
                normal(1 / math.sqrt(feature_width)),
                (feature_width, c.memory_keys),
            )
            memory_value = self.param(
                "memory_value_kernel", normal(1 / math.sqrt(feature_width)), (feature_width, d)
            )
            memory_write = self.param(
                "memory_write_kernel", normal(1 / math.sqrt(feature_width)), (feature_width, 1)
            )
            memory_bias = self.param("memory_write_bias", nn.initializers.constant(-2.0), ())
            memory_output = self.param("memory_output_kernel", normal(1 / math.sqrt(d)), (d, d))
            memory_gain = self.param("memory_gain", nn.initializers.zeros, ())
        elif c.memory == "relational":
            scale_nd = 1 / math.sqrt(n * d)
            memory_key = self.param("memory_key_kernel", normal(scale_nd), (n, d, c.memory_keys))
            memory_query = self.param(
                "memory_query_kernel", normal(scale_nd), (n, d, c.memory_keys)
            )
            memory_value = self.param("memory_value_kernel", normal(scale_nd), (n, d, d))
            memory_write = self.param(
                "memory_write_kernel", normal(1 / math.sqrt(2 * d)), (2 * d, 1)
            )
            memory_bias = self.param("memory_write_bias", nn.initializers.constant(-2.0), ())
            memory_output = self.param("memory_output_kernel", normal(1 / math.sqrt(d)), (d, d))
            memory_gain = self.param("memory_gain", nn.initializers.zeros, ())
        if c.adaptive_halt:
            halt_kernel = self.param("halt_kernel", nn.initializers.zeros, (n, d, 1))
            halt_bias = self.param("halt_bias", nn.initializers.constant(-2.0), (n,))
        edges = jnp.asarray(topology(n, c.degree))

        if valid is None:
            valid = jnp.ones_like(tokens, dtype=bool)

        def token_step(packed, item):
            token, reset, active = item
            previous = packed
            h = packed[:, :n, :] if c.memory != "none" else packed
            if c.memory != "none":
                memory = jnp.where(reset[:, None, None], 0.0, packed[:, n:, :])
            h = jnp.where(reset[:, None, None], 0.0, h)
            x = jnp.broadcast_to(embedding[token, None, :], h.shape)
            if c.node_update == "hierarchical":
                context = jnp.concatenate([embedding[token], h.mean(axis=1)], axis=-1)
                parent = jax.nn.softmax(context @ group_kernel + group_bias, axis=-1)
                child = jax.nn.softmax(
                    jnp.einsum("bi,gij->bgj", context, child_kernel) + child_bias, axis=-1
                )
                probabilities = (parent[:, :, None] * child).reshape((-1, n))
            elif c.node_update == "timescale":
                context = jnp.concatenate([embedding[token], h.mean(axis=1)], axis=-1)
                base_timescale = math.sqrt(d) * (context @ timescale_kernel) + timescale_bias
                raw_rates = jax.nn.sigmoid(base_timescale)
                probabilities = raw_rates / jnp.maximum(raw_rates.sum(axis=-1, keepdims=True), 1e-6)
            else:
                probabilities = jnp.full((tokens.shape[0], n), 1.0 / n)
            rates = jnp.ones_like(probabilities) if c.node_update == "dense" else probabilities

            def graph_round(_, old):
                if not messages:
                    msg = jnp.zeros_like(old)
                elif c.edge_aggregation == "attention":
                    eq = (old @ edge_query)[:, :, None, :]
                    ek = (old @ edge_key)[:, edges, :]
                    efeatures = jnp.tanh(ek + eq)
                    egate = jax.lax.stop_gradient(
                        jnp.where(jnp.all(edge_vector == 0), jnp.ones_like(edge_vector), 0.0)
                    )
                    etangent = efeatures @ egate
                    ediff = etangent - jax.lax.stop_gradient(etangent)
                    eenergy = efeatures @ edge_vector + ediff
                    eweights = jax.nn.softmax(eenergy, axis=2)
                    msg = aggregate(old @ msg_kernel, edges, weights=eweights)
                else:
                    msg = aggregate(old @ msg_kernel, edges)
                inp = jnp.concatenate([x, msg], axis=-1)
                gx = jnp.einsum("bni,nij->bnj", inp, x_kernel) + bias
                gh = jnp.einsum("bni,nij->bnj", old, h_kernel)
                xz, xr, xn = jnp.split(gx, 3, axis=-1)
                hz, hr, hn = jnp.split(gh, 3, axis=-1)
                z = jax.nn.sigmoid(xz + hz)
                r = jax.nn.sigmoid(xr + hr)
                candidate = jnp.tanh(xn + r * hn)
                updated = (1 - z) * candidate + z * old
                if c.node_update == "dense":
                    return updated
                if c.node_update == "timescale":
                    s_features = jnp.abs(updated - old)
                    s_gate = jax.lax.stop_gradient(
                        jnp.where(
                            jnp.all(surprise_kernel == 0),
                            jnp.ones_like(surprise_kernel),
                            0.0,
                        )
                    )
                    s_tangent = jnp.einsum("bnd,ndc->bn", s_features, s_gate)
                    s_diff = s_tangent - jax.lax.stop_gradient(s_tangent)
                    surprise = math.sqrt(d) * (
                        jnp.einsum("bnd,ndc->bn", s_features, surprise_kernel) + s_diff
                    )
                    step_rates = jax.nn.sigmoid(base_timescale + surprise)
                    return old + step_rates[:, :, None] * (updated - old)
                return old + rates[:, :, None] * (updated - old)

            if c.adaptive_halt:

                def halt_round(r, carry):
                    curr_h, h_acc, rem = carry
                    next_h = graph_round(r, curr_h)
                    h_gate = jax.lax.stop_gradient(
                        jnp.where(jnp.all(halt_kernel == 0), jnp.ones_like(halt_kernel), 0.0)
                    )
                    h_tangent = jnp.einsum("bnd,ndc->bn", next_h, h_gate)
                    h_diff = h_tangent - jax.lax.stop_gradient(h_tangent)
                    halt_logits = (
                        math.sqrt(d) * (jnp.einsum("bnd,ndc->bn", next_h, halt_kernel) + h_diff)
                        + halt_bias
                    )
                    lam = jax.nn.sigmoid(halt_logits)
                    is_last = (r == c.rounds - 1) & (c.rounds > 1)
                    p = jnp.where(is_last, rem, rem * lam)
                    next_rem = jnp.where(is_last, jnp.zeros_like(rem), rem * (1.0 - lam))
                    frozen_h = curr_h + rem[:, :, None] * (next_h - curr_h)
                    next_acc = h_acc + p[:, :, None] * frozen_h
                    return frozen_h, next_acc, next_rem

                init_rem = jnp.ones((tokens.shape[0], n), dtype=h.dtype)
                _, h_acc, rem = jax.lax.fori_loop(
                    0, c.rounds, halt_round, (h, jnp.zeros_like(h), init_rem)
                )
                h = h_acc + rem[:, :, None] * h
            else:
                h = jax.lax.fori_loop(0, c.rounds, graph_round, h)
            if c.node_readout == "sum":
                z = jnp.einsum("bni,nij->bj", h, read_kernel) / math.sqrt(n)
            else:
                query = (h.mean(axis=1) @ attention_query)[:, None, :]
                features = jnp.tanh(h @ attention_key + query)
                vec_gate = jax.lax.stop_gradient(
                    jnp.where(jnp.all(attention_vector == 0), jnp.ones_like(attention_vector), 0.0)
                )
                vec_tangent = features @ vec_gate
                vec_diff = vec_tangent - jax.lax.stop_gradient(vec_tangent)
                energy = features @ attention_vector + vec_diff
                weights = jax.nn.softmax(energy, axis=1)
                values = jnp.einsum("bni,nij->bnj", h, read_kernel)
                z = jnp.einsum("bn,bnd->bd", weights, values) * math.sqrt(n)
            if c.memory != "none":
                if c.memory == "delta":
                    features = jnp.concatenate([embedding[token], jnp.tanh(z)], axis=-1)
                    key = jnp.tanh(features @ memory_key)
                    query = jnp.tanh(features @ memory_query)
                    value = jnp.tanh(features @ memory_value)
                    beta = jax.nn.sigmoid(features @ memory_write + memory_bias)[:, 0]
                else:
                    key = jnp.tanh(jnp.einsum("bnd,ndk->bk", h, memory_key))
                    query = jnp.tanh(jnp.einsum("bnd,ndk->bk", h, memory_query))
                    value = jnp.tanh(jnp.einsum("bnd,ndv->bv", h, memory_value))
                    gate_features = jnp.concatenate([h.mean(axis=1), jnp.tanh(z)], axis=-1)
                    beta = jax.nn.sigmoid(
                        math.sqrt(d) * (gate_features @ memory_write) + memory_bias
                    )[:, 0]
                key = key * jax.lax.rsqrt(jnp.square(key).sum(-1, keepdims=True) + 1e-6)
                query = query * jax.lax.rsqrt(jnp.square(query).sum(-1, keepdims=True) + 1e-6)
                memory, recalled = delta_step(memory, key, value, query, beta)
                mem_scale = math.sqrt(d) if c.memory == "relational" else 1.0
                mem_out = (recalled @ memory_output) * mem_scale
                mem_gate = jax.lax.stop_gradient(jnp.where(memory_gain == 0, 1.0, 0.0))
                mem_tangent = mem_gate * mem_out
                mem_diff = mem_tangent - jax.lax.stop_gradient(mem_tangent)
                z = z + (memory_gain * mem_out + mem_diff)
            mean = z.mean(axis=-1, keepdims=True)
            variance = jnp.square(z - mean).mean(axis=-1, keepdims=True)
            z = (z - mean) * jax.lax.rsqrt(variance + 1e-5)
            logits = (z * scale + norm_bias) @ vocab_kernel + vocab_bias
            packed = jnp.concatenate([h, memory], axis=1) if c.memory != "none" else h
            h = jnp.where(active[:, None, None], packed, previous)
            if return_routes:
                routes = jnp.where(active[:, None], probabilities, 0.0)
                return h, (logits, routes)
            return h, logits

        final, output = jax.lax.scan(token_step, state, (tokens.T, resets.T, valid.T))
        if return_routes:
            logits, routes = output
            return logits.transpose(1, 0, 2), final, routes.transpose(1, 0, 2)
        return output.transpose(1, 0, 2), final


class PlainGRU(nn.Module):
    """Single conventional GRU; no graph, node pooling, or message parameters."""

    config: Config

    @nn.compact
    def __call__(self, tokens, state, resets, messages: bool = True, valid=None):
        d = self.config.width
        normal = nn.initializers.normal
        embedding = self.param("embedding", normal(0.1), (VOCAB_SIZE, d))
        x_kernel = self.param("input_kernel", normal(1 / math.sqrt(d)), (d, 3 * d))
        h_kernel = self.param("hidden_kernel", normal(1 / math.sqrt(d)), (d, 3 * d))
        bias = self.param("gru_bias", nn.initializers.zeros, (3 * d,))
        scale = self.param("norm_scale", nn.initializers.ones, (d,))
        norm_bias = self.param("norm_bias", nn.initializers.zeros, (d,))
        vocab = self.param("vocab_kernel", normal(1 / math.sqrt(d)), (d, VOCAB_SIZE))
        vocab_bias = self.param("vocab_bias", nn.initializers.zeros, (VOCAB_SIZE,))
        if valid is None:
            valid = jnp.ones_like(tokens, dtype=bool)

        def token_step(h, item):
            token, reset, active = item
            previous = h
            h = jnp.where(reset[:, None], 0.0, h)
            gx = embedding[token] @ x_kernel + bias
            gh = h @ h_kernel
            xz, xr, xn = jnp.split(gx, 3, axis=-1)
            hz, hr, hn = jnp.split(gh, 3, axis=-1)
            z, r = jax.nn.sigmoid(xz + hz), jax.nn.sigmoid(xr + hr)
            h = (1 - z) * jnp.tanh(xn + r * hn) + z * h
            mean = h.mean(axis=-1, keepdims=True)
            variance = jnp.square(h - mean).mean(axis=-1, keepdims=True)
            readout = (h - mean) * jax.lax.rsqrt(variance + 1e-5)
            logits = (readout * scale + norm_bias) @ vocab + vocab_bias
            h = jnp.where(active[:, None], h, previous)
            return h, logits

        final, logits = jax.lax.scan(token_step, state[:, 0, :], (tokens.T, resets.T, valid.T))
        return logits.transpose(1, 0, 2), final[:, None, :]


def make_model(config):
    return PlainGRU(config) if config.architecture == "gru" else GraphLM(config)
