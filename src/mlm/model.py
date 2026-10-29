"""Causal graph-GRU scan with optional state-conditioned node readout; no growth."""

import math

import jax
import jax.numpy as jnp
from flax import linen as nn

from mlm.associative import delta_step
from mlm.config import Config
from mlm.data import VOCAB_SIZE
from mlm.graph import aggregate, sinkhorn_edge_weights, topology


def initial_state(config: Config, batch_size: int):
    nodes = 1 if config.architecture == "gru" else config.nodes
    if config.memory != "none":
        nodes += config.memory_keys
        if config.symmetric_memory:
            nodes += 4
    if config.episodic_slots > 0:
        nodes += config.episodic_slots
    return jnp.zeros((batch_size, nodes, config.width), dtype=jnp.float32)


class GraphLM(nn.Module):
    config: Config

    @nn.compact
    def __call__(
        self,
        tokens,
        state,
        resets,
        messages: bool = True,
        valid=None,
        return_routes: bool = False,
        return_ponder: bool = False,
        return_mtp: bool = False,
        next_tokens=None,
        router_bias=None,
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
        if c.memory != "none" and c.memory_decay:
            memory_decay_kernel = self.param(
                "memory_decay_kernel", nn.initializers.zeros, (2 * d, 1)
            )
            memory_decay_bias = self.param("memory_decay_bias", nn.initializers.constant(3.0), ())
        if c.memory != "none" and c.symmetric_memory:
            n_tr = 4
            memory_span_kernel = self.param(
                "memory_span_kernel", normal(1 / math.sqrt(2 * d)), (2 * d, n_tr)
            )
            tr_rates = jnp.exp(-jnp.geomspace(1 / 2.0, 1 / 24.0, n_tr))
            tr_log_odds = jnp.log((1.0 - tr_rates) / tr_rates)
            memory_span_bias = self.param("memory_span_bias", lambda *_: tr_log_odds, (n_tr,))
            memory_rev_key = self.param(
                "memory_rev_key_kernel",
                normal(1 / math.sqrt(n_tr * d)),
                (n_tr, d, c.memory_keys),
            )
            memory_rev_value = self.param(
                "memory_rev_value_kernel",
                normal(1 / math.sqrt(n_tr * d)),
                (n_tr, d, d),
            )
        if c.episodic_slots > 0:
            ep_compress = self.param(
                "episodic_compress_kernel", normal(1 / math.sqrt(2 * d)), (2 * d, d)
            )
            ep_write = self.param(
                "episodic_write_kernel",
                normal(1 / math.sqrt(2 * d)),
                (2 * d, c.episodic_slots),
            )
            ep_write_bias = self.param("episodic_write_bias", nn.initializers.constant(-2.0), ())
            ep_query = self.param("episodic_query_kernel", normal(1 / math.sqrt(2 * d)), (2 * d, d))
            ep_pos_bias = self.param(
                "episodic_pos_bias", nn.initializers.zeros, (c.episodic_slots,)
            )
            ep_output = self.param("episodic_output_kernel", normal(1 / math.sqrt(d)), (d, d))
            ep_gain = self.param("episodic_gain", nn.initializers.zeros, ())
        if c.mtp_weight > 0:
            mtp_kernel = self.param("mtp_kernel", normal(1 / math.sqrt(2 * d)), (2 * d, d))
        if c.adaptive_halt:
            halt_kernel = self.param("halt_kernel", nn.initializers.zeros, (n, d, 1))
            halt_bias = self.param("halt_bias", nn.initializers.constant(-2.0), (n,))
        edges = (
            jnp.asarray(c.edges, dtype=jnp.int32)
            if c.edges is not None
            else jnp.asarray(topology(n, c.degree))
        )

        if valid is None:
            valid = jnp.ones_like(tokens, dtype=bool)

        def apply_routing_gates(r_vec):
            s = c.shared_nodes
            b_sz = r_vec.shape[0]
            if 0 < c.top_k_nodes < n - s:
                spec = r_vec[:, s:]
                sel = (
                    spec + jax.lax.stop_gradient(router_bias[None, s:])
                    if router_bias is not None
                    else spec
                )
                _, top_idx = jax.lax.top_k(sel, c.top_k_nodes)
                k_mask = jnp.zeros_like(spec).at[jnp.arange(b_sz)[:, None], top_idx].set(1.0)
                hard_spec = spec * k_mask
                spec_out = spec + jax.lax.stop_gradient(hard_spec - spec)
                if s > 0:
                    return jnp.concatenate(
                        [jnp.ones((b_sz, s), dtype=r_vec.dtype), spec_out], axis=-1
                    )
                return spec_out
            out = r_vec
            if router_bias is not None:
                shift = jax.nn.sigmoid(jax.lax.stop_gradient(router_bias[None, :])) - 0.5
                out = jnp.clip(out + jax.lax.stop_gradient(shift), 0.0, 1.0)
            if s > 0:
                out = out.at[:, :s].set(1.0)
            return out

        def token_step(packed, item):
            token, reset, active = item
            previous = packed
            reset_packed = jnp.where(reset[:, None, None], 0.0, packed)
            h = reset_packed[:, :n, :]
            offset = n
            if c.memory != "none":
                memory = reset_packed[:, offset : offset + c.memory_keys, :]
                offset += c.memory_keys
                if c.symmetric_memory:
                    span_state = reset_packed[:, offset : offset + 4, :]
                    offset += 4
            if c.episodic_slots > 0:
                episodic = reset_packed[:, offset : offset + c.episodic_slots, :]
                offset += c.episodic_slots
            x = jnp.broadcast_to(embedding[token, None, :], h.shape)
            if c.node_update == "hierarchical":
                context = jnp.concatenate([embedding[token], h.mean(axis=1)], axis=-1)
                parent = jax.nn.softmax(context @ group_kernel + group_bias, axis=-1)
                child = jax.nn.softmax(
                    jnp.einsum("bi,gij->bgj", context, child_kernel) + child_bias, axis=-1
                )
                probabilities = (parent[:, :, None] * child).reshape((-1, n))
                det_ctx = jax.lax.stop_gradient(context)
                det_parent = jax.nn.softmax(det_ctx @ group_kernel + group_bias, axis=-1)
                det_child = jax.nn.softmax(
                    jnp.einsum("bi,gij->bgj", det_ctx, child_kernel) + child_bias, axis=-1
                )
                route_probs = (det_parent[:, :, None] * det_child).reshape((-1, n))
            elif c.node_update == "timescale":
                context = jnp.concatenate([embedding[token], h.mean(axis=1)], axis=-1)
                base_timescale = math.sqrt(d) * (context @ timescale_kernel) + timescale_bias
                raw_rates = jax.nn.sigmoid(base_timescale)
                probabilities = raw_rates / jnp.maximum(raw_rates.sum(axis=-1, keepdims=True), 1e-6)
                det_ctx = jax.lax.stop_gradient(context)
                det_base = math.sqrt(d) * (det_ctx @ timescale_kernel) + timescale_bias
                det_raw = jax.nn.sigmoid(det_base)
                route_probs = det_raw / jnp.maximum(det_raw.sum(axis=-1, keepdims=True), 1e-6)
            else:
                probabilities = jnp.full((tokens.shape[0], n), 1.0 / n)
                route_probs = probabilities
            if c.shared_nodes > 0 and c.node_update in {"hierarchical", "timescale"}:
                s = c.shared_nodes
                spec_p = route_probs[:, s:]
                spec_norm = spec_p / jnp.maximum(spec_p.sum(axis=-1, keepdims=True), 1e-6)
                shared_p = jnp.full((tokens.shape[0], s), 1.0 / n, dtype=route_probs.dtype)
                route_probs = jnp.concatenate([shared_p, spec_norm * ((n - s) / n)], axis=-1)
            rates = jnp.ones_like(probabilities) if c.node_update == "dense" else probabilities
            if c.node_update == "hierarchical":
                rates = apply_routing_gates(rates)

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
                    if c.sinkhorn_edges:
                        eweights = sinkhorn_edge_weights(eweights, edges, n)
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
                    step_rates = apply_routing_gates(jax.nn.sigmoid(base_timescale + surprise))
                    return old + step_rates[:, :, None] * (updated - old)
                return old + rates[:, :, None] * (updated - old)

            if c.adaptive_halt:

                def halt_round(r, carry):
                    curr_h, h_acc, rem, kl_acc = carry
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
                    q_r = jnp.where(
                        is_last,
                        (1.0 - c.halt_prior) ** r,
                        c.halt_prior * ((1.0 - c.halt_prior) ** r),
                    )
                    kl_step = p * (jnp.log(jnp.maximum(p, 1e-9)) - jnp.log(jnp.maximum(q_r, 1e-9)))
                    frozen_h = curr_h + rem[:, :, None] * (next_h - curr_h)
                    next_acc = h_acc + p[:, :, None] * frozen_h
                    return frozen_h, next_acc, next_rem, kl_acc + kl_step

                init_rem = jnp.ones((tokens.shape[0], n), dtype=h.dtype)
                init_kl = jnp.zeros((tokens.shape[0], n), dtype=h.dtype)
                _, h_acc, rem, kl_acc = jax.lax.fori_loop(
                    0, c.rounds, halt_round, (h, jnp.zeros_like(h), init_rem, init_kl)
                )
                if c.rounds == 1:
                    kl_acc = kl_acc + rem * (
                        jnp.log(jnp.maximum(rem, 1e-9))
                        - jnp.log(jnp.maximum(1.0 - c.halt_prior, 1e-9))
                    )
                h = h_acc + rem[:, :, None] * h
                ponder_kl = jnp.where(active, kl_acc.mean(axis=-1), 0.0)
            else:
                h = jax.lax.fori_loop(0, c.rounds, graph_round, h)
                ponder_kl = jnp.zeros((tokens.shape[0],), dtype=h.dtype)
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
            z_base = z
            if c.memory != "none":
                if c.memory == "delta":
                    features = jnp.concatenate([embedding[token], jnp.tanh(z_base)], axis=-1)
                    key = jnp.tanh(features @ memory_key)
                    query = jnp.tanh(features @ memory_query)
                    value = jnp.tanh(features @ memory_value)
                    beta = jax.nn.sigmoid(features @ memory_write + memory_bias)[:, 0]
                else:
                    key = jnp.tanh(jnp.einsum("bnd,ndk->bk", h, memory_key))
                    query = jnp.tanh(jnp.einsum("bnd,ndk->bk", h, memory_query))
                    value = jnp.tanh(jnp.einsum("bnd,ndv->bv", h, memory_value))
                    gate_features = jnp.concatenate([h.mean(axis=1), jnp.tanh(z_base)], axis=-1)
                    beta = jax.nn.sigmoid(
                        math.sqrt(d) * (gate_features @ memory_write) + memory_bias
                    )[:, 0]
                key = key * jax.lax.rsqrt(jnp.square(key).sum(-1, keepdims=True) + 1e-6)
                query = query * jax.lax.rsqrt(jnp.square(query).sum(-1, keepdims=True) + 1e-6)
                if c.memory_decay:
                    d_feat = features if c.memory == "delta" else gate_features
                    d_gate = jax.lax.stop_gradient(
                        jnp.where(
                            jnp.all(memory_decay_kernel == 0),
                            jnp.ones_like(memory_decay_kernel),
                            0.0,
                        )
                    )
                    d_tan = d_feat @ d_gate
                    d_diff = d_tan - jax.lax.stop_gradient(d_tan)
                    decay_gate = jax.nn.sigmoid(
                        math.sqrt(d) * (d_feat @ memory_decay_kernel + d_diff) + memory_decay_bias
                    )[:, 0]
                    alpha = 1.0 - 0.1 * beta * (1.0 - decay_gate)
                else:
                    alpha = None
                if c.symmetric_memory:
                    s_feat = features if c.memory == "delta" else gate_features
                    span_gate = jax.nn.sigmoid(
                        math.sqrt(d) * (s_feat @ memory_span_kernel) + memory_span_bias
                    )
                    next_span = (1.0 - span_gate[:, :, None]) * span_state + span_gate[
                        :, :, None
                    ] * jnp.tanh(z_base)[:, None, :]
                    prev_span_key = jnp.tanh(jnp.einsum("bhd,hdk->bk", span_state, memory_rev_key))
                    key = key + 0.5 * prev_span_key
                    key = key * jax.lax.rsqrt(jnp.square(key).sum(-1, keepdims=True) + 1e-6)
                    rev_key = jnp.tanh(jnp.einsum("bhd,hdk->bk", next_span, memory_rev_key))
                    rev_key = rev_key * jax.lax.rsqrt(
                        jnp.square(rev_key).sum(-1, keepdims=True) + 1e-6
                    )
                    rev_value = jnp.tanh(jnp.einsum("bhd,hdv->bv", span_state, memory_rev_value))
                    query = query + 0.5 * rev_key
                    query = query * jax.lax.rsqrt(jnp.square(query).sum(-1, keepdims=True) + 1e-6)
                    span_state = next_span
                else:
                    rev_key = None
                    rev_value = None
                memory, recalled = delta_step(
                    memory,
                    key,
                    value,
                    query,
                    beta,
                    alpha=alpha,
                    rev_key=rev_key,
                    rev_value=rev_value,
                )
                mem_scale = math.sqrt(d) if c.memory == "relational" else 1.0
                mem_out = (recalled @ memory_output) * mem_scale
                mem_floor = jnp.where(
                    jnp.abs(memory_gain) < 0.2,
                    jnp.where(memory_gain >= 0, 0.2, -0.2) - memory_gain,
                    0.0,
                )
                mem_gate = jax.lax.stop_gradient(jnp.where(memory_gain == 0, 1.0, mem_floor))
                mem_tangent = mem_gate * mem_out
                mem_diff = mem_tangent - jax.lax.stop_gradient(mem_tangent)
                z = z + (memory_gain * mem_out + mem_diff)
            if c.episodic_slots > 0:
                ep_feat = jnp.concatenate([h.mean(axis=1), jnp.tanh(z_base)], axis=-1)
                q_ep = ep_feat @ ep_query
                if c.memory != "none" and c.symmetric_memory:
                    q_ep = q_ep + span_state.mean(axis=1)
                ep_scores = (
                    jnp.einsum("bd,bsd->bs", q_ep, episodic) / math.sqrt(d) + ep_pos_bias[None, :]
                )
                ep_attn = jax.nn.softmax(ep_scores, axis=-1)
                ep_recalled = jnp.einsum("bs,bsd->bd", ep_attn, episodic)
                ep_out = (ep_recalled @ ep_output) * math.sqrt(d)
                ep_floor = jnp.where(
                    jnp.abs(ep_gain) < 0.2,
                    jnp.where(ep_gain >= 0, 0.2, -0.2) - ep_gain,
                    0.0,
                )
                ep_gate = jax.lax.stop_gradient(jnp.where(ep_gain == 0, 1.0, ep_floor))
                ep_tangent = ep_gate * ep_out
                ep_diff = ep_tangent - jax.lax.stop_gradient(ep_tangent)
                z = z + (ep_gain * ep_out + ep_diff)
                c_lat = jnp.tanh(ep_feat @ ep_compress)
                w_logits = math.sqrt(d) * (ep_feat @ ep_write) + ep_write_bias
                w_ep = jax.nn.sigmoid(w_logits) * jax.nn.softmax(w_logits, axis=-1)
                episodic = (1.0 - w_ep[:, :, None]) * episodic + w_ep[:, :, None] * c_lat[
                    :, None, :
                ]
            mean = z.mean(axis=-1, keepdims=True)
            variance = jnp.square(z - mean).mean(axis=-1, keepdims=True)
            z = (z - mean) * jax.lax.rsqrt(variance + 1e-5)
            logits = (z * scale + norm_bias) @ vocab_kernel + vocab_bias
            parts = [h]
            if c.memory != "none":
                parts.append(memory)
                if c.symmetric_memory:
                    parts.append(span_state)
            if c.episodic_slots > 0:
                parts.append(episodic)
            packed = parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=1)
            h = jnp.where(active[:, None, None], packed, previous)
            routes = jnp.where(active[:, None], route_probs, 0.0)
            return h, (logits, routes, ponder_kl, z)

        final, (logits_t, routes_t, ponder_t, z_t) = jax.lax.scan(
            token_step, state, (tokens.T, resets.T, valid.T)
        )
        logits = logits_t.transpose(1, 0, 2)
        out_items = [logits, final]
        if return_routes:
            out_items.append(routes_t.transpose(1, 0, 2))
        if return_ponder:
            out_items.append(ponder_t.T)
        if return_mtp:
            z_bt = z_t.transpose(1, 0, 2)
            next_tok = tokens if next_tokens is None else next_tokens
            mtp_inp = jnp.concatenate([z_bt, embedding[next_tok]], axis=-1)
            mtp_hidden = jnp.tanh(mtp_inp @ mtp_kernel)
            mtp_logits = (mtp_hidden * scale + norm_bias) @ vocab_kernel + vocab_bias
            out_items.append(mtp_logits)
        return tuple(out_items)


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
