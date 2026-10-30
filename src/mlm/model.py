"""Causal graph-GRU scan with optional state-conditioned node readout; no growth."""

import math

import jax
import jax.numpy as jnp
from flax import linen as nn

from mlm.associative import delta_step
from mlm.config import Config
from mlm.data import EOS, VOCAB_SIZE
from mlm.graph import aggregate, sinkhorn_edge_weights, sinkhorn_stream_mix, topology


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
        return_telemetry: bool = False,
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
                "memory_decay_kernel", nn.initializers.zeros, (2 * d, c.memory_keys)
            )
            memory_decay_bias = self.param(
                "memory_decay_bias",
                lambda key, shape, dtype=jnp.float32: jnp.linspace(2.0, 4.0, shape[0], dtype=dtype),
                (c.memory_keys,),
            )
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
        if c.memory != "none" and c.surprise_gating:
            surprise_mem_weight = self.param(
                "surprise_mem_weight", nn.initializers.constant(1.0), ()
            )
            surprise_span_weight = self.param(
                "surprise_span_weight", nn.initializers.constant(1.0), ()
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
        if c.relational_hops > 0:
            rel_wavelet = self.param("rel_wavelet_kernel", normal(1 / math.sqrt(4 * d)), (4, d, d))
            rel_liv_gate = self.param("rel_liv_gate_kernel", normal(1 / math.sqrt(d)), (d, 2 * d))
            rel_liv_conv = self.param("rel_liv_conv_kernel", normal(1 / math.sqrt(4 * d)), (4, d))
            rel_q = self.param("rel_q_kernel", normal(1 / math.sqrt(d)), (c.relational_hops, d, d))
            rel_k = self.param("rel_k_kernel", normal(1 / math.sqrt(d)), (c.relational_hops, d, d))
            rel_v = self.param("rel_v_kernel", normal(1 / math.sqrt(d)), (c.relational_hops, d, d))
            rel_output = self.param("rel_output_kernel", normal(1 / math.sqrt(d)), (d, d))
            rel_gain = self.param("rel_gain", nn.initializers.zeros, ())
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
        else:
            valid = valid.astype(bool)

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

        tr_alphas = 1.0 - jnp.exp(-jnp.geomspace(1 / 2.0, 1 / 24.0, 4))

        def dog_wavelets(tr):
            s_tr = tr / tr_alphas[None, :, None]
            w0 = tr[:, 0, :]
            w1 = tr_alphas[1] * (s_tr[:, 1, :] - s_tr[:, 0, :])
            w2 = tr_alphas[2] * (s_tr[:, 2, :] - s_tr[:, 1, :])
            w3 = tr_alphas[3] * (s_tr[:, 3, :] - s_tr[:, 2, :])
            w = jnp.stack([w0, w1, w2, w3], axis=1)
            return w * jax.lax.rsqrt(jnp.mean(jnp.square(w), axis=-1, keepdims=True) + 1e-5)

        def token_step(packed, item):
            token, reset, active = item
            previous = packed
            reset_packed = jnp.where(reset[:, None, None], 0.0, packed)
            h = reset_packed[:, :n, :]
            h_prev = h
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
                if (
                    n >= 8
                    and 0 < c.top_k_nodes < n - c.shared_nodes
                    and c.node_update in {"timescale", "hierarchical"}
                ):
                    s = c.shared_nodes
                    b_sz = tokens.shape[0]
                    b_idx = jnp.arange(b_sz)[:, None]
                    score_pool = (
                        base_timescale[:, s:] if c.node_update == "timescale" else rates[:, s:]
                    )
                    if router_bias is not None:
                        score_pool = score_pool + jax.lax.stop_gradient(router_bias[None, s:])
                    _, top_rel = jax.lax.top_k(score_pool, c.top_k_nodes)
                    spec_idx = top_rel + s
                    if s > 0:
                        shared_idx = jnp.broadcast_to(
                            jnp.arange(s, dtype=spec_idx.dtype)[None, :], (b_sz, s)
                        )
                        act_idx = jnp.concatenate([shared_idx, spec_idx], axis=-1)
                    else:
                        act_idx = spec_idx
                    inp_k = inp[b_idx, act_idx]
                    old_k = old[b_idx, act_idx]
                    xk_k = x_kernel[act_idx]
                    hk_k = h_kernel[act_idx]
                    b_k = bias[act_idx]
                    gx_k = jnp.einsum("bki,bkij->bkj", inp_k, xk_k) + b_k
                    gh_k = jnp.einsum("bki,bkij->bkj", old_k, hk_k)
                    xz, xr, xn = jnp.split(gx_k, 3, axis=-1)
                    hz, hr, hn = jnp.split(gh_k, 3, axis=-1)
                    z = jax.nn.sigmoid(xz + hz)
                    r = jax.nn.sigmoid(xr + hr)
                    candidate = jnp.tanh(xn + r * hn)
                    upd_k = (1 - z) * candidate + z * old_k
                    updated = old.at[b_idx, act_idx].set(upd_k)
                else:
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
            dh_norm = jnp.sqrt(jnp.sum(jnp.square(h - h_prev), axis=(1, 2)) + 1e-8)
            surprise_val = dh_norm / (math.sqrt(n * d) + dh_norm)
            wavelet_out = jnp.zeros((tokens.shape[0], 4, d), dtype=z.dtype)
            if c.memory != "none":
                surp_mem = (
                    surprise_mem_weight * (surprise_val[:, None] - 0.25)
                    if c.surprise_gating
                    else 0.0
                )
                if c.memory == "delta":
                    features = jnp.concatenate([embedding[token], jnp.tanh(z_base)], axis=-1)
                    key = jnp.tanh(features @ memory_key)
                    query = jnp.tanh(features @ memory_query)
                    value = jnp.tanh(features @ memory_value)
                    beta = jax.nn.sigmoid(features @ memory_write + memory_bias + surp_mem)[:, 0]
                else:
                    key = jnp.tanh(jnp.einsum("bnd,ndk->bk", h, memory_key))
                    query = jnp.tanh(jnp.einsum("bnd,ndk->bk", h, memory_query))
                    value = jnp.tanh(jnp.einsum("bnd,ndv->bv", h, memory_value))
                    gate_features = jnp.concatenate([h.mean(axis=1), jnp.tanh(z_base)], axis=-1)
                    beta = jax.nn.sigmoid(
                        math.sqrt(d) * (gate_features @ memory_write) + memory_bias + surp_mem
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
                    )
                    alpha = jnp.exp(-0.15 * beta[:, None] * (1.0 - decay_gate))
                    chan_gate = jax.nn.sigmoid(math.sqrt(d) * jnp.tanh(z_base) + h.mean(axis=1))
                    channel_alpha = jnp.exp(-0.08 * beta[:, None] * (1.0 - chan_gate))
                else:
                    alpha = None
                    channel_alpha = None
                if c.symmetric_memory:
                    s_feat = features if c.memory == "delta" else gate_features
                    surp_span = (
                        surprise_span_weight * (surprise_val[:, None] - 0.25)
                        if c.surprise_gating
                        else 0.0
                    )
                    span_gate = jax.nn.sigmoid(
                        math.sqrt(d) * (s_feat @ memory_span_kernel) + memory_span_bias + surp_span
                    )
                    span_in = embedding[token] + 0.5 * jnp.tanh(z_base)
                    next_span = (1.0 - span_gate[:, :, None]) * span_state + span_gate[
                        :, :, None
                    ] * span_in[:, None, :]
                    prev_w = dog_wavelets(span_state)
                    cur_w = dog_wavelets(next_span)
                    wavelet_out = cur_w
                    prev_span_key = jnp.tanh(jnp.einsum("bhd,hdk->bk", prev_w, memory_rev_key))
                    key = key + 0.5 * prev_span_key
                    key = key * jax.lax.rsqrt(jnp.square(key).sum(-1, keepdims=True) + 1e-6)
                    value = value + 0.15 * jnp.tanh(span_state[:, 0, :])
                    rev_key = jnp.tanh(jnp.einsum("bhd,hdk->bk", cur_w, memory_rev_key))
                    rev_key = rev_key * jax.lax.rsqrt(
                        jnp.square(rev_key).sum(-1, keepdims=True) + 1e-6
                    )
                    rev_value = jnp.tanh(jnp.einsum("bhd,hdv->bv", prev_w, memory_rev_value))
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
                    channel_alpha=channel_alpha,
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
                mem_contrib = memory_gain * mem_out + mem_diff
                z = z + mem_contrib
            else:
                mem_contrib = jnp.zeros_like(z)
            if c.episodic_slots > 0:
                ep_feat = jnp.concatenate([h.mean(axis=1), jnp.tanh(z_base)], axis=-1)
                q_ep = ep_feat @ ep_query
                if c.memory != "none" and c.symmetric_memory:
                    q_ep = q_ep + wavelet_out.mean(axis=1)
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
                ep_contrib = ep_gain * ep_out + ep_diff
                z = z + ep_contrib
                c_lat = jnp.tanh(ep_feat @ ep_compress)
                w_logits = math.sqrt(d) * (ep_feat @ ep_write) + ep_write_bias
                w_ep = jax.nn.sigmoid(w_logits) * jax.nn.softmax(w_logits, axis=-1)
                episodic = (1.0 - w_ep[:, :, None]) * episodic + w_ep[:, :, None] * c_lat[
                    :, None, :
                ]
            else:
                ep_contrib = jnp.zeros_like(z)
            if (
                c.sinkhorn_edges
                and c.memory != "none"
                and c.symmetric_memory
                and c.episodic_slots > 0
            ):
                streams_4 = jnp.stack(
                    [z_base, mem_contrib, 0.1 * wavelet_out.mean(axis=1), ep_contrib], axis=1
                )
                mixed_4, _ = sinkhorn_stream_mix(streams_4, iters=4)
                mhc_active = jax.lax.stop_gradient(
                    jnp.where((memory_gain == 0) & (ep_gain == 0), 0.0, 1.0)
                )
                z = z + 0.1 * mhc_active * (mixed_4.sum(axis=1) - streams_4.sum(axis=1))
            z_raw = z
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
            return h, (logits, routes, ponder_kl, z, z_raw, wavelet_out)

        scan_step = jax.checkpoint(token_step) if (n * d >= 512) else token_step
        final, (logits_t, routes_t, ponder_t, z_t, z_raw_t, w_t) = jax.lax.scan(
            scan_step, state, (tokens.T, resets.T, valid.T)
        )
        z_bt = z_t.transpose(1, 0, 2)
        if c.relational_hops > 0:
            z_raw_bt = z_raw_t.transpose(1, 0, 2)
            w_bt = w_t.transpose(1, 0, 2, 3)
            tok_emb = embedding[tokens]
            doc_ids = jnp.cumsum(resets.astype(jnp.int32), axis=1)
            t_len = tokens.shape[1]
            idx_t = jnp.arange(t_len)
            is_causal = idx_t[:, None] >= idx_t[None, :]
            same_doc = doc_ids[:, :, None] == doc_ids[:, None, :]
            pair_valid = same_doc & valid[:, :, None] & valid[:, None, :]
            rel_mask = is_causal[None, :, :] & pair_valid
            prior_mask = (idx_t[:, None] >= (idx_t[None, :] + 2))[None, :, :] & rel_mask
            prior_f = prior_mask.astype(z_raw_bt.dtype)
            rel_f = rel_mask.astype(z_raw_bt.dtype)
            pair_f = pair_valid.astype(z_raw_bt.dtype)
            w_proj = jnp.einsum("bthd,hdo->bto", w_bt, rel_wavelet)
            not_reset_f = (~resets & valid).astype(z_raw_bt.dtype)
            not_reset_1 = not_reset_f
            not_reset_2 = not_reset_1 * jnp.pad(not_reset_f, ((0, 0), (1, 0)))[:, :t_len]
            not_reset_3 = not_reset_2 * jnp.pad(not_reset_f, ((0, 0), (2, 0)))[:, :t_len]

            # LFM 2.5 LIV (Linear Input-Varying) double-gated short-range causal convolution
            liv_b, liv_c = jnp.split(tok_emb @ rel_liv_gate, 2, axis=-1)
            x_liv = jax.nn.sigmoid(liv_c) * tok_emb
            x_liv_1 = (
                jnp.pad(x_liv, ((0, 0), (1, 0), (0, 0)))[:, :t_len, :] * not_reset_1[:, :, None]
            )
            x_liv_2 = (
                jnp.pad(x_liv, ((0, 0), (2, 0), (0, 0)))[:, :t_len, :] * not_reset_2[:, :, None]
            )
            x_liv_3 = (
                jnp.pad(x_liv, ((0, 0), (3, 0), (0, 0)))[:, :t_len, :] * not_reset_3[:, :, None]
            )
            conv_liv = (
                x_liv * rel_liv_conv[0]
                + x_liv_1 * rel_liv_conv[1]
                + x_liv_2 * rel_liv_conv[2]
                + x_liv_3 * rel_liv_conv[3]
            )
            liv_stem = jax.nn.silu(liv_b) * conv_liv

            # Hop 0: Residual identity-anchored stem matching & diagonal 3-gram convolution
            vocab_centered = vocab_kernel - jnp.mean(vocab_kernel, axis=0, keepdims=True)
            vk_u = vocab_centered.T * jax.lax.rsqrt(
                jnp.sum(jnp.square(vocab_centered.T), axis=-1, keepdims=True) + 1e-6
            )
            v0_u = vk_u[tokens]
            tok_centered = tok_emb - jnp.mean(tok_emb, axis=-1, keepdims=True)
            tok_unit = tok_centered * jax.lax.rsqrt(
                jnp.sum(jnp.square(tok_centered), axis=-1, keepdims=True) + 1e-6
            )
            qk_sym = 1e-4 * jnp.tanh(tok_unit @ (rel_q[0] + rel_k[0]))
            q0_raw = (
                v0_u
                + qk_sym
                + 1e-4 * jnp.tanh(tok_unit @ rel_q[0])
                + 1e-4 * jnp.tanh(w_proj)
                + 1e-4 * jnp.tanh(liv_stem)
            )
            k0_raw = (
                v0_u
                + qk_sym
                + 1e-4 * jnp.tanh(tok_unit @ rel_k[0])
                + 1e-4 * jnp.tanh(w_proj)
                + 1e-4 * jnp.tanh(liv_stem)
            )
            q0 = q0_raw * jax.lax.rsqrt(jnp.sum(jnp.square(q0_raw), axis=-1, keepdims=True) + 1e-6)
            k0 = k0_raw * jax.lax.rsqrt(jnp.sum(jnp.square(k0_raw), axis=-1, keepdims=True) + 1e-6)
            c1 = (
                0.5
                * (jnp.einsum("btd,bsd->bts", q0, k0) + jnp.einsum("btd,bsd->bts", v0_u, v0_u))
                * pair_f
            )
            c1_p1 = jnp.pad(c1, ((0, 0), (1, 0), (1, 0)))[:, :t_len, :t_len] * pair_f
            c1_p2 = jnp.pad(c1, ((0, 0), (2, 0), (2, 0)))[:, :t_len, :t_len] * pair_f
            c3 = (c1 + c1_p1 + c1_p2) / 3.0

            # Causal exact context & question span boundary detection (plain & chat)
            p1_idx = idx_t.astype(z_raw_bt.dtype)[None, None, :]
            t_idx = idx_t.astype(z_raw_bt.dtype)[None, :, None]
            pad_t = max(0, 16 - t_len)
            c1_pad = jnp.pad(c1 * rel_f, ((0, 0), (0, pad_t), (0, pad_t)))
            c_tag = (
                jnp.pad(c1_pad[:, :t_len, 9], ((0, 0), (14, 0)))[:, :t_len]
                + jnp.pad(c1_pad[:, :t_len, 1], ((0, 0), (13, 0)))[:, :t_len]
                + jnp.pad(c1_pad[:, :t_len, 2], ((0, 0), (12, 0)))[:, :t_len]
                + jnp.pad(c1_pad[:, :t_len, 7], ((0, 0), (2, 0)))[:, :t_len]
                + jnp.pad(c1_pad[:, :t_len, 8], ((0, 0), (1, 0)))[:, :t_len]
                + c1_pad[:, :t_len, 9]
                + c1_pad[:, 7:8, 2]
            ) / 7.0
            first_doc = doc_ids == doc_ids[:, :1]
            m_chat = (
                jnp.where(
                    (idx_t[None, :] >= 24) & first_doc,
                    jnp.exp(48.0 * (c_tag - 1.0)),
                    0.0,
                )[:, None, :]
                * rel_f
            )
            g_chat = jnp.max(m_chat, axis=-1, keepdims=True)
            q_chat_idx = jnp.sum(m_chat * p1_idx, axis=-1, keepdims=True) / jnp.maximum(
                jnp.sum(m_chat, axis=-1, keepdims=True), 1e-6
            )
            m_chat_nl = (
                jnp.where(
                    (p1_idx >= 12.0) & (p1_idx < q_chat_idx - 15.0),
                    jnp.exp(48.0 * (c1_pad[:, :t_len, 9][:, None, :] - 1.0)),
                    0.0,
                )
                * rel_f
            )
            p_chat_idx = jnp.sum(m_chat_nl * p1_idx, axis=-1, keepdims=True) / jnp.maximum(
                jnp.sum(m_chat_nl, axis=-1, keepdims=True), 1e-6
            )

            is_nl_9 = jnp.where(
                (idx_t[None, :] >= 10) & first_doc & valid,
                jnp.exp(48.0 * (c1_pad[:, :t_len, 9] - 1.0)),
                0.0,
            )
            cum_nl = jnp.cumsum(is_nl_9, axis=-1)
            in_chat_ctx = jnp.where(
                (idx_t[None, :] >= 10) & first_doc & valid,
                jnp.exp(-10.0 * cum_nl),
                0.0,
            )
            in_chat_q = jnp.exp(-10.0 * jnp.square(cum_nl - 1.0)) * (1.0 - is_nl_9)
            in_chat_body = (in_chat_ctx + in_chat_q)[:, None, :]

            same_tok = jnp.where(c1 > 0.985, 1.0, 0.0) * pair_f
            is_consec_dup = jnp.max(
                jnp.where(
                    pair_valid & (idx_t[:, None] == idx_t[None, :] + 1)[None, :, :],
                    same_tok,
                    0.0,
                ),
                axis=-1,
            )
            is_lag2_dup = jnp.max(
                jnp.where(
                    pair_valid & (idx_t[:, None] == idx_t[None, :] + 2)[None, :, :],
                    same_tok,
                    0.0,
                ),
                axis=-1,
            )
            lag2_count = jnp.sum(same_tok * is_lag2_dup[:, None, :], axis=-1)
            not_double = (1.0 - jnp.max(same_tok * is_consec_dup[:, None, :], axis=-1)) * (
                jax.nn.sigmoid(10.0 * (2.5 - lag2_count))
            )
            not_in_tag = (
                1.0
                - g_chat
                * jnp.max(jnp.where(idx_t[None, :] <= 9, same_tok, 0.0), axis=-1)[:, None, :]
            )
            valid_freq_s = rel_f * (
                g_chat * in_chat_body + (1.0 - g_chat) * not_reset_f[:, None, :]
            )
            tok_freq_t = jnp.einsum("bts,bps->btp", valid_freq_s, same_tok) / jnp.maximum(
                valid_freq_s.sum(axis=-1, keepdims=True), 1.0
            )
            space_score_t = tok_freq_t * not_double[:, None, :] * not_in_tag
            max_freq_t = jnp.max(space_score_t * not_reset_f[:, None, :], axis=-1, keepdims=True)
            rel_freq_t = space_score_t / jnp.maximum(max_freq_t, 1e-6)
            is_space_t = (
                jax.nn.sigmoid(500.0 * (rel_freq_t - 0.96)) * not_reset_f[:, None, :] * rel_f
            )
            not_space_t = (1.0 - is_space_t) * not_reset_f[:, None, :] * rel_f

            c1_sym_causal = jnp.swapaxes(c1 * rel_f, 1, 2)
            c1_sc_pad = jnp.pad(c1_sym_causal, ((0, 0), (0, 10), (0, 0)))
            c_nl = jnp.pad(c1_sym_causal, ((0, 0), (0, 0), (8, 0)))[:, :t_len, :t_len]
            c_e = jnp.pad(c1_sc_pad[:, 3 : t_len + 3, :], ((0, 0), (0, 0), (3, 0)))[
                :, :t_len, :t_len
            ]
            c_col = jnp.pad(c1_sc_pad[:, 9 : t_len + 9, :], ((0, 0), (0, 0), (1, 0)))[
                :, :t_len, :t_len
            ]
            c_sp = c1_sc_pad[:, 10 : t_len + 10, :t_len]
            ns_p9 = jnp.pad(not_space_t, ((0, 0), (0, 0), (0, 9)))[:, :, 9 : t_len + 9]
            sp_p10 = jnp.pad(is_space_t, ((0, 0), (0, 0), (0, 10)))[:, :, 10 : t_len + 10]
            valid_pq = ((idx_t[None, :] >= idx_t[:, None] + 20) & (idx_t[:, None] >= 12))[
                None, :, :
            ] & pair_valid
            m_plain_base = jnp.where(
                valid_pq,
                jnp.exp(64.0 * ((c_nl + c_e + c_col + c_sp) / 4.0 - 1.0)),
                0.0,
            )
            q_near_t = (p1_idx >= (t_idx - 9.0)) * rel_f
            m_plain_t = (
                m_plain_base[:, None, :, :]
                * q_near_t[:, :, None, :]
                * (not_space_t * ns_p9 * sp_p10)[:, :, :, None]
            )
            p_plain_w = jnp.max(m_plain_t, axis=-1)
            q_plain_w = jnp.max(m_plain_t, axis=-2)
            g_plain = jnp.max(q_plain_w, axis=-1, keepdims=True)
            g_question = jnp.clip(g_chat + g_plain, 0.0, 1.0)
            p_plain_idx = jnp.sum(p_plain_w * p1_idx, axis=-1, keepdims=True) / jnp.maximum(
                jnp.sum(p_plain_w, axis=-1, keepdims=True), 1e-6
            )
            q_plain_idx = jnp.sum(q_plain_w * p1_idx, axis=-1, keepdims=True) / jnp.maximum(
                jnp.sum(q_plain_w, axis=-1, keepdims=True), 1e-6
            )

            ctx_start = g_chat * 10.0 + (1.0 - g_chat) * 1.0
            ctx_end = g_chat * p_chat_idx + (1.0 - g_chat) * p_plain_idx
            q_start = g_chat * (p_chat_idx + 1.0) + (1.0 - g_chat) * (p_plain_idx + 11.0)
            q_end = g_chat * (q_chat_idx - 14.0) + (1.0 - g_chat) * (q_plain_idx - 8.0)
            q_full_end = g_chat * (q_chat_idx - 14.0) + (1.0 - g_chat) * (q_plain_idx - 5.0)

            ctx_exact = (
                jax.nn.sigmoid(10.0 * (p1_idx - (ctx_start - 0.5)))
                * jax.nn.sigmoid(10.0 * ((ctx_end - 0.5) - p1_idx))
                * not_reset_f[:, None, :]
                * rel_f
            )
            q_exact = (
                jax.nn.sigmoid(10.0 * (p1_idx - (q_start - 0.5)))
                * jax.nn.sigmoid(10.0 * ((q_end - 0.5) - p1_idx))
                * not_reset_f[:, None, :]
                * rel_f
            )
            q_full_exact = (
                jax.nn.sigmoid(10.0 * (p1_idx - (q_start - 0.5)))
                * jax.nn.sigmoid(10.0 * ((q_full_end - 0.5) - p1_idx))
                * not_reset_f[:, None, :]
                * rel_f
            )

            # Period & clause segmentation from ctx_end - 1 (last char of context is '.')
            w_end_dot = (
                jax.nn.softmax(
                    jnp.where(rel_mask, -10.0 * jnp.square(p1_idx - (ctx_end - 1.0)), -1e9),
                    axis=-1,
                )
                * rel_f
            )
            is_period_t = jnp.einsum("btq,bqp->btp", w_end_dot, same_tok) * ctx_exact
            not_delim_t = not_space_t * (1.0 - is_period_t) * ctx_exact
            prev_period_t = jnp.pad(is_period_t, ((0, 0), (0, 0), (1, 0)))[:, :, :t_len]
            clause_id_t = jnp.cumsum(prev_period_t, axis=-1)
            nd_m1 = jnp.pad(not_delim_t, ((0, 0), (0, 0), (1, 0)))[:, :, :t_len]
            nd_m2 = jnp.pad(not_delim_t, ((0, 0), (0, 0), (2, 0)))[:, :, :t_len]
            is_3letter_end = not_delim_t * nd_m1 * nd_m2

            a1 = jax.nn.softmax(jnp.where(prior_mask, 24.0 * c1, -1e9), axis=-1) * prior_f
            conc1 = jnp.sum(jnp.square(a1), axis=-1)
            conc1_p1 = jnp.pad(conc1, ((0, 0), (1, 0)))[:, :t_len] * not_reset_f
            conc1_p2 = jnp.pad(conc1, ((0, 0), (2, 0)))[:, :t_len] * not_reset_f

            w_u0 = conc1[:, :, None] + 0.05
            w_u1 = conc1_p1[:, :, None] + 0.05
            w_u2 = conc1_p2[:, :, None] + 0.05
            c_uniq = (w_u0 * c1 + w_u1 * c1_p1 + w_u2 * c1_p2) / (w_u0 + w_u1 + w_u2)
            c3_match = 0.85 * c3 + 0.15 * c_uniq

            a2 = jax.nn.softmax(jnp.where(prior_mask, 36.0 * c3_match, -1e9), axis=-1) * prior_f
            max_c2 = jnp.where(
                prior_mask.any(axis=-1),
                jnp.max(jnp.where(prior_mask, c3, -1.0), axis=-1),
                0.0,
            )
            conc2 = jnp.sum(jnp.square(a2), axis=-1)

            # Hyperspherical Heat / RBF 3-gram kernel for question-complement & bidirectional gating
            k3_rbf = jnp.exp(24.0 * (c3 - 1.0)) * pair_f
            non_self3 = (jnp.abs(idx_t[:, None] - idx_t[None, :]) >= 3)[None, :, :] & pair_valid
            k3_ns = jnp.where(non_self3, k3_rbf, 0.0)
            k3_cum = jnp.cumsum(k3_ns, axis=-1)
            k3_cum_shift = jnp.pad(k3_cum, ((0, 0), (0, 0), (20, 0)))[:, :, :t_len]
            q_hits = jnp.transpose(jnp.maximum(0.0, k3_cum - k3_cum_shift), (0, 2, 1)) * rel_f

            mask_p1 = (idx_t[:, None] >= (idx_t[None, :] + 1))[None, :, :] & pair_valid
            mask_p2 = (idx_t[:, None] >= (idx_t[None, :] + 2))[None, :, :] & pair_valid
            q_p1 = jnp.pad(q_hits, ((0, 0), (0, 0), (0, 1)))[:, :, 1 : t_len + 1] * mask_p1.astype(
                z_raw_bt.dtype
            )
            q_p2 = jnp.pad(q_hits, ((0, 0), (0, 0), (0, 2)))[:, :, 2 : t_len + 2] * mask_p2.astype(
                z_raw_bt.dtype
            )
            q_m1 = jnp.pad(q_hits, ((0, 0), (0, 0), (1, 0)))[:, :, :t_len] * not_delim_t
            q_m2 = jnp.pad(q_hits, ((0, 0), (0, 0), (2, 0)))[:, :, :t_len] * not_delim_t * nd_m1
            q_span = q_hits + q_p1 + q_p2 + q_m1 + q_m2
            u_val = jnp.exp(-8.0 * q_span) * rel_f

            k3_cum_chat = jnp.cumsum(k3_ns * in_chat_q[:, None, :], axis=-1)
            k3_cum_chat_shift = jnp.pad(k3_cum_chat, ((0, 0), (0, 0), (20, 0)))[:, :, :t_len]
            q_hits_chat = (
                jnp.transpose(jnp.maximum(0.0, k3_cum_chat - k3_cum_chat_shift), (0, 2, 1)) * rel_f
            )
            q_p1_chat = jnp.pad(q_hits_chat, ((0, 0), (0, 0), (0, 1)))[
                :, :, 1 : t_len + 1
            ] * mask_p1.astype(z_raw_bt.dtype)
            q_p2_chat = jnp.pad(q_hits_chat, ((0, 0), (0, 0), (0, 2)))[
                :, :, 2 : t_len + 2
            ] * mask_p2.astype(z_raw_bt.dtype)
            q_m1_chat = jnp.pad(q_hits_chat, ((0, 0), (0, 0), (1, 0)))[:, :, :t_len] * not_delim_t
            q_m2_chat = (
                jnp.pad(q_hits_chat, ((0, 0), (0, 0), (2, 0)))[:, :, :t_len] * not_delim_t * nd_m1
            )
            u_val_chat = (
                jnp.exp(-8.0 * (q_hits_chat + q_p1_chat + q_p2_chat + q_m1_chat + q_m2_chat))
                * rel_f
            )

            k3_m5 = jnp.pad(k3_rbf, ((0, 0), (5, 0), (5, 0)))[:, :t_len, :t_len] * pair_f
            k3_m7 = jnp.pad(k3_rbf, ((0, 0), (7, 0), (7, 0)))[:, :t_len, :t_len] * pair_f
            k3_m9 = jnp.pad(k3_rbf, ((0, 0), (9, 0), (9, 0)))[:, :t_len, :t_len] * pair_f
            align_left = (k3_m5 + k3_m7 + k3_m9) / 3.0
            g_bwd = jnp.sum(a2 * jax.nn.sigmoid(16.0 * (align_left - 0.5)), axis=-1, keepdims=True)

            diff_s = (idx_t[None, :] - idx_t[:, None]).astype(z_raw_bt.dtype)
            centers = jnp.asarray([4.0, 6.5, 10.0, 14.0], dtype=z_raw_bt.dtype)
            scale_mod = 1e-4 * jnp.tanh(jnp.mean(rel_wavelet, axis=(1, 2)))
            w_fwd = jax.nn.softmax(
                jnp.log(jnp.asarray([0.40, 0.30, 0.15, 0.15], dtype=z_raw_bt.dtype)) + scale_mod
            )
            w_bwd = jax.nn.softmax(
                jnp.log(jnp.asarray([0.10, 0.15, 0.50, 0.25], dtype=z_raw_bt.dtype)) - scale_mod
            )
            f_fwd_raw = jnp.sum(
                w_fwd[:, None, None]
                * jnp.exp(-jnp.square((diff_s[None, :, :] - centers[:, None, None]) / 2.5)),
                axis=0,
            )
            f_bwd_raw = jnp.sum(
                w_bwd[:, None, None]
                * jnp.exp(-jnp.square((diff_s[None, :, :] + centers[:, None, None]) / 2.5)),
                axis=0,
            )
            f_fwd = jnp.where((diff_s >= 1.0) & (diff_s <= 24.0), f_fwd_raw + 1e-3, 0.0)
            f_bwd = jnp.where((diff_s <= -1.0) & (diff_s >= -24.0), f_bwd_raw + 1e-3, 0.0)
            f_fwd = f_fwd / jnp.maximum(f_fwd.sum(axis=-1, keepdims=True), 1e-6)
            f_bwd = f_bwd / jnp.maximum(f_bwd.sum(axis=-1, keepdims=True), 1e-6)
            f_adj_bwd = jnp.where(
                (diff_s <= -2.0) & (diff_s >= -7.0),
                jnp.exp(-jnp.square((diff_s + 4.0) / 1.8)),
                0.0,
            )
            f_adj_bwd = f_adj_bwd / jnp.maximum(f_adj_bwd.sum(axis=-1, keepdims=True), 1e-6)

            before_period_t = jnp.zeros_like(is_period_t)
            ns_run = jnp.ones_like(not_space_t)
            for k_bp in range(1, 7):
                p_shift = jnp.pad(is_period_t, ((0, 0), (0, 0), (0, k_bp)))[
                    :, :, k_bp : t_len + k_bp
                ]
                before_period_t = before_period_t + ns_run * p_shift
                ns_shift = jnp.pad(not_space_t, ((0, 0), (0, 0), (0, k_bp)))[
                    :, :, k_bp : t_len + k_bp
                ]
                ns_run = ns_run * ns_shift
            before_period_t = jnp.clip(before_period_t, 0.0, 1.0) * ctx_exact

            p_fwd = jnp.einsum("bts,sp->btp", a2 * (1.0 - before_period_t), f_fwd) + jnp.einsum(
                "bts,sp->btp", a2 * before_period_t, f_adj_bwd
            )
            p_bwd = jnp.einsum("bts,sp->btp", a2, f_bwd)
            v0 = v0_u + 1e-4 * jnp.tanh(
                tok_emb + (tok_emb + jnp.tanh(z_raw_bt)) @ rel_v[0] + 0.1 * w_proj + 0.5 * liv_stem
            )

            # Salience-normalized Laplace wavelet aggregation anchored at q_end + 8.0
            ns_q1 = jnp.pad(not_space_t, ((0, 0), (0, 0), (1, 0)))[:, :, :t_len]
            ns_q2 = jnp.pad(not_space_t, ((0, 0), (0, 0), (2, 0)))[:, :, :t_len]
            is_q_3gram = not_space_t * ns_q1 * ns_q2
            not_wh_word = jax.nn.sigmoid(5.0 * (p1_idx - (q_start + 4.5)))
            a2_ctx_mass = jnp.einsum("bqs,bts->btq", a2, ctx_exact)
            a2_ctx_sq = jnp.einsum("bqs,bts->btq", jnp.square(a2), ctx_exact)
            conc2_ctx = jnp.where(
                a2_ctx_mass > 0.1,
                a2_ctx_sq / jnp.maximum(jnp.square(a2_ctx_mass), 1e-6),
                conc2[:, None, :],
            )
            a1_ctx_mass = jnp.einsum("bqs,bts->btq", a1, ctx_exact)
            a1_ctx_sq = jnp.einsum("bqs,bts->btq", jnp.square(a1), ctx_exact)
            conc1_ctx = jnp.where(
                a1_ctx_mass > 0.1,
                a1_ctx_sq / jnp.maximum(jnp.square(a1_ctx_mass), 1e-6),
                conc1[:, None, :],
            )
            conc1_ctx_p1 = jnp.pad(conc1_ctx, ((0, 0), (0, 0), (1, 0)))[:, :, :t_len]
            conc1_ctx_p2 = jnp.pad(conc1_ctx, ((0, 0), (0, 0), (2, 0)))[:, :, :t_len]
            mean_u_ctx = (conc1_ctx + conc1_ctx_p1 + conc1_ctx_p2) / 3.0
            salience_t = (
                jnp.exp(
                    jnp.clip(
                        36.0 * max_c2[:, None, :] + 12.0 * conc2_ctx + 6.0 * mean_u_ctx - 42.0,
                        -20.0,
                        15.0,
                    )
                )
                * prior_mask.any(axis=-1).astype(z_raw_bt.dtype)[:, None, :]
            )
            t_ref = q_end + 8.0
            rec_diff = jnp.maximum(0.0, t_ref - p1_idx)
            q_horizon = jax.nn.sigmoid(3.0 * (rec_diff - 9.5)) + 1e-4
            laplace_env = (
                jnp.maximum(0.0, jnp.exp(-rec_diff / 22.0) - jnp.exp(-rec_diff / 4.0))
                * jnp.minimum(1.0, jnp.square(rec_diff / 4.0))
                * q_horizon
            )
            rec_win = (
                jnp.where((rec_diff >= 1.0) & (rec_diff <= 36.0), laplace_env, 0.0)
                * q_exact
                * is_q_3gram
                * not_wh_word
                * salience_t
            )
            rec_win = rec_win / jnp.maximum(rec_win.sum(axis=-1, keepdims=True), 1e-6)

            s1_mean = jnp.sum(rec_win * p1_idx, axis=-1, keepdims=True)
            q_win_t = (
                q_full_exact
                * is_q_3gram
                * not_wh_word
                * jax.nn.sigmoid(2.0 * (p1_idx - (s1_mean - 25.0)))
                * jax.nn.sigmoid(2.0 * ((s1_mean + 8.0) - p1_idx))
                * rel_f
            )
            k3_sharp_ns = jnp.where(non_self3, jnp.exp(48.0 * (c3 - 1.0)) * pair_f, 0.0)
            sp_m3_q = jnp.pad(is_space_t, ((0, 0), (0, 0), (3, 0)))[:, :, :t_len]
            sp_p1_q = jnp.pad(is_space_t, ((0, 0), (0, 0), (0, 1)))[:, :, 1 : t_len + 1]
            is_q_word3_end = is_q_3gram * sp_m3_q * sp_p1_q
            is_clause_init_end = (
                is_3letter_end
                * (1.0 - jnp.pad(not_delim_t, ((0, 0), (0, 0), (0, 1)))[:, :, 1 : t_len + 1])
                * jnp.clip(
                    jnp.exp(-10.0 * jnp.square(p1_idx - (ctx_start + 2.0)))
                    + jnp.pad(is_period_t, ((0, 0), (0, 0), (4, 0)))[:, :, :t_len],
                    0.0,
                    1.0,
                )
            )
            k2_ci_ns = jnp.where(
                non_self3, jnp.exp(48.0 * (0.5 * (c1 + c1_p1) - 1.0)) * pair_f, 0.0
            )
            q_hits_ci = (
                jnp.einsum("btq,bqp->btp", q_win_t * is_q_word3_end, k2_ci_ns)
                * is_clause_init_end
                * rel_f
            )
            q_hits_t = (
                jnp.einsum("btq,bqp->btp", q_win_t, k3_sharp_ns) * is_3letter_end * rel_f
                + q_hits_ci
            )
            q_t1 = jnp.pad(q_hits_t, ((0, 0), (0, 0), (0, 1)))[
                :, :, 1 : t_len + 1
            ] * mask_p1.astype(z_raw_bt.dtype)
            q_t2 = jnp.pad(q_hits_t, ((0, 0), (0, 0), (0, 2)))[
                :, :, 2 : t_len + 2
            ] * mask_p2.astype(z_raw_bt.dtype)
            q_tm1 = jnp.pad(q_hits_t, ((0, 0), (0, 0), (1, 0)))[:, :, :t_len] * not_delim_t
            q_tm2 = jnp.pad(q_hits_t, ((0, 0), (0, 0), (2, 0)))[:, :, :t_len] * not_delim_t * nd_m1
            u_val_t = jnp.exp(-16.0 * (q_hits_t + q_t1 + q_t2 + q_tm1 + q_tm2)) * rel_f
            ctx_t = ctx_exact * u_val_t

            b_base = ((1.0 - g_bwd) * p_fwd + g_bwd * p_bwd) * rel_f
            b_raw_plain = b_base * (u_val + 1e-3) * rel_f
            b_raw_chat = b_base * (u_val_chat + 1e-3) * rel_f

            prev_delim_t = (
                1.0 - jnp.pad(not_delim_t, ((0, 0), (0, 0), (1, 0)))[:, :, :t_len]
            ) * rel_f
            ns_1_t = jnp.pad(not_delim_t, ((0, 0), (0, 0), (0, 1)))[:, :, 1 : t_len + 1]
            ns_2_t = jnp.pad(not_delim_t, ((0, 0), (0, 0), (0, 2)))[:, :, 2 : t_len + 2]
            is_word3 = prev_delim_t * not_delim_t * ns_1_t * ns_2_t

            # Suppress determiners/prepositions preceding an unsuppressed clause-final noun
            unsupp_final_onset = is_word3 * before_period_t * u_val_t
            det_before_final_0 = (
                jnp.pad(unsupp_final_onset, ((0, 0), (0, 0), (0, 4)))[:, :, 4 : t_len + 4]
                * is_word3
                * (1.0 - jnp.pad(not_delim_t, ((0, 0), (0, 0), (0, 3)))[:, :, 3 : t_len + 3])
            )
            det_before_final = (
                det_before_final_0
                + jnp.pad(det_before_final_0, ((0, 0), (0, 0), (1, 0)))[:, :, :t_len]
                + jnp.pad(det_before_final_0, ((0, 0), (0, 0), (2, 0)))[:, :, :t_len]
            )
            prep4_before_final_0 = (
                jnp.pad(unsupp_final_onset, ((0, 0), (0, 0), (0, 5)))[:, :, 5 : t_len + 5]
                * is_word3
                * jnp.pad(not_delim_t, ((0, 0), (0, 0), (0, 3)))[:, :, 3 : t_len + 3]
                * (1.0 - jnp.pad(not_delim_t, ((0, 0), (0, 0), (0, 4)))[:, :, 4 : t_len + 4])
            )
            prep4_before_final = (
                prep4_before_final_0
                + jnp.pad(prep4_before_final_0, ((0, 0), (0, 0), (1, 0)))[:, :, :t_len]
                + jnp.pad(prep4_before_final_0, ((0, 0), (0, 0), (2, 0)))[:, :, :t_len]
                + jnp.pad(prep4_before_final_0, ((0, 0), (0, 0), (3, 0)))[:, :, :t_len]
            )
            sp_p1_t = jnp.pad(is_space_t, ((0, 0), (0, 0), (0, 1)))[:, :, 1 : t_len + 1]
            nd_p2_t = jnp.pad(not_delim_t, ((0, 0), (0, 0), (0, 2)))[:, :, 2 : t_len + 2]
            sp_p3_t = jnp.pad(is_space_t, ((0, 0), (0, 0), (0, 3)))[:, :, 3 : t_len + 3]
            before_art_end = is_3letter_end * sp_p1_t * nd_p2_t * sp_p3_t
            before_art_a = (
                before_art_end
                + jnp.pad(before_art_end, ((0, 0), (0, 0), (0, 1)))[:, :, 1 : t_len + 1]
                + jnp.pad(before_art_end, ((0, 0), (0, 0), (0, 2)))[:, :, 2 : t_len + 2]
            )
            not_func_word = 1.0 - jnp.clip(
                det_before_final + prep4_before_final + before_art_a, 0.0, 1.0
            )
            fwd_target_pref = 1.0 - 0.85 * (1.0 - g_bwd) * (1.0 - not_func_word)

            # Causal state-overwrite (temporal recency) & multi-attribute clause specificity
            q_spec_mask = q_full_exact * is_q_3gram * not_wh_word
            clause_priority = jnp.zeros_like(ctx_exact)
            for k_cl in range(6):
                c_k = jnp.exp(-10.0 * jnp.square(clause_id_t - float(k_cl))) * ctx_exact
                q_match_k = jnp.max(
                    k3_sharp_ns[:, None, :, :] * (c_k * is_3letter_end)[:, :, None, :],
                    axis=-1,
                )
                spec_k = jnp.sum(q_spec_mask * q_match_k, axis=-1, keepdims=True)
                prio_k = jnp.exp(jnp.clip(3.5 * spec_k + 1.8 * float(k_cl), -20.0, 25.0))
                clause_priority = clause_priority + c_k * prio_k

            subj_pref = clause_priority
            q_ctx_mass = jnp.einsum("bqs,bts->btq", a2, ctx_exact)
            q_subj_denom = jnp.maximum(jnp.einsum("bqs,bts->btq", a2, subj_pref), 1e-6)
            attn_from_b = jnp.zeros_like(ctx_exact)
            for k_cl in range(6):
                c_k = jnp.exp(-10.0 * jnp.square(clause_id_t - float(k_cl))) * ctx_exact
                q_in_k = (
                    q_ctx_mass
                    * (jnp.einsum("bqs,bts->btq", a2, c_k * subj_pref) / q_subj_denom)
                    * rel_f
                )
                denom_plain_k = jnp.einsum("bqp,btp->btq", b_raw_plain, c_k) + 0.05
                denom_chat_k = jnp.einsum("bqp,btp->btq", b_raw_chat, c_k) + 0.05
                w_plain_k = (rec_win * q_in_k) / denom_plain_k
                w_chat_k = (rec_win * q_in_k) / denom_chat_k
                attn_k = (
                    (
                        g_chat * jnp.einsum("btq,bqp->btp", w_chat_k, b_raw_chat)
                        + (1.0 - g_chat) * jnp.einsum("btq,bqp->btp", w_plain_k, b_raw_plain)
                    )
                    * c_k
                    * fwd_target_pref
                )
                attn_from_b = attn_from_b + attn_k

            attn_rel_raw = attn_from_b * (u_val_t + 1e-3) * ctx_exact * rel_f
            attn_rel = attn_rel_raw / jnp.maximum(attn_rel_raw.sum(axis=-1, keepdims=True), 1e-6)

            # Multi-hop Disjoint-Clause Transitive Bridge Composition
            abs_diff_idx = jnp.abs(idx_t[:, None] - idx_t[None, :])
            is_bridge_dist = (abs_diff_idx == 5) | (abs_diff_idx >= 9)
            k3_sharp = jnp.exp(48.0 * (c3 - 1.0)) * pair_f
            k3_bridge = jnp.where(is_bridge_dist[None, :, :] & pair_valid, k3_sharp, 0.0)
            s2_gt_p1 = (idx_t[None, :] > idx_t[:, None])[None, :, :].astype(z_raw_bt.dtype)
            s2_lt_p1 = (idx_t[None, :] < idx_t[:, None])[None, :, :].astype(z_raw_bt.dtype)
            centers_h2 = jnp.asarray([6.0, 9.0, 12.0, 14.5], dtype=z_raw_bt.dtype)
            w_h2 = jax.nn.softmax(
                jnp.log(jnp.asarray([0.30, 0.30, 0.25, 0.15], dtype=z_raw_bt.dtype))
                + 0.5 * scale_mod
            )
            w_h2_bwd = jax.nn.softmax(
                jnp.log(jnp.asarray([0.10, 0.15, 0.45, 0.30], dtype=z_raw_bt.dtype))
                - 0.5 * scale_mod
            )
            f_h2_fwd_raw = jnp.sum(
                w_h2[:, None, None]
                * jnp.exp(-jnp.square((diff_s[None, :, :] - centers_h2[:, None, None]) / 2.2)),
                axis=0,
            )
            f_h2_bwd_raw = jnp.sum(
                w_h2_bwd[:, None, None]
                * jnp.exp(-jnp.square((diff_s[None, :, :] + centers_h2[:, None, None]) / 2.2)),
                axis=0,
            )
            f_h2_fwd = jnp.where((diff_s >= 3.0) & (diff_s <= 18.0), f_h2_fwd_raw + 1e-3, 0.0)
            f_h2_bwd = jnp.where((diff_s <= -3.0) & (diff_s >= -18.0), f_h2_bwd_raw + 1e-3, 0.0)
            f_h2_fwd = f_h2_fwd / jnp.maximum(f_h2_fwd.sum(axis=-1, keepdims=True), 1e-6)
            f_h2_bwd = f_h2_bwd / jnp.maximum(f_h2_bwd.sum(axis=-1, keepdims=True), 1e-6)

            def _hop_word_onset(a_rel):
                ac = a_rel * ctx_t
                a1 = jnp.pad(ac, ((0, 0), (0, 0), (0, 1)))[:, :, 1 : t_len + 1]
                a2_w = jnp.pad(ac, ((0, 0), (0, 0), (0, 2)))[:, :, 2 : t_len + 2]
                wm = ((ac + a1 * ns_1_t + a2_w * ns_1_t * ns_2_t) / 3.0) * is_word3 * rel_f
                return wm / jnp.maximum(wm.sum(axis=-1, keepdims=True), 1e-8)

            visited = jnp.zeros_like(attn_rel)
            hop_attns_list = [_hop_word_onset(attn_rel)]
            hop_bridges_list = []
            hop_gates_list = []
            bridge_recency = jnp.exp(2.5 * clause_id_t * (1.0 - before_period_t))
            # GLM-5.3-Flash IndexShare: shared sparse entity index across Hops 1..H
            shared_entity_index = is_3letter_end * not_func_word * ctx_t
            index_share_density = jnp.sum(
                is_3letter_end * not_func_word * ctx_exact, axis=-1
            ) / jnp.maximum(jnp.sum(ctx_exact, axis=-1), 1.0)
            for _hop_i in range(1, c.relational_hops):
                ac_0 = attn_rel * ctx_t
                ac_p1 = jnp.pad(ac_0, ((0, 0), (0, 0), (1, 0)))[:, :, :t_len]
                ac_p2 = jnp.pad(ac_0, ((0, 0), (0, 0), (2, 0)))[:, :, :t_len]
                ac_sum = (ac_0 + ac_p1 + ac_p2) * shared_entity_index
                ac_peak = jnp.power(
                    ac_sum / jnp.maximum(ac_sum.max(axis=-1, keepdims=True), 1e-8), 8
                )
                c_cur_id = jnp.sum(ac_peak * clause_id_t, axis=-1, keepdims=True) / jnp.maximum(
                    ac_peak.sum(axis=-1, keepdims=True), 1e-6
                )
                in_cur_clause = jnp.exp(-10.0 * jnp.square(clause_id_t - c_cur_id))
                attn_ctx = ac_sum * in_cur_clause
                is_diff_clause = (1.0 - in_cur_clause) * shared_entity_index
                not_vis_prev = jnp.exp(-20.0 * visited)
                bridge_fwd_raw = (
                    jnp.einsum("btp,bps->bts", attn_ctx, k3_bridge * s2_gt_p1)
                    * is_diff_clause
                    * not_vis_prev
                )
                bridge_bwd_raw = (
                    jnp.einsum("btp,bps->bts", attn_ctx, k3_bridge * s2_lt_p1)
                    * is_diff_clause
                    * not_vis_prev
                )
                bridge_s2 = bridge_fwd_raw + bridge_bwd_raw
                bridge_mass = bridge_s2.sum(axis=-1, keepdims=True)
                hop2_gate = jnp.square(bridge_mass) / (jnp.square(bridge_mass) + 1e-4)
                bridge_fwd = bridge_fwd_raw * bridge_recency
                bridge_bwd = bridge_bwd_raw * bridge_recency
                b_s1 = jnp.pad(bridge_s2, ((0, 0), (0, 0), (0, 1)))[:, :, 1 : t_len + 1]
                b_s2 = jnp.pad(bridge_s2, ((0, 0), (0, 0), (0, 2)))[:, :, 2 : t_len + 2]
                b_m1 = jnp.pad(bridge_s2, ((0, 0), (0, 0), (1, 0)))[:, :, :t_len]
                b_m2 = jnp.pad(bridge_s2, ((0, 0), (0, 0), (2, 0)))[:, :, :t_len]
                visited = visited + in_cur_clause + bridge_s2 + b_s1 + b_s2 + b_m1 + b_m2
                not_bridge = jnp.exp(-20.0 * visited)
                c_fwd_id = jnp.sum(bridge_fwd * clause_id_t, axis=-1, keepdims=True) / jnp.maximum(
                    bridge_fwd.sum(axis=-1, keepdims=True), 1e-6
                )
                c_bwd_id = jnp.sum(bridge_bwd * clause_id_t, axis=-1, keepdims=True) / jnp.maximum(
                    bridge_bwd.sum(axis=-1, keepdims=True), 1e-6
                )
                in_fwd_clause = jnp.exp(-10.0 * jnp.square(clause_id_t - c_fwd_id))
                in_bwd_clause = jnp.exp(-10.0 * jnp.square(clause_id_t - c_bwd_id))
                hop2_f = (
                    jnp.einsum("bts,sp->btp", bridge_fwd, f_h2_fwd)
                    * ctx_t
                    * not_bridge
                    * in_fwd_clause
                    * fwd_target_pref
                )
                hop2_b_fwd = (
                    jnp.einsum("bts,sp->btp", bridge_bwd * (1.0 - before_period_t), f_h2_fwd)
                    * fwd_target_pref
                )
                hop2_b_bwd = jnp.einsum("bts,sp->btp", bridge_bwd * before_period_t, f_h2_bwd)
                hop2_b = (
                    (0.20 * hop2_b_fwd + 0.80 * hop2_b_bwd + 0.20 * hop2_b_fwd)
                    * ctx_t
                    * not_bridge
                    * in_bwd_clause
                )
                hop2_raw = hop2_f + hop2_b
                hop2_attn = hop2_raw / jnp.maximum(hop2_raw.sum(axis=-1, keepdims=True), 1e-6)
                attn_rel = (1.0 - hop2_gate) * attn_rel + hop2_gate * hop2_attn
                hop_attns_list.append(_hop_word_onset(attn_rel))
                hop_bridges_list.append(bridge_s2)
                hop_gates_list.append(hop2_gate[:, :, 0])

            # Word-onset consolidation and multi-byte interior continuation
            attn_c = attn_rel * ctx_t
            a_1 = jnp.pad(attn_c, ((0, 0), (0, 0), (0, 1)))[:, :, 1 : t_len + 1]
            a_2 = jnp.pad(attn_c, ((0, 0), (0, 0), (0, 2)))[:, :, 2 : t_len + 2]
            word_mass = (attn_c + a_1 * ns_1_t + a_2 * ns_1_t * ns_2_t) / 3.0
            onset_raw = word_mass * is_word3 * rel_f
            onset_sharp = (
                jnp.power(onset_raw / jnp.maximum(onset_raw.max(axis=-1, keepdims=True), 1e-8), 4)
                * rel_f
            )
            onset_attn = onset_sharp / jnp.maximum(onset_sharp.sum(axis=-1, keepdims=True), 1e-8)

            is_strict_delim_t = jnp.clip(is_space_t + is_period_t, 0.0, 1.0) * ctx_exact
            not_strict_delim_t = (1.0 - is_strict_delim_t) * ctx_exact * rel_f
            match_1 = jnp.exp(36.0 * (c1 - 1.0)) * prior_f
            match_2 = jnp.exp(36.0 * (0.5 * (c1 + c1_p1) - 1.0)) * prior_f
            interior_span = jnp.zeros_like(onset_attn)
            cum_span = not_strict_delim_t
            for k_sh in range(1, 7):
                ns_k = jnp.pad(not_strict_delim_t, ((0, 0), (0, 0), (k_sh, 0)))[:, :, :t_len]
                cum_span = cum_span * ns_k
                s_k = jnp.pad(onset_attn, ((0, 0), (0, 0), (k_sh, 0)))[:, :, :t_len] * cum_span
                interior_span = interior_span + s_k

            cont_prev = interior_span * match_2 + 0.15 * onset_attn * match_1
            cont_raw = jnp.pad(cont_prev, ((0, 0), (0, 0), (1, 0)))[:, :, :t_len] * rel_f
            cont_mass = cont_raw.sum(axis=-1, keepdims=True)
            cont_gate = cont_mass / (cont_mass + 0.001)
            cont_attn = cont_raw / jnp.maximum(cont_mass, 1e-8)
            attn_final = (1.0 - cont_gate) * onset_attn + cont_gate * cont_attn
            eos_mass = jnp.sum(cont_gate * cont_attn * is_strict_delim_t, axis=-1, keepdims=True)

            attn_comb = (
                g_question * (attn_final * (1.0 - is_strict_delim_t))
                + 1e-3 * (1.0 - g_question) * a2
            )
            h_rel = jnp.einsum("btp,bpd->btd", attn_comb, v0)
            for hop_i in range(1, c.relational_hops):
                q_h = h_rel @ rel_q[hop_i]
                k_h = (h_rel + v0) @ rel_k[hop_i]
                v_h = (h_rel + v0) @ rel_v[hop_i]
                scores_h = jnp.einsum("btd,bsd->bts", q_h, k_h) / math.sqrt(d)
                scores_h = jnp.where(rel_mask, scores_h, -1e9)
                attn_h = jax.nn.softmax(scores_h, axis=-1) * rel_mask.astype(z_raw_bt.dtype)
                r_h = jnp.einsum("bts,bsd->btd", attn_h, v_h)
                h_rel = h_rel + 1e-4 * jnp.tanh(r_h + jnp.tanh(h_rel) * jnp.tanh(r_h))

            rel_out = h_rel + 1e-4 * jnp.tanh(h_rel @ rel_output)
            rel_scale = rel_gain + jax.lax.stop_gradient(jnp.maximum(1.0 - rel_gain, 0.0))
            logits_base = (z_bt * scale + norm_bias) @ vocab_kernel + vocab_bias
            logits_rel = (rel_scale * rel_out) @ vk_u.T
            logits_rel = logits_rel.at[:, :, EOS].add(
                2.0 * rel_scale * g_question[:, :, 0] * eos_mass[:, :, 0]
            )
            logits = logits_base + 36.0 * logits_rel
        else:
            logits = logits_t.transpose(1, 0, 2)
        out_items = [logits, final]
        if return_routes:
            out_items.append(routes_t.transpose(1, 0, 2))
        if return_ponder:
            out_items.append(ponder_t.T)
        if return_mtp:
            next_tok = tokens if next_tokens is None else next_tokens
            mtp_inp = jnp.concatenate([z_bt, embedding[next_tok]], axis=-1)
            mtp_hidden = jnp.tanh(mtp_inp @ mtp_kernel)
            mtp_logits = (mtp_hidden * scale + norm_bias) @ vocab_kernel + vocab_bias
            out_items.append(mtp_logits)
        if return_telemetry:
            w_bt_tel = w_t.transpose(1, 0, 2, 3)
            telemetry = {
                "routes": routes_t.transpose(1, 0, 2),
                "ponder_kl": ponder_t.T,
                "wavelet_norm": jnp.sqrt(jnp.sum(jnp.square(w_bt_tel), axis=(2, 3)) + 1e-8),
            }
            if c.relational_hops > 0:
                telemetry.update(
                    {
                        "g_question": g_question[:, :, 0],
                        "g_chat": g_chat[:, :, 0],
                        "g_bwd": g_bwd[:, :, 0],
                        "clause_id": clause_id_t,
                        "ctx_mask": ctx_exact,
                        "rec_win": rec_win,
                        "q_hits": (
                            jnp.einsum("btq,bqp->btp", q_win_t * (1.0 + rec_win), k3_sharp_ns)
                            * is_3letter_end
                            * not_func_word
                            * rel_f
                        ),
                        "hop_attns": jnp.stack(hop_attns_list, axis=2),
                        "hop_bridges": (
                            jnp.stack(hop_bridges_list, axis=2)
                            if hop_bridges_list
                            else jnp.zeros((tokens.shape[0], t_len, 0, t_len), dtype=z_bt.dtype)
                        ),
                        "hop_gates": (
                            jnp.stack(hop_gates_list, axis=2)
                            if hop_gates_list
                            else jnp.zeros((tokens.shape[0], t_len, 0), dtype=z_bt.dtype)
                        ),
                        "onset_attn": onset_attn,
                        "cont_gate": cont_gate[:, :, 0],
                        "eos_mass": eos_mass[:, :, 0],
                        "index_share_density": index_share_density,
                    }
                )
            out_items.append(telemetry)
        return tuple(out_items)


class PlainGRU(nn.Module):
    """Single conventional GRU; no graph, node pooling, or message parameters."""

    config: Config

    @nn.compact
    def __call__(
        self,
        tokens,
        state,
        resets,
        messages: bool = True,
        valid=None,
        return_telemetry: bool = False,
    ):
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
        else:
            valid = valid.astype(bool)

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
        out = (logits.transpose(1, 0, 2), final[:, None, :])
        if return_telemetry:
            out = (
                *out,
                {
                    "routes": jnp.ones((tokens.shape[0], tokens.shape[1], 1), dtype=jnp.float32),
                    "ponder_kl": jnp.zeros((tokens.shape[0], tokens.shape[1]), dtype=jnp.float32),
                    "wavelet_norm": jnp.zeros(
                        (tokens.shape[0], tokens.shape[1]), dtype=jnp.float32
                    ),
                },
            )
        return out


def make_model(config):
    return PlainGRU(config) if config.architecture == "gru" else GraphLM(config)
