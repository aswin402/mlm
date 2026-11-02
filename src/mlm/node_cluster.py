"""Multi-Node Specialist Cluster Router for Node-Based Growing MLM v2.

Architecture:
  Main Core Router -> [Node-0: 10M English/Science Node, Node-1: 10M Math Node, ...]

Routing is 100% neural: the Core Router evaluates each specialist node's own learned
token-transition log-likelihood and predictive entropy at the `<|assistant|>` boundary,
with zero hardcoded word lists, character tables, or regexes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mlm.model_v2 import (
    ConfigV2,
    forward_v2,
    generate_v2,
    load_checkpoint_v2,
)
from mlm.tokenizer import BPETokenizer
from mlm.train_v2 import format_chat_prompt_v2


@dataclass
class SpecialistNodeV2:
    """Loaded MLM v2 specialist node checkpoint and metadata."""

    name: str
    domain: str
    checkpoint_path: Path
    params: dict[str, Any]
    config: ConfigV2
    tokenizer: BPETokenizer
    manifest: dict[str, Any]


def score_node_prompt_affinity_v2(
    node: SpecialistNodeV2,
    prompt_text: str,
) -> dict[str, float]:
    """Compute a specialist node's pure neural log-likelihood and next-token sharpness."""
    ids = node.tokenizer.encode(prompt_text, add_bos=True, add_eos=False)
    if len(ids) > node.config.max_seq_len:
        ids = ids[-node.config.max_seq_len :]
    arr = jnp.asarray([ids], dtype=jnp.int32)
    logits = forward_v2(node.params, arr, node.config)[0]  # (T, V)
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)

    if len(ids) > 1:
        target_ids = jnp.asarray(ids[1:], dtype=jnp.int32)
        token_ll = jnp.take_along_axis(log_probs[:-1], target_ids[:, None], axis=-1).squeeze(-1)
        mean_ll = float(jnp.mean(token_ll))
    else:
        mean_ll = float(jnp.max(log_probs[0]))

    next_lp = log_probs[-1]
    next_p = jnp.exp(next_lp)
    next_entropy = float(-jnp.sum(next_p * next_lp))
    vocab_norm_conf = float(1.0 - (next_entropy / max(np.log(float(node.config.vocab_size)), 1.0)))
    prompt_ppl = float(np.exp(np.clip(-mean_ll, 0.0, 20.0)))
    return {
        "mean_log_likelihood": mean_ll,
        "prompt_perplexity": prompt_ppl,
        "next_token_confidence": vocab_norm_conf,
    }


class MLMNodeCluster:
    """Main Core Router connecting independent 10M MLM v2 Specialist Nodes."""

    def __init__(self, nodes: list[SpecialistNodeV2]) -> None:
        if not nodes:
            raise ValueError("MLMNodeCluster requires at least one SpecialistNodeV2")
        self.nodes = {node.name: node for node in nodes}
        self.node_order = [node.name for node in nodes]

    @classmethod
    def from_checkpoints(
        cls,
        node_specs: list[tuple[str, str, Path]],
    ) -> MLMNodeCluster:
        """Load multiple (name, domain, checkpoint_path) specialist nodes into the cluster."""
        loaded: list[SpecialistNodeV2] = []
        for name, domain, ckpt_path in node_specs:
            params, cfg, tok, manifest = load_checkpoint_v2(Path(ckpt_path))
            loaded.append(
                SpecialistNodeV2(
                    name=name,
                    domain=domain,
                    checkpoint_path=Path(ckpt_path),
                    params=params,
                    config=cfg,
                    tokenizer=tok,
                    manifest=manifest,
                )
            )
        return cls(loaded)

    def route_prompt(self, message: str, context: str = "") -> dict[str, Any]:
        """Select the best specialist node purely from neural log-likelihood & confidence."""
        prompt = format_chat_prompt_v2(message, context=context)

        raw_metrics: dict[str, dict[str, float]] = {}
        scores: dict[str, float] = {}
        for name in self.node_order:
            node = self.nodes[name]
            m = score_node_prompt_affinity_v2(node, prompt)
            raw_metrics[name] = m
            # Pure neural routing score: sequence log-likelihood + assistant boundary confidence
            scores[name] = float(m["mean_log_likelihood"] + 0.50 * m["next_token_confidence"])

        selected_name = max(self.node_order, key=lambda n: scores[n])
        return {
            "selected_node": selected_name,
            "selected_domain": self.nodes[selected_name].domain,
            "scores": scores,
            "node_metrics": raw_metrics,
        }

    def generate(
        self,
        message: str,
        *,
        context: str = "",
        max_new_tokens: int = 96,
        temperature: float = 0.0,
        force_node: str | None = None,
    ) -> dict[str, Any]:
        """Route `message` via the Main Core Router and generate from the chosen Specialist Node."""
        routing = self.route_prompt(message, context=context)
        chosen_name = force_node if force_node in self.nodes else routing["selected_node"]
        node = self.nodes[chosen_name]
        prompt = format_chat_prompt_v2(message, context=context)
        reply = generate_v2(
            node.params,
            node.config,
            node.tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return {
            "reply": reply.strip(),
            "selected_node": chosen_name,
            "selected_domain": node.domain,
            "routing": routing,
        }
