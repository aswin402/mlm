"""Validated, immutable experiment settings."""

import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Prevent XLA/OpenMP from saturating all logical CPU threads and freezing laptop UI
if not (os.environ.get("COLAB_GPU") or os.environ.get("COLAB_RELEASE_TAG")):
    os.environ.setdefault("OMP_NUM_THREADS", "6")
    os.environ.setdefault(
        "XLA_FLAGS",
        "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=6",
    )
else:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    architecture: Literal["graph", "gru"] = "graph"
    node_update: Literal["dense", "uniform", "hierarchical", "timescale"] = "dense"
    groups: int = Field(default=4, ge=2, le=64)
    router_balance: float = Field(default=0.01, ge=0, le=1, allow_inf_nan=False)
    edge_aggregation: Literal["sum", "attention"] = "sum"
    node_readout: Literal["sum", "attention"] = "sum"
    memory: Literal["none", "delta", "relational"] = "none"
    memory_keys: int = Field(default=16, ge=4, le=128)
    memory_lr_scale: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)
    local_plasticity: bool = False
    adaptive_halt: bool = False
    ponder_beta: float = Field(default=0.0, ge=0, le=1, allow_inf_nan=False)
    halt_prior: float = Field(default=0.4, gt=0, lt=1, allow_inf_nan=False)
    warmup_steps: int = Field(default=0, ge=0, le=100000)
    top_k_nodes: int = Field(default=0, ge=0, le=128)
    shared_nodes: int = Field(default=0, ge=0, le=64)
    router_bias_gamma: float = Field(default=0.0, ge=0, le=1, allow_inf_nan=False)
    memory_decay: bool = False
    symmetric_memory: bool = False
    surprise_gating: bool = False
    episodic_slots: int = Field(default=0, ge=0, le=128)
    relational_hops: int = Field(default=0, ge=0, le=4)
    mtp_weight: float = Field(default=0.0, ge=0, le=1, allow_inf_nan=False)
    sinkhorn_edges: bool = False
    edges: tuple[tuple[int, ...], ...] | None = None
    optimizer: Literal["adamw", "muon"] = "adamw"
    nodes: int = Field(default=8, ge=2, le=128)
    width: int = Field(default=32, ge=4, le=2048)
    degree: int = Field(default=2, ge=1)
    rounds: int = Field(default=1, ge=1, le=16)
    batch_size: int = Field(default=2, ge=1, le=1024)
    length: int = Field(default=32, ge=2, le=8192)
    learning_rate: float = Field(default=3e-4, gt=0, le=1, allow_inf_nan=False)
    shared_lr_scale: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)
    node_lr_scale: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)
    router_lr_scale: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)
    mature_nodes: int = Field(default=0, ge=0, le=128)
    mature_node_lr_scale: float = Field(default=0.1, gt=0, le=1, allow_inf_nan=False)
    weight_decay: float = Field(default=0.01, ge=0, le=1, allow_inf_nan=False)
    clip_norm: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)

    @model_validator(mode="after")
    def check_graph(self):
        if self.memory != "none" and (self.architecture != "graph" or self.node_readout != "sum"):
            raise ValueError("Delta memory requires graph architecture with sum readout")
        if self.architecture == "gru" and self.node_update != "dense":
            raise ValueError("Recurrent node routing requires graph architecture")
        if self.architecture == "gru" and self.local_plasticity:
            raise ValueError("Local node plasticity requires graph architecture")
        if self.architecture == "gru" and self.adaptive_halt:
            raise ValueError("Adaptive halting requires graph architecture")
        if self.node_update == "hierarchical" and (
            self.nodes % self.groups or self.groups >= self.nodes
        ):
            raise ValueError("Hierarchical groups must divide nodes with at least two children")
        if self.architecture == "gru" and self.edge_aggregation != "sum":
            raise ValueError("Edge attention requires graph architecture")
        if self.architecture == "gru" and self.node_readout != "sum":
            raise ValueError("Node attention requires graph architecture")
        if self.degree >= self.nodes:
            raise ValueError("degree must be smaller than nodes")
        if self.top_k_nodes > 0:
            if self.architecture == "gru" or self.node_update in {"dense", "uniform"}:
                raise ValueError(
                    "top_k_nodes requires routed graph node_update ('hierarchical' or 'timescale')"
                )
            if self.top_k_nodes >= self.nodes:
                raise ValueError("top_k_nodes must be smaller than nodes when active (> 0)")
        if self.shared_nodes > 0:
            if self.architecture == "gru" or self.node_update in {"dense", "uniform"}:
                raise ValueError(
                    "shared_nodes requires routed graph node_update ('hierarchical' or 'timescale')"
                )
            if self.shared_nodes >= self.nodes:
                raise ValueError("shared_nodes must be smaller than nodes")
            if self.top_k_nodes > 0 and self.shared_nodes + self.top_k_nodes > self.nodes:
                raise ValueError("shared_nodes + top_k_nodes must not exceed nodes")
        if self.memory_decay and self.memory == "none":
            raise ValueError("memory_decay requires memory != 'none'")
        if self.symmetric_memory and self.memory == "none":
            raise ValueError("symmetric_memory requires memory != 'none'")
        if self.episodic_slots > 0 and self.architecture == "gru":
            raise ValueError("episodic_slots requires graph architecture")
        if self.surprise_gating and self.memory == "none":
            raise ValueError("surprise_gating requires memory != 'none'")
        if self.relational_hops > 0 and self.architecture == "gru":
            raise ValueError("relational_hops requires graph architecture")
        if self.sinkhorn_edges and (
            self.architecture == "gru" or self.edge_aggregation != "attention"
        ):
            raise ValueError("sinkhorn_edges requires graph architecture with edge attention")
        if self.mtp_weight > 0 and self.architecture == "gru":
            raise ValueError("mtp_weight requires graph architecture")
        if self.edges is not None:
            if self.architecture == "gru":
                raise ValueError("Custom edges require graph architecture")
            if len(self.edges) != self.nodes or any(len(row) != self.degree for row in self.edges):
                raise ValueError("edges must have shape (nodes, degree)")
            for i, row in enumerate(self.edges):
                if len(set(row)) != self.degree or any(
                    not (0 <= j < self.nodes) or j == i for j in row
                ):
                    raise ValueError(
                        "Each node's incoming edges must be distinct valid neighbor indices != self"
                    )
        return self
