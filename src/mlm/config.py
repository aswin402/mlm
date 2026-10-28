"""Validated, immutable experiment settings."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    nodes: int = Field(default=8, ge=2, le=128)
    width: int = Field(default=32, ge=4, le=1024)
    degree: int = Field(default=2, ge=1)
    rounds: int = Field(default=1, ge=1, le=16)
    batch_size: int = Field(default=2, ge=1, le=1024)
    length: int = Field(default=32, ge=2, le=8192)
    learning_rate: float = Field(default=3e-4, gt=0, le=1, allow_inf_nan=False)
    shared_lr_scale: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)
    node_lr_scale: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)
    router_lr_scale: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)
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
        return self
