from __future__ import annotations

from dataclasses import dataclass


_RESPONSES = ("score", "acc")
_SAMPLES = ("random", "all")


@dataclass(frozen=True)
class DataSpec:
    key: int = 4
    response: str = "score"     # "score" | "acc"
    sample: str = "random"      # "random" | "all"
    min_item: int = 2           # >= filter, part of the cache identity
    min_user: int = 2

    def __post_init__(self):
        if self.response not in _RESPONSES:
            raise ValueError(f"Response must be one of {_RESPONSES}")
        if self.sample not in _SAMPLES:
            raise ValueError(f"Sample must be one of {_SAMPLES}")

    @property
    def cache_name(self) -> str:
        return f"{self.key}k_{self.response}_{self.sample}_mi{self.min_item}_mi{self.min_user}"

    def tag(self) -> dict:
        return {"key": self.key, "response_kind": self.response, "sample_set": self.sample}


@dataclass(frozen=True)
class FitConfig:
    spec: DataSpec = DataSpec()
    model: str = "zoi"              # resolved against the registry at run time

    num_steps: int = 8000
    lr: float = 5e-3
    seed: int = 0
    ci_mass: float = 0.95
    squeeze_eps: float = 1e-4       # Beta endpoint squeeze (beta3 / beta4)

    def tag(self) -> dict:
        return {"model": self.model, **self.spec.tag()}