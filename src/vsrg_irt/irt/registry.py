"""Decorator-based model registry."""

from __future__ import annotations

from .base import IRTModel

_REGISTRY: dict[str, IRTModel] = {}


def register(cls: type[IRTModel]) -> type[IRTModel]:
    inst = cls()
    if not inst.name:
        raise ValueError(f"{cls.__name__} must set a non-empty `name`")
    if inst.name in _REGISTRY:
        raise ValueError(f"duplicated model name: {inst.name!r}")
    _REGISTRY[inst.name] = inst
    return cls


def get_model(name: str) -> IRTModel:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(_REGISTRY)}")


def all_models() -> list[IRTModel]:
    return list(_REGISTRY.values())


def valid_combos(responses=("score", "acc")) -> set[tuple[str, str]]:
    """All (model, response) pairs the registered models declare valid."""
    return {
        (m.name, r)
        for m in _REGISTRY.values()
        for r in responses
        if r in m.valid_response
    }
