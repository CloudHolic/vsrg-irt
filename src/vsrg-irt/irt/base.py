"""The IRTModel contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpyro.infer.autoguide import AutoNormal

if TYPE_CHECKING:
    from ..data import Dataset
    from ..specs import FitConfig


class IRTModel(ABC):
    # identity / metadata
    name: str = ""
    valid_response: frozenset[str] = frozenset({"score"})
    difficulty_space: str = "unit"      # "unit" (delta in (0, 1)) | "real" (theta* on R)

    # data wiring
    def make_data(self, dataset: "Dataset") -> dict:
        return dataset.as_model_dict()

    # model + guide
    @abstractmethod
    def model(self, data: dict, cfg: "FitConfig"): ...

    def make_guide(self, data: dict, cfg: "FitConfig"):
        return AutoNormal(self.model)

    # extraction
    @abstractmethod
    def extract_difficulty(self, samples) -> np.ndarray:
        """(n_draws, n_items); larger = harder."""

    def extract_ability(self, samples) -> np.ndarray:
        return np.asarray(samples["theta"])

    def item_extra(self, samples, data: dict) -> dict:
        return {}
