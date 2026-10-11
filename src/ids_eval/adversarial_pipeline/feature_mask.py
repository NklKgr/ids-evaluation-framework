"""Resolves the configured feature mask of a dataset to the boolean array the attacks perturb by."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ids_eval.dto.robustness_config import SweepFeatureMask

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class FeatureMask:
    """Which dataset features an attack may change (column order of X_test)."""

    names: tuple[str, ...]
    perturbable: np.ndarray  # dtype=bool und shape (len(names),) -> passed to ART as `mask`

    @property
    def locked(self) -> tuple[str, ...]:
        return tuple(name for name, free in zip(self.names, self.perturbable) if not free)


def resolve_feature_mask(feature_names: Sequence[str], mask_config: SweepFeatureMask | None) -> FeatureMask:
    names = tuple(feature_names)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"The dataset has duplicate feature names {duplicates}, a mask by name would be ambiguous.")

    #no mask provided-> everything is perturbable 
    if mask_config is None:
        perturbable = np.ones(len(names), dtype=bool)
    else:
        missing = set(mask_config.perturbable) - set(names)
        if missing:
            raise ValueError(
                f"The feature mask of '{mask_config.d_name}' names features that are not in the dataset: "
                f"{sorted(missing)}. Available features: {list(names)}"
            )
        perturbable = np.isin(names, mask_config.perturbable)
        logger.info(
            f"Feature mask of '{mask_config.d_name}' allows {int(perturbable.sum())} of {len(names)} features "
            "to be perturbed."
        )

    if not perturbable.any():
        raise ValueError("The feature mask leaves no feature an attack could perturb.")

    perturbable.setflags(write=False)
    return FeatureMask(names=names, perturbable=perturbable)
