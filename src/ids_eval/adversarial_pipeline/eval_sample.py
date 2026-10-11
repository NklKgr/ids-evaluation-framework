from typing import Self
import numpy as np
import pandas as pd
from dataclasses import dataclass

from ids_eval.dto.robustness_config import BINARY_LABELS
from ids_eval.dto.robustness_config import EvalSampleConfig

@dataclass(frozen=True, eq=False)
class EvalSample:
    positions_in_ds: np.ndarray         #drawn positoins in ds
    y: np.ndarray                       #int8 (label)
    attackable: np.ndarray
    non_sample: np.ndarray
    seed: int

def draw_eval_sample(y_test: pd.Series, sample_config: EvalSampleConfig, perturb_classes: list[int],
                    seed: int, dataset_name: str) -> EvalSample:
    "Here we draw the samples for the sweep"

    labels = y_test.to_numpy()

    #check for all binary only labels
    present = set(np.unique(labels).tolist())
    invalid = present - BINARY_LABELS
    if invalid:
        raise ValueError(f"Only {{0,1}} are allowed, but got {invalid}")

    missing = BINARY_LABELS - present
    if missing:
        raise ValueError(f"Label {missing} was not found, but is needed for teh sweep.")

    #new random generator per set
    rng = np.random.default_rng(seed) #seed from config
    # indices of attacks and benign from dataset
    attack_drawn = _draw(rng, np.flatnonzero(labels == 1), sample_config.n_attack, "attack")
    benign_drawn = _draw(rng, np.flatnonzero(labels == 0), sample_config.n_benign, "benign")

    positions_in_ds = np.sort(np.concatenate([attack_drawn, benign_drawn]))
    non_samples = np.setdiff1d(np.arange(len(labels)), positions_in_ds) #surrogate pool (everyth. thats not in the eval sample)

    #get the labels for drawn samples and compare to mask
    y = labels[positions_in_ds].astype(np.int8)
    attackable = np.isin(y, perturb_classes)

    #immuatable
    for array in (positions_in_ds, non_samples, y, attackable):
        array.setflags(write=False)

    return EvalSample(positions_in_ds=positions_in_ds, y=y, attackable=attackable, non_samples=non_samples, seed=seed)


def _draw(rng: np.random.Generator, candidates: np.ndarray, n: int | None, class_name: str)-> np.ndarray:
    if n is None:
        return candidates
    if n > len(candidates):
        raise ValueError(f"n is {n}, but there are only {len(candidates)} candidates!")
    #drwa uniform without double entries
    return rng.choice(candidates, size=n, replace=False)