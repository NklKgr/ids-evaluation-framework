"""Checks that adversarial examples respect the budget and the feature mask, without repairing them (plan 3.4)."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class PerturbationReport:
    linf: np.ndarray               # float32, largest change per row
    l0: np.ndarray                 # int16, number of changed features per row
    budget_violations: np.ndarray  # bool per row: |δ| > ε somewhere
    mask_violations: np.ndarray    # bool per row: a locked feature changed

    @property
    def is_valid(self) -> bool:
        return not (self.budget_violations.any() or self.mask_violations.any())


def check_perturbation(
    x: np.ndarray,
    x_adv: np.ndarray,
    eps: float,
    perturbable: np.ndarray,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> PerturbationReport:

    #ensure same format
    x = np.asarray(x, dtype=np.float64)
    x_adv = np.asarray(x_adv, dtype=np.float64)
    perturbable = np.asarray(perturbable)

    #ensure same shapes
    if x.ndim != 2 or x.shape != x_adv.shape or perturbable.shape != (x.shape[1],):
        raise ValueError( f"Shapes do not match: x {x.shape}, x_adv {x_adv.shape}, perturbable {perturbable.shape}; ")
    if perturbable.dtype != bool:
        raise ValueError(f"A bool array is needed, but got dtype {perturbable.dtype}.")

    #only changes bigger than machine error count
    changed_idxs = ~np.isclose(x_adv, x, rtol=rtol, atol=atol, equal_nan=True)

    #get changemagnitudes
    abs_delta                       = np.abs(x_adv - x)
    abs_delta[~changed_idxs]        = 0.0 #no change
    abs_delta[np.isnan(abs_delta)]  = np.inf #violation

    linf = abs_delta.max(axis=1)
    l0 = changed_idxs.sum(axis=1)
    budget_violations = linf > eps + atol + rtol * eps
    mask_violations = (changed_idxs & ~perturbable).any(axis=1)

    return PerturbationReport(
        linf=linf.astype(np.float32),
        l0=l0.astype(np.int16),
        budget_violations=budget_violations,
        mask_violations=mask_violations,
    )
