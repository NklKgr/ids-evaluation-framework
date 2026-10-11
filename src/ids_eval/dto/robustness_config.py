import math
import numpy as np

from pydantic import BaseModel, ConfigDict, Field, model_validator
from ids_eval.dto.adversarial_config import AdversarialAttackPluginConfig

BINARY_LABELS = frozenset({0, 1})


class EvalSampleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_attack: int = Field(..., gt=0, description="Number of attack-class test rows the sweep is performed on")
    n_benign: int | None = Field(
        default=None,
        gt=0,
        description="Number of benign test rows; None uses all benign test rows",
    )

class SweepFeatureMask(BaseModel):
    # Later on this can be extended to more sophisticated masks enforcing realistic network constraints,
    # e.g. the direction of a change, caps and arithmetic relations between features [Sheatsley2022!].
    model_config = ConfigDict(extra="forbid")

    d_name: str = Field(..., min_length=1, description="Name of the dataset the mask will be applied to")
    perturbable: list[str] = Field(
        ...,
        min_length=1,
        description="Names of the features an attack may change; all other features stay untouched. "
        "Names must match the dataset columns exactly (case sensitive)!",
    )


class RobustnessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=False,
        description="Enable robustness sweep evaluation",
    )

    eps_grid: list[float] | None = Field(
        default=None,
        description="Equidistant perturbation magnitudes, must be > 0, used for the robustness sweep. eps=0 will be evaluated automatically."
                    "Mandatory if the sweep is enabled.",
    )

    perturb_classes: list[int] = Field(
        default_factory=lambda: [1],  # attack class
        description="Label values (0 = benign, 1 = attack). Rows are perturbed by the attack sweeps.",
    )

    eval_sample: EvalSampleConfig | None = Field(
        default=None,
        description="Number of seeded attack samples (rows) and number of benign samples. Mandatory if the sweep is enabled.",
    )

    attacks: list[AdversarialAttackPluginConfig] = Field(
        default_factory=list,
        description="List of attacks to be performed. Non-empty List is mandatory if the sweep is enabled.",
    )

    feature_masks: list[SweepFeatureMask] = Field(
        default_factory=list,
        description="Per dataset, the features an attack may perturb (-> True). Datasets without a mask are fully perturbable.",
    )

    store_adversarial_examples: bool = Field(
        default=False,
        description="Enables persisting the created adversarial examples."
    )


    @model_validator(mode="after")
    def validate_perturb_classes(self) -> "RobustnessConfig":
        classes = self.perturb_classes
        # set checking of doups and subset(<=) of {0,1}
        if not classes or not set(classes) <= BINARY_LABELS:
            raise ValueError(
                f"'perturb_classes' must be a non-empty subset of {{0, 1}}, got {classes}.")
        return self

    @model_validator(mode="after")
    def validate_feature_masks(self) -> "RobustnessConfig":
        names = [feature_mask.d_name for feature_mask in self.feature_masks]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"'feature_masks' tries to set more than one mask for the dataset(s) {duplicates}.")
        return self

    @model_validator(mode="after")
    def validate_eps_grid(self) -> "RobustnessConfig":
        if self.eps_grid is None:
            return self

        grid = self.eps_grid
        if not grid:
            raise ValueError("'eps_grid' must not be empty for the sweep.")
        if any(eps <= 0 for eps in grid):
            raise ValueError(
                "'eps_grid' values must all be positive and >0."
                "baseline at ε=0 is added by the sweep runner automatically."
            )
        if grid != sorted(set(grid)):
            raise ValueError("'eps_grid' must be strictly increasing (sorted, no duplicates).")

        step = grid[0]
        expected = [step * k for k in range(1, len(grid) + 1)]
        if not np.allclose(grid, expected, rtol=1e-6, atol=1e-9):
            raise ValueError(
                f"'eps_grid' must be equidistant steps: multiples of {step}: "
                f"expected {[round(eps, 6) for eps in expected]}, got {grid}."
            )
        return self

    @model_validator(mode="after")
    def validate_enabled_sweep(self) -> "RobustnessConfig":
        if not self.enabled:
            return self

        required = {"eps_grid": self.eps_grid, "eval_sample": self.eval_sample, "attacks": self.attacks}
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"The robustness sweep is enabled, you need to provide {missing}.")

        if 0 in self.perturb_classes and self.eval_sample.n_benign is None:
            raise ValueError(
                "'eval_sample.n_benign' must be set when 'perturb_classes' contains 0, or every benign test row would be attacked."
            )
        return self