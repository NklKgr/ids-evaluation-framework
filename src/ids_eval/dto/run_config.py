from pydantic import BaseModel, Field, model_validator

from ids_eval.dto.data_manager_config import DataManagerConfig
from ids_eval.dto.evaluation_config import EvaluationConfig
from ids_eval.enumeration.split_method import SplitMethod


class GeneralConfig(BaseModel):
    name: str = Field(..., description="Name of this evaluation run")
    description: str = Field(..., description="Description of this evaluation run")
    seed: int = Field(..., description="Random seed for reproducibility")
    report_path: str = Field(default="out/reports/", description="Path for output reports")
    processed_data_path: str = Field(
        default="out/processed_datasets/", description="Path for storage processed datasets"
    )
    model_storage_path: str = Field(default="out/saved_models/", description="Base path for storing trained models")
    plugin_adversarial_path: str = Field(
        default="plugin_adversarial/", description="Path to adversarial attack plugins"
    )
    plugin_ids_path: str = Field(default="plugin_ids/", description="Path to IDS model plugins")
    plugin_static_metric_path: str = Field(
        default="plugin_static_metric/", description="Path to static metric calculation plugins"
    )
    plugin_runtime_metric_path: str = Field(
        default="plugin_runtime_metric/", description="Path to runtime metric collection plugins"
    )



class RunConfig(BaseModel):
    general: GeneralConfig
    data_manager: DataManagerConfig | None = None
    evaluation: EvaluationConfig | None = None
    # Internal field set by ConfigManager - hash of the entire config file
    _config_file_hash: str | None = None

    def set_config_file_hash(self, hash_value: str) -> None:
        """Sets the hash of the config file (called by ConfigManager after loading)."""
        object.__setattr__(self, "_config_file_hash", hash_value)

    def get_config_file_hash(self) -> str | None:
        """Returns the hash of the config file."""
        return getattr(self, "_config_file_hash", None)

    @model_validator(mode="after")
    def validate_feature_mask_dataset_names(self) -> "RunConfig":
        robustness = self.evaluation.robustness if self.evaluation else None
        if robustness is None or not robustness.feature_masks or self.data_manager is None:
            return self
        dataset_names = {dataset.name for dataset in self.data_manager.dataset}
        unknown = {feature_mask.d_name for feature_mask in robustness.feature_masks} - dataset_names
        if unknown:
            raise ValueError(f"robustness.feature_masks refers to unknown datasets {sorted(unknown)}. Configured datasets: {sorted(dataset_names)}")
        return self

    @model_validator(mode="after")
    def validate_model_saving_for_sweep(self) -> "RunConfig":
        robustness = self.evaluation.robustness if self.evaluation else None
        if robustness is None or not robustness.enabled:
            return self

        unsaved = [model.plugin for model in self.evaluation.anomaly_models or [] if not model.save_model]
        if unsaved:
            raise ValueError(f"The Sweep requires save_model: true for every model, but {unsaved} fail(s) to comply.")
        return self

    @model_validator(mode="after")
    def validate_binary_labels_for_sweep(self) -> "RunConfig":
        robustness = self.evaluation.robustness if self.evaluation else None
        if robustness is None or not robustness.enabled:
            return self

        if self.evaluation.general.is_multiclass:
            raise ValueError("The Sweep is defined for binary labels (benign vs. attack) only. Set evaluation.general.is_multiclass to false.")
        return self

    @model_validator(mode="after")
    def validate_split_method_for_sweep(self) -> "RunConfig":
        robustness = self.evaluation.robustness if self.evaluation else None
        if robustness is None or not robustness.enabled or self.data_manager is None:
            return self

        if self.data_manager.split.method == SplitMethod.KFOLDSPLIT:
            raise ValueError("The Sweep does not support k_fold_split. ")
        return self