"""Visible controls for the November 8 pseudolabeled SEP-C experiment."""

from dataclasses import dataclass
from pathlib import Path


EXPERIMENT_DIRECTORY = Path(__file__).resolve().parent
OCTOBER_DIRECTORY = EXPERIMENT_DIRECTORY.parents[1] / "10-2026" / "10-7-2026-Dan"


@dataclass(frozen=True)
class ExperimentSettings:
    run_name: str
    experiment_directory: Path
    data_directory: Path

    seeds: tuple = (42, 43, 44, 45, 46)
    fold_count: int = 5
    maximum_epochs: int = 500
    early_stopping_patience: int = 100
    training_batch_size: int = 32
    prediction_batch_size: int = 128
    learning_rate: float = 0.001

    maximum_experts: int = 5
    minimum_relative_improvement: float = 0.01
    router_threshold: float = 0.5
    balance_router_classes: bool = False

    saved_model_directory: Path | None = None
    load_saved_model: bool = False
    show_visualizations: bool = False
    expert_mode: str = "direct"

    @property
    def residual(self):
        return self.expert_mode != "direct"

    def __post_init__(self):
        if self.expert_mode not in ("direct", "residual_ones", "residual_weighted"):
            raise ValueError(f"Unknown expert mode: {self.expert_mode}")

        if self.fold_count < 2 or not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Use at least two folds and distinct experiment seeds.")

        positive_settings = (
            self.maximum_epochs, self.training_batch_size, self.prediction_batch_size,
            self.maximum_experts, self.learning_rate,
        )
        if min(positive_settings) <= 0 or self.early_stopping_patience < 0:
            raise ValueError("Training sizes and learning rate must be positive, patience >= 0.")

        if not 0 < self.minimum_relative_improvement <= 1 or not 0 <= self.router_threshold <= 1:
            raise ValueError("Check the relative improvement and router probability thresholds.")

    @property
    def model_directory(self):
        if self.saved_model_directory is not None:
            return Path(self.saved_model_directory)

        return self.experiment_directory / "saved_models" / self.run_name

    @property
    def result_path(self):
        return self.experiment_directory / "saved_results" / f"{self.run_name}.json"
