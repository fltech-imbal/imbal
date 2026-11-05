"""Explicit settings shared by the November 1 SEP-C experiments."""

from dataclasses import dataclass
from pathlib import Path


EXPERIMENT_DIRECTORY = Path(__file__).resolve().parent

# Reuse the latest October pseudolabeled training and testing CSV files.
# Each entry script can point data_directory at another verified dataset.
OCTOBER_DIRECTORY = (
    EXPERIMENT_DIRECTORY.parent.parent / "10-2026" / "10-7-2026-Dan"
)


@dataclass(frozen=True)
class ExperimentSettings:
    """Describe one experiment without hiding its scientific choices."""

    run_name: str

    # Classification gates learn region labels; regression gates learn through
    # the mixture's MSE while the two previously trained experts remain fixed.
    regression_gate: bool = False
    expert_correlation_loss: bool = False
    density_at_inference: bool = False
    density_during_gate_training: bool = False
    sigmoid_gate: bool = False

    # This experimental option gives each expert one ordinary KFold pass to
    # select its epoch count using validation MSE, without a sub-bin/alpha search.
    # The regression gate still performs its own full weight-candidate search.
    mse_only_expert_training: bool = False

    # Keep the October search sizes and training controls unchanged by default.
    seed: int = 42
    fold_count: int = 5
    maximum_epochs: int = 500
    early_stopping_patience: int = 100
    training_batch_size: int = 32
    learning_rate: float = 0.001
    correlation_weight: float = 2.0

    alpha_candidates: tuple = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0)
    rare_class_weight_candidates: tuple = tuple(float(value) for value in range(1, 11))
    subbin_balance_tolerance: float = 0.10

    # Bound both dimensions of the KDE workspace, including direct model calls.
    # A float32 kernel block is at most 128 * 4096 * 4 bytes, or 2 MiB.
    prediction_batch_size: int = 128
    density_query_chunk_size: int = 128
    density_reference_chunk_size: int = 4096

    data_directory: Path = OCTOBER_DIRECTORY
    saved_model_directory: Path | None = None
    load_saved_model: bool = False

    # Plotting is deliberately separate from training and numerical evaluation.
    # Set the limit to None to visualize every row; metrics always use every row.
    show_visualizations: bool = False
    visualization_sample_limit: int | None = 5000
    print_instance_tables: bool = False

    def __post_init__(self):
        positive_settings = (
            self.fold_count,
            self.maximum_epochs,
            self.early_stopping_patience,
            self.training_batch_size,
            self.prediction_batch_size,
            self.density_query_chunk_size,
            self.density_reference_chunk_size,
        )

        if any(value < 1 for value in positive_settings) or self.fold_count < 2:
            raise ValueError("Batch sizes, chunks, and epochs must be positive; folds >= 2.")

        # Reject combinations that would silently describe a different experiment.
        if self.density_during_gate_training and not (
            self.regression_gate and self.density_at_inference
        ):
            raise ValueError("Training-time density requires a regression density gate.")

        if self.sigmoid_gate and not self.regression_gate:
            raise ValueError("The classification experiments require softmax probabilities.")

        if self.mse_only_expert_training and self.expert_correlation_loss:
            raise ValueError("MSE-only expert training cannot also enable PCC.")

        if self.regression_gate and not (
            self.expert_correlation_loss or self.mse_only_expert_training
        ):
            raise ValueError("Choose PCC experts or the explicit MSE-only expert experiment.")

    @property
    def model_directory(self):
        if self.saved_model_directory is not None:
            return Path(self.saved_model_directory)

        return EXPERIMENT_DIRECTORY / "saved_models" / self.run_name
