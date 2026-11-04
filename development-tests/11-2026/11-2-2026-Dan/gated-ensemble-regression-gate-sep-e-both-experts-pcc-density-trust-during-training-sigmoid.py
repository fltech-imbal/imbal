"""Sigmoid regression gate with density during training and inference.

November 2 version of the matching October 16 experiment. Shared modules keep
the memory controls consistent, while the method choices remain visible here.
"""

import os

# Set TensorFlow's logging level before importing the experiment implementation.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from sep_e_experiment import run_experiment
from sep_e_settings import ExperimentSettings, OCTOBER_DIRECTORY


# Keep the existing CSVs as the data source. They are already log-normalized,
# and the original training/validation splits are combined for k-fold selection.
DATA_DIRECTORY = OCTOBER_DIRECTORY

LOAD_SAVED_MODEL = False
SAVED_MODEL_DIRECTORY = None

# None saves/loads this run beneath the November folder. To reuse October's
# matching archive, set this to OCTOBER_DIRECTORY / "saved_models" / MODEL_RUN_NAME.
MODEL_RUN_NAME = "sep_e_regression_gate_both_experts_pcc_sigmoid_density_trust_training"


# These are the scientific choices defining this particular ensemble method.
# The cache never becomes a new input feature for the gate itself.
REGRESSION_GATE = True
EXPERT_CORRELATION_LOSS = True
DENSITY_AT_INFERENCE = True
DENSITY_DURING_GATE_TRAINING = True
SIGMOID_GATE = True


# Preserve October's search effort and training schedule. Bounded inference
# batches are separate from training batches and do not change the loss batches.
MAXIMUM_EPOCHS = 500
EARLY_STOPPING_PATIENCE = 100
TRAINING_BATCH_SIZE = 32
FOLD_COUNT = 5

PREDICTION_BATCH_SIZE = 128
DENSITY_QUERY_CHUNK_SIZE = 128
DENSITY_REFERENCE_CHUNK_SIZE = 4096


# Enable expensive plots and long diagnostic tables only when investigating a
# particular run. Every training/test metric still evaluates the complete split.
SHOW_VISUALIZATIONS = False
VISUALIZATION_SAMPLE_LIMIT = 5000
PRINT_INSTANCE_TABLES = False


def create_settings(seed=42):
    """Collect this file's visible choices into the shared training settings."""

    return ExperimentSettings(
        run_name=MODEL_RUN_NAME,
        regression_gate=REGRESSION_GATE,
        expert_correlation_loss=EXPERT_CORRELATION_LOSS,
        density_at_inference=DENSITY_AT_INFERENCE,
        density_during_gate_training=DENSITY_DURING_GATE_TRAINING,
        sigmoid_gate=SIGMOID_GATE,

        seed=seed,
        fold_count=FOLD_COUNT,
        maximum_epochs=MAXIMUM_EPOCHS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        training_batch_size=TRAINING_BATCH_SIZE,

        prediction_batch_size=PREDICTION_BATCH_SIZE,
        density_query_chunk_size=DENSITY_QUERY_CHUNK_SIZE,
        density_reference_chunk_size=DENSITY_REFERENCE_CHUNK_SIZE,

        data_directory=DATA_DIRECTORY,
        saved_model_directory=SAVED_MODEL_DIRECTORY,
        load_saved_model=LOAD_SAVED_MODEL,

        show_visualizations=SHOW_VISUALIZATIONS,
        visualization_sample_limit=VISUALIZATION_SAMPLE_LIMIT,
        print_instance_tables=PRINT_INSTANCE_TABLES,
    )


def run_two_expert_gated_ensemble(seed=42):
    """Run this variant, returning the ensemble, common expert, rare expert, gate."""

    settings = create_settings(seed)
    return run_experiment(settings)


if __name__ == "__main__":
    run_two_expert_gated_ensemble(seed=42)

