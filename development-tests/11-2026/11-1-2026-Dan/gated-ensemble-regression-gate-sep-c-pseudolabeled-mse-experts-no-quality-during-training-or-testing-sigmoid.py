"""Sigmoid regression gate with MSE-only experts and no density quality.

Each expert uses one five-fold validation-MSE pass to estimate its epoch count,
then refits on its full region. The gate still searches weights using AORE.
"""

import os

# Set TensorFlow's logging level before importing the experiment implementation.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from sep_c_experiment import run_experiment
from sep_c_settings import ExperimentSettings, OCTOBER_DIRECTORY


# Use October 7's pseudolabeled SEP-C files with their original train/test split.
# Their 22 inputs are already normalized; ln_peak_intensity is the target.
DATA_DIRECTORY = OCTOBER_DIRECTORY

# Train this new variant first. Later, True reloads its own November archive.
# An explicit directory can override that location for a matching SEP-C model.
LOAD_SAVED_MODEL = False
SAVED_MODEL_DIRECTORY = None
MODEL_RUN_NAME = "sep_c_pseudolabeled_sigmoid_regression_mse_experts_no_quality"


# Experts use plain, unweighted MSE: no PCC, sub-bin score, or alpha search.
# Their single five-fold pass selects epochs; the gate keeps its AORE search.
REGRESSION_GATE = True
EXPERT_CORRELATION_LOSS = False
MSE_ONLY_EXPERT_TRAINING = True

# Remove quality from mixing in training, validation, and testing. The gate
# still uses normalized sigmoid weights and searches sample importance weights.
DENSITY_AT_INFERENCE = False
DENSITY_DURING_GATE_TRAINING = False
SIGMOID_GATE = True


# Each expert performs five fold fits and one final fit. The gate evaluates
# every alpha below with five folds, then refits using the best mean AORE.
# Inference batches are separate from training batches and preserve loss batches.
MAXIMUM_EPOCHS = 500
EARLY_STOPPING_PATIENCE = 100
TRAINING_BATCH_SIZE = 32
FOLD_COUNT = 5

# These candidates control gate sample importance weights, not expert training.
GATE_ALPHA_CANDIDATES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0)

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
        mse_only_expert_training=MSE_ONLY_EXPERT_TRAINING,
        density_at_inference=DENSITY_AT_INFERENCE,
        density_during_gate_training=DENSITY_DURING_GATE_TRAINING,
        sigmoid_gate=SIGMOID_GATE,

        seed=seed,
        fold_count=FOLD_COUNT,
        alpha_candidates=GATE_ALPHA_CANDIDATES,
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

