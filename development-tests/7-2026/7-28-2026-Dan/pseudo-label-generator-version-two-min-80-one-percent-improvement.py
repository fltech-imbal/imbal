"""Iteratively generate SEP-C regression pseudo-labels.

Workflow
--------
1. Load the model trained by the balanced-fit-val-ae regression script.
2. Predict labels for the currently unlabeled samples.
3. Accept predictions inside [DESIRED_MIN, DESIRED_MAX].
4. Add newly accepted pseudo-labeled samples to the labeled training data.
5. Retrain with the same balanced-fit, validation, reciprocal-weight, and
   decoder-branch structure used by the balanced-fit-val-ae script.
6. Repeat until an iteration improves the accepted count by less than
   MIN_IMPROVEMENT_PERCENT of the original unknown-sample count, or until a
   safety stopping condition is reached.
7. Predict every original unknown sample with the final model, min-max scale
   all predictions into [DESIRED_MIN, DESIRED_MAX], and save one CSV column in
   the same row order as the input unknown-data CSV.

Edit only the Configuration section when file names or thresholds change.
"""

from __future__ import annotations

import json
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

import imbal
from aore_metric import AORE
from imbal.regression import reciprocal_importance


# =============================================================================
# Configuration
# =============================================================================
TARGET_COLUMN = "ln_peak_intensity"
SCRIPT_DIRECTORY = Path(__file__).resolve().parent

# Labeled data used to train the original balanced-fit-val-ae model.
LABELED_TRAIN_CSV_PATH = (
    SCRIPT_DIRECTORY / "sep_10mev_training_original.csv"
)

# Samples whose labels are unknown. An existing target column is ignored.
UNKNOWN_INPUT_CSV_PATH = (
    SCRIPT_DIRECTORY / "sep_10mev_testing_not_pseudo_labeled_yet.csv"
)

# Initial model and parameter file produced by balanced-fit-val-ae training.
INITIAL_MODEL_PATH = (
    SCRIPT_DIRECTORY / "saved_models/pseudo-label-generator-model.keras"
)
INITIAL_PARAMS_PATH = (
    SCRIPT_DIRECTORY
    / "saved_models/best_params_pseudo-label-generator-one-percent.json"
)

# Final artifacts produced by this script.
FINAL_MODEL_PATH = (
    SCRIPT_DIRECTORY / "saved_models/pseudo-label-version-two-one-percent-model.keras"
)
FINAL_PARAMS_PATH = (
    SCRIPT_DIRECTORY / "saved_models/pseudo-label-version-two-one-percent-params.json"
)
OUTPUT_CSV_PATH = (
    SCRIPT_DIRECTORY / "iterative_one_percent_predicted_ln_peak_intensity.csv"
)

# A prediction is accepted as a pseudo-label only when it is in this interval.
# These are inclusive bounds.
DESIRED_MIN = -1.60944
DESIRED_MAX = -0.68278

# Predictions above ln(10) are counted as false positives.
FP_THRESHOLD = np.log(10.0)

# Stop when a single iteration adds fewer than this percentage of the original
# unknown-sample count. Example: 1.0 means fewer than 1% of the original count.
MIN_IMPROVEMENT_PERCENT = 0.1

SEED = 42
MAX_EPOCHS = 500
BATCH_SIZE = 32
PATIENCE = 50
VALIDATION_SIZE = 0.20
ALPHA_CANDIDATES = [0.2, 0.5, 0.8, 0.9, 1.0, 1.1]

# Safety controls. The loop also stops when an iteration accepts no new rows.
MAX_ITERATIONS = 25
MIN_NEW_SAMPLES_PER_ITERATION = 1


# =============================================================================
# Validation and loading helpers
# =============================================================================
def validate_configuration() -> None:
    if DESIRED_MIN >= DESIRED_MAX:
        raise ValueError("DESIRED_MIN must be smaller than DESIRED_MAX.")

    if not 0.0 < MIN_IMPROVEMENT_PERCENT <= 100.0:
        raise ValueError(
            "MIN_IMPROVEMENT_PERCENT must be greater than 0 and at most 100."
        )

    if not 0.0 < VALIDATION_SIZE < 1.0:
        raise ValueError("VALIDATION_SIZE must be between 0 and 1.")

    if MAX_ITERATIONS < 1:
        raise ValueError("MAX_ITERATIONS must be at least 1.")


def load_training_data(csv_path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Labeled training CSV not found: {csv_path}")

    data = pd.read_csv(csv_path)
    if data.empty:
        raise ValueError(f"Labeled training CSV contains no rows: {csv_path}")
    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"Labeled training CSV is missing target column '{TARGET_COLUMN}'."
        )

    feature_frame = data.drop(columns=[TARGET_COLUMN])
    x_train = dataframe_to_finite_float32(feature_frame, "labeled training features")
    y_train = data[TARGET_COLUMN].to_numpy(dtype=np.float32).reshape(-1, 1)

    if not np.isfinite(y_train).all():
        raise ValueError("The labeled training targets contain NaN or infinity.")

    return feature_frame, x_train, y_train


def load_unknown_data(
    csv_path: Path,
    expected_feature_columns: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Unknown-data CSV not found: {csv_path}")

    data = pd.read_csv(csv_path)
    if data.empty:
        raise ValueError(f"Unknown-data CSV contains no rows: {csv_path}")

    feature_frame = data.drop(columns=[TARGET_COLUMN], errors="ignore")

    missing = [column for column in expected_feature_columns if column not in feature_frame]
    extra = [column for column in feature_frame if column not in expected_feature_columns]
    if missing or extra:
        raise ValueError(
            "Unknown-data feature columns do not match the labeled training data. "
            f"Missing columns: {missing or 'none'}; extra columns: {extra or 'none'}."
        )

    # Explicitly restore the exact training-column order.
    feature_frame = feature_frame[expected_feature_columns]
    x_unknown = dataframe_to_finite_float32(feature_frame, "unknown features")
    return feature_frame, x_unknown


def dataframe_to_finite_float32(frame: pd.DataFrame, description: str) -> np.ndarray:
    try:
        values = frame.to_numpy(dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"All {description} must be numeric.") from error

    if not np.isfinite(values).all():
        raise ValueError(f"The {description} contain NaN or infinite values.")
    return values


def load_initial_parameters(params_path: Path) -> dict:
    if not params_path.is_file():
        print(f"Parameter file not found; continuing without it: {params_path}")
        return {}

    with params_path.open("r", encoding="utf-8") as file:
        parameters = json.load(file)

    print(f"Loaded initial parameters from: {params_path}")
    if "best_alpha_index" in parameters:
        print(f"Initial best alpha index: {parameters['best_alpha_index']}")
    if "best_alpha" in parameters:
        print(f"Initial best alpha: {parameters['best_alpha']}")
    return parameters


def load_initial_model(model_path: Path, expected_features: int) -> imbal.regression.Model:
    if not model_path.is_file():
        raise FileNotFoundError(f"Initial saved model not found: {model_path}")

    print(f"Loading initial balanced-fit-val-ae model from: {model_path}")
    model = keras.models.load_model(
        model_path,
        custom_objects={
            "Model": imbal.regression.Model,
            "AORE": AORE,
        },
        compile=False,
    )

    model_features = model.input_shape[-1]
    if model_features is not None and int(model_features) != expected_features:
        raise ValueError(
            f"The model expects {model_features} features, but the training CSV "
            f"contains {expected_features}."
        )

    return model


# =============================================================================
# Training and prediction
# =============================================================================
def compile_for_balanced_training(model: imbal.regression.Model) -> None:
    """Recompile a loaded model whose decoder branch already exists.

    The initial model was saved after ``generate_decoder_branch=True`` had
    already generated its autoencoder decoder. Asking imbal to generate the
    branch again makes it traverse decoder layers as if they were encoder
    layers, which can produce incompatible inverse Reshape operations.
    """
    model.compile(
        loss="mean_squared_error",
        weighted_metrics=[AORE(threshold=np.log(10)), "mae"],
        optimizer="adam",
        generate_decoder_branch=False,
    )


def retrain_balanced_fit_val_ae(
    model: imbal.regression.Model,
    x_all: np.ndarray,
    y_all: np.ndarray,
    iteration_seed: int,
) -> tuple[imbal.regression.Model, int, float]:
    """Retrain one iteration using the original script's balanced-fit logic."""
    tf.keras.utils.set_random_seed(iteration_seed)

    labels_kde = y_all.reshape(-1).copy()
    kde = imbal.regression.fit_kde(labels_kde)
    densities = imbal.regression.get_sample_densities(labels_kde, kde)
    weight_candidates = reciprocal_importance(
        densities,
        alpha=ALPHA_CANDIDATES,
    )

    (x_fit, y_fit, sw_candidates), (x_val, y_val, sw_val) = imbal.regression.split(
        x_all,
        y_all,
        sample_weights=weight_candidates,
        test_size=VALIDATION_SIZE,
        seed=iteration_seed,
    )

    compile_for_balanced_training(model)
    candidate_evaluation_weights = np.ones(len(y_val), dtype=np.float32)

    model.balanced_fit(
        x_fit,
        y_fit,
        validation_data=(x_val, y_val.reshape(-1, 1), sw_val),
        candidate_evaluation_sample_weight=candidate_evaluation_weights,
        sample_weight=sw_candidates,
        batch_size=BATCH_SIZE,
        epochs=MAX_EPOCHS,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=PATIENCE,
                restore_best_weights=True,
            )
        ],
    )

    best_alpha_index = int(model.best_weight_index)
    best_alpha = float(ALPHA_CANDIDATES[best_alpha_index])
    return model, best_alpha_index, best_alpha


def predict_flat(model: keras.Model, x_data: np.ndarray) -> np.ndarray:
    predictions = model.predict(
        x_data,
        batch_size=BATCH_SIZE,
        verbose=1,
    )
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if not np.isfinite(predictions).all():
        raise ValueError("The model produced NaN or infinite predictions.")
    return predictions


def acceptable_mask(predictions: np.ndarray) -> np.ndarray:
    return (predictions >= DESIRED_MIN) & (predictions <= DESIRED_MAX)


def print_false_positive_count(predictions: np.ndarray) -> int:
    """Print and return the number of predictions greater than ln(10)."""
    flattened_predictions = np.asarray(predictions).reshape(-1)
    false_positive_count = int(
        np.sum(flattened_predictions > FP_THRESHOLD)
    )

    print(
        f"False positives (> ln(10), {FP_THRESHOLD:.6f}): "
        f"{false_positive_count}"
    )
    return false_positive_count


def min_max_scale_to_desired_range(predictions: np.ndarray) -> np.ndarray:
    """Scale only the final prediction bounds that violate the desired range.

    The destination range is selected from four cases:
    1. Both bounds violated: [source_min, source_max] -> [DESIRED_MIN, DESIRED_MAX]
    2. Upper bound only:     [source_min, source_max] -> [source_min, DESIRED_MAX]
    3. Lower bound only:     [source_min, source_max] -> [DESIRED_MIN, source_max]
    4. Neither violated:     return the predictions unchanged.
    """
    source_min = float(np.min(predictions))
    source_max = float(np.max(predictions))

    lower_violated = source_min < DESIRED_MIN
    upper_violated = source_max > DESIRED_MAX

    if not lower_violated and not upper_violated:
        print(
            "Final raw prediction bounds are already within the desired range; "
            "no scaling was applied."
        )
        return predictions.astype(np.float64, copy=True)

    target_min = DESIRED_MIN if lower_violated else source_min
    target_max = DESIRED_MAX if upper_violated else source_max

    if lower_violated and upper_violated:
        case_description = "both lower and upper bounds violated"
    elif upper_violated:
        case_description = "only the upper bound violated"
    else:
        case_description = "only the lower bound violated"

    print(
        f"Scaling case: {case_description}. "
        f"Mapping [{source_min:.6f}, {source_max:.6f}] to "
        f"[{target_min:.6f}, {target_max:.6f}]."
    )

    if np.isclose(source_min, source_max):
        # This can only occur for a constant prediction vector outside the
        # desired interval. Clamp that constant to the violated boundary.
        clamped_value = float(np.clip(source_min, DESIRED_MIN, DESIRED_MAX))
        print(
            "All final raw predictions are equal; assigning every prediction "
            f"to {clamped_value:.6f}."
        )
        return np.full(predictions.shape, clamped_value, dtype=np.float64)

    normalized = (predictions - source_min) / (source_max - source_min)
    return target_min + normalized * (target_max - target_min)


# =============================================================================
# Output helpers
# =============================================================================
def save_predictions(predictions: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame({TARGET_COLUMN: predictions.reshape(-1)})
    output.to_csv(output_path, index=False)
    print(f"Saved {len(output)} scaled predictions to: {output_path}")


def save_run_parameters(parameters: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(parameters, file, indent=4)
    print(f"Saved iterative run parameters to: {output_path}")


# =============================================================================
# Main iterative pseudo-labeling procedure
# =============================================================================
def main() -> None:
    validate_configuration()
    tf.keras.utils.set_random_seed(SEED)

    training_feature_frame, original_x_train, original_y_train = load_training_data(
        LABELED_TRAIN_CSV_PATH
    )
    feature_columns = training_feature_frame.columns.tolist()
    _, original_x_unknown = load_unknown_data(
        UNKNOWN_INPUT_CSV_PATH,
        expected_feature_columns=feature_columns,
    )

    load_initial_parameters(INITIAL_PARAMS_PATH)
    model = load_initial_model(
        INITIAL_MODEL_PATH,
        expected_features=original_x_train.shape[1],
    )

    total_unknown = len(original_x_unknown)
    minimum_improvement_count = int(
        np.ceil(total_unknown * MIN_IMPROVEMENT_PERCENT / 100.0)
    )

    # Keep indices so final output remains in exactly the original input order.
    remaining_indices = np.arange(total_unknown, dtype=np.int64)
    accepted_indices: list[int] = []
    accepted_labels: list[float] = []

    x_augmented = original_x_train.copy()
    y_augmented = original_y_train.copy()

    last_best_alpha_index: int | None = None
    last_best_alpha: float | None = None
    completed_iterations = 0
    stopping_reason = "maximum iterations reached"

    print("\nIterative pseudo-labeling configuration")
    print(f"Original labeled samples: {len(original_x_train)}")
    print(f"Original unknown samples: {total_unknown}")
    print(
        f"Acceptable raw prediction range: [{DESIRED_MIN}, {DESIRED_MAX}]"
    )
    print(
        f"Minimum improvement required per iteration: "
        f"{minimum_improvement_count}/{total_unknown} "
        f"({MIN_IMPROVEMENT_PERCENT:.2f}% of original unknown samples)"
    )

    for iteration in range(1, MAX_ITERATIONS + 1):
        if len(remaining_indices) == 0:
            stopping_reason = "all unknown samples accepted"
            break

        completed_iterations = iteration
        print(f"\n{'=' * 72}")
        print(f"Iteration {iteration}")
        print(f"Predicting {len(remaining_indices)} currently unaccepted samples...")

        remaining_predictions = predict_flat(
            model,
            original_x_unknown[remaining_indices],
        )
        print_false_positive_count(remaining_predictions)
        new_mask = acceptable_mask(remaining_predictions)
        new_indices = remaining_indices[new_mask]
        new_labels = remaining_predictions[new_mask]
        new_count = len(new_indices)

        print(f"New acceptable predictions: {new_count}")
        print(
            f"Cumulative accepted after this prediction: "
            f"{len(accepted_indices) + new_count}/{total_unknown} "
            f"({100.0 * (len(accepted_indices) + new_count) / total_unknown:.2f}%)"
        )

        if new_count < MIN_NEW_SAMPLES_PER_ITERATION:
            stopping_reason = (
                "no new acceptable pseudo-labels were found; "
                "further retraining cannot change the dataset"
            )
            break

        # Add each accepted row only once. Its current in-range prediction is
        # frozen as the pseudo-label used in subsequent training iterations.
        accepted_indices.extend(new_indices.tolist())
        accepted_labels.extend(new_labels.tolist())
        x_augmented = np.concatenate(
            [x_augmented, original_x_unknown[new_indices]],
            axis=0,
        )
        y_augmented = np.concatenate(
            [y_augmented, new_labels.astype(np.float32).reshape(-1, 1)],
            axis=0,
        )

        # Remove newly accepted samples from the pool before the next cycle.
        remaining_indices = remaining_indices[~new_mask]

        # Improvement is measured against the original unknown-sample count,
        # not against the number of samples still remaining. The newly accepted
        # rows are retained, but no further retraining is performed once the
        # improvement falls below the configured percentage.
        # Do not apply the improvement stopping criterion until at least
        # 80% of the original unknown samples have been accepted.
        current_acceptance_percent = (
            100.0 * len(accepted_indices) / total_unknown
        )

        if (
            current_acceptance_percent >= 80.0
            and new_count < minimum_improvement_count
        ):
            stopping_reason = (
                f"accepted at least 80% of the unknown samples and iteration "
                f"improvement was below {MIN_IMPROVEMENT_PERCENT:.2f}% "
                "of the original unknown-sample count"
            )
            break

        print(
            f"Retraining on {len(x_augmented)} total labeled + pseudo-labeled rows..."
        )
        model, last_best_alpha_index, last_best_alpha = retrain_balanced_fit_val_ae(
            model,
            x_augmented,
            y_augmented,
            iteration_seed=SEED + iteration,
        )
        print(
            f"Iteration {iteration} selected alpha {last_best_alpha} "
            f"(index {last_best_alpha_index})."
        )


    accepted_count = len(accepted_indices)
    accepted_percent = 100.0 * accepted_count / total_unknown

    print(f"\n{'=' * 72}")
    print("Iterative pseudo-labeling finished")
    print(f"Stopping reason: {stopping_reason}")
    print(
        f"Accepted pseudo-labels: {accepted_count}/{total_unknown} "
        f"({accepted_percent:.2f}%)"
    )

    print("\nPredicting all original unknown rows with the final model...")
    final_raw_predictions = predict_flat(model, original_x_unknown)
    final_scaled_predictions = min_max_scale_to_desired_range(
        final_raw_predictions
    )

    print(
        "Final raw prediction range: "
        f"[{final_raw_predictions.min():.6f}, "
        f"{final_raw_predictions.max():.6f}]"
    )
    print(
        "Final scaled prediction range: "
        f"[{final_scaled_predictions.min():.6f}, "
        f"{final_scaled_predictions.max():.6f}]"
    )

    save_predictions(final_scaled_predictions, OUTPUT_CSV_PATH)

    FINAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(FINAL_MODEL_PATH)
    print(f"Saved final retrained model to: {FINAL_MODEL_PATH}")

    save_run_parameters(
        {
            "seed": SEED,
            "desired_min": DESIRED_MIN,
            "desired_max": DESIRED_MAX,
            "minimum_improvement_percent": MIN_IMPROVEMENT_PERCENT,
            "minimum_improvement_count": minimum_improvement_count,
            "actual_accepted_count": accepted_count,
            "actual_accepted_percent": accepted_percent,
            "completed_iterations": completed_iterations,
            "stopping_reason": stopping_reason,
            "alpha_candidates": ALPHA_CANDIDATES,
            "last_best_alpha_index": last_best_alpha_index,
            "last_best_alpha": last_best_alpha,
            "original_labeled_count": int(len(original_x_train)),
            "original_unknown_count": int(total_unknown),
            "final_augmented_training_count": int(len(x_augmented)),
            "final_raw_prediction_min": float(final_raw_predictions.min()),
            "final_raw_prediction_max": float(final_raw_predictions.max()),
            "final_scaled_prediction_min": float(final_scaled_predictions.min()),
            "final_scaled_prediction_max": float(final_scaled_predictions.max()),
        },
        FINAL_PARAMS_PATH,
    )


if __name__ == "__main__":
    main()
