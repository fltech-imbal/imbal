"""Generate SEP-C regression pseudo-labels from a saved Keras model.

Set the file paths in the configuration section below, then run:

    python generate-pseudo-labels-no-arguments.py

The input CSV must contain one row per sample and the same feature columns,
in the same order, that were used to train the model. If an existing
``ln_peak_intensity`` column is present, it is ignored.
"""

from __future__ import annotations

import json
from pathlib import Path

import keras
import numpy as np
import pandas as pd

import imbal


# ----------------------------
# Configuration
# ----------------------------
TARGET_COLUMN = "ln_peak_intensity"
SCRIPT_DIRECTORY = Path(__file__).resolve().parent

INPUT_CSV_PATH = SCRIPT_DIRECTORY / "sep_10mev_testing_not_pseudo_labeled_yet.csv"
MODEL_PATH = SCRIPT_DIRECTORY / "saved_models/pseudo-label-generator-model.keras"
PARAMS_PATH = SCRIPT_DIRECTORY / "saved_models/best_params_pseudo-label-generator.json"
OUTPUT_CSV_PATH = SCRIPT_DIRECTORY / "predicted_ln_peak_intensity.csv"

BATCH_SIZE = 32

# Desired inclusive range for final pseudo-label predictions.
DESIRED_MIN = -1.60944
DESIRED_MAX = -0.68278

# Predictions greater than ln(10) are counted as false positives.
FP_THRESHOLD = np.log(10.0)


def load_parameters(params_path: Path) -> dict:
    if not params_path.is_file():
        raise FileNotFoundError(f"Parameter file not found: {params_path}")

    with params_path.open("r", encoding="utf-8") as file:
        parameters = json.load(file)

    print(f"Loaded parameters from: {params_path}")
    if "best_alpha_index" in parameters:
        print(f"Best alpha index: {parameters['best_alpha_index']}")
    if "best_alpha" in parameters:
        print(f"Best alpha: {parameters['best_alpha']}")

    return parameters


def load_prediction_model(model_path: Path) -> keras.Model:
    if not model_path.is_file():
        raise FileNotFoundError(f"Saved model not found: {model_path}")

    print(f"Loading saved regression model from: {model_path}")

    # compile=False avoids requiring the training loss and metrics for inference.
    model = keras.models.load_model(
        model_path,
        custom_objects={"Model": imbal.regression.Model},
        compile=False,
    )

    return model


def prepare_samples(input_csv_path: Path, model: keras.Model) -> np.ndarray:
    if not input_csv_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_csv_path}")

    samples = pd.read_csv(input_csv_path)
    if samples.empty:
        raise ValueError(f"Input CSV contains no sample rows: {input_csv_path}")

    # Accept either an unlabeled feature CSV or a labeled SEP-C CSV without
    # accidentally passing the target column into the model.
    feature_data = samples.drop(columns=[TARGET_COLUMN], errors="ignore")

    expected_features = model.input_shape[-1]
    actual_features = feature_data.shape[1]
    if expected_features is not None and actual_features != expected_features:
        raise ValueError(
            "Incorrect number of input features. "
            f"The model expects {expected_features}, but the CSV provides "
            f"{actual_features} after removing '{TARGET_COLUMN}' if present. "
            "Make sure the CSV has the same feature columns in the same order "
            "as the training data."
        )

    try:
        x_samples = feature_data.to_numpy(dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "All input feature columns must contain numeric values."
        ) from error

    if not np.isfinite(x_samples).all():
        raise ValueError("The input features contain NaN or infinite values.")

    print(f"Loaded {len(x_samples)} samples with {actual_features} features each.")
    return x_samples


def print_false_positive_count(predictions: np.ndarray) -> None:
    """Count predictions greater than ln(10) as false positives."""
    flattened_predictions = np.asarray(predictions).reshape(-1)

    false_positive_count = int(
        np.sum(flattened_predictions > FP_THRESHOLD)
    )

    print("\nPrediction count:")
    print(f"FP (> ln(10)): {false_positive_count}")


def min_max_scale_to_desired_range(predictions: np.ndarray) -> np.ndarray:
    """Scale only prediction bounds that violate the desired range.

    Four cases are handled:
    1. Both bounds violated: scale to [DESIRED_MIN, DESIRED_MAX].
    2. Upper bound only: preserve the raw minimum and scale the maximum down.
    3. Lower bound only: scale the minimum up and preserve the raw maximum.
    4. Neither bound violated: return the predictions unchanged.
    """
    flattened = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if not np.isfinite(flattened).all():
        raise ValueError("The model produced NaN or infinite predictions.")

    source_min = float(np.min(flattened))
    source_max = float(np.max(flattened))

    lower_violated = source_min < DESIRED_MIN
    upper_violated = source_max > DESIRED_MAX

    if not lower_violated and not upper_violated:
        print(
            "Prediction bounds are already within the desired range; "
            "no scaling was applied."
        )
        return flattened.copy()

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
        clamped_value = float(np.clip(source_min, DESIRED_MIN, DESIRED_MAX))
        print(
            "All raw predictions are equal; assigning every prediction "
            f"to {clamped_value:.6f}."
        )
        return np.full(flattened.shape, clamped_value, dtype=np.float64)

    normalized = (flattened - source_min) / (source_max - source_min)
    return target_min + normalized * (target_max - target_min)


def save_predictions(predictions: np.ndarray, output_path: Path) -> None:
    flattened_predictions = np.asarray(predictions).reshape(-1)
    output_data = pd.DataFrame({TARGET_COLUMN: flattened_predictions})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data.to_csv(output_path, index=False)

    print(f"Saved {len(output_data)} predictions to: {output_path}")


def main() -> None:
    # These parameters describe the selected training configuration. The alpha
    # value is loaded for reproducibility but is not needed during prediction.
    load_parameters(PARAMS_PATH)
    model = load_prediction_model(MODEL_PATH)
    x_samples = prepare_samples(INPUT_CSV_PATH, model)

    predictions = model.predict(
        x_samples,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    print_false_positive_count(predictions)

    raw_predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    scaled_predictions = min_max_scale_to_desired_range(raw_predictions)

    print(
        f"Raw prediction range: [{raw_predictions.min():.6f}, "
        f"{raw_predictions.max():.6f}]"
    )
    print(
        f"Final prediction range: [{scaled_predictions.min():.6f}, "
        f"{scaled_predictions.max():.6f}]"
    )

    save_predictions(scaled_predictions, OUTPUT_CSV_PATH)


if __name__ == "__main__":
    main()
