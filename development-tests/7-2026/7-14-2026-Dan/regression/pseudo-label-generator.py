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

    save_predictions(predictions, OUTPUT_CSV_PATH)


if __name__ == "__main__":
    main()
