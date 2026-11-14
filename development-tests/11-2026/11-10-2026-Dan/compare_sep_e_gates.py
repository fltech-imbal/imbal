"""Compare the classification and regression gates on SEP-E in PyCharm.

Run this file to train one shared pair of experts and both gates. Each run saves
the experts, gates, assembled ensembles, and test metrics in runs_sep_e/<timestamp>/.
Set LOAD_SAVED_MODEL to True to evaluate the latest saved SEP-E pair.
Set SHOW_VISUALIZATIONS to display and save a separate imbal test prediction
plot for each gate in the run folder.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from compare_sep_c_gates import run_gate_comparison


HERE = Path(__file__).resolve().parent
DATA_DIRECTORY = HERE.parents[1] / "10-2026" / "10-16-2026-Dan"
RUNS_DIRECTORY = HERE / "runs_sep_e"
LOAD_SAVED_MODEL = False
SHOW_VISUALIZATIONS = False
# Optionally choose a specific folder, e.g. RUNS_DIRECTORY / "<timestamp>".
SAVED_MODEL_DIRECTORY = None
TARGET = "p16.4_tplus6"
INPUT_COUNT = 156


def load_sep_e_dataset(data_directory):
    """Pool the log-normalized training and validation rows in their supplied order."""
    splits = {}
    expected_columns = None
    for split in ("training", "validation", "test"):
        path = data_directory / f"sep_e_log_normalized_{split}.csv"
        frame = pd.read_csv(path)
        if TARGET not in frame or len(frame.columns) != INPUT_COUNT + 1:
            raise ValueError(f"{path} must contain {INPUT_COUNT} features and target {TARGET}.")
        if expected_columns is not None and not frame.columns.equals(expected_columns):
            raise ValueError(f"SEP-E column order differs in {path}.")
        expected_columns = frame.columns
        targets = frame.pop(TARGET).to_numpy(dtype="float32").reshape(-1, 1)
        features = frame.to_numpy(dtype="float32")
        if not np.isfinite(features).all() or not np.isfinite(targets).all():
            raise ValueError(f"{path} contains nonfinite features or targets.")
        splits[split] = features, targets

    train_x, train_y = splits["training"]
    val_x, val_y = splits["validation"]
    test_x, test_y = splits["test"]
    return np.concatenate((train_x, val_x)), np.concatenate((train_y, val_y)), test_x, test_y


def main():
    run_gate_comparison(
        *load_sep_e_dataset(DATA_DIRECTORY), target_column=TARGET, runs_directory=RUNS_DIRECTORY,
        load_saved_model=LOAD_SAVED_MODEL, selected_directory=SAVED_MODEL_DIRECTORY,
        show_visualizations=SHOW_VISUALIZATIONS,
    )


if __name__ == "__main__":
    main()
