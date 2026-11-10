"""Load normalized SEP splits, compare both predictors, and select the median seed."""

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from tensorflow import keras

from sep_e_metrics import (
    RARE_THRESHOLD, average_metrics, format_value, print_prefix_metrics, regression_metrics,
)
from sep_e_models import SequentialEnsemble
from sep_e_training import train_ensemble


def load_dataset(data_directory, training_files, testing_file, target_column, input_count):
    """Preserve the supplied feature order, normalization, and test boundary."""

    feature_columns = None

    def read_split(filename):
        nonlocal feature_columns
        path = Path(data_directory) / filename
        dataframe = pd.read_csv(path)
        targets = dataframe.pop(target_column).to_numpy(dtype="float32").reshape(-1, 1)
        columns = list(dataframe.columns)
        if len(columns) != input_count or (feature_columns is not None and columns != feature_columns):
            raise ValueError(f"Unexpected input count or column order in {path}.")

        feature_columns = columns
        features = dataframe.to_numpy(dtype="float32")
        if not np.all(np.isfinite(features)) or not np.all(np.isfinite(targets)):
            raise ValueError(f"{path} contains nonfinite inputs or targets.")

        return features, targets

    training_splits = [read_split(filename) for filename in training_files]
    if len(training_splits) == 1:
        training = training_splits[0]
    else:
        training = tuple(np.concatenate(arrays, axis=0) for arrays in zip(*training_splits))

    testing = read_split(testing_file)
    preprocessing = {
        "feature_columns": feature_columns, "target_column": target_column,
        "normalization": "Inputs already normalized in the supplied CSV files",
        "rare_threshold": float(RARE_THRESHOLD),
    }
    return training, testing, preprocessing


def load_sep_e_dataset(data_directory):
    # Match November 2 by pooling the published training and validation files.
    # The published test file stays separate throughout epoch and depth selection.
    return load_dataset(
        data_directory,
        training_files=["sep_e_log_normalized_training.csv", "sep_e_log_normalized_validation.csv"],
        testing_file="sep_e_log_normalized_test.csv",
        target_column="p16.4_tplus6",
        input_count=156,
    )


def report_split(ensemble, split):
    features, targets = split
    if not np.any(targets >= RARE_THRESHOLD):
        raise ValueError("AORE requires at least one rare evaluation sample.")
    return {
        "routed": regression_metrics(targets, ensemble.predict(features)),
        "full_mean": regression_metrics(targets, ensemble.predict_full(features)),
    }


def print_metric_summary(title, metrics, prefix=""):
    print(f"\n{title}")
    print(f"{'Metric':<20} {'Routed':>12} {'Full mean':>12}")
    for key, label in (("aore", "AORE"), ("overall_mae", "overall MAE"), ("common_mae", "common MAE"),
                       ("rare_mae", "rare MAE"), ("overall_mse", "overall MSE")):
        routed = format_value(metrics["routed"][key])
        full_mean = format_value(metrics["full_mean"][key])
        print(f"{prefix + label:<20} {routed:>12} {full_mean:>12}")


def show_median_predictions(ensemble, testing):
    from imbal.regression import plot_true_vs_predictions

    features, targets = testing
    plot_true_vs_predictions(targets, ensemble.predict(features))


def load_median_ensemble(settings, preprocessing):
    ensemble, metadata = SequentialEnsemble.load(settings.model_directory / "median.zip")
    if metadata["preprocessing"] != preprocessing:
        raise ValueError("The evaluation data schema differs from the saved model schema.")

    return ensemble, metadata["seed"]


def run_experiment(settings, training, testing, preprocessing):
    if settings.load_saved_model:
        ensemble, median_seed = load_median_ensemble(settings, preprocessing)
        print(f"Loaded median seed {median_seed}, {len(ensemble.experts)} accepted experts")
        print_metric_summary("Median model test metrics", report_split(ensemble, testing))
        if settings.show_visualizations:
            show_median_predictions(ensemble, testing)
        return ensemble

    results = []
    training_features, training_targets = training
    for seed in settings.seeds:
        keras.backend.clear_session()
        print(f"Training {settings.run_name}, seed {seed}")
        ensemble, training_details = train_ensemble(training_features, training_targets, seed, settings)
        metadata = {
            "seed": seed, "preprocessing": preprocessing, **training_details,
            "settings": {key: str(value) if isinstance(value, Path) else value
                         for key, value in asdict(settings).items()},
        }
        ensemble.save(settings.model_directory / f"seed_{seed}.zip", metadata)
        result = {
            "seed": seed, "accepted_experts": len(ensemble.experts),
            "attempted_experts": len(training_details["validation_history"]),
            "validation_history": training_details["validation_history"],
            "training": report_split(ensemble, training), "test": report_split(ensemble, testing),
        }
        results.append(result)
        print(f"Seed {seed}: {result['accepted_experts']} accepted / "
              f"{result['attempted_experts']} attempted experts")
        print_metric_summary("Training metrics", result["training"])
        print_metric_summary("Test metrics", result["test"])

    # Test AORE identifies the representative run only after all fits are final.
    ranked_results = sorted(results, key=lambda result: result["test"]["routed"]["aore"])
    median_seed = ranked_results[len(results) // 2]["seed"]
    average_test = {
        method: average_metrics([result["test"][method] for result in results])
        for method in ("routed", "full_mean")
    }
    summary = {"median_seed": median_seed, "average_test": average_test, "runs": results}
    if settings.maximum_experts >= 2:
        comparisons = [result["validation_history"][1]["held_out_comparison"] for result in results]
        average_validation = {
            key: average_metrics([comparison[key] for comparison in comparisons])
            for key in ("expert_1", "two_expert_mean", "oracle")
        }
        summary["average_expert_2_validation"] = average_validation
        print_prefix_metrics(f"Held-out validation averages across {len(results)} seeds", average_validation)

    settings.result_path.parent.mkdir(parents=True, exist_ok=True)
    settings.result_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    # Keep a complete median bundle that can be loaded or moved on its own.
    with TemporaryDirectory(dir=settings.model_directory) as directory:
        median_archive = Path(directory) / "median.zip"
        shutil.copyfile(settings.model_directory / f"seed_{median_seed}.zip", median_archive)
        median_archive.replace(settings.model_directory / "median.zip")
    print_metric_summary(f"Test averages across {len(results)} seeds", average_test, prefix="Mean ")
    median_aore = ranked_results[len(results) // 2]["test"]["routed"]["aore"]
    print(f"Median seed: {median_seed} (routed test AORE {median_aore:.6f})")
    ensemble, _ = load_median_ensemble(settings, preprocessing)
    if settings.show_visualizations:
        show_median_predictions(ensemble, testing)
    return ensemble
