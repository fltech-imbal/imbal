"""Run SEP-C residual experts with unit sample weights."""

from dataclasses import replace

import numpy as np
import pandas as pd

from sequential_ensemble_runner_sep_c import create_settings as baseline_settings
from sep_c_experiment import load_sep_c_dataset, run_experiment
from sep_c_metrics import RARE_THRESHOLD


def create_settings():
    return replace(
        baseline_settings(),
        run_name="sep_c_residual_ones",
        expert_mode="residual_ones",
        load_saved_model=True,
    )


def print_worst_rare_samples(ensemble, testing):
    features, targets = testing
    targets = targets.ravel()
    final, expert_outputs = ensemble.predict(features, return_expert_outputs=True)
    final = final.ravel()
    errors = final.astype("float64") - targets
    rare = np.flatnonzero(targets >= RARE_THRESHOLD)
    worst = rare[np.argsort(-np.abs(errors[rare]), kind="stable")[:5]]

    columns = {"Index": worst, "True y": targets[worst]}
    for number, outputs in enumerate(expert_outputs.T, start=1):
        columns[f"Expert {number}"] = outputs[worst]
    columns["Final prediction"] = final[worst]
    columns["Delta"] = errors[worst]
    print("\nWorst rare test samples by absolute routed error (Delta = final prediction - true y)")
    print(pd.DataFrame(columns).to_string(index=False, na_rep="skipped", float_format="{:.6f}".format))


if __name__ == "__main__":
    settings = create_settings()
    training, testing, preprocessing = load_sep_c_dataset(settings.data_directory)
    ensemble = run_experiment(settings, training, testing, preprocessing)
    print_worst_rare_samples(ensemble, testing)
