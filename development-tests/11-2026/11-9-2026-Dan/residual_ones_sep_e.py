"""Run SEP-E residual experts with unit sample weights."""

from dataclasses import replace

from sequential_ensemble_runner_sep_e import create_settings as baseline_settings
from sep_e_experiment import load_sep_e_dataset, run_experiment


def create_settings():
    return replace(
        baseline_settings(),
        run_name="sep_e_residual_ones",
        expert_mode="residual_ones",
        load_saved_model=False,
    )


if __name__ == "__main__":
    settings = create_settings()
    run_experiment(settings, *load_sep_e_dataset(settings.data_directory))
