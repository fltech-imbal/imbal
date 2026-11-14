"""Run SEP-C residual experts with squared-residual sample weights."""

from dataclasses import replace

from sequential_ensemble_runner_sep_c import create_settings as baseline_settings
from sep_c_experiment import load_sep_c_dataset, run_experiment


def create_settings():
    return replace(
        baseline_settings(),
        run_name="sep_c_residual_weighted",
        expert_mode="residual_weighted",
        load_saved_model=False,
    )


if __name__ == "__main__":
    settings = create_settings()
    run_experiment(settings, *load_sep_c_dataset(settings.data_directory))
