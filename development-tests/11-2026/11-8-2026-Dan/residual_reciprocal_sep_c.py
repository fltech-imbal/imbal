"""Run SEP-C summed residual experts with reciprocal rarity and squared-error weighting."""

from dataclasses import replace

from sequential_ensemble_runner_sep_c import create_settings as baseline_settings
from sep_c_experiment import load_sep_c_dataset, run_experiment


def create_settings():
    return replace(
        baseline_settings(),
        run_name="sep_c_residual_reciprocal",
        expert_mode="residual_reciprocal",
        rarity_candidates=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0),
        minimum_relative_improvement=0.01,
        load_saved_model=False,
    )


if __name__ == "__main__":
    settings = create_settings()
    run_experiment(settings, *load_sep_c_dataset(settings.data_directory))
