"""Run the November 8 RIDE-inspired sequential expert experiment on SEP-C."""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from sep_c_experiment import load_sep_c_dataset, run_experiment
from sep_c_settings import EXPERIMENT_DIRECTORY, OCTOBER_DIRECTORY, ExperimentSettings


def create_settings():
    return ExperimentSettings(
        run_name="sep_c_sequential_experts",
        experiment_directory=EXPERIMENT_DIRECTORY,
        data_directory=OCTOBER_DIRECTORY,
        seeds=(42, 43, 44, 45, 46),
        fold_count=5,
        maximum_epochs=500,
        early_stopping_patience=100,
        training_batch_size=32,
        prediction_batch_size=128,

        maximum_experts=5,
        minimum_relative_improvement=0.01,
        router_threshold=0.5,
        balance_router_classes=False,

        load_saved_model=False,
        saved_model_directory=None,
        show_visualizations=False,
    )


if __name__ == "__main__":
    settings = create_settings()
    run_experiment(settings, *load_sep_c_dataset(settings.data_directory))
