"""Run one five-epoch expert/gate freeze check on each November SEP dataset."""

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from tensorflow import keras

NOVEMBER = Path(__file__).resolve().parent
sys.path.extend(str(NOVEMBER / folder) for folder in ("11-1-2026-Dan", "11-2-2026-Dan"))

import sep_c_experiment
import sep_c_training
import sep_e_experiment
import sep_e_training
from sep_c_settings import ExperimentSettings as SepCSettings
from sep_e_settings import ExperimentSettings as SepESettings


EPOCHS = 5
SEED = 42


def changed_tensors(before, after):
    return sum(
        not np.array_equal(old, new)
        for old, new in zip(before, after, strict=True)
    )


def run_check(dataset, load_dataset, experiment, training, settings_type):
    keras.backend.clear_session()
    training.set_global_determinism(SEED)
    settings = settings_type(
        run_name="freeze_check", seed=SEED, regression_gate=True,
        expert_correlation_loss=True, sigmoid_gate=True,
        density_during_gate_training=True, density_at_inference=True,
    )
    features, targets = load_dataset(settings.data_directory)[:2]
    rare = targets.reshape(-1) >= experiment.RARE_THRESHOLD
    print(f"{dataset} | {len(features)} training rows | seed {SEED}", flush=True)

    # Unit importance weights (alpha=0) permit one fit per model. The gate's
    # density quality is a separate quantity and stays enabled.
    def fit_expert(region_features, region_targets, expert_name, settings):
        expert = training.build_expert(expert_name, region_features.shape[1], settings)
        expert.fit(
            region_features, training.pack_expert_targets(region_targets),
            epochs=EPOCHS, batch_size=settings.training_batch_size, shuffle=False, verbose=0,
        )
        return expert

    print(f"  Stage 1: training each expert for {EPOCHS} epochs", flush=True)
    experts = training.fit_and_freeze_experts(features, targets, rare, settings, fit_expert)
    weights_before = [expert.get_weights() for expert in experts]
    profiles = [experiment.build_density_profile(targets[mask], settings) for mask in (~rare, rare)]
    cache = training.create_prediction_cache(*experts, features, profiles, settings)
    predictions_before = cache.predictions.copy()

    print(f"  Stage 2: training the gate for {EPOCHS} epochs", flush=True)
    gate = training.build_gate(features.shape[1], settings)
    gate_before = gate.get_weights()
    trainer = training.build_cached_gate_trainer(gate, features.shape[1], settings)
    trainer.fit(
        cache.training_inputs(features, slice(None), settings), targets,
        epochs=EPOCHS, batch_size=settings.training_batch_size, shuffle=False, verbose=0,
    )

    gate_after = gate.get_weights()
    gate_changes = changed_tensors(gate_before, gate_after)
    passed = gate_changes > 0 and all(np.isfinite(w).all() for w in gate_after)
    print(f"  Gate weight tensors changed: {gate_changes}/{len(gate_after)}")
    for i, expert in enumerate(experts):
        # Evaluate the live model again on exactly the same ordered training rows.
        after = experiment.predict_in_batches(expert, features, settings.prediction_batch_size)
        delta = np.abs(after.reshape(-1) - predictions_before[:, i])
        weights_changed = changed_tensors(weights_before[i], expert.get_weights())
        rows_changed = np.count_nonzero(delta)
        print(
            f"  {expert.name}: {rows_changed}/{len(features)} predictions changed, "
            f"max absolute change {delta.max():.8g}, {weights_changed} weight tensors changed"
        )
        passed = passed and not expert.trainable_weights and rows_changed == 0 and weights_changed == 0
    print(f"  {'PASS' if passed else 'FAIL'}", flush=True)
    return passed


if __name__ == "__main__":
    sep_c_passed = run_check(
        "SEP-C", sep_c_experiment.load_sep_c_dataset, sep_c_experiment, sep_c_training, SepCSettings
    )
    sep_e_passed = run_check(
        "SEP-E", sep_e_experiment.load_sep_e_dataset, sep_e_experiment, sep_e_training, SepESettings
    )
    sys.exit(0 if sep_c_passed and sep_e_passed else 1)
