"""Regression checks against October's mathematics, plus short training smoke tests.

Run this file directly from the repository's Python environment. Tests extract
only the required original definitions; they never start an October experiment.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import ast
import importlib.util
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import sep_e_training
from sep_e_experiment import load_saved_ensemble
from sep_e_models import (
    ExpertMSEPCCLoss,
    NormalizedGaussianKDE,
    WeightedExpertMSEPCCLoss,
    build_cached_gate_trainer,
    build_ensemble,
    build_expert,
    build_gate,
    gaussian_kde_density,
    predict_in_batches,
)
from sep_e_settings import EXPERIMENT_DIRECTORY, OCTOBER_DIRECTORY, ExperimentSettings
from sep_e_training import (
    GateAOREEarlyStopping,
    aore_components,
    create_prediction_cache,
    mix_cached_predictions,
    select_and_train_gate,
    train_experts,
)


def load_original_definitions(filename, definition_names):
    source_path = OCTOBER_DIRECTORY / filename
    syntax_tree = ast.parse(source_path.read_text(encoding="utf-8"))

    # AST extraction retains the actual old implementation rather than rewriting
    # an expected answer that might accidentally duplicate a new implementation bug.
    selected_definitions = [
        node for node in syntax_tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in definition_names
    ]

    namespace = {
        "np": np,
        "tf": tf,
        "keras": keras,
        "layers": layers,
        "DENSITY_EPSILON": 1e-7,
        "PCC_EPSILON": 1e-7,
        "RARE_PCC_WEIGHT": 2.0,
        "EXPECTED_INPUT_COUNT": 156,
    }

    executable = compile(
        ast.Module(body=selected_definitions, type_ignores=[]), str(source_path), "exec"
    )
    exec(executable, namespace)
    return namespace


def entry_settings():
    """Read every real entry file so its experiment settings are covered too."""

    variants = []

    for entry_path in sorted(EXPERIMENT_DIRECTORY.glob("gated-ensemble*.py")):
        specification = importlib.util.spec_from_file_location(entry_path.stem, entry_path)
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        variants.append(module.create_settings())

    return variants


class EnsembleOptimizationTests(unittest.TestCase):
    def setUp(self):
        keras.backend.clear_session()
        keras.utils.set_random_seed(19)
        self.random_generator = np.random.default_rng(19)

    def test_bounded_density_matches_original_in_eager_and_dynamic_graph(self):
        original = load_original_definitions(
            "gated-ensemble-with-density-trust-sep-e.py", {"NormalizedGaussianKDE"}
        )

        references = self.random_generator.normal(size=257).astype("float32")
        queries = self.random_generator.normal(size=(83, 1)).astype("float32")
        profile = dict(training_targets=references, bandwidth=0.31, maximum_density=0.7)

        original_layer = original["NormalizedGaussianKDE"](**profile)
        bounded_layer = NormalizedGaussianKDE(
            **profile, query_chunk_size=13, reference_chunk_size=61
        )

        expected = original_layer(queries).numpy()
        np.testing.assert_allclose(bounded_layer(queries).numpy(), expected, rtol=2e-6, atol=1e-7)

        # Exercise empty inputs, partial final chunks, and varying traced batch
        # sizes. These are common failure modes for TensorArray/while_loop code.
        compiled_density = tf.function(
            bounded_layer, input_signature=[tf.TensorSpec([None, 1], tf.float32)]
        )

        for row_count in (0, 1, 12, 13, 14, 83):
            actual = compiled_density(queries[:row_count]).numpy()
            np.testing.assert_allclose(actual, expected[:row_count], rtol=2e-6, atol=1e-7)

        numpy_density = gaussian_kde_density(queries, references, 0.31, 13, 61)
        direct_differences = (queries.astype("float64") - references[None, :]) / 0.31
        expected_density = np.mean(np.exp(-0.5 * direct_differences ** 2), axis=1)
        expected_density /= 0.31 * np.sqrt(2 * np.pi)
        np.testing.assert_allclose(numpy_density, expected_density, rtol=1e-12, atol=1e-12)

    def test_both_expert_losses_and_prediction_gradients_match_october(self):
        cases = (
            ("gated-ensemble-with-density-trust-sep-e-both-experts-pcc.py", ExpertMSEPCCLoss, False),
            ("gated-ensemble-regression-gate-sep-e-both-experts-pcc-density-trust-during-training.py",
             WeightedExpertMSEPCCLoss, True),
        )

        for filename, loss_class, weighted in cases:
            original = load_original_definitions(filename, {"ExpertMSEPCCLoss"})
            old_loss = original["ExpertMSEPCCLoss"]()
            new_loss = loss_class()

            for constant_targets in (False, True):
                targets = self.random_generator.normal(size=(17, 1)).astype("float32")

                if constant_targets:
                    targets.fill(1.0)

                if weighted:
                    weights = np.linspace(0.1, 2.0, len(targets), dtype="float32")
                    targets = np.column_stack((targets.reshape(-1), weights))

                predictions = tf.Variable(self.random_generator.normal(size=(17, 1)), dtype=tf.float32)

                with tf.GradientTape(persistent=True) as tape:
                    expected_loss = old_loss(targets, predictions)
                    actual_loss = new_loss(targets, predictions)

                np.testing.assert_allclose(actual_loss.numpy(), expected_loss.numpy(), rtol=1e-6)
                np.testing.assert_allclose(
                    tape.gradient(actual_loss, predictions).numpy(),
                    tape.gradient(expected_loss, predictions).numpy(),
                    rtol=1e-6, atol=1e-7,
                )

                del tape

    def test_cached_gate_predictions_gradients_and_updates_match_all_four_originals(self):
        features = self.random_generator.normal(size=(21, 4)).astype("float32")
        targets = self.random_generator.normal(size=(21, 1)).astype("float32")
        importance_weights = np.linspace(0.2, 2.0, 21, dtype="float32")

        for sigmoid_gate in (False, True):
            for training_density in (False, True):
                suffix = "density-trust-during-training" if training_density else "no-quality-during-training"
                activation_suffix = "-sigmoid" if sigmoid_gate else ""
                filename = (
                    "gated-ensemble-regression-gate-sep-e-both-experts-pcc-"
                    f"{suffix}{activation_suffix}.py"
                )

                original = load_original_definitions(filename, {
                    "NormalizedGaussianKDE", "NormalizeExpertWeights",
                    "build_trainable_gate_ensemble",
                })

                settings = ExperimentSettings(
                    "gradient_check", regression_gate=True, expert_correlation_loss=True,
                    density_at_inference=True, density_during_gate_training=training_density,
                    sigmoid_gate=sigmoid_gate, density_query_chunk_size=8,
                    density_reference_chunk_size=17,
                )

                common_expert = build_expert("common_expert", 4, settings)
                rare_expert = build_expert("rare_expert", 4, settings)
                rare_expert.layers[-1].bias.assign([2.0])

                profiles = [
                    dict(training_targets=np.linspace(-2, 2, 51), bandwidth=0.5, maximum_density=0.5),
                    dict(training_targets=np.linspace(1, 5, 47), bandwidth=0.5, maximum_density=0.5),
                ]

                old_gate = build_gate(4, settings)
                new_gate = build_gate(4, settings)
                new_gate.set_weights(old_gate.get_weights())

                old_trainer = original["build_trainable_gate_ensemble"](
                    common_expert, rare_expert, old_gate, *profiles
                )
                cache = create_prediction_cache(common_expert, rare_expert, features, profiles, settings)
                new_trainer = build_cached_gate_trainer(new_gate, 4, settings)
                cached_inputs = cache.training_inputs(features, slice(None), settings)

                with tf.GradientTape() as old_tape:
                    old_predictions = old_trainer(features, training=True)
                    old_loss = tf.reduce_mean(
                        tf.reshape(importance_weights, [-1, 1]) * tf.square(old_predictions - targets)
                    )

                with tf.GradientTape() as new_tape:
                    new_predictions = new_trainer(cached_inputs, training=True)
                    new_loss = tf.reduce_mean(
                        tf.reshape(importance_weights, [-1, 1]) * tf.square(new_predictions - targets)
                    )

                old_gradients = old_tape.gradient(old_loss, old_gate.trainable_variables)
                new_gradients = new_tape.gradient(new_loss, new_gate.trainable_variables)

                np.testing.assert_allclose(
                    new_predictions.numpy(), old_predictions.numpy(),
                    rtol=2e-5, atol=2e-6,
                )

                for old_gradient, new_gradient in zip(old_gradients, new_gradients):
                    np.testing.assert_allclose(
                        new_gradient.numpy(), old_gradient.numpy(),
                        rtol=2e-4, atol=2e-6,
                    )

                # Also compare an actual compiled Adam step, including Keras'
                # sample-weight reduction rather than only the explicit tape loss.
                old_trainer.train_on_batch(features, targets, sample_weight=importance_weights)
                new_trainer.train_on_batch(cached_inputs, targets, sample_weight=importance_weights)

                for old_weight, new_weight in zip(old_gate.get_weights(), new_gate.get_weights()):
                    np.testing.assert_allclose(new_weight, old_weight, rtol=2e-4, atol=2e-6)

                # Verify that early stopping follows the training-time mixture,
                # particularly the variants that add density only at inference.
                stopping = GateAOREEarlyStopping(
                    new_gate, features, targets, np.arange(21) % 2, cache, settings
                )
                stopping.set_model(new_trainer)
                stopping.on_epoch_end(0)
                expected_aore = aore_components(
                    targets, new_trainer(cached_inputs).numpy(), np.arange(21) % 2
                )[2]
                self.assertAlmostEqual(stopping.best_aore, expected_aore, places=5)

    def test_professor_table_rows_have_the_requested_method_settings(self):
        variants = {settings.run_name: settings for settings in entry_settings()}

        # The four rows require PCC on both experts. Only the two regression
        # rows use independent sigmoid outputs rather than a softmax classifier.
        expected_rows = {
            "density_trust_sep_e_both_experts_pcc": (False, True, False, False),
            "sep_e_classification_gate_both_experts_pcc_no_quality": (
                False, False, False, False
            ),
            "sep_e_regression_gate_both_experts_pcc_sigmoid_density_trust_training": (
                True, True, True, True
            ),
            "sep_e_regression_gate_both_experts_pcc_sigmoid_no_quality_training_or_testing": (
                True, False, False, True
            ),
        }

        for run_name, expected_flags in expected_rows.items():
            with self.subTest(run_name=run_name):
                settings = variants[run_name]
                self.assertTrue(settings.expert_correlation_loss)

                actual_flags = (
                    settings.regression_gate,
                    settings.density_at_inference,
                    settings.density_during_gate_training,
                    settings.sigmoid_gate,
                )
                self.assertEqual(actual_flags, expected_flags)

        # New variants must not overwrite an existing variant's model archive.
        directories = [settings.model_directory for settings in entry_settings()]
        self.assertEqual(len(directories), len(set(directories)))

    def test_quality_free_sigmoid_export_normalizes_the_expert_mixture(self):
        settings = next(
            candidate for candidate in entry_settings()
            if candidate.run_name == (
                "sep_e_regression_gate_both_experts_pcc_sigmoid_no_quality_training_or_testing"
            )
        )

        features = np.zeros((5, 4), dtype="float32")
        common_expert = build_expert("common_expert", 4, settings)
        rare_expert = build_expert("rare_expert", 4, settings)
        gate = build_gate(4, settings)

        # Force expert outputs of 2 and 8, and sigmoid trusts of 0.2 and 0.4.
        # Their weighted average must be 6; an unnormalized export would give 3.6.
        for model in (common_expert, rare_expert, gate):
            model.set_weights([np.zeros_like(weight) for weight in model.get_weights()])

        common_expert.layers[-1].bias.assign([2.0])
        rare_expert.layers[-1].bias.assign([8.0])
        desired_trust = np.array([0.2, 0.4], dtype="float32")
        gate.get_layer("gate_linear_outputs").bias.assign(
            np.log(desired_trust / (1.0 - desired_trust))
        )

        # Neither cache construction nor the exported quality-free model should
        # evaluate a density profile. Importance-weighting code is separate.
        with (
            patch.object(sep_e_training, "NormalizedGaussianKDE") as density_layer,
            patch.object(sep_e_training, "gaussian_kde_density") as numpy_density,
        ):
            cache = create_prediction_cache(
                common_expert, rare_expert, features, None, settings
            )
            density_layer.assert_not_called()
            numpy_density.assert_not_called()

        trainer = build_cached_gate_trainer(gate, 4, settings)
        training_inputs = cache.training_inputs(features, slice(None), settings)
        ensemble = build_ensemble(common_expert, rare_expert, gate, None, settings)

        self.assertFalse(any(
            isinstance(layer, NormalizedGaussianKDE) for layer in ensemble.layers
        ))

        weights_model = keras.Model(
            ensemble.input, ensemble.get_layer("normalized_gate_trust").output
        )
        mixture_weights = weights_model(features).numpy()

        np.testing.assert_allclose(mixture_weights.sum(axis=1), 1.0, atol=1e-7)
        np.testing.assert_allclose(ensemble(features).numpy(), 6.0, atol=1e-6)
        np.testing.assert_allclose(trainer(training_inputs).numpy(), 6.0, atol=1e-6)

        cached_predictions, _ = mix_cached_predictions(
            gate(features).numpy(), cache, use_density=False, normalize_trust=True
        )
        np.testing.assert_allclose(cached_predictions, 6.0, atol=1e-6)

        # Validation must use the same normalized, quality-free prediction as
        # training and export, including when one of the targets is rare.
        targets = np.array([0, 1, 3, 4, 5], dtype="float32").reshape(-1, 1)
        region_labels = np.array([0, 0, 1, 1, 1])
        stopping = GateAOREEarlyStopping(
            gate, features, targets, region_labels, cache, settings
        )
        stopping.set_model(trainer)
        stopping.on_epoch_end(0)

        expected_aore = aore_components(targets, np.full(5, 6.0), region_labels)[2]
        self.assertAlmostEqual(stopping.best_aore, expected_aore, places=6)

    def test_short_training_and_serialization_for_all_nine_entry_points(self):
        variants = entry_settings()
        self.assertEqual(len(variants), 9)

        # A small fixture covers both expert sub-bins and both target regions.
        # Keep real folds and model.fit calls, but avoid the full 138-fit searches.
        features = self.random_generator.normal(size=(48, 4)).astype("float32")
        targets = np.concatenate([
            np.linspace(-2, 2, 24), np.linspace(3, 7, 24)
        ]).astype("float32").reshape(-1, 1)
        region_labels = np.repeat([0, 1], 24)

        for original_settings in variants:
            settings = replace(
                original_settings, fold_count=2, maximum_epochs=1,
                early_stopping_patience=1, alpha_candidates=(0.0,),
                rare_class_weight_candidates=(1.0,), prediction_batch_size=13,
                density_query_chunk_size=7, density_reference_chunk_size=11,
            )

            # Only the expensive weighting-density estimator is stubbed here.
            # Its unchanged library implementation is not the subject of this test.
            with (
                patch.object(
                    sep_e_training,
                    "compute_sample_densities",
                    side_effect=lambda values: np.ones(len(values)),
                ),
                patch.object(
                    sep_e_training,
                    "reciprocal_weights",
                    side_effect=lambda values, alpha: np.ones(len(values), dtype="float32"),
                ),
            ):
                common_expert, rare_expert = train_experts(features, targets, region_labels, settings)

                profiles = None

                if settings.density_at_inference:
                    profiles = [
                        dict(
                            training_targets=targets[region_labels == region],
                            bandwidth=0.5,
                            maximum_density=0.5,
                        )
                        for region in (0, 1)
                    ]

                cache = create_prediction_cache(common_expert, rare_expert, features, profiles, settings)
                gate = select_and_train_gate(features, targets, region_labels, cache, settings)

            ensemble = build_ensemble(common_expert, rare_expert, gate, profiles, settings)
            direct_predictions = predict_in_batches(ensemble, features, 13)
            trust = predict_in_batches(gate, features, 13)
            cached_predictions, _ = mix_cached_predictions(
                trust, cache, settings.density_at_inference, settings.sigmoid_gate
            )

            np.testing.assert_allclose(
                cached_predictions, direct_predictions.reshape(-1),
                rtol=2e-5, atol=2e-6,
            )

            # Temporary archives stay inside this test's explicitly selected
            # directory. The user's October and November model folders are untouched.
            with tempfile.TemporaryDirectory(prefix="sep-e-test-", dir=EXPERIMENT_DIRECTORY) as directory:
                temporary_directory = Path(directory).resolve()
                self.assertEqual(temporary_directory.parent, EXPERIMENT_DIRECTORY)
                ensemble.save(temporary_directory / "ensemble.keras")

                loaded, _, _, _, _ = load_saved_ensemble(replace(
                    settings, saved_model_directory=temporary_directory, load_saved_model=True
                ))

                np.testing.assert_allclose(
                    predict_in_batches(loaded, features, 13), direct_predictions,
                    rtol=1e-6, atol=1e-7,
                )

    def test_imbalanced_expert_search_reuses_fold_densities_across_alphas(self):
        settings = ExperimentSettings(
            "weighted_expert_check",
            regression_gate=True,
            expert_correlation_loss=True,
            fold_count=2,
            maximum_epochs=1,
            early_stopping_patience=1,
            alpha_candidates=(0.0, 0.6),
        )

        # Both regions contain a 16/8 sub-bin imbalance, so this test exercises
        # the weighted search branch that the balanced smoke fixture skips.
        targets = np.concatenate([
            np.linspace(-2, -0.1, 16),
            np.linspace(0.1, 2, 8),
            np.linspace(3, 4.9, 16),
            np.linspace(5.1, 7, 8),
        ]).astype("float32").reshape(-1, 1)

        features = self.random_generator.normal(size=(48, 4)).astype("float32")
        regions = np.repeat([0, 1], 24)

        with (
            patch.object(
                sep_e_training,
                "compute_sample_densities",
                side_effect=lambda values: np.linspace(0.2, 1.0, len(values)),
            ) as density_estimator,
            patch.object(
                sep_e_training,
                "reciprocal_weights",
                side_effect=lambda densities, alpha: np.asarray(
                    densities ** (-alpha), dtype="float32"
                ),
            ),
        ):
            common_expert, rare_expert = train_experts(features, targets, regions, settings)

        # Each expert needs two fold densities and one final-training density.
        # Adding alpha candidates must not multiply those estimator calls.
        self.assertEqual(density_estimator.call_count, 6)
        self.assertTrue(np.all(np.isfinite(common_expert(features).numpy())))
        self.assertTrue(np.all(np.isfinite(rare_expert(features).numpy())))


if __name__ == "__main__":
    unittest.main(verbosity=2)
