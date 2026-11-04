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
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import sep_c_training
from sep_c_experiment import load_saved_ensemble, load_sep_c_dataset, RARE_THRESHOLD
from sep_c_models import (
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
from sep_c_settings import EXPERIMENT_DIRECTORY, OCTOBER_DIRECTORY, ExperimentSettings
from sep_c_training import (
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
        "EXPECTED_INPUT_COUNT": 22,
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

    def test_loader_preserves_the_recent_pseudolabeled_splits(self):
        arrays = load_sep_c_dataset(OCTOBER_DIRECTORY)

        # Compare every feature and target with the original source files. This
        # catches accidental scaling, row merging, and loading SEP-E by mistake.
        for split_name, features, targets, expected_count in (
            ("training", arrays[0], arrays[1], 1531),
            ("testing", arrays[2], arrays[3], 766),
        ):
            dataframe = pd.read_csv(
                OCTOBER_DIRECTORY / f"sep_10mev_{split_name}_pseudo_labeled.csv"
            )
            self.assertEqual(features.shape, (expected_count, 22))
            np.testing.assert_array_equal(
                features, dataframe.drop(columns=["ln_peak_intensity"]).to_numpy(dtype="float32")
            )
            np.testing.assert_array_equal(
                targets[:, 0], dataframe["ln_peak_intensity"].to_numpy(dtype="float32")
            )
            self.assertTrue(np.any(targets >= RARE_THRESHOLD))
            self.assertTrue(np.any(targets < RARE_THRESHOLD))

        # All entry points must use the verified dataset and train fresh models.
        for settings in entry_settings():
            self.assertEqual(settings.data_directory, OCTOBER_DIRECTORY)
            self.assertFalse(settings.load_saved_model)

    def test_bounded_density_matches_original_in_eager_and_dynamic_graph(self):
        original = load_original_definitions(
            "gated-ensemble-with-density-trust-sep-c-pseudolabeled-both-experts-pcc.py", {"NormalizedGaussianKDE"}
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
            ("gated-ensemble-with-density-trust-sep-c-pseudolabeled-both-experts-pcc.py", ExpertMSEPCCLoss, False),
            ("gated-ensemble-regression-gate-with-density-trust-sep-c-pseudolabeled-both-experts-pcc.py",
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


    def test_professor_table_rows_have_the_requested_method_settings(self):
        variants = {settings.run_name: settings for settings in entry_settings()}

        # The four rows require PCC on both experts. Only the two regression
        # rows use independent sigmoid outputs rather than a softmax classifier.
        expected_rows = {
            "sep_c_pseudolabeled_classification_pcc_quality": (False, True, False, False),
            "sep_c_pseudolabeled_classification_pcc_no_quality": (
                False, False, False, False
            ),
            "sep_c_pseudolabeled_sigmoid_regression_pcc_quality": (
                True, True, True, True
            ),
            "sep_c_pseudolabeled_sigmoid_regression_pcc_no_quality": (
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

    def test_mse_experts_use_one_five_fold_pass_and_gate_still_searches_aore(self):
        variants = [settings for settings in entry_settings() if settings.mse_only_expert_training]
        self.assertEqual(len(variants), 2)

        # The new pair matches the two regression table rows, changing only
        # expert loss and epoch selection. Both gates remain normalized sigmoid.
        for settings in variants:
            self.assertTrue(settings.regression_gate)
            self.assertTrue(settings.sigmoid_gate)
            self.assertFalse(settings.expert_correlation_loss)
            self.assertEqual(settings.fold_count, 5)
            self.assertEqual(settings.density_during_gate_training, settings.density_at_inference)
            self.assertEqual(len(settings.alpha_candidates), 9)

        self.assertEqual({settings.density_at_inference for settings in variants}, {False, True})
        settings = replace(
            next(settings for settings in variants if not settings.density_at_inference),
            maximum_epochs=1, early_stopping_patience=1, alpha_candidates=(0.0, 0.6),
        )

        features = self.random_generator.normal(size=(48, 22)).astype("float32")
        targets = np.concatenate([
            np.linspace(-2, 2, 24), np.linspace(3, 7, 24)
        ]).astype("float32").reshape(-1, 1)
        region_labels = np.repeat([0, 1], 24)

        # Exercise real MSE fits with five folds. No expert density estimator,
        # packed PCC targets, or dual-bin callback may participate in this path.
        with (
            patch.object(sep_c_training, "build_expert", wraps=build_expert) as expert_builder,
            patch.object(sep_c_training, "compute_sample_densities", side_effect=AssertionError("Expert weighting")),
            patch.object(sep_c_training, "pack_expert_targets", side_effect=AssertionError("Packed expert targets")),
            patch.object(sep_c_training, "ExpertSubbinEarlyStopping", side_effect=AssertionError("Expert sub-bin selection")),
        ):
            common_expert, rare_expert = train_experts(features, targets, region_labels, settings)

        self.assertEqual(expert_builder.call_count, 12)  # Five folds + one refit per expert.
        self.assertEqual(common_expert.loss, "mse")
        self.assertEqual(rare_expert.loss, "mse")

        cache = create_prediction_cache(common_expert, rare_expert, features, None, settings)
        gate_settings = replace(settings, fold_count=2)
        observed_scores = {candidate: [] for candidate in settings.alpha_candidates}
        final_candidates = []
        original_fit_gate_candidate = sep_c_training.fit_gate_candidate

        def record_gate_fit(*arguments, **keyword_arguments):
            result = original_fit_gate_candidate(*arguments, **keyword_arguments)
            candidate = arguments[6]
            callbacks = keyword_arguments.get("callbacks")

            # Observe the actual validation AORE from each completed gate fit,
            # then compare its mean with the candidate used for the final refit.
            if callbacks:
                observed_scores[candidate].append(callbacks[0].best_aore)
            else:
                final_candidates.append(candidate)

            return result

        with (
            patch.object(sep_c_training, "fit_gate_candidate", side_effect=record_gate_fit),
            patch.object(sep_c_training, "compute_sample_densities", side_effect=lambda values: np.linspace(0.2, 1.0, len(values))),
            patch.object(sep_c_training, "reciprocal_weights", side_effect=lambda densities, alpha: np.asarray(densities ** (-alpha), dtype="float32")),
        ):
            select_and_train_gate(features, targets, region_labels, cache, gate_settings)

        self.assertTrue(all(len(scores) == 2 for scores in observed_scores.values()))
        expected_candidate = min(observed_scores, key=lambda candidate: np.mean(observed_scores[candidate]))
        self.assertEqual(final_candidates, [expected_candidate])

    def test_quality_free_sigmoid_export_normalizes_the_expert_mixture(self):
        settings = next(
            candidate for candidate in entry_settings()
            if candidate.run_name == (
                "sep_c_pseudolabeled_sigmoid_regression_pcc_no_quality"
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
            patch.object(sep_c_training, "NormalizedGaussianKDE") as density_layer,
            patch.object(sep_c_training, "gaussian_kde_density") as numpy_density,
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

    def test_short_training_and_serialization_for_all_six_entry_points(self):
        variants = entry_settings()
        self.assertEqual(len(variants), 6)

        # A small fixture covers both expert sub-bins and both target regions.
        # Keep real folds and model.fit calls, but avoid the full 138-fit searches.
        features = self.random_generator.normal(size=(48, 22)).astype("float32")
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
                    sep_c_training,
                    "compute_sample_densities",
                    side_effect=lambda values: np.ones(len(values)),
                ),
                patch.object(
                    sep_c_training,
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
            with tempfile.TemporaryDirectory(prefix="sep-c-test-", dir=EXPERIMENT_DIRECTORY) as directory:
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

        features = self.random_generator.normal(size=(48, 22)).astype("float32")
        regions = np.repeat([0, 1], 24)

        with (
            patch.object(
                sep_c_training,
                "compute_sample_densities",
                side_effect=lambda values: np.linspace(0.2, 1.0, len(values)),
            ) as density_estimator,
            patch.object(
                sep_c_training,
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
