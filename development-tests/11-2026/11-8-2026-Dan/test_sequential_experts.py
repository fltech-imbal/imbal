"""Regression checks for sample weighting, future-prefix labels, and lazy routing."""

import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from sequential_ensemble_runner_sep_c import create_settings
from sep_c_metrics import compare_second_expert
from sep_c_models import SequentialEnsemble, router_inputs
from sep_c_training import (
    ValidationFold, error_weights, router_targets, router_weights, select_experts,
)


class SequentialExpertTests(unittest.TestCase):
    def test_held_out_comparison_counts_ties_and_uses_best_prefix_for_each_row(self):
        targets = np.array([0, 3, 4, 5])
        first = np.array([0, 0, 3, 4])
        mean = np.array([2, 2, 2, 4])
        comparison = compare_second_expert(targets, first, mean)

        self.assertEqual(comparison["expert_1"]["common_mae"], 0)
        self.assertEqual(comparison["two_expert_mean"]["common_mae"], 2)
        self.assertAlmostEqual(comparison["two_expert_mean"]["rare_mae"], 4 / 3)
        self.assertAlmostEqual(comparison["oracle"]["aore"], (0.75 + 1) / 2)
        self.assertEqual(comparison["rare_improved"], 1)
        self.assertEqual(comparison["rare_worsened"], 1)
        self.assertEqual(comparison["rare_tied"], 1)
        self.assertEqual(comparison["median_rare_error_reduction"], 0)
        self.assertEqual(comparison["largest_share_of_rare_gains"], 1)

    def test_weights_sum_to_row_count_and_handle_perfect_fit(self):
        targets = np.array([[0], [1], [2]], dtype="float32")
        weights = error_weights(targets, np.zeros_like(targets))
        np.testing.assert_allclose(weights, [0, 0.6, 2.4])
        self.assertAlmostEqual(float(weights.sum()), len(targets))
        np.testing.assert_array_equal(error_weights(targets, targets), [1, 1, 1])

    def test_router_looks_past_a_worse_next_expert_and_stops_on_ties(self):
        targets = np.zeros((3, 1), dtype="float32")
        predictions = np.array([[1, 3, -4], [1, 1, 1], [0, 2, 1]], dtype="float32")
        np.testing.assert_array_equal(router_targets(targets, predictions, 1), [1, 0, 0])
        np.testing.assert_array_equal(router_targets(targets, predictions, 2), [1, 0, 0])

    def test_router_input_order_and_zero_placeholder(self):
        inputs = router_inputs(np.array([[7, 8]]), np.array([[2, 4]]))
        np.testing.assert_array_equal(inputs, [[7, 8, 2, 4, 3, 1, 0]])
        np.testing.assert_array_equal(router_inputs(np.array([[7]]), np.array([[2]])), [[7, 2, 2, 0, 0]])

    def test_optional_class_balance_handles_constant_labels(self):
        settings = replace(create_settings(), balance_router_classes=True)
        labels = np.array([0, 0, 0, 1])
        np.testing.assert_allclose(router_weights(labels, settings), [2 / 3, 2 / 3, 2 / 3, 2])
        self.assertIsNone(router_weights(np.ones(3), settings))

    def test_validation_uses_prefix_mean_and_rejects_a_subpercent_gain(self):
        features = np.arange(1, 5, dtype="float32").reshape(-1, 1)
        targets = np.zeros_like(features)
        folds = [
            ValidationFold(np.array(training), np.array(validation), np.empty((4, 0), dtype="float32"))
            for training, validation in [([0, 2], [1, 3]), ([1, 3], [0, 2])]
        ]
        # Prefix predictions become 1.0, 0.5, and 0.499 times the input.
        # The third improves MSE by less than 1% and must be discarded.
        scales = [1.0, 0.0, 0.497]
        fit_weights = []

        def fit(model, inputs, labels, weights, validation_data, settings):
            fit_weights.append(weights)
            return 1

        with patch("sep_c_training.build_expert", side_effect=lambda n, count, s: n), \
             patch("sep_c_training.fit_for_validation", side_effect=fit), \
             patch("sep_c_training.predict_in_batches",
                   side_effect=lambda n, inputs, size: scales[n - 1] * inputs):
            epochs, history = select_experts(features, targets, folds, 42, create_settings())

        self.assertEqual(epochs, [1, 1])
        self.assertEqual([step["accepted"] for step in history], [True, True, False])
        self.assertIsNone(history[1]["held_out_comparison"]["two_expert_mean"]["aore"])
        self.assertEqual(history[1]["held_out_comparison"]["expert_1"]["overall_mse"], 7.5)
        self.assertEqual(fit_weights[:2], [None, None])
        for index, fold in enumerate(folds):
            self.assertEqual(fold.predictions.shape, (4, 2))
            expected = np.square(features[fold.training_indices].ravel())
            np.testing.assert_allclose(fit_weights[2 + index], expected / expected.mean())

    def test_residual_selection_keeps_nonworsening_experts_and_stops_below_one_percent(self):
        features = np.arange(200, dtype="float32").reshape(-1, 1)
        targets = np.zeros_like(features)
        even = np.arange(0, len(features), 2)
        odd = even + 1

        # Expert 1 misses by one on every row. Correcting one row reduces MSE
        # by 0.5%, and correcting two reduces it by exactly 1%.
        for corrected, delta, expected in [
            (1, -1, [True, True]),
            (0, 0, [True, True]),
            (1, 1, [True, False]),
            (2, -1, [True, True, False]),
        ]:
            def predict(number, inputs, batch_size):
                if number == 2:
                    return np.where(inputs < corrected, delta, 0).astype("float32")
                return np.ones_like(inputs)

            for mode in ("residual_ones", "residual_weighted"):
                with self.subTest(mode=mode, corrected=corrected, delta=delta):
                    settings = replace(create_settings(), expert_mode=mode)
                    folds = [ValidationFold(train, val, np.empty((200, 0), dtype="float32"))
                             for train, val in ((even, odd), (odd, even))]
                    with patch("sep_c_training.build_expert", side_effect=lambda n, *args: n), \
                         patch("sep_c_training.fit_for_validation", return_value=1), \
                         patch("sep_c_training.predict_in_batches", side_effect=predict):
                        epochs, history = select_experts(features, targets, folds, 42, settings)

                    self.assertEqual([step["accepted"] for step in history], expected)
                    self.assertEqual(len(epochs), sum(expected))
                    for fold in folds:
                        self.assertEqual(fold.predictions.shape[1], sum(expected))

    def test_only_continuing_rows_reach_later_experts(self):
        calls = []

        def expert(offset):
            def predict(features, training=False):
                calls.append(features[:, 0].tolist())
                return features[:, :1] + offset
            return predict

        def first_router(inputs, training=False):
            return (inputs[:, :1] != 1).astype("float32")

        def second_router(inputs, training=False):
            # Include the row exactly on the decision threshold.
            return (inputs[:, :1] == 2).astype("float32") * 0.5

        ensemble = SequentialEnsemble(
            [expert(0), expert(2), expert(4)], [first_router, second_router], 0.5, 128
        )
        features = np.arange(3, dtype="float32").reshape(-1, 1)
        np.testing.assert_array_equal(ensemble.predict(features).ravel(), [1, 1, 4])
        self.assertEqual(calls, [[0, 1, 2], [0, 2], [2]])
        np.testing.assert_array_equal(ensemble.predict_full(features).ravel(), [2, 3, 4])


if __name__ == "__main__":
    unittest.main()
