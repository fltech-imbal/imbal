from keras.src.metrics import AUC
from typing import List
from numpy.typing import NDArray

from keras.src import ops
from keras.src.metrics import metrics_utils
import tensorflow as tf

class LimitedAUC(AUC):
    def __init__(self,
        *args,
        thresholds : List[float] | NDArray | None = None,
        num_thresholds : int = 200,
        x_min : float | None = None,
        x_max : float | None = None,
        y_min : float | None = None,
        y_max : float | None = None,
        **kwargs
    ) -> None:

        if thresholds is None:
            if num_thresholds <= 1:
                raise ValueError(
                    "Argument `num_thresholds` must be an integer > 1. "
                    f"Received: num_thresholds={num_thresholds}"
                )

            self.num_thresholds = num_thresholds
            thresholds = [
                (i + 1) * 1.0 / (num_thresholds - 1)
                for i in range(num_thresholds - 2)
            ]
            self._thresholds_distributed_evenly = True

        if x_min is not None and x_min not in thresholds:
            thresholds.append(x_min)
        if x_max is not None and x_max not in thresholds:
            thresholds.append(x_max)

        self._x_min = x_min
        self._x_max = x_max
        self._y_min = y_min
        self._y_max = y_max

        super().__init__(*args, thresholds=thresholds, **kwargs)

    def result(self) -> NDArray:
        if (
                self.curve == metrics_utils.AUCCurve.PR
                and self.summation_method
                == metrics_utils.AUCSummationMethod.INTERPOLATION
        ):
            # This use case is different and is handled separately.
            return self.interpolate_pr_auc()

        # Set `x` and `y` values for the curves based on `curve` config.
        recall = ops.divide_no_nan(
            self.true_positives,
            ops.add(self.true_positives, self.false_negatives),
        )
        if self.curve == metrics_utils.AUCCurve.ROC:
            fp_rate = ops.divide_no_nan(
                self.false_positives,
                ops.add(self.false_positives, self.true_negatives),
            )
            x = fp_rate
            y = recall
        else:  # curve == 'PR'.
            precision = ops.divide_no_nan(
                self.true_positives,
                ops.add(self.true_positives, self.false_positives),
            )
            x = recall
            y = precision

        # Find the rectangle heights based on `summation_method`.
        if (
                self.summation_method
                == metrics_utils.AUCSummationMethod.INTERPOLATION
        ):
            # Note: the case ('PR', 'interpolation') has been handled above.
            heights = ops.divide(
                ops.add(y[: self.num_thresholds - 1], y[1:]), 2.0
            )
        elif self.summation_method == metrics_utils.AUCSummationMethod.MINORING:
            heights = ops.minimum(y[: self.num_thresholds - 1], y[1:])
        # self.summation_method = metrics_utils.AUCSummationMethod.MAJORING:
        else:
            heights = ops.maximum(y[: self.num_thresholds - 1], y[1:])

        if self._x_min is not None:
            mask = x[:-1] >= self._x_min
            x = x[tf.concat([[True], mask], axis=0)]
            heights = tf.boolean_mask(heights, mask)
        if self._x_max is not None:
            mask = x[1:] <= self._x_max
            x = x[tf.concat([mask, [True]], axis=0)]
            heights = tf.boolean_mask(heights, mask)
        if self._y_min is not None:
            mask = heights >= self._y_min
            x = x[tf.concat([[True], mask], axis=0)]
            heights = tf.boolean_mask(heights, mask)
        if self._y_max is not None:
            mask = heights <= self._y_max
            x = x[tf.concat([mask, [True]], axis=0)]
            heights = tf.boolean_mask(heights, mask)

        # if tf.size(x) < 2 or tf.size(heights) < 1:
        #     return tf.constant(0.0)

        limited_thresholds = tf.size(x)
        # Sum up the areas of all the rectangles.
        riemann_terms = ops.multiply(
            ops.subtract(x[: limited_thresholds - 1], x[1:]), heights
        )
        if self.multi_label:
            by_label_auc = ops.sum(riemann_terms, axis=0)

            if self.label_weights is None:
                # Unweighted average of the label AUCs.
                return ops.mean(by_label_auc)
            else:
                # Weighted average of the label AUCs.
                return ops.divide_no_nan(
                    ops.sum(ops.multiply(by_label_auc, self.label_weights)),
                    ops.sum(self.label_weights),
                )
        else:
            return ops.sum(riemann_terms)