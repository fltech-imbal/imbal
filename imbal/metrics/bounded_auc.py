from keras.src.metrics import AUC
from numpy.typing import NDArray

from keras.src import ops
from keras.src.metrics import metrics_utils
import tensorflow as tf

class BoundedAUC(AUC):
    """
    An extension of `TensorFlow's AUC class <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/AUC>`_
    which allows for limiting the AUC graph with an optional x minimum, x maximum,
    y minimum, and y maximum.

    For the best approximation of the real AUC, :code:`predictions` should be
    distributed approximately uniformly in the range :code:`[0, 1]` (if
    :code:`from_logits=False`). The quality of the AUC approximation may be poor if
    this is not the case. Setting :code:`summation_method` to 'minoring' or 'majoring'
    can help quantify the error in the approximation by providing lower or upper
    bound estimate of the AUC.

    If :code:`sample_weight` is :code:`None`, weights default to 1.
    Use :code:`sample_weight` of 0 to mask values.

    For use in TensorFlow's :code:`model.compile` function, this class
    can be passed as a class instance or as any of the following string type
    aliases:

    * :code:`"BoundedAUC"`
    * :code:`"bounded_auc"`

    **Note:** Where appropriate, documentation for functions from :code:`tf.keras.Metric` has been
    overridden to be more descriptive. Any other non-descriptive documentation of individual functions
    on this page is due to a lack of documentation in TensorFlow's original source code. Still, TensorFlow's
    documentation and source code for the :code:`Metric` class can be found `here <https://www.tensorflow.org/api_docs/python/tf/keras/Metric>`_.

    Args:
        num_thresholds: Optional, default :code:`200`. The number of thresholds to
            use when discretizing the roc curve. Values must be > 1.
        min_x: Optional. A float between :code:`0` and :code:`1`, specifying the minimum
            x  value at which AUC should be computed. Setting this value should be used for debugging
            purposes only, as ignoring low x values of AUC is generally undesireable.
        max_x: Optional. A float between :code:`0` and :code:`1`, specifying the maximum
            x value at which AUC should be computed.
        min_y: Optional. A float between :code:`0` and :code:`1`, specifying the minimum
            y value at which AUC should be computed.
        max_y: Optional. A float between :code:`0` and :code:`1`, specifying the maximum
            y value at which AUC should be computed. Setting this value should be used for debugging
            purposes only, as ignoring high y values of AUC is generally undesireable.
        curve: Optional, default :code:`'PR'` Specifies the name of the curve to be computed,
            :code:`'ROC'` (default) or :code:`'PR'` for the Precision-Recall-curve.
        summation_method: Optional. Specifies the `Riemann summation method
              <https://en.wikipedia.org/wiki/Riemann_sum>`_ used.
              'interpolation' (default) applies mid-point summation scheme for
              :code:`ROC`.  For PR-AUC, interpolates (true/false) positives but not
              the ratio that is precision (see Davis & Goadrich 2006 for
              details); 'minoring' applies left summation for increasing
              intervals and right summation for decreasing intervals; 'majoring'
              does the opposite.
        name: Optional. String name of the metric instance.
        dtype: Optional. Data type of the metric result.
        thresholds: Optional. A list of floating point values to use as the
            thresholds for discretizing the curve. If set, the :code:`num_thresholds`
            parameter is ignored. Values should be in :code:`[0, 1]`. Endpoint
            thresholds equal to {:code:`-epsilon`, :code:`1+epsilon`} for a small positive
            epsilon value will be automatically included with these to correctly
            handle predictions equal to exactly 0 or 1.
        multi_label: boolean indicating whether multilabel data should be
            treated as such, wherein AUC is computed separately for each label
            and then averaged across labels, or (when :code:`False`) if the data
            should be flattened into a single label before AUC computation. In
            the latter case, when multilabel data is passed to AUC, each
            label-prediction pair is treated as an individual data point. Should
            be set to :code:`False` for multi-class data.
        num_labels: Optional. The number of labels, used when :code:`multi_label` is
            True. If `num_labels` is not specified, then state variables get
            created on the first call to :code:`update_state`.
        label_weights: Optional. List, array, or tensor of non-negative weights
            used to compute AUCs for multilabel data. When :code:`multi_label` is
            True, the weights are applied to the individual label AUCs when they
            are averaged to produce the multi-label AUC. When it's False, they
            are used to weight the individual label predictions in computing the
            confusion matrix on the flattened data. Note that this is unlike
            :code:`class_weights` in that :code:`class_weights` weights the example
            depending on the value of its label, whereas :code:`label_weights` depends
            only on the index of that label before flattening; therefore
            :code:`label_weights` should not be used for multi-class data.
        from_logits: boolean indicating whether the predictions (:code:`y_pred` in
            :code:`update_state`) are probabilities or sigmoid logits. As a rule of thumb,
            when using a keras loss, the :code:`from_logits` constructor argument of the
            loss should match the AUC :code:`from_logits` constructor argument.

    Example:

     .. code-block:: python

            >>> m = keras.metrics.BoundedAUC(num_thresholds=3)
            >>> m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9])
            >>> # threshold values are [0 - 1e-7, 0.5, 1 + 1e-7]
            >>> # tp = [2, 1, 0], fp = [2, 0, 0], fn = [0, 1, 2], tn = [0, 2, 2]
            >>> # tp_rate = recall = [1, 0.5, 0], fp_rate = [1, 0, 0]
            >>> # auc = ((((1 + 0.5) / 2) * (1 - 0)) + (((0.5 + 0) / 2) * (0 - 0)))
            >>> #     = 0.75
            >>> m.result()
            0.75

     .. code-block:: python

            >>> m.reset_state()
            >>> m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9],
            ...                sample_weight=[1, 0, 0, 1])
            >>> m.result()
            1.0

    Usage with `compile()` API:

    .. code-block:: python

        # Reports the AUC of a model outputting a probability.
        model.compile(optimizer='sgd',
                      loss=keras.losses.BinaryCrossentropy(),
                      metrics=[BoundedAUC()])

        # Reports the AUC of a model outputting a logit.
        model.compile(optimizer='sgd',
                      loss=keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=[BoundedAUC(from_logits=True)])
    """
    def __init__(self,
        *args,
        thresholds = None,
        num_thresholds = 200,
        x_min = None,
        x_max = None,
        y_min = None,
        y_max = None,
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

        self._x_min = x_min
        self._x_max = x_max
        self._y_min = y_min
        self._y_max = y_max
        self._direction = 'up'

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