import imbal
import imbal.util.backend as backend
from imbal.util.backend.constants import ModelType

class Model(backend.Model):
    """
    Package location: :code:`imbal.regression.Model`

    An extension of `Keras's Model class <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`_ that
    can perform alternative types of model fits which aim to address common issues when applying
    machine learning algorithms to imbalanced/long-tail data.

    This :code:`imbal.regression.Model` class is intended for use only on regression problems.

    For a comparison of how these fit methods perform compared to a standard fit on imbalanced/long-tail
    data, see the information at the bottom of this page.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mode_enum = ModelType.REGRESSION
        self._mode_subpackage = imbal.regression

    def balanced_fit(
        self,
        x=None,
        y=None,
        sample_density=None,
        sample_weight=None,
        candidate_evaluation_sample_weight=None,
        validation_data=None,
        validation_densities=None,
        validation_split=None,
        epochs=1,
        batch_size=32,
        shuffle=True,
        stratify_batches=False,
        verbose_imbal=1,
        **kwargs
    ):
        """
        Performs a density-balanced fit, where by default, each instance is equally weighted
        in a manner that is inversely proportional to the instance's probability density (reciprocal importance)
        during the model fitting process.

        Args:
            x: Optional, default :code:`None` (Same as `model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_).
                A NumPy array of data points, arranged as a column vector
            y: Optional, default :code:`None` (Same as `model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_).
            sample_weight: Optional, default :code:`None`. A list of sample weights.
                Optionally, a 2D list of sample weights can be provided, in which case
                the model will be fit once for each list of sample weights provided, with the final model weights being set to the
                final weights from the fit which best optimizes the first metric passed during :code:`Model.compile` (default :code:`keras.metrics.MeanAbsoluteError`).
                See "Using Multiple Weight Candidates" below for more details.
            sample_density: Optional, default :code:`None`. Must be specified if :code:`sample_weight` is :code:`None`.
                A list of sample probability densities, which will be
                used to generate sample weights using reciprocal importance.
                If unspecified, :code:`sample_weight` must be specified.
            candidate_evaluation_sample_weight: Optional, default :code:`None`.
                When performing a fit with multiple weights candidates, determines what sample weighting should be used when computing
                the metric to compare weight candidate performance (See "Using Multiple Weight Candidates below for more details).
            validation_data: Optional, default :code:`None` (Same as `model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_).
                The data used to validate the model during training.
                See `Tensorflow's model.fit documentation <https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile>`_.
            validation_densities: : Optional, default :code:`None`. A list of sample probability densities for
                the provided :code:`validation_data`. Only required if :code:`validation_data` is provided
                without sample weights.
            validation_split: Optional, default :code:`None`. A float value representing the proportion of the
                    provided training data to split off into a separate dataset used for model validation.
            epochs: Optional, default :code:`1`. (Same as `model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_).
                The number of epochs to train the model for.
            batch_size: Optional, default :code:`None` (Same as `model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_).
                The batch size to use during training.
            shuffle: Optional, default :code:`True` (Same as `model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_).
                Whether to shuffle the data before each epoch.
            stratify_batches: Optional, default :code:`True`. Whether to stratify data batch-wise during training.
                See :doc:`DatasetWithBatching </imbal/classification/dataset_with_batching>` for details.
                Only used when :code:`multi_output` is :code:`True`.
            verbose_imbal: Optional, default :code:`1`. The verbosity level of debug messages associated with
                imbal functionality. When set to :code:`0`, no imbal debug messages will print. When set to :code:`1`,
                general debug messages will be printed. When greater than :code:`1`, all messages will be printed.
            **kwargs: Any additional keyword arguments accepted by `TensorFlow's model.fit function <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_

        Returns:
            A History object. Its History.history attribute is a record of training loss values and metrics values
            at successive epochs, as well as validation loss values and validation metrics values (if applicable).

        Example:

        .. code-block:: python

            # Assume data has been loaded as '(x_train, y_train), (x_test, y_test)', and
            # we have a compiled model

            kde_bandwidth = imbal.regression.fit_kde(y_train)
            densities = imbal.regression.get_sample_densities(y_train, kde_bandwidth)

            model.balanced_fit(
                x_train,
                y_train,
                sample_densities=densities,
                batch_size=64,
                validation_split=0.2
            )

        Example using multiple class weight candidates:

        .. code-block:: python

            # Assume data has been loaded as '(x_train, y_train), (x_test, y_test)', and
            # we have a compiled model

            kde_bandwidth = imbal.regression.fit_kde(y_train)
            densities = imbal.regression.get_sample_densities(y_train, kde_bandwidth)
            weight_candidates = reciprocal_importance(densities, alpha=[0.2, 0.5, 0.8, 1.0])

            model.balanced_fit(
                x_train,
                y_train,
                sample_weight=weight_candidates,
                batch_size=64,
                validation_split=0.2
            )

            best_threshold = model.best_metric_threshold
            best_sample_weights = model.best_sample_weights
        """
        return super()._balanced_fit(
            x=x,
            y=y,
            sample_density=sample_density,
            sample_weight=sample_weight,
            candidate_evaluation_sample_weight=candidate_evaluation_sample_weight,
            validation_data=validation_data,
            validation_densities=validation_densities,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            stratify_batches=stratify_batches,
            verbose_imbal=verbose_imbal,
            **kwargs
        )

    def rRT_fit(
        self,
        x=None,
        y=None,
        sample_weight=None,
        sample_density=None,
        candidate_evaluation_sample_weight=None,
        validation_data=None,
        validation_split=None,
        validation_densities=None,
        epochs=1,
        batch_size=32,
        shuffle=True,
        stratify_batches=False,
        verbose_imbal=1,
        **kwargs
    ):
        """
        Performs a regressor re-training (rRT) fit, inspired by the
        classifier re-training (cRT) method as described in
        `this paper by Kang et al. (ICLR 2020) <https://arxiv.org/abs/1910.09217>`_.

        Args:
            x: Optional, default :code:`None` (Same as `model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_).
                A Numpy array of data points, arranged as a column vector
            y: Optional, default :code:`None` (Same as `model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_).
                A Numpy array of labels, arranged as a row vector, column vector, or list of one-hot vectors.
            sample_weight: Optional, default :code:`None`. A list of sample weights.
                Optionally, a 2D list of sample weights can be provided, in which case
                the model will be fit once for each list of sample weights provided, with the final model weights being set to the
                final weights from the fit which best optimizes the first metric passed during :code:`Model.compile` (default :code:`keras.metrics.MeanAbsoluteError`).
                See "Using Multiple Weight Candidates" below for more details.
            sample_density: Optional, default :code:`None`. A list of sample probability densities.
                If unspecified, :code:`sample_weight` must be specified.
            candidate_evaluation_sample_weight: Optional, default :code:`None`.
                When performing a fit with multiple weights candidates, determines what sample weighting should be used when computing
                the metric to compare weight candidate performance (See "Using Multiple Weight Candidates below for more details).
            validation_data: Optional, default :code:`None` (Same as `model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_).
                The data used to validate the model during training.
                See `Tensorflow's model.fit documentation <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_.
            validation_densities: : Optional, default :code:`None`. A list of sample probability densities for
                the provided :code:`validation_data`. Only required if :code:`validation_data` is provided
                without sample weights.
            validation_split: Optional, default :code:`None`. A float value representing the proportion of the
                provided training data to split off into a separate dataset used for model validation.
            epochs: Optional, default :code:`1` (Same as `model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_).
                The number of epochs to train for. If an :code:`int`,
                the provided number of epochs will be used during the first stage, and halved for the second stage.
                If a tuple or list of length 2, the value in the first index will be used as the number of
                epochs in the first stage of training, and the second index for the number of epochs
                in the second stage of training.
            batch_size: Optional, default :code:`32` (Same as `model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_).
                The batch size to use during training.
            shuffle: Optional, default :code:`True` (Same as `model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_).
                Whether to shuffle the data before each epoch.
            stratify_batches: Optional, default :code:`True`. Whether to stratify data batch-wise during training.
                See :doc:`DatasetWithBatching </imbal/classification/dataset_with_batching>` for details.
            verbose_imbal: Optional, default :code:`1`. The verbosity level of debug messages associated with
                imbal functionality. When set to :code:`0`, no imbal debug messages will print. When set to :code:`1`,
                general debug messages will be printed. When greater than :code:`1`, all messages will be printed.
            **kwargs: Any additional keyword arguments accepted by `TensorFlow's model.fit function <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_

        Returns:
            A tuple (s1, s2) of History objects. Its History.history attribute is a record of training loss values
            and metrics values at successive epochs, as well as validation loss values and validation metrics
            values (if applicable).

        Example:

        .. code-block:: python

            # Assume data has been loaded as '(x_train, y_train), (x_test, y_test)', and
            # we have a compiled model

            kde_bandwidth = imbal.regression.fit_kde(y_train)
            densities = imbal.regression.get_sample_densities(y_train, kde_bandwidth)

            model.rRT_fit(
                x_train,
                y_train,
                sample_densities=densities,
                batch_size=64,
                validation_split=0.2
            )

        Example using multiple class weight candidates:

        .. code-block:: python

            # Assume data has been loaded as '(x_train, y_train), (x_test, y_test)', and
            # we have a compiled model

            kde_bandwidth = imbal.regression.fit_kde(y_train)
            densities = imbal.regression.get_sample_densities(y_train, kde_bandwidth)
            weight_candidates = reciprocal_importance(densities, alpha=[0.2, 0.5, 0.8, 1.0])

            model.rRT_fit(
                x_train,
                y_train,
                sample_weight=weight_candidates,
                batch_size=64,
                validation_split=0.2
            )

            best_threshold = model.best_metric_threshold
            best_sample_weights = model.best_sample_weights

        """
        return self._decoupled_fit(
            x=x,
            y=y,
            sample_weight=sample_weight,
            sample_density=sample_density,
            candidate_evaluation_sample_weight=candidate_evaluation_sample_weight,
            validation_data=validation_data,
            validation_split=validation_split,
            validation_densities=validation_densities,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            stratify_batches=stratify_batches,
            verbose_imbal=verbose_imbal,
            **kwargs
        )