"""Expert and router networks, conditional prediction, and model archives."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def build_expert(expert_number, input_count, settings):
    expert_name = f"expert_{expert_number}"
    inputs = keras.Input(shape=(input_count,), name=f"{expert_name}_input")
    hidden = inputs

    for layer_number, units in enumerate((18, 12, 8), start=1):
        hidden = layers.Dense(
            units, activation="relu", name=f"{expert_name}_dense_{layer_number}"
        )(hidden)

    representation = layers.Dense(
        6, activation="relu", name=f"{expert_name}_representation"
    )(hidden)
    prediction = layers.Dense(1, name=f"{expert_name}_prediction")(representation)
    expert = keras.Model(inputs, prediction, name=expert_name)
    expert.compile(optimizer=keras.optimizers.Adam(settings.learning_rate), loss="mse")
    return expert


def build_router(router_number, input_count, settings):
    router_name = f"router_{router_number}"
    inputs = keras.Input(shape=(input_count,), name=f"{router_name}_input")
    hidden = inputs

    for layer_number, units in enumerate((18, 12, 8), start=1):
        hidden = layers.Dense(
            units, activation="relu", name=f"{router_name}_dense_{layer_number}"
        )(hidden)

    probability = layers.Dense(1, activation="sigmoid", name=f"{router_name}_continue")(hidden)
    router = keras.Model(inputs, probability, name=router_name)
    router.compile(
        optimizer=keras.optimizers.Adam(settings.learning_rate), loss="binary_crossentropy"
    )
    return router


def prefix_prediction(expert_predictions, residual):
    if residual:
        return expert_predictions.sum(axis=1, keepdims=True)
    return expert_predictions.mean(axis=1, keepdims=True)


def router_inputs(features, expert_predictions, residual=False):
    """Append individual outputs, current prefix prediction, variance, and trust."""

    prediction = prefix_prediction(expert_predictions, residual)
    variance = expert_predictions.var(axis=1, keepdims=True)
    return np.column_stack((
        features, expert_predictions, prediction, variance, np.zeros_like(prediction)
    ))


def predict_in_batches(model, features, batch_size):
    if len(features) == 0:
        return np.empty((0, 1), dtype="float32")

    batches = [
        np.asarray(model(features[start:start + batch_size], training=False))
        for start in range(0, len(features), batch_size)
    ]
    return np.concatenate(batches, axis=0)


class SequentialEnsemble:
    def __init__(self, experts, routers, router_threshold, prediction_batch_size, residual=False):
        if not experts or len(routers) != len(experts) - 1:
            raise ValueError("A sequential ensemble requires one router between adjacent experts.")

        self.experts = experts
        self.routers = routers
        self.router_threshold = router_threshold
        self.prediction_batch_size = prediction_batch_size
        self.residual = residual

    def predict(self, features, *, return_expert_outputs=False):
        """Predict in row order, optionally returning each visited expert's raw output."""

        features = np.asarray(features, dtype="float32")
        predictions = np.empty((len(features), 1), dtype="float32")
        active_rows = np.arange(len(features))
        active_predictions = np.empty((len(features), 0), dtype="float32")
        if return_expert_outputs:
            expert_outputs = np.full((len(features), len(self.experts)), np.nan, dtype="float32")

        for expert_index, expert in enumerate(self.experts):
            if not len(active_rows):
                break

            active_features = features[active_rows]
            next_predictions = predict_in_batches(expert, active_features, self.prediction_batch_size)
            if return_expert_outputs:
                expert_outputs[active_rows, expert_index] = next_predictions.ravel()
            active_predictions = np.column_stack((active_predictions, next_predictions))
            predictions[active_rows] = prefix_prediction(active_predictions, self.residual)

            if expert_index < len(self.routers):
                inputs = router_inputs(active_features, active_predictions, self.residual)
                probability = predict_in_batches(
                    self.routers[expert_index], inputs, self.prediction_batch_size
                )
                continue_mask = probability.ravel() >= self.router_threshold
                active_rows = active_rows[continue_mask]
                active_predictions = active_predictions[continue_mask]

        if return_expert_outputs:
            return predictions, expert_outputs
        return predictions

    def predict_full(self, features):
        predictions = [
            predict_in_batches(expert, features, self.prediction_batch_size)
            for expert in self.experts
        ]
        return prefix_prediction(np.column_stack(predictions), self.residual)

    def save(self, archive_path, metadata):
        """Publish each seed as one complete bundle of ordinary Keras models."""

        archive_path = Path(archive_path)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(
            metadata, expert_count=len(self.experts), router_threshold=self.router_threshold,
            prediction_batch_size=self.prediction_batch_size, residual=self.residual,
        )

        with TemporaryDirectory(dir=archive_path.parent) as directory:
            temporary_archive = Path(directory) / "ensemble.zip"
            with ZipFile(temporary_archive, "w") as archive:
                archive.writestr("experiment.json", json.dumps(metadata, indent=2, allow_nan=False))
                for model in [*self.experts, *self.routers]:
                    model_path = Path(directory) / f"{model.name}.keras"
                    model.save(model_path)
                    archive.write(model_path, model_path.name)

            temporary_archive.replace(archive_path)

    @classmethod
    def load(cls, archive_path):
        with ZipFile(archive_path) as archive, TemporaryDirectory() as directory:
            metadata = json.loads(archive.read("experiment.json"))

            def load_models(prefix, count):
                models = []
                for number in range(1, count + 1):
                    model_path = Path(directory) / f"{prefix}_{number}.keras"
                    model_path.write_bytes(archive.read(model_path.name))
                    model = keras.models.load_model(model_path, compile=False)
                    model.trainable = False
                    models.append(model)
                return models

            expert_count = metadata["expert_count"]
            ensemble = cls(
                load_models("expert", expert_count), load_models("router", expert_count - 1),
                metadata["router_threshold"], metadata["prediction_batch_size"],
                residual=metadata.get("residual", False),
            )
        return ensemble, metadata
