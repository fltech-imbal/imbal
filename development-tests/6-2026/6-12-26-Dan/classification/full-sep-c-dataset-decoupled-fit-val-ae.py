import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from tensorflow.keras import layers
import os

import imbal

seed = 42
tf.keras.utils.set_random_seed(
    seed
)

target_column = "ln_peak_intensity"

max_epochs = 500
batch_size = 32

# ----------------------------
# Data
# ----------------------------
train_data = pd.read_csv("../../../../tutorials/data/SEP-C/sep_10mev_training_classification.csv")
test_data  = pd.read_csv("../../../../tutorials/data/SEP-C/sep_10mev_testing_classification.csv")

y_train = train_data[target_column].values.reshape(-1, 1).astype("float32")
y_test  = test_data[target_column].values.reshape(-1, 1).astype("float32")

x_train = train_data.drop(columns=[target_column]).values.astype(np.float32)
x_test  = test_data.drop(columns=[target_column]).values.astype(np.float32)

# ----------------------------
# Model
# ----------------------------
def build_model(input_shape: int) -> imbal.classification.Model:
    inputs = keras.Input(shape=(input_shape,), name="features")
    hidden1 = layers.Dense(18, activation="relu", name="hidden_layer1")(inputs)
    hidden2 = layers.Dense(12, activation="relu", name="hidden_layer2")(hidden1)
    hidden3 = layers.Dense(8, activation="relu", name="hidden_layer3")(hidden2)
    hidden4 = layers.Dense(6, activation="relu", name="hidden_layer4")(hidden3)
    flatten = layers.Flatten()(hidden4)
    outputs = layers.Dense(1, activation="sigmoid", name="output_layer")(flatten)
    built_model = imbal.classification.Model(inputs=inputs, outputs=outputs, name="sep_model")
    return built_model

MODEL_SAVE_PATH = "saved_models/decoupled-fit-model-val-ae.keras"
LOAD_SAVED_MODEL = True

if LOAD_SAVED_MODEL and os.path.exists(MODEL_SAVE_PATH):
    print(f'Loading saved binary classification model from {MODEL_SAVE_PATH}')
    model = keras.models.load_model(
        MODEL_SAVE_PATH,
        custom_objects={'Model': imbal.classification.Model}
    )
else:
    model = build_model(x_train.shape[1])

    # ----------------------------
    # Validation Set
    # ----------------------------
    (x_train, y_train), (x_val, y_val) =  imbal.classification.split(x_train, y_train, test_size=0.2, seed=seed)

    # ----------------------------
    # Training
    # ----------------------------
    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=[tf.keras.metrics.F1Score(threshold=0.5, name="F1Score"),
                           imbal.metrics.HeidkeSkillScore(threshold=0.5, name="HSS")],
                  generate_decoder_branch=True,
                  )

    PATIENCE = 30

    model.cRT_fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val.reshape(-1, 1)),
        batch_size=batch_size,
        epochs=max_epochs,
        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)]
    )

    # OPTIONAL: Use custom class weights during training
    # Dictionary mapping classes to weights. In this case, 9:1 ratio of common:rare samples,
    # making rare samples more important to the model loss function than with standard sampling.
    # In this case, rare samples will contribute 10% of the loss per epoch, while common samples contribute 90%.
    # NOTE: Comment above call before running the below call.

    # weight pairs represent [common_class_weight, rare_class_weight]
    class_weight_candidates = [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]]

    model.cRT_fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val.reshape(-1, 1)),
        class_weight=class_weight_candidates,
        batch_size=batch_size,
        epochs=max_epochs,
        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)],
        verbose_imbal=2,
    )

    model.save(MODEL_SAVE_PATH)

    import json

    with open("saved_models/best_params_decoupled_fit-val-ae.json", "w") as f:
        json.dump({
            "best_weight_index": int(model.best_weight_index),
            "best_class_weights": [float(x) for x in model.best_class_weights],
            "best_decision_threshold": (
                float(model.best_decision_threshold)
                if model.best_decision_threshold is not None
                else None
            )
        }, f, indent=4)


# ----------------------------
# Evaluation
# ----------------------------
results = model.evaluate(x_test, y_test)
loss, f1_score, hss = results

print(f"Test Loss: {loss:.4f}")
print(f"Test F1Score: {f1_score:.4f}")
print(f"Test HSS: {hss:.4f}")

if model.best_decision_threshold is not None:
    best_threshold = model.best_decision_threshold
    test_predictions = model.predict(x_test)
    test_predictions = test_predictions.reshape(-1, 1)
    test_predictions = (test_predictions > best_threshold).astype(np.float32)

    best_threshold = model.best_decision_threshold
    hss = imbal.metrics.HeidkeSkillScore(threshold=best_threshold)
    hss.update_state(y_test, test_predictions)

    f1 = keras.metrics.F1Score(threshold=best_threshold)
    f1.update_state(y_test, test_predictions)

    print(
        f'Best found threshold: {model.best_decision_threshold}\n'
        f'F1Score using Best Threshold: {f1.result()[0]:.4f}\n'
        f'HSS using Best Threshold: {hss.result()[0]:.4f}\n'
    )
