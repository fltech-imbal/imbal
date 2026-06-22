"""
Import packages
"""
import imbal
import tensorflow as tf
import tensorflow_probability as tfp
import keras
from keras import layers
import numpy as np
import pandas as pd
from tools import FitType, load_sep_ec_data, plot_representation_space

"""
Set script parameters
"""

# tf.config.run_functions_eagerly(True)

ALPHA = 0
MANUALLY_SORT_EVERY_BATCH = False

LEARNING_RATE = 2e-5
FIT = FitType.BALANCED
VALIDATION_DATA = False
AE = False
SINGLE_WEIGHT_ALPHA = 1
WEIGHT_CANDIDATES = False

REPRESENTATION_LAYER_INDEX = -2
EARLY_STOPPING_PATIENCE = 20
EPOCHS = 1000 if VALIDATION_DATA else 1000

DATA_PATH = 'sep_e_no_electron_log_normalized'

# Will be mostly left unchanged
STRATIFY = True
BATCH_SIZE = 2048
KDE_BIN_COUNT=64
SEED = 42

"""
Load data
"""

(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_sep_ec_data(
    f"cleaned-SEP-EC-data/{DATA_PATH}",
)
"""
Build model
"""

if FIT == FitType.REGULAR:
    WEIGHT_CANDIDATES = False

# tf.keras.utils.set_random_seed(
#     SEED
# )

LAYER_DIMS = [128, 128, 64, 64, 32, 32, 32, 2]

inputs = keras.Input(shape=(x_train.shape[1],))

x = inputs
representation_layer = None
for index, num_units in enumerate(LAYER_DIMS):
    if index == len(LAYER_DIMS) - 1:
        x = layers.Dense(num_units)(x)
        representation_layer = x
    else:
        x = layers.Dense(num_units, activation='relu', kernel_initializer='he_normal')(x)

outputs = layers.Dense(1)(x)

model = keras.Model(inputs=inputs, outputs=[outputs, representation_layer], name="SEP_EC")

"""
Generate sample densities
"""

# fit_function = model.fit
# if FIT == FitType.BALANCED:
#     fit_function = model.balanced_fit
# if FIT == FitType.DECOUPLED:
#     fit_function = model.rRT_fit

if VALIDATION_DATA:
    kde_bandwidth = imbal.regression.fit_kde(
        y_train,
        bin_count=KDE_BIN_COUNT
    )

    sample_densities = imbal.regression.get_sample_densities(
        y_train,
        kde_bandwidth,
    )
    sample_weights = imbal.regression.reciprocal_importance(sample_densities, alpha=[0.1*(i+1) for i in range(10)] if WEIGHT_CANDIDATES else SINGLE_WEIGHT_ALPHA)
    val_densities = imbal.regression.get_sample_densities(
        y_val,
        kde_bandwidth,
        distribution=y_train
    )
    w_val = imbal.regression.reciprocal_importance(val_densities, alpha=[0.1*(i+1) for i in range(10)] if WEIGHT_CANDIDATES else SINGLE_WEIGHT_ALPHA)
    val_data = (x_val, y_val, w_val)
else:
    x_train = np.concatenate((x_train, x_val))
    y_train = np.concatenate((y_train, y_val))

    kde_bandwidth = imbal.regression.fit_kde(
        y_train,
        bin_count=KDE_BIN_COUNT
    )

    sample_densities = imbal.regression.get_sample_densities(
        y_train,
        kde_bandwidth,
    )
    sample_weights = imbal.regression.reciprocal_importance(sample_densities)

    val_data = None

# imbal.regression.plot_kde_1d(
#     y_train,
#     kde_bandwidth,
#     bin_count=KDE_BIN_COUNT
# )

# fit_function(
#     x_train,
#     y_train,
#     sample_weight=sample_weights,
#     validation_data=val_data,
#     epochs=EPOCHS,
#     batch_size=BATCH_SIZE,
#     candidate_evaluation_sample_weight=(val_data[2][-1] if VALIDATION_DATA else sample_weights[-1]) if WEIGHT_CANDIDATES else None,
#     callbacks=[keras.callbacks.EarlyStopping(patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True)] if VALIDATION_DATA else None
# )

def compute_regression_loss(y_true, y_pred, sample_weight=None):
    if sample_weight is None:
        mse = tf.keras.losses.MeanSquaredError()
        return mse(y_true, y_pred)
    else:
        return tf.reduce_sum(tf.square(y_true - y_pred) * sample_weight) / tf.reduce_sum(sample_weight)

def compute_representation_loss(labels, representations, weight=None):
    EPSILON = 1e-9
    #
    # if weight is None:
    #     weight = tf.ones(len(labels))
    #
    # tf.debugging.check_numerics(labels, "labels")
    # tf.debugging.check_numerics(representations, "representations")
    #
    # distance_to_next_label = tf.linalg.norm(labels[1:] - labels[:-1], axis=-1) + EPSILON
    # distance_to_last_label = tf.linalg.norm(labels[1:-1] - labels[-1], axis=-1) + EPSILON
    distance_to_first_label = tf.linalg.norm(labels[1:] - labels[0], axis=-1) + EPSILON
    #
    # distance_to_next_representation = tf.linalg.norm(representations[1:] - representations[:-1], axis=-1) + EPSILON
    # distance_to_last_representation = tf.linalg.norm(representations[1:-1] - representations[-1], axis=-1) + EPSILON
    distance_to_first_representation = tf.linalg.norm(representations[1:] - representations[0], axis=-1) + EPSILON
    #
    # combined_label_distances = tf.concat([distance_to_next_label, distance_to_last_label, distance_to_first_label], axis=0)
    # combined_representation_distances = tf.concat([distance_to_next_representation, distance_to_last_representation, distance_to_first_representation], axis=0)
    # combined_weights = tf.concat([weight[1:], weight[1:-1], weight[1:]], axis=0)
    #
    # return tf.reduce_sum(tf.square(combined_representation_distances - combined_label_distances)*combined_weights) / tf.reduce_sum(combined_weights)

    labels_reshaped = tf.reshape(labels, (-1, 1))
    extended_representations = tf.concat([representations, labels_reshaped], axis=1)

    return 1 - tf.reduce_mean(tf.math.abs(tfp.stats.correlation(extended_representations))) #+ tf.reduce_mean(tf.square(distance_to_first_label - distance_to_first_representation))

def combined_loss(
    labels,
    predictions,
    representations,
    sample_weight=None,
    alpha=1
):
    regression_loss = compute_regression_loss(labels, predictions, sample_weight=sample_weight)
    representation_loss = compute_representation_loss(labels, representations, weight=sample_weight)
    total_loss = regression_loss + representation_loss * alpha
    return total_loss, regression_loss, representation_loss

print(np.shape(x_train))
print(np.shape(y_train))
print(np.shape(sample_weights))

train_dataset = tf.data.Dataset.from_tensor_slices(
    (
        x_train,
        y_train,
        None if FIT == FitType.REGULAR else sample_weights,
     )
).shuffle(buffer_size=1000, reshuffle_each_iteration=True).batch(2048)

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

@tf.function
def train_step(x, y, w):
    with tf.GradientTape() as tape:
        predictions, representations = model(x, training=True)

        tf.debugging.check_numerics(predictions, "outputs")
        tf.debugging.check_numerics(representations, "reps")

        total_loss, regression_loss, representation_loss = combined_loss(
            y,
            predictions,
            representations,
            alpha=ALPHA,
            sample_weight=w
        )

    gradients = tape.gradient(total_loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    return total_loss, regression_loss, representation_loss


for epoch in range(EPOCHS):
    print(f"\nStart of epoch {epoch + 1}")

    for step, (x_batch, y_batch, w_batch) in enumerate(train_dataset):

        if MANUALLY_SORT_EVERY_BATCH:
            sort_indices = tf.argsort(tf.squeeze(y_batch))
            x_batch = tf.gather(x_batch, sort_indices)
            y_batch = tf.gather(y_batch, sort_indices)

        loss, reg_loss, rep_loss = train_step(x_batch, y_batch, w_batch)
        if step % 10 == 0:
            print(
                f"  Step {step:3d} | "
                f"Total: {float(loss):.4f} | "
                f"Regression: {float(reg_loss):.4f} | "
                f"Representation: {float(rep_loss):.4f}"
            )

predictions, representations = model.predict(x_train)
predictions = predictions.reshape(-1)
y_train = y_train.reshape(-1)

plot_representation_space(
    representations,
    y_train,
    0,
    1
)

from sklearn.linear_model import LinearRegression
lin_reg_model = LinearRegression()
lin_reg_model.fit(predictions.reshape(-1, 1), y_train.reshape(-1))

imbal.regression.plot_true_vs_predictions(
    y_train,
    predictions
)

adjusted_predictions = lin_reg_model.predict(predictions.reshape(-1, 1))

imbal.regression.plot_true_vs_predictions(
    y_train,
    adjusted_predictions
)

predictions, representations = model.predict(x_test)
predictions = predictions.reshape(-1)
y_test = y_test.reshape(-1)

common_sample_mask = (y_test > -0.5) & (y_test < 0.5)
common_predictions = predictions[common_sample_mask]
rare_predictions = predictions[~common_sample_mask]
common_labels = y_test[common_sample_mask]
rare_labels = y_test[~common_sample_mask]

mae = np.mean(np.abs(predictions - y_test))
common_mae = np.mean(np.abs(common_predictions - common_labels))
rare_mae = np.mean(np.abs(rare_predictions - rare_labels))

imbal.regression.plot_true_vs_predictions(
    y_test,
    predictions,
    title=f'Common MAE: {common_mae:.4f}, Rare MAE: {rare_mae:.4f}, AORE: {(mae + rare_mae)/2:.4f}{f", Alpha: {[0.1*(i+1) for i in range(10)][model.best_weight_index]:.1f}" if WEIGHT_CANDIDATES else ""}',
    save_figure=f"results/{DATA_PATH}_{FIT.name.lower()}_{'w' if VALIDATION_DATA or AE else ''}{'_validation' if VALIDATION_DATA else ''}{'_ae' if AE else ''}.png"
)

plot_representation_space(
    representations,
    y_test,
    0,
    1
)

imbal.regression.tsne_visualization(
    model,
    x_test,
    y_test,
    gradient='jet'
)

adjusted_predictions = lin_reg_model.predict(predictions.reshape(-1, 1))

common_sample_mask = (y_test > -0.5) & (y_test < 0.5)
common_predictions = adjusted_predictions[common_sample_mask]
rare_predictions = adjusted_predictions[~common_sample_mask]
common_labels = y_test[common_sample_mask]
rare_labels = y_test[~common_sample_mask]

mae = np.mean(np.abs(adjusted_predictions - y_test))
common_mae = np.mean(np.abs(common_predictions - common_labels))
rare_mae = np.mean(np.abs(rare_predictions - rare_labels))

imbal.regression.plot_true_vs_predictions(
    y_test,
    adjusted_predictions,
    title=f'Common MAE: {common_mae:.4f}, Rare MAE: {rare_mae:.4f}, AORE: {(mae + rare_mae)/2:.4f}{f", Alpha: {[0.1*(i+1) for i in range(10)][model.best_weight_index]:.1f}" if WEIGHT_CANDIDATES else ""}',
)