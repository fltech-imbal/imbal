import tensorflow as tf
import tensorflow_probability as tfp
import keras
from keras import layers
from imbal.regression import tsne_visualization, plot_true_vs_predictions
# import matplotlib as mpl
# mpl.use('QtAgg')  # or can use 'TkAgg', whatever you have/prefer

import matplotlib
matplotlib.use('QtAgg')  # or 'TkAgg'
import matplotlib.pyplot as plt

import numpy as np
from tools.loss_functions import *
from tools import FitType

FIGURE_NAME = 'temp'
LEARNING_RATE = 2e-4
# FIT = FitType.REGULAR
# SINGLE_WEIGHT_ALPHA = 0
# VALIDATION_DATA = True

FIT_MODE = 'freeze'
ALPHA = 1
MSE_FACTOR = 1
EPOCHS = 1000
SECOND_STAGE_EPOCHS = 5000
REPRESENTATION_DIMS = 2
MANUALLY_SORT_EVERY_BATCH = False
UNIT_REPRESENTATION = False
EXTRA_REGRESSOR_LAYERS = False
THREE_D = False
USE_STRICT_REPRESENTATION = True
REPRESENTATION_LOSS_FUNCTION = augmented_pcc
FIGURE_PICKLE_NAME = 'rectified_semicircle_plot'

RATIO_CONSTRAIN = False
RATIO_CONSTRAIN_ALPHA = 0.05

# tf.config.run_functions_eagerly(True)

def build_sep_ec_model(layer_dimensions, representation_layer_dimension):
    inputs = keras.Input(shape=(x_train.shape[1],))

    x = inputs
    for index, num_units in enumerate(layer_dimensions):
        x = layers.Dense(num_units, activation='relu')(x)
        if index == len(layer_dimensions) - 1:
            x = layers.Flatten()(x)


    representation_layer = layers.Dense(representation_layer_dimension, name='representation' if not UNIT_REPRESENTATION else None)(x)
    if UNIT_REPRESENTATION:
        representation_layer = layers.UnitNormalization(name='representation')(representation_layer)

    if EXTRA_REGRESSOR_LAYERS:
        x = layers.Dense(64, activation='relu')(representation_layer)
        output_layer = layers.Dense(1, name='output')(x)
    else:
        output_layer = layers.Dense(1, name='output')(representation_layer)

    model = keras.Model(inputs=inputs, outputs=[output_layer, representation_layer], name="SEP_EC")
    return model

# 2. Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

def compute_regression_loss(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()
    return mse(y_true, y_pred)

def ratio_loss(labels, representations):
    EPSILON = 1e-9
    distance_to_next_label = tf.linalg.norm(labels[1:] - labels[:-1], axis=1) + EPSILON
    distance_to_next_representation = tf.linalg.norm(representations[1:] - representations[:-1], axis=1) + EPSILON

    ratio = tf.reduce_sum(distance_to_next_label) / tf.reduce_sum(distance_to_next_representation)

    # loss_value = (ratio + 1/ratio)**2 - 4
    loss_value = tf.math.log(ratio) ** 2
    return loss_value * RATIO_CONSTRAIN_ALPHA


def combined_loss(
    labels,
    predictions,
    representations,
    alpha=1
):
    regression_loss = compute_regression_loss(labels, predictions)
    representation_loss = REPRESENTATION_LOSS_FUNCTION(labels, representations)
    if RATIO_CONSTRAIN:
        representation_loss += ratio_loss(labels, representations)
    total_loss = regression_loss*MSE_FACTOR + representation_loss * alpha

    return total_loss, regression_loss, representation_loss

# 5. Dummy data — two label tensors, one per output head
x_train = tf.clip_by_value(tf.random.normal((1000, 2)) + tf.random.normal((1000, 2), stddev=0.1), clip_value_min=-3, clip_value_max=3)
y_reg_train = tf.reshape(tf.linalg.norm(x_train, axis=1), (-1, 1))

allowed_labels = tf.reshape((y_reg_train < 3.5), (-1,))
x_train = x_train[allowed_labels]
y_reg_train = y_reg_train[allowed_labels]

label_min = 0
label_max = 3.5

def f(x):
    return 0.5 * (1 - tf.abs(2 * x - 1))

# def snake_representation(labels, representations):
#     normalized_labels = (labels - label_min) / (label_max - label_min)
#     x = normalized_labels + f(normalized_labels) * tf.sin(4 * np.pi * normalized_labels)
#     y = 1 - normalized_labels + f(normalized_labels) * tf.sin(4 * np.pi * normalized_labels)
#     ideal_representations = tf.concat([x, y], axis=1)
#     return tf.reduce_mean(tf.square(tf.norm(ideal_representations - representations, axis=1)))

def semicircle_representation(labels, representations):
    normalized_labels = (labels - label_min) / (label_max - label_min)
    x = tf.cos(np.pi * normalized_labels)
    y = tf.sin(np.pi * normalized_labels)
    ideal_representations = tf.concat([x, y], axis=1)
    return tf.reduce_mean(tf.square(tf.norm(ideal_representations - representations, axis=1)))

def rectified_semicircle_representation(labels, representations):
    normalized_labels = (labels - label_min) / (label_max - label_min)
    x = 1 - 2 * normalized_labels
    y = tf.sin(tf.math.acos(1 - 2 * normalized_labels))
    ideal_representations = tf.concat([x, y], axis=1)
    return tf.reduce_mean(tf.square(tf.norm(ideal_representations - representations, axis=1)))

def linear_representation(labels, representations):
    normalized_labels = (labels - label_min) / (label_max - label_min)
    x = normalized_labels
    y = 2 - 2 * normalized_labels
    ideal_representations = tf.concat([x, y], axis=1)
    return tf.reduce_mean(tf.square(tf.norm(ideal_representations - representations, axis=1)))

def disturbed_linear_representation(labels, representations):
    normalized_labels = (labels - label_min) / (label_max - label_min)
    x = normalized_labels**2
    y = 2 - 2 * normalized_labels**2
    ideal_representations = tf.concat([x, y], axis=1)
    return tf.reduce_mean(tf.square(tf.norm(ideal_representations - representations, axis=1)))

def nonlinear_representation(labels, representations):
    normalized_labels = (labels - label_min) / (label_max - label_min)
    x = 1 - normalized_labels
    y = 1 - 2 *tf.abs(normalized_labels - 0.5)
    ideal_representations = tf.concat([x, y], axis=1)
    return tf.reduce_mean(tf.square(tf.norm(ideal_representations - representations, axis=1)))

if USE_STRICT_REPRESENTATION:
    REPRESENTATION_LOSS_FUNCTION = rectified_semicircle_representation

train_dataset = tf.data.Dataset.from_tensor_slices(
    (x_train, y_reg_train)
).shuffle(buffer_size=1000, reshuffle_each_iteration=True).batch(200)

model = build_sep_ec_model(
    layer_dimensions=[128, 128, 64, 64, 32, 32],
    representation_layer_dimension=REPRESENTATION_DIMS
)

print(model.summary())

@tf.function
def joint_train_step(x, y):
    with tf.GradientTape() as tape:
        predictions, representations = model(x, training=True)

        total_loss, regression_loss, representation_loss = combined_loss(
            y,
            predictions,
            representations,
            alpha=ALPHA
        )

    gradients = tape.gradient(total_loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    return total_loss, regression_loss, representation_loss

@tf.function
def representation_train_step(x, y):
    with tf.GradientTape() as tape:
        predictions, representations = model(x, training=True)

        total_loss, regression_loss, representation_loss = combined_loss(
            y,
            predictions,
            representations,
            alpha=ALPHA
        )

    gradients = tape.gradient(representation_loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    return total_loss, regression_loss, representation_loss

@tf.function
def regression_train_step(x, y):
    with tf.GradientTape() as tape:
        predictions, representations = model(x, training=True)

        total_loss, regression_loss, representation_loss = combined_loss(
            y,
            predictions,
            representations,
            alpha=ALPHA
        )

    gradients = tape.gradient(regression_loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    return total_loss, regression_loss, representation_loss

if FIT_MODE == 'joint':
    for epoch in range(EPOCHS):
        print(f"\nStart of epoch {epoch + 1}")

        for step, (x_batch, y_reg_batch) in enumerate(train_dataset):

            if MANUALLY_SORT_EVERY_BATCH:
                sort_indices = tf.argsort(tf.squeeze(y_reg_batch))
                x_batch = tf.gather(x_batch, sort_indices)
                y_reg_batch = tf.gather(y_reg_batch, sort_indices)

            loss, reg_loss, rep_loss = joint_train_step(x_batch, y_reg_batch)
            if step % 10 == 0:
                print(
                    f"  Step {step:3d} | "
                    f"Total: {float(loss):.4f} | "
                    f"Regression: {float(reg_loss):.4f} | "
                    f"Representation: {float(rep_loss):.4f}"
                )
elif FIT_MODE == 'freeze':
    for epoch in range(EPOCHS):
        print(f"\nStart of epoch {epoch + 1}")

        for step, (x_batch, y_reg_batch) in enumerate(train_dataset):

            if MANUALLY_SORT_EVERY_BATCH:
                sort_indices = tf.argsort(tf.squeeze(y_reg_batch))
                x_batch = tf.gather(x_batch, sort_indices)
                y_reg_batch = tf.gather(y_reg_batch, sort_indices)

            loss, reg_loss, rep_loss = representation_train_step(x_batch, y_reg_batch)
            if step % 10 == 0:
                print(
                    f"  Step {step:3d} | "
                    f"Total: {float(loss):.4f} | "
                    f"Regression: {float(reg_loss):.4f} | "
                    f"Representation: {float(rep_loss):.4f}"
                )

    for layer in model.layers:
        layer.trainable = False
        if layer.name == 'representation':
           break

    for epoch in range(SECOND_STAGE_EPOCHS if SECOND_STAGE_EPOCHS is not None else EPOCHS):
        print(f"\nStart of epoch {epoch + 1}")

        for step, (x_batch, y_reg_batch) in enumerate(train_dataset):

            if MANUALLY_SORT_EVERY_BATCH:
                sort_indices = tf.argsort(tf.squeeze(y_reg_batch))
                x_batch = tf.gather(x_batch, sort_indices)
                y_reg_batch = tf.gather(y_reg_batch, sort_indices)

            loss, reg_loss, rep_loss = regression_train_step(x_batch, y_reg_batch)
            if step % 10 == 0:
                print(
                    f"  Step {step:3d} | "
                    f"Total: {float(loss):.4f} | "
                    f"Regression: {float(reg_loss):.4f} | "
                    f"Representation: {float(rep_loss):.4f}"
                )
elif FIT_MODE == 'tune':
    for epoch in range(EPOCHS):
        print(f"\nStart of epoch {epoch + 1}")

        for step, (x_batch, y_reg_batch) in enumerate(train_dataset):

            if MANUALLY_SORT_EVERY_BATCH:
                sort_indices = tf.argsort(tf.squeeze(y_reg_batch))
                x_batch = tf.gather(x_batch, sort_indices)
                y_reg_batch = tf.gather(y_reg_batch, sort_indices)

            loss, reg_loss, rep_loss = representation_train_step(x_batch, y_reg_batch)
            if step % 10 == 0:
                print(
                    f"  Step {step:3d} | "
                    f"Total: {float(loss):.4f} | "
                    f"Regression: {float(reg_loss):.4f} | "
                    f"Representation: {float(rep_loss):.4f}"
                )

    for epoch in range(SECOND_STAGE_EPOCHS if SECOND_STAGE_EPOCHS is not None else EPOCHS):
        print(f"\nStart of epoch {epoch + 1}")

        for step, (x_batch, y_reg_batch) in enumerate(train_dataset):

            if MANUALLY_SORT_EVERY_BATCH:
                sort_indices = tf.argsort(tf.squeeze(y_reg_batch))
                x_batch = tf.gather(x_batch, sort_indices)
                y_reg_batch = tf.gather(y_reg_batch, sort_indices)

            loss, reg_loss, rep_loss = regression_train_step(x_batch, y_reg_batch)
            if step % 10 == 0:
                print(
                    f"  Step {step:3d} | "
                    f"Total: {float(loss):.4f} | "
                    f"Regression: {float(reg_loss):.4f} | "
                    f"Representation: {float(rep_loss):.4f}"
                )

sort_indices = tf.argsort(tf.squeeze(y_reg_train))
y_reg_train = tf.gather(y_reg_train, sort_indices)
x_train = tf.gather(x_train, sort_indices)

predictions, representations = model.predict(x_train)

labels_shifted = tf.roll(y_reg_train, shift=-1, axis=0)
representations_shifted = tf.roll(representations, shift=-1, axis=0)

label_distances = tf.linalg.norm(y_reg_train - labels_shifted, axis=1) + 1e-9
representation_distances = tf.linalg.norm(representations - representations_shifted, axis=1) + 1e-9

ratios = label_distances / representation_distances

# print(representations[:50])
# print(model.layers)

x_train = x_train.numpy()
y_reg_train = y_reg_train.numpy()

from sklearn.linear_model import LinearRegression
lin_reg_model = LinearRegression()
lin_reg_model.fit(predictions.reshape(-1, 1), y_reg_train.reshape(-1))

plot_true_vs_predictions(
    y_reg_train,
    predictions
)




def plot_representation_space(representation_vectors, labels, dim_one_index, dim_two_index):
    min_rep_x = np.min(representation_vectors[:, dim_one_index])
    max_rep_x = np.max(representation_vectors[:, dim_one_index])
    min_rep_y = np.min(representation_vectors[:, dim_two_index])
    max_rep_y = np.max(representation_vectors[:, dim_two_index])
    distance_rep_x = max_rep_x - min_rep_x
    distance_rep_y = max_rep_y - min_rep_y
    if distance_rep_x > distance_rep_y:
        diff = distance_rep_x - distance_rep_y
        half_diff = diff / 2
        min_rep_y -= half_diff
        max_rep_y += half_diff
    else:
        diff = distance_rep_y - distance_rep_x
        half_diff = diff / 2
        min_rep_x -= half_diff
        max_rep_x += half_diff

    min_rep_x -= 0.2
    max_rep_x += 0.2
    min_rep_y -= 0.2
    max_rep_y += 0.2

    plt.scatter(
        representation_vectors[:, dim_one_index].reshape(-1),
        representation_vectors[:, dim_two_index].reshape(-1),
        c=labels,
        cmap='jet',
        alpha=0.5
    )
    plt.xlim(min_rep_x, max_rep_x)
    plt.ylim(min_rep_y, max_rep_y)
    plt.colorbar()
    plt.show()

# plot_representation_space(representations, y_reg_train, 0, 1)
# plot_representation_space(representations, 0, 2)
# plot_representation_space(representations, 0, 3)
# plot_representation_space(representations, 1, 2)
# plot_representation_space(representations, 1, 3)
# plot_representation_space(representations, 2, 3)

# tsne_visualization(
#     model=model,
#     data=x_train,
#     labels=y_reg_train,
#     representation_layer_index=REPRESENTATION_LAYER_INDEX,
#     gradient='jet',
#     perplexity=300
# )

plot_representation_space(representations, y_reg_train, 0, 1)

weights, biases = model.get_layer('output').get_weights()

print(weights)
print(biases)

x = np.linspace(-1, 1, 50)
y = np.linspace(0, 1, 50)

# 2. Create a 2D grid from x and y arrays
X, Y = np.meshgrid(x, y)

Z = X * weights[0][0] + Y * weights[1][0] + biases[0]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
plane = ax.plot_surface(X, Y, Z, cmap='jet', alpha=0.3, edgecolor='none')

ax.scatter(
    representations[:, 0].reshape(-1),
    representations[:, 1].reshape(-1),
    y_reg_train.reshape(-1),
    c=y_reg_train,
    cmap='jet',
    alpha=0.5
)

import pickle
pickle.dump(fig, open(f'{FIGURE_PICKLE_NAME}.fig.pickle', 'wb')) # This is for Python 3 - py2 may need `file` instead of `open`

plt.show()

# x_test = tf.clip_by_value(tf.random.normal((300, 2)) + tf.random.normal((300, 2), stddev=0.1), clip_value_min=-3, clip_value_max=3)
# y_reg_test = tf.reshape(tf.linalg.norm(x_test, axis=1), (-1, 1))
# test_predictions, test_representations = model.predict(x_test)
#
# x_test = x_test.numpy()
# y_reg_test = y_reg_test.numpy()
#
# plot_representation_space(test_representations, y_reg_test, 0, 1)
#
# plot_true_vs_predictions(
#     y_reg_test,
#     test_predictions
# )

# adjusted_predictions = lin_reg_model.predict(test_predictions.reshape(-1, 1))
#
# plot_true_vs_predictions(
#     y_reg_test,
#     adjusted_predictions
# )